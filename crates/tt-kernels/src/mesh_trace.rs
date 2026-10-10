//! Traces across a [`Fabric`]: a stretch of mesh work captured once and
//! replayed with the host doing nothing but ordering (`hardware-coverage.md`
//! X4d and R4; [`crate::trace`] is the single-chip machinery this builds on).
//!
//! **What a mesh capture is.** Compute on a chip is a [`Session`] op, which a
//! session trace records and replays from one `CALL` entry per unit. Data
//! between chips is not: an Ethernet transfer
//! ([`Fabric::transfer_tensor`]) is the host starting E1 movers between two
//! chips' GDDR slots and waiting for their acknowledgements. A single trace
//! per chip would therefore replay everything a chip computed with no way of
//! putting a transfer between two of its ops. So a mesh trace is a *sequence*:
//!
//! * [`Step::Phase`]: one session trace per chip that ran work since the last
//!   transfer, replayed together (chips do not depend on each other inside a
//!   phase, only across a transfer) and synchronized before the next step;
//! * [`Step::Transfer`]: the exact route and GDDR ranges a transfer used, run
//!   again by the host after both chips have synchronized -- the same
//!   ordering [`Fabric::transfer_tensor`] has, which is what keeps a replay's
//!   cross-chip data dependencies what the capture's were.
//!
//! The segments are captured by closing every chip's open capture when a
//! transfer starts and reopening it when the transfer ends; a chip that ran
//! nothing since the last transfer keeps its capture open instead (an empty
//! trace is not kept, and what the capture deferred stays deferred).
//!
//! **What a replay may not change** is each tensor's slots: the transfers name
//! them. The sessions' own rule (a trace holds every allocation that existed
//! when its capture ended, a free of one waits for the trace's release) covers
//! the slots a segment names; [`Fabric::end_mesh_trace`] checks that it also
//! covers both ends of every transfer, and refuses the trace
//! ([`MeshTraceError::UnheldTransfer`]) if a slot a transfer names would be
//! free to reuse.
//!
//! Nothing but device work and transfers is recorded. A download or a host
//! write during the capture is refused by the session
//! ([`crate::trace::TraceError::HostTransfer`]), and the host-staged
//! [`Fabric::matmul`] is already refused on a resident fabric.

use std::collections::HashMap;

use tt_device::Transport;
use tt_isa::dram::DramRange;

use crate::session::Session;
use crate::shard::Fabric;
use crate::tensor::{DramTensor, TensorError};
use crate::trace::{TraceError, TraceId};

/// Why a mesh trace operation was refused.
#[derive(Debug)]
pub enum MeshTraceError {
    /// A mesh capture is already open.
    Capturing,
    /// No mesh capture is open.
    NotCapturing,
    /// No mesh trace by this id.
    Unknown(u64),
    /// The capture ran no device work on any chip.
    Empty,
    /// A chip's session refused (`op` names what was being done).
    Session {
        chip: usize,
        op: &'static str,
        error: TensorError,
    },
    /// A transfer failed during the capture or a replay.
    Transfer(String),
    /// The capture holds a transfer whose slots on `chip` no captured trace
    /// holds, so a free followed by an allocation could put another tensor
    /// where a replay's transfer reads or writes: not replayable faithfully.
    UnheldTransfer {
        /// Index of the transfer among the capture's transfers.
        transfer: usize,
        chip: usize,
        /// Whether `chip` sends (`"source"`) or receives (`"destination"`).
        end: &'static str,
    },
}

impl core::fmt::Display for MeshTraceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MeshTraceError::Capturing => write!(f, "a mesh trace capture is already open"),
            MeshTraceError::NotCapturing => write!(f, "no mesh trace capture is open"),
            MeshTraceError::Unknown(id) => write!(f, "no mesh trace {id}: never made or released"),
            MeshTraceError::Empty => write!(f, "the mesh capture ran no device work"),
            MeshTraceError::Session { chip, op, error } => {
                write!(f, "mesh trace: chip {chip} {op}: {error}")
            }
            MeshTraceError::Transfer(e) => write!(f, "mesh trace: Ethernet transfer: {e}"),
            MeshTraceError::UnheldTransfer {
                transfer,
                chip,
                end,
            } => write!(
                f,
                "mesh trace: transfer {transfer}'s {end} slots on chip {chip} are held by no \
                 captured trace there, so a replay could not keep them from being reused; \
                 run captured device work on chip {chip} after the transfer"
            ),
        }
    }
}

impl std::error::Error for MeshTraceError {}

/// One transfer, as run: replayed with the same route and ranges.
struct TransferStep {
    from: usize,
    to: usize,
    route: Vec<usize>,
    ranges: Vec<(DramRange, DramRange)>,
    src: DramTensor,
    dst: DramTensor,
}

enum Step {
    Phase(Vec<(usize, TraceId)>),
    Transfer(Box<TransferStep>),
}

/// A finished mesh trace.
struct MeshTrace {
    steps: Vec<Step>,
    /// Peer matmuls per chip one replay completes (`FabricExecution`).
    matmuls: Vec<u64>,
}

/// What a mesh trace holds, for gates and diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshTraceInfo {
    /// Phases (one session trace per chip that ran work between transfers).
    pub phases: usize,
    pub transfers: usize,
    /// Session traces held, per chip.
    pub segments: Vec<usize>,
}

struct MeshCapture {
    steps: Vec<Step>,
    matmuls: Vec<u64>,
    /// The first failure while closing or reopening segments around a
    /// transfer: the capture is unusable, reported at its end.
    failed: Option<MeshTraceError>,
}

/// A fabric's mesh trace state.
#[derive(Default)]
pub struct MeshState {
    capture: Option<MeshCapture>,
    traces: HashMap<u64, MeshTrace>,
    next: u64,
}

impl MeshState {
    pub(crate) fn capturing(&self) -> bool {
        self.capture.is_some()
    }

    pub(crate) fn note_matmul(&mut self, chip: usize) {
        if let Some(c) = self.capture.as_mut() {
            c.matmuls[chip] += 1;
        }
    }
}

impl<T: Transport> Session<T> {
    /// End the open capture if it recorded device work; with nothing recorded
    /// leave it open (and `None`), so what it deferred stays deferred and a
    /// transfer ahead of the first op needs no trace of its own.
    pub(crate) fn end_trace_segment(&mut self) -> Result<Option<TraceId>, TensorError> {
        let Some(c) = self.capture.as_ref() else {
            return Err(TraceError::NotCapturing.into());
        };
        if c.units.iter().all(|u| u.stream.is_empty()) {
            return Ok(None);
        }
        self.end_trace().map(Some)
    }

    /// Does trace `id` hold `t`'s slots (a free of them waits for its release)?
    pub(crate) fn trace_holds_tensor(&self, id: TraceId, t: &DramTensor) -> bool {
        self.traces
            .get(&id.0)
            .is_some_and(|trace| trace.holds(&t.placement))
    }
}

impl<T: Transport> Fabric<T> {
    /// Is a mesh capture open?
    pub fn mesh_capturing(&self) -> bool {
        self.mesh.capturing()
    }

    /// Start capturing a mesh trace: every chip's session starts a session
    /// capture, and [`Fabric::transfer_tensor`] records itself until
    /// [`Fabric::end_mesh_trace`]. Everything run in between runs as usual.
    pub fn begin_mesh_trace(&mut self) -> Result<(), MeshTraceError> {
        if self.mesh.capture.is_some() {
            return Err(MeshTraceError::Capturing);
        }
        for chip in 0..self.chips.len() {
            if let Err(error) = self.chips[chip].session().begin_trace() {
                self.close_all_discarding();
                return Err(MeshTraceError::Session {
                    chip,
                    op: "could not begin a capture",
                    error,
                });
            }
        }
        self.mesh.capture = Some(MeshCapture {
            steps: Vec::new(),
            matmuls: vec![0; self.chips.len()],
            failed: None,
        });
        Ok(())
    }

    /// End every chip's open capture and release what they made.
    fn close_all_discarding(&mut self) {
        for chip in 0..self.chips.len() {
            let session = self.chips[chip].session();
            if session.capturing() {
                if let Ok(id) = session.end_trace() {
                    let _ = session.release_trace(id);
                }
            }
        }
    }

    /// Drop an open mesh capture and everything it recorded (a panic in the
    /// captured closure, an error in an op): no trace is made.
    pub fn abort_mesh_trace(&mut self) {
        if let Some(cap) = self.mesh.capture.take() {
            for step in cap.steps {
                if let Step::Phase(segments) = step {
                    for (chip, id) in segments {
                        let _ = self.chips[chip].session().release_trace(id);
                    }
                }
            }
            self.close_all_discarding();
        }
    }

    /// Close every chip's capture that recorded work, as one phase.
    fn close_phase(&mut self) -> Result<Vec<usize>, MeshTraceError> {
        let mut segments = Vec::new();
        for chip in 0..self.chips.len() {
            let session = self.chips[chip].session();
            if !session.capturing() {
                continue;
            }
            match session.end_trace_segment() {
                Ok(Some(id)) => segments.push((chip, id)),
                Ok(None) => {}
                Err(error) => {
                    return Err(MeshTraceError::Session {
                        chip,
                        op: "could not end a segment",
                        error,
                    })
                }
            }
        }
        let closed = segments.iter().map(|&(chip, _)| chip).collect();
        if !segments.is_empty() {
            self.mesh
                .capture
                .as_mut()
                .expect("capturing")
                .steps
                .push(Step::Phase(segments));
        }
        Ok(closed)
    }

    /// [`Fabric::transfer_tensor`]'s hops while capturing: the segments end
    /// before them and reopen after, so each replay runs them between the
    /// same two groups of device work.
    pub(crate) fn captured_transfer(
        &mut self,
        from: usize,
        to: usize,
        route: &[usize],
        ranges: &[(DramRange, DramRange)],
        src: &DramTensor,
        dst: &DramTensor,
    ) -> Result<(), String> {
        let closed = match self.close_phase() {
            Ok(closed) => closed,
            Err(e) => {
                let message = e.to_string();
                self.mesh
                    .capture
                    .as_mut()
                    .expect("capturing")
                    .failed
                    .get_or_insert(e);
                return Err(message);
            }
        };
        let ran = self.run_hops(route, ranges);
        let mut reopen = None;
        for chip in closed {
            if let Err(error) = self.chips[chip].session().begin_trace() {
                reopen.get_or_insert(MeshTraceError::Session {
                    chip,
                    op: "could not reopen a capture",
                    error,
                });
            }
        }
        let capture = self.mesh.capture.as_mut().expect("capturing");
        if let Some(e) = reopen {
            capture.failed.get_or_insert(e);
        }
        if let Err(e) = ran {
            let e = MeshTraceError::Transfer(e.to_string());
            let message = e.to_string();
            capture.failed.get_or_insert(e);
            return Err(message);
        }
        capture.steps.push(Step::Transfer(Box::new(TransferStep {
            from,
            to,
            route: route.to_vec(),
            ranges: ranges.to_vec(),
            src: src.clone(),
            dst: dst.clone(),
        })));
        Ok(())
    }

    /// End the capture: wait for what ran, keep each chip's segments, and
    /// check that the slots every transfer names are held. On any failure no
    /// trace is made and what the capture kept is released.
    pub fn end_mesh_trace(&mut self) -> Result<u64, MeshTraceError> {
        let Some(mut cap) = self.mesh.capture.take() else {
            return Err(MeshTraceError::NotCapturing);
        };
        // The last phase: every chip ends, with nothing left to carry over; a
        // chip with nothing in its capture ends with no trace.
        let mut last = Vec::new();
        let mut failure = cap.failed.take();
        for chip in 0..self.chips.len() {
            let session = self.chips[chip].session();
            if !session.capturing() {
                continue;
            }
            match session.end_trace() {
                Ok(id) => last.push((chip, id)),
                Err(TensorError::Trace(TraceError::Empty)) => {}
                Err(error) => {
                    failure.get_or_insert(MeshTraceError::Session {
                        chip,
                        op: "could not end its capture",
                        error,
                    });
                }
            }
        }
        if !last.is_empty() {
            cap.steps.push(Step::Phase(last));
        }
        let release = |fabric: &mut Self, steps: Vec<Step>| {
            for step in steps {
                if let Step::Phase(segments) = step {
                    for (chip, id) in segments {
                        let _ = fabric.chips[chip].session().release_trace(id);
                    }
                }
            }
        };
        if let Some(e) = failure {
            release(self, cap.steps);
            return Err(e);
        }
        if !cap.steps.iter().any(|s| matches!(s, Step::Phase(_))) {
            release(self, cap.steps);
            return Err(MeshTraceError::Empty);
        }
        // Both ends of every transfer must be held on their chip.
        let mut transfer = 0;
        let mut unheld = None;
        'steps: for step in &cap.steps {
            let Step::Transfer(t) = step else { continue };
            for (chip, tensor, end) in [(t.from, &t.src, "source"), (t.to, &t.dst, "destination")] {
                let held = cap.steps.iter().any(|s| match s {
                    Step::Phase(segments) => segments.iter().any(|&(c, id)| {
                        c == chip && self.chips[c].session().trace_holds_tensor(id, tensor)
                    }),
                    Step::Transfer(_) => false,
                });
                if !held {
                    unheld = Some(MeshTraceError::UnheldTransfer {
                        transfer,
                        chip,
                        end,
                    });
                    break 'steps;
                }
            }
            transfer += 1;
        }
        if let Some(e) = unheld {
            release(self, cap.steps);
            return Err(e);
        }
        self.mesh.next += 1;
        let id = self.mesh.next;
        self.mesh.traces.insert(
            id,
            MeshTrace {
                steps: cap.steps,
                matmuls: cap.matmuls,
            },
        );
        Ok(id)
    }

    /// What mesh trace `id` holds.
    pub fn mesh_trace_info(&self, id: u64) -> Option<MeshTraceInfo> {
        let trace = self.mesh.traces.get(&id)?;
        let mut info = MeshTraceInfo {
            phases: 0,
            transfers: 0,
            segments: vec![0; self.chips.len()],
        };
        for step in &trace.steps {
            match step {
                Step::Phase(segments) => {
                    info.phases += 1;
                    for &(chip, _) in segments {
                        info.segments[chip] += 1;
                    }
                }
                Step::Transfer(_) => info.transfers += 1,
            }
        }
        Some(info)
    }

    /// Run mesh trace `id` again, in the capture's order: each phase's segments
    /// replayed on their chips and synchronized, each transfer run between
    /// them. Returns with every chip idle. Between replays the caller writes
    /// new values into the tensors the trace reads (chip 0's session).
    pub fn replay_mesh_trace(&mut self, id: u64) -> Result<(), MeshTraceError> {
        if self.mesh.capture.is_some() {
            return Err(MeshTraceError::Capturing);
        }
        let Some(trace) = self.mesh.traces.remove(&id) else {
            return Err(MeshTraceError::Unknown(id));
        };
        let result = self.replay_steps(&trace.steps);
        if result.is_ok() {
            for (chip, n) in trace.matmuls.iter().enumerate() {
                self.execution.completed_matmuls[chip] += n;
            }
        }
        self.mesh.traces.insert(id, trace);
        result
    }

    fn replay_steps(&mut self, steps: &[Step]) -> Result<(), MeshTraceError> {
        let session_error = |chip, op, error| MeshTraceError::Session { chip, op, error };
        for step in steps {
            match step {
                Step::Phase(segments) => {
                    let mut queued = Vec::new();
                    let mut failure = None;
                    for &(chip, id) in segments {
                        match self.chips[chip].session().replay(id) {
                            Ok(()) => queued.push(chip),
                            Err(e) => {
                                failure = Some(session_error(chip, "could not replay", e));
                                break;
                            }
                        }
                    }
                    // Whatever was queued is waited for, even after a failure,
                    // so no chip is left running behind the error.
                    for chip in queued {
                        if let Err(e) = self.chips[chip].session().sync() {
                            failure.get_or_insert(session_error(chip, "failed a replay", e));
                        }
                    }
                    if let Some(e) = failure {
                        return Err(e);
                    }
                }
                Step::Transfer(t) => {
                    for chip in [t.from, t.to] {
                        self.chips[chip]
                            .session()
                            .sync()
                            .map_err(|e| session_error(chip, "failed before a transfer", e))?;
                    }
                    self.run_hops(&t.route, &t.ranges)
                        .map_err(|e| MeshTraceError::Transfer(e.to_string()))?;
                }
            }
        }
        Ok(())
    }

    /// Give mesh trace `id` back: every session trace it made is released
    /// (deferred frees of what they hold happen then).
    pub fn release_mesh_trace(&mut self, id: u64) -> Result<(), MeshTraceError> {
        let Some(trace) = self.mesh.traces.remove(&id) else {
            return Err(MeshTraceError::Unknown(id));
        };
        let mut failure = None;
        for step in trace.steps {
            if let Step::Phase(segments) = step {
                for (chip, id) in segments {
                    if let Err(error) = self.chips[chip].session().release_trace(id) {
                        failure.get_or_insert(MeshTraceError::Session {
                            chip,
                            op: "could not release a segment",
                            error,
                        });
                    }
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
}
