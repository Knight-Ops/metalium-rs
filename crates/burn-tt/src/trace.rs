//! Traces through Burn (`tt_kernels::trace`, X4d): a forward pass captured
//! once, then run again on new inputs without the host building any of its
//! ops.
//!
//! ```ignore
//! let (trace, first_logits) = burn_tt::Trace::capture(&x, || model_forward(&x))?;
//! for batch in batches {
//!     let logits: Vec<f32> = trace.run(batch)?;
//! }
//! ```
//!
//! What `capture` and `run` give back is the output's values on the host, not
//! a tensor: the output's buffer is rewritten by every run, so a tensor of it
//! would change under whoever held one -- the aliasing a trace must not let
//! in.
//! The input and output are the capture's, kept alive by the trace; dropping
//! the trace releases it on the device.

use burn_backend::TensorMetadata;

use crate::error::TtError;
use crate::server::{self, BufferId};
use crate::tensor::TtTensor;
use crate::TtDevice;

/// A captured trace: see the module documentation.
#[derive(Debug)]
pub struct Trace {
    device: TtDevice,
    id: u64,
    input: BufferId,
    output: BufferId,
    /// The output as stored, `[rows, cols]`.
    output_dims: [usize; 2],
    /// The capture's input and output, holding their buffers.
    _held: [TtTensor; 2],
}

/// Ends a capture that did not end normally (an op's panic in `f`), so the
/// session is not left capturing.
struct Capturing {
    device: TtDevice,
    open: bool,
    /// A training step's capture ([`TracedTrainingStep`]), counted in
    /// [`TRAINING`].
    training: bool,
}

/// The devices capturing a training step now, in this process (a device once
/// for each open training [`Capturing`]).
static TRAINING: std::sync::Mutex<Vec<TtDevice>> = std::sync::Mutex::new(Vec::new());

/// Is `device` capturing a training step? Its replays write the parameters in
/// place (`copy_into`), so a tensor derived from one and kept (a convolution's
/// relaid weights, `TtTensor::derived`) is neither read nor filled then: the
/// capture recomputes it, and so does each replay. An inference trace's
/// capture uses what is kept: its replays read the parameters as they were,
/// and a later in-place write drops what was derived from them.
pub(crate) fn capturing_training(device: TtDevice) -> bool {
    TRAINING
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .contains(&device)
}

impl Capturing {
    fn new(device: TtDevice, training: bool) -> Self {
        if training {
            TRAINING
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(device);
        }
        Capturing {
            device,
            open: true,
            training,
        }
    }
}

impl Drop for Capturing {
    fn drop(&mut self) {
        if self.training {
            let mut c = TRAINING.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(i) = c.iter().position(|&d| d == self.device) {
                c.swap_remove(i);
            }
        }
        if self.open {
            // Whatever it captured, it is no trace: ended, and released.
            if let Some(Ok(id)) = server::run_if_attached(self.device, |e, _| e.end_trace()) {
                server::run_if_attached(self.device, move |e, _| e.release_trace(id));
            }
        }
    }
}

impl Trace {
    /// Capture what `f` computes from `input`, whose device copy the replays
    /// write: `f`'s ops run as usual (the capture is the first run) and its
    /// result, which must be computed on the device, is the trace's output:
    /// its values from the capture's run are returned alongside it, row-major
    /// as stored ([`Trace::output_dims`]).
    ///
    /// Only an F32 tensor the device holds as its own buffer is an input (not
    /// a transposed or row view: a view's slots are another tensor's). An op
    /// in `f` that falls back to the host would need a download, which a
    /// capture refuses: it panics, as a device error in an op does.
    pub fn capture(
        input: &TtTensor,
        f: impl FnOnce() -> TtTensor,
    ) -> Result<(Trace, Vec<f32>), TtError> {
        let device = input.device;
        if !input.is_stored_f32() || input.storage_format() != crate::storage::StorageFormat::F32 {
            return Err(TtError::Device("a trace's input is an F32 tensor".into()));
        }
        let d = input.to_dram();
        if d.transposed || d.buffer.parent.is_some() {
            return Err(TtError::Device(
                "a trace's input must own its buffer: not a transposed or row view".into(),
            ));
        }
        let input_id = d.buffer.id;
        // Each replay writes it.
        server::written_in_place(input_id);
        server::run(device, |e, _| e.begin_trace())?;
        let mut guard = Capturing::new(device, false);
        let out = f();
        let output = match out.dram() {
            Some(d) if !d.transposed && out.device == device && out.is_stored_f32() => {
                Ok((d.buffer.id, [d.buffer.rows, d.buffer.cols]))
            }
            _ => Err(TtError::Device(
                "a trace's output must be F32, computed on its device, untransposed".into(),
            )),
        };
        guard.open = false;
        let ended = server::run(device, |e, _| e.end_trace());
        let ((output, output_dims), id) = (output?, ended?);
        let trace = Trace {
            device,
            id,
            input: input_id,
            output,
            output_dims,
            _held: [input.clone(), out],
        };
        let [r, c] = output_dims;
        let first = server::run(device, move |e, ids| {
            ids.get(output).and_then(|o| e.download(o))
        });
        debug_assert!(first.as_ref().is_ok_and(|v| v.len() == r * c));
        Ok((trace, first?))
    }

    /// The output as stored on the device, `[rows, cols]`: what [`Trace::run`]
    /// returns, row-major.
    pub fn output_dims(&self) -> [usize; 2] {
        self.output_dims
    }

    /// Write `values` (the input's, row-major as stored) into the input,
    /// replay, and read the output back, row-major as stored: one round trip
    /// to the device.
    pub fn run(&self, values: Vec<f32>) -> Result<Vec<f32>, TtError> {
        self.run_timed(values).map(|r| r.output)
    }

    /// [`Trace::run`], with the time the input's write, the replay and the
    /// output's read each took on the device's side.
    pub fn run_timed(&self, values: Vec<f32>) -> Result<server::TraceRun, TtError> {
        let (id, input, output) = (self.id, self.input, self.output);
        server::run(self.device, move |e, ids| {
            ids.healthy()?;
            e.run_trace(id, ids.get(input)?, &values, ids.get(output)?)
        })
    }
}

/// Whether a capture with state named up front writes it in place
/// ([`set_state_in_place`]); on unless a gate turns it off.
static STATE_IN_PLACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Turn off (or back on) writing a capture's state in place, process-wide: off, a
/// [`SharedStateTraces`] capture copies its cache and writes it back after the step, as
/// [`TracedInference::capture_with_state`] does. For gates' negative controls. Dev-facing.
#[doc(hidden)]
pub fn set_state_in_place(on: bool) {
    STATE_IN_PLACE.store(on, std::sync::atomic::Ordering::SeqCst);
}

/// The buffers a capture owns as its state while its step runs ([`server::donate`]), given
/// back when it is dropped, so an error or a panic inside the step cannot leave one donated.
#[derive(Default)]
struct Donation(Vec<server::BufferId>);

impl Donation {
    fn give(&mut self, id: server::BufferId) {
        server::donate(id);
        self.0.push(id);
    }
}

impl Drop for Donation {
    fn drop(&mut self) {
        for &id in &self.0 {
            server::undonate(id);
        }
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        let id = self.id;
        // A drop never opens a card, and never panics: a detached device has
        // nothing left to release.
        server::run_if_attached(self.device, move |e, _| e.release_trace(id));
    }
}

/// A generalized multi-tensor captured inference trace.
///
/// Supports arbitrary numbers of inputs and outputs of any storable dtype
/// (F32, BF16, I32, Bool). On replay, new inputs are written to the card and
/// the trace is executed via a single `CALL` per tile without any host op
/// building or graph evaluation.
#[derive(Debug)]
pub struct TracedInference {
    device: TtDevice,
    id: u64,
    inputs: Vec<BufferId>,
    outputs: Vec<(BufferId, server::OutputKind, [usize; 2])>,
    _held: Vec<TtTensor>,
}

/// Which capture [`TracedInference::capture_pass`] begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// A plain capture (`Session::begin_trace`).
    Plain,
    /// A plain capture that plans the next (`Session::begin_trace_plan_pass`).
    Planning,
    /// One that applies that plan, kept under the key if it holds
    /// (`Session::begin_trace_planned_as`).
    Planned(u64),
    /// One that applies the plan kept under the key, with no plan pass
    /// (`Session::begin_trace_cached`).
    Cached(u64),
}

impl TracedInference {
    /// Capture inference over arbitrary inputs and outputs.
    pub fn capture(
        inputs: &[&TtTensor],
        f: impl FnOnce() -> Vec<TtTensor>,
    ) -> Result<(Self, Vec<server::OutputPayload>), TtError> {
        Self::capture_with_state(inputs, || (f(), Vec::new()))
    }

    /// [`TracedInference::capture`] of a step that carries state from one run
    /// to the next -- a decoder's key/value cache. `f` returns the outputs and
    /// `(new, state)` pairs: an on-device `copy_into` of each `new` over its
    /// `state` is recorded before the capture ends, so every replay leaves the
    /// state as the step computed it, read by the next replay, with no host
    /// copy. Each `state` must be a tensor the step reads (not one it makes):
    /// it is kept alive by the trace.
    pub fn capture_with_state(
        inputs: &[&TtTensor],
        f: impl FnOnce() -> (Vec<TtTensor>, Vec<(TtTensor, TtTensor)>),
    ) -> Result<(Self, Vec<server::OutputPayload>), TtError> {
        Self::capture_donating(inputs, &[], f)
    }

    /// [`TracedInference::capture_with_state`] of a step whose state is named before it runs:
    /// `state`, each tensor owning its buffer, is the capture's to write in place
    /// (`docs/plans/in-place-cache-write.md`). A row scatter into one (`scatter_nd` with
    /// `Assign`, a cache write) then writes the state's own buffer instead of a copy of it,
    /// and a `(new, state)` pair over one buffer needs no write-back. A read of a state tensor
    /// after its write, or of a view of it taken before, is refused by name: the step must
    /// read what the write returned. The donation ends with the step, also on an error.
    pub(crate) fn capture_donating(
        inputs: &[&TtTensor],
        state: &[&TtTensor],
        f: impl FnOnce() -> (Vec<TtTensor>, Vec<(TtTensor, TtTensor)>),
    ) -> Result<(Self, Vec<server::OutputPayload>), TtError> {
        Self::capture_pass(inputs, state, f, Pass::Plain)
            .map(|c| c.expect("a plain capture always begins"))
    }

    /// [`TracedInference::capture_donating`], twice where the device plans captures for L1
    /// residency (`crate::set_residency`, `docs/plans/planned-l1-residency.md`): the plain
    /// capture plans, and a second run of `f` is captured applying the plan, its trace kept in
    /// the plain one's place. The step runs twice, so it must be one whose second run leaves what
    /// its first did: it writes its state, as a decoder's cache row, and does not accumulate
    /// into it. A planned capture that is not its plan's is refused by the session, its reason
    /// printed, and `f` is captured plainly once more: its trace is kept, and the state is what a
    /// plain run leaves.
    ///
    /// `key` names the graph `f` captures (`docs/plans/planned-l1-residency.md` 2h): a plan kept
    /// under it from an earlier capture is applied in one capture, with no plan pass; if the
    /// session refuses it (the graph is not the plan's), it is forgotten and `f` captured twice as
    /// without one.
    pub(crate) fn capture_donating_planned(
        inputs: &[&TtTensor],
        state: &[&TtTensor],
        key: u64,
        mut f: impl FnMut() -> (Vec<TtTensor>, Vec<(TtTensor, TtTensor)>),
    ) -> Result<(Self, Vec<server::OutputPayload>), TtError> {
        match Self::capture_pass(inputs, state, &mut f, Pass::Cached(key)) {
            Ok(Some(cached)) => return Ok(cached),
            Ok(None) => {}
            Err(e) => {
                eprintln!("burn-tt: the cached residency plan was refused; planning again: {e}")
            }
        }
        let (plain, first) = Self::capture_pass(inputs, state, &mut f, Pass::Planning)?
            .expect("a plan pass always begins");
        if !server::run(plain.device, |e, _| e.has_pending_plan())? {
            return Ok((plain, first));
        }
        // The plain trace is released before the planned capture: its programs name addresses
        // planned in the whole arena, the planned capture's in a smaller one, and both held at
        // once filled the program cache of a model on 12 tiles. A refused planned capture is
        // captured plainly again (`f` runs a third time, as it may run twice).
        drop(plain);
        match Self::capture_pass(inputs, state, &mut f, Pass::Planned(key)) {
            Ok(Some(planned)) => Ok(planned),
            refused => {
                if let Err(e) = refused {
                    eprintln!(
                        "burn-tt: the residency-planned capture was refused; capturing plainly: {e}"
                    );
                }
                Ok(Self::capture_pass(inputs, state, &mut f, Pass::Plain)?
                    .expect("a plain capture always begins"))
            }
        }
    }

    /// One capture of `f`, as `pass` says. `None` if planned and there is no plan to apply.
    fn capture_pass(
        inputs: &[&TtTensor],
        state: &[&TtTensor],
        f: impl FnOnce() -> (Vec<TtTensor>, Vec<(TtTensor, TtTensor)>),
        pass: Pass,
    ) -> Result<Option<(Self, Vec<server::OutputPayload>)>, TtError> {
        let Some(first_in) = inputs.first() else {
            return Err(TtError::Device(
                "a trace needs at least one input tensor".into(),
            ));
        };
        let device = first_in.device;
        let mut in_ids = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            if inp.device != device {
                return Err(TtError::Device(
                    "all trace inputs must belong to the same device".into(),
                ));
            }
            if !inp.is_storable() || inp.storage_format() != crate::storage::StorageFormat::F32 {
                return Err(TtError::Device(
                    "a trace input must be a storable tensor".into(),
                ));
            }
            let d = inp.to_dram();
            if d.transposed || d.buffer.parent.is_some() {
                return Err(TtError::Device(
                    "a trace input must own its buffer: not a transposed or row view".into(),
                ));
            }
            in_ids.push(d.buffer.id);
            // Each replay writes it.
            server::written_in_place(d.buffer.id);
        }

        let mut donation = Donation::default();
        for t in state {
            if t.device != device {
                return Err(TtError::Device(
                    "state tensors must belong to the trace device".into(),
                ));
            }
            if !t.is_storable() || t.storage_format() != crate::storage::StorageFormat::F32 {
                return Err(TtError::Device(
                    "state written in place must be storable, unpacked tensors".into(),
                ));
            }
            let d = t.to_dram();
            if d.transposed || d.buffer.parent.is_some() {
                return Err(TtError::Device(
                    "state written in place must own its buffer: not a transposed or row view"
                        .into(),
                ));
            }
            if STATE_IN_PLACE.load(std::sync::atomic::Ordering::SeqCst) {
                donation.give(d.buffer.id);
            }
        }

        match pass {
            Pass::Planned(key) => {
                if !server::run(device, move |e, _| e.begin_trace_planned(key))? {
                    return Ok(None);
                }
            }
            Pass::Cached(key) => {
                if !server::run(device, move |e, _| e.begin_trace_cached(key))? {
                    return Ok(None);
                }
            }
            Pass::Planning => server::run(device, |e, _| e.begin_trace_plan_pass())?,
            Pass::Plain => server::run(device, |e, _| e.begin_trace())?,
        }
        let mut guard = Capturing::new(device, false);
        let (outs, updates) = f();
        // From here nothing reads the state as the step's: the write-back below reads it as
        // the buffer the next replay finds.
        drop(donation);
        let mut out_info = Vec::with_capacity(outs.len());
        for out in &outs {
            if out.device != device {
                return Err(TtError::Device(
                    "all trace outputs must belong to the trace device".into(),
                ));
            }
            let d = out.dram().ok_or_else(|| {
                TtError::Device("a trace output must be computed on the device".into())
            })?;
            if d.transposed {
                return Err(TtError::Device(
                    "a trace output cannot be a transposed view".into(),
                ));
            }
            let kind = if out.is_stored_f32() {
                server::OutputKind::F32
            } else {
                server::OutputKind::Bits
            };
            out_info.push((d.buffer.id, kind, [d.buffer.rows, d.buffer.cols]));
        }
        // The state written back in place, last, as `TracedTrainingStep` writes its parameters.
        let mut held_state = Vec::with_capacity(updates.len() * 2);
        for (new, state) in updates {
            if new.device != device || state.device != device {
                return Err(TtError::Device(
                    "state tensors must belong to the trace device".into(),
                ));
            }
            let (new_d, state_d) = (new.to_dram(), state.to_dram());
            if new_d.transposed || state_d.transposed || state_d.buffer.parent.is_some() {
                return Err(TtError::Device(
                    "state must own its buffer, and neither side be a transposed view".into(),
                ));
            }
            if new_d.buffer.id == state_d.buffer.id {
                // The step wrote the state's own buffer in place: nothing to write back. The
                // buffer is still one whose contents change after it is made.
                server::written_in_place(state_d.buffer.id);
            } else {
                server::copy_into(device, new_d.buffer.id, state_d.buffer.id)?;
            }
            held_state.push(new);
            held_state.push(state);
        }
        guard.open = false;
        let id = server::run(device, |e, _| e.end_trace())?;

        // Download first run's outputs
        let mut first_outputs = Vec::with_capacity(out_info.len());
        for &(out_buf, kind, _) in &out_info {
            let p = server::run(device, move |e, ids| {
                let e_buf = ids.get(out_buf)?;
                match kind {
                    server::OutputKind::F32 => e.download(e_buf).map(server::OutputPayload::F32),
                    server::OutputKind::Bits => {
                        e.download_bits(e_buf).map(server::OutputPayload::Bits)
                    }
                }
            })?;
            first_outputs.push(p);
        }

        let mut held = Vec::with_capacity(inputs.len() + outs.len());
        for inp in inputs {
            held.push((*inp).clone());
        }
        held.extend(outs);
        held.extend(held_state);

        Ok(Some((
            TracedInference {
                device,
                id,
                inputs: in_ids,
                outputs: out_info,
                _held: held,
            },
            first_outputs,
        )))
    }

    /// Name the trace in the profile mode's report (`crate::profile`); it is otherwise reported by
    /// number.
    pub fn set_name(&self, name: impl Into<String>) {
        let (id, name) = (self.id, name.into());
        let _ = server::run(self.device, move |e, _| {
            e.set_trace_name(id, name);
            Ok::<_, TtError>(())
        });
    }

    /// The number and shapes of output tensors, `[rows, cols]`.
    pub fn output_shapes(&self) -> Vec<[usize; 2]> {
        self.outputs.iter().map(|&(_, _, dims)| dims).collect()
    }

    /// The ops the capture recorded, in order: each one's Burn op (or part) as
    /// `label`, and its range of every unit's stream. What a replay runs, op by
    /// op; for profiling a replay (`docs/plans/replay-fixed-cost.md`).
    pub fn ops(&self) -> Result<Vec<tt_kernels::trace::OpRecord>, TtError> {
        let id = self.id;
        server::run(self.device, move |e, _| e.trace_ops(id))
    }

    /// What the trace costs each unit (`None` for a unit it does not use): its stream's GDDR
    /// bytes and the program-cache bytes it holds, beside what every trace on the unit holds
    /// and the cache's size (`Session::trace_footprint`). The program cache is what bounds how
    /// many traces live together.
    pub fn footprint(&self) -> Result<Vec<Option<tt_kernels::trace::UnitFootprint>>, TtError> {
        let id = self.id;
        server::run(self.device, move |e, _| e.trace_footprint(id))
    }

    /// Replay the trace with new input values.
    pub fn run(
        &self,
        inputs: Vec<server::InputPayload>,
    ) -> Result<Vec<server::OutputPayload>, TtError> {
        self.run_timed(inputs).map(|r| r.outputs)
    }

    /// Replay the trace with timing breakdown across PCIe write, card replay, and output readback.
    pub fn run_timed(
        &self,
        inputs: Vec<server::InputPayload>,
    ) -> Result<server::GenericTraceRun, TtError> {
        if inputs.len() != self.inputs.len() {
            return Err(TtError::Device(format!(
                "expected {} inputs, got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }
        let in_pairs: Vec<(BufferId, server::InputPayload)> =
            self.inputs.iter().copied().zip(inputs).collect();
        let out_pairs: Vec<(BufferId, server::OutputKind)> = self
            .outputs
            .iter()
            .map(|&(id, kind, _)| (id, kind))
            .collect();
        server::run_generic_trace(self.device, self.id, in_pairs, out_pairs)
    }

    /// Output `i` as a device tensor of the buffer the trace writes, holding what the last replay
    /// wrote there: a new tensor each call (with no host copy), so an op run on it reads the
    /// card's current data. Kept alive by the trace; an op that writes it in place would change
    /// what the next reader sees, so treat it as read-only. A decode step's logits sampled from on
    /// the card, where the trace left them.
    pub fn output_tensor(&self, i: usize) -> Result<TtTensor, TtError> {
        let held = self
            ._held
            .get(self.inputs.len() + i)
            .filter(|_| i < self.outputs.len())
            .ok_or_else(|| TtError::Device(format!("output {i} of {}", self.outputs.len())))?;
        let dram = held
            .dram()
            .ok_or_else(|| TtError::Device("a trace output must be on the device".into()))?
            .clone();
        Ok(TtTensor::on_device(
            dram,
            held.shape().clone(),
            held.dtype(),
            self.device,
        ))
    }

    /// [`TracedInference::run_timed`] reading back only the outputs `which` names (indices into
    /// the capture's outputs, in the order wanted): a decode step's chosen token without its
    /// logits.
    pub fn run_outputs(
        &self,
        inputs: Vec<server::InputPayload>,
        which: &[usize],
    ) -> Result<server::GenericTraceRun, TtError> {
        if inputs.len() != self.inputs.len() {
            return Err(TtError::Device(format!(
                "expected {} inputs, got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }
        let in_pairs: Vec<(BufferId, server::InputPayload)> =
            self.inputs.iter().copied().zip(inputs).collect();
        let out_pairs = which
            .iter()
            .map(|&i| {
                self.outputs
                    .get(i)
                    .map(|&(id, kind, _)| (id, kind))
                    .ok_or_else(|| TtError::Device(format!("output {i} of {}", self.outputs.len())))
            })
            .collect::<Result<Vec<_>, _>>()?;
        server::run_generic_trace(self.device, self.id, in_pairs, out_pairs)
    }

    /// [`TracedInference::run_timed`] with every unit's data mover stamping the replay's
    /// entries (`tt_kernels::profile`): where a replay's device time goes, op by op when
    /// read beside [`TracedInference::ops`]. Each unit's buffer holds 1,024 events, so
    /// only a small trace fits (a model of one layer). Silicon only; dev-facing.
    #[doc(hidden)]
    pub fn run_profiled(
        &self,
        inputs: Vec<server::InputPayload>,
    ) -> Result<(server::GenericTraceRun, tt_kernels::profile::DeviceProfile), TtError> {
        if inputs.len() != self.inputs.len() {
            return Err(TtError::Device(format!(
                "expected {} inputs, got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }
        let in_pairs = self.inputs.iter().copied().zip(inputs).collect();
        let out_pairs = self
            .outputs
            .iter()
            .map(|&(id, kind, _)| (id, kind))
            .collect();
        server::profile_generic_trace(self.device, self.id, in_pairs, out_pairs)
    }
}

impl Drop for TracedInference {
    fn drop(&mut self) {
        let id = self.id;
        // A drop never opens a card, and never panics: a detached device has
        // nothing left to release.
        server::run_if_attached(self.device, move |e, _| e.release_trace(id));
    }
}

/// Traces of one step at several shapes that carry the same state: a decoder's
/// prefill chunks of several sizes and its one-token step, all reading and writing
/// one key/value cache. Each trace has a key (a chunk length, say) of the caller's
/// type `K`.
///
/// Every capture is handed the same state tensors and returns the state it computed
/// from them, which its replays copy back over them on the card
/// ([`TracedInference::capture_with_state`]): so whichever trace runs, the next one
/// reads what it left, and the order in which they were captured does not matter. A
/// step must read the state it is given and nothing another capture made (a
/// tensor that held one capture's result is that trace's scratch, rewritten by its
/// replays and by no other trace's).
///
/// Two things to know:
/// - A capture runs its step once, so it writes the state. A model whose state is a
///   cache must not rely on what the cache held before a capture.
/// - The traces share the program cache (240 KiB of L1 a unit), and a capture that
///   does not fit fails the attachment for good (its error is sticky). Measure the
///   set with [`SharedStateTraces::footprint`] and reattach to retry with fewer.
#[derive(Debug)]
pub struct SharedStateTraces<K: Ord + Copy + std::fmt::Debug = usize> {
    device: TtDevice,
    state: Vec<TtTensor>,
    traces: std::collections::BTreeMap<K, TracedInference>,
    /// [`SharedStateTraces::set_plan_scope`].
    plan_scope: u64,
}

impl<K: Ord + Copy + std::fmt::Debug> SharedStateTraces<K> {
    /// A set over `state`: tensors on one device, each holding its own buffer (not a
    /// view), resident there.
    pub fn new(state: Vec<TtTensor>) -> Result<Self, TtError> {
        let Some(first) = state.first() else {
            return Err(TtError::Device(
                "shared-state traces need at least one state tensor".into(),
            ));
        };
        let device = first.device;
        for t in &state {
            if t.device != device {
                return Err(TtError::Device(
                    "shared state must belong to one device".into(),
                ));
            }
            if !t.is_storable() || t.storage_format() != crate::storage::StorageFormat::F32 {
                return Err(TtError::Device(
                    "shared state must be storable, unpacked tensors".into(),
                ));
            }
            t.ensure_resident();
            let d = t.to_dram();
            if d.transposed || d.buffer.parent.is_some() {
                return Err(TtError::Device(
                    "shared state must own its buffers: not a transposed or row view".into(),
                ));
            }
        }
        Ok(Self {
            device,
            state,
            traces: Default::default(),
            plan_scope: 0,
        })
    }

    /// Name what the steps belong to (a model), for the residency plans the device keeps for them:
    /// a plan is reused only by a step of the same key, scope and input and state shapes
    /// (`docs/plans/planned-l1-residency.md` 2h). Two sets of steps that could share a key and
    /// shapes, two models on one device, keep apart by scope. A plan reused by another graph is
    /// refused by the device before any op the graph differs at, and planned again; giving the
    /// scope spares that.
    pub fn set_plan_scope(&mut self, scope: u64) {
        self.plan_scope = scope;
    }

    /// The state every trace reads and writes back, in the order given to [`Self::new`].
    pub fn state(&self) -> &[TtTensor] {
        &self.state
    }

    /// Capture the step keyed `key` over `inputs`: `f` reads [`Self::state`] and returns
    /// the outputs and the new state, one tensor for each state tensor in its order and of
    /// its shape and dtype. Returns the capture's own outputs (its first run).
    ///
    /// Where the device plans captures for L1 residency (`crate::set_residency`), `f` runs twice
    /// ([`TracedInference::capture_donating_planned`]): a step that writes its state, not one
    /// that accumulates into it. Once a capture of the same key over inputs and state of the same
    /// shapes has been planned on the device, `f` runs once, under that plan.
    pub fn capture(
        &mut self,
        key: K,
        inputs: &[&TtTensor],
        mut f: impl FnMut() -> (Vec<TtTensor>, Vec<TtTensor>),
    ) -> Result<Vec<server::OutputPayload>, TtError> {
        if self.traces.contains_key(&key) {
            return Err(TtError::Device(format!(
                "a trace keyed {key:?} is already in the set"
            )));
        }
        let state = self.state.clone();
        let mut mismatch = None;
        let donated: Vec<&TtTensor> = self.state.iter().collect();
        // The graph's name for its residency plan: the step's key and what it reads.
        let plan_key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (self.plan_scope, format!("{key:?}")).hash(&mut h);
            for t in inputs.iter().copied().chain(&self.state) {
                (t.shape(), format!("{:?}", t.dtype())).hash(&mut h);
            }
            h.finish()
        };
        let (trace, first) =
            TracedInference::capture_donating_planned(inputs, &donated, plan_key, || {
                let (outs, new) = f();
                if new.len() != state.len() {
                    mismatch = Some(format!(
                        "the step returned {} state tensors for {}",
                        new.len(),
                        state.len()
                    ));
                } else if let Some(i) = (0..new.len()).find(|&i| {
                    new[i].shape() != state[i].shape() || new[i].dtype() != state[i].dtype()
                }) {
                    mismatch = Some(format!(
                        "new state {i} is {:?} {:?}, the state {:?} {:?}",
                        new[i].shape(),
                        new[i].dtype(),
                        state[i].shape(),
                        state[i].dtype()
                    ));
                }
                if mismatch.is_some() {
                    // Nothing written back: the trace is dropped below.
                    return (outs, Vec::new());
                }
                (outs, new.into_iter().zip(state.iter().cloned()).collect())
            })?;
        if let Some(why) = mismatch {
            drop(trace);
            return Err(TtError::Device(why));
        }
        trace.set_name(format!("{key:?}"));
        self.traces.insert(key, trace);
        Ok(first)
    }

    /// The trace keyed `key`.
    pub fn get(&self, key: K) -> Option<&TracedInference> {
        self.traces.get(&key)
    }

    /// The keys captured, ascending.
    pub fn keys(&self) -> Vec<K> {
        self.traces.keys().copied().collect()
    }

    /// What the set costs each unit ([`TracedInference::footprint`], per trace).
    pub fn footprint(&self) -> Result<SetFootprint<K>, TtError> {
        let mut traces = Vec::with_capacity(self.traces.len());
        for (&key, t) in &self.traces {
            traces.push((key, t.footprint()?));
        }
        Ok(SetFootprint { traces })
    }

    /// The device the set runs on.
    pub fn device(&self) -> TtDevice {
        self.device
    }
}

/// What a [`SharedStateTraces`] costs: each trace's per-unit footprint, by key.
#[derive(Clone, Debug)]
pub struct SetFootprint<K = usize> {
    pub traces: Vec<(K, Vec<Option<tt_kernels::trace::UnitFootprint>>)>,
}

impl<K> SetFootprint<K> {
    /// The most program-cache bytes every live trace holds on any one unit, and that
    /// cache's size: the bound on how many traces fit together.
    pub fn cache_held_bytes(&self) -> (u64, u64) {
        self.units()
            .map(|u| (u.cache_held_bytes, u.cache_bytes))
            .max()
            .unwrap_or_default()
    }

    /// The GDDR bytes of every trace's streams and sections, on every unit.
    pub fn gddr_bytes(&self) -> u64 {
        self.units().map(|u| u.gddr_bytes).sum()
    }

    fn units(&self) -> impl Iterator<Item = &tt_kernels::trace::UnitFootprint> {
        self.traces.iter().flat_map(|(_, u)| u.iter().flatten())
    }
}

/// Give `device`'s timestamper `bytes` of L1 per tile, taken from the top of the program
/// cache (`Session::set_profile_buffer`; `None` restores the fixed 1,024-event buffer), so
/// [`TracedInference::run_profiled`] can stamp a larger replay. Call it before the capture:
/// it empties the program cache, and it is refused while a trace lives. Dev-facing.
#[doc(hidden)]
pub fn set_profile_buffer(device: TtDevice, bytes: Option<u64>) -> Result<(), TtError> {
    server::run(device, move |e, _| e.set_profile_buffer(bytes))
}

/// Count what each op of every trace captured on `device` from here on moves through GDDR, as the
/// profile mode does, without it (`Session::set_count_trace_bytes`): read back as each op's
/// `bytes` from [`TracedInference::ops`]. Host work only, so the simulator runs it. Dev-facing.
#[doc(hidden)]
pub fn set_count_trace_bytes(device: TtDevice, on: bool) -> Result<(), TtError> {
    server::run(device, move |e, _| e.set_count_trace_bytes(on))
}

/// Keep `bytes` at the top of each unit's L1 arena for a persistent region in every trace captured
/// on `device` from here on (`Session::set_l1_region`; 0 keeps none): the captured ops plan below
/// it, and a capture with an op that does not is refused by name.
/// `docs/plans/planned-l1-residency.md` Phase 2. Dev-facing.
#[doc(hidden)]
pub fn set_l1_region(device: TtDevice, bytes: u64) -> Result<(), TtError> {
    server::run(device, move |e, _| e.set_l1_region(bytes))
}

/// Plan captures on `device` from here on for L1 residency, in a region of at most `bytes` a unit
/// (0 turns it off; on when the device attaches, unless `TT_RESIDENCY=0`;
/// `Session::set_residency`, `docs/plans/planned-l1-residency.md`). A capture that can run its
/// step twice ([`SharedStateTraces::capture`]) then keeps the planned trace, whose tensors move
/// between units' L1 where the plan says, and its plan, so a later capture of the same step runs
/// it once. Dev-facing.
#[doc(hidden)]
pub fn set_residency(device: TtDevice, bytes: u64) -> Result<(), TtError> {
    server::run(device, move |e, _| e.set_residency(bytes))
}

/// How many captures `device`'s session has begun, refused ones included
/// (`Session::probe_captures`): a gate counting what a build captured. Dev-facing.
#[doc(hidden)]
pub fn probe_captures(device: TtDevice) -> Result<u64, TtError> {
    server::run(device, |e, _| e.probe_captures())
}

/// Every unit's NC waits since they were last taken, in tile cycles, zeroed: what B's waits for
/// its writer core are made of (`tt_kernels::session::Session::probe_nc_waits`; counted only by
/// firmware built with the profile feature). Dev-facing.
#[doc(hidden)]
pub fn probe_nc_waits(device: TtDevice) -> Result<Vec<[u32; 8]>, TtError> {
    server::run(device, |e, _| e.probe_nc_waits())
}

/// Detailed timing breakdown of a traced training step.
#[derive(Clone, Debug, Default)]
pub struct StepTiming {
    pub loss: f32,
    pub write_inputs: std::time::Duration,
    pub replay: std::time::Duration,
    pub read_loss: std::time::Duration,
}

/// A captured training step trace with automated in-place parameter writeback.
///
/// Packages forward pass, loss calculation, backward pass, optimizer parameter
/// updates, and parameter buffer writebacks into a single hardware command
/// stream. On replay, the host only streams new batch inputs and calls `Session::replay`.
#[derive(Debug)]
pub struct TracedTrainingStep {
    device: TtDevice,
    id: u64,
    inputs: Vec<BufferId>,
    loss: BufferId,
    _held: Vec<TtTensor>,
}

impl TracedTrainingStep {
    /// Capture a training step.
    ///
    /// `inputs`: batch input tensors resident on the device.
    /// `f`: training step closure returning `(loss, updates)`, where `updates`
    /// contains `(new_param, orig_param)` pairs. An on-device `copy_into`
    /// is recorded for every pair before ending capture, ensuring hardware
    /// replays update the original parameter buffers in-place.
    pub fn capture(
        inputs: &[&TtTensor],
        f: impl FnOnce() -> (TtTensor, Vec<(TtTensor, TtTensor)>),
    ) -> Result<(Self, f32), TtError> {
        let Some(first_in) = inputs.first() else {
            return Err(TtError::Device(
                "a training trace needs at least one input tensor".into(),
            ));
        };
        let device = first_in.device;
        let mut in_ids = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            if inp.device != device {
                return Err(TtError::Device(
                    "all trace inputs must belong to the same device".into(),
                ));
            }
            if !inp.is_storable() || inp.storage_format() != crate::storage::StorageFormat::F32 {
                return Err(TtError::Device(
                    "a trace input must be a storable tensor".into(),
                ));
            }
            let d = inp.to_dram();
            if d.transposed || d.buffer.parent.is_some() {
                return Err(TtError::Device(
                    "a trace input must own its buffer: not a transposed or row view".into(),
                ));
            }
            in_ids.push(d.buffer.id);
            // Each replay writes it.
            server::written_in_place(d.buffer.id);
        }

        server::run(device, |e, _| e.begin_trace())?;
        let mut guard = Capturing::new(device, true);

        let (loss, updates) = f();

        if loss.device != device {
            return Err(TtError::Device(
                "the loss tensor must belong to the trace device".into(),
            ));
        }
        let loss_dram = loss.dram().ok_or_else(|| {
            TtError::Device("the loss tensor must be computed on the device".into())
        })?;
        if loss_dram.transposed {
            return Err(TtError::Device(
                "the loss tensor cannot be transposed".into(),
            ));
        }
        let loss_id = loss_dram.buffer.id;

        // Perform in-place copy_into for every parameter update before ending trace!
        let mut held_params = Vec::with_capacity(updates.len() * 2);
        for (new_param, orig_param) in updates {
            if new_param.device != device || orig_param.device != device {
                return Err(TtError::Device(
                    "parameter update tensors must belong to the trace device".into(),
                ));
            }
            let new_d = new_param.to_dram();
            let orig_d = orig_param.to_dram();
            if new_d.transposed || orig_d.transposed {
                return Err(TtError::Device(
                    "parameter tensors cannot be transposed views for copy_into".into(),
                ));
            }
            let (new_id, orig_id) = (new_d.buffer.id, orig_d.buffer.id);
            server::copy_into(device, new_id, orig_id)?;
            held_params.push(new_param);
            held_params.push(orig_param);
        }

        guard.open = false;
        let id = server::run(device, |e, _| e.end_trace())?;

        // Download step 0 loss scalar
        let first_loss_data = server::run(device, move |e, ids| e.download(ids.get(loss_id)?))?;
        let first_loss = *first_loss_data.first().unwrap_or(&f32::NAN);

        let mut held = Vec::with_capacity(inputs.len() + 1 + held_params.len());
        for inp in inputs {
            held.push((*inp).clone());
        }
        held.push(loss);
        held.extend(held_params);

        Ok((
            TracedTrainingStep {
                device,
                id,
                inputs: in_ids,
                loss: loss_id,
                _held: held,
            },
            first_loss,
        ))
    }

    /// Replay the entire training step with new batch inputs.
    pub fn step(&self, inputs: Vec<server::InputPayload>) -> Result<StepTiming, TtError> {
        if inputs.len() != self.inputs.len() {
            return Err(TtError::Device(format!(
                "expected {} inputs, got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }
        let in_pairs: Vec<(BufferId, server::InputPayload)> =
            self.inputs.iter().copied().zip(inputs).collect();
        let out_pairs = vec![(self.loss, server::OutputKind::F32)];
        let run = server::run_generic_trace(self.device, self.id, in_pairs, out_pairs)?;
        let loss = run
            .outputs
            .first()
            .and_then(|p| p.as_f32())
            .and_then(|v| v.first().copied())
            .unwrap_or(f32::NAN);
        Ok(StepTiming {
            loss,
            write_inputs: run.write,
            replay: run.replay,
            read_loss: run.read,
        })
    }
}

impl Drop for TracedTrainingStep {
    fn drop(&mut self) {
        let id = self.id;
        // A drop never opens a card, and never panics: a detached device has
        // nothing left to release.
        server::run_if_attached(self.device, move |e, _| e.release_trace(id));
    }
}
