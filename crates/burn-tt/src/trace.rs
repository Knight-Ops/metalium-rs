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

use crate::server::{self, BufferId, EngineError};
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
}

impl Drop for Capturing {
    fn drop(&mut self) {
        if self.open {
            // Whatever it captured, it is no trace: ended, and released.
            if let Ok(id) = server::run(self.device, |e, _| e.end_trace()) {
                server::run(self.device, move |e, _| e.release_trace(id));
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
    ) -> Result<(Trace, Vec<f32>), EngineError> {
        let device = input.device;
        if !input.is_stored_f32() {
            return Err(EngineError("a trace's input is an F32 tensor".into()));
        }
        let d = input.to_dram();
        if d.transposed || d.buffer.parent.is_some() {
            return Err(EngineError(
                "a trace's input must own its buffer: not a transposed or row view".into(),
            ));
        }
        let input_id = d.buffer.id;
        server::run(device, |e, _| e.begin_trace())?;
        let mut guard = Capturing { device, open: true };
        let out = f();
        let output = match out.dram() {
            Some(d) if !d.transposed && out.device == device && out.is_stored_f32() => {
                Ok((d.buffer.id, [d.buffer.rows, d.buffer.cols]))
            }
            _ => Err(EngineError(
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
    pub fn run(&self, values: Vec<f32>) -> Result<Vec<f32>, EngineError> {
        self.run_timed(values).map(|r| r.output)
    }

    /// [`Trace::run`], with the time the input's write, the replay and the
    /// output's read each took on the device's side.
    pub fn run_timed(&self, values: Vec<f32>) -> Result<server::TraceRun, EngineError> {
        let (id, input, output) = (self.id, self.input, self.output);
        server::run(self.device, move |e, ids| {
            ids.healthy()?;
            e.run_trace(id, ids.get(input)?, &values, ids.get(output)?)
        })
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        let id = self.id;
        server::run(self.device, move |e, _| e.release_trace(id));
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

impl TracedInference {
    /// Capture inference over arbitrary inputs and outputs.
    pub fn capture(
        inputs: &[&TtTensor],
        f: impl FnOnce() -> Vec<TtTensor>,
    ) -> Result<(Self, Vec<server::OutputPayload>), EngineError> {
        let Some(first_in) = inputs.first() else {
            return Err(EngineError(
                "a trace needs at least one input tensor".into(),
            ));
        };
        let device = first_in.device;
        let mut in_ids = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            if inp.device != device {
                return Err(EngineError(
                    "all trace inputs must belong to the same device".into(),
                ));
            }
            if !inp.is_storable() {
                return Err(EngineError(
                    "a trace input must be a storable tensor".into(),
                ));
            }
            let d = inp.to_dram();
            if d.transposed || d.buffer.parent.is_some() {
                return Err(EngineError(
                    "a trace input must own its buffer: not a transposed or row view".into(),
                ));
            }
            in_ids.push(d.buffer.id);
        }

        server::run(device, |e, _| e.begin_trace())?;
        let mut guard = Capturing { device, open: true };
        let outs = f();
        let mut out_info = Vec::with_capacity(outs.len());
        for out in &outs {
            if out.device != device {
                return Err(EngineError(
                    "all trace outputs must belong to the trace device".into(),
                ));
            }
            let d = out.dram().ok_or_else(|| {
                EngineError("a trace output must be computed on the device".into())
            })?;
            if d.transposed {
                return Err(EngineError(
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

        Ok((
            TracedInference {
                device,
                id,
                inputs: in_ids,
                outputs: out_info,
                _held: held,
            },
            first_outputs,
        ))
    }

    /// The number and shapes of output tensors, `[rows, cols]`.
    pub fn output_shapes(&self) -> Vec<[usize; 2]> {
        self.outputs.iter().map(|&(_, _, dims)| dims).collect()
    }

    /// Replay the trace with new input values.
    pub fn run(
        &self,
        inputs: Vec<server::InputPayload>,
    ) -> Result<Vec<server::OutputPayload>, EngineError> {
        self.run_timed(inputs).map(|r| r.outputs)
    }

    /// Replay the trace with timing breakdown across PCIe write, card replay, and output readback.
    pub fn run_timed(
        &self,
        inputs: Vec<server::InputPayload>,
    ) -> Result<server::GenericTraceRun, EngineError> {
        if inputs.len() != self.inputs.len() {
            return Err(EngineError(format!(
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
}

impl Drop for TracedInference {
    fn drop(&mut self) {
        let id = self.id;
        server::run(self.device, move |e, _| e.release_trace(id));
    }
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
    ) -> Result<(Self, f32), EngineError> {
        let Some(first_in) = inputs.first() else {
            return Err(EngineError(
                "a training trace needs at least one input tensor".into(),
            ));
        };
        let device = first_in.device;
        let mut in_ids = Vec::with_capacity(inputs.len());
        for &inp in inputs {
            if inp.device != device {
                return Err(EngineError(
                    "all trace inputs must belong to the same device".into(),
                ));
            }
            if !inp.is_storable() {
                return Err(EngineError(
                    "a trace input must be a storable tensor".into(),
                ));
            }
            let d = inp.to_dram();
            if d.transposed || d.buffer.parent.is_some() {
                return Err(EngineError(
                    "a trace input must own its buffer: not a transposed or row view".into(),
                ));
            }
            in_ids.push(d.buffer.id);
        }

        server::run(device, |e, _| e.begin_trace())?;
        let mut guard = Capturing { device, open: true };

        let (loss, updates) = f();

        if loss.device != device {
            return Err(EngineError(
                "the loss tensor must belong to the trace device".into(),
            ));
        }
        let loss_dram = loss
            .dram()
            .ok_or_else(|| EngineError("the loss tensor must be computed on the device".into()))?;
        if loss_dram.transposed {
            return Err(EngineError("the loss tensor cannot be transposed".into()));
        }
        let loss_id = loss_dram.buffer.id;

        // Perform in-place copy_into for every parameter update before ending trace!
        let mut held_params = Vec::with_capacity(updates.len() * 2);
        for (new_param, orig_param) in updates {
            if new_param.device != device || orig_param.device != device {
                return Err(EngineError(
                    "parameter update tensors must belong to the trace device".into(),
                ));
            }
            let new_d = new_param.to_dram();
            let orig_d = orig_param.to_dram();
            if new_d.transposed || orig_d.transposed {
                return Err(EngineError(
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
    pub fn step(&self, inputs: Vec<server::InputPayload>) -> Result<StepTiming, EngineError> {
        if inputs.len() != self.inputs.len() {
            return Err(EngineError(format!(
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
        server::run(self.device, move |e, _| e.release_trace(id));
    }
}
