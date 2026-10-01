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
            if let Ok(id) = server::run(self.device, |e| e.end_trace()) {
                server::run(self.device, move |e| e.release_trace(id));
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
        server::run(device, |e| e.begin_trace())?;
        let mut guard = Capturing { device, open: true };
        let out = f();
        let output = match out.dram() {
            Some(d) if !d.transposed && out.device == device => {
                Ok((d.buffer.id, [d.buffer.rows, d.buffer.cols]))
            }
            _ => Err(EngineError(
                "a trace's output must be computed on its device, untransposed".into(),
            )),
        };
        guard.open = false;
        let ended = server::run(device, |e| e.end_trace());
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
        let first = server::run(device, move |e| e.download(output));
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
        let (id, input, output) = (self.id, self.input, self.output);
        server::run(self.device, move |e| {
            e.run_trace(id, input, &values, output)
        })
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        let id = self.id;
        server::run(self.device, move |e| e.release_trace(id));
    }
}
