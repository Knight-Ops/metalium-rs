//! Generating text with any decoder whose steps have fixed shapes, eagerly or as traces.
//!
//! A decoder that implements [`FixedShapeDecoder`] gives two steps over a cache it owns, both
//! functional (the cache's state in, the new state out):
//! - a **chunk**: `c` prompt tokens from a start position, the logits of one of its rows;
//! - a **step**: one token at a position.
//!
//! The positions are tensors, so the shapes depend only on `c` and on the cache's length. The trait
//! assumes nothing about the architecture (rotary or learned positions, grouped-query or multi-head
//! attention, which norm) beyond a state of rank-4 tensors.
//!
//! [`TracedGenerator`] captures, once, as traces over one shared cache on the card
//! (`burn_tt::SharedStateTraces`), what [`Shapes`] names: a chunk of each size, and a step for
//! each attended length. It then serves any prompt:
//! 1. a prompt is fed as chunks: the smallest size that covers it, or chunks of the largest
//!    then the smallest size that covers the rest, the last chunk padded;
//! 2. each later token is a replay of the step attending to the fewest positions that include
//!    it, so a step costs the context so far, not the cache's length.
//!
//! Token ids go in and logits come out: the embedding lookup runs inside the trace, the cache
//! never leaves the card, and the host builds no op after the captures. [`generate_fixed`] runs
//! the same schedule eagerly, the reference a traced generation equals bit for bit.

use std::time::{Duration, Instant};

use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{
    InputPayload, OutputPayload, SharedStateTraces, Tt, TtBackend, TtDevice, TtError, TtTensor,
};

/// The logits a run of next-token predictions produced.
#[derive(Debug)]
pub struct Generation {
    /// Wall time of each forward pass, the prompt's first, including reading its logits back.
    pub times: Vec<Duration>,
    /// The tokens chosen (greedy), or fed (teacher-forced), after the prompt.
    pub tokens: Vec<i32>,
    /// `logits[i]` is the prediction for the position after `prompt + tokens[..i]`, `[vocab]`.
    pub logits: Vec<Vec<f32>>,
}

impl Generation {
    pub(crate) fn new(steps: usize) -> Self {
        Self {
            times: Vec::new(),
            tokens: Vec::with_capacity(steps),
            logits: Vec::with_capacity(steps),
        }
    }

    /// Run one forward pass, read its logits back and record the wall time of both.
    pub(crate) fn step<B: Backend>(&mut self, forward: impl FnOnce() -> Tensor<B, 2>) {
        let _ = self.step_with(|| Ok::<_, std::convert::Infallible>(read(forward())));
    }

    /// [`Self::step`] for a pass that reads its own logits.
    fn step_with<E>(&mut self, forward: impl FnOnce() -> Result<Vec<f32>, E>) -> Result<(), E> {
        let t0 = Instant::now();
        self.logits.push(forward()?);
        self.times.push(t0.elapsed());
        if std::env::var_os("TT_MODELS_TRACE").is_some() {
            eprintln!("forward {}: {:.2?}", self.times.len() - 1, t0.elapsed());
        }
        Ok(())
    }

    /// Greedy decoding: `steps` tokens, each chosen from the last logits, `forward(token)` run
    /// for each but the last (the last chosen token is not fed back).
    fn decode<E>(
        &mut self,
        steps: usize,
        mut forward: impl FnMut(i32, usize) -> Result<Vec<f32>, E>,
    ) -> Result<(), E> {
        for step in 0..steps {
            let next = argmax(self.logits.last().expect("one prediction per step")) as i32;
            self.tokens.push(next);
            if step + 1 < steps {
                self.step_with(|| forward(next, step))?;
            }
        }
        Ok(())
    }
}

pub(crate) fn argmax(v: &[f32]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
}

pub(crate) fn read<B: Backend>(logits: Tensor<B, 2>) -> Vec<f32> {
    // BF16 logits widen exactly; F32 ones are unchanged.
    logits
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("F32 logits")
}

/// A decoder's cache: state tensors that a step reads and returns anew.
pub trait FixedShapeCache<B: Backend> {
    /// Positions the cache holds.
    fn max_len(&self) -> usize;
    /// The tensors a step reads and whose new values it returns, in a fixed order.
    fn state(&self) -> Vec<Tensor<B, 4>>;
    /// Take a step's new state (an eager step's write-back).
    fn set_state(&mut self, state: Vec<Tensor<B, 4>>);
}

/// A decoder whose steps have shapes that depend only on how many tokens they take: see the
/// module documentation.
pub trait FixedShapeDecoder<B: Backend> {
    type Cache: FixedShapeCache<B>;

    /// An empty cache of `max_len` positions, in the model's dtype.
    fn new_cache(&self, max_len: usize, device: &B::Device) -> Self::Cache;

    /// `tokens` (`[1, c]` ids) from position `start` (`[1]`) on: the logits `[1, vocab]` of chunk
    /// row `last` (`[1]`), and the cache's new state with the chunk's positions written. Rows of a
    /// padded chunk past the prompt may write positions past it (a later step overwrites them
    /// before anything reads them) and positions at or past `max_len` (which are not written).
    fn chunk(
        &self,
        tokens: Tensor<B, 2, Int>,
        start: Tensor<B, 1, Int>,
        last: Tensor<B, 1, Int>,
        cache: &Self::Cache,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>);

    /// One token (`[1, 1]`) at `position` (`[1]`, below `len`), attending only to the cache's
    /// first `len` positions: its logits `[1, vocab]` and the new state. A step's cost should
    /// follow `len`, not the cache's length (`FixedCache::step_write_within`); for a position
    /// below `len`, the logits must not depend on `len`.
    fn step(
        &self,
        token: Tensor<B, 2, Int>,
        position: Tensor<B, 1, Int>,
        len: usize,
        cache: &Self::Cache,
    ) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>);
}

/// Why a generation was refused or failed.
#[derive(Debug)]
pub enum GenerateError {
    /// The request does not fit the cache.
    Capacity { needed: usize, max_len: usize },
    /// A request this driver does not take, named.
    Refused(String),
    /// The device failed (a capture that did not fit the program cache, say: an attachment's
    /// error is sticky, so reattach before retrying).
    Device(TtError),
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::Capacity { needed, max_len } => write!(
                f,
                "the generation needs {needed} cache positions, the cache holds {max_len}"
            ),
            GenerateError::Refused(why) => write!(f, "refused: {why}"),
            GenerateError::Device(e) => write!(f, "device: {e}"),
        }
    }
}

impl std::error::Error for GenerateError {}

impl From<TtError> for GenerateError {
    fn from(e: TtError) -> Self {
        GenerateError::Device(e)
    }
}

/// One chunk of a prompt: `n` real tokens from `start`, fed as a chunk of `c >= n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: usize,
    pub n: usize,
    pub c: usize,
}

/// How a prompt of `len` tokens is fed with chunks of the sizes `buckets` (ascending): the
/// smallest that covers it; past the largest, chunks of the largest, then the smallest that
/// covers the rest.
pub fn schedule(len: usize, buckets: &[usize]) -> Vec<Chunk> {
    let largest = *buckets.last().expect("a bucket");
    let mut out = Vec::new();
    let mut start = 0;
    while start < len {
        let rest = len - start;
        let c = if rest >= largest {
            largest
        } else {
            *buckets
                .iter()
                .find(|&&b| b >= rest)
                .expect("the largest covers it")
        };
        let n = rest.min(c);
        out.push(Chunk { start, n, c });
        start += n;
    }
    out
}

/// The shapes a generation's steps come in: the cache's length, the chunk sizes a prompt is fed
/// in ([`schedule`]), and the lengths a decode step attends to (a step at `position` attends to
/// the smallest length above it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shapes {
    pub max_len: usize,
    pub chunks: Vec<usize>,
    pub steps: Vec<usize>,
}

impl Shapes {
    /// Chunks of `chunks` over a cache of `max_len` positions, decode steps attending to 64, 128,
    /// 256, ... positions below `max_len`, then all of them.
    pub fn new(max_len: usize, chunks: &[usize]) -> Self {
        let steps = std::iter::successors(Some(64usize), |l| l.checked_mul(2))
            .take_while(|&l| l < max_len)
            .chain([max_len])
            .collect();
        Self {
            max_len,
            chunks: chunks.to_vec(),
            steps,
        }
    }

    /// The same, with decode steps attending to `steps` positions (the cache's length is added).
    pub fn with_steps(mut self, steps: &[usize]) -> Self {
        self.steps = steps.to_vec();
        self.steps.push(self.max_len);
        self
    }

    /// Sorted and checked: every chunk and step length a shape the cache can take.
    fn checked(&self) -> Result<Shapes, GenerateError> {
        let sorted = |v: &[usize]| {
            let mut v = v.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        };
        let (chunks, steps) = (sorted(&self.chunks), sorted(&self.steps));
        let max_len = self.max_len;
        if chunks.is_empty() {
            return Err(GenerateError::Refused("no chunk sizes".into()));
        }
        let fits = |v: &[usize]| v[0] > 0 && *v.last().unwrap() <= max_len;
        if !fits(&chunks) {
            return Err(GenerateError::Refused(format!(
                "chunk sizes {chunks:?} for a cache of {max_len} positions"
            )));
        }
        if steps.last() != Some(&max_len) || !fits(&steps) {
            return Err(GenerateError::Refused(format!(
                "decode steps attending to {steps:?} positions of {max_len}: the last must be the \
                 cache's length"
            )));
        }
        Ok(Shapes {
            max_len,
            chunks,
            steps,
        })
    }

    /// The length a decode step at `position` attends to: the smallest above it.
    pub fn step_len(&self, position: usize) -> usize {
        *self
            .steps
            .iter()
            .find(|&&l| l > position)
            .unwrap_or(&self.max_len)
    }
}

/// Positions a generation of `steps` tokens after `prompt` writes: the prompt's, and one for each
/// fed-back token.
fn check_request(prompt: &[i32], steps: usize, max_len: usize) -> Result<(), GenerateError> {
    if prompt.is_empty() || steps == 0 {
        return Err(GenerateError::Refused(format!(
            "a prompt of {} tokens and {steps} steps",
            prompt.len()
        )));
    }
    let needed = prompt.len() + steps - 1;
    if needed > max_len {
        return Err(GenerateError::Capacity { needed, max_len });
    }
    Ok(())
}

fn ids<B: Backend>(tokens: &[i32], device: &B::Device) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(tokens.to_vec(), [1, tokens.len()]), device)
}

fn index<B: Backend>(i: usize, device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::from_data(TensorData::new(vec![i as i32], [1]), device)
}

/// A chunk's tokens, padded with id 0.
fn padded(prompt: &[i32], chunk: Chunk) -> Vec<i32> {
    let mut t = prompt[chunk.start..chunk.start + chunk.n].to_vec();
    t.resize(chunk.c, 0);
    t
}

/// Greedy decoding with `model`'s fixed-shape steps, eagerly: the prompt fed as
/// [`schedule`] says, then a step a token, in `shapes`. What a [`TracedGenerator`] with the same
/// shapes computes, op for op.
pub fn generate_fixed<B: Backend, M: FixedShapeDecoder<B>>(
    model: &M,
    prompt: &[i32],
    steps: usize,
    shapes: &Shapes,
    device: &B::Device,
) -> Result<Generation, GenerateError> {
    let shapes = shapes.checked()?;
    check_request(prompt, steps, shapes.max_len)?;
    let mut cache = model.new_cache(shapes.max_len, device);
    let mut out = Generation::new(steps);
    out.step_with(|| {
        let mut logits = None;
        for chunk in schedule(prompt.len(), &shapes.chunks) {
            let (l, state) = model.chunk(
                ids(&padded(prompt, chunk), device),
                index(chunk.start, device),
                index(chunk.n - 1, device),
                &cache,
            );
            cache.set_state(state);
            logits = Some(l);
        }
        Ok::<_, GenerateError>(read(logits.expect("a chunk")))
    })?;
    out.decode(steps, |token, step| {
        let position = prompt.len() + step;
        let (l, state) = model.step(
            ids(&[token], device),
            index(position, device),
            shapes.step_len(position),
            &cache,
        );
        cache.set_state(state);
        Ok::<_, GenerateError>(read(l))
    })?;
    Ok(out)
}

/// Which trace of a [`TracedGenerator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Key {
    /// The one-token step attending to this many positions.
    Step(usize),
    /// A chunk of this many tokens.
    Chunk(usize),
}

/// A backend whose steps a [`TracedGenerator`] captures: `TtBackend`, and `burn_tt::Tt` (burn-fusion
/// over it), whose tensors resolve to the device's own. Resolving inside a capture runs what the
/// fusion stream holds, so the trace records it.
pub trait TracedBackend: Backend<Device = TtDevice> {
    /// The device tensor of a float primitive.
    fn float_native(t: Self::FloatTensorPrimitive) -> TtTensor;
    /// The device tensor of an int primitive.
    fn int_native(t: Self::IntTensorPrimitive) -> TtTensor;
}

impl TracedBackend for TtBackend {
    fn float_native(t: TtTensor) -> TtTensor {
        t
    }
    fn int_native(t: TtTensor) -> TtTensor {
        t
    }
}

impl TracedBackend for Tt {
    fn float_native(t: Self::FloatTensorPrimitive) -> TtTensor {
        burn_tt::resolve_float_tensor(&t)
    }
    fn int_native(t: Self::IntTensorPrimitive) -> TtTensor {
        burn_tt::resolve_int_tensor(&t)
    }
}

fn float<B: TracedBackend, const D: usize>(t: Tensor<B, D>) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(t) => B::float_native(t),
        TensorPrimitive::QFloat(_) => unreachable!("a quantised tensor is not a step's output"),
    }
}

/// A trace input made on `B`, resident, as the device tensor the trace reads.
fn input<B: TracedBackend, const D: usize>(t: &Tensor<B, D, Int>) -> TtTensor {
    let t = B::int_native(t.clone().into_primitive());
    t.ensure_resident();
    t
}

fn logits(out: Vec<OutputPayload>) -> Result<Vec<f32>, GenerateError> {
    match out.into_iter().next() {
        Some(OutputPayload::F32(v)) => Ok(v),
        other => Err(GenerateError::Refused(format!(
            "a trace's logits came back as {other:?}"
        ))),
    }
}

/// A decoder's step and its prefill chunks, captured once as traces over one cache on the card,
/// then replayed for any prompt and any number of tokens: see the module documentation.
///
/// Building one captures a trace per chunk size and one per step length; they share the program
/// cache (`burn_tt::SharedStateTraces`), so the sizes a model can keep together are bounded by it
/// (`footprint`). A capture runs its step once and writes the cache, which a generation never
/// relies on: every position it reads was written by its own prompt or steps first.
///
/// `B` is the backend the model runs on while it is captured: `burn_tt::Tt` (the default, whose
/// fusion stream each capture runs to its end before the capture closes, and whose fusers make the
/// step `TtBackend`'s bit for bit or faster), or `TtBackend`.
pub struct TracedGenerator<'m, M: FixedShapeDecoder<B>, B: TracedBackend = Tt> {
    model: &'m M,
    shapes: Shapes,
    /// The caches the traces read and write: one, or two for
    /// [`TracedGenerator::with_private_prefill`].
    caches: Vec<M::Cache>,
    sets: Vec<SharedStateTraces<Key>>,
    /// Which of `sets` holds the chunks.
    prefill: usize,
}

impl<'m, B: TracedBackend, M: FixedShapeDecoder<B>> TracedGenerator<'m, M, B> {
    /// Capture `model`'s step at each length and a chunk of each size `shapes` names, over one
    /// cache on `device`. The cache's state must be F32: a packed one would be unpacked by each
    /// replay's write-back.
    pub fn new(model: &'m M, shapes: &Shapes, device: &TtDevice) -> Result<Self, GenerateError> {
        Self::build(model, shapes, device, false)
    }

    /// [`Self::new`] with the chunks capturing over a second cache that the step never reads: a
    /// generation that is wrong by construction, the negative control of the shared cache.
    #[doc(hidden)]
    pub fn with_private_prefill(
        model: &'m M,
        shapes: &Shapes,
        device: &TtDevice,
    ) -> Result<Self, GenerateError> {
        Self::build(model, shapes, device, true)
    }

    fn build(
        model: &'m M,
        shapes: &Shapes,
        device: &TtDevice,
        private: bool,
    ) -> Result<Self, GenerateError> {
        let shapes = shapes.checked()?;
        let max_len = shapes.max_len;
        let mut caches = vec![model.new_cache(max_len, device)];
        if private {
            caches.push(model.new_cache(max_len, device));
        }
        for cache in &mut caches {
            if let Some(t) = cache.state().iter().find(|t| t.dtype() != DType::F32) {
                return Err(GenerateError::Refused(format!(
                    "a cache in {:?}: traced generation keeps an F32 cache",
                    t.dtype()
                )));
            }
            // An eager step of each length and chunk of each size first, so what the model keeps
            // between calls (a weight transposed once for its product, a zero row) is in place
            // before the captures, which then read it rather than make it.
            for &len in &shapes.steps {
                let (l, state) = model.step(ids(&[0], device), index(0, device), len, cache);
                cache.set_state(state);
                let _ = read(l);
            }
            for &c in &shapes.chunks {
                let (l, state) = model.chunk(
                    ids(&vec![0; c], device),
                    index(0, device),
                    index(0, device),
                    cache,
                );
                cache.set_state(state);
                let _ = read(l);
            }
        }
        let mut sets = Vec::with_capacity(caches.len());
        for cache in &caches {
            sets.push(SharedStateTraces::new(
                cache.state().into_iter().map(float).collect(),
            )?);
        }

        let cache = &caches[0];
        for &len in &shapes.steps {
            let token = ids::<B>(&[0], device);
            let position = index::<B>(0, device);
            let (tp, pp) = (input(&token), input(&position));
            sets[0].capture(Key::Step(len), &[&tp, &pp], || {
                let (y, state) = model.step(token, position, len, cache);
                (vec![float(y)], state.into_iter().map(float).collect())
            })?;
        }

        let prefill = caches.len() - 1;
        let cache = &caches[prefill];
        for &c in &shapes.chunks {
            let tokens = ids::<B>(&vec![0; c], device);
            let start = index::<B>(0, device);
            let last = index::<B>(0, device);
            let (tp, sp, lp) = (input(&tokens), input(&start), input(&last));
            sets[prefill].capture(Key::Chunk(c), &[&tp, &sp, &lp], || {
                let (y, state) = model.chunk(tokens, start, last, cache);
                (vec![float(y)], state.into_iter().map(float).collect())
            })?;
        }
        Ok(Self {
            model,
            shapes,
            caches,
            sets,
            prefill,
        })
    }

    pub fn max_len(&self) -> usize {
        self.shapes.max_len
    }

    /// The chunk sizes and step lengths captured, ascending.
    pub fn shapes(&self) -> &Shapes {
        &self.shapes
    }

    pub fn model(&self) -> &M {
        self.model
    }

    /// The cache the traces share (the step's, for [`Self::with_private_prefill`]).
    pub fn cache(&self) -> &M::Cache {
        &self.caches[0]
    }

    /// What the traces cost each unit: the program-cache bytes they hold together (the bound on
    /// which chunk sizes fit beside each other), and their streams' GDDR bytes.
    pub fn footprint(&self) -> Result<Vec<burn_tt::SetFootprint<String>>, GenerateError> {
        self.sets
            .iter()
            .map(|s| {
                let f = s.footprint()?;
                Ok(burn_tt::SetFootprint {
                    traces: f
                        .traces
                        .into_iter()
                        .map(|(k, u)| (format!("{k:?}"), u))
                        .collect(),
                })
            })
            .collect()
    }

    /// Each trace's ops as captured (`burn_tt::TracedInference::ops`), named as `footprint` names
    /// the traces (`Step(64)`, `Chunk(32)`): with their GDDR bytes when the capture counted them
    /// (the profile mode, or `burn_tt::set_count_trace_bytes`).
    pub fn trace_ops(&self) -> Result<Vec<(String, Vec<burn_tt::OpRecord>)>, GenerateError> {
        let mut out = Vec::new();
        for set in &self.sets {
            for key in set.keys() {
                if let Some(t) = set.get(key) {
                    out.push((format!("{key:?}"), t.ops()?));
                }
            }
        }
        Ok(out)
    }

    /// Feed `prompt` from position 0 (as [`schedule`] says) and return the logits after its last
    /// token. Whatever the cache held is overwritten as it is reached.
    pub fn prefill(&self, prompt: &[i32]) -> Result<Vec<f32>, GenerateError> {
        check_request(prompt, 1, self.shapes.max_len)?;
        let mut out = None;
        for chunk in schedule(prompt.len(), &self.shapes.chunks) {
            let trace = self.sets[self.prefill]
                .get(Key::Chunk(chunk.c))
                .expect("a trace for every bucket");
            let tokens = padded(prompt, chunk)
                .into_iter()
                .map(|t| t as u32)
                .collect();
            out = Some(trace.run(vec![
                InputPayload::Bits(tokens),
                InputPayload::Bits(vec![chunk.start as u32]),
                InputPayload::Bits(vec![chunk.n as u32 - 1]),
            ])?);
        }
        logits(out.expect("a chunk"))
    }

    /// One token at `position` (every earlier position already fed): the logits after it, from
    /// the step attending to the fewest positions that include it ([`Shapes::step_len`]).
    pub fn decode(&self, token: i32, position: usize) -> Result<Vec<f32>, GenerateError> {
        if position >= self.shapes.max_len {
            return Err(GenerateError::Capacity {
                needed: position + 1,
                max_len: self.shapes.max_len,
            });
        }
        self.decode_within(token, position, self.shapes.step_len(position))
    }

    /// [`Self::decode`] by the step attending to `len` positions, whatever the position: for a
    /// position at or past `len` the token does not see itself, a wrong result by construction
    /// (the negative control of the step lengths).
    #[doc(hidden)]
    pub fn decode_within(
        &self,
        token: i32,
        position: usize,
        len: usize,
    ) -> Result<Vec<f32>, GenerateError> {
        let trace = self.sets[0]
            .get(Key::Step(len))
            .ok_or_else(|| GenerateError::Refused(format!("no step attending to {len}")))?;
        logits(trace.run(vec![
            InputPayload::Bits(vec![token as u32]),
            InputPayload::Bits(vec![position as u32]),
        ])?)
    }

    /// Greedy decoding: `steps` tokens after `prompt`, as [`generate_fixed`] with the same chunk
    /// sizes computes them (and as the model's growing-cache generation, for a model whose fixed
    /// steps equal it).
    pub fn generate(&self, prompt: &[i32], steps: usize) -> Result<Generation, GenerateError> {
        check_request(prompt, steps, self.shapes.max_len)?;
        let mut out = Generation::new(steps);
        out.step_with(|| self.prefill(prompt))?;
        out.decode(steps, |token, step| self.decode(token, prompt.len() + step))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_is_fed_by_the_smallest_chunk_that_covers_it() {
        let c = |start, n, c| Chunk { start, n, c };
        assert_eq!(schedule(1, &[4, 8]), [c(0, 1, 4)]);
        assert_eq!(schedule(4, &[4, 8]), [c(0, 4, 4)]);
        assert_eq!(schedule(5, &[4, 8]), [c(0, 5, 8)]);
        assert_eq!(schedule(8, &[4, 8]), [c(0, 8, 8)]);
        assert_eq!(schedule(9, &[4, 8]), [c(0, 8, 8), c(8, 1, 4)]);
        assert_eq!(schedule(13, &[4, 8]), [c(0, 8, 8), c(8, 5, 8)]);
        assert_eq!(schedule(16, &[4, 8]), [c(0, 8, 8), c(8, 8, 8)]);
        assert_eq!(schedule(300, &[64, 256]), [c(0, 256, 256), c(256, 44, 64)]);
    }

    #[test]
    fn requests_that_do_not_fit_are_refused_by_name() {
        assert!(matches!(
            check_request(&[1, 2, 3], 6, 7),
            Err(GenerateError::Capacity {
                needed: 8,
                max_len: 7
            })
        ));
        assert!(check_request(&[1, 2, 3], 5, 7).is_ok());
        assert!(matches!(
            check_request(&[], 1, 7),
            Err(GenerateError::Refused(_))
        ));
        assert!(matches!(
            Shapes::new(8, &[]).checked(),
            Err(GenerateError::Refused(_))
        ));
        assert!(matches!(
            Shapes::new(8, &[4, 16]).checked(),
            Err(GenerateError::Refused(_))
        ));
        assert_eq!(Shapes::new(8, &[8, 4, 8]).checked().unwrap().chunks, [4, 8]);
        let shapes = Shapes::new(512, &[64]);
        assert_eq!(shapes.steps, [64, 128, 256, 512]);
        assert_eq!(
            [0, 63, 64, 300, 511].map(|p| shapes.step_len(p)),
            [64, 64, 128, 512, 512]
        );
        assert_eq!(Shapes::new(16, &[4]).steps, [16]);
        assert_eq!(Shapes::new(64, &[4]).with_steps(&[32]).steps, [32, 64]);
        assert!(matches!(
            Shapes {
                steps: vec![32],
                ..Shapes::new(64, &[4])
            }
            .checked(),
            Err(GenerateError::Refused(_))
        ));
    }
}
