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
//! [`TracedGenerator`] captures the step and one chunk per bucket size once, as traces over one
//! shared cache on the card (`burn_tt::SharedStateTraces`). It then serves any prompt:
//! 1. a prompt is fed as chunks: the smallest bucket that covers it, or chunks of the largest
//!    then the smallest bucket that covers the rest, the last chunk padded;
//! 2. each later token is a replay of the step.
//!
//! Token ids go in and logits come out: the embedding lookup runs inside the trace, the cache
//! never leaves the card, and the host builds no op after the captures. [`generate_fixed`] runs
//! the same schedule eagerly, the reference a traced generation equals bit for bit.

use std::time::{Duration, Instant};

use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor, TensorData, TensorPrimitive};
use burn_tt::{
    InputPayload, OutputPayload, SharedStateTraces, TtBackend, TtDevice, TtError, TtTensor,
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

    /// One token (`[1, 1]`) at `position` (`[1]`): its logits `[1, vocab]` and the new state.
    fn step(
        &self,
        token: Tensor<B, 2, Int>,
        position: Tensor<B, 1, Int>,
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

/// Ascending, distinct bucket sizes, each a chunk the cache can take.
fn check_buckets(buckets: &[usize], max_len: usize) -> Result<Vec<usize>, GenerateError> {
    let mut b = buckets.to_vec();
    b.sort_unstable();
    b.dedup();
    if b.is_empty() {
        return Err(GenerateError::Refused("no chunk sizes".into()));
    }
    if b[0] == 0 || *b.last().unwrap() > max_len {
        return Err(GenerateError::Refused(format!(
            "chunk sizes {b:?} for a cache of {max_len} positions"
        )));
    }
    Ok(b)
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
/// [`schedule`] says, then a step a token, over a cache of `max_len` positions. What a
/// [`TracedGenerator`] with the same buckets computes, op for op.
pub fn generate_fixed<B: Backend, M: FixedShapeDecoder<B>>(
    model: &M,
    prompt: &[i32],
    steps: usize,
    max_len: usize,
    buckets: &[usize],
    device: &B::Device,
) -> Result<Generation, GenerateError> {
    let buckets = check_buckets(buckets, max_len)?;
    check_request(prompt, steps, max_len)?;
    let mut cache = model.new_cache(max_len, device);
    let mut out = Generation::new(steps);
    out.step_with(|| {
        let mut logits = None;
        for chunk in schedule(prompt.len(), &buckets) {
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
        let (l, state) = model.step(
            ids(&[token], device),
            index(prompt.len() + step, device),
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
    /// The one-token step.
    Step,
    /// A chunk of this many tokens.
    Chunk(usize),
}

fn float<const D: usize>(t: Tensor<TtBackend, D>) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(t) => t,
        TensorPrimitive::QFloat(_) => unreachable!("a quantised tensor is not a step's output"),
    }
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
/// Building one captures a trace per chunk size and one for the step; they share the program
/// cache (`burn_tt::SharedStateTraces`), so the sizes a model can keep together are bounded by it
/// (`footprint`). A capture runs its step once and writes the cache, which a generation never
/// relies on: every position it reads was written by its own prompt or steps first.
pub struct TracedGenerator<'m, M: FixedShapeDecoder<TtBackend>> {
    model: &'m M,
    max_len: usize,
    buckets: Vec<usize>,
    /// The caches the traces read and write: one, or two for
    /// [`TracedGenerator::with_private_prefill`].
    caches: Vec<M::Cache>,
    sets: Vec<SharedStateTraces<Key>>,
    /// Which of `sets` holds the chunks.
    prefill: usize,
}

impl<'m, M: FixedShapeDecoder<TtBackend>> TracedGenerator<'m, M> {
    /// Capture `model`'s step and a chunk of each size in `buckets`, over a cache of `max_len`
    /// positions on `device`. The cache's state must be F32: a packed one would be unpacked by
    /// each replay's write-back.
    pub fn new(
        model: &'m M,
        max_len: usize,
        buckets: &[usize],
        device: &TtDevice,
    ) -> Result<Self, GenerateError> {
        Self::build(model, max_len, buckets, device, false)
    }

    /// [`Self::new`] with the chunks capturing over a second cache that the step never reads: a
    /// generation that is wrong by construction, the negative control of the shared cache.
    #[doc(hidden)]
    pub fn with_private_prefill(
        model: &'m M,
        max_len: usize,
        buckets: &[usize],
        device: &TtDevice,
    ) -> Result<Self, GenerateError> {
        Self::build(model, max_len, buckets, device, true)
    }

    fn build(
        model: &'m M,
        max_len: usize,
        buckets: &[usize],
        device: &TtDevice,
        private: bool,
    ) -> Result<Self, GenerateError> {
        let buckets = check_buckets(buckets, max_len)?;
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
            // An eager step and chunk of each size first, so what the model keeps between calls
            // (a weight transposed once for its product, a zero row) is in place before the
            // captures, which then read it rather than make it.
            let (l, state) = model.step(ids(&[0], device), index(0, device), cache);
            cache.set_state(state);
            let _ = read(l);
            for &c in &buckets {
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

        let token = ids::<TtBackend>(&[0], device);
        let position = index::<TtBackend>(0, device);
        let (tp, pp) = (
            token.clone().into_primitive(),
            position.clone().into_primitive(),
        );
        tp.ensure_resident();
        pp.ensure_resident();
        let cache = &caches[0];
        sets[0].capture(Key::Step, &[&tp, &pp], || {
            let (y, state) = model.step(token, position, cache);
            (vec![float(y)], state.into_iter().map(float).collect())
        })?;

        let prefill = caches.len() - 1;
        let cache = &caches[prefill];
        for &c in &buckets {
            let tokens = ids::<TtBackend>(&vec![0; c], device);
            let start = index::<TtBackend>(0, device);
            let last = index::<TtBackend>(0, device);
            let (tp, sp, lp) = (
                tokens.clone().into_primitive(),
                start.clone().into_primitive(),
                last.clone().into_primitive(),
            );
            for p in [&tp, &sp, &lp] {
                p.ensure_resident();
            }
            sets[prefill].capture(Key::Chunk(c), &[&tp, &sp, &lp], || {
                let (y, state) = model.chunk(tokens, start, last, cache);
                (vec![float(y)], state.into_iter().map(float).collect())
            })?;
        }
        Ok(Self {
            model,
            max_len,
            buckets,
            caches,
            sets,
            prefill,
        })
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// The chunk sizes captured, ascending.
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
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

    /// Feed `prompt` from position 0 (as [`schedule`] says) and return the logits after its last
    /// token. Whatever the cache held is overwritten as it is reached.
    pub fn prefill(&self, prompt: &[i32]) -> Result<Vec<f32>, GenerateError> {
        check_request(prompt, 1, self.max_len)?;
        let mut out = None;
        for chunk in schedule(prompt.len(), &self.buckets) {
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

    /// One token at `position` (every earlier position already fed): the logits after it.
    pub fn decode(&self, token: i32, position: usize) -> Result<Vec<f32>, GenerateError> {
        if position >= self.max_len {
            return Err(GenerateError::Capacity {
                needed: position + 1,
                max_len: self.max_len,
            });
        }
        let trace = self.sets[0].get(Key::Step).expect("the step's trace");
        logits(trace.run(vec![
            InputPayload::Bits(vec![token as u32]),
            InputPayload::Bits(vec![position as u32]),
        ])?)
    }

    /// Greedy decoding: `steps` tokens after `prompt`, as [`generate_fixed`] with the same chunk
    /// sizes computes them (and as the model's growing-cache generation, for a model whose fixed
    /// steps equal it).
    pub fn generate(&self, prompt: &[i32], steps: usize) -> Result<Generation, GenerateError> {
        check_request(prompt, steps, self.max_len)?;
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
            check_buckets(&[], 8),
            Err(GenerateError::Refused(_))
        ));
        assert!(matches!(
            check_buckets(&[4, 16], 8),
            Err(GenerateError::Refused(_))
        ));
        assert_eq!(check_buckets(&[8, 4, 8], 8).unwrap(), [4, 8]);
    }
}
