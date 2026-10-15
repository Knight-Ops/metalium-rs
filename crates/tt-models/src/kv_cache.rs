//! A key/value cache whose steps have the same shapes at every position, so a step can be traced
//! once and replayed: any decoder that keeps per-layer keys and values `[1, kv_heads, max_len,
//! head_dim]` can use it.
//!
//! A step's position (or a chunk's start) is a tensor. Its keys and values land through
//! `mask_where` where a mask built from that tensor is set, and attention reads all `max_len`
//! positions with those later than each query masked. A masked position adds an exact zero to a
//! softmax's sum and to `P V`, so the step's arithmetic on the positions it attends to is a
//! growing cache's (`step185_fixed_decode`).
//!
//! The cache is functional: a step reads [`FixedCache::state`] and returns the new state rather
//! than writing it, so traces of several steps can share one cache on the card
//! (`burn_tt::SharedStateTraces`) and an eager caller writes it back with
//! [`FixedCache::set_state`].
//!
//! Rows past what a step attends to are never read before a step at their position overwrites
//! them, so a chunk's padding rows (and whatever a trace capture's own run wrote) need no mask
//! of their own.

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, DType, Int, Tensor};

/// Every layer's keys and values, and the positions `0..max_len` a step's position is compared
/// with.
#[derive(Clone, Debug)]
pub struct FixedCache<B: Backend> {
    /// `[k0, v0, k1, v1, ...]`, each `[1, kv_heads, max_len, head_dim]`.
    state: Vec<Tensor<B, 4>>,
    max_len: usize,
    positions: Tensor<B, 1, Int>,
}

/// How a step's keys and values reach a layer's cache, and which positions each query attends to.
#[derive(Clone, Debug)]
pub enum FixedWrite<B: Backend> {
    /// One token at the position set in `here` (the cache's shape: made once a step, not once a
    /// layer); the query attends to the first `len` positions but those set in `later`
    /// (`[1, 1, 1, len]`).
    Masked {
        here: Tensor<B, 4, Bool>,
        later: Tensor<B, 4, Bool>,
        len: usize,
    },
    /// A chunk of `c` tokens at the positions set in `here` (the cache's shape, a run from the
    /// chunk's start): cache row `j` of head `h` takes the chunk's row `index[h max_len + j]` of
    /// its `[kv_heads c, head_dim]` keys and values; query `i` attends to every position but those
    /// set in row `i` of `later` (`[1, 1, c, max_len]`).
    Chunk {
        here: Tensor<B, 4, Bool>,
        later: Tensor<B, 4, Bool>,
        index: Tensor<B, 1, Int>,
    },
}

impl<B: Backend> FixedCache<B> {
    /// An empty cache of `layers` layers, `max_len` positions.
    pub fn zeros(
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        max_len: usize,
        dtype: DType,
        device: &B::Device,
    ) -> Self {
        assert!(layers >= 1, "a cache of no layers");
        assert!(
            max_len >= kv_heads,
            "a cache of {max_len} positions for {kv_heads} key/value heads"
        );
        let shape = [1, kv_heads, max_len, head_dim];
        Self {
            state: (0..2 * layers)
                .map(|_| Tensor::zeros(shape, (device, dtype)))
                .collect(),
            max_len,
            positions: Tensor::arange(0..max_len as i64, device),
        }
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn layers(&self) -> usize {
        self.state.len() / 2
    }

    /// The cache's dtype.
    pub fn dtype(&self) -> DType {
        self.state[0].dtype()
    }

    /// Every layer's keys and values, `[k0, v0, k1, v1, ...]`: the state a traced step writes back.
    pub fn state(&self) -> Vec<Tensor<B, 4>> {
        self.state.clone()
    }

    /// Replace the state with what a step returned (an eager step's write-back).
    pub fn set_state(&mut self, state: Vec<Tensor<B, 4>>) {
        assert_eq!(state.len(), self.state.len(), "a state tensor a layer side");
        for (new, old) in state.iter().zip(&self.state) {
            assert_eq!(new.dims(), old.dims(), "the cache's shape");
        }
        self.state = state;
    }

    /// Layer `i`'s keys and values.
    pub fn layer(&self, i: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        (self.state[2 * i].clone(), self.state[2 * i + 1].clone())
    }

    fn shape(&self) -> [usize; 4] {
        self.state[0].dims()
    }

    /// The write of one token at `position` (`[1]`, below [`Self::max_len`]), attending to every
    /// position of the cache.
    pub fn step_write(&self, position: Tensor<B, 1, Int>) -> FixedWrite<B> {
        self.step_write_within(position, self.max_len)
    }

    /// The write of one token at `position` (`[1]`, below `len`), attending only to the cache's
    /// first `len` positions: what a step costs then follows `len`, not [`Self::max_len`]. The
    /// keys and values are still written into the whole cache; only what attention reads is cut
    /// ([`FixedWrite::attended`]). A position past `len` is not attended to (wrong by construction:
    /// choose `len` above the position). With `len` and `max_len` multiples of 32, attention reads
    /// the cache's prefix in place.
    pub fn step_write_within(&self, position: Tensor<B, 1, Int>, len: usize) -> FixedWrite<B> {
        let max_len = self.max_len;
        assert!(
            (1..=max_len).contains(&len),
            "attending to {len} of {max_len} positions"
        );
        let at = position.clone().expand([max_len]);
        // Expanded to the cache's shape here, once, rather than in each layer.
        let here = self
            .positions
            .clone()
            .equal(at.clone())
            .reshape([1, 1, max_len, 1])
            .expand(self.shape());
        let later = if len == max_len {
            self.positions.clone().greater(at)
        } else {
            self.positions
                .clone()
                .narrow(0, 0, len)
                .greater(position.expand([len]))
        }
        .reshape([1, 1, 1, len]);
        FixedWrite::Masked { here, later, len }
    }

    /// The write of `c` tokens from `start` (`[1]`) on, and each token's own position (`[c]`,
    /// running past `max_len` for a chunk that does: those rows are not written).
    pub fn chunk_write(
        &self,
        start: Tensor<B, 1, Int>,
        c: usize,
    ) -> (FixedWrite<B>, Tensor<B, 1, Int>) {
        let max_len = self.max_len;
        assert!(c <= max_len, "a chunk of at most {max_len} tokens");
        let [_, kv_heads, _, _] = self.shape();
        let positions = self.positions.clone();
        let from = start.clone().expand([max_len]);
        let here = positions
            .clone()
            .greater_equal(from.clone())
            .bool_and(positions.clone().lower(from.clone().add_scalar(c as i64)))
            .reshape([1, 1, max_len, 1])
            .expand(self.shape());
        let own = start.expand([c]) + positions.clone().narrow(0, 0, c);
        let later = positions
            .clone()
            .reshape([1, max_len])
            .expand([c, max_len])
            .greater(own.clone().reshape([c, 1]).expand([c, max_len]))
            .reshape([1, 1, c, max_len]);
        let offset = (positions.clone() - from).clamp(0, c as i64 - 1);
        let base = positions
            .narrow(0, 0, kv_heads)
            .mul_scalar(c as i64)
            .reshape([kv_heads, 1])
            .expand([kv_heads, max_len]);
        let index = (base + offset.reshape([1, max_len]).expand([kv_heads, max_len]))
            .reshape([kv_heads * max_len]);
        (FixedWrite::Chunk { here, later, index }, own)
    }
}

impl<B: Backend> FixedWrite<B> {
    /// `cache` (`[1, kv_heads, max_len, head_dim]`) with the step's `new` keys or values
    /// (`[1, kv_heads, t, head_dim]`) written where the step writes.
    pub fn write(&self, cache: Tensor<B, 4>, new: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = cache.dims();
        match self {
            FixedWrite::Masked { here, .. } => cache.mask_where(here.clone(), new.expand(shape)),
            FixedWrite::Chunk { here, index, .. } => {
                // The chunk's rows laid over the whole cache by a row gather, kept where `here` is
                // set.
                let [_, kv, c, d] = new.dims();
                let placed = new
                    .reshape([kv * c, d])
                    .select(0, index.clone())
                    .reshape(shape);
                cache.mask_where(here.clone(), placed)
            }
        }
    }

    /// The keys or values a query reads from `cache` (`[1, kv_heads, max_len, head_dim]`, as
    /// [`Self::write`] left it): its first `len` positions for a step that attends to fewer than
    /// the whole cache (a view: no copy), else all of it.
    pub fn attended(&self, cache: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            FixedWrite::Masked { len, .. } if *len < cache.dims()[2] => {
                let [n, kv, _, d] = cache.dims();
                cache.slice([0..n, 0..kv, 0..*len, 0..d])
            }
            _ => cache,
        }
    }

    /// The positions each query must not attend to, of those it reads: an attention mask.
    pub fn later(&self) -> Tensor<B, 4, Bool> {
        match self {
            FixedWrite::Masked { later, .. } | FixedWrite::Chunk { later, .. } => later.clone(),
        }
    }
}
