//! A key/value cache whose steps have the same shapes at every position, so a step can be traced
//! once and replayed: any decoder whose attention reads per-layer keys and values `[1, kv_heads,
//! len, head_dim]` can use it.
//!
//! The cache is stored position-major, `[1, 1, max_len, kv_heads head_dim]` (a position's row is
//! every head's key or value, as a projection's row lays them out), when `head_dim` is a multiple
//! of 32 ([`CacheLayout`]): a step's write is then its projection's row as it is, with nothing
//! relaid, and each head's keys are a tile-aligned column block that attention reads in place.
//! Otherwise (a head's block would not be whole tiles) it is head-major, `[1, kv_heads, max_len,
//! head_dim]`. [`FixedWrite::write`] and [`FixedWrite::attended`] take and give the head-major
//! form either way, so a decoder does not depend on the layout.
//!
//! A step's position (or a chunk's start) is a tensor. A step's keys and values are written at
//! the rows that tensor names (`scatter_nd` with `Assign`: one row a head, not the whole cache); a
//! chunk's land through `mask_where` where a mask built from its start is set. Attention reads
//! the positions a step attends to with those later than each query masked. A masked position adds an exact zero to a
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
use burn::tensor::{Bool, DType, IndexingUpdateOp, Int, Tensor};

/// Every layer's keys and values, and the positions `0..max_len` a step's position is compared
/// with.
#[derive(Clone, Debug)]
pub struct FixedCache<B: Backend> {
    /// `[k0, v0, k1, v1, ...]`, each of [`CacheLayout::shape`].
    state: Vec<Tensor<B, 4>>,
    max_len: usize,
    layout: CacheLayout,
    positions: Tensor<B, 1, Int>,
}

/// How a [`FixedCache`] lays out each layer's keys and values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheLayout {
    /// `[1, kv_heads, max_len, head_dim]`: each head's positions together.
    HeadMajor { kv_heads: usize, head_dim: usize },
    /// `[1, 1, max_len, kv_heads head_dim]`: each position's heads together, as a projection's
    /// row holds them.
    PositionMajor { kv_heads: usize, head_dim: usize },
}

impl CacheLayout {
    /// Position-major where each head's block is whole tiles (`head_dim` a multiple of 32), else
    /// head-major.
    pub fn for_heads(kv_heads: usize, head_dim: usize) -> Self {
        if head_dim % 32 == 0 {
            CacheLayout::PositionMajor { kv_heads, head_dim }
        } else {
            CacheLayout::HeadMajor { kv_heads, head_dim }
        }
    }

    /// `(kv_heads, head_dim)`.
    pub fn heads(self) -> (usize, usize) {
        match self {
            CacheLayout::HeadMajor { kv_heads, head_dim }
            | CacheLayout::PositionMajor { kv_heads, head_dim } => (kv_heads, head_dim),
        }
    }

    /// The stored shape of a layer's keys or values over `max_len` positions.
    pub fn shape(self, max_len: usize) -> [usize; 4] {
        match self {
            CacheLayout::HeadMajor { kv_heads, head_dim } => [1, kv_heads, max_len, head_dim],
            CacheLayout::PositionMajor { kv_heads, head_dim } => {
                [1, 1, max_len, kv_heads * head_dim]
            }
        }
    }
}

/// How a step's keys and values reach a layer's cache, and which positions each query attends to.
#[derive(Clone, Debug)]
pub enum FixedWrite<B: Backend> {
    /// One token at a position. Position-major, the token's row of every head goes to row
    /// `rows[0]` (`[1, 1]`, the position) of the cache's `[max_len, kv_heads head_dim]` rows;
    /// head-major, head `h`'s key or value goes to row `rows[h]` (`[kv_heads, 1]`, `h max_len +
    /// position`: made once a step, not once a layer) of its `[kv_heads max_len, head_dim]` rows.
    /// The query attends to the first `len` positions but those set in `later` (`[1, 1, 1, len]`).
    Masked {
        layout: CacheLayout,
        rows: Tensor<B, 2, Int>,
        later: Tensor<B, 4, Bool>,
        len: usize,
    },
    /// A chunk of `c` tokens at the positions set in `here` (the cache's shape, a run from the
    /// chunk's start). Position-major, cache row `j` takes the chunk's row `index[j]` of its `[c,
    /// kv_heads head_dim]` rows; head-major, cache row `j` of head `h` takes the chunk's row
    /// `index[h max_len + j]` of its `[kv_heads c, head_dim]` keys and values. Query `i` attends to
    /// every position but those set in row `i` of `later` (`[1, 1, c, max_len]`).
    Chunk {
        layout: CacheLayout,
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
        Self::zeros_in(
            layers,
            CacheLayout::for_heads(kv_heads, head_dim),
            max_len,
            dtype,
            device,
        )
    }

    /// An empty cache of `layers` layers, `max_len` positions, laid out as `layout` says.
    pub fn zeros_in(
        layers: usize,
        layout: CacheLayout,
        max_len: usize,
        dtype: DType,
        device: &B::Device,
    ) -> Self {
        assert!(layers >= 1, "a cache of no layers");
        if let CacheLayout::HeadMajor { kv_heads, .. } = layout {
            assert!(
                max_len >= kv_heads,
                "a cache of {max_len} positions for {kv_heads} key/value heads"
            );
        }
        let shape = layout.shape(max_len);
        Self {
            state: (0..2 * layers)
                .map(|_| Tensor::zeros(shape, (device, dtype)))
                .collect(),
            max_len,
            layout,
            positions: Tensor::arange(0..max_len as i64, device),
        }
    }

    /// How each layer's keys and values are laid out.
    pub fn layout(&self) -> CacheLayout {
        self.layout
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

    /// Layer `i`'s keys and values, as stored ([`Self::layout`]).
    pub fn layer(&self, i: usize) -> (Tensor<B, 4>, Tensor<B, 4>) {
        (self.state[2 * i].clone(), self.state[2 * i + 1].clone())
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
        let layout = self.layout;
        let rows = match layout {
            CacheLayout::PositionMajor { .. } => position.clone().reshape([1, 1]),
            // Each head's row of the cache's rows, built here once rather than in each layer.
            CacheLayout::HeadMajor { kv_heads, .. } => (self
                .positions
                .clone()
                .narrow(0, 0, kv_heads)
                .mul_scalar(max_len as i64)
                + position.clone().expand([kv_heads]))
            .reshape([kv_heads, 1]),
        };
        let at = position.clone().expand([max_len]);
        let later = if len == max_len {
            self.positions.clone().greater(at)
        } else {
            self.positions
                .clone()
                .narrow(0, 0, len)
                .greater(position.expand([len]))
        }
        .reshape([1, 1, 1, len]);
        FixedWrite::Masked {
            layout,
            rows,
            later,
            len,
        }
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
        let layout = self.layout;
        let positions = self.positions.clone();
        let from = start.clone().expand([max_len]);
        let here = positions
            .clone()
            .greater_equal(from.clone())
            .bool_and(positions.clone().lower(from.clone().add_scalar(c as i64)))
            .reshape([1, 1, max_len, 1])
            .expand(layout.shape(max_len));
        let own = start.expand([c]) + positions.clone().narrow(0, 0, c);
        let later = positions
            .clone()
            .reshape([1, max_len])
            .expand([c, max_len])
            .greater(own.clone().reshape([c, 1]).expand([c, max_len]))
            .reshape([1, 1, c, max_len]);
        let offset = (positions.clone() - from).clamp(0, c as i64 - 1);
        let index = match layout {
            CacheLayout::PositionMajor { .. } => offset,
            CacheLayout::HeadMajor { kv_heads, .. } => {
                let base = positions
                    .narrow(0, 0, kv_heads)
                    .mul_scalar(c as i64)
                    .reshape([kv_heads, 1])
                    .expand([kv_heads, max_len]);
                (base + offset.reshape([1, max_len]).expand([kv_heads, max_len]))
                    .reshape([kv_heads * max_len])
            }
        };
        (
            FixedWrite::Chunk {
                layout,
                here,
                later,
                index,
            },
            own,
        )
    }
}

impl<B: Backend> FixedWrite<B> {
    /// The cache's layout this write was made for.
    pub fn layout(&self) -> CacheLayout {
        match self {
            FixedWrite::Masked { layout, .. } | FixedWrite::Chunk { layout, .. } => *layout,
        }
    }

    /// `cache` (a layer's keys or values, as the cache stores them) with the step's `new` keys or
    /// values (`[1, kv_heads, t, head_dim]`) written where the step writes.
    pub fn write(&self, cache: Tensor<B, 4>, new: Tensor<B, 4>) -> Tensor<B, 4> {
        let shape = cache.dims();
        let (kv, d) = self.layout().heads();
        let [_, _, t, _] = new.dims();
        assert_eq!(
            new.dims(),
            [1, kv, t, d],
            "the new keys or values, head-major"
        );
        match (self, self.layout()) {
            (FixedWrite::Masked { rows, .. }, CacheLayout::PositionMajor { .. }) => {
                // The token's row (every head's, as its projection laid them out) written over the
                // cache's row at its position: nothing else is read, moved or relaid.
                let [_, _, max_len, w] = shape;
                assert_eq!(t, 1, "a step writes one token");
                cache
                    .reshape([max_len, w])
                    .scatter_nd(
                        rows.clone(),
                        new.swap_dims(1, 2).reshape([1, w]),
                        IndexingUpdateOp::Assign,
                    )
                    .reshape(shape)
            }
            (FixedWrite::Masked { rows, .. }, CacheLayout::HeadMajor { .. }) => {
                // The new row of each head written over the cache's: nothing else is read or
                // moved.
                let [_, _, max_len, _] = shape;
                assert_eq!(t, 1, "a step writes one token");
                cache
                    .reshape([kv * max_len, d])
                    .scatter_nd(rows.clone(), new.reshape([kv, d]), IndexingUpdateOp::Assign)
                    .reshape(shape)
            }
            (FixedWrite::Chunk { here, index, .. }, layout) => {
                // The chunk's rows laid over the whole cache by a row gather, kept where `here` is
                // set.
                let rows = match layout {
                    CacheLayout::PositionMajor { .. } => new.swap_dims(1, 2).reshape([t, kv * d]),
                    CacheLayout::HeadMajor { .. } => new.reshape([kv * t, d]),
                };
                let placed = rows.select(0, index.clone()).reshape(shape);
                cache.mask_where(here.clone(), placed)
            }
        }
    }

    /// The keys or values a query reads from `cache` (as [`Self::write`] left it), head-major
    /// (`[1, kv_heads, len, head_dim]`): its first `len` positions for a step that attends to
    /// fewer than the whole cache, else all of it. A view either way: no copy. Position-major,
    /// head `h` is the column block `h head_dim` of the positions' rows.
    pub fn attended(&self, cache: Tensor<B, 4>) -> Tensor<B, 4> {
        let [n, c1, max_len, c3] = cache.dims();
        let len = match self {
            FixedWrite::Masked { len, .. } => *len,
            FixedWrite::Chunk { .. } => max_len,
        };
        let cache = if len < max_len {
            cache.slice([0..n, 0..c1, 0..len, 0..c3])
        } else {
            cache
        };
        match self.layout() {
            CacheLayout::HeadMajor { .. } => cache,
            CacheLayout::PositionMajor { kv_heads, head_dim } => {
                cache.reshape([n, len, kv_heads, head_dim]).swap_dims(1, 2)
            }
        }
    }

    /// The positions each query must not attend to, of those it reads: an attention mask.
    pub fn later(&self) -> Tensor<B, 4, Bool> {
        match self {
            FixedWrite::Masked { later, .. } | FixedWrite::Chunk { later, .. } => later.clone(),
        }
    }
}
