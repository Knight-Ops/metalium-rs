//! The `Dst` tiling pass (`docs/completed-plans/dataflow-compiler.md` F5, H2; the G4 sketch in
//! `docs/learnings/tt-metal-concepts-review.md`): how a block's output tiles are held in `Dst` while
//! their `K` chains accumulate, and how the math and pack roles hand `Dst` over.
//!
//! `Dst` holds eight FP32 tiles (`DEST_REGISTER_FULL_SIZE` 1,024 sixteen-bit rows, two a 32-bit
//! row; `Dst.md`). Under half-sync the math role owns one half while the pack role drains the other,
//! so a **subblock** -- the output tiles accumulated together -- is at most four FP32 tiles, and two
//! are in flight: the `ready`/`free` handshake starts at `(0, 2)` instead of `(0, 1)`.
//!
//! A subblock is one block row's `w` consecutive output tiles (`h` = 1): each `K` step unpacks `A`'s
//! tile once into `SrcB` and holds it while `B`'s `w` tiles pass through `SrcA`, so a pair costs
//! `1 + 1/w` unpacks rather than two, and every output still takes its pairs in ascending `K`
//! (each output's terms are the single-tile kernel's, in the same order, so the bits are). A taller
//! subblock reuses less for the same tiles: one operand only can be held across the inner loop (a
//! `Src` bank holds one tile), so `h x w` unpacks `h + h w` a step against `1 + w` here, and `A`,
//! the FP32 operand, is the one worth holding.

/// What `Dst` holds and how its owners hand it over: half-sync, FP32 accumulation, 32x32 tiles.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DstPlan {
    /// Two halves, math in one while pack drains the other (else one owner at a time).
    pub half: bool,
}

/// FP32 32x32 tiles `Dst` holds whole: 1,024 sixteen-bit rows, a tile 128 of them.
pub const FP32_TILES: usize = 8;
/// `Dst` rows (FP32, sixteen datums each) one tile fills.
pub const TILE_ROWS: u32 = 64;

impl DstPlan {
    /// Half-sync.
    pub const HALF: DstPlan = DstPlan { half: true };

    /// Output tiles one owner holds at a time (`get_dest_max_tiles`'s count for FP32).
    pub const fn capacity(self) -> usize {
        if self.half {
            FP32_TILES / 2
        } else {
            FP32_TILES
        }
    }

    /// Sections in flight, the value `free` starts at.
    pub const fn sections(self) -> u8 {
        if self.half {
            2
        } else {
            1
        }
    }

    /// The first `Dst` row of `half`'s tile `t`.
    pub fn row(self, half: usize, t: usize) -> u32 {
        assert!(half < self.sections() as usize && t < self.capacity());
        (half * self.capacity() + t) as u32 * TILE_ROWS
    }
}

/// `h x w` output tiles accumulated together in one `Dst` section.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Subblock {
    pub h: usize,
    pub w: usize,
}

/// Why a subblock is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DstError {
    /// More tiles than a section holds, or none.
    DoesNotFit { h: usize, w: usize, capacity: usize },
    /// Only one block row a subblock is built (see the module documentation).
    Taller { h: usize },
}

impl std::fmt::Display for DstError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DstError::DoesNotFit { h, w, capacity } => write!(
                f,
                "a {h} x {w} subblock in a Dst section of {capacity} FP32 tiles"
            ),
            DstError::Taller { h } => write!(f, "a subblock {h} tile rows tall: one is built"),
        }
    }
}

impl Subblock {
    /// `h x w` tiles in a section of `plan`, refused past its capacity.
    pub fn new(h: usize, w: usize, plan: DstPlan) -> Result<Subblock, DstError> {
        let capacity = plan.capacity();
        if h == 0 || w == 0 || h * w > capacity {
            return Err(DstError::DoesNotFit { h, w, capacity });
        }
        if h != 1 {
            return Err(DstError::Taller { h });
        }
        Ok(Subblock { h, w })
    }
}

/// A block row of `cols` output tiles cut into subblocks, left to right, and how many block rows
/// one period of the role programs covers so that the sections alternate across the whole block.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RowPlan {
    pub plan: DstPlan,
    pub subblocks: Vec<Subblock>,
}

impl RowPlan {
    /// The fewest subblocks that cover `cols`, as even as can be (wider first): `cols` that no
    /// width divides (the FLUX.2 VAE's 96 channels are 3 tiles; a block of 9, 18 or 7 columns)
    /// take one narrower subblock rather than all narrow ones.
    pub fn choose(cols: usize, plan: DstPlan) -> RowPlan {
        assert!(cols > 0, "a block row of no tiles");
        let cap = plan.capacity();
        let n = cols.div_ceil(cap);
        let (base, wider) = (cols / n, cols % n);
        let subblocks = (0..n)
            .map(|i| Subblock::new(1, base + usize::from(i < wider), plan).expect("at most cap"))
            .collect();
        RowPlan { plan, subblocks }
    }

    /// Block rows a period of the programs covers: one where a row has an even number of
    /// subblocks (its sections alternate 0, 1, ..., 1 and the next row starts at 0 again), else two.
    pub fn rows_a_period(&self) -> usize {
        if self.plan.half && self.subblocks.len() % 2 == 1 {
            2
        } else {
            1
        }
    }

    /// Output tiles a row covers.
    pub fn cols(&self) -> usize {
        self.subblocks.iter().map(|s| s.w).sum()
    }

    /// Each subblock of block row `row`, `(first column, subblock, section)`, the sections taken
    /// in turn across the block's rows.
    pub fn row(&self, row: usize) -> Vec<(usize, Subblock, usize)> {
        let n = self.subblocks.len();
        let sections = self.plan.sections() as usize;
        let mut j0 = 0;
        self.subblocks
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let at = (j0, s, (row * n + i) % sections);
                j0 += s.w;
                at
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_subblock_past_its_section_is_refused() {
        assert_eq!(
            Subblock::new(1, 5, DstPlan::HALF),
            Err(DstError::DoesNotFit {
                h: 1,
                w: 5,
                capacity: 4
            })
        );
        assert!(Subblock::new(1, 8, DstPlan { half: false }).is_ok());
        assert!(Subblock::new(1, 0, DstPlan::HALF).is_err());
        assert!(Subblock::new(2, 2, DstPlan::HALF).is_err());
        assert!(Subblock::new(1, 4, DstPlan::HALF).is_ok());
    }

    #[test]
    fn rows_are_cut_into_the_fewest_even_subblocks() {
        let widths = |cols| -> Vec<usize> {
            RowPlan::choose(cols, DstPlan::HALF)
                .subblocks
                .iter()
                .map(|s| s.w)
                .collect()
        };
        assert_eq!(widths(1), [1]);
        assert_eq!(widths(3), [3]);
        assert_eq!(widths(4), [4]);
        assert_eq!(widths(5), [3, 2]);
        assert_eq!(widths(7), [4, 3]);
        assert_eq!(widths(9), [3, 3, 3]);
        assert_eq!(widths(12), [4, 4, 4]);
        assert_eq!(widths(18), [4, 4, 4, 3, 3]);
        for cols in 1..=40 {
            let p = RowPlan::choose(cols, DstPlan::HALF);
            assert_eq!(p.cols(), cols);
            assert!(p.subblocks.iter().all(|s| s.w <= 4 && s.h == 1));
            assert_eq!(p.subblocks.len(), cols.div_ceil(4));
        }
    }

    #[test]
    fn sections_alternate_across_rows() {
        for cols in 1..=13 {
            let p = RowPlan::choose(cols, DstPlan::HALF);
            let seq: Vec<usize> = (0..6).flat_map(|r| p.row(r)).map(|(_, _, s)| s).collect();
            assert!(seq.windows(2).all(|w| w[0] != w[1]), "{cols}: {seq:?}");
            // A period repeats.
            let period = p.rows_a_period();
            assert_eq!(p.row(0), p.row(period));
        }
        assert_eq!(DstPlan::HALF.row(1, 3), 448);
    }
}
