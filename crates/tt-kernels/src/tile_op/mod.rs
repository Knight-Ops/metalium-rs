//! The tile-generic op (L1 of `docs/learnings/dataflow-compiler.md` §5): an op described by its
//! iteration domain, its operands' tile maps and edges, a body of microkernel calls and its
//! numeric contract, so that distribution, blocking, allocation, channels, the role split and
//! lowering are written once for every op (Stage F3 of `docs/plans/dataflow-compiler.md`, H6).
//!
//! - [`TileOp`] is the description. Its constructors ([`eltwise`], [`reduce`]) check what a client accepts
//!   and build it; nothing past them names a client.
//! - [`Microkernel`] is the only op-specific code a pass calls: how many items a run of the body
//!   holds, and a run's L1 slots and role programs.
//! - [`distribute`] deals the domain's items over units in runs; [`lower`] turns each run into a
//!   job, reading every operand's tiles as its map says ([`TileMap`]).
//!
//! The first commit of the stage pins every job the hand-written builders it replaces emit
//! ([`golden`]): each client is ported word for word against those jobs, then its builder is
//! deleted.

use std::sync::Arc;

use tt_isa::dm::record::TensorRef;
use tt_isa::format::TileFormat;
use tt_isa::isa::Instruction;

use crate::runtime::SemaphoreInit;
use crate::tensor::{DramAlloc, DramTensor, Elem, Overlap, Result, TensorError, Work};

pub mod distribute;
pub mod lower;
mod reduce;
mod sfpu;

pub use reduce::reduce;
pub use sfpu::eltwise;

#[cfg(test)]
mod golden;

/// What a reduction's partial results are carried through (`ReductionContract`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Combine {
    /// One run holds every term of an item: the fold is inside the body, in `Dst`.
    InDst,
    /// The terms in chunks of `chunk` tiles, in order, each chunk's unfolded partial written to
    /// the result's tile in GDDR and reloaded by the next; the last folds. The chunk is the
    /// contract's, never the machine's (S4).
    GddrPartials { chunk: usize },
}

/// How a reduction iterator's terms are combined: in ascending tile order, through
/// [`Combine`]. Part of the op's meaning (S2): a schedule never changes it (S4).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ReductionContract {
    pub combine: Combine,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum IterKind {
    /// Each index is an independent output: free to tile and deal over units (X3).
    Parallel,
    /// Folded into each output, in the contract's order.
    Reduction(ReductionContract),
}

/// One dimension of the iteration domain: its extent in elements (`elems.div_ceil(32)` tiles,
/// the last ragged where `elems % 32 != 0`) and its kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Iter {
    pub elems: usize,
    pub kind: IterKind,
}

impl Iter {
    pub fn tiles(&self) -> usize {
        self.elems.div_ceil(32)
    }

    pub fn parallel(elems: usize) -> Iter {
        Iter {
            elems,
            kind: IterKind::Parallel,
        }
    }

    pub fn reduction(elems: usize, contract: ReductionContract) -> Iter {
        Iter {
            elems,
            kind: IterKind::Reduction(contract),
        }
    }
}

/// One of an edge's two tile dimensions as a function of the domain's indices: one iterator's
/// index (`wrap`: modulo that many tiles, the operand's rows repeated down the domain's), or
/// none, the operand's one tile along it read for every index (a broadcast).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Iter { iter: usize, wrap: Option<usize> },
    Fixed,
}

impl Dim {
    pub fn of(iter: usize) -> Dim {
        Dim::Iter { iter, wrap: None }
    }

    fn iter(self) -> Option<usize> {
        match self {
            Dim::Iter { iter, .. } => Some(iter),
            Dim::Fixed => None,
        }
    }
}

/// Which tile of an edge a domain index reads or writes: a projection of the domain, each
/// coefficient zero or one, with an optional wrap ([`Dim`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileMap {
    pub rows: Dim,
    pub cols: Dim,
}

impl TileMap {
    /// Whether the edge's tile changes with domain iterator `i`.
    pub fn uses(&self, i: usize) -> bool {
        self.rows.iter() == Some(i) || self.cols.iter() == Some(i)
    }
}

/// How an edge's tiles lie in GDDR.
#[derive(Clone, Debug)]
pub enum Layout {
    /// Tiles of its format at the reference's grid (a block of a parent where the reference says).
    Tiles(TensorRef),
    /// A one-row result's data rows, written again packed `[K / 32, 32]` as a product's `A` reads
    /// them: no packing op (`tensor::packed_twin_writes`).
    PackedRow(DramTensor),
}

/// An operand or result: where it lies, which tile each domain index reads ([`TileMap`]), and
/// its element type and format at the edge (formats are edge data: ADR 0004).
#[derive(Clone, Debug)]
pub struct Access {
    pub layout: Layout,
    pub map: TileMap,
    pub elem: Elem,
}

impl Access {
    pub fn format(&self) -> TileFormat {
        match self.elem {
            Elem::Packed(f) => f,
            _ => TileFormat::FP32,
        }
    }
}

/// What may change an op's bits, stated once (S2): every reduction iterator's
/// [`ReductionContract`] is in the domain; the rest is the body's.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NumericContract {
    /// The body's own semantics, fixed by the SFPU interpreter model (`sfpu::ops::reference_op`,
    /// `sfpu::reduce::reference`), rounded once at the pack to the result's format.
    Program,
}

/// The most operands a body reads: `A`, `B` and a ternary op's `C`.
pub const MAX_OPERANDS: usize = 3;

/// A run's L1 and programs ([`Microkernel::run`]).
pub struct RunCode {
    /// Each operand's first slot, `None` for one the body reads no tiles of (or past the
    /// operands it has).
    pub operands: [Option<u64>; MAX_OPERANDS],
    /// Where a chunk reloads its item's partial ([`Combine::GddrPartials`]).
    pub prior: Option<u64>,
    /// The results' first slot.
    pub out: u64,
    pub init: Vec<SemaphoreInit>,
    pub roles: Arc<[Vec<Instruction>; 3]>,
    pub loops: Arc<[Vec<crate::code::Loop>; 3]>,
}

/// A body as the passes see it: the only op-specific code they call.
pub trait Microkernel {
    /// The labels of a job's two lists, the gather's and the scatter's, for a run or (`chunked`)
    /// a chunk of a reduction's terms.
    fn lists(&self, chunked: bool) -> [&'static str; 2];
    /// The most items a run holds, in the whole arena or (`half`) in half of it for a run
    /// double-buffered with the next; `None` where not one item fits.
    fn group(&self, half: bool) -> Option<usize>;
    /// A run of `len` items: its slots and programs, staged in `half` of the arena where
    /// pipelined; `held[o]`: operand `o`'s one tile is read once a run and held.
    fn run(&self, len: usize, half: Option<u8>, held: &[bool]) -> Result<RunCode>;
    /// One chunk of `count` terms of one item ([`Combine::GddrPartials`]): `prior`, a partial to
    /// reload first; `last`, the chunk that folds.
    fn chunk(&self, prior: bool, count: usize, last: bool) -> Result<RunCode> {
        let _ = (prior, count, last);
        Err(TensorError::Shape(
            "this body carries no partials through GDDR".into(),
        ))
    }
}

/// The tile-generic op.
pub struct TileOp {
    pub domain: Vec<Iter>,
    pub operands: Vec<Access>,
    /// What the op writes; the first is the result tensor, allocated when it is built
    /// ([`TileOp::build`]).
    pub results: Vec<Access>,
    pub body: Box<dyn Microkernel>,
    pub contract: NumericContract,
    /// The result tensor's shape and element type.
    pub out: ([usize; 2], Elem),
}

impl TileOp {
    /// The domain's parallel iterators, outermost first.
    pub fn parallel(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.domain.len()).filter(|&i| self.domain[i].kind == IterKind::Parallel)
    }

    /// The domain's reduction iterators, outermost first.
    pub fn reductions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.domain.len()).filter(|&i| self.domain[i].kind != IterKind::Parallel)
    }

    /// Output items: one for each index of the parallel iterators, row-major.
    pub fn items(&self) -> usize {
        self.parallel().map(|i| self.domain[i].tiles()).product()
    }

    /// Tiles of input each item folds: the reduction iterators' tiles.
    pub fn weight(&self) -> usize {
        self.reductions().map(|i| self.domain[i].tiles()).product()
    }

    /// Checks what every pass relies on: each map names iterators of the domain, and each
    /// iterator is read by some operand.
    pub fn check(&self) -> Result<()> {
        let n = self.domain.len();
        if self.operands.len() > MAX_OPERANDS {
            return Err(TensorError::Shape(format!(
                "{} operands: a body reads at most {MAX_OPERANDS}",
                self.operands.len()
            )));
        }
        for a in self.operands.iter().chain(&self.results) {
            for d in [a.map.rows, a.map.cols] {
                if let Dim::Iter { iter, wrap } = d {
                    if iter >= n || wrap == Some(0) {
                        return Err(TensorError::Shape(format!(
                            "a tile map names iterator {iter} (wrap {wrap:?}) of a {n}-iterator \
                             domain"
                        )));
                    }
                }
            }
        }
        if let Some(i) = (0..n).find(|&i| !self.operands.iter().any(|a| a.map.uses(i))) {
            return Err(TensorError::Shape(format!(
                "domain iterator {i} is read by no operand"
            )));
        }
        Ok(())
    }

    /// The op's work: its result allocated, its items dealt over `units` ([`distribute::deal`])
    /// and each run lowered to a job ([`lower::jobs`]).
    pub fn build(&mut self, alloc: &mut DramAlloc, units: usize, overlap: Overlap) -> Result<Work> {
        self.check()?;
        let ([rows, cols], elem) = self.out;
        let out = DramTensor::alloc_elem(alloc, rows, cols, elem)?;
        if let Some(first) = self.results.first_mut() {
            first.layout = Layout::Tiles(out.tensor_ref());
        }
        let built =
            distribute::deal(self, units, overlap).and_then(|deal| lower::jobs(self, &deal, units));
        match built {
            Ok(jobs) => Ok(Work { out, jobs }),
            Err(e) => {
                alloc.free(&out.placement);
                Err(e)
            }
        }
    }
}
