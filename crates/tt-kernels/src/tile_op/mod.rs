//! The tile-generic op (L1 of `docs/learnings/dataflow-compiler.md` §5): an op described by its
//! iteration domain, its operands' tile maps and edge formats, a body of microkernel calls and its
//! numeric contract, so that distribution, blocking, allocation, channels, the role split and
//! lowering are written once for every op (Stage F3 of `docs/plans/dataflow-compiler.md`, H6).
//!
//! The first commit of the stage pins every job the hand-written builders it replaces emit
//! ([`golden`]): each client is ported word for word against those jobs, then its builder is
//! deleted.

#[cfg(test)]
mod golden;
