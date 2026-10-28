---
number: 3
title: "A tile-level dataflow compiler below Burn"
status: "accepted"
affected_paths:
  - "docs/learnings/dataflow-compiler.md"
  - "crates/tt-kernels/src/tensor.rs"
  - "crates/tt-kernels/src/session.rs"
  - "crates/tt-kernels/src/l1.rs"
  - "crates/tt-kernels/src/residency.rs"
  - "crates/burn-tt/src/fusion/mod.rs"
---

# 3. A tile-level dataflow compiler below Burn

## Context
Burn's backend API is op-at-a-time and burn-tt maps each op to a Work of independent jobs that read and write GDDR and end in a cross-unit barrier. Decisions spanning ops, cores or formats had nowhere to be made: three matmul builders by format, 145 generated wrappers widening BFP to F32, ~110 format checks, no iteration-space description an optimization could read. Measured: products at prefill shapes reach 5-18 of ~166 TFLOPS, bound by operand reads; a 1.7B decode step reads at 328 of 512 GB/s with every product ending in a tail; a small decode op costs 9.84 us, ~3 of it its barrier. kernel-fusion-architecture.md section 4 had recorded the opposite decision (no second graph compiler), out of concern for multi-dialect compile latency. The owner asked on 2026-10-27 for a dataflow compiler design document that future work below Burn builds on, generic across formats and ops and evidence-based.

## Decision
Below Burn sits a native-Rust tile-level dataflow compiler, specified by docs/learnings/dataflow-compiler.md. Burn (burn-fusion and trace capture) supplies graphs (L0); ops are described as tile-generic ops with indexing maps, iterator kinds, a body of microkernel calls and a numeric contract (L1); passes in a fixed order lower them to a dataflow graph of processes and typed channels with credits and multicast (L2); and that graph is emitted as the existing machine form: Step/Job/Work, mover records, code::Code and loops, l1::Requirements, the word-keyed program cache (L3). Every resource is accounted before emission; exceeding one is a plan-time choice or typed refusal, never a retry. Eager ops compile in microseconds and are cached by their words; captures may take longer. Every principle is tagged with its evidence (hardware, measured, published, hypothesis) and what would overturn it; Tenstorrent's designs are adopted or rejected on evidence.

## Consequences
Optimizations below Burn are written as transforms over the L1/L2 descriptions, not per op or per format; a change that names a format or a single op in a transform is in the wrong layer. kernel-fusion-architecture.md section 4 and its SfpuMicroOp section are superseded. Hand-written op builders stay until their generic replacement is bit-exact against them, then are deleted. New performance work starts with a hypothesis, prediction and kill criterion in the document's register, and its run IDs are recorded there. The no-multi-dialect concern is kept: no MLIR, no C++ emission, no separate compiler process.
