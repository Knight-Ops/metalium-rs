---
number: 5
title: "Every op has a numeric contract; schedules choose only among bit-equivalent options"
status: "accepted"
affected_paths:
  - "docs/learnings/dataflow-compiler.md"
  - "crates/tt-kernels/src/matmul.rs"
  - "crates/burn-tt/src/matmul_mode.rs"
  - "crates/tt-isa/src/dataflow.rs"
  - "crates/tt-firmware/src/dataflow.rs"
---

# 5. Every op has a numeric contract; schedules choose only among bit-equivalent options

## Context
This stack checks results bit for bit against specification models (the Matrix Unit's group-adder rule of step158, the SFPU interpreter, the mover's copy semantics) and checks Burn's results within derived bounds, never guessed tolerances. The Matrix Unit's accumulation is an order-dependent rounding: an output's bits depend on how its K terms are grouped and in what order group sums reach Dst. Several known speedups (split-K, flash-attention rescaling, packer L1 accumulation, a 16-bit Dst, lower fidelity) change those bits, and Stream-K-style splits change them with core count. A dataflow compiler (ADR 0003) needs a rule for what a schedule may change. Kahn process networks guarantee that channel histories are independent of schedule when processes are deterministic, reads block and no input is chosen by arrival.

## Decision
Every op states a numeric contract: per reduction dimension its chunking and combine order (default: K ascending, hardware groups, partials carried in FP32 Dst), its accumulator width (default FP32 Dst), its fidelity rule (HiFi4 today), its rounding points (every narrowing pack with its mode) and its approximations. The contract belongs to the algorithm. A schedule (blocking, distribution over cores, Dst subblocks, multicast or pulled operand reuse, channel credits instead of barriers, prefetch, spatial placement, fusion that replays every rounding of the unfused graph) may only choose among options bit-equivalent under the contract, and autotuning searches only that set. Splits of a reduction are fixed by the contract, never by core count or card. Every process follows Kahn's discipline: blocking reads, no choice by arrival order, atomics only for credit counts, never for data. A schedule that changes bits is a different contract: opt-in, named, and gated against its own model, as MatmulMode Fast, Exact and Accurate are.

## Consequences
Results never change with the grid, the session's tile count, the card or the schedule. Split-K, flash-attention, packer L1 accumulation, a 16-bit Dst and fidelity below the contract's rule stay opt-in contracts. Every transform's gate compares bit for bit against the unscheduled composition and the specification model, with a negative control. Fused bodies must round to every narrower intermediate format the unfused graph packed to.
