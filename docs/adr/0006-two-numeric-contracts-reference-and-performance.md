---
number: 6
title: "Two numeric contracts: reference for conformance and training, performance as the inference default"
status: "accepted"
amends: 5
affected_paths:
  - "docs/learnings/dataflow-compiler.md"
  - "docs/plans/tt-metal-parity.md"
  - "crates/burn-tt/src/registry.rs"
  - "crates/burn-tt/src/matmul_mode.rs"
  - "crates/burn-tt/src/ops.rs"
  - "crates/tt-kernels/src/matmul.rs"
  - "crates/tt-models/src/generate.rs"
---

# 6. Two numeric contracts: reference for conformance and training, performance as the inference default

## Context
ADR 0005 gave every op a numeric contract. Its default fixes:
- HiFi4;
- FP32 `Dst`;
- K ascending in hardware groups;
- F32 operands in L1, with TF32 in `Src`.

Every other choice is a separately named opt-in contract.

**What the review found (2026-11-09, [tt-metal-parity.md](../plans/tt-metal-parity.md)).** This default binds performance more than any schedule left to tune:
- **BF16 is storage only.** BF16 and F16 arithmetic widens to F32 through separate cast ops (`burn-tt/src/ops.rs`, `cast_native`). So activations move as 4 KiB F32 tiles, and every matmul output packs to an FP32 slot.
- **Prefill sits far below the card.** Its products reach at best ~30 TFLOPS. On p150, tt-metal reaches:
  - 168 TFLOPS at HiFi4;
  - 240 at BF16-HiFi2;
  - 470-554 at BFP8-LoFi.
- **Decode is bound by unpack and math, not bytes.** That is why BFP4 weights bought 1.09x where 1.5x was predicted. *(Corrected 2026-11-09: at `m` = 1 fidelity moves no product, and BFP8 / BFP4 weights take the 1.7B's step 9.83 -> 7.51 / 6.34 ms. The 1.09x of 2026-10-28 was read before the even deal, the column layout and the prefetch; [performance-log](../learnings/performance-log.md#w1-bounded-the-performance-contract-on-this-stack-before-any-plumbing-2026-11-09-cards-0-and-1).)*
- **Most of the order-changing levers stay opt-in by rule:** packer L1 accumulation, lower fidelity, split-K, flash-decode.

tt-metal's default for LLMs is:
- BFP8 weights and KV cache, with BFP4 for the MLP in its performance mode;
- BF16 activations;
- HiFi2 (LoFi for BFP4);
- packer L1 accumulation.

It is validated by teacher-forced top-1/top-5 token match against HuggingFace, and by PCC.

The owner decided on 2026-11-09:
- the parity target is an 8B-class model on one p150, at batch 1 and batch 32;
- inference should default to a faster contract;
- bit-exact results stay available and gated.

## Decision
There are two named contracts. Both are deterministic: Kahn semantics, splits fixed by the contract and never by the grid or the card.

### Reference
ADR 0005's default, unchanged:
- HiFi4;
- FP32 `Dst`;
- K ascending;
- F32 activations;
- bit for bit against the specification models.

It is the default for:
- conformance (`cargo xtask conformance`);
- the training gates (the MNIST golden);
- any user who asks for it (`TtConfig::numerics(Reference)`).

### Performance
The inference default. Its per-op rule:
- **Activations:** BF16 in L1 and GDDR. Arithmetic runs natively on BF16 edges, unpacked to `Src` with no cast op, and narrows once at each op's pack.
- **Weights:** BFP8 by default; BFP4 for the MLP's gate and up as a named option.
- **KV cache:** BF16 by default; BFP8 as a named option.
- **Fidelity:**
  - the fewest phases that keep every mantissa bit of the edge formats (derived fidelity);
  - at most HiFi2 for BF16×BFP8;
  - LoFi for BFP4.
- **Accumulation:**
  - FP32 `Dst` for attention and norms;
  - packer L1 accumulation allowed, its partial format stated;
  - a 16-bit `Dst` only where the op's contract names it.
- **K chunking:** fixed per op by its shape (for example a K block of 8 tiles), never by the unit count. A split-K is allowed only with its combine order stated.
- **Softmax and attention:**
  - flash-style rescaling allowed, its block size fixed per op;
  - statistics in FP32.

### Validation of the performance contract
1. **Each op has its own specification model:**
   - the step158 group-adder rule, applied at the stated fidelity;
   - the SFPU interpreter;
   - the format codecs.

   A gate compares bit for bit against that model, with a negative control. A schedule may still only choose among options bit-equivalent under this contract. ADR 0005's rule holds inside each contract.
2. **Each model has teacher-forced top-1/top-5 token agreement and logit PCC against HuggingFace.** The thresholds are derived from a reference-contract run of the same model plus a stated margin, never guessed. They are recorded beside the gate.

### How a contract is chosen
- A contract is chosen per session or per trace (`TtConfig`). A trace freezes the contract it was captured in.
- Mixing contracts within a graph is allowed only at a named boundary.

### What the switch is, and what Performance means so far (2026-11-10, roadmap W1)
- **The switch:** `burn_tt::Numerics { Reference, Performance }`.
  - `TtConfig::numerics` (or `TT_NUMERICS`) sets it for an attachment; the default is `Reference`.
  - `burn_tt::with_numerics` overrides it for the work a closure builds on its thread.
  - A trace records the contract it was captured under (`TracedInference::numerics`), and the residency-plan key includes it.
  - A capture refuses a `with_numerics` to another contract part way: no named boundary exists yet.
- **Who asks for Performance:** inference does. `tt_models`' `Llama::for_inference` / `Qwen35::for_inference` take the contract, store the weights in its formats (`quant::Plan::for_numerics`) and record it; a `TracedGenerator` captures under the model's contract. Training and conformance keep the attachment's default, the reference contract.
- **Performance today is its first box only:**
  - BFP8 weights (BFP4 `gate|up` as `WeightOptions::mlp_bfp4`);
  - F32 activations and HiFi4, as under the reference contract;
  - no burn-tt op branches on the contract yet.

  Its specification model is therefore exact: the F32 model holding the weights' rounded values (`step240`). The model gate is `step269_performance_contract`. BF16 activations, derived fidelity and packer L1 accumulation extend both as their boxes land.
- **BF16 activations into the products, the default (2026-11-11, roadmap W1):** under the performance contract
  `Llama::for_inference` rounds each norm's output and the MLP's `silu(gate) * up` once to BF16 where it is made
  (`burn_tt::rms_norm_as`, `silu_mul_as`), and the products read them as they lie with F32 results
  (`burn_tt::matmul_as`); the residual stream, the attention block and every op's arithmetic stay F32. Each op is bit
  for bit its F32 op then the cast (`step76`, `step270`), so the specification model is the F32 model holding the
  weights' rounded values with those casts written in. On the 1.7B: decode unchanged, prefill 1.06-1.09x, `step269`
  the same 228 of 232 (performance-log, "BF16 activations into the products"). `WeightOptions::f32_activations` is the
  baseline. A BF16 residual stream (every op on BF16, Burn's BF16 compositions) was tried first and deleted: slower
  and one position under the top-1 threshold (failed-approaches).

## Consequences
**Results under the performance contract change** with the contract's version, never with the grid, the card or the schedule.

**The "bit for bit" claims in the scoreboard and the decode plans** now read "bit for bit under the reference contract". New inference performance rows state their contract.

**The roadmap's parity work (W1-W3)** is written against the performance contract:
- native BF16 arithmetic;
- BFP8 weights;
- LoFi/HiFi2 products;
- 2-D operand multicast;
- batched decode.

**Rows in [failed-approaches.md](../learnings/failed-approaches.md) marked "(revisit)"** are re-bounded under it as new hypotheses. Examples: P7 low fidelity, packer L1 accumulation, flash-decode and split-K.

**Training keeps the reference contract** until a fine-tuning gate (W8) states a BF16 training contract (FP32 master weights and optimizer state) and its loss-curve bound.

**ADR 0005 stays in force for its rule** (schedules choose only among options bit-equivalent under the contract). Its default is now the reference contract, which is the default only outside inference.
