# Roadmap: open work

**Start here.** This file lists open work and nothing else. Reset on 2026-11-09 after the review against tt-metal.
- **Why and in what order:** [tt-metal-parity.md](tt-metal-parity.md).
- **What has been achieved:** [achievements.md](../completed-plans/achievements.md).
- **What was tried and dropped:** [failed-approaches.md](../learnings/failed-approaches.md).
- **Current numbers:** [performance-scoreboard.md](../learnings/performance-scoreboard.md).
- **The roadmap before the reset:** `git show 3e4cefd:docs/plans/roadmap.md`.

**Rules for this file:**
- An item is a box with a link to its plan or evidence.
- When it closes, it moves to achievements.md, or to failed-approaches.md if it was killed. It never grows into a log here.

## Where the project stands (2026-11-09)

- **Foundations are done:**
  - host driver, ttsim, firmware;
  - a native Burn backend with no unsupported float, int or bool trait method;
  - Burn's own suites on silicon: F32/F16 3,001 of 3,025, BF16 1,098 of 1,187, none listed for a missing op.
- **Models run on the card,** traced:
  - ResNet-18 in 12.7 ms and ViT-Tiny in 7.7 ms;
  - SmolLM2-1.7B decodes at ~133 tokens/s (7.52 ms a step) under the performance contract, the inference default since 2026-11-10 (BFP8 weights; top-1 98.28% against HuggingFace, `step269`); 9.81 ms under the reference contract (BF16-stored weights, bit for bit the F32 model);
  - Qwen3.5-4B decodes in 56.9 ms a step under the performance contract (BFP8 weights; no quality gate yet), 62.5 under the reference (~21 of either a greedy pick, W4);
  - a 7,992-token prompt prefills in 5.98 s.

  All of this is single card and batch 1. The 1.7B's 7.52 ms is under the performance contract; everything else is under the bit-exact reference contract.
- **Against tt-metal, same card and model (2026-11-09):**
  - Our batch-1 decode leads: 7.52 (performance) / 9.81 (reference) against 10.55-12.24 ms.
  - tt-metal batches 32 users at 1,289 tokens/s, prefills 2.2-3.1x faster, and runs prefill-shape products 2.4-22x faster (BF16/BFP8, HiFi2/LoFi, 2-D operand multicast).
  - Qwen3.5-4B: tt-metal 38.0 ms on the card and 44-46 host-timed, with BFP4/BFP8 weights at LoFi and the argmax on the host; ours 56.9 with BFP8 weights (62.5 with BF16), or ~36 without our 21 ms greedy pick.
  - Training, since 2026-11-09 traced on every tile by default (`tt-mnist`), beats burn-flex on the CPU: MNIST 1.31 against 2.3-2.6 ms a step, the small transformer 3.75 against 3.9-4.3.
  - [tt-metal-parity.md](tt-metal-parity.md) §1-§3.
- **Multi-chip** is a proof of concept: matmul, conv and attention split across two cards; no all-reduce.

## Goals, in order

1. **Parity with tt-metal on one p150:**
   - an 8B-class model at batch 1 (tt-metal ~38.5 t/s) and batch 32 (42 t/s/user);
   - prefill products at tt-metal's TFLOPS on the same card.
2. **Usable by default:** a Burn user gets the grid, the right contract and traced speed without setting knobs.
3. **LLM fine-tuning on one card:** LoRA/SFT of a 0.5-2B model.
4. **Multi-chip:** tensor parallelism for models past one card; data-parallel training.

A publishable crate is long-term (last section).

## Standing constraints

- **Two numeric contracts** ([ADR 0006](../adr/0006-two-numeric-contracts-reference-and-performance.md)).
  - *Reference* (bit-exact: HiFi4, FP32 `Dst`, F32 activations) is the default for conformance and training gates.
  - *Performance* (deterministic: BF16 activations, BFP8 weights, derived fidelity up to HiFi2, packer L1 accumulation) is the inference default (`burn_tt::Numerics`; tt-models' `for_inference`). So far it is BFP8 weights only; each W1 box below adds a part.
  - Inside each contract a schedule chooses only among bit-equivalent options (ADR 0005).
  - Every performance number states its contract.
- **Every default must scale** (owner, 2026-11-03):
  - **A default holds a weight's bytes once:** no second copy, relaid twin or per-consumer layout beside the stored one.
  - **Extra GDDR a default takes is bounded,** by a constant or by what already scales with the workload (activations, KV cache, traces).
  - **A decode optimization is judged against the largest model in scope** (Qwen3.5-4B in F32, a model that fills one card, a model split across both). The parity 8B model joins this list with W5.
  - **A gain that needs spare GDDR stays opt-in,** documented with its GDDR cost.
- **The big-bet rule** ([tt-metal-parity.md](tt-metal-parity.md) §7):
  - A performance item states its contract and its predicted end-to-end gain on the parity workload before it is built.
  - Under 3% of the step goes to the backlog unless it unblocks a W item.
  - A change that slows a default by more than 0.5% does not land as the default.
- **Validate on card 0** (both cards only for multi-card work). Model workloads never run on ttsim (AGENTS.md).

## Track W: parity with tt-metal

In order; W1-W4 can overlap. Each item's prediction, kill criterion and gate are in
[tt-metal-parity.md](tt-metal-parity.md) §5.

### W0. Same-card parity scoreboard
- [x] Record tt-metal's numbers on card 1 beside ours: microbenches, decode-shape and prefill-shape products, SmolLM2-1.7B through `tt_transformers` at batch 1 and 32 (2026-11-09; [parity table](../learnings/performance-scoreboard.md#parity-with-tt-metal-same-card), [tt-metal-parity.md](tt-metal-parity.md) §2.3). Re-run when a W item lands.
- [ ] `cargo xtask parity` (dev-only): runs both sides on one fenced card (`TT_VISIBLE_DEVICES`), emits the table. Folds in F11's standing check (tt-metal's own DRAM-read binary on the card).
- [ ] F12: does a reader of ours leak NoC VC state? tt-metal saw ~20% DRAM read loss after a DRAM-sharded reader.

### W1. The performance contract
**Bounded 2026-11-09** ([log](../learnings/performance-log.md#w1-bounded-the-performance-contract-on-this-stack-before-any-plumbing-2026-11-09-cards-0-and-1); achievements):
- BFP8 weights alone take the 1.7B's step 9.83 -> 7.51 ms, so the 8.0 ms kill (on the predicted step) does not fire.
- Fidelity moves nothing at `m` = 1.
- A BF16 `A` was slower until a packed tile ran whole in `SrcB` (prefill) and a BF16 row was packed (decode), both 2026-11-10.

BFP8 weights are the inference default since 2026-11-10, behind the contract switch and the model gate `step269` (achievements). The same day (achievements):
- a packed tile whole in `SrcB`: a BF16 `A`'s prefill products 1.13-1.32x the FP32 `A`'s;
- the one-row path for a BF16 `A`: its decode products at the FP32 `A`'s step (BFP8 `B`, HiFi2: 10.74 -> 6.99 ms of products, FP32 `A` 7.00).

A BF16 `A` is now no slower than an FP32 one at any `m`, so native BF16 arithmetic (below) is what turns it into a gain on a model. Packer L1 accumulation first needs the owner's decision.
- [ ] **Packer L1 accumulation: below the 3% rule now; the owner decides whether it moves to failed-approaches.**
  - Ceiling with a packed `A` whole in `SrcB` (2026-11-10): at most 2.5% of a product, none on the 1.7B's 512-row chunk (`matmul::probe_skip_reload`; [log](../learnings/performance-log.md#w1-a-packed-tile-whole-in-srcb-a-bf16-a-past-the-fp32-as-rate-2026-11-10-card-0)). It was up to 24% only while a packed `A` took faces.
  - Needs a ttsim check of `Pack_L1_Acc` (else an isolated silicon gate and a divergence entry), the partial format stated, and a spec model.
  - Delete `probe_skip_reload` when it lands or is killed.
- [ ] **Native BF16 arithmetic,** through the 1.7B on a BF16 stream. Bounded 2026-11-11 ([log](../learnings/performance-log.md#w1-native-bf16-arithmetic-bounded-before-building-2026-11-11-card-0)): bit for bit today's widen-compute-narrow composition, so no op branches on the contract. Predicted a 256-token prefill ~1.08x (chunk products 1.19-1.24x), decode ~0 at short context and at most ~5% at 2K. Phases: BF16 slots in GDDR (done 2026-11-11, `TensorRef::slot`); SFPU ops with BF16 operands and results; products with a BF16 result (quantize on the last K chunk; kill if its cost eats the prefill gain) and a BF16 packed twin; burn-tt routing without `cast_native`; the 1.7B's norm, rope, attention and cache in BF16; tt-models' stream from the contract, `step269`, the bench and the default under the 0.5% rule.
  - BF16 edges unpacked to `Src`, packed to BF16 slots (2 KiB);
  - a one-row BF16 result writes its packed twin in BF16 (`Session::set_packed_twins` writes F32 twins only, so a BF16 `A` is packed again at every product today);
  - `float_matmul` and the element-wise wrappers accept BF16 without `cast_native`;
  - BF16 tensors in BF16 GDDR slots.
- [ ] **Fix the four BF16 defects left from the new architectures** (worked around, never fixed; [new-architectures.md](new-architectures.md)):
  - a BF16 product with a transposed `B` copies it plain (> 40 GB of host bookkeeping at 248K x 2,560);
  - `Precision::Bf16` grows host memory without bound on Qwen3.5;
  - casting a just-loaded tensor uploads the original first;
  - the block copier refuses element-interleaving moves past ~400K elements.
- [ ] **The BF16 embedding lookup without staging** (`gather_indexed_bf16` still stages; `step167` BF16 takes 229 s). B now has the room it lacked (10-14 KiB after the firmware diet). It also ends a second copy in today's defaults: a tied Llama (the 1.7B) keeps its table F32 for the lookup and a stored, untied head beside it.
- [ ] **A BF16 KV cache, and attention reading it without widening** (`attention/widen` widens every call).
- [ ] **A BFP8 KV cache as a named option** (F8).
- [ ] **The contract switch's reach.** `burn_tt::Numerics` landed (2026-11-10): attachment, thread scope, frozen into traces. Still open:
  - ops that branch on it, as BF16 activations, fidelity and packer L1 accumulation land;
  - `with_numerics` cannot reach `Fusion<TtBackend>`'s ops (another thread), as `with_matmul_mode` cannot. Traces record it, and no op branches yet; fix before the first op does.
- [ ] **Per-op specification models** at the stated fidelity (step158's group adder per phase).
- [ ] **The model gate for Qwen3.5-4B** (the 1.7B's is `step269_performance_contract`): a prompt set from `qwen35_reference.py`, the same metrics and margin.
- [ ] **The model gate's thresholds as each W1 box lands:** BFP8's top-1 clears its floor by one position (228 of 232). A box that trips it is reviewed with the owner, never loosened silently.
- [ ] **The BFP4 `gate|up` option** misses tt-metal's floors on the 1.7B (top-1 86.6%, PCC 0.93); it is recorded only. Judge it again on the W5 model.

### W2. Prefill: sharded activations and 2-D operand multicast
- [ ] **A probe.** `[2048,2048]@[2048,16384]` on a 10x11 sub-grid, BF16-HiFi2:
  - `A` multicast along rows and `B` down columns from one sender per strip (`WRITE_L1_MCAST`, credit-gated);
  - compare against the reads-free ceiling (`1793224072`).

  Kill below 60 TFLOPS with operands delivered: then the kernel's push and unpack are next.
- [ ] **Multicast products as an L2 channel transform** in the dataflow compiler (the old prefill P5 / G7, "each operand out of GDDR about once"; [prefill-operand-reuse.md](prefill-operand-reuse.md)).
- [ ] **Activations L1-sharded between consecutive ops** where a capture proves the layout: an L2 pass over the op graph.
- [ ] **A GEMM TFLOPS gate** at BF16-HiFi2 and BFP8-LoFi beside tt-metal's on the same card (F9: fidelity against structure).
- [ ] **Known gaps to fold in:**
  - `[2048,8192]@[8192,2048]` is 1.7x slower through Burn than through the session (14.0-14.2 against 18.5 TFLOPS; never closed);
  - Qwen3.5's grouped attention re-reads `A` 2-32 times (`batched_attention_questions`, unmeasured on the card);
  - F5's partials reload stops the math role on a continuing chunk (`UnpackToDst` beside `MVMUL` is wrong on silicon);
  - the long prompt's passes: the mask read once a head; 64 jobs at chunks of 64.

### W3. Batched decode
- [ ] **Promote [batched-decode.md](../proposals/batched-decode.md) to a plan:**
  - sequences as the rows of a tile row (8/16/32);
  - a batch dimension in `FixedShapeDecoder`;
  - a paged position-major cache and per-sequence row scatter;
  - per-(sequence, head) attention items.
- [ ] **First measurement:** the 1.7B at batch 8 and 32, fixed shapes, against batch 1. Kill above 2x batch 1 at batch 32.
- [ ] **Gate:** each sequence's tokens equal its batch-1 run bit for bit under the same contract; mixed lengths; a swapped-cache negative control.
- [ ] **Later:** bit-exact speculative decode (H12). A draft proposes `k`, the target verifies in one `m = k` step, and greedy acceptance gives plain decode's tokens. Ceiling 2-3x tokens/s.

### W4. Sampling on the grid
- [ ] **Diagnose the wide-vocabulary greedy pick.**
  - Qwen3.5: eight ~2.7 ms parts, ~21.4 ms a step, ~1.5 GB/s; [log](../learnings/performance-log.md#qwen35-4bs-step-on-the-card-a-third-of-it-is-the-greedy-pick-the-even-deals-04-is-the-head-2026-11-08-cards-0-and-1).
  - The code spreads partials over up to 64 units (`Session::argmax_row`); the profile puts each part's gather on unit 0.
  - Profile one part (`TT_PROFILE_DETAIL=pipeline`) before changing code.
- [ ] **One grid-wide argmax for any vocabulary.** Retire the 65,536-column split in `tt-models/src/generate.rs`. Bit for bit, including first max, NaN and `-inf` padding.
- [ ] **Native top-k / top-p over the grid,** trace-compatible (F7). The current on-card sampler is eager and ~112 ms a token.

### W5. An 8B-class model
- [ ] **Choose the model.** Llama-3.1-8B-Instruct (tt-metal publishes it; our token gets a 403) or Qwen3.5-9B, which tt-metal runs on one P150 (`models/demos/blackhole/qwen36`), is ungated, and is the family of our Qwen3.5-4B, which tt-metal's path ran unchanged on card 1. Add it to `fetch-weights` as `on_request`; load it through the generic Llama loader (ADR 0001).
- [ ] **Model coverage it needs** (F15): QKV bias, RoPE scaling, sliding window as required.
- [ ] **On the ladder** (silicon only) at batch 1 and 32, with W1's model gate; a smoke entry.

### W6. Defaults and usability
- [x] **The grid by default** (2026-11-09): `Topology::single` and burn-tt's auto-attach open every tile; an eager op runs at `work:64:4` and a captured one on every unit (`Session::set_capture_spread`); `TT_SPREAD` sets both. On card 0, MNIST eager 2.17 -> 1.55-1.81 ms a step and traced 1.31; the transformer eager 8.5 -> 9.4 (its ops are tiny), traced 3.75 ([performance-log](../learnings/performance-log.md#training-on-every-tile-traced-by-default-2026-11-09-card-0)).
- [ ] **The grid's leftovers:**
  - the tiny transformer's eager step is still 10% slower on every tile than on one (~5 us of host a unit an op): a per-op spread with a cheaper catch-up, or auto-capture;
  - per card for a mesh;
  - an autotune cache keyed by `(op, shape, dtype)`;
  - B16 placement.
- [ ] **One typed configuration** (`TtConfig`, `SessionConfig`):
  - every `TT_*` knob validated the same way (bad values are named errors: `TT_OPERAND_REUSE` panics, `units_from_env` silently defaults);
  - the 56 process-wide atomics moved into the session.
- [ ] **Delete the knobs that lost** ([failed-approaches.md](../learnings/failed-approaches.md), "Knobs to delete").
- [ ] **Auto-capture of a repeated graph** under `Fusion<TtBackend>`.
- [ ] **The eager host-build cost.** An eager product's host build is 10.8 of its 19.7 ms; eager slows after a pulled capture (+85 MB live).
- [ ] **ResNet-18's untraced forward is ~11% slower than recorded** (2026-10-27); bisect.

### W7. Code consolidation (done inside the changes that touch the code)
- [ ] Split `Session` (11.8k lines, 108 fields) into its program-cache, residency, product-planning, trace and transfer parts.
- [ ] **One matmul entry through TileOp.** Retire `matmul_on`'s host-staged path. Make `float_matmul`'s host-download fallback (`burn-tt/src/ops.rs`, the last branch) refuse by name once shown unreachable.
- [ ] **Port** rms_norm, rope, argmax and gather_rows to TileOp as W1-W4 touch them, word for word.
- [ ] **Program and trace size** (F3/F4):
  - programs that do not grow with length (unrolled per-tile programs remain);
  - a trace of 1.5 GB for the 1.7B against tt-metal's 56-70 MB for 8B;
  - sharing identical entries across units.

### W8. Training: LLM fine-tuning
- [x] **Traced training by default in `tt-mnist`** (2026-11-09; `--eager` opts out): on every tile the step fits the program cache. Traced is faster in every mode measured: MNIST 1.31 ms a step (eager 1.55-1.81), BF16 5.3 (75.6), the CNN 31 (71), the transformer 3.75 (9.4).
- [ ] **`TracedTrainingStep` for any Burn training loop** (auto-capture, W6) and trace segmentation for steps that overflow the 240 KiB program cache ([traced-execution.md](traced-execution.md) §8.1; one tile overflowed it on 2026-10-21).
- [ ] **The first traced step costs 0.5-23 s** (MNIST 3.4 s, BF16 23 s): the measured block choice (`9977ad8`) times each product's candidates at capture. Build time is not a target, but a fine-tuning run's first step should not dominate a short run: cache the choices across processes (keyed by shape, route, units).
- [ ] **`tt-mnist --fusion` panics** in burn-fusion (`Should have handle for tensor`, `ordering.rs:65` index out of bounds), on one tile and eager too, at `3e4cefd`; no gate runs it. Bisect, fix, and add a smoke entry.
- [ ] **A fused AdamW** with FP32 master weights and moments.
- [ ] **Fused forward and backward kernels:** RMSNorm and LayerNorm, softmax/attention, cross-entropy, SwiGLU. This covers B13b (the general SFPU chain fuser) and the reduce/norm fuser ([kernel-fusion.md](kernel-fusion.md) Phases 2 and 4). Bound B13c epilogues per model family (H10's kill was the 1.7B's only).
- [ ] **The chunked paths left in training:**
  - conv backward (weight gradient `patches^T @ G`, x gradient as a gather over taps);
  - `max_pool2d_with_indices`;
  - average pooling;
  - grouped and 3-D convolutions;
  - pooling with `C % 32 != 0`.
- [ ] **A BF16 training contract** (BF16 compute, FP32 master) with a loss-curve bound against F32 (ADR 0006).
- [ ] **LoRA adapters** on the generic Llama loader. A gate fine-tunes SmolLM2-1.7B or Qwen3-0.6B, timed beside tt-train on the same card.
- [ ] **B15 leftovers:** a real `Learner` run (stateful optimizer record); one `burn-onnx` model (deferred).
- [ ] **Mixed-precision leftovers:** BFP under autodiff; half formats against a per-matmul reference.

### W9. Firmware and dispatch
- [ ] **Decide the shipped read mode:** safe, or `fast-reads` + `encoded-guard`, measured on the parity workload. Fast-mode P4 (sealed static read records) and P5 (`fast-writes`, sealed `WRITE_RUN`/`SCATTER`).
- [ ] **When W2/W3 need B code past its room:** load cold op families as validated L1 code through T1's `.ttinsn` path, not another diet.
- [ ] **The replay's start** (~170 us: catch-up and `CALL` are two enqueues). Re-measure; a firmware change if it stands.
- [ ] **Host restore residue:** 2.2 / 5.5 ms a generation (~55 us a unit restore).
- [ ] **F16, the per-op control cost:** ~4.1 against tt-metal's 1.29 us traced.
- [ ] **F6:** `SFPLOADMACRO` replay for exp/recip.
- [ ] **A smaller program cache, its L1 given to buffers,** once a consumer is shown L1-bound (with on-card restores only the largest trace's programs must fit: 108 KB a unit for the 1.7B's 256-row chunk).
- [ ] **The `ttinsn_t1` experiment image** overflows T1's slot by 216 bytes: slim or retire it.
- [ ] **F10,** a DRISC prefetcher: in reserve (DRISC has no loader here).

### W10. Multi-chip (after W1-W3)

Design decisions of 2026-10-14 stand:
- a Burn device is one card;
- data-parallel training goes through a native `DistributedBackend` (B14) with a fixed-order all-reduce, bounded against one card;
- a tensor placement that can say one card, replicated or split stays in the design;
- `burn-collective` is a fallback only.

The full text is in `git show 3e4cefd:docs/plans/roadmap.md`, "Track 3".

- [ ] **F2: one Ethernet hop** with the E1 movers, 4-16 KiB, no host. The go/no-go for tensor parallelism (kill ~20 us; tt-metal's all-gather ~5 us). F13: the QSFP topology.
- [ ] **Per-card sessions on every tile, with a spread** (a `Fabric` opens one tile per chip today).
- [ ] **Fast card-to-card `to_device` over Ethernet** (a host round trip today).
- [ ] **Tensor placement:** one card, replicated, split.
- [ ] **Per-card submission;** cross-card send/wait entries in trace streams.
- [ ] **Tensor parallelism** for models past one card (`N`-split products and all-gather; [sram-residency-and-multi-card-decode.md](../proposals/sram-residency-and-multi-card-decode.md)). Qwen3.8-27B is local.
- [ ] **Data-parallel training,** B14 `DistributedBackend`: a host all-reduce (the oracle), then the Ethernet ring.
- [ ] **Pipelining** over card-to-card `to_device`; micro-batching later.
- [ ] **9.11:** a device-resident mesh with chips concurrent, and `copy_into` on the mesh.
- [ ] **Mesh training traces** (mesh inference traces work, step147).
- [ ] **Ethernet pipelining:** both 800G links reach 27 GB/s (27% of the ceiling).
- [ ] **The two-card accuracy gap** (0.9195 against 0.9196 after step 32).
- [ ] **Packed BF16 over the mesh** (widened to F32 today).
- [ ] **Validate on a physical p300.**

## Models: what is left beyond W5

- [ ] **Qwen3.5-4B** ([new-architectures.md](new-architectures.md)):
  - The delta-rule step in fewer ops: ~25 non-product ops a DeltaNet layer, 776 at 16.8 ms a step; tt-metal ~10.
  - The tied head stored BF16 as its only copy, once the BF16 lookup lands (W1). A second copy breaks the scaling rule.
  - Deal evenly only where the uneven deal is not already at the card's read rate: ≤ 0.2 ms; bracket under the 0.5% rule.
  - The trace build: 644-1,487 s, host-bound, cause not found.
  - Chunked (WY) prefill: 3.7 s for 256 tokens today.
- [ ] **Gemma 4 E4B (Nemotron-3.5-Bolt) and MOSS-Image traced.**
- [ ] **Long context** ([long-context.md](long-context.md)):
  - past 32K;
  - `attention::fits` is L1-bound (head_dim 64: ~864 positions; GQA groups x dv/32 > 32 never fit);
  - flash-decode, parked under the reference contract and to be revisited under the performance contract ([split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md)).
  - Qwen3.6/3.8 need DeltaNet with gated output, partial M-RoPE, MoE and 52/89 GB of weights.

## Burn backend and conformance

- [ ] **Fused slicing for `MatmulMode::Accurate`** (19-64x the fast matmul; 127 jobs a `[64,96]@[96,48]`, 64-86% element-wise):
  - a slice SFPU kind;
  - a cheap NaN/infinity classification kind;
  - one Lo matmul over `k`-concatenated slices.

  Then decide whether Accurate becomes the reference contract's matmul default (the 21 F32 and 3 F16 `precision` entries hinge on it). [log](../learnings/performance-log.md#accurate-matmul-cost-2026-10-12-card-0)
- [ ] **A faster exact matmul** (low priority): profile, then a `k`-loop of broadcast multiply-adds (also lifts the chunk limit).
- [ ] **Re-derive the `step68` and `step85` matmul bounds** with step158's group-alignment term.
- [ ] **Conformance follow-ups** ([conformance-to-feature.md](conformance-to-feature.md)):
  - silicon lists for `--fusion`, `--exact`, `--accurate`;
  - the BF16 CTC gradient (-0.4609375 against Flex's -0.45898; trace the chain's dtypes);
  - CI `--strict`;
  - `check-no-flex-in-backend` over ndarray/cpu/candle/nalgebra;
  - refuse the host `StdRng` fallback in `float_random`/`int_random`;
  - the passing F16 modules into `--smoke`;
  - the exact profile's timeouts on silicon.
- [ ] **Bool scalars under `Fusion<TtBackend>`** (burn-fusion 0.21.0 `ScalarIr::to_relative` `todo!()`): refuse by name until upstream fixes it.
- [ ] **Gather without staging:**
  - packed tables (with W1's BF16 lookup);
  - rank 3+;
  - `select` on other dimensions;
  - per-element `gather`;
  - `deform_conv2d`'s corner gathers (2^24 refusal).
- [ ] **Separable `interpolate`** (measure on a real model first); F16 `interpolate` is refused.
- [ ] **B7 batched readback** (`download_many`, batched `tr_execute`).
- [ ] **The profile mode:**
  - prefill chunks replay too few times to be covered whole (32-46%);
  - mesh engines refuse the mode.
- [ ] **To verify, then close or carry:**
  - prefill chunks no longer run `mask_where` past the cache (likely fixed by D1).

## Device-side backlog

- [ ] **I-cache:**
  - hot reload of running cores;
  - `load_and_start_staged` (`SuppressZeroing`/`WaitOutZeroing`, the silicon pair);
  - multi-core loader exclusion;
  - a measured I-cache capacity a core;
  - cross-core invalidation;
  - what occupies the upper 4 KiB of a T-core's local-RAM window.

  T1's own invalidation is shipped. Evidence: [checklist](../completed-plans/phases-0-10-checklist.md) Phase 2.
- [ ] **Ask whether ttsim models the documented RISC-V hardware bugs** (open question 7).
- [ ] **Baby-RISC-V hazards #1, #4, #5, #6, #8 and #10,** neither encoded nor probed.
- [ ] **`Dst` exclusivity** across the three Tensix threads as a type-level guard.
- [ ] **Silicon verification of Blackhole behaviour with only Wormhole pages** (check step93-108's coverage first).
- [ ] **Negative controls** for the step79 scan, step80 integer reduction and step81 rounding gates.
- [ ] **step143 FP16 `measured::SILICON` constants** ("PENDING MEASUREMENT"; step140/141 cite a pending lane constant too).
- [ ] **Re-sync the pinned ISA spec** (`PINS.toml`, still `f848eb66`).
- [ ] **BFP `a` variants, INT8/UINT8, portable `QTensorOps`** ([burn-native-cutover.md](burn-native-cutover.md)).
- [ ] **F18:** an ARC firmware version gate (tt-metal gates on 18.10.0, 19.8.1, 19.12.0). **F19:** reconcile upstream errata (tt-metal#16439, SYS-1419, the VC leak).
- [ ] **Blackhole di/dt run-to-run non-determinism** (tt-metal `tests/didt/`): if a bit-exact gate fails once and passes on rerun, check it against this before the code. Record any occurrence in the [operating notes](../learnings/silicon-operating-notes.md).
- [ ] **`.ttinsn` open questions** (low):
  - T0/T2 spans beside T1's;
  - translating on one core and invalidating another's cache;
  - the L1 data cache on the push loop;
  - kinds other than `gelu`.

## Developer loop and repository hygiene

- [ ] **One target directory a feature set** (`CARGO_TARGET_DIR` per `--profile` / `fast-reads` / `encoded-guard` in `xtask/src/silicon.rs`): ~2 min rebuilds today.
- [ ] **Weights loaded once a session** (a 1.7B reload costs 60-90 s a run).
- [ ] **One command a change:** pick and run the affected gates on card 0, ttsim spikes first.
- [ ] **A single-card silicon filter must not open the other card:** two-chip tests (`with_mesh_device`) open cards `0..chips` whatever `--device` says ([tt-metal-parity.md](tt-metal-parity.md) §6, issue 12).
- [ ] **F20-F26:**
  - the simulator suite on branch pushes;
  - LICENSE and the CC BY-ND question;
  - a pinned toolchain;
  - typed errors for `session.rs` unwraps;
  - tighter ship checks.

## Long-term (not scheduled)

- [ ] **B0** `tracing` targets and per-reason attribution.
- [ ] **B4** honest `dtype_usage`, dtype fallback warnings.
- [ ] **B9** `name`, `memory_cleanup`, persistent allocations, OOM retry.
- [ ] **Silicon gates on a self-hosted nightly runner** (CI runs the simulator only).
- [ ] **Public API and docs review;** firmware embedding for a crates.io build.
- [ ] **Upstream PR to `burn-dispatch`;** a `burn-router`/`burn-remote` gate for `BackendIr` (B12 is implemented, its gate never run).
- [ ] **Serving:** tokenizer, chat template, scheduler, continuous batching, API. After W3, if in scope ([tt-metal-parity.md](tt-metal-parity.md) §8).

## Parked or decided against

Not open work. Each has its record and retry condition in [failed-approaches.md](../learnings/failed-approaches.md):
- split-K and flash-decode under the reference contract;
- X9's two-copy weight layout;
- layer sub-traces;
- the balanced and rate-sized deals;
- bank-affine placement;
- edge credits and spatial pipelining at batch 1;
- register epilogues on the 1.7B;
- `opt-level = 2`;
- weights resident in L1.

## Doc map

| Need | File |
|---|---|
| Strategy, the W program | [tt-metal-parity.md](tt-metal-parity.md) |
| Numeric contracts | [ADR 0005](../adr/0005-every-op-has-a-numeric-contract-schedules-choose-only-among-bit-equivalent-options.md), [ADR 0006](../adr/0006-two-numeric-contracts-reference-and-performance.md) |
| Design below Burn | [dataflow-compiler.md](../learnings/dataflow-compiler.md) |
| Batch-1 decode, what bounds a product | [decode-bytes-in-flight.md](decode-bytes-in-flight.md) |
| Prefill operand reuse | [prefill-operand-reuse.md](prefill-operand-reuse.md) |
| Long context | [long-context.md](long-context.md) |
| New architectures | [new-architectures.md](new-architectures.md) |
| Traces | [traced-execution.md](traced-execution.md) |
| Fusion | [kernel-fusion.md](kernel-fusion.md) |
| Burn parity and coverage | [burn-backend-parity.md](burn-backend-parity.md), [burn-op-coverage.md](burn-op-coverage.md), [burn-op-notes.md](burn-op-notes.md) |
| A new op | [new-op-definition-of-done.md](new-op-definition-of-done.md) |
| A failing conformance test | [conformance-to-feature.md](conformance-to-feature.md) |
| Current numbers, history | [performance-scoreboard.md](../learnings/performance-scoreboard.md), [performance-log.md](../learnings/performance-log.md) |
| Achieved, and tried and dropped | [achievements.md](../completed-plans/achievements.md), [failed-approaches.md](../learnings/failed-approaches.md) |
| Hardware and simulator facts | [silicon-operating-notes.md](../learnings/silicon-operating-notes.md), [ttsim-divergence.md](../learnings/ttsim-divergence.md) |
| The earlier tt-metal comparisons | [tt-metal-comparison.md](../learnings/tt-metal-comparison.md), [tt-metal-concepts-review.md](../learnings/tt-metal-concepts-review.md) |
