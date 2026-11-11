# Parity with tt-metal: review and improvement program

**Review of 2026-11-09.** A zoomed-out review of the whole stack against tt-metal:
- the firmware;
- dispatch;
- the dataflow compiler;
- matmul and numerics;
- sampling, batching and training;
- multi-card;
- the docs and process.

It is the plan behind the roadmap's tracks W0-W10 ([roadmap.md](roadmap.md)).

**Where the evidence comes from:**
- This repository at `3e4cefd` (branch `traced-generation`).
- tt-metal's source at `/mnt/nvme/tt-metal`, `a11d86c19d3` (`v0.81.0-dev20261013-1`), including its git history.
- A same-card tt-metal baseline on card 1 (§2.3).

The earlier comparisons, [tt-metal-comparison.md](../learnings/tt-metal-comparison.md) (2026-10-26) and
[tt-metal-concepts-review.md](../learnings/tt-metal-concepts-review.md) (2026-10-01), are not repeated here. This
document corrects them where tt-metal's commit history says otherwise (§2.4).

**The owner's decisions behind it (2026-11-09):**
1. **Two numeric contracts** ([ADR 0006](../adr/0006-two-numeric-contracts-reference-and-performance.md)): the bit-exact reference contract for conformance and training gates, and a deterministic performance contract as the inference default.
2. **The parity target** is an 8B-class model on one p150, at batch 1 and batch 32.
3. **Training targets** LLM fine-tuning (LoRA/SFT) of a 0.5-2B model on one card.

Approaches already tried and dropped are in [failed-approaches.md](../learnings/failed-approaches.md). What has been
achieved is in [achievements.md](../completed-plans/achievements.md).

## 1. Verdict

**The engineering is strong.** The core is a bit-exact, single-card, batch-1 inference engine:
- it decodes a 1.7B model at ~101 tokens/s;
- on the same card and model, that is **faster than tt-metal at batch 1** (9.90 ms against 10.55 in tt-metal's BFP4/BFP8 performance mode, 11.3 all-BF16 and 12.24 accuracy; §2.3);
- it has a disciplined correctness culture, and no C++ or Python in the shipped path.

**The strategy, not the engineering, is what keeps it from tt-metal everywhere else.** On the same card (§2.3), tt-metal:
- serves 32 users at 1,289 tokens/s, where we serve one at ~101;
- prefills 2.2-3.1x faster;
- runs prefill-shape products 2.4-8.8x faster at the same HiFi4, and 4-22x at its default fidelities.

Three choices made early now bind more than anything left to schedule:

1. **The numeric default.**
   - Every product runs TF32×TF32 at HiFi4 into FP32 `Dst`.
   - Every activation is a 4 KiB F32 tile. BF16 is storage only: BF16 arithmetic widens to F32 through separate cast ops (`burn-tt/src/ops.rs`, `cast_native`; `float_matmul` refuses non-F32 operands, `ops.rs:2559`).
   - tt-metal runs BF16 activations, BFP8 weights and HiFi2/LoFi with packer L1 accumulation.
   - This is most of the prefill gap and part of the decode gap.
2. **One workload.** Batch-1 decode of a 1.7B model was the target for three weeks.
   - At 9.90 ms it is ~1.37x off its BF16 weight floor.
   - With the weights free, the step is 5.33 ms.
   - The rest is many sub-0.3 ms terms. Every hypothesis killed since 2026-10-28 lived there.
   - We already lead tt-metal there.
   - tt-metal's published single-card numbers are 8B-class, at batch 1 and batch 32. Its lead is in batching.
3. **No operand multicast.** Prefill products reach operand reuse by pulling strips from holder units (D5).
   - The best is ~30 TFLOPS against tt-metal's 168 at the same HiFi4 and 240-554 at its default fidelities.
   - The multicast primitive exists (`WRITE_L1_MCAST`). It was rejected for a different pattern: every producer multicasting to all readers.

None of the three is a hardware limit, and the stack's machinery already has most of the parts:
- `SrcRoute::Bf16FromBf16` and `SrcRoute::Bfp`;
- `Fidelity`;
- derived fidelity;
- `WRITE_L1_MCAST`;
- typed channels with credits;
- TileOp.

The program below (§5) turns them into defaults, in an order where each step is measured against tt-metal on the same
card.

## 2. Where we stand

### 2.1 Our numbers (card 1 unless noted; current rows are in the [scoreboard](../learnings/performance-scoreboard.md))

| Workload | Ours | Derived | Contract |
|---|---|---|---|
| SmolLM2-1.7B decode, batch 1, 32 / 256 tokens | 9.90 / 10.95 ms (`1794024534`-`1794025528`) | ~343 GB/s of BF16 weights (3.4 GB) | reference, bit for bit |
| Qwen3.5-4B decode, batch 1 | 62.48 / 62.84 ms (`1794027144`) | ~21.4 ms is the greedy pick; the products ~19.5 ms | reference |
| SmolLM2-1.7B prefill, 32 / 128 / 256 tokens | 1,075 / 3,040 / 3,827 tokens/s (`1793549551`) | ~13 TFLOPS at 256 | reference |
| Products at prefill shapes, through Burn | 30.4 / 24.7 / 10.9 / 14.5 TFLOPS (`1793388902`) | ~18% of a derived HiFi4 peak | reference |
| ResNet-18 / ViT-Tiny, traced, batch 1 | 12.7 / 7.7 ms | — | reference |
| Traced op fixed cost | ~4.1 us an op | tt-metal 1.29 us minimal | — |
| MNIST / transformer training, traced on every tile (the default since 2026-11-09) | 1.31 / 3.75 ms a step (eager, one tile: 2.17 / 8.52) | burn-flex 2.3-2.6 / 3.9-4.4 (CPU, 12 cores, same runs) | reference |
| GDDR reads, card | 487-511 GB/s with large entries | 95-99.8% of 512 | — |

### 2.2 tt-metal's published single-card Blackhole numbers

| Workload | tt-metal | Source |
|---|---|---|
| Llama-3.1-8B, p150, batch 1, greedy argmax | ~38.5 t/s (~26 ms a token); ~22 t/s through the full sampler | commit `0a0510e7fc1` (#48886) |
| Llama-3.1-8B, p150, batch 32 | 42.0 t/s/user (23.76 ms a step), TTFT 53 ms, top-1 91.2 / top-5 98.6 | `models/model_targets.yaml:694-701`; commit `cc9c295fdf0` |
| Decode products, 8B, M=32, p150b, BFP8-HiFi2, 2 readers a bank | W1/W3 129 us (454 GB/s), W2 127 us (462 GB/s), QKV 65 us | commit `f2c0b2623fe` |
| GEMM, p150, 11x10 grid | 239.5 TFLOPS BF16-HiFi2; 472 BFP8-LoFi; 517 BFP4-LoFi; 168.55 HiFi4 | `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:339-430` |
| ResNet-50, p150 | 11,864 / 14,958 img/s at batch 16 / 32 (BFP8, LoFi) | `.../resnet50/blackhole/tests/test_perf_device_resnet50.py:16-21` |
| ViT-base, p150 | ~3,700 img/s at batch 10 (BFP8, HiFi2) | `.../vit/blackhole/README.md:26` |
| Op dispatch, traced | 1.29 us minimal | `pgm_dispatch_blackhole_golden.json` |
| Compute grid | 110 Tensix (11x10; a column goes to dispatch) | `dispatch_core_common.cpp:25-40` |

**What the comparison says:**
- **Per byte, our batch-1 decode is in tt-metal's band.** Ours is ~343 GB/s; tt-metal's 8B at batch 1 is ~330 GB/s (8.5 GB of BFP8 weights in ~26 ms; derived).
- **But tt-metal moves half the bytes** a parameter (BFP8, 1.0625 B), and batches 32 users in about the same step time.
- **The same-model, same-card comparison is §2.3.** We are ahead at batch 1 and far behind at batch 32 and in prefill.

### 2.3 Same-card baseline (card 1, 2026-11-09)

**What ran:**
- tt-metal v0.77 (`d53c2a71fb0`), fenced to card 1, AICLK 1350 MHz, 110-core compute grid;
- the same SmolLM2-1.7B checkpoint through `tt_transformers`' `simple_text_demo`, plus product benches at the model's shapes.

**Where the evidence is:**
- full tables and conditions: [performance-log.md](../learnings/performance-log.md#tt-metal-on-the-same-card-smollm2-17b-and-its-products-2026-11-09-card-1);
- the parity table: the [scoreboard](../learnings/performance-scoreboard.md#parity-with-tt-metal-same-card);
- scripts: `crates/tt-tests/bench/ttnn_parity/`.

| | Ours | tt-metal |
|---|---|---|
| Decode, batch 1 | **9.90 ms** (BF16 weights, bit for bit) | 10.55 performance (BFP4/BFP8), 11.3 all-BF16, 12.24 accuracy |
| Decode, batch 32 | — | 24.82 ms a step, **1,289 tokens/s** (40.3 a user) |
| Prefill, ~128 tokens | 3,084 tokens/s (2026-11-09, `1794260351`; §6 issue 1 settled: no regression) | ~6,600 tokens/s (TTFT 18.3 ms) |
| Products, 24 layers + head (derived) | inside the 9.90 ms step | 7.34 ms BF16-HiFi2, 11.61 BF16-HiFi4, **4.22 BFP8-LoFi** |
| GEMM 2048x8192x2048 | 14.3 TFLOPS (TF32-HiFi4) | 117 BF16-HiFi4, 204 BF16-HiFi2, 315 BFP8-LoFi |
| Qwen3.5-4B decode, batch 1 ([log](../learnings/performance-log.md#tt-metal-on-the-same-card-qwen35-4b-2026-11-09-card-1)) | 62.5 ms host-timed, BF16 weights; ~21.4 of it the on-card greedy pick, ~41 without | 38.0 ms on the card, 44-46 host-timed (`qwen36` path; BFP4 MLP, BFP8 rest, LoFi; argmax on the host) |

**What it changes:**
- **Batch-1 decode is not where the gap is.**
  - tt-metal computes its decode products on 8 DRAM-reader cores. At HiFi4 they are compute-bound: 11.6 ms of products alone.
  - Ours spread over 64-96 units and stay bit-exact.
  - The batch-1 schedule work bought a real lead. Stopping it costs nothing against tt-metal.
- **The prize for W1 at batch 1 is the bytes and the math.**
  - BFP8-LoFi products are a 4.2 ms floor on tt-metal's schedule.
  - On ours, with columns over 64-96 units, the floor should be lower.
- **Qwen3.5-4B is behind only by our greedy pick.** Without it our step is ~41 ms at BF16 against tt-metal's 38.0 on the card at about a third of the weight bytes; tt-metal's DeltaNet path is not bandwidth-bound. W4 alone puts us ahead host-timed; W1 widens it.
- **The gaps are W3 (batching: 13x throughput) and W1+W2 (prefill: 2-3x end to end, up to 22x per product).**

### 2.4 Corrections to the earlier comparison (from tt-metal's commit history)

- **tt-metal's decode matmul runs on 8-24 cores, not 64-104.**
  - In the DRAM-sharded factory, compute sits on the DRAM reader cores: `per_core_N_compute = div_up(N, num_workers)` (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:174,206-209`).
  - The 64-core grid only holds the L1-sharded activation, multicasts it, and receives the result.
  - Our F27/H17 conclusion ("a unit's read rate does not bind") is the mirror image: at M=32 one core absorbs at most ~45-90 GB/s of BFP8, so tt-metal needs 2 readers a bank.
- **Sampling costs ~19.5 ms a token through tt-metal's full sampler** on P150 at batch 1, against ~0 for its multicore argmax fast path. Published tokens/s assume greedy argmax.
- **tt-train exists.** It trains on a single p150:
  - GRPO on Qwen3-0.6B;
  - fused forward/backward SDPA, RMSNorm, cross-entropy and SwiGLU;
  - AdamW with FP32 master weights;
  - BF16 tensors.
- **Blackhole run-to-run non-determinism is documented.** Marginal dies give matmuls that are not bit-exact under di/dt load from ~4 co-active cores (`tests/didt/test_matmul_run_to_run_determinism.py`). A nightly gate runs on `bh_p150b_civ2`. Our bit-exact gates on 120 co-active units should expect it on some dies: §6, issue 9.
- **Untuned `ttnn.matmul` defaults to LoFi,** not HiFi2, when given a program config or grid (`matmul_device_operation.cpp:2938-2947`).

## 3. Architecture, layer by layer

| Layer | Ours | tt-metal | Verdict |
|---|---|---|---|
| **Kernel model** | Fixed resident interpreters: B reads, NC writes, T0-T2 push host-made Tensix streams from a ~240 KiB word-keyed L1 program cache. T1 runs host-made, host-validated fused `.ttinsn` code | Per-op JIT-compiled C++ (reader/compute/writer); compute compiled 3x (unpack/math/pack); disk cache; cold compiles of minutes | **Keep ours.** At 64+ cores reads reach 94-97% of the card, so entry cost does not limit. No kernel compile latency. The cost is B's image cap: four "firmware diets" in a month. W9 |
| **Dispatch** | Host builds lists, rings over PCIe, B orchestrates; traces replay from GDDR via `CALL`; ~4.1 us an op traced, 26-50 us eager | Prefetch/dispatch cores (a Tensix column); 1.29 us an op traced; 2 CQs to overlap I/O | Behind on fixed cost, ahead on cores (120 against 110). Matters most for small-op-heavy steps. W6/W7 |
| **Dataflow between cores** | Per-op GDDR gathers; within a unit, CB-like credits; between ops a barrier (pruned in captures); operand reuse by pulling from holders; residency planned after the fact from captured words | Sharded L1 tensors persist across ops; in0/in1 multicast gated by semaphores; reshards | **The largest structural gap.** No L1-sharded activation type and no operand multicast in products. W2 |
| **Matmul** | One block-stream kernel family with Dst subblocks and half-sync; measured blocks in captures; thin one-row products stream K over 64-96 units; ~10 entry points | DRAM-sharded decode (compute on the 8 reader cores, in0 multicast); 2-D multicast or `minimal_matmul` for prefill | **Ours is better at M=1.** Thin rows, no 32-row padding, and compute spread over many units: tt-metal's BF16-HiFi4 decode products are compute-bound at ~295 GB/s (§2.3). Theirs is better at M>=32 and in prefill, 2.4-22x. W2, W3 |
| **Numerics** | Reference contract only: HiFi4, TF32, FP32 `Dst`, F32 activations, bit for bit | BFP8/BFP4 weights, BF16 activations, HiFi2/LoFi, packer L1 accumulation; top-k and PCC gates | Two contracts (ADR 0006). W1 |
| **Sampling** | Greedy argmax on the card; for vocabularies over 65,536, eight ~2.7 ms parts (Qwen3.5: ~21.4 ms, a third of the step); top-k/top-p sort ~112 ms | Multicore argmax; top-k over the grid; a sampling core per user | W4 |
| **Batching / serving** | Batch 1 only; no paged cache, scheduler, tokenizer or API | Batch 32, paged attention, vLLM integration | W3; serving after |
| **Training** | Burn's autodiff op by op and Burn's SGD as device ops; eager by default. A traced step (`TracedTrainingStep`: forward, backward, update, write-back) and every tile work, opt-in until 2026-11-09 (every tile now the default; traced still opt-in) | tt-train: fused fw/bw kernels, AdamW-FP32 master, LoRA/SFT/GRPO, DDP/TP/FSDP | W8 |
| **Multi-card** | Matmul, conv and attention split over Ethernet as a proof of concept; no all-reduce | Fabric, CCL all-gather ~5 us, TP/DP for 70B on Galaxy | W10, after W1-W3 |
| **Defaults** | Burn opened one tile (`TileChoice::First`) until 2026-11-09; now every tile (`Topology::single`); one spread rule (`Spread::All`) for eager and traced; 73 `TT_*` knobs in shipped crates | Device opens the full grid; per-model JSON configs | W6 |
| **Code shape** | `Session` 11.8k lines and 108 fields; `tensor.rs` 8.9k; 56 process-wide atomics; 54 probe hooks; TileOp covers eltwise/reduce/matmul, while rms_norm, rope, argmax and gather_rows are hand builders | Device-op structs, program factories, Metal 2.0 `ProgramSpec` | W7 |

### What the dataflow compiler (ADR 0003) became

**The design document** ([dataflow-compiler.md](../learnings/dataflow-compiler.md)) is sound: L0 graphs, L1 tile-generic ops, L2 channels, L3 machine form; transforms written once; every principle tagged with evidence.

**In code it is partial:**
- TileOp (5.3k lines) ports pointwise, reduction and matmul word for word.
- Cross-op decisions (residency, barrier pruning, prefetch anchors, measured blocks) work on **captured machine words**, after the fact, not on an L2 dataflow graph.
- So the transforms that need a graph cannot be written once: multicast trees, sharded activation layouts between ops, operand lifetimes across ops.

**The F-stages bought maintainability.** F1-F3 were performance-neutral refactors. They also bought a long tail of decode-schedule hypotheses, mostly killed.

**Recommendation:**
- Keep ADR 0003's direction.
- Spend the compiler effort where a graph pays: an L2 pass that assigns **sharded L1 layouts and multicast channels between consecutive products** (W2).
- Port the remaining hand builders only as they are touched (W7).

### What the firmware approach costs and buys

**Keep:**
- **The interpreters:**
  - Per entry they cost 359 cycles in the safe image, against 90-106 for a bare loop. The `fast-reads` image is 91-175.
  - At scale the data path, not the entry, decides: 487-511 GB/s card reads.
- **The B-reads/NC-writes split.**
- **The host-validated `.ttinsn` path for T1.** It is the stack's JIT and has no compile latency.

**The cost:**
- B's image is capped by its L1 slot.
- Each new capability is a negotiation for bytes: four diets, and `opt-level` changes measured on end-to-end models.
- If W2 (multicast products) or W3 (batched attention) needs more B code than its ~10-14 KiB of room, do not diet again. Make cold op families loadable through the same validated path T1 already uses (W9).

**Also undecided:** which read mode a deployment ships, safe (359 cycles an entry) or `fast-reads` (91-175). W9 decides it.

## 4. Keep, stop, change

**Keep:**
- **Correctness discipline:** independent oracles, negative controls, derived bounds, traced checked against eager, pinned inputs.
- **Device residency and traces.**
- **The profile mode** (ADR 0002), at zero cost unless built in.
- **Architecture-generic LLM machinery** (ADR 0001).
- **The scaling rule:** a default holds a weight once.
- **The fused `.ttinsn` push.**
- **The hypothesis register with predictions and kill criteria:** it is why the kills were cheap.
- **Thin one-row products, the position-major cache, weights laid by column, BF16/BFP storage.**

**Stop:**
- **Schedule tuning of batch-1 decode on the 1.7B under the reference contract.** New items there need a predicted gain of at least 3% of the parity workload's step (§7).
- **Treating "bit for bit" as the only acceptable result for inference.** It remains the reference contract and the gate for every schedule *within* a contract (ADR 0005's rule).
- **Keeping opt-in knobs that lost.** Delete them with their code: [failed-approaches.md](../learnings/failed-approaches.md), "Knobs to delete".
- **Growing roadmap items into logs.** The roadmap holds open work; evidence goes to the performance log and achievements.

**Change:**
- **Default to the grid.**
- **Default inference to the performance contract.**
- **Measure against tt-metal on the same card** before and after each W item.
- **Move the headline target** from the 1.7B at batch 1 to an 8B-class model at batch 1 and 32.

## 5. The improvement program

Each item has:
- a predicted end-to-end gain (derived, to be bounded by its first step);
- a first step;
- a kill criterion;
- its gate.

They run in this order; W1-W4 can overlap.

### W0. Same-card parity scoreboard

- **What:**
  - tt-metal (a pinned built tree, `TT_VISIBLE_DEVICES` fenced to one card) runs the same shapes and, where it can, the same model as we do, on the same card.
  - The scoreboard gets one parity table: ours, tt-metal's, the ratio, and each side's contract.
- **First step:** done 2026-11-09 for microbenches, decode and prefill products, and SmolLM2-1.7B at batch 1 and 32 (§2.3).
- **Then:** add the 8B-class model when W5 lands. Re-run with each W item.
- **Gate:** `crates/tt-tests/bench/ttnn_reference.py` and the parity runner as a dev-only xtask, `cargo xtask parity`, never in a shipped crate.

### W1. The performance contract (ADR 0006)

- **What:**
  1. **Native BF16 arithmetic:**
     - BF16 edges unpacked straight to `Src` (`SrcRoute::Bf16FromBf16` exists);
     - results packed to BF16 slots (2 KiB tiles);
     - no `cast_native` round trip;
     - `float_matmul` and the element-wise wrappers accept BF16.
  2. **BFP8 weights by default** under this contract (the storage exists, [mixed-bfp-storage.md](../completed-plans/mixed-bfp-storage.md)); BFP4 for gate/up as a named option.
  3. **The fidelity rule:** derived fidelity (`TT_DERIVED_FIDELITY` exists), capped at HiFi2 for BF16×BFP8, and LoFi for BFP4.
  4. **Packer L1 accumulation** for K-blocked products, its partial format stated.
  5. **A contract switch:** `TtConfig::numerics(Reference | Performance)`, captured into traces.
- **Predicted gain:**
  - **1.7B decode:**
    - **Bytes.** BFP8 weights are 1.81 GB, a 3.9 ms floor at 470 GB/s.
    - **Operands.** BF16 activations halve `A` and the small ops' bytes.
    - **Math.** HiFi2/LoFi cut the products' unpack-and-math bound. (Measured 2026-11-09: no such bound at `m` = 1; fidelity moves no decode product there, below.)
    - **Prediction:** 9.90 -> 5.5-6.5 ms. tt-metal's BFP8-LoFi products for this model total 4.2 ms on 8 compute cores (§2.3).
    - **Kill:** above 8.0 ms with the products still not read-bound.
  - **Prefill products:** 30 -> >= 60 TFLOPS from BF16 operands and HiFi2 alone, before W2. P7 measured 1.18-1.35x for LoFi with a BF16 `B` even under F32 activations.
- **First step:** a product bench (`matmul_throughput`, `silicon_probe_pair_cost`) at the 1.7B's decode and prefill shapes, with BF16 `A`, BF16/BFP8 `B` and HiFi2/LoFi. Bound the gain before any Burn plumbing.
- **Bounded (2026-11-09,** `silicon_probe_performance_contract` and the 1.7B per weight format, card 1; [log](../learnings/performance-log.md#w1-bounded-the-performance-contract-on-this-stack-before-any-plumbing-2026-11-09-cards-0-and-1)):
  - **The decode kill does not fire.** BFP8 weights alone: 9.83 -> **7.51 ms** a step (BFP4 6.34), with F32 activations at HiFi4.
  - **Fidelity moves nothing at `m` = 1.** It counts in prefill and at `m` = 32.
  - **A BF16 `A` is slower today,** in both:
    - a packed tile is measured whole in `SrcA` only, so a packed `A` (in `SrcB`) takes the faces pair order: prefill at HiFi4 1.4-2.0x slower;
    - a one-row BF16 `A` misses the packed row (`packed_thin_a`) and the weight prefetch: decode products 1.4x slower.
  - **Prefill at BF16-HiFi2:** 25-48 TFLOPS against the predicted >= 60.
  - **H13 killed** (0.94x on the 512-row chunk).
  - **Packer L1 accumulation's ceiling** is up to 24% of a product, only under a packed `A`.
  - **The order of the rest follows:** BFP8 weights (no new kernel); then a packed tile whole in `SrcB` and the one-row path for a BF16 row; then BF16 activations; then packer L1 accumulation.
- **A packed tile whole in `SrcB` (2026-11-10,** card 0; [log](../learnings/performance-log.md#w1-a-packed-tile-whole-in-srcb-a-bf16-a-past-the-fp32-as-rate-2026-11-10-card-0)): bit for bit, licensed. A BF16 `A`'s prefill products at HiFi4 29-70 TFLOPS, 1.13-1.32x the FP32 `A`; at BF16-HiFi2 >= 60 on two of seven shapes. H13's chunk 1.27x (1.34x with BFP8). Packer L1 accumulation's ceiling falls to <= 2.5%, below the 3% rule.
- **The one-row path for a BF16 `A` (2026-11-10,** card 0; [log](../learnings/performance-log.md#w1-the-one-row-path-for-a-bf16-a-its-decode-products-at-the-fp32-as-step-2026-11-10-cards-0-and-1)): a BF16 row packed `[K / 32, 32]` in its own format, read from 32-byte face rows, prefetched and streamed as an FP32 row is, bit for bit. Decode products at `m` = 1: BF16 x BFP8 HiFi2 10.74 -> **6.99 ms** a step's worth (FP32 `A` 7.00), BF16 x BF16 14.33 -> 10.34 (10.19); `down` 178 -> 71-106 us. The prediction (the FP32 row's step) is met within -0.3% to +1.5%; half `A`'s bytes buy nothing measurable, the products being `B`-bound. No model gains until native BF16 arithmetic makes BF16 activations.
- **Native BF16 arithmetic bounded (2026-11-11,** card 0; [log](../learnings/performance-log.md#w1-native-bf16-arithmetic-bounded-before-building-2026-11-11-card-0)): a native BF16 op widens with the expand, computes as today and quantizes once at its pack, so it is bit for bit today's widen-compute-narrow composition (an ADR 0005 schedule choice under both contracts; only the model's activation dtype follows the contract). A BF16 edge equals its widened FP32 edge over all 65,536 patterns (`step270`). The 1.7B's chunk products with a BF16 `A` (BFP8 `B`, HiFi4): 1.19x at 256 rows, 1.24x at 512. **Predicted:** a 256-token prefill ~1.08x (less the quantize's cost, measured with the products); decode ~0 at short context and at most ~5% at 2K (a BF16 cache; attention is not byte-bound). It proceeds because it unblocks W3's `m` = 32 products, the derived fidelity and the BF16 KV cache.
- **Native BF16 through the stack, the 1.7B on a BF16 stream as an option (2026-11-11,** cards 0 and 1;
  [log](../learnings/performance-log.md#w1-burn-tt-on-bf16-and-the-17b-on-a-bf16-stream-an-option-2026-11-11-cards-0-and-1)):
  BF16 slots, element-wise ops and products with BF16 results (bit for bit their compositions), burn-tt's routing,
  BFP storage of BF16. On the 1.7B the BF16 stream's products are the cheaper part of a decode step (4.1 ms of 8.95 on
  the card), but its norms, `silu * up` and the attention block's casts are compositions where the FP32 stream runs one
  op each, so it stays behind the FP32 stream, and it misses `step269`'s top-1 by one position: the default is
  unchanged. Those three fused on BF16 are what judge it again.
- **First box landed (2026-11-10;** [log](../learnings/performance-log.md#w1-bfp8-weights-the-inference-default-behind-the-contract-switch-and-a-model-gate-2026-11-10-cards-0-and-1)):
  - **The switch:** `burn_tt::Numerics`, recorded in traces.
  - **The default:** tt-models' `for_inference` stores BFP8 weights under the performance contract.
  - **The gate:** `step269_performance_contract`, teacher-forced against HuggingFace. BFP8 scores top-1 98.28%, top-5 100%, PCC 0.99968, against the reference's 100% / 100% / 0.999998. The margin is the owner's: 2 points and 0.005.
  - **The 1.7B:** 7.52 ms a step, 0.71x tt-metal's 10.55.
  - **BFP4 `gate|up`** scores 86.6% / 0.93 on the 1.7B, below tt-metal's floors; it is recorded only.
- **Gates:**
  - **Each op:** bit for bit against its specification model at the stated fidelity (step158's group adder, applied per phase), with negative controls.
  - **Each model:** teacher-forced top-1/top-5 against HuggingFace on a fixed prompt set, plus logit PCC. Thresholds are derived from a reference-contract run plus a stated margin; tt-metal accepts top-1 >= 90 and PCC >= 0.86-0.95 for 8B. Record ours with the derivation.
  - **The reference contract** keeps its MNIST golden and conformance lists unchanged.

### W2. Prefill and compute structure: sharded activations and 2-D multicast

- **What:**
  - **A product schedule** in which `A` is multicast along a row of units and `B` down a column (tt-metal's `MatmulMultiCoreReuseMultiCast` / `minimal_matmul` pattern). It is written as an L2 channel transform (`channel::Kind::Broadcast`, credit-gated) over `WRITE_L1_MCAST`, not as pulls from holders.
  - **Activations that stay L1-sharded between consecutive ops**, where a capture can prove the layout: an L2 pass over the op graph, not after the fact over words.
- **Why D5's verdict does not stand here:**
  - D5 rejected *pulling* `B` because of the holder's L1 fan-out.
  - The residency kill rejected *every producer multicasting to all readers*.
  - A rectangle multicast from one sender per strip is neither, and it is what reaches 168-554 TFLOPS on this card.
- **Predicted gain:** prefill products at the 1.7B's shapes from ~30 (W1: >= 60) to >= 150 TFLOPS BF16-HiFi2. A 1.7B prefill at 256 tokens goes from 3,827 to >= 10,000 tokens/s (the roofline row's "realistic").
- **First step:**
  - a probe: one `[2048,2048]@[2048,16384]` product with `A` row-multicast and `B` column-multicast on a 10x11 sub-grid, BF16-HiFi2;
  - TFLOPS against the reads-free ceiling (`1793224072`: 28-30 with today's kernel), to separate the kernel's own rate from the operand delivery.
- **Kill:** below 60 TFLOPS with operands delivered (the kernel binds). In that case the next item is the kernel's push and unpack, not the dataflow.
- **Gate:** a GEMM TFLOPS gate at BF16-HiFi2 and BFP8-LoFi beside tt-metal's on the same card (W0).

### W3. Batched decode

- **What:** promote [batched-decode.md](../proposals/batched-decode.md) to a plan:
  - sequences as the rows of a tile row (batches of 8/16/32);
  - a paged position-major KV cache;
  - per-sequence attention items;
  - on-card sampling per sequence (W4).
- **Why:**
  - A decode step reads every weight once whatever the batch.
  - At M=32 a product's bytes are unchanged and its math grows 32x, which the performance contract's LoFi/HiFi2 absorbs.
  - tt-metal's 8B at batch 32 steps in 23.76 ms, about its batch-1 time.
- **Predicted gain:**
  - At the 8B, a batch-32 step under 24 ms, i.e. >= 42 t/s/user. That is tt-metal's number, and ~1,350 tokens/s aggregate.
  - On the 1.7B, under tt-metal's 24.82 ms batch-32 step (1,289 tokens/s, measured on card 1).
- **First step:** the 1.7B at batch 8 and 32 with a fixed-shape cache. Step time against batch 1 shows how much of the step scales with the batch.
- **Kill:** a batch-32 step above 2x the batch-1 step, which means attention or small ops scale per sequence. Then attention's per-sequence items come first.
- **Gate:** each sequence's tokens equal its batch-1 run under the same contract, bit for bit; a mixed-length batch; a negative control with one sequence's cache swapped.

### W4. Sampling on the grid

- **What:**
  1. **Diagnose the wide-vocabulary greedy pick.**
     - Qwen3.5's eight parts take ~2.7 ms each for ~4 MB of F32 logits, about 1.5 GB/s.
     - The code spreads partials over up to 64 units (`Session::argmax_row`, `session.rs:8483-8499`; clamped in captures by `spread_for`, `session.rs:10704-10716`).
     - The profile attributes each part's "argmax gather" to unit 0 with ~6,084 stamps.
     - Find which is true, then fix the bound.
  2. **One grid-wide argmax for any vocabulary:** partials over every unit, then one choose. The 65,536-column split in `tt-models/src/generate.rs:176-216` goes away.
  3. **Device top-k / top-p** over the grid, in place of the ~112 ms sort.
- **Predicted gain:** Qwen3.5-4B 62.5 -> ~41.5 ms a step (the 21.4 ms greedy pick to < 0.5 ms).
- **Kill:** the pick above 1 ms after the fix.
- **Gate:** the pick equals `argmax` over the whole row bit for bit, including ties, NaN and `-inf` padding, on a 248,320 vocabulary.

### W5. An 8B-class model

- **What:**
  - Llama-3.1-8B-Instruct, or a Qwen 8B-class dense model if the Llama licence blocks it, through the generic Llama loader (ADR 0001);
  - BF16 checkpoint, stored BFP8 under the performance contract;
  - an 8K cache.
  - Add it to `cargo xtask fetch-weights` as `on_request`, and to the model ladder (silicon only).
- **Why:** the only single-p150 LLM figures tt-metal publishes are this size.
- **Predicted:** batch 1 at 8.5 GB of BFP8 weights is an 18 ms floor at 470 GB/s. Matching tt-metal's ~26 ms needs ~330 GB/s, which we stream today. Beating it needs W1's products to read-bound.
- **Gate:** top-1/top-5 against HuggingFace (W1's model gate); a smoke entry.

### W6. Defaults and usability

- **What:**
  1. **The grid by default** (done 2026-11-09): every tile, eager ops at `work:64:4`, captured ops on every unit (`Session::set_capture_spread`). Tests keep naming their gate tile. Left: the tiny transformer's eager step is 10% slower on every tile than on one.
  2. **One typed configuration:**
     - `TtConfig` and `SessionConfig` carry every setting;
     - env vars become a thin, uniformly validated layer (bad values are errors, never a panic or a silent default);
     - the 56 process-wide atomics move into the session.
  3. **A sunset rule:** an opt-in knob that lost is deleted within the change that records its row in [failed-approaches.md](../learnings/failed-approaches.md).
  4. **Auto-capture of a repeated graph** (`Fusion<TtBackend>` already sees it): a Burn user gets trace speed without `Trace`.
- **Gate:**
  - the MNIST golden;
  - conformance unchanged;
  - the model ladder without `TT_TILES`;
  - a test that every `TT_*` knob refuses a bad value by name.

### W7. Code consolidation

- **What:**
  - Split `Session` (11.8k lines) into the program cache, residency, product planning, trace, and transfer modules it already contains.
  - One matmul entry through TileOp. Retire `matmul_on`'s host-staged path and the host-download fallback in `float_matmul` (`burn-tt/src/ops.rs:2611-2660`): refuse by name instead.
  - Port rms_norm, rope, argmax and gather_rows to TileOp as W1-W4 touch them.
- **Kill:** none; it is maintenance. It is done in the changes that touch the code, never as a standalone refactor with no gate moving.
- **Gate:** each port word for word against the builder it replaces, as F3 did.

### W8. Training for LLM fine-tuning

- **What:**
  1. `TracedTrainingStep` as the default for a repeated step (it captures forward, backward, the update and the write-back).
  2. A fused AdamW with FP32 master weights and moments.
  3. Fused RMSNorm, softmax/attention and cross-entropy, forward and backward. The H10 epilogue kill does not carry here: a backward pass removes bytes, not just ops.
  4. A BF16 training contract (BF16 compute, FP32 master), stated as ADR 0006 requires.
  5. LoRA adapters on the generic Llama loader.
- **Target:** LoRA/SFT of SmolLM2-1.7B (or Qwen3-0.6B, which tt-train runs on p150) at a step time recorded beside tt-train's on the same card.
- **First step:** done 2026-11-09 for `tt-mnist`: MNIST and the transformer traced on the grid by default, both faster than burn-flex. Next: the same for any Burn loop (W6 auto-capture), and SmolLM2's fine-tuning step.
- **Gate:**
  - the reference contract's MNIST golden, unchanged;
  - for the BF16 contract, a loss-curve bound derived against an F32 run;
  - an independent oracle per fused kernel.

### W9. Firmware

- **What:**
  - Keep the interpreters.
  - When W2 or W3 needs B code past its room, load cold op families as validated L1 code through T1's `.ttinsn` path, not another diet.
  - Decide the shipped read mode: `fast-reads` + `encoded-guard`, or safe, measured on the parity workload.
- **Gate:** the encoded-read gate and the mover differential gate in both modes (AGENTS.md).

### W10. Multi-card (after W1-W3)

- **What:**
  - Measure an Ethernet hop (F2).
  - Then tensor parallelism across both cards for models past one card, under either contract. Qwen3.8-27B is local.
- **Why later:** the single-card levers above are each worth more than 2x. A second card before them doubles a slow step.

## 6. Bugs and issues found

1. **A possible 1.7B prefill regression: settled 2026-11-09, none.** The ~2,090 tokens/s at 256 tokens were `traced_generation_stream_ceiling`'s rows (its prefill in chunks of 64), which run under the same filter; `traced_generation` reads 3,890 at `HEAD` (`1794260351`). The record below is kept as found.
   - H15 recorded 1,075 / 3,040 / **3,827** tokens/s at 32 / 128 / 256 tokens (2026-11-01, `1793549551`, card 1).
   - Every later run of what looks like the same bench reads ~1,090 / ~2,150 / **~2,090**:
     - the 2026-11-03 C0 probe, "chunks of 64";
     - the `1ff07dc` baseline of 2026-11-04;
     - the firmware diet of 11-06;
     - the even deal of 11-07.
   - The 32-token figure is unchanged.
   - Either the bench's chunking or warm-up differs between those runs (`TT_GEN_CHUNKS`, default 64), or something between `5a1640f` and `f29b104` lost ~45% at 256 tokens.
   - Settle it first: re-run `5a1640f` and `HEAD` with the same command on one card. Bisect if they differ.
   - The scoreboard still quotes 3,827.
2. **The wide-vocabulary greedy pick reads at ~1.5 GB/s** (W4). The docs and the code disagree on its unit count. Confirm with a profiled run (`TT_PROFILE_DETAIL=pipeline`) of one part before changing the code.
3. **BF16 is storage only.**
   - `float_matmul` refuses non-F32 operands (`burn-tt/src/ops.rs:2559`).
   - BF16 element-wise arithmetic widens through cast ops (`cast_native`).
   - A BF16 Burn model pays two extra ops and F32 traffic per op (W1).
4. **Burn's default was one tile** (`burn-tt/src/registry.rs`). **Fixed 2026-11-09:** every tile is the default, eager ops at `work:64:4` and captured ops on every unit; `tt-mnist` trains traced by default. Training now beats burn-flex (MNIST 1.31 against 2.3-2.6 ms a step, the transformer 3.75 against 3.9-4.4). Found on the way: a traced MNIST "regression" bisected to `9977ad8` is the first step's block measurement (3.2 s once), and `tt-mnist --fusion` panics in burn-fusion (roadmap W8).
5. **`float_matmul`'s last fallback stages through the host:**
   - `into_host()`;
   - batch broadcast expanded on the host;
   - one device product per batch element (`burn-tt/src/ops.rs:2611-2660`).

   Confirm it is unreachable on a device engine for any shape the earlier paths accept, or make it refuse by name. AGENTS.md: no staged host compute.
6. **Inconsistent knob validation:**
   - `TT_OPERAND_REUSE` panics on a bad value (`tt-kernels/src/tensor.rs:1998`);
   - `units_from_env` silently uses its default (`tt-kernels/src/session.rs:205-211`);
   - `TT_SPREAD`, `TT_PIPELINE` and `TT_TILIZE` return errors.

   One rule (W6).
7. **Process-wide state:**
   - 56 atomic statics (`DST_SUBBLOCKS`, `BLOCK_LOOPS`, `PAIR_ORDER`, `TRIM_ROWS`, ...) are read once from env and shared by every session in a process;
   - two sessions cannot differ, and a test that sets one leaks into the next (W6).
8. **Untraced ResNet-18 is ~11% slower than recorded**, not bisected (scoreboard, 2026-10-27).
9. **Stale state in docs:**
   - the scoreboard's 1.7B row labelled F32 while quoting BF16-stored weights;
   - `long-context.md` 1.G open but on by default;
   - `layer-sub-traces.md` with open boxes for a retired feature;
   - the roadmap's "1-4 of 120 tiles".

   Fixed in the 2026-11-09 reset.
10. **Bit-exact gates on 120 co-active units can meet Blackhole's di/dt run-to-run non-determinism** on marginal dies (tt-metal's `tests/didt/`). A bit-exact gate that fails once on silicon and passes on rerun should be checked against it before the code is suspected. Record any occurrence in [silicon-operating-notes.md](../learnings/silicon-operating-notes.md).
11. **Reused bit:** `GATHER` bit 2 was reclaimed after a retired meaning (`tt-isa/src/dm/record.rs:39-44`). Traces stored before the change must not replay. Confirm the trace format version guards it.

12. **A single-card silicon run can open both cards.**
    - `cargo xtask silicon --device 1 --filter step12_mnist` ran `the_mlp_trains_sharded_over_two_chips_matching_the_golden`.
    - It opens cards `0..chips` whatever `--device` says (`with_mesh_device`, `crates/tt-tests/src/burn_device.rs`, `Topology::Cards`).
    - A filter meant for one card must not touch the other: skip multi-card tests unless `--device all`, or make the runner refuse them by name.

## 7. Process

- **The big-bet rule:**
  - Every performance item states its predicted end-to-end gain on the parity workload (W0's table) before it is built.
  - Items predicted under 3% of the step go to the backlog unless they unblock a W item.
  - The 0.5% regression rule stays.
- **Measure against tt-metal, not only against our last run.** The parity table is re-run when a W item lands.
- **A kill is recorded once,** in [failed-approaches.md](../learnings/failed-approaches.md), with its retry condition. Its knob and code are deleted in the same change.
- **Documentation stays in its tier:**
  - the roadmap holds open work only;
  - achievements and completed plans are the record;
  - the scoreboard holds current numbers, one line a cell;
  - the performance log holds history.

## 8. Open decisions for the owner

- **The 8B model:** Llama-3.1-8B, which tt-metal publishes, against a Qwen 8B-class model (licence, local weights, architecture spread).
- **The performance contract's model thresholds.** Derived per W1, but the margin over a reference-contract run (tt-metal accepts top-1 ~90%) is a product decision.
- **Serving** (tokenizer, scheduler, API) after W3, or not in scope.
- **Shipped read mode** (W9).
