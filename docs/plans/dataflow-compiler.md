# Plan: the dataflow compiler, staged

**Status:** approved and scheduled 2026-10-28 (owner). Its design principles are in
[dataflow-compiler.md](../learnings/dataflow-compiler.md) (the Bible), and its binding decisions in ADRs
[0003](../adr/0003-a-tile-level-dataflow-compiler-below-burn.md),
[0004](../adr/0004-formats-are-edge-data-converted-only-at-unpack-and-pack.md) and
[0005](../adr/0005-every-op-has-a-numeric-contract-schedules-choose-only-among-bit-equivalent-options.md).
**Branch:** `dataflow-compiler`, in a worktree.
**Silicon:** card 1 only until the owner says otherwise. No `--device all`, and no mesh, multi-card or Ethernet gate.

Every stage opens by writing its hypothesis, prediction and kill criterion into the Bible's register (§13), then
measures its baseline on card 1 at the stage's starting commit. A stage is kept only if its gates are bit-exact and the
prediction holds. Otherwise the refutation is recorded and the stage stops.

The two performance tracks, decode and compute, alternate after the shared foundations, as the owner chose on 2026-10-27. The compute
track's order follows the evidence: products are bound by operand reads (scoreboard, matmul row), so operand reuse
(H1) comes before `Dst` subblocks (H2).

## Stage 0: baselines on card 1

- [x] Release benches on card 1, 2026-10-28 ([log](../learnings/performance-log.md#card-1-baselines-for-the-dataflow-compiler-and-exact-fidelity-2026-10-28-card-1)):
  - matmul at prefill shapes, `[1024,2048]@[2048,8192]` 14.64 TFLOPS (`1793149570`);
  - ResNet-18 / ViT-Tiny traced, host-timed 38.45 / 26.77 ms;
  - decode at `TT_TILES=all`: 135M 6.92 / 7.46 / 8.13 ms (`1793150978`), 1.7B 11.20 / 11.83 / 12.62 (`1793151302`);
  - MNIST 1.398 / 2.882 / 8.290 ms/step (`1793150926`).
  - Prefill tokens/s is in the decode runs' first-token rows.
- [ ] A 1.7B `Step(64)` profile on card 1 (`--profile`), for the card read meter and the tails. This is H4's baseline, taken
  when F4a starts.
- [x] Recorded in the log as card-1 baselines.

## Stage F1: `TileFormat` (ADR 0004), and one matmul for every format pair

- [x] **F1a, the descriptor** (`6dd4444`): `tt_isa::format::TileFormat` with image and slot bytes, family, the
  unpacker's `Src` codes, and the mantissa bits each format brings into `Src`. `full_precision_phases` derives the
  phases from `SrcASrcB.md`'s slices: for one format, the vendor table; for mixed pairs, each operand's bits.
  `TILE_SLOT`/`BF16_TILE_SLOT` are derived from it and asserted equal to 4160/2112. `BfpFormat::slot_bytes` and
  `SrcRoute`'s BFP codes come from it.
  - Still to add, as F2 and F5 need them: `Dst` unpack routes, pack routes and narrowing semantics.
- [x] **F1b, GATHER's `B` by format** (`6dd4444`): bit 29 (a BF16 `B`) became bits 29-31, a format code (FP32, BF16,
  BFP8, BFP4, BFP2).
  - Code 1 is bit 29 alone, so every existing record encodes as before. A packed `B` reads whole slots of its format.
  - The mover's stream cap and the session's L1 ranges read the field.
  - `step172`/`step173` pass in all three read modes on ttsim, and in the safe image on card 1 (`1793150259`).
- [x] **H3 gate** `step238_exact_fidelity`: BFP8 at 2 phases and BFP4/BFP2 at 1 equal HiFi4 bit for bit (groups spread
  over seven binades, signed zeros, K reloads), and BFP8 at 1 phase differs (the negative control). **Passes on ttsim and
  on card 1** (`1793150259`, along with `step172`/`step173`/`step232`/`step233`/`step94`).
  - Still to cover: mixed pairs, once F1c builds them, and NaN/infinity blocks.
  - If it holds on silicon, the default contract's fidelity rule becomes "full precision for the edge formats".
- [x] **F1c, the tensor handle carries its format** (`b9ba4c2`; `step239` on ttsim and card 1 `1793152118`; [log](../learnings/performance-log.md#packed-weights-on-the-block-stream-path-what-bytes-buy-2026-10-28-card-1)). Still to do: route a BFP weight here from burn-tt (today `bfp_matmul` widens a mixed pair), then judge the decode prediction through the model's decode path. In the probe's session one-row products are entry-bound, and the bytes buy nothing there.
- [ ] **H7, whole tiles for packed formats:** BFP weights take the face order today (14.8 TFLOPS against BF16's 22.3 at prefill shapes).
- [ ] **F1c, as first specified** (kept for its detail): Generalize the stored-BF16-weight precedent:
  - `DramTensor::format() -> TileFormat`, with `Elem::Bf16` and a BFP weight as edge formats;
  - `SrcRoute::Tf32WithBf16B` becomes `SrcRoute::Tf32WithB(TileFormat)`, its `b_slot`, `src_formats` and GATHER flags all
    taken from the format.
  - BFP8/4/2 weights beside TF32 activations then take the block-stream path, with its blocking, prefetch and
    pipelining. Today `bfp::matmul` builds one output tile per job with no block reuse.
  - `PairOrder::Auto`'s whole-tile unpacks (P4) are tried for packed formats by a gate, not assumed.
- [ ] **F1d.** One matmul job builder, parameterized by the A, B, out and accumulator edge formats, replacing `tensor::matmul_dram*`, `bf16::matmul` and `bfp::matmul`.
  - Gate: bit-exact against each old builder on every pair it supported, and new mixed pairs (BFP8/4 weights × BF16 or TF32 activations) against the specification model and burn-flex within derived bounds.
  - Then delete the old builders, the `bfp.rs` same-format refusal and their per-format server functions.
- [ ] `step12_mnist`, `cargo xtask conformance --release`, and silicon gates on card 1.
- **Prediction:** a BFP4-weight product is faster than BF16 at decode shapes in proportion to its bytes, at least 1.5× at m = 1. If H3 holds, the compute-bound BFP products run at ≥3× the BF16 HiFi4 rate.

## Stage F2: typed channels and the cross-core credit

- [ ] Add a `Channel` to `tt-kernels`: one producer, N consumers, a depth, and a page of `TileFormat` × tiles. Its kind is `Local`, `Remote`, `Broadcast` or `Gddr`, and its credit words are declared through `l1::Requirements`. The existing `Input`, `Output` and `Transfer` streams are re-expressed as `Local` channels, and every existing gate stays bit-identical.
- [ ] Add the remote credit: a NoC atomic increment into a peer's L1 word, issued after the data's write acknowledgements (T7), plus a local wait.
  - The encoding goes in `tt_isa::dataflow` and `tt_isa::dm`, the firmware in `tt-firmware/src/dataflow.rs`/`mover.rs`.
  - Build both mover modes, and lint with and without `profile`/`encoded-guard`.
  - Run `step172`/`step173` in both read modes.
- [ ] Gates:
  - ttsim, with negative controls: one credit too few stalls and is reported, never races; a credit sent before its data's acknowledgement is caught.
  - An isolated silicon gate on card 1 before any op uses it.
  - A divergence row if ttsim refuses the atomic.
- [ ] Plan-time checks: rate balance and deadlock freedom (every cycle with its reverse credit edges holds enough initial tokens), with a refusal gate for each.

## Stage F3: the tile-generic op (L1) and transforms written once

- [ ] Add `TileOp`: domain with iterator kinds, affine tile maps, edges with `TileFormat` and `OpPadding`, a body of microkernel calls, and a `NumericContract`.
- [ ] Write the passes against it: grid distribution (from `Spread`), L1 blocking (from `block_stream_layout`), allocation (through `l1::Requirements`), channels (F2), the role split, and lowering to `Step`/`Job`/`Work`.
- [ ] Port three clients, one per dataflow class: matmul (from F1), the pointwise SFPU family (`sfpu_eltwise_blocks`, 121 kinds) and a row reduction. Each is bit-exact against its old builder, and the old builder is then deleted.
- **H6 prediction:** no pass contains code specific to one of the three clients. If one does, the description gets the missing concept and the Bible records it.

## Stage F4b (compute): operand reuse across cores, H1

- [ ] Decide D5 by measurement first, on card 1, for one prefill product shape, between:
  - a few holders whose rows or columns are pulled with `READ_L1`;
  - one sender per row or column rectangle multicasting with `WRITE_L1_MCAST` and remote credits;
  - larger blocks re-read from GDDR.
- [ ] Build the winner as F3's reuse pass, so every format and every contraction gets it.
- **Prediction:** `[m,2048]@[2048,16384]` at m = 2,048 from 18.2 to ≥35 TFLOPS.
- **Kill:** GDDR reads drop and TFLOPS do not.

## Stage F4a (decode): channels instead of barriers, H4

- [ ] Derive per-edge channels from the residency plan (`residency.rs`, `Applied::rewrite`) and barrier analysis (`tt_kernels::barriers::analyse`). Keep barriers only where no channel can express the dependency (Bible §7).
- [ ] Let units that finish their share start data-independent reads (the next product's weights) at once.
- **Prediction:** the 1.7B decode step's card read rate ≥400 GB/s (from 328) and the step ≤9.5 ms (from 10.9), bit-exact.
- **Kill:** the step moves <0.3 ms, or small ops lose what products gain.

## Stage F5 (compute): `Dst` subblocks and half-sync, H2

- [ ] Add a `Dst` tiling pass: output subblocks of `h×w ≤ 4` FP32 tiles, and half-sync between math and pack. Relax `retire_batch` (`tt-firmware/src/corpus.rs`) so roles overlap across batches, which needs the configuration to be the same in every batch or fenced.
- **Prediction:** ≥1.4× on the H1 shapes after F4b, bit-exact.
- **Kill:** <1.1×.

## Stage F6 (gated on F4a and F5): fusion through channels and spatial pipelining, H5

- [ ] Fusion through channels: register epilogues and prologues that replay every narrowing rounding (Bible X5), the B13c matmul epilogue fuser, and fusion of L1 and cross-core channels.
- [ ] Spatial pipelining of a decode layer on disjoint core sets, only if H4's tails are not GDDR-queue-bound.

## Not in scope

- Opt-in contracts: split-K, flash attention, packer L1 accumulation, a 16-bit `Dst`, and fidelity below the contract. Each is the owner's call.
- Multi-card schemes.
