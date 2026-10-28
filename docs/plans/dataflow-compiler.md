# Plan: the dataflow compiler, staged

**Status:** approved and scheduled 2026-10-28 (owner). Its design principles are in
[dataflow-compiler.md](../learnings/dataflow-compiler.md) (the Bible), and its binding decisions in ADRs
[0003](../adr/0003-a-tile-level-dataflow-compiler-below-burn.md),
[0004](../adr/0004-formats-are-edge-data-converted-only-at-unpack-and-pack.md) and
[0005](../adr/0005-every-op-has-a-numeric-contract-schedules-choose-only-among-bit-equivalent-options.md).
**Branch:** `dataflow-compiler`, in a worktree.
**Silicon:** cards 0 and 1 (owner, 2026-10-28; card 1 only before that). Each gate is validated on one card, card 0 by
default; benches stay on card 1, where Stage 0's baselines were taken, and a before/after pair always uses one card. No
`--device all`, and no mesh, multi-card or Ethernet gate.

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
  when F4a starts. One exists at F1's close (`1793170516`, BF16 weights, `28ed325`: Step(64) 10.90 ms on the card,
  products 7.64 ms).
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
- [x] **F1c, the tensor handle carries its format** (`b9ba4c2`; `step239` on ttsim and card 1 `1793152118`; [log](../learnings/performance-log.md#packed-weights-on-the-block-stream-path-what-bytes-buy-2026-10-28-card-1)).
- [x] **F1c from burn-tt** (`562b170`, `7daa887`): a BFP weight is one `DramTensor` in burn-tt's one buffer map
  (`DramTensor::as_bfp`; the separate `bfp` map is gone), and any packed weight beside an F32 activation is one product
  (`ops::packed_weight_matmul`). `Plan::stored_weights(format)` and `Llama::with_stored_weights(format)` store a model's
  weights in any packed format; the bench takes `TT_WEIGHT_STORAGE=bf16|bfp8|bfp4|bfp2|f32`. A transposed view stored
  packed is made plain first, so a tied head is read as it lies (`7daa887`: before it, a BFP head was widened at every
  step, 37 ms a 1.7B step).
  - Gate `step240_bfp_weight_storage` (in `--smoke`): synthetic Llamas and SmolLM2-135M with BFP8/4 (and BFP2 on
    silicon) weights generate the F32 model's tokens and logits bit for bit, eager and traced, with no widening; a tied
    model stored as a checkpoint is; unrounded weights differ (the control). ttsim, card 0 `1793160231`, `1793175246`.
  - `step168` was not tightened: its BFP plans store `embed_tokens`, whose lookup widens by design. `step240` holds the
    no-widening claim instead.
- [x] **H7, whole tiles for packed formats** (`eda859f`): confirmed. The licence is the descriptor fact
  `TileFormat::whole_tile_unpack(to, src)`, read by `PairOrder::resolve`; `step239::whole_tile_orders_give_a_packed_weights_bits`
  on ttsim and card 0 alone (`1793160473`). BFP weights at HiFi4 1.51x (`[1024,2048]@[2048,8192]`, 22.4 TFLOPS, BF16's
  rate) and 2.3-2.5x (`[1024,8192]@[8192,2048]`, 26.6-29.1, above BF16's 22.1), card 1 `1793162109` -> `1793167801`.
- [x] **F1c, as first specified:** done by the items above and F1d. `DramTensor` carries its format, the route is the
  two edges' formats, a BFP weight takes the block-stream path, and whole tiles were gated (H7), not assumed.
- [x] **F1d, one matmul builder** (`c5aab9a`, `3bc1d89`, `27c8d38`, `28ed325`).
  - `matmul::Slots {a, b}` through every layout; GATHER bits 22-24 carry `A`'s format (kt is bits 8-21); a packed `A`
    reads whole slots of its format, untransposed, with no thin rows or `pack_row`. Bit 2 (`GATHER_BF16`) is retired and
    refused. `step172`/`step173` pass in the safe, `fast-reads` and `encoded-guard` modes.
  - `SrcRoute::Edges {a, b}`: the `Src` types are derived from the two edge formats; `Bf16FromFp32` stays the one named
    lossy contract. `Session::matmul_dram` is the only entry.
  - Gate `step241_one_matmul_builder` (card 0 `1793162345`): every pair the old builders took, bit for bit against them,
    then the old builders deleted (459 lines: `bf16::matmul`, `matmul_continuing`, `bf16::gather`, `bfp::matmul`,
    `Session::{matmul_bf16, bfp_matmul}`, burn-tt's `Engine::{matmul_bf16, matmul_bfp}` and their twins). Mixed pairs
    are checked against the card's FP32 product of the decoded values (the specification model's oracle), not burn-flex.
    Negative control: `A`'s slot one size off fails it. Card 0 `1793164216`: 143 of 144, the one failure predating the
    branch (`step167`'s BF16 RoPE, fixed in `5b99f6a`).
  - **Derived fidelity is opt-in** (`TT_DERIVED_FIDELITY=1`, `27c8d38`): the phases each edge pair needs equal HiFi4 bit
    for bit for every pair, signed zeros and subnormals included, until a product overflows FP32
    (`step241::derived_phases_are_hifi4_until_a_product_overflows`, ttsim and card 0 `1793163275`). HiFi4 stays the
    default contract.
- [x] `step12_mnist` (8 of 8, the goldens), `cargo xtask conformance --release` (2,973 pass, 52 listed, 0 unexpected),
  and the silicon gates on card 0: `--smoke` 1,236 of 1,236 at `7daa887` (`1793177942`).
- **Prediction:** a BFP4-weight product is faster than BF16 at decode shapes in proportion to its bytes, at least 1.5× at m = 1. If H3 holds, the compute-bound BFP products run at ≥3× the BF16 HiFi4 rate.
- **Result (2026-10-28, card 1): refuted at m = 1.** BFP4 weights make a 1.7B decode step 10.31 / 10.86 / 11.46 ms
  against BF16's 11.25 / 11.89 / 12.64 (1.09x; `1793175898`, `1793165085`). The products' card time falls 7.64 -> 6.45
  ms, but a one-row product is bound by its unpacks and math, not its bytes (profiles `1793170516`, `1793172384`), and
  derived fidelity does not move it (`1793177229`). The routing stays: it reads a quarter of the bytes, is bit for bit,
  and frees GDDR bandwidth for whatever binds next. The compute-bound half is not yet measured (BFP × BFP at prefill).
  [Log](../learnings/performance-log.md#one-matmul-for-every-format-pair-and-bfp-weights-through-the-model-2026-10-28-cards-0-and-1).

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
