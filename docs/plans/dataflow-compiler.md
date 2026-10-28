# Plan: the dataflow compiler, staged

**Status:** approved and scheduled 2026-10-28 (owner). Its design principles are in
[dataflow-compiler.md](../learnings/dataflow-compiler.md) (the Bible), and its binding decisions in ADRs
[0003](../adr/0003-a-tile-level-dataflow-compiler-below-burn.md),
[0004](../adr/0004-formats-are-edge-data-converted-only-at-unpack-and-pack.md) and
[0005](../adr/0005-every-op-has-a-numeric-contract-schedules-choose-only-among-bit-equivalent-options.md).
**Branch:** `traced-generation` (F1 was built on `dataflow-compiler` in a worktree and merged back, `d84ef47`).
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

- **H8** (Bible §13, opened 2026-10-28): a credited handoff between two units costs less than a barrier, ≤ 1.5 us on card 1;
  the in-core streams re-expressed as `Local` channels emit today's words.
- [x] **F2.1, the packet golden first** (`d2024f3`): `session::tests::streamed_packet_words_are_pinned` dumps every
  streamed packet's words for depth-1 and depth-2 regions (apart and shared footprints, past one group), transfers
  of both depths and past one packet, and a mix between lists, against `crates/tt-kernels/golden/stream_packets.txt`
  (`TT_BLESS=1`). It has not changed since.
- [x] **F2.2, `tt_kernels::channel`** (`72f5ce0`):
  - `Channel {producer, consumers, depth, page, kind}`, with `Page::{Batch, Tiles {format, tiles}}` and
    `Kind::{Local(Stream), Remote, Broadcast, Gddr}`. It is not `tt_isa::dataflow::Channel`, the firmware's ownership
    rule.
  - A local channel's credit is the stream's `CB` entry (`Channel::credit`). `streaming_segments` and
    `transfer_segments` make every credit through it, so the literals are gone.
  - `l1::Kind::{Counters, Credit}` through `Requirements::channel`. `enable_dram` declares the three streams' counters
    as channels, the transfer counters now on their own.
  - The golden, the session tests, `step60` and `step33` are unchanged.
- [x] **F2.3, the remote credit** (`656e340`):
  - **`op::CREDIT [x, y, addr, value]`** drains the mover's own requests, then sends one full-width atomic increment
    and waits for its response.
    - B sends on NoC #0, and NC on its writes' NIU.
    - IDs `TXN_CREDIT_B` 9 and `TXN_CREDIT_NC` 10, returning to `CREDIT_RET`.
  - **`op::WAIT_CREDIT [addr, target]`** is a fenced, abort-aware poll, signed modulo 2^32, with wait reason
    `dataflow::CREDIT_WAIT` (40).
  - Both movers run them, on words in `LISTED`, which the card never resets.
  - Host checks: `L1Writes::admits` a credit, `check_pair` lets the writer credit and wait, and `barriers::access`
    treats a credit as a peer-L1 word.
  - `channel::Ring` lowers remote, broadcast (a rectangle, one multicast and a credit each) and GDDR rings to
    `send`/`receive`/`fetch`/`free`.
  - **`Ring::check_send`** is the static ordering check. It refuses a producer list that:
    - writes after its page's credit;
    - writes into an unfreed slot;
    - writes outside its slot;
    - or credits a page not fully written.

    It belongs to the channel's lowering, not `DataMover::check`, because a list alone does not say which writes a
    credit grants.
  - Images (safe): B 35,344 of 43,008 bytes, NC 20,392 of 24,576.
  - Firmware lints pass in safe, `encoded-guard`, `profile` and `fast-reads`.
  - On ttsim, `step172`/`step173` pass in safe, `fast-reads` and `fast-reads`+`encoded-guard`.
- [x] **F2.4, gate `step242_channel_credits`** (`d06c345`, in `--smoke`):
  - Remote, broadcast (2x2) and GDDR rings hand every page over in order across laps, byte for byte.
  - One credit too few holds the consumer at its wait. `channel::stalled` reports have and want, the missing page is
    never copied, and `channel::cancel` fails the list with its code.
  - The host and the tile refuse malformed credits. A planted consumer without its waits copies poison and fails.
  - The isolated run on card 0 came first, the remote ring alone (`1793199629`). Then the full file, 5 of 5
    (`1793199635`). `step172`/`step173` passed in the safe image, 6 of 6 (`1793199651`), and with `fast-reads`,
    together with `step242`, 16 of 16 (`1793199814`).
  - The planted faults of the producer's ordering are `channel`'s unit tests. ttsim cannot race a credit against its
    data's acknowledgement (divergence row 72), and it runs the full-width atomic (row 88), so no new row was needed.
- [x] **F2.5, plan-time checks** (`22bf51f`): `channel::Graph::check`.
  - It gives the least repetition vector from the SDF balance equations, and refuses unbalanced rates.
  - It runs one self-timed iteration with each channel's depth as its reverse edge's credits. A graph that stops is
    refused: a cycle with no initial tokens, or a channel shallower than a firing.
  - Each refusal has a unit test, beside a double-buffered two-unit pipeline that passes.
- [x] **F2.6, H8 measured** (`2f29d9f`, `silicon_bench_channel_credit`, card 1 `1793200301`):
  - a credited handoff is 1.81 us, against a two-unit barrier's 2.66 us polled and 2.59 us released;
  - the prediction (<= 1.5 us) is missed, and the claim (cheaper than a barrier) holds. See the Bible's H8.
- [x] Close-out: `--smoke` on card 0, 1,241 of 1,241 (`1793200324`); `step12_mnist` 8 of 8 on ttsim, goldens unchanged.
- **Not yet built:** no op uses a cross-core channel. F4b (operand reuse) and F4a (channels instead of barriers) are
  the first. `channel::Graph` is not yet run on an op's graph, and a `Gddr` ring's GDDR placement is the caller's.

## Stage F3: the tile-generic op (L1) and transforms written once

**Opened 2026-10-28.** H6's prediction and kill criterion are in the Bible's register. The method:
- a job golden pins every job the three old builders emit;
- each client is ported word for word;
- each old builder is then deleted.

Scope (owner, 2026-10-28): every lowering behind `Session::matmul_dram` (block-stream, blocked/K-block/stream,
thin/prefetch/handoff). `matmul_dram_batched` and attention's `matmul_step` are follow-on clients.

**Before and after, card 1.** `cargo xtask bench-diff <B0.jsonl> <B1.jsonl>` (`61478ca`) joins two collections by test and
key and judges each pair by its own p10..p90 spread. B0 is taken at F3.0, the per-client subsets after each client,
and B1 at the close. The set:
- plain: `host_time_per_op` (now with the call, enqueue, build and a cold first call for add, `exp`, a row broadcast, a
  sum over columns and a matmul), `small_and_large_ops_by_spread`, `matmul_throughput`, `k_block_matmul_baselines`,
  `norm_forward_baselines`, ResNet-18 / ViT-Tiny traced, and `bench_training`;
- `TT_TILES=all`: `traced_generation` on the 135M and the 1.7B;
- `--profile`: `sfpu_pipeline_sweep` and `matmul_pipeline_sweep`, which need stamping firmware.

- [x] **F3.0** (`61478ca`): H6 opened; `host_time_per_op` and the spread bench emit `BENCH` records; `bench-diff`.
  B0 is taken from a worktree at `050cbbd` beside the after-checkpoint, back to back on a quiet host: a first B0 taken
  while the host compiled and ran ttsim gates read 135M decode ~1% above Stage 0's, so it is not used.
- [x] **F3.1, the job golden** (`050cbbd`). `tile_op::golden` pins every job of 409 pointwise cases (every SFPU kind,
  distribution, pipelining, broadcasts, blocks, a packed twin), 90 reductions (every op and axis, finishing kinds, the
  chunked fallback) and 223 products (every `matmul_dram` lowering, every edge-format pair, every pair order), a hash a
  job, in `golden/jobs_{eltwise,reduce,matmul}.txt`; the full dump goes to `target/tile_op_golden/`. The capture-time
  choice of prefetching units stays in `Session::matmul_dram` until F3.8; the golden pins the builder under explicit
  prefetch, busy and handoff arguments, so `step243` moves to F3.8.
- [x] **F3.2-F3.4, `TileOp`, the passes, and the pointwise family** (`33d3562`).
  - `TileOp`: a domain of iterators in elements (parallel, or reduction with its `ReductionContract`); each operand's
    and result's `TileMap` (each tile dimension one iterator, wrapped, or fixed for a broadcast); edges with element
    type and layout (tiles, or a packed row for a one-row result's twin); a `NumericContract`; and a body behind
    `Microkernel` (a run's group, slots and programs), the only op-specific code a pass calls.
  - `distribute::deal`: items in runs, pipelined where `pipelined_runs` says. Runs are kept inside a row where an
    operand reads one tile a row (read off its map) and that moves fewer tiles; `row_runs` takes its transfer counts
    from the caller.
  - `lower::jobs`: each operand's mover access read off its map against the walk order (items, then each item's
    terms), a table of the mover's modes. Any other map is refused by name.
  - `tile_op::eltwise` replaced `sfpu_eltwise_blocks`, which is deleted. The pointwise golden is word for word unchanged,
    and the pointwise ttsim gates pass (`step19/29/45/46/55/121/145/192/200/206/207/231`).
- [x] **F3.5, the row reduction** (`0bfb79e`). `tile_op::reduce`, either axis. The chunked fallback is no longer a
  builder: its contract is `Combine::GddrPartials { chunk: ROW_CHUNK }`, dealt a job an item and lowered by
  `lower::chunked_jobs`. `sfpu_reduce_then`'s body and `reduce_chunked` are deleted. Both goldens are word for word
  unchanged, and `step31/55/67/69` pass on ttsim.
- [x] **The pointwise and reduction checkpoint on silicon.**
  - Card 0: `step19/29/31/45/46/55/67/69/121/145/192/200/206/207/231`, 65 of 65.
  - Card 1: B0 (`050cbbd`) against `0bfb79e`, back to back. The build stage before enqueue was 20-50% slower, up to 3.9%
    of a call, from per-run allocations in the lowering. With those removed (`8011dc7`), every call is within 1.8% of
    B0 (ABBA). Traced decode, ResNet-18, ViT-Tiny and MNIST are within run-to-run drift
    ([log](../learnings/performance-log.md#f3-the-tile-generic-op-pointwise-and-reduction-ported-before-and-after-2026-10-28-cards-0-and-1)).
  - `matmul_pipeline_sweep` had failed on every tree (an unbalanced profile with NC writing). It now falls back to host
    time (`db3081e`).
- [ ] **F3.6-F3.8, matmul:** block-stream, then blocked/K-block/stream, then thin, prefetch and handoff.
- [ ] Allocation through `l1::Requirements` and the op's `channel::Graph::check`, once the matmul's layouts are
  microkernel layouts: today each client's microkernel plans its own L1 (`sfpu::kernel::plan_layout_in`,
  `sfpu::reduce::plan_layout`).
- [ ] **F3.9:** the H6 audit, B1, `--smoke` on card 0, and the close-out.
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
