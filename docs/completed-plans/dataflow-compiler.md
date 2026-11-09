# Plan: the dataflow compiler, staged

**Status:** complete 2026-10-31 (stages F1-F6 closed); approved and scheduled 2026-10-28 (owner). Its design principles are in
[dataflow-compiler.md](../learnings/dataflow-compiler.md) (the design reference), and its binding decisions in ADRs
[0003](../adr/0003-a-tile-level-dataflow-compiler-below-burn.md),
[0004](../adr/0004-formats-are-edge-data-converted-only-at-unpack-and-pack.md) and
[0005](../adr/0005-every-op-has-a-numeric-contract-schedules-choose-only-among-bit-equivalent-options.md).
**Branch:** `traced-generation` (F1 was built on `dataflow-compiler` in a worktree and merged back, `d84ef47`).
**Silicon:** cards 0 and 1 (owner, 2026-10-28; card 1 only before that). Each gate is validated on one card, card 0 by
default; benches stay on card 1, where Stage 0's baselines were taken, and a before/after pair always uses one card. No
`--device all`, and no mesh, multi-card or Ethernet gate.

Every stage opens by writing its hypothesis, prediction and kill criterion into the design reference's register (§13), then
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

- **H8** (design reference §13, opened 2026-10-28): a credited handoff between two units costs less than a barrier, ≤ 1.5 us on card 1;
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
  - the prediction (<= 1.5 us) is missed, and the claim (cheaper than a barrier) holds. See the design reference's H8.
- [x] Close-out: `--smoke` on card 0, 1,241 of 1,241 (`1793200324`); `step12_mnist` 8 of 8 on ttsim, goldens unchanged.
- **Not yet built:** no op uses a cross-core channel. F4b (operand reuse) and F4a (channels instead of barriers) are
  the first. `channel::Graph` runs on each op's in-core graph since F3 (`tile_op::channels`); a `Gddr` ring's GDDR
  placement is the caller's.

## Stage F3: the tile-generic op (L1) and transforms written once

**Opened 2026-10-28.** H6's prediction and kill criterion are in the design reference's register. The method:
- a job golden pins every job the three old builders emit;
- each client is ported word for word;
- each old builder is then deleted.

Scope (owner, 2026-10-28): every lowering behind `Session::matmul_dram` (block-stream, blocked/K-block/stream,
thin/prefetch/handoff). `matmul_dram_batched` and attention's `matmul_step` are follow-on clients.

**Before and after, card 1.** Since the matmul port (owner, 2026-10-28), the check is `bench_training` against Stage 0's
rows. The jobs are word for word, so device time cannot move, and a trace builds once, so only an eager op's build can
cost anything; untraced training is where that shows. The full set below, with its 3% cutoff, is no longer run as a
gate. It stays the tool for finding where a regression is, if `bench_training` shows one. `cargo xtask bench-diff
<B0.jsonl> <B1.jsonl>` (`61478ca`) joins two collections by test and key and judges each pair by its own p10..p90
spread. B0 was taken at F3.0 and the pointwise/reduction subset after `8011dc7`. The set:
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
- [x] **F3.6-F3.8, matmul:** block-stream, then blocked/K-block/stream, then thin, prefetch and handoff. Owner,
  2026-10-28: `matmul_dram_batched` is ported in this stage, so `BlockPlan` goes whole. The opt-in handoff, second job
  and balanced deal are ported, then re-measured under channels in F4a.
  - [x] **F3.6a, the batched golden** (`9dad570`). `tile_op::golden::batched_jobs_are_pinned` pins 58 cases of
    `tensor::matmul_dram_batched_limited` (attention's `Q K^T` and `P V` over heads, ragged and padded row blocks, a row
    a head over a cache, a long prefix by the block-stream model, fixed blocks and off, a `K` limit, refusals) in
    `golden/jobs_batched.txt`. It reaches the plain, block-stream and K-block lowerings. `step243`, which would pin the
    session's capture-time choice of prefetching units, waits for F3.8, and only if that choice moves.
  - [x] **F3.6, block-stream.** `tile_op::matmul` describes a product as the domain `[m P, n P, k R]`, with `A` mapped
    `[m, k]` and `B` `[k, n]` (either stored transposed), behind the `Mma` body. The description gained three concepts,
    none of them named after a client:
    - `Combine::L1Halves { chunk }`: a reduction's chunks continued in L1, alternating the arena's halves, with only the
      last chunk written;
    - 2-D runs: `Deal::blocks` and `Microkernel::block`;
    - a lowering row for the **contraction pair**: two operands sharing the reduction iterator read by one `GATHER`.
      Transposes come from each map's orientation, formats from the edges, and live rows (`gather_rows`) from the
      ragged iterator where the access asks (`Access::live_rows`).

    `Microkernel` gains `block_chunk` and `transposes` (the unpacker's transpose of `B`), and `RunCode` gains the
    results' slot stride. `TileOp::jobs` lowers an op whose result is already placed, which is how the batched
    product's items share one output. `push_block_stream_jobs` is deleted, and `block_stream_plan` still chooses the
    shape (an H6 audit item: the blocking pass's cost input).

    All four goldens are unchanged word for word, and a mutated chunk half fails both matmul goldens. tt-kernels passes
    313 of 313. On ttsim, `step229`, `step60` and `step184` pass. On card 0 (`1793213075`),
    `step229/60/184/241/239/18` pass 24 of 24.
  - [x] **F3.7, blocked, K-block and stream.** One lowering now cuts every contraction:
    `lower::block_jobs`, a job a block, a chunk's three steps as the reduction's contract says.
    - `InDst`: one chunk, halves where the deal is pipelined.
    - `L1Halves`: as in F3.6.
    - `GddrPartials`: `WAIT`, the reads, then the partial's reload from the result.

    `Mma` gains `Schedule::{Blocks, KBlocks, Stream}`. The description gains:
    - `Layout::Packed { tiles, row }`: a one-row operand read from its packed row, a tile holding 32 terms;
    - `Microkernel::holds`: a first operand kept in its slots while a unit's runs need the same tiles there
      (`GATHER_A_RESIDENT`, or no packed read);
    - `RunCode::resident`: the terms its slots hold where they span more than the chunk, the stream's whole `K`;
    - `RunCode::mop`;
    - `Blocks`/`Cols` with the column runs of `thin_runs`;
    - `Chunk`;
    - `TileOp::ahead`: a unit's first run reads its `STREAM`ed operand ahead into a `BUSY_PREFETCH` list where its
      slots lie above the unit's busy top;
    - `TileOp::jobs_from`: job `j` on unit `j % units` across a work of several ops.

    `push_k_jobs` and `push_stream_jobs` are deleted. `BlockPlan::push_jobs` is now a wrapper building the op, kept
    only until F3.8 ports `push_prefetched_jobs`, `push_balanced_jobs` and `second_jobs`, which call it.
    - **For the H6 audit:** the partial's reload sits after the reads in a block's chunk, but before them in the
      linear `chunked_jobs` (the reduction). That is word order only, since a list completes before its kernel. It
      is kept so that both goldens stay word for word.

    All four goldens are unchanged. Two mutations are each caught by the goldens:
    - stream residency off;
    - the K-block reload's item off by one.

    tt-kernels passes 313 of 313.
    - ttsim: `step18/60/68/184/205/216/229/232/239/241`, 44 of 44.
    - Card 0 (`1793213768`): 57 of 58 across those and `step233/238/240/185/188`, the 1.7B at its native context
      included. The one that failed was `step188::traced_generation_equals_the_eager_one`, a timeout. It has run
      127-137 s since it was written, past the runner's 120 s default; alone it passes in 129.9 s (`1793216293`).
      It now has its own 240 s in `xtask` `LONG`.
  - [x] **F3.8, thin, prefetch, handoff, second job, balanced deal, batched.**
    - `BlockPlan` is gone from `tensor.rs`. It is now the product's schedule, `tile_op::matmul::Plan`, beside
      `block_stream_plan`'s role: which columns a unit takes, which go ahead to idle units, and where a handed column
      splits.
    - Every job it places is lowered by `lower::block_jobs`.
    - Its three hand-written splits of a job's contraction reads are now one generic pass, `tile_op::ahead::take_reads`
      (X8): the first `n` gathers' `GATHER` records moved to a list placed ahead, optionally `STREAM` -> `PREFETCH`.
    - `matmul_dram_batched_limited` builds through `Plan` and `tile_op::matmul`.
    - `matmul_dram_limited` stays: a wrapper of `matmul_dram_prefetching` for the tensor tests, not a builder.
    - **For the H6 audit:** the placement over a spread (prefetching, balanced and second jobs) and the handoff's split
      of an `L1Halves` chain into two ops are still client code in `Plan`, written over lowered jobs. The handoff
      retargets a chunk's write to the partials, which is a contract the description lacks: a chain handed between
      units. F4a re-expresses it as a channel (a remote credit), so it is not given a `Combine` here.
    - All four goldens are unchanged, and tt-kernels passes 313 of 313.
    - ttsim: `step60/68/184/205/216/229/241` pass 38 of 38 (`step205` forces the handoff, second job and balanced
      deal).
    - Card 0 (`1793216912`): 52 of 53 with `step18/239/240/185/188`. The failure was
      `step188::device_agrees_with_flex`, a timeout at 120.16 s against its 116-135 s history; alone it passes in
      119.88 s (`1793217290`). It now has its own 240 s in `LONG`.
    - After F3.8: `step12_mnist` passes 8 of 8 on ttsim with the goldens unchanged.
      `cargo xtask conformance --release` gives 2,973 passing, 52 listed, 0 unexpected, as at F1's close.
    - Eager step time, card 1 (`bench_training`, `1793218316`): 1.398 / 2.892 / 8.260 ms/step against Stage 0's
      1.398 / 2.882 / 8.290 (`1793150926`; 1.387 / 2.882 / 8.210 the run before). Within drift, so the generic
      lowering costs untraced training nothing measurable.
- [x] **Allocation and the channel check as passes** (2026-10-28).
  - **The notes pinned first** (`5dc2812`): `tile_op::golden` records each case's `l1::Planned` (the top and need a
    capture keeps in `OpRecord`) in `golden/planned_<client>.txt`, and builds every case twice, from cold and warm
    memos, requiring the same jobs and notes. It found a reduction's group measured noted, so the first reduction of a
    shape in a process noted its probes' top (`56752e3`).
  - **Allocation** (`033fc3f`): `Microkernel::{run, chunk, block_chunk}` became `needs(at)` and `code(at, placed)`.
    A body declares a run's `l1::Requirements` (memoised by shape) and reads its placement back by name
    (`l1::Placed`). `tile_op::allocate::place` alone chooses the region (the arena, half of it moved by `l1::half()`,
    or the ring's end), plans, checks (`l1::check`), memoises and notes. A product's `B` ring against the arena's end
    is a planner property (`Requirements::against_end`) in place of a gap sized from the region.
    `Schedule::{BlockStream, Stream}` carry shapes, not planned layouts. Every golden is unchanged, cold and warm; on
    ttsim, 26 gate files pass 126 of 126.
  - **A schedule's fit tested unnoted** (`90b1cd5`): `stream_chunks`, `packed_k_block` and `k_block_size` noted their
    candidates. Five K-blocked or streamed cases' notes fell; none rose. The default decode schedules are unchanged.
    `step197/205/216/217/218/219/221/222/228` pass 48 of 48 on ttsim.
  - **Channels** (`c6f426e`): `tile_op::channels` checks a unit's graph before lowering:
    - the streams at the depth its runs are staged;
    - the unpacker's batch to the packer;
    - a chained reduction's partial, with one initial page (`Channel::initial`; `Kind::Roles` for a handoff between a
      tile's roles).

    The depth is checked against the lowered kernels' halves. Planted faults: a chain without its first page
    deadlocks, and pipelined runs claimed one page deep fail the goldens.
- [x] **F3.9: the H6 audit, `--smoke` on card 0, and the close-out** (2026-10-28). H6 holds; the result is in the design reference
  (§13 H6), and D1 is decided: staged.
  - Item 2, the reload order: one link form, with wait, reads, then reload (`2f9bc5d`). The reduction's chunked
    fallback was re-blessed: four cases, the same entries reordered. `step31/55/67/68/69` pass 25 of 25 on ttsim.
  - Items 1 and 3 are recorded. The schedule's cost input is §8's, outside the passes. A chain handed between units
    (`Plan`'s spread placement and the handoff) is the concept the description lacks, carried to F4a.
  - The audit's moves (`d280c46`):
    - distribution into `tile_op::distribute`;
    - the arena's halves into `l1`;
    - `block_reads` into `lower`;
    - the read-ahead split into `ahead::split`.

    The exceptions left are named in the design reference.
  - `step12_mnist` on ttsim passes 8 of 8 with the goldens unchanged. `--smoke` on card 0 passes 1,241 of 1,241
    (`1793221000`). `cargo xtask conformance` was not rerun, because nothing under `burn-tt` changed.
  - B1 (`bench_training`, card 1, ABBA against F3.8's tree `a984fa2`): the closing tree gives 1.398 / 2.892-2.903 /
    8.270-8.300 ms/step (`1793222702`, `1793222744`, `1793223083`); F3.8's tree gives 1.398 / 2.892 / 8.270-8.290
    (`1793223007`, `1793223049`). Within drift.
  - `step243` was not needed: the session's capture-time choice of prefetching units did not move.
- **H6 prediction:** no pass contains code specific to one of the three clients. If one does, the description gets the missing concept and the design reference records it. **Held** (the design reference's H6).

## Stage F4b (compute): operand reuse across cores, H1

- [x] **F4b.0, the baseline and the reads-free ceiling** (2026-10-28, card 1, at `f60b228`;
  [log](../learnings/performance-log.md#f4b0-what-operand-reuse-can-buy-the-reads-free-ceiling-at-prefill-shapes-2026-10-28-card-1)).
  - `matmul_throughput` (`1793224018`, `1793224268`): `[2048,2048]@[2048,16384]` 19.36 / 19.17 TFLOPS, 2,062 MB read
    (`A` 971, `B` 1,091) for 151 MB of operands.
  - `silicon_probe_pair_cost` with and without gathers (`1793224072`): the ceiling is 30.3 TFLOPS with an FP32 `B`,
    28.0 with a packed one. Today an FP32 `B` has 1.38-1.51x to gain at the three shapes, a packed one at most 1.13x.
    Skipping only `A`'s reads recovers 92-99% of the ceiling.
  - H1 restated in the design reference before the build (owner's choice): FP32 `B` ≥ 25.8 TFLOPS and ≤ 302 MB read; packed
    `B` ≥ 95% of the ceiling.
  - Found: `[2048,8192]@[8192,2048]` runs at 10.9-11.0 TFLOPS through Burn (80 units) against 18.5 through the
    session in the probe. Open, not F4b's.
- [x] **F4b.1, D5's schemes inside the product** (owner, 2026-10-28: prototypes in the product, measured by
  `matmul_throughput`, in place of a synthetic delivery probe).
  - [x] **Pull** (`TT_OPERAND_REUSE=pull:<h>`, 2026-10-28): the shared deal (`distribute::Shared`), the pass
    (`tile_op::reuse::pull`), the fixed credit words (`tt_isa::dm::CREDITS`, zeroed at mover start, each owner returns
    its own to zero), and aligned segments for linked ops (`session::linked_groups`). Gate `step244_operand_reuse`
    (ttsim; card 0 `1793226376`); model test `tile_op::reuse::tests` (random interleavings, a negative control).
    H1's shape on card 1: 19.1 -> **25.98 TFLOPS** at four holders a group, reads 2,062 -> 1,346 MB (`1793228872`;
    [log](../learnings/performance-log.md#f4b1-a-pulled-from-a-few-holders-the-first-measured-scheme-of-d5-2026-10-28-cards-0-and-1)).
  - [x] `B` shared too (a `GATHER` that skips `B`: a firmware flag), for the reads clause (built; kept opt-in).
    `TT_OPERAND_REUSE=pull:<h>,<hb>`: `record::GATHER_B_RESIDENT` (bit 2, reclaimed from the
    retired packed BF16), a column group a member index of equal-sized row groups
    (`distribute::Shared::b_holder_of`), and the pass run once an operand in credit words 3-5
    (`CREDIT_WORDS` 4 -> 8).
    - *Prediction (before the build):* H1's shape at `pull:2,1` reads about 270 MB (`A` 136, `B` about
      136, from 1,346), inside the ≤ 302 clause, at ≥ 25.98 TFLOPS (pull:4's), predicted 27-28 (the
      probe's "`A` skipped" ran at 29.2 of a 30.3 ceiling).
    - *Kill:* reads drop and TFLOPS fall below 25.98: `B`'s copies cost more than the reads they
      replace.
    - **Result (2026-10-29): correct, and the kill fires** ([log](../learnings/performance-log.md#f4b1-b-pulled-too-the-reads-clause-met-and-the-rate-lost-2026-10-29-cards-0-and-1)).
      `step244` 10 of 10 on card 0 (`1793236301`) and H1's product on every tile bit for bit (`1793236763`).
      On card 1, `pull:2,1` reads **272.6 MB** (inside ≤ 302) at **15.18 TFLOPS** (`1793236805`), against
      `pull:4`'s 1,346 MB at 25.19 in the same session. The rate follows a `B` holder's readers: seven
      14.0-15.2, three 22.2-23.7, one 25.3-25.7 (reads 682-801 MB). Kept opt-in: the credit runs and
      multicast below say whether a holder's credits or its L1's copies bind.
    - *Open, not built (owner, 2026-10-29):* equal groups leave `units % groups` units idle for the
      product (none at H1's 120 over 8 rows). They could run data-independent work meanwhile, but
      every op ends in a barrier across all units, and the credit words' return to zero relies on
      it, so that needs F4a's channels in place of barriers (or F6's disjoint core sets).
  - [-] Multicast: one sender a row rectangle (NC `WRITE_L1_MCAST`, a multicast ready word), against pull.
    **Deferred, not refuted (owner, 2026-10-29): revisit when hunting performance.** Not built; the analysis:
    - A multicast reaches a rectangle of Tensix tiles that excludes its sender (`channel::Ring` refuses otherwise),
      and path-reserved multicasts sharing a rectangle serialize (design reference T7), so the safe form is one sender a
      disjoint rectangle. The deal's groups must then be the grid's rows and columns, not runs of consecutive units.
    - `B` down grid columns (the scheme that would end `B`'s holder fan-out, F4b.1 above): row groups as wide as the
      grid (12 tiles on a 120-tile card). H1's 8 block rows fill 8 of 10 grid rows, 96 units, a ceiling of about
      24.2 TFLOPS (30.3 x 96/120), below `pull:4`'s 25.2-26.0; 10 row groups do not divide 64 tile rows evenly.
    - `A` along grid rows: each row splits at the non-Tensix columns into two rectangles, and `A` pulled already runs
      at ~86% of the 30.3 ceiling, so at most ~1.17x.
    - It is a new protocol on NC inside PAIR packets (NC waits for B's gather on its own tile, then multicasts into
      the readers' slots before their kernel), where a mistake can hang the NoC.
    - Revisit when: a product whose block rows divide the grid's rows (or a card whose grid suits the shape), a
      product whose `B` holder fan-out binds where `A`'s pull leaves headroom, or prefill-scale products after
      F5 (`Dst` subblocks) lifts the kernel's own ceiling.
  - [x] Larger GDDR blocks (fixed `TT_MATMUL_BLOCK` sweeps with packed `B`), 2026-10-29, card 1
    ([log](../learnings/performance-log.md#f4b1-larger-gddr-blocks-the-planners-block-not-reads-binds-a-packed-b-2026-10-29-card-1)).
    - *Prediction:* no block that fits beats the planner's on reads with an FP32 `B`. **Held:** larger ones are
      refused by name (`mb x nb <= 72` FP32 outputs, twice over), and `[8,2,9]` reads least under that bound.
    - *Found:* with a packed `B`, the planner's block binds, not reads: `[6,4,12]` runs BFP8 / BFP4 / BFP2 at 34.4 /
      41.1 / 43.1 TFLOPS against the default's 26.3-26.5 (`1793238988`, `1793238206`), BF16 27.96 at `[4,2,18]`
      (24.69), FP32 21.09 at `[8,4,6]` (20.13). `block_stream_model` charges every tile 4 KiB; with an FP32 `A`
      re-read `nt/nb` times and a packed `B`, a wider `nb` and fewer chunks win.
    - *The planner re-weighed (owner: make it the default), refuted and reverted:* each operand's bytes by its slot,
      with the pair at 0.2 and at 0.15, measured at four shapes on card 1 (`1793240867`, `1793241117`): each gains up
      to 1.4x somewhere and loses up to 14% (slots) or 22% (pair 0.15, FP32 at H1) elsewhere.
  - [x] **A measured block choice** (F4b.2, H9; the default since 2026-10-29, card 1,
    [log](../learnings/performance-log.md#f4b2-a-block-measured-on-the-card-a-pulled-where-it-gains-2026-10-29-card-1)):
    `BlockStream::Measured` (`TT_MATMUL_BLOCK`, `measured` when unset). The first time a capture plans a product's
    question (`tensor::BlockQuestion`: tiles, transposes, units, the format pair, fidelity), the capture is paused
    (`Session::paused`) and the candidates are timed on the card with the real operands; the winner, block and pull
    together, is kept for the process (`tensor::measured_choice`). Build time, not step time (owner, 2026-10-26).
    - *Candidates are measured, not modelled:* every model ranked the sweeps' winners in the hundreds, so they are
      `Auto`'s block unpulled and pulled, 12 listed blocks that won sweeps at ten shapes (`tensor::MEASURED_BLOCKS`),
      and a one-step climb from the fastest (`tensor::block_stream_neighbours`), at most 40 timed.
    - *Timing:* the difference of traces of `R` and `2R` copies, over `R` (`Session::time_product`), so a replay's
      fixed cost cancels; timed alone, blocks that leave units out won and ViT-Tiny ran 11.6% slower.
    - *Eager products run `Auto`'s block:* measured blocks made eager training 1.5-2.9x slower.
    - *A measurement's traces are stored whole*, whatever the capture it pauses stores as layer sub-traces: its `R`
      copies repeat, so as sub-traces they were one body called `R` times. The smoke found it on card 0
      (`step223_sub_traces` with short tables: the measurement's replay failed, the session recovered, and the paused
      capture ended `Stale`); gate `step245::a_measurement_under_sub_traces_times_its_copies_whole` (silicon only,
      fails without the fix).
    - *Pausing a capture* keeps its tensors (`DramAlloc::suspend`), its L1 plan (`l1::unnoted`), its arena and its
      movers' arena, and moves its barrier targets and unit generations past what ran
      (`trace::Capture::barriers_base`, `UnitCapture::generation_base`). A planned capture never pauses: it uses what
      its plan pass measured. Gates: `step245_measured_block` (the chosen candidate runs bit for bit and is timed
      once; a capture paused is the capture unpaused word for word; a pause that does not re-base is caught, ttsim).
    - *Result:* 1.02-2.25x `Auto` over 50 cases (10 shapes x 5 formats), never below; H1 through Burn with nothing set
      19.2 -> 29.0 TFLOPS; 1.7B prefill 1.04-1.34x; models and decode unchanged; 2.7-13.9 s a question.
    - *Open:* an eager product run again after its own shape's pulled capture takes 3-10% more host time (device
      the same), not reproduced through the session alone.
  - [x] Holders' credits without a round trip each (14 a chunk on the holder's B today). `dm::CREDIT_RUN` (word 5
    of `op::CREDIT`), as `MCAST_RUN_ACKS` is a multicast's: a run's first credit waits for the mover's earlier
    requests, none waits for its own response, and the run's responses are counted before the first entry not of it
    and at the list's end. A holder's credits after a gather are one run.
    - *Prediction (before the build):* `pull:1` (14 credits a chunk on one holder) rises to at least `pull:4`'s rate
      (25.19 TFLOPS in the same session), and `pull:2,1` (seven `B` readers a holder) gains measurably over 15.18.
    - *Kill:* under 1% on both: the round trips do not bind, and the flag is dropped.
    - **Result (2026-10-29): refuted and retired.** Built (firmware, `Entry::Credit`, the pass), correct on card 0
      (`step242` with a broadcast ring's credits as one run, `step244`, `step172`/`step173` safe and `fast-reads` +
      `encoded-guard`: `1793242140`, `1793242327`; H1 on every tile bit for bit, `1793242347`). On card 1, off / on, ABBA:
      `pull:1` 25.07 / 25.36, 25.05 / 25.16; `pull:4` 26.01 / 25.64; `pull:2,1` 15.20 / 14.98, 14.85 / 15.31
      (`1793242358`-`1793242444`). No gain: a holder's credits do not bind, so `B`'s fan-out cost is the holder's
      L1 serving its copies. Code, flag and gate case removed.
  - [x] Recovery after a failed linked op restarts every unit it ran on (2026-10-29). It already did: a failed
    `Session::sync` recovers every unit, each mover restarted and its credit words zeroed as it starts; what was
    missing was the gate. `step244::a_stalled_shared_product_leaves_the_session_usable` (both operands shared, each
    reader's last wait one credit past what its holder sends, `reuse::probe_overwait`): the op fails by its
    deadline, and the next shared product is the unshared one bit for bit with every credit word at zero (card 0,
    `1793243728`). Silicon only: ttsim cannot recover a tile whose roles stalled inside a kernel (divergence row 16).
  - [x] The planner prefers block shapes whose rows divide over the units (a group of one runs a whole row). Folded
    into the measured block choice above (2026-10-29): each candidate is timed with the shared deal it gets, so a
    block that leaves groups of one is measured, not assumed. `B` is never pulled automatically, so equal groups
    (`B` shared) do not arise by default.
- [x] Decide D5 by measurement first, on card 1, for one prefill product shape, between:
  - a few holders whose rows or columns are pulled with `READ_L1`;
  - one sender per row or column rectangle multicasting with `WRITE_L1_MCAST` and remote credits;
  - larger blocks re-read from GDDR.
  **Decided 2026-10-29: `A` pulled from a few holders** (card 1, `1793243755`-`1793244204`,
  [log](../learnings/performance-log.md#d5-decided-a-pulled-from-a-few-holders-a-packed-bs-block-is-the-next-lever-2026-10-29-card-1)).
  With an FP32 `B` it is 1.33-1.44x at H1's shape and both confirming shapes (H1 28.38 TFLOPS through the session,
  94% of the ceiling), bit for bit. `B` pulled loses (a holder's fan-out); multicast is deferred (its bound above);
  larger GDDR blocks read no less for FP32. A packed `B` is bound by its block, not reads: at H1, `[6,4,12]` with `A`
  pulled reaches 44.5-45.8 TFLOPS (1.7x), but the best block, and whether to pull, depend on the shape.
  Closing checks on the final tree: `--smoke` on card 0 passes 1,252 of 1,252 (`1793244532`); `step12_mnist` 8 of 8 with
  the goldens unchanged; the job goldens unchanged (the default path emits the same records).
- [x] Build the winner as F3's reuse pass, with a policy for when it pulls, and the measured block choice (F4b.1
  above), which chooses the block and the pull together (F4b.2, 2026-10-29).
  - The pass: `tile_op::reuse::share(reuse, tiles)` decides an op's `share` from the op and the session's `Reuse`, no
    client named. `Reuse::Auto` (`TT_OPERAND_REUSE`, `auto` when unset) shares only what a measurement chose; `off`
    and `pull:<h>[,<hb>]` stay explicit overrides. A static policy (`A` pulled wherever `B` is FP32) was tried and
    dropped: ResNet-18 ran 9.7% slower.
  - The pull survives captures: `dm::L1Writes` has a read window (`reads`) beside the write window. A planned capture
    narrows only the write window, and ending one restores it, so reuse no longer stays off after a planned capture.
    A planned capture's linked ops (a pulled product's holders and readers) are grouped from the plan-rewritten lists;
    before, the rewrite was dropped for them.
  - Every format gets it. Not yet every contraction: the batched block-stream path
    (`matmul_dram_batched_limited`, attention prefill) builds each item's jobs from unit 0 and concatenates them, so
    a pull there needs the items' units numbered globally and each unit's credits returned per item.
- **Prediction (restated after F4b.0, against the measured ceiling):** `[2048,2048]@[2048,16384]` with an FP32 `B`
  from 19.2-19.4 to ≥ 25.8 TFLOPS, reads from 2,062 to ≤ 302 MB; a packed `B` at ≥ 95% of its 28.0 ceiling. As first
  written, ≥ 35 TFLOPS, which is above the 30.3 a product reaches with no reads at all.
- **Kill:** GDDR reads drop and TFLOPS do not.
- **Result (2026-10-29):** the TFLOPS clause holds (`A` pulled: 25.2-26.0 through Burn, 28.38 through the session); the
  reads clause is met only by pulling `B` too (205-273 MB), which loses the rate, so it does not hold with the rate.
  The kill did not fire: where reads dropped (`A` pulled), TFLOPS rose. For a packed `B` the 95% clause holds at the
  default block (`pull:4` 26.56-26.67 of 28.0) and was the wrong ceiling: the block, not reads, binds (up to 45.8).

**Closed 2026-10-29.** Carried to the roadmap as "F4b's leftovers": the pull in the batched block-stream path, and the
eager product's 3-10% host time after a pulled capture.

## Stage F4a (decode): channels instead of barriers, H4

**Opened 2026-10-29 at `9977ad8`.** Sparse waits alone were measured as worth 0.070 ms on the 1.7B
([decode-small-ops.md](decode-small-ops.md) A0, A2 not built): a product's consumers need most of its units, so the time
at its barrier is its slowest unit's work (T10). H4's gain has to come from overlap, units that leave a product early
reading the next product's weights in its tail, so the stage bounds that before building anything (owner, 2026-10-29).
If the bound fires the kill, H4 is refuted, the three schedules below stay opt-in, and the stage stops.

- [x] **F4a.0, the baseline and the overlap bound** (card 1, at the starting commit).
  - H4's baseline (Stage 0's open item): `TT_TILES=all cargo xtask bench --profile --device 1 --filter
    silicon_bench_models::traced_generation_stream_ceiling`, and the same without `--profile` for the host-timed step.
  - The tail split by unit, in the profile mode (`tails.tsv`): each eighth of an op's span, its bytes read by the op's
    own units, by units reading ahead, and by the rest.
  - The overlap bound, in the profile mode (`overlap.tsv`): each (op, unit) at its measured work, started (a) when the
    units it depends on are done; (b) as (a), and each unit's next reads ahead start when its own work ends, at the
    rate the card has left (469 GB/s less what the tail reads) and within its L1 room; (c) as (b), with the small ops
    under reads ahead charged T8's measured slowdown.
  - *Kill before the build:* (c) saves < 0.3 ms a step.
  - **Result (2026-10-29, card 1, at `9977ad8`): the kill does not fire**
    ([log](../learnings/performance-log.md#f4a0-what-overlap-can-buy-a-decode-step-the-tails-split-by-unit-2026-10-29-card-1)).
    - Baseline: `Step(64)` 10.91 ms on the card at 327 GB/s (`1793309460`), host-timed 11.37 / 12.71 ms at 32 / 256
      tokens (`1793310992`). The products' reads by eighth: 581 471 404 342 300 282 197 68 GB/s.
    - The tails (`tails.tsv`, `1793310239`): a product's own units still at work fall 92 -> 20 by its last eighth, and
      the units reading ahead stop by its fourth (119 72 47 5 0 0 0 0 GB/s). In the tail about 70 freed units read
      nothing: each unit's next list (`BUSY_PREFETCH`) sits after its own arrival at the product's barrier
      (`busy_start` places it at the stream length after the product, barrier included).
    - The bound: (b) 2.45 ms of the card's capacity unused in tails a step (2.14 of it the products'); the step with
      every weight read made free (`TT_PROBE_STREAM=skip`) 5.33 / 6.44 ms (`1793311234`), so the products are bound by
      their weights, not their math; the reads' own floor (3,567 MB at 469 GB/s) 7.61 ms. (c): reads under the small
      ops cost them ~3% today (RMSNorm 781.8 against 755.8 us with no reads, `1793311447`), so (c) ≈ 2.4 ms.
    - Consequence for the order: the overlap needs no channel. B pumps a prefetch's reads while it waits at a
      barrier (`mover::barrier`), so a list moved to before the unit's arrival reads in the product's tail with the
      barrier kept. F4a.2's second item is built first, as the cheapest test of H4's mechanism.
- [x] **F4a.1, edge credits in place of kept barriers** (only if F4a.0 clears the kill). Derive per-edge channels
  from the barrier analysis (`tt_kernels::barriers::analyse`, whose `deps` are per edge over every address space, the
  residency plan's L1 moves included). Keep barriers only where no channel can express the dependency, or where a
  barrier is the cheaper lowering of an all-to-all edge (design reference §7). Stored streams only, as `prune`.
  - [x] **Built, opt-in, not faster** (2026-10-30, card 1, lane A2;
    [log](../learnings/performance-log.md#f4a1-f4a3-the-idle-lists-pumped-edge-credits-and-the-three-schedules-2026-10-30-card-1)).
    - `barriers::credit` (`Session::set_edge_credits`, `TT_EDGE_CREDITS=<fan>`), at the prune point: a kept barrier whose
      next run's units each depend on at most `fan` producers, each crediting at most `fan`, becomes credits. A producer
      `op::CREDIT`s each consumer's op-level word (`tt_isa::dm::credit_word` 6 and 7, past `tile_op::reuse`'s six) where
      its arrival was; a consumer `op::WAIT_CREDIT`s for its count before its first op of the run, then credits its word
      back to zero. Every dependency since the last barrier kept as a barrier is lowered (a credit orders nothing else),
      so one word an epoch and at most two epochs, then a barrier, keep any credit from reaching a word before its
      wait. Checked before the capture changes: every dependency covered by a credit or a barrier, and the credits as
      a `channel::Graph` (each unit's runs in order between a source and a sink, a credit an edge) by `Graph::check`; a
      refusal keeps the barriers. B's `WAIT_CREDIT` pumps a queued prefetch as a barrier wait does
      (`dataflow::credit_with`).
    - Gates: `barriers`' unit tests run the lowered streams under 300 random interleavings each and check every
      dependency's order and every word back at zero; their control (the same barriers pruned with no credits)
      races; the coverage check refuses a dropped edge and `Graph::check` an unbalanced credit. `step265_edge_credits`
      (ttsim; card 1 `1793345620`): sparse element-wise deals replay bit for bit with credits (3 of 4 barriers, 63
      credits), both mutations refused at the capture's end, which then keeps its barriers and the same bits; a
      SmolLM2-135M traced decode with credits (711 barriers lowered) equals its eager one (`1793343055`). In `--smoke`.
    - What a 1.7B `Step(64)` admits (`TT_PROBE_BARRIERS`, `1793338608`): of 272 kept barriers, fan 1-4 lowers the 48
      product -> residual-add barriers (64 credits each, one producer a consumer); fan 8 adds 74 more for 21,366
      credits a step; the rest fan in from all of a product's units.
    - ABBA at fan 4, off / on: 10.981 / 12.140 and 11.013 / 12.143 against 11.042 / 12.195 and 11.018 / 12.201 ms
      (`1793347800`-`1793349018`): **not faster** (+0.03 / +0.06). A product's consumer still waits for the producing
      units' work, which is the product's slowest (T10), and the residual add is all that starts early. Opt-in; the
      capture's build grows (the coverage check and graph), which is not a target.
    - On Qwen3.5-4B (2026-10-31, card 1, on the merged tree; [log](../learnings/performance-log.md#the-merged-tree-and-edge-credits-on-qwen35-2026-10-31-card-1)), whose step is ~1,000 small ops a
      barrier each: the build first timed out (the checks were quadratic in a trace's runs; now linear, unit test
      `channel::tests::a_traces_worth_of_runs_checks_in_linear_time`, `step265` and `step246` on card 0
      `1793413336`). ABBA at fan 4: decode unchanged (64.28 / 64.65 and 64.33 / 64.69 off, 64.19 / 64.69 and
      64.24 / 64.71 on), prefill **1.4% faster** in both pairs (3,709 -> 3,656 ms for 256 tokens), bit for bit. Still
      opt-in: a prefill-only gain on a model behind the 1.7B, the decode unmoved.
- [x] **F4a.2, reads ahead started by edges, not anchors:** units that finish their share start data-independent reads
  (the next product's weights) at once.
  - [x] **A product's units, before their arrival** (2026-10-29, cards 0 and 1;
    [log](../learnings/performance-log.md#f4a2-a-products-units-read-the-next-weights-before-their-arrival-2026-10-29-cards-0-and-1)).
    `Session::set_read_ahead_early` (`TT_READ_AHEAD_EARLY=0` off): a unit's `BUSY_PREFETCH` list moves to just before
    its arrival at the product's barrier, the barrier kept, where no other unit of the product touches the L1 its
    reads land in (`barriers::landing`, `barriers::others_touch_l1`; `Capture::move_block` lets the op hold it). B
    pumps it while it waits. 1.7B step 11.32 / 12.69 -> **10.95 / 12.16 ms** at 32 / 256 tokens on card 1, bit for
    bit; `Step(64)` 10.53 ms on the card at 340 GB/s, tail slack 2.45 -> 1.27 ms; the 135M within 0.04 ms. **The
    default** (owner, 2026-10-29: where it gains at real model sizes). Gates: `step216`, and with it on `step188`,
    `step187`, `step205`, `step219` on card 0. Closing checks with it the default: `--smoke` on card 0 passes 1,265 of
    1,265 (`1793318048`), `step12_mnist` 8 of 8 with its goldens unchanged (`1793319760`).
  - Found on the way and fixed: `silicon_profile_mode`'s chain gates (not in `--smoke`; `--feature profile`) failed
    since barrier pruning (2026-10-26) stored their 600-op chain as one op; they keep its barriers now (card 0
    `1793317798`, 4 of 4).
  - [x] *Idle units' lists before their arrival: refuted, reverted.* 0.6 ms slower: an idle unit's `PREFETCH` list is
    plain `GATHER`s that B runs in line, so before an arrival it holds the product's barrier.
  - [x] **Idle units' lists pumped, and before an arrival: refuted again, opt-in** (2026-10-30, card 1, lane A0/A1;
    [log](../learnings/performance-log.md#f4a1-f4a3-the-idle-lists-pumped-edge-credits-and-the-three-schedules-2026-10-30-card-1)).
    - A0, at `715ef1b`: host 11.013 / 12.162 ms at 32 / 256 tokens (`1793332665`); profile `Step(64)` 10,544.6 us,
      340 GB/s, tail slack 1,281 us (1,008 of it the products', `1793332922`).
    - An idle unit's `PREFETCH` list is now a pumped prefetch (`tile_op::ahead::take_reads_pumped`: `GATHER_PREFETCH`,
      the job's first gather ending in `[WAIT, WAIT_PREFETCH]`), and `Session::set_idle_ahead_early`
      (`TT_IDLE_AHEAD_EARLY=1`) moves it to just before the unit's arrival at its anchor's barrier, where no other unit
      of the anchor touches the L1 it lands in (`early_start`). Bit for bit; gate `step216`
      `an_idle_units_list_reads_its_b_before_its_anchors_barrier` (ttsim; card 1 `1793334281`, `1793345620`), whose
      control (the move off) fails the move's claims, and which fails with the move broken.
    - ABBA, on / off: 11.243 / 12.304 and 11.127 / 12.315 against 10.932 / 12.110 and 10.954 / 12.161 ms
      (`1793338877`, `1793339191`, `1793342802`, `1793345003`): **0.24 / 0.17 ms slower**. Pumped, a list no longer
      holds the barrier, but its reads leave the small ops' window, which had the room, for the anchor's tail, which
      the product's own units and the busy lists already read in. Off by default; the pumped lists stay where they
      were (bit for bit, within noise of A0).
    - Gates with it: `step215`, `step216`, `step205`, `step219` 27 of 27 (`1793334281`); `step187` and `step188` 13 of 13
      (`1793335917`).
  - [x] What is left in the tails, 1.27 ms a step: F4a.1's credits (built, not faster: above), the lists' L1 room
    (one job's `B` a unit): **not built**, measured first. B queues one prefetch a unit (`mover.rs`, a second
    `GATHER_PREFETCH` issues the first's reads in line), so more room would buy a second job read ahead; the second
    prefetched job, which reads exactly that ahead in the trace arena's room, is 0.23 / 0.25 ms slower on this
    schedule (F4a.3, below). The cap does not bind; `mover.rs` is unchanged for it.
- [x] **F4a.3,** re-measure the three opt-in decode schedules F3.8 ports, with the dataflow approach (owner, 2026-10-28). Each was
  bit for bit but not faster with a barrier between its ops:
  - the K-chain handoff (`TT_STREAM_HANDOFF`, slower: `1792685519`, `1792686180`), which takes a remote `CREDIT` in
    place of its second op and that op's barrier;
  - the second prefetched job (`TT_PREFETCH_SECOND`, slower: `1792723098`);
  - the balanced deal (`TT_BALANCED_DEAL`, not faster: `1792763914`, `1792764065`).

  Any that is still not faster under channels is retired completely: code, knob, setter and golden cases, with run
  IDs.
  - **Measured (2026-10-30, card 1, lane A4) on F4a's schedule** (pumped idle lists, busy lists early, barriers;
    credits are not faster, so not in it): off H S B B S H off, 1.7B host-timed ms at 32 / 256 tokens:
    off 10.891 / 12.103 and 10.938 / 12.140 (`1793386897`, `1793390634`); handoff 10.900 / 12.164 and
    10.861 / 12.145 (`1793387662`, `1793389927`): within noise; second job 11.137 / 12.372 and 11.144 / 12.380
    (`1793388898`, `1793389659`): 0.23 / 0.25 slower; balanced deal 11.060 / 12.298 and 11.082 / 12.316
    (`1793389154`, `1793389405`): 0.16 / 0.19 slower.
  - **All three retired**: `Reads::{handoff, second_job, balanced_deal}`, `tensor::{Handoff, HANDED_AHEAD,
    balanced_shares}`, `Plan::{second_jobs, push_balanced_jobs}` and the handed columns in
    `push_prefetched_jobs`, the session's second op for handed columns, the setters `set_stream_handoff`,
    `set_prefetch_second_job`, `set_balanced_deal`, the knobs `TT_STREAM_HANDOFF(_AT)`, `TT_PREFETCH_SECOND`,
    `TT_BALANCED_DEAL`, their golden cases (`jobs_matmul.txt`, `planned_matmul.txt`: every case kept is unchanged
    but for its label) and `step205`'s three tests and their unit tests. `step205` 8 of 8 on ttsim.
- **Prediction:** the 1.7B decode step's card read rate ≥400 GB/s (from 328) and the step ≤9.5 ms (from 10.9), bit-exact.
- **Kill:** the step moves <0.3 ms, or small ops lose what products gain.
- **Lane A's close (2026-10-30, card 1, at `8a354a0`): the kill fires.** 10.978 / 12.184 and 10.953 / 12.149 ms
  (`1793392717`, `1793394056`) against A0's 11.013 / 12.162; `Step(64)` 10,548.2 us at 341 GB/s (`1793392975`).
  F4a.2's busy lists (A0 already) are what F4a bought; nothing since moves the step. Gates at the close: 30 of 30
  small (`1793391769`), 21 of 21 models and MNIST with its goldens unchanged (`1793393729`), `step260` / `step261`
  digit for digit A0's (`1793394314`). The tails are the products' slowest units, which F5 and F6 address.

## Stage F5 (compute): `Dst` subblocks and half-sync, H2

- [x] **F5.B0, baselines on card 0 at `715ef1b`** (2026-10-30;
  [log](../learnings/performance-log.md#f5-dst-subblocks-and-half-sync-2026-10-30-card-0)). H1 through Burn `1793332723`
  (FP32 `B` 29.0 / 18.3 / 11.1 / 13.8 TFLOPS); every format through the session at the measured default `1793332760`
  and at `[8,2,9]`, `[8,4,6]`, `[6,4,12]` (`1793335272`, `1793335286`, `1793335298`); a model shape set of 19
  products through Burn (`1793335314`) and the session (`1793335466`): the MOSS-Image DiT at 1,024 tokens (qkv
  `[1024,3072]@[3072,9216]`, out, `@[3072,8192]`, down `[1024,8192]@[8192,3072]`, a head's `Q K^T` and `P V`), Qwen3.5
  and Gemma 4 prefill at 512 tokens (q, gate, down, o; hidden 2560, FF 9216 / 10240), the MOSS text encoder at 512
  (hidden 4096, FF 12288), and the FLUX.2 VAE's patch products `[4096,3456]@[3456,384]`, `[16384,1728]@[1728,192]`,
  `[65536,864]@[864,96]`. `step262` `1793343089` (and its profile `1793387199`).
- [x] **F5.B1, a `Dst` tiling pass and half-sync, the default** (`tile_op::dst`,
  `matmul::block_stream_kernel_subblocked`; `TT_DST_SUBBLOCKS=0` turns it off).
  - `DstPlan` (half-sync: four FP32 tiles a section, `free` starting at two), `Subblock` (refused past the section;
    only one tile row tall, since a `Src` bank holds one tile and `A`, the FP32 operand, is the one worth holding), and
    `RowPlan::choose`: the fewest subblocks a block row, as even as can be, so a row no width divides takes one narrower
    one (3 tiles: `[3]`; 9: `[3,3,3]`; 18: `[4,4,4,3,3]`); a row of an odd number of subblocks makes the programs'
    period two rows, so the sections alternate across the block.
  - Each `K` step unpacks `A`'s tile into `SrcB` once and holds it across the subblock's `B` tiles (`1 + 1/w` unpacks a
    pair); each tile's `MVMUL`s are the tiles order's at its rows (`DEST_TARGET_REG_CFG_MATH_Offset`); the packer reads
    each tile from its rows (`DEST_TARGET_REG_CFG_PACK_SEC0_Offset`); a first chunk's half is cleared tile by tile in
    sixteen-row `ZEROACC`s; a continuing chunk's priors are reloaded at each tile's rows after the math role hands the
    half over, switching only the `Config` words the reload changes (`matmul::reload_switch`: 137 -> 88 words a
    subblock at `[8,2,9]`).
  - The oracle runs all three roles now (`matmul::order::interpret_roles`: semaphores, the pack role scheduled last,
    `Dst` as blocks the packer reads, a reload a prior term): every shape packs each output's canonical order, priors
    first; clearing both halves, overlapping halves, and the one-tile kernel at two sections are caught.
  - Gate `step246_dst_subblocks` (ttsim; card 0 `1793343573`, 6 of 6): the one-tile kernel bit for bit with the special
    values, ragged products, odd columns, `A` or `B` transposed, BF16 and BFP8/4/2 weights, model shapes on every tile,
    exact against host integer sums; both negative controls change thousands of outputs on ttsim and on card 0.
    On the final tree (card 0 `1793386989`, 33 of 34): `step246` 6, `step229` 7, `step241` 4, `step238` 1, `step158` 7,
    `step68` 6 and `step245` 2 of 3. `step245::a_capture_paused_to_measure_is_the_capture_unpaused` fails, on ttsim too
    and with `TT_DST_SUBBLOCKS=0`: the re-swept list (B2) makes its chain choose `[4,4,8]` pulled, on 2 of 8 units, and
    the paused pass then records one more `KERNEL` entry on the six idle units for the op after the product than the
    unpaused one (replays bit for bit). It passed with the list before (`1793337864`). Resolved at the merge
    (`68a0463`): the pause forgets each unit's tile state, so a unit the paused product left idle runs its next
    op's setup once more; the gate's claim is restated as that, at most one `KERNEL` setup entry a pause a unit and
    every other entry equal (ttsim 3 of 3; card 0 in `--smoke`, `1793397733`).
  - Two silicon findings ([notes](../learnings/silicon-operating-notes.md#dst-in-two-halves-math-pack-and-reloads-side-by-side-2026-10-30-card-0-and-ttsim)):
    a `B` base stepped between a subblock's tiles needs the Scalar Unit settled before its `WRCFG` (one tile a run
    wrong without it); and an `UnpackToDst` beside running `MVMUL`s is wrong, so a reload directly from the packer's
    `free` (tried, opt-in) was retired.
- [x] **F5.B2, `MEASURED_BLOCKS` re-swept** for the subblocked kernel (`silicon_probe_block_sweep`, card 0,
  `1793346207`-`1793351145`: the four H1 shapes with FP32, BF16 and BFP8, the DiT's qkv and down and Qwen3.5's gate with
  FP32 and BF16). Ten blocks, with `Auto`'s own, reach 96% or more of each sweep's best (the list before: 79-92% at four
  of the 18 cases); `[6,4,12]` and `[4,4,18]` kept.
- [-] **F5.B3, roles overlapping across batches: not built.** A blocked product's batches are its chunks, and every
  chunk but a job's first reloads the partials the chunk before packed, so the joins the brief keeps are all but one a
  job (32 at H1's `[8,2,9]`): what is left to overlap is a job's first chunk, a few percent at most. Overlapping the
  continuing chunks needs a pack-to-unpack handoff across a `KERNEL` boundary, in the mover's `KERNEL_WAIT`
  (`mover.rs`, F4a's), so it waits for the merge.
- **Prediction:** ≥1.4× on the H1 shapes after F4b, bit-exact. **Kill:** <1.1×.
- **Result (2026-10-30, card 0): bit-exact; neither the prediction nor the kill.** At the measured default over the H1
  shapes and five formats, 0.96-1.35x, geometric mean **1.12x** (`1793332760` against `1793387665`); with an FP32 `B`
  1.06 / 1.29 / 0.96 / 1.03x (geometric mean 1.08x, H1 32.3 -> 34.4 TFLOPS, not the 40 predicted); with every gather
  left out, 1.05-1.46x (`1793339461` against `1793344646`). What still binds a chunk is its reload: the unpacker reloads
  each subblock's priors while the math role waits, and silicon does not allow it beside `MVMUL`s (above), so the
  half-sync hides the packer and not the reload. The lever left is fewer chunks (longer `kq`), which L1 bounds for an
  FP32 `B`, or a handoff that lets a chunk's reloads run under the chunk before (with F4a's channels).

**F4a and F5 merged** (2026-10-30, `traced-generation` at `84f45fe`, with Qwen3.5's traced generation at `a12dc1e` and
`step245`'s restated claim at `68a0463`): the 1.7B decode step 10.93 / 12.14 ms (card 1, `1793396168`; F5's kernel
adds nothing measurable to F4a's schedule, the step being bound by its weights' reads), H1 through Burn 30.6 / 22.5 /
11.4 / 14.3 TFLOPS (card 1, `1793396429`); `--smoke` 1,265 of 1,265 on card 0 (`1793397733`); `step260`-`step263` by
name pass, the models' figures unchanged ([new-architectures.md](../plans/new-architectures.md)). F6 opens on this tree.

## Stage F6 (gated on F4a and F5): fusion through channels and spatial pipelining, H5 and H10

Opened 2026-10-31 at `3613d8f`, the merged F4a/F5 tree. H4's verdict leaves per-unit work balance and the ops between
products as the levers. Two hypotheses (design reference §13): **H10**, register fusion (a product's consumer applied in `Dst`
before the pack, X5a), and **H5**, spatial pipelining (`down` consuming `gate|up` through a cross-core channel on a
disjoint core set). Each is bounded on the card before it is built, and its kill applied to that bound first, as F4a.0
did.

What exists to build on:
- the only epilogue is a reduction's exact `then` (`tensor::finishing`);
- no burn-fusion fuser takes a `Matmul` (the bias `Add` opens `add_relu`'s block);
- `l1::Requirements::fuse` has no production caller;
- every op runs on units `0..k`;
- a product's `SCATTER` already writes into its readers' L1 (residency Phase 3, X5c's write side).

- [x] **F6.0, open the stage (card 1, measure and bound before any build).** Done 2026-10-31
  ([log](../learnings/performance-log.md#f60-register-fusion-and-spatial-pipelining-bounded-on-the-card-2026-10-31-card-1)):
  **H10's kill fires** (decode bound 0.10-0.16 ms; training 1.7-3.2% on the card), **H5's does not** (tails 296-365
  GB/s, `gate|up`'s tail slack 465 us a step). No `Cast` between a decode product and its add (BF16-stored weights).
  Found and fixed: a planned capture of a product half of whose output nothing reads was refused (`step219`).
  - Baselines at `3613d8f`: the 1.7B decode (`traced_generation`, 32 / 256 tokens, and a profile), `bench_training`,
    `qwen35_traced_generation` (information).
  - H10's bound: the 1.7B step with the candidate consumers made free (their lists dropped, timing only), (a) the 48
    residual adds after `o` and `down`, (b) those and the 24 `silu_mul`s; for eager training, the jobs a step that
    matmul + bias (+ relu) as one job saves, priced from the profile.
  - H5's bound: the tails' aggregate read rate against 469 GB/s (`tails.tsv`, `channels.tsv`), and the overlap
    `gate|up`'s tail and `down`'s head give a layer.
  - Check that a BF16-stored weight puts no `Cast` between a decode `Matmul` and its residual `Add` in the IR.
  - *Kill before the build:* H10's decode bound < 0.3 ms and its training bound < 5% (then F6.1-F6.3 are not built);
    H5's tails ≥ 90% of 469 GB/s or its overlap < 0.3 ms (then F6.4 is not built).
- [-] **F6.1, the register epilogue in the contraction lowering** (H10's mechanism; ttsim, then card 0). Not built:
  H10's kill fired at F6.0.
  - An `Epilogue` on the matmul `TileOp`: an optional third operand (a bias row, `rows: Fixed`, or a residual tile),
    SFPU kinds, the result's edge `TileFormat`. It enters `Mma`'s memo shape and the program keys; its operand's slot
    enters `Mma::needs` and the contraction's gathers.
  - `Chunk::last`, the chain's final chunk (the chunk order can be reversed); the epilogue runs on it alone.
  - The math role: after the last `MVMUL` of an output or subblock, wait for the matrix unit, run the SFPU program on
    the output's `Dst` rows, post `ready` after the SFPU. Three sites: the single-output kernel, the looped block, the
    subblocked half-sync kernel. Row loops unrolled (the replay buffer holds the `MVMUL` unit; `.ttinsn` depth ≤ 2);
    staged kinds refused (their rows overlap section 1).
  - The unpack role: the operand to `Dst` by `UnpackToDst`, only after the math role hands the half over (never beside
    running `MVMUL`s, F5's finding); a subblock with no free row refuses an operand or narrows.
  - X5's rounding rule: a fused intermediate is rounded in place to the edge format the unfused graph packed it to.
    FP32 needs nothing; BF16 and FP16 by the SFPU's exact narrowing, through `TileFormat` so the epilogue names no
    format; BFP refused by name (no SFPU replay of the packer's rounding exists).
  - A residual is `(Σ terms) + R`, an SFPU `ADD` after the last term, never a reloaded prior.
  - Oracle: `matmul_order` models the epilogue as an event after the last term and before the pack; values against
    `sfpu::ops::reference_op`. Every job golden unchanged without an epilogue; new pinned cases with one.
  - Gate `step266_matmul_epilogue`: every schedule × {bias + relu, residual, silu, a BF16 edge} bit for bit the product
    and then the op, with the special values, ragged shapes, reversed chunks and a replay; negative controls (the
    epilogue on a non-last chunk, a residual as a prior, the rounding skipped) fail on ttsim and card 0. `SMOKE` entry.
- [-] **F6.2, routing**, each route measured against its F6.0 bound. Not built: H10's kill fired at F6.0.
  - `MatmulEpilogueFuser` (B13c) on `Fusion<TtBackend>`: `Matmul`, then an `Add` of a row or a same-shape tensor, then
    an optional activation; `ChainFuser`'s rules (longest complete match, scalars bound at execution, `run_unfused`).
    `add_relu`'s block stays when it does not match (`step12_mnist`'s counts).
  - Decode on `Tt`: `o` + add and `down` + add, the residual a reader of its resident slot in the residency plan.
  - Training: MNIST's and the transformer's biased linears.
  - Default only where the prediction holds, bit for bit; otherwise opt-in (`TT_MATMUL_EPILOGUE`) or retired, with run
    IDs.
- [-] **F6.3, `silu(gate) * up` in `gate|up`'s epilogue** (only if F6.0's (b) - (a) ≥ 0.15 ms; it was ~0.02, not built): a column pair (`gate`'s
  tile `j`, `up`'s `j + ff/32`) dealt to one job as a constraint of the deal, not a model's relayout; one `SILU_MUL`
  tile packed.
- [x] **F6.4a, the overlap today's placement takes** (2026-10-31, card 1, `tt_kernels::barriers::probe_unorder`): the
  barriers around each `silu_mul` dropped, timing only: **0.17-0.23 ms slower** (`1793458026`, `1793458285`,
  `1793458542`; profile `1793458836`: read-ahead unchanged, products 433 -> 415 GB/s). Early `down` reads slow
  `gate|up`'s slowest units (T8).
- [-] **F6.4, H5: `down` consuming `gate|up` through a cross-core channel** (not built: F6.4a's probe, owner's
  decision 2026-10-31) (only if F6.0's H5 bound clears): `down`
  on a core set disjoint from `gate|up`'s (a placement range beside `Spread`'s count), waiting per `K` chunk on a
  `channel::Ring` credited as `gate|up`'s columns land, checked by `Graph::check`; chunks in increasing `k`, so bit for
  bit. Gate `step267`; ttsim if it takes the credits, else silicon-only with a divergence entry.
- [x] **F6.5, close** (2026-10-31): verdicts in §13, the roadmap, `kernel-fusion.md` (B13c) and the log. No
  arithmetic or schedule changed; the residency fix's gate is `step219` (card 0 `1793424934`).
- **Prediction:** set by F6.0's bounds (H10, H5). **Kill:** both bounds below their thresholds, or a built step
  moving less than half its bound.
- **Result (2026-10-31, card 1): both killed, nothing built.** H10's bound is 0.10-0.16 ms a 1.7B step (every
  residual add and `silu_mul` removed outright) and 1.7-3.2% of an eager training step on the card. H5's bound
  cleared (the tails are not GDDR-queue-bound, `gate|up`'s leave 465 us a step), but the overlap probe made the step
  0.17-0.23 ms slower: reads issued beside a product's slowest units slow them (T8). With F4a's and F5's, the
  evidence is that a 1.7B decode step is bound by the slowest units of its products and by GDDR as one queue, not by
  barriers, small ops or idle capacity. **The plan is complete**: every stage is closed; F4b's leftovers are carried in
  the roadmap.

## Not in scope

- Opt-in contracts: split-K, flash attention, packer L1 accumulation, a 16-bit `Dst`, and fidelity below the contract. Each is the owner's call.
- Multi-card schemes.
