# Performance scoreboard

This is the reference for how fast the firmware is: the data mover, the
B → T0/T1/T2 path, and E1. It holds what is true now, how to measure it and the
stated bottlenecks. Dated runs, change history and the evidence behind each row are in
[performance-log.md](performance-log.md).

**Contract.** AGENTS.md requires performance numbers to be recorded with their benchmark
conditions and run IDs. After each optimization, run `cargo xtask bench` and update the
Scoreboard rows it moved **in place**; history (what the row used to be, the run that
moved it, conditions and raw tables) goes in the [Change log](performance-log.md#change-log)
and the dated sections of the log, never as a second copy here.

## How to measure

- `cargo xtask bench` runs every benchmark in release, one per process. It writes
  the results to `target/silicon/bench/<stamp>.{jsonl,md}`.
- **Device-timed by default:**
  - Tensix: the tile's debug timestamper, one counter for B and T0–T2.
  - Ethernet: E1's own `mcycle`, gated alone in `silicon_eth_clock` (ttsim does not
    model it; divergence row 71).
  - *Host* figures use `Instant` and include host submission and synchronization;
    payload upload/download is included only when the benchmark times it.
- **Each figure:** the median of 9 runs after a warm-up. Every benchmark checks
  that its data arrived intact.
- **Code:**
  - `crates/tt-tests/tests/silicon_bench_{memory,path,eth}.rs`
  - Shared helpers and peaks: `crates/tt-tests/src/bench.rs`
- **Conditions:** card 0 (p150a), AICLK 1350 MHz, GDDR6 16000 MT/s × 8 channels.

| Ceiling | Value | Source |
|---|--:|---|
| NoC link, one NIU, one direction | 86.4 GB/s | 64 B flit/cycle × 1350 MHz (`NoC/README.md:64`) |
| GDDR6, one channel | 64 GB/s | 16000 MT/s × 32 pins (x32 per channel is reasoned, not published) |
| GDDR6, card | 512 GB/s | 8 channels |
| Ethernet, one link | 50 GB/s | 400 GbE per Ethernet tile |
| 800G (both links, one QSFP-DD) | 100 GB/s each way | 2 tiles per port |

## Scoreboard

The single-tile mover and the Ethernet wire are near their ceilings. Card-wide
GDDR6 reads reach 92% of the card with the current ownership rules and in-flight
cap; card-wide writes and 800G are still off their ceilings. Raw mover and Ethernet
rows were refreshed in run `1791060908`; these bypass the streaming executor.

| Target | Now | Of ceiling | Goal |
|---|--:|--:|---|
| Resident matrix chain `(a+b)*b`, two tiles, host | 64×64: 24.6–25.6 µs; 65×70: 41.1–42.4 µs (TF32/BF16, cards 0/1; `1791317719`) | separate calls 35.5–37.1 / 57.8–58.4 µs; 1.38–1.48× | explicit Session composition; no application speedup claim |
| Matrix ELW packed BF16 equal-shape add, 64×64, two tiles, host | 40.2 / 41.9 µs (cards 0 / 1, initial M1) | SFPU adapters 105.1 / 108.6 µs | reduce per-tile dispatch costs; SFPU remains default |
| Matrix ELW row multiply, 65×70, two tiles, host | F32 41.6 / 42.0 µs; BF16 60.8 / 61.3 µs (cards 0 / 1) | initial materialized RHS F32 ~1129 / 1133 µs; BF16 ~1157 / 1161 µs | hold direct broadcast addressing; reduce per-tile dispatch costs |
| GDDR6, one tile, 64 KiB entries | 81.4 GB/s | 94% NoC link | hold |
| GDDR6, one tile, 4 KiB entries | 16.3 GB/s | 26% channel | cut per-entry cost |
| GDDR6, one tile, 4 KiB entries, `fast-reads` build (host-encoded reads, tight loop; list-level trace) | 57.0 GB/s (14.9 in the safe default; 2026-10-13, card 1) | 89% channel | hold: one channel's own rate |
| GDDR6 reads, card, 4 / 8 / 32 tiles, 4 KiB entries: safe -> `fast-reads` | 59 -> 219 / 119 -> 340 / 411 -> 461 GB/s (card 1) | 12% -> 43% / 23% -> 66% / 80% -> 90% | 512 GB/s |
| GDDR6 reads, card, 120 tiles, 2 KiB entries: safe -> `fast-reads` | 401 -> 452 GB/s (card 0; three runs 451-453) | 78% -> 88% | 512 GB/s |
| GDDR6 reads, card, 120 tiles | 469 GB/s | 92% | 512 GB/s |
| GDDR6 reads, card, 120 tiles, a weight's consecutive tiles as `READ_RUN` records, traced (`silicon_probe_decode_stream`, card 0) | **487-489 GB/s**; dealt a column a unit (an `m = 1` product's `B`) 446-458, on 64 or 120 units (`1792261878`) | 95% | the decode products' ceiling: they reach 229-387 GB/s ([decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 0) |
| GDDR6 writes, card, 120 tiles (writes on NoC #1) | 389 GB/s | 76% | 512 GB/s |
| GDDR6 reads + writes, card, 120 tiles (writes on NoC #1) | 403 GB/s | 79% | 512 GB/s |
| Ethernet wire, one link (per-byte slope) | 48–51 GB/s | ~97% | hold |
| 800G, both links streaming | 27.0 GB/s | 27% | 100 GB/s |
| Empty kernel, B → T0–T2 → B (device) | 566 cycles (0.42 µs) | — | lower |
| Device busy, 16 queued 32×256×32 matmuls | 12.6% of wall | — | ~100% |
| MNIST training, card 0, 1 tile (`tt-mnist --host`) | 1.4 ms/step | burn-flex 0.5 | ≤ burn-flex |
| Transformer training, card 0, 1 tile (`tt-mnist --model transformer`) | 8.8 ms/step | burn-flex 4.35 | ≤ burn-flex |
| ResNet-18 forward, 224x224, batch 1, card 1, host (`silicon_bench_models`) | one tile (the default): 782.3 ms, 211 reported ops (`1791996580`; was 16,288 ms, `1791846735`); every tile, `TT_SPREAD=work:64:4`: **131.1 ms** (`1791996716`); every tile, spread `all`: 203.8 ms (`1791996730`) | burn-flex 156.9 / 151.4 / 151.3 ms in the same runs (one tile 5.0x slower; every tile at `work:64:4` 1.15x **faster**) | chunk-free convolution and pooling landed (roadmap Track 2 item 2): the host submits a forward in ~15 ms, the rest is the card. On one tile conv1 alone is ~366 ms (49 taps of a 3-channel table, ~1.2M 64-byte reads) and the image's relayout ~95 ms; the default tile choice (roadmap, "Per-workload tile choice") is what is left |
| ViT-Tiny forward, 224x224, batch 1, card 1, host | every tile, `TT_SPREAD=work:64:4`: **129.1 ms** (`1792026124`; 173.5 before the scores stayed in tile rows, `1792019882`; 283.4 on card 0 before attention-jobs, `1792013104`); one tile: 368.3 ms (`1792026138`); 550 server jobs | burn-flex 109.3 / 109.7 ms in the same runs (1.18x and 3.4x slower) | `attention-jobs.md` done. Left on every tile: `attention/pv` 30 ms of server time, the patch convolution, the heads' merge |
| SmolLM2-135M decode step, embedding lookup excluded, card 1, host | every tile, `TT_SPREAD=work:64:4`: **86.5 ms** (p10 86.0, p90 89.1), 1,696 server jobs (`1792019882`; 99.6 ms and 3,316 jobs on card 0 before attention's products were one job, `1792013124`); one tile: 449.7 ms (`1792019917`) | burn-flex 26.5 ms in the same run (3.3x slower on every tile) | attention 36 ms of server time (was 73): 10 jobs a call, was 64 (`attention-jobs.md`, done; a single query keeps the stacked path). One tile waits on its m = 1 matmuls (1.28 ms each) |
| SmolLM2-135M decode step from a token id, lookup included, card 0, host | 623 ms (p10 525, p90 709), 2,781 reported ops (`1791852097`) | burn-flex 26.4 ms (23.6x slower); the same run without the lookup 622 ms | the lookup is about 1.4 ms of it; the rest is the host submission above |
| SmolLM2-135M embedding lookup, one token from 49,152 x 576, card 0, host (`probe_embedding_cost`) | 260 us, median of 7 warm, validated against the host's rows (`1791851940`); five tokens 278 us, 32 tokens 344 us | was 15.9 s (`1791845218`): `gather_native` staged 28.3 million host addresses a token; now `Session::select_rows` reads only the rows it returns | hold; BF16, F16 and BFP tables still take the staged path (roadmap Track 2, "Gather without staging") |
| SFPU precise `gelu`, three-role kernel, device cycles per tile (`probe_ttinsn_push`, card 0) | 38,053 cycles (T1 push 3.31 cycles/word: push-bound) | `.ttinsn` fused push, experiment only: 13,734 cycles, 2.77x (run-time translated: 2.67x) | decide the follow-on (roadmap Track 2) |
| Accurate matmul (opt-in `MatmulMode::Accurate`), F32 `[256, 256] @ [256, 256]`, card 0, host | 10.96 ms (`1791818553`) | the fast Matrix Unit matmul: 277 us (39.6x); 19.1x to 64.4x over seven shapes, 18.7x at 1024 cubed; the exact mode is 750x slower than this at 256 cubed | the element-wise slicing and NaN/infinity classification are 64% to 86% of it (127 device jobs a matmul): fuse the slicing into one SFPU kind a slice, cheapen the classification, one matmul for the pairs; see roadmap, "Fused slicing for `MatmulMode::Accurate`" |
| Exact matmul (opt-in `MatmulMode::Exact`), F32 `[256, 256] @ [256, 256]`, card 0, host | 8.19 s (about 0.5 us per product, 16 chunks; `1791761944`) | the fast Matrix Unit matmul: 376 us (22,000x); 64x32x64 1,000x, 1024x27x32 3,300x | cut the host-side table cost: a k-loop of broadcast multiply-adds (m + n addresses a step) instead of two `k x outputs` gathers; see roadmap Track 1 |
| Matmul rate at prefill shapes, every tile, TF32/HiFi4, traced replay (`matmul_throughput`, card 0) | `[256,2048]@[2048,2048]` 2.0, `@[2048,8192]` 3.5, `[1024,2048]@[2048,2048]` 3.4, `@[2048,8192]` **4.2 TFLOPS** (`1792258604`) | ~1.2-2.5% of ~166 (derived, HiFi4 on 120 tiles); GDDR reads at 192-390 GB/s | operand reuse: `B` is read `m / 32` times and `A` up to once a unit (3.2 GB for 77 MB of operands) |
| Requests that waited under the in-flight cap (120 tiles × 300 × 16 KiB, one channel) | 8762 per run | — | watch: revisit `MAX_IN_FLIGHT` if it binds below the NoC's own limit |

### Model ladder: now against target

Pretrained models, batch 1, F32, release, medians, burn-flex timed in the same run (`silicon_bench_models`; method and
conditions in the [log](performance-log.md#model-ladder-real-models-on-the-card-2026-10-12-cards-0-and-1)). Update a
row in place when its fix lands, add the run ID, and keep the old row in the log. Work items: roadmap Track 2, items 1-7.

| Workload | burn-flex | Now (run) | Slowdown | Target | Roadmap item |
|---|--:|--:|--:|---|---|
| ResNet-18 forward, 224x224 (card 1), one tile (default) | 156.9 ms | 782.3 ms (`1791996580`; was 16,288 ms on card 0, `1791846735`) | 5.0x | at or below burn-flex | 2 (done), then the default tile choice |
| ResNet-18 forward, 224x224 (card 1), every tile, `TT_SPREAD=work:64:4` | 151.4 ms | 131.1 ms (`1791996716`) | 0.87x | at or below burn-flex: **met** with every tile | default tile choice |
| ViT-Tiny forward, 224x224 (card 1), every tile, `work:64:4` | 109.3 ms | 129.1 ms (`1792026124`; 173.5 after attention-jobs P1-P2, 283.4 on card 0 before; one tile 368.3, `1792026138`) | 1.18x | at or below burn-flex | 9 (done), then the default tile choice |
| SmolLM2-135M decode step, lookup excluded (card 1), every tile, `work:64:4` | 26.5 ms | 86.5 ms (`1792019882`; 99.6 on card 0 before attention-jobs P1-P2; one tile 449.7, `1792019917`) | 3.3x | at or below burn-flex | 9 (attention), then the default tile choice |
| SmolLM2-135M embedding lookup, one token (card 0) | not measured | 260 us (`1791851940`; was 15.9 s, `1791845218`) | n/a | milliseconds: **met** for F32 | 1 |
| SmolLM2-135M decode step from a token id, lookup included (card 0) | 26.4 ms | 623 ms (`1791852097`) | 23.6x | at or below burn-flex | 3 |
| SmolLM2-135M BF16 decoded token, six layers, lookup included (card 0) | not measured | 37.5 s (`1791848957`) | n/a | no slower than F32 | 5 |
| ViT-Tiny forward, 224x224 (card 1), every tile (`TT_TILES=all`), traced replay | 111.9 ms | 11.6 ms (`1792036850`; 12.6 before `replay-fixed-cost.md`, `1792026627`; `work:64:4` 61.9, `1792026634`; one tile does not capture) | 0.10x | at or below burn-flex: **met** (one tile: "Trace segmentation") | 6 |
| SmolLM2-135M decode step, lookup excluded (card 1), every tile (`TT_TILES=all`), fixed shapes, traced replay | 26.4 ms | **26.46 ms**, and with the cache written a row a head 24.40 / 24.84 / 25.53 ms a step of 32 / 128 / 256-token generations (`1792110130`, burn-flex 26.66 / 28.97 / 31.12), and with thin matmul reads 24.14 / 24.57 / 25.25 (`1792118871`, burn-flex 26.05 / 28.24 / 30.44), and fused projections 22.91 / 23.35 / 23.98 (`1792162031`, burn-flex 26.43 / 28.21 / 30.65), and block views (RoPE and SiLU without repacks or copies) 22.46 / 22.95 / 23.63 (`1792174412`, burn-flex 29.52 / 30.95 at 128 / 256), and on `Tt` with the decode fusers (sigmoid, `silu(gate) * up`, RMSNorm's statistic) 20.20 / 20.66 / 21.34 (`1792186583`, burn-flex 26.94 / 30.74 / 31.61), and the statistic over a decode row's row groups 19.94 / 20.42 / 21.11 (`1792189939`, burn-flex 27.96 / 29.87 / 32.60), and the cache written in place 18.59 / 19.07 / 19.79 (`1792263801`, burn-flex 26.11 / 28.53 / 31.02), and element-wise ops on a decode row's row groups 17.51 / 17.97 / 18.76 (`1792269106`, burn-flex 27.3 / 30.1 / 31.2), and a transposed `B` transposed by the unpacker 17.10 / 17.52 / 18.25 (`1792273005`, burn-flex 26.8 / 28.9 / 31.3), and a stacked product's tail rows left unread 16.05 / 16.48 / 17.29 (`1792280238`, burn-flex 28.0 / 29.2 / 31.2), and the batched product's padding off `K` used as it is 15.33 / 15.79 / 16.47 (`1792281384`, burn-flex 26.9 / 29.3 / 31.5), and the softmax in four ops 14.80 / 15.29 / 15.86 (`1792287884`, burn-flex 26.6 / 29.0 / 31.4), and RoPE's sum as one op 14.43 / 14.92 / 15.46 (`1792290681`, burn-flex 26.9 / 29.1 / 31.1), and thin products on 64 units with the scale and mask as one op 14.28 / 14.68 / 15.25 (`1792293687`, burn-flex 26.6 / 29.0 / 31.3), and the cache write's and the embedding's moves spread 14.04 / 14.45 / 15.04 (`1792296202`, burn-flex 26.7 / 29.0 / 31.6), and the position-major cache 13.71 / 14.16 / 14.66 (`1792337842`, burn-flex 27.9 / 35.0 / 31.3), and the decode products' rows, a packed `A`, a streamed `K` and prefetched weights 12.62 / 13.06 / 13.59 (`1792364661`, burn-flex 27.0 / 29.3 / 31.5), and a column operand broadcast by the SFPU 11.37 / 11.76 / 12.29 (`1792372749`, burn-flex 39.7 / 30.5 / 33.1; 11.40 / 11.80 / 12.26 after the rest of the small-ops phase, `1792378993`, burn-flex 27.0 / 29.2 / 31.7), and decode attention as one op 9.58 / 10.11 / 10.96 (`1792425715`, burn-flex 27.2 / 29.8 / 32.4), and the token on the card, RoPE as one op and three copies gone **8.00 / 8.59 / 9.38** (`1792445319`, burn-flex 27.3 / 29.5 / 32.0) (`1792036613`: replay 25.8, logits read 0.47; was 37.1, `1792029144`, see `replay-fixed-cost.md`; before it `work:64:4` 83.5, `1792029158`; one tile 478.3, `1792029169`) | 1.00x | at or below burn-flex: **met** in the same run, by 0.04 ms (burn-flex ranged 25.9-26.6 over the day's runs) | 6 |
| SmolLM2-135M prefill of 256 tokens (card 1), every tile (`TT_TILES=all`), one traced chunk | 1,098.2 ms | 73.1 ms (`1792036724`; a 32-token chunk 34.8, 64 38.7, 128 49.7; was 79.0, `1792031467`) | 0.07x | at or below burn-flex: **met** | 6 |
| ResNet-18 forward, 224x224 (card 1), every tile (`TT_TILES=all`), traced replay | 164.1 ms | 17.2 ms (`1792036818`; 17.6, `1792000003`; untraced, same settings, 205.7 ms) | 0.10x | at or below burn-flex: **met** (one tile does not capture: program cache, item "Trace segmentation") | 6 |
| SmolLM2-135M generation (card 1), every tile, `TracedGenerator`, 512-position cache: 256-token prompt + 64 tokens, ids in, logits out | 3,015 ms (step 31.57; 32-token prompt 1,866, step 26.75) | **1,795 ms** (step 27.31; 32-token prompt **1,688**, step 26.13; `1792078489`, after the GQA fold and step lengths; was 2,110 and 2,077, `1792076190`) | 0.60x (0.90x at 32 tokens) | at or below burn-flex: **met** at every prompt length measured | 6 (step 4 done; left: the whole-cache write, longer chunks) |
| SmolLM2-1.7B-Instruct decode step (card 1), every tile, traced, F32, 512-position cache, 32 / 128 / 256-token prompt | 294.4 / 307.8 / 334.7 ms (3.4-3.0 tokens/s) | **20.95 / 21.41 / 22.18 ms** (47.7-45.1 tokens/s; `1792444936`, the token on the card, RoPE as one op and three copies gone, burn-flex 296.5 / 307.7 / 326.4 in that run; 22.24 / 22.70 / 23.43 before them, `1792432143`, RMSNorm as one op and one region a job, burn-flex 298.3 / 313.7 / 330.8 in that run; 22.78 / 23.26 / 23.98 before them (300.6 GB/s of weights at 32 tokens, 59% of 512; `1792425331`, decode attention as one op, burn-flex 296.0 / 310.2 / 328.3 in that run); 24.04 / 24.48 / 25.39 before it, `1792383988`, packed twins and a barrier released by unit 0; 24.41 / 25.01 / 25.93 before them, `1792378639`, the prefetch keeps its units, burn-flex 296.1 / 312.7 / 328.3 in that run; 24.72 / 25.38 / 26.17 before it, `1792377589`, row gathers dealt by rows, burn-flex 301.1 / 314.2 / 333.3 in that run; 25.38 / 26.03 / 26.91 before it, `1792374654`, RMSNorm's statistic from the packed row, burn-flex 306.3 / 318.9 / 327.3 in that run; 25.48 / 26.18 / 27.01 before it, `1792372396`, a column operand broadcast by the SFPU, burn-flex 294.4 / 305.2 / 329.2 in that run; 26.47 / 27.14 / 27.97 before it, `1792364210`, the decode products' rows, a packed `A`, a streamed `K` and prefetched weights, burn-flex 301.1 / 311.6 / 327.3 in that run; 31.72 / 32.31 / 33.21 before them, `1792337485`, the position-major cache, burn-flex 295.8 / 312.8 / 331.8 in that run; 33.02 / 33.52 / 34.42 before it, `1792298195`; 33.05 / 33.59 / 34.51, `1792295836`, the cache write's and the embedding's moves spread, burn-flex 298.9 / 305.6 / 329.4 in that run; 34.35 / 34.87 / 35.76 before it, `1792292588`, thin products on 64 units and the attention's scale and mask as one op, burn-flex 295.0 / 313.2 / 327.2 in that run; 35.64 / 36.22 / 37.20 before them, `1792290295`, RoPE's sum as one op; 36.12 / 36.52 / 37.54 before it, `1792287482`, the softmax in four ops; 36.39 / 37.02 / 37.98 before it, `1792280991`, the batched product's padding off `K` used as it is; 36.89 / 37.30 / 38.36 before it, `1792279829`, a stacked product's tail rows left unread; 37.24 / 38.66 / 40.50 before it, `1792272596`, burn-flex 310.1 / 308.4 / 325.7 in that run, a transposed `B` transposed by the unpacker; 37.69 / 40.20 / 43.41 before it, `1792268720`, element-wise ops on a decode row's row groups; 38.97 / 41.41 / 44.58 before it, `1792263398`, the cache written in place; 41.83 / 44.25 / 47.40 before it, `1792209913`, a product's later blocks on a unit reading `A` once; 42.97 / 45.41 / 48.59 before it, `1792189445`, RMSNorm's statistic over a decode row's row groups only; 43.42 / 45.85 / 49.00 before it, `1792186180`, on `Tt` with the decode fusers: sigmoid, `silu(gate) * up`, RMSNorm's statistic; 45.23 / 47.71 / 50.78 before them, `1792174499`, block views; 47.90 / 50.37 / 53.53 before them, `1792161623`, fused projections; 49.78 / 52.27 / 55.44 before them, `1792118484`; 57.21 / 59.64 / 62.81 before the thin matmul reads and spread last round, `1792109477`; 73.65 / 76.08 / 79.30 before the cache write of one row a head, `1792082994`) | 0.17x | at or below burn-flex: **met**; next, GDDR's bound (~14.6 ms at 469 GB/s; now 123-138 GB/s). Planned L1 residency, **on by default** since 2026-10-21 (2h: rows held by 2 readers, never-read writes dropped, the region sized under the products' cliff; `TT_RESIDENCY=0` turns it off): 21.00 / 22.25 -> **20.07 / 21.28 ms** at 32 / 256 tokens on card 0 (`1792556241`, `1792554892`); Phase 3, the products' outputs in their readers' L1: **19.50 / 20.71 ms** (`1792595329`, off 21.02 / 22.21 in the same session, `1792593062`); with busy prefetch, once NC stopped draining B's prefetch ID, **on by default** at a prefetch cap of 2: **18.39 / 19.90 ms** (`1792615740`; 19.51 / 20.76 with `TT_BUSY_PREFETCH=0`, `1792615881`); Phase 4 re-scoped and the reader sections read ahead (`dm::TXN_FETCH`): 18.32-18.44 / 19.78-19.83 ms, unchanged at the default (`1792619550`, `1792619858`): the small ops 5.00 -> 4.47 ms and the products 12.92 -> 13.46 (profile `1792620146`), the weight stream taking what they gave up; 19.27 / 20.48 with `TT_BUSY_PREFETCH=0` (`1792620919`); the stream's rate (2026-10-22, card 0, decode-to-bandwidth Phase 8): the prefetch `WAIT` drained at B's read cap 18.28 / 19.84 (`1792631901`), then a weight stream's reads at 2 in flight (`record::GATHER_STREAM`, `dm::STREAM_CAP`), which serves the product's units evenly: **17.27 / 18.71 ms** (`1792637346`; the 135M unchanged), the step's weights at 416 GB/s over its spans (389 before, profiles `1792629809`, `1792638075`) | in bytes (profile `1792210770`, [decode-step-bandwidth.md](../plans/decode-step-bandwidth.md)): the step moves 9.3 GB, 19.8 ms at 469 GB/s, the weights 14.8 ms; products 22.8 of 41.2 ms at 351 GB/s (`gate\|up` 384, `o` 295, `down` 324: not at GDDR's ceiling), `A` 0.94 GB, the cache copies 0.83 GB, the other 1,007 ops 17.4 ms at 70 GB/s; earlier, matmuls 24.0 of 44.7 ms (`1792171171`, [decode-matmul-bandwidth.md](../plans/decode-matmul-bandwidth.md)); the other ~20.7 ms small ops (RMSNorm ~2.8 up to its scale, `1792190118`: the statistic 26.5 us and `DIV` 30.9 a norm; cache write ~5.9 with the key's repack, attention ~4.4; RoPE 1.4, was 4.9: [decode-small-ops.md](../plans/decode-small-ops.md)) |
| SmolLM2-1.7B-Instruct prefill (card 1), every tile, traced, 32 / 128 / 256 tokens | 6.0 / 19.0 / 21.9 tokens/s | **315 / 741 / 861 tokens/s** (101.7 / 172.6 / 297.4 ms to the first token; `1792258449`; 267 / 657 / 771 before the decode work, `1792082994`) | 0.03x | at or below burn-flex: **met**; next, the [roofline](#roofline-smollm2-17b-f32-one-card) (~12,000 tokens/s at 256) | prefill products' reuse (roadmap Track 1) |
| Traced rows for ViT-Tiny and SmolLM2 | n/a | not measured | n/a | measure once segmentation lands | 6 |

### Roofline: SmolLM2-1.7B, F32, one card

The ceiling the 1.7B rows above are working toward (2026-10-17; evidence and the product bench in the
[log](performance-log.md#the-17b-against-its-roofline-and-a-prefill-products-rate-2026-10-17-cards-0-and-1)). The
assumptions:
- **Bytes:** F32, 4 bytes a parameter. Decode moves 6.85 GB of weights plus the cache (32 key/value heads, 0.2 GB at
  512 positions).
- **Bandwidth:** 512 GB/s. Reads measure 469 across 120 tiles, so 14.8 ms is the weights' practical floor.
- **Compute:** ~166 TFLOPS, HiFi4 on 120 tiles. Derived (774 TFLOPS LoFi / 140 tiles / 1,350 MHz = 4,096 FLOP a cycle
  a tile, over four phases), not measured.
- **"Realistic":** 65% of the bandwidth, 45% of the compute.
- **Target for decode:** 90-95% of the bandwidth over the whole step: ~14.5-15.5 ms a 1.7B step
  ([decode-to-bandwidth.md](../plans/decode-to-bandwidth.md)). "Realistic" is a waypoint.

| Case | Theoretical | Realistic | Now | Off realistic | What binds now |
|---|--:|--:|--:|--:|---|
| Decode, batch 1, ~512 positions (256-token prompt) | 13.8 ms | 25.0 ms | 22.18 ms (`1792444936`) | 0.89x | ~7.6 ms of small ops, ~1.9 of them on 1-2 units, and products 15.6 ms at 462 GB/s (at the read ceiling), their weights partly read under the small ops (profile `1792382914`, `Step(64)`; 10.0 ms of small ops before the column broadcast in the SFPU, `1792363365`; [decode-to-bandwidth.md](../plans/decode-to-bandwidth.md#the-larger-changes-compared-2026-10-18)); earlier, ~18 ms and 22.8 ms at 347 GB/s ([decode-step-bandwidth.md](../plans/decode-step-bandwidth.md) B0) |
| Decode, batch 1, short context (32-token prompt) | — | — | 20.95 ms (`1792444936`) | — | the same, attention now one op a layer (35 us at 64 positions, 89 at 512: [log](performance-log.md#decode-attention-as-one-op-2026-10-19-cards-0-and-1)) |
| Decode, batch 4-32 | 174-1,233 tokens/s | 132-863 tokens/s | not supported (batch 1 only) | — | no batched decode |
| Prefill, 64 tokens | 13.4 ms | 20.7 ms | ~102 ms (a 32-token prompt in the 64 chunk) | ~5x | per-op fixed cost (~48 ms a chunk) |
| Prefill, 256 tokens | 13.6 ms | 20.9 ms | 297.4 ms (861 tokens/s) | 14x | products bound by operand re-reads (2.0-4.2 TFLOPS) |
| Prefill, 512-8,192 tokens | 13.8-209 ms | 23.8-464 ms | not captured (chunks stop at 256, the cache at 512) | — | trace size, program cache |

- **Decode's products alone already run near "realistic":** 22.8 ms at 347 GB/s. The gap is the rest of the step.
- **Prefill's products are ~40x from compute-bound,** and GDDR-bound only because they re-read their operands.

### Where each row is evidenced

| Scoreboard rows | Evidence in performance-log.md |
|---|---|
| Resident matrix chain | [Resident matrix chain benchmark](performance-log.md#resident-matrix-chain-benchmark-2026-10-06) |
| Matrix ELW equal-shape / row multiply | [M1 matrix elementwise comparison](performance-log.md#m1-matrix-elementwise-comparison-2026-10-06), [Direct matrix RHS broadcast comparison](performance-log.md#direct-matrix-rhs-broadcast-comparison-2026-10-06) |
| GDDR6 one tile, card reads/writes, in-flight cap | [GDDR6 through the data mover](performance-log.md#gddr6-through-the-data-mover), [GDDR and Ethernet remeasurement](performance-log.md#gddr-and-ethernet-remeasurement) |
| Ethernet wire, 800G | [Ethernet and 800G](performance-log.md#ethernet-and-800g) |
| Empty kernel B → T0–T2 → B | [Core path: B → T0/T1/T2](performance-log.md#core-path-b--t0t1t2) |
| Device busy, MNIST and transformer ms/step | [Streaming ownership rollout](performance-log.md#streaming-ownership-rollout), [Change log](performance-log.md#change-log) |
| Streaming traced/fresh sweeps | [Overlap thresholds for NC ownership](performance-log.md#overlap-thresholds-for-nc-ownership-run-1791079782-2026-10-04) |
| Operation baselines (BF16, packed, mesh) | [Native](performance-log.md#native-operation-baselines-2026-10-05), [BF16 and mesh](performance-log.md#bf16-and-mesh-operation-baselines-2026-10-05), [Packed BF16](performance-log.md#packed-bf16-update-2026-10-05) |
| Accurate matmul against the fast one | [Accurate matmul cost](performance-log.md#accurate-matmul-cost-2026-10-12-card-0) |
| Exact matmul against the fast one | [Exact matmul cost](performance-log.md#exact-matmul-cost-2026-10-11-card-0) |
| Model forward passes (ResNet-18, ViT-Tiny, SmolLM2) | [Model ladder](performance-log.md#model-ladder-real-models-on-the-card-2026-10-12-cards-0-and-1); ResNet-18 since: [Chunk-free convolution and max pooling](performance-log.md#chunk-free-convolution-and-max-pooling-2026-10-14-cards-0-and-1) |
| Embedding lookup, SmolLM2 decode from a token id | [Row select](performance-log.md#row-select-an-embedding-lookup-without-staging-2026-10-13-card-0) |
| Traced generation (prefill buckets and decode over one cache) | [Traced generation, end to end](performance-log.md#traced-generation-end-to-end-2026-10-15-cards-0-and-1), [Decode length and trace size](performance-log.md#decode-length-and-trace-size-2026-10-15-cards-0-and-1), [SmolLM2-1.7B-Instruct](performance-log.md#smollm2-17b-instruct-on-the-card-2026-10-15-card-1), [A cache write of one row a head](performance-log.md#a-cache-write-of-one-row-a-head-2026-10-16-cards-0-and-1) |
| Roofline, matmul rate at prefill shapes, 1.7B prefill | [The 1.7B against its roofline](performance-log.md#the-17b-against-its-roofline-and-a-prefill-products-rate-2026-10-17-cards-0-and-1) |
| SFPU `gelu` kernel, `.ttinsn` push fusion | [`.ttinsn` push fusion](performance-log.md#ttinsn-push-fusion-precise-gelu-2026-10-10-card-0) |
| P2 K-blocking, R3 norm, Approx, X6 | [P2/R3](performance-log.md#p2-k-blocking-and-r3-norm-release-baselines-2026-10-10), [Approx transcendentals](performance-log.md#approx-transcendentals-per-tile-saving-2026-10-10-card-0), [X6](performance-log.md#x6-mover-niu-fast-path-2026-10-10-card-0) |

## Current architecture (streaming ownership)

Streaming ownership is the only GDDR compute scheduler.
The legacy wave, B-only pipeline and two-host-queue NC schedulers are removed.
`Session::enable_dram(b, nc)` requires both images; Burn rejects retired
`TT_EXECUTION` and `TT_SCATTER` settings rather than silently ignoring them.
`TT_PIPELINE=0` and `TT_BATCH=0` change overlap/waiting, not ownership.
Standalone transfer/control primitives and diagnostic movers remain available.
Known fresh-execution regressions are accepted to consolidate optimization work;
the historical rollout gate did **not** pass. Measurements in the
[Streaming ownership rollout](performance-log.md#streaming-ownership-rollout) log predate removal
and remain the legacy comparison baseline. Current sweeps compare serialized
versus overlapped streaming only.

## Where the overhead is

**Current (2026-10-03 measurements):** fresh multi-tile submission remains
host-bound; streaming overlap thresholds still need tuning for NC ownership;
small mover requests pay per-entry setup cost; Ethernet still stores/forwards
one transfer at a time. Concurrent B/NC timestamp exports can be unbalanced,
so host timing and ownership counters are the primary evidence. Streaming
ownership, traced overlap, DMA and asynchronous Burn dispatch are implemented.
The full-reduction addition (2026-10-04) is correctness-gated separately; the
scoreboard above has not been remeasured for it. Full `sum`/`mean` now always
use native SFPU arithmetic, including in exact mode. Exact-mode MNIST uploads
its 256-byte host-produced loss vector for the mean and reads back 4 bytes;
the old 5120-byte steady-state transfer budget is now 5380 bytes. The resident
transformer loss instead removes its 512-byte mean-input download. Mesh
reductions stage through L1 on chip 0 and are limited by L1/program capacity.

**Historical ranking (before streaming consolidation):**

1. Host submission in every op/transfer: about 100 µs per matmul and 3 µs
   per Ethernet send at that baseline.
2. Card-wide writes used only NoC0. NC now writes on NoC1: the current
   raw 120-tile write measurement is 389 GB/s.
3. Per-entry request setup: about 339 cycles, with 116 before touching NIU
   registers at that baseline. B/NC caches were later measured near 4 KiB.
4. Gather/compute/scatter were sequential. Fixed ownership now overlaps them
   where the scheduler selects depth two and storage permits reuse.
5. E1's single staging buffer and a host round trip per transfer remain.
6. Single-port write collapse was fixed by static VC1 (see below).
7. Per-role wake/setup and instruction-push cost remains a tuning target.

**Open issues:**

- [x] **8-bit counter hazard (fixed).** The mover's completion wait reads
  `NIU_MST_REQS_OUTSTANDING_ID`, which is 8 bits and wraps at 256 in flight. A list
  or an on-tile record (up to 65536 tiles) could be reported done before its data
  landed.
  - `noc::issue` now caps each transaction ID at `MAX_IN_FLIGHT` = 128
    (`tt_isa::noc::niu::InFlight`).
  - Waits are counted in the mover mailbox (`THROTTLE_STALLS`, `THROTTLE_CYCLES`).
    You can read them with `DataMover::throttle` or `Session::throttle`.
  - A session prints a `session:` warning the first time it sees waits, and again
    each time they double.
  - Benchmarks report the waits per run; traced lists carry a `THROTTLE` event.
  - Tests: `step49_in_flight` (500 requests in one list), and
    `silicon_bench_memory::gddr_in_flight` (300 requests per tile × 120 tiles on
    one channel, which must stay ≤ the channel's ceiling).
- [x] **Single-port writes (fixed by static VC 1).**
  - Before: 128 KiB write entries collapsed to 27–33 GB/s, and 16 KiB single-port
    writes varied from run to run.
  - After every request went to static VC 1, one port holds 62.9 GB/s at 64 KiB
    and 128 KiB (run 1790970443).
  - The single-port write path changed in nothing else: port hint 0 maps to port
    0 on NoC #0.
- [ ] **Per-request cost** (checklist 9.14). The detailed observations below
  are from the 2026-10-02 profile, before streaming consolidation and the
  direction-specialized images; removed symbols such as `dm::Peer` are historical.
  - **Where a 4 KiB read entry's ~350 cycles go** (2026-10-02). Measured with a
    temporary build stamping the wall clock between stages, about 8 cycles per
    stamp taken off:

    | Stage | Cycles |
    |---|--:|
    | Fetch the entry, record length, trace check | ~30 |
    | Decode and dispatch (`Entry::decode`) | ~88 |
    | Build the request (`Command::registers`) | ~110 |
    | In-flight cap check | ~14 |
    | Initiator-busy poll | ~9 |
    | The ten NIU register writes | ~10 |
    | Issue and read-back | ~6 |
    | Return and trace | ~47 |

  - **The NIU's MMIO is nearly free; the software around it is not.**
    `Command::registers` re-checks every request after `Descriptor::decode`
    has checked the whole move.
  - **Writing only the registers that change was tried and dropped.**
    - Every register reads back as written (`step51_nc_probe`, card 0).
    - But comparing against a copy cost more than the writes it saved: +21
      cycles per entry.
  - **The fixes so far:**
    - Moves are checked once per descriptor and issued from registers
      (`DramMove`, `issue_dram_on`): −40 cycles per entry.
    - **Records walk rows with a cursor.** A software division per gathered
      tile (32 steps) was more than half of a gathered tile's cost. Removing it
      halved matmul gathers.
    - **A record's plain moves skip the general entry path.** Each goes from
      `record::expand` through `Descriptor::decode` straight to the issue
      (`record_entry`), placed beside `issue`.
    - **The firmware divides in hardware** (6–33 cycles). The instruction gate
      had refused `div`/`rem` on the false belief that T2 lacks them; only T2's
      vector divide is missing. Transposed, broadcast and per-row divisions
      gain.
  - **Entry size matters.** Carrying a whole `dm::Mover` in an `Entry` variant
    made every decode about 100 cycles slower, as did decoding rare kinds out
    of line. Both were fixed by a one-byte `dm::Peer` and inline decode.
  - ~350 cycles per entry at any size up to 4 KiB.
  - **Re-measured 2026-10-13** ([read-loop probe](performance-log.md#read-loop-probe-what-the-interpreter-costs-per-read-entry-2026-10-13-card-0)):
    359 device cycles per read entry (64 B or 4 KiB), against 90-106 for a bare loop or a replay of host-encoded
    requests; `Entry::decode` + `DramMove` + issue alone is 189. The host-timed 0.48 us an entry is those 359 cycles
    plus ~0.21 us of list upload.
  - Each request rebuilds and writes ten NIU registers.
  - GATHER records expand into one ~4 KiB read per tile.
- [x] **Resident compute streams, fresh and traced** (checklist 9.15).
  B reads, resident T0–T2 compute and NC writes under batch credits, with one
  role generation per compatible region. Depth-two selection and backend
  retirement still need tuning; long row-sum chunks retain their ordered
  intermediate dependencies.
- [~] **The host queues units one at a time** (checklist 9.17): ~6 µs a unit
  an op, now ~3.4 (PCIe reads, separate barrier lists and program-memo misses
  gone). The rest is three posted writes and CPU work a unit: follow-ups are one
  list fanned out on the card or further reductions in fresh submission work.
  Trace replay is implemented. Low-level `KERNEL` remains for setup/control;
  GDDR compute uses resident streaming regions.
- [ ] **Ethernet moves one transfer at a time** per direction, store and forward
  through one buffer each way (checklist, Ethernet pipelining).
- [x] **Instruction-cache size:** ~4 KiB on B and NC (`probe_icache`).
- [ ] **Not yet measured:**
  - tile-to-tile L1 over the NoC (the mover has no op for it)
  - card 1 (`cargo xtask bench --device all`)
