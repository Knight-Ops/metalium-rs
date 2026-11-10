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
- **The burn-flex baseline changed on 2026-10-29.** Until then burn-flex was built without
  `rayon`, so every "burn-flex" figure below ran on one core. Since then it uses all 12
  host cores (`Cargo.toml`, [test tiers](test-tiers.md#burn-flex-on-every-core)). Compare a
  new run's burn-flex column only with runs from after that date.

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
| GDDR6 reads, card, 120 tiles | 469 GB/s (64 KiB entries, cap 8); `fast-reads` ~505 steady, 495 overall (2026-10-28, card 0, [log](performance-log.md#why-card-wide-reads-plateau-at-469-and-what-it-means-for-decode-2026-10-28-card-0)) | 92% | 512 GB/s |
| GDDR6 reads, card, 120 tiles, a weight's consecutive tiles as `READ_RUN` records, traced (`silicon_probe_decode_stream`, card 0) | **487-489 GB/s**; a column a unit (an `m = 1` product's `B`) 446-458 (`1792261878`) | 95% | the decode products' ceiling ([decode-to-bandwidth.md](../completed-plans/decode-to-bandwidth.md) Phase 0) |
| GDDR6 reads, tt-metal's Blackhole CI gate (`8_dram_adjacent_core_read`) on our movers (`gddr_adjacent_banks`, card 1, [log](performance-log.md#tt-metals-adjacent-core-dram-read-gate-on-our-mover-2026-10-26-card-1)) | **507.7-508.3 GB/s** `fast-reads`, 16 KiB entries; **511.2** with 128 KiB; safe image 414 / 511 (`1793028574`, `1793028275`) | 99% / 99.8% | tt-metal's gate: 490-512 GB/s |
| GDDR6 writes, card, 120 tiles (writes on NoC #1) | 389 GB/s | 76% | 512 GB/s |
| GDDR6 reads + writes, card, 120 tiles (writes on NoC #1) | 403 GB/s | 79% | 512 GB/s |
| Ethernet wire, one link (per-byte slope) | 48–51 GB/s | ~97% | hold |
| 800G, both links streaming | 27.0 GB/s | 27% | 100 GB/s |
| Empty kernel, B → T0–T2 → B (device) | 566 cycles (0.42 µs) | — | lower |
| Cross-core credit handoff, two units, 64 B page (`silicon_bench_channel_credit`, card 1, host, H8) | **1.81 µs** (`1793200301`, 2026-10-28) | two-unit barrier 2.66 µs polled / 2.59 released: 0.70x | per page or batch until H4 measures overlap (D6) |
| Device busy, 16 queued 32×256×32 matmuls | 12.6% of wall | — | ~100% |
| MNIST training, the default: every tile, traced (`tt-mnist --host`; `cargo xtask bench --filter bench_training`), a step after the first | default 784-512-256-10 **1.31-1.39 ms/step**, `--hidden 128` 1.06; eager every tile 1.55-1.81, eager one tile 2.17 (card 0, 2026-11-09, [log](performance-log.md#training-on-every-tile-traced-by-default-2026-11-09-card-0)); one tile eager was 2.882 on 2026-10-27 (`1793128372`, total over the steps) | burn-flex 2.28-2.62 in the same runs | ≤ burn-flex: **met** |
| Transformer training, the default: every tile, traced (`tt-mnist --model transformer`) | **3.75 ms/step**; eager every tile 9.42, eager one tile 8.52 (card 0, 2026-11-09, same log) | burn-flex 3.86-4.36 in the same runs | ≤ burn-flex: **met** |
| ResNet-18 forward, 224x224, batch 1, host (`silicon_bench_models`) | every tile at `work:64:4` **145.9 ms** (not bisected against 131.1 of 2026-10-14), spread `all` 232.4, traced 12.90 (card 1, 2026-10-27, `1793119700`, `1793119618`); one tile (the default) 782.3 (2026-10-14, `1791996580`) | burn-flex 155.5 ms (2026-10-27); 156.9 beside the one-tile run | the default tile choice; on one tile conv1 is ~366 ms, the image's relayout ~95 |
| ViT-Tiny forward, 224x224, batch 1, card 1, host | every tile at `work:64:4` **116.2 ms**, spread `all` 223.3, traced 7.89 (2026-10-27, `1793119700`, `1793119618`); one tile 368.3 (2026-10-15, `1792026138`) | burn-flex 106.0-107.1 ms in the same runs | left on every tile: `attention/pv`, the patch convolution, the heads' merge |
| SmolLM2-135M decode step, embedding lookup excluded, card 1, host | every tile at `work:64:4` **86.5 ms** (p10 86.0, p90 89.1; 2026-10-14, `1792019882`); one tile 449.7 (`1792019917`) | burn-flex 26.5 ms in the same run | attention 36 ms of server time; one tile waits on its m = 1 matmuls (1.28 ms each) |
| SmolLM2-135M decode step from a token id, lookup included, card 0, host | 623 ms (p10 525, p90 709), 2,781 reported ops (`1791852097`) | burn-flex 26.4 ms (23.6x slower); without the lookup 622 ms | the lookup is ~1.4 ms; the rest is host submission |
| SmolLM2-135M embedding lookup, one token from 49,152 x 576, card 0, host (`probe_embedding_cost`) | 260 us, median of 7 warm, validated (`1791851940`); five tokens 278 us, 32 tokens 344 us | was 15.9 s (`1791845218`) before `Session::select_rows` | hold; BF16, F16 and BFP tables still take the staged path (roadmap, "Gather without staging") |
| SFPU precise `gelu`, three-role kernel, device cycles per tile (`probe_ttinsn_push`, card 0) | 38,053 cycles (T1 push 3.31 cycles/word: push-bound) | fused `.ttinsn` push 13,734 cycles, 2.77x; the default since 2026-10-25 ([log](performance-log.md#fused-ttinsn-push-from-the-host-the-default-2026-10-25-card-0)) | done; what stays push-bound is in the profile mode's push table |
| Accurate matmul (opt-in `MatmulMode::Accurate`), F32 `[256, 256] @ [256, 256]`, card 0, host | 10.96 ms (`1791818553`) | the fast matmul 277 us (39.6x); 19.1x-64.4x over seven shapes | slicing and NaN/infinity classification are 64-86% of it; roadmap, "Fused slicing for `MatmulMode::Accurate`" |
| Exact matmul (opt-in `MatmulMode::Exact`), F32 `[256, 256] @ [256, 256]`, card 0, host | 8.19 s (~0.5 us a product, 16 chunks; `1791761944`) | the fast matmul 376 us (22,000x); 64x32x64 1,000x, 1024x27x32 3,300x | cut the host-side table cost (roadmap, "A faster exact matmul") |
| Matmul rate at prefill shapes, every tile, TF32/HiFi4, traced replay, through Burn with an FP32 `B` (`matmul_throughput`) | F5 (`Dst` subblocks, half-sync; the default since 2026-10-30, bit for bit): `[2048,2048]@[2048,16384]` / `[1024,2048]@[2048,8192]` / `[1024,8192]@[8192,2048]` / `[2048,8192]@[8192,2048]` **30.4 / 24.7 / 10.9 / 14.5 TFLOPS** on card 0 (`1793388902`), 30.6 / 22.5 / 11.4 / 14.3 on card 1 (`1793396429`) ([log](performance-log.md#f5-dst-subblocks-and-half-sync-2026-10-30-card-0)) | ~7-18% of ~166 (derived) | the gathers: each operand out of GDDR once, then the performance contract (roadmap W1, W2) |
| Requests that waited under the in-flight cap (120 tiles × 300 × 16 KiB, one channel) | 8762 per run | — | watch: revisit `MAX_IN_FLIGHT` if it binds below the NoC's own limit |

### Parity with tt-metal, same card

tt-metal v0.77 (`d53c2a71fb0`) on card 1, 2026-11-09, against our current rows on card 1. Roadmap W0. Method, conditions
and full tables are in the [log](performance-log.md#tt-metal-on-the-same-card-smollm2-17b-and-its-products-2026-11-09-card-1).
Re-run it when a W item lands (`crates/tt-tests/bench/ttnn_parity/`).

| Workload | Ours (contract) | tt-metal (precision) | Ours / theirs |
|---|---|---|---|
| SmolLM2-1.7B decode, batch 1, ms a step | **9.90** at 32 tokens, 10.95 at 256 (reference, bit for bit); BFP8 weights **7.51** / 8.66, BFP4 6.34 / 7.49 (F32 activations, HiFi4; `1794261034`, `1794261722`, [W1 bound](performance-log.md#w1-bounded-the-performance-contract-on-this-stack-before-any-plumbing-2026-11-09-cards-0-and-1)) | 10.55 performance (BFP4/BFP8); 11.3 all-BF16; 12.24 accuracy (~128-token prompts) | **0.94x reference, 0.71x BFP8, 0.60x BFP4 against 10.55: ahead** |
| SmolLM2-1.7B decode, batch 32 | not supported | 24.82 ms, 40.3 tokens/s/user, **1,289 tokens/s** | ~13x behind in throughput |
| SmolLM2-1.7B prefill, ~128 tokens | 3,084 tokens/s (reference; BFP8 weights 3,245; `1794260351`, `1794261034`). The "~2,150 since 2026-11-03" was the stream-ceiling probe's row, not a regression | TTFT 18.3 ms, ~6,600 tokens/s (performance) | 2.0-2.1x behind |
| GEMM `2048x8192x2048` | 49.2 TFLOPS F32 `A` x BF16 `B` HiFi4; 46.9 BF16-HiFi2; 65.6 BF16 x BFP8 LoFi (session, slope-timed, `1794258133`); 14.3 through Burn with an FP32 `B` | 204 BF16-HiFi2; 117 BF16-HiFi4; 315 BFP8-LoFi | 2.4-4.8x behind |
| GEMM `2048x2048x16384` | 55.6 / 48.4 / 67.6 TFLOPS (as above); 30.6 through Burn | 122 / 74 / 192 | 1.3-2.8x behind |
| Decode products, 24 layers + head, M=32, derived | 11.9 ms BF16 x BFP8 HiFi2, 11.0 LoFi, 17.4 the reference (probe sums, no prefetch, `1794257871`) | 7.34 ms BF16-HiFi2; 11.61 BF16-HiFi4; 6.41 BFP8-HiFi2; **4.22 BFP8-LoFi** | 1.9-2.6x behind: a BF16 `A` takes faces and reads its row tile by tile (W1) |
| Decode product rate, BF16-HiFi2, best | — | 453-479 GB/s (`down`, `gate\|up`, head) | — |
| Qwen3.5-4B decode, batch 1, ms a step ([log](performance-log.md#tt-metal-on-the-same-card-qwen35-4b-2026-11-09-card-1)) | 62.48 host-timed (reference, BF16-stored weights), of which ~21.4 the on-card greedy pick: ~41 without it | 38.0 on the card, 44-46 host-timed with a host argmax (`qwen36` path; BFP4 MLP, BFP8 rest, LoFi) | 1.4x behind host-timed; ~1.08x on the card without the pick (W4, then W1) |

### Model ladder: now against target

Pretrained models, batch 1, F32, release, medians, burn-flex timed in the same run (`silicon_bench_models`; method and
conditions in the [log](performance-log.md#model-ladder-real-models-on-the-card-2026-10-12-cards-0-and-1)). Update a
row in place when its fix lands, add the run ID, and keep the old row in the log. Work items: [roadmap](../plans/roadmap.md) tracks W1-W6.

| Workload | burn-flex | Now (run) | Slowdown | Target | Roadmap item |
|---|--:|--:|--:|---|---|
| ResNet-18 forward, 224x224 (card 1), one tile (default) | 156.9 ms | 782.3 ms (2026-10-14, `1791996580`) | 5.0x | at or below burn-flex | 2 (done), then the default tile choice |
| ResNet-18 forward, 224x224 (card 1), every tile, `TT_SPREAD=work:64:4` | 155.5 ms | 145.9 ms (2026-10-27, `1793119700`, `1793119618`) | 0.94x | **met** | default tile choice |
| ViT-Tiny forward, 224x224 (card 1), every tile, `work:64:4` | 106.0-107.1 ms | 116.2 ms (2026-10-27, `1793119700`, `1793119618`) | 1.09x | at or below burn-flex | 9 (done), then the default tile choice |
| SmolLM2-135M decode step, lookup excluded (card 1), every tile, `work:64:4` | 26.5 ms | 86.5 ms (`1792019882`) | 3.3x | at or below burn-flex | 9 (attention), then the default tile choice |
| SmolLM2-135M embedding lookup, one token (card 0) | not measured | 260 us (`1791851940`) | n/a | milliseconds: **met** for F32 | 1 |
| SmolLM2-135M decode step from a token id, lookup included (card 0) | 26.4 ms | 623 ms (`1791852097`) | 23.6x | at or below burn-flex | 3 |
| SmolLM2-135M BF16 decoded token, six layers, lookup included (card 0) | not measured | 37.5 s (`1791848957`) | n/a | no slower than F32 | 5 |
| ViT-Tiny forward, 224x224, every tile (`TT_TILES=all`), traced replay | 106.0-107.1 ms | **7.89 ms** (card 1, 2026-10-27, `1793119700`, `1793119618`); 8.32 on card 0 (2026-10-25, `1792963996`) | 0.07x | **met** (one tile: "Trace segmentation") | 6 |
| SmolLM2-135M decode step, lookup excluded, every tile (`TT_TILES=all`), BF16 weights, traced, 32 / 128 / 256-token generations | 26.5 / 29.1 / 30.5 ms | **6.92 / 7.47 / 8.14 ms** (card 0, 2026-10-27, `1793059270`, [log](performance-log.md#the-decode-and-prefill-streams-merged-2026-10-27-card-0)) | 0.26x | **met** | 6 |
| SmolLM2-135M prefill of 256 tokens (card 1), every tile (`TT_TILES=all`), one traced chunk | 1,098.2 ms | 73.1 ms (`1792036724`; 32 / 64 / 128-token chunks 34.8 / 38.7 / 49.7) | 0.07x | **met** | 6 |
| ResNet-18 forward, 224x224, every tile (`TT_TILES=all`), traced replay | 155.5 ms | **12.90 ms** (card 1, 2026-10-27, `1793119700`, `1793119618`); 12.72 on card 0 at `38f87a6` (`1793129414`) | 0.08x | **met** (one tile does not capture: "Trace segmentation") | 6 |
| SmolLM2-135M generation (card 1), every tile, `TracedGenerator`, 512-position cache: 256-token prompt + 64 tokens, ids in, logits out | 3,015 ms (step 31.57; 32-token prompt 1,866, step 26.75) | **1,795 ms** (step 27.31; 32-token prompt **1,688**, step 26.13; `1792078489`) | 0.60x (0.90x at 32 tokens) | **met** at every prompt length measured | 6 (left: the whole-cache write, longer chunks) |
| SmolLM2-1.7B-Instruct decode step (card 1), every tile, traced, weights stored BF16 (the F32 model's values, bit for bit), 512-position cache, 32 / 128 / 256-token prompt | 294.4 / 307.8 / 334.7 ms | **9.90 / 10.95 ms** at 32 / 256 tokens (101 tokens/s), the default even deal (2026-11-07, mean of the medians of `1794024534`, `1794025139`, `1794025528`; `TT_EVEN_DEAL=0` 10.14 / 11.18; [log](performance-log.md#the-even-deal-the-default-a-17b-decode-step-1014---990-ms-2026-11-07-cards-0-and-1)). Weights stored F32, dated: 16.79 / 17.19 / 18.09 ms (2026-10-26, card 0, `1793002966`, [log](performance-log.md#re-tuned-for-bf16-weights-2026-10-26-card-0)) | 0.03x | **met**; next, 90-95% of GDDR's bandwidth over the step ([roofline](#roofline-smollm2-17b-f32-one-card)) | 2b ([decode-bytes-in-flight.md](../plans/decode-bytes-in-flight.md)): products bound by their reads in flight ([log](performance-log.md#a-decode-products-pipeline-metered-from-inside-read-bound-by-its-reads-in-flight-2026-11-02-cards-0-and-1)) |
| SmolLM2-1.7B-Instruct prefill (card 1), every tile, traced, BF16 weights, 32 / 128 / 256 tokens | 6.0 / 19.0 / 21.9 tokens/s | **1,075 / 3,040 / 3,827 tokens/s**, paged programs restored on the card (H15, 2026-11-01, `1793549551`, [log](performance-log.md#program-restores-on-the-card-paged-programs-copied-back-by-the-b-movers-2026-11-01-cards-0-and-1)). **Confirmed 2026-11-09:** 1,098 / 3,084 / 3,890 (`1794260351`); the ~1,090 / ~2,150 / ~2,090 quoted since 2026-11-03 were `traced_generation_stream_ceiling`'s rows (prefill in chunks of 64), not a regression ([log](performance-log.md#w1-bounded-the-performance-contract-on-this-stack-before-any-plumbing-2026-11-09-cards-0-and-1)) | 0.006x | **met**; next, the [roofline](#roofline-smollm2-17b-f32-one-card) (~12,000 tokens/s at 256) | roadmap W1, W2 |
| SmolLM2-1.7B-Instruct prefill at long context, every tile, traced, 8,192-position cache, chunks of 512, 1,000 / 7,992 tokens | 1,847 s for 7,992 tokens and 8 more (`1792800061`, `generate_fixed` on the host) | **0.284 / 5.98 s** (3,524 / 1,336 tokens/s; card 0, 2026-10-27, `1793060076`; card 1 with BF16 weights the same, `1793039593`; [log](performance-log.md#the-decode-and-prefill-streams-merged-2026-10-27-card-0)) | ~0.003x | **met** | mostly attention over the prefix (`pv`, the softmax's mask read once a head; [log](performance-log.md#the-long-prompt-softmax-a-column-read-once-a-run-the-statistics-in-the-passes-that-make-their-input-2026-10-26-card-0)) |
| Qwen3.5-4B decode step (card 1), every tile, traced, BF16-stored weights, 32 / 256-token prompt | not measured | **62.48 / 62.84 ms**, the default even deal (2026-11-07, `1794027144`, `1794027958`). Where it goes (2026-11-08, profiled, `1794101823`, `1794105660`; [log](performance-log.md#qwen35-4bs-step-on-the-card-a-third-of-it-is-the-greedy-pick-the-even-deals-04-is-the-head-2026-11-08-cards-0-and-1)): **~21.4 ms the greedy pick** (8 argmax parts, each on one unit), head 5.4-5.6 ms, weight products ~19.5 ms | n/a | ~2x the 17.5 ms BF16 roofline (new-architectures.md) | Track 1, new architectures |
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
  ([decode-to-bandwidth.md](../completed-plans/decode-to-bandwidth.md)). "Realistic" is a waypoint.
- **"Now" stores the weights BF16** (half these bytes, the F32 model's values bit for bit), so a decode below the F32
  theoretical is no contradiction.

| Case | Theoretical | Realistic | Now | Off realistic | What binds now |
|---|--:|--:|--:|--:|---|
| Decode, batch 1, ~512 positions (256-token prompt) | 13.8 ms | 25.0 ms | 10.95 ms (2026-11-07, see the decode row) | 0.44x | each product's reads in flight ([log](performance-log.md#a-decode-products-pipeline-metered-from-inside-read-bound-by-its-reads-in-flight-2026-11-02-cards-0-and-1)) |
| Decode, batch 1, short context (32-token prompt) | — | — | 9.90 ms (the same runs) | — | the same |
| Decode, batch 4-32 | 174-1,233 tokens/s | 132-863 tokens/s | not supported (batch 1 only) | — | no batched decode |
| Prefill, 64 tokens | 13.4 ms | 20.7 ms | ~30 ms (a 32-token prompt, 1,075 tokens/s, `1793549551`) | ~1.4x | a short chunk's products on few units ([log](performance-log.md#a-64-token-prefill-chunk-its-products-on-8-47-units-2026-10-31-card-1)) |
| Prefill, 256 tokens | 13.6 ms | 20.9 ms | ~67 ms (3,827 tokens/s, `1793549551`) | ~3.2x | the products' operand reuse (roadmap W2) |
| Prefill, 512-8,192 tokens | 13.8-209 ms | 23.8-464 ms | 1,000 tokens **284 ms**, 7,992 **5,984 ms** (card 1, `1793039593`) | ~13x at 7,992 | the products' gathers; attention over its prefix |

- **On 2026-10-17, decode's products alone ran near "realistic"** (22.8 ms at 347 GB/s, F32); the gap was the rest of the step.
- **On 2026-10-17, prefill's products were ~40x from compute-bound,** GDDR-bound only because they re-read their operands.

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

**Status:** this section is historical: its content dates from the 2026-10-03 measurements and before streaming
consolidation. Current bottlenecks are in the Scoreboard and Model ladder rows above and their log entries.

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
