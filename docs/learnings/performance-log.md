# Performance log

History and evidence, not current status. Current numbers, the measurement method
and the stated bottlenecks are in [performance-scoreboard.md](performance-scoreboard.md).
Sections here were appended as they were measured; each section keeps its own date
and run IDs. Newest dated sections come first, then the undated baseline sections
(streaming rollout, GDDR6, core path, Ethernet) and the 2026-10-03 reader/writer split.
The chronology is the [Change log](#change-log).

## Index of sections

| Section | Date | Note |
|---|---|---|
| [Change log](#change-log) | 2026-10-02 to 2026-10-10 | table newest first, then dated `###` subsections (2026-10-04, 2026-10-06) |
| [Residency plans over reused GDDR](#residency-plans-over-reused-gddr-2026-10-24-card-0) | 2026-10-24 | G2: chunk captures reuse their GDDR and keep their L1 residency plans; prefill 1.2-2.1% faster at reuse's free GDDR, the first build 2.4-6.6x longer |
| [Prefill products that reuse their operands](#prefill-products-that-reuse-their-operands-2026-10-24-card-0) | 2026-10-24 | 2-D blocked products, `K` streamed, partials in L1 (the default since; `TT_MATMUL_BLOCK=off` turns it off): the 1.7B's prefill products **2.9-3.6x** at `m` >= 512 (`gate\|up` 5.21 -> **18.23** TFLOPS at `m` = 2,048), compute-bound at ~0.4-0.6 us a K-tile pair; its prefill at 8,192 positions with chunks of 512 **1.24 / 6.81 / 18.33 -> 0.48 / 3.70 / 12.59 s** (1,000 / 4,000 / 7,992 tokens) |
| [GDDR reused within a capture](#gddr-reused-within-a-capture-2026-10-24-card-0) | 2026-10-24 | 1.7B at 8,192 positions, chunk captures reusing their intermediates' GDDR (the default since; `TT_CHUNK_REUSE=0` turns it off): GDDR free after the build **2.10 -> 17.78 GB**, the build 226 -> 127 s, prefill unchanged; chunks of 512 now capture: prefill 1,000 / 7,992 tokens 1,236 / 18,323 -> **1,011 / 16,877 ms** |
| [How much of the card's compute a long prefill uses](#how-much-of-the-cards-compute-a-long-prefill-uses-2026-10-24-card-0) | 2026-10-24 | 1.7B prefill at 8,192 positions: **1.1-1.7% of the ~166 TFLOPS derived peak** (1.87 TFLOPS over a 7,992-token prompt); a 64-token chunk's projections 3.73 TFLOPS with the Matrix Unit busy 4% of their span; products plateau at **5.2 TFLOPS** (`gate\|up`) and 2.8 (`down`) even at `m` = 2,048, GDDR-bound at ~400 GB/s: larger chunks do not raise the rate, operand reuse does |
| [A long prompt's prefill: chunks attend to a prefix, the GQA mask read as a block](#a-long-prompts-prefill-chunks-attend-to-a-prefix-the-gqa-mask-read-as-a-block-2026-10-24-card-0) | 2026-10-24 | 1.7B, 8,192-position cache, chunks of 64: prefill of 1,000 / 4,000 / 7,992 tokens **4.69 / 18.46 / 36.63 s -> 1.24 / 6.81 / 18.33 s** (213-218 -> 809 / 587 / 436 tokens/s); the mask's row gather alone was 36% of a chunk; decode unchanged |
| [The 1.7B at 8,192 positions from a 7,992-token prompt](#the-17b-at-8192-positions-from-a-7992-token-prompt-2026-10-24-card-0) | 2026-10-24 | traced prompt and 8 tokens 38.7 s against burn-flex's 1,847 s, the same tokens; chunks of 64 |
| [The 1.7B at its native 8,192 positions](#the-17b-at-its-native-8192-positions-2026-10-23-card-0) | 2026-10-23 | first run past a 512-position cache, after looped element-wise and reduce programs (`long-context.md`): builds (235 s first, 72.5 s second), **17.0 ms** a step after 32 tokens, **26.0 ms** after 1,000 (attention as the composition at `Step(1024)`); past 8,192 the traces do not fit the program cache |
| [Flash-decode's probe: a head's positions in two halves](#flash-decodes-probe-a-heads-positions-in-two-halves-2026-10-23-card-0) | 2026-10-23 | each head's attention as two halves on 64 units with **no exchange** (timing only): isolated, 80.6 -> 60.0 us at 512 positions; in a 1.7B step 16.81 / 18.19 -> **17.08 / 17.71 ms**, 19.7 us a layer at `Step(512)` before any exchange, below the 20 us-after-exchanges stop rule; stopped, A5 parked; today's job does not fit L1 past 512 positions, halves fit to 1024 |
| [A product's units metered, and a balanced deal](#a-products-units-metered-and-a-balanced-deal-2026-10-23-card-0) | 2026-10-23 | each unit's bytes in a product's span against its time (`skew.tsv`): `gate\|up`'s units 0-47 read 1.33 MB each at ~6 GB/s while its 40 prefetching units finish in 13 us; dealing those columns over all 104 units (opt-in, `TT_BALANCED_DEAL=1`) is bit for bit but **not faster** (16.81 / 18.21 -> 16.86 / 18.21 ms): each unit reads slower as more read, the span's aggregate rate stays ~350-390 GB/s, and the head gets slower (830 -> 925 us); the longest unit does not bind |
| [Layer sub-traces: stored bodies and calls](#layer-sub-traces-stored-bodies-and-calls-2026-10-23-card-0) | 2026-10-23 | one layer body a unit and a `CALL_BODY` a layer (opt-in, `TT_SUB_TRACES=1`), bit for bit: the 1.7B's trace streams 675 -> 384 MB (`Step` 19.3 -> 4.9 MB, `Chunk(64)` 107 -> 21), but a 1.7B step **0.15 ms slower** (16.77 / 18.19 -> 16.94 / 18.33) and the 135M 0.05-0.08; not the default |
| [Room for a second job's `B`: the trace arena and second jobs](#room-for-a-second-jobs-b-the-trace-arena-and-second-jobs-2026-10-23-card-0) | 2026-10-23 | a capture's region at the top of `PROGRAMS` and its `B` rings past `DATA`: 16.78 / 18.15 ms, unchanged; idle units taking a second job's `B` there: bit for bit, **slower** (17.18 / 18.54 against 16.77 / 18.14), 266 MB more read ahead but longer barriers; room for `B` no longer binds |
| [The mailboxes packed: the L1 map before room for a second `B`](#the-mailboxes-packed-the-l1-map-before-room-for-a-second-b-2026-10-22-card-0) | 2026-10-22 | the mailboxes moved into 40 KiB above the program slots, 88 KiB of capture-only L1 left beside `DATA`; nothing uses it yet: a 1.7B step 16.79 / 18.20 -> **16.76 / 18.17 ms**, the 135M, ResNet-18 and ViT-Tiny unchanged; a first form that grew `DATA` itself changed every arena-sized choice |
| [Each unit's own prefetch anchor](#each-units-own-prefetch-anchor-2026-10-22-card-0) | 2026-10-22 | each unit's prefetch anchored at the last weight-streaming product it took part in, not the last overall: `gate\|up`'s busy units that sat out `o` read under RoPE, the cache write and attention; a 1.7B step 16.87 / 18.29 -> **16.79 / 18.20 ms**, 429 GB/s, the same 3.48 GB read ahead (L1 holds one job a unit); the 135M 0.01-0.03 ms slower; on by default |
| [A column's `K` chain handed between two units](#a-columns-k-chain-handed-between-two-units-2026-10-22-card-0) | 2026-10-22 | `down`'s whole columns handed at a chunk to spare units, as a second op, bit for bit: a 1.7B step 16.88 / 18.32 -> **17.07 / 18.51 ms** (slower; opt-in): `down` 179 -> 172 us, but the halves contend at GDDR (up to 137 us, not ~88) and `qkv` 51 -> 72 us as the spare units start its read ahead later |
| [`down` prefetched, and the unit counts re-swept](#down-prefetched-and-the-unit-counts-re-swept-2026-10-22-card-0) | 2026-10-22 | each of 40 idle units reads a streamed column's first two chunks ahead: a 1.7B step 17.29 / 18.72 -> **16.87 / 18.30 ms**, 426 GB/s, `qkv` 90 -> 51 us a layer; the prefetch and stream unit counts kept at 40 / 64 / 40; `down`'s 24 busy units now bind it (a whole column each at one unit's rate) |
| [The weight stream metered](#the-weight-stream-metered-what-is-read-ahead-the-prefetchs-wait-and-each-units-own-time-2026-10-22-card-0) | 2026-10-22 | prefetched bytes charged to their product, the prefetch `WAIT` stamped, every unit timed: a 1.7B step's weights stream at **389 GB/s**, 2.97 GB read ahead; unit 0 waits 28 / 67 us a layer for its own prefetch in `qkv` / `gate\|up`; the units read unevenly by where they sit among the readers; a weight stream's reads at 2 in flight even them out: a 1.7B step 18.44 / 19.83 -> **17.27 / 18.71 ms**, 416 GB/s, the 135M unchanged |
| [Residency on by default, its region sized under the products' cliff](#residency-on-by-default-its-region-sized-under-the-products-cliff-2026-10-21-card-0) | 2026-10-21 | a region costs a product only past its plan's floor (110,464 bytes at `K` = 2048: +8-10% one 64-byte step past it); each plan's region sized under its graph's cap, plans cached, residency on by default: a 1.7B step 21.00 / 22.25 -> **20.07 / 21.28 ms**, nothing slower; builds +20-70% (open) |
| [Rows held by a few of their readers, and never-read writes dropped](#rows-held-by-a-few-of-their-readers-and-never-read-writes-dropped-2026-10-20-card-0) | 2026-10-20 | a row many units read whole, held by 2 of them instead of multicast to all, and the writes nothing reads dropped: with `TT_RESIDENCY=1` a 1.7B step 21.03 / 22.25 -> **20.06 / 21.28 ms** (card 0); of the probe's three ways, multicast acknowledgements counted once a run changed nothing |
| [Busy-unit prefetch and the prefetch pump](#busy-unit-prefetch-and-the-prefetch-pump-2026-10-20-cards-0-and-1) | 2026-10-20 | a product's own units read `B` ahead under the ops before it: the products 15.24 -> 11.66 ms, but the small ops' reads queue behind the prefetch (the residual add 14 -> ~100 us): the step 20.97 -> 21.58 ms, at best 20.74 / 21.76 with ID 6's cap at 2 / 1; opt-in |
| [The Phase 8 feasibility probe](#the-phase-8-feasibility-probe-the-step-without-its-weight-stream-and-under-one-2026-10-20-cards-0-and-1) | 2026-10-20 | a 1.7B decode step with its weights free is **7.85 ms** (today 20.92); under a ~300 GB/s stream on idle units its ops cost 1.6x, mostly drains waiting on reads; L1, not bandwidth, bounds a continuous stream; the packed-`A` ring wastes ~254 KB a product unit |
| [The token on the card, RoPE as one op, and three copies gone](#the-token-on-the-card-rope-as-one-op-and-three-copies-gone-2026-10-19-cards-0-and-1) | 2026-10-19 | a row's argmax as one native op and the greedy token read back alone; RoPE as one op; a no-op reshape, the value's copy and the context's repack gone: a 1.7B decode step -> **20.95 / 21.41 / 22.18 ms**, 135M -> **8.00 / 8.59 / 9.38**; sampling on the card works (112 ms a token) |
| [RMSNorm as one op, and one region a job](#rmsnorm-as-one-op-and-one-region-a-job-2026-10-19-cards-0-and-1) | 2026-10-19 | a job's kernels share one semaphore initialisation, so they run as one region (attention 35 -> 25 us a layer); RMSNorm's statistic and `DIV` as one op on 32 units: a 1.7B decode step -> **22.24 / 22.70 / 23.43 ms** |
| [Decode attention as one op](#decode-attention-as-one-op-2026-10-19-cards-0-and-1) | 2026-10-19 | a decode step's attention a key/value head on one unit, seven kernels in L1, bit for bit: a 1.7B decode step -> **22.78 / 23.26 / 23.98 ms**, 135M -> **9.58 / 10.11 / 10.96**; a head's rows at row 0 of its tile, or `SUB_EXP` and `DIV` cost ~8 us a score tile |
| [Decode products: their rows, a packed `A`, a streamed `K`, prefetched weights](#decode-products-their-rows-a-packed-a-a-streamed-k-prefetched-weights-2026-10-18-cards-0-and-1) | 2026-10-18 | the m = 1 kernel computes only its rows, a one-row `A` read packed, a blocked `K` streamed, the next product's weights read on idle units under the small ops, bit for bit: a 1.7B decode step -> **26.47 / 27.14 / 27.97 ms**, 135M -> **12.62 / 13.06 / 13.59** |
| [A position-major key/value cache](#a-position-major-keyvalue-cache-2026-10-18-cards-0-and-1) | 2026-10-18 | the cache stored `[1, 1, L, kv d]`, so a step writes its projection's row as it is (no repack), bit for bit: a 1.7B decode step -> **31.72 / 32.31 / 33.21 ms**, 135M -> **13.71 / 14.16 / 14.66** |
| [The cache write's and the embedding's moves spread](#the-cache-writes-and-the-embeddings-moves-spread-2026-10-18-cards-0-and-1) | 2026-10-18 | a repack's reads in flight together, one batch's row scatter split between units, and a ragged row select over every unit, bit for bit: a 1.7B decode step -> **33.05 / 33.59 / 34.51 ms**, 135M -> **14.04 / 14.45 / 15.04** |
| [Thin products on 64 units; attention's scale and mask as one op](#thin-products-on-64-units-attentions-scale-and-mask-as-one-op-2026-10-18-cards-0-and-1) | 2026-10-18 | a one-tile-row product over a long `K` on at most 64 units (~1.2 ms), and `mask ? -inf : s x` in one pass (~0.1-0.2 ms), bit for bit: a 1.7B decode step -> **34.35 / 34.87 / 35.76 ms**, 135M -> **14.28 / 14.68 / 15.25** |
| [RoPE's sum as one op](#ropes-sum-as-one-op-2026-10-18-cards-0-and-1) | 2026-10-18 | `x cos + rotate(x) sin`'s second product and sum in one pass (`MUL_ADD`, a chain fuser), bit for bit: a 1.7B decode step -> **35.64 / 36.22 / 37.20 ms**, 135M -> **14.43 / 14.92 / 15.46** |
| [Attention's softmax in four ops](#attentions-softmax-in-four-ops-2026-10-18-cards-0-and-1) | 2026-10-18 | `exp(x - max)` in one pass and each clamp on its reduction, bit for bit: a 1.7B decode step -> **36.12 / 36.52 / 37.54 ms**, 135M -> **14.80 / 15.29 / 15.86** |
| [The bound on a decode product's `A` reads](#the-bound-on-a-decode-products-a-reads-2026-10-18-card-0) | 2026-10-18 | every block reading only `B` (a timing probe): a 1.7B layer's products ~143 us faster (`down` 270 -> 198 us), ~3.4 ms a step at most for reading `A` once a product |
| [A stacked product's tail rows left unread](#a-stacked-products-tail-rows-left-unread-2026-10-17-cards-0-and-1) | 2026-10-17 | a batched product's padded left blocks no longer read their tail rows as zeros under a stacked output: a 1.7B decode step -> **36.89 / 37.30 / 38.36 ms**, 135M -> **16.05 / 16.48 / 17.29** |
| [A transposed `B` transposed by the unpacker](#a-transposed-b-transposed-by-the-unpacker-2026-10-17-cards-0-and-1) | 2026-10-17 | a matmul's transposed `B` read as it lies and transposed by unpacker 0 (`Haloize_mode`, faces swapped), not by the mover at ~10 us a tile: bit for bit; a 1.7B decode step -> **37.24 / 38.66 / 40.50 ms**, 135M -> **17.10 / 17.52 / 18.25** |
| [Element-wise ops on a decode row's row groups](#element-wise-ops-on-a-decode-rows-row-groups-2026-10-17-cards-0-and-1) | 2026-10-17 | lanewise SFPU kinds on a `[1, W]` row run 1 of 8 row groups: a 1.7B decode step 38.93 / 41.38 / 44.58 -> **37.69 / 40.20 / 43.41 ms**, 135M -> **17.51 / 17.97 / 18.76** |
| [Captured barriers on the units an op ran on](#captured-barriers-on-the-units-an-op-ran-on-2026-10-17-cards-0-and-1) | 2026-10-17 | a captured op's barrier counts the units it ran on (waits for a unit that sat ops out, in its stream): no step time change (1.7B 38.93 / 41.38 / 44.58 ms); the condition for a cross-op prefetch |
| [A traced cache written in place](#a-traced-cache-written-in-place-2026-10-17-cards-0-and-1) | 2026-10-17 | a traced step's cache donated and written in place (no tile copy, no write-back; a stale read refused by name), the write's indices read as a row: a 1.7B decode step 41.83 / 44.25 / 47.40 -> **38.97 / 41.41 / 44.58 ms**, 135M 20.03-21.19 -> **18.59 / 19.07 / 19.79** |
| [Decode products against their weight stream](#decode-products-against-their-weight-stream-2026-10-17-card-0) | 2026-10-17 | the card reads a weight at **~488 GB/s** (95% of 512) as consecutive tiles on 120 units, 446-458 dealt a column a unit (one channel a column: ~7%, not P5's bound); 64 units saturate; an `m = 1` product takes 1.2-1.8x its weights' stream (`o` 74 against 38 us, `down` 269 against 150): `A`'s reads, a ~15 us kernel tail, the last round |
| [The 1.7B against its roofline, and a prefill product's rate](#the-17b-against-its-roofline-and-a-prefill-products-rate-2026-10-17-cards-0-and-1) | 2026-10-17 | 1.7B prefill re-measured: **315 / 741 / 861 tokens/s** (was 267 / 657 / 771), decode unchanged (41.80-47.44 ms); a prefill-shaped product reaches 2.0-4.2 TFLOPS (~2.5% of HiFi4's ~166), bound by GDDR: `B` re-read per tile row of `A`, `A` per column block (3.2 GB for 77 MB of operands at 390 GB/s) |
| [A decode step in bytes, and the activation once a unit](#a-decode-step-in-bytes-and-the-activation-once-a-unit-2026-10-17-cards-0-and-1) | 2026-10-17 | the profile mode counts every captured op's GDDR bytes and entries (`ops.tsv`, a "Step budget" a trace): a 1.7B step moves 9.6 GB, 20.5 ms at 469 GB/s, against 42.3 ms; its weights 14.8 ms; `A` 1.25 GB; the cache copies 0.83 GB; `gate\|up` at 367 GB/s, not ~490. A unit's later blocks of a product skip `A` (`GATHER_A_RESIDENT`): 1.7B **41.83 / 44.25 / 47.40 ms**. Cross-op prefetch into L1 capped at ~4.4 ms a step |
| [RMSNorm's statistic over a decode row's row groups](#rmsnorms-statistic-over-a-decode-rows-row-groups-2026-10-16-cards-0-and-1) | 2026-10-16 | the statistic kernel runs only the row groups a tile row's rows occupy (1 of 8 for a decode row): its kernel 14.95 -> 5.58 us a norm; a 1.7B decode step 43.42 / 45.85 / 49.00 -> **42.97 / 45.41 / 48.59 ms**, 135M 20.20-21.34 -> **19.94 / 20.42 / 21.11**. Reading only the rows used (a thin `READ_RUN`, ~74% fewer bytes) made the read slower, 19.15 -> 21.01 us, and was reverted |
| [Decode fusers: sigmoid, SiLU and RMSNorm](#decode-fusers-sigmoid-silu-and-rmsnorm-2026-10-16-cards-0-and-1) | 2026-10-16 | decode runs on `Tt` (burn-fusion), with three chain fusers bit for bit `TtBackend`: Burn's sigmoid composition as the native sigmoid, `silu(gate) * up` as one SFPU pass, RMSNorm's statistic as one reduction kernel. A 1.7B decode step 45.23 / 47.71 / 50.78 -> **43.42 / 45.85 / 49.00 ms** (23.0-20.4 tokens/s), 135M 22.46-23.63 -> **20.20 / 20.66 / 21.34**. A norm that also divided in one kernel was 640 us a norm (71.5 ms a 1.7B step): one unit pushed `DIV`'s instructions for 64 tiles |
| [Block views: RoPE and SiLU without repacks or copies](#block-views-rope-and-silu-without-repacks-or-copies-2026-10-16-cards-0-and-1) | 2026-10-16 | element-wise ops read a block of a buffer in place, RoPE runs in the projection's layout (its table laid over the heads once a step, its halves swapped by tile copies): a 1.7B decode step 47.90 / 50.37 / 53.53 -> **45.23 / 47.71 / 50.78 ms** (22.1-19.7 tokens/s), 135M 22.91-23.98 -> **22.46 / 22.95 / 23.63**; RoPE's four repacks a layer gone, the key write now repacks (S1d); F0: the step under burn-fusion is +1.5 / +2.2 ms (its sigmoid is Burn's 8-op composition) |
| [Decode matmuls read only the activation rows they use](#decode-matmuls-read-only-the-activation-rows-they-use-2026-10-16-cards-0-and-1) | 2026-10-16 | P3 (fused `q\|k\|v` and `gate\|up`, `Llama::fused`): 1.7B **47.90 / 50.37 / 53.53 ms**, 135M **22.91-23.98**; `gate\|up` at ~490 GB/s. P0-P2: an `m = 1` product's `GATHER` reads each `A` tile through row `r` of face 1 (one entry, ~27% of the bytes), not the whole tile, and a one-tile-row product's last round is spread over the units: a 1.7B decode step 57.21 / 59.64 / 62.81 -> **49.78 / 52.27 / 55.44 ms** (20.1-18.0 tokens/s), its matmuls 33.8 -> 26.3 ms; two face-row reads a tile instead were *slower* on the 135M (the mover's cost is per entry); what is left is entry-bound on B |
| [A cache write of one row a head](#a-cache-write-of-one-row-a-head-2026-10-16-cards-0-and-1) | 2026-10-16 | `scatter_nd` with `Assign` is one row scatter at resident indices (NC's `SCATTER_ROWS`), and `FixedCache` writes a step with it: a 1.7B decode step 73.65 / 76.08 / 79.30 -> **57.21 / 59.64 / 62.81 ms** (17.5-15.9 tokens/s), the 135M step 25.81-26.99 -> 24.40-25.53 (burn-flex 26.66-31.12); the cache write is 5.0 of 56.8 ms a 1.7B step, was ~20 |
| [Where a SmolLM2-1.7B step goes: the profile mode](#where-a-smollm2-17b-step-goes-the-profile-mode-2026-10-15-cards-0-and-1) | 2026-10-15 | the first profile from the profile mode (`cargo xtask bench --profile`): a 1.7B decode step is 46% matmuls (~200 GB/s), **24% the cache write's `expand`** (32 key/value heads), ~30% small ops; stamping compiled in costs 0.46 ms a 135M step, so it is now a build feature |
| [SmolLM2-1.7B-Instruct on the card](#smollm2-17b-instruct-on-the-card-2026-10-15-card-1) | 2026-10-15 | the first model past 1B parameters, traced, F32, every tile: decode 13.6-12.6 tokens/s (burn-flex 3.4-3.0), prefill 267-771 tokens/s (burn-flex 7.6-24.7); decode reads its weights at 86-93 GB/s, 17-18% of GDDR's 512 |
| [Decode length and trace size](#decode-length-and-trace-size-2026-10-15-cards-0-and-1) | 2026-10-15 | one GDDR region a unit a trace (traces 3,754 -> 563 MB, build 44.9 -> 18.8 s); grouped-query attention without `repeat_kv` and steps attending to 64/128/256/512 positions: decode over a 512-position cache 32.29 -> 26.13-27.31 ms, a generation 0.60x-0.90x burn-flex's time |
| [Traced generation, end to end](#traced-generation-end-to-end-2026-10-15-cards-0-and-1) | 2026-10-15 | `TracedGenerator`, token ids in and logits out, the step and chunks of 64/128/256 over one cache: the four traces hold 81 KB of the 240 KB program cache together; a decode step costs its cache's length (512 positions 32.3 ms, 320 28.5, 64 26.5); over 320 positions a 256-token prompt plus 64 tokens takes 1,860 ms against burn-flex's 2,996, first token 61.7 against 1,020 |
| [A traced replay's fixed cost](#a-traced-replays-fixed-cost-2026-10-15-cards-0-and-1) | 2026-10-15 | the 37.2 ms traced SmolLM2 decode step broken down op by op (unit 0's mover stamps): ~84 ops a layer at ~13 us, not the barrier; five fixes (matmul padding off `K`, a kept zero row, a kept transpose, the model's mask and RoPE, thin downloads untilized by the card) bring it to 26.46 ms, burn-flex 26.42 |
| [SmolLM2 prefill chunk size](#smollm2-prefill-chunk-size-2026-10-15-cards-0-and-1) | 2026-10-15 | a traced prefill chunk (`Llama::step_chunk`) costs ~33 ms + 0.18 ms a token on every tile: 256 tokens in 79 ms (burn-flex 1,040), so a prompt wants one chunk that covers it (buckets), not small chunks; one tile captures only single tokens |
| [SmolLM2 decode step traced](#smollm2-decode-step-traced-2026-10-15-cards-0-and-1) | 2026-10-15 | a fixed-shape decode step (`Llama::step_fixed`) captured once and replayed at every position, the cache written back on the card: 37.1 ms a token on every tile (burn-flex 26.6; eager 86.9 at best) |
| [ViT-Tiny traced](#vit-tiny-traced-2026-10-15-card-1) | 2026-10-15 | a replayed ViT-Tiny forward: 12.6 ms on every tile with the default spread (burn-flex 107.6, 8.5x faster), 61.9 at `work:64:4`; one tile does not capture; a decode step cannot be traced yet (its shapes grow a position a step) |
| [Attention scores kept in tile rows, and a kept causal mask](#attention-scores-kept-in-tile-rows-and-a-kept-causal-mask-2026-10-15-cards-0-and-1) | 2026-10-15 | ViT's scores stay a head per whole tile rows from `Q K^T` to `P V` and the causal mask is kept per attachment: ViT-Tiny 173.5 -> 129.1 ms on every tile (burn-flex 109.3), 481.9 -> 368.3 on one; a prefill call 13 -> 11 jobs |
| [Attention products in one job](#attention-products-in-one-job-2026-10-14-cards-0-and-1) | 2026-10-14 | a batched matmul takes a ragged `m` and heads stacked by rows (one row gather, not a repack a head): attention 64 -> 10 server jobs a decode call, 28 -> 10 a ViT call; ViT-Tiny 283.4 -> 173.5 ms on every tile (burn-flex 107.2), the SmolLM2 decode step 99.6 -> 86.5 |
| [Slices and swaps as views](#slices-and-swaps-as-views-2026-10-14-card-0) | 2026-10-14 | a positive-step slice is a view, a swap of a view stays one, a ragged view is a tile copy: ViT-Tiny 401.4 -> 283.4 ms on every tile (burn-flex 106.7), the SmolLM2 decode step 108.5 -> 99.6 |
| [Cache writes as a tile copy and written rows](#cache-writes-as-a-tile-copy-and-written-rows-2026-10-14-card-0) | 2026-10-14 | `slice_assign` of whole rows copies the destination by tile runs and writes only the value's rows: a SmolLM2 cache write 254 -> 25 us of server time, the decode step 117.3 -> 108.5 ms on every tile |
| [Native `cat`](#native-cat-2026-10-14-card-0) | 2026-10-14 | `cat` and `repeat_dim` are one row gather or one element repack, not a zero-fill and a `slice_assign` a piece: the SmolLM2 decode step 232.6 -> 117.3 ms on every tile (burn-flex 26.3), 623.9 -> 458.1 on one; its 121 `cat`s 196.6 -> 4.6 ms of server time |
| [Decode step: where the host time goes](#decode-step-where-the-host-time-goes-2026-10-14-card-0) | 2026-10-14 | jobs and server-thread time per Burn op: on one tile the SmolLM2 decode (624 ms) waits on m=1 matmuls; on every tile (233 ms) `cat`'s zero-fill and `slice_assign` are 61% of the server thread; ViT-Tiny is attention and repacking `swap_dims`/`slice`, not per-op submission |
| [Chunk-free convolution and max pooling](#chunk-free-convolution-and-max-pooling-2026-10-14-cards-0-and-1) | 2026-10-14 | one tap select and one product a convolution, NHWC views between layers: ResNet-18 16.3 s -> 782 ms on one tile (5.0x burn-flex), 131 ms on every tile at `work:64:4` (burn-flex 151) |
| [tt-metal reference: DRAM reads by core count](#tt-metal-reference-dram-reads-by-core-count-2026-10-13-card-1) | 2026-10-13 | tt-metal's compiled reader takes ~125 cycles a 4 KiB page, ours 359 safe / 175 fast; at scale ours reaches 94-97% of the card, the plain interleaved tt-metal pattern 71-81% |
| [Host-encoded read entries](#host-encoded-read-entries-2026-10-13-card-0) | 2026-10-13 | a build-time mode of the B mover: a read entry 359 -> 175 device cycles (215 with the semantic guard); real flows 1.0x-1.95x; default stays safe |
| [Read-loop probe](#read-loop-probe-what-the-interpreter-costs-per-read-entry-2026-10-13-card-0) | 2026-10-13 | a shipped mover read entry is 359 device cycles; a bare loop 90-106; the host-timed 0.48 us includes ~0.21 us of list upload; the in-flight cap does not matter past 4 |
| [Open-all made strictly better](#open-all-made-strictly-better-2026-10-14-cards-0-and-1) | 2026-10-14 | no per-op loop over every open tile, and a tile's catch-up rides in its first list for the op: 120 tiles at a fixed spread match the best small session (transformer 77.9 against 79.8 on 2 tiles, MNIST 2.2 = 8 tiles), and a per-op spread now pays on mixed sizes (a 2048^2 add 725 us against 1,054 at a fixed 4, small ops 19.4 us both) |
| [Per-op tile spread](#per-op-tile-spread-2026-10-14-card-1) | 2026-10-14 | `Session::set_spread` / `TT_SPREAD`: a session can open every tile and give each op a few; at 120 tiles a fixed spread of 4 gives MNIST 2.2-2.3 ms/step (6.7 on every tile, 2.1 on an 8-tile session) and the transformer ~80 (96.8; 74-76 on 2 tiles); spreads that vary per op lose to churn; the default stays every tile |
| [Barriers on the units an op ran on](#barriers-on-the-units-an-op-ran-on-2026-10-14-card-0) | 2026-10-14 | a tile an op does not use no longer enqueues a barrier list for it: the all-tiles transformer 404-422 -> 96-98 ms/step (now faster than one tile), all-tiles MNIST 12.9-13.4 -> 6.7, one tile unchanged, losses and accuracy identical |
| [Descriptor runs in a trace's prologue](#descriptor-runs-in-a-traces-prologue-2026-10-14-card-0) | 2026-10-14 | `record::POKE_RUN`: a trace's 51 `POKE`s become 9 records, 8,824 -> 4,892 device cycles a replay; a replay is 2.3-4.6 us shorter host-timed, its stream 124 -> 91 entries, and the 8-tile chunk boundary is gone |
| [Trace replay breakdown](#trace-replay-where-a-replayed-ops-device-time-goes-2026-10-14-card-0) | 2026-10-14 | a replayed tiny streaming op costs ~7.4-8.7 us, about half of it control (pair begin ~1.05 us, finish ~1.3-1.6, credits ~0.9, barrier ~0.6-0.7 on many tiles); the 51-`POKE` prologue is 7.2 us a replay; a chunk boundary is 0.75 us; the host adds ~2 us a tile |
| [Trace footprint](#trace-footprint-what-a-trace-costs-a-tile-2026-10-14-ttsim) | 2026-10-14 | a captured op costs ~18 stream entries (~0.6 KiB) but ~8 KiB of GDDR (whole-slot sections) and, per distinct op, ~4 KiB of pinned program cache (~0.6 KiB a program); programs are shared by hash; chunk padding is 0-1% |
| [Records and NC writes](#per-tile-cost-of-the-movers-records-and-ncs-writes-2026-10-13-cards-0-and-1) | 2026-10-13 | A record's tile costs a plain entry's ~340 cycles on B (read) and ~320-345 on NC (write); only plain entries gain from encoding |
| [Firmware image budget](#firmware-image-budget-and-core-assignment-2026-10-13-card-0) | 2026-10-13 | B's 24 KiB slot repacked to 42 KiB at no speed cost; `opt-level` 2 is as fast as 3 and 1.7 KB smaller, `s`/`z` are 1.6-7x slower |
| [Row select: an embedding lookup without staging](#row-select-an-embedding-lookup-without-staging-2026-10-13-card-0) | 2026-10-13 | a SmolLM2 token's lookup 15.9 s -> 260 us; the F32 SmolLM2 gate 575 s -> 33 s; a decode step is the host-bound 623 ms again |
| [Model ladder: real models on the card](#model-ladder-real-models-on-the-card-2026-10-12-cards-0-and-1) | 2026-10-12 | ResNet-18 105x, ViT-Tiny 7.8x and a SmolLM2 decode step 25x slower than burn-flex; a token embedding alone 15.9 s; 32-row chunking with host-built tables in convolution, pooling and gathers is the cost |
| [Accurate matmul cost](#accurate-matmul-cost-2026-10-12-card-0) | 2026-10-12 | opt-in F32-accuracy matmul on the Matrix Unit: 19x to 64x slower than the fast one; the element-wise work around the matmuls dominates |
| [Exact matmul cost](#exact-matmul-cost-2026-10-11-card-0) | 2026-10-11 | opt-in F32 matmul 1,000x to 22,000x slower than the fast one; linear in products |
| [`.ttinsn` push fusion, precise `gelu`](#ttinsn-push-fusion-precise-gelu-2026-10-10-card-0) | 2026-10-10 | experiment: 2.77x on a push-bound kernel; not adopted |
| [X6 mover NIU fast path](#x6-mover-niu-fast-path-2026-10-10-card-0) | 2026-10-10 | |
| [Approx transcendentals, per-tile saving](#approx-transcendentals-per-tile-saving-2026-10-10-card-0) | 2026-10-10 | |
| [P2 K-blocking and R3 norm release baselines](#p2-k-blocking-and-r3-norm-release-baselines-2026-10-10) | 2026-10-10 | |
| [ADC Z/W plane-copy tranche](#adc-zw-plane-copy-tranche-2026-10-06) | 2026-10-06 | contains the 2026-10-07 explicit L1 movement APIs |
| [ADC rectangle-copy instruction tranche](#adc-rectangle-copy-instruction-tranche-2026-10-06) | 2026-10-06 | |
| [Resident matrix chain benchmark](#resident-matrix-chain-benchmark-2026-10-06) | 2026-10-06 | |
| [Native operation baselines](#native-operation-baselines-2026-10-05) | 2026-10-05 | |
| [BF16 and mesh operation baselines](#bf16-and-mesh-operation-baselines-2026-10-05) | 2026-10-05 | |
| [Packed BF16 update](#packed-bf16-update-2026-10-05) | 2026-10-05 | |
| [Streaming ownership rollout](#streaming-ownership-rollout) | 2026-10-03 to 2026-10-04 | undated heading; overlap thresholds 2026-10-04 |
| [GDDR6 through the data mover](#gddr6-through-the-data-mover) | 2026-10-02 to 2026-10-03 | undated heading; pre-streaming baseline |
| [Core path: B → T0/T1/T2](#core-path-b--t0t1t2) | pre-streaming | undated heading |
| [Ethernet and 800G](#ethernet-and-800g) | pre-streaming | undated heading |
| [Reader / writer specialisation](#reader--writer-specialisation-2026-10-03) | 2026-10-03 | refers back to the GDDR6 section above it |

## Change log

Newest first within this table. Run = the `target/silicon/bench/<stamp>` it came from.
The dated `###` subsections after this table are in the order they were appended (oldest
first) and are kept in that order because they cross-reference each other ("continuation
below"); the dated top-level sections of this file are ordered newest-first (see the index at
the top). Use each section's own date, and this table for the chronology.

| Date | Run | Change | Scoreboard effect |
|---|---|---|---|
| 2026-10-17 | 1792258449 / 1792258565 / 1792258604 | Roofline check for SmolLM2-1.7B (F32, one card): the generation re-measured (prefill 315 / 741 / 861 tokens/s, decode 41.80 / 44.24 / 47.44 ms) and a new traced product bench, `matmul_throughput` (2.0-4.2 TFLOPS at prefill shapes, GDDR-bound by operand re-reads). | New Roofline section; 1.7B prefill row and a matmul-rate row updated. |
| 2026-10-16 | 1792170667 / 1792171171 / 1792172763 / 1792174412 / 1792174499 / 1792172342 / 1792174906 | Block views (`decode-small-ops.md` S1, S2a; gate `step192`): element-wise ops on a block of a buffer read in place (`Session::eltwise_blocks`), RoPE in the projection's layout (`Strided::layout_moves`: the table laid over the heads once a step, the halves swapped by tile copies), a regrouped GQA `q` made plain once for attention. SmolLM2-1.7B decode 47.90 / 50.37 / 53.53 -> **45.23 / 47.71 / 50.78 ms**; 135M 22.91-23.98 -> **22.46 / 22.95 / 23.63** (burn-flex 29.5-31.0); card 1, every tile. F0: the same steps on `Fusion<TtBackend>` 47.49 / 49.90 / 53.08 and 24.00 / 24.52 / 25.15. | Decode rows updated. |
| 2026-10-16 | 1792161551 / 1792161623 / 1792162031 / 1792162257 | Fused projections (`decode-matmul-bandwidth.md` P3, `Llama::fused`, gate `step191`): a layer's 7 products become 4, bit for bit. SmolLM2-1.7B decode 49.78 / 52.27 / 55.44 -> **47.90 / 50.37 / 53.53 ms**; 135M 24.14-25.25 -> **22.91 / 23.35 / 23.98** (burn-flex 26.43-30.65); card 1, every tile. | Decode rows updated. |
| 2026-10-16 | 1792114379 / 1792114782 / 1792115180 / 1792115571 / 1792115848 / 1792118484 / 1792118871 / 1792119119 | Decode matmul bandwidth (`decode-matmul-bandwidth.md` P1, P2): thin `A` reads (`record::gather_rows`), first as two face-row reads a tile (1.7B 55.69-61.30 ms, 135M *slower*, 25.43-26.51), then as one prefix read (1.7B 51.15-56.71, 135M 24.07-25.21), then the last round of a one-tile-row product spread over the units (`tensor::thin_runs`). SmolLM2-1.7B decode 57.21 / 59.64 / 62.81 -> **49.78 / 52.27 / 55.44 ms**; 135M 24.40-25.53 -> 24.14-25.25 (burn-flex 26.05-30.44); card 1, every tile. | 1.7B decode row updated. |
| 2026-10-15 | 1792034179 / 1792034180 / 1792034647 / 1792035075 / 1792035466 / 1792035579 / 1792036613 / 1792036724 / 1792036850 / 1792036818 | A traced replay's fixed cost (`replay-fixed-cost.md`): P0 profile (2,418 ops, replay 36.0 ms; per op ~13 us; the barrier 0.6 us of it), then matmul padding off `K`, a kept zero row, a kept transpose for `B`, the model's mask expanded once a step and RoPE's sign in its table, thin downloads untilized by the card. SmolLM2 decode traced 37.2 -> **26.46 ms** (burn-flex 26.42), prefill 256 tokens 79.0 -> 73.1 ms, ViT-Tiny traced 12.6 -> 11.6, ResNet-18 traced 17.6 -> 17.2; card 1, every tile. | Rows updated. |
| 2026-10-15 | 1792031467 / 1792032275 / 1792032323 | SmolLM2 prefill chunk sweep (`smollm2_prefill_chunk_sweep`, traced `Llama::step_chunk`, 512-position cache): every tile 1 / 32 / 64 / 128 / 256 tokens a replay in 43.8 / 40.2 / 44.3 / 55.2 / 79.0 ms (burn-flex prefill of 32 / 128 / 256 tokens 175 / 514 / 1,040 ms), card 1; one tile does not capture 32 and up; `work:64:4` 228-418 ms. | New scoreboard row. |
| 2026-10-15 | 1792029144 / 1792029158 / 1792029169 | SmolLM2 decode step traced (`smollm2_decode_traced`, fixed shapes, cache as written-back state): 37.1 ms a replay on every tile with the default spread, 83.5 at `work:64:4`, 478.3 on one tile (burn-flex 25.5-26.6), card 1, bit for bit the eager fixed step. | New scoreboard row. |
| 2026-10-15 | 1792026627 / 1792026634 / 1792026639 | ViT-Tiny forward traced (`vit_tiny_forward_traced`): 12.6 ms a replay on every tile with the default spread, 61.9 at `work:64:4` (burn-flex 107.6), card 1, bit for bit the eager forward; one tile does not capture (program cache). | New scoreboard row. |
| 2026-10-15 | 1792026124 / 1792026138 | Attention's scores kept a head per whole tile rows (`padded_attention`) and the causal mask kept per attachment: ViT-Tiny 173.5 -> 129.1 ms on every tile at `work:64:4` (burn-flex 109.3), 481.9 -> 368.3 on one tile, card 1; the decode step unchanged (86.9; its single query keeps the stacked path); see "Attention scores kept in tile rows". | ViT rows updated in place. |
| 2026-10-14 | 1792019870 / 1792019882 / 1792019917 | Attention products in one job (ragged `m`, heads stacked by rows gathered once): ViT-Tiny 283.4 -> 173.5 ms and the SmolLM2 decode step 99.6 -> 86.5 ms on every tile at `work:64:4` (burn-flex 107.2 and 26.5), card 1; one tile ViT 481.9, decode 449.7; see "Attention products in one job". | ViT and decode rows updated in place. |
| 2026-10-14 | 1792000003 | ResNet-18 forward traced, every tile: 17.6 ms a replay against 205.7 untraced and burn-flex 151.7, card 1, bit for bit the eager forward; one tile does not capture (program cache); see "Chunk-free convolution and max pooling" below. | New scoreboard row. |
| 2026-10-14 | 1791996580 / 1791996716 / 1791996730 | Chunk-free convolution and max pooling (NHWC tap select, one product, views between layers): ResNet-18 forward 16,288 -> 782.3 ms on one tile, 131.1 ms on every tile at `work:64:4` (burn-flex 151-157 ms), card 1, top-1 validated; see "Chunk-free convolution and max pooling" below. | ResNet-18 rows updated in place. |
| 2026-10-13 | 1791851940 / 1791852097 | Row select (`Session::select_rows`, B's `SELECT_ROWS`): the embedding lookup of SmolLM2's 49,152 x 576 table, 15.9 s -> 260 us a token, validated against the host's rows; a decode step from a token id 623 ms; see "Row select" below. | Scoreboard rows updated in place. |
| 2026-10-12 | 1791846735 / 1791846823 | Model ladder: ResNet-18, ViT-Tiny and SmolLM2-135M forward passes against burn-flex on the same weights, cards 0 and 1 (one job per card), validated; see "Model ladder" below. | New scoreboard rows. |
| 2026-10-12 | 1791818553 | Accurate matmul (`MatmulMode::Accurate`) against the fast Matrix Unit matmul, seven shapes, card 0, validated; see "Accurate matmul cost" below. | New scoreboard row. |
| 2026-10-11 | 1791761944 | Exact matmul (`MatmulMode::Exact`) against the fast Matrix Unit matmul, four shapes, card 0, validated; see "Exact matmul cost" below. | New scoreboard row. |
| 2026-10-10 | 1791597495 | Release card-0 smoke tier after the Phase 10 close-out: 486/486 pass (baseline before the close-out: 258/258, run `1791558187`). | None; correctness record. |
| 2026-10-10 | 1791600377 | X6 mover NIU fast path: request initiator 1 keeps its registers, fast/slow 0.96-0.97 on every shape (about 17 ns of 0.48 us per request); opt-in, see "X6 mover NIU fast path" below. | None; not the default. |
| 2026-10-10 | (`silicon_perf::approx_per_tile_saving`) | `MathMode::Approx` saves 1.24x-8.24x per tile on six transcendentals; see "Approx transcendentals" below. | None; Precise stays the default. |
| 2026-10-10 | 1791593690 / 92 / 99 | P2 K-blocking and R3 norm release baselines, both cards, validated; see "P2 K-blocking and R3 norm release baselines" below. | None; baselines recorded. |
| 2026-10-06 | 1791256590 | Direct matrix RHS tile/face/RWC addressing replaces expanded resident RHS materialization. Same M1 release conditions, both cards, two tiles, validated warmups and seven host-timed samples. | Ragged row multiplication: F32 ~1.13 ms → 42 µs (27×), BF16 ~1.16 ms → 61 µs (19×); zero staging transfer packets. Equal-shape paths remain comparable. No MNIST timing claim. |
| 2026-10-06 | 1791255969 | M1 native matrix ELW add/subtract/multiply; final explicit Dst setup, packed BF16 inputs/F32 accumulation, all RHS broadcasts. Both cards, two tiles, two warmups/seven host-timed samples, validation outside timing. | Equal-shape packed BF16 avoids widening and improves the measured adapter workload; F32 is slower and broadcast materialization dominates. 192 validated comparisons; no universal speedup. Conditions/results below. |
| 2026-10-05 | 1791160155 (gates) | Final BF16/pooling gates include integer/Boolean casts, analytic norm derivatives, mixed F32-loss SGD, BF16 trace temporary holds, matmul-to-pooling ADC reset, partial ceil windows and exact max-pool zero/NaN value/index selection. | Both cards pass 24/24. Max pooling uses SFPU argmax plus a raw-bit OR fold; the arithmetic gather negative control changed signed zero/NaN payloads and failed. This adds correctness coverage, not a speedup. |
| 2026-10-05 | 1791158550 | Packed BF16 versus resident F32→TF32 operands for MNIST's 64×784×128 and 64×128×10 forward GEMMs. Both p150a cards, one gate tile, HiFi4, pipeline off, role profiling off, 1350 MHz AICLK, 16000 MT/s GDDR, eight channels, SHA `0d6ba5227fdb+dirty`; one warm-up, nine completed host-timed samples, output validated every run outside timing. | Card 0: TF32 93.964/18.135 us, BF16 661.742/80.840 us. Card 1: TF32 93.102/18.294 us, BF16 663.076/82.012 us. Current per-tile BF16 gather/mask path is slower; storage savings do not establish an MNIST speedup. No full MNIST performance run. Results: `target/silicon/bench/1791158550.{jsonl,md}`. |
| 2026-10-05 | 1791158962 (gates) | Native BF16 storage/layout/casts, packed MMA, arithmetic adapters, normalization derivatives, F32-loss/BF16 SGD, trace temporary holds and NCHW pooling/backwards. GMPOOL/GAPOOL encodings measure required bit 19 while preserving AddrMod at 15. | Both-card correctness: 22/22. BF16 narrowing refuses pinned ttsim mode `0x105`; those arithmetic gates are silicon-only. Pool window staging remains a coverage path, not a performance optimization. |
| 2026-10-04 | unrun | Reduction/scan and ALU extensions; `silicon_bench_path::tensix_extensions` stages one warm-up and nine completed host-timed runs with per-run output validation | No measured claim: no cards available. Run `cargo xtask bench --device all --filter tensix_extensions`; record run IDs before changing the scoreboard. |
| 2026-10-04 | 1791079782 | Overlap decided per `tensor::Overlap` (off, fresh, captured) with shares measured on NC ownership; `HostStage::Check` splits list checks from the ring write; a trace's program holds carry the cache generation (a release after recovery no longer unholds a newer trace's program) | Fresh 8-tile 2048² add 1.07 -> 1.00 of serial; traced 8/32-tile adds 1.00 -> 0.77-0.95 from 512-1024²; one-tile small reductions 1.00 -> 0.91; enqueue unchanged. Silicon 306/306 (run 1791079893). |
| 2026-10-03 | — | B a pure reader, NC a pure writer. `SIGNAL` / `WAIT_PEER`, `dm::Peer` and B's NoC #1 writes removed; standalone transfers are kernel-less reader/writer packets (`Step::Transfer`, `dataflow::Stream::Transfer`); `Mover::permits` is the one direction table; `FILL_PAD` split with a new `PAD_WRITE`; each image compiles in only its direction. | B 24,240 -> 20,856 B (3.7 KB free from 0.3), NC ~22.9 -> 14.8 KB. Compute and the models unchanged (MNIST 1.3 ms/step, transformer 8.72); large host uploads -31%; small uploads on many tiles +5-8%, a one-tile 64×10 upload +2 µs. Silicon 296/296. |
| 2026-10-03 | 1791063177 / 1791063683 | Consolidate GDDR compute under fixed streaming ownership; remove legacy scheduler selection and reject nonstreamable compute instead of falling back. Both firmware images required by `enable_dram`. | Full initial hardware audit 295/295, final strict ownership/cache/profile/benchmark checks 22/22, full host/simulator suite 710 passed. Mixed stress 60 s each on one/eight tiles, 39971/72561 rounds. Traced identity remains 357 / 329 GB/s combined at 32 / 120 tiles. Known fresh-dispatch regressions accepted; no universal speedup claimed. |
| 2026-10-03 | 1791060908 / 1791061578 / 1791061613 | Remeasure raw GDDR/Ethernet and add a true streaming identity payload benchmark, repeated once. No Ethernet firmware change. | Raw 120-tile read / NoC #1 write / mixed: 469 / 389 / 403 GB/s (device). NC NoC #1 write: 74.94 GB/s on one tile. Streaming traced identity: 355 GB/s combined at 32 tiles, 328 at 120, versus legacy 216 / 244. Fresh still regresses. Two-link Ethernet one-way: 27.03 GB/s host end-to-end. |
| 2026-10-03 | — | Asynchronous dispatch (B8): burn-tt's device calls return at once -- the caller names each result (a process-wide id, never reused) and computes its shape; the server thread translates ids and runs jobs in order; only downloads, traces and queries wait. A failed op poisons its result and is reported, with the op's name, at the attachment's next wait. | Per call, caller's side: 30-40 us -> 0.2 us. **MNIST** 1.8 -> 1.4 ms/step (burn-flex 0.5). **Transformer** 11.1 -> 8.6 ms/step (burn-flex 4.3): 2.0x. What is left is the server's own work per op, now all seen as the step's one wait (MNIST: 1.1 ms of 1.4; the transformer: nearly all of its step) -- building each op's lists and writing them over PCIe; traces and smaller lists are its levers. |
| 2026-10-03 | — | `APPROX_MIN_TILES` and `SOFTMAX_DEVICE_MIN_TILES` removed: approximate ops follow their data at every size (exact mode alone keeps them on the host). Decided, not measured into: data is not to come back to the host to save a call; MNIST is to be made fast otherwise. | **MNIST** 1.1 -> 1.8 ms/step (burn-flex 0.5): 43 device calls a step at ~33 us each are ~1.5 ms of it (B8). **Transformer** 12.0 -> 11.1 ms/step (burn-flex 4.4); 0.16 MB downloaded over 50 steps -- the loss's last `mean` -- from 11.65. |
| 2026-10-03 | — | Indexing on the card (D4): the loss's gather and its backward on the SFPU; `select`/`select_add`, so `nn::Embedding` and its gradient, on the mover (`gather_rows`, `write_rows`, `rows_add`). `tt-mnist --model transformer` prints where the time went per step. A latent race closed in `copy`/`copy_blocks` (a `WAIT` between reads and writes). | **Transformer** 11.9-12.0 ms/step (burn-flex 4.4-4.6), from 8.4-10.3: the embedding on the card costs ~0.9 ms a step against the host's free lookup -- its lists, two 32-byte entries a row a tile column, written over uncached PCIe (a record carrying the indices would cut that ~18x). Per step: 125 element-wise calls 4.5 ms at 36 us each, 16 uploads 1.7 ms, 14 downloads 1.0 ms, 27 matmuls 1.0 ms, 25 reductions 0.8 ms, outside calls 2.8 ms -- ~7 ms of 12 is the per-call round trip (B8). **MNIST** 1.1 ms/step, unchanged. `APPROX_MIN_TILES` at 0 measured again: transformer 10.0-11.0 with almost nothing downloaded, MNIST 1.7-1.8; kept at 8. |
| 2026-10-03 | — | Batched matmuls and strided views (B6, P1b): `Session::matmul_dram_batched` gathers each batch element's operand blocks where they lie (a `TensorRef` moved to the block's first tile); `Session::copy_blocks` moves whole tiles into any arrangement, transposing them with a new `READ_RUN` flag (bit 3, through the silicon-proven `READ_TRANSPOSED`); burn-tt keeps reshapes and swaps as strided views of one buffer (`views.rs`), so attention's heads, `K^T` and their gradients move nothing. Rank-N sums and maxima along the last dimension, sums over leading dimensions in Flex's order, `mean_dim` composed on the card. Full silicon suite 550/550 on both cards with the mover change. | **Transformer** (50 steps, 1 tile): 21.5 -> 8.4-10.3 ms/step (bimodal across runs; burn-flex 4.5-4.7): 2.2x Flex, from 5.5x. Every matmul on the card, none host-staged; 0.63 MB a step over PCIe, from 1.2. **MNIST** unchanged: 1.0-1.1 ms/step, burn-flex 0.5, golden bit for bit.<br>`APPROX_MIN_TILES` re-measured with batched submission: without it the transformer gains (9.0) and MNIST loses (1.0 -> 1.6: its small log-softmax still ends in a host loss), so it stays at 8 until a lookahead (B16) or the loss on the card (D4). |
| 2026-10-03 | — | burn-flex baselines, and a general model beside MNIST. `tt-mnist --model transformer` trains `tt_mnist::transformer` (Burn's `Embedding`, one pre-norm `TransformerEncoder` layer, d 64, d_ff 128, 2 heads, a `Linear` head; batch 4 x seq 32) on the card and on burn-flex from the same weights, and prints burn-tt's per-op report (`burn_tt::report`). Rank-N Linears fold their batch into the rows (`float_matmul`, `linear_{weight,bias}_backward`), so a `[b, s, d]` Linear is one rank-2 device matmul. | **MNIST** (937 steps, 1 tile): burn-tt 1.0 ms/step, burn-flex 0.5 (re-measured; the 9.0 number holds); 91.96% vs 91.97%. Per step, caller's side: 21 server calls, ~28 us each for the queued ones, and the step's one download 240 us (it waits for the queue).<br>**Transformer** (50 steps, 1 tile): burn-tt 21.5 ms/step, burn-flex 3.9 (5.5x); 1.2 MB a step over PCIe. 700 of 1000 matmuls on the card; the other 300 are attention's batched products, host-staged (28 MiB). Host ops between device ops download 26 MiB: the heads' reshapes (13.3), rank-N `sum_dim` (9.4, softmax's and layer norm's), `mean_dim` (Flex; layer norm), `embedding_backward`. Before the fold, nothing ran on the card (every operand stayed a host tensor; matmuls staged 11 MiB per 3 steps on ttsim). |
| 2026-10-03 | — | Tilizing on the card, measured and kept as an option (`Session::set_tilize`, burn-tt `TT_TILIZE=card`): mover ops `TILIZE` / `UNTILIZE` turn a row-major block in L1 into a tile slot's datums and back (16-word face rows unrolled: a `memcpy` call cost 7.4 -> 2.6 µs a tile); rows cross PCIe and each unit tilizes its chunks (a tile row across up to 96 tile columns). **The default stays the host's tilize.** | **Upload to its end, card / host tilize:** 64×784 on 1, 8, 32 tiles 168 / 45, 137 / 56, 373 / 207 µs; 1024² 3.5 / 0.76, 1.02 / 0.48, 0.98 / 0.55 ms; 8192×1024 26.6 / 5.3, 5.6 / 2.7, 4.7 / 2.9 ms; downloads alike. The host copies the rows into pinned memory either way, at about what tilizing them cost, so the card saves the host nothing and adds ~2.6 µs a tile of mover time. Where it would pay: a tilize on the Tensix unpacker (tt-metal's), or rows the card reads from the caller's memory without a copy.<br>**Gates:** `step58_tile_layout` (the ops against the host tilizer, ragged; every host / card combination round-trips, views and a tensor wider than a chunk). |
| 2026-10-03 | — | Host transfers, the host's side.<br>• Each unit's share of a transfer is a contiguous run of tiles, a batch of 64 being one host move of its consecutive slots and one `READ_RUN` / `WRITE_RUN` record, not two entries a tile.<br>• The host tilize and detilize run on up to 16 threads, 256 tiles each.<br>• A large download's output is backed by 2 MiB pages (`MADV_HUGEPAGE`): every 4 KiB first touch faulted.<br>• Uploads and writes are queued behind the ops, with a barrier, as an op is, through a staging ring that syncs only when it wraps onto a transfer still queued; a write through the BAR still syncs first.<br>• The session's in-flight-cap warning is gone: the tile cap is by design, and every large move waits under it (`Session::throttle` still counts).<br>• Every upload takes the DMA, however small (it was 4 tiles and up), and the host's core count is asked once: `available_parallelism` reads cgroup files on every call, ~40 µs. | **`Session` transfers, 8 tiles:** 8192×1024 upload 4.0 -> 11.2 GB/s (the card's part ~20 GB/s), download 1.4 -> 11.0; 1024² 3.7 / 2.8 -> 6.6 / 5.6; 64×784 upload 102 -> 19-33 µs (queued: the host's part), 64×10 download 48 -> 9 µs.<br>**Streamed steps** (`streamed_steps`, upload a batch then multiply, unsynced): 64×784×128 on 8 tiles 63 µs/step queued, 88 with a sync after each upload, 1544 through the BAR (1 tile 101 / 111 / 1663); 512×1024×1024 1.12 / 1.15 / 15.5 ms (1 tile 6.98 / 7.10 / 23.1).<br>**MNIST:** traced inference batch 8 tiles 0.237 -> 0.205 ms (312k images/s), 1 tile 0.305 -> 0.267; training 1 tile 1.1 ms/step, 8 tiles 1.3. |
| 2026-10-03 | — | Tensors move between host and card by the card's own DMA (`dm::op::HOST_READ` / `HOST_WRITE`): the session pins one 1 GiB hugepage (`PIN_PAGES` with `NOC_DMA`; the guest's IOMMU is identity, so contiguous memory is the only kind that pins), tilizes uploads straight into it, and each unit moves its share host -> L1 -> GDDR, or back, in double-buffered batches. The host's own copies through a BAR run uncached under this VM (`ttsim-divergence.md` row M). The host tilize and detilize are now one copy pattern each, checked byte for byte against `tt_layout`'s (which took ~18 µs a tile). On by default (`Session::set_host_dma`); with no pinnable memory, the BAR, said once. | **The card's DMA, raw** (`host_dma_bandwidth`): host -> card 19.4 GB/s on one tile, ~21 on 2-32; card -> host 24.4 / ~27 (Gen4 x16). BAR: 0.15 and 0.04.<br>**`Session` upload / download** (`session_transfers`): 64×784 2.8 / 2.2 GB/s (BAR 0.12 / 0.03); 1024² 3.2-4.2 / 2.3-3.1; 8192×1024 2.8-4.0 / 1.2-1.4 (the host's tilize, detilize and page faults, not the card).<br>**MNIST:** traced inference batch, 8 tiles, 2.6 -> 0.24 ms (input write 2.3 -> 0.09, logits 0.15 -> 0.009; 270k images/s); untraced 0.46 -> 0.33; 60k-image preload ~1.7 -> 0.53 s; training 1 tile 1.2 -> 1.1 ms/step.<br>**Gates:** `step57_host_dma` (ttsim, whose DMA callbacks now serve host regions, then card 0 alone); 116/116 ttsim binaries and 268/268 silicon tests with every transfer on this path. |
| 2026-10-03 | — | Measured: trace replay against queueing fresh (`trace_replay_vs_fresh`), a layer's forward pass (matmul, add a row, relu, matmul), 50 back to back. Captures run plain (pipelining is off while capturing). | **End to end, µs an op, MNIST size (64×784×128×10):** 1 tile fresh 26.1 (pipelined) / 28.7 (plain), replayed 30.3; 8 tiles 28.8 → 11.2; 32 tiles 51.4 → 11.5. Past one tile the replayed pass is the device's, ~45 µs, whatever the tile count.<br>**512×1024×1024×1024:** device-bound, replay within 1% on 1 and 8 tiles, 8% faster on 32 (176 → 161).<br>**A served batch** (input written, replayed, output read): the 200 KB input write 2.3-2.7 ms and the 2.5 KB read 0.15 ms against a 65-123 µs replay. Host↔card copies run uncached under this VM's passthrough (`ttsim-divergence.md` row M): 80 MB/s writes, 28 MB/s reads. That is what `tt-mnist --infer --trace` measures, not the replay. |
| 2026-10-03 | — | Host time per op split (`Session::host_times`, `host_time_per_op`, checklist 9.17), and three fixes it found:<br>• `DataMover::enqueue` read the queue's progress (two uncached PCIe reads) on every list; now only when the host's own count says the ring or slots are full.<br>• A multi-unit op's barrier rides at the end of each unit's last list instead of a list of its own.<br>• Matmul and reduce kernels carry a fresh empty loop table each, so the program memo (keyed by allocation) missed on every one; an empty table is now one key. | **Host µs an op, 50 queued** (before → after): add 64×784 on 8 tiles 64.6 → 28.9, on 32 260.7 → 117.3; matmul 256³ on 8 101.8 → 35.6, on 32 311.6 → 109.9. PCIe reads an op on 32 tiles 127 → 4. One tile unchanged: there the host waits for ring room, which is the device's time.<br>**What is left, a unit:** ~3.4 µs on 32 tiles: the list's three posted writes 1.3, reserve 0.6, segments 0.6, placement 0.3; building the op ~11 µs an op.<br>**MNIST, 300 steps:** 1 tile 1.4 → 1.3 ms/step, 8 tiles 2.1 → 1.6, 32 tiles 2.4; same loss and accuracy.<br>**Gates:** 267/267 silicon, ttsim workspace. |
| 2026-10-03 | — | Measured: one mover image for B and NC, or one specialised to each direction? (`mover_mixed_directions`). The same 128 reads and writes, one tile, as all reads, all writes, alternating, and in blocks of 8 (a gather record's worth, then a scatter's, as pipelined lists run). **Decision: keep the shared image.** | **Mixed against the pure lists' average, per entry:** blocks of 8 1.003-1.006x (~2 cycles of ~337), alternating one by one 1.05x; the same on B and NC, at 64 B, 1 KiB and 4 KiB. Reads 316 cycles an entry, writes 355.<br>The read and write paths barely evict each other in a ~4 KiB instruction cache, and with the split each core only runs one direction's code anyway, so a specialised image could save under 1%. It would cost a second image to keep in step, and B-only mode needs both directions in one. Image size frees nothing: NC's region is reserved either way. |
| 2026-10-03 | — | Scatters on NC, opt-in (`Session::set_scatter_mover`, burn-tt `TT_SCATTER=nc`): in a pipelined group B gathers and launches, and signals NC after each kernel's wait; NC writes that block out on NoC #1 and signals back. Block k's launch waits for NC to have written out block k-2 (same half); B's list ends waiting for NC's last, so B's done means the unit is idle and an NC failure fails B's wait.<br>With NC running, B's timestamper events come out garbled (an entry's end repeated 2 cycles apart): the NC sweeps time from the host. | **Against pipelined B-only, end to end** (`SWEEP_NC=1`): one tile, add 512² and up 0.70-0.80; exp, matmuls ~1.00; reductions 0.95-1.04; adds of 64 tiles or fewer 1.05. Eight tiles: matmuls 1.06-1.18 (a second list a unit to queue, checklist 9.17); 2048² add 0.99, exp 1.02. Off by default: it pays only for write-heavy, mover-bound ops on few tiles.<br>**Gates:** `step56_scatter_on_nc`, 267/267 silicon, `stress_pipeline` with `STRESS_NC=1` 120 s each on 1 and 8 tiles, bits equal. |
| 2026-10-03 | — | Element-wise ops and reductions pipelined too, on by default where `tensor::pipelined_runs`: runs in alternating halves of the arena, the same number on every unit, at least two a unit of 12+ tiles, and a unit's share at least `PIPELINE_SHARE` (48; reductions 192) tiles per unit -- the host queues units one at a time, so the more units, the more each must do before the extra runs pay.<br>• Blocks of resident programs group whatever their loop tables (an element-wise run's repeat is its length); before, runs of unequal length silently ran unoverlapped.<br>• The host encodes and hashes each role program once, not once per kernel enqueued (`stored_program`, `ProgramCache::place_hashed`). | **Device time, on/off** (`sfpu_pipeline_sweep`): one tile, add 0.73-0.84, exp 0.94-0.98, sums and maxima 0.57-0.81 (50 to 4096 tiles); two tiles, add 0.74-0.77, reductions 0.59-0.69 from 1024² up; 8 tiles, 2048² add 1.00, exp 0.96.<br>**Found on the way:** on 8 and 32 tiles these ops are bound by the host's per-unit queueing (a 50-tile max: 54 µs on 8 tiles, 155 on 32), checklist 9.17.<br>**MNIST:** unchanged (its tensors are under the share).<br>**Gates:** 266/266 silicon; `step55_pipeline_sfpu`; `stress_pipeline` (matmul, add, max) 120 s each on 1 and 8 tiles, bits equal. |
| 2026-10-03 | — | Pipelining on by default (`TT_PIPELINE=0` turns it off in burn-tt), where `tensor::pipelining_pays`: K whole in half the arena, at least two half-arena blocks a unit, and the half-arena plan gathering at most twice the plain plan's bytes.<br>• A session queues an op's lists round-robin across units: queuing one unit's whole share before the next blocked the host on that unit's ring (pipelined 1024³ on 8 tiles: 4.8× the plain path, 1.3× after; the rule excludes the rest, degenerate 1×2 half blocks).<br>• `stress_pipeline`: pipelined matmuls and adds queued back to back, every result against the plain path's bits. | **Sweep, on/off, device time** (`matmul_pipeline_sweep`): 1 tile 512³ 0.75, 1024×256×1024 0.75, 256³ 0.87, 64×784×128 0.87; 8 tiles 512³ 0.77, 1024×256×1024 0.89; 32 tiles, nothing pipelines (1.00 ± 0.02 once the arms' order is swapped; the second arm is ~10% slower on 256×1024×256 either way).<br>**MNIST train, 300 steps:** unchanged (1.4 ms/step one tile, 2.2 eight), same loss and accuracy: its matmuls are mostly too small to pipeline.<br>**Gates:** 264/264 silicon tests; `stress_pipeline` 120 s each on 1 and 8 tiles, ~34 000 overlapped blocks each, bits equal. |
| 2026-10-03 | — | Matmul blocks overlap their moves with compute (`Session::set_pipeline`, off by default).<br>• Blocks double-buffered in two halves of the arena.<br>• `LAUNCH` / `KERNEL_WAIT` entries.<br>• A unit's list runs `LAUNCH k, scatter k-1, gather k+1, KERNEL_WAIT k`.<br>A pipelined profile is the mover's alone (`set_profile_roles`): with mover and roles storing to the timestamper at once, card 0's streams lost events. | **matmul 512³, one tile:** 1416 → 1066 µs. Gathers grow (393 → 696 µs, half-size blocks) but hide under compute; the wait for the roles falls to 269 µs; the roles are busy 97% of the op.<br>**Single-block matmuls and element-wise:** unchanged.<br>**Bits:** identical to the plain path on ttsim and card 0 (`step54_pipeline`, one and three tiles, transposes). |
| 2026-10-03 | — | A record's plain moves go straight to the issue (`record_entry`, beside `issue` in `.text.hot`), not through `exec` and `Entry::decode` | **matmul 512³:** 1441 → 1416 µs (gather 414 → 393).<br>**128³:** 29.4 → 28.8.<br>**add, 64 tiles:** 72.1 → 69.7.<br>An out-of-line variant ran the matmul faster (1396) but added 14% to add's runs: the code's placement decides speed. |
| 2026-10-03 | — | Hardware integer divide allowed (the gate's T2 claim was a misreading of the vector `vdiv` caveat); `record::div_rem` removed | **matmul 32×256×32:** 11.9 → 8.5 µs.<br>**128³:** 33.5 → 29.4.<br>**512³:** 1537 → 1441 (gather 493 → 414) |
| 2026-10-03 | — | Records walk rows with a cursor (one division per row, not per tile); GDDR moves checked once per descriptor and issued from registers (`DramMove`, `issue_dram_on`) | **matmul 512³, one tile:** gather 1061 → 493 µs, scatter 162 → 91, op 2176 → 1537.<br>**matmul 128³:** 46.4 → 33.5.<br>**add, 64 tiles:** 139 → 73.5.<br>**exp, 64 tiles:** 579 → 535.<br>**Per entry:** 4 KiB read 354 → 314 cycles, write 397 → 355 |
| 2026-10-02 | — | Option C: NoC #1 owns port 0 on channels 4–7 (port 1 on 0–3), so its writes use eight rows instead of four. The host uses CMFW's endpoint | **Card writes on NoC #1, 120 tiles:** 268 → 378 GB/s.<br>**16 tiles:** 296 → 421.<br>**Reads + writes:** 344 → 396.<br>**Reads:** unchanged (469). |
| 2026-10-02 | — | NC mover (`dm_nc`), `SIGNAL` / `WAIT_PEER`, `copy_pipeline` | NC costs what B does per entry. Split copies run 1.6× at 4 KiB entries on 1–16 tiles and converge at 120 |
| 2026-10-02 | — | Tile movers default to an in-flight cap of 8 (`dm::TILE_IN_FLIGHT_CAP`); `write_noc::ALTERNATE` added | **Card reads:** 120 tiles 425 → 467 GB/s at 64 KiB, 496 at 16 KiB.<br>**One tile:** 3–4% off small reads.<br>**Writes:** unchanged.<br>**Alternating writes:** no gain at 120 tiles (263 against 268 on NoC #1).<br>**Stress (cap 8, all write modes):** 60 s clean. |
| 2026-10-02 | 1790971715 | NoC ownership.<br>• Each GDDR endpoint is one NoC's (NoC #1 port 1; NoC #0 ports 0 and 2), refused otherwise.<br>• Every request on static VC 1.<br>• Writes may go out on NoC #1 (`dm::WRITE_NOC`); reads stay on NoC #0.<br>• The host fences through NoC #0's ports only. | **Card reads:** 120 tiles 127 → 429 GB/s, no collapse past 4.<br>**Card writes:** 52 → 266 GB/s on NoC #1, 161 on NoC #0.<br>**Card reads + writes:** 343.<br>**One port's writes:** 28.5 → 62.9 GB/s.<br>**Per entry:** 4 KiB reads 328 → 339 cycles, `WAIT` 115 → 136 (NIU chosen per request; issue specialised per NIU).<br>**Stress:** 10 min, 120 tiles, both NoCs, 133 GiB, clean. |
| 2026-10-02 | 1790959226 | Experiment: every tile reading on NoC #1, and tiles split across NoCs | NoC #1 reads stuck at one link (4 tiles 86 GB/s, 120 tiles 58). Splitting tiles across NoCs on shared ports hung card 0 (SYS-1419). Replaced by the row above |
| 2026-10-02 | 1790956433 | No arithmetic on B (firmware benchmarks) | Per entry: 4 KiB reads 333 → 328 cycles (baseline 317), `WAIT` 120 → 115 (back to the baseline's 116), 16 KiB unchanged. Card totals, core path, matmul and SFPU device times, Ethernet: within 1%. New: sum over rows on the SFPU, 1 tile 4.7 µs, 64 tiles 87 µs on the device |
| 2026-10-02 | tt-mnist, card 0 | No arithmetic on B: element-wise and the sum over rows on the SFPU only (in-order sum, chunked past 16 row tiles); image gate refuses F instructions | MNIST 1 tile 2.0 → 1.7 ms/step; 8 tiles 1.9–2.0 → 2.2 ms/step (small ops spread thin pay the SFPU kernel's launch, which the old cost model avoided). Golden unchanged |
| 2026-10-02 | 1790953809 | Cap mover NoC requests in flight at 128 per transaction ID (fixes the 8-bit counter wrap); waits counted and reported | Per entry: 4 KiB 317 → 333 cycles, 16 KiB unchanged (357), `WAIT` 116 → 120. Card totals unchanged (reads 319 GB/s at 4 tiles, 129 at 120). The cap binds only under contention (120-tile reads: 2023 waits per run; one channel: 4235) without lowering throughput. 120 tiles × 300 requests on one channel completes exactly: 48 GB/s, 8762 waits per run |
| 2026-10-02 | 1790950685 | Baseline: device-timed benchmarks; trace events in B, T0–T2 and E1 | — |

### R1c / P2 benchmark backlog (2026-10-04)

General-axis repacking and K-blocked resident matmul are implemented and
simulator-gated (`step67`/`step68`). Repacking currently sends per-element
coordinate metadata and may reload a partially assembled tile across several
transfer batches; split-K matmul uses one output tile per job and serial
accumulator reloads. These are coverage paths with no performance claim.
Both-card correctness passed in run `1791145571`. *Superseded 2026-10-10:* the release
baselines were measured; see "P2 K-blocking and R3 norm release baselines" below. The text
that follows is the backlog as written on 2026-10-04.
Before recording a score, run release silicon on both cards, validate outputs,
warm each shape/route/fidelity, report medians with host timing and
`dataflow_stats`, and record run IDs. Compare new general reductions with the
existing matrix-axis paths and forced K blocks with unsplit products where
they fit; include `[64,8192] @ [8192,64]` and fresh/traced execution.


### Convolutional MNIST application observation (2026-10-06)

Run `1791252065`, isolated release runner, card 0, one discovered Tensix tile,
TF32 Src/F32 storage/HiFi4, native `tt_mnist::cnn::Cnn` (938 parameters): Conv2D
1→8, 5×5 stride 4, ReLU, average pool 2×2, Linear 72→10. One epoch of 59,968
resident images, batch 64, 937 SGD steps at lr 0.1; all 10,000 test images.
Native accuracy 78.60%, Flex 78.66%; first/final-100-step-mean losses
2.301632/0.595991 versus 2.301627/0.595428. Training wall time including model/data
preload and one scalar loss per step, excluding accuracy evaluation:
226.410 ms/step native versus 1.040 ms/step Flex. This is one full learning run,
with no warmups or repeated median; simulator regressions and builds ran
concurrently on the host. It establishes working classification and an initial
cost, not a controlled acceleration comparison. Current per-datum patch/overlap
metadata and many small dispatches make fresh CNN training expensive; no
performance improvement is claimed. Outputs and logs are under
`target/silicon/out/1791252065-*` and `target/silicon/cnn-full-epoch.log`.


### Kernel Fusion & Traced Execution Performance Comparison (2026-10-06)

Run `1791257827`, isolated release runner, Blackhole card 0 and card 1, single Tensix tile `(3, 4)`, batch size 64, MNIST MLP forward pass (`784 -> 128 (ReLU) -> 10`). Gate: `step12_mnist::compare_all_four_execution_modes_latency_and_traffic`.

Measured results across all four execution modes on physical hardware:

| Mode | Card 0 Wall (us/batch) | Card 0 Replay (us/batch) | Card 1 Wall (us/batch) | Card 1 Replay (us/batch) | Device PCIe Traffic |
|---|---:|---:|---:|---:|---|
| **1. Unfused Eager** (`TtBackend`) | 488.64 | — | 486.15 | — | 2 uploads, 2 downloads |
| **2. Fused Eager** (`burn_tt::Tt`) | 535.00 | — | 560.00 | — | 2 uploads, 2 downloads |
| **3. Unfused Traced** (`Trace` on `TtBackend`) | 216.06 | 156.94 | 224.94 | 157.25 | Direct buffer writes/reads |
| **4. Fused Traced** (`Trace` on `burn_tt::Tt`) | **212.44** | **149.83** | **208.49** | **149.83** | Direct buffer writes/reads |

**Key Findings:**
- **Combined Fusion + Traced is fastest**: Replaying a fused trace achieves the absolute lowest card execution time (**149.83 us/batch**, down from 156.94 us/batch unfused), saving ~14.5 us per batch in kernel dispatch and L1/DRAM round trips.
- **Trace eliminates host graph overhead**: While eager fusion incurs modest host client overhead for real-time IR pattern matching (535 us vs 488 us wall time), tracing captures the fused graph *once*. Trace replays run the fused compound hardware kernels (`ADD_RELU`) directly on Blackhole without any runtime IR graph overhead.
- **Identical numerical outputs**: All four execution modes produce equivalent logits within derived numerical error bounds. Outputs logged under `target/silicon/out/1791257827-*`.

### M1 matrix elementwise comparison (2026-10-06)

Initial M1 run `1791255969`, both p150a cards, two ARC-discovered compute tiles per
card, release SHA `6d979e7c5606+dirty`. Resident inputs; default streaming/pipeline
policy, no role timestamp profiling. F32 uses TF32 Src, packed BF16 uses BF16 Src,
HiFi4 multiplication. Two warmups and seven completed host-timed samples per
case; dispatch through synchronization is timed, output validation/readback and
freeing are outside timing. Every run validates its output. Inputs are exactly
representable constants 2 and 3, isolating execution cost from numerical error.
Clocks were not independently sampled in this run; this is a baseline under the
runner's existing card policy. Conditions are in the raw child outputs under
`target/silicon/out/1791255969-*`; medians, p10/p90 and `dataflow_stats` are in
`target/silicon/bench/1791255969.{jsonl,md}`. All 192 route/shape/operation/card
comparisons pass, including ADD/SUB/MUL, equal/row/column/scalar RHS, F32/BF16,
aligned 64×64 and ragged 65×70. Repeat with
`cargo xtask bench --device all --filter matrix_vs_sfpu_release_baseline`.

Representative medians (µs); each cell is SFPU / matrix:

| Operation/storage/geometry | Card 0 | Card 1 |
|---|---:|---:|
| Add F32, equal 64×64 | 22.181 / 24.626 | 21.891 / 24.826 |
| Mul F32, equal 64×64 | 21.990 / 24.156 | 21.810 / 24.755 |
| Add F32, equal 65×70 | 25.817 / 42.269 | 26.198 / 42.248 |
| Mul F32, equal 65×70 | 25.737 / 41.826 | 25.957 / 41.778 |
| Add BF16, equal 64×64 | 105.056 / 40.204 | 108.592 / 41.878 |
| Mul BF16, equal 64×64 | 101.708 / 40.486 | 100.575 / 40.795 |
| Add BF16, equal 65×70 | 115.233 / 62.516 | 118.339 / 63.067 |
| Mul BF16, equal 65×70 | 129.600 / 61.384 | 132.274 / 61.663 |
| Mul F32, RHS row 65×70 | 29.024 / 1128.878 | 29.204 / 1132.564 |
| Mul BF16, RHS row 65×70 | 97.691 / 1157.330 | 99.054 / 1161.196 |

The BF16 SFPU baseline includes widening both packed operands, SFPU arithmetic,
and narrowing the result; matrix includes final narrowing only. This comparison
measures the actual selected packed-storage routes, not isolated instruction
throughput. F32 matrix is slower here; the initial broadcasts cost roughly 1–3 ms through
resident coordinate materialization, well above the SFPU broadcast paths. Even
for equal shapes matrix jobs are bounded to one physical tile, and their fixed
packet/setup costs are visible: aligned F32 add records 36 batches/18 pack waits
versus SFPU's 18 batches/0 waits over warmups and samples. BF16 equal add records
72 batches/36 waits versus the SFPU adapters' 126/54. These counters describe
scheduler work, not an NC timestamp estimate. SFPU remains the default; this
baseline establishes a packed BF16 benefit for these equal-shape workloads.
The broadcast staging cost is addressed in the continuation below; per-tile
dispatch remains an optimization opportunity.

### Direct matrix RHS broadcast comparison (2026-10-06)

Run `1791256590` repeats the M1 benchmark after removing expanded RHS tensors.
Both p150a cards, two discovered tiles, release, TF32 Src for F32/BF16 Src for
packed BF16, HiFi4, resident operands, two warmups/seven samples, default
streaming policy. Host timing spans dispatch through sync; every output is
validated outside timing, including packed output narrowing. No NC timestamps
are used. The raw conditions contain SHA `6d979e7c5606+dirty`; clocks were not
independently sampled. The MNIST simulator regression ran concurrently on the
host, so the cross-run ratios are baselines, not an isolated instruction ceiling.
All 192 ADD/SUB/MUL, geometry, storage and card comparisons pass. Artifacts:
`target/silicon/bench/1791256590.{jsonl,md}` and raw
`target/silicon/out/1791256590-*` (same benchmark command as above).

Ragged 65×70 multiplication medians (µs), initial materialized RHS → direct RHS:

| RHS / storage | Card 0 | Card 1 |
|---|---:|---:|
| Row / F32 | 1128.878 → 41.605 | 1132.564 → 41.988 |
| Column / F32 | 1103.410 → 41.958 | 1103.100 → 42.728 |
| Scalar / F32 | 1137.273 → 41.386 | 1139.826 → 42.218 |
| Row / BF16 | 1157.330 → 60.793 | 1161.196 → 61.293 |
| Column / BF16 | 1157.168 → 62.746 | 1166.045 → 61.804 |
| Scalar / BF16 | 1252.574 → 62.376 | 1255.642 → 62.896 |

Direct broadcasts cost about the same as equal-shape matrix arithmetic now:
F32 equal multiply 41.186 / 41.978 µs and BF16 60.091 / 61.594 µs. In the new run,
SFPU row/column/scalar multiply is F32 29.193 / 76.833 / 22.972 µs on card 0
(28.883 / 77.092 / 22.952 on card 1). Matrix still loses F32 row/scalar, but wins
this column workload. Packed BF16 matrix beats the widening/narrowing adapters
in all three measured geometries (card 0 SFPU 123.930 / 154.085 / 93.793 µs;
card 1 123.758 / 153.684 / 87.412). No application speedup is inferred.

Across warmups and samples, F32 row multiplication staging transfer packets fall
108 → 0; compute counters remain 18 regions, 81 batches, 63 pack waits. This
isolates removal of coordinate transfers and expanded GDDR staging while
preserving the bounded tile scheduler. Resident Burn views may still materialize
before entering the kernel. SFPU remains the default.

### Mixed BFP storage: measured conversion overhead (2026-10-06)

Run `1791300393`, `cargo xtask bench --device all --filter
step94_bfp_storage::benchmark_resident_bfp_conversion_and_packed_product`.
Release `c55a447f382e+dirty`, two discovered Tensix tiles, default streaming,
AICLK 1350 MHz, eight GDDR channels at 16000 MT/s. Counter rates measured
1350.080 / 1350.080 MHz (cards 0 / 1). Host `Instant` timing includes completed
submission and synchronization; uploads/downloads and validation are outside
the samples. Two warmups, nine samples, medians below. Inputs are
exact unit values, products validated at every repetition. Shape is
`37x65 @ 65x35`; pack/unpack cover **both** operands, product consumes packed
operands directly with HiFi4 and F32 accumulation/output.

| Format | Slot bytes | Pack card 0 / 1 (µs) | Unpack card 0 / 1 (µs) | Packed product card 0 / 1 (µs) |
| --- | ---: | ---: | ---: | ---: |
| BFP8 | 1152 | 47.936 / 46.342 | 249.233 / 251.037 | 71.568 / 71.036 |
| BFP4 | 640 | 47.825 / 46.622 | 248.643 / 245.226 | 71.958 / 71.166 |
| BFP2 | 384 | 48.046 / 46.833 | 243.583 / 238.293 | 72.328 / 71.326 |

Each format's eleven repetitions produce 1232 scheduled batches and 924 NC
release waits per card, with zero standalone transfer packets: packed and
widening transfers share kernel scheduling. The bounded eight-bank widening
path reuses output scratch only after NC release. Physical images, headers,
exponents and 64-byte slot alignment are included in the allocations; raw
diagnostic readback excludes alignment slack. These sizes reduce resident
operand memory. The conversion overhead is substantial; no model speedup
or larger-GEMM throughput improvement is claimed. Raw results and percentile
spread are in `target/silicon/bench/1791300393.{jsonl,md}`.

### Mixed-storage MNIST policy observations (2026-10-06)

Both-card run `1791300035` agrees with simulator `step96_mixed_bfp_mnist`.
Four real training images, full-batch SGD, four updates at lr=0.05; MLP
784→16→10 with seed 23; CNN uses its existing deterministic Init. F32 master
parameters, gradients and loss inputs; biases/input/logits are explicitly F32.
Mixed columns name hidden/conv weight, ReLU activation and head weight formats.
This tiny training-set diagnostic records compression effects; it does not
establish held-out accuracy or convergence. The F32 golden remains unchanged.

| Model / policy | First → fourth loss | Final training accuracy |
| --- | --- | --- |
| MLP F32 | 2.336746 → 2.067829 | 3/4 |
| MLP BFP2 / BFP4 / BFP8 | 2.320056 → 2.177817 | 1/4 |
| MLP BFP4 / BFP2 / BFP8 | 2.335825 → 2.155071 | 1/4 |
| MLP BFP8 / BFP4 / BFP2 | 2.322205 → 2.221868 | 1/4 |
| CNN F32 | 2.227119 → 2.173344 | 1/4 |
| CNN BFP2 / BFP4 / BFP8 | 2.238755 → 2.192377 | 1/4 |
| CNN BFP4 / BFP2 / BFP8 | 2.264418 → 2.229749 | 1/4 |
| CNN BFP8 / BFP4 / BFP2 | 2.258909 → 2.228090 | 2/4 |

All masters changed, tested gradients/master updates stayed F32, training had
no intermediate downloads, and changed-image traces matched fresh execution
bit-for-bit. Named precision boundaries are shared model code, not a Burn fork.

## Residency plans over reused GDDR (2026-10-24, card 0)

[long-context.md](../plans/long-context.md) 1.G G2. A capture that reuses its own tensors' GDDR is planned for L1
residency again: the planner charges each access to the tensor its address held at that op. Before, a chunk capture
that reused was not planned. `traced_generation_stream_ceiling`, 1.7B, `TT_TILES=all`, `TT_GEN_MAX_LEN=8192`,
`TT_GEN_PROMPTS=1000,7992`, `TT_GEN_NEW=4`, 2 warmups and 9 samples, release, after the blocked products (`5ae40a2`).
"Unplanned" is `TT_RESIDENCY=0`, which also leaves the decode steps unplanned; its prefill is the chunks as they ran
before G2.

| Chunks of 64 | G2 (`1792873151`) | Reuse, unplanned (`1792873658`) | Planned, no reuse (`TT_CHUNK_REUSE=0`, `1792874014`) |
|---|--:|--:|--:|
| Prefill, 1,000 / 7,992 tokens | **841.4 / 15,244 ms** | 853.3 / 15,432 | 841.4 / 15,243 |
| GDDR free after the build | **17.70 GB** | 17.68 | 2.00 |
| Build, first / second | 224.3 / 68.4 s | 91.8 / 47.7 | 230.0 / 70.9 |
| Decode step after 1,000 / 7,992 | 26.01 / 54.91 ms | 23.69 / 53.82 | 26.01 / 54.87 |

| Chunks of 512 | G2 (`1792875893`) | Reuse, unplanned (`1792875430`) |
|---|--:|--:|
| Prefill, 1,000 / 7,992 tokens | **474.0 / 12,580 ms** | 484.3 / 12,629 |
| GDDR free after the build | 15.07 GB | 15.10 |
| Build, first / second | 893.3 / 133.3 s | 135.2 / 91.6 |

- **The plan was the slowdown:** without it a chunk is 1.4% / 1.2% slower at 64 (p10-p90 under 0.1%), 2.1% / 0.4% at
  512. G2 matches the planned capture without reuse to 0.1% while it frees 15.7 GB more.
- **The first build pays:** a planned chunk trace is captured twice (a plan pass and the planned capture), and the host
  expands every move of its graph three times (the plan, each op's check, the verify). At 512 the first build took
  893 s, just under `cargo xtask bench`'s default 900 s limit (`1792874529` timed out at it); a second build applies
  the cached plans.
- **One planned capture is refused in every planned build,** with or without reuse ("op 34 is laid out unlike the plan
  pass's"), and captured plainly; it predates G2.
- **Decode is 1-2 ms faster with `TT_RESIDENCY=0`** at 8,192 positions (23.69 against 26.01 ms after 1,000 tokens).
  G2 does not touch the steps; not yet looked into.

## Prefill products that reuse their operands (2026-10-24, card 0)

[prefill-operand-reuse.md](../plans/prefill-operand-reuse.md) Step 1: a job owns an `mb x nb` block of output tiles and
streams `K` in chunks of `kq` tiles, its FP32 partials in L1 between chunks, each chunk's reads under the chunk before's
compute; each output's block of each role is stored once and repeated by the runner (P1b). Bit for bit the other
paths (`step229`, ttsim and card 0, `1792855491`, `1792857062`). All runs release, every tile, TF32/HiFi4.

**The products** (`matmul_throughput`, `TT_MATMUL_SHAPES`, traced replay, 2 warmups and 7 samples; each replay equal to
the eager product, which equals exact integer arithmetic on a grid). Off `1792857503`, `TT_MATMUL_BLOCK=auto`
`1792857489`, same build:

| Product | Off | Blocked | GDDR read, blocked |
|---|--:|--:|--:|
| `gate\|up` `[64,2048]@[2048,16384]` | 3.40 TFLOPS | **5.01** | 194 GB/s |
| `gate\|up` `[512,2048]@[2048,16384]` | 4.89 | **14.06** | 211 |
| `gate\|up` `[2048,2048]@[2048,16384]` | 5.21 | **18.23** | 339 |
| `down` `[64,8192]@[8192,2048]` | 1.57 | **2.23** | 142 |
| `down` `[512,8192]@[8192,2048]` | 2.73 | **7.07** | 225 |
| `down` `[2048,8192]@[8192,2048]` | 2.80 | **10.06** | 209 |
| `qkv` `[512,2048]@[2048,3072]` | 3.15 | **5.86** | 171 |
| `o` `[512,2048]@[2048,2048]` | 2.69 | **4.21** | 134 |

- **No longer GDDR-bound.** Fixed blocks on the `m` = 2048 products (`1792855574`-`1792855639`, unrolled;
  `1792857227`-`1792857286`, looped): blocks of 24-64 outputs read 120-340 GB/s and are compute-bound (8x8 `gate|up`
  15.48 TFLOPS); blocks of 8-12 read ~400-440 GB/s and are read-bound (`[2,16,4]` 9.20).
- **What a pair and a reload cost** (a `4 x 6` block of `down` at `m` = 2048, `kq` 1 / 2 / 4 / 8: 6.77 / 7.80 / 8.12 /
  8.66 TFLOPS, 5.07 / 4.40 / 4.23 / 3.97 ms a unit's two jobs): **~3-5 us a chunk** for 24 outputs' reloads and packs
  and its batch (~0.15 us an output), and **~0.4-0.6 us a K-tile pair** (540-810 cycles against HiFi4's 64). The pair
  is what binds now: ~16-20 TFLOPS on 120 units.
- **Looped and unrolled programs run alike** (`[6,5,6]` 14.55 against 14.47 TFLOPS); looped, a role is under 400 words
  whatever the block, unrolled the unpack role is ~130 words an output (8x8 does not fit `PROGRAM_MAX`).

**The 1.7B's prefill** (`traced_generation_stream_ceiling`, `TT_GEN_MAX_LEN=8192`, `TT_GEN_PROMPTS=1000,4000,7992`,
`TT_GEN_NEW=4`, 2 warmups and 3 samples, host time of the prompt's chunks and the first token):

| | Before (`1792809505`, chunks of 64) | Chunks of 512, off (`1792856587`) | Chunks of 64, blocked (`1792857529`) | Chunks of 512, blocked (`1792858000`) |
|---|--:|--:|--:|--:|
| 1,000 tokens | 1,236 ms | 1,012 | 853 | **477** (2,095 tokens/s) |
| 4,000 tokens | 6,812 | 5,845 | 5,289 | **3,702** (1,081) |
| 7,992 tokens | 18,327 | 16,879 | 15,290 | **12,588** (635) |
| Build, first / second | 213.5 / 69.3 s | 181.1 / 112.5 | 127.8 / 58.1 | 174.2 / 104.2 |

- **Chunks of 512 now pay:** a chunk's products reuse their operands, so a larger `m` buys rate (`m` 64 -> 512:
  `gate|up` 5.0 -> 14.1 TFLOPS) where before it bought only fewer chunks.
- **Chunks of 512 need the looped kernel:** unrolled, the capture was refused ("a list's programs do not fit the
  program cache ... the list needs 54176 bytes in 9 programs; the capture in progress holds 133712 of its 245760
  bytes", `1792856420`).
- **What is left at 7,992 tokens** is mostly attention over the prefix (the softmax's passes over the scores: 42% of a
  64-token chunk at 8,192 positions before, `1792815504`): the fused long-context softmax is the next lever.

## GDDR reused within a capture (2026-10-24, card 0)

[long-context.md](../plans/long-context.md) 1.G. `traced_generation_stream_ceiling`, 1.7B, `TT_TILES=all`,
`TT_GEN_MAX_LEN=8192`, `TT_GEN_PROMPTS=1000,7992`, `TT_GEN_NEW=4`, 2 warmups and 3 samples, release.

| | No reuse, chunks of 64 (`1792816593`) | Chunk captures reuse (`TT_CHUNK_REUSE=1`, the default since), 64 (`1792817025`) | Reuse, chunks of 512 (`1792817342`) |
|---|--:|--:|--:|
| GDDR free after the build, all / fullest channel | 2.10 / 0.26 GB | **17.78 / 2.22** | 14.79 / 1.85 |
| Build, first / second | 226.3 / 70.5 s | 126.9 / 56.0 | 182.5 / 112.9 |
| Prefill, 1,000 tokens | 1,236 ms | 1,255 | **1,011** |
| Prefill, 7,992 tokens | 18,323 ms | 18,438 | **16,877** |
| Decode step after 1,000 / 7,992 | 26.07 / 54.87 ms | 25.98 / 54.98 | 26.05 / 54.92 |

- The 1.7B's chunk traces held ~15.7 GB of intermediates (every layer's scores, a trace a length); reused, each holds
  about one layer's. Without it, a 512-token chunk's capture at 8,192 positions needs ~48 GB.
- `step188` (card 1, `1792815355`, no reuse): 0.34 GB free after its build, 26.85 GB once its traces are released.
- A 512-token chunk is only 1.09-1.22x faster: its products run at the same GDDR-bound rate as a 64-token chunk's
  ([below](#how-much-of-the-cards-compute-a-long-prefill-uses-2026-10-24-card-0)).

## How much of the card's compute a long prefill uses (2026-10-24, card 0)

Asked: with prefill chunks this large, is the card's compute saturated? No. Peak is the scoreboard's derived ~166 TFLOPS
(HiFi4 on 120 tiles; not measured). FLOPs from the 1.7B's shapes: a 64-token chunk's projections 206.4 GFLOP, its
`qk` and `pv` 51.5 GFLOP each at 8,192 positions (computed over the whole prefix, masked positions included).

- **A whole prefill** (`1792809505`, the D2+D3 bench, 8,192-position cache, chunks of 64):

  | Prompt | TFLOP issued (causal, useful) | Time | TFLOPS issued (useful) | Of ~166 |
  |---|--:|--:|--:|--:|
  | 1,000 tokens | 3.4 (3.3) | 1.236 s | 2.78 (2.69) | 1.7% |
  | 4,000 | 15.1 (14.5) | 6.812 s | 2.22 (2.12) | 1.3% |
  | 7,992 | 34.3 (32.0) | 18.327 s | 1.87 (1.75) | 1.1% |

- **Inside a chunk** (profile mode, `1792815504`; device spans, medians over replays):

  | `Chunk { c: 64, len }` | Replay | Projections | `qk` | `pv` | softmax (6 ops a layer) | scale and mask |
  |---|--:|--:|--:|--:|--:|--:|
  | `len` 1,024 | 82.1 ms | 59.7 ms (3.46 TFLOPS) | 1.5 ms | 2.8 ms | 11.7 ms | 1.1 ms |
  | `len` 8,192 | 186.5 ms | 55.3 ms (**3.73 TFLOPS**, 2.2%) | 10.9 ms (4.75, 2.9%) | 20.6 ms (2.51, 1.5%) | **78.3 ms** (42%) | 11.4 ms |

  The projections' span is 4% kernels and 91% data: they read 20.4 GB a chunk for 6.9 GB of weights (`B` 13.5 GB,
  read once a 32-row tile row; `A` 6.8 GB, once a column block) at 369 GB/s. The softmax's SFPU ops move the
  `[32 x 64, 8192]` scores (67 MB a tensor, six passes a layer) at 172 GB/s, 29% kernels.
- **The products at prefill shapes, by `m`** (`matmul_throughput`, `TT_MATMUL_SHAPES`, every tile, TF32/HiFi4, traced
  replay, 2 warmups and 7 samples, `1792816512`):

  | Product | `m` 64 | 256 | 512 | 1,024 | 2,048 |
  |---|--:|--:|--:|--:|--:|
  | `gate\|up` `[m,2048]@[2048,16384]` | 3.37 TFLOPS (264 GB/s) | 4.78 (374) | 4.97 (389) | 5.14 (403) | **5.22** (409) |
  | `down` `[m,8192]@[8192,2048]` | 1.56 (199) | | 2.76 (352) | | **2.82** (360) |

- **What it says:** a product's GDDR traffic grows with `m` as fast as its work (`B` re-read `m / 32` times), so the rate
  plateaus where GDDR does, at ~3% of the Matrix Unit's peak whatever the chunk size. Larger chunks (1.G) save the
  per-chunk fixed cost and GDDR, not compute; the lever is operand reuse in the product (each unit holding `A` and
  `B` strips and streaming `K`, then multicast along the grid: roadmap, "Prefill products that reuse their operands"),
  and for long prefixes the softmax's passes over the scores (a fused or flash prefill attention).

## A long prompt's prefill: chunks attend to a prefix, the GQA mask read as a block (2026-10-24, card 0)

[long-context.md](../plans/long-context.md) D2 and D3. `traced_generation_stream_ceiling`, 1.7B, `TT_TILES=all`,
`TT_GEN_MAX_LEN=8192`, chunks of 64, `TT_GEN_PROMPTS=1000,4000,7992`, `TT_GEN_NEW=4`, release, 2 warmups and 3 samples
(`TT_GEN_SAMPLES=3`; the conditions line prints the default 9), host time of `generate_tokens`' first token (the
prompt's chunks and the token read back).

| | Before (`1792808294`) | Mask as a block, D3 (`1792809099`, `TT_GEN_PREFIXES=none`) | And a prefix, D2 (`1792809505`) |
|---|--:|--:|--:|
| Prefill, 1,000 tokens (16 chunks) | 4,689 ms | 2,987 | **1,236** (809 tokens/s) |
| Prefill, 4,000 tokens (63 chunks) | 18,463 | 11,759 | **6,812** (587) |
| Prefill, 7,992 tokens (125 chunks) | 36,626 | 23,328 | **18,327** (436) |
| A chunk | ~293 ms at every start | ~187 | follows its prefix |
| Decode step after 1,000 / 4,000 / 7,992 tokens | 26.03 / 36.80 / 54.80 ms | 26.02 / 36.82 / 54.91 | 25.96 / 36.91 / 54.91 |
| Build, first / second | 179.3 / 61.8 s | 139.2 / 39.7 | 213.5 / 69.3 |

- **The mask's row gather was 36% of a chunk** (293 -> 187 ms): `expanded` built a `(0, row)` list of `kv_heads R` rows
  for the `[1, 1, 64, 8192]` mask against `[1, 32, 64, 8192]` scores in every layer, ~25 M read entries a chunk. Read as
  a block (`Broadcast::Block`: plain `READ_RUN`s over the mask's own grid, wrapping every 64 rows) it is a few records a
  run.
- **A prefix** (`Shapes::chunk_len`: the smallest step length holding the chunk's rows) cuts each chunk to its context:
  1,000 tokens 2.4x, 7,992 1.3x (half its chunks still attend to all 8,192). The build captures a chunk trace a length
  (8 at 8,192), +54% build time (accepted).
- Bit for bit throughout (gates `step187`, `step227`, `step224`, `step188`). Next for long prompts: larger chunks, which
  need a capture to reuse GDDR (it holds every layer's scores), and the attention composition's own cost at long
  prefixes.

## The 1.7B at 8,192 positions from a 7,992-token prompt (2026-10-24, card 0)

`step188_smollm2_1_7b::the_native_context_agrees_with_flex`, run `1792800061`, release, every tile, one run (a gate,
not a benchmark: no warmups). `Shapes::new(8192, &[64])`: 125 chunks of 64 tokens, each attending to all 8,192
positions, then 8 decode steps.

| | Prompt (7,992 tokens) and 8 tokens |
|---|--:|
| burn-flex, host (`generate_fixed`, one core busy) | 1,846.7 s |
| burn-tt, traced | **38.7 s** |

The tokens are the same; the largest logit difference 0.166 on 33.7. The prefill is ~0.3 s a chunk: every chunk attends to
the whole cache through the composition (D2 and flash prefill are open), which a timed benchmark should measure next.

## The 1.7B at its native 8,192 positions (2026-10-23, card 0)

[long-context.md](../plans/long-context.md): `traced_generation_stream_ceiling`, `TT_TILES=all`, `TT_GEN_MAX_LEN=8192`,
`TT_GEN_PROMPTS=32,1000`, `TT_GEN_NEW=16`, release, 2 warmups and 9 samples, run `1792793709` (before: refused at
2,048 and 8,192, `1792789393`, `1792789257`).

| | 512-position cache (`1792763914`) | 8,192-position cache |
|---|--:|--:|
| Build, first / second | | 234.7 s / 72.5 s |
| Decode step, 32-token prompt (`Step(64)`, the fused attention op) | 16.81 ms | **17.00 ms** |
| Decode step, 1,000-token prompt (`Step(1024)`, attention as the composition) | | **25.98 ms** |

- The first build's time is the eager warm-up of every bucket at 8,192 positions; the second build captures only.
- One trace's residency-planned capture was refused ("op 34 is laid out unlike the plan pass's") and captured plainly.
- Past the fused op's limit (512 positions at `head_dim` 64) attention costs ~9 ms more a step: flash-decode's case.

## Flash-decode's probe: a head's positions in two halves (2026-10-23, card 0)

[split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md), P0 of its plan: before any exchange is built,
time each head's attention as two halves of its positions on two units, with nothing exchanged between them
(`tt_kernels::attention::probe_split`; the result is then wrong, its time is the point). Release, every tile
(`TT_TILES=all`), 2 warmups and 9 samples.

- **Isolated** (`silicon_probe_attention_rows::attention_rows_split_in_half`, 32 heads of `head_dim` 64, one query row,
  nothing else reading GDDR; `1792787552` and its rerun):

  | Positions | K and V | Today, 32 units | Halves, 64 units | One half a head, 32 units |
  |--:|--:|--:|--:|--:|
  | 128 | 2.10 MB | 33.2 us | 32.3 | 26.2 |
  | 256 | 4.19 MB | 47.6 | 39.1 | 33.0 |
  | 512 | 8.39 MB | 80.6 | 60.0 | 47.8 |
  | 1024 | 16.78 MB | does not fit L1 | 105.9 | 80.7 |
  | 2048 | 33.55 MB | does not fit | does not fit | does not fit |

  Today's op costs ~18 us fixed plus ~4.3 GB/s a unit. A half on 64 units is slower than the same half on 32 (60.0
  against 47.8 us): a unit reads slower as more read, as in the balanced deal.
- **In a 1.7B step** (`traced_generation_stream_ceiling`, `TT_PROBE_ATTENTION_SPLIT=1`, every length split):

  | | 32 / 256-token prompts | Run |
  |---|--:|---|
  | today | 16.81 / 18.19 ms | `1792788575` |
  | halves, no exchange | 17.08 / 17.71 | `1792788725` |

  At `Step(512)` (the 256-token prompt) the halves save 0.47 ms, **19.7 us a layer before any exchange**. At `Step(64)`
  / `Step(128)` they cost 0.27 ms.
- **What it says:** the bit-for-bit split (five exchanges: a run's pre-fold maximum and sum, the head's maximum and
  `l` back, the partial `pv` chain) and the proposal's variant A (three) both start from those 19.7 us, so neither
  clears the probe's stop rule (≥ 20 us a layer after the exchanges). A5 stays parked.
- **What it found besides:** today's job does not fit L1 past 512 positions (`attention::fits`), so a longer cache's
  decode runs attention as the composition. Halves fit to 1024. That, not the speed at 512, is what a split would buy
  (revisit with a context past 512).

## A product's units metered, and a balanced deal (2026-10-23, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, "A product's longest unit". 1.7B, every tile
(`TT_TILES=all`), release, `traced_generation_stream_ceiling`, 2 warmups and 9 samples, validated.

- **The meter** (no behaviour change): `trace::OpRecord::unit_bytes` (each unit's `OpBytes`, its own entries and its
  lists read ahead; they sum to the op's, gate `step216`), joined in `skew.tsv` (`span_mb`, `ahead_mb`, `span_gbps`)
  and in `summary.md`'s "Units against each other" (the slowest unit's MB and GB/s in its span, and the span's bytes at
  469 GB/s).
- **What it shows** (`Step(64)`, profile `1792762439`, per layer, medians):

  | Product | Span us | Barrier us | In span / ahead MB | Span at 469 GB/s | Units |
  |---|--:|--:|--:|--:|---|
  | `qkv` | 54.3 | 33.6 | 0 / 51.1 | 0 | all read ahead |
  | `o` | 22.7 | 5.0 | 0 / 17.0 | 0 | all read ahead |
  | `gate\|up` | 231.9 | 13.4 | 80.9 / 55.4 | 172.6 | 0-47 1.33 MB each (5 columns) at ~6.0 GB/s, 48-63 1.07 MB; 64-103 none, done in 13 us |
  | `down` | 172.0 | 3.6 | 46.9 / 21.3 | 99.9 | 0-23 a whole column (1.07 MB) at 6.8 GB/s; 64-103 half a column, 79 us |
  | the head (once) | 830.5 | 68.4 | 359.9 / 55.4 | 767.3 | 0-63 5.4-5.7 MB at 7.2 GB/s; 64-103 none |

- **The balanced deal** (`Session::set_balanced_deal`, `TT_BALANCED_DEAL=1`; host only): where a `K`-whole product's
  own units would run more than a round, the columns left after the first round and the prefetching units' jobs are
  dealt over both, the remainder on the prefetching units. Bit for bit (`step205`
  `balanced_deals_replay_as_the_ops_run_fresh`, ttsim and card 0; card 0 with it on, 51 of 51, `1792763568`, `step188`
  bit for bit).

  | | Step, 32 / 256-token prompts | Run |
  |---|--:|---|
  | off | 16.81 / 18.21 ms | `1792763914` |
  | on | 16.86 / 18.21 | `1792764065` |

  Profile with it on (`1792764360`): `gate|up` 231.9 -> **229.4 us**, every one of its 104 units now reading 0.80 MB in
  the span, but at 4.2-5.1 GB/s, not 6.0 (units 0-23 190 us median, 212 at most; 64-103 155 / 171), and its barrier 13
  -> 86 us; the head **830 -> 925 us** (104 readers at 4.1-4.5 GB/s each, against 64 at 7.2).
- **What it says:** a unit's rate falls about as the readers grow, so the span's aggregate rate stays where it was
  (`gate|up` 349 -> 353 GB/s over its span, ~380 while its units read; the head 433 -> 389). The longest unit does not
  bind; the aggregate rate inside the span does, and more readers lower it. A column (266 KB, ~60 us at a unit's
  rate) is too coarse to even out the low-numbered units' slower service. Not the default.

## Layer sub-traces: stored bodies and calls (2026-10-23, card 0)

[layer-sub-traces.md](../plans/layer-sub-traces.md). Every tile (`TT_TILES=all`), release, the defaults otherwise, 2
warmups and 9 samples, validated; off and on (`TT_SUB_TRACES=1`) in the same build.

| Workload | Off | On | Runs |
|---|--:|--:|---|
| 1.7B decode step, 32 / 256-token prompts (`traced_generation_stream_ceiling`) | 16.77 / 18.19 ms | **16.94 / 18.33** | `1792734860`, `1792735010` |
| 1.7B decode step, 32 / 128 / 256 (`traced_generation`) | 16.81 / 17.27 / 18.20 | 16.96 / 17.40 / 18.33 | `1792735170`, `1792735711` |
| 135M decode step, 32 / 256 | 7.71 / 9.14 | 7.76 / 9.23 | `1792736431`, `1792736481` |
| ResNet-18 traced forward | 14.18 | 14.22 | same |
| ViT-Tiny traced forward | 9.25 | 9.24 | same |
| 1.7B stream ceiling second build | 12.77 s | 12.04 s | `1792735170`, `1792735711` |

- **Trace streams in GDDR, 1.7B, 512-position cache** (`traced_generation`):

  | Trace | Off | On |
  |---|--:|--:|
  | `Step(64)` (each of 64-512 alike) | 19.3 MB | **4.9 MB** |
  | `Chunk(64)` | 107.4 MB | **20.6 MB** |
  | `Chunk(128)` | 174.8 MB | **27.7 MB** |
  | `Chunk(256)` | 315.6 MB | 315.6 MB |
  | All | 675.0 MB | **383.6 MB** |

  No unit made a body in `Chunk(256)`. Why is not reported by this bench; its tables are the likely cause (126
  entries at most; `Chunk(64)`'s reach 90).
- **What each unit stores** (the space report's `SUBTRACE` lines, the planned captures):
  - 1.7B decode: 104 of 120 units call 22-23 of 23 layers (the others run no repeated op), layers 528,218 entries ->
    bodies 22,966 + tables 52,888 + kept 18,846, **5.6x**.
  - 1.7B prefill `Chunk(64)`: **8.4x**.
  - 135M decode: 114 of 120 units, 28 repeats of 19 ops, 4.5x.
  - 135M `Chunk(64)`: 6.5x.

  A tensor reference costs two table entries, so the tables are most of what is left. Slots taking the same tensor
  in every layer share one argument (2.8x -> 5.6x on the 1.7B decode).
- **Why slower:** a body read from GDDR is fetched once a call, as often as the layer it replaces, and each call adds
  its table's read and a re-read of the rest of its chunk (the stream's own chunks, now shorter, are not fewer than a
  layer's body chunks saved). The fetches a resident body would take off the critical path were 0.23 ms a step
  (planned-l1-residency 5a), so residency could at best win back that, less the table reads.
- **Found by the first card run:** a unit that runs none of the repeated ops was given an empty body, which the mover
  refuses, and every unit then waited at a barrier for its arrival (now `NoBody::Empty`; the host's emulation of the
  stored layout refuses what the mover refuses). The operating notes say how it was read
  ([silicon-operating-notes.md](silicon-operating-notes.md#a-stalled-replay-what-to-read-2026-10-23-card-0)).

## Room for a second job's `B`: the trace arena and second jobs (2026-10-23, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, P3 and R1-R2. 1.7B, every tile (`TT_TILES=all`),
release, `traced_generation_stream_ceiling`, 2 warmups and 9 samples, validated; decode step at 32 / 256-token prompts.

| Configuration | Step | Run |
|---|--:|---|
| defaults | 16.76 / 18.17 ms | `1792711476` |
| trace arena (`TT_TRACE_ARENA=1`) | 16.78 / 18.15 | `1792719606` (off in the same build: 16.78 / 18.16, `1792719760`) |
| trace arena, second jobs on every `Slots` product | 17.25 / 18.66 | `1792721521` (without: 16.76 / 18.17, `1792721671`) |
| trace arena, second jobs where the busy units run more than a round | 17.18 / 18.54 | `1792723098` (without: 16.77 / 18.14, `1792723249`) |

- **Profiles** (`Step(64)`, `--profile`): without second jobs `1792722444` (16.25 ms a replay: products 12.43 ms,
  barriers 2.27, 3,480 MB read ahead); with them on every `Slots` product `1792721989` (16.66: products 12.88, barriers
  3.27, 3,746 MB ahead; `qkv` 51 -> 80 us a layer, `o` 22 -> 33, `gate|up` ~flat, `down` 171 -> 159); with them only
  where the busy units run more than a round `1792723545` (16.66: `gate|up` 231 -> 245 us, 66.0 MB of its 136.3 ahead
  against 55.4).
- **Why it does not pay:** where a product's weights were already read ahead whole (`qkv`, `o`), a second job is only
  compute an idle unit adds to the span; where they were not (`gate|up`), the span is the busy units' longest deal (7
  columns of 432 over 64 units), which 40 columns more on idle units do not shorten (392 still leaves units at 7), and
  the extra reads ahead contend with theirs. More room for `B` a unit does not shorten the step now.

## The mailboxes packed: the L1 map before room for a second `B` (2026-10-22, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, "Room for a second job's `B`", H0 and M2. Every tile
(`TT_TILES=all`), release, the defaults, 2 warmups and 9 samples, validated.

| Workload | Before (`1792702009`) | After (`1792711476`, `1792711624`, `1792711671`) |
|---|--:|--:|
| 1.7B decode step, 32 / 256-token prompts (`traced_generation_stream_ceiling`) | 16.79 / 18.20 ms | **16.76 / 18.17** |
| 135M decode step, 32 / 256 | 7.61 / 9.10 (`1792679220`) | 7.62 / 9.09 |
| ResNet-18 traced forward | 13.94-14.13 | 14.14 |
| ViT-Tiny traced forward | 9.21 | 9.20 |

- **The space report's baseline** (H0, `need` beside `top`, `1792702009`): a step's products reach the arena's top
  (819,200 of `DATA` with the 96 KiB region) and need 807,040 (their probe floor); `down` 578,240; the row scatter and
  select 208,000; RoPE's and the mask's block copies 532,480; RMSNorm's packed row 266,240; attention 33,280.
- **A first form gave `DATA` the 88 KiB** (`0x20000..0x116000`): bit for bit on card 0 (31 of 31 model gates,
  `1792706003`) and 16.73 / 18.15 ms (`1792708053`), but every choice sized from the arena moved: a `K` = 1024 product's
  block (6 columns, its residency cap 97,984 -> 50,816 bytes), an element-wise op's run lengths (36-37 tiles, whose
  layouts met in one half), `[512]^3`'s blocks, and on one tile a traced 135M's column sums, whose programs no longer fit
  beside its traces (`--smoke` `1792706348`, 2 failed). Kept instead: `DATA` unchanged, the room capture-only.

## Each unit's own prefetch anchor (2026-10-22, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, "The GDDR idle before `o`". 1.7B, every tile
(`TT_TILES=all`), release, the defaults, `traced_generation_stream_ceiling`, 2 warmups and 9 samples, validated; decode
step at 32 / 256-token prompts.

- **What changed:** every prefetch was anchored at the last weight-streaming product, so `gate|up`'s reads waited for
  `o`'s barrier even on units that sat `o` out (24-63 take no `o` job; 104-119 run nothing). Now each unit is anchored
  at the last such product it took part in (`Session::set_prefetch_anchor`, `PrefetchAnchor::Own`):
  - an idle unit's `PREFETCH` list waits for its own anchor's barrier;
  - a busy unit's `BUSY_PREFETCH` list moves back as far as its anchor, but never across an op that planned into its
    ring or did not plan at all (`busy_start`).

  Two guards came with it. A `B` the capture allocated or wrote in place is never read ahead; that hole was latent
  under the old anchor too. Probe units are never taken.
- **Measured:**

  | Anchor (idle prefetch units) | ms | Run |
  |---|--:|---|
  | last (40), before | 16.86 / 18.33; 16.87 / 18.29 | `1792694419`, `1792695646` |
  | **own (40)** | **16.80 / 18.21; 16.79 / 18.20** | `1792694569`, `1792695341` |
  | own, the longest-idle units first (40) | 17.13 / 18.67 | `1792694718` |
  | own (48 / 56 / 32) | 17.15 / 18.54; 17.30 / 18.59; 16.84 / 18.24 | `1792694880`, `1792695030`, `1792695180` |

  The other traced workloads, own against last (`1792695341`, `1792695646`):
  - the 135M decode step is 7.62 / 8.31 / 9.10 against 7.61 / 8.28 / 9.07 ms (0.1-0.3% slower);
  - prefill is unchanged;
  - ResNet-18 is 14.12 against 14.14, and ViT-Tiny 9.18 against 9.21 (no packed products, so no anchors).
- **The profile** (`Step(64)`, `1792696105`, against `1792678073`): a layer is 661.6 -> 657.3 us.
  - `gate|up` 235.2 -> 231.7 and `down` 178.9 -> 172.8.
  - RoPE, the row scatter, `o` and `silu_mul` are each 0.5-1.4 us slower: they now run under `gate|up`'s reads.
  - The weights stream at 429 GB/s (426).
  - **The bytes read ahead are unchanged** (3,480 MB; `gate|up`'s 55.4 MB): the anchor moves the same reads earlier
    but cannot add reads.
- **What it says:**
  - A unit holds one job's `B` (one ring in its arena), so the window before `o` carries what units 24-63 already
    read after `o`, sooner. It cannot carry more.
  - Idle units added to fill it (48, 56, or the longest-idle first, which takes units 104-119 for `o`) cost the
    small ops more than they give.
  - The window is bound by L1, as the Phase 8 probe predicted ("~0.6 ms a step for every 100 KB a unit freed").
  - **On by default** (`TT_PREFETCH_ANCHOR=last` restores the old anchor): 0.08-0.11 ms on the 1.7B, against the
    135M's 0.01-0.03.

## A column's `K` chain handed between two units (2026-10-22, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, "The stream's rate", the lever after `down` prefetched.
1.7B, every tile (`TT_TILES=all`), release, the defaults otherwise, `traced_generation_stream_ceiling`, 2 warmups and 9
samples, validated; decode step at 32 / 256-token prompts.

- **What changed** (`Session::set_stream_handoff`, off by default; `TT_STREAM_HANDOFF=1`, `TT_STREAM_HANDOFF_AT=<h>`):
  each of `down`'s columns its own units would read whole (24, on units 0-23) is handed to a unit with no job in the
  product (units 24-47): the column's unit computes chunks `0..h` and writes the FP32 partial; the spare unit reads the
  other chunks' `B` beside it as a prefetch, and after the product's barrier, in a second op, reloads the partial and
  computes the rest. The same chain, bit for bit (`step205`, `step188`; card 0 `1792685194`).
- **Timed:**

  | Setting | ms | Run |
  |---|--:|---|
  | off | **16.88 / 18.32** | `1792685519` |
  | handed at half (2 of 4 chunks) | 17.07 / 18.51 | `1792686180` |
  | handed at chunk 1 | 17.22 / 18.61 | `1792686332` |
  | handed at chunk 3 | 16.88 / 18.33 | `1792686483` |
  | at half, spare units from the top (104-119 first) | 17.33 / 18.74 | `1792687395` |

- **The profile** (`Step(64)`, handed at half, `--profile`, `1792686773`, against `1792678073`), a layer:

  | Op | Span us, before -> handed | Per unit |
  |---|--:|---|
  | `down`, first op | 179 -> **155** | columns' units 0-23 93-137 us (was 138-177), spare units 42-86, idle 75-100 |
  | `down`, second op | -> **17** | the spare units 15 us each; unit 0 (the coordinator) waits 17 |
  | the next `qkv` | 51 -> **72** | its prefetch wait 8 -> 35 us |

- **What it says:**
  - **`down`'s span is bound by GDDR's aggregate rate, not one unit's.** Half a column took a columns' unit up to 137 us,
    not the ~88 us half of 176, because the spare units read the other halves at the same time: `down`'s in-span bytes
    are unchanged and now come from more readers at once.
  - **The spare units cost `qkv`.** Units 24-47 had no `down` job, so their read ahead of the next `qkv` started under
    `down`; handed, it starts after the second op. Taking spare units from the top instead (units no product takes) was
    slower still.
  - **An in-op firmware signal is not worth it here:** it would save the second op's ~17 us a layer, not the contention
    or `qkv`'s later start. Left opt-in.

## `down` prefetched, and the unit counts re-swept (2026-10-22, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, "The stream's rate", Step 2 levers 3, 3a and 4. 1.7B,
every tile (`TT_TILES=all`), release, the defaults (residency, busy prefetch cap 2, stream cap 2),
`traced_generation_stream_ceiling`, 2 warmups and 9 samples, validated; decode step at 32 / 256-token prompts.

- **What changed:** a product whose `K` is streamed (`down`) is prefetched too (`Session::set_stream_prefetch`, on):
  each idle unit takes a column and reads its first two chunks' `B` (the ring's two halves) as soon as the last
  product is done; the counts are settings (`TT_PREFETCH_UNITS`, `TT_THIN_STREAM_UNITS`, `TT_STREAM_PREFETCH_UNITS`).
- **The sweep** (whole-`K` prefetch units : weight-stream units : `down` prefetched : streamed prefetch units):

  | Setting | ms | Run |
  |---|--:|---|
  | 40 : 64 : off (before) | 17.27 / 18.74 | `1792674690` |
  | 48 : 64 : off | 17.43 / 18.87 | `1792674837` |
  | 56 : 64 : off | 17.68 / 19.07 | `1792674983` |
  | 32 : 64 : off | 17.03 / 18.45 | `1792675777` |
  | **40 : 64 : on : 40** | **16.88 / 18.33; 16.87 / 18.32** | `1792675628`, `1792677187` |
  | 32 : 64 : on : 40 | 16.88 / 18.29 | `1792676432` |
  | 24 : 64 : on : 40 | 17.81 / 19.21 | `1792676581` |
  | 16 : 64 : on : 40 | 17.79 / 19.12 | `1792676730` |
  | 32 : 64 : on : 32 | 17.04 / 18.42 | `1792677483` |
  | 40 : 64 : on : 32 | 17.10 / 18.53 | `1792677336` |
  | 32 : 64 : on : 48 | 16.98 / 18.45 | `1792676879` |
  | 32 : 64 : on : 56 | 17.21 / 18.61 | `1792677027` |
  | 56 : 64 : on : 56 | 17.32 / 18.70 | `1792676072` |
  | 40 : 56 : on : 40 | 17.30 / 18.67 | `1792677630` |
  | 40 : 80 : on : 40 | 16.99 / 18.48 | `1792677779` |
  | 40 : 64 : on : 40, `TT_STREAM_CAP` 3 / 4 | 17.09 / 18.50; 17.48 / 18.94 | `1792678560`, `1792678709` |

  The counts stay where they were; `down` prefetched is the gain. Repeats of one setting agree within ~0.02 ms.
- **The other traced workloads, on / off** (`1792679220` / `1792679522`): the 135M decode step 7.62 / 8.29 / 9.10
  against 7.61 / 8.30 / 9.10 ms (its products do not stream), prefill 1589 / 4644 / 5843 against 1589 / 4643 / 5844
  tokens/s; ResNet-18 traced 13.94 against 14.13 ms; ViT-Tiny 9.21 / 9.21. The 1.7B in the same runs 16.87 / 18.30
  against 17.29 / 18.72.
- **The profile** (`Step(64)`, `--profile`, `1792678073`, against `1792638075`): the step's weights at **426 GB/s**
  (416), 3.48 GB read ahead (2.97), the summed spans 16.36 ms (16.76). A layer's ops 647 -> 626 us:

  | Product | Span us, before -> now | Prefetch wait | Per-unit work now |
  |---|--:|--:|---|
  | `qkv` | 90.4 -> **50.8** | 43.9 -> 8.0 | units 0-23 16-45 us, 24-63 9-19, idle 64-103 13-14 |
  | `down` | 156.9 -> **179.1** | 0 | busy units 0-23 138-177 us, idle 64-103 81-94 |
  | `gate\|up`, `o` | unchanged (234, 22) | | |

- **What it says:**
  - **`qkv`'s wait was `down`'s.** With 40 of `down`'s 64 columns on idle units, units 24-63 run no `down` job, and
    their `qkv` prefetch, moved to just after `down`, starts while units 0-23 still read.
  - **What binds now is `down`'s busy units.** Each of units 0-23 reads a whole 1.06 MB column in the span at one
    unit's rate (~6 GB/s at the stream cap of 2), ~176 us, while the idle units, half of theirs read ahead, finish in
    ~90 us: `down` reads ~260 GB/s in its span. More idle units cannot remove the last whole column (56 slower), and
    a higher stream cap slows the rest (3, 4).
  - **The next levers:** a column's `K` chain handed between two units in order, each reading its half of `B` (the
    second ahead), the second reloading the first's FP32 partial (the chain unchanged, bit for bit); and the GDDR
    idle before `o` (~50 us a layer).

## The weight stream metered: what is read ahead, the prefetch's wait, and each unit's own time (2026-10-22, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, "The stream's rate", Step 1. The profile mode now
charges what a product's units read ahead (an idle unit's `PREFETCH`, a busy unit's moved `BUSY_PREFETCH`) to the product
(`OpBytes::ahead`; before, a moved list was no op's), stamps B's prefetch `WAIT` (`trace_mode::PREFETCH_WAIT`, a
`prefetch` column), and times every unit at every op from its own stamps (`skew.tsv`: its work from its last barrier's
release to its arrival, and its wait there). 1.7B, `Step(64)`, `traced_generation_stream_ceiling`, every tile, release,
the defaults (residency, busy prefetch at cap 2), `--profile`, run `1792629809`.

- **The step's weight stream: 389 GB/s** (6,978 MB of `B` over 17.93 ms of spans), of which **2,969 MB is read ahead**
  of its product (the 391 GB/s of the entry below counted 5.26 GB in the products' spans only).
- **By product, a layer** (unit 0's span and phases; work by unit group, medians over the layer's ops):

  | Product | Span us | Barrier | Prefetch wait | Data | `B` MB | Ahead MB | Units 0-63 work, min / median / max us | Units 64-103 |
  |---|--:|--:|--:|--:|--:|--:|--:|--:|
  | `qkv` | 74.4 | 32.3 | 28.2 | 7.2 | 51.1 | 49.0 | 23.7 / 65.3 / 129.2 | 13-24 |
  | `o` | 24.1 | 7.1 | 7.2 | 3.4 | 17.0 | 17.0 | 8.7 / 14.7 / 26.7 | 9-17 |
  | `gate\|up` | 248.3 | 32.7 | 67.3 | 137.1 | 136.3 | 55.4 | 110.4 / 177.5 / 257.4 | 13-19 |
  | `down` | 175.6 | 18.7 | 0 | 138.1 | 68.2 | 0 | 99.4 / 137.3 / 175.2 | (not on it) |

- **What it says:**
  - **Unit 0 waits for its own prefetch** inside `qkv` (28 us) and `gate|up` (67 us): the first job's `B`, read ahead at
    the prefetch ID's cap of 2, has not landed when the product starts, and the prefetch `WAIT` issues the rest at that
    same cap while the kernel waits.
  - **The slow units are the low-numbered ones, every time.** In `down` every unit reads one 1.06 MB column, and
    units 0-31 take ~150 us against ~112 for units 48-63 (by unit modulo 8, i.e. by channel, it is flat: 132-139);
    unit 20 is the slowest in all 24 layers (175 us). `gate|up`: units 0-23 ~230 us, 48-63 ~115. The NoC serves tiles
    unevenly ("What holds card reads at ~430 GB/s: unfair service"), and the deal gives every unit equal bytes.
  - **Idle units finish and wait.** In `gate|up` units 64-103 have only their prefetched job: 13 us of work, then
    ~235 us at the barrier while units 0-63 read the other 81 MB.
  - **`down` is at the read ceiling on its median unit** (68.2 MB in 137 us, ~500 GB/s) and loses ~38 us a layer to
    its slowest; nothing reads ahead for it, and the window before it (`silu_mul`, ~16 us) is short.
- **The meter is consistent:** unit 0's work and wait at each op equal that op's span sample by sample
  (`silicon_profile_mode`), and no unit's stamps failed to line up with its stream (0 of 352 replays x 120 units).
- **Then: the prefetch `WAIT` drained at B's read cap** (ID 6's cap raised to `max(read cap, prefetch cap)` while the
  kernel waits, set back after): the step **18.28 / 19.84 ms** (`1792631901`, against 18.44 / 19.83); profile
  `1792632189`: products 13.46 -> 13.23 ms, `gate|up` 248 -> 242 us and `o` 24 -> 19 a layer, small ops 4.47 -> 4.57 ms;
  unit 0 still waits ~55 us a layer for `gate|up`'s prefetch. The units' order is unchanged to 0.1 us (`down`: units
  0-15 150.4, 16-31 154.1, 32-47 126.2, 48-63 113.4 us). Units are numbered row-major from the grid's top
  (`Tensix::tiles`), so the slow units are the top rows.
- **Card 1 is card 0** (profile `1792632927`): every product's phases and every unit group's work within ~0.1 us.
- **Not the rows themselves.** With the tiles opened in reverse order (a probe, since removed: units 0-63, which run the
  thin products, the bottom rows), the order reverses with them: `down`'s units 0-15 106.5 us, 48-63 155.0
  (`1792633435`); the step 18.40 / 19.53 ms (`1792633155`). A unit's service depends on where it sits among the units
  reading together, so a calibration of every unit alone or all together would not predict a product's.
- **B's in-flight cap, every read** (`TT_IN_FLIGHT_CAP`, a development knob; 1.7B step, 32 / 256-token prompts, ms):

  | Cap | 12 | 8 (the tile cap) | 6 | 4 | 3 | **2** | 1 |
  |---|--:|--:|--:|--:|--:|--:|--:|
  | Step | 18.60 / 20.04 | 18.28 / 19.84 | 18.15 / 19.61 | 17.83 / 19.26, 17.80 / 19.32 | 17.66 / 19.12 | **17.27 / 18.91** | 18.13 / 19.99 |
  | Run | `1792634385` | `1792631901` | `1792634238` | `1792634091`, `1792634834` | `1792634542` | `1792634688` | `1792634992` |

  The profile at 2 (`1792635269`): `down` 175.6 -> 156.4 us a layer with its units even (145-151 us by group, was
  113-154), `gate|up` 242 -> 235, the small ops 4.57 -> 3.80 ms (the idle units' prefetch reads no longer crowd
  theirs), but `qkv` 76 -> 89 and `o` 19 -> 25 (their prefetch drained at 2). **Every read at 2 is not a default:**
  the 135M's decode step 7.57 / 8.25 / 9.05 -> 7.76 / 8.50 / 9.32 ms and prefill 2-3% slower, ResNet-18 traced 14.05 ->
  14.51 (`1792635749` against `1792636067`); ViT-Tiny and the eager forwards even.
- **So a weight stream's reads alone at 2** (`record::GATHER_STREAM` on a packed-`A` product's gathers, idle units'
  prefetch lists included; `dm::STREAM_CAP`, `TT_STREAM_CAP`), every other read at 8, the prefetch drained at 8: the
  1.7B **17.27 / 18.71 ms**, the 135M 7.61 / 8.27 / 9.07 ms and prefill 1589 / 4646 / 5848 tokens/s, unchanged
  (`1792637346`). Stream cap 1: 18.00 / 19.41 (`1792637649`); 3: 17.59 / 19.00 (`1792637796`). The profile
  (`1792638075`): the step's weights at **416 GB/s** over 16.76 ms of spans (389 over 17.93); products 13.01 ms, small
  ops 3.75; `down` 156.6 us a layer, its units even; `qkv` still 94.5 (45 us waiting for its prefetch, 13.5 us of work
  on the idle units). The idle units' prefetch lists at 8 instead: 17.21 / 18.71 against 17.27 / 18.75 in the same
  build (`1792638681`, `1792638827`): no different, so they keep the flag.

## Phase 4 re-scoped and a trace's reader sections read ahead: the small ops shorter, the step not (2026-10-21, card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) Phases 4 and 5a. Conditions as the entry below (1.7B, every
tile, `TT_TILES=all`, release, residency and busy prefetch at cap 2 on, `traced_generation_stream_ceiling`, 2 warmups and
9 samples, validated). The bench needs `TT_TILES=all`: on one tile its traces hold 200 KB of programs and attention's
next list does not fit the 240 KB cache (`1792619251`, refused by name).

- **What changed:** the row scatter reads its indices' face, not the whole index tile (64 B a decode step's position, not
  4,160); the planner also tries leaving out first what spares the least GDDR a byte (it keeps the plan that spares
  more: on the 1.7B the plan is the same, the region 95,680 of 98,304 bytes, and the mask's 8,320 and RoPE's tables'
  4,160 do not fit what is left); B reads a replay's next `PAIR_CALL`'s first reader chunk ahead on a transaction ID of
  its own (`dm::TXN_FETCH`), while it waits for the pair before it.

  | 1.7B step, ms | 32 / 256-token prompt | Run |
  |---|--:|---|
  | before (the default) | 18.39 / 19.90 | `1792615740` |
  | the index read only | 18.32 / 19.78 | `1792619550` |
  | and the sections read ahead | 18.44 / 19.83 | `1792619858` |
  | the same, prefetch cap 3 / 4 | 18.25 / 20.02; 18.31 / 20.20 | `1792620617`, `1792620765` |
  | the same, busy prefetch off | **19.27 / 20.48** (19.51 / 20.76 before, `1792615881`) | `1792620919` |

- **The profile** (`Step(64)`, `--profile`, `1792620146`, against `1792609145`): the small ops **5.00 -> 4.47 ms** (fetch
  0.83 -> 0.23, control 0.57 -> 0.75, barrier 1.32 -> 1.14, writer 0.49 -> 0.32); RMSNorm 0.89 -> 0.67 ms (fetch 35%
  -> 6%), `gamma`'s multiply's fetch 31% -> 15%, the cache write 0.48 -> 0.28 ms. **But the products 12.92 -> 13.46
  ms**, and the summed spans are 17.93 against 17.92.
- **What it says:** under busy prefetch the step is bound by the weight stream the small ops' windows carry, not by the
  small ops. What they give up, the next product waits for (its data share is unchanged at 75%, its barriers 15% ->
  17%). Off, the same changes are worth 0.24 / 0.28 ms. The products take 5.26 GB in 13.46 ms, about 391 GB/s: the
  lever now is the rate the weights stream at, not the small ops' length.
- **135M** (no busy prefetch: its `K` is 576): **7.61 / 9.06 ms** against 7.74 / 9.16 (`1792621076`).

## Busy prefetch held back by a shared transaction count, and fixed (2026-10-21, card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) 3b; [operating
notes](silicon-operating-notes.md#a-transaction-ids-outstanding-count-is-the-tiles-not-the-cores-2026-10-21-card-0).
Conditions as the entry below (1.7B, every tile, release, residency on, `traced_generation_stream_ceiling`).

- **The add's 70.5 us under busy prefetch (cap 2), split** by the profile mode's new `fetch` and `writer` stamps
  (`1792605585`; off `1792605144`): writer 66%, barrier 25%, fetch 2%, data 5%; off, 8.1 us with writer 19%.
- **NC's own waits** (profile image, `TT_PROBE_NC_WAITS=1`, a step, units' mean / busiest; off then on): fetch 98 / 189
  then 88 / 177 us; credits 7.7 / 16.7 then 6.1 / 13.1 ms (the products' compute, shorter with the prefetch); drain 29
  / 65 then 30 / 68 us; L1 write issue 106 / 219 then 106 / 215 us; a `CALL` less its parts ~397 us both ways. B's pump:
  at most 7,488 issuing turns a step, the longest 0.4 us. None of it was the add's wait: NC's list-end drain of the
  prefetch ID (the NIU's count, B's reads) was.
- **After the fix** (only B drains the prefetch ID):

  | Busy prefetch | Off | Cap 1 | Cap 2 | Cap 4 | Cap 8 |
  |---|--:|--:|--:|--:|--:|
  | Step, ms | 19.51 / 20.75 | 19.71 / 20.46 | **18.40 / 19.92** | 18.73 / 20.54 | 19.08 / 20.74 |
  | Run | `1792608298` | `1792608441` | `1792608583` | `1792608724` | `1792608865` |

  Profile at cap 2 (`1792609145`): the add 12.8 us (70.5 before), products 12.92 ms (14.79 off), summed spans 17.92 ms
  (19.01 off); RMSNorm's and `gamma`'s multiply's largest share is now `fetch` (35%, 31%), the trace's own reads.
- **The other traced workloads, busy prefetch off / cap 2** (`1792612512` / `1792612811`): 135M decode step 7.74 / 8.37
  / 9.16 against 7.77 / 8.38 / 9.17 ms, prefill 1575 / 4651 / 5863 against 1572 / 4652 / 5862 tokens/s; ResNet-18
  traced 13.94 against 14.13 ms (between processes: 14.10, 14.11, 14.12, 13.94 alternating off and on,
  `1792613197`-`1792613224`); ViT-Tiny 9.31 against 9.32. No change: none has a one-row product with `K` of 1,024 or
  more, which busy prefetch needs. The 1.7B in the same runs 19.50 / 20.74 against 18.41 / 19.94.
- **On by default** (busy prefetch, prefetch cap 2), no settings: 1.7B **18.39 / 19.90 ms** (`1792615740`, builds 65.4
  / 11.4 s), `TT_BUSY_PREFETCH=0` 19.51 / 20.76 (`1792615881`).

## The products' outputs in their readers' L1 (2026-10-21, card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) Phase 3. A product's `SCATTER` now counts as a plain write,
so its output goes into the L1 of the units that read it (`WRITE_L1`, unicast to at most four readers), and a plan whose
period pinning loses tensors for room is planned unpinned too. Conditions: every tile, release, F32, 512-position cache,
`traced_generation_stream_ceiling`, decode-step medians at 32 / 256-token prompts, 2 warmups and 9 samples, validated.

| 1.7B configuration | ms | First / second build | Run |
|---|--:|--:|---|
| residency off | 21.02 / 22.21 | 54.4 / 8.72 s | `1792593062` |
| Phase 2 (2h, the default before) | 20.07 / 21.28 | 65.0 / 11.1 s | `1792554892` |
| `SCATTER` plain, pinned plan | 20.00 / 21.18 | 64.3 / 11.1 s | `1792592920` |
| **and the `Room` trigger (unpinned)** | **19.50 / 20.71** | 65.3 / 11.2 s | `1792595329` |
| the same, busy prefetch cap 1 | 19.74 / 20.66 | | `1792595603` |
| cap 2 | 19.75 / 21.37 | | `1792595746` |
| cap 4 | 20.36 / 22.20 | | `1792595888` |
| cap 8 | 20.63 / 22.31 | | `1792596032` |

- **Why the pinned plan gained only 0.07 ms.** The residency report (`TT_PROBE_RESIDENCY=1`, `1792594850`) left 143
  tensors out for room: the products' outputs (`qkv` 20,800 B a unit, `gate|up` and `silu_mul`'s output 33,280) and the
  residual stream's two adds (12,480), which the leave-out order drops first because their classes live longest. The
  layer before the period kept all of its own in a 79 KB region. Unpinned (`1792595172`): 393 of 548 tensors resident in
  95,680 of 98,304 bytes, 262 MB of reads and 122 MB of writes a step spared (pinned: 108 and 36).
- **The profile** (`Step(64)`, `--profile`, busy prefetch off, `1792596738`, against `1792538078`): products 14.75 ->
  14.79 ms, their writes 88.1 -> 6.3 MB; the adds 0.66 -> 0.39 ms (13.7 -> 8.0 us), RMSNorm 0.87 -> 0.67, RoPE 0.80 ->
  0.68, `silu_mul` 0.45 -> 0.37; the add, RMSNorm and `silu_mul` read no GDDR. Summed spans 19.58 -> 19.01 ms.
- **Busy prefetch** (cap 2, `1792597186`): the products 14.79 -> 11.25 ms, but the adds 0.39 -> 3.38 ms (70.5 us each)
  with no GDDR read left, RoPE 0.68 -> 1.18, RMSNorm 0.67 -> 1.14: the wait is not at the GDDR controllers (operating
  notes, "GDDR reads are one queue", the 2026-10-21 correction).
- **135M** (same bench): on **7.74 / 9.18 ms** (`1792597666`), off 8.17 / 9.58 (`1792597699`); Phase 2 was 8.12 / 9.57.
- **Gates:** card 0 `1792596373` (`step217`, `step219`, `step220`, `step188`, `step193`, `step209`, `step210`, 27 of 27); `--smoke` 1,180 of 1,180 (`1792597728`).

## Residency on by default, its region sized under the products' cliff (2026-10-21, card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) 2h.

**The cliff, probed** (`silicon_probe_region_cliff`, every tile, release, `1792546337`, `1792546573`): one-row products
under a region swept past each one's predicted limit, every point bit for bit against no region.

| Product | `K` | Limit (bytes of region) | At the limit | 64 bytes past it |
|---|--:|--:|--:|--:|
| 1.7B `qkv` `[2048, 6144]` | 2048 | 110,464 | 148.6 us | 162.0 us (`nc` 2 -> 1) |
| 1.7B `gate\|up` `[2048, 16384]` | 2048 | 110,464 | 342.3 | 376.6 |
| 1.7B head `[2048, 49152]` | 2048 | 110,464 | 991.6 | 1069.7 |
| Qwen2.5-1.5B `gate\|up` `[1536, 17920]` | 1536 | 106,304 | 319.1 | 293.4 at 128 KiB (`nc` 3 -> 2: faster) |
| 8B `qkv` `[4096, 6144]` (`K` blocked) | 4096 | 376,704 | 260.6 | 261.4 |
| 1.7B `down` `[8192, 2048]` | 8192 | 339,264 | 175.7 | 178.6 |
| Qwen2.5-1.5B `down` `[8960, 1536]` | 8960 | 285,184 | 151.1 | 148.6 |
| 8B `down` `[14336, 4096]` | 14,336 | 222,784 | 585.9 | 590.5 |

- **The `K`-whole limit** is `DATA.end` less `matmul::plan_in`'s widest block's *probe* (`A` unpacked:
  `4160 ((kt + 1)(nc + 1) - 1)` bytes over `DATA.base`).
- **The `K`-blocked limits** are their stream chunk layouts'. The plan's width does not bind them.
- **Below each limit,** time and layout are those of no region.

**The step** (1.7B, F32, every tile, `traced_generation_stream_ceiling`, decode step, 32 / 256-token prompts; builds
in the same runs, first then a second in one process):

| Configuration | ms | First / second build | Run |
|---|--:|--:|---|
| residency off | 21.00 / 22.25 | 54.4 / 8.68 s | `1792556241` |
| on, region sized by the plan (what each plan uses) | 20.04 / 21.29 | 64.2 / 11.3 s | `1792546841` |
| on, fixed 96 KiB in both passes | 20.03 / 21.28 | 64.0 / 11.4 s | `1792546985` |
| **on by default** (each plan's budget, `min(cap, 96 KiB)`; plans cached; each op checked before it is queued) | **20.07 / 21.28** | 65.0 / 11.1 s | `1792554892` |

- **The rest, on / off:**
  - 135M decode step 8.12 / 8.76 / 9.57 against 8.20 / 8.84 / 9.63 ms, and prefill 1521 / 4493 / 5717 against
    1517 / 4480 / 5703 tokens/s (`1792555165` / `1792556373`);
  - ResNet-18 traced 14.16 / 14.14 ms, ViT-Tiny 9.34 / 9.35 (`1792556228` / `1792557424`);
  - MNIST MLP traced training 300 steps 984-986 / 968-990 ms;
  - the 135M's build 22.4 / 13.0 s.
- **The builds are the cost,** accepted: a build's time is not an optimization target (2h-v).

## Rows held by a few of their readers, and never-read writes dropped (2026-10-20, card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) 2f and 2g. First a probe of three ways to deliver a row
many units read whole (`silicon_probe_row_delivery`, 120 tiles, `--profile`, `1792533708`; per-op span medians, 8
deliveries a trace, idle / under a GDDR stream on the idle units, us): through GDDR 9.0-17 / 23-80; Phase 2's
multicast from every producer 140-780, unchanged with a producer's acknowledgements counted once (2f (c),
`dm::MCAST_RUN_ACKS`); one gatherer's relay op 16-51 / 28-78; **held by 4 of the readers, 8.1-20 / 11-37** (2f (a);
1, 2 and 8 holders and every case in the plan's table). Then the winner in the plan (`Rows::Held`) and the drops
(`Plan::dropped`).

SmolLM2-1.7B-Instruct, F32, every tile (`TT_TILES=all`), `traced_generation_stream_ceiling`, decode step, 32 / 256-token
prompts, `TT_RESIDENCY=1` unless off; bit for bit (`step188` traced against eager with residency on, `1792535508`; no
planned capture refused in any run):

| Configuration | ms | Run |
|---|--:|---|
| residency off | 21.03 / 22.25 | `1792535978` |
| rows in GDDR (sharded only) | 20.88 / 22.18 | `1792536101` |
| rows held by 2 / 4 / 8 | 20.23 / 21.44, 20.40 / 21.58, 20.53 / 21.77 | `1792536365`, `1792536233`, `1792536496` |
| rows held by 4, drops | 20.18 / 21.43 | `1792536628` |
| **rows held by 2, drops** (the default now) | **20.06 / 21.28** | `1792537293` |
| held by 2, drops, busy prefetch cap 2 | 20.13 / 21.68 | `1792537424` |
| held by 4, drops, busy prefetch cap 2 / 8 | 19.99 / 21.56, 20.95 / 22.60 | `1792536759`, `1792536891` |
| off, busy prefetch cap 2 / 8 | 20.70 / 22.22, 21.64 / 23.27 | `1792537023`, `1792537145` |

The profile (`Step(64)`, off `1792537693`, held by 2 with drops `1792538078`): the summed op spans 20.52 -> 19.58 ms;
the products 15.22 -> 14.75 (their packed `A` from a holder, not GDDR: 7,058 -> 7,001 MB read), RMSNorm 1.24 -> 0.87,
`silu_mul` 0.49 -> 0.45 (26.0 -> 0.8 MB written), `gamma`'s multiply 0.69 -> 0.65; the residual add unchanged (0.66),
since it reads the product's output from GDDR (Phase 3). Busy prefetch still does not pay.

Note: the bench needs `TT_TILES=all` for the 1.7B; on one tile its attention's programs do not fit the program cache
(`1792535149`, at `59e78d1`).

## Planned residency applied: replicated rows by multicast, then sharded only (2026-10-20, card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) Phase 2: a planned capture's lists rewritten so its resident
tensors move between units' L1 (`TT_RESIDENCY=1`, a 96 KiB region). SmolLM2-1.7B-Instruct, F32, every tile,
`traced_generation_stream_ceiling` (`TT_PROBE_STREAM=off`), decode step, 32 / 256-token prompts; bit for bit throughout
(`step188` traced against eager with residency on, `1792526703`).

| Configuration | ms | Run |
|---|--:|---|
| residency off | 20.98 / 22.24 | `1792518158` |
| every plain use resident (204 tensors, 33 KB, 91.5 MB of reads spared a step) | **43.89 / 45.01** | `1792526939` |
| sharded tensors only (79 resident, 25 KB) | 20.92 / 22.14 | `1792528051` |
| busy prefetch, cap 8: residency off / sharded only | 21.66 / 23.26 vs **21.53 / 23.17** | `1792528455`, `1792528191` |
| busy prefetch, cap 2: sharded only | 20.67 / 22.18 | `1792528324` |

The profile (`Step(64)`, residency off `1792527261`, every plain use `1792527601`): the products 15.23 -> 15.57 ms;
`gamma`'s multiply 0.69 -> 12.37, the residual add 0.66 -> 6.77, attention 0.65 -> 4.33, `silu_mul` 0.49 -> 1.27: each an
op whose output is a row its readers take whole on many units (a product's packed `A`, RMSNorm's twin), delivered by its
64 producing units' multicasts into one rectangle at once, which serialize (~250 us a multiply). The reads that copied a
unit's region word by word on B (`COPY_WORDS`) showed too (RMSNorm's data time 0.53 -> 1.02 ms) and became NoC
loopback reads (`READ_L1`). With replicated rows left in GDDR the step is even, and under busy prefetch 0.1 ms
better: the residual add still reads the product's output (`SCATTER`) from GDDR behind the stream (Phase 3).

## What an empty persistent region costs the step (2026-10-20, card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) Phase 2a: the top `TT_L1_REGION` bytes of each unit's arena
kept from every captured op, nothing in them. SmolLM2-1.7B-Instruct, F32, every tile,
`traced_generation_stream_ceiling` (`TT_PROBE_STREAM=off`), decode step, 32 / 256-token prompts:

| Region a unit | ms | Products' gather entries a step | Run |
|---|--:|--:|---|
| none | 20.98 / 22.24 | 196,648 | `1792518158` |
| 64 KB | 21.03 / 22.23 | 196,648 | `1792523682` |
| 96 KB | 20.97 / 22.20 | 196,648 | `1792523958` |
| 128 KB | 22.59 / 23.80 | 307,576 | `1792523804` |
| 256 KB | 22.56 / 23.81 | 307,576 | `1792523535` |

Past ~96-128 KB the products take narrower `K`-whole blocks (`matmul::plan_in`, the widest that fits the arena):
+56% gather entries, +1.6 ms. Under it, a region is free.

## A decode step's L1 residency plan, reported (2026-10-20, card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) Phase 1: what a static plan could keep in L1 between ops,
with nothing made resident. SmolLM2-1.7B-Instruct, F32, every tile, `traced_generation_stream_ceiling` with
`TT_PROBE_RESIDENCY=1` (`1792517890`, `1792518158`). The step is unchanged: 20.98 / 22.24 ms (32 / 256-token prompt).

- **`Step(64)`:** 404 ops, 548 tensors made and used up inside the step. 401 are resident in 47 addresses, sparing **294
  MB of reads and 123 MB of writes a step** (of the 300 / 168 MB they move); the weights and the cache are 7.0 GB
  external. The layer is the period (16 ops x 23 from op 30), and a layer needs 112 KB a unit (171 KB at 512 attended
  positions) against a budget of 339 KB (896 KB less `down`'s 557 KB).
- **The whole step's region is 266 KB, the logits'**: the head's row read whole by the argmax's one unit.
- **147 tensors a step (38.7 MB) are written and never read**: rows written both as tiles and packed, only one read.
- **`Chunk(64)`:** 97 of 691 resident, a 110 KB budget beside the prefill products; `down`'s K-blocked product
  accumulates in GDDR.

## Busy-unit prefetch and the prefetch pump (2026-10-20, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, Steps 3-5 and the pump. SmolLM2-1.7B-Instruct, F32,
every tile, `silicon_bench_models::traced_generation_stream_ceiling` with `TT_PROBE_STREAM=off`.

| Decode step, ms (32 / 256-token prompt), card 0 | |
|---|--:|
| before: the packed-`A` reclaim (`1792466481`) | 20.90 / 22.15 |
| `B`'s ring at the top of `DATA`, busy prefetch off (`1792469766`) | 20.94 / 22.21 |
| busy prefetch on, issued with its record (`1792469641`) | 21.93 / 23.78 |
| busy prefetch on, the pump, ID 6 cap 8 (`1792504143`) | 21.58 / 23.25 |
| cap 4 (`1792505352`) | 21.23 / 23.07 |
| cap 2 (`1792505230`) | **20.74** / 22.25 |
| cap 1 (`1792505109`) | 21.49 / **21.76** |
| off, the same build (`1792505474`) | 20.97 / 22.19 |

`Step(64)` profiles (card 1): off (`1792470407`) products 15.24 ms, the rest 4.63; on, issued with its record
(`1792470019`) products 11.66 (1.7 GB of `B` read ahead), the residual add 0.67 -> 5.15 ms (14 -> 107 us each), RoPE 0.81
-> 1.55; on, the pump (`1792504522`) products 11.59, the add 4.77 (99 us each). The add waits for its own reads behind
the prefetch at GDDR, not for the prefetch's issue: see the [operating notes](silicon-operating-notes.md#gddr-reads-are-one-queue-prefetch-traffic-is-paid-by-the-small-ops-2026-10-20-cards-0-and-1).

## The Phase 8 feasibility probe: the step without its weight stream, and under one (2026-10-20, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8. SmolLM2-1.7B-Instruct, F32, every tile (`TT_TILES=all`),
a 512-position cache, one 64-token chunk; `silicon_bench_models::traced_generation_stream_ceiling` with
`TT_PROBE_STREAM` (timing only: under the probe the products read no weights). Timed on card 0, profiled (`--profile`)
on card 1.

| Mode | Decode step, 32 / 256-token prompt (card 0) | `Step(64)` replay, profile (card 1) |
|---|--:|--:|
| `off`: today | 20.92 / 22.15 ms (`1792463041`) | 20.42 ms (`1792463722`): products 15.79, other ops 4.63 |
| `skip`: no `B` read, no prefetch | **7.85 / 9.10** (`1792463279`) | **7.46** (`1792464102`): products 3.45, other ops 4.02 |
| `skip-bg:24`: and 24 idle units reading 68 MB each | 8.62 / 10.01 (`1792465246`) | |
| `skip-bg:40` | 10.43 / 11.80 (`1792465141`) | 10.02 (`1792465817`) |
| `skip-bg:56` | 12.56 / 13.95 (`1792465032`) | 12.00 (`1792465464`) |

- **Timed `skip-bg` rows are the background's own length** (it outlasts the step): 68 MB a unit in 8.6 / 10.4 / 12.6
  ms is ~190 / ~260 / ~300 GB/s from 24 / 40 / 56 units while the step runs. Per unit that falls from ~7.9 to ~5.4
  GB/s as units are added.
- **The ops under a stream** (profile, `Step(64)`, `skip` against `skip-bg:56`): products' fixed part 3.45 -> 4.72 ms,
  RMSNorm 0.76 -> 1.24, RoPE 0.53 -> 1.06, attention 0.65 -> 1.02, the row scatter 0.46 -> 0.99, `silu_mul` 0.48 -> 0.86,
  `mul` and `add` 0.43 -> 0.84 and 0.80. `control` (the draining `LAUNCH`, `KERNEL`, `CB`, `POKE` entries) ~0.9 -> ~3.5
  ms: latency of the op's own reads, not bandwidth. The stream runs only on units that run no op, so this is GDDR
  queueing alone.
- **Today's prefetch already pays it:** the other ops take 5.28 ms with it (`off`, 20.42 - 15.14) against 4.02 without.
- **First run, for the record:** the background at the front of the streams held up the step's mask (`int_expand`, a
  row gather list on 103 units) behind 68 MB a unit (8.7 ms of barrier); the capture now moves it after each unit's
  last op (`1792462457`-`1792462904` were also one-tile sessions without `TT_TILES=all`, whose traces do not fit the
  program cache).
- **The space audit** (`session::probe_space_report`, the `off` run): the products plan up to 807 KB of `DATA` (the
  packed `A`'s ring planned with `kt` slots: 64 at `K` = 2048, of which `kt / 32 + 1` = 3 are used), the small ops
  12-50 KB, the row scatter and select 208 KB, block copies 532 KB; 404 ops are 32 distinct.

## The token on the card, RoPE as one op, and three copies gone (2026-10-19, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 7.

- **A row's argmax as one native op** (`Session::argmax_row`, `tt_kernels::sfpu::select`; burn-tt's `float_argmax` and
  `float_argmin` on one row). Two ops: each unit's run of tiles to a partial (canonical keys compared as sign-magnitude
  integers, an exact float index a lane), then one unit's choice. One looped body a tile, so a unit's program does not
  grow with its tiles (the first build, a program a tile, took ~11 KB a unit and the 1.7B's traces no longer fit the
  program cache together). `torch.argmax`'s rule (the first maximum, NaN first, `-0 == +0`), which is now the host's
  greedy rule too (`generate::argmax`; it was the last maximum under `total_cmp`).
  - **Cost** (`silicon_probe_sampling`, card 0, a `[1, 49152]` row, traced): the composition 8.95 ms a replay; the op
    about one element-wise op's time, and 10.5 us to read the token against 448 to read the logits.
- **The token on the card** (`TracedGenerator::generate_tokens`, `decode_token`, `prefill_token`): every step and chunk
  trace carries its greedy token as a second output (`TracedInference::run_outputs` reads only what is asked).
  **Sampling** (`generate::Sampler::Sample`: temperature, top-k, top-p, a host-seeded draw) runs on the card over the
  logits where the trace left them (`TracedInference::output_tensor`), eagerly: the device sort's programs, all held by a
  capture, do not fit the program cache beside a model's traces (a 256-wide sort alone nearly fills it; 512 does not
  capture). Top-k beyond one sort is found in rounds of 1,024-wide rows. **112 ms a token** for top-k 50 and top-p 0.9 on
  card 0: correct (`step212` against burn-flex), not yet fast; a native top-k is the follow-up.
- **RoPE as one op** (`burn_tt::rope`, `Session::rope`, `tensor::rope_blocks`; a burn-fusion custom operation on `Tt`):
  `x cos + rotate_half(x) sin` in one job a run of tiles, `MUL` then `MUL_ADD`, `rotate_half` read as `x`'s own tiles
  across each head's halves. Bit for bit the composition. It replaces a `cat` copy and two ops for each of `q` and `k`.
  The first build was 1 ms slower a step: its report bookkeeping asked the result view for its buffer, which copied it
  (35 us a call on 2 units).
- **Three copies gone:** a reshape to the same shape returns its input (`Llama`'s multi-head `q` fold copied a view);
  the cache's row scatter reads the value head's columns of the projection's row where they lie
  (`Session::scatter_rows_into_block`); attention writes its context as the `[1, heads dv]` row the `o` projection reads,
  with its packed twin (no repack, no packing op).

| Decode step, ms (32 / 128 / 256-token prompt), card 1 | 1.7B | 135M |
|---|---|---|
| before (RMSNorm as one op, `1792432143`) | 22.24 / 22.70 / 23.43 | 9.58 / 10.11 / 10.96 (`1792425715`) |
| the token on the card (`1792439035`) | 21.90 / 22.37 / 23.10 | |
| RoPE as one op (`1792443823`) | 21.31 / 21.78 / 22.53 | |
| the three copies (`1792444936`, `1792445319`) | **20.95 / 21.41 / 22.18** | **8.00 / 8.59 / 9.38** |

burn-flex in the last runs: 1.7B 296.5 / 307.7 / 326.4 ms, 135M 27.3 / 29.5 / 32.0. The 1.7B's traces hold 231.5 of
245.8 KB of program cache on the fullest unit.

- **Gates:** `step211_argmax_row` (the native op against the composition and the host rule: ties, NaN beside infinity,
  two NaNs, signed zeros, subnormal maxima, ragged widths, argmin, ranks one and two; unit gates on the specification
  model, watched failing with a later equal key winning); `step212_device_sampling` (greedy tokens equal the logits
  path's on three synthetic Llamas; 144 sampled draws equal burn-flex's, vocabularies 100 to 3,000; a traced sampled
  generation equals the same choices run eagerly); `step213_rope` (bit for bit on `TtBackend` and `Tt`, `head_dim` 64 and
  128 native, 32 composed, one position and four, the tables swapped differ); `step209` now checks the context's twin
  through a product (watched failing with the twin a row off); `step193` checks RoPE as one op on `Tt`. Card 0
  `1792444687` (39 of 39, `step188` among them); `--smoke` 1,152 of 1,152 (`1792447224`). Burn conformance plain 2,973
  passed, 52 as expected, `--fusion` 2,970 passed, 55 as expected; `step12_mnist`.

## RMSNorm as one op, and one region a job (2026-10-19, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phases 6b and 7.

- **One region a job** (`tensor::share_semaphore_init`). Consecutive blocks join one ownership region only when their
  semaphore initialisations are equal (`execution_groups`). The matmul's, SFPU's and reduction's plans number their
  semaphores from zero in the same order with the same starting values (`[0, 1]`, `[0, 1, 0]`, `[0, 1, 0, 1]`), so a job
  of several kinds of kernel now carries their union: one region a job, not one a kernel. The attention op
  (`silicon_probe_attention_rows`, card 0, `1792430708`): **35.2 -> 25.0 us** a layer at 64 positions, 89.4 -> 79.4 at
  512.
- **RMSNorm up to its scale as one op** (`Session::rms_norm`, `tensor::rms_divide`; burn-tt's `float_rms_norm`): every
  unit computes the statistic from the row's packed twin and divides its own tiles, where the statistic ran on one unit
  and the `DIV`'s 63 other units waited on its barrier. Bit for bit the two ops (`step210_rms_divide`).
  - **Isolated** (traced, `K` 2048, card 0, `1792431978`): the two ops 31.7 / 24.9 / 21.5 / 19.8 us on 8 / 16 / 32 / 64
    units; one op 22.8 / 16.4 / **14.4** / 17.5. Every job reads the whole 2-tile packed row, so 64 readers of one channel
    cost more than they save: at least 2 tiles a job (`session::RMS_DIVIDE_TILES`).
  - **In the step** the op took ~34 us at 64 units (profile `1792431191`), against ~36 for the two (`1792428728`): a
    small op costs about twice its isolated time inside the step, the next product's prefetch reading beside it.
- **Measured** (card 1): 1.7B 22.78 / 23.26 / 23.98 -> 22.43 / 22.91 / 23.64 ms with both (`1792430730`), -> **22.24 /
  22.70 / 23.43** with the norm on 32 units (`1792432143`; burn-flex 298.3 / 313.7 / 330.8).
- **Not fused, and why:** `x / stat * gamma`. burn-fusion puts `gamma`'s `Reshape` in a block of its own, so the product
  by it reaches the backend as a separate op that no fuser of the norm's block sees.
- **Gates:** `step210_rms_divide` (ttsim; card 0 `1792430716`, `1792432591`): bit for bit the two ops at `K` 1024, 2048
  and 4096 over signed zeros, subnormals, values near `f32::MAX`, rows holding NaN and infinities; its twin the packed
  result; one region fewer; a captured add-and-norm replayed over new rows equals the two run fresh. Negative control: a
  row with one datum moved gives every other output other bits (one ulp of one square was under the sum's ulp and moved
  nothing). `step209` again bit for bit with the shared initialisation.

## Decode attention as one op (2026-10-19, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 6 (the 2026-10-19 review's first lever). A decode
step's attention for a few query rows a key/value head (`Session::attention_rows`, `tt_kernels::attention`; burn-tt's
`module::attention_rows`) is one op: each head's `Q K^T`, scale and mask, maximum, `SUB_EXP`, sum, `DIV` and `P V`
on one unit, seven kernels in one job, the intermediates in L1. It replaces ten ops a layer (the products, their
stacking gathers, `P`'s gather, the softmax's four) and their barriers. Bit for bit the composition.

**What the kernels cost** (`silicon_probe_attention_rows`, card 0, 32 heads of 64, traced, a kernel left out at a
time; runs `1792424226`-`1792425322`):

| Per layer | 64 positions | 512 positions |
|---|--:|--:|
| first build | 70.9 us | 373.2 us |
| of which `SUB_EXP`, `DIV` | 21.3, 18.3 | 139.8, 144.7 |
| `DIV` swapped for `MUL`, `SUB_EXP` for `SUB` (timing only) | 40.7 | 116.7 |
| **a head's rows read into row 0; `SUB_EXP` row-limited** | **35.2** | **89.4** |
| reads, `Q K^T` and writes alone | 13.4 | 39.0 |

- **The cause was the layout, not the kinds.** `q` stacked by rows puts head `h` at row `h` of its tile; the first
  build computed rows `0..h + 1`, so the head at row 31 ran the whole-tile `SUB_EXP` and `DIV`, ~8 us a score tile.
  On one unit, alone, `DIV` of a `[1, 512]` row costs 29.6 us against `MUL`'s 21.7 (row-limited, ~0.5 us a tile more),
  whatever the divisor (zero, a denormal, NaN, infinity: the same); `SUB_EXP`, not then row-limited, 153.3 us (~8.5
  us a tile). Now each head's query rows are read into rows `0..m` of `A`'s slots (64-byte face rows), and `SUB_EXP` is
  a row-limited kind (`sfpu::ops::row_limited`, unit gate `row_limited_programs_keep_the_computed_rows`).
- **Row-limited reductions** (`reduce::math_programs_rows`) bought ~5 us at 512 positions: the reductions were not
  the cost.
- **Each kernel's block costs ~3.5-4 us** (its launch, credits and the roles' retirement): ~25 us of the 35.
- **Profile** (`1792423305`, before the last fix): `Step(64)` 53-55 us a layer against ~102 us for the ten ops
  (`1792382914`); `Step(512)` 217-224 us.

| Decode step, ms (32 / 128 / 256-token prompt), card 1 | Before (`1792383988`, `1792378993`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 24.04 / 24.48 / 25.39 | **22.78 / 23.26 / 23.98** (`1792425331`; 300.6 GB/s of weights at 32) | 296.0 / 310.2 / 328.3 |
| SmolLM2-135M | 11.40 / 11.80 / 12.26 | **9.58 / 10.11 / 10.96** (`1792425715`) | 27.2 / 29.8 / 32.4 |

- **First build, for the record** (`1792421939`): 1.7B 23.17 / 24.85 / ~27.1 ms, faster at 32 tokens and slower at
  256, from the row offset above.
- **Gates:** `step209_attention_rows` (ttsim, card 0 `1792425779`): bit for bit the composition
  (`burn_tt::set_attention_rows(false)`) at `head_dim` 32 / 64 / 128, MHA, GQA folded and `q` stacked by rows, 32 to
  512 positions, masked keys and values holding NaN and infinities; the report charges each call one job. Negative
  control: the positions permuted (not reversed: a two-tile chain and the lanes' butterfly fold sum a reversal to the
  same bits) give other bits for every output. Watched failing with `P V` reading the undivided exponentials.
- **Regression:** ttsim `step183`, `step184`, `step185`, `step187`, `step193`, `step199`, `step200`, `step12_mnist`, the
  tt-kernels unit tests; Burn conformance plain 2,973 passed, 52 as expected, `--fusion` 2,970 passed, 55 as expected;
  card 0 33 of 33 (`1792425779`: `step183`, `step184`, `step187`, `step188`, `step193`, `step199`, `step200`,
  `step209`); `--smoke` 1,142 of 1,142 (`1792426484`, `step209` added).

## Packed twins, a released barrier, and weights stored transposed (2026-10-19, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 3.

- **Packed twins** (`Session::set_packed_twins`): a one-row element-wise result over a long `K` also writes its
  row packed, in its own scatter, so products and RMSNorm's statistic run no packing op. 121 fewer ops a step; each
  element-wise op ~1 us more; RMSNorm's `DIV` 18.0 -> 25.6 us (its units now wait there on the one-unit statistic).
  1.7B 24.41 / 25.01 / 25.93 -> **23.95 / 24.55 / 25.52 ms** (`1792382455`, burn-flex 311.4 / 317.7 / 335.9).
- **A barrier released by unit 0** (tt-metal's pattern: each unit spins on its own L1, unit 0 multicasts the
  release). Same build, card 1: release **24.04 / 24.48 / 25.39 ms** (`1792383988`) against polling 24.09 / 24.65 /
  25.64 (`1792384336`, `TT_BARRIER_RELEASE=0`). The barrier's ~3.8 ms a step (profile, unit 0) is mostly waiting for
  the slowest unit, not the mechanism.
- **Profile `1792382914`** (twins and release): `Step(64)` 23.22 ms (23.62, `1792379163`), 835 ops (956); products
  15.58 ms at 463 GB/s; everything else 7.64 ms.
- **Weights stored transposed** (`[n, k]`, through the unpacker's transpose, so a column of `B` is consecutive tiles;
  `silicon_probe_decode_stream::decode_products_with_their_weights_stored_transposed`, card 1, isolated products on
  the decode deal): qkv 147.9 -> 157.0 us, o 67.3 -> 68.8, gate|up 341.9 -> 357.1, down 174.9 -> 179.9, head 993.0 ->
  1,021.7. Slower everywhere; not built. Isolated, the products read at 346-412 GB/s: the ~462 in the step is the
  prefetch reading part of their weights earlier.
- **Gates:** `step207_packed_twin`, `step208_barrier_release` (ttsim, card 0 `1792382182`, `1792383833`).

## The prefetch keeps its units (2026-10-19, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 3. In a capture, once a one-row product has streamed,
every op but a product runs on at most that product's units, so the units past them stay idle for the next product's
prefetch. At 512 positions attention had spread to 120 units and taken `o`'s.

| 1.7B, profile | Before | After (`1792379163`) |
|---|--:|--:|
| `Step(512)`: the products' time over their floor | 812 us (`1792363365`) | 392 |
| `Step(512)`: `o` a layer | 52.6 us (`1792363365`) | ~37 |
| `Step(512)`: `qk`, 120 units -> 64 | 27.2 us (`1792363365`) | 27.5 |
| `silu_mul`, 120 units -> 64 | 15.5 us (`1792376868`) | 18.3 |
| the `Step(64)` replay | 23.88 ms (`1792376868`) | **23.62** |

- **Timed** (card 1): 1.7B 24.72 / 25.38 / 26.17 -> **24.41 / 25.01 / 25.93 ms** (`1792378639`, burn-flex 296.1 / 312.7 /
  328.3; 280.5 GB/s of weights at 32 tokens, 55% of 512). 135M 11.40 / 11.80 / 12.26 (`1792378993`, burn-flex 27.0 /
  29.2 / 31.7), unchanged: none of its products prefetch.
- **Where the small-ops phase leaves the step** (four changes, this day): `Step(64)` 25.61 -> 23.62 ms; products 15.60
  ms at 462 GB/s (at the read ceiling); everything else 10.01 -> 8.02 ms; every byte 297 -> 322 GB/s (58 -> 63% of
  512); the timed step's weights 258.6 -> 280.5 GB/s (50 -> 55%).
- **Gates:** `step205::ops_after_a_streaming_product_keep_off_the_prefetch_s_units` (watched failing with the cap off);
  card 0 `1792378401` (28 of 28); `--smoke` on card 0 after the four changes, 1,134 of 1,134 (`1792378998`).

## Row gathers dealt by rows (2026-10-19, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 3. A decode attention's stacked outputs (`[32, 64]`, 2
tiles) were gathered a tile a unit: 64 face-row reads on each of 2 units, 22.3 us. With fewer output tiles than units,
`gather_rows` now deals rows (at least 4 a unit), each one 1,088-byte read and two 64-byte writes.

| 1.7B `Step(64)` | Before (`1792375116`) | After (`1792376868`) |
|---|--:|--:|
| `qk`'s and `pv`'s stacking gathers, 2 units -> 16 | 22.3 us each | 8.0 |
| `P`'s gather, 64 units -> 16 | 5.3 us | 8.0 |
| the replay | 24.58 ms | **23.88** |

- **Timed** (card 1): 1.7B 25.38 / 26.03 / 26.91 -> **24.72 / 25.38 / 26.17 ms** (`1792377589`, burn-flex 301.1 / 314.2 /
  333.3; 276.9 GB/s of weights at 32 tokens, 54% of 512).
- **Gates:** `step62::rows_dealt_over_units_are_the_host_s_selection` (16 units; watched failing); conformance 2,973
  passed, 52 as expected; card 0 `1792376471` (44 of 44).

## RMSNorm's statistic from the packed row (2026-10-19, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 3, "The small ops". A decode row's statistic read its
64 tiles on one unit, ~0.33 us an entry (30.1 us a norm). Packed (`pack_row`, `[K / 32, 32]`), the same squares sum in
the same order down two tiles' rows, bit for bit.

| Per norm, 1.7B `Step(64)` | Before (`1792372923`) | After (`1792375116`) |
|---|--:|--:|
| the packing, 64 units | — | 12.2 us (barrier 4.7, data 4.8) |
| the statistic, 1 unit | 30.1 | **10.9** |
| the `DIV`, 64 units | 14.7 | 18.0 |
| a norm | 44.7 | 41.1 |
| the replay | 24.71 ms | 24.58 |

- **Timed** (card 1): 1.7B 25.48 / 26.18 / 27.01 -> **25.38 / 26.03 / 26.91 ms** (`1792374654`, burn-flex 306.3 / 318.9 /
  327.3; 269.7 GB/s of weights at 32 tokens).
- **Short of the ~0.9 ms estimated:** the packing is an op of its own, with a 64-unit barrier. The saving needs the
  row written packed by the op that makes it (the residual `Add`).
- **Gates:** unit `a_packed_row_s_statistic_is_the_row_s`; `step196` `a_decode_row_s_statistic_reads_it_packed`
  (watched failing with one packed tile too few read); conformance `--fusion` 2,970 passed, 55 as expected; card 0
  `1792374366` (30 of 30).

## Where decode's bandwidth stands, and a column broadcast in the SFPU (2026-10-19, cards 0 and 1)

**The assessment** (1.7B, card 1, before this change: timed `1792364210`, profile `1792363365`):

| Measure | Value | Of 512 GB/s |
|---|--:|--:|
| Weights a step, as the bench counts them | 6.85 GB | |
| Decode step, 32 / 128 / 256-token prompt | 26.47 / 27.14 / 27.97 ms | |
| Weight bandwidth, timed (the bench's "decode weight bandwidth") | 258.6 / ~252 / ~245 GB/s | 50 / 49 / 48% |
| Every GDDR byte over the `Step(64)` replay (7.60 GB in 25.61 ms) | 297 GB/s | 58% |
| The products over their own spans | 462 GB/s | 90% |
| Everything else: 762 ops, 391 MB in 10.01 ms | 39 GB/s | 8% |

- **The products are at the read ceiling.** A layer's four take 591 us against ~584 us for their weights at 469 GB/s;
  the prefetch already reads part of `qkv`'s and `o`'s before them. Only `down` (172 us against ~143, not
  prefetched) is clearly over.
- **The gap to 90-95% is the small ops:** ~424 us a layer in the four windows between products (`gate|up` -> `down`
  ~25, `down` -> `qkv` ~80, `qkv` -> `o` ~237, `o` -> `gate|up` ~82). 4.37 ms a step of it runs on 1-2 units;
  barriers are 4.2 ms of the step.
- **At 512 positions** attention's `qk` and a `pv` gather spread to 120 units and take `o`'s prefetch units (`o`
  35.6 -> 52.6 us a layer).

**The change: a column operand broadcast by the SFPU.** A `[rows, 1]` operand (RMSNorm's division, the softmax's
`SUB_EXP` and division) was read with `READ_BROADCAST_COL`: the mover copied column 0 into every column a datum at a
time, ~10 us of each such op's ~15 us data phase. The mover now reads the tile as it lies
(`record::READ_RUN_COL_AS_IT_LIES`), and the program first copies column 0 across in `Dst`
(`sfpu::Program::broadcast_col0`: seven lane-row rotations, the first lane held, then four stores a row group), bit
for bit the mover's tile.

| Per op, 1.7B `Step(64)` | Before (`1792363365`) | After (`1792372923`) |
|---|--:|--:|
| RMSNorm's `DIV`, 64 units | 24.1 us | 14.7 |
| softmax `SUB_EXP` and `DIV`, 2 units | 24.7 | 15.0 |
| the replay | 25.61 ms | **24.71** |
| everything but the products | 10.01 ms | 9.10 |

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792364210`, `1792364661`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 26.47 / 27.14 / 27.97 | **25.48 / 26.18 / 27.01** (`1792372396`; 268.6 GB/s of weights at 32) | 294.4 / 305.2 / 329.2 |
| SmolLM2-135M | 12.62 / 13.06 / 13.59 | **11.37 / 11.76 / 12.29** (`1792372749`) | 39.7 / 30.5 / 33.1 |

- **Gates:** unit `col_broadcast_tests` (the broadcast is the mover's tile in every computed row group; every kind
  that takes a column is its plain program over the mover's tile; watched failing with six rotations and with a row
  group skipped); `step206_sfpu_col_broadcast` (ttsim, NaN in the column's padding in GDDR; watched failing with the
  broadcast dropped); record unit `records_walk_rows_to_the_same_tiles_as_dividing_per_tile`.
- **Regression:** ttsim `step30`, `step43`, `step121`, `step183`, `step184`, `step187`, `step193`, `step195`, `step196`,
  `step199`, `step200`, `step12_mnist`; `step172`/`step173` in the safe, `fast-reads` and `fast-reads` + `encoded-guard`
  images; firmware clippy plain, `encoded-guard`, `profile`, `fast-reads`; Burn conformance 2,973 passed, 52 as
  expected; card 0 `1792371974` (45 of 45, `step188` among them) and `1792372371` (mover gates and `step206`, guarded
  fast-read image).

## Decode products: their rows, a packed `A`, a streamed `K`, prefetched weights (2026-10-18, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phases 4 and 5: three of the four larger changes compared
there (the fourth, the position-major cache, is the section below). Each is bit for bit, in the backend, for any
one-row product.

| 1.7B `Step(64)` replay (card 1 profile) | ms | Run |
|---|--:|---|
| after the position-major cache | 30.91 | `1792336760` |
| the m = 1 kernel computes only its rows (`LiveRows`, `step203`) | 29.55 | `1792341444` |
| a packed `A` (`pack_row`, `step204`) | 27.90 | `1792342758` |
| a deal rotated over the channels (reverted) | 27.84 | `1792343591` |
| a packed `A`'s K blocks of 128, not 64 | 27.77 | `1792344448` |
| every one-row product's `K` streamed (replaced) | 27.88 | `1792358552` |
| only a blocked `K` streamed | 27.51 | `1792359572` |
| weights prefetched on idle units, 56 / 32 / 20 / **40** | 25.96 / 25.60 / 26.74 / **25.61** | `1792360408`, `1792361675`, `1792362531`, `1792363365` |

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792337485`, `1792337842`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 31.72 / 32.31 / 33.21 | **26.47 / 27.14 / 27.97** (`1792364210`) | 301.1 / 311.6 / 327.3 |
| SmolLM2-135M | 13.71 / 14.16 / 14.66 | **12.62 / 13.06 / 13.59** (`1792364661`) | 27.0 / 29.3 / 31.5 |

- **The 1.7B's first token:** 94.8 / 158.8 / 278.3 ms, unchanged.
- **What the prefetch costs:** the latency-bound small ops it runs under slow by ~60 us a layer at 56 units (silicon
  operating notes), hence the cap.
- **Gates:**
  - ttsim: `step203`, `step204`, `step205`, `step121`, `step190`, `step198`, `step187`, `step191`, `step193`, `step197`,
    `step199`, `step39`, `step176`;
  - Burn conformance: 2,973 passed, 52 as expected; `step12_mnist`;
  - card 0 `1792361154`: 58 of 58, `step188` (the 1.7B traced against eager) among them.

## A position-major key/value cache (2026-10-18, cards 0 and 1)

[decode-attention.md](../plans/decode-attention.md) A-layout (in-place-cache-write D5, decode-small-ops S1d). It is the
first of the four larger changes compared in
[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md#the-larger-changes-compared-2026-10-18).

**What changed:**
- **The layout:** `FixedCache` stores each layer's keys and values as `[1, 1, L, kv d]` when `head_dim` is a
  multiple of 32.
- **The write:** a step's row is its projection's row as it is, one index for every head. Head-major, each step paid
  two 31 us repacks a layer on 2 units.
- **What attention reads:** the same head-major view of the same tiles, so the bits are unchanged.

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792298195`, `1792298556`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 33.02 / 33.52 / 34.42 | **31.72 / 32.31 / 33.21** (`1792337485`) | 295.8 / 312.8 / 331.8 |
| SmolLM2-135M | 14.04 / 14.45 / 15.02 | **13.71 / 14.16 / 14.66** (`1792337842`) | 27.9 / 35.0 / 31.3 |

- **Profile `1792336760`** (`Step(64)`, 1.7B): the replay 32.32 -> 30.91 ms, and the ops 840 -> 810. The cache
  write is 1.94 -> 0.64 ms a step, and per layer it is now:
  - two row scatters of one index, on 64 units, 10.3 us each;
  - the fused `v`'s row copied out of `qkv`, 6 us.
- **The 1.7B's first token:** 94.8 / 158.9 / 278.3 ms.
- **Gates:**
  - `step199::a_position_major_cache_writes_the_projection_s_row_as_it_is` (ttsim);
  - card 0 `1792336651`: `step187`, `step188`, `step193`, `step199`, 21 of 21.

## The cache write's and the embedding's moves spread (2026-10-18, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 1. Profile `1792293993`, `Step(64)`, showed the
cache write's 96 ops at 3.06 ms, each on 2 units, and the embedding's row select at 253 us on 3. Three changes, each
bit for bit:
- **`repack_many`:** up to 32 source tiles are read into slots of their own before any copy. The reads used to run one
  after another, 38 us for a key's or value's row a head.
- **`SCATTER_ROWS` with an index range:** one batch is split between units. A repeated row is still written once,
  from its last index.
- **The row select:** its tile groups are narrowed to the units, because a ragged batch's fill costs ~10 us a tile.

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792292588`, `1792293687`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 34.35 / 34.87 / 35.76 | **33.05 / 33.59 / 34.51** (`1792295836`) | 298.9 / 305.6 / 329.4 |
| SmolLM2-135M | 14.28 / 14.68 / 15.25 | **14.04 / 14.45 / 15.04** (`1792296202`) | 26.7 / 29.0 / 31.6 |

- **Profile `1792296384`** (`Step(64)`): the replay 33.68 -> 32.34 ms. The cache write 3.06 -> 1.94 ms: the row
  scatter 25 -> 8.8 us, the repack only 38 -> 31-33 us. The embedding's select 253 -> 67 us.
- **The repack's reads cut to the span they use** (a row's two face rows, 1,152 of 4,160 bytes), after it: 1.7B
  33.02 / 33.52 / 34.42 ms (`1792298195`), 135M 14.04 / 14.45 / 15.02 (`1792298556`). Neither the reads' latency nor
  their bytes bound the repack.

## Thin products on 64 units; attention's scale and mask as one op (2026-10-18, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 4 ("The deal") and
[decode-attention.md](../plans/decode-attention.md) A4b.
- **The deal:** a one-tile-row product with `K` of at least 32 tiles runs on at most 64 units
  (`session::THIN_STREAM_UNITS`): 64 units saturate the weight stream, and more only add their own `A` reads and an
  uneven last round. Each output column keeps its `K` chain, so the bits are unchanged. It touches every 1.7B product
  (`K` = 64 or 256 tiles) and the 135M's `down` (`K` = 48); the 135M head (`K` = 18) keeps every unit.
- **The scale and mask:** `kind_sfpu::SCALE_MASK_FILL` is `MUL_SCALAR` then `MASK_FILL` in one pass, bit for bit;
  burn-tt's attention applies its first mask (the caller's, else the causal one) with the scale when there is no
  softcap.

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792290295`, `1792290681`) | Scale and mask only | Both | burn-flex |
|---|---|---|---|---|
| SmolLM2-1.7B-Instruct | 35.64 / 36.22 / 37.20 | 35.56 / 36.10 / 37.00 (`1792293194`) | **34.35 / 34.87 / 35.76** (`1792292588`) | 295.0 / 313.2 / 327.2 |
| SmolLM2-135M | 14.43 / 14.92 / 15.46 | — | **14.28 / 14.68 / 15.25** (`1792293687`) | 26.6 / 29.0 / 31.3 |

## RoPE's sum as one op (2026-10-18, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 3.
- **The change:** `x cos + rotate(x) sin`'s second product and its sum run as one pass (`MUL_ADD`, the `MulAdd`
  chain fuser), bit for bit.

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792287482`, `1792287884`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 36.12 / 36.52 / 37.54 | **35.64 / 36.22 / 37.20** (`1792290295`) | — |
| SmolLM2-135M | 14.80 / 15.29 / 15.86 | **14.43 / 14.92 / 15.46** (`1792290681`) | 26.9 / 29.1 / 31.1 |

## Attention's softmax in four ops (2026-10-18, cards 0 and 1)

[decode-attention.md](../plans/decode-attention.md) A4a.
- **The change:** `exp(x - max)` is one pass (`SUB_EXP`), and each clamp rides on its reduction
  (`Session::reduce_then`), bit for bit the seven ops.

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792280991`, `1792281384`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 36.39 / 37.02 / 37.98 | **36.12 / 36.52 / 37.54** (`1792287482`) | — |
| SmolLM2-135M | 15.33 / 15.79 / 16.47 | **14.80 / 15.29 / 15.86** (`1792287884`) | 26.6 / 29.0 / 31.4 |

## The bound on a decode product's `A` reads (2026-10-18, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 4.
- **The probe:** `silicon_probe_decode_stream`, with `tensor::probe_skip_a_reads` making every product block
  `A`-resident, so blocks read only `B`. The products are wrong; only the time counts. Run `1792283215`.
- **The bound:** without `A`'s reads a 1.7B layer's products are ~143 us faster, ~3.4 ms a step:
  - `qkv` 181 -> 157 us;
  - `o` 73 -> 56;
  - `gate|up` 383 -> 353;
  - `down` 270 -> 198;
  - the head 1,051 -> 1,034;
  - the 135M head 307 -> 285.
- **What is left:** even without `A` each product is above its weights' stream on the same units. That is the
  kernel tail and the barrier.

## A stacked product's tail rows left unread (2026-10-17, cards 0 and 1)

[decode-attention.md](../plans/decode-attention.md) A2a.
- **The cost:** the batched product gathers each head's left block into a tile row of its own, and read its 31
  tail rows as zeros, 64 bytes at a time. At `Step(512)` a 1.7B step's `pv` gather was 32,768 read entries (3.06 ms,
  profile `1792278468`).
- **The change:** under a stacked output those `M` rows reach nothing, and they are no longer read
  (`tensor::GATHER_HOLE`).
- **Gate:** `step184`, with NaN-primed staging.

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792272596`, `1792273005`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 37.24 / 38.66 / 40.50 | **36.89 / 37.30 / 38.36** (`1792279829`) | — |
| SmolLM2-135M | 17.10 / 17.52 / 18.25 | **16.05 / 16.48 / 17.29** (`1792280238`) | 28.0 / 29.2 / 31.2 |

**Then the batched product's padding off `K`, used as it is** ([A1](../plans/decode-attention.md)): no fill of
`Q`'s padding rows before `q k^T`.
- 1.7B: **36.39 / 37.02 / 37.98 ms** (`1792280991`).
- 135M: **15.33 / 15.79 / 16.47** (`1792281384`, burn-flex 26.9 / 29.3 / 31.5).

## A transposed `B` transposed by the unpacker (2026-10-17, cards 0 and 1)

[decode-attention.md](../plans/decode-attention.md) A0-A0b.
- **What A0 found:** at `Step(512)` a 1.7B step's `q @ K^T` took 144 us a layer for ~11 read entries a unit. The
  mover transposed each key tile it read, one datum at a time (`READ_TRANSPOSED`, ~10 us a tile).
- **The change:** the mover reads the tiles as they lie (`record::GATHER_B_UNPACK_TRANSPOSE`). Unpacker 0
  transposes within each face (`Haloize_mode`), and the kernel unpacks `B`'s faces in transposed order.
- **Bits:** bit for bit with the mover's transpose, special values included (`step201`, ttsim and card 0).

| Step, ms (32 / 128 / 256-token prompt) | Before (`1792268720`, `1792269106`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct decode | 37.69 / 40.20 / 43.41 | **37.24 / 38.66 / 40.50** (`1792272596`) | 310.1 / 308.4 / 325.7 |
| SmolLM2-1.7B-Instruct first token (prefill) | 101.7 / 172.5 / 297.3 | **98.7 / 169.7 / 293.4** | 7,131 / 7,621 / 12,359 |
| SmolLM2-135M decode | 17.51 / 17.97 / 18.76 | **17.10 / 17.52 / 18.25** (`1792273005`) | 26.8 / 28.9 / 31.3 |

**The context-length slope:** a 1.7B step's cost from 32 to 256 tokens fell from 5.7 to 3.3 ms.

## Element-wise ops on a decode row's row groups (2026-10-17, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 3. A lanewise SFPU kind on a single tile row of
fewer than 32 rows runs its programs over only the row groups its rows occupy: one of eight for a decode step's
`[1, W]`. RMSNorm's statistic already did this. The output's padding is claimed `Undefined`.
- **Kinds:** add, subtract, multiply, divide, the scalar add and multiply, sigmoid, `silu(gate) * up`.
- **Gates:** unit `row_limited_programs_keep_the_computed_rows`; `step200_row_limited_eltwise` (card 0
  `1792268664`).
- **Measured** (card 1, release, every tile, burn-flex in the same runs):

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792266392`, `1792266789`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 38.93 / 41.38 / 44.58 | **37.69 / 40.20 / 43.41** (`1792268720`) | 295.4 / 312.5 / 335.4 |
| SmolLM2-135M | 18.59 / 19.03 / 19.81 | **17.51 / 17.97 / 18.76** (`1792269106`) | 27.3 / 30.1 / 31.2 |

- **Against the prediction:** -1.2 ms a 1.7B step, against ~1 ms predicted (the kernels of `DIV`, `SILU_MUL`, the
  muls and adds were ~1.24 ms a step in profile `1792210770`).

## Captured barriers on the units an op ran on (2026-10-17, cards 0 and 1)

[replay-fixed-cost.md](../plans/replay-fixed-cost.md) P1.
- **The change:** a captured op's barrier counts the units it ran on. A unit that sat ops out waits (a wait-only
  `BARRIER` in its stream) before its next op and at the trace's end, and a replay adds the arrivals the capture
  counted.
- **Gate:** `step176` (ttsim; card 0 `1792266278`, 43 trace gates; `step188` `1792266393`).
- **Measured** (card 1, burn-flex in the same runs):
  - 1.7B 38.97 / 41.41 / 44.58 -> **38.93 / 41.38 / 44.58 ms** (`1792266392`);
  - 135M 18.59 / 19.07 / 19.79 -> **18.59 / 19.03 / 19.81** (`1792266789`).
- **No change, below P0's ~1.5 ms estimate:** an idle unit's arrival at a barrier did not delay the units that
  waited at it.
- **What it is for:** a unit can stream between ops without holding their barriers, which is the condition for
  decode-step-bandwidth B2's prefetch.
- **On silicon, a trace with the waits dropped still gave the right bits** (unit 0 on a 1,024-tile `K` chain, units
  1-3 reading its output next). The replay's start closes that window, so the gate checks the streams' barrier
  entries instead.
- **Under `--profile` the race was real.** The profile mode syncs after every captured op, and the first version
  took the stream's waits from the card's catch-up state, so its streams had none. The 1.7B and 135M traced
  generations then gave garbage tokens.
  - **Fixed:** the capture counts its own barriers, and each wait says how many it stands for, for the profile
    window ([replay-fixed-cost.md](../plans/replay-fixed-cost.md) P1).

## A traced cache written in place (2026-10-17, cards 0 and 1)

[in-place-cache-write.md](../plans/in-place-cache-write.md) D1-D4, Phase 1 of
[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md).
- **The change:** a `SharedStateTraces` capture donates its state before the step runs, and a row scatter into a
  donated cache writes the cache's own buffer. That removes the scatter's tile copy of the cache and the write-back
  after the step.
- **Safety:** buffer generations refuse by name a read of the cache after its write.
- **Also:** the write's index column, a `[1, kv]` buffer read transposed, is read as a row, not copied by a one-tile
  block copy (11.6 us, twice a layer).
- **Gate:** `step199_in_place_cache_write`.

**Measured** (card 1, release, every tile, `silicon_bench_models::traced_generation`, burn-flex in the same run):

| Decode step, ms (32 / 128 / 256-token prompt) | Before (`1792209913`, `1792210419`) | After | burn-flex |
|---|---|---|---|
| SmolLM2-1.7B-Instruct | 41.83 / 44.25 / 47.40 | **38.97 / 41.41 / 44.58** (`1792263398`) | 300.5 / 309.2 / 324.6 |
| SmolLM2-135M | 20.03 / 20.55 / 21.19 | **18.59 / 19.07 / 19.79** (`1792263801`) | 26.11 / 28.53 / 31.02 |

- **Against the prediction:** -2.83 ms a 1.7B step, against -2.26 predicted for the two cache copies (profile
  `1792210770`) plus -0.56 for the index copies.
- **The decode weight bandwidth:** 164 -> 176 GB/s at 32 tokens.
- **Profile `1792264052`** (`Step(64)`): the replay 41.16 -> 38.28 ms, its ops 1,152 -> 1,008. `scatter_nd` is 5.94 -> 3.03 ms: the k/v repack (1.83) and the row scatter (1.2). The step's barriers are 4.77 -> 4.01 ms, and everything but the products is 14.56 ms over 863 ops.
- **Validated:**
  - card 0: `step199`, `step187` (with the 135M checkpoint), `step189`, `step193`, `step186` (`1792263323`) and
    `step188` (`1792263379`);
  - ttsim: those and `step191`, `step197`.

## Decode products against their weight stream (2026-10-17, card 0)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 0. `silicon_probe_decode_stream`, card 0, release,
every tile (120), runs `1792261830` (first), `1792261878` (repeat, within ~1%), `1792261957` (the unit sweep) and
`1792262011` (with the heads).
- **Method:**
  - each op is captured `REPS` = 8 times in one trace and once in another;
  - the op's time is the difference over 7, medians of 9 replays after 2 warm-ups;
  - rates are 4,160-byte slots over that time.
- **Validated:** every replayed product equals the eager product bit for bit, and a stream's last slots equal `B`'s
  tiles in GDDR on the first and last unit.
- **Products:** `Session::matmul_dram`, the decode path (thin `A`, `K` whole or K-blocked for `down`), at
  `Spread::Fixed(u)`.
- **Streams:** `Session::probe_stream`, every tile of `B` read once into 64 slots at the top of `DATA`:
  - *Columns:* a unit reads columns `u, u + units, ..` down their `K` tiles, the product's deal;
  - *Rows:* a contiguous share of consecutive tiles.

| SmolLM2-1.7B product, `m = 1` | Product, 64 units | Product, 120 | `B` alone, Columns, 64 | `B` alone, Columns, 120 | `B` alone, Rows, 120 |
|---|--:|--:|--:|--:|--:|
| `qkv` `[2048, 6144]`, 51 MB | 160 us (320 GB/s) | 180 (284) | 113 (454) | 112 (458) | 105 (489) |
| `o` `[2048, 2048]`, 17 MB | 74 (229) | 73 (232) | 38 (446) | — (64 columns) | 36 (471) |
| `gate\|up` `[2048, 16384]`, 136 MB | 352 (387) | 383 (355) | 300 (454) | 306 (445) | 280 (487) |
| `down` `[8192, 2048]`, 68 MB | 269 (253) | 269 (253) | 150 (453) | — (64 columns) | 141 (485) |

The unit sweep for a product (us; 32 / 48 / 64 / 80 / 96 / 104 / 120 units):

| Product | 32 | 48 | 64 | 80 | 96 | 104 | 120 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `qkv` | 221 | 188 | **160** | 180 | 163 | 171 | 179 |
| `o` | 88 | 91 | 74 | 74 | 74 | 74 | 74 |
| `gate\|up` | 556 | 434 | **352** | 398 | 395 | 391 | 384 |
| `down` | 433 | 449 | 269 | 270 | 270 | 269 | 269 |
| head `[2048, 49152]`, 403 MB | 1,625 | 1,159 | **1,024** | 1,080 | 1,082 | 1,091 | 1,051 |
| 135M head `[576, 49152]`, 113 MB | 343 | **302** | 318 | 307 | 302 | 305 | 308 |

**What it shows:**
- **The ceiling:** reads reach ~488 GB/s, 95% of 512, in the default image.
- **Channels:** a column-dealt stream is ~7% below that, and P5's "one channel a column" is not what holds the
  products back.
- **Units:** 64 units saturate GDDR.
- **The products:** 1.2-1.8x their own weight stream. The profile mode (`1792210770`) splits the gap into `A`'s
  reads (an entry a `K` tile on every unit), a ~15 us kernel tail, and the barrier of an uneven last round.

**Per layer** (1.7B, today's deal):
- the products take ~907 us;
- their weights stream in ~600 us at the column-dealt rate, ~550 us at 488 GB/s.

## The 1.7B against its roofline, and a prefill product's rate (2026-10-17, cards 0 and 1)

Where SmolLM2-1.7B (F32, one card) stands against a roofline for it (the scoreboard's
[Roofline](performance-scoreboard.md#roofline-smollm2-17b-f32-one-card) holds the targets and their derivation).

**The generation, re-measured** (`silicon_bench_models::traced_generation`, `TT_GEN_MODEL=smollm2-1.7b-instruct
TT_TILES=all TT_GEN_FLEX_SAMPLES=1`, card 1, release, run `1792258449`, `c69b9c1+dirty` with B0/B1):

| Prompt + 64 tokens | Prefill (first token) | Decode step | burn-flex: prefill / step |
|---|---|---|---|
| 32 | **315 tokens/s** (101.7 ms) | 41.80 ms | 6.0 tokens/s / 299.1 ms |
| 128 | **741** (172.6) | 44.24 | 19.0 / 309.6 |
| 256 | **861** (297.4) | 47.44 | 21.9 / 328.1 |

- Decode is unchanged from `1792209913` (41.83 / 44.25 / 47.40).
- Prefill was last measured at 267 / 657 / 771 tokens/s (`1792082994`, 2026-10-15). It gained from the decode work
  since (fused projections, block views, the fusers), none of which targeted it.
- A chunk costs about **48 ms + 0.98 ms a token** (from the 128 and 256 prompts). The marginal token is ~3.5 TFLOPS of
  products.

**A prefill product's rate** (new: `silicon_bench_tensix_ops::matmul_throughput`, card 0, every tile, TF32/HiFi4, run
`1792258604`):
- **Validated:** a grid of elements equals exact integer arithmetic, and every replay equals the eager product bit
  for bit.
- **Timing:** the replay of one traced product, with the input write and output read excluded.
- **Bytes:** counted at capture (`set_count_trace_bytes`).

| Product | Replay | TFLOPS | GDDR read (`A` / `B`) | Of the operands' bytes | Read rate | Entries |
|---|--:|--:|--:|--:|--:|--:|
| `[256,2048] @ [2048,2048]` | 1,066 us | 2.0 | 204 MB (68 / 136) | 32x `A`, 8x `B` | 192 GB/s | 49,152 |
| `[256,2048] @ [2048,8192]` | 2,433 us | 3.5 | 801 MB (256 / 545) | 120x `A`, 8x `B` | 329 GB/s | 192,512 |
| `[1024,2048] @ [2048,2048]` | 2,504 us | 3.4 | 818 MB (273 / 545) | 32x `A`, 32x `B` | 327 GB/s | 196,608 |
| `[1024,2048] @ [2048,8192]` | 8,224 us | 4.2 | 3,203 MB (1,022 / 2,181) | 120x `A`, 32x `B` | 390 GB/s | 770,048 |

All four run on 120 units, one op each. Eager, the same products take 1.56 / 5.03 / 5.10 / 16.64 ms.

**Reading:**
- **The products are bound by GDDR, not by the Matrix Unit.**
  - `B` is read once per tile row of `A` (`m / 32` times).
  - `A` is read once per two-column output block, at most once per unit: 32x at `N` = 2048, 120x at 8192.
  - The largest product moves 3.2 GB for 77 MB of operands, at 83% of the 469 GB/s read ceiling.
- **Its arithmetic alone would take ~0.21 ms:** 524,288 tile products over 120 units at ~64 cycles each (HiFi4),
  1,350 MHz. That is ~2.5% of the replay.
- **So the product is ~40x from compute-bound.** About 4x of that is per-unit blocking: with an `A` strip and a `B`
  strip held in L1, each unit reads its operands once, ~0.8 GB in all. The rest is multicast, which shares the
  strips along the grid's rows and columns so each operand leaves GDDR once.
- **The peak of ~166 TFLOPS** (HiFi4, 120 units) is derived from the published figure, not measured: 774 TFLOPS at
  LoFi on 140 tiles at 1,350 MHz is 4,096 FLOP a cycle a tile, and HiFi4 runs four phases.

## A decode step in bytes, and the activation once a unit (2026-10-17, cards 0 and 1)

[decode-step-bandwidth.md](../plans/decode-step-bandwidth.md) B0 and B1.

**B0: the count.** With the profile mode on, each captured op's GDDR traffic is counted when its trace is stored:
- every unit's stream and both sections of each `PAIR_CALL`;
- records expanded as the movers expand them;
- a gather's reads split into its `A` slots and its `B` slots.

`ops.tsv` gains `read_bytes`, `write_bytes`, `reads`, `writes`, `a_bytes`, `b_bytes` and `gbps`, and each trace in
`summary.md` a by-label traffic table and a "Step budget". The gate is `step197_trace_bytes`: a traced step's
products read each weight's tile slots once.

**The 1.7B step in bytes** (profile `1792202359`, `Step(64)`, card 1, 42.27 ms):

| | Ops | ms | GDDR MB (read + written) | GB/s |
|---|--:|--:|--:|--:|
| Products | 97 | 24.00 | 8,218 + 107 (`B` 6,952, `A` 1,247) | 347 |
| Attention's products | 144 | 2.93 | 64 + 26 | 31 |
| The cache write and its write-back | 240 | 5.93 | 423 + 403 | 139 |
| Everything else | 671 | 9.40 | 241 + 122 | 39 |

- **The whole step:** at 469 GB/s its bytes would take 20.5 ms, and the weights alone 14.8.
- **The products by shape:** 120 units 355 GB/s, `o` on 64 units 292, the K-blocked `down` 324.
- **One layer:** `qkv` 335 GB/s, `o` 295, `gate|up` **367** (396 over its data phase), `down` 327.
- **Refuted:** "`gate|up` at ~490 GB/s" (decode-matmul-bandwidth P3, below) does not hold.

**B1: the activation once a unit.** A product's later blocks on a unit skip `A` (`GATHER_A_RESIDENT`), bit for bit
(gate `step198`).
- **Where it applies:** `gate|up`, 272 blocks on 120 units, and the head, 816. The other products run one block a unit.
- **Bytes and entries** (profile `1792210770`):
  - `A` 1,247 -> 944 MB a step;
  - `gate|up` 50,176 -> 40,448 entries, 428 -> 383 us;
  - the head 1,223 -> 1,049 us;
  - the products 24.00 -> 22.83 ms, the replay 42.27 -> 41.16.

**Bench:** `TT_TILES=all TT_GEN_MODEL=<model> TT_GEN_FLEX_SAMPLES=1 cargo xtask bench --device 1 --filter
silicon_bench_models::traced_generation`; release, card 1, every tile, 2 warm-ups, 9 samples, burn-flex in the same
run.

| decode step, ms, 32 / 128 / 256-token prompt | before (`1792189445`) | B1 | burn-flex, same run |
|---|--:|--:|--:|
| SmolLM2-1.7B | 42.97 / 45.41 / 48.59 | **41.83 / 44.25 / 47.40** (`1792209913`) | 301.7 / 310.8 / 329.4 |
| SmolLM2-135M | 19.94 / 20.42 / 21.11 | 20.03 / 20.55 / 21.19 (`1792210419`) | 26.5 / 28.4 / 31.5 |

**The prefetch probe** (`silicon_probe_prefetch_overlap`, card 0, `1792210511` and after):
- Weight tiles streamed by units 64-119 into the top of `DATA` land bit for bit, and adds on units 0-63 under them
  keep their bits.
- Eagerly, overlap cannot be seen. An add takes ~290 us of host time to queue on 64 units, and the stream's ~300 us is
  its own queueing on 56 units, so "both" is the sum (2.53 + 0.30 -> 2.84 ms).
- In a replay, a captured barrier involves every unit, so the prefetch first needs replay-fixed-cost P1.
- **The ceiling:** L1 above a product's `A` is ~630 KB a unit, ~75 MB across the card, against 273 MB of weights a
  layer. That caps the prefetch at ~4.4 ms a step.

## RMSNorm's statistic over a decode row's row groups (2026-10-16, cards 0 and 1)

[decode-small-ops.md](../plans/decode-small-ops.md) S3, "next if it pays". Bench: `TT_TILES=all TT_GEN_MODEL=<model>
TT_GEN_FLEX_SAMPLES=1 cargo xtask bench --device 1 --filter silicon_bench_models::traced_generation`, release, card 1,
2 warmups, 9 samples, burn-flex in the same run; profiles the same with `--profile` (1.7B, `Step(64)`, unit 0).

**What changed.** A decode row is tensor row 0 of each tile: 4 of the SFPU's 32 `Dst` loads a tile. The statistic's
programs (sum of squares, fold, `sqrt(sum / n + eps)`) run only the row groups that hold a tile row's rows
(`sfpu::reduce::row_offsets`); a ragged last tile row has programs of its own. Bit for bit as before (no value moves
between rows); the output's other rows were already claimed `Pad::Undefined`.

| A norm's statistic, `Step(64)`, 1.7B (us, 49 a step) | span | kernels | data |
|---|--:|--:|--:|
| before (`1792188326`) | 35.90 | 14.95 | 19.16 |
| row-limited (`1792190118`) | **26.52** | **5.58** | 19.15 |
| row-limited, rows read thin too (`1792192008`, reverted) | 28.32 | 5.58 | 21.01 |

The `DIV` after it is unchanged: 30.9 us on 64 units (kernels 8.90, data 15.7).

| decode step, ms, 32 / 128 / 256-token prompt | before (`1792186180`) | row-limited | burn-flex, same run |
|---|--:|--:|--:|
| SmolLM2-1.7B-Instruct | 43.42 / 45.85 / 49.00 | **42.97 / 45.41 / 48.59** (`1792189445`) | 303.4 / 311.2 / 329.3 |
| SmolLM2-135M | 20.20 / 20.66 / 21.34 | **19.94 / 20.42 / 21.11** (`1792189939`) | 28.0 / 29.9 / 32.6 |

**The thin read that lost.** `READ_RUN` with a row count read each tile through its used rows (`1024 + 64 r` bytes, as
a thin matmul `GATHER` reads `A`): ~74% fewer bytes, the same 64 entries. The read got slower (19.15 -> 21.01 us) and
the step did not move (1.7B 42.99 / 45.40 / 48.57, `1792191435`; 135M 20.06 / 20.54 / 21.24, `1792191823`). A row
read by one unit costs per read, not per byte, as decode-matmul-bandwidth P1 found for two face-row reads a tile. It
passed the mover gates in both read modes and `--smoke` (1,104 of 1,104, `1792192846`) and was reverted.

**Gates:** unit `sfpu::norm` (rows 1-49, NaN padding, control `a_row_group_too_few_differs`); `step196_row_limited_rms`
(ttsim; card 0 `1792194490`); `step195` and `step193` on ttsim and card 0 (`1792189325`, `1792194490`); `step187` on
ttsim; Burn conformance (`norm`, `rms`) plain and `--fusion`, no unexpected result.

## Decode fusers: sigmoid, SiLU and RMSNorm (2026-10-16, cards 0 and 1)

[decode-small-ops.md](../plans/decode-small-ops.md) F0 (decode on `Tt`), S2b, S3;
[kernel-fusion.md](../plans/kernel-fusion.md) Phases 2 and 4.

**Gates:**
- On card 0: `step194_fused_sigmoid_silu` 4 of 4 (BF16 included, `1792186176`), `step195_fused_rms_norm` 3 of 3
  (`1792186165`), `step193_fusion_decode_trace` 4 of 4 (SmolLM2-135M on `Tt` included, `1792186169`).
- On ttsim: `step12` (`e2e`, `add_relu` still fused) and the Llama gates (`step187`, `step191`, `step192`).
- Burn conformance on ttsim, plain and `--fusion`: no unexpected result.

**Bench:** `TT_TILES=all TT_GEN_MODEL=<model> TT_GEN_FLEX_SAMPLES=1 cargo xtask bench --device 1 --filter
silicon_bench_models::traced_generation`, release, card 1, 2 warmups, 9 samples, burn-flex in the same run. The model
now runs on `Tt` (`TT_GEN_FUSION=0` gives `TtBackend`).

| decode step, ms, 32 / 128 / 256-token prompt | before (`TtBackend`) | `Tt`, three fusers | burn-flex, same run |
|---|--:|--:|--:|
| SmolLM2-1.7B-Instruct | 45.23 / 47.71 / 50.78 (`1792174499`) | **43.42 / 45.85 / 49.00** (`1792186180`) | 304.2 / 316.0 / 335.8 |
| SmolLM2-135M | 22.46 / 22.95 / 23.63 (`1792174412`) | **20.20 / 20.66 / 21.34** (`1792186583`) | 26.9 / 30.7 / 31.6 |
| 1.7B build (capture and warm-up), s | 70.2 | 69.3 | |

**What changed.**
- **Sigmoid.** burn-fusion 0.21 sends `sigmoid` as Burn's 8-op composition. The chain fuser runs `TtBackend`'s
  sigmoid for it, which removes F0's +1.5 / +2.2 ms. A decode step on `Tt` gives `TtBackend`'s logits bit for bit.
- **`silu(gate) * up`:** one SFPU pass (`SILU_MUL`), 49 us, where the sigmoid and two muls were three jobs.
- **RMSNorm** up to its scale: the statistic `sqrt(mean(x x) + eps)` as one reduction kernel (`sfpu::norm`), then
  the existing `DIV`. That is two jobs a norm where there were five.

**The version that lost** (run `1792184225`, profile `1792184874`):
- The norm was one kernel that also divided, the row unpacked a second time.
- It took **640 us a norm** (95% kernel), 31.3 of a 70.8 ms 1.7B step (71.53 / 73.94 / 77.14 ms; 135M 28.90-30.07).
- `DIV`'s body does not fit the 32-entry replay buffer, so each of its ~1,000 instructions a tile is pushed from the
  role program, and one unit pushed them for all 64 tiles. The separate `DIV` spreads those tiles over the units.
- The lesson: fuse a reduction's sequential part, and leave its element-wise map a parallel op.

## Block views: RoPE and SiLU without repacks or copies (2026-10-16, cards 0 and 1)

[decode-small-ops.md](../plans/decode-small-ops.md) S1 and S2a; gate `step192_block_views` (ttsim; card 0 runs
`1792168902`, `1792174393`); `step188` on card 0 (`1792170465`); `--smoke` 1,096 of 1,096 (`1792175641`); Burn
conformance on ttsim with no unexpected result; the MNIST training regression (`step12`, `e2e`) matches its golden. Bench: `TT_TILES=all TT_GEN_MODEL=<model> TT_GEN_FLEX_SAMPLES=1 cargo xtask bench
--device 1 --filter silicon_bench_models::traced_generation`, release, card 1, 2 warmups, 9 samples, burn-flex in the
same run.

**What changed.**

- **Element-wise ops on a block view.** A view that is a block of its buffer (`Strided::as_block`: its elements fill
  a tile-aligned block one to one) runs the op on that block where it lies (`Session::eltwise_blocks`; each operand's
  `READ_RUN` counts over the output's grid at its parent's row stride). Examples: SiLU's `gate` and `up` columns of
  the fused product, and `q`'s heads along a projection's row. No firmware change.
- **RoPE in the projection's layout.** In decode, `q` is `[1, h, 1, d]` with every head in row 0 of the projection.
  Laying it out as `[h, d]` moves data from one row to many, which no tile move does, so both `cat` and `x cos` were
  element repacks (~38 us each on 2 units).
  - Now the table is laid over the heads by tile copies (`Strided::layout_moves`), kept beside it
    (`TtTensor::derived`), once a step.
  - The halves' `cat` is a permutation of the projection's tile columns (`region::cat_in_layout`).
  - `head_dim` 32 (halves of 16 columns) keeps the repack.
- **A regrouped GQA `q` is made plain once** (`operand_blocks`). Without this, SmolLM2-135M's step went 23 -> **40 ms**
  (profile `1792172763`): `q` in the projection's layout, regrouped `[kv, groups, d]`, is not a block, so attention
  fell to Burn's composition and repacked the keys a group, 214 us.

**Result.**

| Step (ms), 32 / 128 / 256-token prompt | before (`1792161623`, `1792162031`) | after (`1792174499`, `1792174412`) |
|---|--:|--:|
| SmolLM2-1.7B | 47.90 / 50.37 / 53.53 | **45.23 / 47.71 / 50.78** (22.1 / 21.0 / 19.7 tokens/s) |
| SmolLM2-135M | 22.91 / 23.35 / 23.98 | **22.46 / 22.95 / 23.63** (burn-flex 29.5 / 31.0 at 128 / 256) |

**Where the 1.7B step changed** (`Step(64)`, profile `1792171171` against `1792162257`):

| Part, ms a step | before | after |
|---|--:|--:|
| `float_cat` (RoPE's halves) | 1.83 | 0.30 |
| `float_mul` (RoPE, SiLU, the norm's `gamma`) | 3.75 | 1.91 |
| `sigmoid` | 1.28 | 1.08 |
| `attention/qk` (q's row gather became a fill-pad) | 1.75 | 1.55 |
| `float_scatter_nd` (the key write now repacks, as the value's did) | 3.89 | 4.80 |

- **The key write's new repack, 0.9 ms.** The key reaches the cache write in the projection's layout and repacks
  there (38 us a layer), as the value already did. Both repacks are S1d, a `SCATTER_ROWS` that reads its rows from a
  strided view.
- **The 135M step gained only 0.4 ms.** Its regrouped `q` is now made plain once (6 us), and its key write repacks
  too.

**F0: the same step on `Fusion<TtBackend>`** (`TT_GEN_FUSION=1`; gate `step193_fusion_decode_trace`, card 0
`1792174399`):

| ms | `TtBackend` | `Fusion<TtBackend>` |
|---|--:|--:|
| 1.7B decode step, 32 / 128 / 256 | 45.23 / 47.71 / 50.78 | 47.49 / 49.90 / 53.08 (`1792172342`) |
| 135M decode step | 22.46 / 22.95 / 23.63 | 24.00 / 24.52 / 25.15 (`1792174906`) |
| 1.7B build (capture and warm-up) | 70,158 | 74,080 |
| 135M build | 16,397 | 17,962 |

- **Capture works, and replays run no Burn op.** But burn-fusion 0.21 does not forward activations, so its
  `sigmoid` is Burn's default composition: eight ops (cast, `x -1`, `exp`, `+1`, `log`, `x -1`, `exp`, cast) where
  burn-tt has one.
- **The tax:** +1.5 / +2.2 ms a step, and logits not always bit for bit `TtBackend`'s (one case of four differed by
  7.9e-6).
- **What burn-fusion fuses today in a decode step: nothing.** The only fuser starts on `Add` or relu.

## Decode matmuls read only the activation rows they use (2026-10-16, cards 0 and 1)

Roadmap Track 1, SmolLM2-1.7B "Next, by that profile": the matmuls' bandwidth
([decode-matmul-bandwidth.md](../plans/decode-matmul-bandwidth.md), P0-P2). They were 59.5% of a 1.7B decode step,
33.8 of 56.8 ms (`1792110380`).

**What the jobs read** (P0, host only, `tt-kernels` test `decode_products_jobs_rounds_and_bytes`, 120 units). Every
`m = 1` product read whole `A` tiles (4,160 bytes) for one row of 32: a step's products moved 12.06 GB, 5.11 GB of
it `A`'s padding, ~357 GB/s at the profile's 33.8 ms. `gate`/`up` (256 output columns, two a job, 128 jobs on 120
units) left 8 units running two jobs while 112 ran one. The L1 plan caps every K-whole `m = 1` block at two columns
of `K` = 2048 (64 `A` + 128 `B` slots of the arena's ~215), so a unit cannot take a third in one job.

**What changed:**
- **P1, thin `A` reads.** `GATHER`'s free flag bits 3-7 carry `r` (1..=16, `record::gather_rows`), set by
  `BlockPlan` on the gathers of `op(A)`'s last tile row when `m % 32` is 1..=16 and `A` is not transposed
  (`tensor::thin_rows`; `Session::set_thin_matmul_reads`, on by default). Each such tile is one read of its datums
  through row `r - 1` of face 1, `1024 + 64 r` bytes (1,088 for `m = 1`, was 4,160). The rest of the L1 slot keeps
  what it held; with face 0's rows past `r` (read from `A`'s padding) it reaches only the output's padding rows,
  whose claim is then `Pad::Undefined`. No zeroing is needed: consumers that need zero padding already refill it
  (`Session::meet`). Packed BF16 reads a tile with at most 16 valid rows the same way, from its metadata, with no
  flag (its fill already zeroes the rest).
- **P2, the last round spread.** For a one-tile-row product with `K` whole and unpipelined, `tensor::thin_runs`
  deals whole rounds of the plan's width, then the remainder as even runs over the units: `gate`/`up` 8 units with
  2 + 2 columns -> 16 with 2 + 1 (the busiest unit 4 -> 3 columns, `ceil(256 / 120)`); the tied head 14 -> 13.

**Read shape, measured (P1's decision).** Two 64-byte face-row reads a tile (2x the entries, 1/32 the bytes) first:
the 1.7B step 57.21 / 59.64 / 62.81 -> 55.69 / 58.13 / 61.30 ms (`1792114379`), and the 135M step **slower**, 24.40 /
24.84 / 25.53 -> 25.43 / 25.81 / 26.51 (`1792114782`). One prefix read a tile instead (the same entry count as a whole
tile, ~27% of the bytes): 51.15 / 53.55 / 56.71 and 24.07 / 24.51 / 25.21 (`1792115180`, `1792115571`). The B
mover's cost is per record-expanded entry as much as per byte, so a read shape must not add entries.

**Measured** (card 1, release, every tile, default spread, burn-flex in the same run):

| Decode step, 32 / 128 / 256-token prompt | Before (`1792109477`, `1792110130`) | P1 (`1792115180`, `1792115571`) | P1 + P2 (`1792118484`, `1792118871`) | burn-flex (same run as P1 + P2) |
|---|---|---|---|---|
| SmolLM2-1.7B | 57.21 / 59.64 / 62.81 ms | 51.15 / 53.55 / 56.71 | **49.78 / 52.27 / 55.44** (20.1 / 19.1 / 18.0 tokens/s) | 294.4 / 307.8 / 334.7 |
| SmolLM2-135M | 24.40 / 24.84 / 25.53 | 24.07 / 24.51 / 25.21 | **24.14 / 24.57 / 25.25** | 26.05 / 28.24 / 30.44 |

- A 256-token prompt plus 64 tokens: 1.7B 4,292 -> **3,830 ms** (burn-flex 31,206); prefill unchanged (267 / 656 /
  771 tokens/s).
- Decode reads the 1.7B weights at 123-138 GB/s, was 109-120.
- The 1.7B traces hold 155.6 KB of the 240 KB program cache (155.5 before: P2's one-column jobs add one program).

**Where the matmul time went** (the profile mode, `Step(64)`, unit 0, 1.7B; `1792110380` before, `1792115848` P1,
`1792119119` P1 + P2):

| Products | Before | P1 | P1 + P2 |
|---|--:|--:|--:|
| `q`/`k`/`v`/`o` (64 units), us each | 94.9 | 73.8 | 73.8 |
| `down` (K-blocked, 64 units) | 343.3 | 267.9 | 267.8 |
| `gate`/`up` and the head (120 units) | 335.6 | 286.4 | 261.3 (`gate`/`up` 241) |
| all 169 products, ms a step | 33.79 | 27.55 | **26.32** |
| the step, ms | 56.81 | 50.61 | 49.38 |

**What bounds them now: entries, not bytes.** A `q` job now reads 64 `B` tiles and 64 thin `A` tiles (337 KB) in
~60 us of data on each of 64 units, ~290 GB/s card-wide, below the ~358 GB/s whole tiles reached. Its 128
record-expanded entries at ~0.45 us each are most of that time. Fewer entries a tile, or more units on the narrow
products (P3's fused `q|k|v` on 120 units; split-K, P4), are what is left; not fewer bytes.

**P3, fused projections** (`Llama::fused`, gate `step191_fused_projections`, card 0 `1792161551`): each layer's
`q|k|v` and `gate|up` one product each, outputs split by column views, bit for bit the unfused model. The 1.7B step
**47.90 / 50.37 / 53.53 ms** (`1792161623`, burn-flex 291.5 / 300.4 / 328.5), the 135M step **22.91 / 23.35 / 23.98**
(`1792162031`, burn-flex 26.43 / 28.21 / 30.65). Profile `1792162257` (`Step(64)`, 47.46 ms):

- **Products 24.01 ms.**
  - `gate|up` is 426 us, two of 241 before: ~490 GB/s card-wide, GDDR's ceiling.
  - `qkv` is 180 us, three of 74 before. 37 us of it is unit 0 at the barrier: an uneven last round of 2 and 1
    columns.
  - `o` (74 us) and `down` (268 us) run on 64 units at ~300 GB/s. 64 x 8 requests in flight do not cover GDDR's
    latency.
- **Everything else 23.4 ms**, now as large as the products:
  - the RMSNorm's composed ops (`mean_dim`, `div`, `sqrt`, `powi`, scalars), ~5 ms;
  - RoPE's repacks, ~3.6 ms;
  - the cache write, ~5 ms;
  - the softmax, ~2 ms.

Gates: `step190_thin_matmul` (ttsim; card 0 runs `1792113956`, `1792115747`, fast-reads `1792116118`), with
`step77`'s BF16 matmuls (ragged rows, NaN-poisoned padding) for the packed path; `step185`/`step187`/`step188` bit for
bit (`1792115747`, `1792118267`); `--smoke` 1,083 of 1,083 (`1792116807`); Burn conformance on ttsim with no unexpected
result (`1792113791`; after P2 `1792117214`).

## A cache write of one row a head (2026-10-16, cards 0 and 1)

Roadmap Track 2 item 6, step 6 (`decode-length-cost.md` B3), from the 1.7B profile below: a fixed cache's write
expanded the new key and value to the cache's whole shape and `mask_where`d them in, 24% of a decode step.

**The change:**
- **`SCATTER_ROWS`**, NC's first record that reads indices (`tt_isa::dm::record`): B stages an index tile and the
  values' tile row; NC writes each staged row's two 64-byte face-rows a tile column at the row the index names, a
  repeated index's last row only, `DOMAIN` before any write.
- **`Session::scatter_rows`**: a tile copy, then the scatter, one job per group of tile columns (every index of those
  columns, in order, so a spread never splits a row's writers).
- **burn-tt**: `scatter_nd` with `Assign` is that op; with one coordinate the indices are read where they are, with
  no flat-row ops.
- **`FixedCache`**: a step writes `rows = h max_len + position` (`[kv_heads, 1]`, built once a step) with
  `scatter_nd`. Prefill chunks keep `mask_where` (their padded tail may run past the cache).

Gates: `step189_scatter_rows` (ttsim; card 0 run `1792109189`, with `step185`, `step187`, `step188` and the mover gates,
30 of 30), Burn's `scatter`/`gather_scatter_nd`/`select` conformance plain and under `Fusion` (186 of 186, ttsim).

**Measured** (card 1, release, every tile, default spread, burn-flex in the same run):

| | Decode step, 32 / 128 / 256-token prompt | Prompt + 64 tokens | burn-flex step |
|---|---|---|---|
| SmolLM2-1.7B before (`1792082994`) | 73.65 / 76.08 / 79.30 ms | 4,789 / 4,992 / 5,332 ms | |
| SmolLM2-1.7B now (`1792109477`) | **57.21 / 59.64 / 62.81** (17.5 / 16.8 / 15.9 tokens/s) | **3,757 / 3,957 / 4,292** | 298.3 / 307.5 / 333.9 |
| SmolLM2-135M before (`1792089070`) | 25.81 / 26.28 / 26.99 | | |
| SmolLM2-135M now (`1792110130`) | **24.40 / 24.84 / 25.53** | 1,580 / 1,615 / 1,682 | 26.66 / 28.97 / 31.12 |

- Prefill is unchanged (1.7B: 267 / 658 / 774 tokens/s).
- Decode reads the 1.7B weights at 109-120 GB/s, was 86-93.
- The 1.7B traces hold 1.10 GB of streams, was 1.54; the driver's build is 67.2 s, was 76.2.

**Where the cache write's time is now** (the profile mode, `Step(64)`, run `1792110380`; 56.8 ms a replay, every op
covered). 48 writes a step (24 layers, keys and values), 5.03 ms in all:

| Part | ops | ms | us each |
|---|--:|--:|--:|
| the row scatter (`SCATTER_ROWS`) | 48 | 1.30 | 27.0 |
| the scatter's tile copy of the cache | 48 | 1.13 | 23.5 |
| the trace's write-back (`copy_into`) | 48 | 1.13 | 23.5 |
| a repack of the value (once a layer) | 24 | 0.91 | 38.1 |
| a block copy of the value | 48 | 0.56 | 11.7 |

The two whole-cache copies (2.26 ms) are what writing in place would remove (the plan's Phase 2, capture-level
donation); the value's repack and block copy (1.47 ms) are its layout after RoPE and the head split, not the cache.
`float_matmul` is now 59.5% of the step (33.8 ms).

## Where a SmolLM2-1.7B step goes: the profile mode (2026-10-15, cards 0 and 1)

The first profile from the profile mode (`TT_PROFILE`, `cargo xtask bench --profile`; `burn_tt::profile`,
`tt_kernels::profile_mode`, gate `silicon_profile_mode`).

**The run:** `TT_TILES=all TT_GEN_MODEL=smollm2-1.7b-instruct TT_GEN_FLEX_SAMPLES=1 cargo xtask bench --profile
--device 1 --filter silicon_bench_models::traced_generation`, run `1792091115`. The profile is in
`target/silicon/profile/1792091115/`.

**Coverage:**

- Every decode step trace is fully covered (1,438-1,443 ops each, 341-693 replays, none overflowed).
- **Their per-op medians sum to 73.5 / 74.5 / 76.0 / 79.3 ms a replay**, against 73.7-79.3 ms for the unprofiled
  decode step (`1792082994`): the attribution accounts for the whole step.
- Prefill chunks get 11-12 replays each, so 32-46% of their ~1,500 ops are covered. The 24 layers are alike, so the
  covered stretch stands for the rest by label.
- Profiled replays are host-timed at ~208 ms against 74: the mode's per-replay window writes and syncs, not the device.

**A decode step (`Step(64)`, 73.5 ms):**

| Label | Ops | ms | Share |
|---|--:|--:|--:|
| `float_matmul` | 169 | 33.7 | 46% |
| `float_expand` | 72 | **17.9** | **24%** |
| `float_mul` | 241 | 3.6 | 5% |
| `float_mask_where` | 48 | 2.2 | 3% |
| `attention/softmax` | 168 | 2.0 | 3% |
| `float_cat` | 48 | 1.8 | 2% |
| `attention/qk` + `/pv` | 144 | 3.1 | 4% |
| the rest (norms, SiLU, adds, masks) | ~550 | ~9 | ~13% |

- **The cache write is a quarter of a decode step.** `FixedCache`'s masked write expands the new key and value to
  the cache's whole shape: a 120-unit row gather of ~355 us, twice a layer, since this model keeps all 32
  key/value heads. SmolLM2-135M has 3, so it never showed. A row write at a device-held position (the deferred B3
  of `decode-length-cost.md`) would replace both the expand and the `mask_where`.
- **The matmuls read the 6.85 GB of weights at ~200 GB/s**, 40% of the 512 peak. The tied output head (402 MB) takes
  1.44 ms, 279 GB/s.
- **The rest is ~1,200 small ops at 6-50 us each:** per-op cost and fusion territory.

**A 256-token prefill chunk** (32% covered): matmuls 62%, the softmax 10%, attention's mask 6.5%, `float_mul` 5%,
`float_select` 5%.

**The cost of stamping** (card 1, the 135M bench; runs `1792088908` and `1792089070`):

| | Decode step, 32 / 128 / 256-token prompt |
|---|---|
| Stamping compiled in, profiling off | 26.28 / 26.74 / 27.45 ms |
| Stamping compiled out (now the default) | **25.81 / 26.28 / 26.99 ms** |

About 0.46 ms a step, 1.8%. So stamping is now the firmware's `profile` feature: the default images carry none of
it, and profiling without it is refused by name.

## SmolLM2-1.7B-Instruct on the card (2026-10-15, card 1)

The first model past 1B parameters (Track 1; `tt_models::llama::SMOLLM2_1_7B_INSTRUCT`). Setup:

- **Model:** 24 layers, width 2048, 32 heads, F32 (6.85 GB of weights).
- **Bench:** `silicon_bench_models::traced_generation` with `TT_GEN_MODEL=smollm2-1.7b-instruct`, card 1, release,
  every tile, default spread, run `1792082994`.
- **Shapes:** chunks of 64/128/256, step lengths 64/128/256/512, a 512-position cache.
- **burn-flex:** the eager growing cache in the same run, 3 samples (`TT_GEN_FLEX_SAMPLES=3`), as a sample takes 23
  to 31 s.
- **Validated:** the traced 32-token prompt bit for bit against the eager schedule on the card. Gate
  `step188_smollm2_1_7b` on card 0 (`1792080640`) agrees with Flex and with the eager generation.

| Prompt + 64 tokens | Prefill (first token) | Decode step | Total | burn-flex: prefill / step / total |
|---|---|---|---|---|
| 32 | **267 tokens/s** (120.0 ms) | **73.65 ms, 13.6 tokens/s** | 4,789 ms | 7.6 tokens/s / 293.4 ms / 22,722 ms |
| 128 | **657** (194.9) | **76.08, 13.1** | 4,992 | 23.0 / 312.0 / 25,260 |
| 256 | **771** (331.9) | **79.30, 12.6** | 5,332 | 24.7 / 332.6 / 31,304 |

- **Traced against burn-flex:** decode 4.0-4.2x faster, prefill 29-35x, a whole generation 4.7-5.9x.
- **Decode is not memory-bound yet.**
  - A step reads every F32 weight once, 6.85 GB. Done in 73.7-79.3 ms, that is **86-93 GB/s, 17-18% of the card's
    512 GB/s** (469 measured for reads across 120 tiles).
  - The floor at that bandwidth is ~14.6 ms a token, so decode is about 5x from it. Its m = 1 matmuls (one job an
    output tile, the `K` tiles a serial chain on one unit) are the first suspect; profile before choosing.
- **Footprint:**
  - the seven traces hold 155 KB of the 240 KB program cache on the fullest unit (SmolLM2-135M: 89 KB) and 1.54 GB
    of streams;
  - the driver's build is 76.2 s.
- **Untraced** (gate run `1792080640`, card 0, every tile): a decoded token is about 530 ms; the first prompt pass is
  36.5 s, cold (the weights are uploaded on first use).
- **The simulator** (one tile) cannot capture a 2048-wide step: "a list's programs do not fit an empty program cache"
  on an element-wise op. The traced gate is silicon only until traces are segmented.

## Decode length and trace size (2026-10-15, cards 0 and 1)

Plan and full tables: [decode-length-cost.md](../completed-plans/decode-length-cost.md). The setup:

- **Benches:** `traced_generation` and `smollm2_replay_profile`, SmolLM2-135M F32 30 layers, card 1, release, every
  tile, default spread, burn-flex in the same run.
- **Gates:** card 0, bit for bit against the growing cache.

**Trace size** (`traced_generation`):

- **Before, `1792074320`:** the step and 64/128/256 chunks took 3,754 MB of streams against 545 MB of entries. Each
  `PAIR_CALL` section had its own slot-rounded region, ~4,360 bytes for ~810 of entries.
- **After, `1792076190`:** one region a unit, sections 64-byte aligned. 563 MB (1.0x the entries), and the driver's
  build went from 44.9 s to 18.8 s. Replay times are unchanged.

**Where a step's length cost went** (one layer, unit 0's device time):

- **64 against 512 positions** (`1792076366`, `1792076371`): +223 us. Of that, `repeat_kv`'s `cat` was +79 (and
  70 us at 64 already), attention's `qk`/`pv`/softmax/mask +77, the `k`/`v` expand +27, the masks +30 (once a
  step), `mask_where` +8 and the write-back copies +3.
- **After the grouped-query fold** (`1792078810`, 512 positions): 1,855.6 -> 1,682.3 us. `cat` went 149 -> 19,
  `qk` 92 -> 69 and `pv` 86 -> 66.

**Generation over a 512-position cache** (prompt + 64 tokens):

| | Decode step, 32 / 128 / 256-token prompt | Total | Run |
|---|---|---|---|
| Before | 32.29 / 32.29 / 32.29 ms | 2,077 / 2,087 / 2,111 ms | `1792076190` |
| Grouped-query fold | 27.32 / 27.31 / 27.32 | 1,761 / 1,770 / 1,795 | `1792078651` |
| Fold and steps of 64/128/256/512 | **26.13 / 26.60 / 27.31** | **1,688 / 1,726 / 1,795** | `1792078489` |
| burn-flex | 26.75 / 29.10 / 31.57 | 1,866 / 2,345 / 3,015 | `1792078489` |

With the step lengths: seven traces (4 steps and 3 chunks), 517 MB of streams, 89.3 KB of the 240 KB program cache on
the fullest unit. The first token is 35.1 / 46.0 / 69.3 ms (was 38.8 / 49.6 / 73.0). What still grows with the
cache is its write (`expand` and `mask_where`, whole-cache): 1.2 ms a step between 64 and 512 positions.

## Traced generation, end to end (2026-10-15, cards 0 and 1)

Roadmap Track 2 item 6, next steps 2 and 3. The setup:

- **What runs:** `tt_models::generate::TracedGenerator` on SmolLM2-135M (F32, 30 layers): the step and
  chunks of 64, 128 and 256 captured as traces over one cache (`burn_tt::SharedStateTraces`). Token ids
  go in (the lookup runs inside the trace) and logits come out.
- **Bench:** `silicon_bench_models::traced_generation`, card 1, release, every tile (`TT_TILES=all`),
  default spread, 2 warm-ups and 9 samples. burn-flex's eager growing-cache `Llama::generate` ran in the
  same run.
- **Validation:** a 32-token prompt's first 4 tokens bit for bit against the eager schedule on the card
  (`generate_fixed`), and every timed run's tokens equal. Gate `step187_traced_generation` on card 0
  (run `1792071747`): SmolLM2 at 30 layers, synthetic Llama shapes and a GPT-2-style decoder, bit for bit.

**The four traces fit together.** Program cache held on the fullest unit, of 245,760 bytes:

- **80,980 bytes** at 512 positions (`1792072109`);
- **63,980** at 320 (`1792072312`).

Per trace, the most on one unit at 512: the step 35,764, `Chunk(64)` 40,680, `Chunk(128)` 45,572,
`Chunk(256)` 59,344 (programs shared between traces are held once). **The streams are large:** 3.75 GB
of GDDR across 120 units at 512 (the step 0.45 GB, `Chunk(256)` 1.46 GB); 3.42 GB at 320. That scales
with ops times units, so it is the cost to watch for a larger model. The build (eager warm-up and four
captures) took 44.6 s at 512 and 40.1 s at 320.

| Cache | Prompt + 64 tokens | First token | Decode step | Total | burn-flex first / step / total | Run |
|---|---|--:|--:|--:|---|---|
| 512 | 32 | 38.7 ms | 32.28 ms | 2,075 ms | 180.1 / 26.57 / 1,856 | `1792072109` |
| 512 | 128 | 49.6 | 32.28 | 2,086 | 513.0 / 28.63 / 2,322 | `1792072109` |
| 512 | 256 | 72.9 | 32.28 | 2,110 | 1,028.1 / 30.82 / 2,985 | `1792072109` |
| 320 | 32 | 35.0 | 28.49 | 1,834 | 175.1 / 26.34 / 1,837 | `1792072312` |
| 320 | 128 | 42.7 | 28.49 | 1,841 | 506.5 / 28.93 / 2,331 | `1792072312` |
| 320 | 256 | 61.7 | 28.49 | 1,860 | 1,019.6 / 30.99 / 2,996 | `1792072312` |

(The first run, `1792071816`, timed with chunks of 64 and 256 only; a 128-token prompt then took the 256
chunk, 73.0 ms.)

**What it says.** A traced step attends to every position of its cache, so it costs the cache's length,
not the context: 26.46 ms over 64 positions (`1792036613`), 28.49 over 320, 32.28 over 512, against
burn-flex's 26.3 to 31.0, whose growing cache attends only to what it holds. The prefill is where the
traces win (5x to 17x on the first token). Decode is at parity at best, and behind once the cache is
longer than the context. Bounding the attended range by the position (step traces per attended
length) is the next step for any LLM, not only this one.

## A traced replay's fixed cost (2026-10-15, cards 0 and 1)

Roadmap Track 2 item 6, next step 1; plan and full tables: [replay-fixed-cost.md](../plans/replay-fixed-cost.md).
Release, every tile (`TT_TILES=all`), default spread, burn-flex in the same run; every replay checked bit for bit against
the eager step by the bench.

**What a replay was** (`smollm2_decode_traced`, card 1, `1792034179`): input write 0.07 ms, replay 36.0, logits read 1.0;
2,418 captured ops, each ending in a barrier all 120 units reach, a median op on 6 units.

**Not the barrier** (`silicon_probe_trace_replay::replay_per_op_cost`, card 0, `1792034180`): a captured chain's cost an op
is 6.2 us on one tile and 6.8 us on 120 for an op on one unit; a `[1, 576] @ [576, 576]` is 21.0 us on 120 tiles.

**Op by op** (`smollm2_replay_profile`, one layer, card 1, `1792034647`; unit 0's mover stamps in 128 KiB at the top of the
program cache, `Session::set_profile_buffer`, an op's span from one of unit 0's `BARRIER` ends to the next, the spans
checked to account for 36.4 of the 36.0 ms): 1,123 us a layer, the head 2,329 us, the start 174 us. A tiny op is 5.7 us on
unit 0, 3.5 of it control. What the layer spent most on: a 16 us `fill-pad` before each of its 7 projections (an m = 1
input's padding rows), two 34 us zero-row `metadata` writes in attention, 8 ops to expand the cache write's operands,
RoPE's `neg` of a slice; the head read its `B` through `READ_TRANSPOSED` (about 10 us a tile, the transpose in software).

| Change (each adds to the one before) | Ops | Replay | Read | Step | burn-flex | Run |
|---|--:|--:|--:|--:|--:|---|
| baseline | 2,418 | 36.0 ms | 1.00 | 37.2 ms | 26.6 | `1792034179` |
| a matmul refills padding only along `K` | 2,207 | 32.6 | 0.96 | 33.8 | 25.9 | `1792035075` |
| one zero row per width kept; a buffer read transposed as `B` transposed once and kept | 2,147 | 28.7 | 1.19 | 30.0 | 26.4 | `1792035466` |
| the model's position mask expanded once a step; RoPE's sign in its `sin` table | 1,911 | 25.8 | 0.99 | 27.0 | 26.3 | `1792035579` |
| a tensor mostly of padding downloads untilized by the card, a unit per 96 tiles | 1,911 | 25.8 | 0.47 | **26.46** | 26.42 | `1792036613` |

- **The download's unit count** (the logits, `[1, 49152]`): untilized by the card on 4 / 8 / 16 / 24 / 32 / 48 / 120 units
  0.66 / 0.55 / 0.46 / 0.49 / 0.51 / 0.60 / 0.95 ms; as tiles detilized by the host on 4 / 16 / 120 units 0.68 / 0.77 / 0.99
  (`1792036386`-`1792036493`). Each unit is a list and a drain for the host.
- **The other traces** (card 1): prefill chunks of 1 / 32 / 64 / 128 / 256 tokens 32.7 / 34.8 / 38.7 / 49.7 / 73.1 ms
  (were 43.8 / 40.2 / 44.3 / 55.2 / 79.0; `1792036724`; burn-flex 184 / 540 / 1,098 ms for 32 / 128 / 256 tokens); ViT-Tiny
  11.6 ms (was 12.6; `1792036850`, burn-flex 112); ResNet-18 17.2 ms (was 17.6; `1792036818`, burn-flex 164). A first
  ViT run at 15.2 ms (`1792036810`) was taken with the conformance suite loading the host; two reruns gave 11.6.
- **Left** (by share): the cache write's `k`/`v` expands and `mask_where`s, ~70 us a layer; the RMSNorm's broadcast `div`,
  27 us twice a layer; the replay's start, ~170 us (a `CALL` must be a list of its own, so a unit's catch-up is a second
  enqueue); ~3.5 us of control an op.

## SmolLM2 prefill chunk size (2026-10-15, cards 0 and 1)

`Llama::step_chunk` (`crates/tt-models/src/llama.rs`) generalises the fixed-shape decode step to `c` prompt tokens from a
start position, both tensors:
- **cache write:** the chunk's keys and values are laid over the cache by a row gather (a one-hot matmul would
  truncate to TF32), then `mask_where` keeps positions `start..start + c`;
- **attention:** each query masks the positions after its own;
- **rotary rows:** selected per token, clamped for padding tokens past the cache;
- **logits:** taken at a `last` row index tensor.

A prompt is its chunks in order, the last padded. A padding token's cache row is overwritten by the decode step at that
position before anything reads it.

Gate `step185_fixed_decode` (ttsim at one layer; card 0 at 30, `1792032276`):
- the chunked prefill's logits, and the decode step's after it, equal the eager prefill's bit for bit, at prompt
  lengths 1, 3, 4, 5 with chunks of 4, and 5 with a chunk of 8;
- one captured chunk replays at the next start bit for bit;
- shifting every token's position by one, watched failing, breaks it.

`smollm2_prefill_chunk_sweep`: card 1, release, the 512-position cache, each chunk captured once with the cache as
written-back state, one replay checked bit for bit against the eager chunk, two warm-up and nine timed replays.
burn-flex's eager prefill was timed in the same run.

| Chunk `c` | every tile, default spread (`1792031467`) | a token | one tile (`1792032275`) | `work:64:4` (`1792032323`) |
|--:|--:|--:|--:|--:|
| 1 | 43.8 ms | 43.8 ms | 1,134.8 ms | 227.8 ms |
| 32 | 40.2 ms | 1.26 ms | does not capture | 248.6 ms |
| 64 | 44.3 ms | 0.69 ms | does not capture | 298.2 ms |
| 128 | 55.2 ms | 0.43 ms | does not capture | 348.8 ms |
| 256 | 79.0 ms | 0.31 ms | does not capture | 418.0 ms |

burn-flex's prefill takes 175 ms for 32 tokens, 514 for 128 and 1,040 for 256, about 4 ms a token.

- **The cost model.** A replay on every tile costs about **33 ms + 0.18 ms a token** (0.13 between 32 and 64). The
  fixed part is the 30 layers' per-op cost, the same as a decode step's. Two consequences:
  - **Small chunks lose.** A token of padding costs about 0.18 ms and one more replay about 33 ms, so a chunk up to
    about 180 tokens larger than the prompt still beats two replays.
  - **The most efficient prefill is one replay of the smallest captured chunk that covers the prompt**, and chunks of
    the largest only past it: buckets, not small chunks.
- **Which buckets.**
  - **64 and 256** cover up to 64 tokens in 44 ms and up to 256 in 79 ms. Past 256, each further 256 is one more
    replay.
  - **Adding 128** saves 24 ms for prompts of 65-128 tokens, at the cost of a third trace's program-cache footprint.
  - The per-token cost had not flattened at 256 (0.31 ms), so 512 is worth measuring with a longer cache.
- **Against burn-flex.** A 256-token prompt in one replay is **13x faster** (79 against 1,040 ms); 32 tokens 4.4x (40
  against 175).
- **One tile captures only single-token steps.** Its cost is attention over all 512 positions (1.13 s a token, against
  478 ms with a 64-position cache). Chunks wait for trace segmentation.
- **`work:64:4` is 5x slower when traced** than the default spread, as for the other traces.

## SmolLM2 decode step traced (2026-10-15, cards 0 and 1)

Roadmap Track 2 item 6. `Llama::forward_embedded` grows its attention by a position a step and writes its cache with
`slice_assign` at a host range, so no two steps have the same shapes. `Llama::step_fixed` (`crates/tt-models/src/llama.rs`)
takes the position as a tensor:
- the rotary rows are a row select from a `max_len` table;
- the key and value land by `mask_where` on a one-hot of the position;
- attention reads the whole cache with the later positions masked.

`burn_tt::TracedInference::capture_with_state` records a `copy_into` of each new cache tensor over the old one, so every
replay leaves the cache as the step computed it, on the card. Gate `step185_fixed_decode` (ttsim at one layer; card 0
at 30, `1792029209`):
- every fixed step's logits equal the growing cache's bit for bit;
- one captured step replays each later position bit for bit;
- without the write-back the replays diverge (negative control);
- masking the query's own position, watched failing, breaks the first claim.

Release, card 1, the 64-position cache, two warm-up and nine timed replays at positions 4-12, each checked bit for bit
against the eager fixed step, burn-flex in the same run (`smollm2_decode_traced`):

| Setting | Traced replay | Eager, fixed shape | Eager, growing cache | burn-flex |
|---|--:|--:|--:|--:|
| every tile, default spread (`TT_TILES=all`) | **37.1 ms** (`1792029144`) | 194.6 ms | not measured | 26.6 ms |
| every tile, `work:64:4` | 83.5 ms (`1792029158`) | 118.7 ms | 86.9 ms (`1792026124`) | 25.9 ms |
| one tile | 478.3 ms (`1792029169`) | 528.9 ms | 449.7 ms (`1792019917`) | 25.5 ms |

- **The replay is 1.4x burn-flex** on every tile, and 2.3x faster than the best eager step.
  - **Eager, the fixed step costs more than the growing one:** attention over all 64 positions, and a `mask_where` copy
    of each layer's cache.
  - **Traced, that host cost is gone.** What is left is device work.
  - **Next:** where the 37 ms of device work goes is not broken down yet; the replay profile
    (`Session::host_times`, the trace replay breakdown) would say.
- **The decode step captures even on one tile**, where ViT and ResNet-18 overflow the program cache. There it is
  device-bound on its m = 1 matmuls, as the eager step is.
- **The default spread wins traced** (37.1 against 83.5 ms) and `work:64:4` wins eager, as for ViT.
- **Regression.**
  - `forward_embedded` shares the layer code (`Write::At`). `step167` F32 and the seven `step168` plans pass on card 0
    (`1792029254`).
  - `tt-models`' host tests (against HuggingFace) pass.

## ViT-Tiny traced (2026-10-15, card 1)

Roadmap Track 2 item 6, after [attention-jobs.md](../completed-plans/attention-jobs.md) cut ViT's attention to 10 jobs
a call. `vit_tiny_forward_traced` makes one eager forward, captures one, then times replays: each writes the image
and reads the logits back with no op submitted by the host. It shares `traced_image_forward` with
`resnet18_forward_traced`. Release, card 1, two warm-ups and nine timed replays, burn-flex in the same run:

| Setting | Traced replay | Eager | burn-flex |
|---|--:|--:|--:|
| every tile, default spread (`TT_TILES=all`) | **12.6 ms** (`1792026627`) | not measured in this setting | 107.6 ms |
| every tile, `TT_SPREAD=work:64:4` | 61.9 ms (`1792026634`) | 129.1 ms (`1792026124`) | 107.6 ms |
| one tile | does not capture (`1792026639`) | 368.3 ms (`1792026138`) | 107.0 ms |

- **The replay's logits equal the eager forward's bit for bit**, and its top-1 equals burn-flex's.
- **ResNet-18 through the shared helper** is unchanged: 17.6 ms traced on every tile (`1792026644`; 17.6 in
  `1792000003`).
- **Capture at `work:64:4`.** ViT's capture fits the program cache at `work:64:4`, where ResNet-18's did not
  (`1792000003`). It still overflows on one tile, which is trace segmentation's case.
- **The default spread is faster when replayed, `work:64:4` when eager.** A replay has no host cost to amortise,
  so spreading every op over every tile wins: 12.6 against 61.9 ms. Eager, the per-unit cost made `work:64:4` the
  better setting. A traced deployment and an eager one want different spreads.
- **A decode step could not be traced as written** (its shapes grow a position a step). The fixed-shape step in the
  section above does it.

## Attention scores kept in tile rows, and a kept causal mask (2026-10-15, cards 0 and 1)

P2b and P3 of [attention-jobs.md](../completed-plans/attention-jobs.md).

- **ViT's scores.** After P1-P2 they were stacked after `Q K^T` and gathered back into whole tile rows for `P V`.
  Now `padded_attention` keeps them a head per whole tile rows throughout:
  - `Session::matmul_dram_batched_with`, `BatchedLayout { stacked: false }`, then `{ lhs_padded: true }`;
  - the scale, the mask and the softmax run on the padded `[heads·mt·32, sk]` tensor row by row.
  - It is taken where padding grows the scores by at most a quarter: ViT, not a decode's single query.
- **The causal mask** is made once an attachment, in the scores' own layout, so a call's mask is one `mask_fill`.

Release, card 1, two warm-ups and nine timed passes, burn-flex in the same run:

| Workload | Before (`1792019882`, `1792019870`) | After | burn-flex |
|---|--:|--:|--:|
| ViT-Tiny forward, every tile at `work:64:4` | 173.5 ms | **129.1 ms** (`1792026124`) | 109.3 ms |
| ViT-Tiny forward, one tile | 481.9 ms | 368.3 ms (`1792026138`) | 109.7 ms |
| SmolLM2 decode step, every tile | 86.5 ms | 86.9 ms (`1792026124`), unchanged | 26.5 ms |

- **ViT's attention server time, every tile.**
  - `qk` 26.5 -> **0.9 ms**: the stacking gather is gone.
  - `pv` 47.2 -> 30.4 ms: the gather back into tile rows is gone. What is left is the product, the scores' column
    padding fill (`meet`) and the stacking gather.
  - `softmax` 2.2 -> 4.0 ms, on 672 rows a call instead of 591.
- **ViT on one tile.** The softmax is 21.4 ms (was 40.9); `pv` is 64.2.
- **The decode step does not take the padded path.** It would be 288 rows for 9. Its attention is unchanged
  (`qk` 20.6, `pv` 13.4 ms).
- **A five-token causal prefill** is 11 jobs a call, was 13 (`step183`, ttsim). There is no prefill bench.
- **Correctness.**
  - Card 0 (`1792025224`): `step183`, `step184`, `step83` (single card), `step60`, `step166`, and `step167` in F32.
  - On ttsim: Burn's conformance has 0 unexpected results, plain (`1792025070`) and `--fusion` (`1792025609`).

## Attention products in one job (2026-10-14, cards 0 and 1)

P0-P2 of [attention-jobs.md](../completed-plans/attention-jobs.md). Before this, a batched matmul whose `m` was not whole tile rows
(decode m = 1, ViT m = 197, a prefill's m = t) ran one head at a time: two element repacks and a matmul a head, then a row
gather, so 28 of a decode call's 64 attention jobs for each product. Now `Session::matmul_dram_batched`
does three things, and the whole product is one server job:
- writes each product from a tile row of its own;
- stacks the products' rows with one row gather;
- gathers an operand whose heads are stacked by rows (`repeat_kv`'s `[9 T, 64]`, the stacked scores) into whole tile rows with zero
  tails.

`module::attention` charges its stages to parts (`attention/qk`, `/softmax`, ...). Release, card 1 (both cards were free;
the gates ran on card 0 at the same time), two warm-ups and nine timed passes, burn-flex timed in the same run:

| Workload | Before (card 0) | After (card 1) | burn-flex (same run) |
|---|--:|--:|--:|
| ViT-Tiny forward, every tile at `work:64:4` | 283.4 ms (`1792013104`) | **173.5 ms** (`1792019882`) | 107.2 ms |
| ViT-Tiny forward, one tile | 663.0 ms (`1792013112`) | 481.9 ms (`1792019870`) | 107.3 ms |
| SmolLM2 decode step, every tile | 99.6 ms (`1792013124`) | **86.5 ms** (`1792019882`) | 26.5 ms |
| SmolLM2 decode step, one tile | 450.4 ms (`1792013133`) | 449.7 ms (`1792019917`) | 25.1 ms |

- **Decode on every tile.**
  - Server jobs: 3,316 -> 1,696.
  - Server time: attention's 73 ms is now 36: `qk` 30 jobs, 20.6 ms; `pv` 30, 13.2 ms; `softmax` 210, 2.1 ms;
    `scale` 30, 0.3 ms.
  - A product job is 0.4-0.7 ms of server time. It runs up to two zero rows, two operand gathers, the product and
    the stacking gather.
- **Decode on one tile is unchanged.** It still waits on its m = 1 matmuls (roadmap, "Per-workload tile choice").
- **ViT on every tile.**
  - Server jobs: 766 -> 550.
  - Server time: `pv` 12 jobs, 47.2 ms; `qk` 12, 26.5 ms; `softmax` 84, 2.2 ms.
  - `pv` is the most expensive because the stacked scores `[3 x 197, 197]` are gathered by rows into whole tile
    rows before the product.
  - **Next:** the stacking gather after `qk` and the gather before `pv` undo each other. Keeping the scores padded
    through the softmax would remove both.
- **Correctness.**
  - Card 0 (`1792019871`): `step183_attention_jobs`, `step184_ragged_batched_matmul`, `step83_attention`, `step60`, `step182`,
    `step166_vit_tiny`, and `step167_smollm2` (F32 23.1 s, was 33; BF16 229 s).
  - Both cards (`1792020158`): the mesh attention, trace and module-reference tests.
  - On ttsim: Burn's conformance has 0 unexpected results (`1792018795`), and the MNIST e2e golden is unchanged.

## Slices and swaps as views (2026-10-14, card 0)

P2 of [decode-host-cost.md](../completed-plans/decode-host-cost.md). Three changes:
- **Slices are views.** A slice whose steps are all positive is a view of the same buffer (`sliced_native`) and
  queues no job.
- **A swap of a view stays a view.** `float_swap_dims` uploads only a tensor with no device copy, where it used to
  materialise any view first.
- **Ragged views are tile copies.** A view made plain is a tile copy wherever each output tile is one source block,
  now including a ragged last tile row or column where the source ends there too (`Strided::tile_moves`).

ViT's attention therefore reads q, k and v as blocks of the fused projection (`matrix_blocks`), with no copy. Same
conditions as "Native `cat`".

| Workload, card 0 | Before | After | burn-flex |
|---|--:|--:|--:|
| ViT-Tiny forward, every tile at `work:64:4` | 401.4 ms (`1792005220`, P0) | **283.4 ms** (`1792013104`) | 106.7 ms |
| ViT-Tiny forward, one tile | 855.9 ms (`1792005205`) | 663.0 ms (`1792013112`) | 112.5 ms |
| SmolLM2 decode step, every tile | 108.5 ms (`1792010467`, P3) | **99.6 ms** (`1792013124`) | 26.2 ms |
| SmolLM2 decode step, one tile | 451.7 ms (`1792010476`) | 450.4 ms (`1792013133`) | 26.6 ms |

- **ViT queues 766 jobs, was 853.** Its 37 slices and 49 swaps queue none, where they cost 83 ms of server time at
  P0. What is left on every tile is `attention`, 336 jobs and 187 of 270 ms (0.56 ms a job, device work). Then come
  the patch embedding's convolution (5 jobs, 44 ms) and the heads merged back by `reshape` (12 jobs, 10 ms). That
  merge is an element repack: flattened, a tile row of the 197-row heads crosses from one head into the next, so no
  tile copy applies.
- **The decode step queues 3,316 jobs, was 3,617.** Its `slice` jobs are gone. SmolLM2's value `v` is now a view
  of parts of rows (a projection split into heads), so a cache write first makes it plain: 30 small repacks, and
  the 90 jobs together take 1.9 ms. Before that fix the cache writes had fallen back to the whole-cache element map,
  at 149 us a job (`1792011825`).
- **Correctness.** Gate `step182_region_views` passes on ttsim and card 0 (`1792012830`); `step165`-`step168` pass on
  card 0 (`1792012839` to `1792012885`). Burn's conformance on ttsim: 3,025 run, 0 unexpected, plain and `--fusion`. The smoke
  tier on card 0 passes 1,052 of 1,052 (`1792013157`).

## Cache writes as a tile copy and written rows (2026-10-14, card 0)

P3 of [decode-host-cost.md](../completed-plans/decode-host-cost.md). A `slice_assign` whose region writes the last dimension whole and in
order (a key/value cache's positions) is now `Session::assign_rows`: a tile copy of the destination (`Session::copy`, a few run
records) with the value's rows written over it (`write_rows`, entries for the written rows only). A destination that is a view is one
row gather; anything else keeps the element map. The destination is never written in place. Same conditions as "Native `cat`".

| SmolLM2 decode step, lookup excluded, every tile at `work:64:4` | `slice_assign` (60 a step) | Step |
|---|--:|--:|
| After the native `cat` (`1792006312`): whole-destination element map | 15.3 ms, 254 us a job | 117.3 ms |
| One row gather over the whole cache (`1792009767`, not kept) | 8.8 ms, 147 us a job | 114.0 ms |
| **Tile copy and written rows** (`1792010467`) | **1.5 ms, 25 us a job** | **108.5 ms** (p10 90.7, p90 125.5) |

The row gather was the first version. It copies the cache row by row, so its host entries (two a row a tile column, about 768 for
this 64-position cache) grow with the cache's length; the tile copy's do not. On one tile the step is 451.7 ms (`1792010476`;
`slice_assign` 5.5 ms there, waiting on the device's matmuls). Gate `step181_region_assign`, ttsim and card 0 (`1792010463`);
`step167` F32, `step168` and `step166` pass on card 0 (`1792010494`, `1792010546`, `1792010734`).

## Native `cat` (2026-10-14, card 0)

P1 of [decode-host-cost.md](../completed-plans/decode-host-cost.md). `float_cat`, `int_cat`, `bool_cat` and the three `repeat_dim`s are
overridden (`crates/burn-tt/src/ops/region.rs`):
- A `cat` along any dimension but the last, of pieces that are whole rows of their buffers, is **one row gather**
  (`gather_rows`). The host names one source row per output row.
- Any other `cat` is **one element repack** over every piece.
- `repeat_dim` is a `cat` of the tensor.

Burn's default was a zero-fill (`Session::metadata`, mover immediates) and one whole-destination `slice_assign` a piece.
Release build, `0ba8d00` plus P0 and P1, card 0, nothing else on either card; `silicon_bench_models`, two warm-ups and
nine timed passes, burn-flex in the same run.

| SmolLM2-135M decode step, lookup excluded | Before (P0) | After | burn-flex |
|---|--:|--:|--:|
| every tile, `TT_SPREAD=work:64:4` | 232.6 ms (`1792005178`) | **117.3 ms** (p10 100.8, p90 136.0; `1792006312`) | 26.3 ms |
| one tile (default) | 623.9 ms (`1792005087`) | 458.1 ms (p10 425.2, p90 475.9; `1792006329`) | 26.5 ms |

Server-thread time and jobs on every tile, one reported pass:

| | Before | After |
|---|--:|--:|
| Jobs a step | 4,279 | 3,617 |
| Server thread busy | 323 ms | 146 ms |
| `cat` (`float_empty` + `float_slice_assign`, then `float_cat`) | 843 jobs, 196.6 ms | 121 jobs, **4.6 ms** (38 us a job) |
| `float_slice_assign` left (the two cache writes a layer) | (in the above) | 60 jobs, 15.3 ms (254 us each) |
| `attention` | 1,920 jobs, 71.9 ms | 1,920 jobs, 72.4 ms |
| `float_matmul` | 211 jobs, 18.6 ms | 211 jobs, 18.0 ms |
| `float_slice` | 361 jobs, 9.4 ms | 361 jobs, 10.4 ms |

- **What is left on every tile is attention.** Its 1,920 jobs (64 a call) take half the server thread. Then come the
  matmuls, and the cache's `slice_assign` (each still maps the whole `[1, 3, 64, 64]` cache on the host: P3).
- **On one tile the step waits on the device.** Matmuls take 268.6 ms (1.27 ms each, unchanged), which is the tile
  choice, not submission.
- **The models are unchanged.** On card 0 `step167_smollm2` F32 (30 layers, 16 greedy tokens equal to burn-flex's)
  passes in 23.4 s (was 33.2 s, `1791851959`; run `1792006367`). `step165`, `step166` and all seven `step168` plans
  pass (`1792006351`, `1792006363`, `1792006391`).
- **The gate.** `step180_native_cat` passes on ttsim and on card 0 (`1792006303`, 6 of 6). Burn's conformance suite on
  ttsim: 3,025 run, 2,973 pass, 52 failed as listed, 0 unexpected (`--fusion`: 2,970 pass, 55 failed as listed, 0
  unexpected). The smoke tier on card 0 passes 1,040 of 1,040 (`1792006807`).

## Decode step: where the host time goes (2026-10-14, card 0)

P0 of [decode-host-cost.md](../completed-plans/decode-host-cost.md). The model benchmarks' reported pass now also counts the
**jobs each Burn op queues on the server thread** (`OpStat::jobs`, charged to the innermost op on the calling thread)
and the **server thread's wall time per op** (`burn_tt::server_report`), so a view no longer counts as device work
and a slow op can be told from a slow device. Release build, `0ba8d00` plus the counters, card 0, nothing else on
either card, `silicon_bench_models`, two warm-ups and nine timed passes, burn-flex in the same run.

| Workload | Tiles | burn-flex | burn-tt | Server jobs | Server busy, reported pass | Run |
|---|---|--:|--:|--:|--:|---|
| SmolLM2-135M decode step, lookup excluded | 1 (default) | 25.8 ms | 623.9 ms (p10 525, p90 705) | 4,279 | 770 of 771 ms | `1792005087` |
| the same | all 120, `TT_SPREAD=work:64:4` | 26.2 ms | **232.6 ms** (p10 178, p90 293) | 4,279 | 323 of 326 ms | `1792005178` |
| ViT-Tiny/16 forward | 1 (default) | 106.8 ms | 855.9 ms | 853 | 846 of 856 ms | `1792005205` |
| the same | all 120, `work:64:4` | 106.5 ms | **401.4 ms** | 853 | 392 of 401 ms | `1792005220` |

The one-tile decode is the 623 ms of `1791852097` again: the Oct 14 barrier and spread changes did not move it (they
help only a session with many tiles open). The 2,779 "reported device ops" are 4,279 jobs; 1,091 `float_device` calls
and 363 of 393 `float_reshape` calls queue nothing.

**Server-thread time by op, decode step** (ms in one pass, then us a job):

| Op | Jobs | 1 tile | all tiles, `work:64:4` |
|---|--:|--:|--:|
| `float_slice_assign` | 722 | 100.5 (139) | 99.4 (138) |
| `float_empty` (`cat`'s zero-fill) | 121 | 224.8 (1,858) | 97.2 (803) |
| `attention` (64 jobs a call) | 1,920 | 100.7 (52) | 71.9 (37) |
| `float_matmul` | 211 | 268.2 (1,271) | 18.6 (88) |
| `float_slice` | 361 | 9.5 (26) | 9.4 (26) |
| `float_neg` | 60 | 13.4 (223) | 0.6 (10) |
| `float_swap_dims` | 90 | 13.1 (146) | 1.8 (20) |
| everything else (incl. 4,278 frees, 1.1 ms) | | 39.9 | 24.6 |

Reading them:

* **On one tile the decode is waiting on the device, not on submission.** `TT_HOST_TIMES=1` (run `1792005154`, the whole
  attachment, 12 passes): of 6,519 ms queueing, 6,055 ms is the `List` stage with 5.5 million PCIe reads, about 70 a list,
  which is the host waiting for room in the one tile's ring. On all tiles (`1792005178`) the same stage is 1,085 ms and
  140 thousand reads. The device work that fills the ring is mostly the m = 1 matmuls (1.27 ms a matmul on one tile, 88
  us on all tiles). The roadmap's "host submission, 2,779 operations at about 220 us each" was the per-op average of a
  device-bound step.
* **On all tiles, `cat` is the cost.** Burn's default `cat` (no override in burn-tt) is a zero-fill and one
  `slice_assign` a piece. `float_empty` and `float_slice_assign` are 196.6 of the 323 ms (61%). A `slice_assign` costs
  ~138 us on either session, so it is host work: the whole-destination element map. A zero-fill costs 800-1,860 us, because
  `Session::metadata` writes the zeros as mover immediates. A native `cat` removes all 121 fills and turns the 720 piece writes
  into 120 jobs (P1).
* **Attention is the next cost after `cat`.** 1,920 jobs, 64 a call, 37-52 us each. That is the composed attention's own job
  count, and outside this plan's P1-P3.
* **The per-op fixed cost is small.** Frees cost 0.3 us each (1.1 ms in all), and element-wise ops 8-30 us a job, so P4 waits.
* **ViT-Tiny is not per-op submission either.** It has 853 jobs. At all tiles the time is `attention` (348 jobs, 211 ms, 0.6 ms
  each, device work on 197 tokens), then `float_swap_dims` (37 jobs, 55 ms, 1.5 ms each) and `float_slice` (37, 28 ms). Both are
  whole-tensor element repacks, which P2 targets. Its convolution (patch embedding) is 44 ms in 5 jobs.

## Chunk-free convolution and max pooling (2026-10-14, cards 0 and 1)

ResNet-18 took 16.3 s a forward (105x burn-flex) because `conv2d` was one `float_matmul` per 32 output positions, each with a
host-built patch gather (`mapped_native`: about one 32-byte `COPY_WORDS` mover entry a datum), plus per-element weight and NCHW
joins, and `max_pool2d` was 32 windows at a time (6,272 chunks of repack, argmax and gather for 64x112x112).

**The change** (roadmap Track 2 item 2; `crates/burn-tt/src/ops/nhwc.rs`, `Session::select_taps`):
- A 2-D convolution is one tap select and one product. The activation's NHWC table `[N H W, C]` and a resident `[taps, P]` I32
  index of table rows (depends on the shapes alone: made once by `server::metadata` and kept per attachment) give the patch
  matrix `[P, taps x C]` through B's existing `SELECT_ROWS` record, one record a tap and output tile row (tap `t`'s indices are row
  `t % 32` of an index tile, reached by moving `idx_at`; no firmware change). Its K order (tap, channel) is burn-flex's im2col
  order. The weights relaid to `[taps x C, O]` are kept on the weight's cell (`TtTensor::derived`; not while capturing, nor for a
  buffer written in place). A padding tap reads the table's zero row (its first padding row, made zero; a table of whole tile rows
  is first copied into a zeroed one a row longer).
- The `[P, O]` result is returned as an NCHW view (`nhwc_strides`), and element-wise ops run on a view that covers its buffer
  one to one in place (`ops::view_op`): BatchNorm's `[1, C, 1, 1]` is a row of the buffer, relu and the residual add are
  same-layout. The next convolution and max pool read the view's buffer as their table: nothing is relaid between layers.
- Max pooling (F32, `C % 32 == 0`) is one select, a padding tap reading its window's first valid datum (the same bits), and a
  fold over the taps' column blocks with exact compares and selects: the first NaN, else the first maximum.

**Card 1, release, `silicon_bench_models::resnet18_forward`** (cats.jpg, pretrained weights, batch 1, 2 warm-ups, median of 9,
burn-flex timed in the same run; top-1 equal to Flex's):

| Session | burn-tt | burn-flex | Run |
|---|--:|--:|---|
| one tile (the default, unchanged) | 782.3 ms (was 16,288 ms, card 0, `1791846735`) | 156.9 ms | `1791996580` |
| every tile, `TT_SPREAD=work:64:4` | **131.1 ms** | 151.4 ms | `1791996716` |
| every tile, spread `all` (the default spread) | 203.8 ms | 151.3 ms | `1791996730` |

A first run of the one-tile row while ttsim gates loaded the host gave 781.8 ms against burn-flex 182.5 (`1791996284`): the
card time does not move, burn-flex does.

**Where the one-tile forward goes** (warm, card 1, device-time per op kind): the host submits the whole forward in ~15 ms; the
other ~768 ms is the card, waited on at the final download. Of it, conv1 alone is ~366 ms from an NHWC view (49 taps of a
3-channel table: each 64-byte face-row read carries 3 useful datums, ~1.2M reads on one tile's mover) and relaying the image
into its table ~95 ms (a per-element repack, the one left in the forward).

**`probe_resnet_op_cost`** (card 1, `1791996580`; one tile, median of 3 warm). The first rows start from a plain NCHW tensor
and read the result back, so they now include relaying the input in and the output out, a datum an entry each; the second
set is as in the model (an NHWC view in, the result left on the card, timed to a `sync`):

| Op | was (card 0, `1791846963`) | NCHW in, read back | NHWC view in, on the card |
|---|--:|--:|--:|
| max pool 3x3/2 on 64x112x112 | 2,817 ms | 558.0 ms | 37.4 ms |
| conv 3x3/1 64->64 on 56x56 | 895 ms | 232.3 ms | 29.9 ms |
| conv 3x3/1 128->128 on 28x28 | 517 ms | 111.8 ms | 15.5 ms |
| conv 3x3/1 256->256 on 14x14 | 529 ms | 56.4 ms | 9.9 ms |
| conv 3x3/1 512->512 on 7x7 | 1,411 ms | 32.8 ms | 8.2 ms |
| add + relu 64x56x56 | 0.6 ms | 0.6 ms | 0.1 ms |

**Accuracy.** `step165_resnet18` on card 0 (`1791996290`): Flex's top-5, max |flex - device| 5.69e-2 against a largest logit of
14.65 (5.13e-2 before; not asserted: no derived bound for a whole network). On ttsim at 64x64: Flex's top-5, 1.45e-1 against
17.11.

**Traced** (`silicon_bench_models::resnet18_forward_traced`, card 1, run `1792000003`, every tile with the default spread;
one eager forward first, which makes and keeps the relaid weights, then one capture; each timed pass writes the image, replays
and reads the logits back):

| `TT_TILES=all` | burn-tt | burn-flex |
|---|--:|--:|
| untraced | 205.7 ms | 156.7 ms |
| traced replay | **17.6 ms** (p10 17.59, p90 17.64) | 151.7 ms |

The replay's logits equal the eager forward's bit for bit. On one tile, or at `TT_SPREAD=work:64:4`, the capture fails: an op's
list of programs does not fit the 240 KB program cache (`a list's programs do not fit an empty program cache`); spread over every
tile it does. The benchmark skips with that reason until traces are segmented (roadmap Track 2, "Trace segmentation").

**What is left:** the default session is one tile, and the grid (`TT_SPREAD=work:64:4`) is what takes ResNet-18 under burn-flex:
roadmap Track 2, "Per-workload tile choice". On one tile, conv1's sparse table is the largest op (a wider table, or the image's
relayout fused into its select, would cut its reads). Backward passes, max pooling with indices, average pooling, grouped and
3-D convolutions still take the chunked path.

## tt-metal reference: DRAM reads by core count (2026-10-13, card 1)

The same measurement on tt-metal (`crates/tt-tests/bench/tt_metal_dram_read/`, v077 `d53c2a71fb0`, fast dispatch): N cores each
read their share of one interleaved DRAM buffer page by page (`noc_async_read_page`, a `TensorAccessor`), `--batch` requests then a
barrier, into a 256 KiB window of their own L1, timed by the tile wall clock inside the kernel (first request to last landing; all
bytes checked against a poisoned window). Card 1, AICLK 1350 MHz; ours is `gddr_aggregate` (Read mix, rotating channels, cap 8)
on the same card in the same session. GB/s for the whole card, and per core in brackets. tt-metal's fast dispatch reserves cores:
**110 here, ours 120**.

| Cores | 2 KiB: ours safe | ours fast | tt-metal | 4 KiB: ours safe | ours fast | tt-metal | 16 KiB: ours safe | ours fast | tt-metal |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 7.4 | 14.4 | 25.5 | 14.7 | 28.8 | 44.2 | 58.2 | 85.1 | 69.2 |
| 4 | 29.4 | 57.1 | 101 | 58.6 | 114 | 174 | 232 | 330 | 257 |
| 8 | 58.7 | 114 | 197 | 117 | 227 | 314 | 463 | 452 | 368 |
| 16 | 117 | 226 | 240 | 234 | 343 | 321 | 483 | 482 | 374 |
| 32 | 233 | 288 | 250 | 397 | 447 | 361 | 484 | 484 | 370 |
| 64 | 335 | 393 | 357 | 462 | 474 | 402 | 488 | 489 | 383 |
| 110 (tt-metal) / 120 (ours) | 386 | 446 | 378 | 480 | 491 | 415 | 496 | 496 | 387 |

Ours: 2 KiB safe / fast `1791910320` / `1791910651`, 4 KiB `1791910902` / `1791911151`, 16 KiB `1791907949` / `1791908197`. tt-metal's
logs: `target/silicon/bench/ttmetal-v077-card1-*.log`.

**Per core, few cores** (the entry cost; 1 core): a 4 KiB page is ~125 cycles on tt-metal (44.2 GB/s), 175 on ours fast (28.8), 359 on
ours safe (14.7); a 2 KiB page ~108 / 175 / 359. Ours fast is within 1.5x of tt-metal's compiled reader at 4 KiB (1.8x at 2 KiB), safe
within 3x (3.4x). At 16 KiB, where the NoC is the limit, ours fast is faster, 85 against 69 GB/s: our window of requests slides, where
tt-metal's loop drains at every barrier.

**Whole card, many cores:** ours reaches 94-97% of the 512 GB/s at 4 and 16 KiB (safe and fast alike) and 77 / 87% at 2 KiB; the plain
interleaved tt-metal pattern reaches 81% at 4 KiB, 76% at 16 KiB, 74% at 2 KiB. The difference is already there at 64 cores (tt-metal 402
against our 462 / 474 at 4 KiB), so it is not the 110 cores.

Variants of tt-metal's side (4 KiB unless noted; whole-card GB/s at 1 / 8 / 110 cores):

| Variant | 1 | 8 | 110 |
|---|--:|--:|--:|
| barrier every 8 pages, NoC0 (the table) | 44.2 | 314 | 415 |
| barrier every 32 pages | 68.0 | 322 | 415 |
| NoC1 (NCRISC, tt-metal's default reader), batch 8 | 42.1 | 96.3 | 249 |
| core i reads bank i mod 8 directly (`--pinned`), batch 8 | 35.8 | 264 | 371 |
| the same, batch 32 | 54.1 | 293 | 339 |
| 16 KiB pinned, batch 8 | 54.1 | 307 | 364 |
| 2 KiB, one request in flight (the stock `6_dram_offchip` pattern) | 5.3 | 42.3 | 299 |

**Read this with its limits.**
- Two things I did not do: tt-metal's tuned DRAM benchmark (its report places one reader per bank next to it to avoid NoC
  congestion; this run uses the first N cores in row order), and a port choice: its interleaved and bank-id addressing reach each bank
  through one port, where ours rotates over the channel's three (`gddr_aggregate_affinity`'s Nearest mixes, one port, reach only
  50-70% on our side too). So "tt-metal tops out at 76-81%" is the idiomatic pattern on this placement, not tt-metal's limit; their
  published figure on Wormhole is 92.7% with the tuned layout. **The port spreading is my hypothesis for the gap, not something tested
  on tt-metal.**
- Their barrier-per-batch loop leaves bubbles that our sliding cap does not; the batch-32 row shows it only helps at one core.
- Timing is by the same wall clock and the bytes are checked on both sides; ours and tt-metal's ran on card 1, one after the other, at 1350 MHz,
  16,000 MT/s, 8 channels. tt-metal's `compute_with_storage_grid_size` is 11x10 under fast dispatch.

## Host-encoded read entries (2026-10-13, card 0)

`op::READ_ENCODED` (`tt_isa::dm::encoded`). The host sends a read as `[READ_ENCODED, TARG, TARG_HI, RET, RET_HI, CHECK, 0,
LEN]` (the same 32 bytes; `TAG` and `CTRL` of a B read are constants the card supplies) and validates it in full
(`encoded::validate`, by reconstruction: it must equal what `Descriptor::decode` + `DramMove` make for some valid move).
The card verifies `CHECK`, an integrity word, and issues. It is a **build-time mode of the B mover**, firmware feature
`fast-reads` (`cargo xtask silicon --feature fast-reads`):

| Mode | Build | The card |
|---|---|---|
| **safe** (default) | no feature | decodes and checks every read itself; `READ_ENCODED` is an unknown op (`error::OP`), refused before any NIU write |
| **fast** | `fast-reads` | verifies a checksum; issues |
| **fast + guard** | `encoded-guard` | the checksum, and `encoded::guard` (usable-endpoint bitmap, return coordinate, bounds, mod-64 congruence) |

The image writes `dm::CAPS` as it starts and the host reads it (`DataMover::fast_reads`): a safe image is never encoded for
(`set_encode_reads(true)` is `DmError::FastReadsNotBuilt`), a fast one is by default. `DataMover::set_encode_reads` rewrites every
plain single-request `READ` after the usual check (outside records; in a packet's reader section too; one entry for one).
Gate `step172_encoded_reads` (ttsim and card 0), in whichever mode is built. `dm_b`: safe 24,412 bytes (24,404 before the
capability word), fast 24,832, fast + guard 24,992; `dm_nc` unchanged.

**Closing the safety gap to safe mode (plan phase 1, 2026-10-13, branch `fast-mode-hardening`).**
- *Context-bound checksum.* The checksum's seed is `encoded::context_seed(key, tile, usable, op)`: `key` is a fresh random word the
  host writes to `dm::ENCODE_KEY` before it starts the mover (the firmware reads it once at boot into a `static`), so an entry
  made for an earlier mover instance (a list left in L1), another tile or chip, or another op fails the card's check
  (`step172 ... refuses_an_entry_made_for_an_earlier_mover`, a valid read in every other respect, so safe to send on silicon). A trace
  replay (`CALL`) flips the seed by `TRACE_DOMAIN`, so an encoded entry inside a trace fails it; the host never encodes in a trace.
  A binding against mix-ups, not a MAC: whoever can write L1 can read the key. Unit tests: every bit of key, x, y, usable and
  op changes the seed; an entry verifies for no other context. Watched failing with the key removed from the seed.
- *`validate` is not circular.* It still reconstructs the entry through `encode` (so the card cannot run a request the safe decode
  would refuse), and now also requires `guard`, a check that shares no code with it, so an encoder bug that `validate` agrees with is
  caught on the host in both builds. The fuzz property "validate accepts => guard accepts" holds over 5 million cases.
- *`enqueue_checked` validates every `READ_ENCODED`* (an unknown op, `error::OP`, on a safe image) and the gates that forge entries on
  purpose use the `#[doc(hidden)]` `enqueue_raw`.
- *A destination moved by 64 bytes is not refusable.* The earlier plan expected `encoded-guard` to refuse an encoded entry with
  `ret += 64` and a recomputed check. It is another valid read (congruent, inside L1): neither `validate` nor the guard can call it
  wrong, because only the caller knows where it was meant to land. What protects against it is that the host only makes entries from
  plain reads it has checked, and the card's checksum is keyed. `step173_mover_differential` holds the end-to-end property instead:
  seeded random lists (mixed sizes, one request and several, every channel and port, `WAIT`s between, disjoint destinations)
  land the host-memcpy oracle's bytes in the safe build, in fast mode with plain entries and with encoded ones. Its negative control
  is that very `ret += 64` entry, which the comparison reports as a mismatch (the comparison has teeth).
- *Cost.* Silicon, card 0 (gate runs `1791925982` safe and `1791926071` fast, entry cost `1791926086`, safe plain entry `1791926103`): all 14 gate runs pass in the safe and fast builds; the encoded
  entry is **180.5 cycles** (175 before: the seed is a load from local RAM instead of an immediate; P2's tight loop hoists it) and the
  safe build's plain entry unchanged (362 / 359 / 360 against 362 / 362 / 361 before at N = 64 / 256 / 448). `dm_b`: safe 24,400
  bytes, fast 24,924, fast + guard 25,092; `dm_nc` 14,252.

**End-to-end, after phases 0-3 and 7 (card 0, release, runs `1791930590` safe, `1791931447` `--feature fast-reads`).** The smoke
selection, one card, one boot each: safe 606 of 607 pass, fast 611 of 612 (the five extra are step173 and the new step172 gates); the one
failure in both is the two-card Ethernet gate `step76 bf16_mesh_product`, which needs both cards and the Ethernet link in a known
state: it failed on `--device all` too, and passed on both cards after `tt-smi -r` of both boards
([silicon-operating-notes](silicon-operating-notes.md#running-tt-metal-after-this-repos-gates-2026-10-13-card-1)), so it is link state
and not the movers. Neither build refused a list.

**Traced execution, with the traces left plain (2026-10-14, card 0; the capture-time encoding of a trace's reads was built, measured and
reverted: `ff39af0`, `2315646`).** Replay time, three runs each, pre-branch commit (`7f18dde`) / this branch's safe build / its fast build: traced
MNIST inference, batch 64, 1 tile 0.493 / 0.494 / 0.469 ms per batch, all tiles 0.510-0.522 / 0.509-0.524 / 0.513-0.524; traced CNN training, 1
tile, 146.2 / 145.6 / 146.8 ms per step. So **the branch's safety and speed commits did not slow traced execution** (safe equals the
pre-branch commit), and **fast mode alone does not speed it up**: a trace holds plain entries, which the card decodes at the safe rate, and the
fast image's one effect on them is the initiator-1 issue path (`READ_FAST`, on by default there: plain reads 361 -> 334 cycles), worth ~5% on a
one-tile replay. Encoding a trace's reads needs the trace's reads to be encodable, and nearly all of a streaming op's reads are records
(a captured layer held 0 plain reads); expanding the records at capture gave -16% (inference, 1 tile) and -17% (CNN training, 1 tile) but grew a
trace 1.3x to 7x in entries (1.0x in the default build), needing a size budget and an unexpanded fallback, for replays that are mostly compute
and synchronisation. Not kept. If a traced workload turns out to be read-bound, the size-neutral route is sealed static read records (the
card validates a record once and issues its tiles at the encoded rate, the record staying three entries), not expansion.

**Containment (plan phase 7, 2026-10-13; both modes, host only).** `DataMover::check_as` now requires every read's or write's L1
range, top-level, encoded or expanded from a record, to lie in the data arena (`tt_isa::l1::DATA`, `0x2_0000` to the mailbox
base): a list cannot be aimed at the mover's key, its list ring, its mailbox or its image, which neither mode's card-side
decode forbids (`Descriptor::decode` bounds by `L1_SIZE`). A survey first (`TT_CONTAIN=report`, the whole simulator suite, 1,484 tests
passing) found **no list that names another region**, so nothing needed a named exception and enforcement is the default
(`TT_CONTAIN=off` or `report` override it). `step172 a_read_cannot_be_aimed_at_the_movers_own_l1`, in both builds: the encode key,
the mailbox, the ring, the image and a read running past the arena's end are refused (`error::RANGE`) before any PCIe, and the same
test fails with `TT_CONTAIN=off` (the read onto the key is accepted). Not covered, and what remains trusted: the other entry kinds
that name L1 (`INDEX_PICK`, `COPY_WORDS`, `FILL`), `MY_X`, `MY_Y` and `USABLE` (host-written in both modes), and the key, which
is a binding against mix-ups and not a MAC. The survey also met one failing test unrelated to the movers, `burn-tt`
`essentials::dtype_capabilities_advertise_what_the_device_computes_and_no_more`, which fails the same on the commit before this change.

**The tight read loop (plan phase 2, 2026-10-13, card 0).** Three steps, each measured on card 0 with a resident list rerun from the
sequence word (device cycles per encoded entry, 4 KiB; 64 B in brackets):

| Step | Fast | Fast + guard | Safe (plain) | Run |
|---|--:|--:|--:|---|
| after phase 1 (keyed checksum) | 180.5 (180.5) | | 361 | `1791926086` |
| 1. request initiator 1 (X6) by default in a fast image | 159.7 (155.2) | | 361 (fast image's plain entry 334) | `1791926392` |
| 2-3. `noc::ReadBatch` and `run_encoded_reads`, a loop of its own | **90.8 (81.0)** | **108.7 (108.6)** | 363 / 358 / 360 (N = 64 / 256 / 448; unchanged) | `1791926751`, `1791926891`, `1791926977` |
| 4. software pipelining | not done: the bare issue loop is ~88 at 4 KiB, 69 at 64 B (read-loop probe) | | | |

- *Step 1* costs nothing but a host write (`READ_FAST = 1` in `start_on` when the image reports `FAST_READS`): each request
  writes the address words, `CTRL` and the command, and the rest of initiator 1 once per list (`step118` holds it
  byte-identical to the slow path on silicon, with its negative controls). It takes the fast image's plain entries from 361 to 334 too.
- *Steps 2-3.* `run_encoded_reads` runs a list's consecutive `READ_ENCODED` entries in one loop, untraced, on the fast path:
  the checksum seed, the guard's tile word, the in-flight room and the initiator's key stay in registers for the run;
  the entry is read word by word, not copied into an array; and `CMD_CTRL` is read back **once per run**
  (`noc::ReadBatch`, over `tt_isa::noc::niu::Batch`) instead of after every request -- the next request's own wait for the
  initiator to be free reads the same register behind the same store, so only the counter reads need the fence (before a
  refill, and when the run ends). Error semantics are the one-at-a-time path's: entries before a bad one are issued, then
  `RESULT = READ_ENCODED` and the code, with no final wait; `step172` holds it for a bad first, middle and last entry of a run
  of eight, watched failing with the issue moved before the check. The accounting is unit-tested on the host against the
  NIU model (the counter never passes the cap, it is read only when the batch is out of room and always after a fence, one
  fence per refill and per non-empty batch). 90.8 is within 3 cycles of the bare issue loop, so there is nothing left for
  software pipelining to win; the guard costs 18 cycles in the loop (it was 40).
- *What it did to the card.* The first card-wide runs after the tight loop read the same as before (1 tile 29.1 GB/s at 4 KiB) and
  were taken as the loop not mattering; they were traced runs, which bypass the loop. Untraced it takes a tile from 15 GB/s to
  57 (see "What bounds one tile's read rate" in this section).

**What bounds one tile's read rate (2026-10-13, cards 0 and 1; this corrects the first card-wide runs after the tight loop, and the first
in-flight cap sweep).** Those read 29 GB/s for one tile at every cap, with the bound unexplained, and were taken as the tight loop not
mattering. They measured the benchmark, not the mover: `gddr_aggregate` times its lists with the timestamper, which stores two events
around every entry (`dm::TRACE`), and a traced run does not take the tight loop (`run_encoded_reads` is for untraced lists) and pays the
events on top. The probe `silicon_probe_read_loop::what_bounds_one_tiles_read_rate` (card 1, fast build, one tile, 256 reads of 4 KiB,
resident lists rerun from the sequence word, every repetition's landed bytes compared; run `1791932966`) isolates it:

| 4 KiB reads, one tile | per entry | rate |
|---|--:|--:|
| one channel, untraced, plain (the safe path's cost) | 346.5 cycles | 16.0 GB/s |
| one channel, untraced, **encoded** | **90.1** | **61.4 GB/s** |
| eight channels in turn, untraced, plain | 354.7 | 15.6 |
| eight channels in turn, untraced, **encoded** | **94.7** | **58.4** |
| one channel, **traced**, encoded | 184.0 | 30.0 |
| eight channels in turn, traced, encoded | 186.2 | 29.7 |
| traced, plain | 359-369 | 15.0-15.4 |

Tracing costs an encoded read ~94 cycles (the events, and the loop it excludes) and a plain read ~13: the benchmark understated the fast mode
by 2x and the safe mode by 4%. A new `dm::trace_mode::LISTS` (`TRACE = 2`: the list's begin and end and every entry but a plain or encoded
read) gives `gddr_aggregate` what it needs (the barrier's release and the list's end) with the reads untraced; it is the default now, and
`AGG_TRACE=full` restores the old runs. The safe image is unchanged by the firmware change (plain 358 / 358 / 358 cycles at N = 64 / 256 /
448; 363 / 358 / 360 before), the fast entry is 91.5.

**Card-wide reads, list-level trace** (GB/s; safe and fast builds, card 1 at 4 KiB, card 0 at 2 KiB; runs `1791933129` and `1791933479`
at 4 KiB, `1791933130` and `1791933480` at 2 KiB):

| Tiles | 4 KiB safe | 4 KiB fast | fast / safe | 2 KiB safe | 2 KiB fast | fast / safe |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 14.9 | 57.0 | 3.8x | 7.5 | 30.0 | 4.0x |
| 2 | 29.7 | 112.1 | 3.8x | 14.9 | 58.9 | 4.0x |
| 4 | 59.4 | 219.3 | 3.7x | 29.8 | 115.0 | 3.9x |
| 8 | 118.8 | 340.0 | 2.9x | 59.6 | 222.1 | 3.7x |
| 16 | 237.4 | 376.0 | 1.6x | 118.8 | 252.3 | 2.1x |
| 32 | 411.2 | 460.7 | 1.12x | 229.7 | 315.5 | 1.37x |
| 64 | 454.0 | 484.0 | 1.07x | 353.3 | 404-414 | 1.15-1.17x |
| 120 | 480.1 | 494.8 | 1.03x | 401.3 | 451-453 | 1.13x |

(That lane's 2 KiB fast run read 312 and 375 GB/s at 64 and 120 tiles, an outlier: three other runs of the same configuration, the cap
sweep's cap 8 included, read 404-414 and 451-453, so those two are not used.) A tile reads 57-61 GB/s, 89-95% of one channel; the card is
3.7-3.8x the safe default up to 4 tiles at 4 KiB, at 66% of its bandwidth with 8 tiles (23% safe) and 90% with 32; the channels, not the
movers, bound it from 16 tiles at 4 KiB and from 32 at 2 KiB.

**In-flight depth** (plan phase 3, not built): with the list-level trace the cap (2, 3, 4, 6, 8; fast; card 0 at 2 KiB, card 1 at 4 KiB)
changes nothing beyond noise at any tile count: 4 KiB, 8 tiles 338 / 326 / 337 / 337 / 335 GB/s, 32 tiles 450 / 455 / 449 / 460 / 459, 120 tiles
492 / 494 / 492 / 494 / 494; 2 KiB, 8 tiles 222 / 223 / 185 / 204 / 222, 120 tiles 447 / 450 / 440 / 450 / 451 (one cap-4 point is low). The plan
gated a per-list cap on 5% at 1-4 tiles with no regression at 16 or more; the cap does not matter either way, so it stays 8 and the machinery
is not built. (The first sweep, with every read traced, showed no effect either, for the wrong reason.)

**End to end (`tt-mnist`, card 0, release, the safe and the `fast-reads` firmware embedded in otherwise identical binaries, alternated, three
runs each; built with `cargo build --release -p tt-mnist -p tt-firmware-images --features tt-firmware-images/fast-reads`).**

| Workload | safe | fast | change |
|---|--:|--:|--:|
| transformer training, 1 tile, 50 steps | 104.7 ms/step (104.6, 104.8, 104.8) | **89.9** (90.1, 89.8, 89.7) | **-14%** |
| transformer training, all tiles | 416 ms/step (413-420) | 434-444 (418-431 in other runs) | +4-6% slower |
| MNIST training, 1 tile, 300 steps | 4.5 ms/step | 4.5 | none (the host bounds it) |
| MNIST training, all tiles | 12.7 ms/step | 13.2 | +4% slower (one run each) |

The fast mode helps where a step is bound by the device reading a tile's operands (one tile, 14%) and loses a few percent at all tiles.
**That loss is not the host's encoding** (my first reading, from a run with encoding switched off, was noise): timing the host's phases in the
all-tiles transformer (temporary counters in `DataMover`, reverted; 50 steps) found **17.7 million enqueues, 355,000 a step -- one per tile per
op -- taking 19.8 s in the safe build and 20.3 s in the fast one, ~390-400 ms of every ~416 ms step**: the host's PCIe writes of the
list, the queue slot and the head to each of 120 tiles, for each of ~3,000 ops. Encoding is 0.17 s of that in total (3.4 ms a step, 0.8%);
list checking ~1 s in both. The fast build's ~20 ms a step over the safe one is mostly the same per-enqueue work (the encoder, and the
`enqueue_checked` walk over the entries' op words) multiplied by 355,000, not a new cost. The all-tiles transformer is bound by the number of
enqueues; the fix is fewer of them (a replayed trace, or a tile subset per op), not a cheaper entry. The one-tile run is not bound by them
(~3,000 enqueues, ~3 ms a step): its ~100 ms is the device busy.
All of these runs are **an order of magnitude slower than the scoreboard's** (transformer
1 tile 8.8 ms/step, MNIST 1.0-1.1) and the same on the commit that recorded them (`7495436`: 329 ms/step), on the branch's merge-base
(`7f18dde`: 105) and on both cards, after a reset of both boards; the host (`burn-flex` 3.7 ms/step) and the cards' PCIe links (Gen4 x16
today, 32 GT/s capable) look normal and the micro-benchmarks' cycle counts are as before. **Unexplained**, and independent of this branch: the
step is the device busy (a 40 ms wait for the last download, 64 ms of caller-side time), so these absolute numbers are comparable only with
each other, not with the scoreboard.

**Per entry, a resident list rerun from the sequence word alone** (no upload; host clock less an empty list's time; it
reproduces the 359 device cycles the probe measured by the tile clock), 256 reads, cap 8, card 0, cycles per entry:

| Entry | 64 B | 4 KiB | run |
|---|--:|--:|---|
| plain `READ` (safe mode) | 359 | 359 | `1791904598` |
| **fast: checksum** (rotate-and-XOR form) | 175 | 175 | `1791904598` |
| fast: checksum (multiply form, shipped) | 176 | 176 | `1791904722` |
| fast, the check removed (not a build) | 138 | 138 | `1791904816` |
| **fast + guard** | 215 | 215 | `1791904922` |

Freshly uploaded lists (`submit_list` + `wait`): plain 651, fast 481, fast + guard 521. The upload (~300 cycles an entry, 32 B
written over PCIe) is now more than the card's time.

- **The check costs ~37 cycles however it is written.** The rotation form and the multiply form cost the same, and 138 cycles
  without either: it is two more words read and combined, not the arithmetic. The semantic guard on top is ~40 more.
- **What a checksum protects.** Any one word damaged, any two swapped (the check word included), a hand-made entry that did not
  come from the encoder (it has the right check only by computing it). It cannot see an entry that is consistent and wrong, which
  only a bug in `encode` makes; the guard is the independent check for that. A first checksum, a plain XOR of an unrotated word,
  let a swap of that word with the check pass: found by the unit test and fixed (every word is mixed with its own constant).
- **Negative controls, watched failing.** With the checksum verification removed, `the_card_refuses_a_damaged_encoded_read_...`
  fails (ttsim) and the other fast tests pass; with the guard removed in an `encoded-guard` build, `the_guard_refuses_a_
  consistent_but_wrong_encoded_read` fails and the others pass. In safe mode the gate holds the opposite: the card refuses a
  *valid* encoded entry. On silicon the entries sent past the host are only ones whose acceptance would still be a valid read into
  the test's own L1/DRAM (a moved address, a flipped checksum bit, a halved length): a bad coordinate is the hang `AGENTS.md`
  warns of, so those are held on ttsim and by `encoded`'s unit tests (the code the card runs).
- An earlier layout (the seven NIU words, and a guard only) measured 194 with the guard and 141 without (runs `1791900861`,
  `1791900958`); the guard written branch-free cost the same as the `&&` chain (`1791901089`), and alone the bitmap lookup
  added 24 cycles and the eight compares 36 (`1791901194`, `1791901286`).

**Real flows, fast mode against safe** (`1791905017` against `1791901818`; host-timed unless marked; card 0):

| Benchmark | safe | fast |
|---|--:|--:|
| `mover_read_sweep`, 64 B to 4 KiB, cycles per entry (device) | 374 | 191 |
| `mover_read_sweep`, 16 KiB entries, cycles per entry | 387 | 355 |
| `mover_overhead`, 256 x 4 KiB, untraced | 128.3 us | 96.2 us |
| `queue_steady_state`, 16 x matmul 32x256x32, host per op | 137.1 us | 123.4 us |
| the same, device first list -> last per op | 128.9 us | 115.5 us |
| `workflow_ops`, matmul 512x512x512 HiFi4 | 1869 us | 1871 us |
| `workflow_ops`, sum rows 64 tiles on sfpu | 789 us | 770 us |
| `copy_pipeline`, 16 to 120 tiles | | within 3% |

Counting entries per upload in six ttsim Session gates, most lists held no plain `READ` at all: their reads are record entries
(`GATHER`, `READ_RUN`), which expand on the card and are not touched; a few packets held 180 to 256 plain reads. Where a flow's
reads are plain entries it gets the saving; where compute or the host dominates it gets nothing. With encoding forced on for
every mover (the earlier layout, with its guard), the whole smoke selection passed on card 0: 1,026 of 1,026.

**Card-wide reads by entry size and tile count** (`gddr_aggregate`, the Read mix: every tile reads its own lists at once; device
clock, GB/s, % of the card's 512 GB/s; `AGG_LEN` sets the entry size, safe image against `--feature fast-reads` on the same
card; the 64 KiB baseline of the scoreboard is unchanged, 466 to 496 GB/s in both). Safe -> fast:

| Tiles | 2 KiB entries (card 0) | 4 KiB (card 0) | 16 KiB (card 1) |
|--:|---|---|---|
| 1 | 7.4 -> 14.4 (1.4% -> 2.8%) | 14.7 -> 28.8 (2.9% -> 5.6%) | 58.2 -> 85.1 (11% -> 17%) |
| 4 | 29.4 -> 57.1 | 58.6 -> 114 (11% -> 22%) | 231 -> 330 (45% -> 65%) |
| 8 | 58.8 -> 114 | 117 -> 227 (23% -> 44%) | 463 -> 452 (90% -> 88%) |
| 16 | 117 -> 226 (23% -> 44%) | 234 -> 344 (46% -> 67%) | 483 -> 482 |
| 32 | 233 -> 290 (46% -> 57%) | 407 -> 446 (79% -> 87%) | 484 -> 484 |
| 64 | 339 -> 392 (66% -> 77%) | 456 -> 464 (89% -> 91%) | 488 -> 489 |
| 120 | 392 -> 442 (77% -> 86%) | 479 -> 489 (94% -> 96%) | 496 -> 496 |

Runs, safe / fast: 2 KiB `1791907945` / `1791908194`; 4 KiB `1791907286` / `1791907626`; 16 KiB `1791907949` / `1791908197`.
Per tile the encoded entry is 1.95x at 2 KiB and 4 KiB (the entry costs 359 against 175 cycles, whatever its size) and 1.45x
at 16 KiB, where the NoC begins to share the time. **Where it matters:** the smaller the entry, the more tiles it takes
to fill the card: 16 KiB entries fill it with 8 tiles in either mode, 4 KiB entries with about 64, and 2 KiB entries
(BF16 tiles) not at all in safe mode (77% with all 120) and 86% in fast mode. On one DRAM channel, 4 KiB entries
fill it (89%) with 2 tiles in fast mode and 4 in safe. The 16 KiB column ran on card 1 and the others on card 0 (two jobs at
once, one card each); a size is compared only with the same size on the same card. The post-run data check of
`gddr_aggregate` compared 64 KiB whatever the entry size; it now compares the entry size run.

**Not changed, and what else this run met** (none of it related to the change):
- Trace lists in GDDR (`CALL` replays) hold plain entries; encoding them is open.
- `silicon_bench_path::matmul_pipeline_sweep` fails with `event 119: an entry began outside a list or inside one`, with the
  committed `mover.rs` as well; it is the role-profile export problem `streaming-dataflow-architecture.md` ("Profiling")
  already records for `sfpu_pipeline_sweep`. `silicon_bench_memory::gddr_aggregate_nocs` fails with `Invalid(10)`
  (`DIRECTION`): it still asks B to write, which the reader/writer split removed.

## Read-loop probe: what the interpreter costs per read entry (2026-10-13, card 0)

`silicon_probe_read_loop::{interpreter_cap_sweep, bare_loop_against_interpreter}` (release, card 0, AICLK 1350 MHz, branch
`fw-architecture-review`; runs `1791859752` (cap sweep, bare loop), `1791859931` (adds the decoded mode)). One tile, one
channel, 256 reads of 64 B or 4 KiB per list. The bare image is `read_loop_b` (`tt_isa::noc::read_loop`), timed by the tile
wall clock around nine repetitions after a warm-up. Every repetition's window checksum is compared with the expected
bytes (the window is poisoned between repetitions) and the last window byte for byte; a probe with the last request
dropped must fail, and does, in every mode.

**Device cycles per request, cap 8 / cap 128** (4 KiB; 64 B):

| Path | cap 8, 4 KiB | cap 8, 64 B | cap 128, 4 KiB | cap 128, 64 B |
|---|--:|--:|--:|--:|
| bare, `issue_dram_on`, address stepped | 95.4 | 90.5 | 88.5 | 78.1 |
| bare, `issue_read` (X6 fast path) | 88.6 | 69.0 | 88.6 | 51.1 |
| replay of encoded requests from L1 | 105.9 | 101.3 | 89.6 | 89.1 |
| `Entry::decode` + `DramMove` + issue per entry | 188.8 | 188.4 | 180.6 | 180.3 |
| shipped mover (list-level stamps, a temporary build) | 359.3 | 358.9 | | |

The temporary build (not committed) also stamped around `exec`: `exec` 312, the list loop outside it 59, a `WAIT` entry 139
in all (151 with the stamps). The host-timed numbers of `mover_read_shapes` and `interpreter_cap_sweep` are 648 cycles
(0.48 us) per entry: the 359 device cycles plus ~290 cycles (0.21 us) for the host to write the 32-byte entry into the
list ring over PCIe, which a host-timed list includes. The same split fits a `WAIT` entry (0.31 us host, 0.11 device).

**In-flight cap sweep** through the shipped mover, host-timed us per entry (64 B / 4 KiB): cap 1 0.585 / 0.680, 2 0.518 /
0.520, 4 0.492 / 0.496, 8 0.479 / 0.483, 16 0.481 / 0.479, 32 0.477 / 0.479, 64 0.477 / 0.475, 128 0.477 / 0.475.

**Read, and what it refutes.**
- The X6 section below concluded that a request's cost is its round trip and that more requests in flight is the lever. The
  X6 measurement stands (the register writes are ~17 ns of an entry); the explanation does not. Past cap 4 more requests in
  flight change nothing, and a bare loop issues a 4 KiB read in 95 cycles at the same cap: the entry is software, 359
  cycles of it, about 95 of them decode and validation and the rest the loop and `exec` around them. Roadmap X7's
  "the cost is the request round trip" is corrected accordingly.
- A replay of host-encoded requests (an entry of the op word and seven request words: the same 32 bytes) is 3.4x under the
  shipped path with the rest of the interpreter intact.
- The probe's device cycles set against the interpreter's host-timed cycles would read 7x; that mixes clocks. The like-for-like
  gap is 3.4-4x; host-visible per fresh list it is smaller, 648 -> ~390 cycles (1.7x), because the upload stays.
- Not measured: more than one tile (the card's GDDR bandwidth limits there: 452-480 GB/s at 4 KiB entries), NC's writes, other
  entry kinds, any model.

## Open-all made strictly better (2026-10-14, cards 0 and 1)

The [per-op tile spread](#per-op-tile-spread-2026-10-14-card-1) left two costs of opening every tile: per-op work that still looped over
every open tile (~5% of a transformer step), and a catch-up that was a list of its own, which made a spread that varies per op lose to churn.

**The changes** (`Session::enqueue_work_inner`):
- Everything an op lays out is sized by the units it runs on: its queues, the units that arrive at its barrier, what each must catch up
  on and which barriers were folded. A capture still involves every unit, as before. Nothing on the fresh path loops over the session's
  other tiles per op.
- A unit that sat ops out catches up inside its first list for the op (`lead_with`): the wait-only `BARRIER` goes first in a packet's
  reader section (its count in the header raised; NC's writes wait on credits only the reader and the roles release, so they follow it
  too) or first in a plain list, with the list's kernel and wait indices moved past it. Only a list already `LIST_MAX` long gets a wait of
  its own, still for the target taken before the op's own barrier was counted. Buffers freed while lists may read them are given back
  only at the next sync, so a write never lands under another unit's read: ordering a unit's reads after the wait is what is needed.

**`tt-mnist`, card 1, one run each (ms/step):**

| Session, spread | transformer | MNIST (300 steps) |
|---|--:|--:|
| 2 tiles, all | 79.8 | 3.5 |
| 8 tiles, all | 81.2 | 2.2 |
| 120 tiles, all | 89.3 (96.2-96.8 before) | 6.4 (6.7) |
| 120 tiles, spread 2 | **77.9** | 3.5 |
| 120 tiles, spread 4 | 79.3 | **2.2** |
| 120 tiles, `work:64:2` / `work:256:4` | 77.8 / 77.2 | 3.0 / 3.0 |

- **120 tiles at a fixed spread now match or beat the best small session** in the same run: the transformer 77.9 at spread 2 against 79.8 on a
  2-tile session, MNIST 2.2 at spread 4 equal to an 8-tile session. (The 2-tile transformer measured 73.9-75.7 in earlier runs, so run to run
  is a few ms.) Every tile with no spread is better too: 96 -> 89 and 6.7 -> 6.4.
- **A catch-up costs no list**: the barrier stage is 0.9 ms over a 300-step MNIST run under either a fixed or a varying spread (24.5 ms before at
  `work:64:4`).
- **Over 300 steps a varying spread still looks worse for MNIST** (2.8-3.2 against 2.2), but that is one-time: starting the movers on the
  tiles a large op first reaches (~240 ms of enqueue time outside every stage, 116 tiles) and their first programs and descriptors (drains 134
  against 16). Over 1,000 and 3,000 steps `work:64:4` is 2.4 against 2.3. `work:16:4` stays at 2.7-2.8: it gives a matmul ~13 tiles at ~5 us of
  host each, so the share has to be large enough for the host's cost a tile.

**Mixed op sizes** (`silicon_bench_spread::small_and_large_ops_by_spread`, card 1, 120 tiles, medians of 7, run `1791988111`): 64 chained
relus of a 64x64 tensor, host-timed per op, and one 2048x2048 add to its end.

| Spread | small op | large op |
|---|--:|--:|
| `All` | 22.7 us | 1,107 us |
| `Fixed(4)` | 19.6 us | 1,054 us |
| **`Work { share: 64, min: 4 }`** | **19.4 us** | **725 us** |
| `Work { share: 16, min: 4 }` | 19.4 us | 1,104 us |

A spread by work is as good as the best fixed one on small ops and 1.45x faster on the large one: `All` is host-bound (120 tiles at ~5 us),
`Fixed(4)` device-bound (1,024 tiles each), and a share of 64 gives the add 64 tiles. A share of 16 clamps to 120 and is host-bound again.

Gates: `step176` holds the fold -- with the wait dropped from the list the reader reads before the write and fails; with every wait a list
of its own the units that caught up show one more list than unit 0 (`[1, 2, 2, 2]`) and fail. The default (`All`) folds too -- an op
with fewer jobs than tiles leaves tiles behind -- so the whole set ran again: 45 simulator binaries, `step12_mnist` with `e2e` (8/8),
and the smoke selection on card 0, 1,027 of 1,027 (run `1791988149`).

## Per-op tile spread (2026-10-14, card 1)

After the [barrier change](#barriers-on-the-units-an-op-ran-on-2026-10-14-card-0) a tile an op does not use costs nothing, so a
session can open every tile and give each op only some. **The best session-wide tile count differs by workload** (card 1, release,
`tt-mnist`, one run each; ms/step):

| Session tiles | 1 | 2 | 4 | 8 | 16 | 32 | all (120) |
|---|--:|--:|--:|--:|--:|--:|--:|
| transformer, 50 steps | 104.3 | **73.9** | 79.2 | 82.0 | 81.9 | 85.5 | 96.8 |
| MNIST, 300 steps | 4.6 | 3.5 | 2.2 | **2.1** | 2.8 | 3.6 | 6.7 |

**The change.** `Session::set_spread` (`tt_kernels::session::Spread`; burn-tt `TT_SPREAD`): `All` (every unit, the default, as
before), `Fixed(k)`, or `Work { share, min }` (one unit for each `share` of the op's work, at least `min`). The work is output tiles
for an element-wise op, input tiles for a reduction, tiles moved for a copy, gather or transfer, and tile products `mt * nt * kt` for
a matmul. Each op builder gets the unit count it builds for, and `execute_on`/`submit_jobs_on` carry the same count to
`enqueue_work`, which deals the jobs round-robin over those units; builders that take no unit count (`repack`, `repack_many`,
`gather_indexed`, ...) are sized by their output (or their job count) and dealt over the units the spread gives them. Jobs of one op
are independent -- two jobs on different units already have no order between them -- so dealing them over fewer units only adds order.

**What it took to make a fixed spread on 120 tiles match a small session.** The first sweep (spread 2 at 120 tiles: 93-96 ms/step
against 74 on a 2-tile session) had two causes, both found with the new `TT_HOST_TIMES=1` (burn-tt prints `Session::host_times`
and lists per tile when it detaches):
- `repack`/`repack_many` (918 calls a transformer step), the host DMA transfers and the matrix-unit ops dealt their jobs over every
  unit whatever the spread; now they follow it.
- `enqueue_work` laid out one segment queue for every unit of the session on every op, empty or not: `Segments` 1,034.5 ms over 50
  steps against 238.9 on the 2-tile session, ~7 us an op against 1.6. Now one queue a unit the op can run on: 388.5 ms.

**At 120 tiles, after both** (card 1, one run each, ms/step):

| `TT_SPREAD` | all | 2 | 4 | `work:16:2` | `work:64:2` | `work:256:2` | `work:16:4` | `work:64:4` |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| transformer | 96.2 | 80.3-80.9 | 80.7-82.0 | 79.7 | 80.6 | 80.6 | 80.6 | 79.0 |
| MNIST | 6.7 | 3.6-3.7 | **2.2-2.3** | 3.1 | 3.0 | 4.3 | 3.2 | 2.9 |

- **A fixed spread on every tile is close to the best small session**: MNIST 2.2-2.3 at spread 4 (2.1 on an 8-tile session, 6.7 on
  every tile), the transformer ~80 at 2-4 (74-76 on 2 tiles, 96.8 on every tile). The ~5 ms a transformer step that is left is the
  per-op work that still scales with the session (per-op vectors of every unit, the barrier loops): 4,069 ms attached over 50 steps
  against 3,832 on 2 tiles, 388 ms of `Segments` against 239.
- **The transformer is bound by its lead tile.** At spread 2 tile 0 takes 151,737 of 283,024 lists and the host waits for ring room
  (394k reads in `List`); a wider spread does not move it (spread 4: 81-82), so what is left there is per-op device time, not the host.
- **A spread that varies per op loses to churn.** `work:16` with no floor put nearly every transformer op on one tile, as slow as a
  one-tile session (106 ms; 154k lists on one tile). A floor (`work:<share>:<min>`) fixes that (79-81) but does not beat a fixed
  spread, and for MNIST it is worse (2.9 at `work:64:4` against 2.2): its lists land on all 120 tiles, units that join a large op and
  sit out the small ones rejoin with a catch-up each time (barrier writes 37,311 against 13,503 over 300 steps) and programs are
  uploaded to more tiles (`Place` writes 2,037 against 907). Varying the spread per op would pay only with a cheaper catch-up (folded
  into the unit's next list) and program placement that knows which tiles hold what.
- **The default is unchanged** (`All`; and burn-tt's auto-attach still opens one tile). Opening every tile with a fixed spread of 4
  would take the transformer from 104 to ~80 ms/step and MNIST from 4.5 to ~2.3 against today's one-tile default; it changes every
  model's tile count, so it is a decision, not part of this change.

Gates: `step177_op_spread` (ttsim and card 0): six ops (a matmul with each operand transposed and `K` in blocks, a row-broadcast add, a
relu, sums over rows and columns, a copy) against the host bit for bit under `All`, `Fixed(1)`, `Fixed(2)` and two work spreads, with
the units each spread chose the only ones that enqueued anything; watched failing with the spread ignored. With every session forced to
`Work { share: 16, min: 1 }`, every numerical check of the 41 multi-tile simulator gates passes (only the checks that count units or
arrivals differ, by design: `step20`'s "every unit worked", `step34`'s arrival count, `step176`'s reader on all four units).

## Barriers on the units an op ran on (2026-10-14, card 0)

The host-time split (`silicon_bench_path::host_time_per_op`, run `1791946223`) showed what the all-tiles transformer's 355,000
enqueues a step were: every op on a multi-tile session ended with a `BARRIER` on **every** unit, and a unit the op gave no job to
got a barrier list of its own -- one PCIe enqueue (three posted writes) a tile an op. A 64x784 add on 120 tiles runs on 50 of them
and paid 70 us and 210 writes for the other 70; the transformer's ops are 2-16 tiles, so nearly all of its 120 enqueues an op were
those lists.

**The change.** The barrier counter now counts arrivals, and an op's barrier adds only the units it runs on (`Session::barriers` is
the latest target). A unit that sat ops out catches up before its next list with one wait-only `BARRIER` for the latest target
(`Session::catch_up`; `dm::op::BARRIER` word 4 = 1 polls without the atomic increment), which the units that ran those ops reach
only after their writes have landed. The catch-up is taken *before* the op's own barrier is counted -- that target includes the
unit's own arrival, which it would otherwise wait for in vain -- and a successful `sync` marks every unit caught up. A capture's
barriers still involve every unit, so a trace's streams and its `k * n` targets are unchanged; a replay catches its units up first
and advances the count by `barriers * n`.

Card 0, release, `tt-mnist` from the commit before (`8205dce`) and with the change, alternated, three runs each (same session):

| Workload | before | after |
|---|--:|--:|
| transformer training, all tiles, 50 steps | 404.2 / 407.9 / 422.2 ms/step | **96.3 / 96.2 / 97.8** |
| transformer training, 1 tile | 105.1 / 105.0 / 105.2 | 105.3 / 105.5 / 105.3 |
| MNIST training, all tiles, 300 steps | 12.9 / 13.1 / 13.4 ms/step | **6.7 / 6.7 / 6.7** |
| MNIST training, 1 tile | 4.5 / 4.5 / 4.5 | 4.5 / 4.5 / 4.5 |

- **The all-tiles transformer is 4.2x faster, and now faster than one tile** (96-98 against 105 ms/step) for the first time. Its losses
  are the same to every printed digit (card 4.4355 -> 3.4883), and MNIST's test accuracy is the same (89.63%) at all tiles.
- **Per op** (`host_time_per_op`, run `1791947516`, against `1791946223`): at 120 tiles the barrier stage is 70.3 -> 0.7 us for the
  64x784 add and 54.9 -> 0.6 for the 256^3 matmul, writes an op 360 -> 150 and 336 -> 192, and the call 331 -> 276 us and 379 ->
  328. An op that fills every tile (the 512^2 add on 120, everything on 8 and 32) is unchanged, as it should be: every unit arrives.
- **What is left at many tiles** is each *participating* unit's own cost, ~4.6 us a unit an op (segments, placement, reserve, check,
  three posted writes): the 64x784 add still takes 50 tiles for 50 output tiles and 276 us. That is plan item 1a (an op on as many
  tiles as its work pays for), and why all-tiles MNIST (6.7 ms/step) is still slower than one tile (4.5).

Gates: `step176_barrier_participants` (ttsim and card 0). A slow op on unit 0 alone (one output tile, `K` 1024), then an op on four
units that gathers its output as `B`: the result is the host's product bit for bit on binary operands (exact in TF32 and FP32) over
three rounds; the slow op ran on one unit and the others enqueued nothing for it; the reader ran on all four. **With the catch-up
skipped, the reader reads the slow op's output before it is written and the gate fails** (round 0, ttsim). A second test captures and
replays a trace between partial ops (behind units before the capture and between replays) against the host. The `BARRIER` decode
refuses a word 4 other than 0 or 1. All 39 multi-tile simulator gates, `step12_mnist` with `e2e` (8/8, golden unchanged), and on card 0
`step33_barrier`, `step39_traces`, `step60_streaming` and `step176` pass. The smoke selection on card 0: 1,027 of 1,027 (the five
two-card tests first failed to attach because a benchmark held card 1, and passed when rerun with it free).

## Descriptor runs in a trace's prologue (2026-10-14, card 0)

The first thing the [replay breakdown](#trace-replay-where-a-replayed-ops-device-time-goes-2026-10-14-card-0) found: every replay
re-establishes the three roles' descriptors with 51 `op::POKE` entries, 8,824 device cycles (6.5 us), the same on every replay.
Each `POKE` writes one word and fences it (`publish`), so the cost is seventeen entries and seventeen fences a role.

`record::POKE_RUN` writes a run of consecutive role-mailbox words in one two-entry record, with one fence for the run
(`mover::poke_run`). A descriptor is three runs -- the program's address and length divide it, since each `KERNEL` and `LAUNCH`
writes those itself, and the live handshake words (`generation`, `ack`) sit between two of them -- so the prologue is 9 records
of 2, 4 and 11 words instead of 51 entries. `trace::UnitCapture::poke_descriptor` emits them; a run of one word stays a `POKE`,
which is the smaller of the two.

Card 0, release, safe image, `silicon_probe_trace_replay::replay_breakdown`, medians of 7 (runs `1791944369` before,
`1791945827` after; the before column is the published breakdown):

| us a replay | chain8, 1 tile | chain8, 8 tiles | layer, 1 tile | layer, 8 tiles |
|---|--:|--:|--:|--:|
| **host-timed, unstamped**: before -> after | 73.5 -> **71.2** | 79.0 -> **76.3** | 129.0 -> **124.4** | 82.5 -> **78.2** |
| device span: before -> after | 69.8 -> 68.1 | 59.3 -> 56.3 | 125.0 -> 120.4 | 64.1 -> 60.0 |
| the prologue in it | 6.54 -> **3.62** | 6.54 -> **3.63** | 6.54 -> **3.63** | 6.54 -> **3.62** |

- **The prologue is 8,824 -> 4,892 cycles (6.54 -> 3.62 us), and a replay is 2.3-4.6 us shorter by the host's clock.** The host
  number is the one to read: the device span also falls because there are 84 fewer stamps (~12 cycles each), which the host-timed
  runs do not pay.
- **A record is ~544 cycles against ~173 for a `POKE`**, so it pays from three words up; the 2-word run is a wash and is left as a
  record for one rule rather than two (worth ~0.4 us a replay to split, not measured).
- **The stream is 124 -> 91 entries and its top level 57 -> 24** (`step174_trace_footprint`), which puts the layer's top-level
  stream back under one 64-entry chunk: the 0.75 us chunk boundary an 8-tile replay paid is gone, and the padding stays zero.
  The pinned program cache and the GDDR a trace holds do not change.
- **What is left of the prologue** is 9 records, 544 cycles each. Writing the descriptor as one run would need the program's
  address and length in it, and those change per op, so the run would be re-emitted for every op rather than once a trace.

Gates: `step175_poke_run` (ttsim) holds the firmware's own loop to the `POKE`s the record expands to -- the same words through
one entry each, bit for bit in all three mailboxes, at every length a record carries -- and watched failing with the loop one word
short. Its controls: a count of none or past the record, a value past the count, an address outside a role mailbox, a run that
starts inside one and ends past it, and NC, which refuses the record as it refuses a `POKE` (`Mover::permits`); each is refused by
the host with the code the tile gives, and the tile, sent it past the check (`enqueue_raw`), writes none of the words.

## Trace replay: where a replayed op's device time goes (2026-10-14, card 0)

`silicon_probe_trace_replay::replay_breakdown` (and `replay_dump`, the raw stamps), release, card 0, safe image, plain traces, branch
`fast-mode-hardening`; runs `1791944369` (breakdown) and `1791944173` / `1791944389` (dumps, 1 and 8 tiles). One replay is profiled at
a time (`profile_start`, `replay`, `sync`, `profile_stop`; the roles do not stamp, `set_profile_roles(false)`), unit 0's mover stamps
are split by what lies between two of them, and the parts are checked to sum exactly to the replay's `LIST_BEGIN`..`LIST_END` span
(a replay whose stream is more than 64 entries is several lists: the gaps between them are the chunk boundaries). Medians of 7.
Two traces: the MNIST-size layer forward (matmul, add row, relu, matmul) and **chain8**, eight relus of one 64x64 tensor, which has
next to no compute or data, so what is left is the replay's own cost.

| Device us per replay (unit 0) | chain8, 1 tile | chain8, 8 tiles | layer, 1 tile | layer, 8 tiles |
|---|--:|--:|--:|--:|
| **span** (per op) | **69.8** (8.73) | **59.3** (7.41) | **125.0** (31.2) | **64.1** (16.0) |
| host, `replay` + `sync`, unstamped | 73.5 | 79.0 | 129.0 | 82.5 |
| `POKE` prologue + setup `KERNEL` | 7.2 | 7.2 | 7.8 | 7.8 |
| chunk boundary (`LIST_END` to next `LIST_BEGIN`) | 0 | 0.75 | 0 | 0 |
| pair begin (first op) + pair gap (later ops) | 22.1 | 7.5 | 10.2 | 4.4 |
| `LAUNCH` | 1.9 | 1.9 | 0.9 | 0.9 |
| `CB` credits | 5.6 | 5.4 | 3.3 | 2.7 |
| data records (`READ_RUN`, `GATHER`) | 12.9 | 6.7 | 63.7 | 19.6 |
| `KERNEL_WAIT` (the roles' run) | 16.7 | 7.2 | 34.4 | 9.7 |
| `BARRIER` | 0 | 9.7 | 0 | 12.7 |
| loop gaps between entries | 1.4 | 11.9 | 2.9 | 6.2 |
| last entry to `LIST_END` | 2.0 | 0.1 | 1.6 | 0.05 |

- **A tiny streaming op costs 7.4-8.7 us on the device, about half of it control.** In chain8 on one tile, data (2.2 k cycles an op) and the
  roles' run (2.8 k) are 42% of an op; pair begin/finish, launch, credits, loop gaps and the last-entry gap are the other ~4.1 us. The
  steady-state op is ~10.6 k cycles (7.8 us) after the prologue.
- **A streaming op's fixed costs, in cycles** (dump, 8 tiles, steady state): starting the `PAIR_CALL` ~1,420 (its mailbox handshakes
  and the reader section's fetch from GDDR; the same ~1,405 for the first op, where nothing precedes it); `LAUNCH` ~320; two `CB`
  entries ~345 + ~560 (they wait on NC and the roles, so part of that is not overhead); finishing the pair ~1,800 (B waits for NC's
  output writes, which are real data movement, 4 tiles here); `BARRIER` ~800-950 once the tiles are aligned (6,100 the first time,
  which absorbs the skew of the `CALL`s' start); ~55 cycles a gap between entries.
- **The 51 `POKE`s cost ~173 cycles each (116 for the entry, ~58 of loop gap), 6.5 us a replay, 9-11% of a small trace.** They write
  role descriptor words one by one and are the same on every replay. Their stamps are ~12 cycles of the 116.
- **A chunk boundary is 1,012 cycles (0.75 us)**: the progress write, the next 64 entries' fetch from GDDR, the patch pass. Seen where
  the top-level stream (51 `POKE`s, a `KERNEL`, the `PAIR_CALL`s, `BARRIER`s) passes 64 entries (8 tiles); each reader section's fetch is
  inside "pair begin". So the chunk loop is not where a replay's time is: one boundary is under 1.3% of chain8 on 8 tiles. The filler
  `WAIT`s the host pads with are zero in these traces ([trace footprint](#trace-footprint-what-a-trace-costs-a-tile-2026-10-14-ttsim)).
- **On many tiles the host is the larger part.** Host minus device is 3.7-4.0 us on one tile (the 4.25 us round trip, with the device's own
  tail) and 18.5-19.7 us on 8: ~2 us a tile for the `CALL` entry's three posted writes and the sync's per-unit polls. A replay of 8
  tiles is 79 us by the host's clock for a 59 us device span; the device is not what bounds the small ones.
- **The barrier is 16-20% of an 8-tile replay.** 4,300 cycles an op in the layer (it includes waiting for the slowest tile), ~1,600 in chain8.

Reading with its limits: unit 0 only; spans include waiting for the other tiles at barriers; NC is not stamped, so the pair gap
and the last-entry gap include the wait for its writes; stamping costs ~12 cycles a stamp (204 stamps, ~2.6% of chain8's span, and
host minus device agrees with the known round trip, so it is not distorting the shape); the safe image, plain entries, one card.

## Trace footprint: what a trace costs a tile (2026-10-14, ttsim)

`Session::trace_footprint` and `step174_trace_footprint` (ttsim; the counts are host-side and do not depend on the card, so silicon
reads the same for a given tile count). A trace is bounded by two things per tile, the program cache (`l1::PROGRAM_CACHE`, 240 KiB of
L1, every program a trace names pinned for its life) and its streams in GDDR. Branch `fast-mode-hardening`, safe image, plain traces.

| Capture | Entries (top level + sections) | Padding | Entry bytes | GDDR allocated | Programs pinned |
|---|--:|--:|--:|--:|--:|
| layer forward 64x784x128x10 (matmul, add row, relu, matmul), 1 tile | 124 (57 + 67) | 0 | 3,968 B | 36 KiB | 28, 16 KiB (6.9% of the cache) |
| the same x2 / x4 in one trace | 195 / 337 | 0 | | 69 / 134 KiB | 28, 16 KiB (unchanged) |
| the same, 2 tiles (each) | 116 | 0 | 3,712 B | 36 KiB | 26, 9 KiB |
| two layer shapes in one trace (x2), 1 tile | 221 | 0 | 7,072 B | 69 KiB | 49, 35 KiB (14.9%) |
| add 2048x2048, 1 tile | 1,668 | 16 (1.0%) | 52 KiB | 77 KiB | 23, 59 KiB (24.6%) |

- **Programs are shared by hash.** The same pass captured again adds entries and pins no program (28, 28, 28), and a pass of another shape
  does (28 -> 49): the control that shows the count can move. So a trace's pinned set is the number of *distinct* (op, shape) programs,
  about 4 KiB each for a layer's four ops (roles and streaming bodies), not the number of ops. At that rate ~60 distinct op shapes fill the
  cache, which is the shape of the model-ladder overflow (roadmap "Trace segmentation"); the per-model count has not been measured.
- **A pass of four ops is ~71 stream entries (~2.3 KiB) but ~32 KiB of GDDR.** Each `PAIR_CALL` stores its reader and writer sections as
  allocations of their own, and an allocation is whole tile slots (4,160 B): a 17-entry section takes a slot. GDDR per op is therefore set
  by the number of allocations (~8 KiB an op here), not by entries. GDDR is not the tight limit (a 2,779-op decode step on one tile at this
  rate is ~22 MB), but it is the one place a trace's size is 13x what its entries need.
- **Chunk padding is small where the sections are short.** 0 entries in the layer forward (every section is under one 64-entry chunk) and
  1.0% in the 2048x2048 add (16 of 1,668 entries), whose sections are long enough to cross chunks. The padding is not a large share of a
  trace's bytes; its cost is the cycles the mover spends running each filler `WAIT` (139 cycles each, from the read-loop probe), which this
  does not measure.
- **The 51 `POKE`s are one per trace, not one per op.** The capture pokes only the descriptor words that changed (`poke_descriptor`), so
  1, 2 and 4 passes all hold 51. A trace prologue for invariant setup would save nothing in entries; what it could save is the
  `KERNEL`/`LAUNCH` round trips, which are per op.
- **Per-op entries are mostly `CB` (credit) records:** 5 a packet in the layer forward, beside a `LAUNCH` and a `KERNEL_WAIT` each and a
  `GATHER`/`SCATTER` or `READ_RUN`/`WRITE_RUN` pair. The 2048x2048 add is `CB` 456, `READ_RUN` 228, `WRITE_RUN` 114 and `RELEASED` 106
  (`op::RELEASED`) in 4 packets.

Not measured: footprints of the three models (they do not capture yet), of CNN training, or on more than 2 tiles.

## Per-tile cost of the movers' records and NC's writes (2026-10-13, cards 0 and 1)

`silicon_probe_read_loop::records_against_entries` (card 0) and `silicon_probe_nc_writes::nc_writes_against_records`
(card 1; `cargo xtask bench --device 1 --filter silicon_probe_nc_writes`, run `1791925106`, release, branch
`fast-mode-hardening`). Resident lists rerun from the sequence word, device cycles by the host clock less an empty list's
time, per tile, N = 64 / 256 / 448 tiles of a tensor over the eight channels. Every repetition poisons the destination,
reruns, and compares every tile's datums (writes: the 4096 datum bytes, header untouched), so a form that is cheap
because it moved nothing fails.

| Per tile, device cycles | N = 64 | N = 256 | N = 448 |
|---|--:|--:|--:|
| B, plain 4160 B `READ` (card 0) | 351-363 | | |
| B, host-encoded `READ_ENCODED` (card 0) | 175-177 | | |
| B, `READ_RUN` record, 32 tiles each (card 0) | 340-351 | | |
| NC, plain `WRITE` (card 1) | 328 | 345 | 343 |
| NC, `WRITE_RUN` record, 32 tiles each (card 1) | 327 | 326 | 320 |

- A record's tile costs what a plain entry does: the record is expanded into the same `DramMove` per tile, so encoding a
  plain entry (175) saves nothing there, and only a validated static record path (plan phase 4/5) can. NC's writes
  cost the same ~330-345 cycles as B's reads; a record saves 0-7% on the write side.
- The probe's first run failed on its own error, not the mover's: a tile's slot holds 4096 datum bytes and 48 bytes of
  padding (`TILE_DATA + 4096 <= TILE_SLOT`), which a `WRITE_RUN` correctly does not write; the probe had compared and written the
  padding. Both forms now move exactly the datums.
- `gddr_aggregate_nocs` is still unusable for NC, and is not fixed here: its lists begin with a `BARRIER`, which only B runs,
  so NC answers `DIRECTION`. The probe above does what it was for without one.

**Where NC's requests come from** (ttsim, temporary instrumentation in `DataMover::check_as`, reverted; the NC list of every
upload, top-level `WRITE` entries against writes a record expands to):

| Gate | Lists | Top-level `WRITE` | Records | Record-expanded writes |
|---|--:|--:|--:|--:|
| `step68_k_block_matmul` | 114 | 0 | 287 | 1,765 |
| `step56_scatter_on_nc` | 64 | 0 | 1,076 | 22,848 |
| `step11_burn` | 91 | 14 | 100 | 367 |
| `step171_select_rows` | 141 | 321 | 87 | 210 |
| `step62_gather_rows` | 97 | 596 | 52 | 212 |
| `step52_nc_mover` (plain writes by design) | 2 | 312 | 0 | 0 |

Matmul, Burn and scatter flows write through records (0-4% top-level); only the row-select and gather gates, whose output rows
are ragged, write plain entries in bulk. An NC fast path therefore has little to win from encoding plain entries (~340 -> ~170
cycles at best) and most from a validated static record path; sealed `WRITE_RUN`/`SCATTER` first, `WRITE_ENCODED` only if a
model flow needs the row writers. This is a tally of gates, not of production flows on silicon.

## Firmware image budget and core-assignment (2026-10-13, card 0)

Why: `dm_b` had 172 bytes left of its 24,576 (silicon-operating-notes, "RISCV B's image has little room
left"). Review and design space: [firmware-architecture-review.md](../proposals/firmware-architecture-review.md).
Conditions: card 0, AICLK 1350 MHz, release, branch `fw-architecture-review` on `7f18dde`; the benches are
`silicon_perf::mover_read_shapes` and `silicon_bench_path::null_kernel`, medians as those tests report them.
Not measured here: live tt-metal/ttnn timings (the installed ttnn 0.72 needs SFPI 7.52.0 and the machine has
7.83.0, so its JIT compile of the dispatch kernel fails; `/home/carl/projects/tt-metal-v077`'s ttnn hangs in device
init). `crates/tt-tests/bench/ttnn_reference.py` is the script for when a matching toolchain exists.

**Whole-image `opt-level`** (the Cargo profile of `crates/tt-firmware`; all five images change together):

| opt-level | `dm_b` | `dm_nc` | `role_t0/1/2` | WAIT entry | 64 B read entry | 16 KiB reads | 64 KiB reads | 1024-NOP role, list begin->end | role push rate | run |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 3 (shipped) | 24,404 | 14,252 | 8,340 / 7,996 / 8,408 | 0.31 us | 0.48 us | 34.0 GB/s | 53.9 GB/s | 3,593 cycles | 2.91 cycles/word | `1791855052` |
| 2 | 22,660 | 13,004 | 8,232 / 7,900 / 8,296 | 0.31 us | 0.48 us | 33.8 GB/s | 53.4 GB/s | 3,604 cycles | 2.93 cycles/word | `1791855317` |
| s | 21,220 | 12,996 | 7,056 / 6,796 / 7,044 | 0.48 us | 0.76 us | 21.6 GB/s | 53.0 GB/s | 22,648 cycles | 21.05 cycles/word | `1791855151` |
| z | 21,164 | 13,824 | 6,872 / 6,500 / 6,920 | 0.52 us | 0.83 us | 19.8 GB/s | 51.1 GB/s | 24,964 cycles | 23.2 cycles/word | `1791855234` |

`opt-level` 2 costs nothing measurable and saves 1,744 bytes on `dm_b`; `s` and `z` save at most 3.2 KB and make
the mover 1.6x slower per entry and the roles' instruction push about 7x slower (the push loop is the
roles' whole job). Size optimisation is not a lever for the roles. `opt-level` 2 was measured on these two
benches only, not adopted: the shipped profile stays at 3.

**Repacked images (V3a), adopted.** The role images are linked in 10 KiB slots packed against NC's slot
(`tt_isa::l1::roles`: T0 `0xA800`, T1 `0xD000`, T2 `0xF800`) instead of at the cores' default reset PCs;
`Device::load_held` already sets the reset-PC override for every core but B, so the host needed no change. B's
slot is now everything below T0's: `dm::IMAGE_MAX` 24,576 -> 43,008 bytes, **+18 KiB of room with the images
byte-identical in size** (`dm_b` 24,404, 18,604 left). No arena or other region moved. Role images use 8.0-8.4 KB
of their 10,240-byte slots. Silicon, same benches: WAIT 0.31 us, 64 B read 0.48 us, 16 KiB 33.7 GB/s, 64 KiB
53.2 GB/s, 1024-NOP list 3,599 cycles, push rate 2.92 cycles/word (run `1791855559`): no change from the
shipped layout. Smoke selection on card 0: 1,030 of 1,030 passed (run `1791855592`, one boot throughout).
ttsim: `step16_dm`, `step52_nc_mover`, `step171_select_rows`, `step24_padding`.

**Negative result: refusing the other direction's ops before decoding them** (`M.permits(w[0])` ahead of
`Entry::decode` in `exec`, so each image could drop the decode arms it never runs). Predicted from symbol sizes:
about 240 bytes on B, 1.9 KB on NC. Built: `dm_b` -4 bytes, `dm_nc` +96. LLVM already shares the decode arms, so
the check only adds a compare chain. Reverted. Symbol-size estimates of what a check will remove are not
evidence; measure the build.

## Row select: an embedding lookup without staging (2026-10-13, card 0)

`select` along dimension 0 of a rank-2 F32, I32 or Bool table now runs `Session::select_rows`
(`crates/tt-kernels/src/index.rs`; B's `SELECT_ROWS` record in `tt_isa::dm::record`): each batch reads
one tile of 32 resident indices, and B turns each into two 64-byte face-row reads a tile column, finding the source
tile on the card, so a lookup reads the rows it returns and builds no host address. Before, `gather_native`
staged every row of the table for every element it returned (576 x 49,152 = 28.3 million host addresses a token).
Gates: `step171_select_rows` (ttsim; card 0 run `1791851883`, 10 of 10), which compares with the host indexing bit
for bit (specials, I32, Bool, views, ragged shapes), refuses an index outside the table with `DOMAIN`, and replays a
trace with changed indices. Release build, git `e17f4c9` plus the change, card 0, nothing else on the card.

### The lookup alone (`probe_embedding_cost`, run `1791851940`)

A 49,152 x 576 table through Burn's own `Embedding`. A warm-up lookup, then seven timed, each of different rows, from
host dispatch through the readback, and each validated bit for bit against the host's copy of the weights.

| Lookup | Before (`1791845218`, 2026-10-12) | After: median of 7 (min to max) |
|---|--:|--:|
| 1 token | 15.9 s | 260.2 us (259.1 to 267.0) |
| 5 tokens | 63.3 s | 278.1 us (273.0 to 371.8) |
| 32 tokens | not measured | 343.7 us (338.0 to 352.1) |

The first lookup of all, 404.5 us, is not a cold table upload: the probe had already downloaded the table
(for the validation), which materialised it. Before, the cost was linear in tokens; now a token adds 3 to 5 us
(4.5 us from 1 to 5 tokens, 2.7 us from 1 to 32) to the fixed cost of a dispatch and a readback.

### SmolLM2-135M on card 0

| | Before | After | Run |
|---|--:|--:|---|
| `step167_smollm2` F32, 30 layers, greedy 16 tokens, agrees with burn-flex | 575 s (`1791844568`) | 33.2 s | `1791851959` |
| Decode step from a token id, median of 9 (p10 to p90) | about 14 s a token (`step167_smollm2`, 2026-10-12; the lookup alone was 15.9 s) | 623.1 ms (525.4 to 709.2), 2,781 reported ops | `1791852097` |
| The same step, the lookup excluded | 619 ms (card 1, `1791846823`) | 621.7 ms (525.7 to 707.4), 2,779 reported ops | `1791852097` |
| burn-flex, the same steps, same run | 25.0 ms | 26.4 ms from a token id, 26.2 ms without the lookup | `1791852097` |

So the lookup is now about 1.4 ms of a 623 ms step, and a decoded token is back to what the roadmap's item 3 describes:
about 24x burn-flex, host submission (2,781 operations at about 220 us each). The two decode numbers before and
after come from different cards (1, then 0); the with-lookup and without-lookup rows of one run are the comparison.

### What did not change

BF16, F16 and BFP tables, tables of rank 3 and up, other dimensions, and `gather` with per-element indices still take
the staged path (`gather_native`), and their gates pass unchanged on card 0 (run `1791852139`): `step167` BF16, six layers, 229 s (441 s in `1791848957`, a different load; the BF16 lookup was not touched), and all seven `step168` mixed-precision plans, 5 to 101 s each. The B image has no room for a packed-BF16 branch
([silicon-operating-notes.md](silicon-operating-notes.md), "RISCV B's image budget").

## Model ladder: real models on the card (2026-10-12, cards 0 and 1)

First measurements of pretrained models (`crates/tt-models`, `tests/silicon_bench_models.rs`), taken to
order the performance work. Release build (`dd8172b769f1+dirty`), two warm-ups and nine timed forward
passes, medians in milliseconds, timed from host dispatch through the final readback, batch 1, the
photograph `cats.jpg`, F32. Each device number's output is validated outside the timing (the top-1 equals
burn-flex's). The two jobs ran at the same time, one per card, each single-card, which is why the ResNet run
is card 0 (`1791846735`) and the other two card 1 (`1791846823`); a number is compared only with the baseline
measured in the same run, on the same card.

| Workload | burn-flex (host) | burn-tt (card) | Slowdown | Reported device ops | Backend calls | Run |
|---|--:|--:|--:|--:|--:|---|
| ResNet-18, 224x224 | 154.8 ms | 16,288 ms | 105x | 211 | 509 | `1791846735` (card 0) |
| ViT-Tiny/16, 224x224, 12 blocks | 104.7 ms | 815 ms | 7.8x | 709 | 764 | `1791846823` (card 1) |
| SmolLM2-135M decode step, 30 layers, the embedding lookup excluded | 25.0 ms | 619 ms (p10 523, p90 706) | 24.8x | 2,779 | 3,874 | `1791846823` (card 1) |
| SmolLM2-135M, the embedding lookup of one token alone | n/a | 15.9 s | n/a | one gather | | `1791845218` (card 0) |
| SmolLM2-135M in BF16, 6 layers, a decoded token end to end (lookup included) | n/a | 37.5 s (prompt 87-90 s) | n/a | | | `1791848957` (card 0) |

Reading them:

* **The embedding lookup is the SmolLM2 decode.** One token from the 49,152 x 576 table takes 15.9 s and five
  take 63.3 s (`probe_embedding_cost`), so a decoded token costs 14 s end to end (`step167_smollm2`, run
  `1791844568`: prompt 66-69 s, then 13.8-14.3 s a token, growing only with the attention's cache length). The
  cause is in `gather_native`: it builds, for each chunk of 32 output elements, a host address list of
  `32 x rows` entries, so one token costs 576 x 49,152 = 28.3 million host-built addresses. It is the
  roadmap's "Gather without staging", now measured on a real vocabulary.
* **ResNet-18's cost is convolution and pooling, and it is chunking, not the count of reported ops.** The
  211 and 709 above count the top-level operations the report lists as run on the device; one reported
  `conv2d` or `max_pool2d` hides many more submissions. Convolution issues one `float_matmul` per
  `PATCH_ROWS = 32` output positions, each with its own host-built patch gather (`ops/conv.rs:5,181-192`: a
  56x56 convolution is 98 matmuls, conv1 on 112x112 is 392), and max pool splits its windows into chunks of
  32, each a host-built `repack`, an argmax and a gather (`ops/pool.rs:153-200`: 6,272 chunks for 64x112x112).
  The weights are resident (588 KiB goes up a pass, the patch tables). Isolated on card 0 (`1791846963`,
  `probe_resnet_op_cost`, median of three): max pool 3x3/2 on 64x112x112 **2,817 ms**; conv 7x7/2 (conv1)
  1,170 ms; conv 3x3 64->64 on 56x56 895 ms; 128->128 on 28x28 517 ms; 256->256 on 14x14 529 ms; 512->512 on
  7x7 **1,411 ms**; against 0.6 ms for an add + relu on 64x56x56 and 3.8 ms for the global average pool (one
  reported op, one submission each). The layer-4 convolution (7x7, few positions, 2.4 million weights)
  costs more than layer 2's (28x28), so the weight gather per chunk matters as much as the chunk count. The
  warm-up passes did not make a forward faster (p10 16,266 ms, p90 16,334 ms), so none of this is
  amortised across calls.
* **BF16 is slower still.** The same decoded token in the checkpoint's own BF16 costs 37.5 s (six layers, so the
  layers are a small part of it) against F32's 14 s, so the BF16 lookup is about 2.7x the F32 one
  (`step167_smollm2` BF16, run `1791848957`); the whole 30-layer BF16 model did not finish inside 1,500 s
  (`1791847160`).
* **A decoded token is about 2,800 reported device operations** at about 220 us each: the decode step is
  host-bound like the transformer training step was (the report does not see submissions nested inside an
  op, so the true count is higher). `cat` is built from `slice_assign` (23,826 `float_slice_assign` calls in
  33 forward passes, 722 a pass: the nine-way `repeat_kv` and the rotary halves), and `float_reshape`,
  `float_slice` and `float_device` calls number in the tens of thousands.
* **ViT-Tiny is the nearest to usable**: 709 reported operations, about 1.1 ms each.

Not measured: traced execution of any of the three (a decode step has the host cost a trace would remove, but
trace segmentation is open, so a forward this size does not fit the 240 KB program cache), the other card for
a model that ran on card 0 (the single-card rule), and BF16. The same three forward passes are the
correctness gates `step165_resnet18`, `step166_vit_tiny` and `step167_smollm2` (runs `1791843753`,
`1791843886`, `1791844568`).

**A tooling defect found on the way:** two `cargo xtask` runs that finish building in the same second took the
same run ID, so the second collected-benchmark file (`target/silicon/bench/<stamp>.jsonl`) replaced the first
(the per-test output files, named by stamp, device and test, were intact). The stamp is now claimed by
creating its log exclusively (`xtask/src/silicon.rs`, `claim_stamp`), so one job per card is safe. Run
`1791846398` is the pair that collided; its numbers are the first measurement without the burn-flex baselines
and agree with the rerun. The runs `1791846735` and `1791846823` label the operation count "device jobs";
it is the reported-operation count described above, and the benchmark now says so.

**A second tooling defect:** when a test passed its per-test time limit the runner killed only the process it had
started. The gates do their device work in a forked child (`tt_ttsim::fork_scope`), which survived, kept running
and kept holding the card, so the next run failed with "card 0 is in use by another process". Each test now runs
in its own process group and a timeout kills the group (`xtask/src/testrun.rs`; the unit test
`a_timeout_kills_the_forked_child_too` fails without the change).

## Accurate matmul cost (2026-10-12, card 0)

`MatmulMode::Accurate` ([`ops/accurate_matmul.rs`](../../crates/burn-tt/src/ops/accurate_matmul.rs)) runs integer slices with
shared exponents at `Fidelity::Lo` on the Matrix Unit and adds twelve slice pairs, with NaN and infinity handled by three more
0/+-1 matmul pairs. It is built from the existing resident ops, so it submits many jobs: one `[64,96]@[96,48]` is **127 device jobs**
(101 element-wise, 18 matmuls, 4 reductions, 4 gathers and copies) against 1 for the fast matmul (counted on ttsim with a temporary
counter in `server::submit`, since removed).

**Conditions.** Run `1791818553` (`cargo xtask bench --device 0 --filter step163_accurate_matmul::benchmark`), card 0, release,
`Config::default()` (one tile, HiFi4 for the fast mode, `Tf32FromFp32`). Host `Instant` around `matmul` and the read of its result,
two warm-ups and `REPS` samples, median (p10 to p90). Every sample validated: the first sixteen rows of the accurate result inside the
derived bound of `step163_accurate_matmul` (a bound the fast matmul leaves), the fast one inside a loose sanity bound.

| `m x k x n` | fast (us) | accurate (us) | accurate / fast |
|---|--:|--:|--:|
| 64 x 32 x 64 | 72.8 (70.6 to 76.8) | 1,393 (1,334 to 1,426) | 19.1 |
| 1024 x 27 x 32 | 146.6 (132.3 to 158.0) | 9,444 (9,377 to 9,471) | 64.4 |
| 128 x 256 x 128 | 126.9 (109.1 to 138.8) | 4,996 (4,994 to 5,000) | 39.4 |
| 256 x 256 x 256 | 277.2 (272.7 to 288.3) | 10,964 (10,956 to 10,981) | 39.6 |
| 32 x 784 x 128 | 101.1 (100.5 to 110.5) | 6,081 (6,077 to 6,092) | 60.2 |
| 512 x 512 x 512 | 1,546 (1,545 to 1,562) | 49,586 (49,526 to 49,641) | 32.1 |
| 1024 x 1024 x 1024 | 14,922 (14,787 to 15,007) | 279,692 (279,286 to 279,948) | 18.7 |

For scale, the exact mode (same card, run `1791761944`) took 70 ms at 64x32x64 and 8.2 s at 256 cubed: accurate is about 50x and 750x
faster than exact at those two shapes, and 19x to 64x slower than fast.

**Where the time goes** (a scratch profile on the same card, medians of the same host timing, not kept): limiting the slice pairs to
the leading one (`set_accurate_matmul_slicing(0, 0)`: 11 matmuls and 11 adds fewer, everything else unchanged) gives

| `m x k x n` | 12 pairs (us) | 1 pair (us) | 11 pairs cost | the rest |
|---|--:|--:|--:|--:|
| 64 x 32 x 64 | 1,448 | 1,133 | 315 us (29 us a pair) | 78% |
| 256 x 256 x 256 | 11,027 | 9,523 | 1,504 us (137 us a pair) | 86% |
| 1024 x 1024 x 1024 | 279,266 | 178,449 | 100,817 us (9.2 ms a pair) | 64% |

**Reading.** The Matrix Unit part (the LoFi matmuls) is not what costs: the rest, which is scaling, slicing and the NaN/infinity
classification, all element-wise on one tile's SFPU, is 64% to 86% of the time, and at 1024 cubed 178 ms of it is spent on
operands of a million elements, about 100 element-wise jobs each. Two things would cut it without touching the arithmetic: fused SFPU
kinds that make a weighted slice in one job (one input and one scale in, one slice out, instead of `trunc`, subtract, scale), and a
cheaper classification of NaN and infinity (about 45 of the 127 jobs). Neither is built.

## Exact matmul cost (2026-10-11, card 0)

`MatmulMode::Exact` ([`ops/exact_matmul.rs`](../../crates/burn-tt/src/ops/exact_matmul.rs)) computes F32
products on the SFPU and sums them in `k` order, in place of the Matrix Unit's group adder
([silicon-operating-notes.md](silicon-operating-notes.md), "Matmul accumulation: the Matrix Unit's group
adder"). It is built from existing resident ops, so what it costs is what a gather, a multiply and a
reduction cost.

**Conditions.** Run `1791761944` (`cargo xtask bench --device 0 --filter
step159_exact_matmul::benchmark`), card 0, release, `Config::default()` otherwise (one tile, HiFi4,
`Tf32FromFp32`). Host `Instant` around `matmul` and the read of its result, two warm-ups and nine samples,
median (p10 to p90). Every sample is validated: the exact result bit for bit against the host oracle, the
fast one within a loose sanity bound.

| `m x k x n` | products | chunks | fast (us) | exact (us) | exact / fast |
|---|--:|--:|--:|--:|--:|
| 64 x 32 x 64 | 131 k | 1 | 68.2 (66.3 to 74.9) | 70,430 (70,180 to 70,569) | 1,033 |
| 1024 x 27 x 32 | 885 k | 1 | 149.1 (145.7 to 155.9) | 495,350 (472,299 to 497,103) | 3,322 |
| 128 x 256 x 128 | 4.2 M | 4 | 164.0 (159.9 to 169.8) | 2,109,880 (2,106,610 to 2,113,014) | 12,863 |
| 256 x 256 x 256 | 16.8 M | 16 | 376.2 (341.0 to 387.2) | 8,192,931 (8,184,219 to 8,199,700) | 21,775 |

**Reading.** The exact path costs about 0.5 us per product (0.54, 0.56, 0.50, 0.49) at every size, so it is
linear in products while the fast one is nearly flat: 22,000x at 256 cubed. Whether the host's two address
tables per chunk (a 16-byte entry per product each) or the device's gathers dominate is **not profiled**. The
untried alternatives that need far fewer addresses: a loop over `k` of `acc = acc + a[:, p] * b[p, :]` (about
`m + n` addresses a step, a few device ops a step, the same sum and the same order, so the same bits), or one
output row at a time. Until one of them is built and measured, the exact mode is for validation and for work
that needs CPU-order accuracy, not for training.

## `.ttinsn` push fusion, precise `gelu` (2026-10-10, card 0)

`probe_ttinsn_push` (`docs/plans/ttinsn-push-experiment.md`), release, card 0, the gate tile (3, 4),
built with `--feature ttinsn-experiment`. One image, `ttinsn_t1` (the T1 runner with two hooks), run
several ways per repetition: the runner's own push (`lw`/`sw`, sixteen words at a time), the same
program pushed as fused `.ttinsn` words built into the image ("baked"), and the same program
**translated into fused code at run time** by the runner and called after invalidating T1's instruction
cache. ABAB after warm-ups, median of 9. T1's `START -> PUSHED -> RETIRED` come from the tile's
timestamper (the roles only, no mover, so no concurrent-export corruption). Result rows are poisoned
before every run (`Dst` keeps its contents between runs on silicon) and every element a program reads is
defined by its prelude, both column parities; every run's result is compared bit-exact between the ways:
`Dst` bands at the push level, every output byte at the kernel level. The results are the same on ttsim
and silicon (the `gelu` result band hashes to `0x0a6ed2c5` on both, for all three ways).

**Baked fused push** (final build; runs `1791654655` push, `1791654657` kernel; earlier build, same
numbers within 1%: `1791651322`, `1791651644`, `1791651513`, `1791651645`):

| What | Runner push | Fused `.ttinsn` | Speedup |
|---|--:|--:|--:|
| Precise `gelu` body, 11,645 words pushed, push phase | 38,578 cycles (3.31/word) | 13,868 (1.19/word) | 2.78x |
| the same, push + retire | 38,630 | 13,914 | 2.78x |
| Kernel of 16 tiles (T0 unpack, T1 math, T2 pack), T1 push | 607,665 | 218,499 | 2.78x |
| Kernel of 16 tiles, whole kernel (cycles per tile) | 608,841 (38,053) | 219,741 (13,734) | 2.77x |

The first build also had a one-tile kernel (39,323 -> 14,824 cycles, 2.65x) and a 130-word program
(610 -> 414, 1.47x); both programs were dropped from the image to make room for the translator.

**Run-time translation** (run `1791654639`; 516 stored words expand to 11,645 pushed; the program is
stored once and the loops are loops in the translated code):

| What | Runner | Baked | Translated | Translated, no opcode check |
|---|--:|--:|--:|--:|
| `gelu` push phase, cycles (of which translating + invalidating) | 38,578 | 13,868 | 25,308 (11,490) | 20,907 (7,089) |
| 194-word program with no loops, push phase | 818 | | 4,791 (4,369) | 3,130 (2,708) |
| Kernel of 1 tile, whole kernel | 39,328 | | 23,219 | |
| Kernel of 16 tiles, whole kernel | 608,840 | 219,741 | 228,421 (2.67x) | |

- **Translating costs 13 to 22 cycles a word**, not the 3-4 I expected: about 13 for the copy loop alone
  (an L1 load and an L1 store per word, with a rotate), 9 more for the per-word opcode check, plus
  about 330 cycles fixed (the load-back and the 24 instructions the invalidation sequence needs).
  Batching sixteen loads, as the runner's push does, changed nothing (26 against 28), so it is not load
  latency. A translator is therefore only worth it for a program that is long once expanded and short
  once stored (loops): `gelu` is 23 times longer expanded, so the cost is 4% of a 16-tile kernel; a program
  with no loops loses (4,791 against 818).
- **The speedup is not instruction gathering.** Blackhole boots with gathering (the hardware fusion of
  adjacent `.ttinsn` words, up to four a push) enabled, and this repo's firmware never touched it, so every
  number above ran with it on. tt-metal's firmware turns it off (`configure_gathering`, tt-metal#16439). Run
  `1791671588`, gathering on against off with the same image: baked `gelu` push 13,868 cycles both ways;
  runner 38,576 / 38,559; translated 20,902 / 20,896; 16-tile kernel baked 219,767 / 219,788, runner
  608,874 / 608,885. Every mode stays bit-exact. A `.ttinsn` push is one store with an immediate, so the
  gain is the removed load and loop bookkeeping at the frontend's one instruction a cycle (1.19 cycles a
  word), not four-a-cycle fusion.
- **Not measured:** the Session level (mover and host around the kernel), kinds other than `gelu`, the
  matmul (backend-bound, `role_push_rate`), card 1, and translation on another core.
- **The kernels are push-bound, now measured.** The earlier "39 of 73 kinds push-bound" was a static
  model (`step121_wait_planner`'s `per_kind_push_table`: 2.8 cycles a word over a one-instruction-a-cycle
  backend). The runner measures 3.31 cycles a word on `gelu` and the retire tail is the same ~45-52 cycles
  either way, so the backend keeps up at the fused 1.19 cycles a word.
- **The per-tile cost of `gelu` in `approx_per_tile_saving` (28.46 us, precise) is this.** 38,053 cycles
  a tile at the card's ~1.35 GHz is 28.2 us: the Session-level slope was T1's push time.

## X6 mover NIU fast path (2026-10-10, card 0)

`silicon_perf::mover_read_fast_path`, release, run `1791600377`: reads through request initiator 1 of
NoC #0 writing only the words that change (`dm::READ_FAST`) against the path that writes every
register of initiator 0, ABAB after three warm-up lists each, median of `REPS`, an empty list's time
subtracted, destination bytes checked against the host's copy. 64 B x 256: 0.481 -> 0.464 us per
entry; 2 KiB: 0.482 -> 0.464 (one port), 0.485 -> 0.471 (all channels); 4 KiB: 0.480 -> 0.462,
0.487 -> 0.473 (3 ports), 0.484 -> 0.470 (all channels); 16 KiB: 0.485 -> 0.471. Fast/slow is
0.961-0.972 everywhere: about 17 ns of the ~0.48 us a request costs. The request's cost is its round
trip, not the register writes, so a gather's share of a step does not move; more requests in flight
is the lever, not cheaper issue.

> **Superseded in part (2026-10-13).** The measurement holds; the explanation does not. A cap sweep and a bare read loop
> show the entry is software (359 device cycles, a bare loop 90-106) and that the cap does not matter past 4: see
> [Read-loop probe](#read-loop-probe-what-the-interpreter-costs-per-read-entry-2026-10-13-card-0).

## Approx transcendentals, per-tile saving (2026-10-10, card 0)

`silicon_perf::approx_per_tile_saving` (lane T8, `MathMode::Approx` against `Precise`; the device
programs equal the interpreter bit for bit): `exp` 8.08 -> 6.51 us per tile (1.24x), `log` 10.96 ->
7.97 (1.37x), `recip` 7.82 -> 0.95 (8.24x), `sigmoid` 12.18 -> 7.40 (1.65x), `tanh` 13.09 -> 8.10
(1.62x), `gelu` 28.46 -> 7.31 (3.89x). Interpreter instruction counts per tile: 1.51x, 2.52x, 2.12x,
2.82x, 3.17x and 7.10x fewer. `sqrt` and `rsqrt` were measured and dropped (1.36x and 1.17x in
instructions, with special-value handling costing as much as Precise's). SFPU math programs are
push-bound at the runner's 2.8 cycles per word (39 of 73 kinds have a push/backend cycle ratio above
1, median 1.17, max 2.87, taking the vector unit at one instruction per cycle as a floor); the
matmul is backend-bound (rows AC/AD). (That ratio is a static model, not a measurement; the
[`.ttinsn` section](#ttinsn-push-fusion-precise-gelu-2026-10-10-card-0) measures the push on silicon.)

## P2 K-blocking and R3 norm release baselines (2026-10-10)

Conditions: release silicon, isolated runner (`cargo xtask bench --device all --filter
silicon_bench_tensix_ops::<name>`), both cards, AICLK 1350 MHz, GDDR 16000 MT/s x 8 channels, git
`c49c2ed` (working tree with the lane commits, runs `1791593690` K-block, `1791593692` norm
forward, `1791593699` norm forward+backward), two warmups and seven host-timed samples, median
(p10-p90 in the run files), validation outside the timing: every K-blocked result equals the
unsplit result bit for bit, and `[64,8192]@[8192,64]` also equals an exact oracle; every norm is
checked against an f64 oracle and burn-flex within a derived bound before timing (F32 forward also
bit-exact against the step70 specification model). No speedup is claimed.

K-block matmul (TF32 Src, F32 storage, HiFi4, two tiles; "unsplit" is the planner's own choice),
median us, card 0 / card 1:

| Shape | unsplit | k_block=1 | 4 | 16 | 64 |
|---|--:|--:|--:|--:|--:|
| [64,8192]@[8192,64] | 751 / 732 | 5821 / 5880 | 1797 / 1791 | 843 / 839 | 725 / 726 |
| [37,4097]@[4097,35] | 496 / 492 | 2935 / 2921 | 955 / 954 | 498 / 494 | 527 / 521 |

A one-tile K block costs 7.7-8x the planner's choice (each block re-packs and reloads the FP32
partial); blocks of 16 tiles or more are within about 5% of unsplit, and the planner default is at
or near the best on both shapes.

Norms (epsilon 1, gamma 1, beta 0), median ms, card 0 / card 1:

| Op | dtype | Shape | forward | forward+backward |
|---|---|---|--:|--:|
| LayerNorm | F32 | [37,8193] | 15.64 / 15.50 | 70.8 / 71.0 |
| LayerNorm | BF16 | [37,8193] | 78.5 / 78.0 | 307.5 / 309.3 |
| RmsNorm | F32 | [37,8193] | 8.81 / 8.88 | 53.1 / 53.4 |
| RmsNorm | BF16 | [37,8193] | 52.6 / 52.5 | 186.8 / 186.3 |
| LayerNorm | F32 | [65,70] | 0.464 / 0.465 | 1.85 / 1.81 |
| LayerNorm | BF16 | [65,70] | 1.47 / 1.48 | 4.81 / 4.69 |
| RmsNorm | F32 | [65,70] | 0.318 / 0.327 | 1.50 / 1.46 |
| RmsNorm | BF16 | [65,70] | 1.08 / 1.07 | 3.30 / 3.22 |

The BF16 norms are 3-5x slower than F32 (each Burn op widens and narrows once); the compositions
are unfused by design (norm fusion is deferred). Norm benchmarks record only downloads per run,
because Burn exposes no `dataflow_stats`.

## ADC Z/W plane-copy tranche (2026-10-06)

Release card-0 run `1791324901`, dirty tree based on `2ca901b2d6be`, two
surviving Tensix units, default pipeline/batch/streaming settings. Resident F32
source shape is `[2,3,33,33]`, stored as 198×33, with values `i % 37`. W/Z origin
is `[0,1]`; selected counts are `[1,1]`, `[1,2]` and `[2,2]`. Native repacking
uses the equivalent precomputed logical indices in W-major/Z-major order.
Host timing runs from dispatch through `Session::sync`, excluding source upload,
index construction, output download/validation and free. Each of nine
invocations per case is validated bit-for-bit; two warmups precede seven samples.
No other silicon runner overlapped this run, and simulator/workspace regression
jobs had finished. NC timestamp exports are not used.

| Selected planes / output storage | Native repack median (µs) | ADC median (µs) | ADC / repack |
|---|---:|---:|---:|
| 1×1 / 33×33 | 48.230 | 111.066 | 2.30× |
| 1×2 / 66×33 | 85.429 | 168.504 | 1.97× |
| 2×2 / 132×33 | 164.877 | 282.738 | 1.71× |

The explicit ADC API is **71–130% slower** for these fixtures. No speedup or
automatic Burn routing is claimed. The fixed 64-row unpack traversal and
per-output-tile gathers are measurable costs even for small ragged outputs.

Across nine invocations, ADC `dataflow_stats` `(regions,batches,transfer_packets)`
is `(18,36,0)`, `(18,54,0)` and `(18,90,0)` respectively. Native repack reports
`(0,0,18)` in each case. Standalone transfer packets do not count kernel
source gathers/output scatters; program audits separately establish NoC0 B
reads, logical-word staging, Z/W instruction use and NoC1 NC output writes.

Collector artifacts: `target/silicon/bench/1791324901.{jsonl,md}`, six validated
records. Correctness: card-0 step99 plus subsequent CNN gates 13/13
(`1791324670`), full release SMOKE 212/212 (`1791324245`), and fresh copies with
`TT_PIPELINE=0` / `TT_BATCH=0` (`1791324866` / `1791324883`). MNIST e2e is 8/8
(271.51 s), with the golden unchanged. Other instruction families and performance
optimization remain deferred.

Preliminary run `1791324740` also validates all six records, but overlapped
simulator workspace testing on the host. Its 1.76–2.33× ratios are retained in
the collector artifacts; the final quiet run above owns the scoreboard.

### Explicit L1 movement APIs (step101/102, 2026-10-07)

Validated release card-0 run `1791389713` (3/3 benchmarks), with no simulator
workloads running; rustc 1.98.1, default Session bring-up/power policy, ARC
watchdog 10, no clock override. Two warmups and seven measured samples; each
output/guard region is validated outside the timed interval. Artifact:
`target/silicon/bench/1791389713.{jsonl,md}`. Earlier validated exploratory run
`1791389291` ran alongside simulator regression checks and is superseded here.

Local measurements use one tile and **128 repeated transfers per sample**.
XMOV setup is done once per resident launch, each issue drains C9; the native
B baseline submits 128 COPY_WORDS descriptors (zero broadcasts a resident zero
word with source stride zero). Both include their different host launch and
completion costs, and neither includes GDDR/staging/readback. These are local
execution costs, not a pure mover-bandwidth measurement.

| Local bytes per issue | B copy, µs / 128 | XMOV copy, µs / 128 | XMOV zero, µs / 128 |
|---|---:|---:|---:|
| 16 | 301.210 | 17.863 | 17.753 |
| 128 | 321.878 | 17.793 | 17.723 |
| 4096 | 1034.564 | 40.735 | 36.970 |

B's 4096-byte broadcast zero baseline is 891.167 µs per 128 issues. Local
throughput alone does not imply a beneficial tensor route.

Whole-tensor F32 measurements use two Session units and time allocation,
build/staging/launch through sync; final download/validation/free are excluded.
Native copy is Session::copy. Native zero is the existing metadata-immediate
zero path (the backend's 32-bit zeros implementation), not a host upload or
arithmetic fallback. The explicit APIs initialize destination padding.

| Shape | Native copy µs | XMOV copy µs | Native zero µs | XMOV zero µs |
|---|---:|---:|---:|---:|
| 32×32 | 11.531 | 14.356 | 516.010 | 14.096 |
| 256×256 | 28.353 | 242.250 | 17949.838 | 207.276 |
| 97×99 (ragged) | 16.882 | 117.959 | 3348.486 | 62.135 |

Over all nine dispatches, XMOV recorded 9/18/18 ownership regions and
9/576/144 batches respectively, with **zero transfer-only packets**. Native
copy recorded 9/18/18 transfer packets, and metadata zero 27/1440/225.
Each XMOV output tile is one bounded local-movement job; the larger native copy
amortizes launches across runs. Ragged movement additionally zeros and copies
face-row fragments. The explicit copy is slower end to end in every measured
shape despite the faster local engine. Keep all four APIs opt-in; automatic
Burn routing remains unchanged, including zero routing.

## ADC rectangle-copy instruction tranche (2026-10-06)

Release card-0 scoreboard: run `1791322068`, dirty tree based on `2ca901b2d6be`,
two surviving Tensix units, default streaming/pipeline/batch settings. Input is
a resident 128×128 F32 matrix with values `i % 37`; origins are `[1,16]` for
16×16 and `[15,16]` for the other rectangles. Timed from host dispatch through
`Session::sync`, excluding input upload, output validation/download and free.
Every invocation is validated bit-for-bit. Two warmups precede seven samples;
the table reports medians in microseconds. No concurrent NC timestamp exports
are used.

| Rectangle | Existing native repack | ADC unpack/Dst/pack | ADC / repack |
|---|---:|---:|---:|
| 16×16 | 15.879 | 28.413 | 1.79× |
| 37×48 | 53.249 | 82.733 | 1.55× |
| 65×80 | 130.622 | 161.030 | 1.23× |

The new route is **23–79% slower** in these cases. This tranche establishes
instruction semantics and a production consumer; it does not establish a
speedup. Automatic Burn ADC slice routing was subsequently removed at the
user's request; float slices again use row views and the original native repack
path. The explicit Session ADC API and instruction gates remain. Future optimization should
reduce per-output-tile setup and per-row unpack/wait/advance issue costs.

Across nine invocations, ADC dataflow `(regions, batches, transfer_packets)` is
`(9,9,0)`, `(18,36,0)`, `(18,81,0)` respectively. Existing native repack is
`(0,0,9)`, `(0,0,18)`, `(0,0,18)`: transfer_packets counts standalone transfers,
not kernel gathers/scatters. Output download is outside both the timed region
and these dataflow counters. Builder audits confirm B reads source tiles
without element copies and NC writes each packed output tile.

Collector artifacts: `target/silicon/bench/1791322068.{jsonl,md}` (six records),
with the validated per-test output under `target/silicon/out/`. This final run
includes counter cleanup and starts after the full smoke runner has exited.
Earlier run `1791321354` measured 22–57% overhead before that cleanup. Run
`1791322057` passed validation but overlapped the smoke runner's final gates;
its timing is excluded from the scoreboard.

## Resident matrix chain benchmark (2026-10-06)

Release run `1791317719`, cards 0 and 1, SHA `08b300ea0fb6+dirty`, two Tensix
tiles per card, resident F32 operands, TF32/BF16 Src precision, HiFi4, normal
Session streaming defaults. Compare `(a+b)*b` as one resident chain against two
matrix calls, with constant operands 2 and 3 (every output must equal 15).
Two warmups precede seven host dispatch-through-sync samples; output downloads
and frees are outside timing. All 144 outputs were validated (including warmups).
No other test or benchmark ran concurrently. Medians in microseconds:

| Card | Shape | Src precision | Two calls (µs) | Chain (µs) | Ratio |
|---|---|---|---:|---:|---:|
| 0 | 64×64 | TF32 | 35.465 | 25.568 | 1.39× |
| 0 | 64×64 | BF16 | 36.718 | 25.537 | 1.44× |
| 0 | 65×70 | TF32 | 57.846 | 41.096 | 1.41× |
| 0 | 65×70 | BF16 | 58.148 | 42.148 | 1.38× |
| 1 | 64×64 | TF32 | 36.558 | 24.637 | 1.48× |
| 1 | 64×64 | BF16 | 37.099 | 25.077 | 1.48× |
| 1 | 65×70 | TF32 | 58.419 | 42.409 | 1.38× |
| 1 | 65×70 | BF16 | 58.368 | 41.928 | 1.39× |

The measured cases improve 1.38–1.48×; this does not establish an application
speedup. Across nine invocations, `dataflow_stats` reports 36 versus 18 regions,
72 versus 36 batches for 64×64, and 162 versus 81 batches for 65×70 (separate
calls versus chain), on both cards and at both precisions. Standalone transfer
packets are zero for both paths; kernel gathers/scatters are not counted by that
field. Production builder audits establish two source reads and one final write
per output tile, only final GDDR allocation and no intermediate GDDR round trip.

Artifacts: `target/silicon/bench/1791317719.{jsonl,md}` (all 16 records), with
per-card validated logs under `target/silicon/out/`. Earlier run `1791317688`
validated outputs but its first BENCH record shared libtest's test-name line and
was omitted by collection. The benchmark now ends that line before reporting;
use the complete final run above.

## Native operation baselines, 2026-10-05

Run `1791235277`, card 0, two Tensix tiles, release, SHA
`83729d294e06+dirty`. Resident operands, two warmups, seven host-timed samples;
timing includes dispatch, synchronization and final output download, with output
validation outside timing. These are operation baselines, not comparative speedups.

| Operation | Shape/conditions | Median (us) | p10/p90 (us) |
|---|---|---:|---:|
| Checked I32 division | [37,70] / 3 | 397.496 | 391.495 / 404.820 |
| F32 average pooling | [1,1,9,10], kernel 4×8 | 391.053 | 385.114 / 404.509 |
| F32 Conv2D | x [1,1,9,10], w [2,1,2,2] | 274.258 | 268.246 / 278.776 |
| F32 attention | Q [1,1,3,32], K/V sequence 64 | 186.376 | 185.745 / 191.133 |

Artifacts: `target/silicon/bench/1791235277.{md,jsonl}`; benchmark source
`silicon_bench_tensix_ops.rs`. BF16 and distributed baselines are recorded below.

## BF16 and mesh operation baselines, 2026-10-05

Run `1791239400`, release SHA `83729d294e06+dirty`, resident operands, two
warmups/seven samples. Single card 0 uses one Tensix tile; the mesh uses cards
0+1 with one tile per card. Host timing includes dispatch through final readback;
validation is outside timing. Every mesh case checks positive product counts on
both cards and acknowledged Ethernet bytes. Packed mesh transport remains deferred.

| Storage / cards | Conv2D median us (p10/p90) | Attention median us (p10/p90) |
|---|---:|---:|
| BF16 / 0 | 3140.697 (3105.041 / 3145.897) | 457.967 (449.772 / 462.185) |
| F32 / 0+1 | 2791.860 (2762.157 / 2818.321) | 565.325 (564.574 / 569.132) |
| BF16 / 0+1 | 3303.418 (3296.014 / 3317.463) | 838.079 (829.765 / 842.718) |

Conv2D: x [1,64,2,3], w [64,64,1,1]. Attention: Q [1,1,3,64], K/V [1,1,64,64].
The earlier single-card F32 baselines use smaller shapes and two tiles; these
rows cannot establish a mesh speedup. Artifacts:
`target/silicon/bench/1791239400.{jsonl,md}` (the JSONL retains both condition records).

## Packed BF16 update, 2026-10-05

Run `1791162075`, card 0 p150a, one Tensix, release, SHA
`0d6ba5227fdb+dirty`, 1350 MHz AICLK, 16000 MT/s GDDR, eight channels, HiFi4,
pipeline/profiling off, resident operands, one warmup, nine synchronized host
samples, output validated outside timing:

| GEMM | Earlier packed BF16 (us) | Compact packed BF16 (us) | Current TF32 (us) |
|---|---:|---:|---:|
| 64×784×128 | 661.742 | 141.092 | 97.741 |
| 64×128×10 | 80.840 | 53.499 | 18.083 |

Compact GATHER descriptors replace per-tile read/fill entries; only ragged lanes
are filled. The larger BF16 case improves about 4.7×. Both BF16 cases remain
slower than TF32, so no MNIST speedup is established. Results:
`target/silicon/bench/1791162075.{md,jsonl}`. The separate BF16 accuracy run
achieves 91.82% after one epoch; see the implementation record for conditions.
Ordinary benchmarks now use one card; test both for actual cross-card behavior.

## Streaming ownership rollout

The current architecture paragraph moved to [performance-scoreboard.md](performance-scoreboard.md#current-architecture-streaming-ownership);
everything below is the rollout evidence.

After consolidation, slot reuse waits only for the earlier batch's pack where
gathers and scatters do not meet in L1, the roles skip the retirement barrier after
a region's last batch, credit waits run the full peer check every 64th poll, and
consecutive transfers share a control list. Sweep run `1791067111` against
`1791060004` (streaming, µs): traced 2048² add 2418.7 -> 2260.5 / 343.9 -> 325.1 /
239.3 -> 228.6 on 1/8/32 tiles; fresh 64² add 18.86 -> 17.15 (legacy best 14.80);
fresh 32-tile 256³ matmul 239.6 -> 239.7 (legacy best 181.1), host-bound at ~130 µs
of enqueueing per op. MNIST 1.4 ms/step (unchanged), transformer 8.8 (burn-flex
4.35). See [the ownership status](streaming-dataflow-architecture.md#validation).

Post-consolidation silicon run `1791063683` rechecks the identity benchmark under
the default ownership scheduler: traced read/write payload is 178.29 GB/s each
(356.58 combined) at 32 tiles, and 164.39 GB/s each (328.78 combined) at 120.
These remain whole-operation host-timed useful-payload rates, not isolated DRAM
read/write bandwidth. The removal does not claim an additional speedup.

2026-10-03, silicon run `1791060004`, device 0. Resident B-reader / NC-writer
ownership uses one host B commit per region and one role generation per compatible
batch group. These are **host end-to-end medians**, not concurrent-core timestamper
estimates: three warmups followed by 15 samples, against the best legacy B/NC path.

| Workload | Tiles | Legacy best (µs) | Streaming (µs) | Ratio |
|---|--:|--:|--:|--:|
| Add 2048², traced | 1 | 3271.76 | 2418.66 | 0.739 |
| Add 2048², traced | 8 | 457.59 | 343.87 | 0.751 |
| Add 2048², traced | 32 | 281.61 | 239.33 | 0.850 |
| Add 64², fresh | 1 | 14.80 | 18.86 | 1.275 |
| Matmul 256³, fresh | 32 | 181.09 | 239.63 | 1.323 |

The historical ≥10% mover-improvement gate passed, but the ≤5% regression gate
failed. No universal speedup is claimed. The initial role stream keeps a
retirement barrier between existing arithmetic batches, and fresh packets still
carry per-batch script and credit work. Reducing that overhead is follow-up work.
The sweep covers add 64²/512²/2048² and matmul 256³/512³ on 1/8/32 tiles, fresh
and traced. Existing `Session::host_times`, transport traffic and
`dataflow_stats`/`dataflow_progress` provide submission and ownership diagnostics.
The known concurrent-NC timestamper corruption remains: streaming profiles can
contain repeated or missing events even with role stamps disabled. A silicon
test checks that profile collection preserves fresh/traced output correctness
and reports export failures diagnostically; it does not certify device timings.

```
cargo xtask silicon --release --include-ignored --filter step60_streaming::streaming_performance_sweep
cargo xtask silicon --release --include-ignored --filter step60_streaming::streaming_stress --timeout-secs 300
```

### Overlap thresholds for NC ownership (run 1791079782, 2026-10-04)

Element-wise and reduce ops overlap their runs where the host's cost for the
extra lists is paid back. The old constants were measured on the B-only pipeline.
Host end to end, one op, serialized against overlapped (< 1 wins), card 0, release:

| Add, tiles a unit | 8 tiles, fresh (forced) | 8 tiles, traced | 32 tiles, fresh (forced) | 32 tiles, traced |
|---|--:|--:|--:|--:|
| 8 (256² on 8, 512² on 32) | 0.99 | 1.00 | 0.99 | 1.02 |
| 32 (512² on 8, 1024² on 32) | 1.20 | 0.90 | 1.39 | 0.95 |
| 128 (1024² on 8, 2048² on 32) | 1.10 | 0.77 | 1.37 | 0.92 |
| 512 (2048² on 8, 4096² on 32) | 1.07 | 0.69 | 1.44 | 0.87 |
| 1152 (3072² on 8, 6144² on 32) | 1.03 | 0.66 | 1.38 | 0.86 |
| 1568 (3584² on 8) | 0.97 | 0.67 | | |
| 2048 (4096² on 8) | 0.87 | 0.67 | | |

(Runs 1791079466 and 1791079480, forced and under the old constants; below 12
tiles a run, nothing overlaps either way.)

Fresh overlap lost on many tiles until a unit holds about 160 tiles for each
unit beyond the first; a replay pays no host time for its runs and gained from
about 32 tiles a unit. `tensor::Overlap` encodes both (`Fresh`: 48 + 180
(units - 1) tiles a unit; `Captured`: 32), replacing the old
`PIPELINE_SHARE * units²` and the reduction's 4x share. With them no sweep cell
is over 3% slower than serial; fresh 8-tile 2048² add 1.07 -> 1.00, traced
8-tile 512²/1024² adds 1.00 -> 0.90/0.77, traced 32-tile 1024²/2048² 1.00 ->
0.95/0.93, one-tile reductions of 64-256² 1.00 -> 0.91. Per-cell noise across
three runs is up to about 0.05, rarely 0.15. `SWEEP_SHARE_PERCENT` (0 forces
overlap above `MIN_PIPELINED_RUN`) scales the shares in both sweeps.

Host enqueue of a fresh 32-tile 256³ matmul, per op (`host_time_per_op`):
143.9 µs, of which segments 34.7, placement 22.2, reservation 16.5, list checks
12.6 (new `HostStage::Check`), list writes 38.8, idle checks 3.5.

### GDDR and Ethernet remeasurement

2026-10-03, device 0, release, AICLK 1350 MHz, GDDR 16000 MT/s.
All benchmark invocations passed. Raw GDDR/Ethernet run `1791060908`, isolated
NC/NoC #1 writes run `1791061578`, streaming runs `1791061555` and `1791061613`.
Each throughput is a nine-sample median; the streaming benchmark uses three
warmups per mode and phase and checks the identity output bit for bit.

**Streaming workload:** an 8192² FP32 identity operation (`MUL_SCALAR 1.0`),
256 MiB useful input and 256 MiB useful output, through B reads on NoC #0,
resident T0/T1/T2 computation, and NC writes on NoC #1. Host timing includes
submission, unpack/compute/pack, credits and synchronization, but excludes
payload upload/download, trace capture and output validation. Read and write
payload rates are the same bytes divided by the same whole-operation time:
they are not independently timed physical DRAM bandwidth. Headers, padding and
command traffic are not counted as useful payload. Concurrent NC device
timestamps are not used.

Results from the repeat run `1791061613`, GB/s:

| Tiles | Streaming fresh read / write (each) | Streaming traced read / write (each) | Streaming traced combined | Best legacy traced combined |
|--:|--:|--:|--:|--:|
| 1 | 9.03 | 10.12 | 20.24 | 15.54 |
| 8 | 53.30 | 80.00 | 160.00 | 111.60 |
| 32 | 62.34 | 177.49 | 354.98 | 215.76 |
| 120 | 72.88 | 163.86 | 327.71 | 244.09 |

Traced streaming is 1.30×/1.43×/1.65×/1.34× the best legacy B/NC throughput,
respectively. 32 tiles outperform 120 for this workload in both runs (354.97
versus 325.86 GB/s in the first run). Fresh streaming still loses to the best
legacy path: combined throughput at 120 tiles is 145.75 versus 186.31 GB/s,
with latency 3.68 versus 2.88 ms. Consolidating the architecture accepts this
regression; it does not overturn the failed historical performance gate.

**Raw transfer baselines, not the streaming executor:**

| Measurement | GB/s | Timing |
|---|--:|---|
| B read, one tile, all channels, 64 KiB entries | 81.36 | device |
| NC write, one tile, NoC #1, all channels, 64 KiB entries | 74.94 | device |
| B read, 120 tiles | 468.59 | device; host end-to-end 310.56 |
| B write on NoC #1, 120 tiles | 389.36 | device; host end-to-end 314.47 |
| B mixed read/write, writes on NoC #1, 120 tiles | 403.15 combined | device; host end-to-end 320.84 |
| Ethernet, one link, one way, 32 × 128 KiB | 19.92 | host; device 20.23 |
| Ethernet, two links, one way | 27.03 combined | host |
| Ethernet, two links, both ways | 33.08 combined across both directions | host |
| Ethernet staged wire slope, 64 → 128 KiB | 49.20 | E1 device |

The isolated NC run has no concurrent B/role stamps. Card-wide raw measurements
use B movers, including the NoC #1 write row: they do not measure 120 NC writers.
Ethernet's E1 path is unchanged by streaming ownership. Its wire is still near
50 GB/s per link, but store-and-forward staging, acknowledgement and host posting
keep actual two-link one-way throughput at only 27% of the 100 GB/s ceiling.

```
cargo xtask bench --filter silicon_bench_memory --filter silicon_bench_eth --keep-going
cargo xtask bench --filter step60_streaming::streaming_gddr_throughput
BENCH_MOVER=nc BENCH_WRITE_NOC=1 cargo xtask bench --filter silicon_bench_memory::mover_write_sweep
```

## GDDR6 through the data mover

| One tile, entry size | Read, 1 ch | Read, all ch | Write, all ch |
|--:|--:|--:|--:|
| 4 KiB | 16.3 GB/s | 16.3 | 14.6 |
| 16 KiB | 62.0 (97% ch) | 63.5 | 57.2 |
| 64 KiB | 62.0 | 81.4 (94% link) | 74.1 |
| 128 KiB | 62.0 | 80.7 | 68.0 |

- **Below 16 KiB the rate is set by the cost per entry, about 339 cycles** (317 at
  the baseline; see the change log). A `WAIT` entry alone costs 136 cycles.
- **One DRAM port now carries a whole channel of writes** (62.9 GB/s at 64 KiB).
  Before static VC 1 one port stopped at about 28.5 GB/s and 128 KiB entries
  fell to 27.5.
- Run 1790971715 (reads) and 1790970443 (writes).

| Tiles at once | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 120 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Reads, card GB/s | 85 | 164 | 296 | 370 | 397 | 404 | 404 | **429** |
| Writes on NoC #0, card GB/s | 76 | 78 | 79 | 83 | 102 | 136 | 150 | 161 |
| Writes on NoC #1, card GB/s | 76 | | 251 | | 293 | | 273 | 266 |
| Reads + writes (NoC #1), card GB/s | 97 | | 322 | | 341 | | 346 | 343 |
| All tiles on one channel, GB/s | 63 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| *Before (reads, run 1790950685)* | *85* | *168* | *318* | *244* | *170* | *161* | *146* | *127* |

Runs 1790970443 and 1790971715.

- **The reads no longer collapse past 4 tiles.** Two changes landed together, and
  they haven't been separated:
  - every request now goes out on static VC 1;
  - port 1 of each channel is now NoC #1's, so NoC #0 traffic that asked for port 1
    goes to CMFW's endpoint instead.
- **Reads belong on NoC #0.** With every tile reading on NoC #1, the card stayed
  at one link's rate: 4 tiles 86 GB/s, 120 tiles 58 (run 1790959226).
  - NoC #1's read data climbs the DRAM column, then runs left along the readers'
    shared row.
- **Writes scale on NoC #1 and not on NoC #0.**
- **Each GDDR endpoint belongs to one NoC** (`tt_isa::dram::DramChannel::owns`):
  - NoC #1 gets port 1; NoC #0 gets ports 0 and 2, including CMFW's.
  - Both NoCs on one endpoint is SYS-1419, which hung card 0 (`ttsim-divergence.md`
    row 73).

### What holds card reads at ~430 GB/s: unfair service, not distance

From `gddr_aggregate_affinity` (runs 1790975421 onward), 120 tiles reading,
each tile's share fixed:

| Placement | Card GB/s |
|---|--:|
| Cycle: every tile rotates over all 8 channels (today) | **429** |
| Column: its half's 4 channels, so no data wraps the torus | 417 |
| Nearest: one channel whose endpoint is at or just above the tile, 15 tiles per channel | 315 |
| Nearest, alternating NoC #0's two endpoints | 216 |

- **Distance doesn't explain the gap.** Nearest scored the same 315 GB/s with two
  different row assignments.
- **The finish times do.** Every tile starts within 27 µs of the others, but the
  last finishes up to 966 µs after the first, in a 1102 µs run. The NoC serves
  tiles unevenly, so the card total is set by the slowest tiles. Session splits
  ops evenly across tiles, so its ops wait on the slowest tile the same way.
- **A lower in-flight cap evens out the service:**

  | Reads, tiles | 1 | 4 | 16 | 120 |
  |---|--:|--:|--:|--:|
  | Default cap (`MAX_IN_FLIGHT` 128) | 84.7 | 294 | 397 | 420 |
  | Cap 32 | | | | 452 |
  | Cap 16 | | | | 460 |
  | **Cap 8** | 84.6 | **323** | **472** | **471** (92%) |
  | Cap 4 | 72.4 | 254 | 471 | 464 |

- Under cap 8, write mixes stay within noise of the default, except one tile mixing
  reads and writes, about 5% slower.
- **The default since 2026-10-02:** `dm::TILE_IN_FLIGHT_CAP`, set by
  `DataMover::start`. E1 keeps `MAX_IN_FLIGHT`.

**Cap 8 against the default, by entry size** (reads, card-wide GB/s; `AGG_LEN`,
`AGG_CAP`):

| Entry | 1 tile | 16 tiles | 120 tiles |
|--:|--:|--:|--:|
| 4 KiB | 16.3 → 15.7 | 255 → 245 | 452 → **480** |
| 16 KiB | 64.9 → 62.1 | 420 → **473** | 464 → **496** (97%) |
| 64 KiB | 84.7 → 84.7 | 396 → **471** | 425 → **467** |
| 128 KiB | 84.5 → 84.3 | 370 → **419** | 398 → **437** |

- **Many tiles: cap 8 wins at every size.** 16 KiB entries, one NoC request
  each, come within 3% of the card.
- **One tile: cap 8 loses 3–4% at 4 and 16 KiB.** Eight requests in flight no
  longer quite cover a small read's latency. From 64 KiB up there's no
  difference.
- **Writes: within noise at every size and tile count.**

### Instruction caches: about 4 KiB on every baby core

Measured by `probe_icache` on card 0 (2026-10-03); the Blackhole size is
documented nowhere. The probe runs the last N bytes of two 8 KiB blocks:
straight-line `nop`s, and a chain of jumps 32 bytes apart.

| Code size | Straight-line, cycles/byte | Jump chain, cycles/jump |
|--:|--:|--:|
| 256 B – 3.5 KiB | 0.25 (one per instruction) | 5.1 |
| 4 KiB | 0.25 | 5.4 |
| 5 KiB | 0.29 | 8.6 |
| 6–8 KiB | 0.31 | 10.6 |

- **All five cores measure the same,** B, NC and T0–T2 alike. T0–T2 were run
  with 6 KiB blocks to fit their 16 KiB slots. Wormhole's were 2 / 2 / ½ / 2 / ½
  KiB for B / T0 / T1 / T2 / NC.
- **A miss costs ~5.5 cycles per 32 bytes,** and straight-line code mostly hides
  it.
- **Why placement matters:**
  - The plain-entry hot path is ~2.5 KiB and fits.
  - A record's tiles also run `run_record` (~4.4 KiB), so their working set
    overflows the cache, and where the code sits changes how many lines
    collide.
- **A taken jump costs ~5 cycles even when cached.**

### RISCV NC's mover costs what B's does

NC runs the same mover image (`dm_nc`). One tile, per entry, `BENCH_MOVER=nc`:

| Entry | B | NC |
|---|--:|--:|
| 4 KiB read | 351.6 cycles | 351.6 |
| 16 KiB read | 363.5 | 363.6 |
| 4 KiB write | 397.4 | 397.3 |
| `WAIT` (256 in a list) | 136.2 | 136.2 |

- **No sign of a smaller NC instruction cache on Blackhole.** On Wormhole NC's
  was a quarter of B's (`riscv-guide-review.md`), but here NC is as fast as B
  at every size.
- **NC can take any share of the moves.**
- **The 4 KiB read is up from 339 cycles** (the cap-8 default; see above).

### What a second mover buys: reads on B, writes on NC

`copy_pipeline` copies GDDR → L1 → GDDR in 64 KiB blocks, double-buffered,
writes on NoC #1, three ways:

- B alone, sequential.
- B alone, pipelined: block k+1's reads go out with block k's writes.
- Split: B reads, NC writes, synchronised by `SIGNAL` / `WAIT_PEER`.

Copy rate in GB/s (each byte read and written once):

| Entries | Scheme | 1 tile | 4 | 16 | 120 |
|---|---|--:|--:|--:|--:|
| 4 KiB | B pipelined | 5.7 | 22.6 | 90 | 171 |
| 4 KiB | **Split** | **9.1** | **36.0** | **143** | 173 |
| 16 KiB | B pipelined | 19.8 | 78.7 | 152 | 164 |
| 16 KiB | **Split** | **26.4** | **84.5** | 156 | 165 |
| 64 KiB | B pipelined | 28.8 | 85.5 | 161 | 164 |
| 64 KiB | Split | 29.7 | 89.9 | 157 | 164 |

- **The split pays where per-entry cost is the limit:** small entries, or few
  tiles busy. 4 KiB copies run 1.6× faster on one to 16 tiles, because each core
  issues half the entries.
- **Large entries:** B alone already overlaps reads and writes by
  software-pipelining, so the second core adds 3–5%.
- **At 120 tiles every scheme meets the card's copy ceiling,** about 165–173
  GB/s each way.
- **Ops that compute:** the further gain is that B blocks on a `KERNEL` entry
  while the roles run, and NC can keep moving meanwhile. Not yet measured: it
  needs the split in `Session`.

### Why card-wide writes stop at about 160 (NoC #0) and 270 (NoC #1) GB/s

**One channel absorbs a full channel of writes from 120 tiles, on either NoC:**
63.9 GB/s, 99.9% (`gddr_aggregate_nocs`, `Mix::OneChannelWrite`). So neither the
GDDR nor one channel's path is the limit. It appears only when all eight channels
are written at once, which points to links the channels share. Inferred from
dimension-order routing (`NoC/RoutingPaths.md`):

- **NoC #0 (X then Y, east then down):**
  - Write data runs east along the writer's row to a DRAM column (raw X 0 or 9),
    then down that column to the endpoint.
  - Channels 0–3 share column 0's downward links and 4–7 share column 9's: about
    2 × 86 GB/s.
  - Four writers in one row also share that row's link: 79 GB/s.
- **NoC #1 (Y then X, up then left):**
  - Write data climbs the writer's own column, then runs left along the endpoint's
    row.
  - NoC #1 owns port 1 only, and port 1 of channel k and channel k+4 sit in the same
    raw row, so eight channels arrive over four rows.
- **Reads are the mirror image:**
  - Their data leaves each DRAM column along its endpoint rows, then fans out down
    the readers' columns.
  - That is why reads scale and writes don't.

| Card writes (64 KiB, cap 8), tiles | 1 | 4 | 16 | 64 | 120 |
|---|--:|--:|--:|--:|--:|
| NoC #0 | 76 | 79 | 102 | 149 | 161 |
| NoC #1 | 77 | 257 | 296 | 273 | 268 |
| Alternating per entry (`write_noc::ALTERNATE`) | 85 | 146 | 182 | 250 | 263 |

**Port option C: NoC #1 on port 0 for channels 4–7 (`DramChannel::noc1_port`).**
Under tt-metal's assignment (port 1 on every channel) eight channels' NoC #1
writes arrived over four rows. Port 0 of channels 4–7 sits in four other rows and
is never CMFW's endpoint, so now each channel gets its own row:

| Card writes on NoC #1 (64 KiB, cap 8), tiles | 4 | 16 | 64 | 120 |
|---|--:|--:|--:|--:|
| Port 1 everywhere (tt-metal) | 257 | 296 | 273 | 268 |
| **Option C** | **299** | **421** | **382** | **378** |
| Reads on NoC #0 + writes on NoC #1, option C | 312 | 427 | 384 | 396 |

- **Card reads are unchanged:** 469 GB/s at 120 tiles.
- **Every endpoint is still owned by one NoC,** and CMFW's endpoint never sees
  NoC #1. The host reads and writes GDDR through CMFW's endpoint.
- **Stress:** 60 s, then a 10-minute soak (1440 rounds, 127 GiB), on 120 tiles clean.

**Alternating is not additive.** One core issues every request in order, so when
NoC #0's path backs up B stalls on its initiator and stops feeding NoC #1 too. An
independent NoC #1 issuer, NC, would not stall that way. The mode stays available,
but it is not the default.

## Core path: B → T0/T1/T2

The following tables are the pre-streaming baseline. Use the ownership rollout
above and the [scoreboard](performance-scoreboard.md#scoreboard) for current end-to-end measurements.

| Hop (empty kernel, `null_kernel`) | Cycles |
|---|--:|
| host writes the list → B begins it | ~4250 (3.1 µs, mostly PCIe) |
| B list begin → `KERNEL` entry → kick | 44 + 129 |
| kick → each role wakes | 2–9 |
| **role wakes → starts pushing** (T0/T1/T2) | **171 / 183 / 183** |
| start → last word pushed (empty program) | 75–82 |
| last push → retired → acknowledged | 9–10, then 24–26 |
| last ack → B sees all three → list end | 43 + 41 |

- **Pushing** costs 2.91 cycles per word on every role, after the fixed start. The
  backend retires within 10 cycles of the last push, so the roles are push-bound.
- **The host round trip is 4.25 µs,** of which the device is 0.42.

| Op, one tile (µs) | Host | Device | Gather | Kernel | Scatter | Roles busy |
|---|--:|--:|--:|--:|--:|--:|
| matmul 32×256×32 HiFi4 | 108.6 | 14.4 | 10.3 | 2.3 | 1.2 | 14% |
| matmul 128³ HiFi4 | 144.0 | 45.8 | 19.6 | 15.7 | 9.7 | 34% |
| matmul 512³ HiFi4 | 3304 | 2166 | 1060 | 944 | 153 | 43% |
| add, 64 tiles, SFPU | 251.4 | 135.8 | 76.0 | 20.9 | 38.3 | 15% |
| exp, 64 tiles, SFPU | 684.3 | 576.4 | 38.0 | 499.4 | 38.3 | 87% |

- **Roles busy** = the busiest role's program time / device time. Gather, kernel and
  scatter never overlap.
- **Queueing hides nothing:** 16 queued matmuls still cost 114 µs of host time each.

## Ethernet and 800G

Both links are up: X 3 ↔ X 3 and X 13 ↔ X 13.

| One link, 128 KiB transfer | Device | Of 50 GB/s |
|---|--:|--:|
| Staged → landed | 37.9 GB/s | 76% |
| Tensix → Tensix | 18.6 GB/s | 37% |
| Each further byte, staged (slope, 16–128 KiB) | 48–51 GB/s | ~97% |
| Each further byte, Tensix → Tensix | 23 GB/s | 46% |

- **Fixed cost per transfer:** 0.75 µs staged (the record and ack round trip), and
  1.37 µs Tensix → Tensix.
- **E1 is store-and-forward:** NoC read in (1.76 µs for 128 KiB), then the link, then
  the landing wait, then NoC write out (1.88 µs). Nothing overlaps.

| Streams of 32 × 128 KiB | Total (host-timed) | Of ceiling |
|---|--:|--:|
| 1 link, 1 way | 20.0 GB/s | 40% of 50 |
| 1 link, both ways | 27.7 GB/s | 28% of 100 |
| **2 links, 1 way (800G)** | **25.8 GB/s** | **26% of 100** |
| 2 links, both ways | 32.1 GB/s | 16% of 200 |

A send takes 3.46 µs on the device, but each send in a stream takes 6.47 µs. The
difference is the host polling for the ack and posting the next send over PCIe.

## Reader / writer specialisation (2026-10-03)

B only reads GDDR and NC only writes it (`dm::Mover::permits`); each image is the
shared `mover.rs` compiled with its direction's half. Standalone transfers
(uploads, copies, block copies, row gathers and writes, padding fills) became
kernel-less reader/writer packets on a transfer credit channel, so B no longer
writes at all. `SIGNAL` / `WAIT_PEER`, `dm::Peer` and B's NoC #1 write path are
gone: the "B writes on NoC #1" rows and the split-copy experiment above are
history, measured on retired paths.

Image sizes (`.text` + `.rodata` + `.bss`; B's limit 24,576 B):

| Step | B | NC |
|---|--:|--:|
| Before | 24,240 (336 free) | ~22,900 |
| SIGNAL / WAIT_PEER removed | 23,288 | 22,508 |
| B's NoC #1 write path removed | 22,108 | 22,508 |
| B never writes | 20,856 (3,720 free) | 23,096 |
| NC never reads, launches, barriers, tilizes or moves PCIe | 20,856 | 14,840 |

Silicon, card 0, release, against the same tree with B writing (the standalone
transfers as B-only lists; end to end from the host, medians):

- **Compute is untouched.** The streaming sweep (1 / 8 / 32 tiles) is within 3%
  of the last run everywhere it measures the same thing; 64² add fresh 17.4 µs
  on one tile (17.2 before).
- **Models:** MNIST 1.3 ms/step on one tile, 0.9 on four (burn-flex 0.5), 91.96%
  as the golden; transformer 8.72 ms/step (burn-flex 4.34), from 8.8.
- **Large uploads got faster:** 1024² and 8192×1024 host-tilized on one tile
  0.74 -> 0.50 ms and 4.97 -> 3.41 ms (-31%), presumably because B reads the next batch in
  while NC writes the last (not isolated). 512×1024×1024 streamed on 8 tiles 1118 -> 1105 µs.
- **Small uploads on many tiles got slower, 5-8%:** 64×784 on 8 tiles 53.5 -> 57.6 µs,
  on 32 tiles 155 -> 165; 1024² on 8 tiles 464 -> 490; 8192×1024 on 8 / 32 tiles
  2731 -> 2901 / 2769 -> 2945; the streamed 64×784×128 step on 8 tiles 74.9 ->
  74.4 queued, 96.7 -> 98.0 synced (the step is unchanged within noise).
  A one-tile 64×10 upload 7.0 -> 9.0 µs (+2 µs): the handoff, B starting NC and
  waiting for its acknowledged write. What costs the many-tile uploads is the
  packet's extra entries written over uncached PCIe (~6.7 ns a byte): a packet
  carries a header and credits, 9 entries against 6; credits that cannot block
  (the first `depth` reserves, the last pop) are left out, which took the 32-tile
  upload from +16% to +6%. The remaining entries are the protocol's.
- Gates: full silicon suite 296/296 on card 0, `step60_streaming`, `step56`,
  `step62`, `step24` on card 1; `stress_noc_ownership` 137 s, `streaming_stress`
  242 s, `stress_pipeline` 122 s clean.
