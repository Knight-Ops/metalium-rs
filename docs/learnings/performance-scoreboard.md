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
| GDDR6 reads, card, 120 tiles | 469 GB/s | 92% | 512 GB/s |
| GDDR6 writes, card, 120 tiles (writes on NoC #1) | 389 GB/s | 76% | 512 GB/s |
| GDDR6 reads + writes, card, 120 tiles (writes on NoC #1) | 403 GB/s | 79% | 512 GB/s |
| Ethernet wire, one link (per-byte slope) | 48–51 GB/s | ~97% | hold |
| 800G, both links streaming | 27.0 GB/s | 27% | 100 GB/s |
| Empty kernel, B → T0–T2 → B (device) | 566 cycles (0.42 µs) | — | lower |
| Device busy, 16 queued 32×256×32 matmuls | 12.6% of wall | — | ~100% |
| MNIST training, card 0, 1 tile (`tt-mnist --host`) | 1.4 ms/step | burn-flex 0.5 | ≤ burn-flex |
| Transformer training, card 0, 1 tile (`tt-mnist --model transformer`) | 8.8 ms/step | burn-flex 4.35 | ≤ burn-flex |
| ResNet-18 forward, 224x224, batch 1, card 0, host (`silicon_bench_models`) | 16,288 ms, 211 reported ops (`1791846735`) | burn-flex 154.8 ms (105x slower) | max pool 2.8 s and 0.5-1.4 s a convolution: one 32-row chunk, one host-built table and one small matmul or reduction at a time; a chunk-free convolution and pooling (roadmap Track 2) |
| ViT-Tiny forward, 224x224, batch 1, card 1, host | 815 ms, 709 reported ops (`1791846823`) | burn-flex 104.7 ms (7.8x slower) | about 1.1 ms an op: host submission; fusers and traces (roadmap Track 2) |
| SmolLM2-135M decode step, embedding lookup excluded, card 1, host | 619 ms, 2,779 reported ops (`1791846823`) | burn-flex 25.0 ms (24.8x slower) | about 220 us an op; `cat` is `slice_assign` (722 calls a step); fusers and traces |
| SmolLM2-135M decode step from a token id, lookup included, card 0, host | 623 ms (p10 525, p90 709), 2,781 reported ops (`1791852097`) | burn-flex 26.4 ms (23.6x slower); the same run without the lookup 622 ms | the lookup is about 1.4 ms of it; the rest is the host submission above |
| SmolLM2-135M embedding lookup, one token from 49,152 x 576, card 0, host (`probe_embedding_cost`) | 260 us, median of 7 warm, validated against the host's rows (`1791851940`); five tokens 278 us, 32 tokens 344 us | was 15.9 s (`1791845218`): `gather_native` staged 28.3 million host addresses a token; now `Session::select_rows` reads only the rows it returns | hold; BF16, F16 and BFP tables still take the staged path (roadmap Track 2, "Gather without staging") |
| SFPU precise `gelu`, three-role kernel, device cycles per tile (`probe_ttinsn_push`, card 0) | 38,053 cycles (T1 push 3.31 cycles/word: push-bound) | `.ttinsn` fused push, experiment only: 13,734 cycles, 2.77x (run-time translated: 2.67x) | decide the follow-on (roadmap Track 2) |
| Accurate matmul (opt-in `MatmulMode::Accurate`), F32 `[256, 256] @ [256, 256]`, card 0, host | 10.96 ms (`1791818553`) | the fast Matrix Unit matmul: 277 us (39.6x); 19.1x to 64.4x over seven shapes, 18.7x at 1024 cubed; the exact mode is 750x slower than this at 256 cubed | the element-wise slicing and NaN/infinity classification are 64% to 86% of it (127 device jobs a matmul): fuse the slicing into one SFPU kind a slice, cheapen the classification, one matmul for the pairs; see roadmap, "Fused slicing for `MatmulMode::Accurate`" |
| Exact matmul (opt-in `MatmulMode::Exact`), F32 `[256, 256] @ [256, 256]`, card 0, host | 8.19 s (about 0.5 us per product, 16 chunks; `1791761944`) | the fast Matrix Unit matmul: 376 us (22,000x); 64x32x64 1,000x, 1024x27x32 3,300x | cut the host-side table cost: a k-loop of broadcast multiply-adds (m + n addresses a step) instead of two `k x outputs` gathers; see roadmap Track 1 |
| Requests that waited under the in-flight cap (120 tiles × 300 × 16 KiB, one channel) | 8762 per run | — | watch: revisit `MAX_IN_FLIGHT` if it binds below the NoC's own limit |

### Model ladder: now against target

Pretrained models, batch 1, F32, release, medians, burn-flex timed in the same run (`silicon_bench_models`; method and
conditions in the [log](performance-log.md#model-ladder-real-models-on-the-card-2026-10-12-cards-0-and-1)). Update a
row in place when its fix lands, add the run ID, and keep the old row in the log. Work items: roadmap Track 2, items 1-7.

| Workload | burn-flex | Now (run) | Slowdown | Target | Roadmap item |
|---|--:|--:|--:|---|---|
| ResNet-18 forward, 224x224 (card 0) | 154.8 ms | 16,288 ms (`1791846735`) | 105x | at or below burn-flex | 2 |
| ViT-Tiny forward, 224x224 (card 1) | 104.7 ms | 815 ms (`1791846823`) | 7.8x | at or below burn-flex | 4 |
| SmolLM2-135M decode step, lookup excluded (card 1) | 25.0 ms | 619 ms (`1791846823`) | 24.8x | at or below burn-flex | 3 |
| SmolLM2-135M embedding lookup, one token (card 0) | not measured | 260 us (`1791851940`; was 15.9 s, `1791845218`) | n/a | milliseconds: **met** for F32 | 1 |
| SmolLM2-135M decode step from a token id, lookup included (card 0) | 26.4 ms | 623 ms (`1791852097`) | 23.6x | at or below burn-flex | 3 |
| SmolLM2-135M BF16 decoded token, six layers, lookup included (card 0) | not measured | 37.5 s (`1791848957`) | n/a | no slower than F32 | 5 |
| Traced rows for the three models | n/a | not measured | n/a | measure once segmentation lands | 6 |

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
| Model forward passes (ResNet-18, ViT-Tiny, SmolLM2) | [Model ladder](performance-log.md#model-ladder-real-models-on-the-card-2026-10-12-cards-0-and-1) |
| Embedding lookup, SmolLM2 decode from a token id | [Row select](performance-log.md#row-select-an-embedding-lookup-without-staging-2026-10-13-card-0) |
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
