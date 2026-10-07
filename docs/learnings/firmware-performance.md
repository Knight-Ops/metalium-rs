# Firmware performance

This is the reference for how fast the firmware is: the data mover, the
B → T0/T1/T2 path, and E1. After each optimization, run `cargo xtask bench`, update
the Scoreboard rows it moved, and add a line to the Change log.

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
| Requests that waited under the in-flight cap (120 tiles × 300 × 16 KiB, one channel) | 8762 per run | — | watch: revisit `MAX_IN_FLIGHT` if it binds below the NoC's own limit |

## Streaming ownership rollout

**Current architecture:** streaming ownership is the only GDDR compute scheduler.
The legacy wave, B-only pipeline and two-host-queue NC schedulers are removed.
`Session::enable_dram(b, nc)` requires both images; Burn rejects retired
`TT_EXECUTION` and `TT_SCATTER` settings rather than silently ignoring them.
`TT_PIPELINE=0` and `TT_BATCH=0` change overlap/waiting, not ownership.
Standalone transfer/control primitives and diagnostic movers remain available.
Known fresh-execution regressions are accepted to consolidate optimization work;
the historical rollout gate did **not** pass. Measurements below predate removal
and remain the legacy comparison baseline. Current sweeps compare serialized
versus overlapped streaming only.

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
and scoreboard above for current end-to-end measurements.

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

## Change log

Newest first. Run = the `target/silicon/bench/<stamp>` it came from.

| Date | Run | Change | Scoreboard effect |
|---|---|---|---|
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
Both-card correctness passed in run `1791145571`; performance remains unmeasured.
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
