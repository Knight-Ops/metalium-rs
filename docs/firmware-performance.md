# Firmware performance

This is the reference for how fast the firmware is: the data mover, the
B → T0/T1/T2 path, and E1. After each optimization, run `cargo xtask bench`, update
the Scoreboard rows it moved, and add a line to the Change log.

## How to measure

- `cargo xtask bench` runs every benchmark in release, one per process. It writes
  the results to `target/silicon/bench/<stamp>.{jsonl,md}`.
- **Device-timed by default:**
  - Tensix: the tile's debug timestamper, one counter for B and T0–T2.
  - Ethernet: E1's own `mcycle`, gated alone in `silicon_eth_clock` (ttsim does not
    model it; divergence row 71).
  - *Host* figures use `Instant` and include PCIe.
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
GDDR6 reads reach 84% of the card since the NoC ownership rules (change log,
2026-10-02); card-wide writes and 800G are still far off.

| Target | Now | Of ceiling | Goal |
|---|--:|--:|---|
| GDDR6, one tile, 64 KiB entries | 81.6 GB/s | 94% NoC link | hold |
| GDDR6, one tile, 4 KiB entries | 16.3 GB/s | 26% channel | cut per-entry cost |
| GDDR6 reads, card, 120 tiles | 429 GB/s | 84% | 512 GB/s |
| GDDR6 writes, card, 120 tiles (writes on NoC #1) | 266 GB/s | 52% | 512 GB/s |
| GDDR6 reads + writes, card, 120 tiles (writes on NoC #1) | 343 GB/s | 67% | 512 GB/s |
| Ethernet wire, one link (per-byte slope) | 48–51 GB/s | ~97% | hold |
| 800G, both links streaming | 25.8 GB/s | 26% | 100 GB/s |
| Empty kernel, B → T0–T2 → B (device) | 566 cycles (0.42 µs) | — | lower |
| Device busy, 16 queued 32×256×32 matmuls | 12.6% of wall | — | ~100% |
| MNIST training, card 0, 1 tile (`tt-mnist --host`) | 1.0 ms/step | burn-flex 0.5 | ≤ burn-flex |
| Transformer training, card 0, 1 tile (`tt-mnist --model transformer`) | 21.5 ms/step | burn-flex 3.9 | ≤ burn-flex |
| Requests that waited under the in-flight cap (120 tiles × 300 × 16 KiB, one channel) | 8762 per run | — | watch: revisit `MAX_IN_FLIGHT` if it binds below the NoC's own limit |

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

## Where the overhead is (ranked)

1. **The host is in every op and every transfer.** About 100 µs of host work per
   matmul, and about 3 µs per Ethernet send.
2. **Card-wide writes reach half the card.** Card-wide reads are at 84% (fixed
   2026-10-02). NC, still held in reset, is the planned NoC #1 writer.
3. **339 cycles per mover entry (317 at the baseline), about 116 of them before
   the NIU is touched.** The read path's hot code is about 3.0 KB.
   - B's and NC's instruction caches are about 4 KiB each, measured below.
   - So per-entry cost depends on code layout, not just instruction count.
4. **Nothing overlaps within an op:** gather, compute and scatter run in sequence.
5. **E1 stores and forwards through one staging buffer,** plus a 0.75 µs round trip
   per transfer.
6. **Writes:** one DRAM port tops out at ~28.5 GB/s; 128 KiB entries fall to
   27–33 GB/s.
7. **About 180 cycles per role from wake to first push.**

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
- [ ] **Per-request cost** (checklist 9.14).
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
- [~] **B stalls for a whole kernel** (checklist 9.15). Done for GDDR matmuls
  (where `tensor::pipelining_pays`), element-wise ops and reductions (where
  `tensor::pipelined_runs`), on by default. Long sums over rows in chunks
  stay plain: each chunk gathers the sums the last one scattered.
- [~] **The host queues units one at a time** (checklist 9.17): ~6 µs a unit
  an op, now ~3.4 (PCIe reads, separate barrier lists and program-memo misses
  gone). The rest is three posted writes and CPU work a unit: next is one
  list fanned out on the card, or trace replay.
  - `KERNEL` drains the moves and waits for all three roles, so nothing moves
    while they compute.
- [ ] **Ethernet moves one transfer at a time** per direction, store and forward
  through one buffer each way (checklist, Ethernet pipelining).
- [x] **Instruction-cache size:** ~4 KiB on B and NC (`probe_icache`).
- [ ] **Not yet measured:**
  - tile-to-tile L1 over the NoC (the mover has no op for it)
  - card 1 (`cargo xtask bench --device all`)

## Change log

Newest first. Run = the `target/silicon/bench/<stamp>` it came from.

| Date | Run | Change | Scoreboard effect |
|---|---|---|---|
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
