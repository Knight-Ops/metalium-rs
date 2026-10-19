# Decode step bandwidth

Roadmap Track 1, SmolLM2-1.7B "Next, by that profile". Written 2026-10-17 against `c69b9c1`. Status: **B0 and B1 done
(2026-10-17; a 1.7B step 42.97 / 45.41 / 48.59 -> 41.83 / 44.25 / 47.40 ms, run `1792209913`); B2 designed, with
B2.0 (replay-fixed-cost P1) recommended next.** Companion plans, which continue under their own names: [decode-matmul-bandwidth.md](decode-matmul-bandwidth.md)
(P5, the channels), [decode-small-ops.md](decode-small-ops.md) (S1d, S4, S5) and
[in-place-cache-write.md](in-place-cache-write.md).

## Why this item

A 1.7B decode step is **42.97 / 45.41 / 48.59 ms** (`1792189445`). Its 6.85 GB of F32 weights take **~14.3 ms** at
GDDR's ~480 GB/s, so the step is 3x its floor. The companion plans each take one part of the step. None of them
accounts for the whole step in bytes, and their per-op rates disagree. decode-matmul-bandwidth says the fused
`gate|up` reads at ~490 GB/s. But its 134 MB of weights in 427 us is ~314 GB/s, or ~359 GB/s counting its activation
reads.

This plan does three things, in order:

1. **B0:** a byte, entry and op accounting of a whole step against the floor, from the trace itself.
2. **B1, lever 2:** the activation read once a unit, a change measurable on its own.
3. **B2, lever 1:** the design of a cross-op weight prefetch, where most of the gap is.

## What the profile shows

Profile `1792190118`, `Step(64)`, card 1: **42.27 ms** a replay over 1,152 ops.

| | Ops | ms | Data | Barrier |
|---|--:|--:|--:|--:|
| Products (`float_matmul`) | 97 | 24.00 | 20.40 | 1.81 |
| Everything else | 1,055 | 18.27 | 11.82 | 2.78 |

One layer, by product:

| Product | Weight | Units | us | Weights' floor at 480 GB/s | Small ops before it |
|---|--:|--:|--:|--:|--:|
| `qkv` | 50.3 MB | 120 | ~180 | 105 us | ~77 us (4 ops) |
| `o` | 16.8 MB | 64 | ~74 | 35 us | ~497 us (32 ops: RoPE, the cache write, attention) |
| `gate\|up` | 134.2 MB | 120 | ~427 | 280 us | ~76 us (4 ops) |
| `down` | 67.1 MB | 64 | ~268 | 140 us | ~49 us (`silu_mul`) |
| **Layer** | **268.4 MB** | | **~949** | **559 us** | **~700 us** |

The head (402 MB, once) is 1.22 ms against an 0.84 ms floor.

**Where the 28 ms between the step and its floor goes:**
- ~9.7 ms is the products running below GDDR's rate.
- ~18.3 ms is the small ops. GDDR streams no weights during them.
- Units 64-119 sit out every op of 64 units or fewer, ~600 us a layer.

## Phases

### B0. A step's bytes, entries and ops (no behaviour change)

**Goal:** a table for each step, built from the captured trace. It gives each op's numbers, and sums them by label and
by model part:
- GDDR read bytes, write bytes and read entries;
- for a product, `A` (activation) and `B` (weights or cache) apart;
- the span from the profile, and the rate achieved.

Then a step budget against the floor:
- weight bytes / 480 GB/s;
- the products' time over that floor;
- the small ops' time with GDDR idle;
- barriers.

Reads are bound by entries as much as by bytes (P1 of decode-matmul-bandwidth: ~0.13 us a request plus ~9 GB/s a
unit). So entries are counted beside bytes.

- [x] **Count at capture** (2026-10-17). `tt_kernels::profile_mode::OpBytes`, in `trace::OpRecord::bytes`:
  - **When:** at `store_trace`, from every unit's stream and both sections of each `PAIR_CALL`, records expanded by
    `tt_isa::dm::record::expand`, as the movers expand them.
  - **What:** GDDR read and write bytes and entries. A gather's reads are split by the L1 slots they land in: `A`'s
    (`rows x kt` slots from `a_at`) and `B`'s.
  - **Switch:** on in the profile mode. `Session::set_count_trace_bytes` (`burn_tt::set_count_trace_bytes`) turns it on
    without the mode, so the simulator runs it. Off, nothing is counted.
  - **Read back:** `TracedInference::ops`, and `TracedGenerator::trace_ops` for a generator's traces.
- [x] **Report it:**
  - `ops.tsv` gains `read_bytes`, `write_bytes`, `reads`, `writes`, `a_bytes`, `b_bytes`, `gbps`;
  - each trace in `summary.md` gains "GDDR traffic by label" and a "Step budget": its bytes at
    `profile_mode::CARD_READ_GBPS` (469, the scoreboard's 120-tile read rate), with the products (ops that gathered
    a `B`) apart from the rest.
- [x] **Host checks** (`decode_products_jobs_rounds_and_bytes`): also the fused `qkv` and `gate|up`.
  - The counter equals the headers' arithmetic, operand by operand, on every job.
  - **Control:** `A` counted as whole tiles does not match it.
- [x] **Gate `step197_trace_bytes`** (ttsim): step193's two synthetic Llamas, unfused and fused.
  - **Claim:** the step's `float_matmul` `B` bytes are the weights' tile slots, computed from the configuration, each
    once.
  - **Controls:**
    - fused reads less `A` than unfused (21.8 against 34.8 KB, and 41.3 against 67.5 KB), with the same `B`;
    - every op moves bytes;
    - counting off counts nothing.
  - **Mutation watched failing:** `A` and `B` attributed the other way round.
  - In `--smoke`.
- [x] **Run** (card 1, the profile mode, `1792202359`). Its `B` bytes are 6,952 MB: the 6.85 GB of weights in
  4,160-byte slots, so the count agrees on silicon. The 135M run is left.
- [x] **The budget** (below).

**The budget, measured** (profile `1792202359`, `Step(64)`, 1.7B, card 1; 42.27 ms a replay):

| | Ops | ms | GDDR MB | Read entries | GB/s | ms at 469 GB/s |
|---|--:|--:|--:|--:|--:|--:|
| Products (`float_matmul`) | 97 | 24.00 | 8,218 read + 107 written | 2.82 M | 347 | 17.8 |
| ... of it weights (`B`) | | | 6,952 | | | **14.8** |
| ... of it activation (`A`) | | | 1,247 (15%) | | | 2.7 |
| Attention's products | 144 | 2.93 | 64 + 26 | 0.12 M | 31 | 0.2 |
| The cache write and its write-back | 240 | 5.93 | 423 + 403 | 0.10 M | 139 | 1.8 |
| Everything else | 671 | 9.40 | 241 + 122 | 0.06 M | 39 | 0.8 |
| **Step** | 1,152 | **42.27** | 8,945 + 658 | 3.10 M | 227 | **20.5** |

- **The cache write:** 48 `scatter_nd` copy lists and 48 `copy_into` lists, each a whole 4.2 MB cache: 0.83 GB a step
  ([in-place-cache-write.md](in-place-cache-write.md)'s copies). The `copy_into`s are the trace's write-back of its
  state, labelled "(outside an op)".
- **The products, by shape:**

  | Shape | Ops | ms | MB read | `A` MB | GB/s |
  |---|--:|--:|--:|--:|--:|
  | 120 units (`qkv`, `gate\|up`, the head) | 49 | 15.81 | 5,619 | 712 | 355 |
  | 64 units (`o`) | 24 | 1.77 | 516 | 107 | 292 |
  | `down`, K-blocked on 64 | 24 | 6.42 | 2,083 | 428 | 324 |

  One layer: `qkv` 59.5 MB in 180 us (335 GB/s), `o` 21.5 MB in 74 us (295), `gate|up` 155.3 MB in 428 us (**367**,
  396 over its data phase), `down` 86.8 MB in 268 us (327).
- **Settled:** `gate|up` does not read at ~490 GB/s, as decode-matmul-bandwidth P3 said.
- **What the step is made of:**
  - the bytes it moves would take 20.5 ms at 469 GB/s;
  - its weights alone 14.8 ms;
  - its products move 347 GB/s over their own time;
  - the other 1,007 ops (18.3 ms, 4.6 ms of it barriers across the step) move 1.4 GB at 70 GB/s, while no weight
    moves.
- **The bytes beside the weights:**

  | What | GB a step | ms at 469 GB/s | Removed by |
  |---|--:|--:|---|
  | `A` | 1.25 | 2.7 | B1 (0.30) and G7 (B2, point 6, the rest) |
  | The cache copies | 0.83 | 1.8 | the in-place cache write |
  | All the rest | 0.6 | | |

- **Targets set by the budget:**
  - **B1:** `gate|up`'s `A` 18.9 -> 8.4 MB a layer and its entries 50,176 -> 40,448 (-19%). The head's `A`
    56.8 -> 8.4 MB, its entries -30%. Bit for bit; worth ~0.6-2 ms a step, depending on whether those products are
    bound by bytes or by entries.
  - **B2:** the 18.3 ms of small ops are where the weights could stream. Their windows a layer are ~77 / ~497 / ~76 /
    ~49 us before `qkv` / `o` / `gate|up` / `down`.

### B1. Lever 2: the activation once a unit (bit for bit)

**What re-reads it.** A unit's later jobs of one op read `A` again only for products of several rounds: `gate|up` (272
jobs on 120 units, units 0-31 run three; ~10.6 MB a layer) and the head (816 jobs, 6-7 a unit). For `qkv`, `o` and
`down` each unit reads `A` once already. What is left there is every unit reading it, which is G7's (B2, point 6).

**Why compute needs no change.**
- Under `Staging::Slots` (every decode product), `A` is planned at `DATA.base` for every job.
  `Requirements::plan` is first-fit in declaration order, and `A` is declared first.
- The unpacker's addresses are baked into the programs.
- So a unit whose next job has the same `A` block can skip reading it.

- [x] **Record** (2026-10-17):
  - `tt_isa::dm::record::GATHER_A_RESIDENT`, bit 25 of `GATHER`'s flags; `gather_kt` reads `kt` past it, and bits
    26-31 make `kt` too large, so they are refused;
  - its expansion skips `A`'s entries, F32 and packed BF16 (`a_resident_gather_reads_only_b`: the same entries less
    `A`'s, every form; a stray high bit refused);
  - `l1_footprint` leaves `A`'s slots out of a resident gather's footprint.
- [x] **Host:** `tensor::Reads { thin, resident_a }` replaces `matmul_dram_limited`'s `thin`.
  - `BlockPlan::push_jobs` sets the flag on a block whose unit's previous block in that call read the same `A` rows to
    the same slots, under `Staging::Slots` only.
  - `Session::set_resident_matmul_a` (on by default) turns it off.
  - The public `tensor::matmul_dram`, and the reference spec it is held to entry for entry, keep it off.
  - Unit test `resident_a_is_read_once_a_unit`, at 120 units: `gate|up` reads `A` once a unit; `B` is unchanged; `q`
    and the K-blocked `down` are untouched.
- [x] **Firmware:** the shared expansion.
  - B's image 26,528 -> 26,692 bytes of 43,008.
  - Clippy plain, with `encoded-guard`, `profile` and `fast-reads`.
- [x] **Gate `step198_resident_activation`** (ttsim; card 0 `1792209694`, fast-reads `1792209888`):
  - **bit for bit** against the flag off, on one unit and on two:
    - `m` of 1 (thin), 17 (whole tiles) and 33 (two tile rows, alternating on a unit);
    - ragged `n`;
    - `B` transposed.
  - **Priming:** before each product, another `A` with NaN padding rows fills the same slots.
  - **Bytes:** captured, the product reads `A` once a unit (two, not four), with `B` unchanged.
  - **Mutations watched failing:**
    - the flag on a unit's first block, which is also what carrying it across ops does (bits: the product computed
      with the other `A`);
    - `B` skipped instead of `A`.
- [x] **Regression:**
  - **ttsim:**
    - `step172`/`step173` safe, `fast-reads`, `fast-reads` + `encoded-guard`;
    - `step18`, `step77`, `step184`, `step185`, `step187`, `step190`, `step191`, `step193`, `step197`;
    - `step12_mnist` (e2e);
    - the `tt-isa` and `tt-kernels` unit tests.
  - **Burn conformance:** 2,973 passed, 52 as expected, none unexpected.
  - **Card 0:**
    - `step185`, `step187`, `step190`, `step191`, `step193` and the mover gates (`1792209694`, fast-reads
      `1792209888`);
    - `step188` (`1792210658`);
    - `--smoke`, 1,105 of 1,105 (`1792210872`).
- [x] **Measured** (card 1, every tile, burn-flex in the same run):
  - **1.7B:** 42.97 / 45.41 / 48.59 -> **41.83 / 44.25 / 47.40 ms** (`1792209913`; 23.9-21.1 tokens/s; burn-flex
    301.7 / 310.8 / 329.4).
  - **135M:** unchanged, 20.03 / 20.55 / 21.19 (`1792210419`). Its products mostly run one round a unit.
  - **Profile `1792210770`:**
    - replay 42.27 -> 41.16 ms;
    - products 24.00 -> 22.83 ms;
    - `A` 1,247 -> 944 MB a step (the predicted -0.30 GB);
    - `gate|up` 428 -> 383 us (entries -19%, now 384 GB/s);
    - the head 1,223 -> 1,049 us.

### B2. Lever 1: cross-op weight prefetch (designed 2026-10-17; build after review)

**Facts** (code, and B0's budget):
- **Every unit waits at every captured barrier,** whether or not it ran the op (`Session::enqueue_work`: a capture's
  `arriving` is every unit).
- **B drains its own reads at every barrier,** `LAUNCH`, `KERNEL`, `WAIT` and chunk end (`TXN`). So a unit streaming
  between two ops holds both ops' barriers, and every unit waits for it.
- **Nothing in L1's `DATA` survives an op;** every builder plans from `DATA.base`.
- **Weights are trace-held,** at fixed GDDR addresses, never written within a step.
- **Who is idle:** units 64-119 sit out every op of 64 units or fewer (~600 us a layer).
- **The windows a layer:** ~77 / ~497 / ~76 / ~49 us of small ops before `qkv` / `o` / `gate|up` / `down`. Only the
  window before `o` (RoPE, the cache write, attention) is long.

**The probe** (`silicon_probe_prefetch_overlap`, card 0, runs `1792210511` on). A unit-range stream,
`Session::probe_prefetch`, lands weight tiles in the top of `DATA` bit for bit, and a chain of adds on units 0-63
computes the same bits under it. It cannot show overlap on the eager path:
- **Eager is host time.** An add's call takes ~290 us to queue on 64 units, and the stream's ~300 us is its own queueing
  on 56 units, so "both" is the sum of the two host costs (2.53 + 0.30 -> 2.84 ms).
- **A replay would show it,** but a captured barrier involves every unit.

So the first phase is the barrier change, and the probe moves into a replay after it.

**The ceiling: L1 caps the prefetch.**
- **Capacity:** a unit has ~630 KB of `DATA` above a product's `A` (266 KB at `K` = 2048). That is ~75 MB across 120
  units, against 273 MB of weights a layer.
- **What could stream, a layer:**
  - `gate|up`'s first round on units 64-119, in the window before `o`: 56 x 532 KB, ~30 MB;
  - `qkv`'s share on units 64-119, during the previous layer's `down` (on units 0-63): ~23 MB.
- **What that is worth:** ~53 MB of 273 off the products' critical path, if their deals then give the prefetched units
  more columns. That is ~185 us of ~950 a layer, **~4.4 ms a step, at most**.
- **What it cannot do:** the small ops' own 18.3 ms is latency (1,007 ops at ~17 us), not GDDR, and streaming weights
  under them does not shorten them. A step near the 14.8 ms floor also needs the small ops shrunk (decode-small-ops,
  the fusers) and the products' own rate raised (P5; G7 below).

**Design, in the order it would be built:**
1. **B2.0, barriers on the units an op ran on in a capture** (replay-fixed-cost P1, open there).
   - A unit that sat out ops waits, wait-only, before its next op, as eager execution's catch-up does.
   - Without it, a prefetching unit holds every barrier it crosses.
   - It is worth ~1.5 ms a step on its own (replay-fixed-cost P0).
   - Then: the probe in a replay, measuring how much of the stream hides and what it costs the small ops.
2. **B2.1, a resident `B`:**
   - `GATHER_B_RESIDENT`, B1's flag's twin: one "resident operand" mechanism;
   - a matmul layout that puts `B`'s slots in a top-of-`DATA` range, `A` and the outputs below;
   - bit for bit, as B1 is.
3. **B2.2, a trace-level L1 reservation:**
   - the top range on the prefetching units, from the prefetch to its product;
   - `l1::Requirements` refuses to plan into it, and the store-time pass checks every op in between.
4. **B2.3, a deal with an offset:** a product's columns dealt from a chosen unit, so the prefetched columns are on the
   units that read them, and the rest is spread to balance the busiest unit.
5. **B2.4, the insertion pass at `store_trace`:**
   - on the `OpRecord`s (X4e's seam), find each product's window on idle units;
   - put `READ_RUN`s into those units' streams at its start, and flip the product's gathers there to `B`-resident;
   - re-chunk;
   - hazard check: nothing between the prefetch and its product writes the weights or plans into the reservation;
   - eager runs are unchanged, and a trace stays bit for bit its eager step.
6. **G7, the activation across units:** `A` is still 1.25 GB a step, 0.95 GB after B1. Every unit of `qkv`, `o` and
   `down` reads it once, and `down` reads 256 `A` entries against 256 `B` a unit.
   - One reader and a multicast to the product's units would remove ~0.9 GB and a third to a half of those
     products' read entries: plausibly ~2-3 ms a step, more than the prefetch's ceiling for less machinery.
   - It needs the same resident flag, plus a semaphore the mover waits on.
   - Non-rectangular multicast is refused on ttsim (`ttsim-divergence.md`).

**Recommendation for review:** build B2.0 next. It pays on its own and is the prerequisite for the rest. Then G7
before B2.1-B2.4: it attacks bytes every product reads, with a ceiling above the prefetch's. The prefetch is worth
building only if B2.0's replay probe shows the stream hiding under the small ops at a useful rate.

- [x] **Probe on the eager path** (above): tiles land, results unchanged, no overlap visible.
- [x] **B2.0** (replay-fixed-cost P1, done 2026-10-17): a captured barrier counts the units its op ran on. No step
  time change (`1792266392`).
- [x] **Superseded (2026-10-18):** B2.1-B2.4 and G7 were not built as designed. In
  [decode-to-bandwidth.md](decode-to-bandwidth.md):
  - G7 is a packed `A`, read as `K / 1024` tiles (Phase 4);
  - the prefetch is the next product's own gather on units idle since the last product, appended to their streams
    at capture (Phase 5). It needs no record change, no L1 reservation and no store-time pass.

## Out of scope

- **BF16 and BFP weights:** they halve or quarter the bytes, and compose with everything here.
- **P5 (channels), S1d/S4/S5 (small ops) and the in-place cache write:** they continue under their own plans. B0's
  accounting measures them too.
- **Split-K** (decode-matmul-bandwidth P4): it changes accumulation order.

## Documentation on completion

The roadmap (this entry and the decode numbers), the performance log and scoreboard with run IDs, silicon operating
notes for any NoC or GDDR finding from the probe, and ttsim divergence for any refusal. Move this file to
`completed-plans/` when done.
