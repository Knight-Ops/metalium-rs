# Decode matmul bandwidth

Roadmap Track 1, SmolLM2-1.7B "Next, by that profile": **the matmuls' bandwidth**. Written 2026-10-16 against
`5ef99ca` plus the row-scatter work (`step189`). Status (2026-10-16): **P0-P2 done** (a 1.7B decode step
57.21 / 59.64 / 62.81 -> 49.78 / 52.27 / 55.44 ms; matmuls 33.8 -> 26.3 ms a step); **P3 done** (47.90 / 50.37 /
53.53 ms; matmuls 24.0 ms). The fused `gate|up` was taken to read at ~490 GB/s, GDDR's ceiling (counted, 367: [decode-step-bandwidth.md](decode-step-bandwidth.md) B0); the narrow products on 64 units
(`o`, `down`) at ~300 GB/s, bound by what the in-flight cap lets a unit hold (next, "What is left").

## Why this item

After the row-scatter cache write, `float_matmul` is **59.5% of a 1.7B decode step: 33.8 of 56.8 ms**
(`Step(64)`, profile mode, card 1, run `1792110380`, `target/silicon/profile/1792110380/`). The step reads the 6.85 GB
of F32 weights at 109-120 GB/s; at the card's measured 469 GB/s for reads (scoreboard) the weights alone take
~15 ms. So the matmuls are about 2x from their floor, and they are the largest lever left on decode.

Targets (set firmly after P0): matmuls 33.8 -> ~18 ms a step, the 1.7B step ~57 -> ~42 ms (~24 tokens/s), every
trace still bit for bit its eager step (`step185`, `step187`, `step188`). Nothing here may change any valid output
bit: each output tile keeps its `K` chain, in the same order.

## What the profile shows (read from `ops.tsv`, run `1792110380`)

One layer is 7 matmuls, all `m = 1` (`[1, 2048]` padded to one tile row):

| Product | Weight | Path | Units | us | Share of step |
|---|---|---|--:|--:|--:|
| `q`, `k`, `v`, `o` | 2048 x 2048, 16.8 MB | `matmul gather`, K whole | 64 | ~95 each | 9.1 ms |
| `gate`, `up` | 2048 x 8192, 67 MB | `matmul gather`, K whole | 120 | ~313 each | 15.0 ms |
| `down` | 8192 x 2048, 67 MB | `K-block matmul gather` | 64 | ~344 | 8.3 ms |
| head (tied, once) | 49152 x 2048, 402 MB | `matmul gather`, `B` read transposed once | 120 | 1,444 | 1.4 ms |

All are 86% data movement, 10% kernels, ~3% barrier. Kernels are not the problem.

## Diagnosis (before P0; P0-P2 below say which held)

1. **Every job re-reads the padded activation from GDDR.** `GATHER` (`tt_isa::dm::record`) reads whole `A`
   tiles (`TILE_SLOT`, 4,160 bytes) for each output block, though `m = 1` uses one row of 32. A `q` job reads 64
   `A` tiles and 64 `B` tiles: **half its bytes are activation padding**, re-read by all 64 units. Counting `A`,
   the products already move ~400-450 GB/s of GDDR traffic (estimates: `q` ~34 MB in 76 us of data, `down`
   ~136 MB in 313 us); only the `B` half is weights. Removing `A`'s padding is worth up to ~1/2 of `q`/`k`/`v`/`o`/
   `down` and ~1/3 of `gate`/`up`.
2. **The activation reads are a convoy.** `A`'s 64 tiles sit on channels `t % 8` (`tensor.rs` placement), and every
   unit walks them in the same `kk` order at the same time, so all units hit one channel, then the next. `B`'s
   column `j` is wholly on channel `j % 8` (64 and 256 tiles wide are multiples of 8): balanced across units, but a
   unit streams from one channel.
3. **Block dealing leaves a tail.** `blocks()` shrinks until there are at least `units` blocks: `gate`/`up`'s 256
   columns become 128 blocks of 2 on 120 units, so 8 units run two jobs (`job j -> unit j % units`) while 112 sit
   idle (inferred from the code; P0 confirms the count). The head likewise.
4. **The 2048-wide products use 64 of 120 units** (64 output tiles, one a job). With 4 KiB entries a safe-mode
   unit reads ~15 GB/s, so 64 units can in principle still saturate GDDR (32 tiles reach 411 GB/s), so this may not
   matter once `A` is gone. P0 decides.

## Phases

### P0. Measure (no behaviour change)

- [x] **Bytes and jobs per product** (2026-10-16, host only: `tt-kernels` unit test
  `decode_products_jobs_rounds_and_bytes`, 120 units, TF32/HiFi4, pipelining on; `-- --nocapture` prints it):

  | Product | Jobs | Busy units | Rounds (units with 2+) | `A` read | `B` read | `A` share |
  |---|--:|--:|--:|--:|--:|--:|
  | `q`/`k`/`v`/`o` | 64 | 64 | 1 (0) | 17.0 MB | 17.0 MB | 50% |
  | `gate`/`up` | 128 | 120 | 2 (8) | 34.1 MB | 68.2 MB | 33% |
  | `down` (K-blocked, 4 gathers a job) | 64 | 64 | 1 (0) | 68.2 MB | 68.2 MB | 50% |
  | head (`B` transposed) | 768 | 120 | 7 (120) | 204.5 MB | 408.9 MB | 33% |

  A 1.7B step's products therefore move **12.06 GB, 5.11 GB (42%) of it `A`'s padding**, in the profile's
  33.8 ms: **~357 GB/s of GDDR traffic**, not the 109-120 GB/s counted from weights alone, and already 76% of
  the 469 GB/s read rate. So the products looked close to byte-bound, and removing `A` (~7 GB a step left) the
  lever, worth ~42% of their bytes at the same rate. (P1 found them bound by entries as much: removing ~37% of a
  `q` job's bytes saved 22% of its time.) `gate`/`up`'s tail is confirmed (8 units run two jobs,
  112 one) and the head's 768 jobs run 6-7 rounds a unit. Left of this item: the achieved rate on silicon
  (the probe below).
- [x] **The silicon probe, replaced by the models.** No separate thin-matmul bench was built. P1 was built with
  both read shapes and measured on the 1.7B and 135M generations directly (below), and the profile mode attributes
  each product's data phase, which answered what this item was for.
- [x] **Answer before P1: bytes or entries?** Both, and entries decide the shape (P1, "Measured"):
  - two face-row reads a tile (2x the entries, 1/32 the bytes) made the 135M step ~1 ms slower;
  - one prefix read (the same entries, ~1/4 the bytes) made both models faster.

  **Decision: one read a tile.**
- [x] **Facts checked in code:**
  - a GDDR slot's datums start at `TILE_DATA` (16), after a 16-byte header (`dm.rs`);
  - `Descriptor::decode` accepts any nonzero length congruent mod 64, and records never reach the fast-reads encoder
    (they expand on the card);
  - an `m = 1` product's `A` is already used with its padding off `K` (`matmul_padding_off_k`), and the output's claim
    is `Undefined` unless `A`'s was zero.

  So unread `A` rows need **no zeroing**: they reach only the output's padding rows, claimed `Undefined`, and consumers
  that need zero padding already refill it (`Session::meet`). This replaced P1's "zero the unread rows" item.

### P1. Read only the activation rows a product uses (done 2026-10-16)

- [x] **Record:** `GATHER` flag bits 3-7 carry `r` (`record::gather_rows`):
  - 1..=16, or 0 for whole tiles;
  - untransposed `A` only; refused with `A` transposed or past 16;
  - `A`'s last tile row of the block is read as one entry a tile: its datums through row `r - 1` of face 1, `1024 + 64 r`
    bytes.

  Packed BF16 needs no flag. A tile with at most 16 valid rows (from its metadata) is read through its last valid row of
  face 1 the same way, for both `A` and `B`, since its fill already zeroes the rest. Unit tests pin addresses and lengths
  to `face_index`.
- [x] **No zeroing** (P0's facts): the output's padding claim is `Pad::Undefined` whenever the thin read was used
  (`Session::matmul_dram`).
- [x] **Host:**
  - `BlockPlan` sets `r` on the gathers of `op(A)`'s last tile row (`push_jobs`, `push_k_jobs`) when
    `tensor::thin_rows(m, a_t)` is nonzero;
  - `Session::set_thin_matmul_reads` (on by default) turns it off for gates;
  - `matmul_dram_batched` (attention) keeps whole tiles: not measured to pay.
- [x] **Firmware:** B's image 25,724 -> 26,528 bytes of 43,008; firmware clippy plain, with `encoded-guard` and with
  `profile`.
- [x] **Gate `step190_thin_matmul`** (ttsim; card 0 runs `1792113956` and `1792115747`, fast-reads `1792116118`):
  - **Bit for bit against whole tiles,** for `m` in {1, 2, 15, 16, 17, 33}, `K` whole, ragged (70) and blocked, ragged
    `n` and `B` transposed. L1 is first primed by another matrix whose padding rows are NaN, so a short or misplaced read
    lands on other values.
  - **`A`'s padding rows in GDDR set to NaN:** whole tiles carry them into the output's padding (the control), and the
    thin read never touches rows 16-31.
  - **Stale padding is invisible downstream:** a thin product's non-finite padding does not reach a row sum, a column
    max, a product with it as `A`, or `C^T @ C`.
  - **Negative control:** a false `Pad::Zero` claim makes `C^T @ C` non-finite.
  - **Mutations caught:** one row too few, face 1 never read, every tile row read thin, the `Undefined` claim dropped.

  BF16 is covered by `step77` (ragged rows, NaN-poisoned padding, exact oracle), unchanged and passing.
- [x] **Regression:**
  - ttsim:
    - `step172`/`step173` in both read modes and with `encoded-guard`;
    - `step18`, `step184`, `step77`, `step187`;
    - `step12_mnist` (e2e).
  - Burn conformance, with no unexpected result (`1792113791`, `1792117214`).
  - Card 0:
    - `step185`, `step187`, `step188` and the mover gates (`1792115747`, `1792118267`);
    - `--smoke`, 1,083 of 1,083 (`1792116807`).
- [x] **Measured** (card 1, every tile, burn-flex in the same run;
  [log](../learnings/performance-log.md#decode-matmuls-read-only-the-activation-rows-they-use-2026-10-16-cards-0-and-1)):
  - **1.7B:** 57.21 / 59.64 / 62.81 -> 51.15 / 53.55 / 56.71 ms (`1792115180`).
  - **135M:** 24.40-25.53 -> 24.07-25.21 (`1792115571`).
  - **The products** (profile `1792115848`):
    - `q`-shaped: 94.9 -> 73.8 us;
    - `down`: 343 -> 268;
    - the 120-unit ones: 336 -> 286.
  - **Now entry-bound:** a `q` job's 128 entries take ~60 us a unit (~0.45 us each, ~290 GB/s card-wide).

### P2. Deal thin products evenly (done 2026-10-16)

- [x] **The plan's premise did not hold.** The L1 plan caps a K-whole `m = 1` block of `K` = 2048 at two columns (64
  `A` + 128 `B` slots of the arena's ~215), so 256 columns cannot go one job a unit on 120 units. Instead,
  `tensor::thin_runs` deals whole rounds of the plan's width, then the remainder as even runs over the units:
  - `gate`/`up`: 8 units with 2 + 2 columns become 16 with 2 + 1, so the busiest unit has `ceil(nt / units)`, the
    least any deal gives;
  - the head: 14 columns become 13.

  It applies only with `K` whole, unpipelined (`Staging::Slots`), one tile row, and more columns than units.
- [x] **Host only;** each output tile's accumulation is unchanged:
  - unit test `thin_runs_give_no_unit_more_than_its_share`;
  - gate `step190` `columns_dealt_evenly_equal_one_units_product`: two units against one (`Spread::Fixed(1)`, where
    the deal is `blocks`'), bit for bit, with thin reads on and off and `B` transposed.

  The host-staged `Session::matmul` is not that oracle at `K` = 2048, because it blocks `K` its own way.
- [x] **Measured:**
  - **1.7B:** 51.15 / 53.55 / 56.71 -> **49.78 / 52.27 / 55.44 ms** (`1792118484`; 20.1-18.0 tokens/s).
  - **135M:** unchanged, 24.14-25.25 (`1792118871`); its products have no tail.
  - **The products** (profile `1792119119`): `gate`/`up` 286 -> 241 us; all products 27.55 -> **26.32 ms** a step.
  - **Program cache:** 155.6 KB of 240 KB.

**Against the targets:** the matmuls were to reach ~18 ms a step and the step ~42 ms. They are at 26.3 and ~49.4, and
the rest is entries. Next:
- **P3** puts `q|k|v` on 120 units in one op (three ops on 64 today), and `gate|up` in one. The estimate from the
  profile is ~5 ms a step.
- **Fewer B entries a tile** (a tighter record-expansion loop, or larger reads where tiles are contiguous) is a
  firmware item beyond this plan.

### P3. Fused projections (model level, bit-identical; done 2026-10-16)

- [x] **`Llama::fused`** (`tt-models/src/llama.rs`):
  - each layer's `q|k|v` becomes one `[h, q + 2 kv]` matrix and `gate|up` one `[h, 2 ff]`, concatenated once after
    strict loading;
  - the outputs are split by column views (`project_split`);
  - the separate weights become views of the fused one, so storage is shared and parameter paths stay as loaded;
  - a group whose matrices differ in dtype stays unfused;
  - not for a model a `quant` plan rewrites (its BFP storage is invisible to generic code, and its rules name the
    separate matrices).

  A layer goes from 7 products to 4. The bench fuses by default (`TT_GEN_FUSED=0` keeps them apart), counting its
  weight bytes before fusing.
- [x] **Gate `step191_fused_projections`** (ttsim; card 0 run `1792161551`, with `step185` and `step187`):
  - fused against unfused, bit for bit, eager (the growing cache) and traced, on step187's Llama shapes and on the
    135M checkpoint (all 30 layers on silicon);
  - `4 layers + 1` products a forward, against `7 layers + 1`;
  - **negative control:** scaling `v_proj` after fusing changes the unfused model's generation and not the fused
    one's.

  Found on the way: a Burn module's lazily initialised parameters are initialised again, to other values, by a
  clone taken before their first use, so the gate materialises them first.
- [x] **Measured** (card 1, every tile, burn-flex in the same run):
  - **1.7B:** 49.78 / 52.27 / 55.44 -> **47.90 / 50.37 / 53.53 ms** (20.9-18.7 tokens/s, `1792161623`).
  - **135M:** 24.14-25.25 -> **22.91 / 23.35 / 23.98** (`1792162031`, burn-flex 26.43 / 28.21 / 30.65).
  - **Profile `1792162257`:**
    - `qkv` 180 us, against three products of 74 (37 us of it unit 0 waiting at the barrier: 192 columns go 2 to
      72 units and 1 to 48);
    - `gate|up` 426 us, against two of 241: ~490 GB/s card-wide, the ceiling. **Refuted by the byte count**
      ([decode-step-bandwidth.md](decode-step-bandwidth.md) B0): 155 MB in 428 us, 367 GB/s;
    - all products 26.32 -> 24.01 ms a step.
  - **Program cache:** 157.2 KB of 240 KB.

**What is left of the products** (24.0 ms; ~9.5 GB a step, ~395 GB/s on average; the weights alone ~14.3 ms at
480 GB/s):
- `o` and `down` run on 64 units at ~300 GB/s. **Not the in-flight cap:** a mover started at 16, 32 or 128 requests
  in flight (`dm::TILE_IN_FLIGHT_CAP` is 8) leaves the 1.7B step at 47.94 / 47.96 / 48.20 ms against 47.90 (runs
  `1792163344`, `1792163754`, `1792164166`; an experiment, not kept).
  - **What the numbers fit:** a `q` job is 128 entries in 76 us with whole tiles and 54 us thin, which fits ~0.13 us
    a request plus ~9 GB/s a unit.
  - **A likely reason, not yet measured:** each unit streams a weight column's tiles from one channel (`j % 8`),
    eight units to a channel.
  - **Next:** the probe P0 planned (a unit's `B` stream alone, `READ_RUN`, on 64 and 120 units, and with the
    columns spread over channels), then a placement that spreads a job's `B` tiles if channels are the bound.
- `qkv`'s uneven last round (2 and 1 columns).
- `A` read again by every job (~70 KB a job, thin). A unit's later rounds are
  [decode-step-bandwidth.md](decode-step-bandwidth.md) B1; every unit reading it once is that plan's B2 (G7).

The original P3 items, done as above (`step188` not rerun: `step191` covers the 135M checkpoint fused):

- In `tt-models`, generic over `LlamaConfig`: `q|k|v` as one `[h, q + 2 kv]` weight and `gate|up` as one
  `[h, 2 ff]`, concatenated once after strict loading (checkpoint names unchanged), outputs split by views. A
  layer goes from 7 products to 4, `qkv` runs on 120 units instead of three ops on 64, and every output tile keeps
  its `K` chain, so logits are **bit for bit** the unfused model's.
- Fuse only matrices that share a format (`quant.rs` names one per matrix); a mixed plan keeps them apart.
- Gate: fused against unfused, bit for bit, on the 135M preset and on `step187`'s shapes (GQA 4:1, head widths
  32 and 128, untied head); the GPT-2-style decoder is already fused. Then `step185`, `step187`, `step188`.

### P5. Why a narrow product's unit reads ~9 GB/s (open; next session)

After P3 the products are 24.0 of a 47.5 ms 1.7B step (`1792162257`): `gate|up` reads at 367 GB/s (byte count, decode-step-bandwidth B0; ~490 was estimated), and `o` (74 us)
and `down` (268 us), on 64 units, move ~300 GB/s, and `qkv` (120 units) ~330. GDDR's ceiling for the step's ~9.5
GB of products is ~19.8 ms, the weights' alone ~14.3. The in-flight cap is ruled out (P3, "What is left").

**Refuted (2026-10-17, [decode-to-bandwidth.md](decode-to-bandwidth.md) Phase 0, `silicon_probe_decode_stream`):** dealt a column a unit, a weight's stream alone reads at 446-458 GB/s on 64 units, ~7% below consecutive tiles (485-489). The products' gap is `A`'s reads, the kernel tail and the deal, not channels. The original hypothesis, kept:

**The leading hypothesis: one channel a column.** Tile `t` of a tensor is on channel `t % 8` (`Placement::slot`)
and tile `(kk, j)` is `t = kk ct + j`, so with `ct % 8 == 0` (every decode weight: 64, 256, 1536 tile columns) a
column's 64 `K` tiles are all on channel `j % 8`, at slot stride `ct / 8`. A job of one or two columns streams from
one or two channels, eight units to a channel at 64 units. `A`'s tiles walk the channels in `kk` order on every
unit at once (a convoy; small since P1).

- [ ] **P5.0 Probe, host only, no firmware change** (`silicon_bench_tensix_ops` or a new
  `silicon_probe_matmul_channels`; release, card 0, warmups and medians, every output validated bit for bit
  against `matmul_dram`):
  - **The control:** `[1, 2048] @ [2048, 2048]` and `[1, 8192] @ [8192, 2048]` with `B` as is (`ct = 64`), on 64
    and 120 units (`Session::set_spread`).
  - **The skew, through the batched path's blocks:** `B` a block of a `[2048, 2080]` parent (`ct = 65`), so tile
    `(kk, j)` sits on channel `(kk + j) % 8` and a column cycles all eight channels. `matmul_dram_batched` takes a
    `Block` of a resident tensor and reads it at the parent's stride (`block_ref`).
  - Compare batched against batched (it keeps whole `A` tiles and `blocks`' deal), and whole against thin is
    already known.
  - **The ceiling:** each unit's `B` tiles alone as `READ_RUN` records (as `silicon_probe_read_loop`
    `records_against_entries`), at both strides.
  - **Attribution** with the profile mode (`--profile`), not new instrumentation.
- [ ] **Answer:** does the skew take the 64-unit products from ~300 toward ~450 GB/s? If not, which of the
  per-request cost (~0.13 us), the per-unit rate (~9 GB/s), or NoC routing bounds them (the operating notes' "What
  holds card reads" and the read-loop probe have the single-tile numbers).
- [ ] **P5.1 If channels are the bound: a skewed placement for weights.** The options, cheapest first:
  - (a) Allocate a matrix whose tile row is a multiple of 8 one tile column wider (`ct + 1`, 1.6% more GDDR at
    `ct = 64`) and use it as a view: `DramTensor` then needs a row stride apart from its width (today `ct` is
    `grid()[1]`). That touches every record's `TensorRef` (`ct` is already a field) and every host builder that
    assumes `ct == cols / 32`. Audit with `find_references(DramTensor::tensor_ref)`.
  - (b) A placement whose channel of tile `t` is `(t + t / ct) % n` (a per-row rotation): a `TensorRef` flag the
    cursor and `tile()` honour, in `tt-isa` (firmware) and `Placement::slot` (host). No padding, but a record
    format change: the mover gates (`step172`, `step173`) in both read modes, and `encoded::validate`.
  - Either way, every output keeps its `K` chain, so the gates hold bit for bit; add a gate for the placement
    itself (tiles land where `slot` says, views and ragged edges).
- [ ] **P5.2 `qkv`'s uneven last round** (192 columns: 72 units of 2, 48 of 1; unit 0 waits 37 us at the barrier).
  Measure whether a deal of 192 as 120 runs that end on the longest unit costs less, or whether `q|k|v` should
  deal its K-blocked form.
- [ ] **Measure** as P1: `traced_generation` 1.7B and 135M on card 1, burn-flex in the same run, then `--profile`.

### P4. Only if P0-P3 leave narrow products short of the floor

- [ ] Split-K for products with fewer output tiles than units (`o`, `down`): partial sums on two units, then one
  F32 add. This **changes accumulation order** (not bit for bit with today's), so it needs a derived bound and an
  explicit decision before it is built. Do not start it unless the measured share justifies it.

## Out of scope

- **BF16 and BFP weights** (the next roadmap item): halve or quarter the bytes. P1 covers `GATHER_BF16`, so they
  compose.
- **Multicast of `A` (G7):** the structural fix for large `m` (prefill), where `A` is real data, not padding.
- **The cache write's two whole-cache copies** ([in-place-cache-write.md](in-place-cache-write.md)).

## Documentation on completion

Roadmap (the "matmuls' bandwidth" entry and the decode numbers in "Where the project stands"), the performance log
and scoreboard with run IDs, silicon operating notes for any GDDR read finding (partial-slot reads, channel
convoys), ttsim divergence for any simulator refusal. Move this file to `completed-plans/` when done.
