# Decode at GDDR's bandwidth

Roadmap Track 1, SmolLM2 single-user decode. Written 2026-10-17 against `c69b9c1` (plus the uncommitted B0/B1 work of
[decode-step-bandwidth.md](decode-step-bandwidth.md)). Status (2026-10-17): **a 1.7B step 41.83 / 44.25 / 47.40 -> 36.39 / 37.02 / 37.98 ms (`1792280991`), 135M
20.03 / 20.55 / 21.19 -> 15.33 / 15.79 / 16.47 (`1792281384`).**
- **Phase 0, done:** the read ceiling is ~488 GB/s, and the products' gap is not the read stream.
- **Phase 1:** the copies are done; the k/v repack moved to Phase 2.
- **Phase 3:** the capture barriers and the row-limited element-wise ops are done.
- **Phase 2:** A0 found the transposed reads of the key cache; `B` is now transposed by the unpacker (A0b), and a
  stacked product's tail rows are no longer read (A2a), nor its padding off `K` filled (A1).
- **Next:** Phase 2's A1-A2 (the batched path's padding, no regathering), then Phase 3's fusers. This is the umbrella plan.
It orders the open items of the companion plans and adds two new ones:
- [in-place-cache-write.md](in-place-cache-write.md) (D1-D5);
- [decode-small-ops.md](decode-small-ops.md) (S1d, S4, S5);
- [decode-matmul-bandwidth.md](decode-matmul-bandwidth.md) (P5);
- [decode-step-bandwidth.md](decode-step-bandwidth.md) (B2, G7);
- [replay-fixed-cost.md](replay-fixed-cost.md) (P1);
- new: [decode-attention.md](decode-attention.md), and Phase 3's row-limited element-wise ops and fusers below.

## Target

**A decode step at 90-95% of GDDR's bandwidth, over the whole step.** Batch 1, F32 weights.

- **The step's bytes:** 6.95 GB of weight slots, plus the cache (≤0.2 GB at 512 positions), plus at most a few tens of
  MB of activations.
- **What 90-95% means:** at 461-486 GB/s (90-95% of 512), **~14.5-15.5 ms a 1.7B step**. Today it is 41.83 / 44.25 /
  47.40 ms (`1792209913`), 2.7-3.1x that.
- **Against the ceiling:** the best whole-card read rate measured is 469 GB/s (92% of 512). The target is at that
  ceiling, or slightly past it.
- **Against the roofline:** the scoreboard's roofline calls 65% "realistic" (25.0 ms). That is a waypoint, not the
  target.
- **135M:** the same problem, made only of fixed cost (~20 ms a step against ~1.2 ms of weights).

**Rules:**
- **Bit for bit:** every phase keeps today's decode exactly.
- **Order changes are decisions:** split-K and split-position attention change accumulation order. They come back
  for a decision with measured numbers and are not built unasked. The target probably needs one of them.
- **Not here:** BF16 weights and batched decode are separate items.

## Where the time goes

Profile `1792210770`, card 1. The table gives trace `Step(64)` (41.16 ms a replay) and `Step(512)` (46.98).

| Part | Step(64) ms | Step(512) ms | At target | What is wrong |
|---|--:|--:|--:|---|
| Products (97 ops) | 22.83 | ~22.8 | ~14.8, all of the step | 351 GB/s overall; `o` and `down` on 64 of 120 units at ~300 GB/s; `A` 0.94 GB; 2.54 M read entries; `qkv`'s uneven last round |
| Attention (~480 ops) | ~5.2 | **~10.9** | ~0.4, hidden | `qk` reads K at ~30 GB/s at 512; row gathers 1.7-3.7 ms; the softmax is 7 ops; fill-pads of operands padded only past `M` |
| Cache write (240 ops) | 5.94 | 5.94 | ~0 | whole-cache copy and write-back 2.26; the k/v repack and block copy 2.40; the row scatter on 2 units 1.29 |
| RMSNorm, RoPE, SiLU, residuals (~520 ops) | ~7.2 | ~7.2 | hidden | 4-7 ops each; kernels run every row group for a 1-row activation |
| *Barriers, inside the above* | 4.77 | 5.2+ | | every unit arrives at all 1,152 barriers |

Attention is the whole context-length slope: `Step(64)` to `Step(512)` adds 5.8 ms, all of it in `qk`, `pv` and their
row gathers.

## Strategy

**Two halves, in this order:**
1. **Make every non-product op small** (Phases 1-3): bit-identical removals and fusions, from ~700-1,000 us a layer
   to ≲100 us.
2. **Run what is left under the weight stream** (Phase 5), with the products at the read ceiling (Phase 4).

**Why the order matters:**
- Hiding 100 us a layer at 469 GB/s takes ~47 MB of weights in flight.
- L1 has ~75 MB of free `DATA` across the units (decode-step-bandwidth B2).
- B2 capped the prefetch at "~4.4 ms at most" against today's ~700 us windows. Once the windows are ≲100 us, L1's
  capacity no longer binds.

## Phases

### Phase 0. The ceiling and the meter (done 2026-10-17)

- [x] **The read ceiling.** Probe `silicon_probe_decode_stream`, with `Session::probe_stream` (card 0, `1792261830`,
  `1792261878`, `1792261957`, `1792262011`;
  [log](../learnings/performance-log.md#decode-products-against-their-weight-stream-2026-10-17-card-0),
  [operating notes](../learnings/silicon-operating-notes.md#a-decode-products-weights-what-the-card-reads-at-2026-10-17-card-0)).
  - **The ceiling:** a weight's consecutive tiles read as `READ_RUN`s on 120 units reach **485-489 GB/s, 95% of 512**,
    in the default image. So 95% is the read stream's own rate, and the 1.7B weights' floor is **~14.3 ms**.
  - **Channels:** dealt a column a unit (one channel a column), 446-458 GB/s on 64 or 120 units: ~7% below. P5's
    channel hypothesis is not the products' bound.
  - **Units:** 64 units saturate GDDR (~7 GB/s a unit). `qkv`, `gate|up` and the 1.7B head are *faster* on 64 than
    on 120 (160 / 352 / 1,024 against 180 / 383 / 1,051 us); the 135M head prefers 48-120.
  - **The products:** an `m = 1` product takes 1.2-1.8x its weights' stream on the same units. `o` is 74 against 38
    us, `down` 269 against 150, `qkv` 160 against 113, `gate|up` 352 against 300. The profile mode attributes the
    rest:
    - `A`'s reads, an entry a `K` tile on every unit: 64 at `K` = 2048, 256 for `down`;
    - a ~15 us kernel tail after the last read;
    - an uneven last round's barrier: `qkv` 38 us, `gate|up` 21, `down` 18.
- [x] **The step's GB/s as one number.** The profile's step budget already gives it: the "all" row, bytes over the
  replay (226 GB/s in `1792210770`). The bench's "decode weight bandwidth" gives the weights' share (144-164 GB/s).
  Both are tracked from here.

### Phase 1. The cache write, 5.9 -> ~1.3 ms (copies done 2026-10-17)

- [x] **In-place write**, [in-place-cache-write.md](in-place-cache-write.md) D1-D4 (gate `step199`). A capture's
  state is donated before its step, and a row scatter into it writes the state's own buffer. A stale read is refused
  by name, and the step needs no write-back.
  - **Also found:** the write's index column was copied at every write; it is now read as the row it is.
  - **Measured (card 1):**
    - 1.7B **41.83 / 44.25 / 47.40 -> 38.97 / 41.41 / 44.58 ms** (`1792263398`);
    - 135M **20.03 / 20.55 / 21.19 -> 18.59 / 19.07 / 19.79** (`1792263801`).
- [ ] **k and v without the repack** (1.83 ms; in-place-cache-write D5, decode-small-ops S1d). It is the cache's
  head-major layout, so it moves to Phase 2: a position-major cache (`[L, kv * d]`) makes the write the
  projection's row as it is.
- [ ] **The row scatter on more units** (1.29 ms on 2 units). Probe it first.

### Phase 2. Decode attention, ~5.2 / 10.9 -> ≲1 ms

[decode-attention.md](decode-attention.md): A0-A5. It all lives in burn-tt's `ModuleOps::attention` override, so any
Burn model gains.

### Phase 3. Each op's fixed cost and the op count, ~7 -> ≲2.5 ms exposed

- [x] **Captured barriers on the units an op ran on** (2026-10-17; replay-fixed-cost P1 = decode-step-bandwidth B2.0;
  gate in `step176`). **Measured: no change** (1.7B 38.93 / 41.38 / 44.58 ms, `1792266392`). A capture that syncs
  mid-run (the profile mode) lost its waits in the first version: fixed, see replay-fixed-cost P1. An idle unit's arrival
  cost nothing measurable. It is kept because Phase 5 needs it: a streaming unit no longer holds other ops'
  barriers. The plan below, kept:
  - **The changes** (from `Session::enqueue_work_inner`):
    - eager's `arriving` set, used in a capture too;
    - a wait-only `BARRIER` for a unit that is behind, where a capture now errors;
    - the trace's cumulative arrivals, recorded for the replay's counter;
    - one closing barrier on every unit.
  - **Gate:** `step186_trace_barrier_participants`.
  - **Worth:** ~1.5 ms. Phase 5 needs it.
- [x] **Row-limited SFPU element-wise kinds** (2026-10-17). The pattern of RMSNorm's statistic:
  - **The limit:** a program built under `sfpu::with_rows(r, ..)` runs each whole-tile row loop
    (`for_each_row_group(64, ..)`) over only `reduce::row_offsets(r)`, unrolled, through the same
    unrolled-loop bookkeeping.
  - **Which kinds:** `sfpu::ops::row_limited` lists the lanewise kinds: add, subtract, multiply, divide, the scalar
    add and multiply, sigmoid, `silu(gate) * up`. `code_for_rows` builds them, and `sfpu_programs` keys on the rows.
  - **When:** `tensor::eltwise_rows` uses the limit for a single tile row of fewer than 32 rows, and the session
    claims the output's padding `Undefined`.
  - **Unit gate:** `row_limited_programs_keep_the_computed_rows`, on the specification model: every kind and
    broadcast at 1-31 rows. The computed rows (whole groups of four) are the whole-tile program's bits, the rest
    are left as `Dst` held them, and the next group is not computed.
  - **Device gate:** `step200_row_limited_eltwise` (ttsim; card 0 `1792268664`; in `--smoke`): bit for bit the
    model over NaN-primed L1 and `Dst`, at 1, 3, 5 and 17 rows, widths 2048 and 70, with row and column broadcasts.
    A ragged result feeds a sum over rows and a matmul along them exactly.
  - **Watched failing:** a row group dropped, on the model and on the device.
  - **Regression:**
    - ttsim: `step187`, `step191`-`step196`, `step199`, `step12_mnist`, the tt-kernels unit tests;
    - Burn conformance: 2,973 passed, 52 as expected;
    - card 0: `step176`, `step187`, `step192`-`step196`, `step199`, `step200` (`1792268664`); `step188` (`1792269107`).
  - **Measured** (card 1, burn-flex in the same runs):
    - 1.7B 38.93 / 41.38 / 44.58 -> **37.69 / 40.20 / 43.41 ms** (`1792268720`);
    - 135M 18.59 / 19.03 / 19.81 -> **17.51 / 17.97 / 18.76** (`1792269106`, burn-flex 27.3 / 30.1 / 31.2).
- [ ] **Chain fusers** (`burn-tt` `fusion/chain.rs`). Each is bit for bit the composition it replaces, with a negative
  control.
  - RMSNorm's `x / stat * gamma` as one body: −49 ops.
  - RoPE's `x cos + cat(x2, x1) sin` as one op over block views (`cat_in_layout`'s tile permutation): −144 ops.
  - The residual `Add` fused into the next norm's statistic: −48 ops, and the norm no longer reads its row on one
    unit.
- [ ] **What is left per op** (decode-small-ops S5). With ~300 ops left, ~3.5 us of control an op is ~1 ms.
  - Measure it.
  - Then decide on a firmware change, such as one `PAIR_CALL` launch and its credits shared by back-to-back small
    ops.

### Phase 4. The products at the read ceiling, 22.8 -> ~15 ms

Reordered by Phase 0. A layer's products take ~907 us, against ~600 us for their weights' column-dealt stream and
~550 at 488 GB/s. The gap is the product's own structure, not channels.

- [ ] **`A` once a product, multicast** (G7, decode-step-bandwidth B2 point 6).
  - **How:** one unit reads `A`'s `K` tiles and multicasts them to the product's units' `A` slots; a semaphore the
    mover waits on releases each unit's compute. Every other unit reads only `B`.
  - **Worth:** `A`'s entries are ~17 us on `qkv`, `o` and `gate|up`, ~70-80 on `down`: ~120 us a layer, ~2.9 ms a
    step.
  - **Constraints:** rectangles only (ttsim refuses non-rectangular multicast), and the reading unit gets fewer
    columns.
- [ ] **Compute under the reads.** The ~15 us kernel tail of each product (~1.4 ms a step) is the compute after the
  last read: under `Staging::Slots` a job's kernel starts when all its `K` tiles are in.
  - **The fix:** start the `K` chain on the tiles that have landed, by credits per `K` chunk, in the same order, so
    bit for bit.
  - Or, for a unit with several jobs, `SlotsHalf`'s double buffering.
- [ ] **The deal.** A product's units should be chosen from its bytes, not "every unit":
  - 64 units saturate;
  - an uneven last round costs its barrier.

  The rule is measured, not guessed (the 135M head wants more units): `qkv`, `gate|up` and the head on 64 units save
  ~1.25 ms a 1.7B step. It goes in `Session::matmul_dram`'s unit choice (`thin_runs`, `blocks`), checked against the
  probe's sweep.
- [ ] **Consecutive tiles.** A column-dealt stream is ~7% under the consecutive-tile ceiling (~45 us a layer).
  - **Options:** a skewed placement (decode-matmul-bandwidth P5.1), or a deal of `K`-contiguous runs that walk the
    channels.
  - Worth doing only after the items above.
- [ ] **Decision point (ask the user):** split-K, if `o` and `down` remain short of the ceiling. Phase 0 shows 64
  units can saturate, so this is now less likely to be needed.

### Phase 5. The weight stream under the small ops, ~3 -> ~0.5 ms exposed

These are decode-step-bandwidth B2.1-B2.4, now sized to fit.
- [ ] **B2.1, `GATHER_B_RESIDENT`** (B1's twin): a matmul layout with `B`'s slots in a range at the top of `DATA`.
- [ ] **B2.2, a trace-level L1 reservation** that `l1::Requirements` refuses to plan into.
- [ ] **B2.3, a deal with an offset:** the prefetched columns go to the units that hold them.
- [ ] **B2.4, the insertion pass at `store_trace`:**
  - `READ_RUN`s into idle units' streams at each window's start;
  - the hazard check;
  - eager runs unchanged, and each trace bit for bit its eager step.
- [ ] **Each window checked at store time,** from the counted bytes: the next product's bytes in L1 must be at least
  the window's length times 469 GB/s.
- **Gates:**
  - a trace with the prefetch is bit for bit the same trace without it;
  - a prefetch into a planned range is refused at store time.

## Expected landing

Estimates, re-measured after every phase.

| After | 32-token prompt | 256-token prompt | Step GB/s |
|---|--:|--:|--:|
| Now | 41.8 ms | 47.4 ms | ~170 |
| Phases 1-3 | ~27 | ~29 | ~260 |
| Phase 4 | ~20 | ~22 | ~350 |
| Phase 5 | **~15.5-16.5** | **~16-17** | **~430-460 (84-90%)** |

- **The last points to 95%:** the decision points (split-K, flash-decode) and whatever Phase 0 shows about the
  ceiling.
- **135M:** ~20 -> ~5-7 ms.

## Verification

**Bits:**
- `step185`, `step187`, `step188`, `step191`, `step193` stay bit for bit.
- Each item gets its own gate, with a negative control watched failing.

**Order of testing:**
- ttsim first, then card 0: `cargo xtask silicon --release --device 0 --filter <gate>`.
- A record or firmware change: `step172` and `step173` in the safe image, `fast-reads`, and `fast-reads` +
  `encoded-guard`; firmware clippy plain, with `encoded-guard`, and with `profile`.
- An arithmetic or routing change: `step12_mnist` (e2e).
- A burn-tt routing change: `cargo xtask conformance --release`, and with `--fusion` for fusers.

**Speed:**
- `silicon_bench_models::traced_generation` on card 1, every tile, for the 1.7B (`TT_GEN_MODEL=smollm2-1.7b-instruct`)
  and the 135M, with burn-flex in the same run.
- Then `--profile`, for the step's GB/s and to confirm the bytes and ops removed.
- `--smoke` on card 0 at the end of each phase.

## Documentation on completion

- The scoreboard and the performance log, with run IDs.
- Each companion plan's checkboxes, and roadmap Track 1.
- The operating notes, for Phase 0's ceiling; ttsim-divergence.md, for any refusal.
- This file and finished companions moved to `completed-plans/`.
