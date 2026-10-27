# Decode at GDDR's bandwidth

Roadmap Track 1, SmolLM2 single-user decode. Written 2026-10-17 against `c69b9c1` (plus the uncommitted B0/B1 work of
[decode-step-bandwidth.md](decode-step-bandwidth.md)). Status (2026-10-19): **a 1.7B step 41.83 / 44.25 / 47.40 -> 24.04 / 24.48 / 25.39 ms (`1792383988`), 135M
20.03 / 20.55 / 21.19 -> 11.40 / 11.80 / 12.26 (`1792378993`).** The four larger changes are built (below), and the
small ops' phase has begun (Phase 3, "The small ops, 2026-10-19"). Then decode attention as one op (Phase 6): 1.7B
**22.78 / 23.26 / 23.98 ms** (`1792425331`), 135M **9.58 / 10.11 / 10.96** (`1792425715`).
- **Phase 0, done:** the read ceiling is ~488 GB/s, and the products' gap is not the read stream.
- **Phase 1:** done. The copies are gone. The repack's reads run in flight together, and the row scatter and the
  embedding's select run on every unit. The k/v repack is gone with the position-major cache (Phase 2's A-layout).
- **Phase 3:** the capture barriers, the row-limited element-wise ops and RoPE's sum fuser are done.
- **Phase 2:** A0 found the transposed reads of the key cache; `B` is now transposed by the unpacker (A0b), and a
  stacked product's tail rows are no longer read (A2a), nor its padding off `K` filled (A1); the softmax is four
  ops (A4a), and the scale and the mask are one (A4b). The cache is position-major (A-layout), so a step writes its
  projection's row as it is.
- **Phase 4:** a thin product over a long `K` runs on 64 units (the deal).
- **Phase 4:** `A` once a product (a packed `A`), compute under the reads (the m = 1 kernel's rows, a blocked `K`
  streamed).
- **Phase 5:** the weight stream under the small ops, on idle units.
- **Bandwidth now** (2026-10-19, [log](../learnings/performance-log.md#where-decodes-bandwidth-stands-and-a-column-broadcast-in-the-sfpu-2026-10-19-cards-0-and-1)):
  the weights stream at **284.8 GB/s over a timed step, 56% of 512** (`1792383988`; 258.6 before the small ops'
  phase); every byte of a `Step(64)` replay at 327 GB/s, 64% (profile `1792382914`).
- **Where the step goes now** (profile `1792382914`, 1.7B `Step(64)`, 23.22 ms; products partly overlap the small
  ops now):
  - the products 15.58 ms, 463 GB/s over their own time: at the read ceiling. Their `B` alone would take 14.88 at
    469 GB/s; only `down` (172 us a layer against ~143, not prefetched) is clearly over;
  - everything else 7.64 ms in 690 ops, in four windows between a layer's products, ~1.9 ms of it on 1-2 units: the
    softmax's SFPU ops and reductions, attention's scale, RMSNorm's statistic.
- **2026-10-19, the review** ([below](#the-2026-10-19-review-what-caps-decode-in-f32)): the deeper prefetch alone is
  worth ≤ ~2 ms (L1 and a unit's read rate cap it); the order is now Phase 6 (attention as one op), Phase 7 (the other
  small ops), Phase 8 (a continuous weight stream). **F32's floor is ~14.3 ms** (6.957 GB of weight slots at 488 GB/s;
  13.6 at 512): ~13 ms needs BF16 storage, a later item.
- **Phase 6, done:** decode attention as one op: 1.7B **22.78 / 23.26 / 23.98 ms** (`1792425331`), 135M **9.58 /
  10.11 / 10.96** (`1792425715`).
- **Then (Phase 7):** RMSNorm as one op, the token on the card, RoPE as one op, three copies gone: 1.7B **20.95 / 21.41 /
  22.18 ms** (`1792444936`), 135M **8.00 / 8.59 / 9.38** (`1792445319`).
- **Phase 8's probe (2026-10-20):** weights free, a step is 7.85 ms; under a ~300 GB/s stream its ops cost 1.6x; go,
  but L1 binds (~2-3 ms realistic). Next: the packed-`A` ring reclaim, then busy units streaming (below).

This is the umbrella plan.
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
  for a decision with measured numbers and are not built unasked
  ([split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md), 2026-10-23). The target probably needs one of them.
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
- [x] **The repack's reads in flight together** (2026-10-18). Profile `1792293993` (1.7B, `Step(64)`): the cache
  write's 96 ops took 3.06 ms. Per layer that is a `native repack` (38 us) and a `resident row scatter` (25 us) for
  each of k and v, every one of them on 2 units.
  - **Why the repack was slow:** `tensor::repack_many` read each source tile into one slot and copied it before it
    read the next, so the 32 reads of an output tile ran one after another (~1.1 us each). The data does not explain
    it: the reads were whole tiles for one row each.
  - **The change:** a batch reads up to `REPACK_SLOTS` (32) source tiles into slots of their own, all before any copy
    (B waits for the preceding reads before a `COPY_WORDS`), then copies the same words in the same order. A tile's
    copies that do not fit a batch read it again, and a later batch reads the output tile back first, as before.
    Every `repack_many` gains (convolutions' relayouts, cats, views), not only decode.
  - **Then only the span read:** each read is the 64-byte span of the slot that holds the words copied from it
    (a row's two face rows, 1,152 bytes of 4,160).
  - **Gate `step202_repack_reads`** (ttsim; in `--smoke`), bit for bit against the host's mapping over NaN-primed
    slots: the decode rows (32 and 9 heads), a transpose (a tile's copies over several batches), an output tile from
    48 source tiles (two batches), and two sources. Watched failing with every copy reading slot 0, and with the span
    64 bytes short.
  - **What it bought: little.** Profile `1792296384` has the repack at 31-33 us, against 38. The span read then took
    the 1.7B step only 33.05 / 33.59 / 34.51 -> 33.02 / 33.52 / 34.42 ms (`1792298195`; 135M unchanged,
    `1792298556`). The repack is bound by neither its reads' latency nor their bytes: 31.0 us on 74 KB (profile
    `1792298755`), ~96 entries a unit at ~0.3 us each. The position-major cache below removes it.
- [x] **The row scatter on more units** (2026-10-18). Each NC write entry costs ~0.37 us to issue, and a unit wrote
  32 rows x 2 face rows.
  - **The record:** `record::SCATTER_ROWS` takes a range of its indices (`scatter_rows_range`). It writes only the rows
    in it, but drops a repeated row unless its index is the last of all `count`. Records splitting one batch therefore
    write each address once, from the same index one record would.
  - **The deal:** `scatter_rows_jobs` splits a batch of 32 or fewer indices (a decode step's row a head) over the units
    left after the column groups, at least `SCATTER_SPLIT_ROWS` (4) a job. Every job reads the whole index tile. More
    batches are not split, since nothing would order one unit's batch after another's.
  - **Gates:**
    - record unit `scatter_rows_ranges_split_one_records_writes`: repeats across ranges, each address once, bad ranges
      refused. Watched failing with each range deduplicated only within itself;
    - `step189::one_batch_split_between_units_writes_each_row_once_from_its_last_index` (16 units). Watched failing
      the same way on ttsim;
    - firmware clippy plain, `encoded-guard` and `profile`; card 0 `1792294692` (54 gates, with `step172`/`step173`).
- [x] **The embedding's row select on every unit** (2026-10-18). Each step began with a 253 us `resident row select`
  on 3 units: a ragged batch fills its staged tiles' other 31 rows one word at a time (~10 us a tile), and the 64 tiles
  went in groups of 24. `select_taps_jobs` now narrows its groups until every unit has one (64 groups of one tile).
  The bits and the `Pad::Zero` claim are unchanged.
  - **Gate:** `step171::a_select_over_many_units_is_the_host_s_indexing_with_zero_padding` (16 units, padding rows
    downloaded). Watched failing with only the first group filled.
- **Regression** for the three: ttsim `step179`, `step180`, `step187`, `step192`, `step193`, `step199`,
  `step12_mnist`; Burn conformance 2,973 passed, 52 as expected; card 0 `1792294692`, `1792295061`, `1792297325`;
  `--smoke` 1,123 of 1,123 (`1792299469`).
- **Measured** (card 1):
  - 1.7B 34.35 / 34.87 / 35.76 -> **33.05 / 33.59 / 34.51 ms** (`1792295836`), with the span read **33.02 / 33.52 /
    34.42** (`1792298195`);
  - 135M 14.28 / 14.68 / 15.25 -> **14.04 / 14.45 / 15.04** (`1792296202`, burn-flex 26.7 / 29.0 / 31.6), with the
    span read 14.04 / 14.45 / 15.02 (`1792298556`).
  - **Profile `1792296384`** (`Step(64)`, before the span read): the replay 33.68 -> 32.34 ms. The cache write's 96
    ops 3.06 -> 1.94 ms (the row scatter 25 -> 8.8 us on 16 units). The embedding's select 253 -> 67 us on 64 units,
    52 of it the barrier; the RoPE tables' selects 24 -> 14 us.
- [x] **k and v without the repack** (in-place-cache-write D5, decode-small-ops S1d; 2026-10-18). It was the cache's
  head-major layout. The cache is now position-major (decode-attention A-layout), and the write is the projection's
  row as it is.

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
- [x] **RoPE's sum as one op** (2026-10-18).
  - **The kernel:** `kind_sfpu::MUL_ADD`, `c + a b`: `MUL` then `ADD`'s instruction with `c` as its left operand,
    two roundings, so bit for bit the two ops.
  - **The fuser:** a `MulAdd` chain (`fusion/chain.rs`, with `Arg::Step(k)`, an earlier step's result) matches
    burn-fusion's `Mul(x, cos)`, `Mul(rotated, sin)`, `Add(first, second)`. The `cat` ends the block before it.
    The first product runs as itself; the second and the sum run as `MUL_ADD`.
  - **The table:** relaid into the operand's layout (`relaid`, the copy `float_mul` keeps). Where it cannot be
    (halves of 16 columns), the sum runs as itself.
  - **Gates:**
    - unit `mul_add_is_mul_then_add_bit_for_bit` (watched failing with one fused `SFPMAD`);
    - the row-limit unit gate, now with ternary kinds;
    - fuser unit `a_sum_of_two_products_is_mul_add` (a product read again, or the sum's operands reversed, is not
      fused);
    - `step193::tt_and_tt_backend_compared`, now with a `head_dim` 64 config: every RoPE sum fuses there, one
      `add` fewer each, and the generations stay `TtBackend`'s bit for bit.
  - **Regression:**
    - ttsim: `step187`, `step191`, `step192`, `step194`, `step195`, `step12_mnist`;
    - Burn conformance: plain 2,973 passed, 52 as expected; `--fusion` 2,970 passed, 55 as expected, 0 unexpected;
    - card 0: 34 gates (`1792290020`); `--smoke` 1,118 of 1,118 (`1792290851`).
  - **Measured** (card 1):
    - 1.7B 36.12 / 36.52 / 37.54 -> **35.64 / 36.22 / 37.20 ms** (`1792290295`);
    - 135M 14.80 / 15.29 / 15.86 -> **14.43 / 14.92 / 15.46** (`1792290681`, burn-flex 26.9 / 29.1 / 31.1).
- **The small ops, 2026-10-19.** From profile `1792363365`: ~424 us a layer of small ops in four windows between
  products, which the prefetch (~43 us of stream a product) cannot hide. In order of worth and risk, each bit for bit
  and re-profiled before the next:
  - [x] **A column broadcast in the SFPU** (~1.0 ms, measured 1.0). A `[rows, 1]` operand was broadcast by the mover a datum at a
    time (`READ_BROADCAST_COL`, ~10 us a tile). The mover now reads its tile as it lies
    (`record::READ_RUN_COL_AS_IT_LIES`), and the program copies column 0 across in `Dst` first
    (`sfpu::Program::broadcast_col0`), bit for bit the mover's tile.
    - **Gates:** unit `col_broadcast_tests` (watched failing with six rotations, and with a row group skipped);
      `step206_sfpu_col_broadcast` (ttsim, the column's padding NaN in GDDR; watched failing with the broadcast
      dropped; card 0 `1792371974`, and `1792372371` in the guarded fast-read image); record unit test.
    - **Regression:** ttsim `step30`, `step43`, `step121`, `step183`, `step184`, `step187`, `step193`, `step195`,
      `step196`, `step199`, `step200`, `step12_mnist`; `step172`/`step173` in three images; firmware clippy x4;
      Burn conformance 2,973 passed, 52 as expected; card 0 45 of 45 (`1792371974`).
    - **Measured:** RMSNorm's `DIV` 24.1 -> 14.7 us, the softmax's 2-unit ops 24.7 -> 15.0 us, the replay 25.61 ->
      24.71 ms (profile `1792372923`). 1.7B 26.47 / 27.14 / 27.97 -> **25.48 / 26.18 / 27.01 ms** (`1792372396`);
      135M 12.62 / 13.06 / 13.59 -> **11.37 / 11.76 / 12.29** (`1792372749`).
  - [x] **RMSNorm's statistic from the packed row** (estimated ~0.9 ms; **measured ~0.15**). `pack_row`'s
    `[K / 32, 32]` row `kk` is tile `kk`'s data row, so a sum down its rows takes each lane's squares in the same
    order (`sfpu::reduce::accumulate_squares_in_order`, `sfpu::norm::math_packed`, `tensor::sfpu_rms_packed`;
    `Session::set_packed_rms`, on, for one row of at least `PACKED_A_K_TILES` tiles). It also lifts the one-job limit
    (4,096 columns) for such a row.
    - **Gates:** unit `a_packed_row_s_statistic_is_the_row_s` (576-8192 columns, specials, two NaN payloads; control:
      the packed rows summed bottom to top differ, on a row whose sum the order changes); `step196`
      `a_decode_row_s_statistic_reads_it_packed` (ttsim; card 0 `1792374366`): packed, unpacked and the model bit for
      bit at 1024-8192 columns, the packing seen to run and not to run; watched failing with the statistic reading one
      packed tile too few. Not observable on the model: the first square taken as it is and the add's operand order
      (kept as `accumulate_squares` has them).
    - **Regression:** ttsim `step187`, `step193`, `step195`, `step197`, `step199`, the tt-kernels unit tests; Burn
      conformance `--fusion` 2,970 passed, 55 as expected; card 0 30 of 30 (`1792374366`, `step188` among them).
    - **Measured** (profile `1792375116`): the statistic 30.1 -> 10.9 us, but the packing is an op of its own on 64
      units, 12.2 us (barrier 4.7), and the `DIV` after it 14.7 -> 18.0. A norm 44.7 -> 41.1 us; the replay 24.71 ->
      24.58 ms. 1.7B **25.38 / 26.03 / 26.91 ms** (`1792374654`).
    - **What would make it pay:** no packing op. The residual `Add` before each norm runs on the row's 64 units and
      could write its row packed too (two face-row writes a tile in its own job), as could `x gamma` for the
      product's packed `A` (97 packing ops a step, ~6-7 us each). Both are a packed twin carried with a one-row
      result.
  - [x] **Row-dealt gathers** in attention (estimated ~0.8 ms, measured ~0.7). `gather_rows` dealt by output tile,
    so a stacked `[32, 64]` output's 64 face-row reads landed on 2 units (22.3 us). With fewer output tiles than
    units it now deals rows (`tensor::GATHER_DEAL_ROWS`, at least 4 a unit): each row one 1,088-byte read of its
    two face rows into a slot of its own and two 64-byte writes into place, only the rows gathered written. `P`'s
    gather (32 rows and their holes over 64 tiles) is dealt the same way.
    - **Gate:** `step62::rows_dealt_over_units_are_the_host_s_selection` (16 units, the output's place primed with
      NaN; stacked, ragged, two sources, a ragged width, and a gather of more tiles than units; watched failing with
      the read landing at the source row's place).
    - **Regression:** ttsim `step61`, `step180`, `step181`, `step183`, `step184`, `step187`, `step190`, `step192`,
      `step193`, `step199`; Burn conformance 2,973 passed, 52 as expected; card 0 44 of 44 (`1792376471`, `step166`
      and `step188` among them).
    - **Measured:** the three gathers a layer 22.3 / 5.3 / 22.3 -> 8.0 us each on 16 units; the replay 24.58 -> 23.88
      ms (profile `1792376868`). 1.7B 25.38 / 26.03 / 26.91 -> **24.72 / 25.38 / 26.17 ms** (`1792377589`, 276.9 GB/s
      of weights at 32 tokens, 54%).
  - [x] **The prefetch keeps its units** (estimated ~0.4 ms at `Step(512)`; measured 0.3 at every length). At 512
    positions attention's `qk` and a `pv` gather spread to 120 units and took `o`'s prefetch units (`o` 35.6 -> 52.6 us
    a layer). In a capture, once a one-row product has streamed, every op but a product runs on at most the units
    that product ran on (`Session::spread_for`, the capture's `weight_units`; `Session::set_prefetch_keeps_units`, on).
    Placement only.
    - **Gate:** `step205::ops_after_a_streaming_product_keep_off_the_prefetch_s_units` (ttsim, 8 units: ops asked to
      spread over all eight stay on the product's four, and the next products prefetch on 4-7, bit for bit; watched
      failing with the cap off). Claim three now turns the cap off to show busy units are not taken.
    - **Regression:** ttsim `step176`, `step186`, `step187`, `step193`, `step197`, `step199`, `step204`; card 0 28 of
      28 (`1792378401`, `step188` among them); `--smoke` on card 0 after the four, 1,134 of 1,134 (`1792378998`).
    - **Measured** (profile `1792379163`): at `Step(512)` `o` ~37 us a layer again and `qk` on 64 units as fast as on
      120 (27.5 us); the products' time over their floor 812 -> 392 us; `silu_mul` on 64 units, 15.5 -> 18.3 us. The
      `Step(64)` replay 23.88 -> 23.62 ms. 1.7B 24.72 / 25.38 / 26.17 -> **24.41 / 25.01 / 25.93 ms** (`1792378639`,
      280.5 GB/s of weights at 32 tokens, 55%); 135M unchanged, 11.40 / 11.80 / 12.26 (`1792378993`, burn-flex 27.0 /
      29.2 / 31.7: none of its products prefetch).
  - **Where the four leave the step** (profile `1792379163`, `Step(64)`, 23.62 ms): products 15.60 ms at 462 GB/s;
    everything else 8.02 ms (10.01 at the start); every byte at 322 GB/s, 63% of 512. 1.7B 26.47 / 27.14 / 27.97 ->
    24.41 / 25.01 / 25.93 ms (-8%).
  - [x] **Decided (2026-10-19):** build the packed twin and the barrier; document RoPE over `q|k` in one pass
    (~0.6 ms, a model-code change) and `x / stat * gamma` as one op (~0.4 ms, burn-fusion's `Reshape` block) and skip
    them for now.
  - [x] **A one-row result written packed by the op that makes it** (`Session::set_packed_twins`, on;
    `tensor::packed_twin_writes`). A `[1, K]` F32 element-wise result over at least `PACKED_A_K_TILES` tiles also
    writes each output tile's data row into a `[K / 32, 32]` twin, from the slots its own scatter writes out (two
    64-byte writes a tile). A product's packed `A` and RMSNorm's statistic read the twin: no packing op. The twin is
    the session's (`Session::packed_twin`), freed with its result and dropped by any write in place (`write`,
    `write_bits`, `copy_into`, `write_rows`, `scatter_rows_into`).
    - **Gate:** `step207_packed_twin` (ttsim and card 0, 8 units): the twin is `pack_row` of the result bit for bit
      (`K` 1024, 1056, 2048; plain and block operands); its readers equal the packing paths' bits and pack nothing; after
      each kind of write in place the readers are the new row's and the twin is gone; GDDR returns to where it was;
      a capture and its replays equal the ops run fresh. Watched failing with the twin's rows a face row off, and with
      `drop_twin` doing nothing (the statistic after a write read the old row).
    - **Regression:** ttsim `step187`, `step190`, `step192`-`step200`, `step204`, `step205`, the tt-kernels unit tests,
      `step12_mnist`; Burn conformance plain 2,973 passed, 52 as expected, `--fusion` 2,970 passed, 55 as expected;
      card 0 34 of 34 (`1792382182`, `step188` among them).
    - **Measured:** 121 fewer ops a step (97 product packings and 49 statistic packings, less the `o` input's, which
      a repack makes); each element-wise op ~1 us more (the two writes), RMSNorm's `DIV` 18 -> 25.6 us (the 63 units
      that sat out the statistic now wait on its barrier there, where the packing op absorbed it). 1.7B 24.41 / 25.01 /
      25.93 -> **23.95 / 24.55 / 25.52 ms** (`1792382455`, 285.8 GB/s of weights).
  - [x] **A barrier released by unit 0** (tt-metal's coordinator-broadcast barrier; `dm::op::BARRIER` word 6,
    `Session::set_barrier_release`, on; `TT_BARRIER_RELEASE=0` turns it off). Every unit polled unit 0's counter with
    NoC0 reads, which queue behind the GDDR reads; tt-metal never polls remotely (`dataflow_api.h`:
    `noc_semaphore_wait` spins on local L1; `barrier_sync.hpp`, "avoids the NOC congestion of polling-read barriers").
    The per-tile RISC-V mailboxes are tile-local and cannot be reached over the NoC, so they are not a barrier.
    - **Now:** arrival is the same NoC atomic increment. Unit 0 waits on the counter in its own L1, then writes the
      target into every other unit's `dm::BARRIER_RELEASE` with one non-posted multicast per rectangle
      (`dm::BARRIER_RECTS`, at most 8: the session's units but unit 0, a row's runs merged down rows, each checked by
      `Rect::new`, so no other tile is written; path-reserved, VC 4, as `noc::multicast::registers_to`). Every other
      unit waits on its own word, with no NoC traffic. Unit 0 arrives at every barrier (a list of its own where the op
      did not run on it), so every target is released; it releases only what it arrives at, so the words only grow. The
      next release waits for the last one's acknowledgements (`NIU_MST_WR_ACK_RECEIVED`), so one is in flight at most;
      the transaction ID's counter is never waited on or cleared (ttsim refuses the clear).
    - **Gates:** `step208_barrier_release` (ttsim, card 0 `1792383833`, 8 units): the slow-op-then-readers case
      exact every round, eager and replayed; every unit's word equals the arrivals after each sync; with release off
      the words stay put; turned on again, every word starts at the count. Negative control (ttsim only): with unit 0
      releasing nothing and every word far ahead, the readers overtake the slow op. On card 0 they did not even at `K`
      8192 and 32768 (the host queues the readers more slowly than unit 0 finishes). Unit `release_rect_tests` (every
      unit but the leader once and nothing else, full and harvested grids, skipped tiles; a checkerboard refused). Record
      unit test for word 6.
    - **Regression:** ttsim `step34`, `step39`, `step62`, `step171`, `step176`, `step183`, `step184`, `step189`,
      `step205`, `step207`; `step172`/`step173`/`step208` in the safe, `fast-reads` and `fast-reads` + `encoded-guard`
      images; firmware clippy x4; card 0: `step172`/`step173`/`step208` in the guarded fast-read image (`1792385599`),
      `--smoke` 1,140 of 1,140 with twins and release on (`1792383989`).
    - **Measured (card 1, release against polling, same build):** 1.7B **24.04 / 24.48 / 25.39** against 24.09 / 24.65 /
      25.64 ms (`1792383988`, `1792384336`): 0.05-0.25 ms. The barrier's ~3.8 ms a step is mostly waiting for the
      slowest unit to arrive, not the release.
  - [x] **Weights stored transposed: tried, slower.** A decode product's column of `B` is `K` tiles a grid row apart;
    stored `[n, k]` (read through the unpacker's transpose, bit for bit) it is consecutive tiles. Probe
    `silicon_probe_decode_stream::decode_products_with_their_weights_stored_transposed` (card 1): every product 2-6%
    slower (`gate|up` 342 -> 357 us, `down` 175 -> 180, the head 993 -> 1022). Not built.
  - **Where the step stands** (1.7B, card 1, `1792383988`): 24.04 / 24.48 / 25.39 ms, 284.8 GB/s of weights at 32
    tokens (56% of 512); profile `1792382914` (twins and release): `Step(64)` 23.22 ms, products 15.58 ms at 463 GB/s,
    everything else 7.64 ms in 690 ops. Alone, the products read at 346-412 GB/s (the probe above): in the step they
    reach ~462 because the prefetch reads part of their weights during the small ops before them. More of the weight
    stream under the small ops is the lever left for the step's bandwidth: a prefetch deeper than one job a unit and
    one product ahead.
- [ ] **The other chain fusers** (`burn-tt` `fusion/chain.rs`):
  - RMSNorm's `x / stat * gamma` as one body. Today `gamma`'s `Reshape` puts the product in a block of its own, so
    the fuser cannot see it (decode-small-ops S3, R0).
  - The residual `Add` fused into the next norm's statistic.
- [x] **What is left per op** (decode-small-ops S5, 2026-10-26). Measured; the unneeded barriers are pruned from a
  capture; one launch for back-to-back ops is not built (Phase 7).

### Phase 4. The products at the read ceiling, 22.8 -> ~15 ms

Reordered by Phase 0. A layer's products take ~907 us, against ~600 us for their weights' column-dealt stream and
~550 at 488 GB/s. The gap is the product's own structure, not channels.

- [x] **The bound on `A`'s reads** (2026-10-18, `silicon_probe_decode_stream` with `tensor::probe_skip_a_reads`,
  card 0 `1792283215`). Every block reads only `B`, so the products are wrong and only their time counts:

  | Product (1.7B), today's deal | With `A` | Without | Saved |
  |---|--:|--:|--:|
  | `qkv`, 120 units | 181 us | 157 | 24 |
  | `o`, 64 | 73 | 56 | 17 |
  | `gate\|up`, 120 | 383 | 353 | 31 |
  | `down`, 64, `K` = 8192 | 270 | 198 | **72** |
  | head, 120 | 1,051 | 1,034 | 17 |

  - **The total:** ~143 us a layer, **~3.4 ms a 1.7B step at most**.
  - **What is left without `A`:** each product is still above its weights' own stream (`o` 56 against 38 us,
    `down` 198 against 150). That is the kernel tail and the barrier, the next items.
- [x] **`A` once a product: a packed `A`** (2026-10-18; G7, decode-step-bandwidth B2 point 6). Neither designed
  form was built: multicast risks the NoC, and a vector layout needs every producer to write it. A third, built on
  tiles, does the job.
  - **`tensor::pack_row`:** lays a `[1, K]` row out as `[K / 32, 32]`, so row `kk` is `K` tile `kk`'s data row. It is
    one op on the product's units: a read and two face-row writes a tile, ~5-7 us.
  - **Each unit reads `A` as `K / 1024` whole tiles** (a `READ_RUN`, 2 at `K` = 2048, 8 at 8192), not `K / 32` thin
    reads at ~0.27 us an entry. Its gather reads only `B` (`GATHER_A_RESIDENT`).
  - **The kernel:** it unpacks `K` tile `kk`'s faces from row `kk % 16` of the packed faces `((kk % 32) / 16, ·)`
    (`matmul::packed_a_pairs`), whose `(·, 0)` and `(·, 1)` are 1,024 bytes apart as a tile's are. `SrcB`'s row 0 holds
    the same values, so it is bit for bit, and the other rows only reach output rows past the product's one. The
    pairs are stepped in runs of 16 (`even_runs`).
  - **K blocks:** a packed `A` takes `ceil(kc / 32) + 1` slots, so `down` fits two K blocks of 128, not four of 64
    (`packed_k_block`).
  - **When:** `Session::set_packed_thin_a` (on): `m = 1`, `A` untransposed, `K` whole tiles and at least
    `PACKED_A_K_TILES` (32), TF32.
  - **Gates:**
    - `step204_packed_thin_a` (ttsim; card 0 `1792342641`, every unit): the packing is the host's reshape at `K` =
      1024, 1056 and 8192, and the products are bit for bit the thin ones, whole, K-blocked and streamed, with `B`
      transposed and NaN-primed L1. Watched failing with the face-row offset dropped;
    - unit `a_packed_a_is_read_in_k_over_1024_tiles`.
  - **Measured** (profile `1792342758`): `qkv` 148 -> 135, `o` 64 -> 53, `gate|up` 349 -> 337, `down` 246 -> 190 us,
    each plus its pack. The step went 29.55 -> 27.90 ms; the larger K blocks took it to 27.77 (`1792344448`).
- [x] **Compute under the reads** (2026-10-18). Reading the code showed the ~15 us "tail" was the last block's
  compute, ~0.23 us a pair, and every earlier block's was serialized with the unit's next reads. Two changes:
  - **The m = 1 kernel computes only its rows** (`matmul::LiveRows`, on every product path and BF16/BFP, by the
    product's own `m`):
    - an output tile whose last tile row holds 1-8 rows skips `A`'s faces `(1, k)` and `SrcB`'s second half: 16
      `MVMUL`s a pair, not 64;
    - with 9-16 rows, it skips the faces `(1, k)` only;
    - the rows left out stay `ZEROACC`'s zeros, which every padding claim allows, and the computed rows take the same
      products in the same order.
    - **Gates:** `step203_live_rows` (ttsim; card 0 `1792341332`): bit for bit against the trim off, `m` 1-33, K whole,
      ragged and blocked, HiFi4 and LoFi, BF16. Watched failing with the last `MVMUL` on the wrong row. `step121`
      checks the trimmed programs' waits.
    - **Measured** (`1792341444`): the tail went 15.0 -> 4.8 us, and the step 30.91 -> 29.55 ms.
  - **A blocked `K` streamed in chunks** (`BlockPlan::push_stream_jobs`, `matmul::StreamLayout`; `Session::set_streamed_k`,
    on):
    - a one-row product with a packed `A` whose `K` would be K-blocked runs each column's `K` as an even number of
      chunk batches, in alternating halves;
    - each chunk reloads the partial the one before packed into its half's output slot (the K block's FP32 reload,
      from L1, not GDDR), so the mover reads chunk `c + 1` while chunk `c` computes, and only the last chunk writes;
    - **why only where `K` is blocked:** each chunk batch costs ~3 us of role work (the reload, the pack, the
      lockstep retire). Streaming every product's `K` lost on units of several columns (`1792358552`: `gate|up` 337 ->
      346 us).
    - **Gates:** `step204` (streamed and not), unit `a_streamed_column_reads_its_k_in_chunks`; watched failing with
      the reload dropped.
    - **Measured** (`1792359572`): `down` 184 -> 173 us, the step 27.77 -> 27.51 ms.
- [x] **The deal** (2026-10-18). A product's units are chosen from its bytes, not "every unit":
  - 64 units saturate;
  - an uneven last round costs its barrier.
  - **The rule:** a one-tile-row product with `K` of at least 32 tiles runs on at most 64 units
    (`session::THIN_STREAM_UNITS`, `THIN_STREAM_K_TILES`; `Session::set_thin_stream_units`, which the probe's sweep
    turns off). The rule comes from the probe's sweep. A short `K` keeps every unit: the 135M head (`K` = 18) was
    faster on 120.
  - **Bits:** unchanged. Each output column keeps its `K` chain; only which unit holds it moves.
  - **Measured** (card 1), with the attention's scale and mask as one op (decode-attention A4b):
    - 1.7B 35.64 / 36.22 / 37.20 -> **34.35 / 34.87 / 35.76 ms** (`1792292588`); without the deal 35.56 / 36.10 /
      37.00 (`1792293194`), so the deal is ~1.2 ms;
    - 135M 14.43 / 14.92 / 15.46 -> **14.28 / 14.68 / 15.25** (`1792293687`, burn-flex 26.6 / 29.0 / 31.3).
- [ ] **Consecutive tiles.** A column-dealt stream is ~7% under the consecutive-tile ceiling (~45 us a layer).
  - **Options:** a skewed placement (decode-matmul-bandwidth P5.1), or a deal of `K`-contiguous runs that walk the
    channels.
  - **Tried, 2026-10-18:** `gate|up`'s rounds dealt rotated, so each unit walked the channel pairs. No gain:
    27.90 -> 27.84 ms (`1792343591`); the barrier fell, the data phase rose. Reverted. The barrier skew is not
    channel locality.
- [ ] **Decision point (ask the user):** split-K, if `o` and `down` remain short of the ceiling. Phase 0 shows 64
  units can saturate, so this is now less likely to be needed. Written up 2026-10-23
  ([split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md)): ~0.2 ms by the handoff's measurement,
  parked.
- [ ] **A product's block from what it runs, not a greedy probe** (found by planned-l1-residency 2h-i, 2026-10-21,
  `1792546573`). `matmul::plan_in` takes the widest block whose probe fits, and the probe charges `A` unpacked (`kt`
  slots) where the kernel runs a packed `A` (`kt/32 + 1`).
  - **What that costs:** at `K` = 2048 the probe holds 807 KB, which is what caps a residency region at 110 KB. At `K` =
    4096 (an 8B's `hidden`) no `K`-whole block fits (257 slots of 220), so those products stream with no weight
    prefetch.
  - **And the widest is not always fastest:** Qwen2.5-1.5B's `gate|up` (`K` = 1536) is 319 us at `nc` 3 and 293 at 2.
  - **To do:** charge the packed `A` in the fit, choose the width by measurement (a bound, not greed, so the freed room
    is not refilled), and measure the products and the step. A separate step from residency: it changes the products'
    layouts with no region at all.

### Phase 5. The weight stream under the small ops (done 2026-10-18)

decode-step-bandwidth B2.1-B2.4 were not built. A simpler design does the job with no record change, no L1
reservation and no store-time pass.
- [x] **Prefetch on idle units** (`Session::set_weight_prefetch`, on; `tensor::BlockPlan::push_prefetched_jobs`).
  - **Which units:** in a capture, a one-row product with a packed `A` and its `K` whole puts its first columns on up
    to `PREFETCH_UNITS` units past its own, ones whose stream has not grown since the last such product (the
    capture's `weight_anchor`). Each takes one job of the plan's width.
  - **What each such unit's stream gets:**
    1. a wait for the last product's barrier;
    2. its job's `B` reads, the job's own `GATHER` with `A` resident (`tensor::PREFETCH`);
    3. the usual catch-up wait;
    4. the job, whose gather reads only the packed `A`.
    So the weights stream while the ops between the two products run.
  - **Why it is safe on any unit:** the list is appended when the product is captured, after everything the unit
    did before, with only the product's own wait after it. Taking only idle units is a policy: a busy unit would
    start late. Eager runs never prefetch.
  - **Gates:**
    - `step205_weight_prefetch` (ttsim with 8 units, ops on 4; card 0 `1792361154`): the capture's run and its
      replays are the ops run fresh, bit for bit, and the second and third products run on units 4-7, but not when
      those units ran the ops between. Watched failing with the rest of the columns not shifted past the
      prefetched ones;
    - card 0: `step188` (the 1.7B traced against eager), 58 of 58 gates.
  - **Measured:**
    - at 56 units (`1792360408`): `o` 53 -> 31, `gate|up` 337 -> 280, the next layer's `qkv` 134 -> 81 us, and the
      step 27.51 -> 25.96 ms;
    - **what it costs the small ops:** the latency-bound ones slow under the prefetch's GDDR traffic (the residual
      adds, RMSNorm's one-unit statistic, `gamma`'s product: ~+60 us a layer);
    - **the cap, by experiment** (`PREFETCH_UNITS`): 20 units 26.74 ms (`1792362531`), 32 25.60 (`1792361675`),
      **40 25.61 / 26.03 / 26.40 / 27.23 at `Step(64)` to `Step(512)` (`1792363365`, the setting)**, 56 25.96
      (`1792360408`).
- [ ] **Left:**
  - ~~prefetch for a K-blocked product (`down`'s column is 1 MB)~~ (2026-10-22: a streamed column's first two chunks,
    Phase 8);
  - the small ops made latency-tolerant of the stream (the one-unit statistic spread over units, decode-small-ops).

## The 2026-10-19 review: what caps decode in F32

Asked: the deeper prefetch, or a review for assumptions that limit us, toward ~13 ms. Three reviews (scheduler,
formats and bandwidth, the op graph; profile `1792382914`, `Step(64)` 23.22 ms: products 15.58, 690 other ops 7.64,
the host ~0.8).

- **The floor.** F32 weight slots are 6.957 GB: **14.3 ms at the 488 GB/s read ceiling**, 14.7 with a 512-position
  cache, 13.6 at the 512 GB/s peak. ~13 ms is not reachable in F32 on one card. The checkpoint is BF16 widened at load
  (`llama.rs` `Precision::F32`), and `SrcA` multiplies at TF32 (9 mantissa bits used), so BF16 storage would be
  bit for bit at half the bytes (~7.2 ms floor): decided (2026-10-19) to come after the F32 levers, then lower formats
  and mixed precision.
- **The deeper prefetch alone: ≤ ~2 ms, likely ~1.** The arena is 896 KiB, so a unit holds one 2-column job at
  `K` = 2048; ~56 idle units hold ≤ ~50 MB against a ~88 MB window; a unit reads at ~7 GB/s; contention slows the
  latency-bound small ops (the cap of 40). Its longest window was attention's.
- **Limiting assumptions, ranked:**

  | # | Assumption | Cost | Phase |
  |---|---|--:|---|
  | 1 | Attention, RoPE and the cache write are 21 ops a layer, 5 of them on 1-2 units (32 heads in one tile row) | ~4.5 ms | 6 |
  | 2 | Weights stream only on idle units, one product ahead: B drains every read before a `LAUNCH` or `BARRIER`, ops are total-ordered behind one cumulative barrier, nothing in L1 survives an op | products ~150 us a layer over their floor (~3.6 ms); isolated they read at 346-412 GB/s | 8 |
  | 3 | ~3.5 us of control an op (`PAIR_CALL`, credits, `LAUNCH`) over ~690 ops, and barrier skew | ~2.4 ms + ~3.8 of waiting | 7 |
  | 4 | RMSNorm is three ops, the residual add a fourth | ~2.3 ms | 7 |
  | 5 | The `[1, 49152]` logits downloaded a step; argmax on the host | ~0.47 ms | 7 |
  | 6 | `down` never prefetched (a 1 MB column) | ~0.7 ms | 8 |
  | — | Not limiting: the in-flight cap (8), `fast-reads` (no effect on `READ_RUN`), NoC1 reads (SYS-1419), slot headers (1.6%), the head (~0.1 ms over) | — | — |

- **Why shrinking is not enough:** once the small ops are small, the step tends to the products' isolated ~17.6 ms
  (732 us a layer against 581); their start, tail and barrier hide only under a stream already running. Hence Phase 8.
- **Expected:** Phase 6 ~20-21 ms, Phase 7 ~18-19, Phase 8 ~15.5-16.5 (32-token prompt).

### Phase 6. Decode attention as one op (done 2026-10-19)

- [x] **6a. In `ModuleOps::attention`** (`Session::attention_rows`, `tt_kernels::attention`; burn-tt
  `module::attention_rows`, `burn_tt::set_attention_rows`). For a few query rows a key/value head under one mask row,
  each head runs on one unit as seven kernels in one job, the intermediates in L1: `Q K^T` (a head's query rows read into
  row 0 of `A`'s slots, the keys through the unpacker's transpose), `SCALE_MASK_FILL`, the row maximum with its clamp,
  `SUB_EXP`, the row sum with its clamp, `DIV`, `P V`; its rows written into the stacked output, two 64-byte face rows a
  row and tile. Every kernel computes only the head's rows (`code_for_rows`, `reduce::math_programs_rows`; `SUB_EXP` now
  a row-limited kind). It replaces ten ops a layer.
  - **Bits:** a product's output row is its `A` row's alone; a reduction over columns and the lanewise kinds never move
    a value between rows; the row-limited programs keep the computed rows' bits. So it is the composition bit for bit.
  - **Gate:** `step209_attention_rows` (ttsim, card 0 `1792425779`); probe `silicon_probe_attention_rows`
    ([log](../learnings/performance-log.md#decode-attention-as-one-op-2026-10-19-cards-0-and-1)).
  - **Measured:** 35 us a layer at 64 positions (was ~102 in ten ops), 89 at 512 (card 0, alone). 1.7B 24.04 / 24.48 /
    25.39 -> **22.78 / 23.26 / 23.98 ms** (`1792425331`); 135M 11.40 / 11.80 / 12.26 -> **9.58 / 10.11 / 10.96**
    (`1792425715`).
  - **Found:** a stacked `q` puts head `h` at row `h`; built for rows `0..h + 1`, the head at row 31 ran the whole-tile
    `SUB_EXP` and `DIV`, ~8 us a score tile (the first build was slower than the ten ops at 512 positions).
- [x] **6b. One region a job** (2026-10-19, `tensor::share_semaphore_init`). The seven blocks were up to seven regions:
  blocks join one only with equal semaphore initialisations, and the matmul's, SFPU's and reduction's differ only by the
  semaphores the others lack. With their union: 35.2 -> **25.0 us** a layer at 64 positions, 89.4 -> 79.4 at 512 (card 0,
  `1792430708`). Fewer kernels (the scale and mask into the maximum's pre-map, `SUB_EXP` into the sum's) are left.
- [ ] **6c. RoPE and the cache's row write in the op** (now ~55 us a layer at `Step(64)`, 5 ops, after RoPE as one op and
  the value's copy gone): a burn-fusion chain fuser
  matching `cat` / `mul` / `MulAdd` / `ScatterNd` / `Attention` (burn-fusion keeps `Attention` one stream op), the
  `v` row's copy out of `qkv` with them.
- [ ] **6d. The context written as `o`'s packed row** (its repack and packing, ~10 us a layer).

### Phase 7. The other small ops

- [x] **RMSNorm's statistic and `DIV` as one op** (2026-10-19, `Session::rms_norm`, gate `step210`): every unit computes
  the statistic from the packed twin and divides its tiles, at least 2 tiles a job (64 readers of one 2-tile row cost
  more than they save). Isolated 19.8 -> 14.4 us a norm; in the step ~36 -> ~30. With 6b: 1.7B 22.78 / 23.26 / 23.98 ->
  **22.24 / 22.70 / 23.43 ms** (`1792432143`). Less than the ~1 ms estimated: inside the step a small op costs about
  twice its isolated time (the prefetch reading beside it).
- [x] `x / stat * gamma` (2026-10-27, decided: model code calls a custom op, as RoPE does): `burn_tt::rms_norm(x, gamma,
  eps)`, on a decode row the scale multiplied inside `rms_divide`'s job, bit for bit (gates `step210`, `step195`); every
  Llama norm calls it. 1.7B 11.31 / 12.63 -> **11.18 / 12.58 ms** at 32 / 256 tokens (card 0, `1793114447`, `1793114653`).
- [x] The residual `Add` with the next norm (2026-10-27, built, **opt-in**: slower). A fuser (`ADD_RMS_NORM`,
  `ADD_RMS_NORM_SCALED`: the `Add`, then Burn's composition or the norm's custom op, the sum kept as a second output);
  one-row products now write their packed twin, so every unit adds the two packed rows for the statistic
  (`Session::add_rms_norm`). Bit for bit (gates `step210`, `step195`, card 0 `1793121892`), but a 1.7B step 11.18 ->
  **11.26 ms** (`1793122290`, `1793122494`): the merged job's five kernels on the norm's 32 units cost more than the
  barrier it removes. `TT_ADD_RMS_NORM=1` turns it on. The estimate (~0.12 ms at A1's ~2.5 us a removed op) assumed
  the merged op cost no more than one of the two.
- [x] **The token on the card** (2026-10-19): a row's argmax as one native op (`Session::argmax_row`, gate `step211`), the
  greedy token every trace's second output, read back alone (`TracedGenerator::generate_tokens`; gate `step212`). The
  greedy rule is now `torch.argmax`'s (the first maximum) on the host and the card. ~0.34 ms a 1.7B step.
- [x] **Sampling on the card** (2026-10-19, `generate::Sampler::Sample`: temperature, top-k, top-p, a host-seeded draw;
  gate `step212` against burn-flex), eagerly over the trace's logits: the device sort cannot be captured beside a model's
  traces (program cache). 112 ms a token: correct, slow.
- [ ] **A native top-k** (`k` ≤ ~64, two phases like the argmax), so that sampling's softmax, top-p and draw are small
  ops inside the trace. Needed before sampled decoding is fast.
- [x] **RoPE as one op** (2026-10-19, `burn_tt::rope`, a burn-fusion custom operation on `Tt`; gate `step213`): ~0.6 ms.
- [x] **Three copies gone** (2026-10-19): the same-shape reshape, the value's copy before its cache write
  (`Session::scatter_rows_into_block`), the context's repack and `o`'s packing (attention writes the row and its twin).
  ~0.36 ms. 1.7B **20.95 / 21.41 / 22.18 ms** (`1792444936`), 135M **8.00 / 8.59 / 9.38** (`1792445319`).
- [x] S5 (2026-10-26, [decode-small-ops.md](decode-small-ops.md) S5): measured per entry and per barrier. A small op
  is 9.8 us on the 135M (11.6 on the 1.7B) on unit 0, ~3 of it its barrier; a host-side analysis of the captured
  streams finds 194 of the 135M's 591 barriers (82 of the 1.7B's 403) unneeded by the data, and a capture now drops
  them (`Session::set_barrier_prune`, on): 135M **7.35 -> 6.97 ms**, 1.7B 11.76 -> 11.59 at 32 tokens (card 0).
  Sparse waits after products (what-if 0.03-0.07 ms) and back-to-back ops in one launch (~0.12 ms at best) are not
  built.
- [ ] The step's mask from the position in one op (8 int ops, ~80 us).

### Phase 8. A continuous weight stream

- [x] **The feasibility probe** (2026-10-20; [log](../learnings/performance-log.md#the-phase-8-feasibility-probe-the-step-without-its-weight-stream-and-under-one-2026-10-20-cards-0-and-1)).
  Two `#[doc(hidden)]` process-wide switches in `tt-kernels`, set by
  `silicon_bench_models::traced_generation_stream_ceiling` (`TT_PROBE_STREAM=off|skip|skip-bg:<units>`, `TT_TILES=all`):
  `tensor::probe_skip_weight_reads` (a packed-`A` product gathers no `B`, nothing prefetches) and
  `session::probe_background_stream` (unbarriered `READ_RUN`s of a scratch tensor on idle units, after each unit's last
  op, at the top of `DATA`). Gate `step214_stream_ceiling_probe` (ttsim; card 0 `1792465030`): no `GATHER` under the
  switch and the products differ from the ops run fresh (watched failing with the switch ignored); the background
  moved after a first op on every unit, bit for bit; nothing left behind. 1.7B, 32 / 256-token prompts:

  | Mode | Timed, card 0 | `Step(64)` replay, card 1 profile |
  |---|--:|--:|
  | today (`off`, `1792463041`, `1792463722`) | 20.92 / 22.15 ms | 20.42 ms: products 15.79 (455 GB/s), 283 other ops 4.63 |
  | weights free (`skip`, `1792463279`, `1792464102`) | **7.85 / 9.10** | **7.46**: products' fixed cost 3.45 (35.5 us each), other ops 4.02 |
  | weights free, 40 idle units streaming (`1792465141`, `1792465817`) | 10.43 / 11.80 (the stream's own end) | 10.02 |
  | weights free, 56 idle units streaming (`1792465032`, `1792465464`) | 12.56 / 13.95 (the stream's own end) | 12.00 |

  - **The ops' tax under a stream.** With ~260 GB/s read on idle units, the step's own work costs 1.34x; with ~300 GB/s,
    1.61x. Almost all of it is latency: `control` (the `LAUNCH`, `KERNEL`, `CB`, `POKE` entries, each of which drains the
    mover's reads) goes from ~0.9 to ~3.5 ms, and the rest is barrier and data wait. Today's prefetch already costs the
    small ops ~1.3 ms (5.28 against 4.02).
  - **The bound and the go/no-go.** With the stream fully overlapped, `max(14.3 ms of weights, the ops under a ~470 GB/s
    stream: ~14-15 by extrapolation)` is ~15 ms, ~6 ms under today. **Go**, by the rule set beforehand (≥ 2 ms).
  - **What binds is L1, not the bound.** A busy unit can stream ahead only as far as its arena holds: ~72 MB over 120
    units against ~290 MB of weights a layer, in windows of ~200-300 us. Realistic: **~2-3 ms**, and **~0.6 ms a step
    for every 100 KB a unit** freed for prefetched `B` (12 MB more a window, 24 layers). So the space audit decides
    how far Phase 8 goes.
- [x] **The space audit** (2026-10-20, `session::probe_space_report`, the same runs: per op kind, the highest `DATA`
  address planned, stream entries and program-cache bytes, and how many ops are alike). `Step(64)`, every tile:
  - **`DATA`:** the small ops reach 12-50 KB (attention 33, RMSNorm 29, RoPE 25, element-wise 12-50); the row scatter
    and select 208 KB; the mask's and RoPE table's block copies 532 KB (four ops at the step's start). **The products
    reach 807 KB**: a packed `A` is read into `kt / 32 + 1` slots, but its ring is planned with `kt` (64 slots, 266 KB,
    at `K` = 2048). `down` (streamed) reaches 578 KB.
  - **Stream:** 404 ops are 32 distinct (label, list, entries a unit): the 24 layers are one layer's ops 24 times, so
    a layer sub-trace with base-relative addresses would hold ~1/12 of the entries.
  - **Programs:** 39-70 KB a trace on its fullest unit (one-tile sessions: argmax 78 KB alone).
  - **Ranked candidates** (space freed, cost):
    1. the packed-`A` ring planned with the slots it fills: ~254 KB a product unit, bit for bit, small;
    2. the block copies' batch sized under a reservation: 532 -> ~266 KB on the units at the step's start, small;
    3. one layer sub-trace called 24 times: ~16x fewer stream entries and GDDR, and fewer stream fetches under load
       (the ops' tax is latency); large (argument refs in the mover). Proposal:
       [layer-sub-traces.md](layer-sub-traces.md) (17.8% of a layer's words differ from the next's, 86%
       of them tensor references). **Built 2026-10-23, opt-in:** 5.6x fewer entries (the argument tables, not
       16x), and the step 0.15 ms slower with the body read from GDDR;
    4. `PROGRAMS` (4 x 32 KiB) and `MAILBOXES` (128 KiB) audited for what they still need: unmeasured;
    5. the timestamper's 16 KiB only under `profile`: small, little.
- [x] **The packed-`A` ring planned with the slots it fills** (2026-10-20, `matmul::plan_layout_packed_a`, candidate 1
  above): a one-row product with a packed `A` plans `A`'s ring with `kt / 32 + 1` slots, not `kt`, so `B` and the outputs
  sit lower. The products' highest `DATA` address 807 -> **553 KB** (254 KB a unit free for prefetched `B`), bit for bit:
  ttsim `step187`, `step190`, `step191`, `step193`, `step203`-`step205`, `step207`, `step209`, `step210`, `step214`; card 0
  `step204`, `step205`, `step207`, `step214` (`1792466471`) and `step188` (`1792466602`, traced against eager). The step is
  unchanged until something uses the room (20.90 / 22.15 ms, `1792466481`).
- [x] **A prefetch transaction ID** (2026-10-20, Step 3 of the plan; `dm::TXN_PREFETCH` = 6, `record::GATHER_PREFETCH`,
  `op::WAIT` word 1 `dm::WAIT_PREFETCH`). B issues a flagged `GATHER`'s reads on ID 6, so the entries that drain its own ID
  (launches, barriers, `WAIT`s, chunk ends, `COPY_WORDS`) leave them in flight; a prefetch `WAIT` drains them, and so does a
  list's end. ID 6's cap is `TILE_IN_FLIGHT_CAP`, set once at the mover's start; the fast read path is told to write the
  tag (which holds the ID) again whenever the ID changes. The record refuses the flag on anything but plain reads (packed
  BF16, a transposing read). Gate `step215_prefetch_txn` (ttsim; card 0 `1792468449`): the flag changes no read; every slot
  holds the oracle's bytes (the host's expansion and a DRAM mirror) after a prefetch `WAIT` and after a list's end; with
  the mover's own cap at one, **a plain gather stalled 127 times, a prefetch 0** (its reads are not on that ID). Whether a
  snapshot after a plain `WAIT` catches a read still in flight is printed, not asserted: 0 of 4 rounds, a read lands ~1 us
  after its issue. Card 0: the safe image 22 of 22 (`step49`, `step118`, `step172`, `step173`, `step188`, `step204`,
  `step205`, `step214`, `step215`; `1792468459`), `fast-reads` 18 of 18 (`1792468676`), `fast-reads` + `encoded-guard` 13 of
  13 (`1792468695`); firmware clippy plain, `encoded-guard`, `profile`, `fast-reads`.
- [x] **`B`'s ring at the top of `DATA`** (2026-10-20, Step 4; `matmul::matmul_requirements_top_b`): a packed-`A` product
  under `Staging::Slots` plans `A` and the outputs at the arena's start and `B` against its end (a gap buffer between), so
  the ops between two products, which plan low (12-208 KB), leave the next product's `B` ring alone. No cost: 20.94 /
  22.21 ms (`1792469766`) against 20.90 / 22.15.
- [x] **Busy units read their first job's `B` ahead: built, correct, slower; opt-in** (2026-10-20, Step 5;
  `Session::set_busy_prefetch`, `TT_BUSY_PREFETCH=1`, `tensor::BUSY_PREFETCH`, `trace::Capture::move_block`). In a capture,
  each of a product's units whose ops since the last product all planned below its `B` ring (`OpRecord::l1_top`, each
  known) splits its first job's `GATHER` into a prefetch list; the capture's own run reads it with the job, and the
  stream moves it back to just after the last product; the job's gather list ends with the prefetch `WAIT`.
  - **Gates:** unit `a_units_first_job_reads_its_b_ahead_where_the_ops_before_plan_below_it` (which jobs, the flag, the
    wait, `B` at the arena's end; not where an op planned higher, is unknown, or `B` is read transposed); unit
    `a_moved_block_leaves_every_op_its_own_entries`; `step216_busy_prefetch` (ttsim, card 0 `1792469431`): bit for bit
    the ops run fresh, on and off; the op before each later product starts ≥ 5 entries later on each unit with its own
    entries, the product 4 fewer; watched failing with the move left out. Card 0 15 of 15 with it on (`step188`
    bit for bit); ttsim `step187`, `step191`, `step193`, `step209`, `step210`.
  - **Measured (it on):** 1.7B **21.93 / 23.78 ms** against 20.94 / 22.21 off (card 0, `1792469641`, `1792469766`). The
    profile (card 1, `Step(64)`, on `1792470019` against off `1792470407`): **the products 15.24 -> 11.66 ms** (1.7 GB of
    `B` read ahead), but **the residual add 0.67 -> 5.15 ms** (14 -> 107 us each) and RoPE 0.81 -> 1.55; RMSNorm and `mul`
    1.92 -> 1.21. **Why:** a `GATHER` is expanded synchronously and B issues in order, so on a busy unit the prefetch's
    ~128 reads (~75 us at a unit's rate) are issued before the next op's entries: the op after the last product (the add)
    waits for all of it to be issued. The reads overlap only the waits that follow.
  - **What would make it pay** (the products' 3.6 ms is there to take): the prefetch's reads issued while the mover would
    otherwise wait, not ahead of the next op. Either (a) **sliced at capture**, no firmware: the `GATHER` cut along `K`
    into chunks of ~8-16 tiles (`b_at` and `B`'s first tile moved by `k0`, as `push_stream_jobs` does), one put before
    each barrier the unit waits at between the two products, so ~2-4 us of issue rides each barrier's wait; or (b) **a
    prefetch pump in the mover**: a flagged `GATHER` queued and expanded a tile at a time from the mover's wait loops
    (barrier, kernel, credits) under ID 6's cap, the prefetch `WAIT` issuing what is left first. (b) hides more and is
    a firmware change of the hot loops; (a) is host only.
- [x] **A prefetch pump in the mover** (2026-10-20; `record::PrefetchCursor`, `InFlight::try_room`, `noc::has_room`,
  `dm::PREFETCH_CAP`). A resident-`A` prefetch `GATHER` is queued, not expanded: B issues its ID's cap of reads when it
  arrives and one a turn of its wait loops (barrier, the roles' acknowledgements, the writer's completion) while ID 6
  has room; the prefetch `WAIT`, a new prefetch and a list's end issue the rest. Bounded a turn because a read completes
  before the next is issued (ttsim, row 72; ~0.27 us an issue on silicon): room alone issued them all at once. The
  cursor's reads are the expansion's (unit `a_prefetch_cursor_reads_what_the_expansion_does`); ID 6's cap is settable
  (`TT_PREFETCH_CAP`, applied as each list starts). `step215` now asserts the deferral on both targets: the last slots
  are unread at a plain `WAIT`, 4 of 4 rounds. Card 0: the safe image 25 of 25 (`1792503652`, `step188` among them),
  `fast-reads` 19 of 19 (`1792503993`), `fast-reads` + `encoded-guard` 13 of 13 (`1792504131`); `--smoke` on card 0
  with the pump and the settable cap, 1,155 of 1,155 (`1792505704`).
  - **Measured, busy prefetch on** (card 0, 32 / 256-token prompts): cap 8 21.58 / 23.25 ms (`1792504143`); cap 4 21.23 /
    23.07 (`1792505352`); **cap 2 20.74 / 22.25** (`1792505230`); **cap 1 21.49 / 21.76** (`1792505109`); off 20.97 /
    22.19 (`1792505474`). Profile at cap 8 (card 1, `1792504522`): products 11.59 ms, the residual add still ~99 us
    (was 107 without the pump, 14 off).
  - **What it showed:** the add was never waiting mostly for the prefetch's issue, but for its own reads behind the
    prefetch traffic at the GDDR controllers (~100 units x 8 requests queued). A window's bytes are not free: whatever
    the prefetch reads, the small ops' latency-bound reads wait behind. At best ~0.2-0.4 ms (cap 2 or 1), against the
    3.6 ms the products give up. **Busy prefetch stays opt-in.**
  - **What would change it:** small ops that do not read GDDR between products -- their activations held in L1 from
    the op that makes them (the "nothing in L1 survives an op" assumption of the 2026-10-19 review) -- so the stream
    could take the windows' whole bandwidth. That is the L1 budget question (the space audit), and a larger design:
    [planned-l1-residency.md](planned-l1-residency.md), now a plan: its Phase 1 (the plan, reported) is done (2026-10-20,
    `step217`): a 1.7B step keeps 401 of 548 tensors resident in 112-171 KB a unit, 294 MB of reads spared a step.
    Phase 2's mechanism done the same day (`step218`, `step219`, bit for bit on the 1.7B): sharded tensors resident,
    the step even and 0.1 ms better under busy prefetch; rows read whole on many units, by multicast, cost 23 ms more
    (contended multicasts). 2f (rows held by two of their readers) and 2g (never-read writes dropped): 21.03 -> 20.06
    ms a step (`1792537293`), but busy prefetch still does not pay (cap 2, 20.13): Phase 3 (the products' outputs)
    decides whether residency frees the windows. **Phase 3 (2026-10-21):** the products' outputs resident, the adds,
    RMSNorm and `silu_mul` reading no GDDR, 19.50 / 20.71 ms (`1792595329`); busy prefetch still does not pay (cap 1,
    19.74 / 20.66; cap 2, 19.75 / 21.37), but **not because of GDDR**: under it the products go 14.79 -> 11.25 ms and
    the add 8 -> 70 us a call with no GDDR read left (`1792597186`). **The cause** (probed 2026-10-21): NC drained the
    prefetch transaction ID at every list's end, and an ID's outstanding count is the tile's NIU's, so NC waited for B's
    prefetch reads while B, waiting for NC, issued more. Only B drains it now, and **busy prefetch pays**: cap 2, 18.40 /
    19.92 ms against 19.51 / 20.75 off (`1792608583`, `1792608298`); caps 1 / 4 / 8 19.71 / 20.46, 18.73 / 20.54, 19.08 /
    20.74 ([planned-l1-residency.md](planned-l1-residency.md) 3b). **On by default** since (2026-10-21), at a prefetch
    cap of 2 (`dm::PREFETCH_CAP_DEFAULT`; `TT_BUSY_PREFETCH=0` turns it off): 18.39 / 19.90 ms with no settings
    (`1792615740`). **Then (2026-10-21)** [planned-l1-residency.md](planned-l1-residency.md) Phase 4 (re-scoped) and 5a, the
    reader sections read ahead: the small ops 5.00 -> 4.47 ms, the products 12.92 -> 13.46, the step unchanged (18.44 /
    19.83, `1792619858`): **under busy prefetch the step is bound by the weight stream the small ops' windows carry.**
- **The stream's rate (2026-10-22).** The step is bound by the weight stream (Phase 5a's finding), so first the meter,
  then the levers it ranks; every lever is bit for bit (a column's result does not depend on which unit computes it or
  when its `B` lands).
  - [x] **Step 1, the meter** (no behaviour change; [log](../learnings/performance-log.md#the-weight-stream-metered-what-is-read-ahead-the-prefetchs-wait-and-each-units-own-time-2026-10-22-card-0),
    profile `1792629809`):
    - what a product's units read ahead is charged to it (`trace::OpRecord::ahead`, `OpBytes::ahead`; a moved
      `BUSY_PREFETCH` list had been no op's, ~1.7 GB a step uncounted); the summary gives the step's weight rate and
      each label's in-span rate;
    - B stamps its prefetch `WAIT` in `OPS` mode (`trace_mode::PREFETCH_WAIT`, the `prefetch` column; firmware
      `profile` feature only);
    - every unit is timed at every op from its own stamps (`profile_mode::unit_times`: the host counts each unit's
      barriers as the mover's window does, from its stream; `skew.tsv`, and a table in `summary.md`).
    - Gates: `step216` `what_a_product_reads_ahead_is_counted_as_its_own` (ttsim, card 0; watched failing with the
      attribution off), `profile_mode` unit `a_units_stamps_are_told_apart_by_its_barriers`, `silicon_profile_mode`
      (unit 0's per-unit times equal its spans sample by sample; a plain `WAIT` is no prefetch's). Card 0: safe 26 of 26
      (`step188` among them), `fast-reads` 14 of 14, `fast-reads` + `encoded-guard` 15 of 15, profile feature 6 of 6.
    - **Found:** the weights stream at **389 GB/s** over the step, 2.97 GB of 6.98 read ahead; unit 0 waits 28 us
      (`qkv`) and 67 us (`gate|up`) a layer for its own prefetch, which drains at the prefetch ID's cap of 2 while the
      kernel waits; the low-numbered units read ~1.35x slower than the high ones whatever the channel (`down`: units
      0-31 ~150 us, 48-63 ~112; unit 20 the slowest in 24 of 24 layers); in `gate|up`, units 64-103 finish their
      prefetched job in 13 us and wait ~235 us at the barrier.
  - **Step 2, the levers, re-ranked by Step 1:**
    1. [x] **The prefetch `WAIT` drains at the read cap** (2026-10-22), not the prefetch cap: once the kernel waits for
       its `B`, no small op is left to protect. The mover raises ID 6's cap to `max(B's read cap, the prefetch cap)`
       for the drain and sets it back after (`InFlight::set_cap_in_flight`: the room left empty, so the next issue
       re-reads the counter; `set_cap` refills it and is only for an idle ID). Unit test
       `a_cap_changed_in_flight_counts_what_is_in_flight`; ttsim `step172`, `step173`, `step215`, `step216` in the three
       images; card 0 safe 26 of 26 (`1792631341`, `step188` bit for bit), `fast-reads` 14 of 14 (`1792631719`),
       `fast-reads` + `encoded-guard` 15 of 15 (`1792631860`). **Found:** at first it took B's cap whatever it was, and
       `step215` (B's own cap set to 1) stalled the prefetch 119 times on silicon (ttsim passed: a read lands before the
       next issue there); hence the `max`. **Measured:** the step 18.44 / 19.83 -> **18.28 / 19.84 ms** (`1792631901`);
       profile `1792632189`: products 13.46 -> 13.23 ms (`gate|up` 248 -> 242 us, `o` 24 -> 19), the small ops 4.47 ->
       4.57. Small: once the product waits, GDDR is already full, so unit 0 still waits ~55 us a layer in `gate|up`.
    2. [x] **The products' units served evenly** (2026-10-22; planned as a deal weighted by a per-card measurement of
       each unit's rate). Card 1 matched card 0 to ~0.1 us (`1792632927`), but with the tiles opened in reverse order
       the order reversed too (`1792633435`): a unit's service depends on where it sits among the units reading
       together, so no measurement of units alone predicts a product's. What evens it is the in-flight cap: every B
       read at 2 gave 17.27 / 18.91 ms (8: 18.28 / 19.84; sweep 1-12 in the
       [log](../learnings/performance-log.md#the-weight-stream-metered-what-is-read-ahead-the-prefetchs-wait-and-each-units-own-time-2026-10-22-card-0)),
       `down`'s units even, but the 135M and ResNet-18 ~3% slower. So only a weight stream's reads take it:
       `record::GATHER_STREAM` (bit 28) on a packed-`A` product's gathers (idle units' prefetch lists included; not
       with `GATHER_PREFETCH`), B lowering its own ID's cap to `dm::STREAM_CAP` (default 2, `TT_STREAM_CAP`) for the
       record and restoring it after. **1.7B 17.27 / 18.71 ms** (`1792637346`), the 135M unchanged; the step's
       weights at **416 GB/s** (profile `1792638075`; 389 before). Gates: record unit test (the flag reads what the
       record reads; refused with `GATHER_PREFETCH`), ttsim `step172`, `step173`, `step205`, `step215`, `step216` in the
       three images; card 0 safe 41 of 41 (`1792636757`: `step185`, `step187`, `step188` bit for bit),
       `fast-reads` 17 of 17 (`1792637187`), `fast-reads` + `encoded-guard` 18 of 18 (`1792637329`); `--smoke` 1,182 of 1,182
       (`1792639145`, `step12_mnist` included).
       `TT_IN_FLIGHT_CAP` (B's own cap, a development knob) stays for further sweeps.
    3. [x] **`PREFETCH_UNITS` / `THIN_STREAM_UNITS` re-tuned** (2026-10-22): the counts are settings
       (`Session::set_prefetch_units`, `set_thin_stream_count`, `set_stream_prefetch_units`; development knobs
       `TT_PREFETCH_UNITS`, `TT_THIN_STREAM_UNITS`, `TT_STREAM_PREFETCH_UNITS`), each a whole-`K` or a streamed
       product's, since the two answer differently. **They stay at 40 / 64 / 40.** 1.7B, card 0, 32 / 256-token prompts:
       - `down` not prefetched: 40 units 17.27 / 18.74 ms (`1792674690`), 48 17.43 / 18.87, 56 17.68 / 19.07, 32 17.03 /
         18.45 (`1792674837`, `1792674983`, `1792675777`);
       - `down` prefetched (4 below), whole-`K` / streamed units: 40 / 40 **16.88 / 18.33, 16.87 / 18.32** (`1792675628`,
         `1792677187`); 32 / 40 16.88 / 18.29 (`1792676432`); 24 / 40 17.81 / 19.21; 16 / 40 17.79 / 19.12; 32 / 32 17.04
         / 18.42; 40 / 32 17.10 / 18.53; 32 / 48 16.98 / 18.45; 32 / 56 17.21 / 18.61; 56 / 56 17.32 / 18.70;
       - a weight stream on 56 or 80 units (40 / 40): 17.30 / 18.67, 16.99 / 18.48 (`1792677630`, `1792677779`); the
         stream cap 3 or 4: 17.09 / 18.50, 17.48 / 18.94 (`1792678560`, `1792678709`).
       The idle prefetch's cap is settled: its lists read at the stream cap; at B's cap of 8 instead, no different
       (`1792638681`, `1792638827`). Gate: `step205` `the_prefetch_and_the_stream_take_the_session_s_unit_counts`
       (ttsim; watched failing with either count ignored).
    3a. [x] **`qkv` waits for its prefetch** (94.5 us a layer, 45 of it unit 0's prefetch `WAIT`). **Closed by 4**, not
       by its own candidates: with `down` prefetched, `down`'s columns move to 40 idle units, its busy units drop to 24,
       and `qkv`'s busy units start their prefetch sooner: `qkv` **90.4 -> 50.8 us** a layer, its prefetch wait 43.9 ->
       8.0 (profile `1792678073` against `1792638075`). Not built, with the reason: the prefetch issued before `down`'s
       barrier (`down` reads at ~434 GB/s in its span, so those reads come out of its share); more idle units on `qkv`
       (48 and 56 measured slower, 3); a wider job a prefetching unit (its wait is gone).
    4. [x] **`down` prefetched** (2026-10-22, below).
- [x] **`down` prefetched** (2026-10-22; `Session::set_stream_prefetch`, on, `TT_STREAM_PREFETCH=0` turns it off).
  `BlockPlan::push_prefetched_jobs` takes a streamed product too: each idle unit (at most
  `set_stream_prefetch_units`, 40) takes a column and its first two chunks' `B` gathers (the ring's two halves) move
  into its `PREFETCH` list; the third chunk refills the first half after the first chunk's kernel, as unprefetched.
  `push_stream_jobs` deals the plan's `col_runs`. Nothing changed in the session or the firmware.
  - **Gates:** unit `a_streamed_product_s_idle_units_read_their_columns_first_two_chunks_ahead` (every `B` tile once,
    chunks 0-1 in the list, halves kept, the idle units' columns, at most the streamed count, none when off);
    `step205` `streamed_products_prefetch_their_first_chunks_and_replay_as_the_ops_run_fresh` (ttsim: under a K-block
    limit of 8 every product streams, the second and third take units 4-7, bit for bit against the ops run fresh;
    **watched failing with three chunks read ahead**, element 0 92.7 against 656.6); card 0 35 of 35 (`1792678887`:
    `step185`, `step187`, `step188`, `step191`, `step205`, `step216`, `step219`); `--smoke` 1,184 of 1,184
    (`1792679835`, `step12_mnist` included).
  - **Measured** (card 0, every tile; on `1792679220`, off `1792679522`): 1.7B **17.29 / 18.72 -> 16.87 / 18.30 ms**;
    the 135M 7.62 / 8.29 / 9.10 against 7.61 / 8.30 / 9.10 (its products do not stream), prefill unchanged; ResNet-18
    traced 13.94 against 14.13, ViT-Tiny 9.21 / 9.21.
  - **The profile** (`Step(64)`, `1792678073`): the step's weights at **426 GB/s** (416 before), 3.48 GB read ahead
    (2.97); a layer's ops 647 -> 626 us: `qkv` 90 -> 51, `down` 157 -> **179** (21.3 of its 68.2 MB ahead). **What binds
    now is `down`'s busy units:** units 0-23 each read a whole 1.06 MB column in its span at one unit's rate (~176 us at
    the stream cap of 2, ~6 GB/s), while the idle units finish their halves in ~85-94 us; `down` reads ~260 GB/s in its
    span. More idle units cannot help while any unit reads a whole column (56 measured slower), nor a higher stream cap
    (3, 4 slower). **Next:** a column's `K` chain handed between two units in order (the second reads its half of `B`
    ahead while the first computes, then reloads the first's FP32 partial: the same chain, bit for bit), so no unit
    reads a whole column in the span (built and measured slower, below); and the window before `o` (~50 us a layer of
    GDDR idle: `o`'s 17 MB lands in ~35 us of a 64 us window, and `gate|up`'s prefetch starts only at `o`'s barrier).
- [x] **A column's `K` chain handed between two units: built, bit for bit, slower; opt-in** (2026-10-22;
  `Session::set_stream_handoff`, off; `TT_STREAM_HANDOFF=1`, `TT_STREAM_HANDOFF_AT=<h>`). With a streamed product
  prefetched, each column its own units would read whole is handed while a unit of the deal has no job: the column's
  unit computes chunks `0..h` (default half) and writes the FP32 partial to a scratch tensor; the spare unit reads the
  packed `A` and the next two chunks' `B` in the same op as a prefetch (`GATHER_PREFETCH`, which the op's barrier
  leaves in flight; `tensor::HANDED_AHEAD`); after the barrier, a second op on the spare units alone reloads the
  partial into the slot chunk `h` reloads from, drains the prefetch and computes the rest (`tensor::Handoff`). The
  same chain in the same order. No firmware change (decided with the user: two ops first, an in-op signal only if the
  profile says).
  - **Gates:** unit `a_streamed_product_s_whole_columns_are_handed_between_two_units` (the column's unit stops at `h`
    and writes the partial, not `C`; the spare unit's first-op list is the packed `A` and two prefetched gathers; the
    second op reloads into the slot the partial was packed from, then `WAIT_PREFETCH`; every `B` tile once; nothing
    handed with no spare unit; watched failing with the other half's slot); `step205`
    `handed_columns_replay_as_the_ops_run_fresh` (ttsim and card 0: units 6-7 spare, handed at chunk 1 and at half,
    bit for bit against the ops run fresh; the second op plans its product's L1; **watched failing** with the partial
    reloaded into the other half's slot and with its reload dropped, element 0 782.9 against 656.6, and without the
    second op's L1 noted). Card 0 with it on, 36 of 36 (`1792685194`: `step185`, `step187`, `step188` bit for bit,
    `step191`, `step205`, `step216`, `step219`); `--smoke` 1,185 of 1,185 (`1792687707`, off). Found by the bench, not a gate at first: the second op did not note its
    L1 layout, which a capture with a persistent region refuses (now asserted in `step205`).
  - **Measured** (card 0, every tile, `traced_generation_stream_ceiling`, 32 / 256-token prompts): off **16.88 /
    18.32 ms** (`1792685519`); handed at half 17.07 / 18.51 (`1792686180`); at chunk 1 17.22 / 18.61 (`1792686332`);
    at chunk 3 16.88 / 18.33 (`1792686483`); at half with the spare units taken from the top (104-119 first) 17.33 /
    18.74 (`1792687395`, not kept).
  - **The profile** (`Step(64)`, handed at half, `1792686773`, against `1792678073`): `down`'s first op 179 -> 155 us
    and its second 17 us (the spare units' tail and its barrier), so `down` 179 -> 172; but the columns' units 0-23
    took 93-137 us each for half a column, not ~88, because the spare units' reads of the other halves run beside
    them: inside `down`'s span the step is bound by GDDR's aggregate rate, not by one unit's. And `qkv` 51 -> 72 us a
    layer (its prefetch wait 8 -> 35): the spare units 24-47 are `qkv`'s, and their read ahead of `qkv` starts only
    after the second op.
  - **The firmware follow-up is not built:** an in-op signal (NC writes the partial into the spare unit's reload slot
    and increments a flag there; B waits on its own flag) would remove only the second op's ~17 us a layer, not the
    contention or `qkv`'s later start. Revisit if `down`'s span ever runs below the aggregate stream.
- [x] **The GDDR idle before `o`: each unit's own prefetch anchor** (2026-10-22; `Session::set_prefetch_anchor`,
  `PrefetchAnchor::Own` on by default, `TT_PREFETCH_ANCHOR=last|own-oldest`). `o`'s 17 MB landed in ~35 us of a 64 us
  window (profile `1792678073`), and `gate|up`'s prefetch started only at `o`'s barrier: every prefetch was anchored at
  the last weight-streaming product, though units 24-63 take no `o` job and 104-119 run nothing.
  - **What changed:** each unit is anchored at the last such product it took part in (its stream grew in it; a unit
    that took part in none, at the first) (`trace::Capture::unit_anchors`).
    - An idle unit's `PREFETCH` list waits for its own anchor's barrier. Which barrier is recorded in
      `OpRecord::ahead_waits`.
    - A busy unit's `BUSY_PREFETCH` list moves back as far as its own anchor, but never across an op that planned into
      its ring or did not plan at all (`session::busy_start`).
  - **Guards:**
    - A `B` the capture allocated or wrote in place (`trace::Capture::written`, the `drop_twin` writers and in-place
      padding fills) is never read ahead. This hole was latent under the old anchor too.
    - Probe-background units are never taken.
    - Changing the anchor drops cached plans.
  - **Gates:**
    - `step205`:
      - `own_anchors_start_a_prefetch_under_the_products_its_units_sat_out`: under `OwnOldest` the third product takes
        units 6-7, and their lists wait for the first product's barrier;
      - `a_b_the_capture_makes_is_not_read_ahead`, **watched failing** with the guard probed off
        (`session::probe_stale_b`): `OwnOldest`'s replays read the `B` before it is made, `Last`'s do not;
      - `a_busy_unit_reads_ahead_from_its_own_anchor_but_not_across_an_op_in_its_ring`, **watched failing** with
        `l1_top` probed away (`session::probe_busy_ignore_tops`): the list moves across a 64-row product in its ring
        and the bits differ;
      - claim five is pinned to `Last`.
    - The unit test `a_moved_block_leaves_every_op_its_own_entries` now covers a move across an op the unit sat out.
    - ttsim: `step205`, `step214`, `step216`.
    - Card 0 under `own-oldest`:
      - `step185`, `step187`, `step188` (bit for bit), `step191`, `1792694043` (claim five failed there, since it
        assumed the old anchor; it is now pinned to `Last`);
      - `step205`, `step216`, `step219`: 22 of 22, `1792694391`.
    - `--smoke` on card 0 with `Own` the default: 1,188 of 1,188 (`1792696786`, `step12_mnist` included).
  - **Measured** (card 0, every tile, 32 / 256-token prompts; [log](../learnings/performance-log.md#each-units-own-prefetch-anchor-2026-10-22-card-0)):
    - last 16.87 / 18.29 ms (`1792695646`), **own 16.79 / 18.20** (`1792695341`);
    - own with the longest-idle units first 17.13 / 18.67 (`1792694718`);
    - own with 48 / 56 / 32 idle units 17.15 / 18.54, 17.30 / 18.59, 16.84 / 18.24;
    - the 135M 0.01-0.03 ms slower; ResNet-18 and ViT-Tiny unchanged.
  - **The profile** (`1792696105`): a layer is 661.6 -> 657.3 us (`gate|up` 235 -> 232, `down` 179 -> 173); the small ops
    in the window are 0.5-1.4 us slower each; the weights stream at 429 GB/s.
  - **What it says:** the bytes read ahead are the same 3.48 GB. A unit's arena holds one job's `B`, so the window
    carries the same reads sooner and cannot carry more. Adding idle units costs the small ops more than it gives.
  - **Next:** what binds the window is L1 per unit, i.e. the space audit's candidates (the block copies' batch, layer
    sub-traces). The next lever of "The stream's rate" is room for a second job's `B`, not an earlier start.
- [x] **`record::PrefetchCursor::new` refuses `GATHER_STREAM | GATHER_PREFETCH`** (2026-10-22, H0 of "Room for a second
  job's `B`" below), as the expansion does (`RANGE`); before, only the host's check stopped the pair. The cursor-against-
  expansion unit test's refusal table has the pair (**watched failing** first: `Ok(())` against `Err(2)`). ttsim
  `step172`, `step173`, `step215` in the three images; card 0 `step215`, `step216` (`1792701998`); firmware clippy plain,
  `encoded-guard`, `profile`, `fast-reads`.
- [x] ~~Busy units stream too~~ (decode-step-bandwidth B2.1-B2.4): **superseded** by what was built since: the prefetch's
  own transaction ID (`dm::TXN_PREFETCH`, waited on by `WAIT_PREFETCH`), `B`'s ring at the top of the arena with the
  rule that the ops between two products plan below it (`matmul_requirements_top_b`, `session::busy_start`), and the
  prefetch pump issuing the reads from the mover's waits instead of slicing them between the small ops' entries
  (`record::PrefetchCursor`).
- [x] Depth and the caps (`PREFETCH_UNITS`, `THIN_STREAM_UNITS`) re-tuned from a profile after Phases 6-7 (2026-10-22,
  "The stream's rate" 3 above: unchanged).

### Room for a second job's `B` (2026-10-22)

A unit's arena holds one job's `B` (the anchor's finding above), and there is no room for a second: `DATA` is 917,504
bytes, the residency region keeps 98,304 of it, and a `K` = 2048 product needs 553,280 (`A`'s 3 slots, 2 outputs and a
128-slot `B` ring at the arena's end), so even a one-column second ring (64 slots, 266,240 bytes) is 320 bytes short,
with nothing left for the small ops. The room is elsewhere in L1: `MAILBOXES` is 128 KiB of which ~28 KiB is used, and
`PROGRAMS` (4 x 32 KiB) is idle while a trace replays (every traced kernel runs from the program cache). Decided
(2026-10-22): take both, then build a second ring. The new map:

| Region | Before | Now |
|---|---|---|
| `DATA` | `0x20000..0x100000` (896 KiB) | unchanged |
| `CAPTURE_ONLY` | - | `0x100000..0x116000` (88 KiB, where the mailboxes were) |
| `PROGRAMS` | `0x120000..0x140000` | `0x116000..0x136000` |
| `MAILBOXES` | `0x100000..0x120000` | `0x136000..0x140000` (40 KiB; each block in `l1::FIXED`) |
| `TRACE_ARENA` (captures only, P1-P3) | - | `0x20000..0x136000` (+216 KiB) |

- [x] **H0** (2026-10-22): the cursor's refusal (above), `DATA need` in the space report (`session::space_report`, beside
  `top`). Baseline (card 0, every tile, `1792702009`): 1.7B **16.79 / 18.20 ms**; a step's products `top` 819,200 /
  `need` 807,040 (the probe floor), `down` 578,240, the row scatter and select 208,000, RoPE's and the mask's block copies
  532,480, RMSNorm's packed row 266,240.
- [x] **M1, every fixed block named** (2026-10-22, nothing moved): `mailbox::PROBE_SCRATCH` (the icache and `.ttinsn`
  probes' literals), `l1::FIXED` (every block of `MAILBOXES` with its extent, B's mailbox through its read endpoints,
  NC's through its stream words), checked in order and inside the region at compile time; unit test
  `every_fixed_block_lies_in_mailboxes_and_none_overlap` with a moved block refused.
- [x] **M2, the mailboxes compacted** (2026-10-22; an ABI change). `MAILBOX_BASE` 0x136000, `DUMP` +0x400, role mailboxes 0x800
  apart from +0x800 (words, guard at +0x100, dump at +0x400: the 16 KiB stride was room for programs that moved to
  `PROGRAMS` long ago), B's mover mailbox +0x2000, NC's +0x3000, the probes' scratch +0x3400, NC's list ring +0x4000;
  `PROGRAMS` moved down against what the mailboxes left (`l1::CAPTURE_ONLY`); `TRACE` and the program cache unchanged.
  - **`DATA` itself stays 896 KiB.** The first form gave `DATA` the 88 KiB (`0x20000..0x116000`) and was measured (card 0:
    31 of 31 model and residency gates, `1792706003`; 1.7B 16.73 / 18.15 ms, `1792708053`), but every choice sized from
    the arena changed with it, which the gates found: a `K` = 1024 product took 6 columns, not 5, and its residency cap
    fell from 97,984 to 50,816 bytes (`step219`'s attention test: the mask in GDDR); a pipelined element-wise op's runs
    became 36-37 tiles, not 32, and a 37-tile run's `B` met a 36-tile run's outputs in one half (`step60`: a release
    wait); `[512]^3`'s blocks changed (`step22`); `step85`'s 113-tile `K` no longer needed a continuation; and on one tile
    a traced 135M's column sums took longer runs whose programs no longer fit beside its traces (`step193`, smoke run
    `1792706348`, silicon only). So the room is capture-only: eager sizes are as before, and a capture's choices are
    pinned to `DATA` (P3).
  - New gate `step221_l1_map`: the capture-only arena (the old mailboxes) and every word of `MAILBOXES` outside `FIXED`
    poisoned on every tile (on ttsim before the session opens), a workload of eager ops, a norm, a product and a
    planned capture with replays, and every poisoned word intact, with a control (a word written at the old
    single-core `STATUS` is reported, and only it); and, on ttsim, a second session opened on the first's L1 computes
    the same bits (**watched failing** with `QUEUE_HEAD` left out of the mover's start: the second session's results
    differ).
  - **Gates.** ttsim: `step3`, `step4`, `step21`, `step22`, `step52`, `step60` (both), `step85`, `step172`/`step173`/
    `step215` in the three images, `step187`, `step193`, `step205`, `step216`-`step221`, `step12_mnist` (golden
    unchanged), the `tt-kernels` unit tests; firmware clippy plain, `encoded-guard`, `profile`, `fast-reads`,
    `ttinsn-experiment`. Card 0, a `Session` bring-up first: `step185`, `step187`, `step188` (bit for bit), `step191`,
    `step193`, `step219`, `step221`, `silicon_probe_region_cliff`, 35 of 35 (`1792709565`); `step221` again in a new
    process (`1792709899`); `--smoke` 1,188 of 1,188 (`1792709902`).
  - **Measured** (card 0, every tile): 1.7B **16.76 / 18.17 ms** (`1792711476`; 16.79 / 18.20 before), the 135M 7.62 /
    9.09 (`1792711624`), ResNet-18 traced 14.14 and ViT-Tiny 9.20 (`1792711671`): unchanged, as nothing uses the room
    yet.
- [x] **P1, the card's list checks take the trace arena** (2026-10-22; firmware). `l1::TRACE_ARENA` (`DATA`, `CAPTURE_ONLY`
  and `PROGRAMS`) and `l1::LISTED`, what a list may name in L1 on the card: `Entry::decode`'s checks of `CHECK_FLAGS`,
  `COPY_WORDS`/`COPY_HALFWORDS`, `WRITE_L1`, `READ_L1`, `WRITE_L1_MCAST` and `INDEX_PICK` take it (plain `READ`/`WRITE`
  were bounded only by L1's size on the card, `DATA` by the host, `tt_kernels::dm::contained`, which is unchanged: an
  eager list stays in `DATA`). The guard's `POKE` stays in `DATA`. Unit test
  `entries_name_the_trace_arena_and_not_the_mailboxes` (every kind, at the capture-only arena's start, the program
  slots' start and last slot: decoded; at the mailboxes' base: refused; **watched failing** with `LISTED` at `DATA`);
  `step218` `the_tile_takes_a_write_into_the_trace_arena` (a raw `WRITE_L1` into a peer's capture-only arena and last
  program slot lands byte for byte; **watched failing** on ttsim with the firmware's bound at `DATA`) and the mailboxes'
  base added to the refused writes. ttsim `step172`, `step173`, `step215`, `step218` in the three images; firmware clippy
  plain, `encoded-guard`, `profile`, `fast-reads`, `ttinsn-experiment`.
- [x] **P2, captures and replays leave the program slots to the trace** (2026-10-22; host). A capture's own setup runs
  (each unit's first kernel after `begin_capture` forgets the semaphores) run from the program cache, where the session
  places the program first (`Resident::pending_setup`, `setup_from_cache`; the descriptor names it, no slot is written),
  never role 0's fixed slot; and the runtime forgets what the slots hold at every replay, every capture's start and end
  (`Resident::forget_program_slots`), so an eager fixed-slot kernel writes its programs again. New gate
  `step222_programs_for_traces`: every unit's slots poisoned, a plan pass and a planned capture of a layer leave every
  poisoned word, the replays bit for bit (**watched failing** with the setup in the slot again); a fixed-slot product
  run after its slots were overwritten and a capture ended gives its own bits (**watched failing**, ttsim only: without
  the forgetting it runs the overwritten words and ttsim refuses opcode 0). ttsim `step17`, `step22`, `step60`, `step187`,
  `step193`, `step205`, `step216`-`step222`, `step12_mnist`, `tt-kernels` unit tests. Card 0 (P1 and P2): `step185`,
  `step187`, `step188` (bit for bit), `step191`, `step193`, `step218`, `step219`, `step221`, `step222`, 46 of 46
  (`1792713314`); `--smoke` 1,189 of 1,189 (`1792713658`); `step221` and `step222` added to it.
- [x] **P3, a capture plans into the trace arena** (2026-10-23; host; opt-in, `Session::set_trace_arena`,
  `TT_TRACE_ARENA=1`). Not as planned (the ops' arena widened, the products' chunk choice pinned): M2 showed that every
  choice sized from the arena moves with it. Instead the capture's **persistent region moves to the top of the trace
  arena** (`0x11E000..0x136000` at 96 KiB, in `PROGRAMS`), and **only the products' `B` rings reach past `DATA`**, up to
  the region's base (`l1::ring_end`; in a plan pass, where the planned capture's region will begin), so ring 1 is
  `0x9C000..0x11E000` (was `0x66000..0xE8000`). Every other layout, and every block width, group and run length, plans in
  `DATA` as outside a capture, so the plan pass and the planned capture agree and the region never narrows a product
  (no cap). The host's containment is per mover (`DataMover::set_arena`: `TRACE_ARENA` during the capture, `DATA`
  otherwise); a layout planned past `DATA` into the trace arena is noted as the op's (`l1::Requirements::plan`; without
  it the product's `l1_top` was 0 and no plan was made); a fixed region (`set_l1_region`, `TT_RESIDENCY_REGION`) stays at
  `DATA`'s top. Gate `step222` `a_capture_in_the_trace_arena_replays_as_the_ops_run_fresh`: a plan pass and a planned
  capture of a layer with a one-row product, in the trace arena and out of it, bit for bit as the ops run fresh, the
  region at or above `DATA.end`, a ring reaching past it (**watched failing** with the movers' host containment left at
  `DATA`: the gathers refused, `RANGE`). `step217`, `step219`'s cap test, `step221` and `step222`'s poison test pin it
  off (they test a region inside `DATA`, or nothing in the capture-only arena). ttsim with it on and off: `step187`,
  `step193`, `step205`, `step214`, `step216`-`step222`, `tt-kernels` unit tests. Card 0 with it on: `step185`,
  `step187`, `step188` (bit for bit), `step191`, `step193`, `step205`, `step216`, `step219`, `step222`, 46 of 46
  (`1792717686`); `--smoke` (off) 1,194 of 1,194 (`1792718017`).
  - **Measured** (card 0, every tile): 1.7B **16.78 / 18.15 ms** on (`1792719606`) against 16.78 / 18.16 off
    (`1792719760`); the 135M 7.57 / 9.08 on (`1792719910`; 7.62 / 9.09 off); ResNet-18 14.15 and ViT-Tiny 9.22 on
    (`1792719944`). Neutral: the ring moved up, but a unit's busy prefetch was already bounded by what it holds, one
    job. **Opt-in until R2** uses the room.
- [x] **R1-R2, idle units take a second job: built, bit for bit, slower; opt-in** (2026-10-23; host;
  `Session::set_prefetch_second_job`, `TT_PREFETCH_SECOND=1`). R1 needed no new layout code: a second ring is the same
  top-`B` layout built while `l1::ring_end` is lowered by the first ring, and every memo of a ring's layout or programs
  is keyed by the ring's end (P3). R2 (`BlockPlan::second_jobs`): each prefetching unit of a `K`-whole product whose `B`
  sits against its ring's end (`Staging::Slots`) takes a one-column second job in round 1, its `B` in ring 2 and read
  ahead in the unit's one `PREFETCH` list; none where ring 2 does not plan (outside the trace arena), none for a
  pipelined product (its jobs stage `B` in a half of the arena, the second's where the first's is: found as wrong bits,
  `step205`), and none where the plan's own units would run no more than a round anyway. Gate `step205`
  `second_jobs_in_the_trace_arena_replay_as_the_ops_run_fresh` (a `[1, 2048] x [2048, 640]` product after another: bit
  for bit, the prefetching units' entries 112 -> 132; **watched failing** with ring 2 on ring 1). ttsim with it on:
  `step187`, `step193`, `step205`, `step216`, `step219`, `step222`; card 0 with it on, 47 of 47 (`1792721192`, `step188`
  bit for bit); off (the default), ttsim `step187`, `step205`, `step216` and card 0 `step188`, `step205`, `step216`, 14 of
  14 (`1792724145`).
  - **Measured** (card 0, every tile, trace arena on): every `Slots` product, 1.7B **17.25 / 18.66 ms** (`1792721521`)
    against 16.76 / 18.17 without (`1792721671`). Profile `1792721989` against `1792722444`: 266 MB more read ahead a
    step (3,746 against 3,480), but the products 12.43 -> 12.88 ms, all of it barrier time (2.27 -> 3.27 ms): `qkv` 51
    -> 80 us a layer and `o` 22 -> 33 (their weights were already read ahead whole: a second job is only a column more
    of compute on an idle unit the others wait for), `gate|up` ~flat, `down` 171 -> 159. With `qkv` and `o` excluded
    (the rule above): 17.18 / 18.54 ms (`1792723098`, against 16.77 / 18.14, `1792723249`); profile `1792723545`:
    `gate|up` 231 -> 245 us a layer with 10.6 MB more of its 136 read ahead. **Why:** `gate|up`'s span is its busy
    units' (432 columns over 64, so some take 7); 40 columns more on idle units leave 392, and some still take 7, while
    the extra reads ahead contend with theirs.
  - **What it says, with P3:** room for `B` is not what binds the window now. P3 gave a capture's ring 216 KiB more
    room and nothing changed; R2 fills it and is slower. The Phase 8 probe's "~0.6 ms a step for every 100 KB a unit
    freed" (made before busy prefetch, residency and the stream's rate) no longer holds. Not built, on this evidence:
    R3-R4 (a two-deep prefetch queue on a second ID, busy units reading a second job ahead) -- a firmware change of the
    mover's hot loops whose premise, that more bytes ahead shorten the products, the measurements contradict. What the
    profile points at instead: a product's longest unit (the column deal against the units' count) and the aggregate
    rate inside its span (`down`, the handoff's finding).

### A product's longest unit (2026-10-23)

The next lever R2 pointed at: a product's longest unit (its column deal against its units) and the aggregate rate
inside its span ([log](../learnings/performance-log.md#a-products-units-metered-and-a-balanced-deal-2026-10-23-card-0)).

- [x] **The meter** (no behaviour change): each unit's bytes in a product's span against its own time
  (`trace::OpRecord::unit_bytes`, `skew.tsv`'s `span_mb`/`ahead_mb`/`span_gbps`, the slowest unit's MB and GB/s in
  `summary.md`). Gate `step216` `what_a_product_reads_ahead_is_counted_as_its_own`: each op's units sum to its bytes,
  and only a unit with a list read ahead counts any ahead (**watched failing** with those units' bytes dropped); card 0
  `1792762291`. Profile `1792762439`: `gate|up` 231.9 us a layer against 172.6 for its 80.9 MB in span at 469 GB/s;
  units 0-47 read 5 columns each (1.33 MB at ~6.0 GB/s) while the 40 prefetching units finish in 13 us; `down` 172.0
  against 99.9, units 0-23 reading whole columns; the head 830.5 against 767.3.
- [x] **The balanced deal: built, bit for bit, not faster; opt-in** (`Session::set_balanced_deal`,
  `TT_BALANCED_DEAL=1`; `BlockPlan::push_balanced_jobs`, `tensor::balanced_shares`). Where a `K`-whole `Slots` product's
  own units would run more than a round, every unit keeps its job read ahead (the prefetching units' and the first
  round), and the columns left are dealt over the plan's units and the prefetching ones, the remainder on the latter.
  Gates: unit `a_balanced_deal_gives_each_unit_its_in_span_share` (each column once, the first round unchanged, no unit
  over its share, the remainder on the prefetching units, `qkv`'s one round unchanged; **watched failing** with the
  later columns shifted one back) and `balanced_shares_are_even_with_the_remainder_last` (**watched failing** with the
  remainder first); `step205` `balanced_deals_replay_as_the_ops_run_fresh` (ttsim and card 0: bit for bit, each unit's
  columns in the span `[6, 6, 6, 6, 0, 0, 0, 0]` -> `[3; 8]`; **watched failing** with the later columns shifted, element
  1248 202.0 against 376.4). Card 0 with it on, 51 of 51 (`1792763568`: `step185`, `step187`, `step188` bit for bit,
  `step191`, `step193`, `step205`, `step216`, `step219`, `step222`, `step223`).
  - **Measured** (card 0, every tile): 1.7B **16.81 / 18.21 ms** off (`1792763914`), **16.86 / 18.21** on
    (`1792764065`). Profile on (`1792764360`): `gate|up` 231.9 -> 229.4 us, each of its 104 units reading 0.80 MB in
    the span at 4.2-5.1 GB/s (was 6.0 on 64), its barrier 13 -> 86 us; the head 830 -> 925 us.
  - **What it says:** the longest unit does not bind. A unit reads slower as more units read, so the span's aggregate
    rate stays at ~350-390 GB/s (`gate|up`) or falls (the head: 433 -> 389). A column (266 KB, ~60 us at a unit's rate)
    is too coarse to even out the low-numbered units' slower service.
- [ ] ~~`down`'s tail: a stream cap set by its readers~~ (firmware). **Not built**, on the evidence above: the bound
  inside a span is the aggregate rate, which more in flight a unit did not raise globally (`STREAM_CAP` 3-4 slower, "The
  stream's rate" 3) and fewer readers do not lift. Revisit only with a measurement of GDDR's occupancy across
  `down`'s span showing it idle while its 24 whole-column units read.

**Where Phase 8 stands:** a 1.7B step at 16.8 / 18.2 ms, its weights at ~430 GB/s over a replay (92% of the
469 GB/s measured ceiling, 84% of 512). Room for `B` (R2), the stored layer (sub-traces) and the deal (above) are all
neutral or slower. What is left between the products and the ceiling is the aggregate rate inside the products' spans
(~350-440 GB/s against 469) and the windows between them. The next decision is the user's: BF16 weight storage (half
the bytes; F32's floor is ~14.3 ms) or the order-changing items (split-K, flash-decode), which come back with numbers
before they are built (Rules): written up in
[split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md) (flash-decode ~0.8-1.1 ms at 512 positions,
recommended opt-in; split-K ~0.2 ms, parked). **Flash-decode probed and stopped** (2026-10-23): two halves a head on
64 units with no exchange save only 19.7 us a layer at `Step(512)` (18.19 -> 17.71 ms) before any exchange
([log](../learnings/performance-log.md#flash-decodes-probe-a-heads-positions-in-two-halves-2026-10-23-card-0)). The
F32 step's remaining lever is BF16 weight storage.

### Phase 9. BF16 weight storage (2026-10-26)

[bf16-weight-storage.md](bf16-weight-storage.md): the weights stored BF16, half the bytes, bit for bit (the checkpoint
is BF16 widened at load, and the Matrix Unit multiplies a BF16 `SrcA` as the same value in TF32, gate `step232`). A
1.7B step **16.80 -> 12.95 ms** at a 32-token prompt (`1792990193`, `1792989603`). The weights take ~7.3 ms of that
at 469 GB/s; the rest is what Phases 3 and 7 left, now exposed: the next work is there, and the stream's re-tuning for
half-size tiles (A5).

- **A5a** (the stream re-tuned): 11.72 ms. **A5b** (2026-10-26,
  [log](../learnings/performance-log.md#compact-bf16-slots-and-where-a-bf16-decode-step-goes-2026-10-26-cards-0-and-1)):
  - **The decision rule, set before building:** build compact BF16 slots if a profile shows GDDR idle in the gaps
    between products with weights left to read ahead. It did: `gate|up` read 32.4 of its 69.2 MB ahead (one 2-column
    job a unit), ~41 MB read in a ~177 us stretch before it (profile `1793023409`). Weights free, a step is 6.26 / 7.34
    ms (`1793024167`).
  - **Built, bit for bit:** BF16 tiles in 2,112-byte L1 slots: 11.78 / 12.96 -> 11.68 / 12.90 ms.
  - **Refuted for BF16 too:** more read ahead shortens no product. 3 columns a job read `gate|up` 48.7 MB ahead and its
    span stayed 125 us; 4 columns put the rings below the small ops' L1 and lost the busy prefetch (13.10 ms). As in
    "A product's longest unit", a product's span is its units' column work.
  - `fast-reads`: 11.74 -> 11.54 ms (card 1), the products only.
  - **What is left:** 283 small ops, ~3.0 ms a step, ~10.5 us each (barrier 20-40%, control 10-20%, one-tile data
    latency), and the products' barriers (2.1-2.5 ms). That is Phase 7's S5 (each op's fixed cost) and
    replay-fixed-cost P1 (barriers on the units an op ran on), and the 135M (7.3 ms, ~0.6 of it weights) measures it.
  - **S5 (2026-10-26):** the products' barrier time is the slowest unit's work: their consumers need most of their
    units (a what-if of sparse waits saves 0.07 ms). Barriers no data needs are pruned: 1.7B 11.59 / 12.17 / 12.95 ms,
    135M 6.97 / 7.54 / 8.27 (card 0). With prefill-operand-reuse P4 merged (2026-10-27, card 0): 1.7B **11.32 / 11.84 /
    12.64 ms**, 135M **6.92 / 7.47 / 8.14**
    ([log](../learnings/performance-log.md#the-decode-and-prefill-streams-merged-2026-10-27-card-0)).

## The larger changes compared (2026-10-18)

Four larger changes were left after Phases 1-4's bit-for-bit items. Each was checked against profile `1792298755`
(1.7B `Step(64)`, 32.32 ms) and the code. The profile, per layer, against `B`'s floor at 469 GB/s:

| Product | us | `B` floor | over | barrier | last block's compute | data over the floor |
|---|--:|--:|--:|--:|--:|--:|
| `qkv` (64 units) | 160 | 109 | 51 | 10 | 15.0 | 23 |
| `o` (64) | 74 | 36 | 38 | 3 | 15.0 | 18 |
| `gate\|up` (64, four 2-column jobs a unit) | 356 | 291 | 65 | 32 | 29.6 | ~0 |
| `down` (64, four K-blocks a unit) | 271 | 145 | 126 | 19 | 15.3 | 87 |
| **layer** | **861** | **581** | **280** | 64 | 75 | 128 |

**What the code showed** (corrections to the estimates above):
- **The "kernel tail" is compute,** not a fixed cost: the last block's compute at ~0.23 us a `K`-tile pair
  ([operating notes](../learnings/silicon-operating-notes.md#a-decode-products-weights-what-the-card-reads-at-2026-10-17-card-0)).
  - The kernel is already launched before the reads.
  - Under `Staging::Slots` (one block in the ring), every earlier block's compute is serialized with that unit's
    next reads too.
  - `down` round-trips each K-block's partial through GDDR.
- **The m = 1 kernel may do ~4x the work it needs.** `matmul_items_reloading` runs all four faces, where row 0 needs
  `fi = 0` and half of `SrcB`; row 0's accumulation order would be unchanged without the rest.
- **The position-major cache removes the two repacks a layer and no attention op.** A2's worth is now -1.36 ms at
  `Step(64)` and -1.56 at `Step(512)`, not -1.7 / -3.7.
- **About 232 us of a layer's ~448 us of small ops run on 1-2 units** (~5.6 ms a step):
  - the stacking row gathers;
  - the softmax's two SFPU ops;
  - RMSNorm's 1-unit statistic;
  - the repacks, gone since.

**The four, compared:**

| Change | Worth | Size | Risk | Verdict |
|---|---|---|---|---|
| Position-major cache (A-layout) | ~1.5 ms | ~100 lines (`kv_cache.rs`) | low: the same tiles in the same order | **done**: 1.2-1.3 ms measured |
| Compute under the reads | ~2-3 ms: the tails (1.8 ms) and `down`'s serialized K-blocks (~1-1.5) | medium, mostly host | medium | **done** as the m = 1 kernel's rows (1.4 ms) and a blocked `K` streamed (0.3 ms) |
| `A` once a product (G7) | <= 3.4 ms (the skip-`A` probe) | large | (a) multicast high (a NoC hang reboots the host); (b) vector layout medium | **done** as a packed `A` on tiles (1.8 ms with its larger K blocks), neither (a) nor (b) |
| The weight stream under the small ops (Phase 5) | ~2-3 ms realistic | ~1,100-1,400 lines | the speed: GDDR contention, and slack that shrinks as the others land | **done** with a simpler design: idle units, no store pass (1.9 ms) |

**Each change in detail:**
- **Compute under the reads, next.**
  - **First, the trimmed m = 1 kernel** (skip `fi = 1` and `SrcB`'s second half). It is small and bit for bit. It
    cuts every block's compute and shows whether a pair is bound by its `MVMUL`s or its unpacks.
  - **Then each job's `K` chain as streamed batches with a ring of two:**
    - each batch gathers `A` and `B` for its `K` chunk;
    - the first clears `Dst`, the last packs;
    - the order is the same.
    The batch format (`[VERSION, batches, capacity, role]`) already allows it. `down` then needs no K-blocking and
    no partial's round trip.
- **`A` once a product.**
  - **The cost:** `A`'s cost is the mover's ~360 cycles an entry (a `READ_RUN` tile costs the same). A thin read
    fetches 1,088 bytes and uses 128.
  - **First, a probe:** `A`'s read order rotated by unit. It tells the entries' cost from 64 units reading one
    channel at once.
  - **Then the vector layout:** the producers write a K-contiguous copy, and unpacker 1 reads it at a 128-byte row
    stride. The stride looks expressible; it is unproven on unpacker 1.
  - **Not multicast:** the leader still pays K entries, and a mistake risks the host.
- **The weight stream.**
  - **A simpler design:** units 64-119 run no op from one layer's `silu_mul` to the next's. So a store-time check
    that a prefetching unit runs nothing before its product can replace B2.2's L1 reservation.
  - **The prefetch is the product's own `GATHER`,** with `GATHER_A_RESIDENT`, so it lands in the slots the product
    reads.
  - **Limits:** `down`'s 1.06 MB column cannot be prefetched, and L1 holds ~30 MB a product.
  - **First:** the replay probe (`probe_prefetch` allowed in a capture, and `OpRecord.barrier` false for unbarriered
    ops).
- **Landing:** the four together reach ~24-26 ms, not 15-16. The rest is the small ops on 1-2 units, a parallel
  track spread the way Phase 1 spread the scatter and the select.

**Landed** (1.7B `Step(64)` replay, card 1 profiles):

| After | ms |
|---|--:|
| start of the comparison (`1792298755`) | 32.32 |
| position-major cache (`1792336760`) | 30.91 |
| the m = 1 kernel's rows (`1792341444`) | 29.55 |
| packed `A`, then its larger K blocks (`1792342758`, `1792344448`) | 27.90, 27.77 |
| a blocked `K` streamed (`1792359572`) | 27.51 |
| weight prefetch, 40 units (`1792363365`) | 25.61 |

Timed (`traced_generation`, card 1): **26.47 / 27.14 / 27.97 ms** (`1792364210`), from 33.02 / 33.52 / 34.42.

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

**Order of testing** (2026-10-20: ttsim only for small spikes of risky changes):
- ttsim: the small gates of a risky change only (a record, the mover, an encoding, a negative control that could
  hang a card). Model-level gates (`step185`, `step187`, `step188`, `step191`, `step193`, `step209`, `step210`, MNIST
  e2e) run on card 0 directly: `cargo xtask silicon --release --device 0 --filter <gate>`.
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
