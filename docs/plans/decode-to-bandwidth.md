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
- [ ] **What is left per op** (decode-small-ops S5). With ~300 ops left, ~3.5 us of control an op is ~1 ms.
  - Measure it.
  - Then decide on a firmware change, such as one `PAIR_CALL` launch and its credits shared by back-to-back small
    ops.

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
  units can saturate, so this is now less likely to be needed.
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
  - prefetch for a K-blocked product (`down`'s column is 1 MB);
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
- [ ] `x / stat * gamma`: burn-fusion puts `gamma`'s `Reshape` in a block of its own, so the product by it is a separate op
  no fuser of the norm's block sees. Needs a decision: a deferred result in burn-tt (risky: much of it tells a device
  tensor by its materialised buffer), a change to burn-fusion's blocks, or model code.
- [ ] The residual `Add` with the next norm (the statistic needs the whole sum on each unit: products write no twin).
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
- [ ] S5: measure the ~3.5 us of control an op, then back-to-back small ops sharing one `PAIR_CALL` and its credits,
  and the next reader section fetched ahead.
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
       [layer-sub-traces.md](../proposals/layer-sub-traces.md) (17.8% of a layer's words differ from the next's, 86%
       of them tensor references);
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
    decides whether residency frees the windows.
- [ ] `down` prefetched: the first chunk batches' `B` gathers of `push_stream_jobs` on idle units. ~0.7 ms.
- [ ] Busy units stream too: prefetch reads on a transaction ID of their own, waited on only by their kernel; a
  reserved top-of-arena range small ops never plan into; the capture slicing a product's prefetch between the small
  ops' entries (decode-step-bandwidth B2.1-B2.4).
- [ ] Depth and the caps (`PREFETCH_UNITS`, `THIN_STREAM_UNITS`) re-tuned from a profile after Phases 6-7.

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
