# More bytes in flight for a decode product (roadmap Track 2 follow-ups, item 2b)

Opened 2026-11-02 on `traced-generation`, after item 2 (`26d26ea`). H11's "then" step in the design reference
([dataflow-compiler.md](../learnings/dataflow-compiler.md) §13, H11; §6, X9 and X10).

## Why

- Item 2's meter ([decode-product-meter.md](../completed-plans/decode-product-meter.md)) found the 1.7B's decode
  products **read-bound by each unit's bytes in flight**. In the slowest unit's work:
  - T0 starves 52-86%;
  - the ring waits 1-2%, and the kernel takes 11-18%;
  - `fast-reads` changes nothing.
- A unit reads ~4-5 GB/s: three 2,112-byte entries in flight (`STREAM_CAP`, packed half) at ~1.3 us a read under load.
- Little's law (T8): the lever is bytes in flight a unit, through either more requests or larger entries.
- Larger entries need a unit's share of the weight to be contiguous in one channel (X9).
- H11's bound (`1793463838`) measured that layout's reads alone: 64 units go from 395 to ~499 GB/s, but unevenly. This
  item measures it inside the product, where the kernel runs too.

This item **bounds** before anything is built. Nothing it adds is a default.

## Phase A0: more requests, today's layout (no build)

`TT_STREAM_CAP_BF16` = 3 (default), 4, 6 and 8, under the pipeline detail. This separates the request count from the
entry size. The knee at ~32 requests a channel (T8) predicts no gain, since 12-15 units read each channel. A5a chose 3
by step time, never with this meter.

## Phase A: contiguous `B` in large entries (timing only)

- **The probe.** `tensor::probe_contiguous_b` is a process-wide, test-only switch, set from `TT_PROBE_CONTIGUOUS_B` in
  `silicon_bench_models::traced_generation_stream_ceiling`.
- **What it changes.** Each `m = 1` product's `B` gather (`GATHER_STREAM`, `A` resident, so it reads only `B`) becomes
  plain `READ`s of the probe's entry size. They come from a scratch copy of the same byte count, laid out as a load-time
  layout would store it:
  - unit `u`'s share in channel `u % 8`;
  - the units sharing a channel stacked by `u / 8`;
  - each unit's batches in order.
- **What is unchanged.** The reads land in the same ring span, so the kernels, credits and barriers are today's.
- **Values.** The scratch holds zeros, so the products are wrong; tokens are not checked.
- **Prefetched jobs** keep today's reads. Their share of the bytes is reported.
- **The cap.** Plain `READ`s run under B's own cap (`TT_IN_FLIGHT_CAP`), so each probe run is paired with a baseline at
  the same cap. The products' stream reads keep `STREAM_CAP` either way.

## Decision rules (fixed 2026-11-02, before any measurement)

Per product (`qkv`, `o`, `gate|up`, `down` in the 1.7B's `Step(64)`, BF16 weights, card 1):
- **Span:** the slowest unit's work, from the pipeline table.
- **`S`, the stream's time:** the product's `B` bytes over the rate at which the same number of units read them alone,
  with the same layout, entry size and cap (`gddr_readers_by_channel`, `Mix::DecodeContiguous`).
- **Gap ratio:** `(span_today − S) / (span_probe − S)`.
- **Span ratio:** `span_today / span_probe`.

Rules:
1. **Kill** if the gap ratio is < 1.2 on every product at every entry size and cap. Record it, close H11's large-entry
   half, and go to item 5's other entries or item 3.
2. Otherwise, the span ratio and the unprofiled step time (32 / 256 tokens) decide whether Phase B is built.
3. A verdict that two runs disagree on is reported as not reproducible and is not tuned.

**Prediction:**
- the 64-unit products (`o`, `down`) reach a gap ratio ≥ 1.2 at 16 KiB, cap 1-2;
- the ~120-unit products (`qkv`, `gate|up`) change little, since the deal already reads ~499 GB/s at 96 units.

## Conditions and the baseline, re-read

**Conditions.** Every run is `traced_generation_stream_ceiling` with `TT_TILES=all`: every tile, as item 2 ran it.
Without it the bench takes one tile, and the decode traces' programs (127-140 KB a trace) leave no room for an argmax
capture.

**How the figures are taken.** Each product's figures come from `pipeline.tsv`'s `Step(64)` rows. The ops are
classed by their order in a layer (`o`, `gate|up`, `down`, then the next layer's `qkv`). Layer 0's `qkv` carries
another label (`float_matmul + rope + rope`), and the last product is the head. Each figure is the median over the
layers of each op's slowest unit.

**Item 2's runs re-read this way** (`1793637811` / `1793638678`):

| product | units | batches | slowest unit, us |
|---|--:|--:|--:|
| `o` | 64 | 1.0 | 16.3 / 16.4 |
| `gate\|up` | 120 | 2.5 | 123.4 / 122.7 |
| `down` | 64 | 4.0 | 116.9 / 116.7 |
| `qkv` | 120 | 1.0 | 34.2 / 34.6 |

- `o`, `gate|up` and `down` equal item 2's table.
- `qkv` does not. Item 2's log gives 51.5 us, 119 units, 1.3 batches and issue 75%. Read this way it is 34.6 us,
  issue 68% and starve 69%: still rule 2, read-bound. The log's row is corrected beside this item's results.

## Results (2026-11-02, card 1)

**Conditions.** Release, safe image, `profile` image, `TT_PROFILE_DETAIL=pipeline`, the 1.7B with BF16 weights, every
tile, a 512-position cache, `Step(64)`. Every figure is the slowest unit's work in us (median over the layers).

**A0: more requests at tile-sized entries buys nothing** (`TT_STREAM_CAP_BF16`, today's layout).

| BF16 stream cap | run | `o` | `gate\|up` | `down` | `qkv` | head |
|--:|---|--:|--:|--:|--:|--:|
| 3 (default) | `1793645082` | 16.6 | 123.8 | 116.7 | 34.3 | 446 |
| 4 | `1793645792` | 16.2 | 125.8 | 114.4 | 34.9 | 460 |
| 6 | `1793646497` | 15.9 | 137.1 | 111.9 | 34.8 | 510 |
| 8 | `1793647203` | 15.9 | 138.7 | 110.7 | 34.9 | 547 |

- The 64-unit `down` gains at most 5%.
- The 120-unit products lose up to 12% (`gate|up`) and 23% (the head): past the ~32-requests-a-channel knee (T8).
- The split is unchanged: T0 starves 82-91% of every large product's slowest unit.

**Phase A: matched pairs** (baseline → contiguous `B` in 16 KiB reads, each pair at the same `TT_IN_FLIGHT_CAP`).

| product | units | cap 1 (`1793650022`, `1793649323`) | cap 2 (`1793647913`, `1793648628`) | cap 3 (`1793650741`, `1793651451`) |
|---|--:|---|---|---|
| `down` | 64 | 116.2 → **81.5** (1.43x) | 115.6 → **83.5** (1.38x) | 116.3 → **86.8** (1.34x) |
| `qkv` | 120 | 46.0 → 39.6 (1.16x) | 46.8 → 32.7 (1.43x) | 39.6 → **28.4** (1.39x) |
| `gate\|up` | 120 | 129.5 → 124.1 (1.04x) | 131.7 → 133.6 (0.99x) | 127.2 → 135.4 (0.94x) |
| `o` | 64 | 17.8 → 17.9 | 17.2 → 17.7 | 16.8 → 17.0 |
| head | 120 | 466 → 453 | 453 → 457 | 450 → 452 |

- **`down`** gains at every cap, most at cap 1, as H11's bench has large entries at their best at cap 1.
  - Its mean unit: 99.2 → 71.6 us (cap 1), 97.8 → 65.7 (cap 2).
  - T0's starve falls from 83% to 75-81%, and the kernel's share rises to 15%.
- **`qkv`** gains 1.4x at caps 2-3. At cap 3 the probe (28.4) also beats the default configuration (34.3).
- **`gate|up`**: its mean unit is 12-14% faster at every cap (e.g. 86.3 → 74.3 us at cap 3), but its slowest unit is
  not. That is H11's evenness caveat: large entries serve 120 units unevenly, and the product waits for the last.
- **`o`** (one batch of 135 KB a unit) does not move.
- **B's own cap matters to the baseline.** `qkv`'s baseline is 34.3 us at the default cap of 8 and 39.6-46.8 at caps
  1-3, because its other reads share that cap. So only same-cap pairs are compared.

**The control: the layout alone buys nothing** (contiguous at 2,112-byte entries, cap 3, `1793652146`).
- `down` 122.7, `qkv` 39.2, `gate|up` 130.3 and `o` 17.1 us, against the cap-3 baseline's 116.3 / 39.6 / 127.2 / 16.8.
- So the gain is the entry size, not where the bytes sit. This agrees with H11's bench. The control's longer lists cost
  `down` 5%.

**Reproducible.** The cap-2 probe, run again (`1793652866`): `down` 83.7, `qkv` 32.9, `gate|up` 133.5 and `o` 17.8 us,
against 83.5 / 32.7 / 133.6 / 17.7.

**`S`, the stream's own time** (`gddr_readers_by_channel`, `1793653566`). GB/s at caps 1 / 2 / 3, 16 KiB entries,
contiguous:

| units | bytes a unit | GB/s |
|--:|--:|---|
| 64 | 541 KB (`down`) | 495.9 / 483.2 / 476.0 |
| 64 | 135 KB (`o`) | 472.4 / 461.4 / 455.5 |
| 120 | 218 KB (`qkv`) | 460.2 / 462.6 / 454.5 |
| 120 | 577 KB (`gate\|up`) | 469.5 / 470.1 / 453.8 |

For comparison, today's deal at tile-sized entries reads 319-399 GB/s on 64 units and 439-497 on 120.

**`S` as the rule fixed it is not computable for three of the products.**
- **The rule's bytes are not the span's bytes.** It takes "the product's `B` bytes", but part of each product's `B`
  is read ahead of its span by prefetching units.
- **For `qkv`, `gate|up` and `o`, a full-weight `S` exceeds the span itself.** `qkv`: 26 MB at ~460 GB/s is ~56 us,
  against a 28-47 us span.
- **The profile's `ahead_bytes` cannot correct it.** It counts what lists *queued* ahead (all of `qkv`), not what
  landed before the span.
- **So for those three only the span ratio is reported.**
- **`down` decides the rule alone, and its verdict holds under either reading of `S`.**

| `S` for `down` | cap 1 | cap 2 | cap 3 |
|---|--:|--:|--:|
| all 34.6 MB: 69.8 / 71.6 / 72.7 us | 3.97 | 3.70 | 3.09 |
| in-span 28.1 MB (less the 6.5 MB queued ahead): 56.7 / 58.2 / 59.1 us | 2.40 | 2.27 | 2.06 |

The figures in the cap columns are gap ratios.

**Verdict, by the rules fixed beforehand.**
- **Rule 1: the kill does not fire.** `down`'s gap ratio is 2.1-4.0, against 1.2.
- **The prediction:** held for `down`; missed for `o` (no gain); `qkv` gained where no gain was predicted.
- **Rule 3:** the repeat agrees.
- **Rule 2 (whether Phase B is built)** is settled by the unprofiled step times below.

**The step, unprofiled** (safe image, every tile, 32 / 256-token prompts, 9 samples, host-timed medians):

| configuration | run | decode step, ms |
|---|---|--:|
| default | `1793653745` | 10.937 / 12.140 |
| contiguous 16 KiB, cap 1 | `1793653969` | **10.545 / 11.954** |
| baseline, cap 1 | `1793654187` | 11.577 / 12.915 |
| contiguous 16 KiB, cap 3 | `1793654412` | 10.743 / 12.000 |
| baseline, cap 3 | `1793654630` | 11.177 / 12.415 |
| default again | `1793654854` | 10.957 / 12.125 |

- **Against its own cap, the probe saves 1.03 / 0.96 ms at cap 1 and 0.43 / 0.42 ms at cap 3.**
- **Against today's default it is 0.40 / 0.18 ms faster, measured.** That holds although it caps every other B
  read at 1, which alone costs the step 0.64 / 0.78 ms (baseline cap 1 against the default).
- **A built version would not pay that.** It would give the weight stream its own cap and keep the other reads at
  B's 8. If the two effects add, it lands near the pair's ~1 ms; the measured 0.18-0.40 ms is its floor.
- **28.4% of the weight bytes stay in today's tile-sized reads ahead** (`GATHER_PREFETCH`, 21.3 of 74.8 GB over the
  run). The built layout should read those large too.
- The default runs bracket the others within 0.02 ms.

**Rule 2: build Phase B.**
- **`down`** is 1.34-1.43x faster on its slowest unit.
- **The step** is 0.18-0.40 ms faster measured, ~1 ms by the pair.
- **`gate|up`** is the open half. Its mean unit is 12-14% faster, but its slowest unit is not, because large
  entries serve 120 units unevenly. So the built form must also deal for evenness: a unit's share sized to the rate
  it gets, or fewer units at a lower cap for `gate|up`. That is a schedule question for the cost model (X10), measured
  with this meter.

## Phase B (only if A's kill does not fire): the relaid product, bit for bit

Started 2026-11-02, after Phase A's kill did not fire.

**Plan, as first written.**
- **Storage.** A per-unit-contiguous `B` placement for a `DramTensor`, relaid on the card in the deal's order. The
  lowering emits Phase A's `READ` runs from it.
- **Gates:**
  - ttsim: decode-shaped `Blocks` and `Stream` products bit for bit against today's;
  - negative control: a unit's share rotated by a tile fails;
  - card 0: the 1.7B's four shapes, `step188`, and `step12_mnist`.
- **Then, as roadmap items:** X9's load-time layout pass, and X10's entry size and reads in flight in the cost model.

**As built** (`Session::set_weight_layout`; on by default since 2026-11-03, `TT_WEIGHT_LAYOUT=0` turns it off).

- **No new encoding: a one-channel `B` is X9's layout.** A `TensorRef` with `n == 1` already puts tile `t` at
  `base + t * slot`. So a region holding a record's tiles in the order it reads them (`kk * cols + j`) is a plain
  one-channel tensor `cols` wide, and the record reads it unchanged but for its `B` reference and `j0 = 0`. (All 32 of
  `GATHER`'s flag bits are taken, so a new mode could not have been a flag.)
- **The card reads such a `B` in large pieces** (`tt_isa::dm::record::BSpan`, shared by the host's checks and B's
  firmware).
  - **Only when the reference is marked.** The relayout sets `REF_LAID_FOR_READER` in word 4 of the reference's
    first entry, which `TensorRef::encode` leaves 0 and `decode` ignores. A one-channel chip's own tensors stay a read
    a tile: the first unmarked form coalesced them too, which `matmul_records_expand_to_the_old_lists` caught under
    its one-channel mask (0x1).
  - Each row of a one-channel, untransposed `B` is one run, and the whole block one run when its rows meet
    (`ct == cols`).
  - The runs are read in pieces of at most one request (16 KiB), the port rotating as a tile's would.
  - `expand_reads` and the read-ahead pump (`PrefetchCursor`) give the same pieces.
  - Any other `B` is read a tile at a time, as before, at no cost a tile: the check is once a record.
  - A stream's pieces keep the stream's own cap (`GATHER_STREAM` -> `STREAM_CAP`). A read ahead keeps the prefetch's
    cap. B's other reads keep theirs. Phase A's probe had to lower B's whole cap.
- **The relayout** (`Session::relay_weights`, called by `matmul_dram` before the product is executed).
  - **Which records.** In a capture, for a one-row product with a packed `A` and a `B` the capture neither allocated
    nor wrote, every streamed or read-ahead `GATHER` of `B` is rewritten to read its region.
  - **The region.** Keyed by unit, header (less its slots and the stream/read-ahead flags) and `B` reference, so the
    decode traces of every length share it. It is allocated in channel `u % 8` outside the capture's log and pool,
    and filled at once.
  - **The fill** (`tensor::relayout_jobs`). The record itself, `B` only, gathers a batch of terms into staging,
    and NC writes them to the region in pieces of 16 KiB. It runs paused (outside the capture) and is waited for.
  - **A planned capture.** It cannot pause, so it makes no region. A record without one reads `B` as it lies, which
    is still bit for bit.
  - **The pause runs movers only** (`Session::paused_movers`). A plain pause forgets every unit's tile state and
    records a setup run where the capture's next op needs one, as a measurement pause must (`step245`). The copies
    run no kernel, so the tile state is as they found it.
    - Before this, a capture that made regions held a few redundant setup runs a unit more than one that found them
      made. `step216_busy_prefetch`'s three structural tests caught it with the layout on: entry counts 43 against
      41, on ttsim and card 0 (`1793666474`).
    - The benches' timed traces came from second builds, which find the regions made, so the Phase B timings above
      stand.
- **Lifetime and staleness.** The regions follow their source (`Session::free_placement`): freed with it, deferred
  with it to a capture or a trace that holds it.
  - **A write in place marks them dirty** (`drop_twin`, which every in-place write calls). They are filled again
    before the next replay or relayout (`refresh_relaid`).
  - **A capture that writes a relaid source is refused at `end_trace`.** Its own replays would write the source
    behind the regions.
  - **Never silently stale**, and still a value cache, not a second weight: Burn's tensors are values, so a weight a
    model holds is never written in place.
- **Gates.**
  - **Host** (`tt-isa`, `a_one_channel_b_laid_for_its_reader_is_read_in_large_pieces`).
    - **The oracle** is the record's own definition: each 64-byte unit read goes to the same slot as a read a tile
      would put it, for whole and row-by-row spans, FP32 and BF16 slots.
    - **The pieces:** each ≤ 16 KiB and congruent; the pump gives the same reads; an unmarked one-channel `B` and an
      interleaved `B` are read a tile at a time.
    - **Watched failing** with a run's GDDR stride wrong.
  - **`step267_weight_layout`** (ttsim; card 0 `1793657310`, every tile):
    - **Bit for bit.** `down`'s and `gate|up`'s schedules (`[1, 8192] @ [8192, 64]` streamed, `[1, 2048] @
      [2048, 256]` `K` whole) with FP32 and BF16 `B`: three replays each equal the eager product and the unrelaid
      replay.
    - **Relaid once each.** The regions hold exactly the weight's slots and are freed with it.
    - **A write reaches the replays.**
    - **The negative control.** With the refill suppressed (`probe_stale_relaid`), a replay after a write gives the
      old product, so the replay reads the regions.
    - **The refusal.** A capture that writes its relaid weight is refused; without the layout it is not.
    - **Watched failing** with each copy batch reading from the weight's first term (caught once a record took
      several batches, which is why `gate|up`'s shape has `K` = 2048).
  - **The mover gates** `step172_encoded_reads` and `step173_mover_differential` pass on ttsim in the safe image,
    with `fast-reads`, and with `fast-reads` and `encoded-guard`. So does `step267` under `fast-reads`.
  - **`step267` is in `SMOKE`.**
  - **On card 0, the final code** (marker, read-ahead cap and source list included):
    - `step188_smollm2_1_7b::traced_generation_equals_the_eager_one` with `TT_WEIGHT_LAYOUT=1`, bit for bit
      (`1793665445`; also `1793664478` and `1793659265` on the earlier forms): the 1.7B's traces read every streamed
      weight through its regions;
    - `step267` and `step12_mnist`, 11 of 11 with the goldens unchanged, the two-chip test included
      (`1793665424`).

**Measured** (card 1, release, safe image, the 1.7B with BF16 weights, every tile, 512-position cache).

*Profiled* (`--profile`, pipeline detail; `Step(64)`, each product's slowest unit, us):

| run | `o` | `gate\|up` | `down` | `qkv` | the step's ops, us |
|---|--:|--:|--:|--:|--:|
| default (`1793658870`) | 16.9 | 124.9 | 119.0 | 35.2 | 10,759 |
| layout, read-ahead cap 2 (`1793659575`) | 12.3 | 110.6 | 89.4 | 18.1 | 10,042 |
| layout, read-ahead cap 1 (`1793660329`) | 12.1 | 106.8 | 88.2 | 18.1 | **9,504** |
| layout, read ahead a slot a piece, cap 2 (`1793661072`) | 17.1 | 126.8 | 84.5 | 29.7 | 10,054 |

- **The products gain more than in Phase A**, because their reads ahead are large too. `qkv`, mostly read ahead,
  is 1.9x faster; `o` 1.4x; `down` 1.3-1.4x; `gate|up` 1.1-1.2x.
- **But large reads ahead at the default cap of 2 slow the ops they run under.** Of the label sums, `rope` with
  the cache's scatter goes 586 -> 1,102 us and `rms_norm` 821 -> 969: GDDR is one queue (T8).
- **At a read-ahead cap of 1** the products keep their gain and `rms_norm` recovers. `rope` stays +351 us, where
  the next products' reads ahead now land once `qkv` ends sooner.
- **A slot a piece for reads ahead** keeps the small ops (`rope` +85 us) but loses half the products' gain. So
  reads ahead stay large, and the layout lowers their cap: `WEIGHT_LAYOUT_PREFETCH_CAP` = 1, applied by
  `set_weight_layout` and at a mover's start; `TT_PREFETCH_CAP` still rules.

*The step, unprofiled* (32 / 256-token prompts, 9 samples; prefill one sample):

| run | decode step, ms | prefill, tokens/s |
|---|--:|--:|
| default (`1793661878`) | 11.068 / 12.229 | 1,091 / 2,084 |
| default, read-ahead cap 1 (`1793662104`) | 11.521 / 12.790 | 1,085 / 2,074 |
| layout, read-ahead cap 2 (`1793662331`) | 11.142 / 12.165 | 1,050 / 2,008 |
| layout, read-ahead cap 1 (`1793662552`, `1793662994`) | 10.621 / 11.625, 10.609 / 11.641 | 1,035 / 1,981 |
| layout, read-ahead cap 1, stream cap 2 (`1793662771`) | 10.566 / 11.599 | 1,049 / 2,010 |
| default again (`1793663219`) | 11.038 / 12.264 | 1,085 / 2,075 |

- **The host's lookups were the rest.**
  - **Where the time went.** A free (`take_relaid`) and an in-place write (`drop_twin`) scanned every region,
    ~11,500 for the 1.7B. Prefill was 4-5% slower although its chunks' device time was unchanged (15,672 against
    15,630 us), and decode paid it too.
  - **The fix.** The relaid weights are now kept in a list of their own (`DramState::relaid_sources`, ~170), and a
    free or write looks there first.

*The final form* (layout on, read-ahead cap 1 by the layout):

| run | decode step, ms | prefill, tokens/s |
|---|--:|--:|
| default (`1793663597`) | 11.072 / 12.264 | 1,085 / 2,076 |
| **layout** (`1793663823`) | **9.886 / 10.932** | 1,072 / 2,048 |
| **layout again** (`1793664027`) | **9.913 / 10.940** | 1,087 / 2,078 |
| default again (`1793664232`) | 11.087 / 12.268 | 1,069 / 2,045 |
| default, the marker's build (`1793665594`) | 11.008 / 12.175 | 1,071 / 2,053 |
| **layout, the marker's build** (`1793665823`) | **9.894 / 10.897** | 1,094 / 2,094 |
| default again (`1793666028`) | 11.006 / 12.156 | 1,091 / 2,089 |

- **Decode 11.08 / 12.27 -> 9.90 / 10.94 ms (-10.7% / -10.8%): 101 tokens/s at a 32-token prompt.** Prefill is
  unchanged within its one-sample noise.
- **Cost: a second copy of the streamed weights.** 0.49 GB more taken on the fullest channel (3.34 -> 2.85 GB free),
  ~3.4 GB in all for the 1.7B's BF16 weights. An F32 model would double its 6.8 GB. To keep that GDDR, open the
  session with `TT_WEIGHT_LAYOUT=0`, or call `Session::set_weight_layout(false)` before the first capture (regions
  already made stay until their weight is freed). A region leaves its channel 64 MiB free (`RELAID_HEADROOM`); with
  less, that record reads its weight as it lies, with the same bits (`step267`'s full-GDDR test).

## Phase C: one copy instead of two (opened 2026-11-03)

The owner's next ask: X9's layout as a weight's only storage, giving back the ~3.4 GB the second copy takes on the
1.7B. Its bound comes first: what every *other* reader of a relaid weight pays to read the layout instead of the
weight as it lies.

### C0: the bound (a bit-exact probe; timing decides)

**The probe.** `tensor::probe_relaid_blocks` (or `TT_PROBE_RELAID_BLOCKS=1`) is test-only.
- **Which products.** Every product over a relaid weight that reads it as it lies: prefill chunks, eager products,
  a measurement's timing captures, a head's `m = 1` product in a planned capture.
- **How each `B` gather is split** (`tensor::split_relaid_gather`):
  - the record keeps `A`, now `GATHER_B_RESIDENT`;
  - each `K` row of the block is cut into runs of columns that sit in one region and follow each other there;
  - each run is plain `READ`s of at most 16 KiB from the region;
  - consecutive rows of one region whose rows are as wide as the block are one span;
  - tiles in no region stay a `B`-only `GATHER` as they lie.
- **Plain reads, not a record a run.** A packet carries a block's gather, kernel and scatter in at most
  `dm::LIST_MAX` (512) entries, and a record a one-column run overflowed it (`step188` refused it, `LENGTH`). A list
  that would still pass 256 entries keeps its gathers, and its bytes are counted as read where they lie.
- **Values.** While both copies exist the probe reads the same bytes into the same slots, so it is **bit for bit**,
  and timing is the only question.

**Gates.**
- **Host.** `tensor::relaid_split_tests`: what each L1 byte holds, against the record's own expansion, for column
  runs, one-column runs, a partly relaid weight, a block that is one run and a block narrower than its run. Three
  mutations were run: two are caught, and a third was caught once the narrow block was added.
- **`step267`** (ttsim in both read modes, and card 0 `1793682326`). After a decode capture relays the weight,
  `[64, k]` products, eager and replayed, equal the products read as they lie. Every tile comes from a region, and
  the control (`probe_relaid_shift`, each weight's first region read one slot off) gives other bits.
- **`step188`'s** traced generation with the probe on is bit for bit (card 0, `1793682326`; its output says the probe split its gathers).
- **Found on the way and fixed:** `meet`'s padding fill in place did not mark a relaid weight dirty (`66aef11`, gate
  `a_padding_fill_in_place_reaches_the_regions`). It is unreachable through today's products, whose packed `A`
  leaves their padding undefined, but not through this probe.

**Conditions.** Card 1, release, safe image, every tile; the 1.7B with BF16 weights and a 512-position cache.
- **Prefill** (`traced_generation_stream_ceiling`): 32 / 128 / 256 tokens at chunks of 64; and 7,992 tokens at chunks
  of 512 with an 8,192-position cache, one sample.
- **Eager products** (`silicon_bench_relaid`): the four weights at `m` = 64 / 256 / 512, ABBA in one process over the
  same regions.
- Decode at 32 / 256 tokens is the sanity check.

**Decision rules (fixed 2026-11-03, before any measurement).** Each rule applies per weight class: *column runs*
(`qkv`, `o`, `gate|up`, the head) and *one-column runs* (`down`, `K` streamed).
1. **Void** if the audit shows less than 99% of a class's tiles held by exactly one region. Fix it and rerun.
2. **The class is killed** if prefill at 128, 256 or 7,992 tokens is more than 2% slower than both brackets
   (off / on / on / off), or an eager product's median is more than 5% slower at any `m`. A killed class keeps two
   copies: the "cheaper middle".
3. **Both classes killed:** close one copy, and go to "regions that yield".
4. **Brackets that disagree:** report the result as not reproducible, and do not tune it.

**Prediction.** Column runs within ±1%. One-column runs 3-10% slower at 7,992 tokens, an entry a tile.

## Work

- [x] A0 on card 1 (2026-11-02, `1793645082`-`1793647203`): more requests at tile-sized entries buy nothing (Results).
- [x] Phase A probe (2026-11-02):
  - **Where it runs.** `tensor::probe_contiguous_b` and `Session::probe_contiguous_b`, applied in
    `enqueue_work_inner` after busy prefetch has flagged its records. A read ahead (`GATHER_PREFETCH`) keeps its
    reads; ops whose units hand data on (`linked`) are left alone.
  - **The scratch.** Each record gets a region of its own, keyed by unit, header and `B` reference. It is allocated
    with `alloc_on(u % channels)` outside any capture's log and pool, and kept, so a re-lowered op reads the same
    bytes.
  - **Host tests** (`tensor::contiguous_b`). The reads cover the slots exactly, front to back, in one channel, each
    congruent mod 64, the last shorter. What would leave `LISTED`, break the congruence or misname an entry is
    refused. Only a weight stream moves: a read ahead or a plain gather stays, and a stream that also reads `A`, or
    a transposed `B`, is refused. The first test was watched failing with the short last read dropped.
  - **The ttsim spike, `probe_contiguous_b`.** It runs `down`'s schedule (`[1, 8192] @ [8192, 32]`) and `gate|up`'s
    (`[1, 1024] @ [1024, 256]`) at 2,112- and 16,384-byte entries. Each runs to its end, every slot of `B` moves
    (1,064,960 bytes), and the output is `+0.0` from the zeroed scratch where the control's is not.
  - **Unchanged when off.** `tt-kernels`' host suite passes 342 of 342.
- [x] `S` from `gddr_readers_by_channel` (`DECODE_BYTES`, added: a product's bytes a unit) at each product's units (`1793653566`).
- [x] Phase A sweep on card 1: matched pairs at caps 1-3, the 2,112-byte control, a repeat; unprofiled pairs and
  the default before and after (2026-11-02, Results). Cap 4 was not run: cap 3 gained less than caps 1-2.
- [x] Rules applied: the kill does not fire; rule 2 builds Phase B. Recorded in the performance log, the design
  reference (H11, X9, X10) and the roadmap.
- [x] Phase B (2026-11-02/03): the one-channel `B` read in large pieces (`record::BSpan`), the relayout in captures
  (`Session::set_weight_layout`, opt-in), regions freed with their source and refilled after a write, the read-ahead
  cap the layout needs, the host's lookups by source. Gates on ttsim and card 0; decode 11.08 / 12.27 -> 9.90 /
  10.94 ms on card 1 (above).
- [x] The default: **on since 2026-11-03**, by the owner's call, for ~3.4 GB more GDDR on the 1.7B (`TT_WEIGHT_LAYOUT=0`
  opts out). A region leaves its channel 64 MiB free (`RELAID_HEADROOM`), else its weight is read as it lies
  (`step267::a_full_gddr_leaves_the_weight_where_it_lies`: watched failing on ttsim with no fallback, and on card 0,
  `1793679359`, with no headroom, where the regions took channel 0's last room and the capture's own stored streams
  were refused). The benches' conditions
  line records `weight_layout`. Before the default, the other models' gates and benches with it on:
  - **Card 0 gates with `TT_WEIGHT_LAYOUT=1`** (`1793666474`, `--keep-going`): 50 of 54 pass. That covers `step187`
    (the 135M and synthetic decoders, traced against eager), `step185`, `step233`, `step234` (BF16 weights), `step240`
    (BFP weights), `step205`, `step219` (residency), `step212` and `step224` (long context).
  - **The four that did not:**
    - `step263`'s Qwen3.5 traced test timed out at the default limit (it takes ~21 minutes);
    - three `step216` structural tests failed: the redundant setup runs above, fixed by the movers-only pause.
  - **On ttsim, with the layout on and off,** `step267`, `step245`, `step216`, `step205`, `step219` and `step187` pass.
  - **After the movers-only pause,** card 0 passes 9 of 9 with the layout on (`1793668937`): `step216`, `step267`, and
    `step263`'s Qwen3.5-4B traced generation, bit for bit against eager and agreeing with Flex.
  - **Benches** (card 1, release, every tile; default / layout / layout / default unless noted):

    | model | default | layout | default again |
    |---|--:|--:|--:|
    | SmolLM2-135M decode, 32 / 256 tokens, ms (`1793670594`-`1793670752`) | 7.024 / 8.261 | 6.996 / 8.224, 6.987 / 8.213 | 7.044 / 8.264 |
    | Qwen3.5-4B decode, 32 / 256 tokens, ms (`1793670805`, `1793671626`; one run each) | 64.11 / 64.45 | **62.85 / 63.26** | |
    | Qwen3.5-4B prefill, 32 / 256 tokens, ms | 464.2 / 3,712 | 464.5 / 3,716 | |
    | ResNet-18 forward, traced, ms (`1793670307`-`1793670481`) | 12.37 | 12.73, 12.54 | 12.66 |
    | ViT-Tiny forward, traced, ms (`1793670540`-`1793670581`) | 7.68 | 7.48, 7.66 | 7.66 |

    - **Qwen3.5-4B decodes 1.2-1.3 ms faster (-2%).** Its step is dominated by what roadmap items 3 and 4 name (~20 ms
      in no op's span, the F32 head, ~780 small ops), not its products; prefill is unchanged; its build takes 30 s
      more (596 -> 626 s).
    - **The 135M gains 0.04 ms (0.5%).** Its products are too small to be read-bound.
    - **ResNet-18 and ViT-Tiny are unchanged within run-to-run noise** (±0.3 ms). They have no one-row product but
      their classifier heads.
- [ ] **One copy instead of two** (the owner's next ask, 2026-11-03): X9 as the weight's only storage. Prefill and
  eager products read the same layout a column run at a time (a decode unit's share is whole runs, so one `GATHER` a
  run, no new encoding); the tied head's token lookups, in-place writes and downloads need their own paths; the split
  is fixed at load from the decode deal. **Bound first:** prefill and an eager `m > 1` product over the relaid layout,
  card 1, against today's.
- [ ] If one copy stalls: relay only `down` and `qkv` (Phase A's gains), roughly half the extra GDDR, measured.
- [ ] Regions that yield: on a refused allocation, free relaid regions no live trace reads before failing it; drop
  them on `set_weight_layout(false)`.
- [ ] `gate|up`'s evenness: its mean unit is faster but not its slowest. A deal that sizes each unit's share to the rate
  it gets, measured with the pipeline detail (X10).
- [ ] X9 at load time: the regions are made at a capture's first sight of a weight; a model loader could make them
  when the weights are stored, with the deal the decode traces will use.
