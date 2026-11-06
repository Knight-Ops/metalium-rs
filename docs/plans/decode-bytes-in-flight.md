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

**As built** (`Session::set_weight_layout`; opt-in, `TT_WEIGHT_LAYOUT=1`. It was on by default for part of 2026-11-03,
then reverted because the second copy does not scale; see "The default reverted" below).

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

**The default reverted (owner, 2026-11-03): opt-in again, because a second copy does not scale.**
- **What changed.** `TT_WEIGHT_LAYOUT=1` (or `Session::set_weight_layout(true)`) turns the layout on. By default
  each weight is held once, as it lies, and a read ahead's cap is the mover's 2 again.
- **Why.** The second copy's GDDR grows with the weights: ~3.4 GB on the 1.7B in BF16 and 6.8 GB in F32. A model
  that fills a card, or is split across both, cannot afford it, and the 1.7B is a test model. C0 (below) showed that
  one copy in this form slows every other reader.
- **The rule** now on the roadmap
  ([constraint](roadmap.md#constraint-every-default-must-scale-to-larger-models)): a default holds a weight's bytes
  once. A gain that needs spare GDDR stays opt-in, with its cost named.
- **Kept.** The code, `step267` (in `--smoke`, so the opt-in path is still exercised), `silicon_bench_relaid` and
  the C0 probes. They may serve prefill and decode on different cards, where each card holds one layout.

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

### C0 results (2026-11-03, card 1): the kill fires for both classes; one copy closes in this form

**Eager products** (`silicon_bench_relaid`, `1793682506`). Medians of 14 host-timed products, ABBA in one process
over the same regions, each product bit for bit. Every tile was read from a region, and each weight's regions held
every tile exactly once.

| weight (its runs) | `m` = 64, us | `m` = 256, us | `m` = 512, us |
|---|--:|--:|--:|
| `qkv` (2 columns) | 5,306 → 7,093 (1.34x) | 2,230 → 5,124 (2.30x) | 5,512 → 8,694 (1.58x) |
| `o` (1 column) | 3,466 → 4,672 (1.35x) | 3,598 → 5,724 (1.59x) | 2,129 → 6,066 (2.85x) |
| `gate\|up` (1-2 columns) | 5,760 → 11,032 (1.92x) | 6,557 → 11,809 (1.80x) | 18,953 → 29,116 (1.54x) |
| `down` (1 column) | 13,325 → 18,265 (1.37x) | 13,097 → 22,036 (1.68x) | 6,620 → 22,383 (3.38x) |

**Prefill** (`traced_generation_stream_ceiling`, `TT_TILES=all`, chunks of 64; tokens/s at 32 / 128 / 256 tokens).

| run | probe | prefill | decode step, ms |
|---|---|--:|--:|
| `1793682568` | off | 1,093 / 2,148 / 2,089 | 9.73 / 10.24 / 10.90 |
| `1793682781` | on | 948 / 1,872 / 1,826 | 9.80 / 10.29 / 10.94 |
| `1793683019` | on | 1,046 / 2,062 / 2,008 | 9.80 / 10.31 / 10.94 |
| `1793683256` | off | 1,090 / 2,141 / 2,081 | 9.80 / 10.30 / 10.94 |

- **What the probe moved.** 2.30 M / 2.04 M gathers became 48.7 M / 48.3 M entries, reading 107 / 109 GB of `B` from
  regions and none as it lies.
- **The audit.**
  - One-column runs: 48 weights and 491,520 tiles, all held once.
  - Column runs: 49 weights and 1,179,648 tiles, all held, but **98,176 (8.3%) held twice**.
  - Run widths: 10,752 runs of one column and 8,447 of two.
- The 7,992-token prefill was not run: the eager rule had already decided both classes.

**Verdict, by the rules fixed beforehand.**
- **Rule 2, eager.** Every eager product over the regions is 1.34-3.38x slower, against a 5% kill. Both classes are
  killed.
- **Rule 2, prefill.**
  - The one-column class (audit valid) is 3.7-13% below both brackets at 128 and 256 tokens: killed again.
  - The column-run class's prefill is **void by rule 1**, since only 91.7% of its tiles are held exactly once. Its
    eager kill stands, from a run whose audit was exact.
- **Rule 3:** both classes are killed. **One copy closes in this form.** Next is "regions that yield".
- **Decode** did not move: 9.73-9.80 / 10.90-10.94 ms.
- **The prediction:**
  - column runs, within ±1%: missed (eager up to 2.3x);
  - one-column runs, 3-10% slower: low (prefill 4-13%, eager up to 3.4x).

**What it found.**
- **The cost is the entries, not the bytes.**
  - A decode deal's runs are one or two columns wide, so a prefill block (several columns, a `K` chunk) becomes an
    entry for every row and run: ~21 entries for each 5-entry gather, ~4x the list.
  - Each one is built and sent by the host for an eager product, and stored and walked by B for a replay.
  - The bytes read are the same, in reads of one or two tiles: no larger than today's.
- **A layout chosen for the decode reader is the wrong layout for a block reader.** For both to read one copy in
  large pieces, either the deal's runs must be as wide as a prefill block, which Phase A's evenness argues against,
  or a record must cut a region's rows into a block with an L1 pitch, which is a new encoding.
- **Today's decode deals do not partition a weight.** 8.3% of the column-run weights' tiles sit in two regions:
  prefetch units' shares and the thin runs of different captures overlap. That is ~0.2 GB of the 1.7B's ~3.4 GB
  second copy. A one-copy design would first need one deal a weight.

## Phase C1: by-column weights, the scalable form (opened 2026-11-04)

**The form.** A stored weight `[K, N]` (`Kt x Nt` tiles) is laid **by column**: column `j`'s `Kt` tiles are contiguous
in channel `channels[j % n]`, at `base[c] + ((j / n) * Kt + kk) * slot`. The tile contents and the bytes are those of
today's interleaved weight. It is held once and made at load time.
- **Why not C0's prescription.**
  - *One deal a weight:* the decode deal depends on the capture. The prefetch units, the spread and `thin_runs`'
    remainders differ between captures, which is what put 8.3% of the column-run tiles in two regions. Fixing a deal
    would change decode's schedule and its evenness.
  - *A pitched read a region:* each 1-2-column run would still be its own `GATHER`, so a block would cost several.
- **What the form gives.**
  - **Decode.** Any deal's column run is a union of whole columns, and each column is one run in one channel (128 KiB
    at `K` = 2048 in BF16). Decode reads it in 16 KiB pieces, reads ahead included. A K-whole decode job places `B` in
    L1 by column, `(j * kt + kk)`: the kernel takes per-pair addresses (`matmul::slots_layout`), so this needs no new
    microkernel. Streamed `down` jobs are one column already.
  - **Blocks** (prefill, eager, measured-block captures) keep one `GATHER` a block and read a tile at a time, as today:
    the same entries and the same channel spread along a block row. Only the order within a channel differs.
  - **Nothing else to maintain.** No relayout, regions, headroom, staleness or partition.
- **Encoding.** Two bits in word 4 of the `B` reference's first entry, which `TensorRef::encode` leaves 0:
  - `REF_BY_COLUMN`, the placement;
  - `REF_L1_BY_COLUMN`, this reader's L1 order.

  Every record but `GATHER` refuses a by-column reference by name. Records expand alike on the host and the card and
  are never encoded, so the read modes are untouched.
- **Results.** Bit for bit, since each output's `kk` pairs stay in ascending order.
- **The two-copy form stays opt-in** (`TT_WEIGHT_LAYOUT=1`, `step267`).

**Decision rules (fixed 2026-11-04, before any measurement).**

*Conditions:* card 1, release, safe image, every tile (`TT_TILES=all`), and the 1.7B with BF16 weights and a
512-position cache. Each comparison is bracketed off / on / on / off.

0. **Void** if the audit fails. Every tile of a by-column weight must be held exactly once, and the GDDR it takes must
   be within one column a channel of the interleaved bytes.
1. **Blocks are killed** if either of these holds:
   - an eager product over a by-column weight is more than 5% slower (median) at any `m` = 64 / 256 / 512;
   - prefill at 128, 256 or 7,992 tokens is more than 2% below both brackets.
2. **Decode is killed** if the step at 32 / 256 tokens does not gain at least half of Phase B's 1.1 / 1.3 ms, that is,
   if it is slower than ~10.45 / 11.55 ms against ~11.0 / 12.2.
3. **The outcome.**
   - Neither killed: the default for stored weights.
   - Blocks killed only: opt-in, its prefill cost named.
   - Decode killed: close the form, and go to item 5's copy-free levers.
4. **Brackets that disagree:** the result is reported as not reproducible, and nothing is tuned.

**Prediction.** Eager products and prefill within ±2%, and decode at or below Phase B's 9.9 / 10.9 ms.

**Order.**
1. The encoding, with host tests against a read-a-tile oracle.
2. The firmware.
3. A ttsim gate (`step268_column_weights`) with a negative control (the column-to-channel map rotated), then that gate
   alone on card 0.
4. The eager bound (`silicon_bench_columns`, card 1; rule 1's cheap half, and the decode-shaped `m = 1` product).
5. The load-time path (`TT_WEIGHT_COLUMNS=1`).
6. The model bound.

### C1 as built (2026-11-04)

- **The encoding** (`tt_isa::dm::record`).
  - `TensorRef::col_tiles`, written as `REF_BY_COLUMN` in word 4 with the length in word 5.
    - A `GATHER`'s by-column `A` or `B` read a tile at a time is dispatched once a record to one out-of-line walk,
      `record::walk_by_column` (untransposed or transposed; a walk past a row's last column goes on at the next row, as
      the tile numbers do, and a row past the column's length is refused). `TensorRef::tile`, `Cursor` and the
      read-ahead cursor stay the interleaved code, byte for byte (first built with a by-column branch in them; that
      cost every model 2-3%, below).
  - `REF_L1_BY_COLUMN` on a `GATHER`'s `B`: `BSpan::of_columns`, one run a column on its own channel, in 16 KiB pieces
    taken from the columns in turn. `PrefetchCursor` gives the same reads.
  - Refusals, with a new code, `error::LAYOUT`:
    - a by-column reference in any record but a `GATHER`;
    - the L1 order by column over an interleaved, transposed or resident `B`, or on `A`;
    - `REF_LAID_FOR_READER` beside a by-column layout;
    - a read ahead of a by-column `B` read a tile at a time (the host never asks for one: `tile_op::ahead::split`).
  - All of these are checked before anything is read.
- **The firmware** needs no change of its own: the B mover runs the shared `expand_reads` and `PrefetchCursor`. Clippy
  is clean in every feature set; the B image's room is below.
- **The host** (`tt_kernels::tensor`).
  - `Layout::{Interleaved, ByColumn}` and `DramAlloc::alloc_laid`. A by-column grid takes the interleaved grid's
    regions, `cols.div_ceil(n) * rows` slots a channel.
  - `Placement::slot` and `tensor_ref` follow the layout. Every host path that addresses through `slot()` follows too:
    BF16 and BFP compression, widening and downloads, BF16 uploads, and `write_images` / `download_any`, tile by tile.
  - The DMA's records take interleaved tensors only, so a by-column tensor goes through its slots.
  - A one-row product over a by-column `B` sets `REF_L1_BY_COLUMN` (`tile_op::lower`). Its `Blocks` pairs name `B` at
    `(j * kt + kk)`, with a program variant of their own (`0x100`), so the sum is in the same order.
  - The two-copy relayout skips a by-column weight.
- **Load time.** `Session::set_column_weights` (`TT_WEIGHT_COLUMNS`, on by default since the verdict; `=0` off) lays
  what `store_bf16`, `store_bfp` (now burn-tt's BFP storage cast) and `upload_bf16_weight` store. X9's two-copy form
  (`TT_WEIGHT_LAYOUT=1`) takes precedence: with it on, weights are stored interleaved for it to relay. F32 weights
  (no storage cast) stay interleaved, a follow-up.
- **Gates.**
  - **Host.** Four `tt-isa` tests against the layout's definition:
    - the reference and its walk;
    - a block a tile at a time;
    - the decode read in pieces, for FP32, BF16 and BFP8, a stream's chunk and a view's columns;
    - the refusals, each other record against its interleaved twin.

    Four mutations were each caught: the channel, the column step, the span's channel, and the span's row offset.
    A `tt-kernels` placement test checks every tile in a slot of its own, on a full and a harvested grid.
  - **`step268_column_weights`** (ttsim in the safe, `fast-reads` and `fast-reads` + `encoded-guard` images; card 0
    `1793760854`, every tile):
    - one-row products (`down`'s and `gate|up`'s schedules), eager and three replays, and blocks at `m` = 64 / 40,
      each bit for bit against the interleaved weight, for BF16 and BFP8;
    - the stored bits and the widened tensor equal;
    - the control: names rotated a channel give other bits.

    A mutation that keeps the gather's order but drops the pairs' fails the one-row test.
  - `step172` and `step173` pass in every read mode. `step267` (the two-copy form) is unchanged.
  - **Card 0 with `TT_WEIGHT_COLUMNS=1`** (`1793760979`): `step188`'s traced generation against the eager one,
    `step234`, `step240` and `step187`, 18 of 18, bit for bit.

### C1 results (2026-11-04, card 1)

**Eager products** (`silicon_bench_columns`, `1793760911` and `1793760944`; medians of 7 host-timed products, ABBA in
one process, each product bit for bit against the first interleaved one).

| weight | `m` = 1 | `m` = 64 | `m` = 256 | `m` = 512 |
|---|--:|--:|--:|--:|
| `qkv` `[2048, 6144]` | 1,320 → 1,208 (-8.5%); 1,309 → 1,217 (-7.1%) | -0.0% / +0.1% | +0.5% / +0.5% | -0.3% / -0.6% |
| `o` `[2048, 2048]` | 829 → 777 (-6.3%); 811 → 775 (-4.4%) | -0.8% / -0.1% | +0.3% / +0.6% | +0.5% / +0.4% |
| `gate\|up` `[2048, 16384]` | 2,400 → 2,197 (-8.5%); 2,338 → 2,191 (-6.3%) | -0.3% / -0.3% | +0.3% / +0.6% | +1.9% / -0.3% |
| `down` `[8192, 2048]` | 2,946 → 2,700 (-8.3%); 2,893 → 2,705 (-6.5%) | +0.1% / -0.0% | +0.1% / -0.0% | -0.3% / -0.3% |

- **Rule 1, eager: not killed.** The block products are within -0.8% to +1.9% of interleaved, against 5%.
- **GDDR:** each weight takes exactly the interleaved bytes (3,244,032 / 1,081,344 / 8,650,752 / 4,325,376). Every
  `Nt` is a multiple of 8, so there is no rounding.
- **The one-row products**, eager, are 4-8.5% faster, from their columns read in pieces.

**The model, first build: decode killed, and the meter found why.**

Conditions: `traced_generation_stream_ceiling` with `TT_TILES=all`, prompts of 32 / 128 / 256 tokens, and the 1.7B with
BF16 weights.

| run | decode step at 32 / 128 / 256 tokens, ms | prefill, tokens/s |
|---|--:|--:|
| off (`1793761260`) | 11.261 / 11.747 / 12.420 | 1,065 / 2,092 / 2,033 |
| by column, read-ahead cap 2 (`1793761504`) | 11.784 / 12.341 / 12.967 | 1,054 / 2,072 / 2,018 |
| by column, read-ahead cap 2 (`1793761723`) | 11.768 / 12.345 / 12.977 | 1,060 / 2,081 / 2,026 |
| off (`1793761943`) | 11.213 / 11.743 / 12.420 | 1,067 / 2,091 / 2,035 |
| off (`1793762211`) | 11.257 / 11.778 / 12.450 | 1,066 / 2,090 / 2,034 |
| by column, `TT_PREFETCH_CAP=1` (`1793762453`) | 11.114 / 11.583 / 12.174 | 1,059 / 2,080 / 2,025 |
| by column, `TT_PREFETCH_CAP=1` (`1793762670`) | 11.156 / 11.573 / 12.174 | 1,053 / 2,073 / 2,017 |
| off (`1793762886`) | 11.216 / 11.714 / 12.372 | 1,072 / 2,110 / 2,052 |

- **The first build left out Phase B's read-ahead cap of 1.** Its 16 KiB reads ahead at the mover's cap of 2 slowed
  the small ops behind them, as in Phase B: decode was 4.7% slower.
- **At cap 1 the gain was 1-2%,** where rule 2 wants about 7%. As built, rule 2's kill fired.
- **The profiled pair** (`--profile`, `TT_PROFILE_DETAIL=pipeline`, 32 tokens; `Step(64)`, each product's slowest
  unit, us, the median over layers):

  | product | off (`1793763203`) | by column, column after column (`1793763811`) | by column, pieces in turn (`1793765738`) |
  |---|--:|--:|--:|
  | `qkv` | 36.3 | 18.7 | 18.5 |
  | `o` | 17.7 | 12.6 | 12.6 |
  | `gate\|up` | 380.6 | **418.2** | 350.8 |
  | `down` | 495.9 | 365.4 | 359.1 |
  | head | 3,559.5 | **3,972.3** | 3,542.0 |

  (The work figures sum each unit's batches, so they are not comparable with Phase B's table; the ratios are.)
- **What was wrong: a channel hotspot.** `qkv`, `o` and `down` gained what Phase B measured, but the wide-run 120-unit
  products lost.
  - A unit's run of `c` columns starts at column `u * c` and sits on consecutive channels.
  - The span read a whole column before the next, so every unit in step read channel `(u * c) % 8`. At four columns a
    run that is channels 0 and 4 only.
  - Phase B's regions put each unit on its own channel, `u % 8`.
- **The fix is the read order only:** the span now reads its columns' pieces in turn (piece 0 of each column, then
  piece 1, ...). It reads the same bytes into the same slots, so the bits are unchanged.
  - Re-gated on ttsim in every read mode and on card 0 (`1793764546`), with the per-read channel check pinning the
    order.
  - The cap of 1 is now built in: `set_column_weights` lowers a read ahead's cap as `set_weight_layout` does.
- **This is a defect found by the meter, not a parameter tuned to the rule.** The rule is applied to the built form
  below, with both reported.

**The model, the built form** (by column, pieces in turn, read-ahead cap 1 by the layout).

| run | decode step at 32 / 128 / 256 tokens, ms | prefill, tokens/s | first build, s |
|---|--:|--:|--:|
| off (`1793764793`) | 11.256 / 11.747 / 12.423 | 1,064 / 2,091 / 2,035 | 90.5 |
| **by column** (`1793765041`) | **10.101 / 10.628 / 11.266** | 1,064 / 2,085 / 2,030 | 75.1 |
| **by column** (`1793765257`) | **10.066 / 10.590 / 11.232** | 1,065 / 2,088 / 2,032 | 75.1 |
| off (`1793765473`) | 11.235 / 11.768 / 12.442 | 1,071 / 2,105 / 2,048 | 89.6 |

- **Decode is 10.4% / 9.7% faster** at 32 / 256 tokens: 11.25 → 10.08 and 12.43 → 11.25 ms, 99 tokens/s. Phase B's
  two copies gave 10.7%.
- **Rule 2, decode: not killed.** 10.08 / 11.25 ms is within the ≤ 10.45 / 11.55 bar.
- **Rule 1, prefill:** 128 and 256 tokens are at most 0.3% below the brackets, against 2%. The 7,992-token prefill is
  below.

- **GDDR:** 26.754 / 26.75 GB free, against 26.75 with the default. The fullest channel is 3.344 GB free either way.
  The weights are held once.
- **Builds are 17% faster** (90 → 75 s first, 39.5 → 31.3 s second): the decode lists are shorter.

**The long prompt** (7,992 tokens at chunks of 512, an 8,192-position cache, one sample, 4 new tokens; the runner's
limit raised to 2,400 s, since the first build alone takes ~12 minutes and the default 900 s cut off the first attempt,
`1793766347`-`1793769051`).

| run | prefill, ms | decode step after 7,992, ms | first / second build, s |
|---|--:|--:|--:|
| off (`1793769996`) | 5,956.0 | 47.94 | 720.5 / 191.6 |
| **by column** (`1793771014`) | 5,958.4 | **46.54** | 680.9 / 174.4 |
| **by column** (`1793771974`) | 5,959.1 | **46.60** | 679.3 / 174.1 |
| off (`1793772933`) | 5,953.7 | 47.99 | 717.8 / 192.2 |

- **Rule 1, prefill at 7,992: not killed.** 0.05% slower, against 2%. The prefill's blocks (`m` = 512) read the
  layout a tile at a time, as the eager bound showed.
- **Decode after 7,992 tokens is 2.9% faster**: the products are a smaller share of a step whose attention reads
  8,192 positions.
- **GDDR:** 16.18 GB free either way; the fullest channel 2.02 GB.

**The other models** (card 1, release, every tile; off / by column / by column / off, each a process of its own).

| model | off | by column | off again |
|---|--:|--:|--:|
| Qwen3.5-4B decode, 32 / 256 tokens, ms (`1793773961`-`1793776432`) | 65.48 / 65.84 | **63.41 / 63.84**, **63.45 / 63.84** | 65.46 / 65.87 |
| Qwen3.5-4B prefill, 32 / 256 tokens, ms | 466.0 / 3,730.5 | 466.1 / 3,729.7, 466.2 / 3,729.1 | 466.1 / 3,730.3 |
| Qwen3.5-4B build, s | 614.7 | 584.2, 581.9 | 611.0 |
| SmolLM2-135M decode, 32 / 256 tokens, ms (`1793777272`-`1793777433`) | 7.190 / 8.484 | 7.239 / 8.451, 7.143 / 8.386 | 7.228 / 8.472 |
| ResNet-18 forward, traced, ms (`1793777487`-`1793777688`) | 12.97 | 12.97, 12.98 | 13.01 |
| ViT-Tiny forward, traced, ms (`1793777755`-`1793777800`) | 7.97 | 7.94, 8.00 | 7.93 |

- **Qwen3.5-4B decodes 3.1% faster** (2.0 ms a step; X9's two copies gave 1.2-1.3 ms), with prefill unchanged and
  its build 30 s shorter.
- **The 135M, ResNet-18 and ViT-Tiny are unchanged.** The 135M's products are too small to be read-bound; the image
  models store no weight (F32, interleaved).

**The verdict (rules 0-3): both halves pass, so the layout becomes the default for stored weights.**
- Rule 0: every tile held once (`a_grid_laid_by_column_holds_each_tile_once_where_the_layout_puts_it`,
  `a_weight_laid_by_column_holds_what_was_stored`), and GDDR taken equals the interleaved bytes.
- Rule 1: eager blocks within -0.8% to +1.9% (bar 5%); prefill at 128 / 256 / 7,992 within 0.3% (bar 2%).
- Rule 2: decode 10.08 / 11.25 ms at 32 / 256 tokens (bar 10.45 / 11.55).
- Rule 4: every bracket agrees with its twin within 0.4%.

**The off path paid for the layout: found and fixed (2026-11-04).** Every "off" run above was 2-3% slower than the
day before on the same card (the 1.7B's decode 11.25 ms against 10.96, `1793723839`), and so were ResNet-18 and ViT-Tiny,
which store no weight. A bracket of the committed `1ff07dc` against this tree with the layout off, on card 1, put it
in the code, not the card:

| | `1ff07dc` (`1793778018`, `1793778978`) | this tree, off (`1793778338`, `1793778662`) |
|---|--:|--:|
| 1.7B decode, 32 / 128 / 256 tokens, ms | 11.00 / 11.50 / 12.18 | 11.22 / 11.77 / 12.44 |
| 1.7B prefill, 32 / 128 / 256 tokens, tokens/s | 1,086 / 2,136 / 2,076 | 1,065 / 2,089 / 2,035 |
| ResNet-18 forward, traced, ms (`1793777945`, `1793778904` against `1793778256`, `1793778581`) | 12.54 / 12.56 | 12.93 / 12.99 |
| ViT-Tiny forward, traced, ms | 7.66 / 7.49 | 7.74 / 7.74 |

- **Where:** the movers' per-tile loops. A hybrid (the baseline with only the new record code) was as slow; the
  baseline with only the new reference fields and their per-record decoding was not (12.53 / 12.55 and 7.65 / 7.70
  ms). The profile (`1793780198` against `1793780286`) showed every op slower, gathers 2-4% and fused eltwise groups
  5-8%, with the compute cores' push cycles a word up 8-12% while their images were unchanged: the movers' larger,
  branchier per-tile code, which shares L1 with the compute cores' instruction fetches.
- **Moving the by-column branches out of line was not enough** (ResNet-18 12.69 / 12.90 ms, ViT-Tiny 7.92 / 7.91):
  any test in the per-tile path cost.
- **The fix:** `TensorRef::tile`, `Cursor` and the read-ahead cursor are the baseline's code again, and a by-column
  operand is dispatched once a record to one out-of-line walk (`record::walk_by_column`): a `GATHER`'s `A` or `B`,
  untransposed (a row a walk, a step a tile, no division) or transposed. A read ahead of a by-column `B` read a tile at
  a time is refused by the card and never asked for by the host (`tile_op::ahead::split`); a by-column `B` read by
  column keeps its span. Off, ResNet-18 12.56 / 12.58 ms and ViT-Tiny 7.50 / 7.56 against the baseline's 12.56 /
  12.56 and 7.68 / 7.67 (`1793781126`-`1793781431`).
- **Found on the way:** a stored weight can be a product's `A` (`step241`'s every pair of edge formats), which the
  first build read by column through the generic cursor and the fix first refused; the walk reads it.
- **It came back, and the fix had to go further** (the first full run on card 1, `1793788972`-`1793797887`). With `A`
  walked too, the dispatch sat inside the `GATHER` arm, between its loops, and the tree was again slower than the
  baseline whatever the flag: ViT-Tiny 7.80-7.89 ms against 7.48-7.72 in eight runs each, ResNet-18 up to 1.5%, the
  1.7B's off runs 1.0-1.4%. The profile (`1793799184` against `1793799228`) showed the same mark, the fused eltwise
  groups' push cycles a word up 10-12%. The read-ahead cap was not it (`TT_PREFETCH_CAP=2` changed nothing,
  `1793797995`-`1793798606`).
  - **The form that held:** the `GATHER` arm is the baseline's code byte for byte; at its top, a record with either
    operand laid by column, or a mark of the L1 order, leaves once for `record::gather_by_column` (cold, out of line),
    which checks `B`'s marks before any read and walks each operand in its own layout (`walk_by_column` steps an
    interleaved operand as well, a channel at a time). `run_record` now starts at the baseline's address with 112
    bytes more. ViT-Tiny 7.680 / 7.681 ms against the baseline's 7.612 / 7.719, ResNet-18 12.545 / 12.547 against
    12.358 / 12.545 (`1793799852`-`1793800085`).
  - A host test pins the interleaved operand's reads in the cold path to the arm's (each of three mutations caught).
- **The B image's room.** The cold path is 3.6 KB, and the `profile` image built with `fast-reads` no longer fit
  `IMAGE_MAX` (1,580 bytes over; `encoded-guard` more). Two changes:
  - **Smaller code:** one decode body for both reference kinds, the span's flag a `u32` (a `bool` was its `Option`'s
    niche and made every copy of a span byte by byte), the slots passed in, one loop body in the walk.
  - **The role slots gave B 1,920 bytes** (`tt_isa::l1::roles::SLOT` 10,240 -> 9,600; the largest role image is
    8,884). B's room left: 4,584 bytes (safe), 2,420 (`encoded-guard`), 220 (`profile` + `fast-reads` +
    `encoded-guard`). The next firmware growth must find its room.
- **Weights whose columns do not fill the channels stay interleaved** (`tensor::Layout::for_weight`). By column, every
  channel's region holds `cols.div_ceil(n)` columns, and the 135M's 18-column weights took 38 MB more (33.235 -> 33.197
  GB free, `1793797728` against `1793797781`): GDDR that grows with the model. A weight is now laid by column only
  where that takes no more slots a channel than interleaved, which every weight of the 1.7B and Qwen3.5-4B does.
- **The rules are applied again to the final code** (below).

### C1 final: the rules on the code as built (2026-11-04, card 1)

The tree as it will be committed, against the committed baseline `1ff07dc` built in its own worktree, card 1, release,
safe image, every tile (`TT_TILES=all`), each run a process of its own. Card 0 had passed the gate list with the
layout on by default (71 of 71, `1793801167`) and `--smoke` (1,283 of 1,283, `1793805483`) first.

**The off path** (models that store no weight; baseline / tree / tree / baseline).

| model | `1ff07dc` | tree (default, on) | `1ff07dc` again |
|---|--:|--:|--:|
| ResNet-18 forward, traced, ms (`1793807243`-`1793807427`) | 12.516 | 12.577, 12.576 | 12.553 |
| ViT-Tiny forward, traced, ms (`1793807486`-`1793807528`) | 7.691 | 7.507, 7.704 | 7.663 |

The slowdown of the first full run is gone: ResNet-18 is within 0.3% of its baseline, and ViT-Tiny sits inside its
brackets.

**Eager products** (`silicon_bench_columns`, `1793807545`; interleaved → by column, medians, ABBA in one process, bit
for bit).

| weight | `m` = 1 | `m` = 64 | `m` = 256 | `m` = 512 |
|---|--:|--:|--:|--:|
| `qkv` `[2048, 6144]` | 1,239 → 1,164 us (-6.1%) | +0.1% | -1.1% | -0.8% |
| `o` `[2048, 2048]` | 808 → 784 us (-2.9%) | +1.4% | +0.9% | -0.6% |
| `gate\|up` `[2048, 16384]` | 2,294 → 2,205 us (-3.9%) | -0.1% | -0.0% | -2.5% |
| `down` `[8192, 2048]` | 2,777 → 2,697 us (-2.9%) | +0.3% | +0.1% | -0.3% |

GDDR: each weight takes the interleaved bytes exactly (3,244,032 / 1,081,344 / 8,650,752 / 4,325,376).

**The 1.7B** (`traced_generation_stream_ceiling`, BF16 weights, a 512-position cache; baseline / off / on / on / off /
baseline).

| run | decode step at 32 / 128 / 256 tokens, ms | prefill, tokens/s | first / second build, s | GDDR free, GB |
|---|--:|--:|--:|--:|
| `1ff07dc` (`1793807556`) | 11.004 / 11.527 / 12.179 | 1,094 / 2,152 / 2,092 | 86.7 / 37.6 | 26.760 |
| off (`1793807793`) | 11.054 / 11.605 / 12.247 | 1,071 / 2,109 / 2,052 | 85.1 / 37.5 | 26.752 |
| **on** (`1793808030`) | **10.069 / 10.568 / 11.213** | 1,096 / 2,155 / 2,095 | 74.8 / 31.1 | 26.760 |
| **on** (`1793808246`) | **10.056 / 10.549 / 11.207** | 1,100 / 2,161 / 2,098 | 74.0 / 30.9 | 26.760 |
| off (`1793808461`) | 11.090 / 11.571 / 12.244 | 1,087 / 2,139 / 2,082 | 85.0 / 37.4 | 26.750 |
| `1ff07dc` (`1793808695`) | 10.947 / 11.476 / 12.151 | 1,075 / 2,118 / 2,056 | 86.2 / 37.6 | 26.752 |

- **Decode: 10.06 / 11.21 ms** at 32 / 256 tokens, 99.4 tokens/s. That is 9.1% / 8.5% faster than the layout off
  and 8.4% / 7.9% faster than the baseline.
- **Prefill** with the layout on is at or above every bracket at every length.
- **Builds are 13-18% shorter.**
- **GDDR:** the fullest channel is 3.344 GB free with the layout on and 3.343 GB off.
- **The opt-out pays about 0.5%:** off, decode is 11.05-11.09 / 12.24-12.25 ms against the baseline's 10.95-11.00 /
  12.15-12.18. It is the cold dispatch's test at the top of every `GATHER`. With the layout on, every run beats the
  baseline.

**The long prompt** (7,992 tokens at chunks of 512, an 8,192-position cache, 4 new tokens; off / on / on / off).

| run | prefill, ms | decode step after 7,992, ms | first / second build, s |
|---|--:|--:|--:|
| off (`1793808930`) | 5,909.1 | 46.99 | 681.7 / 183.7 |
| **on** (`1793809900`) | 5,897.3 | **45.85** | 663.5 / 171.0 |
| **on** (`1793810840`) | 5,898.2 | **45.94** | 646.3 / 171.3 |
| off (`1793811764`) | 5,908.8 | 47.08 | 682.0 / 183.5 |

GDDR is 16.18 GB free either way, and the fullest channel 2.02 GB.

**The other models** (off / on / on / off).

| model | off | on | off again |
|---|--:|--:|--:|
| Qwen3.5-4B decode, 32 / 256 tokens, ms (`1793812735`-`1793815192`) | 64.22 / 64.58 | **62.88 / 63.24**, **62.89 / 63.24** | 64.20 / 64.59 |
| Qwen3.5-4B prefill, 32 / 256 tokens, ms | 464.0 / 3,712.9 | 464.0 / 3,712.3, 464.4 / 3,712.9 | 464.3 / 3,713.4 |
| Qwen3.5-4B build, s | 607.6 | 582.6, 582.4 | 598.6 |
| SmolLM2-135M decode, 32 / 256 tokens, ms (`1793816020`-`1793816179`) | 7.066 / 8.294 | 7.032 / 8.250, 7.029 / 8.257 | 7.060 / 8.254 |

- **Qwen3.5-4B decodes 2.1% faster** (1.3 ms a step), with prefill unchanged.
- **The 135M is unchanged within 0.5%.** Its 18-column weights stay interleaved (`Layout::for_weight`), its GDDR is
  33.235 GB free either way, and its products are too small to be read-bound.

**The verdict (rules 0-4, final code): both halves pass. The layout is the default for stored weights**
(`TT_WEIGHT_COLUMNS=0` opts out).
- **Rule 0:** every tile is held once. GDDR taken equals the interleaved bytes for every weight laid by column, and the
  models' free GDDR is unchanged.
- **Rule 1:** eager blocks are within -2.5% to +1.4% (bar 5%). Prefill at 128 / 256 / 7,992 tokens is at or above
  both brackets (bar 2% below).
- **Rule 2:** decode is 10.06 / 11.21 ms (bar 10.45 / 11.55), within 2% of Phase B's two copies (9.90 / 10.94), from
  one.
- **Rule 4:** each run agrees with its twin within 0.2% (ViT-Tiny's within its run-to-run noise).

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
- [x] The default: **on 2026-11-03** by the owner's call (reverted the same day, below), for ~3.4 GB more GDDR on the 1.7B (`TT_WEIGHT_LAYOUT=0`
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
- [x] **One copy instead of two: bounded and closed in this form** (2026-11-03, Phase C0 above; card 1
  `1793682506`, `1793682568`-`1793683256`). The probe was bit for bit, but eager products over the regions are
  1.34-3.38x slower and prefill over one-column runs is 4-13% slower, from ~4x the list entries. Reopen only with a
  record that cuts a region's rows into a block (an L1 pitch: a new encoding) and one deal a weight.
  The entry as first written: X9 as the weight's only storage. Prefill and
  eager products read the same layout a column run at a time (a decode unit's share is whole runs, so one `GATHER` a
  run, no new encoding); the tied head's token lookups, in-place writes and downloads need their own paths; the split
  is fixed at load from the decode deal. **Bound first:** prefill and an eager `m > 1` product over the relaid layout,
  card 1, against today's.
- [x] **The default reverted** (2026-11-03, owner): opt-in again (`TT_WEIGHT_LAYOUT=1`), because the second copy
  grows with the weights (above).
- [x] **The scalable form** (Phase C1, by-column weights, 2026-11-04; the default for stored weights): one stored layout, made at load time,
  each column contiguous in one channel. Decode reads it in large pieces; prefill and eager blocks read it at today's
  entries. No deal is fixed. Bound against the one-copy default on prefill and eager as well as decode (C1's rules).
  - [x] Encoding (`REF_BY_COLUMN`, `REF_L1_BY_COLUMN`), host tests, mutations watched failing
  - [x] Firmware (no change of its own: the shared expansion), clippy in every feature set
  - [x] `step268_column_weights` on ttsim (every read mode), then card 0 alone (`1793760854`)
  - [x] Eager bound on card 1 (`silicon_bench_columns`, `1793760911`, `1793760944`): not killed
  - [x] Load time (`TT_WEIGHT_COLUMNS=1`), other readers supported or refused by name (`error::LAYOUT`)
  - [x] Model bound on card 1; the verdict by the rules (C1 final, `1793807243`-`1793816179`): both halves pass,
    the default; card 0's gate list and `--smoke` with it on (`1793801167`, `1793805483`)
  - [ ] F32 weights (no `with_storage` today) stay interleaved: a follow-up if an F32-stored model is read-bound
- [ ] Item 5's copy-free levers (bank-affine placement, atomic column claims, a dedicated prefetcher), ranked against
  the scalable form after its bound. **Ranked after C1 (2026-11-04):** C1 took nearly all of 2b's gain (10.06 against
  two copies' 9.90 ms), so these levers now compete for what is left beyond it, not for 2b's ~10%.
  - **First: bank-affine placement. Bounded and killed 2026-11-06** (H17, card 1, `1793988029`): 2-3 readers a
    channel at 16 KiB read `down`'s bytes 1.04x and `o`'s 1.06x faster than C1's deal, under 0.1 ms a step.
  - **Second: `gate|up`'s evenness (below).** Its slowest unit sets the product.
  - **Third: atomic claims and a dedicated prefetcher.** They need new firmware: B had 4.6 KB of room (safe) and 220
    bytes in the tightest build, so the firmware diet (2026-11-06) made room first (`cargo xtask fw-size`).
- [ ] `gate|up`'s evenness: its mean unit is faster but not its slowest. A deal that sizes each unit's share to the rate
  it gets, measured with the pipeline detail (X10). Layout-independent.
- Retired 2026-11-06 with the two-copy form itself (firmware diet; code at `61604b9`; `TT_WEIGHT_LAYOUT` refused by
  name). These refined it:
  - relay only `down` and `qkv` (Phase A's gains), roughly half the extra GDDR;
  - regions that yield: on a refused allocation, free relaid regions no live trace reads before failing it, and drop
    them on `set_weight_layout(false)`;
  - two-copy regions made at load time, not at a capture's first sight.
- Withdrawn 2026-11-06, the parked use: prefill and decode on different cards, each holding one layout (needs the decode card's other readers
  to read the regions, C0's gap).
