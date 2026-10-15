# Trace size, and a decode step that costs the context, not the cache

Roadmap Track 2, item 6, step 4, and "Trace size in GDDR". Written 2026-10-15 against `0d36bdb`.
Owner direction: every fix general to decoders, not SmolLM2-shaped.

## Why

- **Trace size.** SmolLM2's step and its 64/128/256 chunks (`TracedGenerator`, 512-position cache, 120 units) held
  **3.75 GB** of trace streams in GDDR (run `1792072109`). A 1B model's F32 weights alone take 4-5 GB of a card's 32.
- **Decode cost.** A traced decode step attends to its whole cache, so it costs the cache's length, not the context:
  26.46 ms over 64 positions, 28.49 over 320, 32.28 over 512, against burn-flex's 26.3-31.0.

## Phases

### A0. Measure the trace size (done)

- [x] `silicon_bench_models::traced_generation` reports each trace's entry bytes (`UnitFootprint::stream.entries * 32`)
  beside its GDDR bytes, and its section count.

Card 1, run `1792074320`:

| Trace | GDDR | Entry bytes | Ratio | Sections | Allocated a section |
|---|--:|--:|--:|--:|--:|
| Step | 452.8 MB | 84.2 MB | 5.4x | 103,702 | 4,361 B |
| `Chunk(64)` | 777.6 MB | 104.4 MB | 7.4x | 180,060 | 4,316 B |
| `Chunk(128)` | 1,063.5 MB | 144.1 MB | 7.4x | 243,476 | 4,366 B |
| `Chunk(256)` | 1,459.7 MB | 212.2 MB | 6.9x | 334,100 | 4,368 B |

**Why:** each `PAIR_CALL` stored its reader and writer sections in an allocation of their own, at least one
`TILE_SLOT` (4,160 bytes), and a section averages ~810 bytes (~25 entries), not the few entries first assumed. The
gain is therefore ~6.6x, not 30-60x.

### A1. One region per unit per trace (done)

- [x] **Sections wait on the host during a capture.** They are only read on replay; the capture's own run executes
  the live lists. `UnitCapture::pending_sections` holds them, and `pair_calls` records where each `PAIR_CALL` sits.
- [x] **`Session::store_stream` lays out one region a unit.** The stream comes first, then each section from a
  64-byte boundary: a mover's chunk read needs `tt_isa::dram::ALIGN`, not just the 32 the entry decode checks. It
  makes one `alloc_on` and one `dram_write`, and fills in each `PAIR_CALL`'s offsets. `UnitTrace.regions` is gone,
  since the stream's placement is the region.
- [x] **Gates:**
  - `step174_trace_footprint` bounds `gddr_bytes` by the entries plus 64 bytes a part, rounded to one slot. It was
    watched failing on the old allocation (2 of 3 tests).
  - Unchanged and passing on ttsim: `step60_streaming`, `step147_mesh_trace`, `step171`, `step174`, `step179`,
    `step185`, `step186`, `step187` and `step12_mnist` (e2e).
  - Silicon: `step174` `1792076071`, `step60` `1792076076` (18), `step171` `1792076097`, `step179` `1792076107`,
    `step185` `1792076112`, `step186` `1792076145`, `step187` `1792076147`, card 0; `step147` on both cards
    `1792076169`.

**Result** (card 1, run `1792076190`):

- the four traces' streams are **563 MB, was 3,754** (each now 1.0x its entry bytes);
- the driver's build is **18.8 s, was 44.9** (one GDDR write a unit, not one a section);
- replay times are unchanged.

### B0. Where a step's length cost goes (done)

`smollm2_replay_profile` with `TT_PROFILE_MAX_LEN` (new), one layer, card 1, runs `1792076366` (64 positions) and
`1792076371` (512). This is unit 0's device time by Burn op, in microseconds, for one layer plus the head and the
step's prologue:

| Label | 64 | 512 | Growth |
|---|--:|--:|--:|
| `float_cat` (`repeat_kv`) | 69.7 | 149.1 | **+79.4** |
| `attention/pv` | 53.7 | 85.8 | +32.1 |
| `float_expand` (the new `k`, `v` to the cache's shape) | 52.9 | 80.2 | +27.3 |
| `attention/qk` | 68.6 | 91.5 | +22.9 |
| `attention/softmax` | 80.2 | 99.8 | +19.6 |
| `int_expand`, `bool_expand`, `int_equal`, `int_greater` (the step's masks: once a step) | 299.0 | 328.6 | +29.6 |
| `float_mask_where` (the cache write) | 18.0 | 25.9 | +7.9 |
| `(outside an op)` (the write-back `copy_into`s) | 8.9 | 12.1 | +3.2 |
| everything else | 981.4 | 977.6 | 0 |
| **total** | 1,632.4 | 1,855.6 | +223.2 |

What it decided:

- `repeat_kv` is the largest share, and costs 70 us a layer even at 64 positions (B1);
- attention's reads are next (B2);
- the write-back copy is negligible, so B2 cuts what attention reads and leaves the write and the write-back
  whole-cache: no copy into a view.

### B1. Grouped-query attention without `repeat_kv`

- [x] **The fold** (`llama.rs`, `Write::Fixed`): the query heads of a group are folded into the rows,
  `[1, heads, t, d] -> [1, kv_heads, groups t, d]`, which is `repeat_kv`'s order. Attention then reads the cache's
  own key/value heads, and the mask's rows are repeated per group once a step. The growing cache keeps `repeat_kv`
  and is the reference.
- [x] **Gates:** `step185` and `step187` pass bit for bit against the growing cache, on ttsim (SmolLM2 at all seven
  lengths, and the synthetic GQA 4:1 and multi-head shapes) and on card 0 (`1792078436`, `1792078467`).
- [x] **Profile at 512 positions after it** (`1792078810`), one layer: 1,855.6 -> 1,682.3 us. In that:
  - `float_cat` 149.1 -> 19.1 us (the two left are RoPE's);
  - `attention/qk` 91.5 -> 69.0 and `attention/pv` 85.8 -> 65.9: fewer repeated heads to gather and stack.

### B2. Steps that attend only to the positions in use

- [x] **`FixedCache::step_write_within(position, len)`** writes the whole cache. Its mask covers `0..len`, and
  `FixedWrite::attended` gives attention the `0..len` prefix (a view; read in place when `len` and `max_len` are
  multiples of 32).
- [x] **`FixedShapeDecoder::step` takes `len`.** `Shapes` holds the cache's length, the chunk sizes and the step
  lengths: 64, 128, ... below `max_len`, then `max_len`. `TracedGenerator` captures one step a length, and a decode
  at `position` replays the smallest length above it. `generate_fixed` follows the same shapes.
- [x] **`step187` new tests** (ttsim; card 0, `1792078467`):
  - `decode_steps_attend_only_to_the_positions_in_use`: generations crossing from the 32- to the 64-position step
    equal the growing cache and the whole-cache schedule (a Llama, and the GPT decoder);
  - `a_step_attending_to_too_few_positions_diverges`: the 32-position step at position 36 differs from the
    teacher-forced reference, which the right step equals.
- [x] **Bench `traced_generation`** with step lengths (`1792078489`) and without (`TT_GEN_STEPS=`, `1792078651`).
  Both run on card 1 at a 512-position cache, with burn-flex in the same run:

| | Decode step (32 / 128 / 256-token prompt) | Prompt + 64 tokens | Streams | Program cache | Build |
|---|---|---|--:|--:|--:|
| Before B1 (`1792076190`) | 32.29 / 32.29 / 32.29 ms | 2,077 / 2,087 / 2,111 ms | 563 MB | 81.0 KB | 18.8 s |
| B1, the whole cache a step (`1792078651`) | 27.32 / 27.31 / 27.32 | 1,761 / 1,770 / 1,795 | 407 MB | 78.1 KB | 15.5 s |
| B1 and B2, steps of 64/128/256/512 (`1792078489`) | **26.13 / 26.60 / 27.31** | **1,688 / 1,726 / 1,795** | 517 MB | 89.3 KB | 18.7 s |
| burn-flex | 26.75 / 29.10 / 31.57 | 1,866 / 2,345 / 3,015 | | | |

The decode steps of these prompts run over positions 32-95, 128-191 and 256-319, so they attend to 64 or 128, 256
and 512 positions. The fold is most of the gain (5 ms a step); the step lengths add 1.2 ms on short contexts. Traced
generation is now faster than burn-flex at every prompt length: 0.90x, 0.74x and 0.60x of its time. The first token
also improved (35.1 / 46.0 / 69.3 ms, was 38.8 / 49.6 / 73.0), since the chunks fold too.

### B3. Deferred: a row write at a resident index

Per B0, the cache write's `expand` (+27 us a layer at 512) and `mask_where` (+8) are what still grow after B2: a step
over 512 positions is 1.2 ms slower than one over 64 (26.13 against 27.31 ms). A native row write at a device-held
position would replace both. It is a new mover record and a new op, worth about 1 ms a step at long contexts; it is
left for when longer contexts or a larger model make it a larger share.

## Status

**Done (2026-10-15):**
- A0, A1 (trace size 3,754 -> 563 MB, and 517 MB with four step lengths);
- B0, B1, B2 (decode at or below burn-flex at every length measured).

B3 is deferred. This plan moves to `completed-plans/` when the roadmap entries cite it.
