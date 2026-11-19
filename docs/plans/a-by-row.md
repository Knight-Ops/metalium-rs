# `A` laid by row, read a row a request

Roadmap W2's box "`A` laid by row" ([roadmap](roadmap.md)): a matmul's activation operand `A` is read by B's mover a
tile a request today, interleaved over the channels (tile `t` on channel `t % n`). A weight `B` is already stored by
column and read a column a request (`BSpan`, [decode-bytes-in-flight.md](decode-bytes-in-flight.md) C1, the
2026-11-14 log). This plan does the same for `A`: activations laid by row, so that a block's `A` strip is one run a row.

**Why:** at the 1.7B's 128-256-token prefill, `A`'s reads are the lever left in the products (P0 of the short-chunk box,
[log](../learnings/performance-log.md#w2-short-chunk-products-mapped-before-building-spreading-refuted-as-cost-the-lever-2026-11-17-card-0)):
576 -> 336 us a layer at 128 rows with them free.

**The layout** (`tt_isa::dm::record::TensorRef::by_row`, `REF_BY_ROW`): tile `t` is row `r = t / ct`, column
`c = t % ct`; row `r`'s `ct` tiles are consecutive slots on `channels[r % n]`, from slot `(r / n) * ct`. FP32 or BF16
slots (unlike by column, BF16 activations are the performance contract's). Only for tensors of at least `n` tile rows:
fewer would waste slots and put a tensor on few channels, and decode's one-row activations stay interleaved, so decode
is untouched by construction.

**Lessons carried from by column:** a test in the per-tile loops cost every model 2-3% (so each layout is dispatched
once a record, out of line); a unit's runs read one after another hot-spotted two channels (so a span reads its runs'
pieces in turn).

## Phases

- [x] **P0: the encoding and the bound** (2026-11-18).
  - `REF_BY_ROW` (word 4 bit 3), refused beside `REF_BY_COLUMN` or over no columns; BF16 slots allowed.
  - `BSpan::of_rows`: a by-row `A`'s rows as runs, the same addressing and pieces-in-turn read as `of_columns`
    (run `r` on `channels[r % n]`, `stride` a row's bytes, `off` the first column's). A `GATHER` reads an untransposed
    by-row `A` through it, out of line (`gather_a_rows`), before `B`; transposed, it takes `walk_by_column`, which
    steps a by-row tensor too. A read ahead refuses a by-row `B`; every record but a `GATHER` refuses a by-row tensor.
  - Host tests (`tt-isa`): the reference and the walk against the layout's definition (along a row, down a column);
    a gather's 64-byte units against a read-a-tile oracle (FP32, BF16, a view inside the grid, a ragged last row
    read whole), pieces in turn, one request a read, `B` unchanged; the refusals. Four mutations each caught: the
    span's row, its stride, the piece order, the walk's step.
  - `dm_b` (profile + encoded-guard, `cargo xtask fw-size`): 48,268 -> 49,584 bytes, room 2,640; `run_record`
    10,936 -> 11,200.
  - **The bound** (`silicon_bench_memory::mover_gather_rate`, profile image, card 1, `1795029525`, `1795029592`;
    BF16 tiles, cycles a tile's bytes):

    | strip | interleaved | by row |
    |---|--:|--:|
    | `A` alone, 7 rows of 4 tiles | 301.9 | 143.1 |
    | `A` beside `B` read a column a request | 191.3 | 115.8 |
    | `A` alone, 2 rows of 48 tiles (a model's row length) | 239.1 | **57.9** |

    Kill was "above ~130 cycles a tile": at a model's row length it is 57.9, 4.1x. With 181 cycles a tile saved (the
    bound assumed 212), the bound is ~2.1-3.1 ms of a 128-token chunk (8-12%).
- [x] **P1: every record arm accepts a by-row tensor** (2026-11-18). Not arm by arm: every arm addresses a tensor
  through three things, and each learned the layout.
  - `TensorRef::cursor`: a by-row tensor's cursor (`row_cursor`, once a row, out of line) is the interleaved step over
    one channel (`Cursor::k`, `Cursor::n` 1), so the per-tile step has no test; `tile_in` sends a by-row tensor to
    `tile_by_row` (out of line).
  - `READ_RUN` / `WRITE_RUN`'s plain run starts a new cursor where it wraps to the next row (inside the wrap branch).
  - Encoded reads need nothing: they encode entries after the expansion.
  - Host test `every_record_addresses_a_tensor_laid_by_row_as_its_interleaved_twin`: `SCATTER`, `READ_RUN` (across
    rows, broadcast down and across, read as it lies, column-major, transposed), `WRITE_RUN`, `SELECT_ROWS`,
    `SCATTER_ROWS` (face rows, a whole tile row), `FILL_PAD`, `PAD_WRITE`, FP32 and BF16, each entry's tile and byte
    equal to its interleaved twin's (each layout's inverse from its definition). Three mutations each caught.
  - Room at the tightest (profile + encoded-guard): `dm_b` 1,596 bytes, `dm_nc` 604 (from 2,640 / 1,560).
- [x] **P2: host placement** (2026-11-18; opt-in `TT_ACT_ROWS=1`, the gate's `=all`). Rule as built: grids of at
  least `n` tile rows **and two tile columns** (a one-column grid laid by row is interleaved, and its `[r, 1]` <->
  `[1, r]` view names another width). **The 128-token chunk's `A` (`[4, 64]` tiles) stays interleaved:** on 4 of 8
  channels its rows would carry ~68 GB/s a channel under the products' re-reads, past a channel's ~64; splitting a
  short row over several channels needs a per-tile test in the cursor (the 2-3% lesson). So this build reaches
  256-token and longer chunks (the 512 / 1,024 buckets). Also: a `GATHER`'s by-row `B` (the KV cache) and transposed
  `A` stay in line (the cursor and `tile_in` address the layout), not the out-of-line walk.
  **Defect found by P3's gate, not P2's tests:** `write_images` / `download_any` special-cased only by-column
  placements, so a by-row tensor's upload and download laid tiles interleaved, consistently with each other; the
  card's reads disagreed. Every non-interleaved placement now goes tile by tile.
- [ ] **P3: gates.** Done on ttsim: `step276_rows_activations` (eltwise, product with `A` by row and `B` by row, both
  reductions, a chained product, a row view, a transposed `A`; `m` 64 / 128 / 200 / 256; the session's rule and every
  grid; the rotated control) in the safe, `fast-reads` and `fast-reads` + `encoded-guard` images, with `step172` /
  `step173`. Left: card 0 (with the BF16 casts, silicon only), conformance under `TT_ACT_ROWS=all`, `step12_mnist`.
  Planned: A ttsim gate (safe, `fast-reads`, `fast-reads` + `encoded-guard`): every op class writing and
  reading a by-row tensor, products at `m` = 64 / 128 / 256 with ragged rows followed by a reduction and a matmul,
  bit for bit against interleaved; the rotated control gives other bits. Card 0. `step172` / `step173` in all three
  read modes, firmware lints with and without `profile`, `step12_mnist`, conformance.
- [ ] **P4: the model A/B** (ABBA by process, card 0, performance contract): the 1.7B's prefill at 32 / 128 / 256
  (and the bucket default's larger chunks), decode, ViT-Tiny, ResNet-18, Qwen3.5-4B's prefill. Kill: under 3% of a
  128-token chunk, or any default slower by more than 0.5%. Win: on by default (`TT_ACT_ROWS=0` off). Loss: a row in
  `failed-approaches.md`, the knob and the code deleted.
