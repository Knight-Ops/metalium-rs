# Decode host cost: copies as region moves, not element maps

Roadmap Track 2, item 3 ("The decode step's host cost (25x)"). Written 2026-10-14 against `0ba8d00`.
Status: **done (2026-10-14).** P0-P3 and P5 are done; P4 is deferred to roadmap Track 2 item 9 (attention), which is next.

## Why this item, and what it is not

Track 2's items are ordered by what the model ladder measured. Items 1 (row select) and 2 (chunk-free
convolution) are done, so item 3 is next. A SmolLM2-135M decode step is **623 ms against burn-flex's 26 ms**
(card 0, run `1791852097`, one tile, F32, 30 layers). The roadmap read this as host submission: 2,779
operations at about 220 us each. P0 below corrects that. On one tile the step mostly waits on m = 1 matmuls. On
all tiles it is 233 ms, and 61% of the server thread is `cat` built from a zero-fill and per-element
`slice_assign`s.

This plan makes the operations that SmolLM2 calls cheap, inside the backend. Rules for the scope:

- **The model code does not change.** `crates/tt-models/src/llama.rs` is ordinary Burn code (`Tensor::cat`,
  `slice`, `slice_assign`, `swap_dims`). Everything here has to help any Burn model that calls the same ops,
  and must never be specific to SmolLM2.
- **Not in scope:**
  - **A fused RoPE op.** RoPE's negate, multiply and add are a job for the general chain fuser (B13b).
  - **A decode trace.** That needs trace segmentation, which is a separate roadmap item. This plan removes
    two obstacles to one (see "Hand-off" below) and leaves the rest.
  - **The BF16 lookup.** That is item 5.

## What a decode step does today (verified in the code)

Per layer, with `t = 1` and a cache length of `L` (`llama.rs:88-232`):

| Model code | Burn lowering | burn-tt today |
|---|---|---|
| `Rope::apply` on q and k: two half `slice`s, `neg`, `cat(.., 3)` | `float_slice` x2, `float_cat` -> `float_empty` + 2 `float_slice_assign` | 4-D slices take `sliced_native` (`ops.rs:565`): an element repack. The cat is a zero-fill plus two whole-destination repacks. |
| `repeat_kv` on keys and values: 3 head `slice`s, then a 9-way `cat(.., 1)` | `float_slice` x3, `float_empty`, 9 `float_slice_assign` | 9 repacks, each of the **whole** `[1, 9, L, 64]` destination |
| Cache write: `cache.k.clone().slice_assign(..)` for k and for v | `float_slice_assign` | Repacks the whole `[1, 3, max_len, 64]` cache to write one row |
| `heads()`: `reshape` then `swap_dims(1, 2)` | `float_reshape`, `float_swap_dims` | `swap_dims` calls `to_dram()` first (`ops.rs:2315`), which materialises a view |

That accounts for 24 `slice_assign` calls a layer: 2 x 2 for RoPE, 2 x 9 for `repeat_kv` and 2 for the cache.
Over 30 layers that is 720, plus 2 for the angle table: the **722** in the profile.

**Root cause.** `slice_assign_native` (`ops.rs:1001-1072`) builds a host vector with one `(source, address)`
entry per element of the *destination*. `repack_many` then has the server thread turn that vector into a
per-output-tile map of word copies, and the result goes to a new buffer. So a `k`-way `cat` costs:

- one zero-fill job,
- `k` device copies of the whole destination,
- `k` host maps of that size, built element by element.

`float_slice` of rank > 2 does the same through `sliced_native`.

None of this is arithmetic. Every copy in the table moves whole stored rows or whole 32-column tile blocks,
because a tensor is stored as `[prod(leading dims), last]` (`tensor.rs:757`). Primitives that move rows and
blocks already exist:

- `gather_rows`: several sources, face-row reads (`tt-kernels` `tensor.rs:3555`).
- `copy_blocks`: tile blocks, one source (`tensor.rs:3296`).
- `copy_into`: a whole tensor into an existing buffer (`tensor.rs:3227`).
- `rows_view`: zero-copy whole tile rows (`tensor.rs:623`).

## The design: one lowering for every copy-shaped op

Add a **region planner** in `burn-tt`. It takes a destination shape and a list of `(source view, destination
region)` pairs, and lowers them to the cheapest moves that are correct, in this order:

1. **A view, no job:** the result is a sub-range of whole tile rows of one buffer (`rows_view`), or a stride
   that `views::Strided` already expresses.
2. **Row runs:** every piece covers whole stored rows (a `cat` or `slice` on any dim but the last; a cache
   write of whole positions). Lower to one multi-source `gather_rows` job. The host builds one
   `(source, row)` entry per row, not one per element: a factor of the row width (64 for SmolLM2's heads), not
   a constant. If P0 shows the per-row entries still matter, add a run form `(source, first_row, count)`.
3. **Tile blocks:** the pieces split the last dim at multiples of 32 (RoPE's halves are 32 wide). Lower to one
   `copy_blocks` job, extended to take several sources.
4. **Element repack:** what is left (unaligned last-dim splits, steps other than 1, transposed or packed
   sources). This is today's path, but **one** `repack_many` for the whole op, not one per piece.

`float_cat`, `float_repeat_dim`, `float_slice` and `float_slice_assign` (and their int and bool versions) all go
through the planner. The result is one device job per op, or none, and host work proportional to the number of
rows or blocks moved rather than the number of elements.

**Ownership rules for `slice_assign`:**

- **In place**, only when the destination buffer is uniquely owned: `Arc::strong_count` on the buffer and on
  every view parent, no trace hold, no cache pin. Then it writes the region with a row or block `copy_into`
  at an offset.
- **Otherwise copy**, which is SmolLM2's `cache.k.clone().slice_assign(..)`: the clone keeps the old buffer
  alive. The copy is a tile-run copy of the destination (`copy_blocks`, no per-element map) followed by the
  region write. It is still two cheap jobs instead of one element repack.
- Both paths follow the AGENTS.md rules on deferred frees and padding: repairing a view must not change its
  parent's padding claims.

## Phases

### P0. Re-baseline and count jobs (done 2026-10-14; it orders P1-P4)

The 623 ms predated "barriers on the units an op ran on" and the open-all spread (2026-10-14). Also, "reported
device ops" counts zero-cost views as device operations (`report.rs` `on_device`), so it is not a job count.

- [x] A count of **jobs queued per Burn op** (`OpStat::jobs`, `report::queued`, charged to the innermost op on
  the calling thread when `server.rs` queues a job), and **server-thread wall time per op**
  (`report::served`, `burn_tt::server_report`; frees under `(free)`). Unit test
  `a_job_is_charged_to_the_innermost_op_and_timed_under_it`. `silicon_bench_models::measure` prints both
  tables and reports `server jobs` and `server thread busy`.
- [x] Re-measured on card 0, release (runs `1792005087`, `1792005154` with `TT_HOST_TIMES=1`, `1792005178`,
  `1792005205`, `1792005220`). The tables are in
  [performance-log.md](../learnings/performance-log.md#decode-step-where-the-host-time-goes-2026-10-14-card-0):

  | | 1 tile | all tiles, `work:64:4` | burn-flex |
  |---|--:|--:|--:|
  | SmolLM2 decode step | 623.9 ms | 232.6 ms | 26 ms |
  | ViT-Tiny forward | 855.9 ms | 401.4 ms | 107 ms |

**What P0 found, and the order it sets:**

- **On one tile the decode waits on the device.** 93% of queueing is the `List` stage polling for ring space, and
  the m = 1 matmuls take 1.27 ms each. That is the tile-choice item (roadmap "Per-workload tile choice"), not this
  plan. **This plan measures on all tiles at `work:64:4`.**
- **On all tiles, `cat` is 61% of the server thread.** `float_empty` (a zero-fill written as mover immediates, 803 us)
  and `float_slice_assign` (138 us each, host work) take 196.6 of 323 ms. **P1 comes first**, then P3, because
  the cache writes are 60 of the 722 `slice_assign`s.
- **Attention is next after `cat`** (1,920 jobs, 72 ms on all tiles). It is out of this plan's scope; the roadmap
  records it.
- **P2 matters for ViT more than for the decode.** ViT-Tiny's `swap_dims` and `slice` are 83 of 392 ms on all tiles
  (repacks at 0.8-1.5 ms each); the decode's are 11 ms.
- **P4 waits.** Frees cost 1.1 ms a step, and element-wise jobs 8-30 us.

### P1. The region planner, and a native `cat` (done 2026-10-14)

- [x] `crates/burn-tt/src/ops/region.rs`: `cat_rows` (off the last dimension, pieces that are whole rows of
  their buffers: one `(piece, buffer row)` an output row) and `cat_elements` (anything else: one
  `(piece, [row, column])` an element). Host unit tests compare both with a reference built the slow way, as
  `slice_assign` would, including a head view at a row offset, a column offset and a transpose.
- [x] `float_cat`, `int_cat`, `bool_cat` are overridden and in `OVERRIDDEN`; `gen-burn-ops --check` and
  `burn-coverage` are regenerated (now native).
  - Lowering: **one `gather_rows` job** for row cats; **one `repack_many`** over every piece for the rest.
  - Fallback: Burn's composition (`cat_composed`) for an empty result, a dtype or device that differs, or a
    device without resident storage.
  - Pieces that are views of one buffer share a source.
- [x] **`repeat_dim` moved here from P3**: Burn's default for it is the same composition, so it is a `cat` of
  the tensor (`float_`/`int_`/`bool_repeat_dim`, overridden).
- [x] Packed dtypes:
  - BF16 and F16 take the element path: `gather_rows` reads 4-byte face rows, not packed buffers.
  - BFP pieces are decoded and the result re-encoded, as the generated wrappers do for one operand.
- [ ] **Not built: multi-source `copy_blocks`.** Last-dimension cats take one element repack. On the decode
  step all 121 `cat`s (60 of them RoPE's last-dimension halves) cost 4.6 ms of server time in all, so a block
  copy has nothing to win there. Revisit only if a profile shows a large last-dimension `cat`, such as a
  prompt-length RoPE.
- [x] **Gate `step180_native_cat`** (ttsim and card 0, run `1792006303`, 6 of 6; in `SMOKE`):
  - **Oracle:** burn-flex bit for bit over signed zeros, subnormals, NaN payloads, infinities and `f32::MAX`.
  - **Shapes:** every dimension of ranks 1-4, at last dimensions 5, 7, 32, 33 and 64; the 9-way `repeat_kv`
    and RoPE's halves.
  - **Inputs:** view pieces (a swap, a slice of a reshape, a reshaped row view, a transpose); I32 and Bool
    `cat`; `repeat_dim` of all three on every dimension; BF16 and F16.
  - **Padding and replay:** a ragged producer, then a `cat`, then `sum_dim` and a matmul; a trace replay of
    changed inputs.
  - **Cost:** each F32 `cat` is **one job** of the expected kind (a row gather or an element repack), with no
    `float_empty` or `float_slice_assign` and no download.
  - **Negative controls**, each watched failing:
    - pieces reversed per outer index;
    - a run one row short;
    - the element map's column off by one;
    - a piece's rows read from the next piece's buffer.

    The plan's "block offset off by 32" does not apply, since there is no block path.
- [x] Residency: the `cat` downloads nothing (`tensor_traffic`), and its result is on the device. Under
  `TT_TRACE_FALLBACK=1` the only downloads are the gate's own `into_data` read-backs.
- [x] Regression checks:
  - Burn's conformance on ttsim: 3,025 run, 0 unexpected, both plain and `--fusion`.
  - Smoke tier on card 0: 1,040 of 1,040 (run `1792006807`).
  - `step12_mnist` e2e: 8 of 8, golden unchanged.
  - `step127`, `step162`, `step171`, `step179` on ttsim.
  - `step165`-`step168` on card 0.
- [x] **Measured** ([log](../learnings/performance-log.md#native-cat-2026-10-14-card-0)):

  | Decode step | Before | After |
  |---|--:|--:|
  | every tile at `work:64:4` | 232.6 ms | **117.3 ms** (`1792006312`) |
  | one tile | 623.9 ms | 458.1 ms (`1792006329`) |

  burn-flex takes 26.3 ms. On every tile the server thread is busy 146 ms, and half of that is `attention`'s
  1,920 jobs.

### P2. `slice` and `swap_dims` without repacks (done 2026-10-14)

- [x] **Slices are views.** A slice whose steps are all positive, of any rank, F32, I32, Bool, BF16 or F16, is a
  view of the same buffer: `sliced_native` moves the base to the first element and scales each stride by its
  step, and queues no job. A negative step stays one copy. The `int_` and `bool_slice` paths share it.
- [x] **Why `swap_dims` materialised.** The eager `to_dram()` came with the convolution commit (`6d979e7`), and
  its purpose was to upload a host tensor so that the swap could be a view of it. Materialising a view was a
  side effect. It now uploads only a tensor with no device copy and no view.
- [x] **Ragged tile copies.** `Strided::tile_moves`, which makes a view plain, takes a ragged last tile row or
  column where the source block ends at its own buffer's edge too. Such a view is now one tile copy, not an
  element repack: a column block of a `[197, 576]` projection, for example.
  - The old check that each output tile's row run stays inside one row of the view relied on
    `last % 32 == 0`. It is now explicit.
  - Unit test `ragged_edges_are_tile_moves_where_the_source_ends_too`, including two refusals: a ragged edge
    inside the source, and a tile row crossing a row of the view.
- [x] **Consumers.** Views of every kind already reached the element-wise path (`view_op`, else
  `plain_dram` -> `to_dram`), the batched matmul (`matrix_blocks`, which reads a head of the projection in place),
  reductions, `cat` and `slice_assign`.
  - Gate `step182` reads a stepped, column-offset slice through each of them.
  - Conformance runs every Burn slice test with no unexpected result.
  - One consumer needed a change: `slice_assign` of whole rows now makes a value that is a view of parts of rows
    plain first, which costs the value's elements, not the destination's. SmolLM2's `v` became such a view when
    `swap_dims` stopped materialising; without this, its cache writes fell back to the whole-cache element map.
    It has its own `step181` test, watched failing.
- [x] **Gate `step182_region_views`** (ttsim and card 0, run `1792012830`, 6 of 6; in `SMOKE`). The plan called it
  `step181`.
  - **Slices:** ranks 2-4, steps and column offsets queue no job, and a negative step one; bit for bit over every
    special value.
  - **Consumers:** each one above, on small integers (exact).
  - **Ragged copies:** a ragged column block made plain by **one tile copy, no repack**; rows 0..197 of a 300-row
    buffer by a repack, not a tile copy.
  - **ViT's heads:** `q k^T` with no slice, swap, reshape or transpose job.
  - **Dtypes:** I32, Bool, BF16 and F16 slices.
  - **Padding and replay:** a ragged producer, then a slice, then reductions; a trace through views.
  - **Negative controls**, each watched failing:
    - the slice's base one row down;
    - the column base applied as rows;
    - a step's stride left unscaled;
    - a ragged tile copy accepted inside the source.
- [x] Regression checks:
  - ttsim: `step84`, `85`, `127`, `152`, `155`, `157`, `160`, `162`, `171`, `178`, `179`, `180`, `181`, `62` and
    `step12_mnist` e2e (golden unchanged).
  - Card 0: `step165`-`step168`.
  - Burn's conformance on ttsim, plain and `--fusion`: 0 unexpected.
- [x] **Measured** (card 0):

  | | Every tile at `work:64:4` | One tile |
  |---|--:|--:|
  | ViT-Tiny | **283.4 ms** (`1792013104`; 401.4 at P0) | 663.0 ms (`1792013112`; 855.9 at P0) |
  | SmolLM2 decode step | **99.6 ms** (`1792013124`; 108.5 after P3) | 450.4 ms (`1792013133`) |

  burn-flex takes 106.7 ms for ViT and 26.2 ms for the decode step. Neither model queues a slice or swap job
  any more.

### P3. `slice_assign` through the planner (done 2026-10-14)

- [x] `float_slice_assign` (and int, bool, which share `slice_assign_native`), when the region writes the last
  dimension whole and in order (a cache's positions):
  - **A plain destination** gets a tile copy with the value's rows written over it. That is one job,
    `Session::assign_rows` = `Session::copy` then `write_rows`, through `Engine::assign_rows` on the card, mesh
    and simulator engines. The host builds a few run records and entries for the written rows only
    (`region::assign_written`).
  - **A destination that is a view** is one row gather over it and the value (`region::assign_rows`).
  - **Any other region** keeps the element map (`region::assign_elements`, the old code moved).
  - BF16 and F16 take the element path.
- [x] **No in-place write (a decision, not an omission).** The result is always a new buffer, so the
  ownership rules above are not needed:
  - The pattern that matters, SmolLM2's `cache.k.clone().slice_assign(..)`, never owns its destination
    uniquely, so an in-place path would not run there.
  - The tile copy costs a few records on the host, so being shared costs little.
  - An in-place write stays the trace work's question (see "Hand-off"), where fixed buffers matter.
- [x] `float_repeat_dim` (and int, bool): done in P1, as a `cat` of the tensor.
- [x] **Gate `step181_region_assign`** (ttsim and card 0, run `1792010463`, 5 of 5; in `SMOKE`). The plan
  called it `step182`; it is numbered in landing order.
  - **Oracle:** bit for bit against burn-flex over every special value.
  - **Regions:** cache positions (8- and 64-position caches), runs across a tile row, stepped and reversed rows,
    a head, a middle dimension, the whole tensor; half the last dimension and a reversed last dimension (the
    element path).
  - **Aliasing:** the SmolLM2 pattern over 16 steps keeps every earlier version intact.
  - **Inputs:** a view destination (asserted to take the row gather) and view values, a transposed value, I32,
    Bool, BF16, F16.
  - **Padding and replay:** a ragged producer, then `slice_assign`, then `sum_dim` and a matmul; a trace of a
    cache write replayed with changed keys.
  - **Cost:** each F32 case is one job of the expected kind, with no download.
  - **Negative controls**, each watched failing:
    - the written row one down;
    - the value's rows reversed;
    - the view path's two sources swapped;
    - a half row let through by rows.

    The plan's controls for an in-place write do not apply, since there is no in-place write.
- [x] Regression checks:
  - ttsim: `step84`, `step85`, `step152`, `step180`, `step62`, `step127`, `step162`, `step12_mnist` e2e (golden
    unchanged), `stale_buffer`, `burn_server`.
  - Card 0: `step167` F32, `step168` (7 of 7), `step166`.
  - Burn's conformance on ttsim, plain and `--fusion`: 0 unexpected.
- [x] **Measured** (card 0):

  | Decode step | Cache writes | Step |
  |---|--:|--:|
  | every tile, before P3 | 15.3 ms (254 us a job) | 117.3 ms |
  | every tile, after P3 | **1.5 ms (25 us a job)** | **108.5 ms** (`1792010467`) |
  | one tile, after P3 | | 451.7 ms (`1792010476`) |

  An intermediate version (one row gather over the whole cache, 147 us a job, `1792009767`) was replaced: its
  host entries grow with the cache's length.

### P4. The fixed cost per op (deferred to roadmap item 9, 2026-10-14)

The per-op path takes the `DIMS`, `TIMES` and `ATTACHED` mutexes. `supports_dram` takes `ATTACHED` twice and is
called several times an op. Every dropped `Buffer` sends its own free job, and `report::enter` locks a global map
on every Burn call (`server.rs:2060-2098`, `2151`, `report.rs:188-218`). On the server thread, each job runs L1
planning and a program-cache lookup.

- [-] **Not built; re-profiled after P1-P3** (`1792013124`). The server thread is still busy for 125 of the pass's
  126 ms, so caller-side savings would not move the wall time; the 4,278 frees take 1.1 ms in all. What is left is
  about 25-40 us of server time a job, mostly attention's 1,920 jobs (73 of 125 ms). Fewer jobs (item 9) wins more
  than cheaper ones there. Revisit the candidates after item 9, measured the same way:
  - batch the free jobs into the next submitted job;
  - cache `supports_dram` per attachment;
  - keep report counters thread-local, merging only on `with_report` exit;
  - cache the L1 plan and work shape per `(op kind, shapes, dtypes, spread)`.

### P5. Close-out (done 2026-10-14)

- [x] Measured on card 0, release, with burn-flex in the same runs and every output validated: the decode step and
  ViT-Tiny at one tile and every tile (P2's table; the history in `performance-log.md`, current rows in
  `performance-scoreboard.md`).
- [x] Correctness on card 0: `step167` F32 gives the same 16 greedy tokens, and `step168` (all seven plans),
  `step165` and `step166` pass after each phase.
- [x] `step180`, `step181` and `step182` are in `SMOKE`; the smoke tier on card 0 passes 1,052 of 1,052 (run
  `1792013157`).
- [x] Roadmap items 3 and 4 carry the numbers. The decode step is still host-bound on every tile, and the next
  lever is measured: attention's jobs (item 9). On one tile it is the device's m = 1 matmuls ("Per-workload tile
  choice").

## Checks this change needs (from AGENTS.md)

| Change | Check |
|---|---|
| Every phase | `cargo fmt --all --check`; `cargo clippy --workspace --all-targets -- -D warnings`; the new gates and the affected step gates on ttsim |
| `OVERRIDDEN` changed | `cargo xtask gen-burn-ops --check`; `cargo xtask burn-coverage` |
| `burn-tt` routing changed | `cargo xtask conformance --release` with no unexpected result; also `--fusion` (the fuser sees new ops) |
| Execution order or routing | `cargo test -p tt-tests --features e2e --test step12_mnist` (golden unchanged) |
| Silicon | Card 0 only, through `cargo xtask silicon --release --device 0 --filter <gate>`. Ask first if other agents may be using the cards. |
| Shipping | `cargo xtask check-no-sim-in-ship`; `check-no-flex-in-backend` |

## Risks

- **Views reaching consumers that assume plain buffers (P2).** This is the main correctness risk: a consumer
  that ignores a stride reads the wrong data and the result still looks plausible. The mitigations are the
  per-consumer chain gates and keeping `to_dram` as the fallback for any consumer not yet audited.
- **In-place `slice_assign` (P3).** Burn's value semantics hide aliasing. The uniqueness test must count view
  parents, trace holds and cache pins, not only the tensor's own `Arc`. Gate it with a test that keeps the
  old tensor alive.
- **Multi-source `copy_blocks` (P1).** This is a `tt-kernels` change and touches the mover's list encoding.
  If it reaches `mover.rs` or `tt_isa::dm::encoded`, run the encoded-read and differential gates in both read
  modes (AGENTS.md).

## Hand-off to a decode trace (not in this plan)

A traced decode step also needs two things this plan does not do:

- **Trace segmentation**, because the step's programs exceed the 240 KB program cache.
- **Positions supplied as a device input.** `Rope::new` uploads positions with `from_data` each step, which a
  capture refuses (`HostTransfer`). That belongs to the model or to a Burn-level position input, and has to be
  decided with the trace work.

P3's in-place write gives a replay a fixed buffer to write, but only when the destination is uniquely owned.
SmolLM2's `clone().slice_assign` is not. Whether a trace can rebind the copied cache, or needs the model to
own its cache uniquely, is a question for the trace work.
