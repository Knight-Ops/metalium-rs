# Long context: traces that build at any length, prefill on long prompts, flash-decode past the fit limit

Status (2026-10-24): **Part 1's gates pass: the 1.7B at its native 8,192 positions agrees with burn-flex (a
7,992-token prompt), and synthetic GQA models generate at 32,768, traced equal to eager. D2 and D3 done: a 1.7B prefill
of 1,000 / 7,992 tokens at 8,192 positions 4.69 / 36.63 -> 1.24 / 18.33 s, bit for bit. Open: larger chunks at long
context (1.G, GDDR reuse in a capture), the BF16 cache, and contexts past 32K. Part 2 (split flash-decode) parked by
decision (user, 2026-10-23).** Since (2026-10-25): attention's `SUB_EXP` and `DIV` replay instead of being pushed word by word, and a
prefill of 7,992 tokens takes 12.28 / 10.00 s at chunks of 64 / 512 (was 15.24 / 12.58; roadmap, "The long-context
softmax"). Since (2026-10-26, 1.H): a column operand read once a run and the softmax's statistics computed in the passes that
make their input: 7,992 tokens in **10.92 / 8.42 s** at chunks of 64 / 512. Roadmap Track 1, "Long context".
SmolLM2-1.7B's native context is 8,192.

**Where it stopped:**
- **8,192 positions work.** The 1.7B builds and decodes at 8,192 positions (card 0, `1792793709`): 17.0 ms a step after
  a 32-token prompt, 26.0 ms after 1,000 tokens (attention as the composition past the fused op's limit). The
  synthetic GQA models (`head_dim` 128 and 256) generate from a 7,992-token prompt, traced equal to eager bit for bit
  (`step224`, `1792794346`).
- **Past 8,192 the traces do not fit:** at 32,768 positions the `head_dim` 128 model's traces hold 241 of a unit's
  246 KB of program cache, and the prefill chunk's capture is refused. Each doubling of the cache adds a step trace,
  so shrinking programs (A1, A2, A3's pack) buys a doubling at a time, not 128K. **What unblocks it:** a trace's
  programs kept in GDDR and loaded into the cache when the generator switches traces (only the active trace need
  fit; a few ms a bucket change), or flash-decode, whose programs do not depend on the length.
- **Part 2 is parked at P0** (user, 2026-10-23): the Matrix Unit model (`tt_tests::matrix_unit`, step158's) is not bit for bit on
  full-mantissa operands (ttsim: 33 of 128 outputs of a plain `[4, 64] @ [64, 32]` product 1-2 ulp off, nearly
  always below the model; sweeps of the quantum, product rounding, phase order, `Dst` grouping and rounding found
  nothing better; `step225`'s `probe_*` tests). Not yet run on silicon. The recommendation on the table: exact checks
  on data the model covers, a derived bound against burn-flex elsewhere, the model gap its own investigation.
- `step225` (Part 2) is `#[ignore]`d with the reason while it is parked; `step224` is Part 1's target and fails until
  32,768 positions fit.

## Why

`TracedGenerator` for the 1.7B fails to **build** at `max_len` 2048 (`element-wise 0x1f5`, the scale and mask,
`1792789393`) and 8192 (`Max over Cols`, `1792789257`): "a list's programs do not fit an empty program cache". The
failing op is not oversized: the error is raised in a `Chunk(64)` capture at 2048 and the first `Step(4096)` capture
at 8192 (the `stream_ceiling` outputs under `target/silicon/out/`).
- **The binding limit is the 240 KiB program cache a unit** (`tt_isa::l1::PROGRAM_CACHE`), not GDDR. A generator
  holds every trace at once (a `Step(len)` a bucket 64, 128, ..., `max_len`, and a `Chunk(c)` a chunk size), and a
  held program is never evicted (`program_cache.rs`). "Empty" in the error means empty of programs no trace holds.
  Unit 0 holds the most (job `j` runs on unit `j % spread`, and a one-output reduction is one job): 232 of 240 KB at
  `max_len` 512 already (`target/silicon/bench/1792735711.md`).
- **Program bytes grow with length.** An element-wise op's unpack and pack roles are unrolled a tile
  (`sfpu/kernel.rs` `roles_code`, only math loops); a reduction's three roles are unrolled over its input tiles
  (`sfpu/reduce.rs` `roles_at`); the `qk` product's blocks grow with `N`. Held programs a trace on unit 0: 39.5, 44.6,
  52.3, 67.6, 59.4 and 75.0 KB for `Step(64)` to `Step(2048)`.
- **The sizing rules ignore what the cache holds:** `measure_sfpu_group`, `measure_reduce_group` and the segment
  grouping budget against `PROGRAM_MAX` or the whole region.
- **Prefill costs `max_len` a chunk.** A chunk attends to every position of the cache (`FixedWrite::Chunk`'s
  `attended`), and writes it by a whole-cache `mask_where` and a copy of the whole state back (3 GiB a chunk at 8192).
  The GQA mask's broadcast is a row gather list (6.5 M entries in the 2048 chunk trace).
- **The fused decode attention op is L1-bound** (`attention::fits`): a head's whole `K` and `V` on one unit, 512
  positions at `head_dim` 64 and 128 in a capture, 256 at 256, never 8 groups at 256; and one unit reads ~4.3 GB/s.
  Past it the composition runs.

GDDR is not the limit: the 1.7B's F32 weights (6.9 GB), an 8,192-position F32 cache (3.2 GB) and its traces fit a card.

## Decisions (user, 2026-10-23)

- **Past today's fit limit, attention may change order:** flash-decode, a head's positions split over units, then a
  combine, with an exact host oracle and a derived bound. Up to the limit, `attention_rows` stays, bit for bit.
- **The gates prove 8,192 positions on the real 1.7B** (a prompt of ~8,000 tokens) and **32K on synthetic GQA Llamas**
  (`head_dim` 128 and 256, two layers).

Part 1 is bit for bit and needs no firmware change; it also makes the composition run at every length, which is
Part 2's unsplit baseline and fallback.

## Part 1: traces that fit at any length, and prefill

- [ ] **1.0 The harness** (the failing evidence first).
  - [x] Host tests (`tt-kernels`): a looped role's stored words do not depend on its tile count, for `roles_code`
    (2 to 64 tiles, every `Operands`) and the reduce roles (`per` 3 to 200, 1 to 5 outputs, a continuation's
    chunks); a matmul block's is A3's.
  - [x] `place_programs`' refusal names the unit, the bytes the list needs in how many programs, the bytes traces
    hold, and the largest run they leave (`ProgramCache::room`); `SharedStateTraces::capture` names the key it
    captured and the keys beside it.
  - [x] Gate `step224_long_context` (card 0, no weights): two-layer GQA Llamas (`head_dim` 128 at 8:2 heads, 256 at
    4:1) build `TracedGenerator` at `max_len` 8192 and 32768 with chunks `[64, 512]`; it prints
    `SharedStateTraces::footprint` and asserts the build, the fullest unit's held bytes and each bucket's added bytes
    against budgets set from the first measurement (resource limits, reasons beside them).
- [ ] **1.A Program bytes independent of length** (bit for bit: the same math, data, addresses and order; only how an
  address reaches its configuration register changes).
  - [ ] A1, element-wise: unpack and pack take their L1 addresses from Tensix GPRs stepped by `ADDDMAREG` (matmul's
    `gpr_retarget`, `gpr_step`, `write_bases`, gate `step38_gpr_add`), so each tile's block is the same words, looped by
    the runner's existing block repeats (`mailbox::loops`).
  - [ ] A2, reductions: first, a loop over the identical middle tiles, the last (`valid`) tile and the finish, an outer
    loop over outputs.
  - [ ] A3, matmul, as far as `step224` shows growth: a fixed role budget in `chunk_fits_in` and `k_block_size`, then an
    output-tile loop only if needed.
  - Gates: a host evaluator over `Code::expand()` shows the old builder and the looped one present the same address at
    every `UNPACR`/`PACR`; the element-wise and reduce suites on ttsim with a stride off by one slot watched failing;
    `step12_mnist`, conformance, `--smoke` on card 0.
  - **A1 and A2 done** (2026-10-23): `sfpu::kernel::roles_looped` and `sfpu::reduce::roles_looped` (an outer loop over
    outputs, the identical middle inputs looped inside, split into sibling loops past one loop's 128), each against
    the unrolled builder by `code::transfers` (`looped_roles_unpack_and_pack_what_the_unrolled_ones_do`,
    `looped_reduce_roles_...`; a wrong stride is seen, `the_evaluator_sees_a_wrong_stride`). The evaluator found that
    the unrolled roles ignore `b_once` for `RowBroadcast` (never built so); the looped ones match it. ttsim: `step19`,
    `step30`, `step31`, `step32`, `step43`, `step55`, `step63`, `step67`, `step69`, `step149`, `step200`, `step206`.
    **Silicon found what ttsim did not:** every multi-tile gate failed on card 0 (`1792793167`) until a `STALLWAIT` on
    the Scalar Unit sat between each `ADDDMAREG` and the `WRCFG` reading its GPR (`datapath::gpr_settled`; ttsim
    divergence row 104); then 39 of 39 (`1792793443`, `1792793456`).
  - **First measurement** (`step224`, card 0, `1792793599`): `head_dim` 128 at 8,192 builds in 3.3 s holding 155 KB
    of a unit's 240; `head_dim` 256 at 8,192 in 2.4 s, 125 KB, and at 32,768 in 5.8 s, 177 KB. `head_dim` 128 at
    32,768 is refused at its `Chunk(64)` capture: the chunk's `qk` product (a block of ~60 outputs) needs 50 KB in 3
    programs beside the 179 KB the 10 step traces hold. Past 1,024 positions a step bucket still adds ~2-8 KB: the
    `qk` product (A3) and the softmax's run lengths. Dedup by hash already halves what the traces hold (313 KB summed
    a trace, 155 held).
- [x] **1.E A trace's programs paged** (2026-10-23). The traces of a generator need not fit a unit's program cache
  together, only each alone: a held program is evictable once nothing queued can run it (a unit's queue empty), a
  capture's own programs are protected until it ends, a trace keeps its programs' words, and a replay puts back at
  their addresses whatever another trace's took the room of (`ProgramCache::restore`, after a drain if what it evicts
  may be queued), skipping the work when nothing left the cache since (a change count).
  - Gate `step226_paged_programs` (ttsim and card 0, `1792796637`): two traces of 4,476 and 5,800 program bytes in a
    cache cut to 6,832 replay by turns, bit for bit the eager ops, 6,480 bytes restored over 8 replays; restoring
    switched off (ttsim), the mover refuses the stale program's list (code 8).
  - `step224` budgets each trace alone (half the cache). All four builds pass on card 0 (`1792796637`): `head_dim`
    128 at 8,192 in 2.8 s and at 32,768 in 7.9 s (refused before), `head_dim` 256 in 1.9 / 4.1 s.
- [x] **1.F GDDR held by nothing** (2026-10-24; found by `step224`'s 32K eager reference, which ran a card's 31 GB out).
  - A free waited for a sync while any unit had queued lists, and an eager chain that never syncs (255 prefill
    chunks, each a new 128 MB copy of the cache) kept every one: a free now syncs when more waits to be freed than is
    free on the fullest channel (`Session::free_placement`).
  - A trace held any placement whose slots were in use when it ended, including a later buffer on slots freed and
    reused since: now only placements allocated before its end (`Placement::serial`, `FreeSnapshot::was_free`; unit
    test `a_placement_allocated_after_a_snapshot_was_free_then`).
  - A GDDR refusal names the refusing channel's free bytes and largest run; `burn_tt::gddr_free_bytes` reports what
    is free (dev-facing).
  - Card 0 (`1792799289`): both synthetic models generate traced equal to eager bit for bit at 8,192 and 32,768
    positions, a prompt of `max_len - 200` and decode steps across the 4,096 bucket's edge (traced 32,568-token prompt:
    11.8 s at `head_dim` 128, 6.2 s at 256), GDDR unchanged after each.
- [x] **The 1.7B at its native context** (2026-10-24, `step188::the_native_context_agrees_with_flex`, card 0,
  `1792800061`): an 8,192-position cache, chunks of 64, a 7,992-token prompt (the chat prompt repeated) and 8 greedy
  tokens. The card's traced generation gives burn-flex's tokens (`[504, 3575, 282, 4649, 314, 7042, 30, 2]`), the top-1
  equal at every step, the largest logit difference 0.166 on 33.7, the closest margin 3.97; traced equal to eager bit for
  bit. Time: **38.7 s** on the card for the prompt and 8 tokens against burn-flex's 1,847 s on the host. A prompt this
  long is fed in chunks of 64: a 512-row chunk's attention holds `[32 heads, 512, 8,192]` scores several times over, and
  a capture keeps every layer's (about 48 GB), so the largest chunk must shrink as the cache grows.
- [ ] **1.B Cache-aware sizing:** `ProgramCache::room()` (held, free, largest free extent), and the segment grouping
  budgets against it. Unit 0's start rotated only if `step224` still shows it over budget.
- [ ] **1.C Step buckets:** keep doubling from 64 (8 at 8K, 10 at 32K); identical programs are shared by hash. A coarser
  ratio only on `step224`'s numbers.
- [ ] **1.D Prefill on long prompts.**
  - [x] D1: a chunk writes the cache as a row scatter (`scatter_nd(Assign)`, in place in a capture), as a step does:
    `FixedWrite::Chunk` is gone, a chunk is `FixedWrite::Masked` of `c` rows. `Shapes` requires `max_len` a multiple
    of the largest chunk, so every row is in range.
  - [x] D2 (2026-10-24): a chunk attends to a prefix, `Key::Chunk { c, len }`. `Shapes::prefixes` (the step lengths
    by default, `with_prefixes(&[])` the whole cache) and `Shapes::chunk_len` (the smallest holding `start + c`, a
    padded chunk's padding rows included) are what capture, prefill and `generate_fixed` all use;
    `FixedShapeDecoder::chunk` takes `len`, `FixedCache::chunk_write_within`, `Llama::fixed_chunk_within`. Masked
    positions add exact zeros, so a chunk at `L` equals the chunk at `max_len` bit for bit. Gates `step187`
    `chunks_attend_only_to_the_positions_in_use` (prefixes 8, 32, 64 against the growing cache and the whole-cache
    schedule, a Llama and the GPT decoder) and `a_chunk_attending_to_too_few_positions_diverges` (negative control);
    ttsim and card 0. Unblocked by 1.E: one trace a size and length (8 for `[64]` at 8,192) need not fit together.
  - [x] D3 (2026-10-24): the GQA mask as a block (`sfpu::ops::Broadcast::Block`, `tensor::block_repeats`): a second
    operand `[R, C]`, `R` whole tiles dividing `A`'s rows, read as plain `READ_RUN`s over its own grid, one a stretch
    between wraps (`tensor::block_reads`); host only, no kernel or firmware change. burn-tt routes it (`ops.rs`
    `block_of`: 1 on a run of leading dimensions, `A`'s on the rest) before `expanded`'s row gather, now reported as
    the part `broadcast/expand`. Gates: unit `a_block_broadcast_reads_b_s_tile_rows_again_down_a` (every slot through
    `record::expand`; **watched failing** with the wrap a tile row late); `step227_block_broadcast` (`mask_fill`, an
    addition and `attention` against the expanded operand bit for bit, small integers against burn-flex; a per-head
    row and a 16-row block still gathered and right). `step224` prints each trace's row gathers: a chunk of the
    two-layer GQA models runs 5, made once a chunk (its mask's positions expanded), 7 with blocks switched off.
  - **Found on the way** (`step224` at 32,768, a chunk trace a length): a streamed segment's scripts were placed
    without evicting what other traces hold, so with the traces holding nearly the whole cache a streamed reduction
    was refused ("a list's programs do not fit", 82 KB free once evictable programs go, 4.7 KB needed). They now
    evict as the kernels' programs do (`place_programs`); a replay restores them as it does its programs. Still
    open: placement is first fit without compaction, so programs a capture protects can leave a run too small for
    a program that fits the bytes (seen with `step226`'s cache cut to the larger trace exactly).
  - Measured (card 0, `1792808294` -> `1792809099` -> `1792809505`,
    [log](../learnings/performance-log.md#a-long-prompts-prefill-chunks-attend-to-a-prefix-the-gqa-mask-read-as-a-block-2026-10-24-card-0)):
    a 1.7B chunk of 64 at 8,192 positions 293 -> 187 ms with D3 (the gather was 36%), then each chunk at its prefix:
    prefill of 1,000 / 4,000 / 7,992 tokens **4.69 / 18.46 / 36.63 -> 1.24 / 6.81 / 18.33 s**, decode unchanged.
- [ ] **1.G GDDR reuse within a capture** (found 2026-10-24). A capture defers every free (`Session::free_placement`), so a
  trace holds every layer's intermediates: a 512-token chunk's scores at 8,192 positions (~48 GB over 24 layers) do not
  fit, so a long prompt runs in chunks of 64; and since D2 a generator holds a chunk trace a length, so the 1.7B's
  eager schedule no longer fits beside them (`step188` releases them first).
  - **Why it is safe on the card** (read off the code, 2026-10-24): on a session of more than one unit every captured op
    ends in a barrier whose target counts every earlier arrival (`Session::enqueue_work_inner`, `dm.rs` barrier
    targets), and a unit that sat an op out waits for the latest target before its next list (`trace_behind`,
    `catch_up_to`). So no unit starts op `m` before ops `0..m` are done everywhere, on replay and in the capture's own
    run, and the host frees a tensor only after enqueuing its last reader: a later op may take its slots. The lists
    that run ahead of barriers (`PREFETCH`, `BUSY_PREFETCH`) read only a `B` the capture neither allocated nor wrote.
  - **What it must keep:**
    - only a tensor born in the capture (its `Placement::serial` past the capture's start) and not uploaded during it
      (an upload is written once, through the BAR, and not by a replay) gives its slots back; a tensor made before the
      capture and freed in it is read again by the next replay, so its free stays deferred;
    - one unit has no barriers: no reuse there;
    - the trace owns every slot its internal tensors used: at its end, what of them is free is claimed for the trace
      and given back once at `release_trace` (no double free: a placement over reused slots is never freed itself);
      a tensor alive at the end keeps today's hold (`Trace::holds`);
    - the residency planner attributes GDDR slots to tensors by address (`residency::Index`, one tensor an address):
      with reuse it must find the tensor an address held at an op (the latest allocated by then).
  - [x] G0: the meter (2026-10-24): the GDDR left after a build, reported by `traced_generation_stream_ceiling` and
    printed by `step188`. The 1.7B at 8,192 positions with a chunk trace a length: **0.34 GB free after the build, 26.85
    once its traces are released** (`step188`, card 1, `1792815355`); 2.10 GB in the bench's build (`1792816593`).
  - [x] G1 (2026-10-24): `DramAlloc`'s capture pool (`start_pool`, `give_to_pool`, `take_pool`, `free_ranges`; uploads
    and host writes through `uploading` and `note_host_written` never enter it), `Session::set_capture_reuse` (and
    `TT_CAPTURE_REUSE=1`; nothing on one unit), the trace's `pool` given back at `release_trace`; a capture that reuses
    makes no residency plan and applies none. `burn_tt::set_capture_reuse`; `TracedGenerator` turns it on for its
    chunk captures only (the steps keep their residency plans), by default since G3 (`TT_CHUNK_REUSE=0` turns it
    off). Gate `step228_capture_reuse`
    (4 units, ttsim and card 0 `1792816577`): a chain freeing as it goes replays bit for bit as it ran, its trace
    holding 299,520 bytes of GDDR against 965,120 without reuse, every byte back at the release; a tensor made before
    the capture and a constant uploaded in it keep their slots (**watched failing** with `give_to_pool` taking them:
    both replays diverge, `[false, false]`, since the capture's own run already wrote over them).
    - Measured (1.7B, 8,192 positions, card 0; `1792816593` / `1792817025` / `1792817342`): GDDR free after the build
      **2.10 -> 17.78 GB**, the build 226 -> 127 s, prefill unchanged (1,000 / 7,992 tokens 1,236 / 18,323 -> 1,255 /
      18,438 ms); and **chunks of 512 now capture at 8,192** (14.79 GB free): 1,011 / 16,877 ms, 1.22x / 1.09x, the
      products' plateau (below) capping what a larger chunk buys.
  - [x] G2 (2026-10-24): a capture that reuses is planned. `OpRecord::allocs` (the capture's tensors allocated when the
    op was queued); `residency::Index::extend` puts each allocation over the dead extents it overlaps, and
    `Graph::from_capture` grows the index op by op, so an access is charged to the tensor its address held at that op
    (the planned capture's `apply_plan` already grew it with the log). A tensor freed in the capture is matched by its
    allocation (`Placement::same_serial`), not its address: by address, a chunk's output over a freed intermediate's
    slots looked freed, never read, its writes dropped. `begin_applied` and `end_trace` no longer refuse a reusing
    capture's plan. Gates: unit `an_address_is_the_tensor_that_held_it_at_the_op` (**watched failing** without the
    overlap removed: the dead tensor named); `step228` `a_planned_capture_over_reused_slots_replays_as_it_ran` (ttsim and
    card 0 `1792870177`: bit for bit, 663,552 B moved against 2,248,704 unplanned, 299,520 B held against 965,120
    planned without reuse, the same plan either way; **watched failing** with the whole log's index: five of six
    intermediates seen never read, one resident, the planned capture fails; and with freedom by address: the last op's
    capture refused); `step219` again claims fewer bytes for every planned trace, chunks included (card 0 `1792870184`:
    a chunk 3,387,264 -> 1,482,624 B). Card 0, all passing: `step185`, `step187`, `step212` (`1792870223`), `step224`,
    `step188` with the native context, the same tokens as burn-flex, 16.91 GB free after its build (`1792870439`).
    - Measured (1.7B, 8,192 positions, card 0, [log](../learnings/performance-log.md#residency-plans-over-reused-gddr-2026-10-24-card-0)):
      prefill of 1,000 / 7,992 tokens 853.3 / 15,432 -> **841.4 / 15,244 ms** at chunks of 64 (`1792873658` ->
      `1792873151`; the planned capture without reuse 841.4 / 15,243, `1792874014`), 484.3 / 12,629 -> **474.0 / 12,580**
      at 512 (`1792875430` -> `1792875893`); GDDR free after the build 17.70 / 15.07 GB. The first build 92 -> 224 s at
      64, 135 -> 893 s at 512 (two captures a chunk trace, and the host expands each graph's moves three times).
  - [x] G3 (2026-10-24): on by default for chunk captures, without G2 (the chunk captures are unplanned). An engine
    without it (the mesh engine) builds without reuse, unless `TT_CHUNK_REUSE=1` asked for it (then refused by name). With reuse
    on, 22 of 22 traced gates passed on card 0 (`1792817785`: `step185`, `step187`, `step212`, `step224`, `step188`
    with the native context). Prefill at chunks of 64 measured 0.6-1.5% slower in one run each (`1792816593` against
    `1792817025`), not confirmed beyond noise. The price: on more than one unit a chunk's capture is no longer
    residency-planned (no L1-resident intermediates; G2 restored it), so `step219` claimed fewer bytes for the
    planned steps and no more for the chunks until G2. One rule (`Session::reuses_captures`: asked for and more than one unit)
    decides both whether a capture reuses and whether a plan is refused for it; before it, a one-unit chunk ran a plan
    pass every build, refused its plan and captured plainly again.
  - [ ] Larger default chunks at long context (`TT_GEN_CHUNKS` 128-512 at 8,192 fit now); worth little until the
    products reuse their operands (1.22x / 1.09x at 512).

- [x] **1.H Long-prompt softmax passes** (2026-10-26, card 0; roadmap "Next for long prompts: the data passes";
  [log](../learnings/performance-log.md#the-long-prompt-softmax-a-column-read-once-a-run-the-statistics-in-the-passes-that-make-their-input-2026-10-26-card-0)).
  The profile (`1792967661`, chunk of 64 at 8,192) had the scale and softmax at 44% of a replay:
  - `SUB_EXP` and `DIV` were data-bound at ~290 GB/s, half their reads the `[rows, 1]` column read once a tile;
  - the maximum and sum were full passes, chunked past one run's 219 tiles on 64 units, each with a `CLAMP_MIN` op.
  - [x] H0, the anchor: `step31_reduce` `chunked_reductions_are_the_programs`. Lines of 230 and 240 tiles, both axes,
    specials; equal to `sfpu::reduce::reference` bit for bit; the sum folded a chunk at a time differs (control);
    **watched failing** with the carried accumulator dropped. Now in `--smoke`.
  - [x] H1, a column read once a run (`tensor::row_runs`, `TT_COL_ONCE=0` off, `set_col_once`): a column broadcast's
    runs keep to tile rows and read their row's `B` tile once into one slot (`b_once`). They are taken only where a
    model of the busiest unit's transfers (two a tile and one for `B`, against three a tile) beats the runs they
    replace. Gates:
    - host: `row_runs_keep_to_tile_rows`, `a_column_broadcast_reads_its_row_s_tile_once_a_run` (off: `B` a tile);
    - `step206` `a_long_row_reads_its_column_tile_once_with_the_same_bits`: `SUB`, `DIV`, `SUB_EXP` over rows of 90 and
      121 tiles, equal to the expanded-operand model and to the old runs; **watched failing** with the next row's tile
      read.
    - The first rule (fewest runs) cost RMSNorm's chunk division and decode rows their spread (8% at 1,000 tokens, 5%
      on the long decode step); costed, both recovered.
  - [x] H2, the statistics in the passes that make their input (`softmax::Mode::Written`, `Session::set_softmax_mode`):
    - passes: `SCALE_MASK_FILL` with the maximum, staged `SUB_EXP` writing `e` with the sum, staged `DIV` by the sum
      (`sfpu::softmax::{sub_exp_sum_out, div_by_sum}`, `ops::staged_at` with stage rows clear of the statistics);
    - burn-tt routes attention there on rows past one reduction run (`TT_SOFTMAX_ROWS` unset: auto; `1` every row,
      `0` never).
    - Unit gates: `the_written_passes_are_the_composition_bit_for_bit`, `the_written_passes_are_staged`.
    - `step230`: both modes against their models and against the six ops, a row of 230 tiles with a repeated mask
      included; **watched failing** with the sum taken from `s`.
  - Card 0, all passing:
    - with the first rule: `step31`, `step183`, `step185`, `step187`, `step206`, `step209`, `step227`, `step230`,
      `step224`, `step188` (`1792970468`, `1792972687`);
    - with the costed rule: `--smoke` 1,213 of 1,213 (`1792981502`), `step188` and `step224` (`1792983113`). The native
      context gave burn-flex's tokens, 0.166 on 33.7, traced 12.31 s.
  - Also passing: `step12_mnist` (ttsim); conformance on ttsim, 2,973 of 3,025 with the 52 listed.
  - Measured: 7,992 tokens 12.06 -> **10.92 s** at chunks of 64 and 9.84 -> **8.42 s** at 512; 1,000 tokens 746 -> 745 and
    399 -> 386 ms; the long decode step 53.3 -> 52.6 ms; short context, ViT-Tiny and ResNet-18 unchanged.
  - Left, on numbers:
    - pass 1 still reads the mask once a head (32 x 2 MB a layer at `c` = 64);
    - pass 2's in-order sum keeps one job a tile row, so a chunk of 64 runs the passes on 64 of 120 units (the maximum
      alone could split a row over units: exact in any order, once signed zeros and NaN payloads are shown to agree);
    - attention's `pv` is now the largest part of a chunk of 512 (77 of 286 ms before this work).

## Part 2: flash-decode past the fit limit

- **A run** is one L1-resident block of a head's positions, today's job without the `DIV`: `qk`, the scale and mask,
  the row maximum clamped (`m_i`), `SUB_EXP` against `m_i`, the row sum (`l_i`) and an unnormalized `pv` (`o_i`),
  written to a scratch tensor.
- **The combine** is a second op (op, barrier, op; no in-op peer wait): `M = max m_i`, `w_i = exp(m_i - M)`,
  `l = clamp(sum w_i l_i)`, `o = sum w_i o_i`, `o / l`, and the row's packed twin.
- **What changes against the composition:** each `exp` is taken twice (`E(T - m_i) E(m_i - M)`), the division moves
  after `pv`, `l` is a sum of runs' folds, and `o` is `s` `Dst` chains scaled and added. Unchanged: the scores, the
  scale and mask, the head's maximum, and the groups of eight in each `MVMUL` (runs are whole tiles). Fallback if the
  bound is ever too wide: a keys pass against the exact maximum, then values (every `exp` today's, ~5% slower).
- **Specials:** a run entirely masked gives `m_i = -MAX`, `l_i = 0`, `o_i = 0`, so `w_i = 0` and no `-inf - -inf`
  reaches the combine; a row entirely masked gives 0, as today; no `exp` argument is positive.
- **The split is a function of shapes and constants only:** a block `nr` a `head_dim` against a fixed 180-slot budget
  (8 tiles at 64, 4 at 128, 2 at 256; refused, never shrunk, below it); runs a head `s` a power of two up to 512, chosen
  from a pinned cost model (fixed cost and a unit's read rate, measured in P2); the switch-over is `fits` against a fixed
  196-slot arena, since an arena-dependent boundary would let eager and traced choose different ops and bits. Limit:
  `512 x nr x 32` positions (128K / 64K / 32K at 64 / 128 / 256).
- **The output check** is `2 m dvt <= 128` (no twin a run), which admits 8 groups at `head_dim` 256.

- [ ] **P0** The composition's time at 2048 and 8192 (after Part 1), and the oracle's unsplit order equal to `step209`'s
  device output bit for bit. Stop and ask if the Matrix Unit model is not bit for bit on full-mantissa F32 through TF32.
- [ ] **P1** The oracle (`tt-tests/src/matrix_unit.rs`, step158's model lifted out as a `Dst` chain; `flash_oracle.rs`,
  `unsplit` and `split`, the SFPU steps through `sfpu::ops::reference_op` and `sfpu::reduce::reference`), the derived
  bound, and a unit test pinning `s` for the reference shapes.
- [ ] **P2** The runs op (`tt-kernels/src/flash_decode.rs`), gated on its partial rows; the probe
  `silicon_probe_attention_rows::flash_scaling` fixes the cost model and the units. Stop if the runs make less than
  250 GB/s at 8192.
- [ ] **P3** The combine and `Session::flash_decode`.
- [ ] **P4** burn-tt routing (`part("attention/flash")`, `set_flash_decode`, `TT_FLASH_DECODE`); the uncommitted
  `probe_split` removed.
- [ ] **P5** Card 0 gates and benches. Stop if 512 is not bit for bit and time-neutral, or flash does not beat the
  composition at every length past the limit. Then the user decides default or opt-in.
- Gate `step225_flash_decode` (new-op definition of done): device equal to `split` bit for bit (`head_dim` 32 to 256,
  `m` 1, 4 and 8, `s` 2 to 8, a ragged tail); within the derived bound of `unsplit` and of burn-flex; watched failing: a
  run dropped, `w_i = 1`, the unsplit order; specials (a run or a row masked, NaN keys masked and not, scores of +-300
  across runs); routing and residency; eager and a residency capture choose the same `s` and bits.

## End to end

- `step224` extended: traced against `generate_fixed` bit for bit at 8192 and 32768 (a prompt of `max_len - 200`, and
  one crossing a bucket's edge), D2's equality, no state copy in a chunk trace, the synthetic models against burn-flex.
- The 1.7B at 8192 (`step188`): a prompt of ~8,000 tokens of real text, traced against `generate_fixed` bit for bit,
  and top-1 against burn-flex at every step, Flex's tokens teacher-forced.
- Benches: `traced_generation_stream_ceiling` with `TT_GEN_MAX_LEN` 512 (unchanged), 2048 and 8192, flash on and off,
  then `--profile` at 8192; rows in the scoreboard and the log.

## Not in this plan

A BF16 cache read without `attention/widen`; `max_position_embeddings` and RoPE scaling in `LlamaConfig`; runs of
several blocks past the limit; Qwen3-class architectures (Gated DeltaNet, MoE, 262K caches).
