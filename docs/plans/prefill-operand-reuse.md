# Prefill products that reuse their operands

Status (2026-10-24): **Step 1 built and on by default (`TT_MATMUL_BLOCK=off` turns it off).** Bit for bit the other
paths (`step229`, ttsim and card 0; the model gates and the smoke set on card 0 with it on). On card 0 the 1.7B's prefill products run **2.9-3.6x** faster at `m` >= 512
(`gate|up` 5.21 -> 18.23 TFLOPS at `m` = 2,048) and 1.4-1.5x at `m` = 64; its prefill at 8,192 positions, chunks of 512,
takes **0.48 / 3.70 / 12.59 s** for 1,000 / 4,000 / 7,992 tokens (1.24 / 6.81 / 18.33 before)
([log](../learnings/performance-log.md#prefill-products-that-reuse-their-operands-2026-10-24-card-0)). The products
are now bound by the kernel's compute, ~0.4-0.6 us a K-tile pair. Open: P4 (the pair) and P5 (Step 2). Roadmap Track 1, "The roofline", "Prefill products that reuse their operands".

## Why

Prefill-shaped products plateau at **5.2 TFLOPS** (`gate|up`) and **2.8** (`down`) from `m` = 512 to 2,048, GDDR-bound at
~400 GB/s (`1792816512`); a long prefill uses 1.1-1.7% of the card's ~166 TFLOPS (`1792815504`,
[log](../learnings/performance-log.md#how-much-of-the-cards-compute-a-long-prefill-uses-2026-10-24-card-0)). Larger
chunks do not raise the rate: a product's traffic grows with `m` as fast as its work.

What the traffic is made of, from the code:
- `matmul::plan_uncached` keeps all of `K` in the ~215-slot arena. At `K` = 2048 the largest block is one tile row by two
  columns, so `B` is read `mt` times and `A` about once a unit.
- At `K` = 8192 (`down`) nothing fits whole: `push_k_jobs` makes one output tile a job, four K blocks each re-gathering
  `A` and `B`, the F32 partial written to GDDR and read back. `A` is read `nt` times and `B` `mt` times.

## The design (Step 1: a K-streamed 2-D blocked product)

A job owns an `mb x nb` block of output tiles and streams `K` in chunks of `kq` tiles:
- **Chunk `c`** gathers `A`'s `mb x kq` tiles and `B`'s `kq x nb` into input half `c % 2`; its kernel computes every
  output tile of the block, row-major, each over the chunk's `kq` pairs in `k` order, reloading the tile's FP32 partial
  (none in the first chunk) and packing it again.
- **Partials alternate between two output sets** (`out[c % 2]`, reloaded from `out[1 - c % 2]`), as
  `matmul::StreamLayout`'s do: a unit's batches are double-buffered (`Step::Kernel::half`), and the session only orders
  a batch's gather after the scatter two batches back (`streaming_segments`' `RELEASED`), so a partial packed in
  place could land on the slots the previous job's last scatter still reads.
- **An even number of chunks a job**, so every job starts in half 0 and the halves alternate throughout a unit.
- **Only the last chunk is written out.** No partial goes to GDDR.
- **Bit for bit with the other paths:** each output tile takes the same products in the same `k` order, through FP32
  partials reloaded as a K block's are (step68). Split-`K` is not used: it changes the order.

L1: `2 mb nb` output slots and `2 kq (mb + nb)` input slots (8x8 with `kq` 2, 6x6 with 5, 4x4 with 10: ~192 slots).

**Traffic:** `MNK (1/mb + 1/nb)` tiles, against `MNK (1 + 1/2)` at `K` = 2048 and `MNK (1 + 1)` for `down` before.

**Compute (measured):** ~0.4-0.6 us a K-tile pair at HiFi4 (540-810 cycles against the Matrix Unit's 64), and ~3-5 us a
chunk of 24 outputs for their reloads, packs and the chunk's batch (~0.15 us an output). So a block wants enough
outputs to keep its reads under its compute (24-64), and the planner scores whole rounds of jobs (`down` at `m` = 2048
has 64 x 64 output tiles: 8x8 blocks are 64 jobs on 120 units).

## Phases

- [x] **P1. The kernel and its layout** (`matmul.rs`, 2026-10-24): `BlockStreamLayout` (two input halves, two output
  sets, the handshakes), declared through `l1::Requirements`; `matmul_items_continuing` reloads every output of a
  block, each from its own prior (`Dst` cleared between outputs; a K block's one-output words unchanged);
  `pair_code` shares a pair's words between the builders. `block_stream_kernel`, unrolled per output.
  - [x] **P1b. Programs that do not grow with the block** (2026-10-24, `block_stream_kernel_looped`, on by default,
    `matmul::set_block_loops`). Unrolled, a continuing chunk's unpack role is ~130 words an output (the prior's unpack
    to `Dst` and the `Src` configuration restored after it): 4,846 words for 6x6, 8,350 for 8x8, which `PROGRAM_MAX`
    refuses, and chunks of 512 did not capture ("a list's programs do not fit the program cache", `1792856420`). A
    role program has no branches (its loops are the replay buffer and the MOP, `crate::loops`), so each output's block
    is stored once and repeated by the runner (`crate::code::Loop`: the unpack role a column inside a row, the math
    and pack roles an output), its addresses stepped in GPRs (`A`'s row, `B`'s column, the prior, the output). Under
    400 words a role whatever the block; as fast as unrolled. ttsim caught the one bug before silicon (`ADDDMAREG`
    names its result first: an unpack out of L1's bounds).
- [x] **P2. Plan, jobs and selection** (`tensor.rs`, `session.rs`, 2026-10-24): `block_stream_plan` (a model of
  reads, compute and rounds over `[mb, kq, nb]`, the fit in the arena and the programs checked; cached a process),
  `push_block_stream_jobs`, chosen in `matmul_dram_prefetching` for `mt >= 2`, `kt >= 2`, no packed `A` and no forced K
  block; `Session::set_block_stream`, `TT_MATMUL_BLOCK=off|auto|mb,kq,nb` (`auto` by default since P3). A fixed shape
  that does not fit says so on stderr and runs the other paths.
- [x] **P3. Gates and measurement** (2026-10-24).
  - [x] `step229_block_stream_product` (ttsim, card 0 `1792855491` and `1792857062`): bit for bit against the other
    paths over ten block shapes (8x8 looped), LoFi and HiFi4, `A` or `B` transposed, ragged in every dimension, the
    special values; looped and unrolled; exact against host integer sums; the negative control
    (`tensor::probe_reverse_chunks`: chunks last first) changes the outputs. Host tests: each job reads its strips
    once and writes its block once, halves alternate, looped programs do not grow, the model's choices for the 1.7B's
    products. step18, 68, 203 and 204 unchanged on ttsim.
  - [x] `matmul_throughput` at prefill shapes, fixed-block and `kq` sweeps, the model's constants from them
    ([log](../learnings/performance-log.md#prefill-products-that-reuse-their-operands-2026-10-24-card-0)).
  - [x] The 1.7B's prefill end to end (chunks of 64 and 512).
  - [x] `step12_mnist` (`--features e2e`, ttsim) with it on: the goldens hold. Burn's conformance on ttsim with it
    the default (`cargo xtask conformance --release`, `target/conformance/1792865927`): 2,973 of 3,025, the 52 listed
    failures and no other.
  - [x] With it on, card 0: `step187` (10 tests, `1792858426`), `step227` (`1792858452`), `step224` (`1792858840`),
    `step188` (`device_agrees_with_flex` `1792858577`; `the_native_context_agrees_with_flex` `1792859080`, the 1.7B at
    8,192 positions against burn-flex). Then `auto` the default, and the smoke set on card 0: 634 of 635
    (`1792861197`); the one, `step205`'s busy-prefetch control, assumed its 64-row product plans most of `DATA`, which
    a blocked one does not (the read ahead rightly crosses it, the bits agree): it pins the other paths.
  - [x] The gates of the other paths' own features pin them (`set_block_stream(Off)`): `step20` (its every-unit deal),
    `step54`, `step60_streaming`, `step190`, `step198`, `step201`, `step203`, and `step205`'s control. `step229` covers
    the blocked path on 2-4 units, `auto`'s own shapes included.
  - Found, not from this work (fail the same on `2371aae`): `step190` hangs 2 runs in 3 on ttsim with parallel test
    threads (passes with `--test-threads=1`); `step60_streaming`'s `a_streaming_trace_holds_its_bodies_under_cache_pressure`
    and `releasing_a_stale_trace_leaves_a_newer_traces_programs_held` see no program-cache evictions.
  - The model (0.4 us a pair, 0.15 an output's reload, 2 a chunk, GDDR's useful rate 320 GB/s) ranks shapes; it is not
    exact (fixed blocks measured within ~10-25% of it). A sweep of a product's shapes is `TT_MATMUL_BLOCK=mb,kq,nb`.
- [ ] **P4. Compute a pair** (the bound now): what makes a K-tile pair ~540-810 cycles against HiFi4's 64 `MVMUL`
  cycles -- the twelve face unpacks of two F32 tiles a pair, the bank handoffs between unpack and math, the pushes --
  measured with profile stamps; then `SrcB` reuse of an `A` face across a row of outputs in `Dst` (G4 subblocks).
  Each output's `K` chain stays in order.
- [ ] **P5. Step 2: each operand out of GDDR about once** (multicast along the grid's rows and columns, or strips held
  by a few units and pulled): needs a cross-unit ready/free credit, which does not exist. Worth it once P4 makes the
  products read-bound again.
