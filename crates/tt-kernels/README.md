# tt-kernels

Host-side Tensix kernels and everything that runs them: instruction streams for the
three Tensix roles (unpack on T0, math on T1, pack on T2, as tt-metal's LLK splits
them), the runner, the chip session, GDDR-resident tensors, the L1 planner, and the
on-tile and chip-to-chip data movers. Shippable; depends on `tt-isa`, `tt-device`,
`tt-layout`, `tt-kmd`, never on the simulator.

## Modules

| Module | What it holds |
|---|---|
| `session` | `Session<T>`: owns a chip for compute. Bring-up order, per-tile reset, resident roles. `TileChoice::{Exactly, First, Count(n), All}`; `Session::open_card(index, ROLES, choice)` for silicon. `enable_dram`, `upload`/`download`/`free`, `matmul_dram`, `eltwise`, `sum_rows`. Also `matmul_on`, the reset-per-run host-staged matmul. |
| `tensor` | `DramTensor`: row-major `[rows, cols]` FP32 stored as 32x32 tiles interleaved over the usable GDDR channels. `DramAlloc`. The GDDR matmul, element-wise and column-sum ops, built as `Work` (jobs) dealt over the session's tiles. |
| `bf16` | `Bf16Tensor`: separate two-byte storage in 2112-byte aligned slots. Raw upload/download, native bit copies/views, device ties-even narrowing and SrcA/MOVA2D widening, compact packed gathers and K-block matmul with F32 accumulation. Allocation and trace holds retain the physical slot size. |
| `bfp` | `BfpFormat::{Bfp8, Bfp4, Bfp2}` and distinct `BfpTensor`: shared exponents, native pack/unpack and packed K-block matmul. Format participates in cache keys and physical allocation/ownership. |
| `fpu` | Opt-in 16×16 GMPOOL/GAPOOL block max/sum/mean, and BF16 window sums with F32 continuation and explicit mean divisors. General F32 reductions keep their SFPU semantics. |
| `dm` | `DataMover`: the `dm_b` image (a pure GDDR reader) on RISCV B of one tile, or the `dm_nc` image (a pure writer) on RISCV NC. `read`/`write`, `run_list`/`submit_list`/`wait`, and `enqueue`, which checks a list as the mover will before writing it. Lists carry op records (`tt_isa::dm::record`), `KERNEL`/`LAUNCH` entries that drive the resident roles, and `PAIR` packets: a reader and a writer section joined by circular-buffer credits, one host commit. |
| `runtime` | `Kernel` (three role programs + `Schedule`), `run` (stage, start, wait, collect), `Resident` (role images left running between kernels), `Profile`. |
| `program_cache` | `ProgramCache`: the host's mirror of one tile's resident programs in `tt_isa::l1::PROGRAM_CACHE` (exact-word keys, first fit, LRU, pinned while a list names them, programs over half the region bypassed, cleared on reset). A `KERNEL` entry names each role's resident program, so one list runs kernels of any shape. `Session::program_cache_stats` gives hits, misses, uploads and evictions per tile. |
| `l1` | The L1 planner. A kernel declares `Requirements` (buffers: scratch or circular buffers with pages and one producer/consumer; live stages; semaphores). `Requirements::plan` places them in `tt_isa::l1::DATA`, sharing bytes between buffers never live together; `check` verifies a plan independently; `Requirements::fuse` merges two kernels, unifying producer and consumer buffers. |
| `matmul` | The Matrix Unit datapath: `Src` staging, role preludes, `Fidelity` (`Lo`, `HiFi2..4`), `SrcRoute` (`Tf32FromFp32`, `Bf16FromFp32`, `Bf16FromBf16`, `Bfp(format)`), chunk planning (`plan`, `chunk_fits`), `matmul_requirements`, `tilize_f32` / `detilize_packed`. |
| `adc_copy` | `Session::copy_rect_adc`: exact F32 rectangular copies using unpacker XY counters and Dst (16-column boundaries). `Session::copy_planes_adc`: arbitrary nonempty F32 Y/X planes selected from `[W,Z,Y,X]` using Z/W counters and bounded staging. Both preserve raw bits and produce zero padding. |
| `datapath` | Unpacker/packer configuration for flat FP32 runs; thread state reset. |
| `link` | `Link`, `Mover`, `discover`: chip-to-chip moves over Ethernet with the `eth_e1` image. |
| `shard` | `Fabric`, `Chip`: a matmul split along `N` across cabled chips, bit-identical to one chip. |

## Many tiles

BF16 narrowing is silicon-only with the pinned simulator, which refuses late
pack mode `0x105`. Raw BF16 storage/copy, packed matmul and BF16 pooling window
staging have simulator gates. Device conversion rounds ties-even, quiets NaNs
while retaining sign/high payload and flushes BF16 subnormals to signed zero;
raw storage/copies preserve every bit. See `step74`–`step78` and
[the implementation record](../../docs/plans/tensix-next-features.md).

A session with `TileChoice::Count(n)` or `All` runs one unit (resident roles + data
mover) per surviving tile. A GDDR op is a set of independent jobs dealt round-robin
over the units, each unit's work queued as one packet (B reads, the roles compute, NC
writes); no job's arithmetic depends on where it runs, so results are the
same bits for any tile count. Callers pick the count through `burn_tt::Topology`
(`TT_TILES`, `tt-mnist --tiles`).

## Test

```bash
cargo test -p tt-kernels
```

Unit tests here: the L1 planner, op-record expansion against the list builders it
replaced, job dealing over units, list batching, matmul planning. Device gates
are in `tt-tests`; the BFP extension is covered by step92–96. The active
inventory and silicon evidence are in `docs/plans/hardware-coverage.md`.

## Gotchas

- **Open a chip through `Session`.** It reads the Tensix grid from the ARC before
  touching any tile, refuses a tile the chip lacks (`SessionError::NoSuchTile`),
  registers the driver's cleanup write, and resets each tile's cores, backend and
  per-thread state. Skipping any of these has cost silicon runs (a fused-off tile
  hangs the NoC; stale `Config` and `Dst` leak between runs).
- Kernels never name L1 addresses in `tt_isa::l1::DATA`; they go through `l1`.
  A fused kernel that does not fit is a planning error, never an overlap.
- Encoder `unwrap`s in the builders are bugs in this crate, not caller errors. The
  runner returns `runtime::RunError`.
- `Topology::Cards` (via `shard`) stages operands from the host per matmul and
  computes on one tile per card. Only `Session` keeps tensors in GDDR.

## General reductions and K blocking

`Session::repack` constructs a matrix from source-coordinate metadata using
aligned B reads, local word copies and NC writes, preserving bits and parent
padding claims. Ragged output padding is undefined. Burn uses it for general
axis reductions and materializing non-tile-coherent views. Long column sums
and row/column maxima preserve an unfolded accumulator across bounded chunks.

Resident ordinary and supported tile-aligned batched matmuls automatically
split K when necessary. Each output tile stays on one unit, with FP32 prior
accumulators reloaded in product order. `set_matmul_k_block_limit` optionally
forces a maximum K tile count; `None` restores automatic planning. Existing
unsplit pipelining remains. `step67`/`step68` cover simulator and both-card
execution (`1791145571`); BFP K reloads are covered separately by step94.
Release BFP measurements are recorded in `docs/learnings/firmware-performance.md`.

## Current extension status

Inclusive min/max scans use raw F32 total ordering. Integer reductions use
wrapping sum/product and signed min/max without F32 conversion. Hardware BF16/
TF32 precision modes are opt-in and preserve documented Blackhole rounding
behavior; ordinary BF16 casts retain their separate contract. Checked integer
division/remainder is enabled and validated by step82 on both cards. See
[the current handoff](../../docs/plans/tensix-next-features.md).

## Resident BFP storage

After `enable_dram`, `Session::bfp_from_f32` packs an F32 `DramTensor` into
`BfpTensor`; `bfp_to_f32` widens on device. `download_bfp` returns decoded F32,
`download_bfp_raw` returns diagnostic physical tile images, and `free_bfp`
preserves trace holds/deferred frees. Conversion repairs dirty input padding
without changing a parent view's padding claim.

Each 32×32 tile carries a 16-byte header and 64 exponent bytes. BFP8/4/2 image
sizes are 1104/592/336 bytes; aligned GDDR slots are 1152/640/384 bytes. B/NC
perform transfers only. `bfp_matmul` consumes same-format nontransposed operands
directly, with F32 accumulation and bounded K reloads; mixed/transposed inputs
widen on device. `copy_into_bfp` retains physical bytes and exponent groups.
Initial execution is single-card; packed mesh transport remains deferred.

Step92/94 cover independent encoding/arithmetic oracles, special values,
ragged padding, physical sizes and changed-input replay on simulator and both
cards. Pinned ttsim refuses direct BFP2 matmul; that arm is silicon-only. See
[the delivered contract and validation](../../docs/plans/mixed-bfp-storage.md)
and [conversion/matmul release medians](../../docs/learnings/firmware-performance.md).
