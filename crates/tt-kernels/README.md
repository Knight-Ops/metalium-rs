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
| `dm` | `DataMover`: the `dm_b` image on RISCV B of one tile. `read`/`write`, `run_list`/`submit_list`/`wait`. Lists carry op records (`tt_isa::dm::record`) and `KERNEL` entries that drive the resident roles. |
| `runtime` | `Kernel` (three role programs + `Schedule`), `run` (stage, start, wait, collect), `Resident` (role images left running between kernels), `Profile`. |
| `l1` | The L1 planner. A kernel declares `Requirements` (buffers: scratch or circular buffers with pages and one producer/consumer; live stages; semaphores). `Requirements::plan` places them in `tt_isa::l1::DATA`, sharing bytes between buffers never live together; `check` verifies a plan independently; `Requirements::fuse` merges two kernels, unifying producer and consumer buffers. |
| `matmul` | The Matrix Unit datapath: `Src` staging, role preludes, `Fidelity` (`Lo`, `HiFi2..4`), `SrcRoute` (`Tf32FromFp32`, `Bf16FromFp32`, `Bf16FromBf16`), chunk planning (`plan`, `chunk_fits`), `matmul_requirements`, `tilize_f32` / `detilize_packed`. |
| `datapath` | Unpacker/packer configuration for flat FP32 runs; thread state reset. |
| `link` | `Link`, `Mover`, `discover`: chip-to-chip moves over Ethernet with the `eth_e1` image. |
| `shard` | `Fabric`, `Chip`: a matmul split along `N` across cabled chips, bit-identical to one chip. |

## Many tiles

A session with `TileChoice::Count(n)` or `All` runs one unit (resident roles + data
mover) per surviving tile. A GDDR op is a set of independent jobs dealt round-robin
and run in waves; no job's arithmetic depends on where it runs, so results are the
same bits for any tile count. Callers pick the count through `burn_tt::Topology`
(`TT_TILES`, `tt-mnist --tiles`).

## Test

```bash
cargo test -p tt-kernels
```

Unit tests here: the L1 planner, op-record expansion against the list builders it
replaced, job dealing over units, list batching, matmul planning. Device gates for everything in this crate are in
`tt-tests` (`step8` to `step21`).

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
