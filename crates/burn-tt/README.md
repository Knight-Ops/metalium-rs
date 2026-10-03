# burn-tt

A [Burn](https://burn.dev) backend (`burn-backend` 0.21.0, pinned exactly) for
Tenstorrent Blackhole. Every op forwards to `burn-flex` on the host except the ones
it runs on the card. Shippable: no simulator dependency (the simulator engine used
by the gates lives in `tt-tests`).

## What runs on the device

With an engine that keeps tensors in GDDR (`KmdEngine`, and the ttsim engine in
`tt-tests`), on 2-D F32 tensors:

| Burn op | On the card as |
|---|---|
| `float_matmul` | `Session::matmul_dram` (operands may be transposed views) |
| `float_add` / `sub` / `mul`, `float_mul_scalar` | SFPU element-wise (`tt_kernels::kind`); `add` of a `[1, n]` row broadcasts (`ADD_ROW`) |
| `relu`, `relu_backward` | `RELU`, `RELU_BACKWARD` |
| `float_sum_dim(0)` | SFPU sum over rows, in Flex's summation order |
| `float_transpose` / `float_swap_dims` (2-D) | a view: same buffer, read transposed |
| `float_slice` of whole rows on 32-row bounds | a view, no copy |

Element-wise ops go to the device only when an operand is already there. A rank-N
`float_matmul` whose right side has no real batch (a `Linear` over `[b, s, d]`) folds
the batch into the rows and runs as the 2-D product; so do `linear_weight_backward` and
`linear_bias_backward`. The table above is the first ops; `hardware-coverage.md`'s Burn
op tables are the full list, and `TT_REPORT=1` says what a given model ran where. Any F32
`float_matmul` the above does not cover (batched, or an engine without GDDR such as
`Topology::Cards`) still runs on the card, staged from the host (`Engine::matmul`).
Everything else downloads its inputs once and runs on Flex.

The hand-written ops are `OVERRIDDEN` in `xtask/src/gen_burn.rs`; the forwarding
of every other op is generated into `src/generated/delegate.rs` by
`cargo xtask gen-burn-delegate` (never hand-edit; CI runs `--check`).

## Key types

| Item | What it is |
|---|---|
| `TtBackend`, `TtDevice { chip }` | The backend marker and a chip by index. |
| `TtTensor` | A shared cell with a lazily filled host copy (Flex) and device copy (GDDR buffer). |
| `attach(device, factory)` | Starts the device's server thread and runs the `Engine` factory on it; returns an `AttachGuard`. Attaching a chip twice is refused. |
| `Engine`, `kmd_engine`, `kmd_mesh_engine` | What a device can do; the silicon engines (one card on a `Session`, or a `Fabric` of cards). |
| `Topology`, `attach_topology` | `Single { card, tile }` or `Cards { .. }`; the one call training code makes. |
| `Topology::from_env` (`TT_TOPOLOGY`, `"0"` or `"0,1"`), `tiles_from_env` (`TT_TILES`, `n` or `all`) | Environment-driven choice, used by the `tt-tests` silicon harness. |
| `tensor_traffic`, `device_traffic`, `record_transfers` | PCIe traffic accounting. |

## Test

```bash
cargo test -p burn-tt
```

Host only: `tests/delegation.rs` checks every non-device op is Flex's answer byte for
byte, `tests/server.rs` drives the server with a host engine. Device behaviour is
gated in `tt-tests` (`step11_burn`, `step19_eltwise`, `step20_many_tiles`,
`step12_mnist`).

## Environment variables

Every variable the workspace reads, by who reads it. burn-tt's own switches
(`TT_PIPELINE`, `TT_SCATTER`, `TT_HOST_DMA`, `TT_TILIZE`, `TT_TOPOLOGY`, `TT_TILES`) refuse a
value they do not accept, with an error naming it; `TT_BATCH` treats anything but
`0` as on, `TT_EXACT` anything but `1` as off, and `TT_SILICON_DEVICE` an
unparsable value as `0`.

### burn-tt and the session (what a training or inference run sees)

| Variable | Values (default first) | Effect |
|---|---|---|
| `TT_PIPELINE` | `1`, `0` | Ops overlap a tile's data moves with its compute where that pays (`Session::set_pipeline`): GDDR matmuls, element-wise ops and reductions. `0` runs every block one after another. Same bits either way. |
| `TT_SCATTER` | `b`, `nc` | Where pipelined ops write their outputs out from: each tile's RISCV B (with its gathers), or RISCV NC, writing on NoC 1 while B gathers (`Session::set_scatter_mover`). `nc` pays for write-heavy ops on one or two tiles and costs host time on many. |
| `TT_HOST_DMA` | `1`, `0` | Tensors cross PCIe by the card's own DMA through a pinned 1 GiB hugepage (`Session::set_host_dma`), or with `0` by the host's stores and loads through a BAR -- uncached under VM passthrough, ~100x slower. Without a free 1 GiB hugepage the session uses the BAR and says so once. |
| `TT_TILIZE` | `host`, `card` | Where tensors take the tile layout on their way to the card's GDDR and lose it on the way back (`Session::set_tilize`): the host's cores, or each tile's data mover. Burn sees row-major data either way, with the same bits. `host` is faster at every size measured: the host copies the rows into pinned memory either way, at about the cost of tilizing them, and a mover tilizes ~2.6 us a tile. |
| `TT_BATCH` | `1`, `0` | Ops are queued on the tiles' movers and synced only when the host needs a result (`Session::set_batching`); `0` waits for every op. |
| `TT_EXACT` | unset, `1` | `1`: only ops that reproduce `burn-flex`'s bits exactly run on the card; the approximations held to derived bounds (division, `exp`, `log`, sums over columns, softmax) run on the host. For runs checked against a host golden. |
| `TT_PROFILE` | unset, a path | Records a device-side profile of everything an attachment runs and writes it as Chrome trace JSON on detach (`{chip}` in the path becomes the card). Silicon only. |
| `TT_REPORT` | unset, `1` | `1`: print the per-op report at exit (`burn_tt::report`): for every op, calls, results made on the device and on the host, bytes downloaded, uploaded and host-staged, and whether `burn-tt` implements it (`tt`) or Flex runs it (`flex`). The worklist for a new model. |
| `TT_STRICT` | unset, `1` | `1`: a download a host op causes, or a host-staged matmul, panics with the op's name (`burn_tt::set_strict`; one thread's block: `burn_tt::strictly`). Reading results back (`into_data`) is never one; `burn_tt::host_ok(..)` marks intended host work. |
| `TT_TRACE_FALLBACK` | unset, any | Prints a backtrace whenever a device tensor is downloaded for a host op: the way to find what still crosses PCIe. |
| `TT_TOPOLOGY` | unset, `0`, `0,1` | Which card(s) `Topology::from_env` attaches: one card, or several sharing each matmul. Read by callers that use it (the `tt-tests` harness), not by `attach` itself. |
| `TT_TILES` | unset, `n`, `all` | How many Tensix tiles a card computes on (`tiles_from_env`). Read by callers that use it (the `tt-tests` harness; `tt-mnist` takes `--tiles` instead). |
| `TT_ELTWISE` | must be unset | Retired: element-wise ops always run on the SFPU. Set, it is refused so a script that still sets it finds out. |

### Simulator and test harness

| Variable | Default | Effect |
|---|---|---|
| `TT_TTSIM_LIB` | `vendor/libttsim_bh.so` | The simulator library ttsim-backed tests load. |
| `TT_TTSIM_LIB_X2`, `TT_TTSIM_LIB_X4` | `vendor/libttsim_bh_x2.so`, `_x4.so` | The two- and four-chip builds the multi-chip gates load. Pointing one at the single-chip build checks those gates are not vacuous. |
| `TT_SILICON_DEVICE` | `0` | Which `/dev/tenstorrent/N` the silicon tests use (`cargo xtask silicon --device` sets it). |
| `TT_BLESS` | unset | `1`: `step12_mnist`'s ttsim run rewrites the loss golden the silicon run must reproduce. |
| `TT_ISA_DOCS` | `vendor/tt-isa-documentation` | A local checkout of the ISA documentation for the `xtask` generators; still checked against the pinned digest. |
| `STEP26_CASE` | unset | Runs only the named case of `step26_sfpu_isa`. |
| `STRESS_SECS`, `STRESS_TILES`, `STRESS_NC` | 60 s on silicon; `1,8`; unset | The soak tests' length, tile counts, and (`stress_pipeline`) scatters on NC. |

### Benchmarks (`cargo xtask bench`)

| Variable | Read by | Effect |
|---|---|---|
| `SWEEP_TILES`, `SWEEP_NC` | `silicon_bench_path` | The pipeline sweeps' tile counts; `SWEEP_NC` compares pipelined B-only against scatters on NC. |
| `PIPELINE` | `silicon_bench_path` | `1`: the one-unit path benchmarks run pipelined. |
| `BENCH_MOVER` | `silicon_bench_memory` | `nc`: the single-tile mover benchmarks run on RISCV NC. |
| `AGG_TILES`, `AGG_CAP`, `AGG_LEN`, `AGG_MIXES` | `silicon_bench_memory` | The card-wide GDDR benchmarks' tile counts, in-flight cap, entry size and read/write mixes. |
| `COPY_ENTRY` | `silicon_bench_memory` | The entry size `copy_pipeline` moves a block in. |
| `SOFTMAX_SIZE`, `SOFTMAX_DEVICE_ONLY` | `silicon_perf` | The softmax benchmark's shapes; device time only. |
| `NO_FENCE`, `NO_LANDING_WAIT` | `silicon_eth_bench` | Drop the zero-fill fence and the landing wait (to measure what they cost). |
| `SRC_X`, `SRC_CHIP`, `QUEUE` | `probe_eth` | The Ethernet probe's source tile, chip and queue. |

## Gotchas

- **Device errors panic.** Burn ops return tensors, not results, so a device op on
  an unattached device or a failed run panics. It never silently falls back.
- `TT_PROFILE` records each tile's mover lists, entries and records, and its role
  runs, by the tile's own cycle counter (Perfetto, `chrome://tracing`). Silicon
  only: ttsim does not model the timestamper's event stream.
- `Topology::Cards` keeps nothing in GDDR yet (matmuls only, host-staged) and
  computes on one tile per card; `on_tiles` refuses more.
- Random ops run on Flex, seeded through `Flex::seed`.
