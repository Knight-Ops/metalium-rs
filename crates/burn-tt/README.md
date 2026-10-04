# burn-tt

A [Burn](https://burn.dev) backend (`burn-backend` 0.21.0, pinned exactly) for
Tenstorrent Blackhole. Compute uses native Tenstorrent implementations; missing
operations, layouts, and dtypes fail with operation and tensor metadata.
`burn-flex` is an external validation backend and is absent from this crate's
normal, build, and dev dependencies. Shippable: no simulator dependency (the simulator engine used
by the gates lives in `tt-tests`).

## What runs on the device

With an engine that keeps tensors in GDDR (`KmdEngine`, and the ttsim engine in
`tt-tests`), supported operations include:

| Burn op | On the card as |
|---|---|
| `float_matmul` | `Session::matmul_dram` (operands may be transposed views) |
| `float_add` / `sub` / `mul`, `float_mul_scalar` | SFPU element-wise (`tt_kernels::kind`); `add` of a `[1, n]` row broadcasts (`ADD_ROW`) |
| `relu`, `relu_backward` | `RELU`, `RELU_BACKWARD` |
| `float_sum_dim` / `float_mean_dim` / `float_max_dim` | all F32 axes, including ragged rank-N views; SFPU reductions with native repacking |
| `float_sum`, `float_mean` (rank one or more) | Native full F32 reduction, bounded column chunks followed by the chunked row sum; mean divides on the SFPU |
| `float_transpose` / `float_swap_dims` / `float_permute` | strided views at any rank; downstream native copies use whole-tile moves or ragged word repacking |
| `float_argmax` / `float_argmin` | native selection, first tie/NaN; rank-one/two F32, I32 indices, axes up to 2^23 |
| `bool_equal` / `bool_equal_elem` | Boolean XOR/NOT or identity, native broadcasts |
| `bool_into_float` / `bool_into_int` | exact native 0/1 conversion to F32/I32 |
| `float_any` / `float_all` and dimensional variants | Burn defaults over comparisons, Boolean-to-F32 and supported native sums |
| `float_expand` / `int_expand` / `bool_expand` | native byte-preserving gathers and transposes |
| `float_slice` of whole rows on 32-row bounds | a view, no copy |

Supported element-wise ops upload inputs as needed and compute on the device. A rank-N
`float_matmul` whose right side has no real batch (a `Linear` over `[b, s, d]`) folds
the batch into the rows and runs as the 2-D product; so do `linear_weight_backward` and
`linear_bias_backward`. Reshapes and dimension swaps of a resident F32 tensor are
strided views of its buffer at any rank (`src/views.rs`): a batched matmul reads its
operands' blocks where they lie (attention's heads, `K^T`), and an op that needs a plain
matrix gets one by a native block copy or word repacking on the card. The table above is the first ops; `hardware-coverage.md`'s Burn
op tables are the full list, and `TT_REPORT=1` says what a given model ran where. Any F32
`float_matmul` the above does not cover (batched, or an engine without GDDR such as
a minimal custom engine) still runs on the card, staged from the host (`Engine::matmul`).
Full `sum`/`mean` always run natively: they upload host F32 inputs and use
native arithmetic. Their arithmetic is checked against
derived bounds, not guaranteed to match Flex's bits. Empty/rank-zero tensors,
non-F32 dtypes, engines without native reduction support and resident views
that cannot be copied on the card fail explicitly. Required methods without
a native implementation fail explicitly; Burn defaults compose our primitives.
Mesh engines retain buffers in chip 0's GDDR. Rank-two matmuls split output
columns across the fabric, moving tile slots over Ethernet, directly or through
relay tiles. Other primitives and full reductions run on chip 0, retaining
intermediates on the device. Batched matmuls currently compute on chip 0.
The two-chip and four-chip MLP gates match the native single-chip golden and
reject intermediate host transfers and staged computation.

Native implementations are `OVERRIDDEN` in `xtask/src/gen_burn.rs`. Dispatch and
explicit unsupported methods are generated into `src/generated/ops.rs` by
`cargo xtask gen-burn-ops` (never hand-edit; CI runs `--check`).
`cargo xtask check-no-flex-in-backend` guards dependency independence.

## Key types

| Item | What it is |
|---|---|
| `TtBackend`, `TtDevice { chip }` | The backend marker and a chip by index. |
| `TtTensor` | A shared cell with a lazily filled host copy (owned tensor bytes) and device copy (GDDR buffer). |
| `attach(device, factory)` | Starts the device's server thread and runs the `Engine` factory on it; returns an `AttachGuard`. Attaching a chip twice is refused. |
| `Engine`, `kmd_engine`, `kmd_mesh_engine` | What a device can do; the silicon engines (one card on a `Session`, or a `Fabric` of cards). |
| `Topology`, `attach_topology` | `Single { card, tile }` or `Cards { .. }`; the one call training code makes. |
| `Topology::from_env` (`TT_TOPOLOGY`, `"0"` or `"0,1"`), `tiles_from_env` (`TT_TILES`, `n` or `all`) | Environment-driven choice, used by the `tt-tests` silicon harness. |
| `tensor_traffic`, `device_traffic`, `record_transfers` | PCIe traffic accounting. |

## Test

```bash
cargo test -p burn-tt
```

`tests/essentials.rs` checks owned storage, constructors, transaction ordering,
seed reproducibility and contextual unsupported failures without hardware.
`tests/stale_buffer.rs` guards attachment and buffer lifetimes. External Flex
comparisons live in `tt-tests`, including `burn_server`, `step11_burn`,
`step12_mnist`, `step59_burn_transformer`, `step64_burn_native_broadcast`, and
`step66_burn_small_ops`.
The MLP and transformer gates reject host arithmetic and staged model compute.

## Dtypes and physical formats

Constructors and compute currently support F32, I32 and Bool as implemented by
the native primitives. F32 reports accelerated arithmetic; Bool reports storage
and logic arithmetic; I32 reports storage and conversion support. Integer
arithmetic is unsupported.
F16, BF16, other integers and quantized tensors are unsupported.

Burn's logical dtype and shape belong to `TtTensor`; the engine owns the physical
device layout. Host byte counts are not device allocation sizes. Tenstorrent
BFP2/BFP4/BFP8 and their `a` variants are separate physical tile formats with
shared exponents, rather than aliases for Burn's scalar dtypes. Their layout and
decoders already exist in `tt-isa::tile`, but backend storage and compute support
need separate gates. See [the cutover and dtype backlog](../../docs/burn-native-cutover.md).

## Environment variables

Every variable the workspace reads, by who reads it. burn-tt's own switches
(`TT_PIPELINE`, `TT_HOST_DMA`, `TT_TILIZE`, `TT_TOPOLOGY`, `TT_TILES`) refuse a
value they do not accept, with an error naming it; `TT_BATCH` treats anything but
`0` as on, and `TT_SILICON_DEVICE` an
unparsable value as `0`.

### burn-tt and the session (what a training or inference run sees)

| Variable | Values (default first) | Effect |
|---|---|---|
| `TT_EXECUTION` | must be unset | Retired: GDDR compute always uses resident B-reader/NC-writer ownership, including traces. Set, it is refused with migration guidance. |
| `TT_PIPELINE` | `1`, `0` | Ops overlap a tile's data moves with its compute where that pays (`Session::set_pipeline`): GDDR matmuls, element-wise ops and reductions. `0` runs every block one after another; streaming still uses NC and waits for its output release before slot reuse. Captures retain the chosen schedule. Same bits either way. |
| `TT_SCATTER` | must be unset | Retired: NC owns compute-region output writes on NoC1, even with `TT_PIPELINE=0`. Set, it is refused with migration guidance. |
| `TT_HOST_DMA` | `1`, `0` | Tensors cross PCIe by the card's own DMA through a pinned 1 GiB hugepage (`Session::set_host_dma`), or with `0` by the host's stores and loads through a BAR -- uncached under VM passthrough, ~100x slower. Without a free 1 GiB hugepage the session uses the BAR and says so once. |
| `TT_TILIZE` | `host`, `card` | Where tensors take the tile layout on their way to the card's GDDR and lose it on the way back (`Session::set_tilize`): the host's cores, or each tile's data mover. Burn sees row-major data either way, with the same bits. `host` is faster at every size measured: the host copies the rows into pinned memory either way, at about the cost of tilizing them, and a mover tilizes ~2.6 us a tile. |
| `TT_BATCH` | `1`, `0` | Ops are queued on the tiles' movers and synced only when the host needs a result (`Session::set_batching`); `0` waits for every op. |
| `TT_EXACT` | must be unset | Retired with the Flex cutover. Native numerical bounds apply; unset this variable. |
| `TT_PROFILE` | unset, a path | Records a device-side profile of everything an attachment runs and writes it as Chrome trace JSON on detach (`{chip}` in the path becomes the card). Silicon only. |
| `TT_REPORT` | unset, `1` | `1`: print the per-op report at exit (`burn_tt::report`): for every op, calls, results made on the device and on the host, bytes downloaded, uploaded and host-staged, and whether `burn-tt` implements it (`tt`) or the method is unsupported (`unsupported`). The worklist for a new model. |
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
| `STRESS_SECS`, `STRESS_TILES` | 60 s on silicon; `1,8` | The soak tests' length and tile counts. NC ownership is always enabled. |

### Benchmarks (`cargo xtask bench`)

| Variable | Read by | Effect |
|---|---|---|
| `SWEEP_TILES` | `silicon_bench_path` | The pipeline sweeps' tile counts; both arms use NC ownership. |
| `PIPELINE` | `silicon_bench_path` | `1`: the one-unit path benchmarks run pipelined. |
| `BENCH_MOVER` | `silicon_bench_memory` | `nc`: the single-tile mover benchmarks run on RISCV NC (the writer: read benchmarks skip). B is the default and only reads. |
| `BENCH_WRITE_NOC` | `silicon_bench_memory` | `0` (default) or `1`: single-tile write routing; `1` needs `BENCH_MOVER=nc`, since only NC writes on NoC #1. These diagnostic benchmarks bypass the session executor. |
| `AGG_TILES`, `AGG_CAP`, `AGG_LEN`, `AGG_MIXES` | `silicon_bench_memory` | The card-wide GDDR benchmarks' tile counts, in-flight cap, entry size and read/write mixes. |
| `COPY_ENTRY` | `silicon_bench_memory` | The entry size `copy_pipeline` moves a block in. |
| `SOFTMAX_SIZE`, `SOFTMAX_DEVICE_ONLY` | `silicon_perf` | The softmax benchmark's shapes; device time only. |
| `NO_FENCE`, `NO_LANDING_WAIT` | `silicon_eth_bench` | Drop the zero-fill fence and the landing wait (to measure what they cost). |
| `SRC_X`, `SRC_CHIP`, `QUEUE` | `probe_eth` | The Ethernet probe's source tile, chip and queue. |

## Gotchas

- **Device errors panic, at the next wait.** Device ops return before they run
  (asynchronous dispatch), so a failed run panics where a result is next waited
  for -- a download, a trace's replay -- naming the op that failed and the engine's
  error; the first failure on an attachment is reported by every later wait. An op
  on an unattached device panics at once. Nothing silently falls back.
- `TT_PROFILE` records each tile's mover lists, entries and records, and its role
  runs, by the tile's own cycle counter (Perfetto, `chrome://tracing`). Silicon
  only: ttsim does not model the timestamper's event stream.
- `Topology::Cards` uses resident buffers and one compute tile per card;
  `on_tiles` refuses more. Non-matmul primitives execute on chip 0.
- Native argmax and argmin support rank-one/two F32 input, I32 output, and reduced axes
  of at most 2^23 elements, with first-tie and first-NaN semantics.
- Random construction uses independent host RNG streams per `TtDevice`, seeded
  through `TtBackend::seed`. Draws are reproducible on this backend; matching
  Flex's random sequence is not required. Unseeded streams start at seed 0.

General reductions and K-blocked resident matmul have simulator gates in
`step67_general_reduce` and `step68_k_block_matmul`; both-card silicon validation
is pending. K continuations reload FP32 accumulators and retain the original
product order. Supported batched layouts retain their tile-alignment rules.
