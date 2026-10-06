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
| `float_add` / `sub` / `mul`, `float_mul_scalar` | SFPU by default; explicit matrix mode uses `ELWADD`/`ELWSUB`/`ELWMUL` with RHS row, column or scalar broadcasts |
| `relu`, `relu_backward` | `RELU`, `RELU_BACKWARD` |
| `float_sum_dim` / `float_mean_dim` / `float_max_dim` | all F32 axes, including ragged rank-N views; SFPU reductions with native repacking |
| `float_sum`, `float_mean` (rank one or more) | Native full F32 reduction, bounded column chunks followed by the chunked row sum; mean divides on the SFPU |
| `float_transpose` / `float_swap_dims` / `float_permute` | strided views at any rank; downstream native copies use whole-tile moves or ragged word repacking |
| `float_argmax` / `float_argmin` | native selection, first tie/NaN; rank-N F32, I32 indices, axes up to 2^23 |
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
intermediates on the device. Batched matmuls also partition products across cards; convolution and attention
reuse this path, including backward products. Elementwise work and reductions
remain on chip 0. Distributed trace capture is unsupported.
The two-chip and four-chip MLP gates match the native single-chip golden and
reject intermediate host transfers and staged computation.

Native implementations are `OVERRIDDEN` in `xtask/src/gen_burn.rs`. Dispatch and
explicit unsupported methods are generated into `src/generated/ops.rs` by
`cargo xtask gen-burn-ops` (never hand-edit; CI runs `--check`).
`cargo xtask check-no-flex-in-backend` guards dependency independence.

Native Conv2D supports groups/depthwise, stride, padding, dilation, bias and
F32/BF16 gradients. Conv1D uses Burn's singleton-spatial composition; transposed
Conv2D supports output padding below the maximum of stride and dilation. Unfold and stepped
slice assignment support native copies. Gather/select accept resident I32 indices
on arbitrary axes; float scatter/select-add preserve logical duplicate ordering. I32 updates wrap
and Boolean scatter/select-OR runs natively with resident indices.
BF16 convolution and scatter accumulate in F32 until the output boundary.

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

## Explicit matrix elementwise mode

SFPU remains the default. Configure one card before attaching its engine:

```rust,no_run
use burn_tt::{attach_topology_with_elementwise, ElementwiseMode, Fidelity,
    SrcPrecision, SrcRoute, Topology, TtDevice};
let device = TtDevice::new(0);
let guard = attach_topology_with_elementwise(
    device, Topology::single(0), SrcRoute::Tf32FromFp32, Fidelity::HiFi4,
    ElementwiseMode::Matrix { precision: SrcPrecision::Tf32, fidelity: Fidelity::HiFi4 },
)?;
# Ok::<(), burn_tt::EngineError>(())
```

`kmd_engine_with_elementwise` and the consuming
`KmdEngine::with_elementwise_mode`/`DramBuffers::with_elementwise_mode` builders
provide the same selection for custom factories. Custom engines expose it through
`Engine::elementwise_mode`. Attachment snapshots it once; floating arithmetic
keeps asynchronous dispatch. Mesh opt-in fails explicitly. Configuration cannot
be changed through an attached engine's public dispatch interface.

Floating add/subtract/multiply and their scalar forms select this route before
BF16 widening. F32 operands use TF32 or BF16 Src truncation; physical BF16 operands
stay packed and always use BF16 Src. Scalar BF16 buffers use the standard resident
BF16 conversion. Outputs accumulate/store F32 and a BF16 result narrows once at
its output boundary. Integer operations and other families keep their existing
routing. Equal shapes and RHS row/column/scalar broadcasts are supported, including
materialized resident views; other stored geometry is refused.

This mode permits reduced precision and the measured matrix special-value
contract. Add/subtract align at a shared 10-fraction-bit quantum even with BF16 Src;
TF32 multiply ignores SrcA's last fraction bit. Fidelity changes multiplication
only. Tested zeros/subnormals normalize to positive zero, tested signed NaNs act
like signed infinities against finite nonzero values, and opposite infinities
have non-IEEE outcomes. See `docs/learnings/silicon-operating-notes.md` and step90
for exact scope and bounds. Native RHS broadcasts read the original tile and
select its SrcB face/row directly, without an expanded GDDR tensor. Resident
views still use their existing materialization path. Benchmarks show large
broadcast improvements but do not justify changing the default.

`OpStat::matrix_eltwise` reports matrix dispatches. The numerical gates and the
role-stream instruction audit establish execution; this field is not a hardware
instruction counter. Ttsim covers packed inputs/F32 outputs; its existing PACR
`0x105` refusal leaves BF16 result narrowing and BF16 Burn gradients silicon-only.

## New Tensix primitives

Direct F32 `prod`/`prod_dim` support negative inputs and zero. Full products
reduce logical axes in descending order; dimensional products use the declared
SFPU accumulation/fold order. Boolean `any`/`all` reduce canonical bits directly.
Inclusive F32 `cumsum`/`cumprod` traverse logical indices in order, carrying
prefixes between tiles. Arbitrary-axis scans and arg-reductions use native word
repacking; flip and stepped slices (including reversal) copy on the card.

I32 add/sub/mul wrap modulo 2^32; signed comparisons never convert to F32.
Bitwise shifts mask counts modulo 32, with arithmetic right shift. F32 `round`
is ties-even; floor/ceil/trunc preserve special values and signed zeros.
F32-to-I32 truncates and saturates, with NaN mapped to zero. These programs do
not use SFPSTOCHRND's differing rounding/format semantics.

Dedicated LayerNorm/RMSNorm gates validate existing Burn compositions without
introducing a fused API. Gates `step69`–`step72` pass the simulator and both
cards (`target/silicon/1791145571.log`). The pinned `Autodiff` backend still inherits
Burn's log/exp product default, so negative-valued direct products are currently
an eager TtBackend capability. Its cumprod backward also has Burn's documented
zero-input limitation; nonzero scan gradients are gated natively.

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

Constructors and compute support F32, BF16, I32 and Bool as implemented by
the native primitives. F32/BF16 report accelerated arithmetic; Bool reports storage
and logic arithmetic. I32 retains the conservative `Storage` capability flag
until general integer tensor coverage is complete; its implemented native
operations include conversion, wrapping arithmetic, bitwise ops, signed
comparisons, shifts, wrapping sum/product and signed min/max reductions.
Checked integer division/remainder and integer mean are native and simulator-gated;
zero divisors report device DOMAIN errors. Card-0 silicon gates pass (`1791232718`).
F16, other integers and quantized tensors are unsupported.

BF16 uses 2112-byte physical slots with two-byte datums. Raw upload/download,
views and layout copies preserve payloads. Device casts round ties-even, quiet
NaNs and flush BF16 subnormals to signed zero. Packed rank-two matmul reads BF16
directly and accumulates in F32; most other floating-point operations widen on
Tensix, compute through native F32 primitives, then narrow at the operation
boundary. Operands must share a dtype. BF16 mesh execution uses resident Ethernet
execution after native device widening; network operand storage is currently F32.
Packed single-card products support K continuations, batches and broadcast views.
Pooling geometry/index metadata is replayable, enabling F32 and BF16 pool traces.
I32 sum/product and signed min/max reductions are native. Axis integer mean
wraps its sum before checked division by the logical count.
Burn BF16 arithmetic is silicon-gated because ttsim refuses late narrowing.

Native NCHW average/adaptive pooling and max pooling with resident spatial indices
support overlap backwards. BF16 averages use GAPOOL; F32 averages and general max
use SFPU to retain their numerical contracts. All-padding windows fail explicitly.
Pooling geometry constants use replayable metadata, enabling trace capture. The GMPOOL
block API is opt-in. `step74`–`step78` cover these paths; see
[the implementation record](../../docs/plans/tensix-next-features.md) for runs and limits.
BF16 currently saves storage rather than time: the first packed gather path is
slower than TF32 on the measured MNIST GEMMs.

Burn's logical dtype and shape belong to `TtTensor`; the engine owns the physical
device layout. Host byte counts are not device allocation sizes. Tenstorrent
BFP2/BFP4/BFP8 and their `a` variants are separate physical tile formats with
shared exponents, rather than aliases for Burn's scalar dtypes. Their layout and
decoders already exist in `tt-isa::tile`, but backend storage and compute support
need separate gates. See [the cutover and dtype backlog](../../docs/plans/burn-native-cutover.md).

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
- Native argmax and argmin support rank-N F32 input, I32 output, and reduced axes
  of at most 2^23 elements, with first-tie and first-NaN semantics.
- Random construction uses independent host RNG streams per `TtDevice`, seeded
  through `TtBackend::seed`. Draws are reproducible on this backend; matching
  Flex's random sequence is not required. Unseeded streams start at seed 0.

General reductions and K-blocked resident matmul have simulator gates in
`step67_general_reduce` and `step68_k_block_matmul`; both-card silicon validation
is pending. K continuations reload FP32 accumulators and retain the original
product order. Supported batched layouts retain their tile-alignment rules.

Native rank-four attention preserves pinned Burn's scale/softcap, Boolean and
bottom-right causal masks, additive bias, and zero fully-masked rows. Forward
BF16 computes its intermediates in F32 and narrows once at the output; autodiff
retains Burn's primitive composition. Ragged F32 batches materialize on device.
Mesh batches dispatch each product through distributed output-column partitions.
`step83` covers F32 simulator forward, analytic gradients, single-card traces and
two-chip forward. All six attention silicon gates pass (`1791232718`), including
BF16 resident training on card 0 and forward execution on cards 0 and 1.
Release benchmarks and broader attention acceptance remain outstanding.
Convolution and general resident-index routing remain unsupported.
