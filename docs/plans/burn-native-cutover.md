# Native Burn cutover

`burn-tt` owns its tensor storage, constructors, seed state, dtype reporting and
readback. Its normal, build and dev dependency graphs contain no `burn-flex`.
Flex remains in the workspace for external reference comparisons, including
`tt-tests` and `tt-mnist --host`.

`cargo xtask gen-burn-ops` reads only the pinned `burn-backend` operation traits.
Required methods dispatch to native implementations or fail with the operation,
shape and dtype. Burn's composed defaults remain in use; defaults that are bare
unsupported placeholders receive contextual failures. The generated methods are
the authoritative operation backlog. There is no Flex fallback or exact mode.

## Model gates

The linear gradient, MLP forward/training, and transformer training gates run
through native device arithmetic. Reports reject host arithmetic, intermediate
downloads and staged model compute. Input construction, host input layout
packing, index metadata and explicit scalar readback are permitted. MLP training
pins the native loss curve and a steady-state transfer budget; Flex remains an
independent numerical reference. Every restored primitive also has an operation
gate, including native scalar broadcasting and ragged transpose copies.

## Remaining operation coverage

Missing FFT, sorting, portable quantized compute, and
other required methods fail explicitly. Unsupported shapes of implemented
operations also fail explicitly. Mesh engines retain buffers in chip 0's GDDR. Rank-two matmuls split output
columns across chips and move tile slots over Ethernet without host tensor
staging. Distributed batched products and F32/BF16 module gradients are gated (step83/step88); the remaining training primitives execute on chip 0. The
two-chip and four-chip sharded MLP gates reproduce the native golden and reject
intermediate downloads, host arithmetic and staged computation. Direct and
relayed links have native residency and numerical gates.

`tt-mnist` trains and evaluates with native argmax and a bit-preserving I32
column-to-vector reshape. Prediction indices are explicitly read back for
application accuracy bookkeeping. Native argmax and argmin preserve first ties and first
NaNs for rank-N F32 input with I32 output and axes up to 2^23 elements.

Boolean equality, Boolean-to-F32/I32 conversions, and I32/Boolean expansion
run natively. Boolean-to-F32 also enables Burn's default float `any`/`all`
through native dimensional and full sums. `float_permute` creates
strided views through dimension swaps; downstream operations retain the
native whole-tile copies or ragged word repacking. `step66_burn_small_ops` checks
parity, residency and strict-mode execution and belongs to the silicon smoke
suite. Burn's minimum defaults now compose argmin and gather, retaining the
existing gather axes and signed-zero limitation.

## General reductions and K blocking

F32 sum/mean/max dimensional reductions accept arbitrary axes and ragged
rank-N views through native word repacking. Long sums/maxima carry the
unfolded accumulator between chunks. Ordinary and supported tile-aligned
batched resident matmuls now support K beyond one L1 block by reloading
FP32 partial accumulators in original product order. `step67`/`step68`
validate simulator execution, residency, numerical models, autodiff and
changed-input traces; both-card silicon passed in run `1791145571`.
Other reduction kinds remain separate work; ragged and strided batched matmul
is native (`materialized_batched_matmul`).

## Reduction and integer extensions

Direct products, raw-bit Boolean reductions, rank-N arg-reductions and inclusive
F32 scans are native (`step69`). Stepped F32 slices and flips keep scan backward
resident. LayerNorm/RMSNorm compositions have dedicated numerical, gradient and
residency gates (`step70`). I32 wrapping ALU/comparisons/shifts and deterministic
F32 rounding/I32 conversion are gated by `step71`/`step72`. Both-card validation
passed in run `1791145571`. Pinned Burn Autodiff's log/exp product default and zero-input
cumprod backward limitation are unchanged; see the backend README.

## BF16 storage and pooling (2026-10-05)

BF16 physical storage, native conversion and layout copies, direct packed rank-two
matmul, native F32 arithmetic adapters, normalization and NCHW pooling/backwards
are implemented. BF16 averages use GAPOOL; general max retains SFPU index/NaN
semantics. `step74`–`step78` validate resident computation on both cards; late
BF16 narrowing is refused by pinned ttsim, so those arithmetic gates are silicon-only.
Mesh BF16, pooling traces, compact packed gathers and packed K continuations
are now gated; batched BF16 products and integer reductions are resident.
Checked integer division/remainder and axis mean now pass `step82` in ttsim;
card-0 silicon gates also pass (`1791232718`). Native attention, ragged F32
batches and actual two-chip batched products pass the `step83` simulator and
silicon gates, including BF16 resident training. The remaining items are listed in `hardware-coverage-closeout.md`.
See `tensix-next-features.md` for the current handoff and validation evidence.
BF16 is slower on the measured MNIST GEMMs; see `tensix-next-features.md`.

## Tenstorrent BFP formats (delivered 2026-10-06)

BFP2/BFP4/BFP8 are explicit per-tensor storage of logical F32 through
`TensorStorageExt::{with_storage, storage_format}`. Physical allocation derives
from `TileImage`, never ordinary row-major host byte counts. Tensix performs
packing/unpacking and arithmetic; B/NC transfer bytes. Same-format ordinary
rank-two operands use packed matmul directly; mixed/transposed/batched products
widen on device through existing native scheduling.

Unary/scalar results preserve storage, binary arithmetic/matmul use the higher
input precision, and reductions return F32. Views preserving exponent groups
share storage; regrouping materializes decoded F32. Casts have identity backward;
gradients, master parameters and optimizer state remain F32. Fusion preserves
compression boundaries; parameter copies and changed-input traces retain
physical storage. Ordinary readback/serialization returns decoded F32.

Step91–96 validate physical layouts, numerical oracles, ragged edges, K reloads,
Burn propagation and actual MNIST MLP/CNN precision policies on both cards.
Full SMOKE passed 364/364 (`1791300502`), final BFP controls 28/28
(`1791301208`), and PRNG diagnostics 4/4 (`1791302401`). See
[the contract and completed checklist](mixed-bfp-storage.md) and the
[measured conversion costs and accuracy observations](../learnings/firmware-performance.md).
BFP `a` variants, INT8/UINT8, packed mesh transport and portable Burn
`QTensorOps` remain deferred. Burn random uses the existing per-device seeded
host construction; hardware distribution generation remains deferred despite
the now-reproducible diagnostic RISC-V seed-store path.

### Convolution and resident-index continuation (2026-10-05)

Native module convolution/unfold routes and gradients, resident arbitrary-axis
indices and raw slice assignment are now overridden/generated. Default cat,
repeat, Conv1D, embedding and autodiff compose these primitives. Mesh batched
matmul now partitions products, so convolution/attention forwards and gradients
execute on both cards. See tensix-next-features.md for gate run IDs and open
acceptance items; this does not close the entire backend inventory.

### M1 matrix elementwise continuation (2026-10-06)

- [x] Explicit `ElementwiseMode::Matrix { precision, fidelity }` before single-card
      attachment; constructors still default to SFPU. The mode snapshot preserves
      asynchronous dispatch and does not survive detachment/reattachment.
- [x] Native floating add/subtract/multiply and scalar dispatch intercept packed
      BF16 before widening, accumulate in F32 and narrow at the output boundary.
- [x] Equal-shape and RHS row/column/scalar geometry, resident views, ragged
      padding, trace holds and deferred frees; unsupported geometry/mesh mode
      fails explicitly. Integer and other operation families retain routing.
- [x] Step90 records matrix dispatches, audits actual role instructions and checks
      native arithmetic/analytic gradients with no intermediate downloads.
- [x] Final continuation acceptance is recorded in hardware-coverage.md: both-card
      targeted gates 24/24, full smoke 322/322, release baseline `1791255969`,
      workspace/Clippy/generator/shipping checks and unchanged MNIST golden.

Benchmarks and matrix numerical/special-value contracts are in the learnings
files. The opt-in permits reduced Src/alignment precision; no universal speedup
or IEEE special-value behavior is promised.
