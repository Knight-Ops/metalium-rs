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

Missing convolution, FFT, sorting, quantized compute, integer arithmetic, and
other required methods fail explicitly. Unsupported shapes of implemented
operations also fail explicitly. Mesh engines retain buffers in chip 0's GDDR. Rank-two matmuls split output
columns across chips and move tile slots over Ethernet without host tensor
staging. Other training primitives and batched matmuls execute on chip 0. The
two-chip and four-chip sharded MLP gates reproduce the native golden and reject
intermediate downloads, host arithmetic and staged computation. Direct and
relayed links have native residency and numerical gates.

`tt-mnist` trains and evaluates with native argmax and a bit-preserving I32
column-to-vector reshape. Prediction indices are explicitly read back for
application accuracy bookkeeping. Native argmax and argmin preserve first ties and first
NaNs for rank-one/two F32 input with I32 output and axes up to 2^23 elements.

Boolean equality, Boolean-to-F32/I32 conversions, and I32/Boolean expansion
run natively. Boolean-to-F32 also enables Burn's default float `any`/`all`
through the supported dimensional and full sums. `float_permute` creates
strided views through dimension swaps; downstream operations retain the
existing whole-tile materialization limits. `step66_burn_small_ops` checks
parity, residency and strict-mode execution and belongs to the silicon smoke
suite. Burn's minimum defaults now compose argmin and gather, retaining the
existing gather axes and signed-zero limitation.

## Tenstorrent BFP formats

Support for BFP2/BFP4/BFP8 (often called BF2/BF4/BF8) and their `a` variants is
a separate milestone. Logical Burn dtype is not a physical device tile format.
The current `Cell` records logical dtype and shape; `Engine` and its device
buffers own physical layout and allocation. `HostBuffer` contains ordinary
row-major host bytes. Its byte size must never determine a BFP device allocation.

Build on the existing `tt-isa::tile::L1Format`, `TileImage`, bit addressing,
exponent-section layouts and decoders. Before advertising backend support:

1. Add a device storage descriptor for physical format and shared exponent
   layout, independently of the logical Burn dtype and accumulator precision.
2. Implement packing, unpacking, staging, readback and format conversion using
   the physical descriptor. Preserve logical shape and ragged edge semantics.
3. Gate device allocation sizes, exponent groups, rounding, zero, extremes and
   ragged tiles against the ISA model, then validate the same cases on silicon.
4. Enable matmul and subsequent primitives with explicit input/storage/output
   formats and accumulation policy. Add accuracy bounds appropriate to each
   format and model gates using identical logical inputs.
5. Decide the public Burn mapping separately: reduced physical storage of an
   F32 tensor versus a quantized primitive. Burn quantization schemes must not
   be assumed to match Tenstorrent block exponent semantics.

Until those gates pass, BFP formats are not reported as supported by `burn-tt`.
