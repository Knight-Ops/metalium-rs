# Tensix feature implementation record

2026-10-05. Model coverage first; BF16 is the new storage format.
Normalization keeps Burn's native compositions rather than adding fused kernels.

| Milestone | Implemented | Remaining completion work |
|---|---|---|
| Reduction breadth | Products, Boolean reductions, argextrema, sum/product scans; min/max scans with raw total ordering (`step79`); full and axis I32 wrapping sum/product and signed min/max (`step80`); observed carry/identity mutants | Remaining hardware coverage as tracked below |
| Normalization | Native LayerNorm/RMSNorm compositions and analytic gradient gates | Fusion deferred |
| Integer and rounding | Full-width I32 ALU, deterministic rounding/casts; opt-in hardware BF16/TF32 nearest, stochastic and toward-zero modes (`step81`) | Checked division/remainder and arbitrary-axis integer mean pass ttsim and silicon (`step82`); benchmark run `1791235277` |
| BF16 storage | Packed gathers, packed K continuations, batched/broadcast/view matmul, native adapters, two-card mesh execution, full MNIST accuracy run | Mesh transport still widens operands to F32; further performance tuning |
| FPU pooling | GMPOOL/GAPOOL block kernels, BF16 GAPOOL averages, native max/indices/backwards; F32 and BF16 pooling traces (`step78`) | Window staging is bounded/reused. GMPOOL differs from Burn and its packed ArgMax path exposes no indices; general pooling stays SFPU |
| FPU transpose | Instruction probes (`step73`), existing mover view materialization | 16×16 explicit Src transpose and eligible TF32 matmul preparation pass (`step87`). General payload-preserving transpose remains native raw copy; M3 stays partial |

Ordinary silicon validation now uses card 0. Use both cards when testing actual
mesh/Ethernet execution or investigating device differences, per the user's
2026-10-05 instruction. Earlier two-card logs remain historical evidence.

Shipping code remains native Rust. B copies/reorders datums and NC writes them;
all arithmetic executes on Tensix. New Burn overrides are maintained in
`xtask/src/gen_burn.rs` and generated with `cargo xtask gen-burn-ops`.

## Numerical and execution contracts

Products multiply directly, including negative values; they do not use log/exp.
Reduction identities are 1 for products/AND and 0 for OR. Ragged lanes are
masked with the identity before folding, and long axes preserve the unfolded
accumulator. Boolean storage and outputs remain canonical raw I32 0/1.

Scans are inclusive, in increasing logical axis order. Each tile continues
from the previous tile's last prefix. The next gather waits for NC completion;
it reads the prior result from GDDR rather than depending on persistent Dst
state. Scan padding is undefined. Trace replay and downstream reductions test
changed inputs and padding ownership. Arbitrary-axis routing constructs address
coordinates on the host, without downloading or calculating tensor values.

I32 multiplication reconstructs the full low 32-bit product using 16-bit
limbs and SFPMUL24. Arithmetic wraps modulo 2^32; signed shifts mask counts
modulo 32. Comparisons never convert through F32. Scalar immediates are raw
I32 bits. Rounding preserves signed zeros, infinities and NaN payloads;
F32-to-I32 truncates and saturates, with NaN mapping to zero. These deterministic
contracts use raw-bit SFPU programs rather than SFPSTOCHRND.

Gates use independent Rust integer/rounding results, analytic normalization
gradients, exact finite products/prefixes, and specified Blackhole arithmetic
or the SFPU interpreter for execution order and special values. Bounds are
derived beside the gates. Deliberately changing product identity, scan carry,
multiply reconstruction, ties-even routing, Boolean AND identity or square-root
routing was observed to fail its gate.

Pinned Burn's `Autodiff` wrapper implements full `prod` using its own log/exp
default. The direct-product guarantee applies to `TtBackend`'s eager overrides;
it does not repair that external wrapper. Its existing cumprod backward also
divides by input values, so zero-containing input gradients remain outside this
gate's guarantee. Nonzero cumsum/cumprod backwards execute natively.

TRNSPSRCB permutes reduced-precision Src datums, not arbitrary F32 words.
Its encoding retains Wormhole-only provenance until measured on Blackhole.
The BF16 no-transpose control in the new 512-datum probe disagreed before any
transpose; see the unresolved observation in `ttsim-divergence.md`. It is not
evidence of a Blackhole instruction defect and is not an enabled BF16 route.

## Validation and performance

Both p150a cards passed `step67`–`step73`: 58/58 cases, run
`target/silicon/1791145571.log`. BF16 storage and block pooling passed 4/4
cases in `1791147578.log`. Native Burn BF16, direct packed matmul and pooling
passed 10/10 in `1791158237.log`; normalization and expanded pooling passed
8/8 in `1791158730.log`. Final `step74`–`step78` validation passed 24/24 in
`1791160155.log`, including BF16 integer/Boolean casts, mixed F32-loss SGD,
trace holds, matmul-to-pooling state changes and exact max selection bits.
The tile-row view extension and halfword fill bounds passed both cards in
`1791160473.log`. Workspace unit/integration tests, separate documentation
tests, default/silicon Clippy, firmware RISC-V Clippy, generator checks and
shipping dependency checks pass. The five F32 MNIST training regressions pass
with the existing golden unchanged.
All gates are in SMOKE.

BF16 conversion rounds ties-even, quiets NaNs while retaining sign/high payload,
and flushes BF16 subnormals to signed zero. Raw BF16 transfer and layout copies
preserve every bit. Widening uses BF16 SrcA followed by MOVA2D; narrowing uses
the late packer, whose mode `0x105` the pinned simulator refuses. These conversion
and Burn BF16 arithmetic gates are therefore silicon-only. Raw storage and
packed matmul have simulator gates. Matrix accumulation and output are F32;
Burn rounds the output once to BF16. K-continuation and batched products retain packed BF16 operands. Mesh products
widen on device for the existing resident Ethernet route and narrow the result
once at the Burn boundary; mesh traffic is not yet packed BF16.

NCHW pooling supports padding, dilation, ceil mode, adaptive windows and overlap
backwards. F32 pooling retains native SFPU semantics. BF16 average pooling stages
16-lane GAPOOL chunks, accumulates in F32 and divides by the declared valid or
padded count before narrowing. Host work constructs coordinates and constants;
tensor values, indices and gradients remain resident. All-padding windows fail
explicitly. Max uses SFPU argmax followed by a raw-bit OR selection fold, so
values and indices select the same first tie/NaN and retain signed zeros and
NaN payloads. Ordinary arithmetic gather's zero/NaN canonicalization does not
apply to pooling selection.
The GMPOOL block API remains opt-in. Operation geometry and index constants are replayable metadata descriptors,
so F32 and BF16 pooling traces capture without tensor uploads and replay changed
inputs. Tensor values still execute entirely on device.

Repeat with the isolated runner:

```bash
cargo xtask silicon --release --device 0 --filter step67 --filter step68
cargo xtask silicon --release --device 0 --filter step69 --filter step70 --filter step71 --filter step72 --filter step73
cargo xtask silicon --smoke --release --device 0
cargo xtask bench --device 0 --filter bf16_mnist_matmul
```

The benchmark excludes uploads/readback from timing, synchronizes every run,
validates every output, warms up once and reports nine-run host medians. Record
conditions and run IDs in `firmware-performance.md`. Run `1791162075` measures
one Tensix, release, resident operands, HiFi4, pipeline off, one warmup and nine
validated host-timed samples on card 0: packed BF16 takes 141.092/53.499 us for
64×784×128 and 64×128×10, versus TF32 97.741/18.083 us. The larger BF16 product
improves about 4.7× from the earlier 661.742 us, but remains slower than TF32;
this does not establish an MNIST speedup.

## 2026-10-05 wrap-up and next starting point

New silicon evidence: packed K/ragged products plus pooling traces 7/7
(`1791162575`); batched BF16 1/1 (`1791163126`); actual two-card BF16 mesh 1/1
(`1791163264`); extremum scans 1/1 (`1791163532`); integer reductions 1/1
(`1791163872`); hardware precision modes 2/2 (`1791165365`). All ordinary gates
used card 0. Corresponding simulator gates passed where ttsim supports the mode.
The packed-gather unit test was watched failing with the wrong slot stride.
New scan/integer-reduction/hardware-rounding gates still need observed deliberate
mutations before treating their complete definition-of-done checklist as closed.

Full BF16 MNIST accuracy: card 0, four Tensix, release, one epoch, 60,000 training
and 10,000 test examples; 59,968 training examples processed in 937 steps.
Loss 2.3211 → 0.3781, test accuracy **91.82%**. This is an accuracy run, without a
matched F32 performance comparison. Log: `target/silicon/bf16-mnist-accuracy.log`.
All five F32 training regressions pass with the original golden unchanged;
transfer expectations now omit index metadata uploads that capture can replay.
Log: `target/silicon/tensix-mnist-regression.log`.

**Integer division/remainder is unfinished and disabled.** The full-width SFPU
program agrees with an independent I64 oracle in the interpreter, including
MIN/-1 wrapping and Python-sign remainder. The experimental NC `CHECK_FLAGS`
path correctly detects a zero divisor but also rejects valid tensor divisors
with DOMAIN=11 in ttsim. This is an unresolved implementation observation, not
a proven simulator divergence. The builder refuses execution, Burn overrides
are removed, and the two `step82` integration gates are explicitly ignored and
excluded from SMOKE. No silicon correctness is claimed. Resume by isolating
packed C_ROW status flags, checking their publication/lifetime and ragged lane
mapping, then rerun both simulator gates before enabling the route or testing
silicon. Draft protocol/firmware and builder code remain for this investigation.

Next: (1) resolve checked integer division/remainder and native integer mean;
(2) observe the new negative controls; (3) general GMPOOL semantics and window
staging; (4) resolve transpose control and integrate tensor routing. Norm fusion
remains deliberately deferred. The implementation plan is **not fully complete**.

Earlier workspace-wide validation above belongs to its recorded revision. The
wrap-up checks and any limitations are recorded below; do not infer that the
entire current workspace has been rerun from the historical statement.

Final wrap-up checks: host and firmware formatting pass; default and silicon
workspace Clippy pass with `-D warnings`; separate RISC-V firmware Clippy passes;
Burn generator `--check` and both shipping dependency checks pass. `tt-isa` and
`tt-kernels` library tests pass 310 cases (one existing ignored test). Selected
simulator gates `step77`–`step81` pass nine cases; `step82` has two explicitly
ignored unfinished cases. The entire workspace test suite was not repeated at
wrap-up; the five training regression cases already passed earlier this turn.
Logs: `target/silicon/tensix-wrap-{unit,gates,clippy,silicon-clippy,firmware-clippy}.log`.

## 2026-10-05 continuation: simulator and silicon validation

This section supersedes the disabled-division starting point above. The former
valid-input DOMAIN rejection did **not** reproduce with current code. The original
failure's cause has not been established; no simulator defect is claimed.
`step82` separately checks C_ROW production and packing for every physical face,
then exercises the full SFPU body. Status now uses the layout's separately declared
C scratch slot (Requirements lifetime covers gather, kernel and NC validation),
instead of overwriting B. NC checks only logical rows/columns after pack completion.
The gates pass valid full-width divisors, MIN/-1, scalar and row/column broadcast,
face-boundary zeros and ragged padding. Native Burn tensor/scalar division and
Python-sign remainder are restored. Axis mean uses wrapping sum then checked
logical-count division; full mean retains Burn's composition. All five step82
simulator tests pass and are in SMOKE. All five gates also pass on card 0
in silicon run `1791232718`.

Observed negative controls, restored before verification:
- `step79`: replace prior-tile carry with zero; gate fails.
- `step80`: replace I32 product padding identity 1 with 0; gate fails.
- `step81`: configure the hardware rounding source as ZERO; output comparison fails.
- `step83`: remove bottom-right causal offset; analytic result fails (0 vs 12).
Logs: `target/silicon/tranche-mutant-{scan-carry,integer-identity,rounding-mode,attention-causal}.log`.
The additional silicon mode-bit mutant changed requested Nearest to TowardZero;
`step81` failed its Bf16/Nearest comparison in run `1791233262`. Restoring the
mode passed in run `1791233389`. ttsim refuses other rounding modes.

Ragged F32 batches now use the native materialization route already used by packed
BF16. Each matrix is repacked on device, preserving its parent's padding, and
uses existing K blocking. Results stay F32 until the final storage conversion.
Eligible aligned batches keep the existing direct route. Host inputs to transpose
establish their device storage before constructing a view, so pinned attention's
autodiff composition does not stage that transpose on the host.

`ModuleOps::attention` is native: QK^T, scale, optional positive softcap,
Boolean/causal masks, additive bias, NaN-safe softmax and multiplication by V.
Causal geometry is replayable metadata; true masks positions and unequal sequence
lengths align bottom-right. Fully masked rows produce zeros. BF16 forward widens
Q/K/V and optional bias and narrows only the final output. Pinned Burn autodiff
remains composed from primitives (its BF16 intermediate boundaries remain Burn's).
Five step83 simulator gates pass: analytic causal/masked ragged heads, custom
scale/softcap/bias against an independent f64 result with a derived bound, analytic
Q/K/V/bias gradients, changed-value trace replay after temporary frees, and two-chip
aligned/ragged forward execution. BF16 forward with bias and resident BF16 SGD with
F32 loss are silicon-only gates; both pass on card 0 in `1791232718`.
All six step83 gates pass in that run and are in SMOKE.

Mesh aligned batched products now materialize each block and call the existing
Fabric output-column partitioner, rather than computing only on chip 0. Ragged
products take that same distributed route through ordinary materialized matmul.
Mesh trace capture remains unsupported. The two-chip simulator uses widths 64 for
both products, requiring two nonempty partitions. Per-chip arithmetic/Ethernet
telemetry and mesh gradient comparison were initially outstanding; step88 and
the later continuation sections record their validation. The forward gate
passes on actual cards 0 and 1 in silicon run `1791232718`.

The first sandboxed silicon attempt could not see `/dev/tenstorrent/0`.
That was a sandbox visibility limitation, not unavailable hardware. Repeating
with elevated access via the isolated runner executes successfully: run
`1791232718`, **11/11 passed**, ordinary gates on card 0 and mesh forward on
cards 0 and 1. At that initial run, convolution, resident indexing, slice assignment, bounded
pooling, GMPOOL semantics and tensor transpose integration were still pending.
Subsequent sections record their implementations and measured limitations.

Active acceptance checklist:
- [x] Separate packed-domain diagnostic and logical-domain simulator gates.
- [x] Native checked division/remainder and arbitrary-axis integer mean routing.
- [x] Integer column broadcast, scalar zero and changed-divisor/domain trace on two tiles.
- [x] Integer silicon validation on card 0 (`1791232718`).
- [x] Integer release benchmark (`1791235277`).
- [x] Observe scan carry and integer reduction identity mutants.
- [x] Observe hardware rounding-mode mutant on silicon (`1791233262`); restored gate passes (`1791233389`).
- [x] Native ragged F32 batched materialization and attention forward/analytic gradients.
- [x] Single-card F32 causal attention trace replay and two-chip simulator forward.
- [x] Attention card-0 forward/gradient/F32 trace and BF16 resident training (`1791232718`).
- [x] Attention external oracle, expanded masks/views/large-K, BF16 one/two-tile trace coverage (`1791249159`, `1791249486`).
- [x] Actual two-card aligned/ragged attention forward (`1791232718`).
- [x] Per-chip distributed telemetry, F32/BF16 mesh gradient/reference gates and release benchmarks (`1791239948`, `1791240702`, `1791239400`).
- [x] Bounded pooling, resident indexing and slice assignment.
- [~] Src transpose: validated 16×16 conversion route integrated into eligible matmul; general payload-preserving tensor contract remains on native copies.
- [x] Planned convolution family, gradients, unfold, resident training, one/two-tile traces and distributed reference gates (`1791249486`).

Continuation verification: `cargo fmt --all --check`, default/silicon workspace
Clippy with `-D warnings`, separate firmware RISC-V Clippy, Burn generator
`--check`, and both shipping dependency checks pass. Silicon workspace
`cargo test --workspace --features tt-tests/silicon --no-run` compiles the gates.
The five MNIST training regressions pass (308.49 s) with the golden unchanged.
Targeted step82/step83 pass ten simulator cases; the two-chip attention gate also
passes after releasing temporary operand buffers between batch products.
The full default workspace test run passes, including documentation tests.
The final integer trace/scalar/column extensions were also rerun in the targeted
ten-case step82/step83 run; the final mesh operand-lifetime/bounds refinement
passes its two-chip gate. Logs are under
`target/silicon/tranche-{workspace,gates,mesh-attention,silicon-build,clippy,silicon-clippy}.log`.

### Transpose control and bounded pooling continuation (2026-10-05)

The signed BF16 zero-transpose control passes both cards, followed independently
by one transpose and its inverse: `step73`, 4/4 in run `1791233413`. ttsim
still fails before any transpose, even with the math prelude removed; the
silicon BF16 probe is explicitly gated by `silicon`. TF32 remains simulator-covered.
This does not enable a tensor transpose kernel or close M3.

F32 pooling now stages at most 32 equal-sized windows per descriptor group.
BF16 average pooling reuses one 16x16 packed staging allocation, releases consumed
F32 partials after each continuation, and retains only final group outputs.
Arithmetic and overlap backward order are preserved. Additional two-tile traces
cross both output batching boundaries and the 16-element continuation boundary.
Validation results for this staging change are recorded after the gates finish.

Pooling staging validation: `step75`/`step78` passed both cards, 18/18 in
`1791233597`, including both new two-tile traces. GMPOOL signed-zero,
subnormal, signed NaN, infinity and value-tie characterization independently
matches its integer magnitude model in ttsim and both cards (`1791233808`, 2/2).
General Burn max pooling stays on its exact SFPU value/index route. Hardware
index encoding/first-eight-row limitations still need a separate gate.

### Shared static copies and convolution implementation (2026-10-05)

Multi-source repack assembles raw F32/I32/Bool words or packed BF16 halfwords
on device. Every source coordinate is checked before dispatch; ragged output
padding is declared undefined and parent claims remain untouched. Slice assignment
now covers arbitrary logical axes and stepped/permuted views. Native empty
initialization enables Burn's composed cat/repeat without host staging, including
physical BF16 zeros. `step84` preserves signed zeros, subnormals and NaN payloads,
checks I32 extremes/Bool, parent immutability and analytic gradients: all three
simulator gates and both cards pass (`1791233990`, 6/6). Dynamic resident indices are implemented below; broader acceptance stays open.

Conv2D now uses bounded 32-row im2col matrices, existing K continuation, grouped
products, bias and raw NCHW repacking. Input gradients use product blocks plus
bounded deterministic overlap folds; weight gradients accumulate chunk products
in F32, and bias gradients reduce natively. BF16 widens before arithmetic and
narrows only each operation's output. Conv1D preserves Burn's singleton-spatial
composition; transposed Conv2D reuses the inverse patch geometry. Output padding
requires less than the maximum of stride and dilation. Native unfold4d and float_unfold preserve
payloads; Burn's composed unfold backward uses native slice assignment.
Five independent integer/analytic simulator cases pass grouped/depthwise, ragged
batch boundaries, dilation/padding/stride, transposed output padding and gradients,
Conv1D and unfold gradients. Both cards pass the initial six silicon cases (`1791234267`, 12/12). Expanded
card-0 gates pass F32/BF16 two-tile changed-input traces and BF16 resident SGD
with F32 loss; actual two-card convolution forward and all three gradients use
64-wide partitions (`1791234542`, 12/12 including two attention gates). External
numerical comparisons, larger K and broader layout acceptance remain pending.
These are implementation records, not closure of the convolution milestone.

### Resident indices and distributed evidence (2026-10-05)

`INDEX_PICK` validates resident I32 indices before addressing a source tile. B
performs raw selection; SFPU performs arithmetic. Arbitrary-axis multi-index
gather/select preserve F32/BF16/I32/Bool payloads. Float scatter/select-add fold
duplicate indices in logical order, with BF16 arithmetic widened until the final
output. Embedding and its backward remain Burn compositions; no index download
is needed. `step86` passes four simulator cases and ten tests across both cards
(`1791235065`): device-produced indices, face/ragged boundaries, raw values,
negative/end DOMAIN failures, duplicate ordering, BF16 late narrowing and
changed-argmax-index two-tile trace replay. Existing step61 Flex comparisons pass.
I32 wrapping scatter/select-add and Boolean scatter/select-OR are implemented
and validated by the later step86 continuation.

Mesh batched products now dispatch across cards. Fabric counters report products
submitted on each card and acknowledged Ethernet packets/bytes, independently of
PCIe traffic accounting. `step83` checks actual two-card attention forward and
analytic Q/K/V/bias gradients, requiring at least six products per card for the
backward workload; `step85` checks convolution gradients. These pass in
`1791234542`. Mesh trace capture remains unsupported.

Release card-0 operation baselines are recorded in firmware-performance.md, run
`1791235277`; they establish conditions and medians, without speedup claims.
Observed reversed-convolution-column and shifted-resident-index mutants fail
their independent simulator gates. Later continuation sections record large-K/permuted attention and convolution
coverage, the measured GMPOOL index limitation and the bounded Src transpose route.

The expanded resident transformer trace exposed a watchdog regression: one CALL
exceeded the simulated no-queue-completion budget. Completed trace chunks now
publish progress, preserving the stall deadline. Previously failing step40
passes simulator and card-0 silicon (`1791235705`); corrected full workspace rerun passes
(`target/silicon/tranche-workspace-fork-fixed.log`). The large-K planner assertion
is guarded by outside_fork to avoid inheriting its parent cache lock.

Additional continuation gates: forced ragged K=3609 (113 tiles, plan asserts K
is split) checks convolution forward and all gradients on both cards
(`1791235857`, 22/22 full step85 selections). A fractional permuted input matches
external Flex under an independently derived operand/phase accumulation bound;
BF16 attention replays changed V across ragged heads/sequences on two tiles. Both
cards pass these additions (`1791235944`, 4/4). Reversed convolution kernel-column
and shifted resident-index mutants were observed failing simulator oracles.

BF16 actual two-card convolution forward/all gradients and attention Q/K/V/bias
gradients pass (`1791237741`, four selections including the existing BF16 mesh
product). Each gate checks per-card product counts and acknowledged Ethernet
activity. Generalized transposed output padding below max(stride,dilation),
including padding at least stride, passes a grouped analytic forward/gradient
case in ttsim and card 0 in the same run. The isolated both-card smoke suite
passes 268/268 (`1791236644`); later additions have their own gates.

### Src transpose and GMPOOL contract limits (2026-10-05)

`Session::transpose_src_block` converts a resident 16×16 F32 storage face through
TF32 or BF16 Src and transposes it using TRNSPSRCB. The instruction operates only
on SrcB rows 16..32, so B copies the operand into that half of a flat 512-datum
unpack. Requirements declare the operand/output slots and semaphores. Zero and
subnormal inputs normalize to positive zero; infinities and tested NaN payloads
survive. Both cards pass changed-input two-tile trace replay and native TF32
matmul-reference equality (`1791239869`, 2/2). Eligible 16×16 transposed TF32 right
operands use this route. Arbitrary F32/I32/Bool and packed BF16 payload-preserving
materialization continues through native raw copying; M3 remains partial.

GMPOOL ArgMax mode is refused by ttsim. Both cards return max values but zero
packed index bits, including distinct finite winners in every first-eight row,
ties, zeros, NaNs and lower-half winners (`1791240702`, 8/8 combined selections).
An attempted integer Dst read/store also exposed no indices (`1791240373`).
The diagnostic MaxIndexProbe returns raw words and promises no decoded indices.
This measured path cannot implement Burn's max-with-indices contract; Burn's
existing SFPU route stays in use. No Blackhole index encoding is inferred.

The same combined run validates generalized transpose padding, external Flex
attention comparison and direct F32/BF16 single-card versus two-card forwards and
all gradients. MNIST rerun passes all five gates with unchanged golden
(`target/silicon/tranche-mnist-final.log`).

### Typed resident updates and final contract gates (2026-10-06)

I32 scatter/select-add now uses wrapping Tensix addition and integer mask
selection; Boolean scatter/select-OR uses native logical operations. Both reuse
the resident-index domain dependency and ordered update geometry. Step86 checks
I32 extremes/overflow, duplicate indices and Boolean updates, then replays
device-produced indices across 0/15/16/31/32/34 with two tiles and no index
download. The host decoder independently rejects invalid slot ranges, alignment,
bounds, chunks, rows and word widths before unsafe access.

Both cards pass 18 latest selections (`1791249159`), including those additions,
large K=3609 attention on permuted heads with causal/broadcast/all-row masking,
and F32/BF16 transposed-Conv1D/unfold backward compositions. Convolution and BF16
attention traces are further widened to require multiple output tiles and run
with one and two Tensix tiles. The deliberate omitted-slice-assignment mutant
fails the raw-copy oracle (`target/silicon/step84-mutant-omitted-assignment.log`).


The widened convolution and BF16 attention traces pass changed-input replay with
one and two Tensix tiles, with output widths requiring both tiles. Step88 also
passes strengthened nonzero Q/K gradient oracles on both output partitions and
bitwise single-card/mesh comparisons (`1791249486`, 8/8). Observed negative
controls fail for shifted resident index columns, a one-bit integer selection
mask, Boolean AND in place of OR, omitted slice assignment and reversed convolution
kernel columns (`target/silicon/step{84,85,86}-mutant-*.log`); restored gates pass.


### Final continuation verification (2026-10-06)

The updated isolated release smoke selection passes **288/288** on cards 0 and 1
(run `1791250791`), including step82–88, one/two-tile changed-input traces,
resident training, checked domains and actual distributed forward/backward
products. Direct single-card/mesh comparisons cover F32/BF16 outputs and all
module gradients; the attention case has nonzero Q/K contributions on both
partitions and acknowledged Ethernet traffic.

The full default workspace run including doctests passes
(`target/silicon/tranche-workspace-final.log`). Final default/silicon workspace
Clippy, separate RISC-V firmware Clippy, formatting, silicon no-run compilation,
all three generator checks and both shipping dependency checks pass
(`target/silicon/tranche-checks-final.log`). The five MNIST e2e regressions pass
in 264.20 s with the golden unchanged
(`target/silicon/tranche-mnist-current.log`). Release operation baselines and
conditions are recorded in `docs/learnings/firmware-performance.md`; no speedup
guarantee is made. M3 remains partial for the measured Src payload-conversion
limit; ND indexing and excluded convolution/attention variants are not closed.


### Convolutional MNIST application continuation (2026-10-06)

`tt-mnist --model cnn` selects the shared `tt_mnist::cnn::Cnn`: Conv2D 1→8,
5×5 stride 4, ReLU, average pool 2×2 and Linear 72→10 (938 parameters).
It uses the existing dataset, optimizer, optional Flex comparison, F32/BF16
storage, inference and single-card inference/training trace paths. Evaluation
uses bounded 64-image batches. Traced training now indexes image batches by
`from * 784` and keeps trace inputs F32 before native storage conversion.

The resident-label gate exposed missing unaligned integer slicing; I32/Bool now
reuse native logical slicing and general strided swap views. Packed BF16
training traces also needed `copy_into_bf16` for parameter updates: logical
halfword copies retain buffer identity and initialize padding with zero, using
the existing Requirements-declared repack staging. Generic trace replay remains
single-card only. Step89 passes F32/BF16 actual MNIST learning and changed-batch
replay versus fresh updates on one/two tiles, both cards (`1791252026`, 8/8).
The zero-pooled-features mutant fails the learning oracle
(`target/silicon/cnn-mutant-zero-features.log`); restored simulator tests pass.
Full-epoch card-0 acceptance passes (`1791252065`): 59,968 training images, 937 steps, then all 10,000 test images. Accuracy is 78.60% versus Flex 78.66%; loss falls 2.301632→0.595991 versus 2.301627→0.595428. Single release timing including preload/scalar losses is 226.410 ms/step versus Flex 1.040 ms/step; concurrent simulator/build checks ran on the host, so this is an initial application observation, not a controlled median benchmark or speedup claim.


CNN continuation final verification: card 1 passes 15/15 slicing/index/CNN
selections (`1791252326`), and card 0 passes the same 15/15 (`1791252404`).
Restored step89 simulator learning/replay gates pass; the new permuted I32/Bool
slice oracle passes. All five existing MNIST MLP e2e regressions pass in 267.53 s
with the golden unchanged. Workspace default/silicon Clippy, formatting, Burn
generator and both shipping dependency checks pass. Logs are
`target/silicon/cnn-{simulator-final,slice-simulator,mlp-regression,clippy-final,default-clippy}.log`.

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
