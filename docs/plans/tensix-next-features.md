# Tensix feature implementation record

2026-10-05. Model coverage first; BF16 is the new storage format.
Normalization keeps Burn's native compositions rather than adding fused kernels.

| Milestone | Implemented | Remaining completion work |
|---|---|---|
| Reduction breadth | Products, Boolean reductions, argextrema, sum/product scans; min/max scans with raw total ordering (`step79`); full and axis I32 wrapping sum/product and signed min/max (`step80`) | Watch new scan/reduction gates reject deliberate mutants |
| Normalization | Native LayerNorm/RMSNorm compositions and analytic gradient gates | Fusion deferred |
| Integer and rounding | Full-width I32 ALU, deterministic rounding/casts; opt-in hardware BF16/TF32 nearest, stochastic and toward-zero modes (`step81`) | Division/remainder execution disabled pending domain-flag validation; integer mean depends on division |
| BF16 storage | Packed gathers, packed K continuations, batched/broadcast/view matmul, native adapters, two-card mesh execution, full MNIST accuracy run | Mesh transport still widens operands to F32; further performance tuning |
| FPU pooling | GMPOOL/GAPOOL block kernels, BF16 GAPOOL averages, native max/indices/backwards; F32 and BF16 pooling traces (`step78`) | General GMPOOL routing with exact NaN/index semantics; window staging optimization |
| FPU transpose | Instruction probes (`step73`), existing mover view materialization | Tensix tensor transpose integration; resolve the BF16 control mismatch first |

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
