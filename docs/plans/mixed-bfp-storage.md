# Mixed BFP tensor storage and seeded PRNG

Current status (2026-10-06): resident BFP8/BFP4/BFP2 conversion and packed
matmul, Burn propagation and identity backward pass on both cards
(`1791299490`). Expanded acceptance, fusion and actual MNIST policies pass
both cards (`1791300035`); full SMOKE passes 364/364 (`1791300502`); final
rounding/control/propagation gates pass 28/28 (`1791301208`). Diagnostic
RISC-V PRNG restarts pass both cards (`1791302401`, 4/4 with advancement and
predication). D2 is complete for the delivered BFP8/4/2 formats; S7 (native seeded
random) was completed in the Phase 10 close-out.

## Hardware prerequisites

- [x] Independent arithmetic packing oracle for BFP8/BFP4/BFP2, including
  exponent groups and low-first nibble/bit ordering (`step92_bfp_formats`).
- [x] Raw signed decode corpus over exponents 0, 1, 6, 127, 254, 255; Src
  normalization flushes decoded subnormals. All datum patterns are exercised.
- [x] Separate histogram reset gate (`step93_exponent_history`); no assertion
  that CLREXPHIST controls each block's chosen exponent.
- [x] Checked diagnostic seed-register sequence (`backend::write_prng_seed`),
  advancement/predication gate and simulator/silicon reseed characterization.
- [x] Establish a reproducible diagnostic silicon restart path: direct RISC-V
  full-width configuration store, fence and 512 RISC-V NOP iterations
  (`step91_seeded_prng::riscv_seed_store_characterization`). WRCFG continues
  the stream. The application-level device random built on it landed with S7 (step140-142).
- [x] Full conversion sweeps at rounding/clamping thresholds, independent
  integer encoding model and conversion-to-matmul error propagation.

The physical probe uses 16-byte headers, 64 exponent bytes and 1024 datums:
BFP8/BFP4/BFP2 image sizes are 1104/592/336 bytes before GDDR slot alignment.
Measured codes 6/7/15 are exposed by `L1Format::code`; resident conversions
pass physical-byte and decoded-value oracles on both cards (`1791299101`). The finite-input absolute
compression bound is derived beside the gate, separately from special-value
bit-exact expectations. The executed missing-exponent-section mutant is rejected.

## Resident storage

- [x] Add `BfpFormat::{Bfp8, Bfp4, Bfp2}` and distinct `BfpTensor`, with
  physical allocation sizes derived from TileImage, placement and padding.
- [x] Tensix pack/unpack conversion, decoded and raw Session readback and free.
  B and NC perform transfers only. Deliver BFP8, then BFP4 and BFP2.
- [x] Direct same-format packed matmul, including ragged edges and K reloads
  (`step94`, both cards). Transposed, mixed-format and batched Burn inputs
  widen on device and use the existing native F32 scheduler.
- [x] Broader transposed/batched BFP product acceptance and performance audit.
- [x] Include format in program/cache/replay keys and ownership. Validate NC
  completion, trace holds, deferred frees and repair of a view's padding.
- [x] Explicitly refuse mesh requests in the initial single-card implementation.

## Burn contract

Logical tensors remain F32; storage casts create new tensors and invalidate
cached source-value host copies. Logical BF16 enters BFP through an explicit
conversion to logical F32. Native operations decode on device, compute and
accumulate in F32, then pack using these rules:

| Operation | Result storage |
| --- | --- |
| Explicit cast | Requested format |
| Unary/scalar arithmetic | Input storage |
| Binary arithmetic and matmul | Higher precision: F32 > BFP8 > BFP4 > BFP2 |
| Reductions | F32 |
| Comparisons/index outputs | Existing Boolean/integer types |
| Views retaining exponent groups | Share original storage |
| Rearrangements creating new groups | Exact decoded F32; explicit cast to compress |

Compound operations follow constituent rules. F32 biases/constants/reductions
can promote subsequent results. Storage casts use identity backward, a documented
straight-through approximation. Gradients, accumulation, master parameters and
optimizer state remain F32. Sensitive loss examples explicitly use F32 inputs.
Serialization returns decoded F32 values; precision policies restore compression.

- [x] Tensor `with_storage`/`storage_format` extension for TtBackend and autodiff.
- [x] Central adapters for existing floating operations; no host arithmetic or
  intermediate downloads. Existing unsupported operations continue to fail.
- [x] Preserve storage in traces/parameter copies and compression boundaries
  under fusion; no change to rounding frequency.
- [x] Precision-policy example for named weights/activations, with explicitly
  selected ordinary-F32 policy for other backends; executable API/README examples.

## Acceptance

- [x] Conversion, packed matmul, Burn propagation, mixed-storage training and
  changed-input replay gates; SMOKE includes each delivered gate.
- [x] Executed negative controls for wrong exponent/order/edge masks,
  accumulator reload and output propagation.
- [x] Instruction, allocation and traffic audits; independent decoded-operand
  arithmetic oracle and separate compression comparison against F32 inputs.
- [x] MNIST MLP/CNN layer policies over all formats, gradients and trace replay;
  record accuracy changes without modifying the F32 golden.
- [x] Both-card silicon validation, required workspace/firmware/generator and
  shipping checks, training regression and validated release benchmarks.

BFP `a` variants, portable Burn QTensorOps, DOTPV and packed mesh transport remain
deferred. No compressed-gradient policy, INT8 path or dependency fork is added.

Final validation: workspace tests with `tt-tests/e2e` pass, including the eight
existing MNIST regressions with unchanged golden. Default/silicon Clippy,
firmware RISC-V Clippy, formatting, all three generator checks, silicon
compilation and shipping dependency checks pass. Ordinary-F32 and native
policy API examples compile/run as `tt-mnist` doctests. Benchmarks use release
outputs, two warmups and nine validated samples per format on both cards;
conditions, medians and policy accuracy observations are in
`docs/learnings/firmware-performance.md`.

The accumulator negative control executes the final K tile alone (the numerical
result of dropping the preceding accumulator). Exponent/order/padding controls
execute corrupted physical streams or a deliberately false padding claim. The
output-storage control computes the uncompressed scalar result and demonstrates
that it differs from the required compressed output. B/NC paths contain byte
transfers/geometry copies only; Tensix performs pack/unpack/arithmetic. Cached
source-value controls, physical readback and model traffic reports prevent a
host cache from hiding skipped compression or arithmetic.

## Resume checkpoint (2026-10-06)

The delivered tranche is complete: BFP8/4/2 resident storage, native conversions
and products, Burn storage control/propagation, autodiff, fusion boundaries,
parameter copies/traces, model policies and diagnostic PRNG characterization.
There is no active implementation or required acceptance check left pending
for that scope. The deferred items above are future tranches. D2 is complete
only for delivered formats. S7 was completed in the Phase 10 close-out: Burn random
draws on the device (`burn-tt/src/random.rs`, step140-142).

Resume by reading this file, the backend/kernel READMEs and the BFP/PRNG sections
of `silicon-operating-notes.md`, `ttsim-divergence.md` and
`firmware-performance.md`. The latter retain empirical results even if local
`target/silicon` logs are removed. Do not reinterpret historical unchecked
entries as missing implementations. Important retained limits:

- Initial BFP storage is single-card; packed mesh requests fail explicitly.
- Direct packed products require same-format ordinary nontransposed operands;
  mixed/transposed/batched Burn products widen on device.
- Replay input buffers require ordinary F32 storage; capture compression inside
  the computation. Compressed outputs/parameter copies are supported.
- Cast backward is a straight-through approximation; training state stays F32.
- WRCFG seed writes continue silicon's stream. Direct RISC-V stores with a fence
  and 512 NOP-loop iterations restart it in the diagnostic gate; the interval
  is conservative, not a measured minimum/completion protocol. Simulator lane
  initialization is offset; adjacent lanes are correlated and seed 0xffffffff
  is absorbing. This is not an application hardware RNG contract.
- MNIST policy accuracy measurements use four images/four steps, not held-out
  accuracy or full convergence. Existing F32 goldens remain unchanged.

Validation references: full both-card SMOKE `1791300502` (364/364), final BFP
controls `1791301208` (28/28), finalized PRNG suite `1791302401` (4/4), and
release benchmarks `1791300393`. The new RISC-V probe is selected by SMOKE's
step91 filter; the full 364-case run preceded that added probe. Required checks
passed as described above; the final eight-case MNIST regression passed in
268.05 s. Subsequent documentation reconciliation changed no executable code.

The tranche's source, tests and documentation are committed together under
`Add mixed BFP tensor storage and PRNG diagnostics`. This file is the durable
handoff; resuming does not require the chat context. Hardware run logs remain
local artifacts, with their findings and validation references retained in
the documentation above.
