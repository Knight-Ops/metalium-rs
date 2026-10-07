# ADC row-window copies — 2026-10-06

Implement the related INCADCXY/ADDRCRXY instruction pair with a resident kernel
and validate it alongside native Burn slices. Ordinary acceptance uses card 0 under the user's
one-card policy. Other instruction families remain deferred.

## Current contract

`tt_isa::adc::{Targets, Xy, Coordinates, increment, advance_cursor}` checks
three-bit increments, makes an empty target selection unrepresentable, and
always uses the issuing thread. Live increments leave cursor anchors unchanged;
cursor advances update selected anchors and restore the selected live values.
The helpers retain the generated Wormhole provenance. Empty cursor masks are
silicon no-ops but simulator refusals (divergence 74).

`Session::copy_rect_adc(source, origin, dims)` accepts in-bounds nonempty F32
rectangles, any row origin/count and column origin/width divisible by sixteen.
The source is an ordinary DramTensor, including tile-row views; tensor
transposition/strided mapping is outside this API. Invalid geometry or other
logical dtypes fail before execution.

B reads complete source tiles into planned L1 slots. Math clears Dst and hands
it to unpacker 0, whose live/cursor XY counters select complete sixteen-datum
rows. The packer waits for unpack retirement, packs the complete output tile,
and releases Dst; NC writes the result. The unpacker retires its last operation
and restores zero XY/ZW live counters and cursor anchors before handoff, so
later Src programs cannot inherit the final rectangle row. Runs split at
source/output faces and at seven rows. Whole mailbox descriptors, existing program cache/scheduler,
trace holds and deferred frees apply. Input padding is unrestricted and its
claims remain unchanged; all output padding is zero.

Burn preserves row views first, then uses its original native repack path for
float slices. Automatic ADC routing was removed after its measured regression;
the engine capability, server plumbing and report counter were removed with it.
The explicit `Session::copy_rect_adc` API and instruction gates remain.
Default slice gates cover aligned/unaligned rectangles, stepped/transposed
inputs, residency, raw bits and gradients. ADC optimization and performance-based
Burn routing are deferred to a later pass.

## Acceptance checklist

- [x] Typed helper encoding/range checks and no cross-thread override.
- [x] Independent counter and indexing oracles, all nonempty coordinate masks,
  target combinations, unpacker 1/packer observations and thread isolation.
- [x] Logical exceptional bits, face/tile boundaries, ragged output rows,
  borrowed row views, input padding claims and downstream reductions.
- [x] Invalid geometry/dtype rejection, changed-input traces, deferred frees.
- [x] Restore original Burn repack routing; retain residency/gradient gates
  for aligned/unaligned rectangles, views, stepped and transposed paths.
  Existing generated float_slice override retained.
- [x] Safe live-add mutant differs from the correct cursor restoration oracle;
  zero increments restore the selected live values to their anchors.
- [x] Add step98 to SMOKE; record empty-mask refusal and partial-Dst-row finding.
- [x] Final expanded simulator and one-card silicon gates; record final run ID.
- [x] Workspace tests/lints, generator/shipping checks and MNIST regression.
- [x] Full release silicon SMOKE on card 0: 203/203, run `1791321817`.
- [x] Validated release native-repack comparison: warmups, medians, host timing,
  dataflow counts and collector artifacts in firmware-performance.md.

Final isolated benchmark `1791322068` validates all six records, with ADC
23–79% slower than native repack for the three measured rectangles. This
performance cost is recorded explicitly; no speedup is claimed.

Final `cargo test --workspace` passes, including doctests. Default and silicon
workspace Clippy, separate RISC-V firmware Clippy, formatting, silicon no-run
compilation, ISA/Burn generator checks and both shipping dependency checks pass.
All eight step12 MNIST e2e regressions pass (277.93 s), with no golden change.

Expanded card-0 run `1791321793` passes nine semantic/kernel/Burn gates and
four CNN learning/trace gates after the counter-lifetime correction. The final
simulator suite passes nine gates. Fresh-copy gates also pass with both
`TT_PIPELINE=0` and `TT_BATCH=0`; trace capture requires batching and is tested
with the default settings.
No BF16/FP16/BFP kernels, application RNG or additional instruction families are
included, and no speedup is a completion requirement.

## Native slice performance restoration (2026-10-06)

The original Burn `float_slice` dispatch and all affected engine/server/report
code match the pre-ADC implementation. This removes automatic ADC execution
without introducing a new selector or setting. The ADC Session API and its
counter cleanup remain intact. The updated step98 slice gate checks native
residency, exceptional bits and gradients instead of an ADC dispatch counter.

Restoration validation: simulator step98 9/9; card-0 step98 plus step89 CNN
13/13 (`1791323039`); default/silicon workspace Clippy, formatting, Burn generator
and both shipping dependency checks pass. All eight step12 MNIST e2e
regressions pass again (264.59 s), with the golden unchanged.
