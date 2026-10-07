# ADC Z/W plane copies — 2026-10-06

## Current contract

`tt_isa::adc::{Zw, ZwCoordinates, increment_zw, advance_cursor_zw}` reuse
`Targets`, check three-bit increments, and always address the issuing thread.
Blackhole live counters and cursor anchors are thirteen bits. Live addition
leaves anchors intact; cursor-relative updates replace selected live values
with updated anchors. Unpacker input addresses use only the low eight bits.
Generated encoding provenance remains unchanged. Empty masks were investigated
independently: ttsim refuses them, while card 0 accepts them as no-ops.

`Session::copy_planes_adc(source, [W,Z,Y,X], [w,z], [nw,nz])` interprets source
storage as `[W*Z*Y,X]` and returns `[nw*nz*Y,X]`, in W-major then Z-major order.
All dimensions must be nonzero, storage must match, selection must be in bounds
and products must fit. Only F32 is supported. Validation precedes output
allocation and submission. Ordinary row views are supported. All logical bits
are preserved, including NaN payloads and signed zeros. Input padding is `Any`,
output padding is `Zero`, and parent padding claims remain intact.

Every output tile has 64 sixteen-datum fragments in face order. Each fragment
is mapped independently to a logical source plane and physical source tile.
B groups fragments by source tile, reads one tile into scratch on NoC0, then
copies logical words into a zero-initialized staging slot. The staging slab has
eight local Z rows per W group, independent of global plane dimensions. All
scratch/output slots and semaphores are allocated through `Requirements` using
the existing layout planner. The largest mover list is 258 entries, below 512;
no expanded mailbox or mover ABI is introduced.

Math clears Dst before unpack. An uncompressed FP32 descriptor sets input
strides; explicit byte output strides (Z=64, W=512) select complete Dst rows.
INCADCZW traverses each Z group; ADDRCRZW restores Z anchors and advances W
anchors between groups. Every counter mutation follows unpack retirement. The
final unpack retires and XY/ZW live counters and anchors return to zero before
handoff. T2 packs the complete tile, NC writes it on NoC1. Existing Work,
scheduler, cache pins, trace holds, deferred frees and ownership settings apply.

Burn routing is inapplicable to this explicit Session-only API. Automatic Burn
adoption, other dtypes, application RNG, transpose and other instruction
families remain deferred. RMWCIB0..3 remain deliberately omitted.

## Inventory reconciliation

Direct counts in the current mnemonic checklist are **7 matrix/source-state**,
1 SFPU, 8 unpacker/packer, 2 frontend, 2 configuration and **19 DMA/register/
atomic** rows before this tranche: 39 total. The supplied plan's 8/18 grouping
was incorrect; moving those group counts back to the actual rows preserves the
total. Completing these two Z/W rows leaves 37 pending, including REG2FLOP_ADC.

## Acceptance

- [x] Checked encodings, all target/mask combinations, increments 0–7 and
  rejection of 8; no thread override.
- [x] Independent thirteen-bit counter and input/output address models;
  zero-increment restoration, preservation of unselected coordinates,
  unpacker 0/1 and packer observations, thread isolation.
- [x] Ragged heights/widths 1/17/33, tile-crossing plane boundaries, single and
  multiple selections beyond local staging dimensions, borrowed row views,
  exceptional raw bits and invalid geometry/dtype rejection.
- [x] Changed-input trace replay and deferred frees; repeated copies followed
  by ragged matmul and reduction; program audit requires both instructions.
- [x] Safe negative controls: omitted increment, swapped Z/W output strides,
  live addition replacing cursor restoration.
- [x] Add step99 semantic/kernel gates to SMOKE; simulator 9/9.
- [x] Fresh copies with TT_PIPELINE=0 and TT_BATCH=0.
- [x] Expanded card-0 silicon gates and full release SMOKE (including CNN).
- [x] Workspace tests/lints, silicon compilation/lints, generator and shipping
  checks, step12 MNIST with the golden unchanged.
- [x] Validated release native-repack benchmark: warmups, medians, host timing
  and dataflow counts, with run IDs recorded. A speedup is not required.

Initial card-0 run `1791323973` passes the four initial semantic/kernel gates.
Final simulator suite passes 9/9. Card-0 step99 followed by CNN gates passes
13/13 (`1791324670`); full release SMOKE passes 212/212 (`1791324245`).
Fresh copies pass with TT_PIPELINE=0 (`1791324866`) and TT_BATCH=0
(`1791324883`) on silicon and in the simulator. Benchmark `1791324901`
validates all six records and measures ADC at 1.71–2.30× native repack cost.
Conditions and dataflow counts are in firmware-performance.md.

Eight step12 MNIST e2e gates pass (271.51 s) with the golden unchanged.
Formatting, default/silicon Clippy, silicon no-run compilation, separate
firmware Clippy, all three generator checks and both shipping checks pass.
The full workspace test suite, including doctests, passes.
