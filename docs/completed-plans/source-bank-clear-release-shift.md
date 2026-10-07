# Stage C — source-bank clearing, release and SrcB shifting

Started and completed 2026-10-07. Acceptance gates allocated as `step103_source_banks`.
The starting tracker is 87 completed groups, 21 pending, six excluded; Stage B
is accepted in step101/102. All four constrained families are accepted: 91 completed, 17 pending, six excluded.
Stage D is next, as a separate healthy-bank handover tranche, never recovery.

## Supported forms

`matrix::Banks` retains lockstep ownership. Existing SETRWC releases are unchanged.
ZEROSRC distinguishes current-unpacker banks (Empty/Filling → Empty) from current
matrix banks (Loaded → Loaded), with A/B/both-operand selection. Both operands
means one current bank per operand, never both physical banks. Matrix clearing
of unpacker data requires retired unpack work and an explicit cross-role handoff.
CLEARDVALID releases A/B/both (Loaded → Empty), with Reset=0 and
KeepReadingSameSrc=0. UNPACR_NOP_ZEROSRC uses selected unpacker, WaitLikeUnpacr=1,
BothBanks=0, clear value=0; refill and handover use regular UNPACR. SHIFTXB retains
Loaded B, checks row 0..63 and modifier 0..7, and selects rotate or zero-fill.

Deferred variants: ZEROSRC negative infinity and UNPACR wider nonzero clear values
need format characterization; both-physical-bank clearing needs a quiescent-state
proof; retained-bank release needs new ownership states. Reset remains excluded.
No automatic Burn route, Session tensor API, performance claim or runner change.

## Implementation and acceptance checklist

- [x] Checked ownership APIs, encoding/range tests and compile-fail ownership gates.
- [x] Independent physical 19-bit Src-array model: both pointer pairs, ownership,
  staged claims, clear effects, wraparound and modifier increment.
- [x] Kernel builder with declared scratch/semaphores, explicit format/ADC/RWC/
  modifiers, unpack retirement before matrix clear, retirement before publication
  and release before reuse.
- [x] Simulator matrix clear A/B/both, separate/both CLEARDVALID release, alternating
  banks, surviving control and wrong-clear-selection mutant.
- [x] Isolated current-unpacker ZEROSRC refusals, each with surviving controls;
  corrected UNPACR_NOP_ZEROSRC executes in ttsim.
- [x] Release silicon: all four families, TF32/BF16 transitions, partial staging,
  preserved operand/rows, rotate/zero-fill, row-63 wrap, repeated shifts,
  modifier entries 1/4, specials and deterministic data mutants.
- [x] Existing matmul/pooling programs after clear/release on the same tile.
- [x] Accepted gates in SMOKE; isolated card-0 release SMOKE 242/242 (`1791397621`).
- [x] MNIST execution-order regression: 8/8 with `TT_BLESS` unset; golden unchanged.
- [x] Workspace format/Clippy/tests; silicon compile/Clippy; generators and
  shipping dependency checks. No firmware changes, so separate firmware checks
  are unnecessary.
- [x] Measured learnings, authoritative tracker and remaining plan close-out;
  moved this accepted tranche to completed-plans.

## Evidence

Simulator: 6/6. Card-0 release semantic acceptance: 9/9 (`1791397029`).
Current-unpacker ZEROSRC refuses `tensix_zerosrc: write_mode=0`; SHIFTXB remains
silicon-only under row 50. UNPACR_NOP_ZEROSRC's initial `bank_clr_ctrl=1` refusal
was refuted as the wrong layout. Measured BH WaitLikeUnpacr/BothBanks positions
are 5/4, with clear-value bits 2..3; the generator input and provenance now carry
the independent bank-selection/held-bank gate evidence. Checked zero executes on
both targets. Wider clear code 2 was characterized as TF32 1.0 solely to establish
the gained field bit; exposing nonzero values across formats remains deferred.

Every physical row is observed. Two initial handovers initialize all rows and seed
both banks; four subsequent rounds clear each bank, preserve the opposite-bank
sentinel and re-read previously cleared rows. No source reset occurs before the
appended existing matmul/pooling instruction sequences. Data mutants retain waits.

Full release SMOKE: card 0, 242/242 (`1791397621`), including final explicit ADC
initialization and all nine step103 gates. The MNIST silicon smoke gates pass
with the unchanged golden; the required simulator execution-order gate is below.

Final checks: `cargo test --workspace` passes (including ownership compile-fail
doctests); workspace Clippy passes with warnings denied, with and without
`tt-tests/silicon`; the full silicon suite compiles with `--no-run`.
`gen-isa`, `gen-cfg`, `gen-burn-ops` checks and both shipping dependency checks
pass. The generated table inventory includes 164 encodings and 16 measured
layouts; the matrix-address LLK cross-check continues to cover its 15 matrix
layouts and does not classify the new unpacker encoding as a matrix instruction.

MNIST execution-order regression: 8/8 (single-tile, four-tile, two-chip and
four-chip simulator configurations), with `TT_BLESS` removed from the command
environment. `crates/tt-tests/tests/golden/mnist_reduced.txt` is unchanged.
All checklist items are accepted. Stage D remains separate work.
