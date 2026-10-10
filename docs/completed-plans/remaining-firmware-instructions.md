# Remaining Tensix firmware instructions — implementation plan

> **Status 2026-10-10:** Stages E-H and the research rows are closed by lane (see [hardware-coverage-closeout.md](../plans/hardware-coverage-closeout.md)): `ATGETM`/`ATRELM`, `ATCAS`, `ATSWAP` (group form), `ATINCGET`, `ATINCGETPTR`, `SFPLOADMACRO` and `MOVDBGA2D` are `[x]` on silicon; `PACR_SETREG`, `UNPACR_NOP_SETREG`, `STREAMWAIT`, `STREAMWRCFG` and `GATESRCRST` are `[-]` with evidence; `LOADREG`, `STOREREG` and `STOREIND_MMIO` are `[x]` on the allowlisted PIC words (silicon probes and gates pass on card 0). The list below is historical.

Review date: 2026-10-07. Source of completion status:
[hardware-coverage.md](../plans/hardware-coverage.md#tensix-coprocessor-instruction-implementation-checklist),
including the step97–104 acceptance and the current working-tree code.
This is an implementation sequence; measured Stage D acceptance is recorded in
its completed checklist and operating notes. Five pending rows are now
deliberately excluded from support; the rationale and alternatives are below.

## Scope and findings

The mnemonic checklist now contains **92 completed groups, 16 pending groups and
six deliberately omitted groups** (`RMWCIB0..3`, `DOTPV`, `SHIFTXA`,
`SETDVALID`, `REG2FLOP_ADC`, `FLUSHDMA`). Groups include multiple forms:
register/immediate operands, bank selections and micro-modes must be tracked
individually during implementation. Step100 closes seven groups (card-0 run `1791379895`); see the
[completed tranche](scalar-config-foundation.md).
A generated encoder is not a checked API,
semantic gate or kernel implementation.

Here “firmware instructions” means Tensix coprocessor instructions issued by the
T0/T1/T2 firmware runners. Most changes belong in `tt-isa` helpers and
`tt-kernels` program builders. `role_t0.rs`, `role_t1.rs` and `role_t2.rs` already
use the generic `corpus::run_role`; adding an opcode does not normally require
an opcode-specific firmware handler. Firmware changes are needed only when new
state must be initialized, described, fenced or restored by the runner.

Evidence checked: `tt-isa/src/{backend,matrix,adc,sfpu}.rs`, generated ISA
definitions, generator provenance, step5/step9/step38/step97–99 gates, and the
pinned `vendor/tt-isa-documentation/{BlackholeA0,WormholeB0}` instruction pages.
Paths below are relative to `crates/` unless otherwise specified.

Already delivered and excluded from this backlog: matrix ELW operations,
`MOVD2A/MOVD2B/MOVB2A`, ADC XY/ZW increments and cursor updates, delivered LUT
forms, `ADDDMAREG` address stepping, `CLREXPHIST`, and the seven step100 scalar/configuration
groups, the four step101/102 L1 groups and the four step103 source-bank families. Application PRNG,
payload-preserving tensor transpose, FP16/INT8 formats, ND indexing and broader
Burn operation coverage remain feature work; they do not add pending mnemonics
to this count. S7's diagnostic PRNG gates do not establish application RNG.

The unchecked inventory needs these semantic corrections:

- `SETDVALID` gives Src banks to the matrix unit; it does not set a Dst scoreboard.
  Its Blackhole implied-format interaction is explicitly unsupported.
- `CLEARDVALID` gives Src banks to the unpackers. Its non-reset forms can be
  implemented; its reset form remains excluded from the safe surface.
- `GATESRCRST` invalidates the one-slot SrcB operand cache.
- `XMOV` starts an asynchronous bulk copy/zero operation in aligned 16-byte
  units, not an L1-to-register move.
- `ATGETM/ATRELM` operate on Tensix mutexes, not L1 atomic words. `ATSWAP`
  performs masked stores and does not return the old memory value.

## Complete pending inventory and disposition

Stages refer to the sequence below. Research rows stay open until they have an
implemented or deliberately excluded disposition. Stage C is accepted in step103; see
[source-bank clearing, release and shifting](source-bank-clear-release-shift.md).
Stage D is accepted as a separate healthy-bank handover tranche; see
[its completed checklist](explicit-unpacker-handover.md).
Stage B is accepted in step101/102; its detailed plan is completed.

| Group | Pending instructions | Next disposition |
|---|---|---|
| Matrix/source: 2 | `MOVDBGA2D` | Diagnostic helper only; manually establish bank ownership/format |
| | `GATESRCRST` | Diagnostic cache-invalidation helper; production use needs an observable cache-state oracle |
| SFPU: 1 | `SFPLOADMACRO` | Stage H: macro configuration and scheduling, silicon-only |
| Unpacker/packer: 2 | `PACR_SETREG` | Stage F: restricted, explicitly configured MMIO target |
| | `UNPACR_NOP_SETREG` | Research: weak semantics and nonstandard TDMA-RISC address-base state |
| Frontend: 2 | `STREAMWAIT`, `STREAMWRCFG` | Stage G, after a minimal stream-overlay lifecycle exists |
| DMA/register/atomic: 9 | `ATGETM`, `ATRELM`, `ATCAS`, `ATSWAP`, `ATINCGET`, `ATINCGETPTR` | Stage E |
| | `LOADREG`, `STOREREG`, `STOREIND_MMIO` | Stage F |

All 16 pending groups are accounted for. This count covers the authoritative
tracker's grouped rows, not every generated ISA variant or retired instruction.

## Deliberate exclusions (2026-10-07)

`[-]` means excluded from the supported implementation backlog. Existing raw
encoders/probes remain; this does not assert that hardware refuses the opcode.
The following decisions rely on pinned specification text and existing gates,
not new hardware experiments. Pages are under
`vendor/tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/`.

| Excluded group | Evidence and reason | Supported alternative |
|---|---|---|
| `SHIFTXA` | `SHIFTXA.md` declares UnsupportedFunctionality and a noncontractual prior-instruction-dependent input-row bug. A predictable general row-shift API cannot be promised. This is Wormhole-tree documentation, not a new BH measurement. | Explicit staging/repacking, or validated SFPU lane movement where suitable; neither promises identical cost or direct SrcA semantics. |
| `SETDVALID` | `SETDVALID.md` explicitly identifies **Blackhole** implied-format handover as ill-specified/stale. | Regular UNPACR final FlipSrc today; separately gated `UNPACR_NOP_SETDVALID` in Stage D. |
| `REG2FLOP_ADC` | `REG2FLOP_ADC.md` declares UnsupportedFunctionality and weak functional-model confidence. No concrete consumer justifies establishing a new dynamic ADC contract now. | Checked SETADC/SETADCXX/XY/ZW with descriptors; not a claim of equivalent runtime GPR-to-ADC throughput. |
| `DOTPV` | `DOTPV.md` says identical to MVMUL without SrcB broadcast and explicitly recommends MVMUL. Step9 already retains BH encoding evidence (divergence 50). | Non-broadcast MVMUL; no new production helper or equivalence tranche. |
| `FLUSHDMA` | `FLUSHDMA.md` prefers STALLWAIT in almost every case: FLUSHDMA monopolizes the shared Scalar Unit while waiting. This is a redundant scheduling choice, not a proven universal performance inequality. | STALLWAIT with the same C0–C3 conditions and all block bits; current-thread wait does not monopolize the other threads' Scalar Unit. |
| `RMWCIB0..3` | Existing simulator omission; no handler, already excluded. | Whole-word WRCFG, plus gated CFGSHIFTMASK for supported mask operations. |

Reopen an exclusion only for a concrete consumer that the supported paths cannot
serve adequately and an independent Blackhole semantic gate. A simulator refusal
alone does not justify excluding a useful hardware instruction: SHIFTXB and
SFPLOADMACRO remain scheduled. Likewise, CLEARDVALID's unsafe reset form and
STREAMWRCFG's ordering bug do not exclude their entire families; constrained
forms and explicit waits can preserve their useful behavior.

## Shared implementation contract

- Add checked helpers to `tt-isa`; keep it dependency-free and `no_std`.
  Helpers validate semantic limits, alignment, register groups, address units,
  clobbers and ownership, rather than merely accepting encodable bit fields.
- Build independent reference models from pinned functional pseudocode.
  Compare physical/raw representations where appropriate, not an epsilon.
  Numerical kernels also need instruction-sequence prediction and separately
  derived approximation bounds, including zeros, subnormals, NaNs and infinities.
- Use generated layouts as their provenance permits. For new BH measurements,
  change `xtask/src/gen_isa/Bits32_BH.lua` and `measured.rs`, regenerate and run
  `gen-isa --check`. Do not silently promote Wormhole layouts or hand-edit outputs.
  Existing step9 evidence confirms selected fields, not every mode of an opcode.
- Probe simulator support per form in `fork_scope`; new refusals get an explicit
  control and a divergence entry. Do not assume all M4 instructions share row 50:
  that row establishes refusal for DOTPV, SHIFTXB and MOVDBGA2D, not SHIFTXA.
- Declare L1 storage/semaphores through `Requirements`; name GPR scratch ranges
  and configuration clobbers. Start each program from explicit state. Preserve
  scheduler ownership, B/NoC0 reads, NC/NoC1 writes, cache pins, trace holds and
  deferred frees. Packing retirement does not establish NC release.
- Use actual completion waits for asynchronous loads/stores and config changes.
  Fixed DMANOP counts are not a memory completion mechanism. Mutants should
  corrupt deterministic data/mapping, not remove waits or access protections.
- Ordinary acceptance uses one discovered card under the tracker's current
  policy. Use both cards for mesh or an actual device-specific investigation.
  All silicon execution uses `cargo xtask silicon`; no plain silicon `cargo test`.

## Implementation sequence

### A — Configuration readback and scalar register foundation (7 groups)

Tranche implementation and acceptance: [completed tranche](scalar-config-foundation.md).
`DMANOP` state-preservation coverage is accepted in Stage B. `FLUSHDMA` is
deliberately excluded in favor of STALLWAIT.

Dependencies: existing `backend::set_gpr`, `write_word`, config bank selection,
`STALLWAIT` and runner readback. Extend `tt-isa/src/backend.rs`; put substantial
scalar models/helpers in a new module if keeping them there obscures config APIs.

- [x] `RDCFG`: checked GPR/config index, current bank only, no ThreadConfig
  indexing; return a sequence that waits for Configuration Unit completion
  before consuming the GPR. Gate both config banks and all three issuing threads.
- [x] `CFGSHIFTMASK`: typed eight ALU modes, replace/preserve mask semantics,
  scratch selection (index 3 means current thread), rotations 0–31 and widths
  representing 1–32 bits. Respect its eight-bit destination index; reject
  reset/special-effect destinations. Gate preserved bits and wraparound arithmetic.
- [x] `SUBDMAREG`, `MULDMAREG`, `CMPDMAREG`, `SHIFTDMAREG`, `BITWOPDMAREG`:
  checked register and immediate forms; subtraction wraps, multiply uses only
  each operand's low 16 bits, compares are unsigned and shifts logical with
  register counts masked to five bits. Cover operand/result aliasing, same and
  different four-GPR groups, boundary immediates and invalid modes.
- [x] Use independent integer/config models and a swapped operand/mask mutant.
  First consumer: a replayable descriptor/address calculation with changed
  parameters. Avoid changing tensor arithmetic or matmul ordering in this stage.

### B — L1 access, local bulk movement and synchronization (4 groups)

Detailed execution checklist: [l1-scalar-movement.md](l1-scalar-movement.md).

Dependencies: completed A. Add checked scalar memory helpers and kernel-side declared
buffer/address descriptors; firmware runner changes only if required for setup.

- [x] `DMANOP`: checked fixed opcode and state-preservation diagnostic only.
  Use STALLWAIT C0 for real outstanding scalar requests; never fixed bubble
  counts. FLUSHDMA remains excluded.

- [x] `LOADIND`/`STOREIND_L1`: byte/halfword/word/128-bit forms; base in 16-byte
  units, offset in bytes, half-register offset increments 0/2/4/16. Reject
  silent alignment rounding and misaligned four-GPR groups; validate the whole
  transfer range and offset wrap. Model partial loads preserving upper GPR bits.
- [x] Drain scalar memory requests with C0 before dependent reads or handing
  writes to another agent. Gate host-visible readback and cross-role consumption.
- [x] `XMOV`: initially expose L1-to-L1 copy and zero-to-L1 only. Check aligned
  source/destination/count, full extents, count-field limits and overlap policy
  (reject overlap initially). Configure THCON mover fields explicitly, drain
  conflicting configuration and wait on Blackhole C9 (`cond::MOVER_OUTSTANDING`) for completion.
  C12 (`cond::CONFIG_BUSY`) drains setup writes, not the transfer; the
  Wormhole XMOV page's C12 advice must not be copied to Blackhole.
  Leave configuration/NC instruction-RAM destinations outside the initial API.
- [x] Add a bounded resident copy/fill consumer with raw exceptional bits,
  guard regions, ragged tails, invalid ranges and changed-input traces. Compare
  release medians with existing B-core copies before considering routing.

### C — Source clearing, release and SrcB movement (4 groups)

Dependencies: existing `matrix::Banks`; A/B supply readback and diagnostics.
Step103 accepts all four constrained families (card-0 `1791397029`, 9/9;
simulator 6/6). Both-physical-bank, nonzero clear and retained-bank release
variants remain explicitly deferred in the [tranche plan](source-bank-clear-release-shift.md).

- [x] `ZEROSRC`: typed A/B selection and current-bank ownership; distinguish
  unpacker-selected versus matrix-selected banks. Allow both-bank clearing only
  with a quiescent-state proof. Start with zero; characterize negative-infinity
  physical encodings/implied formats separately before exposing them.
- [x] `CLEARDVALID`: implement only non-reset release/flip semantics. Keep
  `Reset` unavailable. A keep-reading-same-bank form must have a separate state
  model; it must not pretend to satisfy `Banks`' normal lockstep transition.
- [x] `UNPACR_NOP_ZEROSRC`: current unpacker bank only, `WaitLikeUnpacr` selected
  explicitly. Reject `BothBanks`; characterize the wider Blackhole clear-value
  field rather than copying Wormhole's negative-infinity assumptions.
- [x] `SHIFTXB`: loaded-B consumer with typed rotate/zero-fill variants, RWC
  row wrapping and address-modifier coverage including entry 4. Silicon-only
  under divergence 50; build an independent physical-row permutation oracle.
- [x] Gate alternating banks and programs, partial staging, final release,
  subsequent matmul/pooling, raw specials and changed-input replay. Use a
  wrong-row or wrong-clear-value mutant. Preserve the existing recovery path.

### D — Explicit unpacker handover (1 group)

Dependencies: C's ownership model and the operating notes for the earlier failed
UNPACR_NOP_SETDVALID recovery attempt that took down the host.

- [x] Implement `UNPACR_NOP_SETDVALID` for an already established, healthy
  unpacker bank after partial regular UNPACRs. Explicitly configure output
  format, retire preceding reads and establish the required bank-access wait;
  this instruction does not automatically perform that wait.
- [x] Extend `Banks<Filling, ...>` with an explicit handover transition;
  prevent duplicate and empty/cleared-staging handover. Format correctness is
  an explicit builder invariant and low-level precondition, not a Banks proof.
- [x] Gate equivalence to regular UNPACR's final FlipSrc, TF32/BF16 format
  changes, alternating banks and downstream consumers. Use data mutants only.
  This is not a replacement unwedge/reset mechanism.

### E — Mutexes and L1 atomic protocols (6 groups)

Dependencies: A/B. Begin with isolated protocol tests; existing scheduler
semaphores and credits are not replaced as part of instruction coverage.

- [ ] `ATGETM`/`ATRELM`: checked BH mutex indices **0, 2, 3, 4** (1 and values
  above 4 wait forever); explicit per-thread acquire/release scopes and state
  cleanup. Gate controlled contention and round-robin handoff, not timing guesses.
- [ ] `ATCAS`: typed four-bit compare/set values on a checked L1 word. It
  blocks/retries until equality; it is not a conventional nonblocking CAS that
  returns success. Ensure a progress path outside the blocked Scalar Unit.
- [ ] `ATSWAP`: typed eight-halfword mask over an aligned 16-byte region,
  single/four-GPR forms and preserved unmasked words. Assert no old-value return.
- [ ] `ATINCGET`: explicit 1–32-bit field width, preserved upper bits, wrapping
  increments, old-value GPR return and C0 wait before consumption.
- [ ] `ATINCGETPTR`: checked adjacent read/write counters, bounded counter
  width, push/pop/no-increment and batch size; reject free-running counters and
  increments incompatible with the capacity. Gate empty/full transitions and
  wrap with an independent FIFO model and a producer that can make progress.
- [ ] Allocate protocol storage through `Requirements`; test three-thread
  uniqueness/serialization and reset between programs. Blocking protocols need
  isolated runner deadlines and recovery; do not intentionally strand hardware.

### F — Restricted MMIO and packer-sequenced writes (4 groups)

Dependencies: A/B, a documented harmless writable register target and ownership
of TDMA-RISC state. No arbitrary MMIO-address API.

- [ ] `LOADREG`, `STOREREG`, `STOREIND_MMIO`: allowlisted aligned register
  targets; reject the forbidden region below `0xFFB11000`, unknown/destructive
  targets and unintended address truncation. Model STOREIND's shifted half-GPR
  offset separately from its L1 variant. Drain/read back before external use.
- [ ] `PACR_SETREG`: initialize `SetRegBase` and `SetRegHiScaler` explicitly
  through their documented path; check address/value construction and format
  dependence. Verify late pack conversion and buffer flush ordering, not merely
  an eventual MMIO write. If a harmless target/setup cannot be established,
  keep this row research-only rather than probe arbitrary registers.
- [ ] Gate config isolation across programs, guard targets, fresh/trace replay
  and restored state. Production adoption needs a measured consumer benefit.

### G — Minimal stream-overlay lifecycle (2 groups)

Dependencies: A/F and stream register definitions from the pinned Blackhole
tree. Current software GDDR streaming is not proof of overlay support.

- [ ] Declare exclusive overlay stream allocation, initialization, teardown,
  register access and thread-local stream selector/high target configuration.
- [ ] `STREAMWAIT`: typed phase/message conditions, full target splitting,
  selector 0–3 and B0–B8 consumer block mask; model zero mask's B6 default and
  latched wait behavior. Use a controlled producer for unmet-to-met transitions.
- [ ] `STREAMWRCFG`: checked stream register and config destination, reject
  config reset, and append Configuration Unit completion wait. Explicitly
  cover the documented reordering bug with later Configuration Unit operations.
- [ ] Probe ttsim support independently; add divergence-backed silicon arms if
  unavailable. Gate repeated sessions and changed-state replay. Automatic NoC
  dataflow adoption is a separate architecture/performance decision.

### H — Scheduled SFPU macros (1 group; S9)

Dependencies: mature SFPU interpreter/builder and explicit macro state lifecycle;
A readback helps diagnosis. Does not require stream/atomic stages.

- [ ] Extend `SFPCONFIG` beyond existing constants to checked macro misc,
  sequence and template setup. Declare persistent configuration in program keys
  and descriptors, initialize before use, and drain before reconfiguration.
- [ ] Add a macro schedule model for load plus Simple/MAD/Round/Store, delays,
  substituted operands, LReg16 and predication. Reject scheduling collisions,
  invalid template classes and undefined encodings. Account for the coupled
  VDHi/address bit, SFPSWAP restrictions and Simple/Round destination conflicts.
- [ ] Extend the step5 load-half evidence with each sub-unit independently,
  then a small load/compute/store chain compared bit-for-bit against the
  ordinary SFPU sequence/interpreter. Add swapped template/delay mutants.
- [ ] Silicon-only device execution (divergence 7); host model tests still run
  everywhere. Add changed-input trace and back-to-back different macro programs.
  Benchmark validated release medians before any SFPU default changes.

## Research and diagnostic closure

- [ ] `MOVDBGA2D`: extend step9 probes to one/eight-row semantics and format
  selection, with explicit valid ownership or a separately established format.
  Keep it a diagnostic surface; normal kernels already use MOVA2D's automatic wait.
- [ ] `GATESRCRST`: checked cache-invalidation diagnostic and a controlled
  changed-SrcB experiment. A no-effect comparison alone cannot prove invalidation;
  if no observable cache-state oracle exists, leave semantic completion open.
- [ ] `UNPACR_NOP_SETREG`: establish TDMA-RISC base/accumulator semantics and
  a harmless MMIO target before executing; otherwise document deferral.

The six deliberate exclusions above have accepted planning dispositions and do
not block tranche completion. They must not be relabeled semantic completions.

## Acceptance and handoff

For each stage, allocate the next available `stepNN` test filenames when work
starts (do not reserve numbers against concurrent work). Add supported semantic
and consumer gates to `xtask/src/silicon.rs` SMOKE. Instruction-only/diagnostic
stages state why Burn routing, gradients or tensor padding do not apply; stages
that add tensor APIs must test residency, parent padding, downstream consumers,
one/two-tile changed-input traces, deferred frees and ownership switches.

Run appropriate host unit/model tests and simulator gates, watched failing under
a safe deterministic mutant, then isolated release silicon gates on one card.
For kernel integration also run the selected smoke/model regressions and
`cargo test -p tt-tests --features e2e --test step12_mnist` for arithmetic,
routing or execution-order changes. Preserve the MNIST golden unless there is
an explained deliberate arithmetic change.

Required checks for affected work: workspace formatting/Clippy/tests, silicon
no-run compilation and Clippy, separate RISC-V firmware Clippy when firmware
changes, relevant generator `--check` commands and both shipping dependency
checks. If silicon is unavailable, leave acceptance open and report the unrun
gate. Benchmark new performance consumers with release output validation,
warmups, host timing, medians and `dataflow_stats`.

Update the authoritative coverage rows and active stage checkboxes with code
and run IDs. Record only newly measured facts in silicon operating notes,
simulator divergence entries and firmware performance conditions/results.
Move this plan to `docs/completed-plans/` only after every stage/research row has
an accepted implemented or deliberately deferred disposition.

**Recommended next tranche: Stage E.** Stages A–D are accepted (step100–104).
Stage D supplies bounded explicit healthy-bank handover after partial regular
UNPACR; recovery is unchanged. E/F/G and H retain their stated dependencies.


L1 movement close-out (2026-10-07): step101/102 deliver checked scalar memory
access, diagnostic DMANOP and explicit single-card tensor copy/zero. Card-0
semantic gates pass 12/12 (`1791389018`), SMOKE 233/233 (`1791389331`) and release
benchmarks 3/3 (`1791389713`). The completed tranche records the measured STOREIND
width correction and the minimal T2 output-ownership shell. Stage B entries
above are reconciled with this acceptance; Stage C is also accepted below.

Source-bank close-out: step103 provides typed current-bank zero/release/shift,
independent physical ownership/data models, bounded role sequences and clean
matmul/pooling reuse. Card-0 semantic acceptance is 9/9 (`1791397029`). The
measured UNPACR_NOP correction is in the generator, not a handwritten encoding.
Full card-0 release SMOKE passes 242/242 (`1791397621`); MNIST 8/8 with
unchanged golden. Workspace tests, format/Clippy, silicon compilation/Clippy,
generators and shipping checks pass. The accepted tranche plan is archived;
inventory at that acceptance was 91/17/six.

Explicit handover close-out: step104 measures Blackhole non-clearing mode 0x1e9
for both unpackers and provides Filling-only helpers plus bounded diagnostics.
Combined format/bank/replay gates pass `1791413233`; nonzero SrcB row reset and
direct consumers pass `1791413291`, final fractional inputs `1791413841`.
Card-0 release SMOKE passes 258/258 (`1791413478`); MNIST 8/8 retains its golden.
Workspace tests, format/Clippy, silicon no-run/Clippy, generator and shipping
checks pass. The old Wormhole mode 7 encoding is the isolated reboot trigger;
the internal ARC reset mechanism remains unproven. The Stage D checklist alone
is archived; this overall plan remains active at 92/16/six. Arbitrary disjoint
partial placement, hidden SrcA counter observation, other NOP modes and
production/performance/recovery adoption are deferred.
