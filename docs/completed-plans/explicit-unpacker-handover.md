# Stage D — Explicit healthy-bank unpacker handover

Accepted on selected card 0, 2026-10-07. Inventory is 92 completed, 16 pending
and six excluded groups. Preserve step103 and keep the overall remaining-instructions
plan active. No production tensor, Burn, double-buffering or recovery changes.

## Implementation and acceptance

- [x] Allocate step104 (already present in the resumed working tree).
- [x] Add Filling-only A/B helpers returning three instructions and Loaded:
  drain C1/C2, wait for current-bank unpacker ownership C5/C6, then NOP.
  Both waits block Before::EVERYTHING. C10/C11 on the Wormhole page are not
  Blackhole ownership conditions.
- [x] Document configuration-bank, output-format, source-base and configuration
  retirement preconditions. Banks cannot validate external format writes.
- [x] Compile-fail empty, duplicate and cleared-staging publication on A/B.
- [x] Build bounded explicit/regular diagnostics with Requirements scratch and
  semaphores, retirement before publishing to math and explicit matrix release.
- [x] Independent physical data, ownership and pointer oracle; model format and
  source-row reset, special values, untouched rows and wrong format/row mutants.
- [x] Simulator regular controls and isolated A/B refusal probes.
- [x] Establish healthy-bank silicon regular control before candidate probes.
- [x] Measure A/B separately and together, both physical banks, TF32/BF16,
  multiple partial UNPACRs, source-row reset, repeated programs and data mutants.
- [x] Add matmul/pooling reuse after release in the same program, including
  changed-input replay (regular simulator control passes; explicit is silicon-only).
- [x] Validate format-sensitive computation and source-row reset on silicon.
  Readback and the host state model alone do not establish these.
- [x] Record measured encoder provenance and regenerate: bounded Blackhole
  non-clearing mode 0x1e9 replaces the legacy Wormhole mode 7. Only unpacker
  selection varies in the checked encoder.
- [x] Complete workspace checks, silicon compilation/lint, generator and shipping
  dependency checks, release SMOKE, step103 and unchanged-golden MNIST regression.
- [x] On acceptance only: 92 completed / 16 pending / six excluded; record run
  IDs and archive only this checklist. No diagnostic performance claim.

## Resumed failure and bisect

The historical sections below retain earlier session constraints and hypotheses.
Current contract: card-0 silicon confirms mode 0x1e9 for A/B, format changes,
alternating banks and direct matmul/pooling consumers. Run `1791413291` confirms
nonzero SrcB row reset; SrcA's required address override hides its counter.
No unsupported SrcA override mode was probed. Multipart coverage re-stages the
same range across multiple partial UNPACRs; disjoint placement is deferred.
The legacy mode 7 isolated reboot trigger is recorded in operating notes;
the exact ARC reset mechanism is unproven. Final regression checks pass.

The user reported a whole-system reboot during the previous attempt. Run
1791402960 contains START without END for regular_unpacr_control and
explicit_a_healthy_bank under boot a4e3b1bb-3f6c-4431-8a6e-dbd68cbe174a.
The resumed boot is 77ee3dd9-2c4e-4700-971b-3cbc2dd1a4e4. This does not isolate
which probe caused the reboot or prove an ARC cause. Do not bless the encoder.
The NVMe repository is read-only in the current sandbox (the host mount state
has not been established) and some compiled executables are data, not ELF; a fresh recovery copy is /home/carl/metalium-stage-d-recovery.

Use normal watchdog settings as explicitly requested by the user; do not change
recovery. The present session rejects outside-sandbox execution, and its sandbox
has no card device. Run each boundary separately on one discovered card. Do not
use --keep-going; any failure/timeout stops candidate probes pending diagnosis.

1. Established step103 surviving_uncleared_control_and_wrong_selection_mutant.
2. step104 bisect_configuration_only (no source instructions or role semaphore loop).
3. step104 bisect_single_bank_regular_control (one partial bank, regular final
   FlipSrc, serialized roles, A/B row readback and both releases).
4. step104 regular_unpacr_control (the original concurrent alternating-bank control).
5. Only after the controls pass: bisect_single_bank_explicit_a, then
   bisect_single_bank_explicit_b, then the full explicit gates.

Invoke each with cargo xtask silicon --release --device 0 --filter
step104_unpacker_handover::<test> (step103 uses its own binary). Discover the
available card first; device 0 is the previous selected card, not a grid oracle.
Do not execute silicon tests with cargo test. Test processes and build artifacts
must come from the fresh copy, not reboot-corrupted binaries.

## Recovery-copy checks

Step103 simulator regressions: 6/6. Step104 host/simulator controls, downstream
reuse and refusal assertions: 6/6. ISA doctests: 34/34, including all six
handover compile-fail cases. gen-isa --check passes; no NOP provenance promotion.
Workspace Clippy, silicon no-run compilation and silicon workspace Clippy pass
after the final downstream-test addition. Both shipping dependency checks,
format and diff checks pass. MNIST e2e passes 8/8 against the unchanged golden (263.51 s). Full workspace
tests pass (20:28 UTC); hardware acceptance and release SMOKE remain pending. The user explicitly approved card-0 access, but
the execution tool rejects require_escalated because sandbox_approval is false;
no filesystem permission-request tool is available in this session.

Resume note (updated 20:28 UTC): full workspace tests passed;
MNIST e2e subsequently passed all eight tests, including the four-chip ring
golden (20:22 UTC). Logs are /tmp/stage-d-workspace-tests.log and /tmp/stage-d-mnist.log.
Both suites have finished with exit status 0.
The resumed-change patch is /home/carl/stage-d-resume.patch; git apply --check
succeeded against the original tree. User changed sandbox configuration to add
cards 0 and 1, but this session still cannot see them; restart/resume is advised.
Do not repeat permission questions: card access and hardware bisect are authorized.

Configuration diagnosis: the user pasted a different config containing
[sandbox] allowed_paths, which is not a documented Codex setting. The supported
replacement is a named permissions profile extending :workspace with explicit
read/write filesystem entries and top-level default_permissions. Any loaded
sandbox_mode or --sandbox flag takes precedence over permission profiles.
The file visible in this execution environment has no permission entries and
an October 4 modification time. Actual device access is still pending; do not
infer host device absence from the sandbox's private /dev mount.

## Silicon bisect result (latest, 2026-10-07)

Device access is available through the elevated isolated runner. The historical
recovery-copy/device-visibility notes above describe earlier sessions.

- [x] Re-establish step103 regular control (`1791406158`).
- [x] Configuration-only (`1791406171`), one-bank regular (`1791406183`) and
  alternating regular (`1791406197`) boundaries pass card 0 in sequence.
- [x] Identify the independent failed boundary: one-bank explicit A,
  `1791406213`, START without END before the second reported reboot.
- [x] Bisect the waits independently with regular publication: C1/C5 A
  (`1791407483`) and C2/C6 B (`1791407513`) pass. Boot remains stable past
  the normal 10-second watchdog interval after both controls.
- [x] Record the isolated software trigger and limits in
  [silicon operating notes](../learnings/silicon-operating-notes.md#stage-d-silicon-reboot-bisect-2026-10-07-latest-evidence).
- [x] Ignore all explicit candidate/consumer silicon gates by default. No
  handover provenance promotion, smoke entry or inventory acceptance.
- [ ] Obtain independent Blackhole encoding evidence before another candidate
  probe. No arbitrary opcode-bit sweep or wedged-tile probe is authorized by
  this finding. The inherited `0x43000007` is not an accepted handover.
- [ ] If finer causality is required, distinguish NOP issue/retirement from its
  first matrix consumer using durable phase records, and obtain hypervisor/ARC
  reset evidence. Guest journal alone does not prove the reset mechanism.

The isolated trigger is replacing healthy-bank regular publication with the
inherited A handover word in the single-bank program. Its internal effect and
ARC watchdog causation remain unresolved; do not record either as measured.
Stage D stays active at 91 completed / 17 pending / six excluded.

Independent encoding explanation: the official Blackhole opcode macro places
Set_Dvalid in bits 8..11 and Unpack_Pop in bits 0..1. The failed Wormhole word
has Set_Dvalid=0, Src_ClrVal_Ctrl=1, Unpack_Pop=3. The encoding mismatch is now
identified; the correct Blackhole handover mode still needs independent semantic
and silicon acceptance. See the source link and limits in the operating notes.

Latest verification after the bisect: step103 and step104 host/simulator tests
pass (6/6 each); silicon-feature Clippy for tt-tests and its dependencies passes.
Candidate A/B and explicit downstream gates are ignored on silicon by default.

## Measured mode and final validation (supersedes earlier candidate status)

Official immutable Blackhole LLK revision
`201312fe7960b3711a420e2d655d420a39b0230e` documents clear-to-one format
selector 3 as DVALID-only. Measured A/B words are `0x430001e9` /
`0x438001e9`. Generator fixed-mode replacement credentials preserve opcode
and named-field protection; only WhichUnpacker is exposed. Gates using the
measured profile are enabled; rejected delay-mode research gates stay ignored.
Operating notes retain all documented hypotheses, failures and recovery run IDs.

- A and B independent raw semantics: `1791412640`, `1791412708`.
- Full checked alternating-bank A/B: `1791412921`, `1791413005`.
- Combined formats, multiple partials, changed-input replay/data mutant:
  `1791413233`.
- Direct explicit matmul/pooling and nonzero SrcB row reset: `1791413291`.
- The initial multipart failure `1791413005` was in the regular control:
  each partial restarted output addressing. Re-staging the same range corrects
  the control and retains bounded multiple-partial coverage. Arbitrary disjoint
  placement is deferred.

The measured row reset is SrcB with update enabled and Base=1, observing the
next bank at row 16 rather than row 32. SrcA's supported address override hides
its counter, so no direct hidden-counter claim or unsupported override probe is
made. Special-value physical readback, source format changes and independent
row/format oracle mutants remain part of the gates. Recovery behavior and
normal ARC watchdog timeout 10 are unchanged. The legacy mode 7 encoding error
is the isolated reboot trigger; internal watchdog/NoC causation is unresolved.

## Acceptance record

Selected card 0 release SMOKE: 258/258 (`1791413478`), including step103 9/9
and step104 16/16. Final fractional TF32/BF16 direct-consumer gate:
`1791413841`, 1/1. No new reboot; boot ID remains
`14d13ced-d7bb-4c81-ba18-505d1dd8c540`.

Workspace tests, workspace format/Clippy, silicon no-run compilation and
workspace Clippy, gen-isa --check, generator tests (46), and both shipping
dependency checks pass. Step104 host/simulator passes 8/8, step103 6/6, and ISA
compile-fail doctests pass. MNIST e2e passes 8/8 (275.03 s), golden unchanged.
The final fractional regular consumer also passes its targeted simulator gate.
Logs: /tmp/stage-d-final-{workspace,clippy,silicon-clippy,no-run,generator,host,mnist,smoke}.log.
No firmware changes require a separate firmware lint. Only this checklist is
archived; the remaining-instructions plan is active. Inventory: 92 completed,
16 pending, six excluded. No performance claim or production adoption.
