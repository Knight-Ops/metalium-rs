# Configuration reads and scalar register arithmetic

Scope: RDCFG, CFGSHIFTMASK, SUBDMAREG, MULDMAREG, CMPDMAREG,
SHIFTDMAREG and BITWOPDMAREG. Explicit instruction foundation; default tensor
routing stays unchanged. DMANOP and FLUSHDMA belong to the transfer tranche.
Burn routing, gradients and tensor padding are inapplicable to these diagnostic
instruction programs. Descriptors are data and never submitted to a memory engine.

- [x] Checked configuration read with full Configuration Unit wait.
- [x] Checked mutation: eight ALU modes, two mask modes, four scratch selectors,
  width 1–32, right rotation 0–31, allowlisted scratch and THCON base words.
- [x] Checked scalar register/immediate APIs with unsigned, wrapping and low-16
  multiply semantics; reject invalid registers and immediates.
- [x] Independent pinned-pseudocode integer/configuration models and host boundaries.
- [x] Simulator per-form refusal probes with surviving controls; retain supported
  register multiplication, RDCFG and full-width unrotated preserved Add gates.
- [x] Card 0 semantic matrix for modes, selectors, banks, threads and aliasing.
- [x] Changed-parameter descriptor consumer, unrolled/replay structural assertion,
  intervening configuration and thread-local GPR isolation.
- [x] Deterministic data negative controls preserving waits and protections.
- [x] Final acceptance: isolated release card 0 suite including all scratch targets.
- [x] Final formatting, workspace tests, default/silicon Clippy, silicon no-run,
  generator checks and shipping dependency checks; MNIST golden unchanged.
- [x] Update coverage inventory and move this plan to completed-plans.

Implementation: `tt-isa/src/{backend,scalar}.rs`, `step100_scalar_config.rs`.
Only register MULDMAREG may be promoted through generator CONFIRMED: it passes
both simulator and silicon, exercising every field. Other scalar forms retain
WormholeOnly status because simulator refusal prevents satisfying that contract.
No firmware changes or benchmark claim.

Hardware scheduling evidence: card 0 replay run 1791342637 exposed the preceding
shift result at the final WRCFG when it immediately followed the scalar XOR.
An explicit full Configuration Unit barrier before publication passed changed
parameters (1791343224). Read and mutation helpers already include their waits;
the consumer also establishes the scalar-to-configuration publication boundary.
Do not interpret this narrow observation as a general scalar encoding correction.


Acceptance completed 2026-10-07:

- Simulator step100: 7/7, including refusal controls and supported semantic arms.
- Isolated release card 0: 9/9 (`1791379895`); preceding full run also 9/9
  (`1791343719`). Included in SMOKE. Card 1 was not needed.
- `cargo test --workspace`, both default/silicon workspace Clippy variants,
  silicon workspace no-run compilation, formatting and diff whitespace checks pass.
- `gen-isa --check`, `gen-cfg --check`, `gen-burn-ops --check`,
  `check-no-sim-in-ship` and `check-no-flex-in-backend` pass.
- MNIST e2e: 8/8, golden unchanged. No firmware changes, so separate firmware
  lint was inapplicable. No benchmark or speedup threshold for this foundation.
- Instruction inventory: 83 completed, 30 pending, one omitted family.
