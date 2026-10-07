# L1 scalar access and bulk movement — next instruction tranche

Plan date: 2026-10-07. Depends on the completed
[scalar/configuration foundation](../completed-plans/scalar-config-foundation.md)
(step100, card-0 run `1791379895`). Instruction inventory and exclusion policy:
[remaining-firmware-instructions.md](remaining-firmware-instructions.md).
Implemented in step101/102. Twelve isolated release card-0 semantic gates pass
(`1791389018`); validated release benchmarks are `1791389713`. Card-0 SMOKE passes 233/233
(`1791389331`). Final close-out evidence is recorded below.

## Scope and outcome

Implement checked `LOADIND`, `STOREIND_L1`, and `XMOV` for declared L1 buffers;
add diagnostic `DMANOP` coverage. These four related groups extend the delivered
scalar register foundation into memory access. They provide a bounded local
copy/zero primitive and readback infrastructure for later atomics/MMIO work.
`FLUSHDMA` is deliberately excluded: the supported completion mechanism is
`STALLWAIT`, which does not monopolize the shared Scalar Unit while waiting.

Initial consumer: an explicit single-card Session copy/zero operation over a
bounded staging buffer, with scalar transfers for tails smaller than 16 bytes.
B reads GDDR on NoC0, a Tensix role performs the local transfer, NC writes the
assembled result on NoC1. Add the API only after isolated instruction gates pass.
Keep automatic Burn copy/repack dispatch a separate performance decision; step98
already showed that fewer host-side copies do not establish a faster tensor path.
Do not change arithmetic, matmul order, scheduler credits or existing barriers.

## Delivered contract and evidence

The four explicit Session APIs are implemented, without automatic Burn routing.
`local_movement` declares one source/destination tile and two semaphores through
Requirements. Full tiles use one bulk copy. Ragged copies zero output, then copy
valid face-row spans with XMOV and naturally aligned scalar tails. Scalar tails
rebase each address and use increment None, so a final 16-bit offset update cannot
wrap. GPR 0 stages configuration; 8–11 hold data; 12/13 and 14/15 hold bases and
offsets. Outputs claim Pad::Zero (BF16 padding is inspected physically).

**Measured amendments to the initial plan:** Blackhole STOREIND width values are
0=16, 1=2, 2=4, 3=1 bytes; LOADIND retains the pinned mapping. This correction
changes semantic values in the checked helper, not generated field positions.
An empty T2 publishes NC's output credit too early: T2 must carry a minimal
ownership semaphore shell. Its runner reserves output, its shell posts ready to
T0 and waits for T0's done post, and its runner then publishes. T0 waits ready
before writing any configuration/data and posts done after full C0/C9 drains.
T1 is empty and every movement instruction remains on T0. All semaphores return
to their initial zero values. Firmware, mailbox ABI and scheduler barriers are
unchanged. The observed failure and correction are in the operating notes.

Instruction/reference gates: `step101_l1_movement`. Resident gates:
`step102_xmov_tensor`. Host byte-array interpreter/extent/stream audits:
`local_movement::tests`. Semantic run `1791389018` passes 12/12 on card 0;
card 1 is not claimed under the current one-card policy. Simulator probes refuse
LOADIND/STOREIND in every width, XMOV itself and mover configuration register 88;
NOP and DMANOP controls survive. Generated provenance stays unchanged.

## Sources and implementation locations

Pinned functional models: `LOADIND.md`, `STOREIND_L1.md`, `DMANOP.md`, `XMOV.md`
and `ScalarUnit.md` under
`vendor/tt-isa-documentation/WormholeB0/TensixTile/TensixCoprocessor/`, plus
`WormholeB0/TensixTile/Mover.md`. These are specification inputs, not proof of BH
encodings. Generated layouts currently retain WormholeOnly provenance.
Blackhole completion and block masks come from
`BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md` and existing
`tt-isa/src/backend.rs` constants.

| Layer | Planned change |
|---|---|
| `tt-isa/src/scalar.rs` | Typed transfer width, offset half-register/increment, checked load/store helpers and DMANOP diagnostic helper; split memory helpers into a module if needed |
| `tt-isa/src/backend.rs` | Paired scalar/mover waits with explicit consumers, using existing BH condition constants |
| `tt-kernels/src/l1.rs`, new local-copy builder and `session.rs` | Declared transfer extents, GPR/config clobbers, bounded copy/zero descriptors and Session integration |
| `tt-firmware/src/corpus.rs`, role runners | Audit end-of-program completion and replay setup; change only if the generic runner cannot express the required drain/publication contract |
| `tt-tests/tests/stepNN_*` | Independent models, isolated semantics, simulator support probes and resident consumer gates; choose free numbers when implementation starts |
| `xtask/src/gen_isa/{Bits32_BH.lua,measured.rs}`, `xtask/src/silicon.rs` | Only measured encoding corrections, if required; supported semantic/consumer gates in SMOKE |

## Checked memory contract

LOADIND/STOREIND_L1 form an address as `16 * base_gpr + offset_half_gpr`.
The offset is a selected 16-bit half of a GPR; its increment is 0, 2, 4 or 16
**bytes**, not a transfer-width selector. LOADIND width encodings are 0=128 bits,
1=32 bits, 2=16 bits and 3=8 bits. Measured Blackhole STOREIND exchanges values
1/2 (16/32 bits); the helper applies this correction. Partial loads preserve high result bits;
partial stores use only low input bits. A 128-bit transfer uses four GPRs.

- [x] Check GPR 0–63, offset-half index 0–127, four-GPR group alignment and
  complete destination/source allocation. Reject hardware's silent address and
  register-group rounding. Validate address addition/multiplication without
  wrapping, natural alignment, and the whole range inside the declared buffer,
  not merely inside physical L1.
- [x] Separate encoding validation in tt-isa from descriptor extent validation
  in tt-kernels. GPR contents are runtime state: an encodable register index is
  not proof of a safe address. Initialize addresses from checked descriptors;
  reject arbitrary address GPR contents in the Session surface.
- [x] Bound every loop iteration and offset increment, including the final
  update. Initially reject 16-bit offset wrap; split/rebase larger transfers.
  Preserve the unused offset half-register and unrelated GPRs.
- [x] Initially keep result/data groups disjoint from address and offset GPRs.
  LOADIND updates the offset before its asynchronous result arrives; aliasing
  can overwrite that update. STOREIND's offset update can also alter source
  data if aliased. Model these cases independently, reject them in the consumer,
  and expand the helper contract only with explicit semantic evidence.
- [x] Treat zero-length copy/zero as a checked no-op. For partial 16-byte tails,
  use aligned scalar words/halfwords/bytes without touching adjacent padding.
  Reject overlapping copy ranges initially; promise no memmove semantics.

## Completion and mover ownership

Blackhole **C0** drains scalar memory requests for the issuing thread;
**C9** drains the hardware mover across threads and TDMA-RISC;
**C12** drains configuration writes. The Wormhole XMOV page recommends C12
for mover completion; that bit number is wrong for Blackhole.

- [x] Add/use a scalar C0 wait that blocks Scalar Unit issue and every next
  consumer. Before host/RISC-V observation or semaphore publication, use a full
  barrier and the existing runner fence/publication sequence. A wait on one
  role does not drain another role's requests.
- [x] Drain configuration with C12 before XMOV reads the active bank's setup;
  program all THCON_SEC0_REG6 source/destination/count/direction fields explicitly.
  Do not inherit these fields across programs or config-bank changes.
- [x] Copy uses mover mode 3; zero uses mode 0. Both addresses/count are in
  16-byte units. Check the low-16 count limit (maximum 65,535 blocks per issue),
  field widths and full source/destination extents. Skip zero count. Restrict
  destinations to declared L1; backend configuration and NC instruction RAM
  targets are excluded from this tranche.
- [x] Establish exclusive ownership of the tile's hardware mover and its setup
  state through transfer completion. Audit TDMA-RISC users during bring-up and
  runtime; the B/NC software data movers are distinct from this hardware block.
  Serialize conflicting roles/config-bank mutations rather than treating C9 as
  a thread-local lock.
- [x] After XMOV, wait on C9, blocking new mover instructions (B4 or B0) and
  every consuming unit. Before overwriting mover setup, include Configuration
  Unit blocking. Publish/release buffers only after this boundary. A C12-only
  setup barrier never proves transfer completion.
- [x] Gate DMANOP's unchanged GPR/config/memory state. It occupies the Scalar
  Unit for one cycle; repeated DMANOP is never used to prove a load/store drain.

## Execution and acceptance

1. **Independent models and checked helpers.**
   - [x] Build byte-array/GPR reference models directly from pinned pseudocode,
     separate from encoding helpers. Cover each width, offset half and increment,
     high-bit preservation, alignment, boundaries, alias rejection and overflow.
   - [x] Build an independent copy/zero model and config address/count conversion
     checks; use checked errors for invalid extents without issuing hardware.
   - [x] Validate encoding fields/provenance; correct only measured BH layouts
     through the generator inputs and run `gen-isa --check`.
2. **Isolated instruction semantics.**
   - [x] Probe support per opcode/form in `tt_ttsim::fork_scope`, with a surviving
     supported control; keep parent lock-taking work in `outside_fork`.
     Record refusals as divergences rather than inventing simulator support.
   - [x] Gate loads/stores on T0/T1/T2: every width/increment and both offset
     halves, group/register boundaries, raw special-value bits and guard regions.
     Compare preserved state, not only copied output.
   - [x] Gate copy/zero over multiple sizes and both configuration banks, repeated
     programs and role handovers, with independently predicted guard bytes and
     explicit reinitialization. Include immediate dependent scalar reads and
     cross-role/host observation after real requests and completion barriers.
   - [x] Watch deterministic wrong-offset, wrong-width/high-bit and wrong-source
     mutants fail. Never remove waits/ownership checks as a negative control.
3. **Resident consumer.**
   - [x] Implement declared bounded staging and complete replayable descriptors.
     Allocate buffers/semaphores through Requirements; reject firmware/fixed-region
     overlap, and retain cache pins, trace holds and deferred frees.
   - [x] Validate byte-exact F32/I32/Bool and raw BF16 copies/zero, ragged tails,
     parent padding unchanged, and downstream reduction/matmul. Require positive
     evidence that Tensix performs the copy, alongside download/traffic audits.
   - [x] Test one/two-tile changed-input traces, deferred input/output frees and
     both TT_PIPELINE/TT_BATCH ownership settings. Keep the initial API explicit;
     Burn routing/gradients are inapplicable until automatic dispatch is proposed.
4. **Silicon and performance.**
   - [x] Run isolated release instruction/consumer gates and SMOKE through
     `cargo xtask silicon`; ordinary acceptance follows the tracker's current
     one-card policy. Both cards only for mesh/device-specific investigations.
     If unavailable, leave silicon acceptance open.
   - [x] Use `cargo xtask bench` with output validation, warmups, host-timed medians
     and dataflow_stats. Compare local transfer time separately from full GDDR
     end-to-end time against existing B-core/native repack paths; include small,
     large and ragged copies and actual staging/launch/drain costs.
   - [x] Record conditions/run IDs in firmware-performance.md. Adopt automatic
     routing only for measured beneficial cases with the same bit/padding contract;
     a slower correct explicit implementation can remain supported without routing.
5. **Close-out.**
   - [x] Run workspace formatting/lints/tests, silicon no-run compilation/lints,
     both shipping checks and relevant generator checks. If firmware changes,
     also run its separate RISC-V Clippy/instruction-image gate.
   - [x] Run step12_mnist for execution-order or routing changes; preserve the
     golden unless arithmetic deliberately changes.
   - [x] Update coverage checkboxes only for accepted forms, with gate/run IDs;
     log new simulator/hardware findings in the appropriate learnings. Move this
     tranche plan to completed-plans only after its acceptance checklist closes.

After this tranche, take source clear/release and SHIFTXB (Stage C), followed by
healthy-bank explicit unpacker handover (Stage D). Atomics, restricted MMIO,
stream overlay and SFPU macros remain separate dependency-driven tranches.

## Close-out verification (2026-10-07)

- `cargo fmt --all --check`, both workspace Clippy variants, silicon no-run
  compilation, all three generator checks and both shipping checks pass.
- Workspace unit/integration tests pass. The initial aggregate run's tt-mnist
  doctest could not resolve burn_tt (E0463); separate `cargo test --workspace
  --doc` passes. No source workaround or doctest suppression was introduced.
  Final tt-isa/tt-kernels library tests and the step101 simulator/model tests pass.
- MNIST e2e regression passes all eight gates, including two/four-chip simulator
  training; golden unchanged. No firmware source change or separate lint needed.
- Release semantic gates: `1791389018`, 12/12; final construction/submission
  failure cleanup gate `1791389916`, 1/1. Both 32-bit and BF16 rejected outputs
  return all GDDR allocation bytes; the session remains usable after cache refusal.
- Release SMOKE: `1791389331`, 233/233, card 0. Benchmarks: `1791389713`, 3/3,
  with conditions/medians/dataflow counts in firmware-performance.md. Copy is
  slower end to end than native copy; all APIs remain explicitly selected.

This closes the tranche under its one-card acceptance policy. Stage C and later
instructions remain separately scheduled in remaining-firmware-instructions.md.
