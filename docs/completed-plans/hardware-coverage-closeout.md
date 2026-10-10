# Hardware coverage close-out

> **Status: complete (2026-10-10).** Moved here from `docs/plans/`. The open items are carried in
> [hardware-coverage.md](../plans/hardware-coverage.md) ("Where things stand"); the permanent
> silicon-handling rules are in [silicon-operating-notes.md](../learnings/silicon-operating-notes.md).
> Code paths named below are as of the lanes; the lane scaffolding was later folded away (see
> "Scaffold lanes use").

Record of the effort that finished [hardware-coverage.md](../plans/hardware-coverage.md) and
[remaining-firmware-instructions.md](remaining-firmware-instructions.md). Started 2026-10-09.
Done means every coverage row is `[x]` or a documented `[-]` with evidence and a supported
alternative. A `[-]` exit is a legitimate outcome when the hardware cannot be driven safely or has no
observable oracle; a guess presented as a result is not.

Legend as in the coverage tracker: `[x]` done and gated · `[~]` partial · `[ ]` not started ·
`[-]` deliberately not done, reason given.

## Working rules

- **Lane agents** work in their own git worktree. They run host unit tests, ttsim gates (inside
  `fork_scope`) and Clippy for the crates they touch. **They never run silicon.**
- **The coordinator** owns integration and every silicon run, one at a time through
  `cargo xtask silicon --release --device 0 --filter <name>`. Both cards only for mesh work (T9) and
  device-specific measurements (T6 PRNG, T7 FP16 packer rounding).
- **Silicon risk order:** documented, then measured, then UNVERIFIED/WormholeOnly (always an
  isolated minimal probe first, after a ttsim pass or a recorded refusal), then NoC-hang class (one
  probe per session, nothing queued behind it). A hang is fixed in the API, never retried.
- **A lane ends with a report** holding: the diff summary; a silicon request (isolated probe, the
  semantic gate, a mutant expected to fail); proposed text for SMOKE lines, `measured.rs` /
  `Bits32_BH.lua` rows, divergence rows, operating notes and performance rows; and the negative
  control it watched fail.
- **Coordinator-only files** (lanes propose text, never merge it): `xtask/src/silicon.rs` (SMOKE),
  `xtask/src/gen_isa/{Bits32_BH.lua,measured.rs}`, `crates/tt-isa/src/isa/{generated.rs,mod.rs}`
  (provenance counts at `mod.rs:511-524`), `crates/burn-tt/src/generated/ops.rs`,
  `docs/learnings/*`, `docs/plans/hardware-coverage.md`, `remaining-firmware-instructions.md`.
  A lane that needs a measured layout to compile may edit the Lua/`measured.rs` locally; the
  coordinator re-applies it after rebase and regenerates. Generated files are never merged as text.
- **Definition of done** for any op is the AGENTS.md list: independent oracle (never an epsilon),
  ttsim gate with a negative control watched failing (or a divergence row plus a silicon-only
  gate), silicon validation, native Burn routing, residency and padding checks, changed-input trace
  replay where applicable, a SMOKE entry, learnings updated.

## Scaffold lanes use (Wave 0, done)

*Historical.* The scaffold below existed so parallel lanes never edited the same lines. It was
removed after the lanes finished: Burn overrides are one `OVERRIDDEN` list in
`xtask/src/gen_burn.rs`, the lane op modules are `crates/burn-tt/src/ops/{index,scan,sort}.rs` plus
`ops.rs` (re-exported by name), and the `lane:` anchors in `server.rs` are gone.

- **Burn overrides:** each tensor lane lists its methods in its own
  `xtask/src/gen_burn/overrides/<lane>.rs` (`pub const OVERRIDDEN`). `gen_burn::overridden()`
  concatenates the base list and every lane; a trait may appear in several lists, a method in only
  one. Regenerate with `cargo xtask gen-burn-ops`; on a merge conflict in
  `generated/ops.rs`, take either side and regenerate.
- **Burn implementations:** each lane writes its code in `crates/burn-tt/src/ops_<topic>.rs`
  (`pub mod float`, `pub mod int`, `pub mod bool`); `ops.rs` already glob re-exports all of them,
  so lanes do not edit `ops.rs`. Files: `ops_intbool` (T1), `ops_index` (T2), `ops_scan` (T3),
  `ops_rem` (T4), `ops_sort` (T5), `ops_dtype` (T7).
- **Engines:** `crates/burn-tt/src/server.rs` has a `lane:<name>` anchor for each tensor lane in
  the `Engine` trait, `KmdEngine` and `MeshEngine`. Add engine methods under your own anchor only.
  `tt-kernels` lanes add `impl<T: Transport> Session<T>` blocks in their own new module.

## Pre-assigned numbers

Step test files, divergence rows and SFPU kind ranges are assigned here so concurrent lanes do not
collide. Gaps are fine and are noted at close-out. The earlier rule "allocate step numbers when work
starts" (`remaining-firmware-instructions.md`) is superseded for this effort.

| Lane | Steps | Divergence rows |
|---|---|---|
| E mutex/atomics | 105–106 | 76–77 |
| F MMIO/PACR_SETREG | 107–108 | 78–79 |
| G stream overlay | 109 | 80–81 |
| H SFPLOADMACRO | 110 | 82 |
| MD matrix diagnostics | 111 | 83–84 |
| PU packer/unpacker modes | 112–114 | 85–87 |
| N NoC multicast/atomics/IRQ | 115–117 | 88–90 |
| X6 mover fast path | 118 | 91 |
| FENCE (X7) | 119 | – |
| TS tag search | 120 | 92 |
| P9 wait planner/.ttinsn | 121–122 | 93 |
| T1 int/bool | 125–126 | 94 |
| T2 indexing | 127–129 | 95 |
| T3 scans/arg-extremes | 130–132 | 96 |
| T4 remainder | 133–134 | 97 |
| T5 sort | 135–139 | 98 |
| T6 random | 140–142 | 99–100 |
| T7 FP16 | 143–144 | 101 |
| T8 MathMode | 145–146 | 102 |
| T9 mesh trace | 147 | 103 |
| T10 benches/dispositions | 148–149 | – |

At close-out: steps 109 and 122-124 were never used (G and P9 closed `[-]` without a gate file;
123-124 were never assigned), and 148-149 (T10) went into `silicon_bench_tensix_ops.rs` and
`silicon_perf.rs`. Divergence rows 80-81, 91, 93-98 and 103 were reserved and not needed.

SFPU kind ranges in `tt-kernels/src/sfpu/ops.rs` (all free at the time of assignment):
T4 `0x141–0x14f`, T3 `0x150–0x15f`, T1 `0x161–0x16f`, T7 `0x195–0x19f`, T5 `0x1b1–0x1bf`,
T6 `0x1c0–0x1cf`, T8 `0x1d0–0x1df`.

## Hardware/ISA lanes

- [x] **E: mutexes and L1 atomics** (steps 105-106; all six groups `[x]` on card 0 with ATSWAP single form `[-]`; blocking forms are freed by the host or a RISC-V poke, never by a Tensix thread, confirmed on silicon; Light is the default poll design, Full hangs after its first poll; measured 34.7M polls/s; `harness` preamble now uses the session reset).
  Instructions: `ATGETM`, `ATRELM`, `ATCAS`, `ATSWAP`, `ATINCGET`, `ATINCGETPTR`. Owned new
  `tt-isa/src/sync/mutex.rs`, `tt-isa/src/scalar/atomic.rs` (the role guard is
  `tt-isa/src/mailbox/guard.rs`), `tt-kernels/src/atomics.rs`, and
  `tt-firmware/src/corpus.rs`. Needs a bounded role-side deadline, a host-visible blocked status and
  release from another thread or the host (today `unwedge()` only posts semaphores). `Mutex` is
  constructible only for indices {0,2,3,4}. Probe order: `ATINCGET`, `ATSWAP`, `ATCAS` with the
  compare already met, uncontended `ATGETM/ATRELM`, then blocking forms with the deadline active.
  The four Wormhole-only instructions are probed in isolation. Exit: an op whose isolated probe
  misbehaves becomes `[-]`; a blocking form is never exposed without the deadline path.
- [x] **F: restricted MMIO** (steps 107-108; `PACR_SETREG` and `UNPACR_NOP_SETREG` `[-]`; the MMIO trio `[x]` on the allowlisted PIC words, host, ttsim and card-0 silicon gates pass, run one probe at a time with no reboot).
  Instructions: `LOADREG`, `STOREREG`, `STOREIND_MMIO`, `PACR_SETREG`,
  `UNPACR_NOP_SETREG`. Owned `tt-isa/src/scalar/mmio.rs`, not `datapath.rs`.
  Research first: one documented harmless, writable, host-readable register (reject anything below
  `0xFFB11000`); targets are an allowlist enum, never raw addresses. Exit: `[-]` for the MMIO
  trio if no harmless target exists; `[-]` for `PACR_SETREG` without a documented
  `SetRegBase`/`SetRegHiScaler` path; `UNPACR_NOP_SETREG` expected `[-]`.
- [x] **H: `SFPLOADMACRO` (S9)** (step110; probes s00-s13 pass on card 0; five mutations watched failing on the host model; the early hangs were a tile wedged by an earlier aborted E gate, fixed by the harness preamble now using the session's semaphore-releasing reset). Owned `tt-isa/src/sfpu/load_macro.rs`,
  `tt-kernels/src/sfpu/macro_sched.rs`. Schedule model for load plus Simple/MAD/Round/Store,
  delays, LReg16, predication; rejects collisions, the VDHi coupling and `SFPSWAP` restrictions.
  Gate: bit-for-bit against the ordinary sequence in the interpreter; mutants swap template and
  delay. Silicon-only (divergence 7). Exit: a contradicting sub-unit becomes a `[-]` sub-form.
- [x] **MD: matrix diagnostics** (step111; `MOVDBGA2D` `[x]` for 1/8-row, formats, flush, bank ownership on card 0; `GATESRCRST` `[-]` `NoObservableOracle` after a clean isolated probe; the increment-1 `AddrMod` cases are resolved: bit 14 widens a one-row move to four rows, the model and oracle match, and the six cases pass on card 0).
  Instructions: `MOVDBGA2D`, `GATESRCRST`. Owned `tt-isa/src/matrix/debug.rs`.
  `GATESRCRST` needs an observable stale-versus-fresh difference with SrcB loaded through
  `MOVD2B`. Exit: `GATESRCRST` becomes `[-]` "no observable oracle" if both arms match.
- [x] **PU: packer and unpacker modes** (steps 112-114; card 0: BF16 `UnpackToDst`, packer ReLU, edge masking, tileize all pass; unpacker transpose is not payload-preserving so M3/D5 are `[-]`; the 16-bit Dst read path preserves normals only; five negative controls watched failing on ttsim; not routed into Session/Burn ops).
  Owns `tt-kernels/src/datapath.rs`. In order: packer ReLU,
  edge masking, BF16 `UnpackToDst` (silicon-only, row 31; closes the D1 row), then unpacker
  transpose/tilize/broadcast with every new mode value isolated first (the Wormhole mode-7 precedent
  rebooted the host). The last sub-tranche also decides M3 and D5. Exit for M3/D5: `[-]` with the
  mover's `READ_TRANSPOSED`/repack as the payload-preserving contract and host or mover tilize as
  the alternatives.
- [x] **N: NoC multicast, atomics, completion interrupts** (steps 115-117; host/ttsim 16/16; NoC atomics `[x]` on card 0 (21 forms); completion polling `[x]` with the RTZ source sticky until cleared; IRQ handler `[-]`; multicast `[x]` on cards 0 and 1, probes run one at a time with no reboot).
  Owns `tt-isa/src/noc.rs` and a new
  probe bin; does not touch `mover.rs` until its gates pass. Multicast rectangle only from the
  ARC-discovered grid, writes only, completion by counting acknowledgements from the known
  recipients. Atomics from BH `NoC/Atomics.md`, L1 targets only. Interrupts: the
  `NIU_TRANS_COUNT_RTZ` polling form first. Silicon order: unicast atomic to a neighbour, 1×2
  multicast, full grid. Exit: the IRQ handler `[-]` if it cannot be isolated; device multicast
  stays `[~]` host-TLB only if the 1×2 probe is not clean.
- [x] **FENCE: X7 posted-write fence** (step119; unit 11 + ttsim 5; probe on card 0 and both-card Ethernet gates pass; three negative controls watched failing; `silicon_eth_link` gates need both cards so they are not in SMOKE). Owns `tt-device/src/device.rs`. A `FencedWrite` API bundling
  posted writes with one read-back; migrate call sites that race another agent.
- [-] **TS: `L1CacheTagSearchAccel`** (step120; host/ttsim 8/8; silicon: every armed trigger load hangs the baby core and leaves the block armed across resets (all ten scenarios stop at step 0 LOAD_ISSUED, and the earlier passing minimal probe fails afterwards) — `[-]` with that evidence; silicon tests `#[ignore]`d).
  Owns `tt-isa/src/tag_search.rs` and a probe bin. Done when
  search, invalidate-all and the bit-vector query are gated; adoption not required.
- [-] **G: stream overlay** (research complete 2026-10-10: no pinned Blackhole overlay register map, no controlled producer, host-reboot precedent; `STREAMWAIT` and `STREAMWRCFG` are `[-]` with the missing facts listed in `hardware-coverage.md`; steps 109 and rows 80-81 unused).
  Instructions: `STREAMWAIT`, `STREAMWRCFG` (wave 2, after F's register research).
  Likely `[-]`: Blackhole has no NoC Overlay documentation pinned, and Wormhole offsets are not
  adopted.
- [x] **X6: mover NIU fast path** (step118; ttsim and card 0, persistence probes then gate then both
  controls one at a time, no reboot; release medians run `1791600377`: fast/slow 0.96-0.97 on every
  shape, ~17 ns of ~0.48 us per request, so opt-in and not the default). The registers persist; the gain
  is small because the request's round trip, not its register writes, is the cost.
- [-] **P9: wait planner and `.ttinsn` fusion** (step121, host gate, no card; checker `tt_isa::hazard` over 954 builder role programs finds 0 missing waits and 1,724 redundant waits, so a planner has nothing to insert; `.ttinsn` needs a run-time code generator outside the instruction gate; seven negative controls watched failing).
  (wave 3, after E/F/H/PU). The planner's waits must
  be a superset of today's hand-written `Before::` waits and keep the MNIST golden. `.ttinsn` is
  measured first; `[-]` if no measurable push bottleneck or if it weakens the build-time instruction
  gate.

## Tensor and Burn lanes

- [x] **T1: int and bool wiring** (steps 125-126; simulator and card 0 15/15, runs `1791560877`,
  `1791560886`; four negative controls watched failing on ttsim; `int_abs` has no mutant; divergence row 94 unused). `int/bool_{permute,flip,unfold}`, `int/bool_mask_{where,fill}`
  (raw-bit scalar kind, since the f32 scalar loses bits above 2^24), `int_abs` (wrapping; check
  Flex's `i32::MIN` first), `int_cast` (I32→I32 native, other widths `[-]`). Makes `int_clamp*`,
  `int_sign`, `int_max_abs*` work. Negative control: fill through the f32 scalar must fail.
- [x] **T2: indexing compositions** (steps 127-129; simulator 19/19 and card 0 20/20 including the first
  silicon BF16 gather; seven negative controls watched failing on ttsim; scatter_nd Mul/Min/Max `[-]`;
  divergence row 95 unused). `float/int_gather_nd`, `float/int_scatter_nd` (Add via
  `select_add`; Assign with a stated last-writer-in-index-order contract; Mul/Min/Max `[-]`),
  `float_cross` (oracle uses the device's mul/sub rounding), `int_matmul` (exact mod 2^32; size
  budget refusal), a `prelu` weight-shape-[1] residency test.
- [x] **T3: scans and integer arg-extremes** (steps 130-132; simulator 11/11; card 0 11/11, runs
  `1791560292`, `1791560296`, `1791560300`; eight negative controls watched failing on ttsim; silicon
  mutants not run; SMOKE entries added; divergence row 96 unused). Owns `tt-kernels/src/sfpu/scan.rs`. New
  `ScanOp::{MinNaN,MaxNaN}` with Flex semantics (NaN propagates; on equal values including ±0 the
  earlier element is kept), integer `ISum/IProd/IMin/IMax`, `int_argmax/argmin`. The existing
  total-order Min/Max must fail the new gate on `[+0,-0]` and NaN inputs.
- [x] **T4: exact remainder** (steps 133-134; simulator 11/11 and card 0 11/11; four controls watched
  failing; deviation: `1e30 % 3` is exactly 0 so it cannot fail the trunc form, `4 % -2` and `1e30 % 7` do;
  divergence row 97 unused) (`float_remainder{,_scalar}`). Bit-exact to Flex's
  `((a%b)+b)%b`; exact SFPU fmod by exponent alignment and chunked integer long division. Confirm
  S1 add is IEEE round-to-nearest-even first, else the oracle uses `fma_bh`. Negative control:
  `a - b*trunc(a/b)` must fail on `1e30 % 3` and `-4 % 2`.
- [x] **T5: device sort family** (steps 135-139, later merged into 135-136; simulator and card 0 all pass; three network-mutation
  controls watched failing on ttsim; BF16 value gate corrected to expect the documented narrowing
  flush; performance not measured; axis bound 1024 by design). Orderable integer key
  `x ^ ((x>>31) as u32 >> 1)` reproduces `total_cmp`; bitonic network along the axis with an index
  tile; ties break on original index; sentinels sort last; explicit axis-length bound. Routes
  `float/int_sort`, `_sort_with_indices`, `_argsort`, `_argtopk`; `topk` follows through Burn's
  default. Negative controls: a dropped stage; a missing tie-break.
- [x] **T6: native random and dropout (S7)** (steps 140-142; ttsim 9+6+8 and cards 0/1: lane model 8 seeds x 32 lanes exact, kernel 6/6, Burn 8/8; three negative controls plus three statistical controls watched failing; traces refuse random rather than repeat seeds; divergence rows 99-100 proposed in the lane report: lane offset 96 vs 98 and the settle interval).
  Per-tile seeding through a RISC-V config store, fence
  and 512 NOPs (step91). Bit-exact host model of the hardware stream; pinned χ² and
  Kolmogorov–Smirnov tests; Bernoulli, uniform, normal (Box–Muller over the S4 programs),
  `int_random`. Replay advances a GDDR seed buffer or random is refused inside traces.
- [x] **T7: FP16 (D1)** (steps 143-144; ttsim 7+6 and card 0 gates pass; shipped casts are exact SFPU programs, the packer's own FP16 conversion is measured and not used; `native_conversions_match_the_measured_model` still has its `measured::SILICON` constants to fill from the recorded `T7-MEASURE` lines).
  `Elem::F16` and `StorageFormat::F16`, packing to `L1Format::Fp16`; measure
  packer rounding/overflow/subnormal behaviour on both cards; `float_cast` F32/BF16↔F16; F64 `[-]`.
- [x] **T8: `MathMode::{Precise, Approx}` (S10)** (steps 145-146; ttsim and card 0 10/10; six ops 1.24x-8.24x per tile).
  Session setting `TT_MATH=approx`, in the program
  memo keys, Precise default. Bounds derived beside each gate. Negative controls: Approx must break
  the Precise bound somewhere; alternating modes must give different bits.
- [x] **T9: mesh trace capture (R4)** (step147; ttsim 5/5 and card 0/1 silicon 3/3; per-chip session traces between host-run Ethernet transfers, `UnheldTransfer` refusal; two negative controls watched failing; mesh training traces and `copy_into` on a mesh still unsupported).
  `MeshEngine::begin_trace`; changed-input replay matches a
  fresh run with no uploads. Both cards.
- [x] **T10: benchmarks and dispositions** (K-block and norm release benchmarks run on both cards, runs `1791593690`/`92`/`99`, validated, recorded in `firmware-performance.md`; M2/M3/D5 dispositions applied; the lane's stale-statement audit list for `burn-native-cutover.md` and `burn-backend-parity.md` is not yet applied).
  Release K-block and norm benchmarks (P2/R3); M2 `[-]`
  GMPOOL diagnostic-only (packed ArgMax index bits missing on both cards, run `1791240702`); record
  PU's M3/D5 dispositions.

## Out of scope, with explicit failure

These keep their `unsupported::fail` stubs and get a `[-]` disposition with a test that the panic
names the operation: `interpolate` (+ backward), `conv3d`, `conv_transpose3d` (+ backwards),
`deform_conv2d` (+ backward), `rfft`, `irfft`, `float_grid_sample_2d`, `ctc_loss`, and
`QTensorOps` (D2 is backend storage, not portable quantization).

## Wave order and integration

Wave 1 (parallel): E, F (research first), H, MD, PU, N (host/ttsim only), FENCE, TS, T1–T10.
Wave 2: G, X6, T3b (`int_argtopk` on T5), T5b (topk in an attention trace). Wave 3: P9, then
coordinator integration.

Merge order: FENCE, TS, MD, T1, T2, T10, H, T3, T4, T8, F, E, PU, T7, T9, N, G, X6, T5, T6, P9. After
each merge: rebase; regenerate and `--check` `gen-isa`, `gen-burn-ops`; run the lane's ttsim
gates; run its silicon queue; update provenance counts and docs in the same commit as the code.

Silicon queue order: FENCE, TS, H load-half, `ATGETM/ATRELM`, packer ReLU, BF16 `UnpackToDst`;
then `MOVDBGA2D`; then UNVERIFIED singletons (`ATINCGET`, `ATSWAP`, `ATCAS`, `GATESRCRST`, the MMIO
allowlisted target, `PACR_SETREG`, unpacker mode values); NoC probes last, one per session.

Close-out: every coverage row `[x]` or documented `[-]`; move this file and
`remaining-firmware-instructions.md` to `docs/completed-plans/`; rewrite "Where things stand" in
`hardware-coverage.md`.

## Baseline and Wave 0

- [x] Baseline on `e5ce106` plus the Wave 0 scaffold (which changes no behaviour): release card-0
  `cargo xtask silicon --smoke` **258/258** (run `1791558187`); `cargo test -p tt-tests --features
  e2e --test step12_mnist` 8/8 in 263 s, golden unchanged. Workspace Clippy (default and
  `tt-tests/silicon`), `fmt`, `gen-burn-ops --check`, `burn-coverage --check`,
  `check-no-sim-in-ship` and `check-no-flex-in-backend` pass.
- [x] Wave 0: per-lane override lists and op modules, engine anchors, `cargo xtask burn-coverage`
  (generated `burn-op-coverage.md`, run by CI and prek), refusal tests for the out-of-scope module
  ops (`crates/burn-tt/tests/out_of_scope.rs`), stale tracker lines fixed.
- Concurrency limit: 12 cores / 62 GB. Run at most four lane agents at a time, each with its own
  worktree `target/` and `CARGO_BUILD_JOBS=3`; each worktree needs `vendor/` copied from the main
  checkout (it is git-ignored).

## Final state (2026-10-10)

Branch `phase10-closeout`. Verification at the final commit: the whole default workspace test suite
passes, workspace Clippy (default and `tt-tests/silicon`), firmware Clippy, `fmt`, `gen-burn-ops
--check`, `burn-coverage --check`, `check-no-sim-in-ship` and `check-no-flex-in-backend` pass, the
eight MNIST end-to-end regressions pass with the golden unchanged (265 s), and the release card-0
smoke tier passes **486/486** (run `1791597495`; baseline before this effort 258/258, `1791558187`).

### Disposition of every lane
| Lane | Outcome |
|---|---|
| T1 int/bool, T2 indexing, T3 scans, T4 remainder, T5 sort | `[x]`, card 0 |
| T6 random/dropout | `[x]`, cards 0 and 1 |
| T7 FP16, T8 MathMode | `[x]`, card 0 (T7 packer behaviour measured; T8 saving measured) |
| T9 mesh traces | `[x]`, ttsim and cards 0/1 |
| T10 benchmarks and dispositions | `[x]` benchmarks and baselines recorded; M2/M3/D5 `[-]`; the lane's stale-statement audit for `burn-native-cutover.md` and `burn-backend-parity.md` is NOT applied |
| E mutexes/atomics | `[x]` card 0 (ATSWAP single form `[-]`) |
| H SFPLOADMACRO | `[x]` card 0 |
| MD matrix diagnostics | `[x]` MOVDBGA2D (incl. the `AddrMod` cases: bit 14 widens a one-row move to four rows, now in the model); GATESRCRST `[-]` |
| PU packer/unpacker modes | `[x]` ReLU, edge masking, BF16 `UnpackToDst`, tileize; unpacker transpose `[-]` (M3); not routed into Session/Burn ops |
| FENCE (X7) | `[x]` card 0 and both-card Ethernet gates |
| N NoC | atomics `[x]`; completion polling `[x]`; IRQ `[-]`; **multicast `[x]`** (cards 0 and 1, run one probe at a time, no reboot) |
| F MMIO | `PACR_SETREG`, `UNPACR_NOP_SETREG` `[-]`; `LOADREG`/`STOREREG`/`STOREIND_MMIO` `[x]` on the allowlisted PIC words (card 0, one probe at a time, no reboot) |
| TS tag search | `[-]` (armed trigger load hangs the baby core, block stays armed) |
| G stream overlay | `[-]` (no pinned Blackhole register map) |
| P9 planner, `.ttinsn` | `[-]` (checker shows 0 missing waits; fusion needs a run-time code generator) |
| **X6 mover NIU fast path** | `[x]` card 0; persistence holds, byte-identical, 3-4% faster, opt-in |

### Still open
Carried in [hardware-coverage.md](../plans/hardware-coverage.md) "Where things stand":

1. Mesh training traces and `copy_into` on a mesh.
2. Both-card runs of the lanes validated on card 0 only (including T3's cummin/cummax).
3. X7 small-transfer latency (`[~]`).
4. T7's `measured::SILICON` constants (`step143`'s `native_conversions_match_the_measured_model`
   panics `PENDING MEASUREMENT` until they are filled from the recorded `T7-MEASURE` lines).
5. `step111b` was renamed `probe_addr_mod_sweep` (a diagnostic that asserts nothing); the
   `step137`-`step139` sort gates now live in `step136_burn_sort`.

Operational notes for whoever continues: see
[silicon-operating-notes.md](../learnings/silicon-operating-notes.md) ("Handling silicon runs").
