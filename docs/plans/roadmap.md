# Roadmap: what is left for a full Burn-based Tenstorrent backend

**Start here.** This is the single list of open work. It holds only open items, each
linking to its evidence and to the detailed plan. Closed work lives in
[`completed-plans/`](../completed-plans/) and the archived phase docs below; permanent facts live in
[`learnings/`](../learnings/). Last reviewed against the code on 2026-10-10.

## Where the project stands

- **Hardware instructions: done.** Phase 10 is closed
  ([close-out record](../completed-plans/hardware-coverage-closeout.md)). The Burn operation
  traits have no unsupported float, int or bool methods
  ([burn-op-coverage.md](burn-op-coverage.md)).
- **Models: toy scale only.** MNIST MLP (91.96%), a 938-parameter CNN and a one-layer
  TinyTransformer, all trained from scratch in `tt-mnist`. Nothing loads pretrained weights.
- **Performance: behind the CPU unless traced.** Untraced training is 2–3× slower than
  burn-flex ([scoreboard](../learnings/performance-scoreboard.md)); the cost is host-side
  submission, not the device. Traced MNIST inference already beats the CPU.
- **Multi-chip: proof of concept.** Matmul, conv and attention split across two cards over
  Ethernet; every other op runs on chip 0; no all-reduce.

The goals, in order: **real models run**, then **performance parity**, then **multi-chip**.
A publishable crate is a long-term goal (last section), not scheduled.

## Track 1: first real models

Goal: off-the-shelf Burn code runs on a card without knowing about burn-tt internals.
Detail: [burn-backend-parity.md](burn-backend-parity.md) (B-numbers).

- [x] **B1** typed `TtError` (2026-10-11, `burn-tt/src/error.rs`): `try_into_data`/`Backend::sync` return the sticky
  error, strict mode keeps the panic, a panicking engine reports its message.
- [x] **B2** lazy attach, engine registry (`set_default_engine`), `init`/`detach`/`TtConfig`, card lock
  (`Kmd::lock_card` in every `Session::open_card`), `TtDevice::enumerate`/`device_count` (2026-10-11).
  The default tile is `TileChoice::First`; see "Use the grid" for choosing a tile count per workload.
- [ ] **B10/B11** vendored Burn conformance suite on ttsim with expected-failures, and a
  subset in `cargo xtask silicon --smoke`. Never run to date.
- [ ] **B15** `burn-store` safetensors load (F32 and BF16), `Learner` checkpoint round trip,
  one `burn-onnx` model. Nothing in the code uses `burn-store` yet.
- [ ] **Model ladder**, each rung checked against burn-flex; every failure becomes an op or
  shape item here:
  1. pretrained small CNN/ResNet inference
  2. small ViT
  3. small decoder-only LLM inference (RMSNorm, RoPE, SiLU, KV-cache `slice_assign`)
  4. fine-tuning
- [ ] **Partial ops** (from the old coverage tracker): `float_matmul` edge shapes still
  staged through the host, `mean_dim`, gather/scatter/select, `slice_assign`/`cat`, conv
  acceptance, attention variants, `int_div`/`remainder` on card 1, ND indexing, GMPOOL
  max-with-indices. These are closed by the ladder, not separately.

## Track 2: performance (host-bound first)

Goal: training at or below burn-flex per step on MNIST and the transformer, then on a
Track 1 model. Evidence and bottlenecks:
[performance-scoreboard.md](../learnings/performance-scoreboard.md) ("Where the overhead is").

- [ ] **Trace segmentation.** Training traces overflow the 240 KB program cache
  ("a list's programs do not fit"). Split into segments so training runs traced
  ([traced-execution.md](traced-execution.md) §8).
- [ ] **B13b** general SFPU micro-op chain fuser (today one fuser, `ADD_RELU`).
  [kernel-fusion.md](kernel-fusion.md) lines 195–197.
- [ ] **B13c** matmul + bias + activation epilogue (kernel-fusion.md lines 200–204).
- [ ] **Reduce/norm fuser**, L1 circular-buffer streaming for multi-tile row reductions
  (kernel-fusion.md lines 207–210).
- [ ] **Per-workload tile choice.** Auto-attach opens one tile (`TileChoice::First`), and a session's tile
  count is fixed at attach. Open `All` tiles and choose a subset per op: an autotune-style cache keyed by
  `(op, shape, dtype)` that measures candidate counts on first use (`[512,512]^2`: 8 tiles 1.32 ms against
  120 tiles 2.0 ms, checklist 9.7c) and keeps the best. Needs a per-op tile subset in `tt-kernels`; related to
  B16 (placement).
- [ ] **Use the grid.** Models run on 1–4 of 120 tiles: default tile count from the grid,
  Dst subblocks (G4), multicast matmul operands (G7; the primitive is done and
  silicon-gated, nothing in `tt-kernels` uses it).
  [tt-metal-concepts-review.md](../learnings/tt-metal-concepts-review.md).
- [ ] **B7** batched readback (`download_many`, batched `tr_execute`). (The `Backend::sync` barrier of B8 is
  done with B1.)
- [ ] **X7** small-transfer latency; 9.14 per-request cost (note X6 found the fast
  path gains only 3–4%: the cost is the request round trip; the remaining lever is more
  requests in flight, plus the GATHER/SCATTER-direct and multicast-list ideas).
  **9.13** role `START` traced before the role's descriptor and config work, so the push
  timings leave part of the launch out; measure instruction-heavy kernels (SFPU `exp`, roles
  87% busy) before rewriting the push loop. Evidence:
  [checklist](../completed-plans/phases-0-10-checklist.md) Phase 9, "Performance backlog".
- [ ] **Later:** L1-sharded tensor residency (G12) and FlashAttention
  ([future-fusion-features.md](../proposals/future-fusion-features.md)); on-card dispatch
  ([x280-on-card-dispatch.md](../proposals/x280-on-card-dispatch.md)).
- **Fused `.ttinsn` push for SFPU kernels: parked, evidence-triggered.** Measured 2.7x on a push-bound
  kernel; not worth scheduling until a real model shows SFPU-bound device time. Everything is in
  [Parked: fused `.ttinsn` push](#parked-fused-ttinsn-push-for-sfpu-kernels).

## Track 3: multi-chip (after the single-chip path is fast)

Evidence: [phases-0-10-checklist.md](../completed-plans/phases-0-10-checklist.md) Phase 8/9.

- [ ] **9.11** device-resident mesh with chips running concurrently, and `copy_into` on the
  mesh (today `MeshEngine` keeps buffers in chip 0's GDDR).
- [ ] Gradient all-reduce over Ethernet, data-parallel training, **B14**
  `DistributedBackend`.
- [ ] Mesh training traces (mesh inference traces work, step147).
- [ ] Ethernet pipelining: both 800G links together reach 27 GB/s (27% of ceiling).
- [ ] Investigate the two-card accuracy gap (0.9195 vs 0.9196 after step 32).
- [ ] Packed BF16 over the mesh (operands are widened to F32 today).
- [ ] Validate on a physical p300 (all silicon work used two cabled p150a cards).

## Parked: fused `.ttinsn` push for SFPU kernels

**Status:** experiment complete 2026-10-10 (card 0); follow-on **parked by owner decision**, to be
revisited only when the trigger below fires. Plan and full record:
[ttinsn-push-experiment.md](ttinsn-push-experiment.md). Numbers and run IDs:
[performance-log.md](../learnings/performance-log.md#ttinsn-push-fusion-precise-gelu-2026-10-10-card-0).

### What it is

An SFPU math program is pushed to the coprocessor by RISC-V T1, which today reads each word from L1 and
stores it (`lw` + `sw`, sixteen words at a time, 3.31 cycles a word measured). The long programs (precise
`gelu`: 516 words stored, 11,645 pushed) make the math role **push-bound**: the backend retires within ~50
cycles of the last push. `.ttinsn` encodes the same push as a single store of a compile-time immediate
(the Tensix word rotated left two bits), 1.19 cycles a word. tt-metal does exactly this by default: its LLK
`TTI_*` macros emit `.ttinsn` with constants, built per kernel configuration by a cached JIT compiler.

### What was measured (card 0, release, medians, every result bit-exact against the runner)

| Way of pushing | `gelu` push phase | 16-tile `gelu` kernel (whole) |
|---|--:|--:|
| Runner (today) | 38,578 cycles | 608,841 |
| Baked fused words, built into the image | 13,868 (2.78x) | 219,741 (2.77x) |
| Translated at run time by the runner | 20,907-25,308 (1.5-1.8x) | 228,421 (2.67x) |

The 1-tile kernel went 39,328 -> 23,219 (translated). A 194-word program with no loops was *slower*
translated (4,791 against 818 cycles). The matmul is backend-bound and unaffected; host-bound training sees
nothing.

### What was learned that holds whatever is decided

- **The gain is not hardware gathering.** Blackhole boots with instruction gathering (adjacent `.ttinsn`
  fused, up to four a push) on; tt-metal disables it on every RISC (`configure_gathering`, tt-metal#16439: a
  `REPLAY` record window can be corrupted). Measured: on or off gives identical cycles. **Every shipped
  Tensix image now turns it off** (2026-10-11, `tt_firmware::disable_gathering`, read back and checked at
  start; silicon-verified on both cards), so a `.ttinsn` path inherits it. The other chicken bits of `cfg0`
  (L1 data cache bit 3, relaxed ordering bit 0) are untested here.
- **T1 can write code at run time and invalidate its own instruction cache** (`RISCV_IC_INVALIDATE`, mask
  bit 2), with the documented sequence. A rewrite *without* it runs stale code (verified), so the control is
  real. ~330 cycles fixed. Invalidating *another* core's cache by mask is documented, not tested.
- **Translating costs 13 cycles a word** for the copy plus ~9 for a per-word opcode check, plus ~330 fixed;
  batching loads did not help (it is not load latency). Break-even for one launch without a cache:
  `expanded_words * 2.1 > stored_words * (13..22) + 330`. `gelu` (23x longer expanded than stored) clears it
  easily; a loop-free program does not.
- **The build-time instruction gate had a hole**: the assembler marks `.word` as data, so a stray
  compressed-space word passed in any image. Fixed for every image (data in code is refused; none existed).
- **Silicon traps for any run-to-run comparison:** `Dst` keeps its contents between runs (poison the rows
  first, or a push that wrote nothing matches the last run); `SFPSTORE` writes one column parity per
  address (`a` even, `a + 2` odd), so define both; a checksum must be able to fail (the first cancelled to
  zero on every program).
- **ttsim executes `.ttinsn` and run-time-written code** and matches silicon bit for bit (divergence row
  AQ), so the translator's emitted RISC-V is checked on ttsim; it has no instruction cache and no timing.
  Tests that fork a ttsim hung about one run in five when run concurrently (unexplained; the experiment's
  tests take a mutex).
- The "39 of 73 kinds push-bound" figure was a static model (`per_kind_push_table`); 23 scalar-free unary
  kinds sit at the model's 2.8 ceiling (`gelu` 368 stored words, `erf`, `asinh`, `sin`, `tan`, ...).

### Why it is parked

1. **The bottleneck today is the host, not the SFPU.** Untraced training is 2-3x slower than the CPU and the
   device is 12.6% busy; trace segmentation, the fusers and using the grid come first (Track 2).
2. **Amdahl.** Overall speedup is `1 / (1 - f + f / 2.7)` for a fraction `f` of device time in push-bound SFPU
   kernels: 1.07x at 10%, 1.23x at 30%, 1.46x at 50%. There is no real model yet to say which (Track 1).
3. **A cheaper lever exists.** Approx math mode already cuts per-tile time (precise -> approx, host-timed,
   card 0): `gelu` 28.5 -> 7.3 us (3.9x), `recip` 7.8 -> 0.95 (8.2x), `sigmoid` 1.65x, `tanh` 1.62x, `log`
   1.37x, `exp` 1.24x. It stacks with `.ttinsn`, at a precision cost, with no new machinery.
4. **Adoption is not free:** a gate that ships, I-cache invalidation, a planned L1 code buffer
   and cache, a streamed-path hook, Session-level and card-1 validation.
5. tt-metal solves this with a compiler we do not have; our stand-ins (baked images, run-time translator) are
   work to maintain.

### Trigger: revisit when

A profile of a real Track 1 model (the ViT or LLM rungs) shows push-bound SFPU kernels above **~20% of
device time** on the card, with the model device-bound (host overhead already handled by traces).
How to measure: traced runs record `START`/`PUSHED`/`RETIRED` per role through the timestamper
(`silicon_perf::role_push_rate` shows the method); a role whose `START -> PUSHED` span is long and whose
`PUSHED -> RETIRED` tail is ~50 cycles is push-bound. `per_kind_push_table` (in
`step121_wait_planner.rs`) lists which kinds can be.

### If unparked: design and work list

Prefer, in this order (each step is cheaper than the next and may be enough):

1. **Approx mode first** wherever the precision contract allows (no `.ttinsn` needed).
2. **Host-side expansion (untried, recommended experiment):** the host builder emits the fused code directly
   (rotated words plus a fixed `addi`/`bne`/`ret` loop skeleton, the same emitter as the generator in
   `probe_ttinsn_push.rs`) into the program slot; the device only invalidates its cache and calls it. No
   device translation cost. The gate's rules (word validity, no data directives, only the skeleton's
   RISC-V instructions) move into a host validator covered by the same tests.
3. **Device translation** (`tt_firmware::ttinsn`, measured) on T1, or on idle T0/T2 (B is the worst
   choice: busy reader, small I-cache), with a translation cache keyed by the program's words.
4. **Baked images** only for a short manifest of the heaviest kinds (the generator makes adding one a
   one-line change); limited by the 16 KiB T1 slot and by scalars and tile counts being baked in.

Work items whichever way (gathering is already off in every shipped image): hook both push paths (`run_in` and the streamed
`push_body`); plan the code buffer through `l1::Requirements` (the experiment used the last 16 KiB of
L1 by fiat); a size threshold from the break-even formula; a Session-level measurement (mover and host
around the kernel; the earlier Session-level per-tile slope for `gelu`, 28.46 us, equals the kernel's push
time, so large ops should follow); card 1; ttsim gate with the existing controls; docs and a scoreboard row.

### Open questions (not run)

- T0 and T2 spans next to T1's for a heavy kernel, a light kernel and a matmul tile in both formats
  (offered, not run): whether T0/T2 or their units ever limit, and so whether translating there is free.
- Translating on one core and invalidating another's cache.
- The L1 data cache (`cfg0` bit 3) on the runner's ordinary push loop: it might speed `lw` without any
  `.ttinsn`; untested, with coherence risks (see the Tier-1 L0-cache note in the checklist).
- Whether any other kind than `gelu` behaves the same, and the matmul unpack role since batching.

### Where it lives and how to rerun

| What | Where |
|---|---|
| Experiment image (T1 runner + hooks), behind cargo feature `ttinsn-experiment`, off by default | `crates/tt-firmware/src/bin/ttinsn_t1.rs`, hooks in `corpus.rs` (`experiment`), `crates/tt-firmware/src/ttinsn.rs` (translator, invalidation) |
| Gate allowance (validates `.ttinsn` runs; refuses data in code for every image) | `crates/tt-firmware-images/build.rs` (`check_ttinsn_blocks`) |
| Generated fused programs and opcode bitmap | `crates/tt-firmware/src/ttinsn_programs.in` (`TT_BLESS=1 cargo test -p tt-tests --test probe_ttinsn_push ttinsn_programs_are_current`) |
| Tests, rungs and measurements | `crates/tt-tests/tests/probe_ttinsn_push.rs`, `per_kind_push_table` in `step121_wait_planner.rs` |
| ttsim gates | `cargo test -p tt-tests --features ttinsn-experiment --test probe_ttinsn_push` |
| Silicon (one test per process, card 0) | `cargo xtask silicon --release --device 0 --feature ttinsn-experiment --include-ignored --filter probe_ttinsn_push::experiment::<test>` with `<test>` one of `rung0_icache_invalidation`, `rung4_gelu_matches`, `rung5_translate_nops`, `rung7_translated_gelu_matches`, `measure_push_phase`, `measure_kernel_level`, `measure_translator`, `measure_gathering` |

## Device-side backlog (not on the critical path)

- [ ] I-cache invalidation and hot reload of running cores; `load_and_start_staged`
  (`SuppressZeroing`/`WaitOutZeroing`, plus the silicon pair); multi-core loader exclusion;
  **measure I-cache capacities** (open question 1; undocumented for Blackhole, sets hot-loop
  sizing and whether T1 is viable for large code). Also open: what occupies the upper 4 KiB
  of a T-core's 8 KiB local-RAM slow-path window (silicon only). Evidence:
  [checklist](../completed-plans/phases-0-10-checklist.md) Phase 2; ttsim models none of reset zeroing or I-cache invalidation
  (open question 6).
- [ ] **Ask whether ttsim models the documented RISC-V hardware bugs** or the intended
  behaviour (open question 7); either answer changes what a simulator gate proves. Evidence:
  [checklist](../completed-plans/phases-0-10-checklist.md) Phase 3 and "Open questions".
- [ ] **Documented baby-RISC-V hazards still neither encoded nor probed:** #1 interrupt
  handlers cannot write CSRs, #4 `pmacfg0/1` range selection broken, #5 `intp_restore_pc` is
  only a copy, #6 operand-forwarding starvation hang, #8 store-then-load with
  non-overlapping byte ranges unordered, #10 Configuration Unit cross-thread starvation.
  Evidence: [checklist](../completed-plans/phases-0-10-checklist.md) "Tier 1" hardware bugs.
- [ ] `Dst` exclusivity across the three Tensix threads as a type-level guard (the other
  hazard encodings in the original list are done). Evidence: [checklist](../completed-plans/phases-0-10-checklist.md) Phase 5.
- [ ] Silicon verification of Blackhole behaviour that has no Blackhole documentation page
  (Wormhole page not architecture-conditionalized): all packer sub-pages, unpacker
  `FormatConversion`/`FlushCache`/`IncrementContextCounter`, `RWCs`, `SrcASrcB` fidelity
  phases, `FloatBitPatterns`, `WaitGate`, Miscellaneous Unit. Not reconciled with the Phase 10
  matrix: check what step93-108 already cover before starting. Evidence:
  [checklist](../completed-plans/phases-0-10-checklist.md) "Silicon-verification backlog".
- [ ] Observe deliberate mutations (negative controls) for the step79 scan, step80 integer
  reduction and step81 hardware-rounding gates; the 2026-10-05 wrap-up says their
  definition of done was not yet closed. Evidence:
  [tensix-next-features.md](../completed-plans/tensix-next-features.md) wrap-up section.
- [ ] step143 FP16 `measured::SILICON` constants (the test panics "PENDING MEASUREMENT").
- [ ] Re-sync the pinned ISA spec (`PINS.toml`, still f848eb6).
- [ ] BFP `a` variants, INT8/UINT8, portable `QTensorOps` (deferred in
  [burn-native-cutover.md](burn-native-cutover.md)).

## Long-term goals (not scheduled)

**Publishable crate.** The code is public on GitHub and usable as is; the polish below is
deferred to later cleanup passes. B1 and B2 stay in Track 1 because models need them.

- [ ] **B0** `tracing` targets and per-reason attribution.
- [ ] **B4** honest `dtype_usage`, dtype fallback warnings.
- [ ] **B9** `name`, `memory_cleanup`, persistent allocations, OOM retry.
- [ ] Silicon gates on a self-hosted nightly runner (CI runs the simulator only).
- [ ] Public API and docs review; confirm firmware-image embedding works for a crates.io
  build.
- [ ] Upstream PR to `burn-dispatch` (a closed enum) and `burn-router`/`burn-remote` gate
  for `BackendIr` (B12 is implemented, gate never run).

## How the other docs relate

| Doc | Role now |
|---|---|
| [burn-backend-parity.md](burn-backend-parity.md) | B0–B16 detail |
| [kernel-fusion.md](kernel-fusion.md), [traced-execution.md](traced-execution.md), [mixed-bfp-storage.md](mixed-bfp-storage.md) | Feature specs |
| [burn-op-coverage.md](burn-op-coverage.md) | Generated operation status |
| [ttinsn-push-experiment.md](ttinsn-push-experiment.md) | Done, result: go. Experiment record for `.ttinsn` push fusion |
| [new-op-definition-of-done.md](new-op-definition-of-done.md) | Standing: what a new op must have before it counts as done |
| [burn-op-notes.md](burn-op-notes.md) | Standing: per-op notes for the Burn backend |
| [phases-0-10-roadmap.md](../completed-plans/phases-0-10-roadmap.md) | Archived: architecture and phase rationale (historical design) |
| [phases-0-10-checklist.md](../completed-plans/phases-0-10-checklist.md) | Archived: phase 0–10 tick-list and evidence |
| [tensix-next-features.md](../completed-plans/tensix-next-features.md) | Archived: dated 2026-10-05 sprint record |
| [hardware-coverage.md](../completed-plans/hardware-coverage.md) | Archived: Phase 10 instruction coverage matrix |
| [performance-scoreboard.md](../learnings/performance-scoreboard.md) | Learnings: current performance baselines and "Where the overhead is" |
| [performance-log.md](../learnings/performance-log.md) | Learnings: dated benchmark run log and change history |
