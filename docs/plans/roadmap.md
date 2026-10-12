# Roadmap: what is left for a full Burn-based Tenstorrent backend

**Start here.** This is the single list of open work. It holds only open items, each
linking to its evidence and to the detailed plan. Closed work lives in
[`completed-plans/`](../completed-plans/) and the archived phase docs below; permanent facts live in
[`learnings/`](../learnings/). Last reviewed against the code on 2026-10-12.

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
- [x] **B10/B11** (2026-10-11, [`burn-tt-conformance`](../../crates/burn-tt-conformance/README.md),
  `cargo xtask conformance --release`): Burn 0.21's own backend tests (1,838 tensor and autodiff
  tests; upstream's `#[ignore]`d ones excluded) run on ttsim one test per process. **1,806 pass; 32
  are in `expected-failures.toml`, each group with a category and a reason** (it began at 163: rounds
  one to five closed 131, see [conformance-to-feature.md](conformance-to-feature.md)). The passing
  tests are in `--smoke`, validated on card 0. A second profile, `cargo xtask conformance --exact` (matmuls
  accumulate in exact F32, `TT_EXACT_MATMUL=1`), has its own list, `expected-failures-exact.toml`: **1,818
  pass, 20 listed** (3 F64, 1 product the exact mode refuses by name, 16 that ttsim is too
  slow for and silicon passes). The run on `Fusion<TtBackend>` (B13a, `--fusion`, 2026-10-12) passes
  1,803 with 35 listed (the 32 above and three bool-scalar tests that die in burn-fusion 0.21.0, below), and with `--exact` 1,815 with
  23 listed; it found and
  fixed a fuser bug (below). The F16 suites (D1, `tensor_f16` and `autodiff_f16`, 1,187 tests, 2026-10-12) run in every profile: 1,165 pass with 22 listed (1,167 and 20 with `--exact`, and the same under fusion); see round seven in [conformance-to-feature.md](conformance-to-feature.md).
- [x] **`interpolate` / `interpolate_backward`**, nearest, bilinear, bicubic and Lanczos3, forward and
  backward (2026-10-11, [`ops/interpolate.rs`](../../crates/burn-tt/src/ops/interpolate.rs), gate
  `step155_interpolate`, card 0, smoke tier 814 of 814, log `1791733008`). No new kernel: shape-only tap tables
  on the host, a device gather, SFPU multiplies and the in-order row sum. All 22 of Burn's tests pass;
  nearest and bilinear equal Flex bit for bit, bicubic and Lanczos3 are within a derived bound, Lanczos3
  forward is also checked against the `image` crate, and Lanczos3 backward (which Flex lacks) against the
  adjoint of the forward map. It also fixed a `tensor::metadata` batching overrun that panicked the server
  thread on a table of many distinct values. Procedure: [conformance-to-feature.md](conformance-to-feature.md),
  "round two". Left: F16 is refused, and a resize staging more than 2^24 host addresses is refused by name
  (Track 2, "Separable `interpolate`").
- [x] **`scatter_nd` with Mul, Min and Max** (2026-10-11, [`ops/index.rs`](../../crates/burn-tt/src/ops/index.rs)
  `scatter_rows`, gate `step156_scatter_nd_reduce`, card 0, log `1791741849`, smoke tier 858 of 858, log `1791742419`). The refusal was a deferral:
  the `Assign` loop already applies tuples in index order, so the same loop gives Flex's order for the
  rest. All 26 of Burn's tests pass (20 are gradients, which needed nothing new); `Min`/`Max` equal Flex
  bit for bit over every special, F32 `Mul` equals the specification's multiply in index order.
- [x] **`conv3d` and `conv_transpose3d`**, forward and all three gradients (2026-10-11,
  [`ops/conv.rs`](../../crates/burn-tt/src/ops/conv.rs), gate `step157_conv3d`, card 0, log `1791751188`,
  smoke tier 875 of 875, log `1791751984`). No new kernel: the 2-D patch tables became generic over the number of spatial
  dimensions (`Geometry<D>`), and the 2-D results are unchanged (`step85`, `step154`, the MNIST golden). Output
  and the x, weight and bias gradients equal Flex bit for bit on small integers, including `padding_out` past
  `max(stride, dilation)`, groups, a batch of two and more positions than one 32-row chunk; a mirrored-kernel
  control fails as it should. 18 of Burn's 20 tests pass. The other two (`test_conv3d_groups`, `_complex`)
  are `precision`: **not TF32 truncation** (their operands are exact) but the Matrix Unit's group adder
  (decomposed below, gate `step158_matmul_accumulation`), identical on silicon. Left: BF16 forward is
  silicon-only (PACR `0x105`).
- [x] **Exact matmul, opt-in** (2026-10-11, `MatmulMode::{Fast, Exact}`,
  [`ops/exact_matmul.rs`](../../crates/burn-tt/src/ops/exact_matmul.rs), gate `step159_exact_matmul`, ttsim and card 0,
  log `1791761930`; smoke tier 890 of 890, log `1791763156`). The fast Matrix Unit matmul stays the default, with the
  tolerance now stated (the group adder's derived bound, `step158`). Exact means F32 SFPU products summed in
  increasing `k` order from `+0.0`, no FPU and no TF32. Selected by `TtConfig::matmul(..)` or
  `TT_EXACT_MATMUL=1` for an attachment, and by `with_matmul_mode(..)` / `exact_matmul(..)` for the ops a closure builds
  on this thread (not reaching `Fusion<TtBackend>`, whose ops run on another thread; a trace freezes the mode it was
  captured in). No new kernel: a gather into `[k, outputs]`, an SFPU multiply and `float_sum_dim` over
  dimension 0 (the only in-order layout), chunked over outputs; every route is covered by one hook (conv, linear,
  attention, gradients, batch broadcast, views, BF16 widened). Equal to an independent sequential-SFPU oracle bit
  for bit on random operands, all special values, every route, a trace and autodiff `backward()`; two mutations
  (reversed `k`, a wrong join) are caught. Refused by name: a mesh, a `k` beyond one chunk, more than 1024 chunks.
  **Price (card 0, `1791761944`): 1,000x to 22,000x slower than the fast matmul, about 0.5 us a product**
  (`[256,256]^2`: 8.2 s against 376 us; [performance-log.md](../learnings/performance-log.md#exact-matmul-cost-2026-10-11-card-0)).
  **What it settled:** every one of the 21 `precision` entries passes in exact mode (on silicon for the large LU
  tests), so they were the Matrix Unit matmul and not separate defects. Next: a faster formulation (below).
- [x] **`deform_conv2d` and `deform_conv2d_backward`** (2026-10-12,
  [`ops/deform_conv.rs`](../../crates/burn-tt/src/ops/deform_conv.rs), gate `step160_deform_conv2d`, ttsim and card 0,
  silicon log `1791767178`; smoke tier 908 of 908, log `1791767187`). The refusal was a deferral. No new kernel: the host builds only what the
  shapes fix; the card computes the sample positions, bilinear fractions and corner validity, four resident gathers read
  the corners (an invalid corner reads an appended zero row), then Flex's term order and the mask; the product, bias and
  weight gradient are `conv`'s own (`conv_products`, `weight_gradient`, now shared), and the x gradient is a one-hot
  matrix times the column gradient. All 17 of Burn's tests pass in both profiles. Those tests are loose (`relative(0.5)`
  on every gradient), so the gate is the evidence: forward and all five gradients **equal Flex bit for bit** on exact data
  (7 cases covering groups, offset groups, stride, padding, dilation, a batch of two, more positions than a 32-row chunk,
  with and without mask and bias; plus samples exactly on the `-1` and `H` borders), and a NaN or infinity in x
  reaches only the samples that read it (exact matmul mode equals Flex NaN for NaN; the fast mode's Matrix Unit
  turns NaN into infinity, a documented property). Five mutations of the implementation (corner swap, dropped
  validity, wrong one-hot weight, swapped offset-gradient axes, swapped sampling axes) each fail the gate. **Limit:**
  a gather stages its whole table per element, so the host holds `4 x samples x channels x (rows + 1)` addresses and
  more than 2^24 is refused by name (Track 2, "Gather without staging").
- [x] **F16 CTC gradients** (2026-10-12, ttsim, fusion and card 0 (`step162`, run `1791818535`)): `has_ctc_loss_backward`, native `ctc_loss` and
  `ctc_loss_backward` ([`ops/ctc.rs`](../../crates/burn-tt/src/ops/ctc.rs)), widened to F32 and rounded once; the two
  `autodiff_f16` CTC gradient entries are gone from every list. The alpha recursion is Burn's (private there) and beta is alpha of
  the time- and label-reversed problem; gate `step162` `ctc_gradient_in_f16_meets_the_f32_reference`. F32 CTC takes the same path
  (its gradient equals Burn's composition to 1e-5).
- [ ] **Faster exact matmul.** The exact path is linear in products at about 0.5 us each; whether its two
  `k x outputs` host address tables or the device gathers dominate is not profiled. The untried
  formulations need far fewer addresses and give the same sum in the same order: a loop over `k` of
  `acc = acc + a[:, p] * b[p, :]` (about `m + n` addresses a step), or one output row at a time. Profile first,
  then build the cheaper one and re-measure (`cargo xtask bench --filter step159_exact_matmul::benchmark`); a
  `k`-loop also removes the chunk limit that refuses the 4.3e10-product transaction test.
- [ ] **Conformance findings** (rounds one to five are closed: softmax of any rank, I32/Bool
  `reshape`, broadcast `mask_where`/`powf`, empty `cat`, the planner bug, `bool_argwhere`, `interpolate`,
  `scatter_nd`, `conv3d`/`conv_transpose3d`, `deform_conv2d`; see [conformance-to-feature.md](conformance-to-feature.md)).
  32 tests remain, each a group in `expected-failures.toml`; **none is a missing op**: what is left is the Matrix
  Unit's rounding (21, passing in the exact profile), simulator speed (8) and F64 (3). The next conformance work is
  therefore the runs that were open, both now done (B13a, `Fusion<TtBackend>`, round six; D1, the F16 suites, round seven), not more list entries.
  In the order the model ladder needs them:
  1. Twenty-one `precision` failures in the default profile (gelu gradient 2, conv2d 1, conv3d 2, LU and
     determinant 16). **All pass in the exact profile**, so they are the Matrix Unit matmul (the fast mode's
     documented rounding and TF32 truncation), not separate defects. Decomposed: the two `conv3d` ones are the
     group adder (2026-10-11, [silicon-operating-notes.md](../learnings/silicon-operating-notes.md), "Matmul
     accumulation: the Matrix Unit's group adder"; `step158_matmul_accumulation`, run `1791754387`: `MVMUL`
     adds its 16 products in two groups of eight, aligned once to a quantum of 2^(largest operand-exponent sum
     - 10) with half-up rounding; the model equals the device bit for bit); the gelu gradient is TF32
     truncation (reproduced by hand); the LU, determinant and conv2d ones are the Matrix Unit matmul with
     truncation and alignment **not separated** (apply the step158 model with TF32 truncation to the matmuls they
     compose if it matters). Decision (2026-10-11): keep the fast matmul as the default and offer the exact one
     opt-in (the entry above); the 21 stay listed in the default profile, `precision`, each reason saying the
     exact profile passes it. Never loosen a vendored tolerance.
  2. **Re-derive the `step68` and `step85` bounds.** They charge `2^-9 sum|ab|` for TF32 truncation and a few F32
     additions, and nothing for the group alignment, which can be larger with exact operands (1024 beside seven
     0.5s in one group: 1031 for 1027.5, `2^-8.2`; step158). They hold on their own similar-sized data, but are
     not a guarantee. Add the alignment term from the measured rule (the gate's `dot` returns it) wherever a gate
     derives a matmul bound.
  3. Eight `timeout` tests the simulator is too slow for; each passes on silicon, so nothing to fix.
  4. Three F64 tests (not a Tensix dtype; two end as timeouts because upstream
     `burn-tensor-testgen` 0.21.0's `#[might_panic]` deadlocks when the panic message differs).
- [x] **Fuser scalar guard** (2026-10-12, [`fusion/eltwise.rs`](../../crates/burn-tt/src/fusion/eltwise.rs) and `fusion/mod.rs`,
  gate `step161_fusion_scalar_guard`, ttsim and card 0, log `1791772236`). Found by the suite on `Fusion<TtBackend>`:
  burn-fusion gives a fuser every scalar as an id, and the fuser took every id for zero, so `thresholded_relu(x, a)` and
  `mask_fill(x, x <= t, v)` ran as `relu` (3 of Burn's tests). The optimization now records the ids and checks
  `Context::scalars` when it executes, falling back to the original operations. Real `relu` and `add` + `relu` stay fused
  (`step12_mnist` unchanged). See [conformance-to-feature.md](conformance-to-feature.md), "round six".
- [ ] **Bool scalars under `Fusion<TtBackend>`.** burn-fusion 0.21.0's `ScalarIr::to_relative` is `todo!()` for a Bool
  scalar, so `bool_equal_elem`, `bool_not_equal_elem` and a bool `mask_fill` kill the fusion server thread on `burn_tt::Tt`
  (three tests, `bug` in `expected-failures-fusion*.toml`; the same ops pass on `TtBackend`). It is upstream's code: either an
  upstream change (keep the bool as a literal) or, meanwhile, a refusal by name in `burn-tt` that points at it, since a
  user of `Tt` meets a dead thread today.
- [x] **D1: F16 arithmetic** (2026-10-12, generator `is_half`, `float_compute_input`, `expanded`, `step162_f16_arithmetic`, ttsim;
  `step162` and `step144` on card 0, log `1791814459`; the conformance failures on card 0, run `1791811392`). F16 widens to F32 on the card, computes and rounds once, as BF16
  does; `dtype_usage(F16)` is `general`. Found two latent BF16 defects on the way (`expand` of a half-precision tensor, an
  optional bias left unwidened) and fixed both. Listed: five `precision` (three are the Matrix Unit and pass in the exact
  profile; two are CTC gradients, per-operation binary16 rounding in Burn's default composition, matched by burn-flex's F16),
  15 `timeout` (all pass on silicon) and two F64. Left: the exact-profile timeouts were not run on silicon (the exact matmul is
  about 1,000x slower), and the F16 conformance modules that pass are not in `--smoke` (only the gate is).
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
- [ ] **Separable `interpolate`.** Today one pass of up to 36 taps stages `taps x N x C x outputs` host
  addresses and refuses more than 2^24 (a ViT position-embedding resize, `[1, 768, 14, 14]` to
  `[32, 32]` bicubic, is 12.6 million and fits; a Lanczos3 one does not). Two 1-D passes (rows, then
  columns) cut taps per output from 16/36 to 4/6 and the staged addresses with them, at the cost of
  Flex's single-sum term order for bicubic and Lanczos3 (both already bound-checked, not bit-exact).
  Chunk the gather over planes. Measure on a real model before building it.
- [ ] **Gather without staging.** `gather_native` stages every row of its table for every element it returns
  (`select`, `gather`, `embedding` and `deform_conv2d` all go through it), so a lookup costs
  `elements x table rows` host addresses. `deform_conv2d` refuses more than 2^24 of them by name, which is far
  below a real detector head (a 64x64 plane with thousands of samples); an embedding of a real vocabulary has the
  same shape of problem. A device-side indexed read that takes the table once, instead of one staged row per
  element, is the fix; profile `embedding` and `deform_conv2d` first.
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
