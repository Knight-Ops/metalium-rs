# Roadmap: what is left for a full Burn-based Tenstorrent backend

**Start here.** This is the single list of open work. It holds only open items, each
linking to its evidence and to the detailed plan. Closed work lives in
[`completed-plans/`](../completed-plans/) and the archived phase docs below; permanent facts live in
[`learnings/`](../learnings/). Last reviewed against the code on 2026-10-16.

## Where the project stands

- **Hardware instructions: done.** Phase 10 is closed
  ([close-out record](../completed-plans/hardware-coverage-closeout.md)). The Burn operation
  traits have no unsupported float, int or bool methods
  ([burn-op-coverage.md](burn-op-coverage.md)).
- **Models: four pretrained networks run on a card.** ResNet-18, ViT-Tiny and SmolLM2-135M (a
  30-layer Llama with a KV cache) run inference through `burn-tt` and agree with burn-flex on silicon
  (`crates/tt-models`, 2026-10-12), in F32, with SmolLM2 also in every mix of F32, BF16, F16 and BFP8/4/2. The
  first past 1B parameters, SmolLM2-1.7B-Instruct (2026-10-15), decodes at 22.1 tokens/s traced against
  burn-flex's 3.3 (13.6 on 2026-10-15; the cache's row write, thin matmul reads, fused projections and block views since), and prefills 29-35x faster.
  They were 105x, 7.8x and 24x behind burn-flex per forward pass
  ([profile](../learnings/performance-log.md#model-ladder-real-models-on-the-card-2026-10-12-cards-0-and-1)); ResNet-18 is now
  5.0x on one tile and faster than burn-flex on every tile (131 ms against 151, Track 2 item 2).
  A token's F32 embedding lookup, 15.9 s then, is 260 us since the row select
  ([log](../learnings/performance-log.md#row-select-an-embedding-lookup-without-staging-2026-10-13-card-0));
  BF16 tables still take the staged path.
  The MNIST MLP, CNN and TinyTransformer still train from scratch in `tt-mnist`.
- **Performance: behind the CPU unless traced.** Untraced training is 2–3× slower than
  burn-flex ([scoreboard](../learnings/performance-scoreboard.md)); the cost is host-side
  submission, not the device. Traced MNIST inference already beats the CPU, and traced, the three
  models match or beat burn-flex on every tile: the SmolLM2 decode step 26.46 ms against 26.42,
  ViT-Tiny 11.6 ms against 112, ResNet-18 17.2 against 164 (2026-10-15, Track 2 item 6,
  [replay-fixed-cost.md](replay-fixed-cost.md)). A whole traced generation (ids in, logits out, any
  fixed-shape decoder) beats burn-flex at every prompt length measured: 0.60x to 0.90x of its time over a
  512-position cache, a decode step 26.1-27.3 ms against 26.8-31.6 (Track 2 item 6, step 4,
  [decode-length-cost.md](../completed-plans/decode-length-cost.md)).
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
  fixed a fuser bug (below). **`--accurate`** (2026-10-12, `MatmulMode::Accurate`, `expected-failures-accurate.toml`): **2,989 of 3,025 pass, 36
  listed** (5 F64 and 31 ttsim timeouts, every one passing on card 0 in that mode: F32 run `1791819292`, F16 run `1791820653`), so no
  `precision` entry is left; `--fusion --accurate` 2,986 with 39 listed. The F16 suites (D1, `tensor_f16` and `autodiff_f16`, 1,187 tests, 2026-10-12) run in every profile: 1,165 pass with 22 listed (1,167 and 20 with `--exact`, and the same under fusion); see round seven in [conformance-to-feature.md](conformance-to-feature.md).
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
- [x] **Accurate matmul, opt-in** (2026-10-12, ttsim and card 0: `step158` `lofi_*` run `1791818525`, `step163` and `step162` run `1791818535`,
  benchmark run `1791818553`). `MatmulMode::Accurate`
  ([`ops/accurate_matmul.rs`](../../crates/burn-tt/src/ops/accurate_matmul.rs), gate `step163_accurate_matmul`, `step158` `lofi_*`):
  the answer to "can a matmul stay on the Matrix Unit and lose its error". Per-row and per-column power-of-two scales and integer
  slices of 7 and 5 bits make every product and every sum of eight exact at `Fidelity::Lo` (the unit's quantum is then 1), twelve
  slice pairs are added small to large, and NaN and infinity follow IEEE by three 0/+-1 matmuls. New SFPU kinds `POW2_FLOOR` and
  `POW2_FLOOR_INV` (an exponent-field mask, bit for bit against the host on every special); a per-call fidelity override
  (`Engine::matmul_dram_at`); `TT_ACCURATE_MATMUL=1`, `TtConfig::matmul(Accurate)`, `accurate_matmul(..)`, `--accurate`. ttsim: error
  about 1e-7 of the largest entry against the fast matmul's 1e-3 (0.21 of the derived bound; the fast one leaves the bound 22,072
  times on the same data), integer data exact, 41 of 41 selected precision tests pass under `--accurate` (F32 and F16). Domain:
  `|x| <= 2^64`, subnormals zero; a row's small entries are kept to the row's scale. **Cost on card 0: 19x to 64x the fast matmul** (seven shapes, 18.7x at 1024 cubed; 50x to 750x faster than exact). One
  `[64,96]@[96,48]` submits **127 device jobs** (101 element-wise, 18 matmuls, 4 reductions, 4 gathers and copies) against 1, and 64% to
  86% of the time is the element-wise slicing and classification, not the LoFi matmuls (profile in
  [performance-log.md](../learnings/performance-log.md#accurate-matmul-cost-2026-10-12-card-0)). Open: fused slicing kinds and a
  cheaper NaN/infinity classification, and whether it should become the default (decide from the cost above and the model numbers).
  Closed: the 300 to 500 LU tests, the determinants and the transaction pass on card 0 under `--accurate` (the LU and determinant
  ones fail their tolerance in the fast mode there), the full `--accurate` lists, and the BF16 gate (`1791818535`).
- [ ] **Fused slicing for `MatmulMode::Accurate`** (opt-in stays until this is measured; decided 2026-10-12). The mode is correct and
  gated but costs **19x to 64x the fast matmul** on card 0 (run `1791818553`, seven shapes, 18.7x at 1024 cubed), because one
  `[64,96]@[96,48]` submits **127 device jobs** (101 element-wise, 18 matmuls, 4 reductions, 4 gathers and copies) and 64% to 86% of
  the time is that element-wise work, not the LoFi matmuls (profile: [performance-log.md](../learnings/performance-log.md#accurate-matmul-cost-2026-10-12-card-0)).
  The arithmetic does not change, so `step163_accurate_matmul` must keep passing unchanged; what shrinks is the job count. In order of
  expected saving:
  1. **A slice kind**: `x` and the row or column power of two in, slice `s` out, already weighted. Today a slice is `trunc`, subtract,
     multiply by the radix, then a weight multiply (about 3 to 4 jobs a slice, 9 slices); one SFPU program can compute the earlier
     slices in registers and store only slice `s`. This also folds in the scaling `x * u * 2^k` (2 more jobs an operand).
  2. **A cheaper NaN and infinity classification**: about 45 of the 127 jobs (15 an operand: `is_nan`, `is_inf`, `== 0`, three
     bool-to-float, `bool_not`, `mask_fill`, `sign`, `mul`, two `sub`, a `sum_dim`, a second `mask_fill`, `clamp`, then six 0/+-1
     matmuls and about a dozen combining ops). One kind per operand producing the encodings (`inf_sign`, `finite_sign`, the zero and
     nonzero indicators, the finite part) would cut it to a few jobs; the six small matmuls could share one launch.
  3. **One matmul for the pairs**: twelve Lo matmuls and eleven adds (each add rounds in F32 and is a job) could be one Lo matmul over
     the slices concatenated along `k` (each block padded to a multiple of 32 so no group of eight straddles two scales), the unit's
     own `Dst` accumulation doing the adds; needs a cheap concatenation (a row gather costs host address tables, which is what made the
     exact mode slow, so measure it).
  Each new kind needs what `POW2_FLOOR` has: a program in `tt-kernels/src/sfpu/ops.rs`, `operands`/`sig`/`accuracy` entries, and a
  bit-for-bit test against the host on every special (zeros, denormals, `f32::MAX`, infinities, NaNs). Measure with
  `cargo xtask bench --device 0 --filter step163_accurate_matmul::benchmark`, record the new rows beside the 2026-10-12 ones in the
  performance log and scoreboard, and set the target from the first fused measurement (the exact mode took 0.5 us a product; this one
  is already 50x to 750x faster than that). Only then revisit making `Accurate` the default: today the default profile still lists the
  21 F32 and 3 F16 matmul `precision` tests, which pass only under `--accurate`.
- [x] **F16 CTC gradients** (2026-10-12, ttsim, fusion and card 0 (`step162`, run `1791818535`)): `has_ctc_loss_backward`, native `ctc_loss` and
  `ctc_loss_backward` ([`ops/ctc.rs`](../../crates/burn-tt/src/ops/ctc.rs)), widened to F32 and rounded once; the two
  `autodiff_f16` CTC gradient entries are gone from every list. The alpha recursion is Burn's (private there) and beta is alpha of
  the time- and label-reversed problem; gate `step162` `ctc_gradient_in_f16_meets_the_f32_reference`. F32 CTC takes the same path
  (its gradient equals Burn's composition to 1e-5).
- [ ] **Faster exact matmul.** (`MatmulMode::Accurate`, above, answers the cost question for most users; this is for the exact CPU order.) The exact path is linear in products at about 0.5 us each; whether its two
  `k x outputs` host address tables or the device gathers dominate is not profiled. The untried
  formulations need far fewer addresses and give the same sum in the same order: a loop over `k` of
  `acc = acc + a[:, p] * b[p, :]` (about `m + n` addresses a step), or one output row at a time. Profile first,
  then build the cheaper one and re-measure (`cargo xtask bench --filter step159_exact_matmul::benchmark`); a
  `k`-loop also removes the chunk limit that refuses the 4.3e10-product transaction test.
- [ ] **Conformance findings** (rounds one to five are closed: softmax of any rank, I32/Bool
  `reshape`, broadcast `mask_where`/`powf`, empty `cat`, the planner bug, `bool_argwhere`, `interpolate`,
  `scatter_nd`, `conv3d`/`conv_transpose3d`, `deform_conv2d`; see [conformance-to-feature.md](conformance-to-feature.md)).
  32 tests remain, each a group in `expected-failures.toml`; **none is a missing op**: what is left is the Matrix
  Unit's rounding (21 F32, plus 3 in F16; they pass in the exact profile and under `--accurate`), simulator speed (8) and F64 (3). The next conformance work is
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
     exact profile passes it. Never loosen a vendored tolerance. **Update (2026-10-12):** `--accurate` passes them too
     (`MatmulMode::Accurate`, entry above), at 19x to 64x the fast matmul's cost instead of the exact mode's thousands; the default stays
     fast and the 21 (and the 3 F16) stay listed until "Fused slicing for `MatmulMode::Accurate`" is measured.
  2. **Re-derive the `step68` and `step85` bounds.** They charge `2^-9 sum|ab|` for TF32 truncation and a few F32
     additions, and nothing for the group alignment, which can be larger with exact operands (1024 beside seven
     0.5s in one group: 1031 for 1027.5, `2^-8.2`; step158). They hold on their own similar-sized data, but are
     not a guarantee. Add the alignment term from the measured rule (the gate's `dot` returns it) wherever a gate
     derives a matmul bound.
  3. Eight `timeout` tests the simulator is too slow for; each passes on silicon, so nothing to fix.
  4. Three F64 tests (not a Tensix dtype; two end as timeouts because upstream
     `burn-tensor-testgen` 0.21.0's `#[might_panic]` deadlocks when the panic message differs).
- [x] **D2: Burn's suites with a BF16 float element, and a silicon conformance lane** (2026-10-12; BF16 on card 0 in `--strict`: run
  `1791830816`, **1,187 tests, 1,098 pass, 89 listed, 0 unexpected**, 2,691 s). BF16 is the half type this hardware prefers, and
  upstream ships no BF16 suite (`tensor_f16.rs` has `TODO: bf16`), so `tests/tensor_bf16.rs` and `tests/autodiff_bf16.rs` are the F16
  entries with `bf16` (948 + 239 tests). ttsim refuses PACR mode `0x105`, the narrowing every BF16 result goes through
  ([ttsim-divergence.md](../learnings/ttsim-divergence.md), "BF16 late narrowing"), so they are **silicon only**: the ttsim lane skips them
  and names that if asked for one. What it added:
  - **`cargo xtask conformance --silicon`**: card 0, one test at a time, 600 s default timeout, the fsync'd `target/silicon/<stamp>.log`
    (the stamp is the run ID), compared with `expected-failures-silicon{-fusion}{-exact}{-accurate}.toml`; needs no ttsim. It is also the
    first full silicon pass lane for the F32 and F16 suites (not yet run, below).
  - **`cargo xtask conformance --reference-flex`**: the suite on burn-flex on the host, no list, exit 0, to tell a test whose tolerance the
    type cannot meet from a burn-tt defect. Dev only (`reference-flex` feature of the conformance crate, which refuses to compile with `silicon`
    or `fusion`); never part of a pass count.
  - **One defect: `grid_sample_2d`.** Burn's default bilinear composition ran in BF16, rounding the coordinates, weights and sum;
    `should_pad_reflection_grid_sample_2d` returned 1.3047 where the BF16 grid gives 1.3003 (1.296875 as BF16). It is now overridden
    (`ops.rs` `float_grid_sample_2d`), so the generated wrapper widens to F32, composes and rounds once. Gate `step164_grid_sample_bf16`
    (card 0, run `1791830746`): equal to an f64 oracle rounded once over zeros, border and reflection padding, and the default
    composition, run on the same operands, differs (negative control). All 15 `grid_sample` conformance tests pass on card 0 (run `1791830775`).
  - **The other 89 listed, in `expected-failures-silicon.toml`:** 2 F64; 22 that pass under `--accurate` (the Matrix Unit matmul, runs
    `1791829833`, `1791830481`); 63 that fail under `--accurate` too and that burn-flex fails in BF16 (run `1791827087`), so the type's
    own resolution (BF16 steps at 2^-8 against tolerances of 1e-3 to 5e-3); 2 CTC tests (`test_ctc_loss_matches_pytorch`,
    `test_ctc_loss_grad`) that burn-flex passes, where burn-tt's value is the correctly rounded one for the BF16 log-probabilities (reproduced
    in f64 by hand). **Open:** why burn-tt's CTC gradient keeps the unrounded value before the `log_softmax` backward where burn-flex's BF16
    chain rounds it (the test's value needs that rounding); settle it with a trace of the chain, not by an entry.
  - **Smoke:** 13 BF16 modules and `step164` are in `SMOKE`.
  - **First full silicon pass of F32 and F16** (default profile, `--strict`, card 0, run `1791833700`): **3,025 tests, 3,001 pass, 24 listed,
    0 unexpected, 6,460 s**, nothing downloaded or staged on the host. All 24 are already listed for ttsim (5 F64, 19 `precision`);
    the 23 ttsim `timeout` entries pass on silicon, as their reasons say. Five small LU and determinant tests listed as `precision` on
    ttsim pass on silicon (three runs): they draw `Tensor::random` inputs, which differ per target
    ([ttsim-divergence.md](../learnings/ttsim-divergence.md) row 99), so their outcome follows the data, not a numerical difference.
    `expected-failures-silicon.toml` holds the 24 and the 89 BF16 entries.
  Left: the `--fusion` silicon passes (`conformance --silicon --fusion`, about 45 to 110 minutes each, card 0), the `--exact` and `--accurate`
  silicon lists (empty files today, so those profiles are unverified on silicon beyond the cited targeted runs), and whether CI runs
  conformance with `--strict` (it does not; both silicon passes above did).
- [ ] **Conformance follow-ups (low priority; nothing here blocks a model or a gate).** Where conformance stands after D2: every listed
  test has a cause and none is a missing op (F32 and F16 on silicon: 3,001 of 3,025 pass; BF16: 1,098 of 1,187; both `--strict`). What is left:
  1. **Silicon lists for the other profiles.** `conformance --silicon --fusion`, `--exact` and `--accurate` have empty lists
     (`expected-failures-silicon-{fusion,exact,accurate}.toml` do not exist), about an hour a profile on card 0 (single-card rule). Run each
     with `--update`, triage as in [conformance-to-feature.md](conformance-to-feature.md) round nine, and cite the run IDs. `--exact` is
     slow on silicon (the exact matmul costs thousands of times the fast one), so start with `--fusion` and `--accurate`.
  2. **The CTC gradient in BF16.** burn-tt's `test_ctc_loss_grad` value (-0.4609375) is the correctly rounded one for the BF16
     log-probabilities, but burn-flex's BF16 chain gives the test's -0.45898, which needs the CTC gradient rounded to BF16 before the
     `log_softmax` backward. Why burn-tt's gradient keeps the unrounded value there is not determined (the entry in
     `expected-failures-silicon.toml` says so). Trace the autodiff chain (`ctc_loss_backward`, then `log_softmax` backward) with intermediate
     dtypes before deciding whether it is a defect or only a difference.
  3. **Hardening from the 2026-10-12 audit** (no cheating was found: `burn-tt` has no host arithmetic on tensor values, and the
     conformance path links no CPU backend). Decisions still open:
     - CI runs `cargo xtask conformance --release` without `--strict` (`.github/workflows/ci.yml`); both silicon passes used it and
       nothing was downloaded or staged. Add it to CI, or a strict lane.
     - `check-no-flex-in-backend` forbids only the crate name `burn-flex`; extend it to `burn-ndarray`, `ndarray`, `burn-cpu`, `burn-candle`
       and `nalgebra`, and decide whether the conformance crate's `reference-flex` feature stays (it is dev-only and refuses to compile
       with `silicon` or `fusion`).
     - `float_random` and `int_random` fall back to a host `StdRng` when the engine is unattached or has no DRAM (`ops.rs`, `random.rs`);
       the conformance engines never take that path. Refuse by name instead, or drop `rand` and `rand_distr`.
     - Strict mode and the residency report exempt `*_into_data` (Burn's LU pivots and `one_hot` round-trip through the host, moving
       indices only), `bool_argwhere`'s 4-byte count read (`host_ok`), and host-resident byte repacks; the vendored-tests digest is
       checked only by `cargo xtask conformance`, not by `cargo xtask silicon --smoke`; the summary does not count the upstream tests
       that are never run (`#[ignore]`d ones and cfg-gated ones such as the Flex-only NaN tests).
  4. **The 24 matmul-precision tests** stay listed in the default profile until "Fused slicing for `MatmulMode::Accurate`" is measured
     (above); they pass under `--accurate`.
  5. **Not ours:** 3 Bool-scalar tests under `Fusion<TtBackend>` (burn-fusion 0.21.0 `todo!()`, "Bool scalars under `Fusion<TtBackend>`"
     below), the F64 tests (not a Tensix dtype) and the ttsim `timeout` entries (each passes on silicon).
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
- [x] **B15, loading and checkpoints** (2026-10-12; [`crates/tt-models`](../../crates/tt-models/README.md)).
  `burn-store` (safetensors only) loads ResNet-18, ViT-Tiny (F32) and SmolLM2-135M (BF16) strictly
  ([`load.rs`](../../crates/tt-models/src/load.rs)): `burn-store`'s own `load_from` leaves a tensor the file lacks at
  its random initialisation and skips a shape mismatch, so every disagreement is an error naming the tensors. Two
  `burn-store` behaviours found and worked around: it lists a norm layer's `weight` and `bias` as unused although
  it applied them as `gamma` and `beta`, and its PyTorch adapter renames only batch, layer and group norms, so
  `RmsNorm` needs explicit remaps. The weights and five test photographs are pinned in `PINS.toml` and fetched by
  `cargo xtask fetch-weights` (340 MB, SHA-256 verified). **The BF16 note in [burn-backend-parity.md](burn-backend-parity.md)
  3.4 ("a BF16 model that runs entirely on the host") is stale**: a BF16 checkpoint loads as a BF16 model and
  computes on the device since D1/D2; the B4 warning it asked for names no cast. Gate `step170_checkpoint_round_trip`
  (ttsim 208 s, card 0 2.6 s, run `1791847160`): train, save with Burn's `BinFileRecorder` (what the `Learner`'s file
  checkpointer uses), load into a fresh model, continue, bit-identical to an uninterrupted run, with controls (a model
  that was not loaded does not match; save, load, save is byte-stable).
- [ ] **B15, left:** a real `Learner` run (the gate uses the recorder directly, with plain SGD, which keeps no
  state, so a stateful optimiser's record is unclaimed), and one `burn-onnx` model (`burn-onnx` is not in the lock or
  the registry; deferred by decision 2026-10-12).
- [x] **Model ladder, rungs 1 to 3** (2026-10-12, card 0, `--strict`-style `assert_native_model`: no staged compute,
  no host arithmetic, no downloaded operand). Each compares `TtBackend` with burn-flex on the same weights and
  input, and each host model is first checked against the world (the cat photo is a cat) or the reference
  implementation (SmolLM2 reproduces HuggingFace `transformers`: the same top-10 next tokens on two prompts, logits
  to 5e-5):
  1. **ResNet-18**, `step165_resnet18` (run `1791847160`): the same top-5 in the same order as Flex; largest logit
     difference 5.1e-2 on 14.6 (the fast matmul's TF32 rounding). 17.8 s including the host reference.
  2. **ViT-Tiny/16**, `step166_vit_tiny` (`1791847160`): the same top-5; 6.5e-2 on 12.5; 2.1 s. Layer norm, GELU,
     softmax and Burn's `attention` all native.
  3. **SmolLM2-135M**, `step167_smollm2` (`1791844568`, F32, all 30 layers): the same 16 greedy tokens as Flex,
     teacher-forced top-1 equal at every position, largest logit difference 6.9e-2 on 33.2. RoPE's `sin` and `cos`
     run on the device, grouped-query attention reads a cache filled by `slice_assign`, the output head is the
     tied embedding table. 575 s, almost all of it the embedding lookup (below). **In the checkpoint's own BF16**
     (BF16 parameters, activations and cache; 6 layers and 4 tokens, `1791848957`, 441 s): teacher-forced top-1
     equal at every position, the four greedy tokens identical to Flex, largest logit difference 0.75 on 63.3 (BF16's
     8-bit mantissa). A BF16 token costs **37.5 s against F32's 14 s**: the BF16 lookup is 2.7x slower still; the
     whole 30-layer BF16 model timed out at 1,500 s (`1791847160`).
  4. **Fine-tuning** is open.
- [x] **Mixed precision in one model** (2026-10-12; requested: BFP2/4/8, BF16, FP16 and F32 together):
  [`quant.rs`](../../crates/tt-models/src/quant.rs) names a format per matrix (and per activation boundary) by
  parameter path; `step168_smollm2_mixed_precision` (card 0, `1791845324`, 6 layers, 7 plans, all pass) runs each plan
  natively and against an independent host model (`bfp_round_trip`, the packer's rule decoded with `tt_isa`'s
  unpackers; plain F32 from there, which is what the device computes with after decoding). BFP8, BFP4, BFP2 land
  0.097, 0.094 and 0.045 from their references against 0.59, 16.5 and 34.9 from the unquantised model (the F32
  plan's own device-vs-Flex noise is 0.12), so the device really compressed; one model holding all six formats
  (weights in BFP8/4/2, BF16 and F16, activations in BF16 and BFP8) lands 0.38 from its reference against 34.9.
  **Caveat:** BF16 and F16 are weaker evidence. Their effect on the logits (0.26 and 0.30) is the size of the
  difference between two valid roundings of a half type, and for F16 the device is nearer the unquantised model
  (0.07) than Flex's F16 run (0.34): burn-tt computes in F32 and rounds once per op (D1), Flex rounds per
  primitive. The gate asserts native execution, finite logits and top-1 agreement where the margin allows, and
  nothing more for them. What the formats cost the model (host, 30 layers, one position): F16 0.017, BF16 0.26,
  BFP8 1.4, BFP4 19.8, BFP2 23.0 (top-1 holds through BFP8). Left: the full 30 layers per plan (the lookup makes each
  plan two minutes), the half formats against a per-matmul reference, and BFP in a trace or under autodiff.
- [x] **A real LLM of 1B parameters or more: SmolLM2-1.7B-Instruct** (2026-10-15). SmolLM2-135M is a test vehicle,
  not a model worth running; performance work is now measured on this one. It is the same Llama architecture at 24
  layers of width 2048, 32 heads (multi-head), vocabulary 49,152, tied head, F32 (6.8 GB of weights). What it took:
  - **Pinned:** `smollm2-1.7b-instruct`, 3.4 GB BF16 (the fetch timeout now scales with size).
  - **The preset** `SMOLLM2_1_7B_INSTRUCT` and `Llama::smollm2_1_7b_instruct`, with no model code changes.
  - **A reproducible reference:** `crates/tt-models/reference/llama_reference.py`, a uv script pinning `transformers`
    5.19.0 and torch 2.14.1 (CPU).
  - **The host test `llama_1_7b_host`:** the preset equals `config.json`; all 218 tensors load; the top-10 on a raw
    and a chat-templated prompt equal `transformers`, the logits within 3.3e-5; and the same greedy answer, "The
    capital of France is Paris."
  - **Gate `step188_smollm2_1_7b`** (card 0, run `1792080640`):
    - device against Flex: the same 16 greedy tokens, the largest logit difference 0.116 on 42.4;
    - traced against eager: bit for bit across chunks of 32 and 64 and step lengths 64 and 128.

    It is not in `--smoke` (3.4 minutes); run it with `--filter step188` and `--timeout-secs 1800`. The traced test is
    silicon only: the simulator's one tile cannot hold a 2048-wide capture's programs (trace segmentation). Shared
    helpers are in `crates/tt-tests/tests/ladder_support`.
  - **Measured** (card 1, every tile, 512-position cache, run `1792082994`,
    [log](../learnings/performance-log.md#smollm2-17b-instruct-on-the-card-2026-10-15-card-1)):
    - **decode 13.6 / 13.1 / 12.6 tokens/s** against burn-flex's 3.4 / 3.2 / 3.0, at 32 / 128 / 256-token prompts
      (4.0-4.2x);
    - **prefill 267 / 657 / 771 tokens/s** against 7.6 / 23.0 / 24.7 (29-35x);
    - a 256-token prompt plus 64 tokens in **5.3 s against 31.3 s**.

    Decode reads its weights at only **86-93 GB/s, 17-18% of GDDR's 512**. The bandwidth floor is ~14 ms a token,
    against 74-79 ms measured, so decode is far from bound by memory: the next performance target. The traces hold
    155 KB of the 240 KB program cache and 1.5 GB of streams.

  **Profiled** (2026-10-15, the profile mode, run `1792091115`,
  [log](../learnings/performance-log.md#where-a-smollm2-17b-step-goes-the-profile-mode-2026-10-15-cards-0-and-1)). A
  decode step is:
  - matmuls 46% (~200 GB/s);
  - **the cache write's `expand` 24%** (all 32 key/value heads expanded to the cache's shape twice a layer);
  - ~30% in ~1,200 small ops.

  **Next, by that profile:**
  - [ ] **Decode at GDDR's bandwidth** ([decode-to-bandwidth.md](decode-to-bandwidth.md), 2026-10-17): the umbrella
    plan. The target is a step at **90-95% of GDDR's bandwidth** (~14.5-15.5 ms a 1.7B step, F32, batch 1, bit for
    bit), not the roofline's 65%. It orders the items below:
    - Phase 0: the read ceiling;
    - Phase 1: the cache write;
    - Phase 2: [decode-attention.md](decode-attention.md);
    - Phase 3: barriers, row-limited element-wise ops, fusers;
    - Phase 4: the products at the ceiling;
    - Phase 5: the weight prefetch under the small ops.
    - **2026-10-18:** the four larger changes compared
      ([there](decode-to-bandwidth.md#the-larger-changes-compared-2026-10-18)), and all four built, bit for bit:
      - the position-major cache;
      - the m = 1 kernel's rows and a blocked `K` streamed (compute under the reads);
      - a packed `A` (`A` once a product);
      - the next product's weights prefetched on idle units.
      - **Result:** a 1.7B step **33.02 / 33.52 / 34.42 -> 26.47 / 27.14 / 27.97 ms** (`1792364210`), 135M 14.04 /
        14.45 / 15.02 -> **12.62 / 13.06 / 13.59** (`1792364661`).
      - **Bandwidth (2026-10-19):** the products run at the read ceiling (462 GB/s over their spans); the step's
        weights stream at 56% of 512 GB/s (284.8 GB/s, `1792383988`) because ~7.6 ms of small ops sit between the products
        ([log](../learnings/performance-log.md#where-decodes-bandwidth-stands-and-a-column-broadcast-in-the-sfpu-2026-10-19-cards-0-and-1)).
      - **The small ops, under way** (decode-to-bandwidth Phase 3, "The small ops, 2026-10-19"): a column operand
        broadcast by the SFPU, RMSNorm's statistic from the packed row, row gathers dealt by rows, the prefetch keeping
        its units, packed twins, a barrier released by unit 0: 1.7B **24.04 / 24.48 / 25.39 ms** (`1792383988`), 135M
        **11.40 / 11.80 / 12.26** (`1792378993`). Weights stored transposed were slower (not built). Next there: a
        deeper weight prefetch and prefetch for the K-blocked `down`; RoPE over `q|k` in one pass and
        `x / stat * gamma` as one op are documented and skipped (~1 ms together).
  - [ ] **The step's bandwidth budget** ([decode-step-bandwidth.md](decode-step-bandwidth.md), 2026-10-17).
    - **B0, done:** the profile mode counts every captured op's GDDR bytes and entries (gate `step197`).
      - A 1.7B step moves 9.6 GB, which at 469 GB/s takes 20.5 ms, against 42.3 ms measured.
      - Its weights alone take 14.8 ms.
      - The products run at 347 GB/s; `gate|up` at 367, not ~490.
      - The activation is 1.25 GB a step, and the cache copies 0.83 GB.
      - The other 1,007 ops take 18.3 ms and move little.
    - **B1, done:** a unit's later blocks of a product read `A` once (`GATHER_A_RESIDENT`, gate `step198`, bit for
      bit). A 1.7B step **42.97 / 45.41 / 48.59 -> 41.83 / 44.25 / 47.40 ms** (`1792209913`); the 135M step is
      unchanged.
    - **B2, designed:** a cross-op weight prefetch into L1.
      - Capped at ~4.4 ms a step: L1 holds ~75 MB, against 273 MB of weights a layer.
      - It needs barriers on the units an op ran on first (replay-fixed-cost P1).
      - Recommended next: P1, then G7 (the activation multicast), then the prefetch if a replay probe shows it hides.

    The items below continue under their own plans.
  - [x] a row write at a device-held position for the cache (2026-10-16; decode-length-cost B3): `scatter_nd` with
    `Assign` is one row scatter at resident indices (NC's `SCATTER_ROWS`, `Session::scatter_rows`, gate `step189`),
    and `FixedCache` writes a step's `kv` rows with it. A 1.7B decode step **73.65 / 76.08 / 79.30 -> 57.21 / 59.64 /
    62.81 ms** (17.5-15.9 tokens/s, run `1792109477`); the 135M step 24.40-25.53 ms, below burn-flex's 26.66-31.12
    ([log](../learnings/performance-log.md#a-cache-write-of-one-row-a-head-2026-10-16-cards-0-and-1)). The write is now
    5.0 of 56.8 ms a 1.7B step; its two whole-cache copies (2.3 ms) are left (Track 2 item 6, step 6);
  - [ ] the matmuls' bandwidth ([decode-matmul-bandwidth.md](decode-matmul-bandwidth.md); P0-P3 done 2026-10-16,
    [log](../learnings/performance-log.md#decode-matmuls-read-only-the-activation-rows-they-use-2026-10-16-cards-0-and-1)).
    - An `m = 1` product reads each activation tile through its used rows (gate `step190`), and its last round is
      spread over the units.
    - `Llama::fused` makes `q|k|v` and `gate|up` one product each, bit for bit (gate `step191`).
    - **Result:** a 1.7B decode step **57.21 / 59.64 / 62.81 -> 47.90 / 50.37 / 53.53 ms** (20.9 tokens/s, run
      `1792161623`), its products 33.8 -> 24.0 ms; the 135M step 22.91-23.98 (burn-flex 26.4-30.6).
    - `gate|up` now reads at GDDR's ceiling. `o` and `down`, on 64 units, move ~300 GB/s, and **not** because of the
      in-flight cap (16/32/128 measured, unchanged).
    - **Open: P5**, a probe of one channel a column, then perhaps a skewed weight placement.
  - [ ] **The small ops** ([decode-small-ops.md](decode-small-ops.md)): 23.4 of a 47.5 ms step, now ~20.7 of 44.7.
    - **S1 and S2a are done** (2026-10-16, gate `step192`).
      - Element-wise ops read a block of a buffer in place.
      - RoPE runs in the projection's layout: its table is laid over the heads once a step, and its halves are swapped
        by tile copies.
      - A 1.7B decode step **47.90 / 50.37 / 53.53 -> 45.23 / 47.71 / 50.78 ms** (22.1-19.7 tokens/s, run
        `1792174499`); the 135M step 22.46-23.63 (`1792174412`).
    - **Open: S1d.** The key write now repacks, as the value's did; reading both in place saves ~1.8 ms.
    - **F0 measured** (gate `step193`). On `Fusion<TtBackend>` the step captures and replays, but costs
      +2.2 / +1.5 ms (1.7B / 135M): burn-fusion turns `sigmoid` into Burn's 8-op composition, and fuses nothing in
      the step.
    - **Decided: burn-fusion fusers are the end state** (2026-10-16). They are reliable, and serve any model on TT;
      individual ops are still made as fast as they go.
    - **The three fusers are done** (2026-10-16, `burn-tt` `fusion/chain.rs`; gates `step194`, `step195`, `step193`
      on ttsim and card 0).
      - A 1.7B decode step on `Tt` is **45.23 / 47.71 / 50.78 -> 43.42 / 45.85 / 49.00 ms** (23.0-20.4 tokens/s, run
        `1792186180`); the 135M step 22.46-23.63 -> **20.20-21.34** (`1792186583`)
        ([log](../learnings/performance-log.md#decode-fusers-sigmoid-silu-and-rmsnorm-2026-10-16-cards-0-and-1)).
      - Burn's sigmoid composition runs as `TtBackend`'s sigmoid. A decode step on `Tt` is `TtBackend`'s bit for bit
        (step193 asserts the logits), and decode moves to `Tt` (`TracedGenerator`'s default; the bench's
        `TT_GEN_FUSION=0` opts out).
      - `silu(gate) * up` is one SFPU pass (`SILU_MUL`), bit for bit the three ops, reading a fused projection's
        halves in place (S2b).
      - RMSNorm up to its scale is two ops (S3): its statistic in one kernel (`sfpu::norm`: the sum of squares in
        `sum_dim`'s order, then `* 1/n`, `+ eps`, `sqrt`), then the existing `DIV`. Bit for bit the five ops. A
        version that also divided ran 640 us a norm on one unit (silicon, the 1.7B step 71.5 ms). The scale stays its own op:
        burn-fusion puts its `Reshape` in another block.
      - The statistic runs only the row groups a tile row's rows occupy (2026-10-16, gate `step196`): a decode row's
        kernel 14.95 -> 5.58 us. 1.7B step **42.97 / 45.41 / 48.59 ms** (`1792189445`), 135M **19.94-21.11**
        (`1792189939`). Reading only those rows too was slower (19.2 -> 21.0 us) and was reverted.
      - Found: a chain must be able to begin one op into a block (a residual `Add` holds the norm's first op), and
        `ElementWiseFuser` lost its match to any later op while another fuser was open.
  - [x] **The cache write's two whole-cache copies** ([in-place-cache-write.md](in-place-cache-write.md) D1-D4,
    2026-10-17, gate `step199`): donation declared before capture, buffer generations, then `scatter_rows_into`;
    the write's indices read as a row. A 1.7B step **41.83 / 44.25 / 47.40 -> 38.97 / 41.41 / 44.58 ms**
    (`1792263398`), 135M **18.59 / 19.07 / 19.79** (`1792263801`). D5 (the k/v repack) moved to
    [decode-attention.md](decode-attention.md).
  - [ ] **The roofline** (2026-10-17, [scoreboard](../learnings/performance-scoreboard.md#roofline-smollm2-17b-f32-one-card),
    [log](../learnings/performance-log.md#the-17b-against-its-roofline-and-a-prefill-products-rate-2026-10-17-cards-0-and-1)).
    For F32 on one card, the "realistic" targets are 25 ms a decode step and ~21 ms for a 256-token prefill; we are at
    47.44 ms and 297.4 ms (`1792258449`). Decode closes through the items above. Two items are new:
    - [ ] **Prefill products that reuse their operands.**
      - **Now:** a prefill-shaped product reaches 2.0-4.2 TFLOPS against ~166 at HiFi4 (`matmul_throughput`,
        `1792258604`). It is bound by GDDR because each unit gathers its own tiles: `B` is read `m / 32` times, `A`
        once per two-column block, up to once a unit. The largest product reads 3.2 GB for 77 MB.
      - **Step 1:** a 2-D blocked product. Each unit holds an `A` strip and a `B` strip in L1 and streams `K`,
        keeping each tile's `K` chain in order, so it stays bit for bit. This is ~4x fewer bytes.
      - **Step 2:** multicast along the grid's rows and columns, so each operand leaves GDDR about once. This is the
        same mechanism as G7 in [decode-step-bandwidth.md](decode-step-bandwidth.md). ttsim refuses
        non-rectangular multicast.
      - **Target:** the products compute-bound, ~40x from today.
    - [ ] **Batched decode** (design: [batched-decode.md](../proposals/batched-decode.md), 2026-10-18; unscheduled:
      batch-1 prefill and decode come first, and are built with it in mind).
      - **The largest knob for total throughput:** a step reads every weight once whatever the batch, and up to 32
        rows (one tile row) stay bound by the weight stream at HiFi4.
      - **The design:** sequences are the rows of one tile row; the batch sizes are 8, 16 and 32 (`MVMUL`'s row
        halves, `LiveRows`); the cache is a pool of position-major 32-position pages with a page table a sequence;
        attention is batched items, one per (sequence, head); sampling happens on the card.
      - **Now:** `TracedGenerator` is batch 1, so the roofline's batch 4-32 rows (132-863 tokens/s "realistic") are
        unreachable.
      - **Why it pays:** a step is mostly per-op fixed cost, and an `m <= 16` product's thin reads cost the same
        entries. So batch 2-16 should cost little more a step.
      - **It needs:** a batch dimension in `FixedShapeDecoder`; a cache per sequence, with each sequence's position
        and attended length; and the row scatter per sequence.
  - BF16 weights (half the bytes a token);
  - Llama-3.2-1B or Qwen2.5-1.5B for grouped-query attention, RoPE scaling and QKV biases.
- [ ] **Partial ops** (from the old coverage tracker): `float_matmul` edge shapes still
  staged through the host, `mean_dim`, gather/scatter/select, `slice_assign`/`cat`, conv
  acceptance, attention variants, `int_div`/`remainder` on card 1, ND indexing, GMPOOL
  max-with-indices. These are closed by the ladder, not separately.

## Track 2: performance (host-bound first)

Goal: training at or below burn-flex per step on MNIST and the transformer, then on a
Track 1 model. Evidence and bottlenecks:
[performance-scoreboard.md](../learnings/performance-scoreboard.md) ("Where the overhead is").

**What the model ladder measured (2026-10-12, [profile](../learnings/performance-log.md#model-ladder-real-models-on-the-card-2026-10-12-cards-0-and-1)),
and so the order to work in.** Against burn-flex on the same machine: ResNet-18 105x slower (16.3 s), SmolLM2 decode
25x (619 ms a step) and ViT-Tiny 7.8x (815 ms), with a token embedding alone 15.9 s. The first two have a single cause
each, and the cheapest wins; ViT and decode are host submission, the fusers' territory. The `.ttinsn` trigger (SFPU
push above ~20% of device time with the model device-bound) has not fired: no model is device-bound yet.

- [x] **1. Gather without staging** (2026-10-13, F32, I32 and Bool; ttsim and card 0). `select` along dim 0 of a
  rank-2 table with 1-D I32 indices runs `Session::select_rows` (`tt-kernels/src/index.rs`): per batch one tile of 32
  resident indices is read and B's new `SELECT_ROWS` record (`tt-isa/src/dm/record.rs`) turns each into the table's
  two 64-byte face-row reads a tile column, the source tile found on the card, so a lookup reads the rows it returns and
  builds no host address (`select_native` routes to it, `ops.rs` `select_rows_native`; `embedding` reaches it, asserted by the
  gate; `gather_nd` also calls `select_native` and its gate, `step127`, still passes, but which path it takes is not asserted). Gate `step171_select_rows` (10 tests; card 0 run `1791851883`): bit for bit the host indexing
  (signed zeros, subnormals, NaN payloads, infinities, `f32::MAX`; I32; Bool; ragged `n` and widths; a table of 25 tile
  columns, past one batch; a view starting mid-grid), `DOMAIN` for an index at the row count or negative, residency and
  padding chains, and a **trace replay with changed indices**; four mutations of the record (halves swapped, source row
  mod 31, `bound` one too large, output row off by one) each fail it, and the record's unit tests hold the addresses to
  the face layout independently. **Measured, card 0, release:** one token from SmolLM2's 49,152 x 576 table
  **260 us, was 15.9 s** (five tokens 278 us, was 63.3 s; `probe_embedding_cost`, `1791851940`, validated against the host's
  rows); `step167_smollm2` F32, 30 layers, **33 s, was 575 s** (`1791851959`); a decode step from a token id 623 ms with
  the lookup about 1.4 ms of it (`1791852097`): a decoded token is the host-bound 24x of item 3 again. Burn's own
  select, gather, embedding and index tests: 200 of 200 on ttsim, plain and under `Fusion<TtBackend>`, `--strict`.
  **Left:** BF16, F16 and BFP tables, rank 3 and up, other dims and `gather` with per-element indices still stage
  (item 5 and the entry below). Cost to know: the mover image had 172 bytes left of its 24 KiB (since raised to 42 KiB by repacking the role images, which unblocks a B-side packed-BF16 branch)
  ([operating notes](../learnings/silicon-operating-notes.md#riscv-bs-image-budget-2026-10-13-ttsim-and-card-0));
  also found: `cargo clippy` inflated every firmware image by 900 bytes (`build.rs` now strips `RUSTC_WORKSPACE_WRAPPER`).
- [x] **2. Chunk-free convolution and pooling** (2026-10-14, F32/BF16/F16 forward; ttsim and cards 0 and 1;
  [log](../learnings/performance-log.md#chunk-free-convolution-and-max-pooling-2026-10-14-cards-0-and-1)). A 2-D convolution is one
  tap select and one product (`ops/nhwc.rs`): `Session::select_taps` (`tt-kernels/src/index.rs`) builds the patch matrix `[P, taps x C]`
  from the NHWC table with B's existing `SELECT_ROWS` (a `[taps, P]` index of table rows, geometry-only, kept per attachment; tap `t`'s
  row reached by moving `idx_at`, no firmware change), in Flex's im2col order; the weights relaid to `[taps x C, O]` are kept on the
  weight (`TtTensor::derived`). The `[P, O]` result is an NCHW view of NHWC, and element-wise ops run on such views in place
  (`ops::view_op`: BatchNorm's `[1, C, 1, 1]` is a row), so nothing is relaid between layers. Max pooling (F32, `C % 32 == 0`) is one
  select (a padding tap reads its window's first valid datum) and an exact fold over the taps. Gates: `step178_select_taps` (bit for
  bit against host indexing, specials, 49 taps across index tile rows and staging, the zero row of a dirty table and of a whole-tile
  one, `DOMAIN`), `step179_nhwc_conv_pool` (scalar oracle on integers for conv1-like, 3x3/64, whole-tile-row, 1x1 s2, dilated and
  biased cases; max pool bit for bit with NaN payloads, signed-zero ties, subnormals, ceil and dilation; a warm ResNet block with no
  repack; a trace replay), each watched failing under five mutations; both in `SMOKE`; step85/78/162/165 and MNIST e2e unchanged;
  smoke on card 0 1,024 of 1,024 (run `1791997050`, the selection less its 10 mesh tests); Burn's conformance on ttsim with no
  unexpected result in the plain, `Fusion<TtBackend>`, `--exact` and `--accurate` profiles.
  **Measured (card 1, release):** ResNet-18 forward **16,288 -> 782.3 ms on one tile** (5.0x burn-flex's 156.9, `1791996580`), and
  **131.1 ms on every tile at `TT_SPREAD=work:64:4`, faster than burn-flex's 151.4** (`1791996716`); per op, NHWC in and on the card,
  max pool 2,817 -> 37 ms and a 3x3 convolution 30 / 16 / 10 / 8 ms (was 895 / 517 / 529 / 1,411). The host now submits a forward
  in ~15 ms: on one tile the rest is the card, conv1 ~366 ms of it (a 3-channel table read 64 bytes a row) and the image's relayout
  ~95 ms. **Left (chunked path still):** backward passes (weight gradient = `patches^T @ G`, one product; x gradient as a gather over
  taps), `max_pool2d_with_indices`, average pooling, grouped and 3-D convolutions, pooling with `C % 32 != 0`, and the non-Fast
  matmul modes. The default tile choice is what takes ResNet-18 under burn-flex (below).
- [x] **3. The decode step's host cost (25x; closed 2026-10-14, 3.8x on every tile; left: item 9).** 2,779 reported operations (more submitted) at about 220 us each:
  `cat` is built from `slice_assign` (722 `float_slice_assign` calls a step: the nine-way `repeat_kv`, the rotary
  halves), plus tens of thousands of reshapes, slices and `float_device` calls over 33 passes. A native `cat`, a
  fused RoPE and the fusers below shrink it; a trace would remove it, which needs trace segmentation. ViT-Tiny
  (709 operations at 1.1 ms) is the same cost at a smaller scale. Plan: [decode-host-cost.md](../completed-plans/decode-host-cost.md)
  (copy-shaped ops as row and block moves instead of per-element host maps; a native `cat`).
  **Re-measured 2026-10-14** (P0, card 0, jobs and server-thread time per op, [log](../learnings/performance-log.md#decode-step-where-the-host-time-goes-2026-10-14-card-0)):
  - **One tile:** 624 ms (`1792005087`), mostly waiting on the device's m = 1 matmuls (1.27 ms each). The 2,779
    "operations" are 4,279 jobs.
  - **Every tile with `work:64:4`:** 233 ms (`1792005178`). `cat`'s zero-fills and `slice_assign`s are 61% of the
    server thread and `attention`'s 1,920 jobs 22%.
  - **Native `cat` (P1, 2026-10-14):** `cat` and `repeat_dim` are one row gather or one element repack (`ops/region.rs`,
    gate `step180_native_cat`, ttsim and card 0). The decode step is **117.3 ms on every tile** (`1792006312`; burn-flex
    26.3, so 4.5x) and 458 ms on one tile. `step167` F32 takes 23.4 s, was 33.2.
  - **Cache writes (P3, 2026-10-14):** a `slice_assign` of whole rows is a tile copy with the value's rows written
    over it (`Session::assign_rows`, gate `step181_region_assign`). A cache write costs 25 us of server time (was 254).
    The step is **108.5 ms on every tile** (`1792010467`, 4.1x burn-flex).
  - **Views (P2, 2026-10-14):** slices with positive steps are views, `swap_dims` no longer materialises a view,
    and a ragged view is made plain by a tile copy (gate `step182_region_views`). The step is **99.6 ms on every
    tile** (`1792013124`, 3.8x burn-flex) and 450.4 ms on one tile.
  - **What is left:** `attention`'s 1,920 jobs (73 of 125 ms of server time) is item 9. On one tile, the m = 1
    matmuls (270 ms) are "Per-workload tile choice".
- [x] **4. ViT-Tiny forward (7.8x, 815 ms; closed 2026-10-14, 2.7x on every tile; left: item 9).** 709 reported operations at about 1.1 ms, first read as host submission, the same
  cost as item 3 at a smaller scale. Layer norm is composed (`float_mean_dim`, `sqrt`, `div`: 50 `mean_dim` and 25 `sqrt` calls a
  pass); GELU and attention are native. Benchmark: `silicon_bench_models::vit_tiny_forward`, against burn-flex's 104.7 ms.
  **Re-measured 2026-10-14** (card 0, `1792005205`, `1792005220`): 856 ms on one tile and 401 ms on every tile at `work:64:4`
  (burn-flex 107), from 853 jobs.
  - It is **not per-op submission**. On every tile the time is `attention` (348 jobs, 211 ms of device work), then the element
    repacks of `swap_dims` and `slice` (74 jobs, 83 ms; [decode-host-cost.md](../completed-plans/decode-host-cost.md) P2).
  - **After P2 (2026-10-14):** **283.4 ms on every tile** (`1792013104`, 2.7x burn-flex's 106.7) and 663.0 on one tile.
    Its slices and swaps queue nothing, and attention reads the heads in place. Attention is now 187 of 270 ms of
    server time (item 9), then the patch embedding's convolution (44 ms) and the heads' merge (`reshape`, 12 jobs, 10 ms).
- [x] **9. Attention (closed 2026-10-15; ViT-Tiny 1.18x burn-flex on every tile; [attention-jobs.md](../completed-plans/attention-jobs.md)).** `module::attention`
  (`ops.rs`) composes `matmul(q, k^T)`, a scale, the masks, a softmax (`max_dim`, `clamp_min`, `sub`, `exp`, `sum_dim`,
  `clamp_min`, `div`) and `matmul(p, v)`. On card 0 it is now the largest cost of both transformers:
  - **SmolLM2 decode step, every tile:** 1,920 jobs, **64 a call**, and 72 of the 146 ms of server-thread time after the
    native `cat` (`1792006312`).
  - **ViT-Tiny, every tile:** 348 jobs and 211 of 392 ms, device work at 0.6 ms a job (`1792005220`).

  First find which of the 64 jobs are batched-matmul splits per head, broadcast copies for the `sub` and `div`, and the
  causal mask's upload. P0's counters give the jobs per op inside the call. Then fuse what the numbers favour: one
  batched product per matmul, a softmax over rows in one job or a few (the reduce/norm fuser below), and the mask as
  geometry. The arithmetic order stays Burn's: the existing attention gates must hold bit for bit or within their derived
  bounds. FlashAttention ("Later") is the end state, not this item. Plan: [attention-jobs.md](../completed-plans/attention-jobs.md).
  - **P0 (2026-10-14):** `module::attention` charges its stages to parts (`report::part`). Of a decode call's 64 jobs,
    56 were the two products, each run a head at a time because `m` was not whole tile rows (two repacks and a matmul a
    head). The softmax is 7 and the scale 1; a causal mask adds 3.
  - **P1-P2 (2026-10-14, ttsim, cards 0 and 1):** a batched matmul takes a ragged `m` and heads stacked by rows,
    gathered once into whole tile rows. Each product is one job, bit for bit `matmul_dram`'s
    (`step184_ragged_batched_matmul`, `step183_attention_jobs`).
    - **Jobs a call:** decode 64 -> 10, ViT 28 -> 10, prefill 67 -> 13.
    - **Every tile:** ViT-Tiny **173.5 ms** (was 283.4; burn-flex 107.2) and the decode step **86.5 ms** (was 99.6; burn-flex
      26.5), card 1, runs `1792019882` and `1792019870` ([log](../learnings/performance-log.md#attention-products-in-one-job-2026-10-14-cards-0-and-1)).
  - **P2b and P3 (2026-10-15, ttsim, cards 0 and 1).** Where padding grows the scores by at most a quarter (ViT, not a
    decode's single query), they stay a head per whole tile rows from `Q K^T` to `P V` (`padded_attention`). The
    causal mask is kept per attachment in the scores' layout: a prefill call is 11 jobs, was 13.
    - **ViT-Tiny:** every tile **129.1 ms** (burn-flex 109.3, run `1792026124`); one tile 368.3.
    - **The decode step** is unchanged at 86.9 ms.
  - **Tabled, with their numbers in the plan:**
    - **The softmax's clamps and a fused softmax (P4):** 2.1 of decode attention's 36 ms of server time on every tile.
    - **A padded view every op accepts:** only attention was measured to need it.
  - **What is left in the decode step is not attention:** its single-query products (`qk` 20.6, `pv` 13.4 ms of
    server time, mostly the operand gathers and zero rows of P1-P2) and the default tile choice.
- [ ] **5. BF16 lookup (open; the row select does not take it).** `step167` BF16 (six layers) takes 229 s on card 0 (run
  `1791852139`; its 441 s was `1791848957`) because the BF16 lookup still stages through `gather_indexed_bf16`. A packed
  BF16 row select needs about 400 bytes the B image does not have; the split design (B reads the pair, NC writes the
  half) and the other options are in the
  [operating notes](../learnings/silicon-operating-notes.md#riscv-bs-image-budget-2026-10-13-ttsim-and-card-0).
- [ ] **6. Traced execution of the three models.** ResNet-18 measured (2026-10-14, card 1,
  `silicon_bench_models::resnet18_forward_traced`, run `1792000003`): with every tile open (`TT_TILES=all`, default spread) a
  replayed forward is **17.6 ms against 205.7 ms untraced and burn-flex's 151.7 ms (8.6x faster than the CPU)**, its logits the
  eager forward's bit for bit. It captures only when the programs are spread over many tiles: on one tile, or at
  `TT_SPREAD=work:64:4`, a list's programs do not fit the 240 KB program cache (the bench says so and skips), which is trace
  segmentation (next item). An inference capture reads the relaid weights an eager forward kept (a training capture remakes
  them: its replays write the parameters in place; `step179`'s trace test, watched failing).
  - **ViT-Tiny** (2026-10-15, card 1, `vit_tiny_forward_traced`): a replay is **12.6 ms** on every tile with the default
    spread, against burn-flex's 107.6 (run `1792026627`), and bit for bit the eager forward. At `work:64:4` it is 61.9 ms
    (`1792026634`), where it now captures. One tile does not capture
    ([log](../learnings/performance-log.md#vit-tiny-traced-2026-10-15-card-1)).
  - **SmolLM2 decode step** (2026-10-15, card 1, `smollm2_decode_traced`, gate `step185_fixed_decode`).
    - **How it traces:** `Llama::step_fixed` gives the step fixed shapes. The position is a tensor; the rotary rows
      are a row select; the cache is written by `mask_where` and attention masks the later positions.
      `TracedInference::capture_with_state` writes the cache back on the card after each replay.
    - **Result:** a replayed token is **37.1 ms** on every tile, against burn-flex's 26.6 (run `1792029144`), and bit
      for bit the eager fixed step. That is 83.5 ms at `work:64:4` and 478.3 on one tile, where it also captures
      ([log](../learnings/performance-log.md#smollm2-decode-step-traced-2026-10-15-cards-0-and-1)).
    - **Since (2026-10-15, [replay-fixed-cost.md](replay-fixed-cost.md)):** **26.46 ms a token against burn-flex's 26.42** in
      the same run (`1792036613`), from the profile and five fixes below (next step 1).
  - **SmolLM2 prefill** (2026-10-15, card 1, `smollm2_prefill_chunk_sweep`, gate `step185`).
    - **How:** `Llama::step_chunk` feeds the prompt `c` tokens a replay, the start a tensor. It is bit for bit the
      eager prefill, and the decode step after it matches too.
    - **Cost model:** a replay on every tile costs about **33 ms + 0.18 ms a token**, so a prompt wants one chunk
      that covers it: buckets of 64 and 256, plus 128 if its trace fits.
    - **Result:** 256 tokens take **79 ms** against burn-flex's 1,040, and 32 tokens 40 ms against 175. One tile
      captures only single tokens ([log](../learnings/performance-log.md#smollm2-prefill-chunk-size-2026-10-15-cards-0-and-1)).
  - **Where traced generation stands** (card 1, every tile, F32, batch 1, 2026-10-15). End to end, ids in and logits
    out (`TracedGenerator`, 512-position cache, run `1792078489`, after step 4), with burn-flex in the same run:

    | Prompt + 64 tokens | Traced | burn-flex | Decode step, traced / burn-flex |
    |---|--:|--:|---|
    | 32 | **1,688 ms** | 1,866 | 26.13 / 26.75 ms |
    | 128 | **1,726** | 2,345 | 26.60 / 29.10 |
    | 256 | **1,795** | 3,015 | 27.31 / 31.57 |

    The first token comes 5x to 15x sooner. The parts, before the driver (embedding lookup excluded; after
    [replay-fixed-cost.md](replay-fixed-cost.md), runs `1792036613` and `1792036724`):
    - **Decode:** 26.46 ms a token with a 64-position cache (**38 tokens/s**), burn-flex 26.42 in the same run. A
      one-token chunk over a 512-position cache (`step_chunk`, not `step_fixed`) is 32.7 ms.
    - **Prefill, one replay a bucket:**
      - 32 tokens in 34.8 ms (about 920 tokens/s);
      - 64 in 38.7 ms (1,650 tokens/s);
      - 128 in 49.7 ms (2,580 tokens/s);
      - 256 in 73.1 ms (**3,500 tokens/s**).

      burn-flex reaches about 175-240 tokens/s.
  - **Next steps, in order** (each measured as before: release, card 1, every tile with the default spread, burn-flex in
    the same run; correctness bit for bit against the eager path, as in `step185`):
    1. [x] **Profile the fixed ~33 ms a replay** (done 2026-10-15, [replay-fixed-cost.md](replay-fixed-cost.md)). Unit 0's
       mover stamps, op by op (`smollm2_replay_profile`, since replaced by the profile mode, `cargo xtask bench --profile`): ~84 ops a layer at
       ~13 us, not the barrier (0.6 us an op) and not the m = 1 products' arithmetic. Fixed: a matmul refills padding only
       along `K`; one zero row per width kept; a buffer read transposed as `B` transposed once (the tied head, 2.3 ms);
       the model's mask expanded once a step and RoPE's sign in its table; thin downloads untilized by the card. Decode
       37.2 -> 26.46 ms (burn-flex 26.42); prefill 256 tokens 79.0 -> 73.1. Left in the plan: the cache write's
       expands (~2 ms a step; a native row assign at a resident index), the RMSNorm's broadcast `div`, the replay's
       start, each op's ~3.5 us of control (step 6).
    2. [x] **A prefill-and-decode driver, generic over decoders** (2026-10-15).
       - **The pieces:**
         - `tt_models::generate::TracedGenerator` over any `FixedShapeDecoder`: a chunk and a one-token step, both
           functional over a cache;
         - `burn_tt::SharedStateTraces`, several traces over one state;
         - `tt_models::kv_cache::FixedCache` and `rope::Rope`, architecture-neutral;
         - `Llama` driven by `LlamaConfig`, with SmolLM2 its preset `SMOLLM2_135M`.
       - **How it runs:** the step and chunks of 64, 128 and 256 are captured once. A prompt replays the smallest
         chunk that covers it (chunks of the largest past that), then the step a token. The cache stays on the card.
       - **All four traces fit together:** 81 KB of the 240 KB program cache on the fullest unit, though 3.75 GB of
         trace streams in GDDR.
       - **Gate `step187_traced_generation`** (ttsim; card 0, run `1792071747`): bit for bit, token for token, against
         the eager schedule (`generate_fixed`) and against the growing cache. It covers:
         - SmolLM2 at 30 layers;
         - Llama shapes SmolLM2 lacks (multi-head attention, GQA 4:1, head widths 32 and 128, untied heads, ragged
           widths);
         - a GPT-2-style decoder;
         - one driver serving many prompts;
         - a negative control (chunks over an unshared cache diverge).
       - **Measured** ([log](../learnings/performance-log.md#traced-generation-end-to-end-2026-10-15-cards-0-and-1)):
         over a 320-position cache, a 256-token prompt plus 64 tokens takes **1,860 ms against burn-flex's 2,996**
         (first token 61.7 against 1,020), 128 + 64 1,841 against 2,331, and 32 + 64 1,834 against 1,837.
    3. [x] **The token lookup inside the trace** (2026-10-15, with step 2): the traces take I32 ids and the embedding's
       row select replays with them.
    4. [x] **A decode step that costs the context, not the cache** (2026-10-15,
       [decode-length-cost.md](../completed-plans/decode-length-cost.md)). A step over 512 positions cost 32.29 ms. A
       per-layer profile at 64 and 512 positions showed `repeat_kv`'s `cat` growing most, then attention's reads; the
       write-back copy barely grew. Two fixes, both general to decoders:
       - **Grouped-query attention without `repeat_kv`:** query head groups are folded into rows, attending to the
         cache's own key/value heads. This saves 5 ms a step.
       - **Step traces per attended length** (`Shapes`: 64, 128, 256, ... then the cache's length,
         `FixedCache::step_write_within`): attention reads only the cache's first `L` positions, a view with no copy.
         This saves a further 1.2 ms on short contexts.

       Decode is now 26.13 / 26.60 / 27.31 ms at 32, 128 and 256-token prompts (burn-flex 26.75 / 29.10 / 31.57), and a
       generation 0.60x-0.90x burn-flex's time (run `1792078489`). Gates: `step185` and `step187` (new: steps crossing
       lengths, and a step attending to too few positions diverging), ttsim and card 0. **Left:**
       - chunks of 512 (the per-token prefill cost had not flattened at 256: 0.29 ms, `1792036724`) and chunks that also
         attend only to what they use;
       - ~~the cache write's `expand` and `mask_where`~~ (2026-10-16: a step writes one row a head, step 6).
    5. **One tile and `work:64:4`:** prefill chunks and ResNet-18/ViT capture there only once traces are segmented
       ("Trace segmentation" below).
    6. [ ] **What is left of a replay's fixed cost** (each measured in [replay-fixed-cost.md](replay-fixed-cost.md),
       P0 and P2; decode is at burn-flex, not below it). In order of share:
       - [x] **The cache write's operands** (2026-10-16). Burn 0.21's `select_assign` panics for `Assign` before a
         backend sees it, so the reachable op is `scatter_nd` with `Assign`: now one row scatter at resident indices
         (`Session::scatter_rows`, NC's `SCATTER_ROWS`, gate `step189`; ttsim and card 0), which `FixedCache` uses for a
         step. Left of the write (`Step(64)` profile, 1.7B, `1792110380`, 5.03 ms a step): the scatter's tile copy and
         the trace's write-back, two whole-cache copies (2.26 ms; writing in place inside a capture would remove both:
         how to make that safe is [in-place-cache-write.md](in-place-cache-write.md)),
         and the value's repack and block copy after RoPE (1.47 ms). Prefill chunks still `mask_where` (a padded chunk
         may run past the cache, and a scatter cannot drop rows).
       - **The RMSNorm's broadcast `div`** (27 us, twice a layer): the mover broadcasts the `[1, 1]` column in software
         (`READ_BROADCAST_COL`), and the SFPU divides.
       - **The replay's start** (~170 us): every unit's catch-up and its `CALL` are two enqueues, since the mover takes
         a `CALL` only as a list of its own (a firmware change).
       - **P1, barriers on the units an op ran on** (~0.6 us an op, ~1.5 ms a step), and each op's ~3.5 us of control
         (descriptor pokes, the `PAIR_CALL`'s launch and credits).
- [x] **7. SmolLM2 back in `SMOKE`** (2026-10-13): `step167` F32 (33 s) and `step168` (about 3 minutes, all seven plans pass on
  card 0, run `1791852139`) are in; the BF16 `step167` gate (229 s) waits for item 5 (`xtask/src/silicon.rs`, `SMOKE`).
- [x] **8. Firmware image budget** (2026-10-13, ttsim and card 0; [review](../proposals/firmware-architecture-review.md),
  [measurements](../learnings/performance-log.md#firmware-image-budget-and-core-assignment-2026-10-13-card-0)). The role
  images are repacked into 10 KiB slots (`tt_isa::l1::roles`), so B's mover slot is 42 KiB, not 24: 18 KiB free, same speed
  (smoke 1,030 of 1,030, run `1791855592`). Found: `opt-level` 2 is as fast as 3 and 1.7 KB smaller (measured on two
  benches); `s`/`z` slow the mover 1.6x and the roles' push loop 7x. **Left:** an end-to-end check of `opt-level` 2, then adopt
  it; the loadable-kernel prototype (only if mover per-entry cost becomes the bottleneck: it is not today); live tt-metal
  timings once a ttnn whose SFPI matches the machine's is available (`crates/tt-tests/bench/ttnn_reference.py`).
- [x] **The profile mode** (2026-10-15). One switch for where the time goes, eager and traced:
  - **How:** `TT_PROFILE=<dir>` or `cargo xtask bench|silicon --profile`, from `burn_tt::profile` and
    `tt_kernels::profile_mode`.
  - **What it records:**
    - each Burn op's (and op part's) device time, split into barrier, kernels, control and data;
    - each trace's ops by name: a replay stamps a window of its ops (`dm::TRACE_WINDOW`), the next replay the next,
      so a trace of any length is covered without a replay being repeated or extended;
    - replay splits, and the host's server time and counters.
  - **Output:** `summary.md`, `ops.tsv` and `trace.json`.
  - **Overflow:** a stamp overflow is reported, never a failed run.
  - **The firmware's `profile` feature:** stamping compiled in cost 0.46 ms a 135M decode step, so the default images
    carry none of it.
  - **Gates:** `silicon_profile_mode` (eager attribution, a 600-op trace covered window by window, windows stitched
    against whole profiles, the window's negative control) and `step23` (refused without the feature).
  - **Retired:** the custom `smollm2_replay_profile`.
  - **Left:** prefill chunks replay too few times in the bench to be covered whole (32-46%); mesh engines refuse
    the mode.
- [ ] **Trace segmentation.** Training traces overflow the 240 KB program cache
  ("a list's programs do not fit"). Split into segments so training runs traced
  ([traced-execution.md](traced-execution.md) §8).
- [x] **Trace size in GDDR** (2026-10-15, [decode-length-cost.md](../completed-plans/decode-length-cost.md) A0-A1).
  SmolLM2's step and 64/128/256 chunks held **3.75 GB** of trace streams; now **563 MB**, each trace exactly its
  entries. Every `PAIR_CALL` section had been an allocation of its own of at least one 4,160-byte slot, holding ~810
  bytes. A unit's stream and sections are now one region (sections from 64-byte boundaries, which a mover's chunk read
  needs), written once. The driver's build went from 44.9 s to 18.8 s. Gate `step174` (a bound watched failing on the
  old allocation); every trace gate on ttsim and silicon. What remains is entries themselves, about 25 a section;
  fewer ops is the lever. The original entry, kept for the record:
  SmolLM2's step and its 64/128/256 prefill chunks hold **3.75 GB** of trace streams
  across 120 units (the step 0.45 GB, `Chunk(256)` 1.46 GB; run `1792072109`), about 8 KB an op a participating
  unit.
  - **Likely cause (not yet measured):** each `PAIR_CALL`'s reader and writer sections are allocated separately in
    whole GDDR slots (`Session::store_region`, at least `TILE_SLOT`, 4,160 bytes, each) while holding a few entries.
  - **Measure first:** the entries `UnitFootprint::stream` counts, times 32 bytes, against `gddr_bytes`.
  - **The fix if so:** one contiguous arena per unit per trace, its sections packed at entry alignment.
  - **When:** before the 1B-parameter model (Track 1), whose F32 weights alone take 4-5 GB of a card's 32.
- [ ] **B13b** general SFPU micro-op chain fuser. The decode chains are done (2026-10-16): Burn's sigmoid
  composition, `silu`, `silu(gate) * up` (`fusion/chain.rs`, a declarative chain matcher). Open: the general
  micro-op bytecode ([kernel-fusion.md](kernel-fusion.md) Phase 2).
- [ ] **B13c** matmul + bias + activation epilogue (kernel-fusion.md lines 200–204).
- [ ] **Reduce/norm fuser.** RMSNorm's statistic is done (2026-10-16, `sfpu::norm`, one reduction job a tile
  row, over only the row groups its rows occupy; the division stays a multi-unit op). Open: LayerNorm, softmax, rows past 128 tiles (kernel-fusion.md Phase 4).
- [x] **Barriers on the units an op ran on (2026-10-14).** A multi-tile op's barrier was on every unit, and a unit
  it gave no job to enqueued a barrier list of its own: one PCIe enqueue a tile an op, 355,000 a step for the
  all-tiles transformer. Now only the units an op runs on arrive, and the others catch up with a wait-only
  `BARRIER` before their next list: the all-tiles transformer 404-422 -> 96-98 ms/step (faster than one tile,
  105), all-tiles MNIST 12.9-13.4 -> 6.7 ([performance log](../learnings/performance-log.md#barriers-on-the-units-an-op-ran-on-2026-10-14-card-0)).
  What is left at many tiles is each participating unit's own cost, ~4.6 us a unit an op: the next item.
- [ ] **Per-workload tile choice.** `Session::set_spread` / `TT_SPREAD` (2026-10-14): a session opens every
  tile and each op runs on a fixed count (`Fixed(k)`) or one sized by its work (`Work { share, min }`). At 120
  tiles a fixed spread of 4 is within ~5% of the best small session (MNIST 2.2-2.3 ms/step, transformer ~80);
  spreads that vary per op lose to catch-up and program-placement churn
  ([performance log](../learnings/performance-log.md#per-op-tile-spread-2026-10-14-card-1)). Since then a
  catch-up rides in the unit's first list and no per-op work loops over every open tile: 120 tiles at a fixed
  spread match the best small session, and `Work { share: 64, min: 4 }` is as fast on small ops and 1.45x on a
  2048^2 add ([performance log](../learnings/performance-log.md#open-all-made-strictly-better-2026-10-14-cards-0-and-1)).
  Open: the default (still `All`, and auto-attach still opens one tile; every tile with `Work { 64, 4 }` is the
  measured candidate, after the models and conformance run on it), and the same per card for a mesh. **ResNet-18 since item 2
  (card 1): one tile 782.3 ms, every tile with `work:64:4` 131.1 ms (burn-flex 151.4), every tile with `all` 203.8 ms**
  ([log](../learnings/performance-log.md#chunk-free-convolution-and-max-pooling-2026-10-14-cards-0-and-1)). Auto-attach opens one tile (`TileChoice::First`), and a session's tile
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
- [ ] **Gather without staging, what is left** (the dim-0 F32/I32/Bool lookup is done: item 1). `gather_native` still stages
  every row of its table for every element it returns, for: packed tables (BF16, F16, BFP: item 5), rank 3 and up
  (`select` of a `[V, a, b]` table: an index would yield `a` rows, which do not line up with 32-row output tiles; a
  `[V*a, b]` select with the index expanded by the host is the cheap fix), `select` along other dims, `gather` with
  per-element indices (`float_gather`, `nd_flat_rows`'s metadata, `scatter_add_native`'s domain check, `deform_conv2d`'s
  four corner gathers: the 2^24-address refusal there is unchanged), and a `[1, n]` index layout is the only one
  `SELECT_ROWS` reads (a column of indices would be a flag). Each needs the new-op gates.
- [ ] **B7** batched readback (`download_many`, batched `tr_execute`). (The `Backend::sync` barrier of B8 is
  done with B1.)
- [ ] **X7** small-transfer latency; 9.14 per-request cost (note X6 found the fast
  path gains only 3–4%. **Corrected 2026-10-13:** that is not the request round trip. A read entry is 359 device cycles of
  software and a bare loop issues one in 90–106; the in-flight cap does not matter past 4 (see the
  [read-loop probe](../learnings/performance-log.md#read-loop-probe-what-the-interpreter-costs-per-read-entry-2026-10-13-card-0)).
  Host-encoded read entries are in as a build-time mode (firmware feature `fast-reads`, safe by default; 2026-10-13): 359 ->
  175 device cycles an entry, then 91 (109 with `encoded-guard`) with the tight loop of fast-mode hardening phase 2, 1.95x on mover read lists, ~10% on 16 queued small matmuls, nothing
  on a 512^3 matmul ([measurements](../learnings/performance-log.md#host-encoded-read-entries-2026-10-13-card-0)). Left: trace
  lists, records, which mode a deployment ships, the GATHER/SCATTER-direct and multicast-list ideas).
  **Fast-mode hardening** (branch `fast-mode-hardening`): done: P0 baselines (a record's tile costs a plain entry's ~340 cycles
  on B's reads and NC's writes; NC's writes are 0-4% top-level `WRITE` in the matmul, Burn and scatter flows), P1 safety (the
  checksum bound to a per-start key, tile, chip and op; `validate` also runs the guard; `enqueue_checked` validates;
  `step173_mover_differential`; the encoded entry is 180.5 cycles with the seed load), P2 tight read loop (X6 default,
  `ReadBatch`, `run_encoded_reads`: 91 cycles an entry, 109 with the guard; **one tile reads 57 GB/s at 4 KiB, 3.8x the safe
  default, and 8 tiles 340 GB/s, 2.9x**: the 29 GB/s the aggregate benchmark first showed was its own per-read tracing, now a
  list-level `dm::trace_mode::LISTS`), P3 per-list in-flight cap: not built, the honest sweep shows no effect of the cap (2-8) at any tile count. Open: P4 sealed static read records, P5 NC writes
  (`fast-writes`; sealed `WRITE_RUN`/`SCATTER` first). P7 L1 containment is done: reads and writes name only the data arena, enforced
  on the host in both modes (a survey of the whole suite found no violator).
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

### Design decisions (2026-10-14)

Training and inference both have to run across cards, so data parallelism, pipelining and
(eventually) tensor parallelism must all stay possible. What is decided:

- **A Burn device is one card.** Each card is its own `TtDevice` -- what both of Burn's
  multi-device systems assume, and what pipelining (`to_device` between layers) needs. This answers
  [burn-backend-parity.md](burn-backend-parity.md) open question 5 for now; the mesh as one device
  stays possible as a separate device type later (below).
- **Data-parallel training goes through `DistributedBackend`, implemented natively (B14).** Autodiff
  hands each gradient to the backend as it is computed (`submit_gradient_sync`), and the backend's
  `all_reduce` runs once every card has submitted, so gradient traffic overlaps the backward pass.
  Native means an Ethernet ring driven by the E1 movers, in an order we fix, and later inside per-card
  traces. It is needed regardless: Burn's `distributed` feature (turned on by `burn-train`'s DDP and
  by `burn-core`/`burn-optim`'s `distributed`) makes `Autodiff<TtBackend>` require the trait, whose
  `all_reduce` and `sync_collective` default to `unimplemented!()`.
- **Numerics contract for data-parallel training: deterministic, fixed order.** The all-reduce sums in a
  fixed ring order: the same bits run to run, and bit-exact against a host all-reduce in the same order
  (the oracle). It is not bit-identical to one card, since the gradient sums are reordered; the
  difference from one card is bounded and documented beside the gate, as the numerics rules ask.
- **A single tensor across several devices stays in the design, not built yet.** Today a device tensor
  belongs to one card's session (and `MeshEngine` splits a matmul's operands per op). The placement a
  tensor carries should be able to say *one card*, *replicated on these cards* or *split along a
  dimension across these cards*, with the buffers in per-card sessions, so tensor parallelism can be
  added later without redoing the foundation. **Tensor parallelism (one op across cards, a custom mesh
  device) is an eventual large goal, deliberately not pursued now**; nothing built here should rule it out.
- **`burn-collective` reviewed (0.21.0 source), not the primary path.** It is a library outside the backend,
  generic over any Burn backend, used by `burn-optim`'s `collective` feature (`GradientsParams::all_reduce`
  after backward, one gradient at a time). It moves data with ordinary backend ops (`float_to_device` to copy
  a slice between devices, `float_add`, `float_slice`, `float_cat`), so it works with burn-tt today at the
  speed of our card-to-card `to_device` (a host round trip), with no overlap with backward and no hook for
  our own transport. Its peers are held in a `HashMap`: the Ring and Centralized strategies take their order
  from it (it differs run to run), Tree sorts by device id. With two cards every strategy is deterministic
  (a two-term sum and a halving are exact either way round); from three, only Tree meets the contract above.
  It becomes a usable fallback once card-to-card `to_device` runs over Ethernet. It also brings a
  multi-machine layer (`tokio`, `burn-communication`) this project does not need.

### Foundation (shared by every scheme), in order

- [ ] **Per-card sessions on every tile, with a spread.** Each card's session opens all its tiles and gives
  each op as many as its work pays for (`Session::set_spread`, Track 2). Barriers, catch-ups, program caches
  and the barrier counter are per chip (two cards share no NoC); ordering between cards is explicit Ethernet
  transfers and their acknowledgements. Today a `Fabric` opens one tile per chip (`TileChoice::Exactly`).
- [ ] **Fast card-to-card `to_device` over Ethernet.** The shared primitive: pipelines hand activations over
  with it, `burn-collective` gets fast with no further work, and the native all-reduce builds on the same
  movers. Today it is a host round trip.
- [ ] **Tensor placement** able to express one card, replicated or split along a dimension (see above); only
  the one-card case needs to work at first.
- [ ] **Per-card submission.** One server thread drives every card today, and mesh traces run phase by phase
  with the host doing each Ethernet transfer. Cards are independent between Ethernet sync points: a
  submission queue per card, and cross-card send/wait entries inside each card's trace stream, so a
  training step does not return to the host mid-step.

### Schemes

- [ ] **Data-parallel training: B14 `DistributedBackend`**, a host all-reduce first (the oracle; download, sum
  in the fixed order, upload), then the Ethernet ring.
- [ ] **Pipelining** (layers on different cards) over card-to-card `to_device`; micro-batching later.
- [ ] **Tensor parallelism / a mesh device** (one op across cards): an eventual goal, not scheduled. Today's
  `Topology::Cards` (a matmul split along `N`, bit-identical to one card) is the proof of concept.

### Existing items

- [ ] **9.11** device-resident mesh with chips running concurrently, and `copy_into` on the
  mesh (today `MeshEngine` keeps buffers in chip 0's GDDR).
- [ ] Gradient all-reduce over Ethernet, data-parallel training, **B14**
  `DistributedBackend` (see Schemes above).
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
| [replay-fixed-cost.md](replay-fixed-cost.md) | Track 2 item 6, steps 1 and 6: where a traced replay's time goes, op by op, and the fixes |
| [decode-length-cost.md](../completed-plans/decode-length-cost.md) | Done: trace size in GDDR, and a decode step that costs the context (GQA fold, step lengths) |
| [traced-execution.md](traced-execution.md) §3 | Several traces over one state (`SharedStateTraces`) and the generic decoder driver (`tt_models::generate`) |
| [new-op-definition-of-done.md](new-op-definition-of-done.md) | Standing: what a new op must have before it counts as done |
| [burn-op-notes.md](burn-op-notes.md) | Standing: per-op notes for the Burn backend |
| [phases-0-10-roadmap.md](../completed-plans/phases-0-10-roadmap.md) | Archived: architecture and phase rationale (historical design) |
| [phases-0-10-checklist.md](../completed-plans/phases-0-10-checklist.md) | Archived: phase 0–10 tick-list and evidence |
| [tensix-next-features.md](../completed-plans/tensix-next-features.md) | Archived: dated 2026-10-05 sprint record |
| [hardware-coverage.md](../completed-plans/hardware-coverage.md) | Archived: Phase 10 instruction coverage matrix |
| [performance-scoreboard.md](../learnings/performance-scoreboard.md) | Learnings: current performance baselines and "Where the overhead is" |
| [performance-log.md](../learnings/performance-log.md) | Learnings: dated benchmark run log and change history |
