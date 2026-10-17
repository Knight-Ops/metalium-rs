# Decode small ops

Roadmap Track 1, SmolLM2-1.7B "Next, by that profile". Written 2026-10-16 against `7e9def9`. Status: **S1, S2a, F0, S2b and S3
done (2026-10-16; a 1.7B step 42.97 / 45.41 / 48.59 ms on `Tt` with S3's row-limited statistic, run `1792189445`);
S1d, S4, S5 open.** Companion plans: [decode-matmul-bandwidth.md](decode-matmul-bandwidth.md) (the products, P5 open) and
[in-place-cache-write.md](in-place-cache-write.md) (the cache write's copies). The whole step's byte budget, and the
weight prefetch that would overlap these ops, are [decode-step-bandwidth.md](decode-step-bandwidth.md).

## Why this item

After the decode matmul work, a 1.7B decode step is **47.90 / 50.37 / 53.53 ms** (run `1792161623`). Its products
are 24.0 ms; **everything else is 23.4 ms** (profile `1792162257`, `Step(64)`, 47.46 ms, unit 0), as large as the
products. The weights alone would take ~14.3 ms at GDDR's ~480 GB/s, so the small ops are now half of what stands
between the step and its floor. The 135M step (22.91-23.98 ms, already under burn-flex's 26.4-30.6) has the same ops
at 30 layers and narrower tensors, so it gains proportionally more.

## Where the 23.4 ms goes, by what the model does

From `ops.tsv` of `1792162257` (label | path, the op count a step, ms a step). Allocating the 193 `float_mul | sfpu
gather` ops: 49 `x gamma`, 96 RoPE (two a call), 48 SiLU (`x sigmoid(x)` and `x up`).

| Model part | Ops a step (where) | ms |
|---|---|--:|
| **RMSNorm** (49: two a layer and the final) | `powi_scalar` 0.44, `mean_dim` (a sum, `reduce gather`) 1.63, `mul_scalar` (the `1/n`) 0.29, `add_scalar` (eps) 0.29, `sqrt` 0.56, `div` (a column broadcast) 1.50, `x gamma` ~0.44 | **~5.2** |
| **RoPE** (48 calls: `q` and `k`) | `float_cat \| native repack` 1.83 (the swapped halves), `float_mul \| native repack` 1.82, the SFPU muls ~0.85, the adds ~0.38 | **~4.9** |
| **Attention, not the products** | softmax 1.62 + 0.34, the `qk`/`pv` operands' row gathers 1.12 + 1.13, mask 0.18, scale 0.15 | **~4.5** |
| **The cache write** | [in-place-cache-write.md](in-place-cache-write.md) | 5.0 |
| **SiLU and `x up`** | `sigmoid` 1.08 (45 us each), its `block copy list` 0.20 and a mul's 0.20, the two muls ~0.43 | **~1.9** |
| **Residual adds** | 48 `float_add` | ~0.38 |
| Other | `reshape` repack 0.18, the token/rotary `select`s 0.30, `int_expand` 0.15 | ~0.6 |

Per op: an element-wise op on a `[1, 2048]` activation costs ~6-9 us, a reduction or a broadcast `div` ~30 us (one
unit reads the whole row), and an element repack ~38 us.

## What the code allows (checked 2026-10-16)

- **A trace records lowered device lists, not Burn ops.** `TracedInference::capture_with_state` (`burn-tt/src/trace.rs`)
  runs the step eagerly. Each Burn op goes through `server::submit` as an opaque closure, and `Session::begin_trace`
  appends every segment's mover entries to per-unit streams (`capture_segment`, `session.rs`). The `OpRecord` list
  (`tt-kernels/src/trace.rs`) holds labels and entry ranges only. **There is no peephole at capture**: by the time an
  op is recorded, its kernels are built and have run.
- **Decode traces do not use burn-fusion.** `TracedGenerator` is `FixedShapeDecoder<TtBackend>`
  (`tt-models/src/generate.rs`), not `Tt = Fusion<TtBackend>`. Fusion and traces do compose elsewhere: MNIST captures
  under `Tt`, with `resolve_float_tensor` flushing the stream inside the capture.
- **burn-fusion fusers bind scalars at execution.** A fuser sees `ScalarIr` ids, not values (`fusion/eltwise.rs`), so
  a pattern with an `eps` or a `1/n` reads them from `context.scalars` in `execute`, as `ADD_RELU`'s zero check does.
  The only fuser today is `ADD_RELU`/`RELU`.
- **A coarse backend override is the other precedent.** Burn's softmax is overridden (`ops.rs`, `float_softmax` ->
  `device_softmax_matrix`). But RMSNorm (`burn-nn` `RmsNorm::forward`), SiLU (`activation::silu`, `x * sigmoid(x)`) and
  RoPE (`tt-models/src/rope.rs` `apply`) are composed above the backend: no backend method sees them whole.
- **SFPU programs compose.** One kind per op today, but bodies are functions over a `Program` (`sigmoid_program`,
  `sqrt_program(.., inverse)`, `binary_body`, `divide`), and `ADD_RELU` is a two-op body. The program cache keys on the
  program's words, so a new body needs no cache change. Row and column broadcasts exist (`RowBroadcast` lays `B`'s
  row into Dst; `ColBroadcast` has the mover expand the column, `READ_BROADCAST_COL`). Reductions over columns store
  the result into every column of the output tile, "ready to be broadcast" (`sfpu/reduce.rs`).
- **Bit identity is available for element-wise chains.** Intermediates live in F32 in Dst and in L1 alike, so the
  same SFPU instruction sequence run in registers gives the composed ops' bits (`ADD_RELU` is gated that way). A
  fused reduction is bit-identical only if it sums in `sum_dim`'s order. Where a fusion cannot keep bits, it needs a
  derived bound beside its gate (AGENTS.md: never guess tolerances).

## The decision to make first: where fusion lives

- **(A) burn-fusion fusers** ([kernel-fusion.md](kernel-fusion.md) B13b's SFPU micro-op chain, B13d's reduce-map for
  RMSNorm, LayerNorm and softmax):
  - **For:** general to any Burn model, which suits a general backend; it is the design kernel-fusion.md already
    chose.
  - **Against:** decode must run and trace on `Tt`; scalars arrive at execution; burn-fusion's own overhead under a
    trace is unmeasured.
- **(B) Model-level ops with a native path:** a tt-models trait with the composed Burn default for any backend, and a
  burn-tt implementation behind an entry point (`rms_norm`, `silu_mul`, `rope`).
  - **For:** simple, visible, and gated per op.
  - **Against:** only models that call it benefit (tt-models' Llama, not arbitrary Burn code), and it is a public API
    surface.
- **(C) No fusion: make the slow paths fast.** Several of the largest items are repacks and copies, not op count
  (RoPE's two repacks 3.7 ms, the block copies, the row gathers). They are bit-identical backend fixes that help every
  model.

**Recommendation:** C first (S1, S2a; it needs no API decision and is measured on its own), then F0 to decide between
A and B with numbers, then the fusions.

## Phases

### S1. RoPE without element repacks: done (2026-10-16)

**The premise above was wrong for decode.** In a decode step `x` is `[1, h, 1, d]` with every head in row 0 of the
`[1, W]` projection. Laying it out as `[h, d]` moves data from one row to many rows, which no tile move does
(`tile_moves` fails at the first tile). A tile-aligned `cat` in row-major order could therefore never be a block copy
there.

**What was built instead: RoPE runs in the projection's own layout.** Element-wise results do not depend on layout, so
the bits are unchanged.
- [x] **Element-wise ops on block views** (shared with S2a, below): `x cos` reads `x`'s heads where they lie.
- [x] **The table laid over the heads.** A broadcast operand that is neither a row nor a column of the view's block is
  relaid into the view's layout by tile copies (`ops.rs` `relaid`, `Strided::layout_moves`). It is kept beside the
  table (`TtTensor::derived`), so it is made once a step and read by every layer.
- [x] **The halves' `cat` as a tile permutation.** A `cat` of views of one buffer laid out alike keeps their layout and
  is one block copy (`region::cat_in_layout`).
- [x] **`head_dim` 32 keeps the repack** (halves of 16 columns).
- [x] **A regrouped GQA `q` is made plain once** (`operand_blocks`). In the projection's layout, `[kv, groups, d]`'s
  groups lie along columns and are not blocks. Without this, attention fell to Burn's composition: the 135M step went
  23 -> 40 ms (profile `1792172763`).
- [x] **Gates:**
  - `step192_block_views`: RoPE against burn-flex bit for bit at `head_dim` 64 / 128 / 32, decode and prefill shapes,
    MHA and GQA; no repack at 64 and 128; the table kept for the next layer; the GQA attention gate, watched failing
    without the fix (6 repacks for 1).
  - `step185`, `step187`, `step191`, `step180`, `step182`, `step183`, `step184`, `step188` (card 0 `1792170465`).
  - Two mutations watched failing: a block read without its column offset, and a `cat` reading every piece from the
    first.

### S1d. The cache write reads its rows in place (open; ~1.8 ms a 1.7B step)

The key now reaches `FixedWrite::write` in the projection's layout and repacks there (38 us a layer, 0.9 ms a step),
as the value already did (0.9 ms). This is [in-place-cache-write.md](in-place-cache-write.md) D5's fix, for both: a
`SCATTER_ROWS` whose value rows are read from a strided view (each head's row is a tile-aligned column block of the
projection), or `FixedWrite` taking the rows before the head split.

### S2. SiLU and `x up` (~1.9 ms)

- [x] **S2a (C): the two block copies: done (2026-10-16).** They did materialise the fused product's `gate` and `up`
  column views. An element-wise op on a block view now reads the block where it lies (`Strided::as_block`,
  `Session::eltwise_blocks`, `tensor::sfpu_eltwise_blocks`): each operand's tiles are read at its parent's row
  stride, through `READ_RUN`'s grid width.
  - Gate: `step192` `element_wise_ops_read_column_blocks_in_place` (raw bits, every special value, against the same
    ops on plain copies; ragged at the buffer's edge; a block inside a tile still copied).
  - `sigmoid` 1.28 -> 1.08 ms a step.
- [x] **S2b (fusion): `silu(gate) up` as one SFPU body** (2026-10-16): `kind_sfpu::SILU_MUL` runs `SIGMOID`'s
  program, then `MUL(x, s)` and `MUL(t, up)` in registers. Bit for bit the three ops: the sigmoid and the product
  are never denormal, so their FP32 store and reload change nothing.
  - **Where:** the chain fuser (`burn-tt` `fusion/chain.rs`) runs it on `Tt`. Its halves of a fused projection are
    read in place, one job where there were three.
  - **Gates:** unit `sfpu::ops::silu` (two mutations watched: the sigmoid of `y` fails; not reloading `x` passes,
    since `sigmoid_program` keeps `x` in L0); `step194`.

### F0. Measure before choosing A or B: measured (2026-10-16)

- [x] **The decode step runs and traces on `Tt`.** `TracedGenerator<'m, M, B = TtBackend>` is generic over
  `TracedBackend`, and the bench switch is `TT_GEN_FUSION=1`. Gate `step193_fusion_decode_trace` (ttsim; card 0
  `1792174399`): on `Tt`, traced equals eager bit for bit (synthetic Llamas, MHA and GQA, fused and unfused; the 135M
  checkpoint), and a nudged RMSNorm `gamma` diverges.
- [x] **Replay, capture and warm-up**, card 1, every tile:

  | | `TtBackend` | `Fusion<TtBackend>` |
  |---|--:|--:|
  | 1.7B decode step, ms | 45.23 / 47.71 / 50.78 (`1792174499`) | 47.49 / 49.90 / 53.08 (`1792172342`) |
  | 135M decode step, ms | 22.46 / 22.95 / 23.63 (`1792174412`) | 24.00 / 24.52 / 25.15 (`1792174906`) |
  | 1.7B build, s | 70.2 | 74.1 |
  | 135M build, s | 16.4 | 18.0 |

  - **Why `Tt` is slower:** burn-fusion 0.21 does not forward activations (`burn-fusion-0.21.0/src/ops/activation.rs`),
    so `sigmoid` becomes Burn's default composition: eight ops (cast, `x -1`, `exp`, `+1`, `log`, `x -1`, `exp`,
    cast) in place of one SFPU op.
  - **Bits:** the logits are not always `TtBackend`'s (one case of four differed by 7.9e-6).
  - **Fusion today:** none in a decode step; the one fuser (`ADD_RELU`) starts only on `Add` or relu.
- [x] **What the stream shows:**
  - **RMSNorm is contiguous:** `Mul(x, x)` (not `powi_scalar`), `MeanDim`, `AddScalar(eps)`, `Sqrt`, `Div`, the
    `gamma` `Reshape`, `Mul`. The `1/n` is inside burn-tt's `mean_dim`, and `Drop`s can fall between the ops.
  - **SiLU is contiguous:** the 8-op sigmoid, `Mul(gate, sig)`, `Mul(., up)`.
  - **Scalars are ids at optimisation time** (`ScalarIr`), bound at execution as `ADD_RELU` binds its zero.
- [x] **The decision (2026-10-16): A, burn-fusion fusers, as the end state.**
  - **Why:** fusers serve any model that runs on TT through Burn, which is Burn's real strength. A model-level fused
    API serves only tt-models.
  - **Meanwhile:** the individual ops are made as fast as they go (S1d, S4, S5). They are what every fuser falls back
    to, and what a model not yet fused runs on.
  - **What a fuser must be, to be reliable:**
    - bit for bit the composition it replaces, or a bound derived beside its gate;
    - gated with a negative control;
    - covered by `cargo xtask conformance --fusion`.
  - **Order** (each a [kernel-fusion.md](kernel-fusion.md) item):
    1. Recognise Burn's generic sigmoid and run the native one. This brings decode on `Tt` to `TtBackend`'s time and
       bits.
    2. SiLU, `silu(gate) up` as one SFPU body (S2b, B13b).
    3. RMSNorm as one reduce-map (S3, B13d).
  - **Decode moves to `Tt`** in the bench and tt-models once (1) puts it at parity, so that each later fuser
    shows in the step. **Done (2026-10-16):**
    - (1) holds: step193 asserts a decode step's logits on `Tt` equal `TtBackend`'s.
    - `TracedGenerator` defaults to `Tt`; the bench runs `Tt`, and `TT_GEN_FUSION=0` gives `TtBackend` ("on
      TtBackend" rows).

### S3. RMSNorm as one reduce-map (~5.2 ms -> target under 1)

- [x] **Two ops a norm, up to its scale** (2026-10-16, `tt_kernels::sfpu::norm`, `Session::rms`, the chain
  fuser's `RMS_NORM`): the statistic in one kernel, then the existing `DIV`.
  - **R0, observed on `Tt` (`BURN_FUSION_LOG=full`):** F32 has no casts (`Tensor::cast` skips a same-dtype cast),
    and `square` is `Mul(x, x)`. `gamma`'s `Reshape` is a block of its own, and the product by it joins neither
    block, so `* gamma` stays one SFPU op (~0.44 ms a 1.7B step).
  - **The kernel:** a reduction's kernel (`reduce::roles`), one job a tile row.
    - It squares each tile as it accumulates it, in `sum_dim`'s order, then `reduce::finish` folds.
    - The statistic: `MUL_SCALAR`, `ADD_SCALAR` and `SQRT`'s bodies, lanewise, before the pack.
    - The output is the `[rows, 1]` tensor those ops leave. `DIV` reads its column 0 through the mover's broadcast,
      as before.
  - **A first version also divided, and lost** (silicon, run `1792184225`):
    - It unpacked the row a second time and ran `DIV`'s program per tile: 640 us a 2048-wide norm, against 106 us
      for the five ops.
    - The 1.7B step went 45.2 -> 71.5 ms. `DIV`'s body does not fit the replay buffer, so each instruction is
      pushed from the role program, and one unit pushed all 64 tiles'.
    - It also needed column 0's value copied to every lane by bits, since the fold leaves each lane its own rounding.
    - The lesson: fuse the sequential part (the sum) and leave a parallel op parallel.
  - **Found on the way:** a residual `Add` keeps `ElementWiseFuser` open, so the norm's `Mul(x, x)` joined its
    block, and only 6 of 30 norms fused. A chain may now begin one op into a block, if that op is an `Add` (or
    relu's compare) that `ElementWiseFuser` stays open on. A first version passed over any op, and `add_relu` stopped
    fusing in MNIST (step12): the matmul's block grew over the bias `Add` that must begin one.
- [x] **Gate:**
  - Unit `sfpu::norm`: the statistic equals the composed ops' on the specification model, at widths 37-4096,
    `eps` 1e-5 and 1e-6, and special values. The control is the sum reversed. The longest row's programs fit a slot.
  - `step195`: bit for bit `TtBackend` for decode `[1, 1, 2048]` and `[1, 1, 576]`, ragged, several tile rows and
    Burn's `RmsNorm` module; no `mean_dim`, `sqrt` or `float_div` call (the division is the fused op's own). Near misses run as themselves: another `eps` (bound), a mean over another
    dimension, a kept `x * x`.
  - `step193`: every norm of a decode step fused; the logits are `TtBackend`'s.
- [x] **Measured** (card 1, with S2b and the sigmoid fuser):
  - 1.7B step **45.23 / 47.71 / 50.78 -> 43.42 / 45.85 / 49.00 ms** (`1792186180`).
  - 135M **22.46-23.63 -> 20.20-21.34** (`1792186583`; burn-flex 26.9-31.6).
- [x] **The statistic over only the row groups a tile row's rows occupy** (2026-10-16). A decode row is tensor row 0
  of each tile, 4 of the 32 `Dst` loads a tile (`sfpu::reduce::row_offsets`). `accumulate_squares`, `finish_rows` and
  `norm::statistic` take the tile row's row count, and `sfpu_rms` gives a ragged last tile row programs of its own.
  No value moves between rows, so the computed rows keep their bits; the output's other rows were already
  `Pad::Undefined`.
  - **P0, the profile first** (`1792188326`, `Step(64)`, 1.7B, a norm's two ops): the statistic 35.9 us, **kernels
    14.95**, data 19.16; the `DIV` 30.9 us on 64 units (kernels 8.90, data 15.67).
  - **Measured** (card 1, burn-flex in the same run): kernels 14.95 -> **5.58 us**, the statistic 35.9 -> 26.5 us
    (profile `1792190118`). 1.7B step **43.42 / 45.85 / 49.00 -> 42.97 / 45.41 / 48.59 ms** (`1792189445`, burn-flex
    303.4 / 311.2 / 329.3); 135M **20.20 / 20.66 / 21.34 -> 19.94 / 20.42 / 21.11** (`1792189939`, burn-flex 28.0 /
    29.9 / 32.6).
  - **Tried and reverted: reading only the rows used.** `READ_RUN` with a row count (one `1024 + 64 r`-byte read a
    tile, as a thin matmul `GATHER` reads `A`) made the data phase **slower, 19.15 -> 21.01 us**, with ~74% fewer
    bytes, and the step no faster (1.7B 42.99 / 45.40 / 48.57, `1792191435`; 135M 20.06-21.24, `1792191823`;
    profile `1792192008`). A row's 64 reads from one unit cost per read, not per byte. It passed every gate (mover
    gates in both read modes, `--smoke` 1,104 of 1,104, `1792192846`) and was not kept.
  - **Gates:** unit `sfpu::norm` (rows 1-49 across group and face boundaries, NaN padding; control
    `a_row_group_too_few_differs`); `step196_row_limited_rms` (ttsim, card 0 `1792194490`; L1 and `Dst` primed with
    NaN; watched failing with the last tile row programmed 4 rows short); `step195`, `step193`, `step187` (ttsim)
    and `step195`, `step193` on card 0; Burn conformance (`norm`, `rms`) plain and `--fusion`.
  - **What is left of a norm:** the statistic's read (19 us, one unit, 64 entries) and the `DIV` (31 us). The same
    row limit applies to every SFPU element-wise op on a `[1, W]` decode activation (`DIV` here, 8.9 us of kernel on
    each of 64 units, `SILU_MUL`, `sigmoid`, the residual adds), whose decode row is one of the eight row groups their
    kernels run (what that saves there is unmeasured).

### S4. Attention's non-product ops (~4.5 ms)

Superseded by [decode-attention.md](decode-attention.md) (2026-10-17), which takes both items below.

- [ ] The `qk`/`pv` row gathers (1.12 + 1.13 ms) gather the operands into tile rows every call
  ([attention-jobs.md](../completed-plans/attention-jobs.md) P1-P2). For a decode step's single query, check whether
  the cache's own layout can be read in place.
- [ ] The softmax (1.96 ms): attention-jobs P4 (tabled there: "the softmax's clamps and a fused softmax"). Re-measure
  first; it is now ~4% of the step.

### S5. Each op's fixed cost

- [ ] What is left per op once S1-S4 shrink the count: ~6 us for an element-wise op on 64 units (barrier, control,
  one tile in and out). roadmap Track 2 item 6, step 6's "P1, barriers on the units an op ran on" and "each op's
  ~3.5 us of control" are the items; re-measure with the profile mode after S1-S4.

## Measurement and documentation

Each phase: `silicon_bench_models::traced_generation` for 1.7B and 135M on card 1, every tile, burn-flex in the same
run, then `--profile`. Baselines: 1.7B 47.90 / 50.37 / 53.53 ms (`1792161623`), 135M 22.91 / 23.35 / 23.98
(`1792162031`), profile `1792162257`. Rows go into the performance log and scoreboard. On completion: the roadmap,
kernel-fusion.md (if A is chosen, its B13b/B13d checklists), and this file moved to `completed-plans/`.
