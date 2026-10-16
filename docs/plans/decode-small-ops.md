# Decode small ops

Roadmap Track 1, SmolLM2-1.7B "Next, by that profile". Written 2026-10-16 against `7e9def9`. Status: **planned, not
started.** Companion plans: [decode-matmul-bandwidth.md](decode-matmul-bandwidth.md) (the products, P5 open) and
[in-place-cache-write.md](in-place-cache-write.md) (the cache write's copies).

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

### S1. RoPE without element repacks (C; ~3.7 ms a 1.7B step)

- [ ] **Find what repacks.** `apply` slices `x` into halves, `cat`s them swapped, and computes `x cos + swapped sin`
  (the sign is in `sin`'s table), with `cos` and `sin` `[1, 1, t, d]` broadcast over heads. `x` is a view (`reshape` +
  `swap_dims` of a projection's column block). Read the labels to see which input each repack materialises.
- [ ] **The half swap is tile-aligned for `head_dim` 64 and 128** (halves of 32 and 64 columns are whole tile columns):
  a `cat` of whole-tile column blocks of a view should be a block copy, not an element repack. Fix it in
  `ops/region.rs` / `cat` for any tile-aligned pieces, as a general rule.
- [ ] **The broadcast multiply:** either a row-broadcast over heads that reads the table in place (no repack), or the
  model building `[1, heads, t, d]` tables once a step and sharing them across layers. Prefer the backend fix; it
  helps every model.
- [ ] **Gates:** bit for bit (only data movement changes): `step185`, `step187`, `step191`, plus a `cat`/broadcast gate
  on views with ragged and tile-aligned halves (`step180`'s style).

### S2. SiLU and `x up` (~1.9 ms)

- [ ] **S2a (C): the two block copies.** `sigmoid | block copy list` and `float_mul | block copy list`, 24 each,
  likely materialise the fused `gate|up` product's column views (P3's split) before an element-wise op. Element-wise
  ops run on some views in place (`ops::view_op`); make a tile-aligned column block one of them.
- [ ] **S2b (fusion): `silu(gate) up` as one SFPU body** (`x sigmoid(x) y`, the same instructions as the three ops
  in registers), bit for bit the composition. Through A or B per F0.

### F0. Measure before choosing A or B

- [ ] Run the decode step under `Tt` (`Fusion<TtBackend>`): `TracedGenerator` over a `FixedShapeDecoder<Tt>`.
  - Does it capture?
  - What does burn-fusion add to a replay? (Nothing should remain: a trace replays lowered lists.)
  - What does it add to the capture and the warm-up?
- [ ] Record the answer here and pick A or B for S2b, S3 and S4.

### S3. RMSNorm as one reduce-map (~5.2 ms -> target under 1)

- [ ] **One op a norm:** read the row of 64 tiles once, square and sum in `sum_dim`'s order (bit for bit), then
  `x / sqrt(mean + eps) gamma`, writing one row: Burn's composition (`x.cast(F32).square().mean_dim(..) + eps`,
  `.sqrt()`, `x / rms`, `* gamma`), in the same order. kernel-fusion.md B13d's design (stats in Dst rows or an L1
  circular buffer), F32 first.
- [ ] **Gate:** bit for bit against the composed norm on ragged widths, signed zeros, subnormals, `eps` 1e-5 and 1e-6
  (both SmolLM2 presets), and NaN/inf in one element of a row; the Llama gates (`step185`, `step187`, `step191`) bit
  for bit; negative control: a reduction in another order differs.

### S4. Attention's non-product ops (~4.5 ms)

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
