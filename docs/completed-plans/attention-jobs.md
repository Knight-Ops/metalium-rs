# Attention in fewer jobs

Roadmap Track 2, item 9 ("Attention"). Written 2026-10-14 against `b812e38`.
Status: **P0-P3 and P2b done; P4 and the general padded view tabled with their numbers (2026-10-15).**

## Why this item, and what it is not

After [decode-host-cost.md](decode-host-cost.md), `module::attention` is the largest cost left
in both transformers. On card 0, every tile at `work:64:4`:

- **SmolLM2 decode step:** 1,920 jobs (64 a call), 73 of 125 ms of server-thread time. The step takes 99.6 ms
  against burn-flex's 26.2 (run `1792013124`).
- **ViT-Tiny forward:** 336 jobs, 187 of 270 ms. The forward takes 283.4 ms against burn-flex's 106.7
  (run `1792013104`).

Scope:

- **The model code does not change.** Everything here must help any Burn model that calls `attention` or a
  batched matmul.
- **The arithmetic stays Burn's.** Each product is still `matmul_dram`'s bit for bit. The softmax keeps its
  order: max, clamp, subtract, `EXP`, tree-order sum, clamp, `DIV`. The existing attention gates hold bit for bit
  or within their derived bounds.
- **Not in scope:**
  - **FlashAttention.** It is the end state ([future-fusion-features.md](../proposals/future-fusion-features.md)).
  - **Grouped-query attention without `repeat_kv`.** It needs a model or API change.
  - **A decode trace.** It needs trace segmentation.

## What an attention call does today (P0, measured on ttsim)

`module::attention` (`crates/burn-tt/src/ops.rs`) charges each stage's jobs to a part of its own
(`report::part`, `crates/burn-tt/src/report.rs`). The parts sum to what `attention` alone counted. Gate
`step183_attention_jobs`:

| Shape | qk | scale | mask | softmax | pv | total |
|---|--:|--:|--:|--:|--:|--:|
| decode `[1,9,1,64]` over a cache of 37 | 28 | 1 | 0 | 7 | 28 | **64** |
| ViT `[1,3,197,64]`, heads read from one projection | 10 | 1 | 0 | 7 | 10 | **28** |
| prefill `[1,9,5,64]`, causal | 28 | 1 | 3 | 7 | 28 | **67** |

- **The matmuls are 87% of decode's jobs.**
  - `batched_matmul` refuses a batch whose `m` is not whole tile rows. This is because product `i`'s output
    starts at row `i·m` of one `[batch·m, n]` buffer (`tt_kernels::tensor::matmul_dram_batched_limited`).
  - Decode (m = 1), ViT (m = 197) and prefill (m = t) all fall to `materialized_batched_matmul`.
  - For each head, that path makes a host-built element repack of A, one of B, and a `matmul_dram`, then a
    `gather_rows` stacks the results: 3·heads + 1 jobs.
- **The softmax is 7 jobs:** `max_dim`, `clamp_min`, `sub`, `exp`, `sum_dim`, `clamp_min`, `div`. The `sub` and
  `div` are native column broadcasts.
- **A causal mask is 3 jobs:** a metadata write of a host bit vector, an `expanded` gather copy to the scores'
  shape, and `mask_fill`.

## Phases

### P0. Attribution (done 2026-10-14, ttsim)

- [x] `report::part`, so an op charges its stages' jobs and server time to rows of their own.
- [x] `module::attention` in parts.
- [x] `step183_attention_jobs`: jobs per part for the decode, ViT and prefill shapes (table above). Every job is
  charged to a part.
- [x] Per-part server time from `silicon_bench_models` (the parts show in its tables; card 1, after P1-P2, above).

### P1 and P2. Ragged batched products, and blocks stacked by rows (done 2026-10-14, ttsim)

Built as one mechanism inside `Session::matmul_dram_batched` (`crates/tt-kernels/src/session.rs`), so a
batched product is still **one server job** whatever its shape. Everything downstream stays a plain tensor.
The padded-view alternative was rejected: it would have taught every element-wise op and reduction a
second layout, for about three jobs a call.

- [x] `tensor::matmul_dram_batched_limited` writes product `i` from tile row `i·mt` of a
  `[batch·mt·32, n]` output, so the `m % 32` refusal is gone. That is the stacked `[batch·m, n]` when `m`
  is whole tiles.
- [x] When `m` is not whole tiles, one `gather_rows` stacks the products' rows into `[batch·m, n]`.
- [x] An operand whose blocks do not fit (`tensor::block_fits`) but are each whole rows of it (`Session::rows_padded`):
  - **Which operands:** decode's `q` `[9, 64]`; `K` and `V` `[9·T, 64]` after `repeat_kv`'s `cat`;
    the stacked scores `P`.
  - **What happens:** one `gather_rows` puts it into a block per whole tile rows. Every tail row reads one
    zero row (`Session::metadata`; the XMOV zero fill stops ttsim), so the padding is zero and safe as `K`.
  - **Shared blocks:** blocks at the same place (grouped heads) are gathered once.
  - **How the product reads it:** through `padded_block_ref`, which, unlike `block_ref`, lets a block end
    inside the tensor.
- [x] `Strided::matrix_blocks` (`crates/burn-tt/src/views.rs`) accepts an operand whose blocks all fit, or
  are all whole rows. `batched_matmul` (`ops.rs`) dropped its `m % 32` refusal. The per-head
  `materialized_batched_matmul` is left for a batch broadcast from other than 1, and for packed dtypes.
- [x] **Gate `step184_ragged_batched_matmul`** (ttsim):
  - **Claim:** each product equals `matmul_dram` bit for bit on its block uploaded alone.
  - **Cases:** ViT's heads read in place (m = 197) with the stacked `P`; decode (m = 1) and prefill
    (m = 5) heads stacked by rows over T ∈ {1, 31, 32, 33, 70}, with an infinity starting each `V` head;
    three key heads shared by nine query heads; and a transposed left operand stacked by rows.
  - **Mutations:**
    - Stacking from row `i·m` fails all three tests.
    - A padded block one tile row off fails all three.
    - Tail rows reading the source's next row fail only the transposed-`K` test. With only `V`'s tail
      junk, `P`'s zero `K` padding meets it, and the Matrix Unit gives zero even against an infinity on
      ttsim (not checked on silicon).
- [x] **Gate `step183_attention_jobs`** (ttsim):
  - **Claim:** every call equals the same attention run a head at a time, bit for bit, and `qk` and
    `pv` are one job each.
  - **Server jobs a call:**

    | Shape | before | after |
    |---|--:|--:|
    | decode | 64 | **10** |
    | ViT | 28 | **10** |
    | prefill | 67 | **13** |

  - **What is left:** `scale` 1, `softmax` 7, and in prefill `mask` 3.
  - **Device work inside the two product jobs:** at most two zero rows, two operand gathers, the products
    and one stacking gather.
- [x] **`step60`'s refusal changed.** A `[35, 32]` block that ends inside its tensor and is not whole rows of
  it is still refused ("not whole tiles"). Two such products are no longer refused for sharing a tile row.
- [x] **Regression on ttsim:**
  - Unchanged and passing: `step60`, `step68`, `step83` (mesh included), `step182`, and the burn-tt and
    tt-kernels unit tests.
  - Conformance `--release`: 3,025 tests, 2,973 pass, 52 expected, **0 unexpected** (run `1792018795`).
  - With `--fusion`: 2,970 pass, 55 expected, **0 unexpected** (run `1792019988`).
  - The Burn-level gates that reach matmul or attention, unchanged and passing:
    - step11, 40, 59, 65, 76, 85, 90, 95, 125, 127-129, 144, 146-149, 151;
    - step157-160, 162, 163, 171, 180, 181.
  - The MNIST e2e golden is unchanged.
  - **step155 (`interpolate`) deadlocks intermittently on ttsim, and it predates this work.** It hung in 2 of 4
    runs here, and in 3 of 5 on the unchanged `b812e38`; `interpolate` never calls a matmul.
    - **Symptom:** in the forked child both the test thread and the server thread wait on futexes. That fits a
      lock held by another parent thread at fork time.
    - **Suspect, not confirmed:** libtest's stdout lock, since the child flushes stdout.
    - **Status:** not fixed here; it needs its own item.
- [x] **Silicon on card 0** (`1792019871`): `step183`, `step184`, `step83`, `step60`, `step182`, `step166`, and
  `step167` in F32 (23.1 s, was 33) and in BF16.
- [x] **Both cards** (`1792020158`): the mesh attention, trace and module-reference tests (`step83`, `step147`,
  `step88`).
- [x] **Benches**, card 1, release ([log](../learnings/performance-log.md#attention-products-in-one-job-2026-10-14-cards-0-and-1)):
  - **ViT-Tiny:** every tile 283.4 -> **173.5 ms** (burn-flex 107.2); one tile 663.0 -> 481.9.
  - **Decode step:** every tile 99.6 -> **86.5 ms** (burn-flex 26.5); one tile 449.7, unchanged (it waits on its
    m = 1 matmuls).
  - **Decode attention's server time:** 73 -> 36 ms (`qk` 20.6, `pv` 13.2, `softmax` 2.1).
  - **ViT's:** `pv` 47.2, `qk` 26.5, `softmax` 2.2.
- [x] MNIST e2e golden unchanged (ttsim).
- [x] **Next, from those numbers:** ViT's `qk` stacked its products' rows and `pv` gathered the same scores back
  into whole tile rows. Done attention-local in P2b, below. The general padded view stays tabled.

### P2b. Attention's scores kept a head per whole tile rows (done 2026-10-15, ttsim)

Built inside `module::attention` (`padded_attention`, `crates/burn-tt/src/ops.rs`) rather than as a layout every op
understands. The general version is tabled below.

- [x] **The product calls:**
  - `Session::matmul_dram_batched_with` takes a `BatchedLayout`: `stacked: false` leaves the output a product per
    whole tile rows, and `lhs_padded: true` reads a left operand already laid out that way, with no gather.
  - The engine method is `Engine::matmul_dram_batched_layout`. The capability `supports_batched_layout` is asked once
    an attachment, and is `false` for a mesh.
- [x] **What attention does with them:**
  - `Q K^T` leaves the scores as a plain `[heads·mt·32, sk]` tensor.
  - The scale, the causal mask and the softmax run on it row by row, with the same kinds in the same order.
  - `P V` reads it in place and stacks once.
- [x] **When it is taken:**
  - **Required:** the `Fast` matmul mode, no caller mask or bias, and padding of at most a quarter more rows.
    ViT's 197 tokens qualify (672 rows against 608). A decode's single query (288 against 32) and a five-token
    prefill do not.
  - **The rest** keep the stacked path from P1-P2.
- [x] **`step183`:**
  - **Cases:** a causal ViT-shaped case is added. Each call still equals the same attention a head at a time, bit for
    bit; a single head never pads, so the reference took the stacked layout.
  - **Mutations:** `P V` reading head `i` from row `i·m`, and the padded causal mask laid out at the stacked stride.
    Each fails the gate.

### P3. Causal mask as cached geometry (done 2026-10-15, ttsim)

- [x] **Kept per attachment.** The mask is made once (`nhwc::cached_metadata`, generalized from the convolution's tap
  index to any I32 or Bool tensor).
- [x] **In the scores' own layout.** Each `[sq, sk]` block is repeated for every batch and head, or laid a head per
  whole tile rows for `padded_attention`. So `mask_fill` is one same-shape job: no host bit vector a call, no metadata
  job, no `expanded` copy.
- [x] **Large masks.** Past a million elements only one block is kept, and `mask_fill` broadcasts it as before.
- [x] **Jobs:** prefill `[1,9,5,64]` 13 -> **11** jobs a call; `attention/mask` is 1 on a warm call (`step183`).
- [x] **`step83` unchanged.** Mutations:
  - A mask shifted one column fails `step83`'s analytic cases.
  - The kept mask's blocks after the first left unmasked fails `step183`.

### P4. Softmax in fewer jobs: tabled (2026-10-15)

- **Measured, card 1, every tile:**
  - Decode attention's 36 ms of server time is `qk` 20.6, `pv` 13.2, `softmax` 2.1 (210 jobs at about 10 us) and
    `scale` 0.3 (`1792019882`).
  - ViT's softmax is 2.2 of 76 ms.
  - On one tile the softmax is device time: 40.9 ms of ViT's 473, 84 jobs at 0.49 ms.
- **Folding the two `clamp_min`s:** saves about 60 jobs a decode step (about 0.6 ms on every tile). It needs new SFPU
  kinds, a max-and-clamp reduce output and a divide by a clamped sum, each with the new-op definition of done.
- **A fused softmax row program:** saves up to 6 jobs a call. It is B13d's first piece
  ([kernel-fusion.md](../plans/kernel-fusion.md), "Reduce-Map"), and must reproduce `EXP`, `DIV` and `sfpu_reduce`'s tree-order
  sum bit for bit.
- **Revisit when** the softmax is more than about 30% of attention's time on the configuration that matters: one tile
  with a fixed tile choice, or a longer context where `sk` grows the softmax's tiles.

### Tabled: a padded view every op accepts

The general form of P2b: a `Strided` view whose batch stride is `mt·32` rows, accepted by `covering_view`/`view_op`
(element-wise ops), by last-dimension reductions, and by `matrix_blocks` as a fitting operand. Any batched matmul's
consumer could then skip the stacking gather.

- **Why not now:**
  - Every op on such a view must agree that the per-block tail rows are not elements. That is a second layout for
    element-wise ops, reductions, `expanded` broadcasts, `tile_moves` and reshapes, and padding claims (`Pad`) that
    hold inside a buffer.
  - The one consumer measured to pay, attention's softmax, is covered attention-locally by P2b.
- **Revisit if** another composition between two batched products shows the stacking and padding gathers in its
  server time (a cross-attention, a batched LoRA product).

### P5. Close-out (done 2026-10-15)

- [x] **Re-measured on card 1**, release, every tile and one tile (`1792026124`, `1792026138`):
  - **ViT-Tiny:** 129.1 ms on every tile (burn-flex 109.3) and 368.3 on one tile.
  - **Decode step:** 86.9 ms, unchanged.
  - **Prefill:** counted on ttsim only (`step183`); there is no prefill bench.
- [x] **Silicon, card 0** (`1792025224`): `step183`, `step184`, `step83`, `step60`, `step166`, and `step167` in F32.
- [x] **Conformance on ttsim:** 0 unexpected, plain (`1792025070`) and `--fusion` (`1792025609`).
- [x] **Docs:** the performance log, the scoreboard, roadmap item 9 and `burn-op-notes.md` are updated.
- [x] **Moved** to `completed-plans/`.

**Hand-off.** The decode step's attention is now its two single-query products. Each is one job, but inside it the
operand gathers and zero rows take 0.4-0.7 ms of server time. Cheaper candidates, in order:
- keep the zero row per attachment (`cached_metadata`);
- read `q` (one row a head) without a gather, as a whole-row block of a `[heads, 64]` buffer whose `m` is 1;
- after that, the default tile choice (roadmap, "Per-workload tile choice") on one tile.
