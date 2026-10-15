# Attention in fewer jobs

Roadmap Track 2, item 9 ("Attention"). Written 2026-10-14 against `b812e38`.
Status: **P0, P1 and P2 done on ttsim (2026-10-14); silicon validation and measurement open.**

## Why this item, and what it is not

After [decode-host-cost.md](../completed-plans/decode-host-cost.md), `module::attention` is the largest cost left
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
- [ ] **Next, from those numbers:** ViT's `qk` stacks its products' rows, and `pv` gathers the same scores back
  into whole tile rows. Keeping the scores padded through the scale and softmax would remove both gathers: a padded
  view that element-wise ops and last-dimension reductions accept.

### P3. Causal mask as cached geometry

- [ ] Cache the `[sq, sk]` mask per attachment.
- [ ] Apply it as a row broadcast over the padded layout, not the `expanded` copy.
- [ ] `step83` unchanged.

### P4. Softmax in fewer jobs (decided from P1 to P3's numbers)

- [ ] **First, cheap:** fold the clamps and the scale where the arithmetic is identical.
- [ ] **Then, if the numbers favour it:** one SFPU row program for max, sub, exp, sum, div, in the same order
  (B13d's first piece, [kernel-fusion.md](kernel-fusion.md)). It needs the full new-op definition of done.

### P5. Close-out

- [ ] Re-measure on card 0, decode, ViT and prefill, on one tile and every tile.
- [ ] Update the performance log, the scoreboard, roadmap item 9 and `burn-op-notes.md`.
- [ ] Move this plan to `completed-plans/`.
