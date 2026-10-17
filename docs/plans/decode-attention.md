# Decode attention

Phase 2 of [decode-to-bandwidth.md](decode-to-bandwidth.md). Written 2026-10-17 against `c69b9c1`. Status:
**planned; A0 next.** It follows S4 of [decode-small-ops.md](decode-small-ops.md) (the row gathers and the softmax),
which it replaces.

## Why this item

A 1.7B decode step spends **~5.2 ms on attention at `Step(64)`** and **~10.9 ms at `Step(512)`** (profile
`1792210770`). Its floor is reading the cache once: 24 layers x 2 x 4.2 MB at 512 positions, 0.2 GB, **~0.43 ms at
469 GB/s**. Attention is also the whole context-length slope: `Step(64)` to `Step(512)` adds 5.8 ms to the step.

| `Step(512)`, 24 layers | Ops | ms |
|---|--:|--:|
| `qk`, matmul gather (K at ~29 GB/s on 64 units) | 24 | 3.46 |
| `pv`, row gather list | 48 | 3.07 |
| softmax, SFPU and reduce gathers (7 ops a call) | 168 | 2.43 |
| `pv`, matmul gather | 24 | 0.73 |
| `qk`, row gather list | 24 | 0.59 |
| `qk`, fill-pad list | 24 | 0.39 |
| mask, scale | 48 | 0.44 |

## What the code does (checked 2026-10-17)

- **The entry point.** Burn's `ModuleOps::attention` is overridden (`burn-tt/src/generated/ops.rs` -> `ops.rs`
  `module::attention`). The labels are `report::part` scopes inside it.
- **Decode never takes `padded_attention`.**
  - `FixedWrite` passes a mask and folds GQA into rows: `q` is `[1, kv, groups·t, d]`.
  - `padded_attention` is skipped whenever a mask is passed.
  - Its padding rule also refuses `m` = 3.
- **What a call runs:**
  - `qk`: `float_matmul` -> `batched_matmul` -> `Session::matmul_dram_batched`, stacked layout;
  - scale: `mul_scalar`;
  - mask: `mask_fill`;
  - softmax: `max_dim`, `clamp_min`, `sub`, `exp`, `sum_dim`, `clamp_min`, `div`;
  - `pv`: the same batched matmul as `qk`.
- **Row gathers.** `rows_padded` relays an `A` whose blocks are not tile-aligned (`q`'s and `P`'s few-row blocks)
  into 32-row blocks; tail rows read the kept `zero_row`. An output with `m % 32 != 0` is restacked.
- **Fill-pads.** `meet` calls `fill_pad` on both operands, unconditionally. Unlike `matmul_dram`, the batched path has
  no `matmul_padding_off_k` rule.
- **K and V are read in place** when `block_fits` holds (`tensor.rs`): a head's block starts on a tile and is whole
  tiles, or ends at the tensor's end. Otherwise the whole K and V are row-gathered. Whether a decode attended length
  that is not a multiple of 32 (`kv_cache.rs` `attended`) takes that path is unchecked.

## Phases

Every phase is bit for bit with today's attention, and each head's positions keep their order.

- [ ] **A-layout. The cache's layout, decided first.** Today it is head-major, `[1, kv, L, d]`.
  - **The cost:** a step's key and value repack into it at every write (38 us each, 1.83 ms a 1.7B step;
    in-place-cache-write D5). Each head's new row is a column block of the projection's one row.
  - **The alternative:** a position-major cache, `[L, kv * d]`. The write is then the projection's row as it is (a
    one-index row scatter), and each head's keys are a tile-aligned column block over all positions.
  - **What attention would read:**
    - `qk` as `K_h q_h^T`: `A` is the head's block, untransposed;
    - `pv` with `V_h` as an untransposed block `B`.

  Decide it with A0's numbers: it changes `FixedCache`, the prefill chunks' write and attention's operand forms.
- [ ] **A0. Profile by sub-op** at `Step(64)`, `Step(256)` and `Step(512)` (the profile mode; no new instruments).
  - Why `qk` reads K at ~30 GB/s: `block_fits` at attended lengths on and off multiples of 32, and the 64-unit deal
    over 32 heads.
- [ ] **A1. The batched path's padding:** `matmul_dram_batched` gets `matmul_padding_off_k`.
  - `q` and `P` have padding past `M` only.
  - **Worth:** −0.38 ms.
  - **Gate:** `step24_padding`'s pattern on the batched path, with dirty padding past `M` and past `N`. Watched
    failing with the rule applied along `K`.
- [ ] **A2. No regathering of `q`, `P` or the output.** Either:
  - `padded_attention` takes the caller's mask and a small `m`;
  - or the batched product reads thin `A` rows in place, as the decode projections already do (`thin`), and writes
    thin output rows.

  **Worth:** −1.7 ms at `Step(64)`, −3.7 at `Step(512)`.
- [ ] **A3. K and V read in place at the read ceiling.**
  - The cache keeps tile-aligned per-head blocks at every attended length.
  - `qk` and `pv` are dealt over 120 units by head, with each head's positions on one unit.
  - **Target:** ≥300 GB/s on the cache. **Worth:** −3 ms at 512.
- [ ] **A4. Softmax, scale and mask as one op, in the same order:** the existing device softmax
  (`device_softmax_matrix`), or a reduce-map as RMSNorm's statistic is (S3). Scale and mask fold in where the bits
  allow.
  - **Worth:** −1.5 ms.
  - **Gate:** bit for bit the 7-op composition, with masked rows, `-inf`, and a row of all-equal scores. The
    control: the sum reversed.
- [ ] **A5. Decision (ask the user):** split-position (flash-decode) attention, if A0-A4 leave attention exposed. It
  changes the softmax's and `pv`'s accumulation order, so it needs a derived bound.

## Gates, every phase

- **Burn-flex:** attention against burn-flex at `head_dim` 32, 64 and 128, MHA and GQA, with attended lengths on and
  off multiples of 32.
- **Bit for bit:** `step185`, `step187`, `step191`, `step193`, and `step188` on card 0.
- **Conformance:** `cargo xtask conformance --release` (attention and softmax).
- **Measured:** `traced_generation` 1.7B and 135M on card 1, then `--profile`.
