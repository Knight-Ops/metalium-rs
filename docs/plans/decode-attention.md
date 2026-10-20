# Decode attention

Phase 2 of [decode-to-bandwidth.md](decode-to-bandwidth.md). Written 2026-10-17 against `c69b9c1`. Status:
**A0, A0b, A1, A2a, A4a, A4b and the A-layout done (2026-10-18); A2 and A3 superseded for decode by attention as one
op (2026-10-19, [decode-to-bandwidth.md](decode-to-bandwidth.md) Phase 6, `Session::attention_rows`: a head a unit, no
regathering, bit for bit); A5 open.** It follows S4 of [decode-small-ops.md](decode-small-ops.md) (the row gathers and the softmax),
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

- [x] **A-layout. The cache is position-major** (2026-10-18). It was head-major, `[1, kv, L, d]`.
  - **The cost was:** a step's key and value repacked into it at every write (31 us each, 1.5 ms a 1.7B step;
    in-place-cache-write D5). Each head's new row is a column block of the projection's one row.
  - **Now:** `FixedCache` stores `[1, 1, L, kv * d]` when `head_dim` is a multiple of 32
    (`tt_models::kv_cache::CacheLayout::for_heads`). Otherwise it stays head-major, since a head's block would not
    be whole tiles.
    - **The write:** the projection's row as it is, `new.swap_dims(1, 2).reshape([1, kv * d])`, a one-index row
      scatter.
    - **A chunk:** writes its `[c, kv * d]` rows by position.
  - **What attention reads:** `FixedWrite::attended` gives the same head-major view, `[1, kv, len, d]`.
    - Head `h` is the column block `h d` of the positions' rows, read in place.
    - `qk` stays `q K^T` (`K` through the unpacker's transpose, A0b) and `pv` reads `V`'s block. The tiles are the
      same in the same order.
    - Not `K_h q_h^T`: swapping the operands between `SrcA` and `SrcB` is not known to keep the bits.
  - **Unchanged:** `Llama`, the GPT decoder and Burn's API.
  - **The fallbacks:**
    - A ragged attended length makes its operands plain (a repack). Traced steps attend multiples of 32.
    - The fused `v` sits at a column offset in `qkv`, so its row is copied once (6 us a layer). A row scatter that
      reads values at a column offset would remove it.
  - **Gate:** `step199::a_position_major_cache_writes_the_projection_s_row_as_it_is` (ttsim; card 0
    `1792336651`):
    - **No repack:** a synthetic GQA Llama's decode steps have no `native repack`.
    - **The negative control:** the same model over a head-major cache (`Llama::fixed_cache_in`) repacks (6 in
      its 3 step traces: the value's row in each layer).
    - **Bits:** its generation is the head-major one's, token for token and bit for bit, at step lengths 32, a
      ragged 48 and 128, after a two-chunk prefill.
  - **Regression:**
    - ttsim: `step185`, `step187`, `step193` (with the weights), `step197`, `step199`, the `tt-models` tests;
    - card 0 `1792336651`: `step187`, `step188` (the 1.7B, traced against eager), `step193`, `step199`.
    - `--smoke` on card 0: 1,124 of 1,124 (`1792338013`).
  - **Measured** (card 1):
    - 1.7B 33.02 / 33.52 / 34.42 -> **31.72 / 32.31 / 33.21 ms** (`1792337485`, burn-flex 295.8 / 312.8 / 331.8);
    - 135M 14.04 / 14.45 / 15.02 -> **13.71 / 14.16 / 14.66** (`1792337842`);
    - profile `1792336760` (`Step(64)`): the replay 32.32 -> 30.91 ms, and the cache write's ops 96 -> 72 at 1.94
      -> 0.64 ms.
- [x] **A0. Profile by sub-op** (profile `1792264052`). **Why `qk` reads K at ~30 GB/s:**
  - **Not the deal or `block_fits`:** at `Step(512)` the product is ~11 read entries a unit on 120 units, and
    still 138 us of data.
  - **The cause:** `q @ K^T` reads the cache as a transposed `B`, and the mover transposed each tile one datum at a
    time (`READ_TRANSPOSED`, ~10 us a tile).
- [x] **A0b. The unpacker transposes `B`** (2026-10-17).
  - **The flag:** `record::GATHER_B_UNPACK_TRANSPOSE` has the mover read each of a transposed `B`'s tiles as it
    lies.
  - **The kernel:** `matmul_items_reloading(.., b_transposed)` sets unpacker 0's `Haloize_mode`, which transposes
    within each face, and unpacks `B`'s face `(k, fj)` from the slot's face `(fj, k)`.
  - **When:** the TF32 route, every product path (whole, thin, K-blocked, batched). On by default;
    `Session::set_unpack_transposed_b(false)` keeps the mover's transpose.
  - **Bits:** the unpack converts each datum as it does untransposed, so `SrcA` holds the same values. Measured bit
    for bit on silicon, special values included.
  - **Gate `step201_unpacker_transposed_b`** (ttsim; card 0 `1792271895`, `1792272312`; in `--smoke`). Bit for bit
    against the mover's transpose and against `B^T` uploaded plain:
    - whole, ragged, thin, K-blocked and `A`-transposed products;
    - a batched `q k^T`.
  - **Watched failing:** the faces in the untransposed order; `Haloize_mode` off.
  - **Also checked:** `step121`'s wait planner over the transposed programs; record unit test
    `an_unpacker_transposed_b_is_read_as_it_lies`.
  - **Regression:**
    - ttsim: `step172`/`step173` in the safe, `fast-reads` and `fast-reads` + `encoded-guard` images; `step18`,
      `step183`, `step184`, `step186`, `step187`, `step190`, `step193`, `step198`; `step12_mnist`;
    - firmware clippy in four configurations;
    - Burn conformance: 2,973 passed, 52 as expected;
    - card 0: 45 gates (`1792272312`), the mover gates and `step201` in `fast-reads` (`1792272565`), and
      `--smoke` 1,115 of 1,115 (`1792276969`).
  - **Measured** (card 1):
    - 1.7B 37.69 / 40.20 / 43.41 -> **37.24 / 38.66 / 40.50 ms** (`1792272596`). The context slope from 32 to 256
      tokens fell from 5.7 to 3.3 ms;
    - 135M 17.51 / 17.97 / 18.76 -> **17.10 / 17.52 / 18.25** (`1792273005`);
    - 1.7B prefill's first token 101.7 / 172.5 / 297.3 -> 98.7 / 169.7 / 293.4 ms.
- [x] **A2a. The stacked product's tail rows left unread** (2026-10-17). After A0b, profile `1792278468`:
  - **The cost:** `pv`'s row gather was 1.12 ms a 1.7B step at `Step(64)` and **3.06 ms at `Step(512)`**. That is
    32,768 read entries a step, ~97% of them each head's 31 zero tail rows, read from the kept zero row 64 bytes at
    a time.
  - **Why they need not be zero:** an untransposed left block's tail rows are `M` rows, and they make only output
    rows past each product's `m`, which a stacked output leaves out.
  - **The change:** `rows_padded` leaves them as the staging slot held them (`tensor::GATHER_HOLE`, nothing read)
    in exactly that case. A `K` row (a transposed left block, any right block) and an output kept padded still read
    zero.
  - **Gate:** `step184` now fills the row gather's staging with NaN before every check; every product stays bit for
    bit its block computed alone. Watched failing with the stacking reading one row late (a NaN tail row).
  - **Regression:**
    - ttsim: `step183`, `step187`, `step193`;
    - card 0 (`1792279773`): those, `step199`, `step201`, ViT-Tiny's `step166`;
    - Burn conformance: 2,973 passed, 52 as expected;
    - `step12_mnist`.
  - **Measured** (card 1):
    - 1.7B 37.24 / 38.66 / 40.50 -> **36.89 / 37.30 / 38.36 ms** (`1792279829`). The 32-to-256-token slope is
      1.5 ms (5.6 when this plan began);
    - 135M 17.10 / 17.52 / 18.25 -> **16.05 / 16.48 / 17.29** (`1792280238`, burn-flex 28.0 / 29.2 / 31.2).
- [x] **A1. The batched path's padding** (2026-10-17). `matmul_dram_batched_with` filled both operands' padding
  every time.
  - **The rule now** (`matmul_padding_off_k`'s): an operand needs zero padding only when its stored `K` dimension is
    ragged, and otherwise it is used as it is. Its padding past `M` or `N` reaches only the output's padding,
    declared `Undefined`, or rows a stacked output leaves out.
  - **Gate:** `step184::padding_off_k_is_used_as_it_is`. A `Q` whose padding rows hold a bias queues the clean `Q`'s
    ops and stays bit for bit each product alone. With `K` ragged, the padding is filled and the products still
    match. Watched failing with the rule off (the extra fill).
  - **Regression:** card 0 `1792280747` (`step183`, `step184`, `step187`, `step193`, `step166`, `step188`);
    Burn conformance 2,973 passed, 52 as expected; `step12_mnist`; `--smoke` 1,116 of 1,116 (`1792281614`).
  - **Measured** (card 1):
    - 1.7B 36.89 / 37.30 / 38.36 -> **36.39 / 37.02 / 37.98 ms** (`1792280991`);
    - 135M 16.05 / 16.48 / 17.29 -> **15.33 / 15.79 / 16.47** (`1792281384`, burn-flex 26.9 / 29.3 / 31.5).
- [x] **A2. No regathering of `q`, `P` or the output** (decode: done by attention as one op, 2026-10-19; the
  composition, which a prefill and other shapes still run, keeps its gathers). Either:
  - `padded_attention` takes the caller's mask and a small `m`;
  - or the batched product reads thin `A` rows in place, as the decode projections already do (`thin`), and writes
    thin output rows.

  **Worth:** −1.36 ms at `Step(64)`, −1.56 at `Step(512)`. **Partly taken (2026-10-19):** the stacking gathers and
  `P`'s are dealt by rows over 16 units (22.3 / 5.3 / 22.3 -> 8.0 us each, decode-to-bandwidth Phase 3, ~0.7 ms); the
  regathering itself remains. That is re-measured after A2a (profile `1792298755`):
  the two stacking gathers on 2 units (22 us each a layer), `P`'s gather (5 us) and the context's repack (6-8 us).
- [x] **A3. K and V read in place at the read ceiling** (decode: attention as one op reads each head's keys and values
  in place on its own unit, 2026-10-19; 89 us a layer at 512 positions, card 0).
  - The cache keeps tile-aligned per-head blocks at every attended length.
  - `qk` and `pv` are dealt over 120 units by head, with each head's positions on one unit.
  - **Target:** ≥300 GB/s on the cache. **Worth:** −3 ms at 512.
- [x] **A4a. The softmax in four ops, not seven** (2026-10-18). Each fusion keeps the composition's bits, and
  the order-sensitive sum stays its own reduction.
  - **`kind_sfpu::SUB_EXP`:** `exp(x - max)` in one pass. Every `SFPMAD` result is flushed and the FP32 store is
    exact, so it is the subtract then `EXP`.
  - **`Session::reduce_then`:** a reduction with an exact unary kind (`sfpu::ops::exact_in_place`) applied to its
    result before the pack. Each of the softmax's clamps (`clamp_min(max, -MAX)`, `clamp_min(sum, MIN_POSITIVE)`)
    rides on its reduction. `tensor::reduce_then_fits` keeps a reduction too long for one tile on the two ops.
  - **Where:** both softmaxes in burn-tt's attention (`float::reduce_clamp_min`, `float::float_sub_exp`), and the
    device softmax's numerator.
  - **Gates:**
    - unit `sub_exp_is_sub_then_exp_bit_for_bit` (watched failing with `exp` on the wrong register);
    - `step200::sub_exp_is_sub_then_exp_on_the_card`;
    - `step200::a_reduction_then_a_clamp_is_the_two_ops_on_the_card` (all-`-inf` and all-zero rows, NaN; watched
      failing with the clamp dropped).
  - **Regression:** card 0 `1792287238` (`step183`, `step184`, `step187`, `step193`, `step166`, `step188`,
    `step200`); Burn conformance 2,973 passed, 52 as expected; `step12_mnist`; `--smoke` 1,118 of 1,118
    (`1792288287`).
  - **Measured** (card 1):
    - 1.7B 36.39 / 37.02 / 37.98 -> **36.12 / 36.52 / 37.54 ms** (`1792287482`);
    - 135M 15.33 / 15.79 / 16.47 -> **14.80 / 15.29 / 15.86** (`1792287884`, burn-flex 26.6 / 29.0 / 31.4).
- [x] **A4b. The scale and the mask as one op** (2026-10-18). Both could ride in `SUB_EXP`'s pass only if the
  maximum were taken after them, so the scale and the mask became one op of their own, and the reduction reads it.
  - **The kernel:** `kind_sfpu::SCALE_MASK_FILL`, `mask ? fill : x * scale`: `MUL_SCALAR`'s load, product and FP32
    store, with `MASK_FILL`'s select on the raw mask between them, so bit for bit the two ops. A fill that an FP32 store
    would change (a denormal, a NaN's payload) is refused (`code_for`), since `MASK_FILL` moves it as raw bits. It is
    row-limited, and it takes a row or a column mask.
  - **Where:** burn-tt's attention, when there is no softcap between them, applies its first mask (the caller's, else
    the causal one) with the scale. `padded_attention`'s causal mask does the same. A second mask runs as `MASK_FILL`.
  - **Gates:**
    - unit `scale_mask_fill_is_mul_scalar_then_mask_fill_bit_for_bit`: every broadcast, scales whose products flush or
      overflow, fills of `±inf`, `±0`, `MAX`. Watched failing with the select taken before the scale.
    - the row-limit unit gate, now with this kind.
    - `step200::scale_mask_fill_is_scale_then_mask_on_the_card`: a `[1, attended]` mask over 1, 9 and 32 rows,
      over NaN-primed L1. Watched failing with the select dropped.
    - `step183`: the scale holds the mask's job, and every case is bit for bit one head at a time. Card 0
      `1792292589`, `1792293081`.
  - **Regression:** ttsim `step187`, `step191`, `step193`; Burn conformance 2,973 passed, 52 as expected.
  - **Measured** (card 1, alone): 1.7B 35.64 / 36.22 / 37.20 -> 35.56 / 36.10 / 37.00 ms (`1792293194`).
    With the thin-product deal (decode-to-bandwidth Phase 4), see there.
- [ ] **A5. Decision (ask the user):** split-position (flash-decode) attention, if A0-A4 leave attention exposed. It
  changes the softmax's and `pv`'s accumulation order, so it needs a derived bound.

## Gates, every phase

- **Burn-flex:** attention against burn-flex at `head_dim` 32, 64 and 128, MHA and GQA, with attended lengths on and
  off multiples of 32.
- **Bit for bit:** `step185`, `step187`, `step191`, `step193`, and `step188` on card 0.
- **Conformance:** `cargo xtask conformance --release` (attention and softmax).
- **Measured:** `traced_generation` 1.7B and 135M on card 1, then `--profile`.
