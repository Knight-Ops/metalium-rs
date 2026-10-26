# BF16 weight storage: half the bytes a decode step reads, bit for bit

Status (2026-10-26): **A0-A4 done: built, gated on ttsim and card 0, measured (decode 16.80 -> 12.95 ms), the benches' default; A5 open.** Roadmap Track 1, "The roofline";
[decode-to-bandwidth.md](decode-to-bandwidth.md) Phase 9. Decided by the user on 2026-10-26, once the F32 levers were
spent (a 1.7B step at 16.8 / 18.2 ms, its weights at ~430 GB/s, 92% of the 469 GB/s read ceiling; F32's floor ~14.3 ms).

## Why it is bit for bit

The SmolLM2 checkpoints are BF16, widened to F32 at load (`Precision::F32`). A BF16 value is an FP32 one with its low
sixteen bits clear, and the unpacker's TF32 keeps the top ten mantissa bits, so in `Src` a weight stored BF16 is the same
datum as the same weight stored F32. The Matrix Unit's fidelity phases slice `Src` by bit position, so the products
agree, and nothing else changes: the activation stays F32 (TF32 into `SrcB`), every output's `K` chain keeps its order.
Measured, not assumed (A0): the vendored Blackhole unpacker page is a stub, and the BF16-to-`Src` rule
(`WriteSrcBF16(x) = WriteSrcTF32(x << 3)`) is Wormhole's.

## What was built

- [x] **A0. The probe** (gate `step232_mixed_src_formats`, ttsim and card 0 `1792986764`): `A` FP32 into `SrcB` as TF32
  (code 4), `B` BF16 in L1 moved as BF16 (code 5): **the FP32 product over the same values, bit for bit**, at LoFi,
  HiFi2 and HiFi4, with signed zeros, BF16 subnormals, `bf16::MAX`, infinities and NaN in `B`; the control (weights with
  bits below BF16's) differs. The other way to write it, BF16 in with TF32 out, is undefined by the specification and
  ttsim refuses it (`in_data_format=5 mismatches out_data_format=4`).
- [x] **A1. The mover** (`tt_isa::dm::record::GATHER_B_BF16`, bit 29): a `GATHER`'s `B` read from its BF16 tensor's
  2,112-byte GDDR slots, each whole into the start of the 4,160-byte L1 slot an FP32 `B`'s tile takes; with
  `A_RESIDENT`, `STREAM`, `PREFETCH` and thin rows, and in a `PrefetchCursor`; refused with a transposed `B` and with
  packed BF16. Record test `a_bf16_b_reads_its_own_slots_whole` (watched failing with `B`'s slot stride F32's);
  `step172` and `step173` in the safe image, `fast-reads` and `fast-reads` + `encoded-guard`; firmware clippy plain,
  `encoded-guard`, `profile`.
- [x] **A2. The kernels and the session.**
  - `matmul::SrcFormats` (each operand's L1 format and `OutDataFormat`) through `pair_code` and every product builder;
    `SrcRoute::Tf32WithBf16B`; `matmul_kernel_with`, `matmul_roles_with`.
  - `tensor::Elem::Bf16`: a `DramTensor` of BF16 datums in BF16 slots, which only a product reads, as its `B`; every
    other op refuses it by its elem. `Session::store_bf16` (on the card, ties-even), `upload_bf16_weight` (host bits),
    `widen_bf16`. `Session::matmul_dram` takes a stored `B` on every path: whole `K`, a thin tile row, `K` blocks, a
    packed one-row `A`, a streamed `K`, 2-D blocks with `K` streamed, prefetch and busy prefetch.
  - Gate `step233_bf16_weight_products` (ttsim; card 0): each path with `B` stored BF16 equals `B` in FP32 bit for bit,
    ragged included; a captured chain whose weights idle units read ahead replays as the FP32 chain; the control
    differs; a transposed stored `B` and other routes are refused.
  - **Found and fixed on the way:** `bf16::expand` (BF16 widened to F32) reset only unpacker 0's `X` counter, so after
    any product (whose face selects leave `Z` on a later face) every widening read the wrong datums, on ttsim and
    silicon alike. It now clears the counters (`step233`'s widening check, watched failing before the fix).
- [x] **A3. burn-tt and tt-models.**
  - `burn_tt::storage::StorageFormat::Bf16` (logical F32, BF16 storage): `with_storage` stores on the card (a
    transposed view is made plain first); a product with an F32 `A` reads it directly (`ops::bf16_weight_matmul`:
    `A`'s leading dimensions folded, one `matmul_dram`); every other op widens it, by the storage contract.
  - `tt_models::quant::Format::Bf16Stored`, `Plan::bf16_weights` (every matrix but the embedding table),
    `Llama::untied` (a tied head made its own projection, the table transposed), `Llama::with_bf16_weights`.
    `TracedBackend` now requires `StorageBackend`.
  - Gate `step234_bf16_weight_storage` (card 0 `1792988204`): synthetic Llamas (multi-head, GQA 4:1, untied heads,
    head width 128, ragged widths and vocabularies) and SmolLM2-135M's checkpoint, all 30 layers: **traced, the stored
    model's tokens and logits equal the F32 model's bit for bit**; its matrices are BF16 storage and no op widens one
    (watched failing with the native route disabled); the control (weights not rounded first) differs. Silicon only:
    storing narrows through the packer's BF16 mode, which ttsim refuses.
  - The benches take `TT_WEIGHT_STORAGE=bf16|f32`.
- [x] **A4. Measured, and the benches' default** (card 0, every tile, 512-position cache;
  [log](../learnings/performance-log.md#bf16-weight-storage-half-the-bytes-a-decode-step-reads-2026-10-26-card-0)): a 1.7B
  decode step **16.80 / 17.19 -> 12.95 / 13.35 / 14.13 ms** at 32 / 128 / 256-token prompts (`1792990193`, `1792989603`),
  first token 40.6 -> 37.2 ms at 32. `TT_WEIGHT_STORAGE` defaults to `bf16` in `silicon_bench_models` (`f32` keeps
  them); the library keeps it explicit (`Llama::with_bf16_weights`), since only the caller knows its checkpoint is
  BF16. Prefill products gain less (a pair is bound by `B`'s unpacks, which BF16 halves: ~1.07x at HiFi4,
  `silicon_probe_pair_cost`).
- [x] **A5a. The stream re-tuned** (2026-10-26, card 0,
  [log](../learnings/performance-log.md#re-tuned-for-bf16-weights-2026-10-26-card-0)): a BF16 weight stream on 96 units
  (`THIN_STREAM_UNITS_BF16`; F32 keeps 64) at 3 reads in flight (the high half of `dm::STREAM_CAP`, taken by the mover
  for a `GATHER_B_BF16` record; F32 keeps 2, where 3 is slower). The prefetch units (40) and cap (2) stay. A 1.7B step
  12.89 -> **11.72 ms** at a 32-token prompt.
- [ ] **A5b.** Compact BF16 L1 slots for the rings and the prefetch (twice the window a unit); then the small ops and
  per-op cost, now ~4.4 ms of an 11.7 ms step beside ~7.3 ms of weights at 469 GB/s.

## Limits

- **Exact only for BF16-valued weights.** Storing rounds ties-even (NaNs quieted, BF16 subnormals to signed zero, as
  `Src` flushes them anyway); an F32 checkpoint's weights would change. Apply it to BF16 checkpoints.
- A stored weight is read untransposed: `with_storage` makes a transposed view plain first.
- ttsim cannot store on the card (the BF16 pack), so model-level gates are silicon only; the products are gated on ttsim
  from host-made BF16 tiles (`step233`).
- Not built: BF16 `A` (activations), the BF16 cache, a BF16 embedding lookup (Track 2 item 5: the lookup keeps its F32
  table, so a tied model holds the table in F32 and its head in BF16).
