# BF16 weight storage: half the bytes a decode step reads, bit for bit

Status (2026-10-26): **A0-A5 done: built, gated on ttsim and card 0, measured (decode 16.80 -> 11.73 ms), the benches' default.** A5b's compact L1 slots are bit for bit and free L1, but do not shorten the step. Roadmap Track 1, "The roofline";
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
- [x] **A5b. Compact BF16 L1 slots: built, bit for bit, the default; the room does not shorten the step**
  (2026-10-26, card 0, [log](../learnings/performance-log.md#compact-bf16-slots-and-where-a-bf16-decode-step-goes-2026-10-26-cards-0-and-1)).
  - **Measured first** (profile `1793023409`): a 1.7B `Step(64)` replay is 11.35 ms, the products 8.37 ms at 426 GB/s
    and 283 other ops 2.98 ms; 1,771 of the step's 3,555 MB of weights are read ahead, one job's `B` a unit (`gate|up`
    32.4 of 69.2 MB a layer). Weights free (`TT_PROBE_STREAM=skip`, `1793024167`) a step is 6.26 / 7.34 ms.
  - **Built:** `record::GATHER_B_BF16` now puts each BF16 tile in a 2,112-byte L1 slot (`b_at + (kk cols + j) 2112`;
    an ABI change of the record, no `mover.rs` edit), and every layout that holds a BF16 `B` steps it by
    `SrcRoute::b_slot`: `plan_layout_route` / `plan_layout_packed_a` (`plan_slots`), `k_block_layout_with`,
    `k_stream_layout`, `block_stream_layout` (the looped kernel steps `B`'s column by a register of its own,
    `B_SLOT_GPR`), the block-stream planner's `fits_arena`, R2's ring and the session's L1 footprint. A K-whole BF16
    product's ring is 270 KB, not 532.
  - **Plan widths** (`matmul::Bf16Plans`, `TT_BF16_PLANS`): `f32` (the default) plans an FP32 `B`'s columns; `ring` as
    many as fit an FP32 ring's bytes (3 at `K` = 2048); `wide` as many as fit (4).
  - **Gates:** record unit `a_bf16_b_reads_its_own_slots_whole` (the compact stride), `step172`/`step173` in the safe,
    `fast-reads` and `fast-reads` + `encoded-guard` images; firmware clippy plain, `encoded-guard`, `profile` and the
    `fast-reads` images; `step233` (ttsim, both 2-D block kernels now) **watched failing** with the record's stride left
    at 4,160; unit `a_bf16_b_plans_compact_slots`; card 0 21 of 21 (`1793029808`: `step234`, `step233`, `step209`,
    `step219`, `step188` traced = eager and = Flex).
  - **Measured** (card 0, stream-ceiling bench, 32 / 256 tokens): the commit before 11.78 / 12.96 ms (`1793028095`);
    compact at FP32 widths **11.68 / 12.90** (`1793027730`; traced generation, tokens checked, 11.73 / 12.23 / 12.95,
    `1793030086`); `ring` 11.81 / 12.99 (`1793029006`); `wide` 13.10 / 14.12 (`1793026815`); compact with the second
    ring (`TT_TRACE_ARENA=1 TT_PREFETCH_SECOND=1`) 11.88 / 13.08 (`1793027910`).
  - **Why the room does not pay:** `wide`'s 4-column rings reach below the tops of the ops between products, so busy
    units stop reading them ahead (`qkv` 26.0 -> 3.2 MB ahead, `gate|up` 32.4 -> 3.2; profile `1793028276`). `ring` reads
    `gate|up` 48.7 MB ahead (was 32.4) and its span stays 125 us, its barrier 10.9 -> 27.9 us (profile `1793029342`): a
    product's span is its units' own column work, not the bytes left to read, as in F32 (decode-to-bandwidth, "A
    product's longest unit"). Compact rings stay for the L1 they free.
  - **`fast-reads`** (card 1, same card; 21 of 21 gates with it, `1793030396`): a 1.7B step 11.74 / 12.26 / 12.98 ->
    **11.54 / 12.04 / 12.72 ms** (`1793031435`, `1793030673`), the 135M unchanged (7.32 / 7.90 / 8.53 -> 7.34 / 7.82 /
    8.44). Profiled, all of it is the products' 1.68 M read entries (8.37 -> 8.13 ms); the small ops are 2.98 -> 3.01 ms.
  - **Next:** the 283 small ops' ~3 ms and the products' barriers (~2.1-2.5 ms a step), not the weight stream.

## Limits

- **Exact only for BF16-valued weights.** Storing rounds ties-even (NaNs quieted, BF16 subnormals to signed zero, as
  `Src` flushes them anyway); an F32 checkpoint's weights would change. Apply it to BF16 checkpoints.
- A stored weight is read untransposed: `with_storage` makes a transposed view plain first.
- ttsim cannot store on the card (the BF16 pack), so model-level gates are silicon only; the products are gated on ttsim
  from host-made BF16 tiles (`step233`).
- Not built: BF16 `A` (activations), the BF16 cache, a BF16 embedding lookup (Track 2 item 5: the lookup keeps its F32
  table, so a tied model holds the table in F32 and its head in BF16).
