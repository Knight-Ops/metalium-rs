# Compact block-float tiles in GDDR

Block-float tiles stored in GDDR without their header and slot padding.

## Layout

| | L1 image | slot today | compact |
|---|---|---|---|
| BFP8 | 16 header + 64 exponents + 1,024 datums = 1,104 | 1,152 | 1,088 (17 x 64) |
| BFP4 | 16 + 64 + 512 = 592 | 640 | 576 (9 x 64) |

- Tile `i` of a channel's extent lies at `base + 16 + i * 1088` (BFP8), so it is always 16 mod 64.
- It is read to `slot + 16` in L1, which is also 16 mod 64, so every transfer stays congruent mod 64.
- The header bytes in L1 are not written. The unpacker skips them (`(1 + DigestSize) * 16` bytes).
- Writes back to GDDR (`store_bfp`, the pack output, a KV row's `SCATTER_ROWS`) use the same offsets.

## Gains (Qwen3.8-27B, gate/up BFP4, the rest BFP8, embedding in host memory)

| | today | compact | saved |
|---|---|---|---|
| weights | 23.14 GB | 21.55 GB | 1.60 GB |
| KV cache, BFP4, 512K positions | 10.74 GB | 9.66 GB | 1.07 GB |

## What changes

- `TensorRef` gets a GDDR stride beside its L1 slot.
- A single helper gives a tile's `(channel, offset, length)`.
- About 100 sites move to that helper:
  - `tt_isa::dm::record`'s expansion, which also runs in the NC firmware;
  - in `tt-kernels`: `matmul`, `bfp`, `residency`, `session`, `attention`;
  - `burn-tt::storage`;
  - the weight cache format.
