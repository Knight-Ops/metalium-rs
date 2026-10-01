# tt-layout

Host-side tilization: a strided row-major tensor to the L1 image the unpacker
reads, and back. Shippable, `#![forbid(unsafe_code)]`, depends only on `tt-isa`.

The tile shape is not decided here. `tt_isa::tile::TileDescriptor` decides it, and
this crate walks the tensor in whatever order the descriptor implies. The familiar
32x32 tile of four 16x16 faces is one descriptor, `Layout::tt_metal_32x32`.

## API

| Item | What it is |
|---|---|
| `tilize(&TensorView, &Layout) -> Vec<u8>` | Tensor to a contiguous run of tile images, row-major tile order. |
| `detilize(...)` | The inverse into a `TensorViewMut`; drops padding. |
| `Layout` | A `[batch, rows, cols]` tensor mapped onto a grid of tile images. |
| `TensorView` / `TensorViewMut` | Strided views over host buffers (the shape Burn hands tensors in). |
| `HostDtype` | `F32`, `Bf16`, `F16`. |
| `PadValue` | What ragged edges are padded with (`Zero`). |

Used by `tt_kernels::matmul` (`tilize_f32`) to stage operands, including every
`DramTensor` upload.

## Test

```bash
cargo test -p tt-layout
```

- `tests/roundtrip.rs` (G1): tilize then detilize is the identity over a proptest
  corpus of shapes.
- `tests/placement.rs` (G2): every datum lands at the byte the unpacker would fetch
  it from.

## Gotchas

- No block-float (BFP) encoder exists; only FP32, BF16 and FP16 are converted.
- The device does not read every FP16 bit pattern the way the host writes it (see
  `HostDtype::F16`'s doc).
- `PadValue` has only `Zero`: right for accumulation, wrong for a min-reduction.
