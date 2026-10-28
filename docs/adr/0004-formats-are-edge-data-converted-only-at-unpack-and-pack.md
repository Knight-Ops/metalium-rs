---
number: 4
title: "Formats are edge data, converted only at unpack and pack"
status: "accepted"
affected_paths:
  - "crates/tt-isa/src/tile.rs"
  - "crates/tt-isa/src/dm.rs"
  - "crates/tt-kernels/src/matmul.rs"
  - "crates/tt-kernels/src/bfp.rs"
  - "crates/tt-kernels/src/bf16.rs"
  - "crates/burn-tt/src/storage.rs"
  - "xtask/src/gen_burn.rs"
---

# 4. Formats are edge data, converted only at unpack and pack

## Context
The Matrix Unit computes in three numeric families only (e8: TF32/BF16 Src, FP32/BF16 Dst; e5: FP16; int8/int32), and every L1 storage format is converted into one of them by the unpacker and out of one by the packer (vendor/tt-isa-documentation, Unpackers/FormatConversion.md, Packers/FormatConversion.md, SrcASrcB.md). Yet format facts were spread over seven types in four crates (L1Format, TileImage, Elem, BfpFormat, SrcRoute/SrcFormats, StorageFormat, HostDtype) plus bare slot constants (TILE_SLOT 4160, BF16_TILE_SLOT 2112); matmul had three builders by format and refused mixed BFP formats (bfp.rs); 145 generated Burn wrappers widened BFP inputs to F32. The owner required on 2026-10-27 that optimizations serve F32, F16, BF16 and BFP8/4/2 alike, without a process per format.

## Decision
One TileFormat descriptor in tt-isa carries every format-dependent fact: L1 code, datum bits, exponent block, image and slot bytes, numeric family, legal unpack routes to Src and Dst and pack routes (early and late conversion), the mantissa bits each fidelity slice consumes, and the narrowing semantics (rounding mode, denormal flush, -0, NaN to infinity). A format is an attribute of an edge (an operand, result, channel or buffer), never of an op; op bodies are written against numeric families; conversion happens only at unpack and pack. The compiler writes every unpacker and packer configuration a program relies on and never trusts a claim about the previous format. A BFP exponent block (16 datums) is never split by a view without an explicit widening. Fidelity is derived from the edge formats under the op's numeric contract, not set per format.

## Consequences
Adding a format is adding a descriptor row and a format-matrix gate row; op code changes only for a new numeric family. A transform, cost model entry or job builder that names a specific format is in the wrong layer. The three matmul builders are to be replaced by one parameterized by its edge formats (mixed formats such as BFP8 weights with BF16 activations included), bit-exact against each. Slot sizes come from the descriptor (G9). The widen-compute-narrow wrappers are removed op by op as each op accepts its family's formats directly.
