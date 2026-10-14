# tt-models: the model ladder

Pretrained models, written with Burn's own layers and generic over the backend, run on `burn-tt`
and checked against `burn-flex` on the same weights and the same input. The point is to find what
a model past MNIST needs that the backend does not yet do, and to profile it so the performance
work (`docs/plans/roadmap.md`, Track 2) is ordered by a real workload instead of a guess.

Dev-only (`xtask/src/ship.rs`): `burn-flex` is the host reference. Nothing here ships.

## Fetching what it runs on

```bash
cargo xtask fetch-weights                 # everything below, 340 MB
cargo xtask fetch-weights --model resnet18
```

Every file is pinned in `PINS.toml` (`[weights]`) by Hugging Face commit, length and SHA-256, and
verified before it is left under its final name. `vendor/weights/` is not committed. The tests that
read it are ignored unless `--features weights` (silicon implies it), so a plain
`cargo test --workspace` passes on a checkout that has not fetched them.

| Rung | Model | Source | Ops it exercises |
|---|---|---|---|
| 1 | ResNet-18 | `timm/resnet18.a1_in1k` | conv2d, batch norm (inference), max pool, adaptive average pool, linear |
| 2 | ViT-Tiny/16, 224 | `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k` | patch convolution, layer norm, GELU, `attention`, class token, position embedding |
| 3 | SmolLM2-135M (Llama architecture) | `HuggingFaceTB/SmolLM2-135M` (BF16) | RMSNorm, SiLU, RoPE (`sin`/`cos` on the device), grouped-query attention, `slice_assign` KV cache, embedding of a 49,152-token vocabulary, tied output head |

The test photographs (`vendor/weights/images/`) are five natural images, so a top-1 comparison is
between real classes and not near-tied logits of noise.

## Loading (`load.rs`, B15)

`burn-store` reads the safetensors files. Its own `load_from` is lenient (a tensor the file lacks
keeps its random initialisation; a shape mismatch is skipped), which for a pretrained model means a
network that runs and answers nonsense. `load_safetensors` makes every disagreement an error that
names the tensors: missing parameters, shape or dtype mismatches, and file tensors the model does not
consume (unless the caller lists them: PyTorch's `num_batches_tracked`, the blocks beyond a
truncated depth). Two `burn-store` behaviours it works around, both found by this check:

* it lists a norm layer's `weight` and `bias` as unused although the adapter applied them as
  `gamma` and `beta`;
* its PyTorch adapter renames batch, layer and group norms only, so Burn's `RmsNorm` (`gamma`) needs
  explicit remap rules.

The dtype is the file's, and nothing here casts: a BF16 checkpoint is a BF16 model. The old claim
that this runs "entirely on the host" predates BF16 becoming a general device dtype.

## What the gates claim

All compare `TtBackend` with `Flex` on the same strictly loaded weights, and run the device side
under `assert_native_model` (no staged compute, no host arithmetic, no downloaded operands).

* `step165_resnet18`, `step166_vit_tiny`: top-1 equal to Flex's on a photograph; the error is
  measured and printed, not turned into a tolerance. The host model is first checked against the
  world (`tests/resnet18_host.rs`, `vit_host.rs`: the cat photo is a cat).
* `step167_smollm2`: teacher-forced top-1 agreement at every position (both backends are fed the
  same tokens), and the free-running greedy decode is printed with the position of the first
  divergence. The host model reproduces HuggingFace `transformers` (`tests/llama_host.rs`: the same
  top-10 next tokens on two prompts, logits to 5e-5).
* `step168_smollm2_mixed_precision`: a different number format per matrix. See below.
* `step170_checkpoint_round_trip`: train, save with Burn's `BinFileRecorder`, load into a fresh
  model, continue, and end bit-identical to an uninterrupted run (and not to a model that was not
  loaded).

## Mixed precision (`quant.rs`)

F32, BF16 and F16 (logical dtypes: the matrix is computed in that dtype) and BFP8, BFP4 and BFP2
(block floating point: physical storage of a logical F32 tensor, decoded by the device as it
computes) in one model. A `Plan` names a format for each matrix by a substring of its path, and a
second plan names formats for the activations at `layers.N.attn_ctx` and `layers.N.mlp_hidden`.

`quant::Native` applies a plan on `burn-tt`. `quant::Emulated` applies it on any backend: block
formats are rounded on the host by `bfp_round_trip`, an independent model of the packer and the
unpacker (the oracle of `step92_bfp_formats`, decoded with `tt_isa`'s decoders), and kept as F32.
The same model code runs on both, so the comparison measures the device's arithmetic on identical
quantised values.

## Where it stands

See the roadmap entry for the numbers. The short version: all three models run natively on card 0 and
agree with Flex, and the profile is what the performance work needed: one embedding lookup of a real
vocabulary costs 15 s (`gather_native` stages `32 x rows` host addresses per chunk), a ResNet-18
forward costs seconds, and a decoded token costs about 3,000 device operations.
