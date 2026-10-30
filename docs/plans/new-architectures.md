# New architectures: hybrid recurrence, per-layer embeddings, a diffusion transformer

Three checkpoints that break an assumption the Llama-shaped `tt_models::llama` makes, brought up
on burn-flex against their reference implementations first (phase A), then on the card against
burn-flex (phase B). Branch `models-hybrid`, worked beside the F4a lane; nothing here changes
`tt-kernels`, `tt-firmware`, `tt-isa` or `burn-tt`.

| Model | Code | What is new |
|---|---|---|
| `Qwen/Qwen3.5-4B` | `tt_models::qwen35` | Gated DeltaNet layers: a recurrent state per head instead of a cache that grows, a depthwise causal convolution with carried inputs, softplus/exp decay, L2-normed q/k, a SiLU-gated RMSNorm; every fourth layer full attention with an output gate, q/k norms, a partial rotary embedding; zero-centred norms (`1 + w`). |
| `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` | `tt_models::gemma4` | Gemma 4 E4B: per-layer embeddings (a 262,144 x 42 x 256 table) gated into every layer, sliding-window and full attention 5:1 with different head widths and rotary embeddings ("proportional" on the full ones), the last 18 layers sharing earlier layers' keys and values, norms around every sub-block, logits soft-capped. |
| `OpenMOSS-Team/MOSS-Image-Base` | `tt_models::moss_image` | Text to image: the Qwen3.5 stack at width 4096 as the encoder (`qwen35::MOSS_TEXT_ENCODER`, headless), a 42-layer single-stream DiT (shared adaLN modulation, a four-axis rotary embedding, the bounded "situ" activation `4 tanh(g/4) sigmoid(g) 25 tanh(h/25)`, non-causal attention), a flow-matching Euler schedule with guidance 3 whose negative branch skips block 22, the FLUX.2 VAE decoder. |

## Phase A: on the host, against the reference (done 2026-10-30)

The pattern is `llama_1_7b_host.rs`: a pinned `uv` reference script in
`crates/tt-models/reference/` prints the checkpoint's shape and fixed-input results; the host test
loads the pinned weights strictly and must reproduce them. Each guard is far looser than what is
measured (printed by the tests); it catches a wrong layer, not an error bound.

- [x] Weights pinned in `PINS.toml`/`xtask/src/pin.rs`, `on_request` (fetched only by
  `--model`, so a bare `fetch-weights` stays 3.8 GB); sharded checkpoints load through
  `load::load_safetensors_shards`, which checks the shards together.
- [x] **Qwen3.5-4B** (`tests/qwen35_host.rs`): the top-10 on a raw and a chat prompt in order,
  logits within 2.6e-5 and 2.7e-5 of `transformers` 5.19.0, and its 24 greedy tokens through the
  carried state. The recurrence runs a token at a time; the reference sums a prompt in chunks of 64
  (WY form), the same function in another order.
- [x] **Nemotron-Bolt** (`tests/gemma4_host.rs`): the top-10 on a raw prompt, a 704-token prompt
  past the 512-position window, and a chat prompt, logits within 4.2e-5, 1.3e-4 and 3.0e-5, and the
  greedy answer ("The capital of France is **Paris**.") to its end token. The checkpoint's
  `model-frozen-shared-kv.safetensors` holds only the sharing layers' unused key/value weights
  and is not fetched.
- [x] **MOSS-Image** (`tests/moss_image_host.rs`, against the checkpoint's own
  `pipeline_moss_image.py` on `diffusers` 0.38.0): the prompt's and the negative prompt's
  encodings, one DiT step on a synthetic latent and context with and without block 22, a 4-step
  guided denoising of synthetic noise, and the VAE's images of a synthetic latent and of the
  denoised one, each within 2.4e-5 of its RMS at the largest of its mean, RMS and sixteen samples;
  the Euler schedule bit for bit. Found on the way: torch 2.14's CPU `linspace` computes each
  point as one fused multiply-add (`moss_image::linspace` does the same; a separate product and sum
  differ in the last bit at 239 of 1,000 points, which moved one snapped sigma).

Not covered by a host test: the BF16 paths (the references are F32), a sliding window on a prompt
longer than the window in decode steps (the cache is a growing one masked by position, the
704-token check is a prompt), and the vision, audio and multi-token-prediction towers (text only;
their tensors are left in the files by name).

## Roofline on one card (2026-10-30)

What no implementation on one Blackhole card can beat, from the models' shapes
(`crates/tt-models/reference/roofline.py`, standard library only; it prints every number here). The card is the
scoreboard's: GDDR6 at 512 GB/s peak and 488 measured as the read ceiling; the matrix unit's 4,096 FLOP a cycle a
tile a fidelity phase on 120 tiles at 1.35 GHz, so **166 TFLOPS at HiFi4** (what an exact product of F32/TF32 or
BF16 operands takes), 332 at HiFi2 and 664 at LoFi (lossy; prefill-grade fidelity is opt-in only). Derived, not
measured; the best measured kernel is 35.5 TFLOPS with an FP32 `B` and 49.5 with BF16. The script reproduces the
scoreboard's SmolLM2-1.7B roofline (7.05 GB and 13.8 ms a F32 decode step at 512 GB/s) as its calibration.

**Decode, batch 1, is bandwidth-bound for all three, by two orders of magnitude.** A step's intensity is 0.5-3.5
FLOP a byte against the card's ridge of ~340 (166 TFLOPS / 488 GB/s); the compute is 0.05-0.06 ms of a 5-38 ms step.

| Model | Weights | GB a step (512 / 8,192 positions) | ms at 488 GB/s (512 / 8,192) | tokens/s at 488 |
|---|---|--:|--:|--:|
| SmolLM2-1.7B (calibration) | F32 | 7.05 / - | 14.44 / - | 69 |
| Qwen3.5-4B | F32 | 16.96 / 17.46 | 34.75 / 35.78 | 29-28 |
| Qwen3.5-4B | BF16 | 8.55 / 9.05 | 17.52 / 18.55 | 57-54 |
| Qwen3.5-4B | BFP8 | 4.61 / 5.11 | 9.44 / 10.47 | 106-96 |
| Qwen3.5-4B | BFP4 | 2.50 / 3.01 | 5.13 / 6.16 | 195-162 |
| Nemotron-3.5-Bolt-E4B | F32 | 18.63 / 18.89 | 38.18 / 38.70 | 26 |
| Nemotron-3.5-Bolt-E4B | BF16 | 9.35 / 9.60 | 19.15 / 19.67 | 52-51 |
| Nemotron-3.5-Bolt-E4B | BFP8 | 4.99 / 5.24 | 10.23 / 10.75 | 98-93 |
| Nemotron-3.5-Bolt-E4B | BFP4 | 2.67 / 2.92 | 5.47 / 5.99 | 183-167 |

(The cache is F32 as the models run today, the embedding tables' rows BF16. The project's decode target is 90-95%
of the bandwidth, `bandwidth-target`: these divided by ~0.93.)

What the shapes say about where the bytes go:
- **The tied heads are 14-15% of every step**: 0.64 B parameters of Qwen3.5's 4.21 B, 0.67 B of Nemotron's 4.64 B
  (the 248K- and 262K-token vocabularies). A head stored apart from the table, in a narrower format, is worth as much
  as the same change to a sixth of the layers. Nemotron today widens its BF16 table to F32 for the head every step
  (`Gemma4::forward`): 1.34 GB read, 2.68 GB written and read again, ~6.7 GB and ~14 ms, most of a BF16 step's
  whole budget.
- **Context barely matters**: Qwen3.5 keeps keys and values in 8 of 32 layers (4 heads), Nemotron in 4 full layers of
  42 (the sliding ones capped at 512 positions, the shared ones none). At 8,192 positions the cache is 0.54 and
  0.31 GB, 6% and 3% of a BF16 step; the SmolLM2-1.7B's would be 3.2 GB.
- **Qwen3.5's recurrent state is cheap in bytes and expensive in ops**: 105 MB read and written a step (0.2 ms), but
  its decay and correction are element-wise passes over 12.6 M F32 values a step, which run on the SFPU, not the
  matrix unit, and today as ~13 separate ops a layer (312 a token). At the 1.7B's measured ~10.5 us a small op
  (283 ops in 2.98 ms, scoreboard), that is **~3.3 ms a token of fixed cost outside this roofline**, a fifth of a
  BF16 step. The fused one-op delta-rule step is the lever, as RMSNorm and RoPE were for the Llama.
- **Nemotron's 2.8 B-entry per-layer table costs nothing a step** (one 21.5 KB row a token), but it is 5.6 GB of
  GDDR in BF16, and its lookup is the BF16 row select that still stages (roadmap Track 2 item 5).

**Prefill is compute-bound** past a few hundred tokens (reading the BF16 weights once is 17-19 ms):

| Model | Tokens | TFLOP | ms at 166 TFLOPS | ms at the measured BF16 kernel (49.5) |
|---|--:|--:|--:|--:|
| Qwen3.5-4B | 512 / 2,048 / 8,192 | 3.7 / 15.1 / 63.5 | 22 / 91 / 383 | 75 / 304 / 1,283 |
| Nemotron-3.5-Bolt-E4B | 512 / 2,048 / 8,192 | 4.1 / 16.8 / 70.1 | 25 / 101 / 422 | 83 / 339 / 1,416 |

Qwen3.5's figure assumes the DeltaNet's prefill is matrix-shaped. Run a token at a time, as `qwen35` does today, it
is sequential: ~3.3 ms a token of small ops (above), **~6.7 s for 2,048 tokens against a 91 ms roofline**. The
chunked WY form (`torch_chunk_gated_delta_rule`: chunks of 64, a unit-lower-triangular solve a chunk, then
products) is what makes the prefill a matrix workload; it is the one change here that moves a bound, not a
constant.

**MOSS-Image is compute-bound from 512 x 512 up**, the only one of the three that is:

| Resolution | Image tokens | TFLOP a DiT forward (attention's share) | ms a forward at 166 TFLOPS | s an image, 50 steps x 2 (CFG) | at the measured BF16 kernel | at LoFi (lossy) | VAE decode, ms of compute / of activation traffic |
|---|--:|--:|--:|--:|--:|--:|--:|
| 256 x 256 | 256 | 2.6 (1%) | 16 | 2.1 (weights-bound: 21 ms a forward) | 5.3 | 2.1 | 2 / 6 |
| 512 x 512 | 1,024 | 10.5 (5%) | 63 | 6.3 | 21.2 | 2.1 | 9 / 22 |
| 1024 x 1024 | 4,096 | 47.9 (18%) | 288 | 28.8 | 96.7 | 7.2 | 36 / 92 |

- The DiT's 5.08 B parameters are 10.2 GB in BF16, 21 ms to read; a forward at 1024 x 1024 is 288 ms of HiFi4
  compute, an intensity of ~4,700 FLOP a byte. **At the default 1024 x 1024 an image cannot take less than 28.8 s
  on one card at exact fidelity** (96.7 s at today's best BF16 kernel, 7.2 s if LoFi were acceptable).
- The two guidance forwards share their weights: run as a batch of two they read them once, which takes the
  weight-bound 256 x 256 case from 2.1 to ~1.6 s (then compute-bound, 32 ms a step) and changes nothing where compute
  binds.
- Attention is 18% of the work at 1024 x 1024 (4,113 tokens attend to all others, non-causal); a fused attention
  kernel matters here, unlike the decoders.
- The text encoder runs once a prompt: 6.92 B parameters, 13.8 GB of BF16 read for the prompt and the negative
  prompt, ~28 ms. The VAE decode is bound by its F32 activations' traffic (~92 ms at 1024 x 1024 if each op reads
  and writes once), not its 6 TFLOP: fusing its norms and SiLUs into the convolutions is the lever there.
- **Memory**: the encoder (14.8 GiB) and the DiT (9.5 GiB) fit on one 32 GiB card together in BF16; in F32 (27.7 and
  18.9 GiB) they do not, which is why the host test and the gate run them in turn.

What this roofline leaves out: per-op fixed cost (the reason the 1.7B is at 11.2 ms against its 13.8 ms F32 / ~7 ms
BF16 bound), the SFPU's own ceiling for the element-wise work, the cache's writes, and two cards (decode and the DiT
split over both would see twice the bandwidth and compute, less the Ethernet transfers).

## Phase B: on the card (started 2026-10-30, cards 0 and 1)

Results so far (logs under `target/silicon/` of the `models-hybrid` worktree):

- [x] **`step260_qwen35`, F32** (card 0, `1793320075`): passes. Native (`assert_native_model`: nothing staged,
  nothing downloaded but the logits), the teacher-forced top-1 equal at all 8 positions, the free-running 8 tokens
  identical to Flex's (and the first 8 of `transformers`' continuation); largest logit difference 8.6e-2 against a
  largest logit of 33.1 (closest top-1 margin 2.46). Eager decode ~4 s a token (~2,800 ops; roofline 35 ms).
- [x] **`step260_qwen35`, BF16 weights stored, F32 arithmetic** (card 0): passes. Native, teacher-forced top-1
  equal at all 8 positions, the free-running 8 tokens identical; largest logit difference 8.9e-2 against 33.1.
  The F32 gate rerun with the same head (card 1, after the head became `table @ last^T`) passes with the same
  largest difference to five digits, 8.8649e-2: consistent with the stored weights giving the F32 model's logits
  bit for bit (as `step240` shows for the Llama), not a check of it.
- [x] **`step261_gemma4`** (card 0): passes, with both tables F32 and every other matrix stored BF16 (F32
  arithmetic on the checkpoint's values). Native, teacher-forced top-1 equal at all 8 positions (closest margin 13.75), the free-running
  8 tokens identical ("The capital of France is **Paris"); largest logit difference 7.9e-2 against 29.9. Eager
  decode ~3 s a token.
- [x] **`step262_moss_image`** (card 0): passes after the upsampling fix (below). Native; the DiT step on the real
  weights through its first 1, 8 and all 42 blocks, and the VAE's image, against Flex:

  | Run | largest difference / RMS | RMS of the difference / RMS |
  |---|--:|--:|
  | 1 block | 4.5e-3 | 1.2e-3 |
  | 8 blocks | 3.1e-3 | 7.3e-4 |
  | 42 blocks | 3.0e-2 | 5.2e-3 |
  | the VAE's 256 x 256 image | 2.2e-2 | 6.8e-3 |

  The error starts at a block's TF32 rounding (~1e-3: the matrix unit computes F32 products in TF32) and grows
  about as the square root of the depth, with no jump at a block, which is what rounding does and a wrong op does
  not. Evidence, not a bound: the gate guards at a quarter of the RMS (a wrong block moves an output by its whole
  size). Open: a derived bound, from a Flex run that rounds the products' operands to TF32 as the unpacker does.

What `burn-tt` did that the host could not show (each a roadmap item; none is changed here):

1. **A BF16 product with a transposed right operand copies it plain first** (`bf16_matmul` calls `plain_dram`; the
   F32 path passes the view's `transposed` flag to `matmul_dram` instead). For Qwen3.5's tied head,
   `last @ table^T` over the 248,320 x 2,560 BF16 table, that copy's host bookkeeping passed 40 GB and was killed
   (`silicon_probe_qwen35_bf16_memory`). Worked around: `qwen35` computes a tied head as `table @ last^T`.
2. **`Precision::Bf16` (BF16 activations) still holds host memory without bound on the whole 32-layer model**: the
   probe was killed at a 16 GB cap during the prompt; one layer, and three or four, stay flat over repeated passes,
   so the cause is not yet located. Not gated; BF16 weights go through `Bf16Stored` storage instead (the shipped
   1.7B configuration).
3. **A BF16 tensor takes an F32 tile's slot in GDDR** (`DramTensor::alloc_elem` allocates `TILE_SLOT`, the FP32 slot,
   for every element type): BF16 halves bandwidth, not capacity. Nemotron's 5.6 GB BF16 per-layer table needs 11.3
   GB of GDDR, and its model did not fit. (`Bf16Stored` storage has compact slots.)
4. **A cast of a freshly loaded tensor uploads the original first**: burn-tt holds a loaded tensor on the host until
   first use; a cast to F32 on the card uploads the BF16 tensor (in an F32 slot), writes the F32 one, frees the
   first, and the holes left the 11.3 GB F32 table no contiguous run (2.9 GB free, largest run 816 MB in the
   refusing channel). Worked around: `gemma4` widens on the data (exact) before anything reaches the card.
5. **The block copier refuses moves that interleave single elements past ~400 K elements** ("descriptor refused
   before submission (code 2)"): a nearest upsampling written as a broadcast of a trailing axis of one
   (`[.., w, 1] -> [.., w, 2]`) works at `[1, 384, 32, 32]` and is refused from `[1, 384, 64, 64]` on
   (`silicon_probe_moss_image_ops`). The refusal is typed but surfaces only at the next download. Worked around:
   `vae::upsample_nearest` repeats whole rows, then the transpose's.

Left:
- **No `SMOKE` entries, by decision (owner, 2026-10-30)**: `step260`-`step262` stay out of `--smoke`, an exception
  to the model-gate rule. Their weights are `on_request` (52 GB), a smoke run on a checkout without them would fail,
  and each takes 5-7 minutes and 26-40 GB of host memory. They are run by name when this work is tested:
  `cargo xtask silicon --release --device 0 --filter step260_qwen35` (and `step261_gemma4`, `step262_moss_image`),
  each inside a memory cap (`systemd-run --user --scope -p MemoryMax=40G`) so a host-memory fault stays in its run.
- **Under F4a and F5** (merged tree `68a0463`, card 0, 2026-10-30): `step262_moss_image` (`1793400458`) and
  `step261_gemma4` (`1793400112`) print the figures they printed before both, digit for digit; `step260_qwen35`
  (`1793399450`, F32 and BF16-stored weights) prints 7.7931e-2 of 33.1234, the figure the F32 tied head's
  `last @ table^T` gives (Qwen3.5 traced, below), not one F4a or F5 moved. `step263_qwen35_traced` passes on card 0
  (`1793400913`).
- [ ] Performance, after correctness: traced generation (a fixed-shape `HybridCache` for Qwen3.5), the fused
  delta-rule step, the DeltaNet's chunked prefill, a window-sized cache for Gemma 4's sliding layers.
  - [x] **Qwen3.5 traced generation** (branch `qwen35-traced`): `qwen35::HybridCache` (the eight attention layers'
    keys and values in a `FixedCache`, each DeltaNet's carried convolution inputs and recurrent state, the rotary
    table) and `FixedShapeDecoder for Qwen35`, so `TracedGenerator` serves it as it serves the Llama. A recurrent
    state cannot be overwritten later as a key can: a padded chunk's rows past the prompt get decay 1 and write
    strength 0 (exact), and the convolution keeps the inputs ending at the prompt's last token, selected by `last`;
    a chunk from position 0 starts from zeros. Inside a chunk the recurrence runs a token at a time, unrolled.
    Gate `step263_qwen35_traced` (card 1, `1793339453`, BF16-stored weights, every tile): traced greedy tokens and
    logits equal the eager growing state bit for bit at prompts of 17, 5, 16 and 61 tokens (chunks of 8 and 16, the
    first two padded; decode across the 64-position step's end), top-1 equal to burn-flex everywhere (largest logit
    difference 7.8e-2 of 33.1); without the padding mask (`qwen35::UnmaskedPadding`) the decode logits move by up
    to 11.9. Its `the_first_layers_trace_as_they_run_eagerly` is the same claim on four layers in ~85 s.
  - [x] **The step's constants made once** (`Qwen35::prepared`, applied on load): `1 + w` of the zero-centred
    norms, `qkv|z|b|a`, `gate|up` and the attention's `q|k|v` as one product each, the taps as rows,
    `-exp(A_log)`; on the card the norms are `burn_tt::rms_norm`, the queries' and keys' L2 norms an RMS norm with
    `eps / k_dim` and a scale row, a one-token convolution its window times the taps summed. The eager model's
    arithmetic changed with it (it shares the code); burn-flex's is unchanged but the decode step's convolution sum.
  - Measured (bench `silicon_bench_models::qwen35_traced_generation`, card 1, `1793341445`, `Tt`, every tile,
    512-position cache, chunks of 16): **68.5 / 68.9 ms a decode step at 32 / 256 tokens** (14.6 tokens/s), against
    eager's ~1.9 s on every tile (~4 s on one) and the 17.5 ms BF16 roofline; 1,009 ops a step, each with a barrier.
    Profile (`1793343574`): the products 19.6 ms of the 41 ms the profile covers (7.4 GB at 379 GB/s), 748 other ops
    21.4 ms; the recurrence ~560 us a DeltaNet layer (~13 ms a step), 200 us of it the decay broadcast to the whole
    state; the head (2.5 GB of F32 table) outside the covered window. With the decay a column of the state
    (`0547fa4`): **64.2 / 64.6 ms** (`1793346925`, 15.6 tokens/s), `step263` passing again (`1793345534`, eager
    ~1.6 s a token on every tile, traced 67.0 ms on `TtBackend`).
  - What it found outside `tt-models`: (6) the argmax of a 248,320-wide row does not fit a unit's L1 arena ("reduce
    inputs (1010880 bytes) does not fit the 917504-byte L1 arena"); `generate::greedy` takes a row past 65,536
    columns in single-row parts (the Llamas' traces unchanged; `step187` and `step188`'s traced test pass on card
    1, `1793344290`, as do `step260`'s BF16-weight test there, largest logit difference 7.79e-2 and eager ~3 s a
    token on one tile, and `qwen35_host` against `transformers`). (7) A tied
    head as `table @ last^T` returns a transposed view, which a trace cannot output: an F32 table's head is now
    `last @ table^T` (the BF16 table keeps the workaround of (1)). (8) **The build is host-bound and
    superlinear in a trace's ops**: 1,487 s for two steps and two chunks (`step263`), 1,108 s for four steps and a
    chunk of 16 (7,056 ops), one core at 100%; the four-layer model builds in seconds.
  - Left, in order of what they buy: the fused delta-rule step (the recurrence, decay and norms of a DeltaNet are
    ~25 of its ~34 ops a step, a new SFPU op by `new-op-definition-of-done.md`); a BF16 copy of the tied head
    (2.5 GB of F32 table read a step, ~5 ms at the bandwidth; 1.3 GB more GDDR); the capture's host cost; the
    DeltaNet's chunked (WY) prefill (a 256-token prompt takes 5.0 s traced).
