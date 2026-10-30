# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers==5.19.0",
#     "diffusers==0.38.0",
#     "torch==2.14.1",
#     "safetensors",
#     "accelerate",
#     "huggingface_hub",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""What the MOSS-Image reference pipeline computes for the pinned checkpoint, part by part: the
reference `crates/tt-models/tests/moss_image_host.rs` holds `tt_models::moss_image` to.

Dev-only (nothing builds or ships it). Run with uv, which installs the pinned versions above:

    uv run crates/tt-models/reference/moss_image_reference.py \
        --revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3 \
        --weights vendor/weights/moss-image-base \
        --prompt "A red apple on a wooden table" --steps 4

The model code is the checkpoint's own `transformer/pipeline_moss_image.py` at the pinned revision
(downloaded from the Hub, imported as a module); `diffusers` supplies the FLUX.2 VAE and the Euler
scheduler underneath it, `transformers` the Qwen3.5 encoder. The weights are the local files `cargo
xtask fetch-weights --model moss-image-base` verified. Everything runs in float32 on the CPU, one
part at a time (the encoder is dropped before the transformer loads), each part's tensors copied
in one at a time.

The inputs that are not text are synthetic and computed identically in the Rust test:
`synth(n, a, b, c)[i] = sin(a i) + 0.7 sin(b i + 1.3) + 0.5 cos(c i)` in float64, rounded to
float32. A tensor is printed by its element count, mean, root mean square and sixteen evenly
spaced elements (`stats`).

Prints, as Rust: the encoder's shape as a `Qwen35Config` literal; the prompt's and the empty
negative prompt's chat-templated ids and encodings; one transformer step on a synthetic latent and
context, with and without block 22; the schedule for `--steps`; the latents after denoising
synthetic noise for the prompt over that schedule (guidance 3, block 22 skipped in the negative
branch, as the pipeline's defaults); the VAE's image of a synthetic latent and of those latents.
"""

import argparse
import importlib.util
import math
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.initialization import no_init_weights
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

REPO = "OpenMOSS-Team/MOSS-Image-Base"


def rust_list(values, fmt="{}"):
    return "[" + ", ".join(fmt.format(v) for v in values) + "]"


def synth(n, a, b, c):
    return torch.tensor(
        [math.sin(a * i) + 0.7 * math.sin(b * i + 1.3) + 0.5 * math.cos(c * i) for i in range(n)],
        dtype=torch.float64).to(torch.float32)


def stats(name, t):
    flat = t.detach().reshape(-1).to(torch.float32)
    n = flat.numel()
    picks = [flat[(k * n) // 16 + n // 32].item() for k in range(16)]
    mean = flat.double().mean().item()
    rms = flat.double().square().mean().sqrt().item()
    print(f"{name}: n {n}, mean {mean:.6e}, rms {rms:.6e}")
    print(f"{name} samples: {rust_list(picks, '{:.6e}')}")


def copy_into(module, paths, prefix="", skip=()):
    params = dict(module.state_dict())
    loaded = set()
    for path in paths:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                if not name.startswith(prefix):
                    continue
                target = name[len(prefix):]
                if target in skip:
                    continue
                assert target in params, target
                params[target].copy_(f.get_tensor(name))
                loaded.add(target)
    left = sorted(set(params) - loaded - set(skip))
    assert not left, left
    return len(loaded)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", required=True)
    p.add_argument("--weights", required=True, help="vendor/weights/moss-image-base")
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    w = a.weights
    print(f"// transformers {__import__('transformers').__version__}, diffusers "
          f"{__import__('diffusers').__version__}, torch {torch.__version__}, CPU float32; "
          f"{REPO} at {a.revision}")

    # The pipeline's own module, at the pinned revision.
    source = hf_hub_download(REPO, "transformer/pipeline_moss_image.py", revision=a.revision)
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", source)
    moss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(moss)

    # 1. The text encoder.
    config = AutoConfig.from_pretrained(REPO, subfolder="text_encoder", revision=a.revision)
    text = config.text_config
    rope = text.rope_parameters
    print(f"// encoder: hidden {text.hidden_size}, ff {text.intermediate_size}, layers "
          f"{text.num_hidden_layers}, heads {text.num_attention_heads}/{text.num_key_value_heads} "
          f"of {text.head_dim}, linear {text.linear_num_key_heads}x{text.linear_key_head_dim} / "
          f"{text.linear_num_value_heads}x{text.linear_value_head_dim}, conv "
          f"{text.linear_conv_kernel_dim}, layer types {sorted(set(text.layer_types))} every "
          f"{text.full_attention_interval}, rope {rope!r}, eps {text.rms_norm_eps}, vocab "
          f"{text.vocab_size}, tied {config.tie_word_embeddings}")
    tokenizer = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer", revision=a.revision)
    with no_init_weights():
        encoder = Qwen3_5TextModel._from_config(text, dtype=torch.float32)
    encoder.eval()
    shards = sorted(os.path.join(w, "text_encoder", f) for f in os.listdir(os.path.join(w, "text_encoder")))
    n = copy_into(encoder, shards, prefix="language_model.")
    print(f"// encoder tensors: {n}")
    embeds = {}
    for kind, value in [("prompt", a.prompt), ("negative", "")]:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": value}], tokenize=False, add_generation_prompt=True,
            enable_thinking=True)
        ids = tokenizer(formatted).input_ids
        out = encoder(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
        embeds[kind] = out
        print(f"\n// {kind}: {value!r}, chat-templated ({len(ids)} tokens)")
        print(f"{kind} ids: {rust_list(ids)}")
        stats(f"{kind} encoding", out)
    del encoder

    # 2. The transformer.
    with no_init_weights():
        dit = moss.MOSSImageTransformer().to(torch.float32)
    dit.eval()
    shards = sorted(os.path.join(w, "transformer", f) for f in os.listdir(os.path.join(w, "transformer"))
                    if f.endswith(".safetensors"))
    n = copy_into(dit, shards, skip=("rope.bands",))
    dit.rope.bands = dit.rope._default_bands(torch.device("cpu"))
    print(f"\n// transformer tensors: {n}")
    latent = synth(128 * 16 * 16, 0.91, 2.17, 0.123).reshape(1, 128, 16, 16)
    context = (0.5 * synth(12 * 4096, 0.37, 1.61, 0.071)).reshape(1, 12, 4096)
    for label, skip in [("step", None), ("step without block 22", [22])]:
        v = dit(latent, 750.0, context, skip_layers=skip).sample
        stats(label, v)

    scheduler = moss.MOSSImageScheduler()
    scheduler.set_timesteps(a.steps)
    print(f"\nsigmas: {rust_list(scheduler.sigmas.tolist(), '{!r}')}")
    print(f"timesteps: {rust_list(scheduler.timesteps.tolist(), '{!r}')}")
    latents = synth(128 * 16 * 16, 1.13, 0.59, 0.287).reshape(1, 128, 16, 16)
    positive, negative = embeds["prompt"], embeds["negative"]
    for timestep in scheduler.timesteps:
        cond = dit(latents, timestep, positive).sample.float()
        uncond = dit(latents, timestep, negative, skip_layers=[22]).sample.float()
        prediction = uncond + 3.0 * (cond - uncond)
        latents = scheduler.step(prediction, timestep, latents, return_dict=False)[0]
    stats("denoised", latents)
    del dit

    # 3. The VAE.
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_config(
        AutoencoderKLFlux2.load_config(REPO, subfolder="vae", revision=a.revision)).to(torch.float32)
    vae.eval()
    path = os.path.join(w, "vae", "diffusion_pytorch_model.safetensors")
    with safe_open(path, framework="pt") as f:
        vae.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=True)

    def decode(z):
        # As `MOSSImagePipeline.__call__` un-normalises and unfolds the latent tokens.
        bn, eps = vae.bn, vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1)
        b, c, h, w_ = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w_).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w_)
        return vae.decode(z, return_dict=False)[0].clamp(-1, 1)

    print()
    stats("decoded latent", decode(latent))
    stats("image", decode(latents))


if __name__ == "__main__":
    main()
