# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers==5.19.0",
#     "torch==2.14.1",
#     "safetensors",
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
"""What HuggingFace `transformers` computes for a pinned Gemma 4 checkpoint's language model: the
reference `crates/tt-models/tests/gemma4_host.rs` holds `tt_models::gemma4::Gemma4` to.

Dev-only (nothing builds or ships it). Run with uv, which installs the pinned versions above:

    uv run crates/tt-models/reference/gemma4_reference.py \
        --repo nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 \
        --revision 49843eed8db1f1c616c47af75cfd36e07aec0667 \
        --weights vendor/weights/nemotron-3.5-bolt-e4b/model-00001-of-00001.safetensors \
        --prompt "The capital of France is" \
        --repeat "The quick brown fox jumps over the lazy dog. " 64 \
        --chat "What is the capital of France?" \
        --generate 16

The checkpoint is a vision-audio-language model (`Gemma4ForConditionalGeneration`); for text its
language model is `Gemma4ForCausalLM` on the `text_config`, built here (eager attention, float32,
CPU) from the checkpoint's `model.language_model.*` tensors (renamed `model.*`), copied in one at
a time so the F32 model (about 30 GB) is the peak. The layers that share another's keys and values
have no key/value weights, so `model-frozen-shared-kv.safetensors` is not read. The weights come
from the local file `cargo xtask fetch-weights` verified; `config.json` and the tokenizer come from
the Hub at the same revision. Prints, as Rust:

- the checkpoint's text shape as a `Gemma4Config` literal, to compare with the preset;
- for each prompt, its token ids and the ten largest next-token logits; `--repeat TEXT N` is a
  prompt of TEXT's ids N times (the beginning-of-text token once), longer than the sliding window;
- for the last prompt, the greedy continuation's token ids and text.
"""

import argparse

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

PREFIX = "model.language_model."


def rust_list(values, fmt="{}"):
    return "[" + ", ".join(fmt.format(v) for v in values) + "]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--weights", required=True, help="the pinned model-00001-of-00001.safetensors")
    p.add_argument("--prompt", action="append", default=[], help="raw text")
    p.add_argument("--repeat", nargs=2, action="append", default=[], metavar=("TEXT", "N"))
    p.add_argument("--chat", action="append", default=[], help="a user message, chat-templated")
    p.add_argument("--generate", type=int, default=0, help="greedy tokens after the last prompt")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    config = AutoConfig.from_pretrained(a.repo, revision=a.revision)
    text = config.text_config
    text._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    # The initialiser's values are all overwritten below; skipping it saves minutes.
    from transformers.initialization import no_init_weights
    with no_init_weights():
        model = Gemma4ForCausalLM._from_config(text, dtype=torch.float32)
    model.eval()

    params = dict(model.state_dict())
    loaded, skipped = set(), []
    with safe_open(a.weights, framework="pt") as f:
        for name in f.keys():
            if not name.startswith(PREFIX):
                skipped.append(name)
                continue
            target = "model." + name[len(PREFIX):]
            assert target in params, target
            params[target].copy_(f.get_tensor(name))
            loaded.add(target)
    assert all(("vision" in n or "audio" in n) for n in skipped), [
        n for n in skipped if "vision" not in n and "audio" not in n]
    left = sorted(set(params) - loaded - {"lm_head.weight"})
    assert not left, left
    model.tie_weights()
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    print(f"// transformers {__import__('transformers').__version__}, torch {torch.__version__}, "
          f"CPU float32, eager attention; {a.repo} at {a.revision}; {len(loaded)} tensors")
    rope = text.rope_parameters
    full = [i for i, kind in enumerate(text.layer_types) if kind == "full_attention"]
    every = full[0] + 1
    assert full == list(range(every - 1, text.num_hidden_layers, every)), text.layer_types
    sliding_cfg, full_cfg = text.per_layer_config[0], text.per_layer_config[full[0]]

    def shape(cfg, params):
        head_dim = cfg.head_dim
        rotated = int(params.get("partial_rotary_factor", 1.0) * head_dim // 2)
        return f"""AttentionShape {{
        head_dim: {head_dim},
        kv_heads: {cfg.num_key_value_heads},
        rope_theta: {float(params["rope_theta"])!r},
        rotated: {rotated},
    }}"""

    print(f"""Gemma4Config {{
    hidden: {text.hidden_size},
    heads: {text.num_attention_heads},
    sliding: {shape(sliding_cfg, rope["sliding_attention"])},
    full: {shape(full_cfg, rope["full_attention"])},
    full_every: {every},
    sliding_window: {text.sliding_window},
    kv_shared_layers: {text.num_kv_shared_layers},
    ff: {text.intermediate_size},
    vocab: {text.vocab_size},
    layers: {text.num_hidden_layers},
    per_layer: {text.hidden_size_per_layer_input},
    eps: {text.rms_norm_eps!r},
    logit_cap: Some({float(text.final_logit_softcapping)!r}),
}}""")
    print(f"// rope: {rope!r}; act: {text.hidden_activation}; tied: {text.tie_word_embeddings}; "
          f"k_eq_v: {text.attention_k_eq_v}; double-wide MLP: {text.use_double_wide_mlp}; "
          f"MoE: {text.enable_moe_block}; per-layer vocab: {text.vocab_size_per_layer_input}; "
          f"bidirectional: {text.use_bidirectional_attention}")

    prompts = [("prompt", t, tokenizer(t).input_ids) for t in a.prompt]
    for t, n in a.repeat:
        once = tokenizer(t, add_special_tokens=False).input_ids
        bos = tokenizer("").input_ids
        print(f"\n// repeated: {t!r}: begin {rust_list(bos)}, then {rust_list(once)} x {n}")
        prompts.append(("repeated", f"{t!r} x {n}", bos + once * int(n)))
    prompts += [
        ("chat", t, tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], add_generation_prompt=True, tokenize=True))
        for t in a.chat
    ]
    for kind, t, ids in prompts:
        # A chat template comes back as an encoding (`input_ids` and a mask), raw text as a list.
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        logits = model(torch.tensor([ids])).logits[0, -1]
        top = torch.topk(logits, 10)
        print(f"\n// {kind}: {t!r} ({len(ids)} tokens)")
        if len(ids) <= 64:
            print(f"ids: {rust_list(ids)}")
        print(f"top10: {rust_list(top.indices.tolist())}")
        print(f"logits: {rust_list(top.values.tolist(), '{:.5f}')}")
        last = ids

    if a.generate and prompts:
        out = model.generate(torch.tensor([last]), max_new_tokens=a.generate, do_sample=False)
        new = out[0, len(last):].tolist()
        print(f"\n// greedy, {a.generate} tokens: {tokenizer.decode(new)!r}")
        print(f"continuation: {rust_list(new)}")


if __name__ == "__main__":
    main()
