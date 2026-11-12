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
"""What HuggingFace `transformers` computes for a pinned Qwen3.5 checkpoint's language model: the
reference `crates/tt-models/tests/qwen35_host.rs` holds `tt_models::qwen35::Qwen35` to.

Dev-only (nothing builds or ships it). Run with uv, which installs the pinned versions above:

    uv run crates/tt-models/reference/qwen35_reference.py \
        --repo Qwen/Qwen3.5-4B \
        --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
        --weights vendor/weights/qwen3.5-4b/model.safetensors-00001-of-00002.safetensors \
        --weights vendor/weights/qwen3.5-4b/model.safetensors-00002-of-00002.safetensors \
        --prompt "The capital of France is" \
        --chat "What is the capital of France?" \
        --generate 24

The checkpoint is a vision-language model (`Qwen3_5ForConditionalGeneration`); for text its
positions are the same on the three mRoPE streams, so its language model is
`Qwen3_5ForCausalLM` on the `text_config`, built here with the checkpoint's
`model.language_model.*` tensors (renamed `model.*`) and the head tied to the embedding as the
top-level config says. The vision tower and the multi-token-prediction head are not loaded. The
weights come from the local files `cargo xtask fetch-weights` verified (their SHA-256 pinned in
`PINS.toml`); `config.json` and the tokenizer come from the Hub at the same revision. Prints, as
Rust:

- the checkpoint's text shape as a `Qwen35Config` literal, to compare with the preset;
- for each prompt, its token ids and the ten largest next-token logits (`float32`, CPU), and, with
  `--hidden`, the first four values of the last position's hidden state after every layer;
- for the last prompt, the greedy continuation's token ids and text.
"""

import argparse

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

PREFIX = "model.language_model."


def rust_list(values, fmt="{}"):
    return "[" + ", ".join(fmt.format(v) for v in values) + "]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--weights", action="append", required=True, help="a pinned shard")
    p.add_argument("--prompt", action="append", default=[], help="raw text")
    p.add_argument("--chat", action="append", default=[], help="a user message, chat-templated")
    p.add_argument("--generate", type=int, default=0, help="greedy tokens after the last prompt")
    p.add_argument("--hidden", action="store_true", help="print each layer's last hidden state")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    config = AutoConfig.from_pretrained(a.repo, revision=a.revision)
    text = config.text_config
    text.tie_word_embeddings = config.tie_word_embeddings
    tokenizer = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    model = Qwen3_5ForCausalLM._from_config(text, dtype=torch.float32)

    state = {}
    skipped = []
    for path in a.weights:
        for name, tensor in load_file(path).items():
            if name.startswith(PREFIX):
                state["model." + name[len(PREFIX):]] = tensor
            else:
                skipped.append(name)
    assert all(n.startswith(("model.visual.", "mtp.")) for n in skipped), skipped
    # `load_state_dict` copies each BF16 tensor into the F32 parameter, widening it exactly.
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state
    assert not unexpected, unexpected
    assert missing == (["lm_head.weight"] if text.tie_word_embeddings else []), missing
    model.tie_weights()
    model.eval()

    print(f"// transformers {__import__('transformers').__version__}, torch {torch.__version__}, "
          f"CPU float32; {a.repo} at {a.revision}")
    rope = text.rope_parameters
    full = [i for i, kind in enumerate(text.layer_types) if kind == "full_attention"]
    every = full[0] + 1
    assert full == list(range(every - 1, text.num_hidden_layers, every)), text.layer_types
    print(f"""Qwen35Config {{
    hidden: {text.hidden_size},
    heads: {text.num_attention_heads},
    kv_heads: {text.num_key_value_heads},
    head_dim: {text.head_dim},
    rotary_dim: {int(text.head_dim * rope.get("partial_rotary_factor", 1.0))},
    rope_theta: {float(rope["rope_theta"])!r},
    linear_k_heads: {text.linear_num_key_heads},
    linear_k_dim: {text.linear_key_head_dim},
    linear_v_heads: {text.linear_num_value_heads},
    linear_v_dim: {text.linear_value_head_dim},
    conv_kernel: {text.linear_conv_kernel_dim},
    full_every: {every},
    ff: {text.intermediate_size},
    vocab: {text.vocab_size},
    layers: {text.num_hidden_layers},
    eps: {text.rms_norm_eps!r},
    tied_head: {str(text.tie_word_embeddings).lower()},
}}""")
    print(f"// rope: {rope!r}, act: {text.hidden_act}, attention bias: {text.attention_bias}, "
          f"output gate: {getattr(text, 'attn_output_gate', None)}")

    prompts = [("prompt", t, tokenizer(t).input_ids) for t in a.prompt]
    prompts += [
        ("chat", t, tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], add_generation_prompt=True, tokenize=True))
        for t in a.chat
    ]
    for kind, t, ids in prompts:
        # A chat template comes back as an encoding (`input_ids` and a mask), raw text as a list.
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        out = model(torch.tensor([ids]), output_hidden_states=a.hidden)
        logits = out.logits[0, -1]
        top = torch.topk(logits, 10)
        print(f"\n// {kind}: {t!r} ({len(ids)} tokens)")
        print(f"ids: {rust_list(ids)}")
        print(f"top10: {rust_list(top.indices.tolist())}")
        print(f"logits: {rust_list(top.values.tolist(), '{:.5f}')}")
        if a.hidden:
            for i, h in enumerate(out.hidden_states):
                print(f"hidden[{i}]: {rust_list(h[0, -1, :4].tolist(), '{:.6f}')}")
        last = ids

    if a.generate and prompts:
        out = model.generate(torch.tensor([last]), max_new_tokens=a.generate, do_sample=False)
        new = out[0, len(last):].tolist()
        print(f"\n// greedy, {a.generate} tokens: {tokenizer.decode(new)!r}")
        print(f"continuation: {rust_list(new)}")


if __name__ == "__main__":
    main()
