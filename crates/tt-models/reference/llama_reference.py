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
"""What HuggingFace `transformers` computes for a pinned Llama-architecture checkpoint: the
reference `crates/tt-models/tests/*_host.rs` hold `tt_models::llama::Llama` to.

Dev-only (nothing builds or ships it). Run with uv, which installs the pinned versions above:

    uv run crates/tt-models/reference/llama_reference.py \
        --repo HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --revision 31b70e2e869a7173562077fd711b654946d38674 \
        --weights vendor/weights/smollm2-1.7b-instruct/model.safetensors \
        --prompt "The capital of France is" \
        --chat "What is the capital of France?" \
        --generate 16

The weights come from the local file `cargo xtask fetch-weights` verified (its SHA-256 pinned in
`PINS.toml`), so the reference runs on the same bytes; `config.json` and the tokenizer come from
the Hub at the same revision. Prints, as Rust:

- the checkpoint's shape as a `LlamaConfig` literal, to compare with the preset;
- for each prompt, its token ids and the ten largest next-token logits (`float32`, CPU);
- for the last prompt, the greedy continuation's token ids and text.
"""

import argparse

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def rust_list(values, fmt="{}"):
    return "[" + ", ".join(fmt.format(v) for v in values) + "]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--weights", required=True, help="the pinned model.safetensors")
    p.add_argument("--prompt", action="append", default=[], help="raw text")
    p.add_argument("--chat", action="append", default=[], help="a user message, chat-templated")
    p.add_argument("--generate", type=int, default=0, help="greedy tokens after the last prompt")
    a = p.parse_args()

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(a.repo, revision=a.revision)
    tokenizer = AutoTokenizer.from_pretrained(a.repo, revision=a.revision)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    state = {k: v.to(torch.float32) for k, v in load_file(a.weights).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = getattr(config, "tie_word_embeddings", False)
    assert not unexpected, unexpected
    assert missing == (["lm_head.weight"] if tied else []), missing
    model.tie_weights()
    model.eval()

    print(f"// transformers {__import__('transformers').__version__}, torch {torch.__version__}, "
          f"CPU float32; {a.repo} at {a.revision}")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    print(f"""LlamaConfig {{
    hidden: {config.hidden_size},
    heads: {config.num_attention_heads},
    kv_heads: {config.num_key_value_heads},
    head_dim: {head_dim},
    ff: {config.intermediate_size},
    vocab: {config.vocab_size},
    layers: {config.num_hidden_layers},
    rope_theta: {float(getattr(config, "rope_theta", None) or config.rope_parameters["rope_theta"])!r},
    eps: {config.rms_norm_eps!r},
    tied_head: {str(tied).lower()},
}}""")
    scaling = getattr(config, "rope_scaling", None)
    bias = getattr(config, "attention_bias", False) or getattr(config, "mlp_bias", False)
    print(f"// rope_scaling: {scaling!r}, attention/mlp bias: {bias}")

    prompts = [("prompt", text, tokenizer(text).input_ids) for text in a.prompt]
    prompts += [
        ("chat", text, tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True))
        for text in a.chat
    ]
    for kind, text, ids in prompts:
        # A chat template comes back as an encoding (`input_ids` and a mask), raw text as a list.
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        logits = model(torch.tensor([ids])).logits[0, -1]
        top = torch.topk(logits, 10)
        print(f"\n// {kind}: {text!r} ({len(ids)} tokens)")
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
