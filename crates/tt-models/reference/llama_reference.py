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

With `--prompt-set OUT.rs` it writes instead the teacher-forced reference of the performance
contract's model gate (ADR 0006; `crates/tt-tests/tests/step269_performance_contract.rs`) as a Rust
module: for each chat message of `PROMPT_SET` below, the chat-templated prompt's ids, the greedy
continuation (at most `--generate` tokens, ending at the end-of-turn token if it comes first), and
at every position that predicts a continuation token (the prompt's last, then each continuation
token but the last) the five largest logits and their ids, from one forward pass over the whole
sequence:

    uv run crates/tt-models/reference/llama_reference.py \
        --repo HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --revision 31b70e2e869a7173562077fd711b654946d38674 \
        --weights vendor/weights/smollm2-1.7b-instruct/model.safetensors \
        --generate 32 --prompt-set crates/tt-models/src/prompt_sets/smollm2_1_7b.rs
"""

import argparse

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def rust_list(values, fmt="{}"):
    return "[" + ", ".join(fmt.format(v) for v in values) + "]"


# The model gate's prompts: one chat message each, of different kinds (fact, code, arithmetic,
# summary, explanation, translation, logic, advice). Changing them changes the gate's thresholds,
# which are derived from a reference-contract run over exactly these.
PROMPT_SET = [
    "What is the capital of France?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is 17 multiplied by 24? Show your steps.",
    "Summarize the causes of the French Revolution in three sentences.",
    "Explain what a hash table is to a beginner.",
    "Translate 'Good morning, how are you?' into Spanish and German.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? Why?",
    "Give me three tips for writing clear technical documentation.",
]


def rust_f32(value):
    """`value` to five decimals, without the trailing zeros clippy calls excess precision."""
    text = f"{value:.5f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def rust_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_prompt_set(path, model, tokenizer, a):
    """The teacher-forced reference over `PROMPT_SET` as a Rust module (see the module doc)."""
    import transformers

    cases = []
    for text in PROMPT_SET:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        out = model.generate(torch.tensor([ids]), max_new_tokens=a.generate, do_sample=False)
        continuation = out[0, len(ids):].tolist()
        logits = model(torch.tensor([ids + continuation])).logits[0]
        # Position p predicts token p + 1: the prompt's last position predicts the first token.
        rows = logits[len(ids) - 1 : len(ids) - 1 + len(continuation)]
        top = torch.topk(rows, 5, dim=-1)
        cases.append((text, ids, continuation, top.indices.tolist(), top.values.tolist()))
        print(f"// {text!r}: {len(ids)} prompt tokens, {len(continuation)} continuation: "
              f"{tokenizer.decode(continuation)!r}")

    lines = [
        "//! The performance contract's model gate's reference (ADR 0006): what HuggingFace",
        "//! `transformers` computes for SmolLM2-1.7B-Instruct, teacher-forced over a fixed prompt set.",
        "//! **Generated; do not edit.** Written by `crates/tt-models/reference/llama_reference.py",
        "//! --prompt-set` (see its doc for the command):",
        f"//! transformers {transformers.__version__}, torch {torch.__version__}, CPU float32,",
        f"//! {a.repo} at {a.revision}, the pinned `model.safetensors`, greedy continuations of at most",
        f"//! {a.generate} tokens.",
        "",
        "use super::Case;",
        "",
        f"pub const CASES: [Case; {len(cases)}] = [",
    ]
    for text, ids, continuation, top_ids, top_logits in cases:
        lines.append("    Case {")
        lines.append(f"        chat: {rust_str(text)},")
        lines.append(f"        prompt: &{rust_list(ids)},")
        lines.append(f"        continuation: &{rust_list(continuation)},")
        lines.append("        top5: &[")
        for row in top_ids:
            lines.append(f"            {rust_list(row)},")
        lines.append("        ],")
        lines.append("        logits: &[")
        for row in top_logits:
            lines.append("            [" + ", ".join(rust_f32(v) for v in row) + "],")
        lines.append("        ],")
        lines.append("    },")
    lines.append("];")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"// wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--weights", required=True, help="the pinned model.safetensors")
    p.add_argument("--prompt", action="append", default=[], help="raw text")
    p.add_argument("--chat", action="append", default=[], help="a user message, chat-templated")
    p.add_argument("--generate", type=int, default=0, help="greedy tokens after the last prompt")
    p.add_argument("--prompt-set", help="write the model gate's reference to this Rust file")
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

    if a.prompt_set:
        assert a.generate > 0, "--prompt-set needs --generate"
        write_prompt_set(a.prompt_set, model, tokenizer, a)
        return

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
