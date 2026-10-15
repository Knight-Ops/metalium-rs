---
number: 1
title: "LLM support is generic across architectures, not model-specific"
status: "accepted"
affected_paths:
  - "crates/tt-models/src/generate.rs"
  - "crates/tt-models/src/kv_cache.rs"
  - "crates/tt-models/src/rope.rs"
  - "crates/tt-models/src/llama.rs"
  - "crates/burn-tt/src/trace.rs"
  - "crates/tt-tests/tests/step187_traced_generation.rs"
---

# 1. LLM support is generic across architectures, not model-specific

## Context
SmolLM2-135M was the first language model on the card, and the first traced-generation and decode optimisations were written against it (hard-coded shapes, one model's checkpoint). It is a test vehicle, not a model worth running: its decode step is dominated by fixed per-op costs, and its shapes hide costs that larger models expose (SmolLM2-1.7B's 32 key/value heads made the KV-cache write a quarter of a decode step, invisible at 135M's 3). The owner directed on 2026-10-15 that optimisations be generalised across LLMs, not built for SmolLM2.

## Decision
Machinery for language models is architecture-neutral and lives behind generic names: burn_tt::SharedStateTraces (several traces over one state), tt_models::generate (FixedShapeDecoder, TracedGenerator, generate_fixed, Shapes), tt_models::kv_cache::FixedCache and tt_models::rope::Rope. A concrete architecture implements the decoder trait; Llama is the name of the concrete Llama architecture only, configured by LlamaConfig (SmolLM2-135M and SmolLM2-1.7B-Instruct are presets). Gates check generic code on shapes the reference model lacks (synthetic Llama configs, a GPT-2-style decoder written in the test) as well as on real checkpoints, and performance work is measured on a model worth running (SmolLM2-1.7B-Instruct, then larger).

## Consequences
New model families add a FixedShapeDecoder implementation (or a LlamaConfig preset), not a new driver. An optimisation that only helps one model's shapes is not accepted without a general form (for example grouped-query attention folded into rows rather than a SmolLM2-specific path; decode steps attending to the positions in use for any decoder over FixedCache). step187_traced_generation must keep covering synthetic shapes and a non-Llama decoder. Model-specific constants live only in presets and their reference tests (llama_host.rs, llama_1_7b_host.rs, reference/llama_reference.py).
