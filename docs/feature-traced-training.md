# Generalized Traced Training & Inference Architecture

This document has been expanded into the comprehensive specification for **Generalized Traced Execution Architecture (Inference & Training)**:

👉 **See [`docs/feature-traced-execution.md`](feature-traced-execution.md)**

### Summary of Unified Capabilities:
- **Generalized Traced Inference (`TracedInference<M>`)**:
  - $N$ input tensors of any shape and dtype (`F32`, `BF16`, `I32`, `Bool`).
  - $M$ output tensors (e.g. logits, bounding boxes, KV caches).
  - Read-only parameters.
  - Zero host op construction on replay.
- **Generalized Traced Training (`TracedTrainingStep<M, O>`)**:
  - Full forward pass + loss + backward pass + optimizer step.
  - Generic parameter reflection via Burn's `ModuleVisitor` by `ParamId`.
  - On-device in-place parameter writeback via `copy_into(src, dst)` before `end_trace()`.
  - Stateful (Adam/AdamW) and stateless (SGD) optimizer state writeback.
  - Scalar loss readback after replay without trace abort.
  - Evaluated on base MNIST MLP and TinyTransformer workloads.
