# Generalized Traced Execution Architecture for Burn on Tenstorrent: High-Performance Traced Inference & Training

## 1. Architectural Motivation & Scope

In standard Burn backend execution, every model invocation (inference forward pass or training step) incurs substantial host-side latency:
- Constructing and traversing computational graph nodes and Autodiff tapes.
- Allocating intermediate tensor descriptors on the host heap.
- Dispatching commands across the `burn-tt` server thread boundary.
- Multiple PCIe MMIO round-trips and synchronization barriers per step.

For small networks like MNIST, host overhead accounts for ~0.4 ms of a ~1.0 ms step. Even for larger transformer models, host-side dispatch limits Tensix tile utilization.

Hardware traces (`Session::begin_trace` / `Session::end_trace`, `hardware-coverage.md` X4d) compile the entire sequence of Tensix kernels and data mover transfers into a single GDDR command stream. Once captured, execution is triggered by a single `CALL` entry per tile (`Session::replay`), eliminating all host op construction and dispatch latency.

### The Problem: Current Trace Limitations

Until now, `burn_tt::Trace` (`crates/burn-tt/src/trace.rs`) only supported a restricted form of **inference**:
1. **Strictly 1 Input**: `Trace::capture(input: &TtTensor, ...)` holds only one input buffer.
2. **F32 Only**: Explicitly checks `is_stored_f32()`, rejecting token IDs (`I32`), attention masks (`Bool`), or `BF16`.
3. **Strictly 1 Output**: Only downloads a single F32 output buffer.
4. **No Training Support**: As documented in `hardware-coverage.md:594`:
   > *"A training step is replayable once its weights are updated in place (the optimizer writing the same buffers) and the loss stays on the device; neither holds today (B16, and Burn's tensors are immutable). Until then a training loop traces its forward pass at most."*

This document defines a **fully unified, model-agnostic traced execution architecture** covering both:
- **Generalized Traced Inference (`TracedInference<M>`)**: $N$ inputs, $M$ outputs, arbitrary dtypes (F32, BF16, I32, Bool), read-only parameters.
- **Generalized Traced Training (`TracedTrainingStep<M, O>`)**: Full forward + loss + backward + optimizer update with automated in-place parameter writeback.

---

## 2. Unified System Architecture

```
               ┌────────────────────────────────────────────────────────┐
               │         Core Trace Runtime (tt-kernels & server)       │
               │  - Session::begin_trace / end_trace                    │
               │  - Multi-Buffer Replay (PAIR_CALL)                     │
               │  - Session::write / write_bits (F32, BF16, I32, Bool)   │
               │  - Session::copy_into (on-device buffer updates)       │
               └───────────────────────┬────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
     ┌─────────────────────────────┐       ┌─────────────────────────────┐
     │      TracedInference        │       │     TracedTrainingStep      │
     │  - N inputs (any dtype)     │       │  - Forward + Loss + Bwd     │
     │  - M outputs (any dtype)    │       │  - Optimizer update         │
     │  - Read-only parameters     │       │  - In-place param writeback │
     │  - Streaming outputs        │       │  - Scalar loss readback     │
     └─────────────────────────────┘       └─────────────────────────────┘
```

---

## 3. Generalized Traced Inference (`TracedInference<M>`)

### Design

Inference models take arbitrary input structures and return arbitrary output structures:
- **Vision Models (CNNs / ResNets / ViT)**: `Image [B, C, H, W]` $\rightarrow$ `Logits [B, Classes]`
- **Object Detection**: `Image [B, C, H, W]` $\rightarrow$ `(BoundingBoxes [B, N, 4], Scores [B, N, Classes])`
- **Language Models / Transformers**: `(Tokens [B, S], AttentionMask [B, 1, 1, S])` $\rightarrow$ `(Logits [B, S, V], KeyValueCache)`
- **Multi-Modal Models**: `(Image [B, C, H, W], PromptTokens [B, S])` $\rightarrow$ `Embeddings`

### API Specification

```rust
pub struct TracedInference<M> {
    model: M,
    trace_id: u64,
    input_buffers: Vec<BufferDescriptor>,
    output_buffers: Vec<BufferDescriptor>,
    _held_tensors: Vec<TtTensor>,
}

impl<M: Module<TtBackend>> TracedInference<M> {
    /// Capture any model inference with arbitrary inputs and outputs.
    pub fn capture<In: TraceableInputs, Out: TraceableOutputs, F>(
        model: &M,
        sample_inputs: &In,
        forward_fn: F,
    ) -> Result<(Self, Out::HostData), EngineError>
    where
        F: FnOnce(&M, &In) -> Out;

    /// Replay inference with zero host op construction.
    pub fn run<In: TraceableInputs, Out: TraceableOutputs>(
        &self,
        inputs: &In,
    ) -> Result<Out::HostData, EngineError>;
}
```

### Trace Execution Lifecycle (Inference)
1. **Capture Phase**:
   - `Session::begin_trace()`
   - Run `forward_fn(model, sample_inputs)`: all Tensix kernels, GEMMs, and mover transfers are recorded.
   - `Session::end_trace()`
   - Download the outputs of the sample run and return `(TracedInference, first_outputs)`.
2. **Replay Phase**:
   - Write new input payloads into fixed GDDR input slots using `write` (F32) or `write_bits` (I32/Bool/BF16).
   - Enqueue a single `CALL` entry per tile (`Session::replay`).
   - Download the $M$ output buffers back to host memory.

---

## 4. Generalized Traced Training (`TracedTrainingStep<M, O>`)

### Core Challenges & Generalized Solutions

#### A. Automated In-Place Parameter Writeback (`copy_into`)
- **The Problem**: Burn tensors are functional/immutable. `optim.step(lr, model, grads)` allocates fresh `Param` tensors in new GDDR buffers. If captured statically, subsequent replays would continue reading the *initial* parameter buffers, never accumulating weight updates.
- **The Generalized Solution**:
  Every parameter in Burn implements `Param<Tensor<B, D>>` with a unique `ParamId`.
  Using Burn's generic `ModuleVisitor`:
  1. **Pre-step**: Traverse `model` to collect `(ParamId, OrigBufferId)`.
  2. **Post-step**: Traverse `new_model` returned by `optim.step()` to collect `(ParamId, NewBufferId)`.
  3. Match by `ParamId` and emit an on-device `copy_into(src: NewBufferId, dst: OrigBufferId)` for every parameter.
  4. Because `copy_into` is executed *before* `Session::end_trace()`, it is captured into the trace command stream.
  5. Every hardware replay automatically updates `OrigBufferId` in GDDR at the conclusion of the step.
  6. The user's `model` struct continues pointing to `OrigBufferId`, so after $N$ replay steps, the model holds the fully trained weights in-place.
  - **Generality**: Works for any parameter shape, rank, and module hierarchy without custom code.

#### B. Optimizer State Handling (Adam, AdamW, Momentum SGD)
- **Stateless Optimizers (SGD)**: Only parameter tensors are written back.
- **Stateful Optimizers (Adam, AdamW)**: Maintain first and second moment tensors ($m, v$) keyed by `ParamId`.
  Burn's `OptimizerAdaptor` exposes these state tensors. During capture, updated state tensors are similarly mapped and written back to their persistent GDDR slots via `copy_into`, enabling stateful optimizers to execute entirely in hardware across replays.

#### C. Loss Readback Without Trace Abort
- `Session::begin_trace` rejects downloads during capture (`TraceError::HostTransfer`).
- In traced training, the loss tensor remains resident in GDDR during capture. The loss buffer is registered as the trace's scalar output.
- During capture, `into_scalar()` is avoided until after `end_trace()`.
- On each replay step, `s.replay(id)` runs, and only after NC completion is the 4-byte scalar loss downloaded to the host.

### API Specification

```rust
pub struct TracedTrainingStep<M: AutodiffModule<TtBackend>, O: Optimizer<M, TtBackend>> {
    model: M,
    optim: O,
    trace_id: u64,
    input_buffers: Vec<BufferDescriptor>,
    loss_buffer: BufferId,
    _held_tensors: Vec<TtTensor>,
}

impl<M: AutodiffModule<TtBackend>, O: Optimizer<M, TtBackend>> TracedTrainingStep<M, O> {
    /// Capture a training step for any Burn model, optimizer, batch, and loss function.
    pub fn capture<B: TraceableBatch, F>(
        model: M,
        optim: O,
        lr: f64,
        sample_batch: &B,
        step_fn: F,
    ) -> Result<(Self, f32), EngineError>
    where
        F: FnOnce(&M, &B) -> Tensor<TtBackend, 1>;

    /// Replay the entire training step for a new batch in hardware.
    pub fn step<B: TraceableBatch>(&mut self, batch: &B) -> Result<StepTiming, EngineError>;

    /// Finish training and return the model holding updated weights.
    pub fn finish(self) -> M;
}
```

---

## 5. Dynamic Embedding Lookups in Language Models

In deep learning models with token inputs (Transformers, LLMs):
- Standard Tensix ops (GEMMs, normalizations, activations, softmax, attention, reductions) operate dynamically on matrix values in GDDR.
- In `burn-tt`, generic CrossEntropy already uses `index_mask` and `MASK_WHERE`, which dynamically evaluate changed target indices on-device.
- However, `burn::nn::Embedding` currently uses `float_select` (`gather_rows`) and `float_select_add` (`rows_add`), which bake row indices into mover commands.

### Generalized Solutions:
1. **One-Hot GEMM Path**:
   For vocabulary $V$ and batch tokens $N$, express the lookup as $X_{\text{one\_hot}} \times W_{\text{embed}}$, with backward $X_{\text{one\_hot}}^T \times dY$. Both passes are standard 2D GEMMs that operate dynamically over arbitrary token batches in GDDR without recompiling the trace.
2. **Backbone Tracing**:
   Alternatively, embedding lookup runs in the data loader / pre-step, and the trace captures the TransformerEncoder + Head + Loss + Backward + Optimizer (representing >98% of model FLOPs).

---

## 6. Proposed Code Changes

### Layer 1: Core Runtime (`tt-kernels`)
1. **[`crates/tt-kernels/src/tensor.rs`](file:///mnt/nvme/metalium-rs/crates/tt-kernels/src/tensor.rs)**:
   - Implement `pub fn copy_into(src: &DramTensor, dst: &DramTensor, units: usize) -> Result<Work>`.
     Reuses `dst.tensor_ref()` as the destination `ro`, emitting `record::READ_RUN` from `src` and `record::WRITE_RUN` to `dst`.
2. **[`crates/tt-kernels/src/session.rs`](file:///mnt/nvme/metalium-rs/crates/tt-kernels/src/session.rs)**:
   - Add `pub fn copy_into(&mut self, src: &DramTensor, dst: &DramTensor) -> Result<(), TensorError>`.
   - Add `pub fn write_bits(&mut self, t: &DramTensor, values: &[u32]) -> Result<(), TensorError>`.

### Layer 2: Burn Backend (`burn-tt`)
1. **[`crates/burn-tt/src/server.rs`](file:///mnt/nvme/metalium-rs/crates/burn-tt/src/server.rs)**:
   - Expose `copy_into` and `write_bits` on `Server`.
   - Add unified multi-buffer trace replay: `run_generic_trace`.
2. **[`crates/burn-tt/src/trace.rs`](file:///mnt/nvme/metalium-rs/crates/burn-tt/src/trace.rs)**:
   - Refactor into generic trace traits: `TraceableInputs`, `TraceableOutputs`, `TraceableBatch`.
   - Implement `TracedInference<M>`.
   - Implement `TracedTrainingStep<M, O>`.
3. **[`crates/burn-tt/src/lib.rs`](file:///mnt/nvme/metalium-rs/crates/burn-tt/src/lib.rs)**:
   - Export `TracedInference` and `TracedTrainingStep`.

### Layer 3: Model Workloads (`tt-mnist`)
1. **[`crates/tt-mnist/src/main.rs`](file:///mnt/nvme/metalium-rs/crates/tt-mnist/src/main.rs)**:
   - Add `--train-trace` flag.
   - Implement `train_traced` for MNIST MLP via `TracedTrainingStep`.
2. **[`crates/tt-mnist/src/transformer.rs`](file:///mnt/nvme/metalium-rs/crates/tt-mnist/src/transformer.rs)**:
   - Implement `train_traced` for `TinyTransformer` via `TracedTrainingStep`.

---

## 7. Verification Plan

1. **Unit & Runtime Tests**:
   - `cargo test -p tt-kernels --lib copy_into`
   - `cargo test -p tt-tests --test step40_burn_trace` (multi-input inference and multi-step training).
2. **Model Parity**:
   - `cargo test -p tt-tests --features e2e --test step12_mnist` (confirm golden numerical trajectory).
   - `cargo test -p tt-tests --test step59_burn_transformer`.
3. **Hardware Execution**:
   - `cargo run --release -p tt-mnist -- --steps 50 --train-trace`
   - `cargo run --release -p tt-mnist -- --model transformer --steps 10 --train-trace`
   - Verify steady-state time per step drops by ~0.2–0.4 ms/step over untraced mode.
