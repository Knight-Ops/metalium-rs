# Traced Execution Architecture for Burn on Tenstorrent

> Open work lives in [roadmap.md](roadmap.md); the section 8 scaling roadmap below is part of it.

Current BFP extension (2026-10-06): captured conversions, compressed outputs,
parameter copies and changed-input replay pass step94–96 on both cards. Trace
holds and deferred frees retain physical slot sizes and exponent groups.
Replay input buffers still require ordinary F32 physical storage; perform
explicit BFP casts inside the captured computation. Decoded output readback
widens on device. Training keeps gradients, masters and optimizer state F32.
See [the current storage contract](mixed-bfp-storage.md); the motivation and
original limitations below describe the earlier trace design.

Status (2026-10-10): sections 3, 4 and 6 describe what `crates/burn-tt/src/trace.rs` and
`crates/tt-mnist/src/trace.rs` contain. The generic, `Module`-based API that an earlier revision of
this document specified was **not built**; it is kept, labelled, at the end of section 4.

## 1. Architectural Motivation & Scope

In standard Burn backend execution, every model invocation (inference forward pass or training step) incurs substantial host-side latency:
- Constructing and traversing computational graph nodes and Autodiff tapes.
- Allocating intermediate tensor descriptors on the host heap.
- Dispatching commands across the `burn-tt` server thread boundary.
- Multiple PCIe MMIO round-trips and synchronization barriers per step.

For small networks like MNIST this host-side cost, not the device, dominates an untraced step (the
scoreboard in [performance-log.md](../learnings/performance-log.md) records the device busy for
only 12.6% of wall time on a queue of small matmuls).

Hardware traces (`Session::begin_trace` / `Session::end_trace`, `hardware-coverage.md` X4d) compile the entire sequence of Tensix kernels and data mover transfers into a single GDDR command stream. Once captured, execution is triggered by a single `CALL` entry per tile (`Session::replay`), eliminating all host op construction and dispatch latency.

### The original limitation (historical)

The first `burn_tt::Trace` (`crates/burn-tt/src/trace.rs`) supported only a restricted form of **inference**,
and still exists in that form:
1. **One F32 input**: `Trace::capture(input: &TtTensor, ...)` holds a single input buffer.
2. **One F32 output**, downloaded as `Vec<f32>`.
3. **No training**: a training step is replayable only if the optimizer writes the same buffers
   (Burn's tensors are immutable) and the loss stays on the device.

`TracedInference` and `TracedTrainingStep` (below) lift those limits:
- **`TracedInference`**: $N$ inputs, $M$ outputs, F32 or raw-bit (BF16, I32, Bool) payloads, read-only parameters.
- **`TracedTrainingStep`**: forward + loss + backward + optimizer update with in-place parameter writeback.

---

## 2. System Architecture

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
     │  - N inputs (F32 or bits)   │       │  - Forward + Loss + Bwd     │
     │  - M outputs (F32 or bits)  │       │  - Optimizer update         │
     │  - Read-only parameters     │       │  - In-place param writeback │
     │  - Streaming outputs        │       │  - Scalar loss readback     │
     └─────────────────────────────┘       └─────────────────────────────┘
```

Both are non-generic structs in `burn-tt` that work on `TtTensor` primitives and a caller-supplied closure.
Burn-level concerns (walking a `Module`, matching parameters, choosing the optimizer) belong to the caller;
`tt-mnist` is the one caller today.

---

## 3. Traced Inference (`TracedInference`, as built)

### API (`crates/burn-tt/src/trace.rs`)

```rust
pub struct TracedInference { /* device, trace id, input and output buffer ids, held tensors */ }

impl TracedInference {
    /// Capture inference over arbitrary inputs and outputs. `inputs` are resident,
    /// buffer-owning tensors (not transposed or row views); `f` runs the forward pass
    /// and returns the output tensors, which must be computed on the device.
    pub fn capture(
        inputs: &[&TtTensor],
        f: impl FnOnce() -> Vec<TtTensor>,
    ) -> Result<(Self, Vec<server::OutputPayload>), TtError>;

    /// Output shapes as stored, `[rows, cols]`.
    pub fn output_shapes(&self) -> Vec<[usize; 2]>;

    /// Replay with new input payloads (`InputPayload::F32` or `Bits`); one `CALL` per tile.
    pub fn run(&self, inputs: Vec<server::InputPayload>)
        -> Result<Vec<server::OutputPayload>, TtError>;
    pub fn run_timed(&self, inputs: Vec<server::InputPayload>)
        -> Result<server::GenericTraceRun, TtError>;
}
```

`Trace::capture(&TtTensor, FnOnce() -> TtTensor)` / `Trace::run(Vec<f32>)` remain as the one-input,
one-output F32 form. A capture refuses a host fallback (a download would be needed) with a panic that ends
the capture cleanly, and an input or output of another device, a view, or a non-storable dtype with an
`TtError`. Outputs come back as `OutputPayload::F32` or `OutputPayload::Bits` (raw words for BF16, I32,
Bool), never as tensors, because the output buffer is rewritten by every replay.

### Trace Execution Lifecycle (Inference)
1. **Capture Phase**:
   - `Session::begin_trace()`
   - Run `f()`: all Tensix kernels, GEMMs, and mover transfers are recorded.
   - `Session::end_trace()`
   - Download the outputs of the sample run and return `(TracedInference, first_outputs)`.
2. **Replay Phase**:
   - Write new input payloads into fixed GDDR input slots using `write` (F32) or `write_bits` (I32/Bool/BF16).
   - Enqueue a single `CALL` entry per tile (`Session::replay`).
   - Download the $M$ output buffers back to host memory.

### Several traces over one state (`SharedStateTraces`, 2026-10-15)

A decoder needs more than one trace over the same key/value cache: its one-token step, and prefill
chunks of a few sizes. `SharedStateTraces<K>` (`crates/burn-tt/src/trace.rs`) holds them, keyed by
the caller's `K`. Each capture works like this:

- it receives the same state tensors;
- it returns the new state, one tensor for each state tensor, of the same shape and dtype;
- `capture_with_state` records the on-card `copy_into(new → state)`.

So whichever trace replays, the next reads what it left. The rules:

- **The step is functional.** It reads the state it was given and returns the new state. It must not
  read a tensor that an earlier capture produced, because that tensor is the earlier trace's
  scratch: its own replays rewrite it, and no other trace's do. A model whose step mutated its cache
  object in place would capture the second trace over the first one's scratch. Under this rule, the
  order of capture does not matter.
- **A capture runs its step once, so it writes the state.** Nothing may rely on what the state held
  before a capture. A cache read only at positions the generation itself has written meets this
  rule.
- **Several traces fit only together, or fail.** They share the program cache (240 KiB of L1 a unit;
  programs are refcounted, so a program shared by two traces is held once). A capture that does not
  fit fails the attachment for good (the error is sticky), so reattach before retrying with fewer
  traces. `SharedStateTraces::footprint` and `TracedInference::footprint` (`Session::trace_footprint`)
  report what each unit holds.

The model side is `tt_models::generate`:

- `FixedShapeDecoder` is a decoder's chunk and step over a cache, both functional.
- `TracedGenerator` captures the step and one chunk per bucket size, feeds a prompt with the smallest
  bucket that covers it (chunks of the largest past that), then replays the step a token. Token ids go
  in, and the lookup runs inside the trace.
- `generate_fixed` is the same schedule run eagerly.

The gate is `step187_traced_generation`, run on Llama shapes SmolLM2 does not have and on a GPT-2-style
decoder: bit for bit against the eager schedule and the growing cache.

---

## 4. Traced Training (`TracedTrainingStep`, as built)

### A. In-place parameter writeback (`copy_into`)
- **The problem**: Burn tensors are functional/immutable. `optim.step(lr, model, grads)` allocates fresh `Param` tensors in new GDDR buffers. If captured statically, subsequent replays would continue reading the *initial* parameter buffers, never accumulating weight updates.
- **The solution, split across two crates**:
  1. *Caller (`tt-mnist`, `crates/tt-mnist/src/trace.rs`)*: `collect_parameter_updates(&model, &new_model)` runs Burn's `ModuleVisitor` (`ParamCollector`) over the model before and after `optim.step`, matches parameters by `ParamId` and returns `Vec<(new_param, orig_param)>` of `TtTensor` primitives. `ensure_resident(&model)` uploads every parameter beforehand, because a capture forbids host-to-device uploads.
  2. *`burn-tt`*: `TracedTrainingStep::capture(inputs, f)` calls `f`, which returns `(loss, updates)`. For every `(new, orig)` pair it records an on-device `copy_into(new, orig)` (`server::copy_into`) **before** `Session::end_trace()`, so the writeback is part of the command stream.
  3. Every replay therefore updates the original buffers in GDDR at the end of the step; the user's `model` keeps pointing at them, so after $N$ replays it holds the trained weights in place.
- Parameters that are transposed views are refused (`copy_into` needs whole buffers).

### B. Optimizer state
- **Stateless optimizers (SGD)** are what is built and gated (`tt-mnist` uses `SgdConfig`): only parameter tensors are written back.
- **Stateful optimizers (Adam, AdamW)**: not built. Their moment tensors would have to be returned in the `updates` vector as further `(new, orig)` pairs; nothing in `tt-mnist` or `burn-tt` collects them and no gate covers it.

### C. Loss readback without trace abort
- `Session::begin_trace` rejects downloads during capture (`TraceError::HostTransfer`).
- The loss tensor stays resident during capture and its buffer is the trace's one scalar output.
- `capture` downloads step 0's loss after `end_trace()`; each `step` replays, then downloads the 4-byte loss (`StepTiming { loss, write_inputs, replay, read_loss }`).

### API (`crates/burn-tt/src/trace.rs`)

```rust
pub struct TracedTrainingStep { /* device, trace id, input buffer ids, loss buffer id, held tensors */ }

impl TracedTrainingStep {
    /// `inputs`: the batch tensors (resident, buffer-owning). `f` runs forward, loss,
    /// backward and optimizer step and returns `(loss, updates)` with
    /// `updates = [(new_param, orig_param), ..]`.
    pub fn capture(
        inputs: &[&TtTensor],
        f: impl FnOnce() -> (TtTensor, Vec<(TtTensor, TtTensor)>),
    ) -> Result<(Self, f32), TtError>;

    /// Replay the whole step for a new batch.
    pub fn step(&self, inputs: Vec<server::InputPayload>) -> Result<StepTiming, TtError>;
}
```

`tt-mnist` shows the call pattern (`train_traced` in `crates/tt-mnist/src/main.rs` and
`transformer.rs`): upload parameters, build the batch tensors, `capture(&[&xp, &yp], || { loss; grads;
new_model = optim.step(..); (primitive_float(loss), collect_parameter_updates(&model, &new_model)) })`, then
`step(vec![InputPayload::..])` per batch. There is no `finish()`: the model never moved, and holds the
trained weights in place.

### Not built: the generic design (historical proposal)

An earlier revision of this document specified `TracedInference<M: Module<TtBackend>>`,
`TracedTrainingStep<M: AutodiffModule<TtBackend>, O: Optimizer<M, TtBackend>>`, traits
`TraceableInputs` / `TraceableOutputs` / `TraceableBatch`, `capture(model, optim, lr, sample_batch, step_fn)`
and `finish() -> M`, with `ModuleVisitor` parameter collection and optimizer-state mapping inside
`burn-tt`. None of these exist. If a generic wrapper is wanted, it would be a layer above the current
structs (generalising `collect_parameter_updates` and covering Adam state), not a change to them.

---

## 5. Dynamic Embedding Lookups in Language Models (design note, 2026-10-05)

In deep learning models with token inputs (Transformers, LLMs):
- Standard Tensix ops (GEMMs, normalizations, activations, softmax, attention, reductions) operate dynamically on matrix values in GDDR.
- In `burn-tt`, generic CrossEntropy already uses `index_mask` and `MASK_WHERE`, which dynamically evaluate changed target indices on-device.
- `burn::nn::Embedding` composes `float_select` (gather rows) and `float_select_add` (`embedding_backward`), which at the time of writing baked row indices into mover commands. `tt-mnist`'s traced transformer step captures the embedding with token ids as a trace input; whether changed ids take effect on replay through this path was not asserted by the gates in section 7. **It is now** for the lookup of a rank-2 F32, I32 or Bool table (2026-10-13): `step171_select_rows` captures a `select` whose index is the `argmax` of the traced input and replays three inputs that pick different rows, equal to the host's rows on ttsim and card 0 (`SELECT_ROWS` reads the index where it runs, so the job list depends on shapes alone). The BF16 and BFP lookups still stage and are not covered.

### Alternative solutions:
1. **One-Hot GEMM Path**:
   For vocabulary $V$ and batch tokens $N$, express the lookup as $X_{\text{one\_hot}} \times W_{\text{embed}}$, with backward $X_{\text{one\_hot}}^T \times dY$. Both passes are standard 2D GEMMs that operate dynamically over arbitrary token batches in GDDR without recompiling the trace.
2. **Backbone Tracing**:
   Alternatively, embedding lookup runs in the data loader / pre-step, and the trace captures the TransformerEncoder + Head + Loss + Backward + Optimizer (representing >98% of model FLOPs).

---

## 6. Implemented Architecture & Code Additions (2026-10-05)

### Layer 1: Core runtime (`tt-kernels`)
1. **`crates/tt-kernels/src/tensor.rs`**: `pub fn copy_into(src: &DramTensor, dst: &DramTensor, units: usize) -> Result<Work>`, emitting `record::READ_RUN` from `src` and `record::WRITE_RUN` to `dst`: executed in hardware without host interaction and recorded cleanly into traces.
2. **`crates/tt-kernels/src/session.rs`**: `Session::copy_into` (plus `copy_into_bfp`, `copy_into_bf16`) and `Session::write_bits(&DramTensor, &[u32])`, enabling non-F32 replay writes (`I32`, `Bool`, `BF16`).

### Layer 2: Burn backend (`burn-tt`)
1. **`crates/burn-tt/src/server.rs`**: `copy_into` and `write_bits` handlers; unified multi-buffer replay `run_generic_trace` (multi-input write, single-entry `CALL` replay, multi-output and scalar download); `InputPayload` / `OutputPayload` / `OutputKind`.
2. **`crates/burn-tt/src/trace.rs`**: `Trace`, `TracedInference`, `TracedTrainingStep` and `StepTiming`, as described in sections 3 and 4. No generic traits, and no `ModuleVisitor` use.
3. **`crates/burn-tt/src/tensor.rs`**: `ensure_resident()` and `download_device()` on `TtTensor`.
4. **`crates/burn-tt/src/lib.rs`**: re-exports `Trace`, `TracedInference`, `TracedTrainingStep`, `StepTiming`, `InputPayload`, `OutputPayload`, `GenericTraceRun`.

### Layer 3: Model workloads (`tt-mnist`)
1. **`crates/tt-mnist/src/trace.rs`**: `ParamCollector` (a `ModuleVisitor`), `collect_parameter_updates`, `ensure_resident`, and primitive-extraction helpers.
2. **`crates/tt-mnist/src/main.rs`**: `--train-trace` and `--hidden <HIDDEN_SPEC>` (e.g. `512,256`); `train_traced` for the MNIST MLP.
3. **`crates/tt-mnist/src/transformer.rs`**: `train_traced` for `TinyTransformer`.

---

## 7. Verification & Results

### 7.1. Gates
- **`crates/tt-tests/tests/step40_burn_trace.rs`** (5 tests):
  - `a_traced_forward_pass_is_the_fresh_one`: single-input trace replays the fresh forward pass.
  - `a_capture_that_downloads_to_the_host_is_refused_and_ends`: a host fallback is refused without leaving the session capturing.
  - `generalized_inference_trace_multi_input_multi_output`: multi-input/multi-output tracing across dtypes.
  - `traced_training_step_updates_weights_in_place_and_reduces_loss`: in-place parameter writeback and loss decrease across replays.
  - `transformer_traced_training_runs_and_loss_decreases`: `TinyTransformer` traced training for three steps.
- `step12_mnist` (golden trajectory and `compare_all_four_execution_modes_latency_and_traffic`, which includes traced replay with and without fusion) and the `step59_burn_transformer` regression.

### 7.2. Silicon measurements

Measured numbers live in [performance-log.md](../learnings/performance-log.md) and are not repeated here.
The relevant entries: the scoreboard rows "MNIST training, card 0, 1 tile" (1.4 ms/step against burn-flex 0.5) and
"Transformer training" (8.8 ms/step against burn-flex 4.35); the 2026-10-03 change-log rows for traced MNIST
inference (a batch on 8 tiles in about 0.2 ms); and "Kernel Fusion & Traced Execution Performance Comparison
(2026-10-06)", run `1791257827`, which compares eager and traced, fused and unfused forward passes on both cards.

*Historical, withdrawn:* an earlier revision of this section tabulated "99.45 ms/step beats the CPU" for a 535k-parameter MLP, a
1.12 ms CPU MNIST step, 145,150 / 205,301 img/s inference and "4.19 ms/step traced transformer training".
None of these are recorded on the scoreboard or in the change log, and the CPU step time contradicts the
burn-flex baseline there (0.5 ms), so they are removed rather than carried forward.

---

## 8. Scaling Roadmap & Future Work

Everything in this section is open work (it is also tracked in [roadmap.md](roadmap.md)).
Measured traced-execution results are on the scoreboard and in the change log of
[performance-log.md](../learnings/performance-log.md): MNIST training is 1.4 ms/step on one
tile against 0.5 ms for burn-flex on the host, the transformer 8.8 against 4.35, and a traced MNIST
forward pass replays in about 150 us a batch on the card (run `1791257827`). Hardware traces remove host op
construction, not the device cost, and they do not yet make training faster than the CPU.

To scale traced execution from single-tile benchmarks to large models across many Tensix cores, a full chip
(120 compute tiles on a p150a; ttsim models 140) and multi-card clusters (p150a mesh), four architectural
frontiers must be addressed:

### 8.1. Trace Cache Segmentation & Instruction Streaming

#### Physical Constraint: 240 KB L1 Program Cache
Each Tensix tile provides 1.5 MiB of local L1 SRAM (`tt_isa::tensix::L1_SIZE`), of which **240 KiB** is left for the firmware program cache (`tt_isa::l1::PROGRAM_CACHE`: L1 after the program slots and the 16 KiB trace buffer).
When capturing a large enough network's entire training step (forward pass, loss, backward pass, optimizer updates, and parameter writebacks) on a single tile, the unrolled kernel instruction stream can exceed 240 KiB. `Session` then refuses with:
```text
a list's programs do not fit an empty program cache
```
The MNIST and `TinyTransformer` traces captured today fit; deeper training graphs would need program cache management.

#### Architectural Solution:
1. **Instruction Segment Chunking**:
   - Rather than requiring the entire training step to reside concurrently in L1 SRAM, segment the trace into logical execution phases:
     - Segment 0: Forward pass ($L_1 \to L_2 \to \dots \to L_N$).
     - Segment 1: Loss & backward gradient propagation ($dL_N \to \dots \to dL_1$).
     - Segment 2: Optimizer step & in-place `copy_into` writebacks.
   - The firmware `CALL` dispatcher pages in program chunks from GDDR on demand, evicting completed segments while preserving intermediate data buffers in L1/GDDR.
2. **Spatial Program Sharding**:
   - Distribute operations across multiple Tensix tiles ($N \ge 4$). Because each tile only receives the mover and compute instructions for its assigned tensor shards, the per-tile program footprint drops proportionally by $1/N$, fitting within the 240 KiB threshold.

---

### 8.2. Distributed Data Parallel (DDP) Training

Current multi-tile execution divides each operation's tile units across the tiles of one chip (`TileChoice`, `Session::enable_dram`; see [streaming-dataflow-architecture.md](../learnings/streaming-dataflow-architecture.md)). While ideal for inference latency, training large batch sizes achieves maximum hardware efficiency through **Data Parallelism**.

#### Architectural Design:
1. **Tile-Group Replicas**:
   - Partition the 120 compute tiles of a p150a into independent replica groups (e.g. 4 groups of 30 tiles, or 8 groups of 15 tiles).
   - Replicate model weights across all groups; partition each minibatch across groups along the batch dimension $B$.
2. **On-Chip AllReduce via Tensix NoC Rings**:
   - At the completion of the backward pass, each group holds local parameter gradients $\nabla W_k$.
   - Execute an on-device ring or tree `AllReduce` across the high-bandwidth on-chip Network-on-Chip (NoC0/NoC1) routers to compute the mean gradient $\frac{1}{K}\sum \nabla W_k$.
   - Optimizer step and weight writebacks execute locally on the averaged gradients with zero host interaction.
3. **Trace Integration**:
   - The `AllReduce` communication primitives are captured directly into the trace command stream alongside compute kernels. The entire multi-tile data-parallel training iteration replays with a single host dispatch call.

---

### 8.3. Multi-Card Scaling (Inter-Card Ethernet Fabric)

Tenstorrent Blackhole cards (such as the dual-p150a PCIe platform) are connected via 400 Gbps Ethernet links (one link measures 48-51 GB/s on the wire, see the scoreboard). The workspace already provides foundational inter-card building blocks:
- `tt_kernels::shard::Fabric` and `burn_tt::MeshEngine` (`kmd_mesh_engine`). Mesh traces cover inference only (`step147`); mesh training traces are unsupported.
- Multi-device sessions (`--cards 0,1`).
- Tensix Ethernet mover cores (E1 customer core).

#### Next Steps for Traced Multi-Card Training:
1. **Cross-Card Gradient AllReduce**:
   - Implement pipelined Ring-AllReduce across the inter-card 400G Ethernet links, driven directly by the E1 Ethernet cores.
   - Overlap inter-card gradient communication with backward-pass computation of earlier layers (bucketed AllReduce).
2. **Unified Multi-Card Trace Capture**:
   - Synchronize trace capture across multiple card sessions, assigning aligned `trace_id` handles.
   - Provide a unified `TracedTrainingStep` coordinator that triggers multi-card execution via non-blocking PCIe queues, preventing host-side synchronization bubbles.

---

### 8.4. L1 SRAM Operator Fusion (Norm, Activations, Attention)

Status: only the MVP `ADD_RELU` / `RELU` fuser exists; the general fuser, matmul epilogue and reduce-map fusers are specified in [kernel-fusion.md](kernel-fusion.md) and are open.

In standard eager execution, every intermediate operation writes its output back to GDDR and re-reads it in the subsequent kernel:
$$\text{Linear} \xrightarrow{\text{GDDR}} \text{LayerNorm} \xrightarrow{\text{GDDR}} \text{GELU} \xrightarrow{\text{GDDR}} \text{Linear}$$
On modern accelerators like Blackhole, memory bandwidth (GDDR) is the primary performance bottleneck for non-GEMM operators.

#### Architectural Design:
1. **Fused Matmul + Epilogue**:
   - Fuse bias addition, activation functions (ReLU, GELU, SiLU), and dropout directly into the packer/NC writer pipeline before writing out of Tensix L1 SRAM.
   - Tensix math/pack cores can evaluate element-wise SFPU functions on Dst registers directly before packing to GDDR, eliminating intermediate round-trips.
2. **Fused Normalization**:
   - Replace composed multi-kernel LayerNorm/RMSNorm (mean reduction $\to$ variance reduction $\to$ normalize $\to$ affine scale) with a single-pass L1 fused kernel that computes statistics within the tile's 1.5 MiB SRAM.
3. **Fused FlashAttention for Transformers**:
   - Implement tiled online softmax attention ($Q K^T / \sqrt{d} \to \text{Softmax} \to V$) operating entirely in Tensix L1 staging buffers.
   - Prevents materializing the $O(S^2)$ attention matrix in GDDR, yielding major speedups for language models and Vision Transformers (`TinyTransformer`).

---


---

### 8.6. Pipelined Tracing: Historical Investigation & Protocol Rules

Implemented (2026-10-03), and part of the unified design in
[`streaming-dataflow-architecture.md`](../learnings/streaming-dataflow-architecture.md#traces):
streaming ownership is the only GDDR compute scheduler, fresh and traced, and
captures retain both the reader and the writer stream. The generation rebasing,
program holds, chunk-boundary behaviour, NC fetch and recovery rules are described
there, with the tests that cover them and the overlap thresholds a capture uses
([measurements](../learnings/performance-log.md#streaming-ownership-rollout)).

`Session::enable_dram(b, nc)` is the only setup; `TT_PIPELINE=0` serializes slot
reuse but keeps NC as the writer. The legacy wave, B-only pipeline and
two-host-queue NC schedulers are removed.

#### Original Investigation & Protocol Findings:
- **Rebase generations**: The firmware’s CALL replay path recognizes and rebases `KERNEL`, `LAUNCH`, and `KERNEL_WAIT`. `capture_segment()` converts `KERNEL` generations into offsets relative to the capture, and converts `LAUNCH` and `KERNEL_WAIT` as well, preserving each launch/wait pairing so replay uses newly reserved generations.
- **Program cache retention**: Capture code holds program-cache references from `LAUNCH` entries in addition to `KERNEL` entries, preventing eviction until the trace is released.
- **Trace-chunk boundary handling**: Replay fetches commands in 64-entry chunks and drains NoC requests between chunks. When launch and wait land in different chunks, the active generation and staging-buffer ownership must be preserved across boundaries.
- **NC reader/writer streams**: Captures retain both the B reader and NC writer streams (via `PAIR_CALL` shared packets) and track NC lifetime and completion status.
