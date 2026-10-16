# Implementation Plan: Burn Backend Kernel Fusion (B12 & B13)

This plan defines the concrete specification, architecture, and verification checklists for implementing kernel fusion in `burn-tt` using Burn's `burn-fusion` framework and native Tensix execution in `tt-kernels`.

It tracks milestones **B12** and **B13a–B13c** from [`burn-backend-parity.md`](burn-backend-parity.md).

**Decision (2026-10-16): fusion lives in burn-fusion fusers.** The alternative was model-level fused ops in tt-models.
Fusers are the end state, reliable and benefiting any model that runs on TT through Burn. Individual ops are still
optimised as far as they go, since every fuser falls back to them. Evidence:
[decode-small-ops.md](decode-small-ops.md) F0.
- A decode step on `Fusion<TtBackend>` captures and replays (gate `step193`), and fuses nothing yet.
- It costs +1.5 / +2.2 ms a step (135M / 1.7B), because burn-fusion 0.21 expands `sigmoid` into Burn's 8-op
  composition.
- RMSNorm's and SiLU's ops are contiguous in the stream, with scalars bound at execution.

Current BFP contract (2026-10-06): `StorageBackend` also supports
`Fusion<TtBackend>`. Explicit casts resolve pending work and register a new
handle, preserving compression boundaries. Compressed add/ReLU executes its
constituent storage rules rather than collapsing their rounding into the F32
fused path. Step95 gates storage propagation and residency on both cards.
See [the delivered storage and training rules](mixed-bfp-storage.md).

---

## 1. Architectural Strategy

Rather than building an independent graph optimization compiler beneath Burn, `burn-tt` directly implements Burn's `FusionBackend` trait. Burn's graph capture and greedy pattern matching engine serves as our single graph optimization layer, directly emitting typed execution descriptors for `tt-kernels`.

```mermaid
flowchart TD
    subgraph Client["Burn Client Thread"]
        Op1["Tensor Ops (Matmul, Add, ReLU)"] --> FusionClient["burn-fusion::FusionClient"]
    end

    subgraph Runner["Burn Fusion Runner Thread"]
        FusionClient --> PatternMatcher["OperationFusers (burn-tt)"]
        PatternMatcher -->|Emit| TtOpt["TtOptimization"]
        TtOpt -->|execute()| BackendExec["server::submit(KmdEngine)"]
    end

    subgraph Device["Tenstorrent Blackhole Device"]
        BackendExec --> Tensix["Tensix Core (Single Fused Execution)"]
    end
```

---

## 2. Milestone Breakdown & Specification

### Milestone B12: `BackendIr` Implementation
**Objective**: Unify `burn-tt`'s tensor representations into a single `Handle` type to satisfy the Burn IR contract.

- **Dependencies**: None.
- **Components Modified**: `crates/burn-tt/src/tensor.rs`, `crates/burn-tt/src/lib.rs`.
- **Key Types**:
  ```rust
  #[derive(Clone, Debug)]
  pub enum TtHandle {
      Tensor(TtTensor),
      Quantized(TtQTensor),
  }
  ```
- **Trait Implementation**:
  Implement `burn_ir::BackendIr` for `TtBackend`:
  - `type Handle = TtHandle;`
  - Implement the 8 primitive-to-handle conversions (`float_tensor_into_handle`, `handle_into_float_tensor`, `int_tensor_into_handle`, etc.).
  - Implement tensor cloning and handle management.
- **Status (2026-10-10)**: `BackendIr` is implemented (`crates/burn-tt/src/tensor.rs`).
  The router/remote gate below has never been run: neither `burn-router` nor `burn-remote`
  is a dependency of any workspace crate.
- **Verification Gate** (not yet run):
  - `burn-router` with `(TtBackend, Flex)` passes delegation tests. Flex is a test-only
    comparison backend (`tt-tests`), never a `burn-tt` dependency.

---

### Milestone B13a: `FusionBackend` Baseline (Zero Fusers)
**Objective**: Integrate the `burn-fusion` runtime pipeline with eager fallback, proving client-server thread safety and zero regressions before enabling pattern matching.

- **Status (2026-10-10)**: delivered as specified, then extended: `TtFusionRuntime::fusers`
  now returns one MVP fuser (`ElementWiseFuser`, `ADD_RELU` and `RELU`; see B13b), so "zero
  fusers" below is the original baseline, not today's behavior. The `Tt = Fusion<TtBackend>`
  alias and default-on `fusion` feature exist (`crates/burn-tt/src/lib.rs`).
- **Dependencies**: B12.
- **Components Modified**: `crates/burn-tt/Cargo.toml`, `crates/burn-tt/src/lib.rs`, new `crates/burn-tt/src/fusion/mod.rs`.
- **Cargo Feature**:
  Introduce `fusion` feature to `burn-tt/Cargo.toml` (default on, mirroring `burn-cuda`):
  ```toml
  [features]
  default = ["std", "fusion"]
  fusion = ["dep:burn-fusion", "dep:burn-ir", "dep:serde"]
  ```
- **Exported Type**:
  ```rust
  #[cfg(feature = "fusion")]
  pub type Tt = burn_fusion::Fusion<TtBackend>;
  #[cfg(not(feature = "fusion"))]
  pub type Tt = TtBackend;
  ```
- **Runtime Skeleton**:
  - Define `TtFusionRuntime` implementing `burn_fusion::FusionRuntime`.
  - Define `TtOptimization` (empty enum initially) implementing `burn_fusion::Optimization<TtFusionRuntime>`.
  - Define `OptimizationState` with `serde` serialization for optimization caching.
  - Implement `fusers(device) -> vec![]` (zero fusers).
  - Implement `burn_fusion::FusionBackend for TtBackend`.
- **Behavior**:
  All operations stream through Burn's fusion client thread and execute eagerly on `TtBackend`.
- **Verification Gate**:
  - `cargo test -p tt-tests --features e2e --test step12_mnist` passes bit-for-bit when run with `Tt = Fusion<TtBackend>`.
  - Transformer regression `step59_burn_transformer` passes.
  - Step latency and memory usage remain within baseline noise.

---

### Milestone B13b: Element-Wise Chain Fuser & SFPU Micro-Op Engine
**Objective**: Eliminate combinatorial recipe explosion ($A+B=AB$, $A+B+C=ABC$) by building a generalized SFPU micro-op bytecode engine that fuses arbitrary pointwise DAGs into a single Tensix execution pass over L1 tiles.

- **Dependencies**: B13a, [`tt_kernels::sfpu`](../learnings/kernel-fusion-architecture.md).
- **Status (2026-10-10)**: only an MVP exists: `ElementWiseFuser` (`crates/burn-tt/src/fusion/eltwise.rs`)
  recognises `add` followed by Burn's lowered ReLU (`lower_equal_elem(0.0)` + `mask_fill(0.0)`) as
  one `ADD_RELU` SFPU dispatch, and a lone ReLU as one `RELU` dispatch. The micro-op bytecode
  engine, register allocator and the fusible-operation list below are not built (Phase 2
  checklist, unchecked items).
- **The Combinatorial Trap vs. Micro-Op Bytecode**:
  - Rather than hardcoding distinct hardware opcodes (`ADD_RELU`, `ADD_GELU`, `MUL_ADD_SIGMOID`), `tt-kernels` defines a **generic parameterized SFPU kernel** driven by a compact micro-op sequence passed in the program descriptor.
  - Bytecode operations operate across the 8 local SFPU vector registers (`L0..L7`):
    ```rust
    #[repr(u8)]
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum SfpuMicroOp {
        LoadDst { src_row: u8, dst_reg: u8 },
        StoreDst { src_reg: u8, dst_row: u8 },
        Add { src1: u8, src2: u8, dst: u8 },
        Mul { src1: u8, src2: u8, dst: u8 },
        Mad { src1: u8, src2: u8, src3: u8, dst: u8 },
        Relu { src: u8, dst: u8 },
        Gelu { src: u8, dst: u8 },
        Silu { src: u8, dst: u8 },
        Tanh { src: u8, dst: u8 },
        Sigmoid { src: u8, dst: u8 },
    }
    ```
- **Virtual Register Allocation**:
  - `ElementWiseFuser` runs a linear-scan register allocator over the pointwise sub-DAG targeting registers `L0..L7`.
  - **The 8-Register Invariant**: If live variables exceed 8, the fuser automatically splits the chain into two kernels.
- **Fusible Operations**:
  - Arithmetic: `float_add`, `float_sub`, `float_mul`, `float_div`.
  - Scalars: `float_add_scalar`, `float_sub_scalar`, `float_mul_scalar`, `float_div_scalar`.
  - Activations: `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `hard_sigmoid`.
  - Backward activations: `relu_backward`, `gelu_backward`.
  - Broadcasts: Row/column broadcasts without host-staged copies.
- **Verification Gate**:
  - Mathematical parity: bit-identical output compared to eager execution for arbitrary test DAGs.
  - Device traffic: `device_traffic().submissions` reduced by factor of $N$ (where $N$ is chain length).
  - Silicon validation on one card.

---

### Milestone B13c: Matmul Epilogue Fuser (Unified FPU + SFPU)
**Objective**: Fuse Matrix Multiplication with Bias Addition and Activation functions into a single pass through Tensix `Dst` registers, sharing the exact same SFPU micro-op engine.

- **Dependencies**: B13b, [`tt_kernels::matmul`](../../crates/tt-kernels/src/matmul.rs).
- **Fusible Pattern**:
  $$\text{Output} = \text{Activation}(\text{Matmul}(A, B) + \text{Bias})$$
  - Supported Activations: `ReLU`, `GELU`, `SiLU`, `None`.
- **Implementation**:
  - Create `MatmulEpilogueFuser` implementing `burn_fusion::OperationFuser`.
  - Matches `float_matmul` $\to$ row-broadcast `float_add` $\to$ activation.
  - Emits `TtOptimization::MatmulEpilogue { a, b, bias, epilogue_ops, out }`.
  - In `Optimization::execute()`, invokes `server::matmul_epilogue`.
- **Hardware Realization**:
  - TRISC1 computes GEMM into `Dst` rows `0..64`.
  - TRISC0 unpacks the bias row into `Dst` rows `64..128` (`B_ROW`) or `192..256` (`C_ROW`).
  - TRISC1 SFPU evaluates the epilogue micro-op bytecode directly on `Dst`.
  - TRISC2 packs finalized result once to L1 output circular buffer.
  - NCRISC writes final tensor to GDDR over NoC1.
- **Verification Gate**:
  - Linear + ReLU layer emits exactly **1 device job** instead of 3.
  - GDDR traffic shows 2 writes and 2 reads eliminated per layer.
  - Validated on silicon and ttsim via `SMOKE` suite.

---

### Milestone B13d: Fused Reductions & Normalizations (Reduce-Map)
**Objective**: Fuse composite reduction and normalization layers (`RMSNorm`, `LayerNorm`, `Softmax`) into a single circular-buffer stage in L1 SRAM, preventing GDDR round-trips across reduction boundaries.

- **Dependencies**: B13b, B13c.
- **Fusible Patterns**:
  - **RMSNorm**: $\text{Square} \to \text{Mean} \to \text{Rsqrt} \to \text{Mul(Scale)}$.
  - **LayerNorm**: $\text{Mean} \to \text{Variance} \to \text{Normalize} \to \text{Mul(Scale)} \to \text{Add(Bias)}$.
  - **Softmax**: $\text{Max} \to \text{Sub} \to \text{Exp} \to \text{Sum} \to \text{Div}$.
- **Hardware Realization**:
  - TRISC1 computes row reduction into `Dst`.
  - Intermediate stats (mean, max, sum) stay resident in `Dst` rows `192..256` or L1 CB.
  - Normalization and broadcast scaling evaluated on tiles in-place without GDDR spill.
- **Verification Gate**:
  - Transformer encoder step latency reduction on silicon.
  - Zero intermediate GDDR transactions in `tensor_traffic()`.

---

## 3. Execution Checklist

### Phase 1: Core Trait Plumb-Through (B12 & B13a)
- [x] Add `burn-ir = { version = "=0.21.0" }`, `burn-fusion = { version = "=0.21.0" }`, and `serde` to `crates/burn-tt/Cargo.toml`.
- [x] Implement `burn_ir::BackendIr` for `TtBackend` in `crates/burn-tt/src/tensor.rs`.
- [x] Implement `TtFusionRuntime` and `burn_fusion::FusionBackend` in `crates/burn-tt/src/fusion/mod.rs`.
- [x] Add `pub type Tt = Fusion<TtBackend>` behind `fusion` feature flag in `crates/burn-tt/src/lib.rs`.
- [x] Verify MNIST tests pass with zero fusers enabled.

### Phase 2: Element-Wise Fuser & SFPU Micro-Op Engine (B13b)
- [x] Implement `ElementWiseFuser` MVP in `crates/burn-tt/src/fusion/eltwise.rs` (fusing `Add` + `LowerEqualElem(0.0)` + `MaskFill(0.0)` into `TtOptimization::AddRelu` / `Relu`).
- [x] Implement native hardware SFPU `ADD_RELU` op (`kind_sfpu::ADD_RELU = 0x1f0`) in `crates/tt-kernels/src/sfpu/ops.rs`.
- [x] Add backend execution path in `crates/burn-tt/src/ops.rs` (`float_add_relu`).
- [x] Support Traced Execution + Kernel Fusion simultaneously in `burn_tt::Trace` and `crates/tt-mnist`.
- [x] Validate on silicon hardware across both Blackhole devices (`compare_all_four_execution_modes_latency_and_traffic`).
- [x] **Burn's default `sigmoid` composition** (cast, `x -1`, `exp`, `+1`, `log`, `x -1`, `exp`, cast) runs the
  native `SIGMOID` (2026-10-16, `fusion/chain.rs`). Decode on `Tt` is `TtBackend`'s bit for bit.
  - **How:** `ChainFuser` matches a declarative pattern (`Pattern`: steps and the step counts where it may end).
    - Every intermediate must die inside the match (read `ReadWrite`, or dropped).
    - Exact scalars (`-1`, `1`) are checked at execution; a miss runs the operations.
    - The casts are required, so a hand-written chain stays as written.
    - A chain may begin one op into its block when that op is one `ElementWiseFuser` stays open on (an `Add`); the
      op runs first as itself. Passing over any other op broke `add_relu`: it kept a matmul's block open over the
      bias `Add` that must begin one.
  - **Gate:** `step194` (bit for bit `TtBackend::sigmoid` over special values; near misses: `+2`, a kept `exp(-x)`,
    no casts) and `step193` (a decode step's logits equal `TtBackend`'s).
  - **Fixed on the way:** `ElementWiseFuser` lost its match to any later op while another fuser kept the block open.
- [x] **SiLU** (`x sigmoid(x) y`, decode-small-ops S2b) as one SFPU body (`SILU_MUL`, `SILU`), bit for bit the
  three ops (unit gate `sfpu::ops::silu`; `step194`, a fused projection's halves read in place). BF16 and the
  matrix element-wise mode run the separate ops.
- [ ] Define `SfpuMicroOp` instruction enum and parameterized execution kernel in `tt_kernels::sfpu`.
- [ ] Implement greedy virtual register allocator (`L0..L7`) with 8-register budget splitting.
- [ ] Generalize `ElementWiseFuser` to emit arbitrary micro-op bytecode chains.

### Phase 3: Matmul Epilogue Fuser (B13c)
- [ ] Implement `MatmulEpilogueFuser` in `crates/burn-tt/src/fusion/matmul.rs`.
- [ ] Add epilogue configuration to `tt_kernels::matmul::MatmulConfig`.
- [ ] Wire `Dst` SFPU micro-op bytecode execution in `matmul.rs` role generation.
- [ ] Add silicon smoke test running fused Linear+ReLU / Linear+GELU layer.
- [ ] Measure throughput and memory bandwidth savings in benchmark suite.

### Phase 4: Reduce-Map Normalization Fuser (B13d)
- [x] RMSNorm up to its scale as two ops (2026-10-16): the chain fuser's `RMS_NORM` pattern (`Mul(x, x)`,
  `MeanDim` of the last dimension, `AddScalar(eps)`, `Sqrt`, `Div`).
  - **The statistic** runs as one reduction kernel (`tt_kernels::sfpu::norm`, `Session::rms`, one job a tile row):
    the sum of squares in `sum_dim`'s order, then `MUL_SCALAR`, `ADD_SCALAR` and `SQRT`'s bodies before the pack.
  - **The division** is the existing multi-unit `DIV`.
  - Bit for bit the five ops (unit gate `sfpu::norm`, with a reversed-order control; `step195`; `step193`).
  - **Measured:** a single-kernel version that also divided cost 640 us a norm on silicon. `DIV`'s instructions are
    pushed one by one, all on one unit. A reduce-map whose map is long must spread its map over the units.
  - **The scale is not in it:** its `Reshape` is a block of its own in burn-fusion, and the product joins neither
    (observed).
  - **Limit:** rows of up to 128 tiles (4096 columns); longer rows run the separate ops.
- [ ] LayerNorm and softmax as reduce-maps.
- [ ] Add L1 circular-buffer streaming path for rows too long for one job.
- [ ] Benchmark fused transformer encoder layer in `step59_burn_transformer`.
