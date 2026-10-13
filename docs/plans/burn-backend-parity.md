# Burn backend parity — what `burn-tt` needs to feel like `burn-cuda`

> Open work lives in [roadmap.md](roadmap.md); this file is the B0-B16 design record and status
> table (section 6). Sections 1, 2 and 4 are the 2026-10-01 design, written before the native
> cutover: every statement about Flex, host fallback or "today" in them is **pre-cutover
> history** unless a current-status note says otherwise.

> Native cutover: `burn-tt` no longer delegates to Flex. Unsupported methods fail
> explicitly, and `TT_EXACT` is retired. Historical Flex fallback descriptions
> below are superseded by [the current backend contract](../../crates/burn-tt/README.md)
> and [the cutover backlog](burn-native-cutover.md).

Current implementation record (2026-10-05): native grouped/depthwise Conv2D,
Conv1D composition, transposed Conv2D, unfold and gradients now have resident
F32/BF16 paths; output padding requires less than the maximum of stride and dilation. Resident
arbitrary-axis gather/select and float scatter/select-add remove index readback.
Slice assignment enables native cat/repeat. Mesh batched products partition across
cards, including convolution/attention backwards. Both-card gates, changed-input
traces, analytic/Flex oracles and remaining acceptance gaps are recorded in
[tensix-next-features.md](../completed-plans/tensix-next-features.md). Historical unchecked inventory
rows below are not evidence that these paths are absent.

Current storage extension (2026-10-06): BFP8/BFP4/BFP2 store logical F32 with
native conversions/products, identity cast backward, fusion boundaries and
traced parameter copies. Named MNIST MLP/CNN precision policies pass both cards;
masters, gradients and optimizer state remain F32. This does not implement
portable Burn `QTensorOps`. Random draws run on the device (S7, step140-142), seeded per
device by `TtBackend::seed`; the RISC-V seed store needs a settling interval, and
WRCFG and simulator lane initialization differ from silicon. See [the current storage contract](mixed-bfp-storage.md).

Implementation guide for the Burn-facing half of `burn-tt`: the trait surface, composition
with Burn's wrappers and tooling, and the developer experience, measured against the
CubeCL backends (`burn-cuda`, `burn-wgpu`, both `burn_cubecl::CubeBackend`). It does **not**
track which op runs on which Tensix unit -- that is `hardware-coverage.md` (items F, S, M,
R, D), referenced by id below. Written 2026-10-01 against the pinned Burn 0.21.0 sources in
the cargo registry (`~/.cargo/registry/src/index.crates.io-*/burn-*-0.21.0`, abbreviated
`reg:`) and, for crates not in the registry (`burn-backend-tests`, `burn-train`,
`burn-remote`), `github.com/tracel-ai/burn` at tag `v0.21.0` (abbreviated `gh:`).

Rule this document applies throughout (from the request, and the repo's "hazards in the
API, not comments" rule): **a sharp edge is filed off in code, not documentation.** In
order of preference: (1) automate invisibly when it costs nothing or there is no
alternative; (2) automate, with a once-per-key warning that names the op, shape, reason
and cost; (3) a typed, actionable error at the earliest point. Never an opaque panic or
hang mid-model.

**Status, 2026-10-05** (current code; dated comparisons below are historical):
Native execution refuses unsupported operations explicitly. Full and arbitrary-axis
F32 sums/maxima, direct products, Boolean AND/OR reductions, rank-N argextrema,
inclusive sums/products, full-width I32 ALU and deterministic rounding/casts
have resident paths. The extensions pass simulator and both-card gates in `step69`–
`step72`. LayerNorm/RMSNorm use Burn's
native compositions, including backward gates, without new fused kernels.
BF16 storage, casts, layout copies, packed matmul and native arithmetic adapters
are implemented, with silicon gates for late narrowing. NCHW pooling and its
backwards are resident; BF16 average pooling uses GAPOOL. Packed BF16 batched
products, BF16 mesh execution and F32/BF16 pooling traces are gated. Mesh operands
widen on device for Ethernet transport. Integer reductions are resident;
checked division/remainder and integer mean now pass simulator and card-0 gates
(`1791232718`). Native attention passes card-0 forward/gradient/F32 trace and
BF16 training gates; distributed batched products pass actual two-card forward
(distributed gradients remain unvalidated). See [the implementation record](../completed-plans/tensix-next-features.md)
and [the backend contract](../../crates/burn-tt/README.md) for limitations.
B0 reporting and B3 stale-id protection are implemented. Fusion has its first phase (B12
`BackendIr`, `FusionBackend`, the `Tt = Fusion<TtBackend>` alias and one MVP element-wise
fuser, see [kernel-fusion.md](kernel-fusion.md)); typed errors (B1, 2026-10-11) and the
`Backend::sync` barrier are done, and so are lazy attach, the engine registry, device
enumeration and the card lock (B2, 2026-10-11); `tracing` targets, the conformance suite and
placement remain separate backend work.

Next backend order: the Burn conformance suite (B10/B11).
General model coverage continues in `hardware-coverage.md`.

---

## 1. Historical comparison (2026-10-01)

*Pre-cutover snapshot.* The "burn-tt today" column describes the Flex-delegating backend as
of 2026-10-01 and is kept as history. Since then, native cutover removed every Flex fallback
(unsupported ops, shapes and dtypes fail explicitly), rank-N storage and views, batched
matmul, asynchronous dispatch, stale-buffer protection (B3), the report and strict mode (B0)
and typed, sticky errors (B1) landed, `dtype_usage` is half corrected (B4) and fusion has
its first phase. The table's explicit `attach` with a panic when unattached, its attached-only
`device_count` and its missing driver lock were closed by B2 (2026-10-11); still open as the
table says: no conformance run (B10/B11) and no `DistributedBackend` (B14). Section 6 holds the
per-item status.

| Capability | burn-cuda / burn-wgpu | burn-tt today | Gap | Pri |
|---|---|---|---|:-:|
| Device creation | `CudaDevice::default()` just works: first `R::client(device)` lazily runs `DeviceService::init` on a per-device runner thread (`reg:cubecl-cuda-0.10.0/src/runtime.rs:47-48,344-346`; `reg:cubecl-runtime-0.10.0/src/client.rs:69`). WGPU adds an opt-in `init_setup` (`reg:cubecl-wgpu-0.10.0/src/runtime.rs:232,250`) and `CUBECL_WGPU_DEFAULT_DEVICE` | explicit `attach(device, factory)` required (`server.rs:324-369`); unattached device ops panic (`server.rs:384-386`; pinned by `tests/delegation.rs:114-119`) | lazy auto-attach with an engine registry; `attach` kept as the `init_setup` analogue | **P0** |
| Errors | `try_into_data`/`sync` return `ExecutionError`, documented as "any error since the last sync" (`reg:burn-tensor-0.21.0/src/tensor/api/base.rs:1904-1918`); init failures still `unwrap` | *(was `EngineError(String)` and a panic in the op; since B1: `TtError`, deferred to `try_into_data`/`sync` as `ExecutionError`, strict mode panics)* | typed `TtError` with hints; deferred (sticky) errors through `ExecutionError` | done |
| Fallback visibility | n/a (everything runs on device) | silent host fallback; `TT_TRACE_FALLBACK=1` prints a backtrace per *download* to stderr (`tensor.rs:223-230`); counters `tensor_traffic`, `device_time` (`traffic.rs:38,104`) | per-op report, `tracing` targets, strict mode | **P0** |
| Conformance | `burn-backend-tests` via `burn-dispatch` (`cargo test-cuda`, `gh:crates/burn-backend-tests/.cargo/config.toml`); Flex: "all pass" (`reg:burn-flex-0.21.0/README.md:158`) | not run; hand-written batteries (`tests/delegation.rs`, `step11_burn`, `step19_eltwise`) | vendored suite against `TtBackend` on ttsim | **P0** |
| `dtype_usage` | from the runtime's type table, `Accelerated` only where MMA exists (`reg:burn-cubecl-0.21.0/src/backend.rs:115-142`) | Flex's answer verbatim (`lib.rs:112-115`): claims F16/BF16/ints equal to F32 | `Accelerated` only for what has a device path | P1 |
| `device_count` | all devices the runtime sees (`backend.rs:144-147`) | number *attached* (`lib.rs:108-110`, `server.rs:295-297`) | count `/dev/tenstorrent/N` | P1 |
| `sync` | blocks on the client queue, maps errors (`backend.rs:69-74`) | default `Ok(())` | barrier + sticky error once dispatch is async | P1 |
| `memory_cleanup`, `memory_persistent_allocations`, `staging` | pool cleanup, persistent pool, pinned staging (`backend.rs:76-100`) | defaults (no-op) | cleanup = flush frees + allocator compaction; persistent = weights region | P2 |
| Readback | `tr_execute` batches every read into one copy (`reg:burn-cubecl-0.21.0/src/ops/transaction.rs:18`) | downloads happen *before* the future is returned (`ops.rs:359-363`), one round trip per tensor (`convert.rs:211-221`) | one server job per transaction; real futures | P1 |
| Rank / shape | any rank, any stride | device path only for rank-2 F32 (`tensor.rs:299-301`); rank-1 biases, rank>2 activations, `reshape` all on the host | rank-N stored as `[prod(lead), last]`; reshape views | **P0** |
| Batched matmul | one kernel | resident `matmul_dram_batched` for tile-aligned blocks; ragged, broadcast and strided batches repack on the device (`materialized_batched_matmul`) | resident batched path | P1 |
| Fusion | `Cuda = Fusion<CubeBackend<..>>` by default (`reg:burn-cuda-0.21.0/src/lib.rs`), four fusers (`reg:burn-cubecl-0.21.0/src/fusion.rs:147-154`) | none | `BackendIr` + `FusionBackend`; zero-fuser first, then Tensix fusers | P2 |
| Router / remote | via `BackendIr` (`reg:burn-router-0.21.0/src/types.rs:33`; `gh:crates/burn-remote/src/server/base.rs:171-178`) | no `BackendIr` | same impl as fusion step 1 | P2 |
| Distributed / DDP | `DistributedBackend` (`reg:burn-cubecl-0.21.0/src/ops/distributed.rs:12`) | none; enabling Burn's `distributed` feature anywhere makes `Autodiff<TtBackend>` stop being an `AutodiffBackend` (`reg:burn-autodiff-0.21.0/src/backend.rs:155-156`) | implement it (host all-reduce first) | P1 |
| Multi-device | one `CudaDevice` per GPU, `to_device` peer copies | `TtDevice{chip}`; cross-chip `to_device` is download + re-tag + upload (`ops.rs:25-30,348-357`); a mesh is one device, host-staged, no GDDR (`server.rs:630-643`) | enumerate cards; P2P later (checklist 9.11) | P2 |
| Feature flags | `default = [std, fusion, autotune]`, `tracing`, `distributed` (`reg:burn-cuda-0.21.0/Cargo.toml`) | none | `tracing`, `fusion`, `distributed`, `strict` | P2 |
| Tile choice | n/a | default `Exactly(3,4)` -- a coordinate valid on *these two* cards (`topology.rs:36,40-45`); tile count from `TT_TILES` only when the caller asks (`topology.rs:108-111`) | default from the chip's own grid | **P0** |
| Process exclusivity | driver arbitrates contexts | two processes may open the same card; nothing refuses (`tt-kmd/src/lib.rs:18-20` -- `Kmd::lock` exists, unused) | take a driver lock at open; typed "in use" error | P1 |
| Stale buffers | n/a | buffer ids restart at 1 per engine (`server.rs:106-126`), so a tensor that outlives a detach can name a *different* buffer after a re-attach; `supports_dram` cache outlives the attachment (`server.rs:415-432`) | attachment generation in every `Buffer` | **P0** (correctness) |

---

## 2. Trait surface and design proposals (2026-10-01)

### 2.1 `Backend` / `BackendTypes`

**Burn 0.21.** `Backend` (`reg:burn-backend-0.21.0/src/backend/base.rs:92-179`) requires
`name`, `seed`, `dtype_usage`, `device_count`; provides `ad_enabled` (110),
`memory_persistent_allocations` (116), `memory_cleanup` (130), `sync` (144), `staging`
(153), `supports_dtype` (168, derived from `dtype_usage`). `BackendTypes` (`base.rs:20-40`)
carries the device and the four primitives. `CubeBackend` overrides all of them
(`reg:burn-cubecl-0.21.0/src/backend.rs:56-147`).

**burn-tt (2026-10-01, pre-cutover).** `lib.rs:87-116`: one primitive (`TtTensor`) for
float/int/bool as Flex does, `TtQTensor` for quantized; `seed` seeds Flex's global generator
(`lib.rs:103-106` -- correct, since every random op is Flex's until S7); `device_count` counts
attached devices; `dtype_usage` is Flex's. `sync`, `memory_*`, `staging` are defaults.

**burn-tt (2026-10-10, current).** `TtTensor` is the float/int/bool primitive and `TtQTensor` is
metadata only. `seed` calls `random::seed`, which seeds the device's native generator (S7;
no Flex). `device_count` still counts attached devices (`server::attached_count`). `dtype_usage`
is native and half-honest: `F32`/`BF16` are `Accelerated`, `Bool` is general, `I32`/`F16` are
`Storage`, everything else (including `F64` and `QFloat`) is empty (`lib.rs`, gate
`step144_burn_fp16`). `sync`, `memory_*` and `staging` are still the defaults.

**Design.**

| Method | Recommended `burn-tt` behaviour |
|---|---|
| `name` | `tt<card 0, 8 tiles, gddr>` -- from the attached engine (a new `Engine::describe()`), so logs and `burn-train` summaries say what actually ran. Unattached: `tt<card 0, not attached>`; must not trigger an attach. |
| `seed` | keep seeding Flex; additionally store the seed per device for the device PRNG (S7). |
| `sync` | after B8 (async dispatch): a barrier job on the server queue that returns the device's sticky error, mapped to `ExecutionError::WithContext` with the `TtError` text. Before B8 every op already blocks, so it only flushes queued `free`s (`server.rs:461-465`) and reports. |
| `memory_cleanup` | barrier, then ask the engine to coalesce its GDDR allocator and drop host copies of tensors that are device-resident and unreferenced elsewhere (the `OnceLock` host copy in `tensor.rs:150` lives as long as the cell). |
| `memory_persistent_allocations` | run `func` with a thread-local "persistent" flag so uploads inside it go to a weights region of the allocator (fragmentation control, the CUDA meaning). P2. |
| `staging` | no-op is correct. Optionally pre-tilize the `TensorData` bytes on a worker (checklist 9.10 is the real fix). |
| `dtype_usage` | `F32`: `Storage | Arithmetic | Accelerated`. Every other dtype Flex supports: `Storage | Arithmetic` (they work, on the host) -- *not* `Accelerated`. `BF16` gains `Accelerated` with D1, ints with D3/S5. `supports_dtype` stays "it works", which is true; `Accelerated` is the honest "it runs on the card". |
| `device_count(type_id)` | count numeric entries of `/dev/tenstorrent/` (it also holds `by-id/`; `xtask/src/silicon.rs` `all_devices` already filters), plus devices attached to registered non-silicon engines (ttsim). Must not open anything. |
| `ad_enabled` | default (`Autodiff` overrides). |

**Gate.** Unit tests in `burn-tt/tests/backend.rs`: `name` before/after attach;
`dtype_usage(BF16)` lacks `Accelerated`; `device_count` equals the numeric node count with
nothing opened (checked by `device_traffic` being `None`).

### 2.2 Devices: `DeviceOps`, ids, defaults, enumeration

**Burn 0.21.** `DeviceOps: Clone + Default + PartialEq + Send + Sync + Debug + Device`
(`reg:burn-backend-0.21.0/src/backend/device.rs:30-43`); `Device::{to_id, from_id}` with
`DeviceId { type_id, index_id }`. Default dtypes per device live in a global registry,
initialised once (`device.rs:45-150`, `set_default_dtypes`). CUDA: `CudaDevice { index }`,
`Default` = 0, `type_id` 0 (`reg:cubecl-cuda-0.10.0/src/device.rs`). WGPU: an enum whose
`#[default]` is `DefaultDevice`, resolved at first use, overridable by
`CUBECL_WGPU_DEFAULT_DEVICE` (`reg:cubecl-wgpu-0.10.0/src/device.rs:16-51`).

**How CUDA "just works".** `ComputeClient::load(device)` creates a `DeviceHandle` whose
first use runs `CudaServer::init(device_id)` on a runner thread owned by that device
(`reg:cubecl-common-0.10.0/src/device/base.rs:49-61`, `handle/mod.rs:19-22`,
`handle/channel.rs:214-377`). That is *exactly* `burn-tt`'s server-thread model, made lazy.
`burn_backend` re-exports this machinery (`DeviceHandle`, `DeviceService`), and `burn-tt`
already links `cubecl-common` through it -- but `DeviceService: Send` and `init` is
infallible, while the ttsim engine is `!Send` and opening a card can fail. So keep
`server.rs`'s thread (it is proven and lets a `!Send` engine be built in place) and add the
laziness, rather than adopting `DeviceHandle`.

**burn-tt.** `TtDevice { chip }`, `Default` = chip 0, id `(0, chip)` (`lib.rs:58-83`);
`DeviceOps` empty (`lib.rs:85`). The engine is chosen by whoever calls `attach`
(`server.rs:324`) / `attach_topology` (`topology.rs:127-147`).

**Design: an engine registry plus lazy attach** (item B2).

```rust
// burn-tt/src/registry.rs
pub type EngineFactory = Box<dyn FnOnce(Serve) -> Result<(), TtError> + Send>;
/// What to open for a device nobody attached explicitly.
pub fn set_default_engine(f: impl Fn(TtDevice) -> Result<EngineFactory, TtError> + Send + Sync + 'static);
/// Explicit, fallible, configurable: the `init_setup` analogue. Idempotent for an
/// identical config; a conflicting config on an attached device is `TtError::AlreadyAttached`.
pub fn init(device: TtDevice, cfg: TtConfig) -> Result<(), TtError>;
```

* `server::run` (`server.rs:378-394`) stops panicking on "not attached": it calls
  `ensure_attached(device)`, which takes the `ATTACHED` lock, and if absent builds the
  factory from the registry -- default: `kmd_engine` on `/dev/tenstorrent/{chip}` if the
  node exists -- and attaches, holding the guard in a process-lifetime table (statics are
  never dropped; the driver's cleanup write already covers process exit, `tt-kmd/src/lib.rs:21-23`).
* Failure is remembered per device (`Failed(TtError)`), so the second op fails fast with the
  same message instead of re-opening a card 100 times; `init` clears it.
* `tt-tests` registers the ttsim factory once (a `ctor` or the existing `with_device`),
  so `burn-tt` still never depends on the simulator (`check-no-sim-in-ship` unchanged).
* `attach`/`AttachGuard` stay public for scoped use (tests, benchmarks); an explicit attach
  of an auto-attached device is refused with a message naming `init`.
* `TtConfig` replaces positional `(route, fidelity)` arguments: `tiles`, `route`,
  `fidelity`, `budget`, `topology`, read from env (`TT_TILES`, `TT_TOPOLOGY`) when unset.
  **Default tiles: `TileChoice::All` or a measured best count** -- not `Exactly(3, 4)`
  (`topology.rs:36`), which is a coordinate from the two cards in this lab and fails with
  `NoSuchTile` on a card harvested differently. Measured: `[512,512]^2` is 1.32 ms on 8 tiles
  and 2.0 ms on 120 (checklist 9.7c), so "all" is not automatically best -- see Q2.
* Card exclusivity: `Session::open_card` takes a driver lock (`Kmd::lock`,
  `tt-kmd/src/lib.rs:341-349`) per card and returns `TtError::CardBusy { card }` if another
  process holds it. Today a second process silently shares the chip.

**Enumeration.** `TtDevice::enumerate() -> Vec<TtDevice>` (numeric `/dev/tenstorrent/*`),
used by `device_count` and by `burn-train`'s multi-device strategy. Keep `type_id = 0` for
cards; reserve `type_id = 1` for "a mesh as one device" if Q5 decides to keep that shape.

**Gate.** `burn-tt/tests/server.rs` (host engine registered as default):
`Tensor::ones([64,64], &TtDevice::default()).matmul(..)` works with no `attach`; a
registry that fails gives `TtError` text on the first *and* second op with one factory call;
a `[0x7FFF]` device with no node gives `NoSuchCard { card, present: [0, 1] }`. ttsim: the
same through `tt-tests`. Silicon: `tt-mnist` with its attach code deleted still trains.

**Implemented (B2, 2026-10-11).** As designed, with these differences: the registry lives in
`burn-tt/src/registry.rs` and the attach table in `server.rs` (`ensure_attached`, one `OPENING` lock so
a first use calls the factory once, failures remembered per device); the default tile is
`TileChoice::First`, since a harvest-proof tile *count* needs the per-workload choice recorded in the
roadmap; `detach(device)` was added; the positional `attach_topology*`/`kmd_engine*` functions were
replaced by `attach_with(device, TtConfig)`; `supports_dram` is cached in the attachment record; an
attachment made before a `fork` is treated as absent (a forked child has none of the parent's server
threads); and the lock is `Kmd::lock_card` (driver lock 63) taken in `Session::open_card`, so every
session, not only burn-tt's, owns its card. The gate is `burn_server.rs`, `step148_lazy_attach`,
`silicon_card_lock` and `silicon_lazy_attach`.

### 2.3 Dtypes

*The "burn-tt" paragraph below is the 2026-10-01 state; BF16 and FP16 storage and native casts
since landed (see 2.1 for the current `dtype_usage`), and unsupported dtypes now fail
explicitly instead of running on Flex.*

**Burn 0.21.** `DType` and `FloatDType`/`IntDType` (`burn_std`); a tensor's dtype comes
from `TensorCreationOptions` or the device default (`reg:burn-tensor-0.21.0/src/tensor/api/base.rs:1942-1954`).
CUDA supports F32/F16/BF16/Flex32/ints/bool/QFloat (`reg:burn-cuda-0.21.0/src/lib.rs` tests).

**burn-tt.** Device path is F32 only: `is_matrix_f32` (`tensor.rs:299-301`), matmul's dtype
check (`ops.rs:157-161`), `on_device` hard-codes F32 (`tensor.rs:209`). Anything else is
Flex, correctly. The hazard is *silent*: `set_default_dtypes::<TtBackend>(&d, BF16, ..)`,
`tensor.cast(BF16)`, or a BF16 safetensors file (§3.4) puts the *whole model* on the host
with no signal.

**Design.** Honest `dtype_usage` (§2.1); a once-per-(op,dtype) warning on the first device-
capable op that falls back for dtype (via the report, §4.2); `TT_STRICT` makes it an error.
BF16/F16 storage on device is D1; until then, *do not* silently upcast (it changes results
and memory) -- warn with the fix ("`.cast(FloatDType::F32)` or load with F32").

### 2.4 Readback, `TensorData`, transactions

**Burn 0.21.** `into_data` = `try_into_data().expect(..)`, `try_into_data` = blocking read
of `into_data_async` (`base.rs:1898-1918`). `TransactionOps::tr_execute`
(`reg:burn-backend-0.21.0/src/backend/ops/transaction.rs:43-46`) reads many tensors at once;
`Transaction` (`reg:burn-tensor-0.21.0/src/tensor/api/transaction.rs:25`) is used by
`burn-train` metrics (e.g. `gh:crates/burn-train/src/metric/auroc.rs`). CubeCL batches all
reads into one copy (`reg:burn-cubecl-0.21.0/src/ops/transaction.rs:18-...`).

**burn-tt.** `float_into_data` downloads synchronously *while building* the future
(`ops.rs:359-363`); `tr_execute` converts each primitive with `into_flex`, one blocking
download each (`convert.rs:211-221`, `ops.rs:477-481`).

**Design (B7).** A `server::download_many(device, ids)` job (one queue round trip; the engine
can already read many buffers in one session) used by `tr_execute`; `float_into_data`
returns `async move { .. }` that sends the job and awaits a oneshot, so `into_data_async`
is truly async and errors come back as `ExecutionError` rather than a panic. **Gate:** a
transaction of 8 resident tensors is one `device_time("download")` call; a failing engine
yields `Err(ExecutionError)` from `try_into_data`, not a panic.

### 2.5 `TensorPrimitive`, `to_device`, multi-device

`TensorPrimitive::{Float, QFloat}` is handled in `convert.rs:171-197`. `float_to_device`
(`ops.rs:348-357`) re-tags via the host and, for an F32 matrix on a GDDR engine, uploads --
"`to_device` is how a caller says this lives on the card", which is the right semantic and
matches CUDA's eager upload. Int/bool `to_device` only re-tag (`ops.rs:406-411,427-432`).
Cross-card copies go through the host; Ethernet P2P belongs to checklist 9.11.
`float_matmul` with operands on different devices `assert!`s (`ops.rs:152-156`) -- keep the
refusal, but as a typed message naming both devices and the `to_device` fix (Burn's own
`TensorCheck` does not cover device mismatch).

### 2.6 `QTensorOps`

*Pre-cutover (2026-10-01):* Flex's, on the host (checklist Phase 7: out of scope); `TtQTensor`
wrapped `FlexQTensor`; `dtype_usage(QFloat)` was `Storage`, as Flex returned.

*Current (2026-10-10):* `TtQTensor` carries only shape, scheme and device. The generated
`QTensorOps` impl fails explicitly for quantized construction and compute, and
`dtype_usage(QFloat)` is empty. Per-tensor BFP8/4/2 storage of logical F32 exists, but it is
not Burn's portable quantization and is not routed through `QTensorOps`
([mixed-bfp-storage.md](mixed-bfp-storage.md)). Portable `QTensorOps` remains deferred.

---

## 3. Composition

### 3.1 `Autodiff<B>`

Works today (`step11_burn::a_linear_layers_gradients_are_within_the_bound`, `step12_mnist`,
`tt-mnist`). Clones share both copies of a cell, so what the forward pass uploaded the
backward pass finds (`tensor.rs:120-127`). Two hazards:

* **The `distributed` feature.** `impl AutodiffBackend for Autodiff<B, C>` requires
  `B: DistributedBackend` when `burn-autodiff/distributed` is on
  (`reg:burn-autodiff-0.21.0/src/backend.rs:155-156`), and `burn/distributed`,
  `burn-core/distributed`, `burn-optim/distributed` and `burn-train/ddp` all switch it on
  (`reg:burn-0.21.0/Cargo.toml:111-119`, `gh:crates/burn-train/Cargo.toml`). A user who
  enables DDP gets a trait-bound error on `Autodiff<TtBackend>` that names neither crate.
  Fix in code: implement `DistributedBackend` (B14). Its defaults are mostly usable;
  `all_reduce` and `sync_collective` are `unimplemented!()` by default
  (`reg:burn-backend-0.21.0/src/backend/distributed/ops.rs:104-119`) and need a host
  implementation first (download, sum, upload), Ethernet later. Note
  `distributed/ops.rs:1` imports `cubecl::device::DeviceId`; whether the feature builds
  without a cubecl backend feature must be checked when wiring it (Q6).
* **Checkpointing** (`Autodiff<B, BalancedCheckpointing>`) recomputes forward ops; it is
  exercised by the conformance suite's `checkpointing` module
  (`gh:crates/burn-backend-tests/tests/common/autodiff.rs`), which is how we learn whether
  recomputation interacts badly with residency (it should not: recomputed ops see resident
  inputs).

### 3.2 `Fusion<B>` -- step by step

**Burn 0.21.** `Fusion<B: FusionBackend>` (`reg:burn-fusion-0.21.0/src/backend.rs:21`);
`FusionBackend: BackendIr<Handle = FusionHandle<R>, Device = FusionDevice<R>>` with
`FusionRuntime`, `cast_float`, `FullPrecisionBackend` (`backend.rs:198-209`);
`FusionRuntime { OptimizationState: Serialize + DeserializeOwned, Optimization,
FusionHandle: Clone + Send, FusionDevice: DeviceOps; fn fusers(device) }` (`182-194`);
`OperationFuser<O> { fuse(&OperationIr), finish, reset, status, properties{score, ready},
len, clone_dyn }` (`130-149`); `Optimization<R> { execute(&mut Context<Handle>,
&OrderedExecution<R>), to_state, from_state }` (`164-172`). The fusion server is an
*upstream* `DeviceService` on its own runner thread (`reg:burn-fusion-0.21.0/src/client.rs:24-37`),
so graph capture overlaps the downstream backend. `BackendIr` (`reg:burn-ir-0.21.0/src/backend.rs:17-38`)
is eight conversions between primitives and one `Handle` type. Reference implementation:
`reg:burn-cubecl-0.21.0/src/fusion.rs:103-186` (`BackendIr`, `FusionCubeRuntime`, fusers
`ElementWiseFuser`, `MatmulFuser`, `ReduceFuser`, `ReduceBroadcastedFuser`, and a
`FallbackOperation` wrapper for ops a fused kernel cannot absorb).

**Steps for `burn-tt`.** *(Detailed specification and checklists in [`kernel-fusion.md`](kernel-fusion.md); architectural ground truth in [`kernel-fusion-architecture.md`](../learnings/kernel-fusion-architecture.md)).*

1. **`impl BackendIr for TtBackend`** (B12). `Handle` must be one type for all four
   kinds; `TtTensor` covers three, quantized is `TtQTensor`. Use
   `enum TtHandle { Tensor(TtTensor), Quantized(TtQTensor) }` (CubeCL's single
   `CubeTensor` is the alternative: fold `FlexQTensor` into a `TtTensor` cell variant).
   Gate: `burn-router` with `(TtBackend, Flex)` and a `burn-remote` server on a card host
   pass the delegation battery. This alone unlocks running models on a remote Tenstorrent
   box from a laptop -- a large DX win for little code.
2. **`TtFusionRuntime` with zero fusers** (B13a). `Optimization = TtOptimization` (an empty
   enum to start), `OptimizationState` a serde enum (adds `serde` to `burn-tt`),
   `FusionHandle = TtHandle`, `FusionDevice = TtDevice`, `fusers() -> vec![]`.
   `impl FusionBackend for TtBackend { cast_float -> float_cast + into handle;
   FullPrecisionBackend = TtBackend }`. Every op then streams through the fusion server and
   executes eagerly on `TtBackend`. Gate: the conformance suite (§5) on `Fusion<TtBackend>`
   gives the same bytes as on `TtBackend`; ms/step for MNIST within noise.
3. **Element-wise chain fuser** (B13b). Accepts `float_add/sub/mul{,_scalar}`, `relu`,
   `relu_backward`, row-broadcast add on F32 rank-N (after B5) operands of one shape;
   `finish` emits one `TtOptimization::Eltwise { program, inputs, outputs }`; `execute`
   gets handles from `context.handles.get_float_tensor::<TtBackend>(..)`
   (`reg:burn-ir-0.21.0/src/handle.rs:124`), sends **one** server job, registers outputs
   (`register_float_tensor`, `handle.rs:159`). Depends on a device op that runs a chain in
   one pass: F3/F4 (`record::SFPU`) or a chained `ELTWISE` record. `score` = ops fused;
   `ready` when every input is resident.
4. **Matmul epilogue fuser** (B13c). `float_matmul` -> row `float_add` -> `relu`: the
   Linear+ReLU of every MLP, run as one pass through `Dst` (the plan's "unpack -> math ->
   pack chain"). Depends on S1 (SFPU `ADD_ROW`/`RELU` on the matmul's output tiles).
5. **Reduce fusers** (later): softmax/log-softmax (R2) as one optimization.
6. **`to_state`/`from_state`**: burn-fusion caches optimizations by stream pattern; state
   must rebuild the program from serialised parameters, not hold device ids.

**Gate for 3-5.** Bit-identical to the unfused run (same kernels, same order), fewer server
jobs (`device_time` counts), and the MNIST golden unchanged; `Fusion<TtBackend>` exported
as `burn_tt::Tt` behind a default-on `fusion` feature, mirroring `burn_cuda::Cuda`
(`reg:burn-cuda-0.21.0/src/lib.rs`).

### 3.3 `burn-router`, `burn-remote`, `burn-dispatch`

Router and remote need only `BackendIr` (step 1 above). **`burn-dispatch` is a closed enum**
(`reg:burn-dispatch-0.21.0/src/device.rs:23-63`): code written against `burn::Dispatch`
cannot select `burn-tt` without a fork or an upstream PR adding a `tt` variant (Q4). This
matters because `burn-backend-tests` is written against `Dispatch` (§5).

### 3.4 `burn-train`, `burn-store`, ONNX

* **`Learner`** (`gh:crates/burn-train/src/learner/`): strategies `single`, `multi`, `ddp`
  (`learner/supervised/strategies/`). `single` needs nothing new. `multi` moves gradients
  between devices with `to_device` (host round trip today). `ddp` needs B14. Metrics read
  through `Transaction` (B7 makes that one round trip). `sys-metrics` uses NVML; a
  `burn-tt` metric (tensor traffic and fallbacks per step from the report, §4.2) is the
  analogue, behind a `train` feature.
* **Checkpointing.** `Record` for a tensor is `into_data()` then `from_data(data, device)`
  (`reg:burn-core-0.21.0/src/record/tensor.rs:117-133`): downloads on save (correct; the
  device copy survives), and loads as a *host* tensor at the device's default dtype, uploaded
  on first device use. Correct. Gate: save after N steps, load into a fresh model, continue,
  bit-identical to an uninterrupted run (on ttsim).
* **`burn-store` (safetensors, PyTorch, burnpack)** applies
  `Tensor::from_data(data, (device, snapshot.dtype))` (`reg:burn-store-0.21.0/src/applier.rs:240`)
  -- *the file's dtype*, not the device default. A BF16 Hugging Face checkpoint therefore
  becomes a BF16 model. **Updated 2026-10-12:** it no longer runs on the host: BF16 is a general device dtype (D1,
  D2), so SmolLM2's BF16 checkpoint loads as a BF16 model that computes on the card, with BF16 activations and cache
  (`step167_smollm2` BF16 on card 0, run `1791848957`, `assert_native_model`). Nothing casts on load, so the
  warning sharp edge #5 asked for would name no cast; it is withdrawn.
* **ONNX** moved to `burn-onnx` (`gh:burn-book/src/onnx-import.md:7,98`): generated Rust
  generic over `B: Backend`, weights via `burn-store`. Every op works (Flex); conv/softmax/
  norm models interleave host and device ops, which can be *slower* than pure Flex
  (ping-pong, sharp edge #11). Gate: one small CNN from `burn-onnx` examples runs on
  `TtBackend` matching Flex, with its report printed.

---

## 4. Ergonomics and sharp edges

*Section 4 is the 2026-10-01 design. Its "Today" column predates the native cutover: "host" /
"silent fallback" entries are pre-cutover (those cases now run natively or fail explicitly);
edges 9, 12, 20 and 21 are closed (batched matmul, views, B3); edges 1, 2, 18, 19, 22, 23, 24
are still open as written.*

### 4.1 Inventory: every way a model hits an opaque failure, silent fallback or cliff

Each with today's behaviour, and the handling tier: **(1)** automatic, **(2)** automatic +
once-per-key warning, **(3)** typed early error.

| # | Edge | Today | Fix | Tier |
|--:|---|---|---|:-:|
| 1 | Device not attached | panic in the first device op (`server.rs:384-386`), only for matmul/`to_device`; eltwise on host tensors silently runs on host | lazy auto-attach (B2) -- **done (B2, 2026-10-11)** | 1  |
| 2 | No card / no driver / no permission | `tt-mnist` prints a hint itself (`tt-mnist/src/main.rs:331-346`); library users get `EngineError` text | `TtError::{NoDriver, NoSuchCard{present}, Permission{path}}` whose `Display` carries the hint; panics at first op name the error and `burn_tt::init` -- **done (B1/B2, 2026-10-11)** | 3  |
| 3 | Hard-coded tile `(3,4)` fused off on another card | `NoSuchTile` at attach | default from the chip's grid -- **done (B2: `TileChoice::First`, 2026-10-11)** | 1  |
| 4 | `TT_TILES` default 1 | 1 tile unless the caller reads the env (`topology.rs:108-111`) | `TtConfig` reads env; default many tiles (Q2); `name()` reports it -- **done (B2, 2026-10-11)** | 1  |
| 5 | Non-F32 dtypes (BF16 weights, `set_default_dtypes`, `cast`) | whole model on host, silent | warn once per (op, dtype) with the cast fix; strict = error; D1 removes it for BF16 | 2 |
| 6 | Rank-1 tensors (biases, `[n]` params) | host; uploaded on every use as SGD updates them (checklist 9.5: the six tensors per step) | store rank-1 as `[1, n]` (B5) | 1 |
| 7 | Rank > 2 (`[b, s, d]` activations) | every op host, incl. element-wise | store as `[prod(lead), last]`; same-shape eltwise on device (B5) | 1 |
| 8 | `reshape`/`unsqueeze`/`flatten` | generated delegate -> download (`generated/delegate.rs:336-342`) | view when the last dim is kept; otherwise device copy (D4) or download, reported | 1/2 |
| 9 | Batched matmul | resident `matmul_dram_batched` for tile-aligned blocks; ragged, broadcast and strided batches repack on the device (`materialized_batched_matmul`) -- done | resident batched path (B6) | 1 |
| 10 | Shapes not a multiple of 32 | already invisible: `DramTensor` zero-pads and crops (`tt-kernels/src/tensor.rs:7,237`) | report padding overhead (e.g. `[64,10]` is 3.2x) at info level | 1 |
| 11 | Host<->device ping-pong (device op between host ops) | each crossing is a download + upload; can be slower than Flex | report bytes per op; cost-aware placement (B16, Q3) | 2 |
| 12 | Transposed view into eltwise/sum/slice | native strided views and device materialization, no host fallback (P1a; step66/step84) -- done | materialise the transpose on device (M3) or report; check before upload | 2 |
| 13 | Slice not on whole tile rows | full download of the parent (`ops.rs:331`) -- for a preloaded dataset, the whole dataset | partial download of the rows the slice reaches (small downloads already read only what they touch, checklist 9.4b) then upload; D4 does it on device | 2 |
| 14 | Broadcasts other than `[1,n]` (`[m,1]`, scalars as tensors, rank-N) | host (`ops.rs:80-87`) | report; kernels in S1/D4 | 2 |
| 15 | `sum_dim(1)`, mean, max, softmax, ... | host | report; R1/R2 | 2 |
| 16 | Int/bool ops (labels, masks, argmax) | storage, views and the logic ops on the device since D3 (`hardware-coverage.md` 10.2b); arithmetic host until S5 | report under `Reason::NoDeviceKernel`; *not* warned when operands are host-resident (no cost) | 2 |
| 17 | Mesh topology | all host-staged, no GDDR (`server.rs:630-643`; 233 vs 5.8 ms/step, checklist 9.4b) | warn at attach with the measured cost until 9.11 | 2 |
| 18 | Device error mid-model (timeout, run error) | panic in the op (`server.rs:405-512`) | sticky error -> `ExecutionError` at `sync`/`try_into_data`; strict = immediate typed panic -- **done (B1, 2026-10-11)** | 3  |
| 19 | Engine panics on the server thread | thread dies; every later op panics "server thread has stopped" without the cause (`server.rs:391-393`) | `catch_unwind` around each job; device moves to `Failed(cause)` -- **done (B1, 2026-10-11)** | 3  |
| 20 | Tensor outlives its attachment, device re-attached | `BufferId`s restart at 1 (`server.rs:106-126`): **reads another tensor's data** | `Buffer` carries an attachment generation; mismatch = `TtError::StaleTensor` -- **done (B3, 2026-10-11)** | 3  |
| 21 | `supports_dram` cache keyed by device only (`server.rs:415-432`) | stale after re-attach with a different engine | cache in the attachment record -- **done (B2, 2026-10-11)** | 1  |
| 22 | Operands on different chips | `assert!` (`ops.rs:152-156`) | typed panic message with the `to_device` fix (Burn ops cannot return `Result`) | 3 |
| 23 | Another process on the same card | silent sharing | driver lock, `TtError::CardBusy` -- **done (B2, 2026-10-11)** | 3  |
| 24 | GDDR exhausted | engine error -> panic with allocator text | `TtError::OutOfMemory { requested, live, largest_free }`; `memory_cleanup` then retry once | 3 |
| 25 | `distributed` feature on | `Autodiff<TtBackend>` is not `AutodiffBackend` | implement `DistributedBackend` (B14) | 1 |
| 26 | Determinism | bit-exact run to run (golden); Flex-seeded random | keep; document nothing -- assert it in conformance (seeded random test) | -- |

### 4.2 The mechanism: a fallback report, `tracing`, strict mode

**Where to hook.** The generated delegate (`xtask/src/gen_burn.rs`) emits every forwarding
function; it should emit, before converting arguments, `let _op = crate::report::enter("float_reshape", Reason::NoDeviceKernel);`
-- a guard that sets a thread-local *current op*. `TtTensor::host()` (`tensor.rs:216-246`),
on an actual download, attributes the bytes to the current op; `to_dram` attributes uploads
the same way. Hand-written ops call `report::device(op)` on their device branch and
`report::enter(op, reason)` on each fallback branch with the precise `Reason`. Because the
generator writes the hook, no forwarded op can be missed, and the generator's `--check`
keeps it that way (fits "generated delegation stays").

```rust
pub enum Reason { NoDeviceKernel, DType(DType), Rank(usize), TransposedView,
                  Broadcast { lhs: Vec<usize>, rhs: Vec<usize> }, UnalignedSlice,
                  EngineWithoutGddr, NotResident }
pub struct OpStat { pub op: &'static str, pub reason: Option<Reason>, pub calls: u64,
                    pub on_device: u64, pub downloaded: u64, pub uploaded: u64,
                    pub example_shapes: Vec<Vec<usize>> }
pub fn report() -> Report;            // snapshot, Display = a table sorted by bytes
pub fn report_reset();
pub fn with_report<R>(f: impl FnOnce() -> R) -> (R, Report);  // gates use this
```

`tensor_traffic`, `record_transfers` and `device_time` (`traffic.rs`) become views of the
same store (keep them; gates use them).

**Logging.** Depend on `tracing` (what Burn itself uses: `#[cfg_attr(feature = "tracing",
tracing::instrument(level = "trace"))]` in `reg:burn-cubecl-0.21.0/src/ops/base.rs:58-59`,
`reg:burn-autodiff-0.21.0/src/ops/tensor.rs:52`). Targets:

| Target | Level | What |
|---|---|---|
| `burn_tt::attach` | info | which engine, card, tiles, GDDR, time to open; warn for a mesh (edge 17) |
| `burn_tt::fallback` | warn, **once per (op, reason, dtype)** | `float_sum_dim(dim=1) [64,128] F32 ran on the host: no device kernel (hardware-coverage R1); downloaded 32 KiB` |
| `burn_tt::transfer` | debug | every upload/download with shape and op (replaces `TT_TRACE_FALLBACK`; backtrace at trace level) |
| `burn_tt::pad` | debug | padding overhead per shape |
| `burn_tt::device` | trace | each server job with duration (feature `tracing` adds spans per op, as CubeCL does) |

A user with no subscriber must still see warnings: if `tracing::dispatcher::has_been_set()`
is false, the warn path falls back to one `eprintln!` per key. `TT_TRACE_FALLBACK=1` stays
as an alias for `burn_tt::transfer=trace` so existing workflows keep working. `TT_REPORT=1`
prints the report at exit (a `libc::atexit` hook; libc is already a workspace dependency,
`Cargo.toml:47`).

**Strict mode.** `TT_STRICT=1` or `burn_tt::set_policy(Policy { fallback: Deny, .. })`:
any fallback *that moves bytes* (a download caused by a host op, or a dtype/rank fallback of
a device-capable op) panics with the `OpStat` and the fix. Host ops on host-resident data
(labels, int masks) are not fallbacks. An allowlist scope, `burn_tt::host_ok(|| loss(..))`,
marks intended host work (MNIST's loss until R2). CI and every gate run strict.

**Conformance against residency.** A test helper
`burn_tt::assert_resident(steps, |step| ..)` generalises `step12_mnist`'s per-step transfer
assertion: after warm-up, a step may only move the tensors the test lists. And a
"should be on device" table -- generated from `OVERRIDDEN` -- runs each overridden op on
resident operands under strict mode, so an op whose device branch silently stops matching
(a widened `Reason`) fails CI. This is `hardware-coverage.md` F6's coverage generator
with teeth.

### 4.3 Errors: panics vs `Result`

* `pub enum TtError` (thiserror-style, no new dependency needed): `NoDriver`,
  `NoSuchCard { card, present }`, `Permission { path }`, `CardBusy { card }`,
  `NoSuchTile { .. }` (from `SessionError`), `Timeout { op, budget }`, `Device(String)`,
  `OutOfMemory { .. }`, `StaleTensor`, `AlreadyAttached`, `Fallback(OpStat)` (strict).
  `EngineError` becomes `TtError` (keep a type alias one release).
* Fallible entry points return `Result`: `init`, `attach`, `try_sync` helpers.
* Inside ops (no `Result` in Burn's signatures): device errors are recorded as the device's
  sticky error and the op returns a *poisoned* tensor (a cell whose device copy is
  `Err(TtError)`); `try_into_data`/`sync` return `ExecutionError::WithContext { reason }`,
  matching the documented contract of `try_into_data` (`base.rs:1904-1906`). Using a
  poisoned tensor in a later op propagates the poison without running. Strict mode panics
  immediately instead. This is also the prerequisite for async dispatch (B8): once jobs do
  not block, errors *must* be deferred.
* Hangs: per this repo's rule, a hang is fixed in the API that hung (budgets in
  `tt_kernels::runtime`), not by a watchdog around runs; `Timeout` carries which op and tile.

### 4.3a Exact mode and approximate ops (Phase 10)

Some device ops are approximations held to derived bounds rather than to Flex's bits
(`hardware-coverage.md` S3, S4, R1, R2): division and the reciprocal (one ulp), `exp`,
`log` and the rest of S4's transcendentals (10.2d-f, trigonometry included), sums over
columns (tree order), softmax. (Historical: the exact mode, `burn_tt::set_exact` and `TT_EXACT`, was retired; ops are native within derived bounds.) Full `sum`/`mean` are native
only and retain their device arithmetic in this mode, within derived bounds
rather than guaranteeing Flex's bits. The MNIST device golden pins that policy.
Which legacy ops are which is data:
`tt_kernels::sfpu::ops::accuracy` -- S2's compare, select and sign ops are exact and run
on the device in exact mode and at any size (`hardware-coverage.md` 10.2c). Below eight tiles the
approximate ops ran on the host whatever the mode until 2026-10-03 (`APPROX_MIN_TILES`, removed: data stays on the card; measured: their
fixed cost outweighs a download) -- a placement heuristic B16 should replace.

### 4.4 `sync`, async dispatch, determinism

Today every job blocks the calling thread (`server.rs:378-394`), so `sync` is trivially
correct and each op pays a queue round trip. CUDA's model is enqueue-and-return. B8:
assign `BufferId`s on the client (a per-attachment atomic counter), so `matmul_dram`,
`eltwise`, `sum_rows`, `slice_rows` enqueue and return immediately with shapes computed
host-side (they are deterministic from the inputs); only downloads, `sync` and
`device_traffic` wait. This overlaps Burn's host-side work (autodiff graph, Flex ops) with
device work, and composes with 9.7's one-launch records. Determinism is unchanged: one
queue per device keeps program order. Gate: the MNIST golden bit for bit; ms/step measured.

### 4.5 Feature flags and env vars

*Proposal (2026-10-01).* Current state: `burn-tt` has features `std` and `fusion` (both default;
`fusion` provides `Tt = Fusion<TtBackend>`). There is no `tracing`, `distributed`, `train` or
`strict` feature. Of the proposed environment variables, `TT_STRICT`, `TT_REPORT`,
`TT_TRACE_FALLBACK`, `TT_TILES` and `TT_TOPOLOGY` exist; `TT_DEVICE` and `TtConfig` do not.

| Flag | Default | Effect |
|---|:-:|---|
| `fusion` | on (after B13a) | `burn_tt::Tt = Fusion<TtBackend>`, as `burn_cuda::Cuda` |
| `tracing` | off | per-op spans (warnings are always on) |
| `distributed` | on | `DistributedBackend` (avoids edge 25 by construction) |
| `train` | off | `burn-train` metric for traffic/fallbacks |

Env vars, all read once into `TtConfig` and echoed by `name()`/the attach log:
`TT_TILES`, `TT_TOPOLOGY` (existing), `TT_STRICT`, `TT_REPORT`, `TT_TRACE_FALLBACK`
(alias), `TT_DEVICE` (default card, like `CUBECL_WGPU_DEFAULT_DEVICE`).

---

## 5. Testing strategy

### 5.1 How Burn's backends run the shared suite

`burn-backend-tests` (`gh:crates/burn-backend-tests`, unpublished) is ~404 test files:
`tests/tensor.rs` (float/int/bool op suites, `clone_invariance`, `multi_threads`),
`tests/tensor_f16.rs`, `tests/autodiff.rs` (75 files, with and without checkpointing),
`tests/autodiff_f16.rs`, plus cube/fusion-only binaries. The backend is chosen by cargo
feature through **`burn-dispatch`**: `common/backend.rs` defines
`pub type TestBackend = burn_dispatch::Dispatch`, sets default dtypes for
`DispatchDevice::default()` in a `#[ctor]`, and the autodiff suite builds its device with
`DispatchDevice::autodiff(DispatchDevice::default())`
(`gh:crates/burn-backend-tests/tests/common/{backend,autodiff}.rs`). Aliases:
`test-flex = "test --release --no-default-features --features flex,std"`,
`test-cuda = ".. cuda,std,fusion"` (`.cargo/config.toml`). There is no per-backend `testgen!`
instantiation macro in 0.21; `burn-tensor-testgen` only provides the `#[might_panic]`
attribute (`gh:crates/burn-tensor-testgen/src/lib.rs:44-45`).

### 5.2 How `burn-tt` runs it (B10)

Implemented as `crates/burn-tt-conformance` and `cargo xtask conformance` (see the crate's README).
`burn-dispatch` cannot name `TtBackend` (closed enum), so the suite is run from our own entry
binaries:

1. `cargo xtask fetch-burn-tests` downloads the `v0.21.0` tarball and keeps
   `crates/burn-backend-tests/tests/` (404 files) in `vendor/burn-backend-tests/` (gitignored, like
   the other pinned inputs), pinned in `PINS.toml` by the SHA-256 of the extracted tree, not of the
   tarball (GitHub does not promise it is byte-stable). It is fetched rather than copied into the
   crate: it is third-party, and the digest check is what makes "passes" mean the same thing twice.
2. Three entry binaries replace upstream's `common/backend.rs` and `common/autodiff.rs`:
   `tensor` (`TestBackend = TtBackend`), `autodiff` (`Autodiff<TtBackend>`) and
   `autodiff_checkpointing` (`Autodiff<TtBackend, BalancedCheckpointing>`), with
   `AutodiffDevice::new() -> TtDevice`. Autodiff is a *type* here and a property of a
   `DispatchDevice` upstream, so the checkpointing run is a third binary, not a second module of
   one. The vendored modules are `#[rustfmt::skip]`: `cargo fmt` follows `#[path]` modules and
   would rewrite the pinned files.
3. `crates/burn-tt-conformance/expected-failures.toml` groups the failing tests by reason, each group
   with a category. The runner fails on an unlisted failure, on a listed test that passes, on an
   entry whose test is gone, and on a `todo` entry. Empty is the goal.
4. **No eager-upload policy was needed.** The design assumed test inputs start on the host, so ops
   would run there. Measured: `float_from_data` stays on the host until the first op, which then
   uploads its operands and computes on the device (`TT_REPORT=1`: `float_add` 1 call, 1 on the
   device, 48 B up). The suite exercises the path a user's code takes.
5. **One process per test**, not one simulator per binary: a process costs about 20 ms and 26 MB, and
   an isolated simulator means one failure never poisons the next (Q1 answered). Upstream's own
   `#[ignore]`d tests are not run.
6. The runner sets `TT_REPORT=1`, so each test's output carries the residency table (the
   coverage census); `--strict` adds `TT_STRICT=1`. (`TT_STRICT=report` does not exist: strict mode
   is `1` or off.)

**Gate (validation tier, ttsim):** `cargo xtask conformance --release` matches the expected-failures
list; F32 tensor and both autodiff binaries first, F16 and `--fusion` are done (D1, B13a).
Plain `cargo test --workspace` neither builds nor runs it (every test target needs the `run`
feature).

### 5.3 Silicon smoke tier (B11)

`SMOKE` (`xtask/src/silicon.rs`) lists whole upstream modules that have no expected failures:
matmul, element-wise, relu, aggregation (`sum_dim`..), transposes/permute/reshape/expand,
`slice_assign`, int add/matmul/slice, `multi_threads`, and the autodiff `matmul`/`relu`/`add`/`slice`.
The silicon runner builds `burn-tt-conformance` with its `silicon` feature beside `tt-tests`, and
runs these on card 0 only (`TtDevice::default()`), skipping them on other cards. The suite's literals
are the ones Burn's CPU backend meets, so a pass is a comparison with it. Full-suite runs on silicon
are occasional, not smoke.

---

## 6. Roadmap

Ordered; each item's done-criterion is its gate. `HC:` = `hardware-coverage.md` item.

| Id | Item | Depends | Done when |
|---|---|---|---|
| **B0** (report, strict, `host_ok`, `with_report`, `TT_REPORT`/`TT_STRICT` done 2026-10-03: `burn-tt/src/report.rs`, a guard the generator emits in every op; `tracing` targets, per-`Reason` attribution and the once-per-key warnings open) | Report + `tracing` + strict mode (§4.2); generator emits the hook | -- | every delegated op appears in `report()` with bytes; `TT_STRICT=1` turns `step12_mnist`'s known host ops into failures unless in `host_ok`; `TT_TRACE_FALLBACK` still works |
| **B1** (done 2026-10-11: `burn-tt/src/error.rs`; `EngineError` removed; `try_into_data`/`sync` return the sticky error, strict mode panics, `catch_unwind` per server job; `NoDriver`/`NoSuchCard`/`Permission` classify opening a card, `CardBusy` waits for B2's lock) | `TtError`, sticky errors, poisoned tensors, `catch_unwind` on the server thread (edges 2, 18, 19, 22, 24) | -- | `a_host_staged_engine_error_panics_in_the_op_with_it`, `an_engine_error_is_try_into_data_s_error`, `strict_mode_panics_where_try_into_data_would_return_the_error`, `a_panicking_engine_is_reported_by_every_later_wait` (`tt-tests/tests/burn_server.rs`) |
| **B2** (done 2026-10-11: `burn-tt/src/registry.rs`, `server::ensure_attached`; the default tile is `TileChoice::First`, not a measured count -- the per-workload tile optimizer is a Track 2 item in the roadmap; the card lock is `Kmd::lock_card` taken by every `Session::open_card`, `SessionError::CardBusy` -> `TtError::CardBusy`; `kmd_engine`, `attach_topology*` and `kmd_mesh_engine` are gone, `attach_with(TtConfig)` replaces them; an attachment from before a `fork` is treated as absent) | Engine registry, lazy auto-attach, `init`/`detach`/`TtConfig`, default tile, card lock, `device_count`/`enumerate` (edges 1, 3, 4, 20, 21, 23) | B1 | `burn_server.rs` (registry, remembered failure, `init`, refusal, `enumerate`), `step148_lazy_attach` (ttsim, residency, fork negative control), `silicon_card_lock`, `silicon_lazy_attach` (smoke) |
| **B3** (done via never-reused process-wide ids, 2026-10-03) | Stale buffers and capability cache scoped to the attachment; ids are translated by that server | -- | both stale-buffer gates pass; old tensors cannot read or free a new attachment's buffer |
| **B4** (half done: `dtype_usage` reports `F32`/`BF16` as `Accelerated`, `I32`/`Bool` as `Storage | Arithmetic` (so `set_default_dtypes(.., I32)` works, found by B10) and `F16` as `Storage | Arithmetic` (`lib.rs`, 2026-10-12: arithmetic widens to F32 and rounds once; not `Accelerated`, there is no packed F16 matmul); the once-per-(op, dtype) warnings are missing; the `burn-store` BF16 warning is withdrawn, 2026-10-12: a BF16 checkpoint loads as a BF16 model and computes on the device, so there is no cast to name) | Honest `dtype_usage`; dtype fallback warnings (edge 5); BF16 safetensors warning naming the cast | B0 | a BF16 file loads as BF16 and computes on the device under `assert_native_model` (`step167_smollm2` BF16, run `1791848957`) |
| **B5** (storage, views and element-wise done: `hardware-coverage.md` P1a; partial-row download open -- measured: a 1000-row batch of a resident set is a host slice and a 3 MB re-upload every batch, 46 ms a batch against 2.1 on the host, row X) | Rank-N and rank-1 storage as `[prod(lead), last]`; `reshape`/`unsqueeze`/`flatten` keeping the last dim as views; partial-row download for unaligned slices (edges 6, 7, 8, 13) | B0 | MNIST steady step moves only `dL/dlogits` and the logits (biases and SGD stay resident, needs `float_sub` with `mul_scalar` on `[1,n]`); a `[b,s,d]` element-wise chain downloads nothing |
| **B6** (done 2026-10-03 for tile-aligned operands: a rank-N lhs against an unbatched rhs -- every `Linear` -- folds into the 2-D product, as do `linear_{weight,bias}_backward`; a real batch on both sides runs as `Session::matmul_dram_batched` over blocks of the operands' buffers, reshapes and swaps being strided views (`burn-tt/src/views.rs`); untiled shapes are still host-staged) | Batched matmul on resident operands (edge 9) | B5 | `[8,64,64]@[8,64,64]` under strict: zero downloads; bit-identical to per-batch host-staged |
| **B7** | `download_many`, real readback futures, batched `tr_execute` | B1 | one server job per transaction |
| **B8** (asynchronous calls done 2026-10-03; `Backend::sync` barrier done 2026-10-11 (`server::try_sync`, B1); batched readback is B7: `server::submit` -- the caller names each result by a process-wide id and computes its shape, the server thread translates ids (`server::Ids`), only waits wait; a failed op poisons its result and the attachment's next wait. Per call 30-40 us -> 0.2 us; MNIST 1.8 -> 1.4 ms/step, the transformer 11.1 -> 8.6) | Async dispatch with client-assigned ids; `sync` as barrier. Measured (2026-10-01, `ttsim-divergence.md` row Z): each call's round trip to the server is 32-49 us, ~0.2 ms of a 0.61 ms batch-64 inference and ~0.7 ms of a 2.2 ms training step | B1, B3 | MNIST golden bit for bit; ms/step recorded; `tt-mnist --infer`'s per-call breakdown shows the calls returning without the round trip |
| **B9** | `name`, `memory_cleanup`, `memory_persistent_allocations`; OOM retry | B2 | unit tests §2.1 |
| **B10** (done 2026-10-11: `burn-tt-conformance`; no eager policy was needed, see 5.2) | Fetched, digest-pinned conformance suite on ttsim, expected-failures | B2, B0 | §5.2 gate: `cargo xtask conformance --release` |
| **B11** (done 2026-10-11: 163 tests, card 0) | Conformance subset in `SMOKE` (curated modules, not generated) | B10 | `cargo xtask silicon --smoke` runs it on card 0 |
| **B12** (partial: `impl BackendIr for TtBackend` with `TtHandle` exists, `tensor.rs`; the router/remote gate has never run -- neither `burn-router` nor `burn-remote` is a dependency or a test anywhere in the workspace) | `BackendIr` (`TtHandle`); router + remote gates | -- | `burn-remote` server on a card host passes the delegation battery from a client |
| **B13a** (partial: `FusionBackend`, `TtFusionRuntime`, `burn_tt::Tt` and the default-on `fusion` feature exist; `step12_mnist` exercises `Tt`; "zero fusers" was superseded by the MVP fuser of B13b; conformance on `Fusion<TtBackend>` ran 2026-10-12: 1,803 pass, 35 listed, `expected-failures-fusion*.toml`; three bool-scalar tests die in burn-fusion 0.21.0, so "green" is open on that upstream `todo!()` alone) | `FusionBackend`, zero fusers, `burn_tt::Tt` alias, `fusion` feature | B12, B10 | conformance green on `Fusion<TtBackend>` |
| **B13b** (MVP only: `ElementWiseFuser` matches `add` + Burn's lowered ReLU as one `ADD_RELU` dispatch and a lone ReLU as `RELU`; no general chain fuser, no micro-op engine; see [kernel-fusion.md](kernel-fusion.md)) | Element-wise chain fuser | B13a, HC:F3/F4 (or chained `ELTWISE`) | one job per chain; golden unchanged |
| **B13c** | Matmul + bias + ReLU epilogue fuser | B13b, HC:S1 | Linear+ReLU is one device job |
| **B14** | `DistributedBackend` (host all-reduce), `distributed` feature on by default | B2 | `burn-train` DDP over two cards trains MNIST (slowly); Ethernet all-reduce with checklist 9.11 |
| **B15** (partly done 2026-10-12: strict `burn-store` safetensors load of F32 and BF16 checkpoints, `tt-models/src/load.rs`; the checkpoint round trip with Burn's `BinFileRecorder`, `step170_checkpoint_round_trip`; three pretrained models against Flex, `step165` to `step168`. Open: a real `Learner` run, one `burn-onnx` model) | Integration gates: `Learner` run with checkpoint round trip; `burn-store` safetensors load (F32 + BF16); one `burn-onnx` model | B0, B2, B4 | each matches Flex; reports archived |
| **B16** (proposed) | Cost-aware placement/lookahead for isolated ops and model-level decisions; the old size thresholds were removed 2026-10-03, so supported approximate ops now follow resident data at every size (exact mode is retired) | B0 report data | evaluate end-to-end models and PCIe traffic; full loss mean is now resident (R1b), while other shape fallbacks and host-created operands remain |

Coverage work proper (S1-S9, M1-M4, R1-R4, D1-D6) proceeds in parallel; each landed item
shrinks the report and the expected-failures list, and B0's report is how its "downloads
nothing" line in the Definition of done is checked.

---

## 7. Open questions

1. **ttsim and the shared suite.** One simulator per test binary (fast, shared state,
   one crash kills the binary) or a process per test (isolated, pays simulator start each
   time)? Measure simulator open time before choosing; `fork_scope` per test is the
   repo's current pattern.
2. **Default tile count.** "All surviving tiles" is slower than 8 for today's MNIST and
   `512^3` past ~64 (checklist 9.6/9.7c). A per-op tile choice inside the session (by job
   count) would make the default irrelevant; until then pick a measured default.
3. **Placement policy.** "Device op whenever possible" (today: matmul always uploads,
   element-wise follows the data) versus a cost model using tensor bytes and FLOPs. A
   lookahead exists only under `Fusion` (B13); without it, a simple rule (upload only if an
   operand is resident or the op's work exceeds a measured threshold) needs a decision on
   whether that counts as a "silent fallback" (it would be reported).
4. **`burn-dispatch` upstream.** Is a `tt` variant upstream worth pursuing so
   `burn::Dispatch` users and `cargo test-tt` work without vendoring?
5. **What is a Burn device on a multi-card host:** each card a `TtDevice` (Burn's model,
   DDP-friendly) or the mesh as one device (today's `Topology::Cards`)? Possibly both, as
   distinct `type_id`s.
6. **`burn-backend/distributed` without a cubecl feature** -- `distributed/ops.rs:1`
   imports `cubecl::device::DeviceId`; compile-check before making `distributed` default-on.
7. **Poisoned tensors vs. Burn's expectations.** Does any Burn wrapper (autodiff,
   fusion) read tensor *metadata* in a way that a poisoned cell must still satisfy? Shapes
   are known, so likely fine; confirm under the conformance suite with an injected engine
   error.

### M1 matrix elementwise continuation (2026-10-06)

The checklist and its acceptance record are kept once, in
[tensix-next-features.md](../completed-plans/tensix-next-features.md#m1-matrix-elementwise-continuation-2026-10-06);
the kernel-level checklist is in [hardware-coverage.md](../completed-plans/hardware-coverage.md). Matrix-mode
elementwise is an explicit opt-in (`ElementwiseMode::Matrix`); SFPU remains the default.
