# tt-metal concepts review: what we are missing

A review of this stack against tt-metal (TT-Metalium, LLK, TTNN), as of 2026-10-01 (after
9.7c). It asks which **Tenstorrent-system concepts** tt-metal is built around that we lack,
have only in part, or do differently, and how to close each gap here. It is a guide, not a
tracker: work it creates is ticked in `implementation-checklist.md` (Phase 9) or
`hardware-coverage.md` (Phase 10, items F1..F6, S1..S9, M1..M4, R1..R4, D1..D6), which
this file cites rather than repeats.

Sources. tt-metal paths are relative to the local install
`~/.tenstorrent-venv/lib/python3.12/site-packages/ttnn/tt_metal/` (written `tt_metal/`) and
its sibling `ttnn/ttnn/cpp/ttnn/` (written `ttnn/`). GitHub documents are under
`https://github.com/tenstorrent/tt-metal/blob/main/` (`METALIUM_GUIDE.md`,
`tech_reports/{tensor_layouts,tensor_sharding,matrix_engine,data_formats,GEMM_FLOPS,
memory,AdvancedPerformanceOptimizationsForModels,MetalProfiler,Debugging,TT-Fabric,
EthernetMultichip,Saturating_DRAM_bandwidth,Handling_Special_Value}/`). Every claim about
this repo was checked against the code. The one measured figure (the K limit, G3) came
from a scratch probe that called `matmul::chunk_fits_in` and `plan_in`.

---

## 1. Summary

Status: **have** · **partial** · **different** (different by design, and fine; see §4) ·
**missing**. Impact: C correctness, P performance, X developer experience. Priority: P0
before Phase 10's first new op, P1 alongside Phase 9/10, P2 when a model needs it.

| tt-metal concept | What tt-metal does | Ours | Status | Impact | Pri |
|---|---|---|---|---|---|
| Logical vs padded shape, pad value | `Tensor` carries `logical_shape` and `padded_shape`. Tile padding is undefined unless an op fills it (`ttnn/operations/data_movement/fill_pad`, `kernel_helper_functions/pad_tile.hpp`) | `DramTensor` stores only `rows, cols` and assumes zero padding (`tensor.rs:6-8`). Nothing tracks or restores it | **partial, with a live bug** (G1) | C | **P0** |
| Rank > 2 tensors in tile layout | `[..., H, W]`, each leading slice padded to `ceil32(H)`, tiles over the last two dims (`tech_reports/tensor_layouts`) | `DramTensor` is 2-D. `burn-tt` sends anything not `is_matrix_f32` (`burn-tt/src/tensor.rs:183`) to the host | **missing** (G2) | P, C | **P0** |
| Row-major vs tile layout | `Layout::ROW_MAJOR` / `TILE`, with `tilize`/`untilize` ops on the device | Host tilize only (`tt-layout`, `matmul::tilize_f32` `matmul.rs:452`). GDDR holds tiles only | partial (D5, 9.10) | P | P1 |
| K blocking (`in0_block_w`), spill or reload, packer L1 accumulation | Any K: partial sums go to an intermediate CB and are reloaded into Dst (`copy_tile`) or accumulated by the packer (`PACKER_L1_ACC`) (`ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm.cpp:38-107`) | All of K must fit in L1 at once. `matmul_dram` refuses a split (`tensor.rs:478-481`) and `burn-tt` panics (`server.rs:511`) | **missing** (G3) | C (panic) | **P0** |
| Dst capacity, subblocks, `SyncHalf` | 16 tiles of 16-bit Dst, 8 at FP32. Half of that per `tile_regs_acquire` in `SyncHalf`. `out_subblock_h*w` ≤ capacity (`tt-llk/tt_llk_blackhole/common/inc/ckernel.h:836-840`; `tensix_types.h:191-193`) | One output tile in Dst at a time, all of Dst cleared for each (`matmul.rs:333-338, 375`) | partial (G4) | P | P1 |
| Circular buffers (page FIFO) | `cb_reserve_back/push_back/wait_front/pop_front` over 16-bit page counters, `NUM_CIRCULAR_BUFFERS` = 64 on BH (`tt_metal/api/tt-metalium/circular_buffer_constants.h`; `hw/inc/api/dataflow/dataflow_api.h:208, 474`) | Declared and planned (`tt-kernels/src/l1.rs:201`). No runtime: one-shot gather, compute, scatter | partial, designed (G5; 9.8) | P | P1 |
| Double buffering | CB of 2× block pages; reader fills one half while compute drains the other | None | missing (G5; 9.8) | P | P1 |
| Reader / compute / writer split on two NoCs | BRISC and NCRISC each run a data-movement kernel, by default on different NoCs; TRISC0..2 run compute | B core does all movement on NoC 0. NC unused (`tt-isa/src/l1.rs:35`) | partial (G6) | P | P1 |
| Multicast (`noc_async_write_multicast`, mcast matmul in0/in1) | 1D/2D mcast matmul: in0 read once per core row and multicast across, in1 likewise per column (`.../dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp:340-373`) | None. Every block re-reads its operands from GDDR | missing (G7) | P | P1 |
| L1 ("global") semaphores, NoC atomics | `CreateSemaphore`: L1 words, `noc_semaphore_inc` / `set_multicast` / `wait` across cores | Only the Tensix Sync Unit's 8 per tile (`sync.rs:43`), planned by `Requirements::semaphore` | missing (G7) | P | P1 |
| `CoreRange` / `CoreRangeSet` | Rectangles of logical cores: placement, mcast destinations, per-range args | `TileChoice` (`session.rs`), a count of the first `n` tiles. No rectangles | partial (G7) | P, X | P1 |
| Harvesting, coordinate spaces | logical / virtual (translated) / physical | Newtypes per space (`noc.rs`), grid read from ARC, a fused-off tile refused (`session.rs:1-20`) | **have** | -- | -- |
| Command queues, fast dispatch | Host writes commands into a hugepage ring. Prefetcher and dispatcher cores pull them, write worker L1, send GO signals, and report completion into host memory (`tt_metal/impl/dispatch/kernels/cq_commands.hpp:19-62`) | Every op is synchronous: the host writes over uncached BAR MMIO and polls `DONE` by MMIO read, wave by wave (`session.rs:666-728`; `dm.rs:232-257`) | **missing** (G8) | P | P1 |
| Non-blocking enqueue, events | `EnqueueProgram(blocking=false)`, `EnqueueRecordEvent` / `EventSynchronize`; the host runs ahead | Each Burn op blocks on `rx.recv()` (`server.rs:380-394`) | missing (G8) | P | P1 |
| Traces | `BeginTraceCapture`/`ReplayTrace`: a step's commands kept in DRAM, replayed by `CQ_PREFETCH_CMD_EXEC_BUF` (`cq_commands.hpp:27`) | None. Op records (9.7b) are the raw material | missing (G8) | P | P1 |
| Host pinned memory (sysmem) | Issue and completion queues in pinned host memory, reached by the device over PCIe (`PIN_PAGES`) | `tt-kmd` has no `PIN_PAGES`. All traffic is host MMIO (`RUST_IMPL_PLAN.md:234-237` names the ioctl) | missing (G8) | P | P1 |
| Program cache, program hash | `compute_program_hash` → cached `Program`; on a hit only `override_runtime_arguments` runs | Programs memoised by shape (`matmul.rs:906-925`), resident in L1 keyed by words (`program_cache.rs:70`), DRAM addresses carried in records | **have** (different form, §4) | -- | -- |
| Compile-time vs runtime args | CT args baked into the kernel binary; RT args written per core per launch | Programs bake L1 addresses; records carry DRAM refs and scalars (`dm/record.rs`). No per-launch arguments for role programs | partial (F3's `SETDMAREG`) | P | P2 |
| Interleaved DRAM buffers, page size | Page = one tile in its format, round-robin over banks at the **same offset** in every bank | Tile slot of 4160 B (`dm.rs:149`), round-robin over channels at **per-channel** bases (`tensor.rs:6-12, 213-228`) | **have** (different form, §4) | -- | -- |
| Sharded tensors (height/width/block), L1 memory config | `MemoryConfig{L1/DRAM, INTERLEAVED/*_SHARDED}`: activations live in L1 across cores between ops (`api/tt-metalium/buffer_types.hpp:11-39`; `tech_reports/tensor_sharding`) | GDDR-interleaved only. L1 is per-op scratch | missing (G12) | P | P2 |
| Data formats per tensor | `DataType` on every tensor (FP32, BF16, BFP8_B, BFP4_B, INT32, UINT8...). Unpack and pack convert | FP32-only tensors. Codes measured for 0, 1, 4, 5, 8 (`tile.rs:138-147`) | partial (G9; D1–D3) | P, C | P1 |
| Compute kernel config | `MathFidelity`, `fp32_dest_acc_en`, `math_approx_mode`, `packer_l1_acc`, `dst_full_sync_en` per op | `Fidelity` and `SrcRoute` per engine, FP32 Dst always | partial (G10) | P, X | P1 |
| `*_tile_init` / `*_tile` and LLK init/uninit state | Each op's init reprograms unpack/math/pack state; mixing ops needs `*_init_short` / reconfig (`bmm_large_block_zm.cpp:54-60`) | Every role program carries the full state it needs (`matmul.rs:357-368`), and a tile reset clears thread state (`session.rs:13-16`) | different now, **hazard with fusion** (G11) | C | P1 |
| Eltwise via FPU vs SFPU | Binary: FPU `ELW*` (`add_tiles`). Unary and transcendental: SFPU | B core scalar FP32 (`dm_b.rs:86-151`) | missing (F1–F5, S1, M1) | P | P1 |
| reduce / bcast / transpose_wh APIs | `reduce_tile` (with a scaler tile), `*_tiles_bcast`, `transpose_wh_tile` | `COL_SUM` on the B core; transpose in the B core | missing (R1, M2, M3) | P | P1 |
| Tilize/untilize on the device | `tilize_block` / `untilize_block` compute APIs, or unpacker tilize mode | Host | missing (D5, 9.10) | P | P1 |
| Tile shapes (tiny tiles) | `Tile({16,32})` etc., `DstTileShape` | `TileDescriptor` models any shape (`tile.rs:1-18`). Kernels fix 32×32 | partial (G15) | P | P2 |
| Watcher, waypoints, DPRINT | NoC address sanitising, waypoints (`WAYPOINT("CWFW")`), kernel asserts, `DPRINT` ring buffers | Host-side `decode` refusals before the device and again on it, heartbeat, `PANICKED` and error codes, no message text (`tt-firmware/src/lib.rs:167-180`) | partial (G13) | X | P1 |
| Device profiler (Tracy) | Per-RISC zone timestamps into L1, merged on the host | `runtime::Profile` (host phases), `Device::traffic`, the timestamper buffer (`tt-device/src/trace.rs`) | partial (G13) | X | P2 |
| `noc_async_*_barrier`, flush, posted writes | Per-NoC transaction counters, read/write/atomic barriers | Per-`TxnId` wait (`tt-firmware/src/lib.rs:438`), fence through `publish()` | **have** | -- | -- |
| Allocator | Banked L1 and DRAM allocators, lock-step bank offsets, L1_SMALL, TRACE regions | Coalescing per-channel GDDR allocator (`tensor.rs:114-170`); L1 planned by liveness (`l1.rs`), stronger than tt-metal's (§4) | **have** | -- | -- |
| Mesh device, fabric, CCL | `MeshDevice`, TT-Fabric routing, `all_gather`, `reduce_scatter`, `all_reduce` | `shard.rs` splits N over chips through Ethernet movers, host-staged (`shard.rs:1-16`) | partial (G14; 9.11) | P | P2 |
| TTNN device-op structure | `validate`, `compute_output_specs`, `create_program`, `override_runtime_arguments`, `compute_program_hash` | Free functions returning `Work` (`tensor.rs:447, 592, 662`). Checks inline as `TensorError::Shape` | partial (G16) | X | P1 |

---

## 2. Gaps of consequence

### G1. Padding semantics: logical shape, padded shape, pad value (P0, correctness)

**(a) tt-metal.** A tile-layout tensor whose dims are not multiples of 32 has a
*logical* shape and a *padded* one. What sits in the pad is **unspecified** unless an op
says otherwise. Ops whose result depends on it fill it first: `ttnn.fill_implicit_tile_padding`
(`ttnn/operations/data_movement/fill_pad/`), with face-level helpers in
`ttnn/operations/kernel_helper_functions/pad_tile.hpp`. Reductions and softmax mask by the
logical shape or fill the pad with the identity of the reduction (0 for a sum, -inf for a
max). The hardware has no notion of a logical shape: `MVMUL`, `ELW*`, the SFPU and the
packer compute whole faces.

**(b) Ours.** `DramTensor` (`tensor.rs:200-205`) holds `rows, cols` and a placement, and
its documentation states the invariant: "zero-padded at the ragged edges, which is the
identity for every accumulation" (`tensor.rs:6-8`). Upload establishes it (tt-layout pads
with zero). **Nothing maintains it**:
- `ADD_ROW` adds the bias row to all 32 rows of every tile, padding rows included
  (`dm_b.rs:121-151`, line 146). The `ELTWISE` record does not mask
  (`dm/record.rs:226-289`). After `[37, 70] + [1, 70]`, rows 37..63 hold the bias.
- `COL_SUM` sums all 32 rows of every tile (`dm_b.rs:109-119`), and `SUM` passes every
  tile row (`dm/record.rs:291-296`). So `sum_rows(add_row(x, b))` for a ragged `x` adds
  `27·b[c]` to column `c`. In Burn that is `(x + b).sum_dim(0)` on a device-resident `x`
  with a batch that is not a multiple of 32. It is silent: download crops
  (`tensor.rs:319-367`). The gates never chain a ragged `ADD_ROW` into a sum
  (`step19_eltwise.rs:140-187`, `step20_many_tiles.rs:220-270` each start from fresh
  uploads). MNIST's batch of 64 hides it. *Found by reading, not yet reproduced. The repro
  is exactly that chain at `[37, 70]`.*
- `MUL_SCALAR` by ±inf or NaN turns zero padding into NaN (`0·inf`). A matmul that takes
  that tensor as a ragged-`K` operand then gets `NaN·0 = NaN` in **every** output.
- Phase 10 makes this the common case: `exp(0) = 1`, `sigmoid(0) = 0.5`, `x + s` (S1's
  `float_add_scalar`), `log(0) = -inf`. A softmax over a ragged row (R2) would count the
  pad columns `exp(0)` into its denominator.

**(c) Design.** Make the pad a typed property of the tensor, and make every op state what
it needs and what it leaves:

```rust
pub enum Pad { Zero, Value(f32), Undefined }        // tt_kernels::tensor
pub struct DramTensor { rows, cols, pad: Pad, placement }
trait OpPadding {                                    // per op kind, data not comments
    fn requires(&self, input: usize) -> PadNeed;     // Any | Zero | Value(identity)
    fn produces(&self, inputs: &[Pad]) -> Pad;       // ADD_ROW: Undefined for rows; RELU(Zero)=Zero ...
}
```
The op constructor (G16's `validate`) compares `requires` with each input's `pad`. When
they differ it inserts a **fill-pad job** before the op. Only edge tiles change: the
`ct` tiles of the last tile row and the `rt` tiles of the last tile column, so the cost
is O(perimeter), never O(area). The fill-pad job is a mover `COMPUTE` kind today and an
SFPU masked store (S2) or packer edge masking (`Packers/EdgeMasking.md`, D4) later.
`produces` comes from the op's algebra (`f(0) == 0` is a fact about `f`), so `RELU` and
`MUL` on `Zero` stay `Zero` and cost nothing. Cheaper still where it applies: ops mask
by the logical shape themselves. `COL_SUM` can stop at the last valid row of the last tile
row for free, and a reduction's `PadNeed::Value(identity)` lets the planner choose.

**(d) Steps.** 1. Add `pad` to `DramTensor`, `Pad::Zero` on upload and alloc-by-op set by
`produces`. 2. Add a `FILL_PAD` record (edge tiles, value, logical rows/cols) to
`dm/record.rs` and `dm_b.rs`. 3. `COL_SUM` and `SUM` take the valid row count of the last
tile row. 4. Give `matmul_dram` `requires(Zero)` on the `K` edge of both operands only (the
M and N edges of the output just inherit). 5. Every new Phase 10 op fills in `OpPadding`
as part of the definition of done (add a line to `hardware-coverage.md` "Definition of
done").

**(e) Gate.** A ttsim gate, `step2x_padding`: for each pair (producer, consumer) over
{`ADD_ROW`, `MUL_SCALAR(inf)`, `RELU`} × {`sum_rows`, matmul with ragged `K`}, at
`[37, 70]`, against `burn-flex`, bit for bit. Watch it fail first (it should fail today
on `ADD_ROW → sum_rows`). Add a property test that every edge tile of an op's output holds
what its `Pad` claims (download the raw slots, not the cropped tensor). Run on silicon.

**(f) Slot.** Before 10.0 (S1 adds `add_scalar`, which breaks `Zero`). The `ADD_ROW → SUM`
bug is a fix now.

### G2. N-D tensors in tile layout (P0, performance and generality)

**(a)** tt-metal tiles the last two dims and treats leading dims as a stack of matrices:
logical `[B, H, W]` is padded to `[B, ceil32(H), ceil32(W)]`, each slice padded on its
own, so a reshape that merges `B` into `H` is free only when `H % 32 == 0`
(`tech_reports/tensor_layouts`). Batched matmul, broadcast and reductions over leading
dims work on that view.

**(b)** `DramTensor` is `[rows, cols]` only. `burn-tt` routes to the device only when
`is_matrix_f32` (`burn-tt/src/tensor.rs:182-185`, `ops.rs:70, 165, 248, 282, 314`). Any
rank-3 or rank-4 tensor (attention, conv, a batched `Linear` input `[B, S, D]`) is a
download, a Flex op and an upload. Batched matmul is host-staged per batch element
(`ops.rs:176-237`). Only the download is reported, and only under `TT_TRACE_FALLBACK`
(`burn-tt/src/tensor.rs:107-115`). The upload in `to_dram` is silent.

**(c) Design.** `DramTensor { shape: Vec<usize> (logical), pad: Pad, placement }`, stored
as `prod(leading)` stacked `[ceil32(H), ceil32(W)]` tile grids. `TensorRef` gains a batch
stride in tiles (`rt * ct`). The existing kernels already work on tile ranges, so most of
the change is in the records' tile indexing. Reshapes split by cost. One that keeps the
last dim and merges leading dims where `H % 32 == 0` is a view, like `rows_view`
(`tensor.rs:292`). Anything else is a device re-tile (D4/D5). Broadcast over leading
dims is a `TensorRef` with batch stride 0, the tt-metal `bcast` idea applied in the
record rather than in the kernel.

**(d) Steps.** `DramTensor::shape`. Batch stride in `TensorRef` (it has the room: 64
bytes). `ELTWISE`/`SUM`/`GATHER` iterate batches. `burn-tt`: replace `is_matrix_f32` with
`is_tileable_f32` (rank ≥ 1, `F32`). A batched `float_matmul` becomes one op with
`batch` jobs, dealt like blocks.

**(e) Gate.** `step20`-style: `[3, 37, 70]` element-wise and `[2, 3, 64, 40] @ [2, 3, 40, 33]`
against Flex, bit for bit, at 1 and 8 tiles. A residency gate: a rank-3 MLP step downloads
nothing (`tensor_traffic`).

**(f) Slot.** With 10.0. It decides whether Phase 10's ops are reachable from real models
at all (R2 softmax is over `[B, S, V]`, R4 attention is rank 4).

### G3. K blocking without a K limit (P0, a panic mid-model)

**(a)** tt-metal's matmul streams `K` in blocks of `in0_block_w` tiles. With more than one
block (`spill`), each output subblock is packed to an intermediate CB and **reloaded into
Dst** with `copy_tile` before the next block accumulates on top
(`bmm_large_block_zm.cpp:38-107`). Alternatively the packer accumulates into L1 itself
(`PACKER_L1_ACC`, `llk_pack_reconfig_l1_acc`, `bmm_large_block_zm_fused_bias_activation.cpp:287, 379-405`).
L1 then holds two K blocks (double-buffered), not all of K.

**(b)** Ours keeps all of `K` in L1: `plan_uncached` halves `kc` only as a last resort
(`matmul.rs:960-1009`), and the GDDR path refuses any split (`tensor.rs:476-482`)
because the host-side partial-sum add reorders rounding. `burn-tt` turns the refusal into
`panic!("matmul on {device}: ...")` (`server.rs:511`). **Measured** with `chunk_fits_in(
[1, kt, 1], Tf32FromFp32, _, Slots)`: the most K tiles that fit are **102 at HiFi4**
(K ≤ 3264) and **109 at Lo** (K ≤ 3488). The binding limit is the 8192-instruction
program slot (`mailbox.rs:156`) as much as the 896 KiB data arena: the unpack and math
programs are fully unrolled per `(output, k)` pair (`matmul.rs:383-421`). So `[*, 4096] @
[4096, *]`, an ordinary MLP width, panics.

**(c) Design.** Device-side K blocking that keeps the bits. Dst accumulates `MVMUL`
products in K order. Packing the FP32 partial exactly (FP32 Dst → FP32 L1 is lossless)
and unpacking it back into Dst (`UnpackToDst` FP32, F1) before the next block continues
**the same sequence of FP32 additions**. Expect bit-identity with the unsplit run, which
keeps the Phase 7 golden as the oracle. `PACKER_L1_ACC` does *not* keep the bits (it
adds block sums, so it reorders), so it is opt-in under a `Tolerance` and never the
default. This is the same argument `shard.rs:11-16` makes for splitting N rather than K.
The program-size half of the limit goes with loops (MOP/REPLAY, checklist Phase 9) or with
one program per K block reused from the program cache (9.7c): the K-block program is
the same for every block, only the L1 addresses differ, which is G16's runtime-argument
point.

**(d) Steps.** 1. F1 (whole-tile `UnpackToDst` FP32). 2. A `matmul_roles` variant: math
first loads the partial into Dst (`k_block > 0`), then accumulates. The pack target is the
partial slot. 3. `plan_in` returns `[mc, kc, nc]` with `kc < kt` allowed on `Slots`, and
`matmul_dram` emits per output block `kt / kc` gather/kernel steps on one tile, with a
`WAIT` between, in one list. 4. Until it lands, replace the panic with a host fallback that
logs once (§3). 5. Then the K-block ring becomes a CB (G5) and the gathers overlap.

**(e) Gate.** Where an unsplit run fits, K blocking must not change a bit: K = 3200
(`kt` = 100) at `kc ∈ {100, 50, 25}`, plus a ragged `[37, 3000] @ [3000, 70]`, all
bit-identical to `kc = kt`. Where it does not fit (`[64, 8192] @ [8192, 64]`), check the
result against Flex within the derived HiFi4 bound (`step11_burn.rs`'s `bound`), with no
download (`tensor_traffic`). Watch it fail with the reload skipped.

**(f) Slot.** Needs F1. Belongs in 10.0 next to S1, because F1 is the same work.

### G4. Dst capacity: subblocks and `SyncHalf` (P1, performance)

**(a)** Blackhole Dst is `DEST_REGISTER_FULL_SIZE = 64 * 16 = 1024` 16-row units: 16
32×32 tiles at 16-bit, **8 at FP32** (`fp32_dest_acc_en`). With `SyncHalf` (the default;
`dst_full_sync_en=false`), math owns one half while pack drains the other, so **4 FP32 tiles**
per acquire (`ckernel.h:836-840`, `tensix_types.h:191-193`, `reg_api.h:45-90`). A matmul
picks `out_subblock_h × out_subblock_w ≤ 4` (FP32) or `≤ 8` (BF16 Dst). Each `SrcA` tile
(in1) is unpacked once and reused across `subblock_h`, and each `SrcB` across
`subblock_w`.

**(b)** `matmul_roles` keeps one output tile in Dst. Math waits for `free`, clears **all**
of Dst (`ZEROACC` mode 3, `matmul.rs:374-376`), accumulates, posts `ready`. Pack drains,
posts `free` (`matmul.rs:333-431`). So math and pack never overlap, and every `MVMUL`
re-unpacks both operands (`matmul.rs:383-390`): 2 unpacks per `MVMUL`.

**(c) Design.** A typed `DstPlan { mode: Full | Half, acc: Fp32 | Fp16, tile: TileShape }`
whose `capacity()` is computed the way `get_dest_max_tiles` computes it, with a `DstSlot`
newtype that cannot index past it. `matmul_roles` takes a `Subblock { h, w }` refused at
construction if `h*w > capacity`. Math alternates halves (`ZEROACC` of the half only),
and the semaphore pair becomes the tt-metal `math → pack` section handshake with two
sections in flight. The `ready/free` semaphores already start at (0, 1) and become (0, 2).

**(d) Steps.** `DstPlan` in `tt-isa` next to `Dst.md`'s facts. Subblock loop order in
`matmul_roles` (reuse `SrcA` across `w`, `SrcB` across `h`; check `Banks` allows it).
Half-Dst addressing for pack (`pack_rows` with a base row). The chooser picks `(h, w)`
from the block shape the way ttnn's `get_matmul_subblock_params` does, as a pure function
with a unit test.

**(e) Gate.** MNIST's products and a `[256, 256] @ [256, 256]` bit-identical to today's
(accumulation order per output tile is unchanged). `silicon_perf` shows fewer unpacks per
`MVMUL` and a lower ms for the 512³ product. Watch it fail with both halves cleared.

**(f) Slot.** 9.8 (overlap inside the tile), together with the hazards-as-data wait
planner.

### G5. The circular-buffer runtime (P1; designed, 9.8)

**(a)** A CB is a ring of `fifo_num_pages` pages with two counters, `tiles_received`
(producer) and `tiles_acked` (consumer), **16-bit and wrapping**
(`hw/inc/internal/circular_buffer_interface.h:89-109`). The rules, each a known footgun
(`dataflow_api.h:180-205, 440-472`):
1. A push never wraps: the producer writes contiguous memory, and the pushes in one cycle
   of the ring must **sum to exactly** `fifo_num_pages` (5+7 on 12 is legal, 7+7 is not).
2. `cb_wait_front(n)` without an intervening pop waits **cumulatively** (8, 16, 24, 32), not
   four times 8.
3. Writing a page does not publish it; `push_back` does. Reading does not free; `pop_front`
   does.
Double buffering is a ring of 2× the block, nothing more.

**(b)** `Requirements::cb(name, page, pages, align, from, to, live)` (`l1.rs:201`) declares
rings with endpoints (`Endpoint::{Mover, Unpack, Math, Pack}`, `l1.rs:39`). The planner
places them, `fuse` joins them. The runtime protocol is decided (`RUST_IMPL_PLAN.md`,
"L1 planning and circular buffers", item 6) and not built: today each "ring" is filled
completely before the kernel starts (`dm_b.rs:213-217`: `KERNEL` waits for every move).

**(c) Design.** Put the three rules in the type, not the comments:
- `CbProducer<'p>` / `CbConsumer<'p>` handles, one each per CB, minted by the `Plan`
  (endpoint ownership is already declared, so the plan can refuse a second producer).
- A `push(n)` builder that refuses an `n` which would cross the ring's end (rule 1). The
  planner can make it impossible instead, by choosing `pages` as a multiple of every
  push size the kernel declares.
- `wait_front` takes `n` relative to the last pop, and the emitted code converts it to
  the cumulative form (rule 2), so the cumulative form never reaches user code.
- Counters are L1 words with 16-bit wrapping arithmetic, polled through `publish()`
  (Tier 1 #7). **Not the Sync Unit semaphores**: those are 4-bit and saturate at 15
  (`sync.rs:14, 96-110`), so a ring of more than 15 pages would deadlock silently. The
  Tensix side (T0 waiting for "a page arrived") polls the L1 word from the role runner
  before pushing the `UNPACR`. The B side increments it after `noc::wait` (`tt-firmware/src/lib.rs:438`).

**(d) Steps.** Follow 9.8's order: a mover-side `PUSH`/`WAIT_FREE` list entry. A role-runner
`WAIT_PAGES(cb, n)` / `POP(cb, n)` pseudo-op between program chunks (so a role program
becomes a sequence of chunks separated by CB operations). Then the matmul's A/B rings at
`2 × kc`-tile blocks (G3). Hazards-as-data supplies the Tensix-side ordering.

**(e) Gate.** ttsim: a mover → unpack → math → pack → mover pipeline over 3× the ring's
pages, bit-identical to today's. A property test of the counter arithmetic across the
16-bit wrap (start counters at `0xFFF0`). A test that a 16-page ring works, which would
fail if anyone "simplifies" to Sync Unit semaphores. Silicon: the 512³ matmul time.

**(f) Slot.** 9.8. It is the prerequisite for G3's streaming, G6 and G7.

### G6. Reader / compute / writer on two data movers and two NoCs (P1, performance)

**(a)** tt-metal gives each Tensix tile two data-movement RISCs (BRISC = RISCV_0, NCRISC =
RISCV_1), each with its own default NoC, so a reader kernel and a writer kernel run at
once on separate NoC paths while TRISC0..2 compute (`METALIUM_GUIDE.md`, "Kernel types").
NoC 0 and NoC 1 route in opposite directions, so reads and writes do not share links.

**(b)** RISCV B does every move, the scalar element-wise compute and kernel sequencing
(`dm_b.rs`). NC is held in reset with its slot kept clear (`tt-isa/src/l1.rs:35`). No code
issues on NoC 1 (`Noc1` appears only in `tt-isa/src/noc.rs`'s types).

**(c) Design.** An `nc_w` writer image under the same `tt_isa::dm` contract, taking
`SCATTER` and CB-pop entries, issuing on NoC 1. B keeps gathers and `KERNEL`. The record
expander (`record::expand`) already produces entries per side. Split them by op kind at
expansion time, so the host still sends one record. A typed `NocId` on `niu::Command`
issue (it already exists as a type parameter) keeps a NoC-1 command from being issued
through NoC-0 registers.

**(d) Steps.** Survey NC's IRAM/L1 constraints (Blackhole page). Port `dm_b`'s issue loop.
Use a CB (G5) between the packer and the writer. Gate the NC's reset handling as
`step16_dm` did for B.

**(e) Gate.** ttsim: an element-wise run and a matmul with gathers on B and scatters on NC,
bit-identical. Silicon: read and write overlap measured in `silicon_perf`.

**(f) Slot.** After 9.8. It overlaps with checklist "Three-thread pipelining".

### G7. Multicast, L1 semaphores and core ranges (P1, performance)

**(a)** The 2D-mcast matmul (`MatmulMultiCoreReuseMultiCastProgramConfig`) puts a grid of
cores on the output. The first core of each row reads its in0 block from DRAM and
`noc_async_write_multicast`s it to the row. The first of each column does the same for
in1. Sender and receivers handshake through **L1 semaphores**: receivers
`noc_semaphore_inc` the sender's when their CB has space, and the sender waits for
`num_dests`, multicasts the data, then `noc_semaphore_set_multicast(VALID)`
(`reader_bmm_tile_layout_in0_sender_padding.cpp:340-373`,
`reader_bmm_tile_layout_in0_receiver.cpp:42-75`). DRAM traffic falls from `O(cores ×
K)` to `O(sqrt(cores) × K)`. The 1D variants do the same along one axis.

**(b)** `tensor::blocks` shrinks the block until every unit has one (`tensor.rs:419`).
Every job gathers its own A rows and B columns (`tensor.rs:500-535`). MNIST's first layer
plans as `[1, 25, 4]` (measured): each block re-reads its 25 K tiles of A and the 100
tiles of B. With 64 tiles on a 512³ product, the B matrix is read from GDDR once per
block row. Semaphores are only the Tensix Sync Unit's 8 (`sync.rs:43`), local to a tile.
`TileChoice::Count(n)` takes the first `n` tiles row by row, with no rectangles.

**(c) Design.**
- `CoreRange` / `CoreRangeSet` in **translated** coordinates (Blackhole harvests columns,
  and translation keeps the survivors contiguous). Built only from `grid::Tensix`, so a
  rectangle containing a fused-off tile cannot be constructed (row 35's hang becomes a
  type error).
- A second semaphore kind in the planner: `Requirements::l1_semaphore(name, initial)`,
  an L1 word in the data arena, incremented by NoC atomic (BH `NoC/Atomics.md`) or set by
  multicast. The plan keeps the two kinds apart (`Sem` vs `L1Sem`), so a Tensix
  `SEMWAIT` cannot be pointed at an L1 word.
- `niu::Command::Multicast { range: CoreRange, ... }`, refusing a range containing the
  sender on NoC paths where that is illegal, and an L1 → L1 congruence rule (C16).
- A matmul config enum, decided by a pure function of shape and grid like
  `ttnn::operations::matmul::create_matmul_program_config`: `OneTile`, `Mcast1D { axis }`,
  `Mcast2D { grid }`, each with `(in0_block_w, subblock, per_core_m/n)`.

**(d) Steps.** Measure first: is the 64-tile matmul GDDR-bound once 9.8 overlaps? (The
`Saturating_DRAM_bandwidth` report is the target figure.) Then NoC atomics and L1
semaphores (ttsim row check), a multicast probe (`TLB strided` is row 4; the NIU
broadcast is untested), `Mcast1D` along N, then 2D.

**(e) Gate.** ttsim: the mcast matmul bit-identical to the per-block one (the accumulation
order per output tile is unchanged). A probe gate that a multicast reaches exactly the
rectangle. Silicon: GDDR bytes read per op (NoC counters, already `[x]`) and ms.

**(f) Slot.** Phase 9, "multi-tile distribution with NoC multicast". Needs G5.

### G8. Dispatch: async queue, completion in host memory, traces (P1, performance)

**(a)** Fast dispatch puts the host's commands in a ring in **pinned host memory**
(hugepage). A prefetcher core *pulls* them over PCIe and a dispatcher core writes worker
L1, multicasts launch messages and GO signals, waits for workers, and writes completion
records **into host memory** (`cq_commands.hpp:19-62`: `RELAY_LINEAR`, `EXEC_BUF`,
`WRITE_PACKED` with `MCAST`, `WAIT`, `SEND_GO_SIGNAL`, `WRITE_LINEAR_H_HOST`). The host
never reads device MMIO in steady state. `EnqueueProgram` is non-blocking. Events order
work across queues. **Traces** capture a whole step's commands once into DRAM and replay
them with one command (`EXEC_BUF`), which removes host dispatch from the step entirely
(`tech_reports/AdvancedPerformanceOptimizationsForModels`).

**(b)** Ours, per op: the host builds records, writes them through uncached BAR MMIO
(~150 MB/s, measurement M), writes a descriptor per tile, and **polls `DONE` by MMIO
read** per unit per wave (`dm.rs:232-257`, `session.rs:692-728`). The Burn op blocks
until then (`server.rs:380-394`), although the output's GDDR placement is known on the
host before the device runs (`tensor.rs:485`). `tt-kmd` has no `PIN_PAGES`. 9.7c's
measurement names the floor: past 8 tiles, "per-tile submission, a descriptor write and
a list per tile".

**(c) Design**, in three steps that each pay alone:
1. **Async enqueue in `burn-tt`.** `server::eltwise/matmul_dram/sum_rows` return the
   `BufferId` at once. The host-side allocator already decided the placement, so the
   server thread executes in order behind them, and only `download` (and `free`) waits.
   Errors surface at the next sync point as a typed `EngineError` naming the op that
   failed. This is the `blocking=false` + `Finish` model, and it overlaps Burn's
   autodiff graph building with device time.
2. **Completion into host memory.** `PIN_PAGES` a page. B's `DONE` becomes a NoC write
   to the PCIe tile's host window (or both, while gated), and the host polls cached RAM.
   Under this VM, where MMIO reads run at 5 MB/s, this probably helps more than any
   other dispatch change. Later, the issue ring in host memory with B (or a dispatcher
   tile) pulling from it gets uploads out of the uncached-write path too
   (measurement M's 226 MB/s ceiling).
3. **Traces = recorded op records in GDDR.** A training step is the same sequence of
   records every step (9.5 asserts the bytes are identical). `Session::begin_trace` /
   `end_trace` capture each unit's lists into GDDR. `replay` writes one descriptor per unit
   (or one to a dispatcher tile that multicasts the GO, G7). Runtime arguments that change
   per step (the batch view's first tile) are patched in place: the TTNN
   `override_runtime_arguments` idea applied to records.

**(d) Steps.** (1) is host-only and gated by the existing golden. (2) needs `PIN_PAGES` in
`tt-kmd` (ABI test like `abi_layout.rs`), the NoC → PCIe address map
(`PCIExpressTile/README.md` "device→host 64-bit NoC map"), and a ttsim answer (it may not
model host memory: check, log a divergence row). (3) needs (2) and 9.7b's records.

**(e) Gates.** (1) The golden bit for bit, and a test that an op error surfaces at the next
download with the op's name. (2) `Device::traffic` reads per steady MNIST step → ~0. (3)
A replayed step equals a dispatched one bit for bit, and PCIe writes per step are a
constant independent of op count.

**(f) Slot.** (1) can go now. (2) and (3) are new Phase 9 items after 9.8. They are the
real answer to "per-op cost is the host's".

### G9. Per-tensor data formats and format-dependent pages (P1)

**(a)** Every tt-metal tensor has a `DataType`. DRAM pages and CB pages are sized by format
(FP32 tile 4096 B, BF16 2048 B, BFP8_B `1024 + 64`, BFP4_B `512 + 64`:
`api/tt-metalium/constants.hpp:13-21`). Unpack converts L1 → `Src`/Dst and pack converts
Dst → L1 (`tech_reports/data_formats`). BH's enum (`hw/inc/internal/tt-1xx/blackhole/tensix_types.h:213-235`)
agrees with our measured codes 0, 1, 4, 5, 8. It is a **hypothesis list** for the rest
(Bfp8_b = 6, Bfp4_b = 7, Int8 = 14, UInt8 = 30, UInt16 = 9, UInt32 = 24), to measure as
`tile.rs:118-125` demands, not to transcribe.

**(b)** `DramTensor` is FP32 with a fixed 4160-byte slot (`dm.rs:149`). `Staging::Slots`
refuses anything but FP32 in L1 (`matmul.rs:885-887`). The burn-tt BF16 path computes on
Flex (`ops.rs:157-163`).

**(c) Design.** `DramTensor { format: L1Format }` and `TileSlot::for(format)` =
`round_up(16 + datum_bytes·1024 [+ 64 exponent bytes], 64)`, a const fn, so the C64 rule
holds per format. `TensorRef` carries the slot size. The op definition names its accepted
input formats and its output format. Mismatches are an explicit typed cast job (pack
conversion, D1), never an implicit reinterpretation.

**(d)–(f)** As D1–D3. Add the `TileSlot` function and the ttsim refusals (row 31:
`UnpackToDst` for 16-bit) to the definition: the `F1` route choice (via `Src` + `MOVA2D`
on ttsim) is what makes a BF16 SFPU op gateable on ttsim at all.

### G10. Compute kernel configuration as a typed, per-op value (P1)

**(a)** `ComputeKernelConfig { math_fidelity, math_approx_mode, fp32_dest_acc_en,
packer_l1_acc, dst_full_sync_en }` is given per op, defaulted per architecture, and part
of the program hash.

**(b)** `Fidelity` and `SrcRoute` are fixed per engine (`server.rs` `KmdEngine`). Dst is
always FP32 (`matmul.rs:362-365`, `runtime.rs` `DST_FMT_FP32`). Approximation modes do not
exist yet (no SFPU ops).

**(c) Design.** `ComputeConfig` in `tt_kernels`, folded into every program memo key (the
matmul's key already holds route and fidelity, `matmul.rs:917`). Its consequences
are typed: `fp32_dest_acc` sets `DstPlan::acc` (G4), so the capacity changes with it and
a subblock that no longer fits is refused at construction. `approx` selects the SFPU
program variant, and the derived error bound comes with it (the Phase 10 oracle policy).
Burn has no per-op config, so `burn-tt` maps one engine-wide `ComputeConfig`, with an
override per op kind. When an op runs below FP32 precision (TF32 `Src` today), the
backend says so once (§3).

**(f) Slot.** Introduce with F3/F4 (the SFPU kernel's config is the first new user).

### G11. LLK init/uninit state hazards under fusion (P1, correctness)

**(a)** In tt-metal a compute kernel calls `mm_init`, `copy_tile_to_dst_init_short`,
`*_tile_init`, `unpack_reconfig_data_format` between ops because each reprograms shared
unpacker, math (`ADDR_MOD`, MOP) and packer state that the other op assumed
(`bmm_large_block_zm.cpp:54-60` re-inits matmul after every `copy_tile`). Forgetting one
gives wrong results, not an error.

**(b)** Today every role program brings its whole configuration (`matmul_roles`:
`src_thread_config`, `config_program`, `math_prelude`, `matmul.rs:357-371`), and a tile
reset clears per-thread state (`session.rs:13-16`; rows 47 and 49). That is correct and
costs words. With fusion (`Requirements::fuse`) and with the program cache running
programs back to back without a reset (9.7c), a program inherits the previous program's
`Config`/`ThreadConfig`/`ADDR_MOD`/RWCs.

**(c) Design.** This is hazards-as-data for *state*, not ordering. Each program builder
declares the state it **reads** (config words, ADDR_MOD slots, RWCs, `SFPCONFIG`
constants, LReg 11–14) and **writes**. A program's prologue is then the diff from a known
predecessor state, or the full set when the predecessor is unknown (the safe default,
which is today's behaviour). In debug builds the runner checks the declared read set
against a register snapshot (`cfg` reads) and fails with the first stale field's name. It
is generated from `cfg/generated.rs`, never transcribed.

**(e) Gate.** A ttsim test that runs matmul → SFPU op → matmul through the cache with no
reset and is bit-identical to running each after a reset. Watch it fail with the SFPU
program's `ADDR_MOD` write removed.

**(f) Slot.** Due when F3's SFPU kernel shares a session with the matmul, before fusion.

### G12. Sharded, L1-resident tensors (P2)

tt-metal keeps activations in L1, height/width/block sharded over a core grid
(`buffer_types.hpp:11-19`, `tech_reports/tensor_sharding`), so a chain of ops never touches
DRAM. Ours always round-trips GDDR between ops. The right shape here is the plan's
fusion (`Requirements::fuse` drops the GDDR legs) plus a `Residency::L1Sharded {
range: CoreRangeSet, shard: [h, w] }` placement for a `DramTensor`'s successor type, with
ops declaring which shard specs they accept, as ttnn's `validate` does. Defer until
G5/G7 exist and a measured op chain is GDDR-bound.

### G13. Debug tooling: watcher, waypoints, device print (P1, developer experience)

tt-metal's watcher sanitises every NoC address, records per-RISC waypoints (`WAYPOINT("CWFW")`
around every CB wait, `dataflow_api.h:479-484`), catches kernel asserts and reports *which
core is stuck where* on a hang. DPRINT streams formatted values through L1 ring buffers.
The device profiler records zones per RISC (`tech_reports/MetalProfiler`).

Ours is stronger before the device (every descriptor decoded on the host first, then
again by the firmware, `dm_b.rs:1-6`) and weaker on it: a hang is "`TimedOut { seq, done
}`" (`dm.rs:250-253`), and a panic is a code with no location (`tt-firmware/src/lib.rs:167-180`).
Recommended, cheapest first:
- **Waypoints**: one word per core in its mailbox, written at every wait site (mover list
  index, entry kind, CB id, role program counter chunk). `RunError::Roles` and
  `DmError::TimedOut` read and print them, so a hang names the entry and the wait.
- **Panic location**: write `file` id + `line` (two words, a `const` table generated at
  build time) instead of dropping `info.location()`.
- **Device print**: a fixed-format ring (`u32` tag + payload words), decoded on the host
  against a tag table generated from the firmware source. No formatting on the core.
- **Zones**: the timestamper already works on silicon. Emit begin/end events per list
  entry into `TRACE` and export Chrome-trace JSON from `runtime::Profile`.

### G14. Mesh, fabric, collectives (P2; 9.11)

tt-metal: `MeshDevice`, TT-Fabric routing over Ethernet, CCL ops (`all_gather`,
`reduce_scatter`, `all_reduce`) as device ops scheduled like any other (`tech_reports/TT-Fabric`,
`EthernetMultichip`, `Programming_Mesh_of_Devices`). Ours: `shard.rs` splits N across chips
through host-orchestrated Ethernet movers, chips in turn, not resident (checklist 9.11).
For data-parallel training, the collective needed is an all-reduce of gradients, and
its order is part of the result. Keep `shard.rs`'s discipline and state the reduction order
in the op (a ring all-reduce sums in ring order, so the golden becomes "bit-identical to
the same ring order on the host", the `COL_SUM` approach). Build it on 9.11's resident
mesh, with the trace (G8) spanning chips.

### G15. Tile shapes other than 32×32 (P2)

tt-metal supports tiny tiles (16×32, 32×16, 16×16, and 1/2/4/8×32 for some ops), mostly
for decode-time matmuls with M = 1 (`DstTileShape`, `ckernel_defs.h`). `TileDescriptor`
already models any `W/Z/Y/X` (`tile.rs:1-18`), but every kernel and `DramTensor` assume
32×32 (`TILE_SLOT`, `face_index`, `matmul_roles`'s face loops). A `[1, n]` bias today
occupies whole 32-row tiles (and 9.4's "small downloads" work around it). When it is
worth it: `TileShape` as a type parameter of `DstPlan` and `TileSlot`, M = 16 first.

### G16. A device-op trait (P1, developer experience)

TTNN's device operation is a fixed protocol: `validate_on_program_cache_miss/hit`,
`compute_output_specs`, `create_output_tensors`, `select_program_factory` →
`create(...)`, `override_runtime_arguments`, `compute_program_hash`. Ours are free
functions (`tensor::matmul_dram`, `eltwise`, `sum_rows`) that mix validation
(`TensorError::Shape` strings), allocation and job building. Phase 10 adds dozens of ops,
so fix the shape now:

```rust
pub trait DeviceOp {
    type Args;                                       // shapes, formats, ComputeConfig
    fn validate(&self, inputs: &[&DramTensor]) -> Result<(), OpError>;    // typed, actionable
    fn output_specs(&self, inputs: &[&DramTensor]) -> Vec<TensorSpec>;     // shape, format, Pad (G1)
    fn padding(&self) -> &dyn OpPadding;                                   // G1
    fn program_key(&self) -> ProgramKey;             // what the program cache keys on
    fn jobs(&self, inputs, outputs, units) -> Work;  // records; DRAM refs are the runtime args
}
```
`Session::run_op` then owns the fill-pad insertion (G1), the format casts (G9), the
fallback report (§3) and the async enqueue (G8). F4's host-side registry is this trait
with op ids.

---

## 3. Hardware constraints and sharp edges: put them in code, not in docs

Policy, in order of preference: **(1)** handle it invisibly when that costs no performance
or when there is no alternative. **(2)** Handle it with a once-per-call-site warning that
names the op, the shape and the cost. **(3)** Refuse it with a typed error at
construction time that says what to change. Never fail opaquely, and never hang
mid-model.

**Mechanism** (none exists today: no `log`/`tracing` dependency in any `Cargo.toml`, and
five ad hoc `eprintln!`s in `tt-kernels`/`burn-tt`/`tt-device`):
- Add `tracing` to `tt-kernels` and `burn-tt` (not `tt-isa` or the firmware). Targets
  per subsystem: `tt::fallback`, `tt::pad`, `tt::precision`, `tt::l1`, `tt::dispatch`,
  `tt::mover`, `tt::program_cache`. `RUST_LOG=tt::fallback=warn` replaces
  `TT_TRACE_FALLBACK` (keep the variable as an alias).
- `warn_once!(key, ...)`: a `static ONCE: OnceLock<Mutex<HashSet<Key>>>` keyed by `(op,
  shape class, reason)`, so a training loop warns once, not per step.
- A structured **fallback/padding report**, `burn_tt::report() -> Report { fallbacks:
  Vec<(op, shape, dtype, reason, bytes_moved, count)>, pads_filled, precision_notes }`,
  extending `tensor_traffic` (`burn-tt/src/traffic.rs`). The residency gates (9.5) assert
  on it, and `tt-mnist --report` prints it.
- `OpError` replaces `TensorError::Shape(String)`, with variants carrying the numbers
  (`KTooLarge { k, max_k, fidelity }`). `Display` says what to do.
- `burn-tt` never panics on a refusal it could have served on the host. A device refusal
  of a supported op falls back with a (2)-level warning. Only transport and hardware faults
  panic (Burn ops cannot return errors), and they first print the waypoint dump (G13).

| Constraint | Source | Enforced today | Recommended |
|---|---|---|---|
| 32×32 tiles, 16×16 faces: ragged shapes pad | `UNPACR`/`PACR` face addressing; `tile.rs` | (1) on upload (tt-layout zero-pads); crop on download | (1) stays. But the pad must be tracked (G1) |
| Pad contents after an op | G1 | **silent wrong result** (`ADD_ROW`, `COL_SUM`) | (1) the fill-pad job on edge tiles only, from `OpPadding`; `tt::pad` debug event per fill |
| Rank > 2 on the device | G2 | silent host fallback; only downloads logged | (1) N-D `DramTensor`; meanwhile (2) `warn_once` per op and rank |
| Row slice not on tile rows (`first % 32 != 0`) | `rows_view` (`tensor.rs:292-306`) | silent host fallback (`ops.rs:331`) | (1) a device re-tile copy (D4); meanwhile (2) |
| Matmul K beyond one tile's L1 + program slot (K > 3264 at HiFi4, > 3488 at Lo; measured) | `tensor.rs:476-482`, `mailbox.rs:156` | `TensorError::Shape` → **panic** in burn-tt (`server.rs:511`) | (1) K blocking with Dst reload (G3); meanwhile (2) host-staged fallback + warn naming K and the limit |
| Batched matmul | `ops.rs:176-237` | (1) host-staged per batch, silent | (2) warn with the bytes per call until G2 |
| Dst capacity (8 FP32 tiles; 4 per half in `SyncHalf`) | `ckernel.h:836-840`; BH `Dst.md` | by convention: 1 tile used | (3) `DstPlan::capacity()` and `DstSlot` newtype, refused at program construction |
| Sync Unit semaphores: 8 per tile, 4-bit, saturate at 15 | `SyncUnit.md`; `sync.rs:43, 96-110` | (3) `Semaphore::new` refuses ≥ 8; the planner shares by liveness (`l1.rs:226`) | add: (3) refuse any use as a counter with a possible value > 15 (a CB of > 15 pages); CB counters are L1 words (G5) |
| L1: 1536 KiB per tile, 896 KiB data arena, 252 KB program cache | `tensix.rs:193`, `tt-isa/src/l1.rs:52-56, 81-85` | (3) `PlanError` / `RunError::DoesNotFit` at plan time | keep. Name the op and the buffer in the message (`what` is a `&'static str` today, so the shape is lost) |
| Program slot: 8192 instructions | `mailbox.rs:156` | (3) `RunError::ProgramTooLong`; the chunk planner avoids it | keep. Program size becomes a planner input, not a probe (`chunk_fits_in` builds every candidate, `matmul.rs:879-897`) |
| DRAM ↔ L1 congruence: C64 read, C16 write; L1 ↔ L1 C16; ≤ 16 KiB per request | rows 64; `noc.rs:276, 449-466`; tt-metal `DRAM_ALIGNMENT` = 64 (`core_config.h:41`) agrees | (3) refused on the host by `niu::Command` and again in firmware | keep. Make `TileSlot::for(format)` (G9) round to 64 so no format can trip it |
| `NOC_CMD_WR_INLINE` to L1; `L1_ACC_AT_EN` | plan "Encode hazards" | typed (`noc.rs`) | keep |
| Harvested (fused-off) tiles hang the NoC | row 35 | (3) the grid from ARC; `TileChoice` refuses | (3) `CoreRange` built only from the grid (G7) |
| Fused-off DRAM channels | 9.1 | (3) a `DramChannel` cannot be named | keep |
| L0 cache not coherent: polls need a fence | Tier 1 #7 | `publish()` in every poll (`dm_b.rs`) | (3) a `Polled<u32>` firmware type whose only read fences, so a bare poll does not compile |
| ttsim refuses BF16/BFP `UnpackToDst`, `SFPLOADMACRO`, `DOTPV` | rows 31, 7, 50 | the gates are silicon-only, by hand | (3) a `Target::{Ttsim, Silicon}` capability query that the op builder consults, choosing the `Src`+`MOVA2D` route on ttsim (F1), and a typed `Unsupported { on: Ttsim }` |
| TF32 `Src` precision for an FP32 matmul | `SrcRoute::Tf32FromFp32` (`matmul.rs:779-797`) | documented, silent at run time | (2) one `tt::precision` note per engine at attach: "matmul multiplies at TF32, HiFi4" |
| Denormals flush, NaN canonicalised | numerics rows C, D | gates record it | (2) once, in the report, when an op's input held a denormal (cheap on the host side only; otherwise just document) |
| GDDR exhausted | `DramAlloc` (`tensor.rs:133-145`) | `OutOfMemory` → panic in burn-tt | (3) the message names live bytes and the largest tensors; (2) evict-to-host is not worth it |
| CB rules (no wrap mid-push, pushes sum to ring size, cumulative waits) | `dataflow_api.h:180-205, 440-472` | n/a (no runtime yet) | (3) by construction in `CbProducer`/`CbConsumer` (G5) |
| Config and thread state survive programs | rows 47, 49 | full prologue in every program; reset between sessions | (1) declared state and generated prologues (G11) |
| A mover or role that hangs | `dm.rs:244-254`, row 65 | `TimedOut` after a deadline; recovery resets the tile | add the waypoint dump (G13) to the error; never hang without a deadline |

---

## 4. Deliberately different, and fine

Future readers: do not "fix" these toward tt-metal.

- **Instruction streams as data, not compiled C++ kernels on the TRISCs.** Role runners
  push host-built `Instruction` sequences from L1 (`mailbox.rs:133-156`). That is what
  lets an op be a builder plus a gate with no firmware change (F4), makes programs
  hashable and cacheable by their words (9.7c), and keeps all hazard checks on the host.
  The cost (program size, G3) is paid back with MOP/REPLAY and K-block reuse, not by
  switching to compiled kernels.
- **Program cache keyed by words, addresses in records.** tt-metal hashes op attributes
  and patches runtime args. We memoise by shape (`matmul::programs`), cache by encoded
  word (`program_cache.rs`), and carry GDDR addresses in records (`TensorRef`). Same
  effect, and the cache key cannot drift from the program.
- **The L1 planner** (`tt-kernels/src/l1.rs`) is stricter than tt-metal's hand-indexed CBs
  (`c_in0 = 0`, `c_out0 = 16`, global CB addresses per program): liveness sharing,
  unforgeable handles, fusion as a merge, and an independent `check`.
- **Tile slots with a 16-byte header, 4160 B.** Any slot copies to any other under C64,
  and an output slot is an operand slot (`matmul.rs:560-568`). 1.6% of GDDR for no
  alignment bugs is a good trade. Keep it per format (G9).
- **Per-channel bases** in `TensorRef` rather than tt-metal's lock-step bank offsets: no
  allocation must fit in every bank at the same address, so less fragmentation. The record
  expander computes the address once per tile.
- **Bit-exactness as the oracle.** tt-metal tests with PCC/ulp tolerances. We keep
  accumulation orders equal to Flex's or to a stated order (`COL_SUM`, `shard.rs`'s split
  along N), which is why G3 prefers Dst reload over `PACKER_L1_ACC` and G14 states the
  ring order.
- **Element-wise on the B core today** is a staging point (S1 moves it), not a design; but
  keep it as the reference and the fallback (S1 says so).
- **No slow-dispatch vs fast-dispatch split.** There is one path, gated on ttsim and
  silicon alike. G8 evolves it; it does not add a second one.
- **ttsim-first gates.** tt-metal's tests are silicon-first. The two-gate rule stays.

---

## 5. Open questions

1. Does ttsim model device → host writes into pinned host memory (G8 step 2)? If not, that
   path is silicon-only and needs a divergence row and a ttsim stand-in.
2. Is MVMUL's Dst accumulation strictly sequential FP32 per `MVMUL`, so that reload is
   bit-identical (G3)? The math says yes. Check that the 8-row halves and fidelity phases
   do not hold an internal wider accumulator. Probe: compare `[1, 64, 1]` in one run
   against two K blocks with reload, on ttsim and silicon.
3. Does NC on Blackhole have the IRAM/L1 instruction-fetch constraints that Wormhole's
   NCRISC had (G6)? The BH `BabyRISCV` pages decide.
4. NIU multicast on Blackhole: which NoC and which rectangle orientations are legal, and
   does ttsim model it (TLB `strided` is row 4 and refused)? Decides whether G7's gate is
   ttsim at all.
5. Do NoC atomics into L1 work on ttsim (G7's L1 semaphores)?
6. Is packer edge masking (`Packers/EdgeMasking.md`, WH-only page) present on Blackhole?
   It would make G1's fill-pad free inside the producing op.
7. For N-D tensors (G2), pad each leading slice (tt-metal) or pack slices densely when
   `H % 32 != 0`? Per-slice padding keeps every kernel a 2-D kernel. Dense packing saves
   memory but makes tiles straddle slices. Recommend per-slice.
8. Should `burn-tt` refuse rather than fall back when a user asks for strict residency
   (`TT_STRICT=1` → a panic naming the op)? This would be useful in gates: 9.5 asserts
   by shape today.
