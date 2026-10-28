# The Dataflow Compiler: Design Principles

**Crates:** `tt-isa`, `tt-kernels`, `tt-firmware`, `burn-tt`
**Classification:** Tier 1. A living architecture document, the reference that every optimization below Burn builds on. Its binding decisions are ADRs [0003](../adr/0003-a-tile-level-dataflow-compiler-below-burn.md), [0004](../adr/0004-formats-are-edge-data-converted-only-at-unpack-and-pack.md) and [0005](../adr/0005-every-op-has-a-numeric-contract-schedules-choose-only-among-bit-equivalent-options.md). The staged work is in [the plan](../plans/dataflow-compiler.md).
**Hardware target:** Blackhole A0, 120 Tensix tiles a card on ours (12 of 14 columns).
**Written:** 2026-10-28. Card numbers are from card 0 unless a run says otherwise.

---

## 0. How to read and change this document

**Evidence decides.** A principle is in here because evidence supports it, not because Tenstorrent did it, and not because Tenstorrent did not. Where Tenstorrent's design and an alternative disagree, both are recorded with their evidence (§10), along with the experiment on our hardware that decides between them.

Every principle carries an evidence tag:

| Tag | Meaning |
|---|---|
| **[HW]** | A hardware fact: `vendor/tt-isa-documentation`, `cfg_defines.h`, or a measured entry in [silicon-operating-notes.md](silicon-operating-notes.md) |
| **[MEAS]** | Measured in this repository; cites a run ID or a [performance-log](performance-log.md) entry |
| **[PUB]** | Published result (paper, tech report, source tree); cited in §14 |
| **[HYP]** | A hypothesis. Names the experiment that confirms or refutes it, and stays provisional until that runs |

Every principle also says what would **overturn** it. To change a principle, record the measurement or source that overturns it, edit the principle and its tag, and, if an ADR states it, supersede the ADR. A refuted hypothesis is kept here, struck through or marked, with its run, as `roadmap.md` does for refuted ideas.

## 1. What "dataflow" means here, and why a compiler

**A dataflow program is a graph of sequential processes that exchange tiles over bounded channels, placed in space.** Data moves explicitly from one core's SRAM to another's. A process runs when its inputs have arrived and its outputs have room. Contrast the SIMD/GPU model that Burn's backend API assumes: an op is a kernel launch over global memory, and the next launch starts once the previous one has finished. Caches and the runtime hide the movement. Blackhole has no caches between its L1s and GDDR, and five processors per core that move, unpack, compute and pack independently [HW] (§2). It runs the SIMD model correctly, but only by making every op a round trip to GDDR with a barrier at the end.

**What this repository already does dataflow-style:**
- Within a core, reads, compute and writes overlap through credits and double-buffered L1 halves (`Session::enable_dram`, [streaming-dataflow-architecture.md](streaming-dataflow-architecture.md)).
- Traces remove host dispatch.
- The residency planner keeps graph-internal tensors in L1 and writes producers' outputs into their readers' L1 ([planned-l1-residency.md](../plans/planned-l1-residency.md)).
- Weight prefetch reads the next product's operand while earlier ops run.

**What it does not do** (survey of 2026-10-27):
- An op is a `Work` of independent jobs that each read and write only GDDR (`tensor.rs` `Work`). Every op ends in a cross-unit barrier.
- One output tile sits in `Dst` at a time (`matmul.rs`, `matmul_roles`).
- The roles retire every batch together (`tt-firmware/src/corpus.rs`, `retire_batch`).
- No op reuses an operand across cores, and no core grants another a credit.
- Each format has its own path: three matmul builders (`tensor::matmul_dram*`, `bf16::matmul`, `bfp::matmul`), 145 generated wrappers that widen BFP to F32 and back, and ~110 hand-written format-equality checks.
- No op has a description of its iteration space that a transform could read, so every optimization is written once per op.

**Measured consequences** [MEAS], from [performance-scoreboard.md](performance-scoreboard.md):
- Products at prefill shapes reach 5-18 TFLOPS of ~166 (HiFi4, 120 tiles). The scoreboard's own diagnosis: "a blocked product is bound by what it reads".
- A 1.7B BF16 decode step reads 3,568 MB at 328 GB/s, which is 64% of 512, and runs at 90%+ for only 1.9 of 10.9 ms. Every product ends in a tail (profile of 2026-10-27, [log](performance-log.md#the-cards-gddr-reads-over-a-decode-step-2026-10-27-card-0)).
- A small decode op costs 9.84 µs, ~3 µs of it its barrier ([log](performance-log.md#which-barriers-a-decode-step-needs-2026-10-26-card-0)).

**Why a compiler, not more hand-written ops.** These gaps share a cause: decisions that span ops, cores or formats have nowhere to be made. Fixing them op by op, format by format, multiplies the work by (ops × formats × transforms). Fixing them once needs:
- a description of an op that transforms can read (§5),
- formats as data (§4),
- one channel abstraction (§7),
- one numeric contract that bounds what any transform may change (§3).

That is the compiler. It is **not** a multi-dialect stack. Compile time stays in microseconds for eager ops, and longer only for captures (trace builds may be slow; steps may not). It is native Rust over the existing L3 (§5). This reverses "Why We Do Not Need a Second Graph Compiler" in [kernel-fusion-architecture.md](kernel-fusion-architecture.md) §4 and keeps that section's concern (ADR 0003).

---

## 2. Fundamental truths of the machine

Each truth is a fact the compiler models. None of them is a preference.

**T1. The memory hierarchy** [HW] [MEAS]

| Level | Capacity | Rate / cost | Source |
|---|---|---|---|
| `SrcA`, `SrcB` | 2 banks each of 64 rows × 16 columns × 19 bits | Read by every `MVMUL` | `WormholeB0/.../SrcASrcB.md` |
| `Dst` | 512 × 16 × 32 bit (8 FP32 tiles), or 1024 × 16 × 16 bit (16 tiles); same storage | Accumulates at the matrix unit's rate | `BlackholeA0/.../Dst.md` |
| L1 | 1536 KiB a tile. `DATA` is 896 KiB at `0x20000..0x100000`, plus `CAPTURE_ONLY` 88 KiB for captures; the rest is images, programs, mailboxes and the program cache | Single-cycle-class to the tile's cores | `BlackholeA0/TensixTile/L1.md`; [notes](silicon-operating-notes.md#the-tensix-l1-map-and-changing-it-2026-10-22-ttsim-and-card-0); `tt_isa::l1` |
| NoC (2 of them) | 64 B a flit at 1.35 GHz | Point-to-point and rectangle multicast | `BlackholeA0/NoC/README.md` |
| GDDR6 | 32 GiB in 8 channels | 512 GB/s peak, and our movers reach it (507-511 with `fast-reads`). A decode step averages 328 | Scoreboard ceilings table; F27 notes |

**T2. Compute** [HW] [MEAS]
- **Matrix unit.** One `MVMUL` multiplies an 8×16 `SrcB` block by a 16×16 `SrcA` block into `Dst`. It issues once a cycle per fidelity phase (`MatrixUnit.md`). That is 4,096 FLOP a cycle a tile at one phase, ~166 TFLOPS at HiFi4 on 120 tiles at 1.35 GHz (derived; scoreboard "Roofline").
- **The SFPU is a vector unit,** orders of magnitude below the matrix unit (`MatrixUnit.md` quotes `SFPMAD` at 1/64 of `MVMUL`'s Wormhole rate). Every non-contraction op is bound by the SFPU, by data movement, or by instruction push (`.ttinsn` pushes at 1.19 cycles a word, [notes](silicon-operating-notes.md#ttinsn-push-fusion-2026-10-10-card-0)).

**T3. Formats are converted only at the unpacker and packer boundaries.** Compute happens in three numeric families [HW] (`Unpackers/FormatConversion.md`, `Packers/FormatConversion.md`, `SrcASrcB.md` "Data types"):

| Family | In `Src` | In `Dst` | L1 formats that unpack into it |
|---|---|---|---|
| **e8** | TF32 or BF16 | FP32 or BF16 | FP32 (to TF32 or BF16), BF16, BFP8/4/2. BFP→BF16 is lossless |
| **e5** | FP16 | FP32 or FP16 | FP16, FP8 (e5m2), BFP8a/4a/2a, all lossless |
| **int** | Integer "8" | Integer "32" | INT8, UINT8 (and INT32 to `Dst` only) |

`MVMUL` accepts only same-family operand pairs (`SrcASrcB.md`, the combination table). So **a storage format changes only three things: the tile's bytes, the unpacker's and packer's configuration, and the cost model. It never changes the schedule of compute.** Narrowing at pack has fixed semantics, and they belong in the contract:
- round-to-nearest, ties away from zero, or stochastic;
- denormals flushed;
- −0 becomes +0;
- an e8 NaN becomes infinity on rounding (`Packers/FormatConversion.md`).

**T4. The fidelity phases an exact product needs follow from the operand formats** [HW] (`SrcASrcB.md`, "Fidelity phases"):

| Operand in `Src` | Phases for full precision |
|---|---|
| FP8, BFP4(a), BFP2(a) | 1 |
| BFP8(a) | 2 |
| BF16 | 4 |
| TF32, FP16 | 4, still one `SrcA` mantissa bit short |

Narrower formats need fewer phases for the same exactness. `Dst` holds full precision whatever the phase count.

**T5. The matrix unit's accumulation is a defined, order-dependent rounding** [HW] [MEAS] ([notes](silicon-operating-notes.md#matmul-accumulation-the-matrix-units-group-adder-2026-10-11-card-0-and-ttsim), gate `step158_matmul_accumulation`):
- An `MVMUL` is a 16-term dot product made of two groups of 8.
- Each group is aligned to a quantum `2^(max exponent sum − 10)`, its products rounded half-up, and summed exactly.
- Each group sum is added to FP32 `Dst` with ties toward +∞, phases in order, faces in order.

So **an output's bits depend on how its K terms are grouped and in what order they are added**. Anything that preserves K order and grouping per output preserves the bits. Packing an FP32 partial and unpacking it again to FP32 `Dst` is an identity [HW]. That is why K chunking with partials reloaded between chunks is exact.

**T6. Five processors a tile, sharing the coprocessor** [HW] [MEAS]
- The roles are B (reads, NoC0), NC (writes, NoC1), and T0/T1/T2 (unpack/math/pack).
- T0-T2 share the configuration unit and 8 Tensix semaphores (`SEMINIT`/`SEMPOST`/`SEMGET`). Unpack hands to math through `Src` data-valid flags.
- Today all three retire each batch together "because they share configuration and semaphores" (`corpus.rs`). Any overlap across batches must make the configuration batch-invariant or fence it.

**T7. NoC ordering, atomics and multicast** [HW] [MEAS]
- **Ordering:**
  - Packets on the same VC that share a route never reorder (`WormholeB0/NoC/Ordering.md`).
  - A write followed by an atomic has no ordering guarantee. Requests on different NoCs have none either.
  - Ordering a *credit after data* therefore needs the data's write acknowledgements (a fence), or a linked or same-path transaction.
- **Atomics:** an atomic increment targets any Tensix L1, including a multicast rectangle (`BlackholeA0/NoC/Atomics.md`).
- **Multicast:** rectangles of Tensix tiles only. Path-reserved multicasts that share a rectangle **serialize**: 64 units multicasting into one 104-tile rectangle took ~250 µs against ~14 for the GDDR writes [MEAS] ([notes](silicon-operating-notes.md#nc-writes-another-tiles-l1-2026-10-20-ttsim-and-card-0)). One sender per rectangle is the safe form.
- **Endpoint ownership:** each GDDR endpoint is owned by one NoC per channel (the SYS-1419 rule; B reads on NoC0, NC writes on NoC1).
- **Transaction IDs:** their outstanding counters are the tile's, not the core's [MEAS].

**T8. GDDR is one queue, and service is unfair** [MEAS] ([notes](silicon-operating-notes.md#gddr-reads-are-one-queue-prefetch-traffic-is-paid-by-the-small-ops-2026-10-20-cards-0-and-1))
- Prefetch traffic is paid for by the small ops that wait behind it.
- Units reading together are served by position. Two reads in flight per unit evens it out.
- A unit's read rate is set by its entry size: ~6.5 GB/s for 2 KiB entries on the safe image, 20-35 with `fast-reads`. Bank-local readers add nothing (F27, 2026-10-27).

**T9. Silicon keeps state; harvested tiles hang the NoC** [HW] [MEAS]. L1 and configuration survive between programs and processes. A program establishes every piece of state it relies on. Grids come from ARC (`ENABLED_TENSIX_COL`). An access to a harvested tile can reboot the host ([notes](silicon-operating-notes.md)).

**T10. Fixed costs** [MEAS]
- **Host enqueue:** ~3.4 µs a unit an op untraced. Trace is the normal execution model for anything repeated.
- **Traced small op:** 9.84 µs on the 135M, of which the barrier and its gap are ~3 µs. Removing the barriers that order nothing the data needs took the 135M step from 7.35 to 6.97 ms. Sparse waits after products would save only 0.03-0.07 ms, because the barrier's time there is the slowest unit's work ([log](performance-log.md#which-barriers-a-decode-step-needs-2026-10-26-card-0)).

---

## 3. Semantics that never change

**S1. Every process follows Kahn's discipline** [PUB]:
- A process is sequential and deterministic.
- A read blocks until its data has arrived.
- No process chooses between inputs by which arrived first.
- Atomics carry credit counts only, never data.

Kahn's theorem then gives every channel the same history under any schedule, any speed and any placement (Kahn 1974; proved by Faustini 1982 and Stark 1995). **This is the guarantee that lets the compiler schedule freely without touching bits.** Increments commute, so counting credits atomically is deterministic. Accumulating data atomically is not, and is forbidden. *Overturned by:* nothing short of giving up bit-exactness.

**S2. Every op has a numeric contract.** It is part of the algorithm, not of the schedule [MEAS: the stack's standing rule, `tt-metal-concepts-review.md` §4; [PUB] Stream-K and FlashAttention show that splits change bits]. A contract states:
- **Reduction order:** for each reduction dimension, its chunk size and combine tree. Default: K in ascending tiles, `MVMUL` groups as the hardware forms them, partials carried through FP32 `Dst`.
- **Accumulator:** FP32 `Dst` by default. A 16-bit `Dst` is a different contract.
- **Fidelity rule:** HiFi4 today. "Exact for the operand formats" is the candidate rule (H3, §13).
- **Rounding points:** every pack that narrows, with its mode (T3).
- **Approximation:** an SFPU program's own semantics, fixed by the SFPU interpreter model.

The reference for "bit-exact" is the op's **specification model**: the matmul rule of T5, the SFPU interpreter, and the mover's copy semantics. Burn's own results (burn-flex) are checked within bounds derived from the contract, never a guessed tolerance (AGENTS.md).

**S3. A schedule chooses only among bit-equivalent options.** Tiling and distributing *parallel* dimensions is always legal. A reduction dimension may be split or reordered only in the way its contract states. A transform that cannot prove equivalence under S2 is either a **different contract**, opt-in and named (as `MatmulMode::{Fast, Exact, Accurate}` are today), or it is not done. Autotuning searches only within the equivalent set, so tuning never changes results.

**S4. Splits are fixed by the contract, never by core count** [PUB, inference from Stream-K]. A result must not change when the grid, a session's tile count or the card changes. If a load-balancing split is wanted, its split points (K-chunk size, combine order) are part of the contract and independent of the machine.

**S5. Determinism in values, not in time** [PUB: Groq buys time-determinism with a fully static machine]. Kahn semantics fix values. Timing is left to credits and is measured, not scheduled to the cycle.

---

## 4. Formats are data

**F1. One descriptor.** A `TileFormat` descriptor in `tt-isa` holds everything format-dependent:
- the L1 code, datum bits, and exponent block (16 datums for BFP);
- the image layout and bytes (exponent section, then mantissas);
- the slot bytes (16-byte header, C64 congruence);
- the numeric family (T3);
- the legal unpack routes to `Src` and to `Dst`, and pack routes with early and late conversions;
- the mantissa bits each fidelity slice consumes (T4);
- the special-value behaviour on narrowing (T3).

It replaces what is scattered today across seven types in four crates: `L1Format`, `TileImage`, `Elem`, `BfpFormat`, `SrcRoute`/`SrcFormats`, `StorageFormat`, `HostDtype`, plus the bare constants `TILE_SLOT = 4160` and `BF16_TILE_SLOT = 2112`. [PUB] tt-mlir parameterizes its tile type by element type (`!ttcore.tile<32x32, bfp_bf8>`). TT-Metal keeps one `DataFormat` enum but checks legality at run time, and its implicit format state is a known hazard (`reconfig_data_format` trusts the caller's claim about the old format).

**F2. A format belongs to an edge, not an op.** An op's body is written against numeric families. Each operand and result edge carries a `TileFormat`, and the unpacker and packer convert at the boundary. A mixed-format product such as BFP8 weights × BF16 activations is one op with two edge formats in one family, not a fourth matmul builder. Buda attached formats to edges (`in_df`/`out_df` per operand) and that part worked [PUB]. Its precision *policy*, set by environment-variable heuristics (`PYBUDA_AMP_LIGHT`), did not work.

**F3. The compiler owns all format state.** Every unpacker and packer configuration a program relies on is written by that program (T9). Nothing trusts a claim about the previous format.

**F4. Never split a BFP exponent block.** An exponent covers 16 datums, one face row. A view, slice or pad that cuts a block must convert through a wider format on purpose and say so. It must never reinterpret.

**F5. Adding a format is adding a descriptor row.** New code is needed only for a new numeric family (none is planned) or for a conversion the hardware does not do (`HostDtype`/`tt-layout` on the host). The test for any optimization: *does it name a format?* If it does, it is in the wrong layer.

**F6. Generic fidelity** [HYP: H3]. Fidelity comes from the edge formats (T4), not from a knob for each format. Under the default contract, "exact for these formats" would make BFP4×BFP4 run 4× fewer phases than HiFi4 at the same exactness. Whether it gives the *same bits* as HiFi4 is H3.

---

## 5. The IR stack

The stack is lowered progressively, and each level keeps what the next needs ("don't lose information too quickly" [PUB] MLIR Linalg rationale). There is no raising.

**L0, the Burn graph.** burn-fusion's lazy stream, and a trace's capture (`OpRecord`, the residency graph). The compiler takes graphs from here. It does not ask Burn to know about placement.

**L1, the tile-generic op.** This is a design sketch; names will change in code.
```
TileOp {
  domain:   [Iter { extent_tiles, kind: Parallel | Reduction(ReductionContract) }],
  operands: [Access { map: TileMap /* domain index -> operand tile index, affine */,
                      edge: Edge { format: TileFormat, pad: OpPadding } }],
  results:  [Access { .. }],
  body:     Body /* microkernel calls on Src/Dst: Mma, FpuEltwise, Sfpu(program), Reduce, Copy */,
  contract: NumericContract,
}
```
Indexing maps plus iterator kinds are what `linalg.generic` and tt-mlir's `d2m.generic` use [PUB]. Tiling, fusion and distribution are written once against this interface. The body is the only op-specific part, and its microkernels are the only arch-specific part. That is BLIS's split [PUB], and it holds here because formats stop at the boundary (T3).

**L2, the dataflow graph.**
```
Process { role: Reader | Compute | Writer, cores: CoreSet, program }
Channel { producer: Process, consumers: [Process], depth, page: TileFormat × tiles,
          kind: Local(CB) | Remote(credit) | Broadcast(multicast) | Gddr }
```
A channel has one producer and N consumers. Broadcast lowers to multicast, and distribute, join and link are channel operations, as in MLIR-AIE's `objectFifo` [PUB]. The existing in-core streams (`Input`, `Output`, `Transfer` in `tt_isa::dataflow`) are the single-core case. **The movement program is first-class, readable IR**: Buda's undocumented overlay was a black box [PUB].

**L3, the existing machine form.** `Step`/`Job`/`Work`, mover records, `code::Code` and `loops`, `l1::Requirements`, credits and semaphores, and the word-keyed program cache. The compiler emits these. It does not replace them.

**Pass order** (after D2M's pipeline [PUB], adapted to our L3):
1. generic op (L1)
2. grid selection
3. fusion
4. L1 blocking
5. allocation (`l1::Requirements`, residency)
6. explicit channels (L2)
7. `Dst` tiling: subblocks, half-sync
8. role split
9. credits and multicast
10. mover schedule
11. L3 words

**Every resource is accounted exactly before code is emitted:** L1, `Dst` rows, the 8 semaphores, the 8192-word program slot, mover image room, entry counts and in-flight limits. Buda found its limits after the fact: up to 20 placer retries and recompiles on overlay overflow [PUB]. Here, exceeding a limit is a plan-time decision or a typed refusal, never a retry.

**Compile time.** An eager op compiles in microseconds and is cached by its words. A capture may take longer, because trace build time is not a target, but a replay may not. Dynamic shapes are handled by bucketing captures [PUB: TT-Metal traces are frozen by shape].

---

## 6. Transforms, written once

Each transform states its legality under S2/S3, its expected effect, and the evidence for it.

| # | Transform | Legal under the default contract? | Status |
|---|---|---|---|
| X1 | **L1 blocking**: output block `mb×nb`, K in chunks with FP32 partials reloaded | Yes. K order and groups unchanged (T5) | [MEAS] exists (`block_stream_layout`), modelled per shape. The widest block is not always best: Qwen's `gate\|up` ran faster narrowed ([notes](silicon-operating-notes.md#a-region-at-the-top-of-l1-costs-a-product-only-past-its-plans-floor-2026-10-21-card-0)), so the model is calibrated, not trusted |
| X2 | **`Dst` subblocks and half-sync**: `h×w ≤ 4` FP32 tiles a half; math fills one half while pack drains the other | Yes. Each output's K order is unchanged | [HYP] H2. Not first: a K-tile pair already runs in 0.14-0.23 µs and products are bound by their reads (scoreboard row 75) |
| X3 | **Grid distribution**: parallel dimensions over cores | Yes (S4: never a reduction split) | [MEAS] exists (`Spread`). Generic over a work measure |
| X4 | **Operand reuse across cores**: one reader per row/column rectangle, multicast or pulled | Yes. Moves bytes only | [HYP] H1. Contended multicast is measured slow (T7). Pulling a row from a few holders measured cheaper than GDDR ([notes](silicon-operating-notes.md#nc-writes-another-tiles-l1-2026-10-20-ttsim-and-card-0)). Which wins for products is decision D5 |
| X5a | **Register fusion** (prologue/epilogue on `Dst`) | Yes, **if** it replays every rounding the unfused graph does. Where the unfused intermediate was packed narrower than FP32, the fused body rounds to that format in place | [MEAS] compound SFPU kinds (`SILU_MUL`, …) are gated bit-for-bit against their composition |
| X5b | **L1 channel fusion** (producer and consumer on one core) | Yes, under the same rounding rule | [MEAS] `Requirements::fuse` exists, with no production caller |
| X5c | **Cross-core channel fusion** (producer writes the consumer's L1) | Yes | [MEAS] residency Phase 3, in captures |
| X6 | **Barriers become channel edges**: a unit starts when its own inputs are ready | Yes (S1) | [HYP] H4. The evidence cuts both ways (T10): a barrier's time after a product is the slowest unit's work, so removing the barrier alone buys little. The gain has to come from overlap, below |
| X7 | **Spatial pipelining**: ops on disjoint core sets at once, linked by channels | Yes (S1) | [HYP] H5. Only on exact resource accounting. tt-mlir brought it back as `d2m.spatial` [PUB] |
| X8 | **Prefetch** of data-independent operands (weights) | Yes | [MEAS] exists. It competes with small ops for the one GDDR queue (T8) |
| — | Split-K, flash-attention rescaling, packer L1 accumulation, 16-bit `Dst`, fidelity below exact | **No**: different contracts | Opt-in only, by name. Packer L1 accumulation was rejected as a default because it reorders the sum (G3); split-K and flash-decode are parked ([proposal](../proposals/split-k-and-flash-decode.md)) |

**X5's rounding rule makes fusion format-generic.** A fused body equals the unfused graph exactly when every pack of the unfused graph that narrows is replayed as an explicit rounding to that edge's `TileFormat`. Fusion then never depends on which format the intermediate had.

**X6 and overlap.** A barrier orders three things: (a) data dependencies, (b) reuse of L1 and programs, and (c) nothing at all (the unneeded ones, already dropped in captures). Channels express (a) exactly, as a credit per edge, and planned lifetimes express (b). What remains to gain is **overlap**:
- A unit that finishes its share of a product starts work that does not depend on the product, such as the next product's weight reads.
- A consumer starts as soon as *its* producers have written, not when all units have.

The 1.7B profile shows where this could pay: products start near 634 GB/s and end in tails near 68-89 GB/s, at 90%+ of the card's rate for only 1.9 of 10.9 ms. Whether channels fill those tails, or the GDDR queue (T8) just moves the wait, is H4.

---

## 7. Scheduling, buffers and deadlock

- **Rates.** Every channel declares its tokens per firing at both ends: synchronous dataflow, or cyclo-static for phased kernels (Lee & Messerschmitt 1987; Bilsen et al. 1996) [PUB]. The balance equations give how often each process fires per graph iteration. An inconsistent graph is refused at plan time.
- **Credits are reverse edges.** A channel of depth `d` is a forward data edge plus a reverse edge holding `d` initial credit tokens [PUB] (SDF3). This is exactly the `reserve`/`push` and `wait`/`pop` protocol of circular buffers, which Buda and TT-Metal both kept [PUB], and which our streams already implement within a core.
- **Deadlock freedom is checked before emission.** Every cycle in the graph with reverse edges must hold enough initial tokens. A graph that fails is refused, not run. Buda's rule that a blocking op's output buffer hold a block "to prevent deadlocks" [PUB] is one instance of this.
- **Depth.** Depth 2 (double buffering) is the default, as for TT-Lang and `objectFifo` [PUB] and our own `pipelined_runs`. More depth is chosen from the buffer-versus-throughput trade-off [PUB] (Stuijk et al.) and L1 room, and modelled throughput is the reciprocal of the iteration bound.
- **Credit granularity is a cost decision.**
  - Today a credit is a compute-sized batch: depth 1 or 2, at most 40 batches a region.
  - A credit per tile is TT-Metal's default [PUB].
  - Finer credits overlap more and synchronize more. Which granularity pays is decision D6.
- **Cross-core credits** are NoC atomic increments into a consumer's or producer's L1 word, ordered after the data by its write acknowledgements (T7).
  - Their L1 words and semaphores are declared in `l1::Requirements` like any buffer, and their lifetimes cover trace holds.
  - Each program writes their initial values (T9).
  - TT-Metal's compiler needed a "forge_backdoor" to place global semaphores [PUB]. Here placement belongs to the planner from the start.
- **Barriers remain only** where no channel can express the dependency:
  - a dependency through GDDR whose reader or writer sets are unknown at plan time;
  - a host synchronization point;
  - reuse of a fixed region that the planner does not track.

---

## 8. The cost model

- **A roofline whose inputs come from the format descriptors.** For each candidate schedule of an op instance, it estimates:
  - bytes moved per edge (tile bytes from `TileFormat`, times reads per tile under the blocking and reuse);
  - `MVMUL`s × phases (T4);
  - SFPU and other instruction words × push rate;
  - unpack and pack cycles per tile per format route;
  - mover entries × entry cost (~359 cycles safe, ~91 with `fast-reads` [MEAS]);
  - the card's GDDR and NoC ceilings [MEAS];
  - fixed per-op costs [MEAS].
- **Calibrated, not trusted.** Each coefficient comes from a microbenchmark per (microkernel, numeric family, route), run in the profile mode (ADR 0002), with its run ID kept beside the coefficient. Analytic models can match expert tuning (BLIS [PUB]). Learned models win where effects interact (TPU [PUB]). Ours has already been wrong once (X1), so it is calibrated and checked.
- **Greedy per-op decisions first.** tt-mlir removed its chain solver: over 50+ models, greedy per-op choices were right in every case they analysed, and spills were driven by constraints, not memory [PUB]. A global solver comes only when a measurement shows greedy losing (D4).
- **An autotune cache is keyed by `(op, shape, edge formats, contract)`.** It only chooses among bit-equivalent candidates (S3). Per-model, hand-picked configurations, like TT-Metal's matmul program configs, are an anti-pattern [PUB]. The search space comes from the transforms, not from people.

---

## 9. How to extend

**A new format:**
1. Add its `TileFormat` row: code, image, slot, family, routes, fidelity slices, narrowing semantics, each from `vendor/` or measured.
2. Add host conversion in `tt-layout` if the host must produce it.
3. Add a format-matrix gate row. It must pass bit-exact on every op that accepts its family.

No op code changes.

**A new op:**
1. Write its `TileOp`: domain, maps, body over families, and contract.
2. Write its specification model, which predicts bits.
3. Pass the new-op contract ([new-op-definition-of-done.md](../plans/new-op-definition-of-done.md)): oracle, simulator gate with a negative control, silicon on one card, Burn routing, padding and residency.

Tiling, distribution, fusion, reuse, channels and formats come from the passes.

**A new microkernel** (a body primitive):
1. Write its semantics and implementation pair, so passes can match it (TensorIR [PUB]).
2. Give its resource use (`Dst` rows, `Src` banks, semaphores, program words).
3. Supply a cost-model calibration.

**A new transform:**
1. State its legality under S2/S3.
2. Write a falsifiable prediction with a kill criterion.
3. Gate it bit-exact against the unscheduled composition and against the previous schedule, with a negative control that breaks equivalence on purpose.

---

## 10. What other systems teach, and the decisions they leave us

**Adopt, with evidence:**

| From | What | Why |
|---|---|---|
| Kahn, SDF/CSDF, SDF3 | Process/channel semantics, rates, credits as reverse edges, deadlock and buffer analysis | [PUB] theorems; bit-exactness under any schedule |
| `linalg.generic`, `d2m.generic` | Indexing maps and iterator kinds; staged lowering | [PUB] transforms written once; tt-mlir's own pipeline |
| BLIS, CUTLASS | Only the microkernel and the converters are arch- or format-specific; convert at the operand boundary | [PUB]; matches T3 |
| MLIR-AIE `objectFifo` | One producer, N consumers, depth, broadcast/distribute/join | [PUB]; lowers to locks and DMA like our credits and movers |
| Buda | Per-edge formats; spatial epochs with core-to-core pipes | [PUB] the part that worked |
| TT-Metal | Circular buffers plus `noc_semaphore_inc` credits; reader/compute/writer split; trace | [PUB] and [MEAS]: ours already does this within a core |
| TT-Lang | Depth-2 buffers by default; a verifier for pipe roles | [PUB]; channel roles checked at compile time |
| SambaNova SN40L | Cores in proportion to work; static lifetime memory planning; a static bandwidth model | [PUB] |
| Graphcore Poplar | Running out of memory is a compile error | [PUB]; matches `PlanError` |
| Cerebras | Keep a layer-at-a-time fallback for graphs that do not fit spatially | [PUB] layer-pipelined mode only fits models under ~1B parameters |

**Avoid, with evidence:**
- Discovering backend limits by retrying (Buda) [PUB].
- Per-model, hand-picked program configurations (TT-Metal matmul configs) [PUB].
- Precision policy in environment variables or heuristics (Buda AMP) [PUB].
- Implicit format state (`reconfig_data_format`) [PUB].
- Two sets of defaults for one config (TT-Metal `ComputeKernelConfig` LoFi vs a helper's HiFi4) [PUB].
- Tests by correlation or ULP where a model can predict bits [PUB]. None of Buda, TT-Metal or tt-mlir promises bit-exact results; we do, so every conversion and rounding point is specified here.
- Compile times of minutes to hours (Cerebras weight streaming) [PUB].
- Cycle-level static scheduling that gives up flexibility (Groq) [PUB].

**Decisions between Tenstorrent's design and an alternative:**

| # | Question | Tenstorrent | Alternative | Status and deciding evidence |
|---|---|---|---|---|
| D1 | Lowering | Staged (D2M pipeline) | One builder per op (ours today) | **Provisional: staged.** Information-preserving lowering is what makes transforms generic. Confirmed if the first three clients (§13 H6) need no per-op transform code |
| D2 | Partials across K chunks | Packer L1 accumulation | Reload into FP32 `Dst` | **Decided: reload** under the default contract (L1 accumulation changes rounding, G3). L1 accumulation stays available as a named contract |
| D3 | Decode weight readers | DRAM-sharded, one reader beside each bank | Interleaved `READ_RUN` stream | **Decided 2026-10-27: no gain.** At equal entry sizes, 8 readers a channel equal today's deal; with reads 5× cheaper the step moves only 10.82 → 10.60 ms (F27, [log](performance-log.md#f27-probed-bank-local-reads-and-what-does-not-bind-a-decode-product-2026-10-27-card-0)) |
| D4 | Layout search | Greedy (tt-mlir, after removing a solver) | Global solver | **Provisional: greedy** [PUB]. Revisit only on a measured loss |
| D5 | Operand reuse in products | 2D multicast matmul | A few holders pulled by `READ_L1`; or GDDR re-reads with larger blocks | **Open.** Contended multicast is slow (T7); pulled rows are measured cheaper for rows. Decided by H1 |
| D6 | Credit granularity | A page (often a tile) | A compute-sized batch (ours) | **Open.** Decided per channel class by the cost model once H4 measures sync cost against overlap |
| D7 | Replicating rows to many readers | Multicast | Sharded rows | **Decided: sharded.** Replicated rows by contended multicast took 43.9 ms a step against 20.92 (roadmap, residency 2e/2f) |
| D8 | `Dst` width | BF16 `Dst`, 8 tiles a half, often the default | FP32 `Dst`, 4 a half | **Decided: FP32** in the default contract (S2). 16-bit is an opt-in contract |
| D9 | Fidelity | A per-op knob, LoFi..HiFi4 | Derived from edge formats | **Open: H3** |

---

## 11. What stays, and what this supersedes

**Stays** (each measured or decided elsewhere):
- instruction streams as data;
- the word-keyed program cache;
- the L1 planner and `PlanError`;
- 4160-byte slots with C64 congruence (generalized to a slot per `TileFormat`, G9);
- B reads on NoC0 and NC writes on NoC1, with SYS-1419 endpoint ownership;
- the safe and `fast-reads` movers, with `encoded::validate` and `guard`;
- the build-time instruction gate and `tt_isa::ttinsn::validate`;
- traces and planned residency;
- bit-exactness against specification models;
- ADR 0001 (generic across LLMs) and ADR 0002 (one profile mode).

**Supersedes:**
- [kernel-fusion-architecture.md](kernel-fusion-architecture.md) §4, "Why We Do Not Need a Second Graph Compiler". Its concern, no slow multi-dialect stack, is kept as §5's compile-time rule.
- §6.2's `SfpuMicroOp` engine, which describes code that was never built. A body's SFPU microkernels are L1 `Body` calls (§5).
- That document's L1 budget figure ("~1.2 MB"). The arena is 896 KiB plus 88 KiB capture-only (T1).

---

## 12. Verification doctrine

- **Equivalence first.** Every transform is gated bit-for-bit against two references: the unscheduled composition (or the code path it replaces), and the specification model. Both run over a format matrix: every edge-format pair the op's family admits, plus signed zeros, subnormals, NaN, infinities, extremes and ragged padding.
- **Negative controls.** Each gate has a planted fault it must catch: a wrong subblock stride, a dropped credit, a channel with one credit too few (which must stall and be detected, never race), or a reordered K.
- **ttsim, then silicon on one card.** A primitive ttsim refuses gets a divergence entry and a silicon-only gate. Credit, multicast and NoC-atomic changes run an isolated silicon gate before any op uses them.
- **Performance claims are predictions first.** A hypothesis, a predicted effect and a kill criterion are written down before the build. The measurement is a release benchmark with validated output, warmups and medians. Its run ID is recorded here and in the scoreboard and log.

---

## 13. Hypotheses register

Each entry: the claim, the prediction, how it is measured, and what kills it. Results are recorded in place.

- **H1 Operand reuse.** Products at prefill shapes are bound by operand reads, so reading each operand from GDDR about once (D5's winner) lifts them toward the compute bound, for every format at once.
  - *Prediction:* `[m,2048]@[2048,16384]` at m = 2,048 from 18.2 to ≥ 35 TFLOPS. The BFP8-weight case improves in proportion to its bytes.
  - *Measured by:* `matmul_throughput` and the profile's card read rate.
  - *Kills it:* reads drop and TFLOPS do not.
- **H2 `Dst` subblocks and half-sync.** Once reads stop binding (after H1), the kernel's math utilization rises from ~41-68% of HiFi4 issue toward ~90%. That range is derived from the 0.14-0.23 µs measured per K-tile pair against 128 cycles of `MVMUL`.
  - *Prediction:* ≥ 1.4× on the same shapes, bit-exact.
  - *Kills it:* < 1.1×.
- **H3 Exact fidelity.** Running only the phases the operand formats need (T4) gives the same bits as HiFi4 for those formats, because the skipped phases' slices are zero and adding +0 to FP32 `Dst` is an identity.
  - *Measured by:* a ttsim and silicon gate across the format matrix, including −0 and the group adder's alignment quirk ("a zero slice still sets its exponent", T5).
  - *Prediction:* bit-identical, and BFP4×BFP4 products ≥ 3× the BF16 HiFi4 rate when compute-bound.
  - *Kills it:* any bit difference. Exact fidelity is then a separate contract.
- **H4 Channels instead of barriers.** With per-edge credits and data-independent reads started as units free up, a 1.7B decode step's card read rate rises from 328 GB/s (64%).
  - *Prediction:* ≥ 400 GB/s average and step ≤ 9.5 ms (from 10.9), bit-exact.
  - *Measured by:* the profile's card read meter.
  - *Kills it:* the step moves < 0.3 ms, or small ops slow by as much as the products gain (the one-queue effect, T8).
- **H5 Spatial pipelining.** Running a layer's ops on disjoint core sets at once beats running them one after another for decode at batch 1.
  - *Prediction:* none until H4 has run.
  - *Kills it:* H4's tails turn out to be GDDR-queue-bound.
- **H6 Generic ops.** The L1 description covers matmul, the 121 pointwise SFPU kinds and a row reduction with no transform code specific to any of them.
  - *Measured by:* lines of per-op transform code, and the three old builders deleted bit-exact.
  - *Kills it:* a transform needs per-op special cases. The description is then missing a concept, and it is amended here.

---

## 14. Bibliography

**Models of computation**
- G. Kahn, "The semantics of a simple language for parallel programming", IFIP 1974; summary and proofs: https://en.wikipedia.org/wiki/Kahn_process_networks, https://ptolemy.berkeley.edu/books/Systems/chapters/ProcessNetworksandRendezvous.pdf
- T. Parks, bounded scheduling of process networks: https://ptolemy.berkeley.edu/papers/parksThesis
- E. A. Lee, D. G. Messerschmitt, "Static scheduling of synchronous data flow programs", IEEE TC 1987: https://ptolemy.berkeley.edu/publications/papers/87/staticscheduling
- G. Bilsen et al., "Cyclo-static dataflow", IEEE TSP 1996
- S. Stuijk, M. Geilen, T. Basten, buffer sizing for SDF/CSDF: https://www.es.ele.tue.nl/sdf3/publications/tc_csdf_buffersizing.php

**Algorithm/schedule separation**
- Halide: https://people.csail.mit.edu/jrk/halide12/halide12.pdf
- MLIR Linalg rationale: https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/ and dialect https://mlir.llvm.org/docs/Dialects/Linalg/
- Ansor: https://arxiv.org/abs/2006.06762
- MetaSchedule: https://arxiv.org/abs/2205.13603
- TensorIR: https://arxiv.org/abs/2207.04296
- Exo (PLDI 2022)
- Triton: https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

**Spatial compilers and machines**
- Spatial/Plasticine: https://dawn.cs.stanford.edu/publications/spatial
- SambaNova SN40L: https://arxiv.org/html/2405.07518v2
- Cerebras execution modes: https://training-api.cerebras.ai/en/1.8.0/wsc/cerebras-basics/cerebras-execution-modes.html
- Graphcore glossary: https://docs.graphcore.ai/projects/graphcore-glossary/
- Groq TSP (ISCA 2020, 2022)
- MLIR-AIE objectFIFO: https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/section-2/section-2a/README.md

**Kernels**
- BLIS: https://www.cs.utexas.edu/~flame/pubs/sc13.pdf, https://www.cs.utexas.edu/~flame/pubs/TOMS-BLIS-Analytical.pdf
- Stream-K: https://arxiv.org/abs/2301.03598
- FlashAttention: https://arxiv.org/abs/2205.14135
- CUTLASS EVT: https://research.colfax-intl.com/epilogue_visitor_tree/
- ReproBLAS: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.html
- Learned TPU cost model: https://arxiv.org/abs/2008.01040
- XLA cost model: https://openxla.org/xla/lhs_cost_model

**Tenstorrent.** Local reference copies are at `/mnt/nvme/tt-metal` and `/home/carl/projects/tt-lang`. They are references only, never dependencies.
- Buda netlist: https://github.com/tenstorrent/tt-budabackend/blob/main/docs/public/netlist.md
- Buda ops and buffering: https://github.com/tenstorrent/tt-buda/blob/main/docs/public/backend/op.rst
- PyBuda compile loop: https://github.com/tenstorrent/tt-buda/blob/main/pybuda/pybuda/compile.py
- PyBuda data formats: https://docs.tenstorrent.com/docs-test/pybuda/latest/dataformats.html
- METALIUM_GUIDE: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
- Matrix engine: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md
- LLM tech report (matmul program configs): https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md
- Saturating DRAM bandwidth: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md
- `reconfig_data_format`: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/data_formats/reconfig_data_format.md
- Accuracy tips: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/op_kernel_dev/accuracy_tips/accuracy_tips.md
- D2M ops: https://github.com/tenstorrent/tt-mlir/blob/main/include/ttmlir/Dialect/D2M/IR/D2MOps.td
- D2M region ops: https://github.com/tenstorrent/tt-mlir/blob/main/include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td
- D2M pipeline: https://github.com/tenstorrent/tt-mlir/blob/main/lib/Dialect/D2M/Pipelines/D2MPipelines.cpp
- TTNN optimizer spec: https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/specs/ttnn-optimizer.md
- TT-Lang specification: https://github.com/tenstorrent/tt-lang/blob/main/docs/sphinx/specs/TTLangSpecification.md
