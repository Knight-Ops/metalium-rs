# Planned L1 residency: activations that survive between ops

Status: **scheduled; Phase 1 and Phase 2's mechanism done, 2f open** (2026-10-20; a proposal until then). Asked for after Phase 8 of
[decode-to-bandwidth.md](decode-to-bandwidth.md): the space audit, busy-unit prefetch and the prefetch pump.
It is the "sharded, L1-resident tensors" gap of [tt-metal-concepts-review.md](../learnings/tt-metal-concepts-review.md)
(G12). That gap was deferred "until a measured op chain is GDDR-bound", and that measurement now exists. It is designed
together with [layer-sub-traces.md](../proposals/layer-sub-traces.md): a planned layer has the same L1 addresses in every repeat.
Roadmap: Track 1, "Decode at GDDR's bandwidth".

## Why

Every op today reads its operands from GDDR into L1, computes, and writes its result back to GDDR. Nothing in L1
outlives an op. Phase 8 measured what that costs (1.7B, F32, every tile, cards 0 and 1, 2026-10-20):

- **The small ops wait on GDDR.** A decode step's ops other than the products read ~290 MB and write ~170 MB a step,
  almost none of it needed. Their reads are latency-bound: 40-75% of each op's time is waiting for data (profile
  `1792470407`).
- **They pay for any weight stream beside them.** With the next product's weights read ahead on the product's own
  units:
  - the products went from 15.24 to 11.6 ms;
  - the residual add went from 14 to ~100 us a call;
  - the step gained at best ~0.2-0.4 ms (`1792504522`, `1792505230`, `1792505109`).

  GDDR is one queue: whatever a window streams, the ops in it wait for
  ([operating notes](../learnings/silicon-operating-notes.md#gddr-reads-are-one-queue-prefetch-traffic-is-paid-by-the-small-ops-2026-10-20-cards-0-and-1)).
- **Weights free is 7.85 ms.** With the products' weight reads skipped, a step is 7.85 ms against 20.92 (the Phase 8
  probe). Most of the step is the weight stream, and the small ops cannot get out of its way.

If the activations between products never touched GDDR, the weight stream could take every window's whole bandwidth
with nothing waiting behind it.

## The mixed planner

**Today:** `l1::Requirements::plan` places one op's buffers first-fit from the bottom of `DATA` (896 KiB). The same
shape always gets the same addresses, and the arena is empty again after the op. Programs have their L1 addresses
compiled in, so they are cached by shape: one copy serves all 24 layers, and a trace replays unchanged.

**Proposed:** keep that planner for each op's scratch. Add a **persistent region** whose addresses are planned
statically, once, over a whole captured graph:
- **Statically:** each tensor's lifetime across the graph's ops is computed, then each tensor gets an address, as XLA's
  buffer assignment does.
- **Kept from the per-op planner:** determinism. The same graph always gets the same addresses, so programs stay cached
  by shape and traces need no patching.
- **Gained:** data that survives from the op that makes it to the ops that read it.
- **Not built:** a runtime heap with free lists and fragmentation in 896 KiB, and launch-time clash checks.

### Scope: planned graphs, not eager ops

- **What qualifies:** any captured graph. That is decode steps, prefill chunks, traced inference (ResNet, ViT) and
  **traced training steps**. Untraced eager execution keeps today's GDDR round trip, so it gains nothing here but
  risks nothing either.
- **Training:**
  - Activations that the forward pass keeps for backward live across most of the step. They mostly stay in GDDR
    (see "What lives in L1").
  - Tensors that live briefly, within the forward or within the backward, qualify.
  - Parameter and optimizer updates write GDDR in place, as they do now.
  - Training comes after decode, because its traces first need trace segmentation (they do not fit the program
    cache). This proposal and the sub-traces both help there.

### Two passes per capture

A capture runs its ops live: the capture is the first run. The plan must therefore exist before the planned capture
runs.
1. **Plan pass:** the graph is captured as today, all GDDR. Its op records are the graph: each op's inputs, outputs,
   units, L1 scratch top (`OpRecord::l1_top`, from the space audit) and layout. The plan pass reads them and plans.
2. **Planned capture:** the same function is captured again with the plan applied. As each op is captured, its op
   index selects the plan's decisions for its inputs and outputs.
   - The planned capture must produce the same sequence of op signatures as the plan pass, or it is refused and the
     plain capture is kept, with the reason named.
   - The plan is cached by the graph's signature. Traces that share a body (a decode step's attended lengths: 64,
     128, 256, 512) plan their shared ops once.
3. **API:** the capture entry points (`burn_tt::Trace::capture`, `TracedInference::capture{,_with_state}`,
   `SharedStateTraces::capture`, `TracedTrainingStep`) take `FnOnce` today. A planned capture needs to call the
   function twice, so it takes `FnMut`. Callers that can only give `FnOnce` keep the plain capture.
   `tt_models::generate::TracedGenerator` already captures by calling model code, which can run twice.

### What lives in L1

A tensor is resident only if all of these hold:
- the graph both produces and consumes it, so it is not one of the graph's inputs, outputs or state (logits, the
  token, the KV cache, weights);
- every consumer has an L1 input path in the layout the producer can write (the layout contract, below);
- it fits the persistent region beside everything live with it.

When it does not fit, the plan leaves out the tensors with the largest size times lifetime first, in a deterministic
order, until the rest fit. Those tensors keep their GDDR round trip. This is how training's saved activations end up in
GDDR without a special case.

### The layout contract

Each op kind declares which L1 layouts it can read and write. Three to begin with:
- **Replicated row:** a `[1, K]` row and its packed twin, whole on every unit that reads it. A product's packed `A`
  needs this on each of its units, and so do RMSNorm's statistic and the residual add's operands.
- **Tile-sharded:** a run of tiles a unit, as element-wise ops already deal them (`runs`).
- **Per head:** attention's query, key and value rows, and its context, a head to a unit.

A producer writes in the layout its consumer reads:
- the writer core sends its output tiles into the consumers' L1, with one multicast for a replicated row;
- where two consumers want different layouts, the plan either writes both or leaves that tensor in GDDR;
- a relayout op exists only if measured to pay.

### Addresses: one layer, the same in every repeat (sub-traces)

The space report found a decode step's ops repeat with a period of one layer (16 ops; 365 of 388 ops equal the op one
period later). The plan uses that:
- it plans **one period**: one layer's tensors, with their lifetimes inside it;
- every repeat gets the **same addresses**;
- tensors that cross a period boundary are pinned to the same slot at both ends.

The residual stream is the case that matters:
- `x` lives in slot S;
- the first residual add writes `x + attention` into slot T, because S is still being read;
- the second add writes `x'` back into S, because `x` died at the first add.

So layer `i + 1` finds its `x` where layer `i` found its own. With every repeat's L1 addresses equal, the layer
sub-trace body (`layer-sub-traces.md`) needs no L1 arguments: its arguments are only the GDDR references (the layer's
weights, its cache, the tensors left in GDDR). Moving activations into L1 also shrinks those argument tables. Today
86% of the words that differ between layers are tensor references, and many of them are activations.

Anything that does not repeat (the embedding at the head, the norm and the output product at the tail) is planned
around the periodic part, in the same region.

### The arena on each unit

```
DATA.base                                                        DATA.end
| per-op scratch (Requirements::plan) -> | ... | <- prefetched B ring | persistent region |
```

- **Persistent region:** at the top, a fixed size per graph, from the plan's peak. At 1.7B a replicated row and its
  twin are 16 KB, a `gate|up` shard is under 1 KB a unit, and a layer's live set is tens of KB a unit. It scales with
  the hidden size: 8B's rows are 16 KB.
- **Per-op scratch:** planned in `[DATA.base, region.base)`. `Requirements::plan` takes that arena; the layouts placed
  by hand (`MATMUL_STAGE`, `rms_divide`, `rope_blocks`) assert that they stay below it.
- **The products' `B` ring:** placed against `region.base` instead of `DATA.end` (`matmul_requirements_top_b` is
  given the reduced arena). Busy-unit prefetch keeps its rule: the ops between two products must plan below the ring.
- **Budget:** products need ~553 KB at `K` = 2048 since the packed-`A` reclaim, and small ops 12-50 KB. The 254 KB that
  reclaim freed is what this region and the sub-traces' stored bodies (~7 KB a unit) use.

### Hazards

- **Read after write.** A consumer runs after its producer's barrier, as every op does today. A barrier already means
  the units that ran the producer have finished, including their writes into other tiles' L1. The writer completes
  before its unit arrives, and that rule is gated for each new write record.
- **Write after read.** A slot is reused only by an op strictly after its last reader's barrier. The plan's lifetimes
  end at the last reader's op index, and the allocator reuses a slot only after that. This is checked when the plan is
  made and again over the planned capture's records: every reader of a slot comes before the next writer.
- **Within one op:** an op never reads and writes the same slot. The residual stream's S/T slots exist for this.
- **Between replays:** nothing in the persistent region is live from one replay to the next. A trace's outputs and
  state are in GDDR. So eager ops queued between replays may plan over the whole arena, and no reservation outlives a
  replay. Every replay starts from the plan's first writer.

### What the mover and the ops need

- **Reading in place:** an op whose input is already in its own unit's L1 reads it there, with no mover read. Its
  input ring is bound to the slot, as tt-metal binds a CB to a sharded buffer (`GATHER_A_RESIDENT` is the first case
  of this).
- **Writing to other tiles' L1:** the writer core (NC) sends output tiles into another tile's L1. This is a new write
  record, plus a multicast form for replicated rows, each rectangle checked as the barrier release's are
  (`dm::BARRIER_RECTS`, `Rect::new`). It is the hang class that can reboot the host, so: ttsim first, small spikes, and
  a refusal for any rectangle that is not the op's own units.
- **Ownership rules:** `check_pair` learns L1 endpoints: no GDDR channel is involved, and the NoC that writes is the
  writer's.
- **Twins:** `packed_twin_writes` writes into the region when the twin is resident.

### Burn and the residency gates

- **A resident tensor still looks like a tensor to burn-tt.** Its placement says L1, and its GDDR buffer is allocated
  only if something forces a spill.
- **Inside a planned capture:** a download, or an op without an L1 path, is a plan error found by the plan pass. It
  never becomes a silent host or GDDR fallback during the planned run.
- **After the capture:** outputs are in GDDR as now.
- **Gates:** `assert_native_model` and the residency checks gain one more claim: the planned run writes no GDDR for a
  resident tensor (the profile mode's per-op bytes), so a cached copy cannot hide a fallback.

## Phases

### Phase 1. The plan, reported only (done 2026-10-20)

- [x] **`l1_top` exact, and a need beside it.** A layout planned in the first half and moved to the second
  (`Staging::SlotsHalf`, the reductions' halves) is noted where it ends up (`l1::planned_shifted`); a reduction's memo hit
  notes its layout again. `OpRecord::l1_need` is what an op cannot do without: a product's `B` ring placed against the
  arena's end leaves a gap (`Requirements::slack`), and the need leaves it out. Decode now has no unplanned op.
- [x] **The graph off the streams** (`tt_kernels::residency::Graph::from_capture`). A capture logs the tensors it allocates
  (`DramAlloc::start_log`); every op's records and moves, sections included, are expanded as the movers expand them and
  each GDDR access is charged to the tensor and tile it touches. The rest is external (weights, the cache, inputs).
- [x] **The plan** (`residency::Plan::make`): the period (the space report's, now shared: `residency::period`); tensors
  at the same place of two repeats (the same op of the layer, the same of its tensors) in one class with one address, so
  the residual stream is pinned across a layer's end and the head's `x` sits where every layer's does; first-fit
  addresses over inclusive lifetimes; the costliest class (`bytes x lifetime`) left out first when the budget is
  short. **The check** (`Plan::check`) is independent: no overlap while both live, the period's ops see the same
  addresses, the region fits.
- [x] **The report** (`residency::probe_residency_plan`, `RESIDENCY` lines; `TT_PROBE_RESIDENCY=1` on
  `silicon_bench_models::traced_generation_stream_ceiling`).
- [x] **Gates.** Unit tests in `residency.rs`: the S/T alternation, every layer's `x` at one address, determinism, the
  leave-out order, outputs and ops without a path, and the negative controls (a slot reused before its last reader, an
  op that would read and write one slot, an address that differs a period later) refused. `step217_residency_plan`
  (ttsim; card 0 `1792518267`): over a 4-layer Llama's traces, fused and not, every op's bytes read off its streams
  equal what the trace counts (`OpBytes`), nothing unattributed; every plan checks; a layer is the period and a resident
  tensor crosses it; only tensors freed inside the capture are resident; a second build plans identically; a resident
  tensor moved onto another live one is refused. Card 0, same run: `step216`, and `step188` (against Flex, and traced
  against eager).

**What the 1.7B's plan says** (card 0, every tile, `1792518158`; the step unchanged, 20.98 / 22.24 ms):

| Trace | Internal tensors | Resident | Addresses | Region (within the repeats) | Budget | GDDR avoided |
|---|--:|--:|--:|--:|--:|--:|
| `Step(64)` | 548 | 401 | 47 | 266 KB (112 KB) | 339 KB | 294 MB read / 123 MB written of 300 / 168 |
| `Step(128)` | 548 | 401 | 47 | 266 KB (121 KB) | 339 KB | 300 / 123 |
| `Step(256)` | 548 | 401 | 47 | 266 KB (137 KB) | 339 KB | 313 / 123 |
| `Step(512)` | 544 | 398 | 44 | 266 KB (171 KB) | 339 KB | 339 / 123 |
| `Chunk(64)` | 691 | 97 | 53 | 83 KB (4 KB) | 110 KB | 81 / 32 |

- **The period is a layer:** 16 ops x 23 repeats from op 30 (365 of 388 ops repeat). The residual stream's two
  addresses hold 24 and 23 tensors each.
- **A step's whole graph fits, with every op given an L1 path:** all but the outputs and the never-read tensors. The
  budget is set by `down`'s streamed product (557 KB of the 896).
- **The whole-step region is the logits'** (266 KB): the head's `[1, 49152]` row read whole by the argmax's one unit.
  The layers need 112-171 KB, growing with the attended length. Left out, the logits would leave ~170-225 KB a unit for
  prefetched `B` instead of 73.
- **147 tensors a step (38.7 MB) are written and never read** (`NeverRead`): an op that writes a row both as tiles and
  as its packed twin, where the next op reads only one (`gamma`'s multiply, RMSNorm's divide, `silu_mul`, attention's
  rows). Phase 2's resident twin replaces both writes.
- **The consumers' layouts differ:** `x` is read by RMSNorm's divide on 32 units and by the add on 64, both sharded, and
  its packed twin replicated on the product's 64-104 units. RoPE reads `qkv` mixed. So a producer writes each tile to
  every unit that reads it ("write both"), as the union per unit that the plan sizes.
- **Prefill:** `down`'s K-blocked product accumulates in GDDR (`ReadAndWritten`), and the products leave 110 KB, so a
  chunk keeps most of its tensors in GDDR.

### Phase 2. The residual stream resident

Add, RMSNorm, `gamma`'s multiply, and the next product's packed `A`. This is what busy-unit prefetch hurt most.

- [x] **2a. A persistent region, still empty** (2026-10-20). `l1::arena()`, a thread-local `Region` that is `DATA`
  except while a capture keeps a region (`Session::set_l1_region`, `burn_tt::set_l1_region`, `TT_L1_REGION=<bytes>`):
  then it ends at the region's base. Every `Requirements::plan` in `DATA` plans in it; what was sized from `DATA`
  is sized from it (the SFPU, reduction and chunk groups, `rms_divide`, `rope_blocks`, `matmul::half()`, the
  host-staged product's outputs, and the copy and fill-pad staging, `arena_group`); every memo of a layout, or of
  programs that name its addresses, is keyed by its end. `end_trace` refuses a capture with an op that planned past
  the base, or did not plan while captured (`TraceError::IntoRegion`). Measuring an SFPU group's programs plans in
  `DATA` and is not noted as the op's (`l1::unnoted`): its lengths do not depend on where it sits, and a small arena
  made it panic.
  - **Gates:** `step217` (ttsim; card 0 `1792523328`, run with `TT_L1_REGION=262144`): with 256 KB kept, every
    captured op plans below it (the products' `B` rings move down) and the traced generation is bit for bit the one
    without; a region leaving one slot fails the op by name and keeps no trace, and the op after the capture has the
    whole arena. `step188` with a 256 KB region on card 0 (same run): against Flex, and traced against eager, bit for
    bit. `step216` without a region on card 0 (`1792523314`); with one it moves no prefetch, as it should: its element-wise op's
    group fills the arena up to the lowered `B` ring.
  - **Measured (1.7B, card 0, 32 / 256-token prompts):** no region 20.98 / 22.24 ms (`1792518158`); 64 KB 21.03 /
    22.23 (`1792523682`); 96 KB 20.97 / 22.20 (`1792523958`); **128 KB 22.59 / 23.80** (`1792523804`); 256 KB 22.56 /
    23.81 (`1792523535`). Past ~96-128 KB the products' blocks narrow (`plan_in` takes the widest `K`-whole block the
    arena holds): their gather entries 196,648 -> 307,576 a step, +1.6 ms. **So Phase 1's budget (339 KB, from the
    ops' need) overstates what is free: the room that costs nothing is under ~100 KB a unit.** The residual stream's
    group is well under it. A larger region pays only if what it saves exceeds the products' 1.6 ms; and the
    two-pass rule refuses a region that reshapes an op by itself (an op's signature holds its entries a unit).
- [x] **2b-i. `WRITE_L1`, unicast** (2026-10-20). `op::WRITE_L1` `[WRITE_L1, x, y, to, from, len, 0, 0]`: NC writes
  `len` bytes of its L1 at `from` to `to` in tile `(x, y)`'s L1, one response-marked `noc::Command::Write` through the
  NIU its writes use (`issue_l1_write`), so the drain before `DONE` -- and so the barrier -- waits for it. The card's
  decode: a Tensix coordinate, both ranges in `DATA`, congruent mod 16, at most one request; B refuses it
  (`DIRECTION`). The host: `DataMover::set_l1_writes` names the tiles and the window (`L1Writes`); every other
  `WRITE_L1` is refused before it is queued, and none is admitted until a session names them; `check_pair` admits it in
  NC's section. `l1_footprint` does not model it yet (a slot it reads waits for NC's release, as before).
  - **Gates:** `tt-isa` unit `an_l1_write_decodes_only_whole_and_only_on_nc`; `tt-kernels` unit
    `a_writer_section_may_write_another_tiles_l1`; `step218_l1_write_records` (ttsim, then card 0 in the safe image
    `1792524683`, `fast-reads` `1792524825`, `fast-reads` + `encoded-guard` `1792524963`): writes land in two peers'
    windows on NoC #1 and #0 with the guard and a bystander untouched (watched failing with the firmware writing 64
    bytes off); the host refuses a tile not the session's, a harvested column's, an address outside the window, any
    write before the session names its tiles; the tile refuses seven malformed entries sent raw and writes nothing.
    `step172`, `step173` (and `step50`) in all three images, ttsim and card 0. Firmware clippy plain, `encoded-guard`,
    `profile`, `fast-reads`.
  - **Image sizes:** B 30,404 -> 30,612 bytes; NC 15,540 -> 16,704 (7,872 of 24,576 left).
- [x] **2b-ii. `WRITE_L1_MCAST`** (2026-10-20; brought ahead of 2c: a replicated row to ~104 units is ~200 unicasts of
  64 B a producing unit, ~55 us an add at ~0.27 us an issue, against the add's 14 us today, where multicast is a few
  requests). `[WRITE_L1_MCAST, x0 | y0 << 8 | x1 << 16 | y1 << 24, 0, to, from, len, 0, 0]` (`dm::mcast_corners`): NC
  drains its earlier writes, reads NoC #1's write-ack counter, issues one response-marked, path-reserved multicast on
  static VC 4 with the corners swapped for NoC #1 (`multicast::corners_hi`, as tt-metal does) on an ID nothing drains
  (`dm::TXN_NC_MCAST` = 7), and waits for the counter to advance by the rectangle's tiles before the next entry. NoC #1
  only: NoC #0's counter is B's barrier release's (`DIRECTION`). The card refuses a rectangle holding its own tile,
  a reversed one, one across a non-memory or DRAM column or outside the Tensix rows; the host admits only a
  rectangle all of whose tiles are the session's.
  - **Gates:** `tt-isa` unit `a_multicast_decodes_only_over_tensix_columns`; `step218` (ttsim, then card 0
    `1792525425`): a 2x2 rectangle holds every byte at two skews, in one list with a unicast before and after it,
    the tiles beside it as their own unicasts left them (watched failing with the firmware writing 64 bytes off); the
    host refuses a rectangle with a tile not the session's or over a harvested column; the tile refuses four
    multicasts sent raw (holding itself, reversed, across column 8, through NoC #0) and writes nothing. Card 0:
    `step50`, `step172`, `step173`, `step208`, `step218` in the safe image 19 of 19 (`1792525443`), `fast-reads` 16 of
    16 (`1792525591`), `fast-reads` + `encoded-guard` 17 of 17 (`1792525736`); firmware clippy plain,
    `encoded-guard`, `profile`, `fast-reads`.
  - **Image sizes:** B 30,612 -> 30,996 bytes; NC 16,704 -> 17,872 (6,704 of 24,576 left).
- [x] **2c. L1 paths, by rewriting the ops' lists, not their builders** (2026-10-20). The Phase 1 graph already maps
  every GDDR move to a tensor and tile, so a planned capture rewrites each op's lists as they are queued
  (`residency::Applied::rewrite`, `Session::apply_plan`), whatever op made them: a whole move that reads a resident
  tensor (a plain `READ`, or one a `READ_RUN` without flags expands to) becomes a read of the unit's own region, and a
  whole move that writes one (a plain `WRITE`, or a `WRITE_RUN`'s) becomes `WRITE_L1` to the writing unit and
  `WRITE_L1_MCAST` to each rectangle of the other units that read the tile at the same address
  (`residency::rectangles`). Each resident tensor holds, on every unit that reads it, the tiles that unit reads, in
  tile order, a slot each. A record that touches a resident tensor must be wholly plain moves of resident tensors, or
  the plan stops (below). A use whose moves all are is `Use::plain`; the plan makes only those resident. The region's
  reads first used `COPY_WORDS`, B copying a slot a word at a time; now **`op::READ_L1`** (`[READ_L1, x, y, from, to,
  len, 0, 0]`, B's NoC read of a tile's L1, its own a loopback, under its read ID; host admission as `WRITE_L1`'s;
  B 30,996 -> 31,328 bytes, NC 17,872 -> 17,908).
- [x] **2d. The two-pass capture** (2026-10-20). `Session::set_residency(bytes)` (`TT_RESIDENCY=1`,
  `TT_RESIDENCY_REGION`, default `RESIDENCY_REGION` = 96 KiB): every capture keeps that region, so the plan pass and
  the planned capture lay their ops out alike, and a plain capture ends by planning itself (`plan_of`:
  `Plan::make_within` the region, `Plan::check`). `Session::begin_trace_planned` applies that plan: `L1Writes` set on
  every mover to the session's tiles and the region, each op's lists rewritten. `OpRecord::shape` hashes each op's
  lists as laid out before the rewrite; `end_trace` refuses a planned capture (`TraceError::Planned`) unless its ops'
  shapes are the plan pass's, its allocations the same tensors in order, nothing stopped the rewrite, and its own graph
  shows no GDDR move of a resident tensor (`Applied::verify`). burn-tt: `SharedStateTraces::capture` takes `FnMut` and
  captures twice when the session plans (`TracedInference::capture_donating_planned`), keeping the planned trace, or
  on a refusal the plain one, replayed once so the state is what the plain run left; `burn_tt::set_residency`;
  `TracedGenerator` clones its inputs per run. The period's pinning is kept only where it costs no tensors: the planner
  also plans without it and keeps whichever spares more GDDR bytes (`Plan::pinned`). Left: allocating a resident
  tensor's GDDR only on spill (it is allocated, never written), and a plan cache by signature (each trace plans once).
- [x] **2e. Gates and measurement** (2026-10-20).
  - **Gates:** `step219_residency_decode` (ttsim; card 0 `1792528883`): on 8 units, an add, RMSNorm, `gamma`'s multiply
    and the next add, captured plain then planned, replay bit for bit as the ops run fresh, the planned trace moving
    1.88 MB of GDDR against 3.01; a planned capture of another graph is refused by name and the session runs on;
    **negative control:** the plan with a resident tensor moved onto one still to be read is refused by `Plan::check`
    and, applied anyway, gives other bits. Through Burn, a 4-layer Llama's traced generation with residency on equals
    it without bit for bit, `Step(16)` 3.24 -> 2.14 MB and `Chunk(4)` 4.09 -> 2.52 MB of GDDR. Card 0 with
    `TT_RESIDENCY=1` (`1792526703`): `step188` (the 1.7B, every tile: against Flex, and traced against eager, no
    planned capture refused), `step193`, `step209`, `step210`, 11 of 11. `--smoke` on card 0, residency off: 1,167 of
    1,167 (`1792528895`).
  - **Measured: replicated rows by multicast cost more than they save** (1.7B, card 0, every tile, busy prefetch off).
    With every plain use resident (204 tensors in a 33 KB region, 91.5 MB of reads a step spared) a step is **43.89 /
    45.01 ms** against 20.98 / 22.24 (`1792526939`). The profile (`Step(64)`, off `1792527261`, on `1792527601`): the
    products unchanged (15.23 -> 15.57 ms); `gamma`'s multiply 0.69 -> **12.37 ms**, the residual add 0.66 -> 6.77,
    attention 0.65 -> 4.33, `silu_mul` 0.49 -> 1.27: the ops whose output is a row read whole by many units (a product's
    packed `A` on 64-104 units, RMSNorm's twin on 32). ~250 us a multiply: its 64 units each multicast face rows into
    the same 104-tile rectangle at once, and path-reserved multicasts into one rectangle serialize.
  - **So rows read whole on many units stay in GDDR for now** (`TT_RESIDENCY_REPLICATED=1` keeps them, for
    comparison). The sharded tensors alone (79 resident in a 25 KB region): **20.92 / 22.14 ms** (`1792528051`)
    against 20.98 / 22.24, even; with busy prefetch at ID 6's cap 8, **21.53 / 23.17** against 21.66 / 23.26 off
    (`1792528191`, `1792528455`); at cap 2, 20.67 / 22.18 (`1792528324`). The residual add still reads the product's
    output from GDDR (a `SCATTER`, which the rewrite does not redirect) behind the stream: Phase 3.
- [ ] **2f. Replicated rows, delivered without contended multicasts.** Candidates, to be measured on the multiply:
  (a) a few holders: each producer unicasts its 128 B of a row to `h` holders and each reader pulls the whole row
  from its holder with one `READ_L1` (8 KB a unit, `h` spreading the load); (b) multicasts serialized by design: one
  producer gathers the row (unicasts in), then multicasts it whole (two 4 KB requests a rectangle) after the op's
  barrier, as a step of the reader; (c) multicast acknowledgements counted once a run, not an entry (firmware), which
  helps only if the NoC is not the bound.

### Later phases

3. **Products into their consumers:** `qkv` → RoPE, `gate|up` → `silu_mul`, `o` → the residual add (tile-sharded).
4. **Attention's rows and context** (per head). The KV cache stays in GDDR.
5. **Sub-traces on the plan** (`layer-sub-traces.md`): one stored layer whose only arguments are GDDR references.
6. **Prefill chunks, traced inference (ViT, ResNet blocks), then traced training** once trace segmentation exists.

## Expected

These are estimates to be measured at each phase.
- **Phase 2 alone:** the small ops between products lose most of their data time: the add and RMSNorm become L1
  reads. Below 4.6 ms of small ops a step.
- **With busy-unit prefetch:** the products' 11.6 ms with the weights read ahead, plus small ops no longer queued behind
  the stream (~4-5 ms), is a 1.7B step of roughly **16-17 ms**, against 20.9. The F32 floor is 14.3.
- **Storage, with phase 5:** a step's trace about 16x smaller.

## Risks

- **A new hazard class.** Write-after-read across tiles was hidden by the GDDR round trip. Plan-time and record-time
  checks and silicon negative controls are the defence.
- **Multicast into other tiles' L1.** A wrong rectangle can hang the NoC and reboot the host: checked rectangles only,
  ttsim spikes before silicon, never a hand-made record on silicon whose refusal ttsim has not shown.
- **Two captures per trace.** Building takes longer: the 1.7B driver's 18.8 s build becomes ~2x for its first
  capture of each signature. The plan cache keeps later builds at one capture.
- **Breadth.** Every op on the decode path needs an L1 input path and an L1 output path. The phases bring them in one
  group at a time, each measured.
- **What it does not change:** eager execution, the F32 floor, and the products' own reads.
