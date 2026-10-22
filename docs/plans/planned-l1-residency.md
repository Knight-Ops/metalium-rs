# Planned L1 residency: activations that survive between ops

Status: **scheduled; Phases 1, 2 and 3 done** (2026-10-21; a proposal until 2026-10-20): rows read whole on many units
held by two of their readers (2f), never-read writes dropped (2g), the region sized by its plan under the products' cliff,
plans cached, and residency **on by default** (2h): a 1.7B step 21.00 / 22.25 -> 20.07 / 21.28 ms; slower builds
accepted. Phase 3, the products' outputs written into their readers' L1: **19.50 / 20.71 ms**, the residual stream,
RMSNorm and `silu_mul` reading no GDDR; and, found by probing why busy-unit prefetch still did not pay, NC no longer
waits for B's prefetch reads at a list's end, so busy prefetch is on by default at a prefetch cap of 2: **18.39 / 19.90
ms**. Phase 4, re-scoped (q and the context were resident already): the cache write's index read narrowed and a second
leave-out order; Phase 5a, a replay's next reader section read ahead: the small ops 5.00 -> 4.47 ms, **the step
unchanged** (18.32-18.44 / 19.78-19.83 ms), the weight stream under busy prefetch taking what they gave up. Next: the
rate the weights stream at, not the small ops (Phase 5b, the stored layer, deferred on that finding). Asked for after Phase 8 of
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
- **Since the mailboxes were packed** (2026-10-22, decode-to-bandwidth.md "Room for a second job's `B`", M2): `DATA` is
  unchanged; the 88 KiB the mailboxes left (`l1::CAPTURE_ONLY`) and the program slots after it are for captures only
  (P1-P3 there), whose block widths stay pinned to `DATA`'s.

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
- [x] **2f. Replicated rows, delivered without contended multicasts** (2026-10-20). Three candidates, probed on
  the multiply's geometry first, then the winner built into the plan and the rewrite.
  - **A peer's L1 read** (`op::READ_L1` of another tile, which (a) needs; nothing ran one before): `tt-isa` unit
    `an_l1_read_decodes_only_whole_and_only_on_b`; `step218` `a_read_pulls_a_peers_window`, the host's and the tile's
    refusals (ttsim; card 0 safe `1792531966`, `fast-reads` `1792532114`, `fast-reads` + `encoded-guard`
    `1792532259`).
  - **(c) in firmware:** `dm::MCAST_RUN_ACKS` (word 6 of `WRITE_L1_MCAST`): a run of marked multicasts is drained
    once before its first and its acknowledgements counted together before the next other entry and at a list's end
    (`settle_multicasts`, NC only). Gate `step218` `a_run_of_multicasts_lands_before_the_next_entry` (one run between
    unicasts, two runs split by a lone multicast, a run ending the list; ttsim, card 0); `step50`, `step172`,
    `step173`, `step218`, `step220` on card 0 in the safe image 23 of 23 (`1792533202`), `fast-reads` 28 of 28
    (`1792533364`), `fast-reads` + `encoded-guard` 29 of 29 (`1792533532`). B 31,328 -> 31,548 bytes, NC 17,908 ->
    18,108 (6,468 left).
  - **The probe** (`Session::probe_row_delivery`, `residency::RowScheme`; gate `step220_row_delivery`: every reader
    ends with the twin in every scheme, eager and replayed, and a reader sent to no holder reads poison; ttsim and
    card 0 `1792532765`). Timed by `silicon_probe_row_delivery` (card 0, 120 tiles, `--profile`, `1792533708`): per-op
    span medians of 8 deliveries a trace, idle / under a GDDR stream on every unit the delivery leaves idle (56; 16
    for 104 readers), in us:

    | Row, producers -> readers | GDDR | multicast (Phase 2) | (c) run acks | (a) 1 / 2 / **4** / 8 holders | (b) relay |
    |---|--:|--:|--:|--:|--:|
    | `[1, 2048]`, 64 -> 32 | 9.0 / 47 | 140 / 140 | 130 / 140 | 8.2 / 7.5 / **8.1** / 9.9; 21 / 17 / **14** / 21 | 16 / 34 |
    | `[1, 2048]`, 64 -> 64 | 14 / 70 | 210 / 210 | 200 / 200 | 12 / 9.2 / **8.7** / 10; 24 / 23 / **17** / 21 | 19 / 37 |
    | `[1, 2048]`, 64 -> 104 | 17 / 23 | 240 / 240 | 240 / 240 | 16 / 11 / **9.8** / 10; 20 / 14 / **11** / 11 | 23 / 28 |
    | `[1, 8192]`, 64 -> 64 | 16 / 80 | 780 / 780 | 760 / 760 | 33 / 23 / **20** / 28; 55 / 66 / **37** / 58 | 51 / 78 |

    **(a) wins;** (c) changes nothing (the NoC is the bound, not NC's waits), (b) pays an op and a barrier. Through
    GDDR the readers' time quadruples under the stream; from holders it at most doubles, and the writes (unicasts on
    NoC #1) slow too.
  - **(a) in the plan** (`residency::Rows`, `Plan::make_rows`, `Plan::held`; `Session::set_residency_rows`,
    `TT_RESIDENCY_ROWS=gddr|mcast|held:<h>`, default `held:2`; `TT_RESIDENCY_REPLICATED` retired). A resident tensor
    some op reads whole on more than one unit (`Tensor::replicated`) is held by `h` of its readers (`residency::holders`:
    contiguous runs of the readers, each held by its middle one), if every reader reads it alike
    (`Tensor::read_alike`; otherwise it stays in GDDR, so a holder's bytes are never more than a unit's were). A write
    is a `WRITE_L1` to each holder, a read a `READ_L1` from the reader's holder, consecutive slots one request.
    `Plan::check` refuses a reader with no holder or one sent to a unit that is not one (`Hazard::NotHeld`), and the
    rewrite one that holds nothing. The prefetch lists cannot move a held read ahead of its writer: they take the
    `GATHER` record only (its `A` resident), and the packed `A`'s `READ_RUN` stays in the op's own list.
  - **Gates:** `residency.rs` units `holders_split_the_readers_evenly`,
    `a_held_row_is_written_to_its_holders_and_read_from_them` (the first direct tests of the rewrite),
    `rows_choose_the_delivery_and_a_misrouted_holder_is_refused`, `a_row_not_read_alike_is_not_held`; `step219`
    `a_row_read_whole_is_held_by_two_units_bit_for_bit` (multiply -> one-row product, the twin held by 2 of 8: bit for
    bit, 3.83 -> 3.23 MB of GDDR; the negative control refused by the check and the planned capture by name), the
    Llama test with the default. Card 0, `TT_RESIDENCY=1`: `step219`, `step220`, `step188`, `step193`, `step209`,
    `step210`, 20 of 20 (`1792535508`, held by 4), and with `TT_BUSY_PREFETCH=1` (`1792535737`); with the default
    of 2, `step219`, `step220`, `step188` 11 of 11 (`1792538756`); no planned capture refused. `--smoke` on card 0,
    residency off, 1,173 of 1,173 (`1792538973`).
  - **Measured (1.7B, card 0, every tile, 32 / 256-token prompts, ms):** off 21.03 / 22.25 (`1792535978`); sharded
    only 20.88 / 22.18 (`1792536101`); held by 2 **20.23 / 21.44** (`1792536365`), 4 20.40 / 21.58 (`1792536233`), 8
    20.53 / 21.77 (`1792536496`). With 2g: held by 4 20.18 / 21.43 (`1792536628`), **held by 2 20.06 / 21.28**
    (`1792537293`). Busy prefetch still does not pay: held by 2 with drops, cap 2, 20.13 / 21.68 (`1792537424`); held
    by 4 with drops, cap 2 19.99 / 21.56 and cap 8 20.95 / 22.60 (`1792536759`, `1792536891`); off, cap 2 20.70 /
    22.22 and cap 8 21.64 / 23.27 (`1792537023`, `1792537145`). The whole model prefers 2 holders where the probe
    preferred 4: a step's other rows are read by fewer units, and two holders' writes cost each producer less.
- [x] **2g. Never-read writes dropped** (2026-10-20). Phase 1's `NeverRead` tensors (a row written as tiles and as its
  packed twin, where the next op reads one): `Plan::dropped` holds the internal ones whose every write is a plain
  move; the rewrite removes their writes (a list left empty keeps a `WAIT` in its place); `Applied::verify` refuses a
  GDDR move of one, `Plan::check` a dropped tensor that is read, resident, or the graph's output
  (`Hazard::ReadDropped`), and the rewrite a read of one. On by default under residency
  (`Session::set_residency_drop`, `TT_RESIDENCY_DROP=0` turns it off). Gates: unit
  `a_never_read_tensor_s_writes_are_dropped`; `step219` `never_read_writes_are_dropped_bit_for_bit` (bit for bit,
  548,864 -> 532,480 B written; the negative control refused by the check and the planned capture by name). Measured
  above: -0.17 ms a step held by 2, -0.22 held by 4.
- [x] **2h. The region sized by the plan, and residency on by default** (done 2026-10-21; asked for ahead of larger
  models). Today every planned capture keeps a fixed region (`RESIDENCY_REGION`, 96 KiB), chosen from the 1.7B: free
  up to 96 KiB, +1.6 ms a step past ~128 KiB, where the products take narrower blocks (2a). The 1.7B's plans use tens
  of KB of it. Past the cliff, a region costs the products more than residency saves, and the cliff depends on the
  products' `K`: a `K`-whole column of weights is ~266 KB at `K` = 2048 and ~532 KB at 4096. So on a larger model the
  fixed region may cost more than it saves. Phase 1's budget (`DATA.end` less the neediest op's `l1_need`: 339 KB at
  the 1.7B) overstated the room ~3x. Where the region does not fit, the plan leaves tensors in GDDR (the costliest
  first) and never fails, so the risk is a loss of speed, not of correctness. In order:
  - [x] **2h-i. The cliff, measured** (2026-10-21). `silicon_probe_region_cliff` (card 0, every tile, release,
    `1792546337`, `1792546573`): eight one-row products (the 1.7B's and Llama-3.2-1B's `qkv`, `gate|up` and head at `K`
    = 2048, Qwen2.5-1.5B's `gate|up` at 1536, an 8B's `qkv` at 4096, `down` at 8192, 8960 and 14,336), each under
    regions from 0 to past its predicted limit, the limit and 64 bytes either side of it. It reports time, read and
    stream entries a unit, the block (`matmul::probe_plan_at`) and the recorded need; every point is bit for bit
    against the product with no region.
    - **The cliff is exact, and the planner predicts it.** `plan_in` takes the widest block whose probe layout fits,
      and the probe charges `A` unpacked (`kt` slots, where the kernel runs a packed `A` of `kt/32 + 1`). So a `K`-whole
      product keeps its block while the arena ends at or above `DATA.base + 4160 * ((kt + 1)(nc + 1) - 1)`: at `K` =
      2048, a region of at most **110,464 bytes**. At 110,464 the 1.7B's products are as with no region; 64 bytes past
      it they narrow (`nc` 2 -> 1) and slow 8-10% (`qkv` 148 -> 161 us, `gate|up` 342 -> 376, head 991 -> 1069).
    - **`K` blocked, `A` packed:** the jobs are a column each, so `plan_in`'s width binds nothing. The streamed chunk
      layout binds: `down` 339,264 (8192), 285,184 (8960), 222,784 (14,336); 8B `qkv` 376,704. Past each, under 2%.
    - **Why `l1_need` overstated the room:** it never saw the probe. `plan_in`'s probes were noted only on a cache
      miss, and a capture after an eager run hit the cache, so the products recorded the packed layout's need. Phase
      1's budget was then `down`'s (339 KB) where the K = 2048 products' is 110 KB.
    - **Findings for later** ([decode-to-bandwidth.md](decode-to-bandwidth.md)): Qwen2.5-1.5B's `gate|up` is
      *faster* narrowed (319 -> 293 us), and `K` = 4096 is never `K`-whole (257 slots of 220), so an 8B's products
      stream with no weight prefetch. Charging the packed `A` in `plan_in`, with the width chosen by measurement rather
      than greed, is a separate step.
    - Recorded in the [operating notes](../learnings/silicon-operating-notes.md#a-region-at-the-top-of-l1-costs-a-product-only-past-its-plans-floor-2026-10-21-card-0).
  - [x] **2h-ii. The region from the plan** (2026-10-21).
    - **Every op notes its floor.** That is the lowest arena end at which its layouts are the ones it has
      (`l1::Planned::need`, `OpRecord::l1_need`); `need` may now pass `top`.
      - `matmul::plan_floor` gives the chosen block's probe. `plan_in` notes it on every call, without its probes,
        fixing the warm/cold defect.
      - `pipelining_pays` notes both stagings' floors where both decide.
      - A packed `A` with `K` blocked notes `packed_k_block`'s and the stream's layouts, not the plan's, whose width
        its jobs ignore.
      - The other arena-sized layouts already noted their chosen layout. The gates below would find one that does not,
        as a refusal, never as wrong bits.
    - **The cap:** `residency::cap` is `DATA.end` less the graph's highest need (110,464 bytes for a 2048-wide
      model).
    - **The plan pass keeps no region** (`Session::begin_trace_plan_pass`), so its trace is laid out as a plain one,
      and that is what a refusal keeps.
    - **The plan** is made within `min(cap, RESIDENCY_REGION)`. `RESIDENCY_REGION` is now the ceiling. The planned
      capture keeps that whole budget, not only what the plan uses: every op is laid out alike either way, and a
      model's traces then share one arena end. Two consequences:
      - **The programs.** The programs name addresses planned in the arena, so one set serves every trace. Kept to what
        each plan used (the first build of this step), each trace held its own set, and the 135M on 12 tiles filled the
        program cache (`step193`, "a list's programs do not fit an empty program cache").
      - **Later plan passes.** For the same reason, once a planned capture has held, later plan passes keep its region
        (`Session::planned_base`): a plan pass in the whole arena held a second set beside the kept traces'. So the
        first trace's plan sizes a session's region, and a later graph whose ops would narrow under it plans narrower
        rather than fail.
    - **Only plan passes plan.** `begin_trace_plan_pass` marks them, so single-pass captures (`Trace::capture`,
      `TracedInference::capture`, `TracedTrainingStep`) neither keep a region nor plan.
    - **The plain trace is released before the planned capture** (burn-tt), and a refused planned capture is captured
      plainly again.
    - `TT_RESIDENCY_REGION=<bytes>` (`Session::set_residency_fixed`) keeps a fixed region in both passes, for
      measurement.
    - **Each planned op checked before it is queued.** `Applied::check_op` checks two things before the rewrite:
      - the allocations so far are the plan's;
      - every move of every list lands on the plan pass's tensors and tiles.

      A planned capture whose plan stops queues nothing more: the next ops would read from GDDR what the plan put in
      L1. A cached plan applied to another graph had made an op divide by zero on the card (`dm::error::DOMAIN`), and
      that stops the queue.
    - **Faster rewrite.** `Applied::rewrite` runs only on the lists that moved a resident or dropped tensor in the plan
      pass (`Applied::touches`), and the capture's `Index` grows with its log instead of being rebuilt every op.
    - **Gates:**
      - `matmul` unit `a_plan_stands_down_to_its_floor_and_narrows_below_it` (the probe's rows: the same block at the
        floor, narrower 64 bytes below, noted on cached calls too);
      - `step219` `the_region_is_sized_by_its_plan_and_never_past_its_cap`: a `K` = 2048 product on 8 units; its cap
        is 110,464; bit for bit in its own region and in exactly the cap. **Negative control:** 64 bytes past the cap
        (`Session::probe_planned_region`), the product narrows and the planned capture is refused by name.

      ttsim, then card 0 (`1792552842`).
    - **Measured** (1.7B, card 0, every tile, 32 / 256-token prompts, ms; before the region took the whole budget):
      off 20.97 / 22.21 (`1792546706`); sized by the plan 20.04 / 21.29 (`1792546841`); fixed 96 KiB 20.03 / 21.28
      (`1792546985`). Even, as expected. The 1.7B's cap is 110,464 (the `K` = 2048 products), so its budget is the 96
      KiB ceiling.
  - [x] **2h-iii. The plan cache** (2026-10-21).
    - **The key.** A session cannot know a graph before capturing it, so plans are kept by the caller's key.
      - `Session::begin_trace_planned_as(key)` keeps the plan if its capture holds;
      - `Session::begin_trace_cached(key)` applies it in one capture, refused (and forgotten) if the graph is not the
        plan's;
      - any residency setting change forgets every plan.
      - burn-tt keys a step by its key, its inputs' and state's shapes and dtypes, and the set's scope
        (`SharedStateTraces::set_plan_scope`). `TracedGenerator` scopes by its model, so two models on one device
        keep apart.
    - **Gates:**
      - `step219` `a_cached_plan_is_applied_once_and_forgotten_for_another_graph`: one capture under a kept plan, bit
        for bit. **Negative controls:** another graph under the key is refused and the key forgotten; a graph rewired
        at its fourth op is refused before that op is queued, and the session runs on bit for bit.
      - The Llama test: a second `TracedGenerator` build captures each trace once (`burn_tt::probe_captures`), bit
        for bit, moving the same bytes.

      ttsim, card 0 (`1792552842`).
    - **Measured** (the 1.7B driver, card 0; first build, then a second in the same process):

      | Residency | First build | Second build | Run |
      |---|--:|--:|---|
      | off | 45.6 s; 54.4 s | 8.76 s; 8.68 s | `1792546706`; `1792556241` |
      | on, the rewrite rebuilding its index every op | 64.2 s | 11.3 s | `1792546841` |
      | on, the index grown and only touched lists rewritten | 63.2 s | 10.2 s | `1792548011` |
      | on, each op also checked before it is queued (`check_op`) | 65.0 s | 11.1 s | `1792554892` |

      - **The first build** captures every trace twice: +20-40% (the off build varies by ~9 s between runs).
      - **The second build** captures each trace once, under its cached plan. Its +2.4 s is the host's work on each
        planned capture:
        - the check of each op before it is queued;
        - the rewrite of the lists it touches;
        - `Applied::verify`'s graph of the whole capture: ~85 ms a step trace, ~570 ms the prefill chunk.
      - **Shared bodies.** The step's attended lengths differ in their attention op, so their plans are not one, and
        they are captured apart.
  - [x] **2h-iv. On by default** (2026-10-21). Residency is on unless `TT_RESIDENCY=0`, for the session and so for
    `burn-tt`'s device. Rows are held by 2 and drops are on, as before. Only captures that can run twice
    (`SharedStateTraces`, so `TracedGenerator`) plan; the single-pass captures are unchanged.
    - **Measured, on against off** (card 0, every tile, release; on / off):

      | Workload | On | Off | Runs |
      |---|--:|--:|---|
      | 1.7B decode step, 32 / 256-token prompt | **20.07 / 21.28 ms** | 21.00 / 22.25 | `1792554892` / `1792556241` |
      | 135M decode step, 32 / 128 / 256 | 8.12 / 8.76 / 9.57 ms | 8.20 / 8.84 / 9.63 | `1792555165` / `1792556373` |
      | 135M prefill, 32 / 128 / 256 | 1521 / 4493 / 5717 tok/s | 1517 / 4480 / 5703 | same |
      | ResNet-18, traced | 14.16 ms | 14.14 | `1792556228` / `1792557424` |
      | ViT-Tiny, traced | 9.34 ms | 9.35 | same |
      | MNIST MLP training, traced, 300 steps (`tt-mnist --train-trace --tiles all`) | 986 / 984 ms | 968 / 990 | two runs each |
      | 1.7B build, first / second | 65.0 / 11.1 s | 54.4 / 8.68 s | `1792554892` / `1792556241` |
      | 135M build (`traced_generation`) | 22.4 s | 13.0 s | `1792555165` / `1792556373` |

      - **No workload runs slower**, so none keeps residency off.
      - **The builds are slower,** which misses this item's bar of no slower than off. Accepted (2026-10-21): a
        build's time is not an optimization target, so residency stays on by default (2h-v).
      - **The 1.7B's `traced_generation`** hit its 900 s limit in the Flex baseline, on and off alike; its step is
        `traced_generation_stream_ceiling`'s.
    - **Gates with it on:** `--smoke` on card 0, 1,178 of 1,178, the model ladder included (`1792553102`); `step188`
      (`1792552873`; no planned capture refused); `step193`, `step197`, `step209`, `step210`, `step217`, `step219`,
      `step220`, 24 of 24 (`1792552842`); `step12_mnist` 8 of 8 and `cargo xtask conformance --release` (2,973 passed,
      52 failed as expected, none unexpected), ttsim.
      - `step197` (the plain trace's bytes) and `step217`'s Phase 1 report turn residency off: their claims are about
        a plain capture.
    - **Found on the way:**
      - **One tile.** `tt-mnist --train-trace` on one tile fails at commit `59e78d1` too: "a list's programs do not
        fit an empty program cache". It is not residency's; on every tile it trains.
      - **A wrong plan can fault the card.** A plan applied to another graph made an op divide by zero on the card
        (`step193`, before `check_op`).
  - [x] **2h-v. The build's cost: accepted, not pursued** (2026-10-21). Residency's first build captures every trace
    twice (+20-40%), and a second build in one process still costs +2.4 s of host checks. A build's time is not an
    optimization target, so residency stays on by default. Should it ever matter, the options, each to be measured:
    - plan from a trace's eager warm-up's ops instead of a plan pass;
    - make `Applied::verify` share `check_op`'s walk;
    - keep plans across processes, keyed as the cache is.

### Phase 3. The products' outputs into their readers' L1 (done 2026-10-21)

`qkv` → RoPE and the cache write, `gate|up` → `silu_mul`, `o` and `down` → the residual adds. A product writes its
output with NC's `SCATTER` record (`tensor.rs` `push_jobs`, `push_stream_jobs`, `push_k_jobs`), which the plan did not
count as a plain move, so every product's output had `Reason::NoPath` and stayed in GDDR (2e: "the residual add still
reads the product's output from GDDR").

- [x] **3a. `SCATTER` as a plain write** (`residency::plain`). A `SCATTER` expands to whole-tile `WRITE`s exactly as a
  `WRITE_RUN` does, and `Applied::rewrite` already expands records on the host, so the output side needed no new record,
  no firmware change and no change to the products' builders: each output tile becomes a `WRITE_L1` to each unit that
  reads it. Every reader's input was already plain (RoPE's `READ_RUN`s and partner `READ`s, `silu_mul`'s and the add's
  flagless `READ_RUN`s, the cache write's `READ`s). A K-blocked product that reloads its partials (prefill's `down`)
  still reads and writes `C` in one op (`ReadAndWritten`) and stays in GDDR.
- [x] **Unicast to few readers** (`residency::UNICAST_READERS` = 4). A tile at most four units besides its writer read
  goes to each by `WRITE_L1`, not by a multicast over their rectangles: RoPE reads each `qkv` tile on two neighbouring
  units, and many producers' path-reserved multicasts serialize (2e). Not measured apart from 3a.
- [x] **A pinning that costs room is not kept** (`Plan::make_rows`). Pinned, an address holds its class's largest member
  in every repeat, and the costliest class leaves first: with the products' outputs as candidates, the 1.7B's pinned plan
  left 143 tensors out for room, the residual stream first (its classes live longest), and kept only 254 resident (the
  layer outside the period kept all of its own). `Room` now also makes the planner plan without the pinning and keep the
  plan that spares more: **393 of 548 tensors resident** (Phase 1's estimate was 401), 262 MB of reads and 122 MB of
  writes a step spared (was 108 and 36), a region of 95,680 bytes of the 98,304 budget. **Cost:** a step's layers no
  longer share addresses, which Phase 5's stored layer wants; programs are unaffected (the region's addresses are in
  the movers' entries, not in kernels). Making the pinned assignment fit as tightly is part of Phase 5.
- [x] **The report names what room left out** (`Plan::report`: `left xN Room (B a unit): ...`).
- [x] **Gates** (ttsim, then card 0):
  - `residency.rs` units `a_scatter_writes_into_its_readers_l1` (a `SCATTER`'s writes are plain and become a
    `WRITE_L1` to each reader, by unicast to two neighbours; a reader reads its own region; watched failing with
    `SCATTER` not plain and with the unicast rule off) and `a_pinning_that_costs_room_is_not_kept` (watched failing
    without the `Room` trigger).
  - `step219` `a_product_writes_into_the_residual_add_bit_for_bit`: norm, `gamma`, a `K` = 2048 product, the
    residual add, on 8 units: bit for bit, the output resident, the planned trace moving fewer GDDR bytes by at least
    what the output moved (20.6 -> 18.4 MB); **negative control:** the output placed over a tensor still to be read is
    refused by `Plan::check` and, applied anyway, gives other bits. Watched failing with `SCATTER` not plain.
  - `step219` `products_under_busy_prefetch_are_bit_for_bit`: two such layers with busy-unit prefetch on.
  - `step219`'s Llama test now asserts the products' outputs resident in every trace (24 a trace, read by RoPE,
    `silu_mul`, the adds, the cache write and attention): `Step(16)` 3.24 -> 1.53 MB of GDDR (2.14 at Phase 2),
    `Chunk(4)` 4.09 -> 1.90 (2.52).
  - Card 0: `step217`, `step219`, `step220`, `step188` (the 1.7B against Flex, and traced against eager; no planned
    capture refused), `step193`, `step209`, `step210`, 27 of 27 (`1792596373`); `--smoke` 1,180 of 1,180 (`1792597728`, residency on,
    `step12_mnist` included).
- [x] **Measured** (1.7B, card 0, every tile, release, 32 / 256-token prompts, ms):

  | | Step | Run |
  |---|--:|---|
  | residency off | 21.02 / 22.21 | `1792593062` |
  | Phase 2 (2h) | 20.07 / 21.28 | `1792554892` |
  | 3a, pinned plan | 20.00 / 21.18 | `1792592920` |
  | **3a and the `Room` trigger** | **19.50 / 20.71** | `1792595329` |
  | the same, busy prefetch cap 1 / 2 / 4 / 8 | 19.74 / 20.66; 19.75 / 21.37; 20.36 / 22.20; 20.63 / 22.31 | `1792595603`, `1792595746`, `1792595888`, `1792596032` |

  The profile (`Step(64)`, `1792596738`, against Phase 2's `1792538078`): products 14.79 ms (unchanged) writing 6.3 MB
  (was 88.1); the adds 0.66 -> **0.39 ms** (8.0 us each), RMSNorm 0.87 -> 0.67, RoPE 0.80 -> 0.68, `silu_mul` 0.45 ->
  0.37; the add, RMSNorm and `silu_mul` read no GDDR, RoPE only its tables. 19.01 ms of summed spans, was 19.58.
  The 135M: **7.74 / 9.18 ms** against 8.17 / 9.58 off (`1792597666`, `1792597699`; Phase 2 8.12 / 9.57).
- **Finding: busy prefetch is no longer held back by GDDR.** With it on (cap 2, `1792597186`) the products go 14.79 ->
  11.25 ms, but the adds 0.39 -> **3.38 ms (70.5 us each) while reading nothing from GDDR**, RoPE 0.68 -> 1.18, RMSNorm
  0.67 -> 1.14. Phase 8's account ("GDDR is one queue": the add waits behind the stream at the controllers) does not
  explain an op whose reads are all `READ_L1` of its own unit's region.
- [x] **3b. The probe, and the fix** (2026-10-21, card 0). The wait was not the `READ_L1`s (the kernel entries that wait
  for them stayed at 1%):
  - **The profile mode stamps more** (firmware `profile` feature, `OPS` mode): B's fetches of a trace's stream chunks and
    reader sections (`fetch`), and its waits for NC to finish a pair (`writer`), as entries headed by `CALL` and
    `PAIR_CALL`, which no real stamped entry is (`tt_isa::dm::trace_mode::{FETCH, WRITER_WAIT}`); new columns in
    `summary.md`, `ops.tsv` and `trace.json`. Under busy prefetch the add's 70.5 us were **66% `writer`**, 2% `fetch`
    (`1792605585`; off, `1792605144`: 19% of 8.1 us).
  - **NC counts its own waits** (a profile image only): its section fetches, credit waits, drains, L1 write issues and
    whole `CALL`s, and B's pump turns, summed into `tt_isa::dataflow::NC_WAITS` (`Session::probe_nc_waits`,
    `burn_tt::probe_nc_waits`, `TT_PROBE_NC_WAITS=1` on `traced_generation_stream_ceiling`). None changed with busy
    prefetch; B's pump turns were at most 0.4 us each. What was left was NC's list-end `noc::wait(PREFETCH_TXN)`: a
    transaction ID's outstanding count is the tile's NIU's, so NC waited for **B's** prefetch reads, which B kept in
    flight by pumping more while it waited for NC
    ([operating notes](../learnings/silicon-operating-notes.md#a-transaction-ids-outstanding-count-is-the-tiles-not-the-cores-2026-10-21-card-0)).
  - **Fixed:** only B drains the prefetch ID at a list's end (`mover.rs` `finish_prefetch`). Busy prefetch pays:

    | 1.7B step (ms), residency on | Busy prefetch off | cap 1 | cap 2 | cap 4 | cap 8 |
    |---|--:|--:|--:|--:|--:|
    | before the fix | 19.50 / 20.71 | 19.74 / 20.66 | 19.75 / 21.37 | 20.36 / 22.20 | 20.63 / 22.31 |
    | after (`1792608298`-`1792608865`) | 19.51 / 20.75 | 19.71 / 20.46 | **18.40 / 19.92** | 18.73 / 20.54 | 19.08 / 20.74 |

    The profile at cap 2 (`1792609145`): the add 12.8 us, products 12.92 ms (14.79 off), summed spans 17.92 ms (19.01
    off). What the stream still costs the small ops is mostly the trace's own fetches (RMSNorm 35% `fetch`, `gamma`'s
    multiply 31%): Phase 5's stored layer.
  - **Gates** (the mover changed): `step50`, `step172`, `step173`, `step215`, `step216`, `step218`, `step219`, `step220`
    on ttsim and on card 0 in the safe image (35 of 35, `1792609649`), `fast-reads` (40 of 40, `1792609825`) and
    `fast-reads` + `encoded-guard` (41 of 41, `1792610007`); `step188`, `step193`, `step209`, `step210` (11 of 11,
    `1792610184`) and, with busy prefetch at cap 2, `step188`, `step193`, `step216`, `step219` (17 of 17, `1792610404`, no
    planned capture refused); `--smoke` 1,180 of 1,180 (`1792610630`); `silicon_profile_mode` (now asserting `fetch` and
    `writer` are attributed, inside each op's span) and `step23` with the `profile` feature (6 of 6, `1792612369`). Firmware clippy plain, `profile`, `fast-reads`,
    `encoded-guard`. **Not gated:** no test fails if NC waits on the prefetch ID again; the evidence is the profile
    before and after.
  - **The other traced workloads with it** (card 0, every tile, release; busy prefetch off / on at cap 2, `1792612512` /
    `1792612811`): the 135M's decode step 7.74 / 8.37 / 9.16 against 7.77 / 8.38 / 9.17 ms, its prefill 1575 / 4651 /
    5863 against 1572 / 4652 / 5862 tokens/s, its build 27.1 / 26.9 s; ResNet-18 traced 13.94 / 14.13 ms, ViT-Tiny 9.31 /
    9.32. **None changes, as expected:** busy prefetch engages only for a one-row product with a packed `A`, which needs
    `K` of at least 1,024 (`PACKED_A_K_TILES`): the 135M's width is 576, ViT's and ResNet's products have many rows or a
    short `K`, and MNIST's training products have a batch of rows and `K` = 784. ResNet's 0.2 ms is between processes,
    not the setting: four alternating runs gave 14.10 off, 14.11 on, 14.12 off, 13.94 on (`1792613197`-`1792613224`).
    The 1.7B in the same runs: 19.50 / 20.74 off, **18.41 / 19.94** on. Gates with it on: `step165`, `step166`,
    `step167` F32, `step185`, `step187`, 15 of 15 (`1792613099`), besides 3b's.
  - [x] **Busy prefetch on by default, at a prefetch cap of 2** (2026-10-21): `Session`'s `busy_prefetch` is on
    (`TT_BUSY_PREFETCH=0` turns it off) and a mover's prefetch ID starts at `dm::PREFETCH_CAP_DEFAULT` = 2
    (`TT_PREFETCH_CAP` overrides; 0 is `TILE_IN_FLIGHT_CAP`). **Correction (2026-10-22):** the cap is not the idle
    units' prefetch's: their `PREFETCH` lists carry no `GATHER_PREFETCH` (`tensor::push_prefetched_jobs`), so they read
    on the mover's own ID at its own cap; only a `BUSY_PREFETCH` list is flagged. With busy
    prefetch off, cap 2 and the old default of 8 measured alike (19.51 / 20.75, 19.50 / 20.71). `step215` now sets the
    prefetch ID's full cap itself: its claim is that the ID is not the mover's own, measured at that cap. With no
    settings: the 1.7B **18.39 / 19.90 ms** (`1792615740`), 19.51 / 20.76 with `TT_BUSY_PREFETCH=0` (`1792615881`);
    gates `step165`-`step167`, `step185`, `step187`, `step188`, `step193`, `step209`, `step210`, `step215`-`step217`,
    `step219`, `step220`, 44 of 44 (`1792613854`, no planned capture refused); `--smoke` 1,180 of 1,180
    (`1792614168`); ttsim `step215`, `step216`, `step217`, `step219`.

### Phase 4. Attention's operands (re-scoped, done 2026-10-21)

Planned as "attention's rows and context, per head", but Phases 2 and 3 had made them resident already, with nothing
attention-specific: RoPE's query is read by attention from L1, attention's packed context twin is held for `o`'s packed
`A`, and the context's own row, which nothing reads, is never written (2g). What attention and its neighbours still read
from GDDR in a 1.7B step (`Step(64)`, profile `1792609145`): the KV cache (state, by design), the mask (6.4 MB a step at
64 positions, 51 MB at 512; `Room`), RoPE's cos and sin (25.6 MB; `Room`) and the cache write's I32 index tile (12.8 MB:
64 units x 48 ops, a whole tile for one position).

- [x] **The index read narrowed** (`index::scatter_rows_jobs`): sixteen indices or fewer sit in face 0, which
  `SCATTER_ROWS` reads alone (row 0 of a `[1, n]` tensor's tile, column 0 of an `[n, 1]` one's first rows), so a batch
  of them reads 64 bytes (`[1, n]`) or 64 a row (`[n, 1]`) from `TILE_DATA`, one `READ` as before. Gate `step189`, now
  with 16 and 17 indices, row and column: bit for bit; **watched failing** with the column form read as 64 bytes (the
  mover refuses the stale words as indices). The cache write 0.48 -> 0.28 ms a step.
- [x] **A second leave-out order** (`Plan::make_rows`, `LeaveOut::LeastSpared`): where a class finds no room, each plan
  is made again leaving out first the class that spares the fewest GDDR bytes a byte of its address, and the plan that
  spares more is kept (the first on a tie). Unit `a_short_region_keeps_what_spares_most_a_byte` (a one-tile tensor
  read by eight ops kept over a three-tile one read once): **watched failing** without the new order. **On the 1.7B it
  changes nothing:** the region is 95,680 of 98,304 bytes and the mask (8,320) and the tables (4,160 each) do not fit the
  2,624 left in either order. They would at the products' floor (110,464), not measured: the step is not bound by the
  small ops (Phase 5a's finding).
- [ ] **Not built: the mask read as row 0.** `SCALE_MASK_FILL`'s row broadcast unpacks only row 0 of each mask tile
  (`sfpu::kernel`, `Operands::RowBroadcast`), so 32x fewer bytes would do, but as plain `READ`s that is two entries a
  tile against one tile read: a `READ_RUN` flag would be needed to make it cheaper, a mover change not justified while
  the step is bound by the weight stream.
- [x] **Gate** `step219` `a_decoder_s_attention_operands_stay_in_l1_bit_for_bit` (card 0): a Llama with 64-wide heads
  and `K` = 1,024, every tile, so the native RoPE, attention as one op and `o`'s packed `A` are the 1.7B's paths. RoPE's
  query, attention's context twin, the mask and RoPE's tables resident; attention's unread rows dropped; bit for bit
  against the unplanned traces, `Step(64)` 61.6 -> 48.3 MB of GDDR. Found on the way: a step attends whole tiles of
  positions to run attention as one op (`attention::fits`), and a product of `K` under 1,024 gathers its `A`, which has
  no L1 path, so on the 135M attention's context goes to `o` through GDDR.

### Phase 5a. A replay's next reader section read ahead (done 2026-10-21)

The layer sub-traces' cheaper first step ([layer-sub-traces.md](../proposals/layer-sub-traces.md)): every chunk a replay
runs was read, then waited for, in turn (`call_chunks`). B now reads the first chunk of the next `PAIR_CALL`'s reader
section ahead, into `dataflow::AHEAD_CHUNK` (the free 2 KiB after `OUTER_CHUNK`), on a transaction ID of its own
(`dm::TXN_FETCH` = 8), and the pair waits only for that ID. It is issued when an outer chunk lands (its first pair) and
when a pair's reader section ends (the next one, found as the first `PAIR_CALL` in the outer chunk whose section is
stored later), so it lands while B waits for NC. A chunk read ahead is used only by the pair it was read for, and is
drained by B alone, before it is used and at a list's end (an ID's count is the tile's NIU's, 3b). No format change; B's
safe image 33,152 of 43,008 bytes.

- [x] **Gates** (ttsim, then card 0): `step39`, `step172`, `step173`, `step215` on ttsim in the safe, `fast-reads` and
  `fast-reads` + `encoded-guard` images, `step216` in the safe one; on card 0 the safe image 63 of 63 (`1792618297`: `step39`,
  `step50`, `step172`, `step173`, `step185`, `step187`, `step188`, `step193`, `step209`, `step210`, `step215`,
  `step216`, `step218`-`step220`), `fast-reads` 57 of 57 (`1792618769`), `fast-reads` + `encoded-guard` 58 of 58
  (`1792619005`); `silicon_profile_mode` and `step23` with the `profile` feature, 6 of 6 (`1792622726`, the `fetch`
  column now the wait left); `--smoke` 1,181 of 1,181 (`1792621118`, `step12_mnist` included). Phase 4's gates with
  `step188`, `step193`, `step209`, `step210`, `step217`, `step220`: 28 of 28 (`1792617802`). Firmware clippy plain,
  `encoded-guard`, `profile`, `fast-reads`. **Watched failing:** the read ahead
  taken for the pair before (`>=` for `>`) and used without the match check: `step216` differs.
- [x] **Measured** (1.7B, `1792619550`-`1792620919`, [log](../learnings/performance-log.md)): a step's fetch 0.83 -> 0.23 ms
  and the small ops 5.00 -> 4.47 ms, with Phase 4's index read, but the products 12.92 -> 13.46 ms: 18.44 / 19.83 ms
  against 18.39 / 19.90 before and 18.32 / 19.78 with the index read only; prefetch caps 3 and 4 no better. With busy
  prefetch off, 19.51 / 20.76 -> 19.27 / 20.48. **The step is bound by the weight stream that busy prefetch reads in the
  small ops' windows:** what the small ops give up, the next product waits for. The 135M (no busy prefetch) 7.74 / 9.16
  -> 7.61 / 9.06.
- **So Phase 5b (the stored layer body) is deferred:** what it would take from the small ops (fetch 0.23 ms and part of
  control's 0.75 a step) moves into the products the same way. The lever is the stream's rate under the small ops (the
  products reach about 391 GB/s; the target is 460-486).

### Later phases

5. **Sub-traces on the plan** (`layer-sub-traces.md`, 5b): one stored layer whose only arguments are GDDR references.
   Deferred by 5a's finding; first make the pinned plan spare as much as the unpinned one (Phase 3's cost).
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
