# Planned L1 residency: activations that survive between ops

Status: **proposal, unscheduled** (2026-10-20). Asked for after Phase 8 of
[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md): the space audit, busy-unit prefetch and the prefetch pump.
It is the "sharded, L1-resident tensors" gap of [tt-metal-concepts-review.md](../learnings/tt-metal-concepts-review.md)
(G12). That gap was deferred "until a measured op chain is GDDR-bound", and that measurement now exists. It is designed
together with [layer-sub-traces.md](layer-sub-traces.md): a planned layer has the same L1 addresses in every repeat.
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

1. **The plan, reported only.** The plan pass, liveness, period detection, slot assignment and the hazard checks, run
   on the captures of the decode step, printed beside the space report. Nothing is resident yet.
   - Gates: the plan is the same on every run; the period plan puts every repeat at the same addresses; a mutated plan
     that reuses a slot before its last reader is refused.
2. **The residual stream:** add, RMSNorm, `gamma`'s multiply, and the next product's packed `A`, as a replicated row and
   twin. This needs the NC multicast record (ttsim spike first) and L1 inputs for those four ops. It is the part
   busy-unit prefetch hurt most.
   - Gates: bit for bit the plain capture (`step188`, `step193`, `step209`, `step210`, on card 0); no GDDR bytes for
     the resident tensors; a negative control writing a slot before its last reader's barrier fails on silicon.
   - Measured with busy-unit prefetch on and off.
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
