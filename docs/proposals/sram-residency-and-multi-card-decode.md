# SRAM-resident weights and multi-card decode: what Cerebras' layout teaches Blackhole

Status: **proposal, unscheduled** (2026-10-25). A review of Cerebras' "How we deploy trillion-parameter frontier models
on Cerebras" (2026-10-22, Kimi K3 as the worked example) against this stack. Nothing here is measured on our cards
yet. It records:
- one idea rejected with its arithmetic: weights resident in L1;
- one candidate batch-1 lever: exact tensor parallelism across the two cards;
- two notes for [batched-decode.md](batched-decode.md): pipelined micro-batches, and KV capacity as the concurrency
  ceiling.

Roadmap: Track 3, multi-chip ([roadmap.md](../plans/roadmap.md#track-3-multi-chip-after-the-single-chip-path-is-fast)).

## What the post describes

It is **weight-resident pipelining, not weight streaming**. No external weight store (MemoryX) appears in it.

- **Weights live on the wafer.**
  - A wafer has 44 GB of SRAM. A 3T model at 4-bit weights is ~1.5 TB, ~16 GB a layer over its 93 layers.
  - So whole layers sit on a wafer and "only activations cross from one wafer to the next", over RDMA over Ethernet
    at ~2 us a hop.
- **Placement is by layer kind:**
  - A dense (full-attention) layer gets a wafer to itself. The SRAM left after its weights is that layer's KV cache.
  - Linear-attention layers keep a fixed state a request, so two share a wafer.
  - 24 + 35 = 59 wafers.
- **Throughput comes from the pipeline.**
  - Every stage works on a different request or micro-batch, so the requests in flight are about the stages times the
    micro-batch size.
  - A smaller micro-batch makes each request faster; a larger one raises total throughput.
  - Requests stream through continuously and do not wait for each other, unlike a GPU batch.
- **Concurrency is capped by memory.** Every request needs its own KV cache on every dense-layer wafer, so the SRAM
  left after the weights sets how many requests can run.
- **Precision:** 4-bit weights and an 8-bit KV cache.
- **Metric:** tokens/s per megawatt, compared at the same per-user generation speed. Batch size and concurrency are
  not efficiency measures on their own.
- **Not covered:** prefill, MoE, speculative decoding, tensor parallelism.

## Our figures beside the wafer's

| | Cerebras wafer | Blackhole p150a (one card) | Source |
|---|---|---|---|
| On-chip SRAM | 44 GB | 1,536 KiB × 120 tiles = 189 MB raw; `DATA` 917,504 B × 120 = 110 MB | [tt-metal-concepts-review.md](../learnings/tt-metal-concepts-review.md), `tt_isa::l1::DATA` |
| Off-chip memory | none used in the post | GDDR6, 512 GB/s peak, 469-489 GB/s read | [scoreboard](../learnings/performance-scoreboard.md) |
| Chip-to-chip | ~2 us a hop (RDMA over Ethernet) | 50 GB/s a link; a hop's latency **not measured**; `to_device` is a host round trip today | scoreboard, [roadmap.md](../plans/roadmap.md) Track 3 |

A wafer has ~230x a card's raw L1 and ~400x its data area.

## Rejected: weights resident in L1

- **Whole models.** The 1.7B's F32 weight slots are 6.957 GB ([decode-to-bandwidth.md](../plans/decode-to-bandwidth.md),
  "The 2026-10-19 review"). Divided by one card's 110 MB data area:
  - ~63 cards at F32;
  - ~32 cards at BF16;
  - ~9 cards at BFP4 (592 B a tile against 4,096).

  We have 2 cards, and every tile's data area is also each op's working memory.
- **A pinned slice of each product's weights.** There is no L1 to pin it in:
  - A `K` = 2048 product needs 553,280 B of the 917,504-B arena. A second job's weight ring is already 320 B short
    (decode-to-bandwidth, "Room for a second job's `B`"). That ring is being built from `MAILBOXES` and `PROGRAMS`
    instead.
  - The residency region (98,304 B) already sits under the products' cliff of 110,464 B. 64 B past the cliff, the
    1.7B's products narrow and slow by 8-10% ([planned-l1-residency.md](../plans/planned-l1-residency.md) 2h-i).
  - Even if 300 KB a tile were free, that is 36 MB: 0.5% of the 1.7B's weights, ~0.07 ms of a 16.8-ms step.
- **Small models do not need it.** The 135M's F32 weights are ~0.54 GB, ~1.2 ms at 469 GB/s, yet its step is
  8.00-9.38 ms. Small ops and fixed per-op cost bound it, not the weight stream.
- **Verdict:** L1 belongs to activations (planned residency) and to the weight stream's rings. Weights stay in GDDR
  and are streamed. Revisit only for a model that fits a mesh's data area many times over, i.e. tens of cards.

## Candidate: exact tensor parallelism across two cards (batch 1)

The post's principle, that only activations cross the link, fits two cards. A decode row is 8 KB at F32 (the 1.7B's
2,048-wide hidden state). The weights are 6.96 GB.

### The split

Each card streams half of every layer's weights. No partial sums cross the link.

- **Products are split along `N` (output columns), never along `K`.** Each output element is computed whole on one
  card, in its single-card `K` order. `Topology::Cards` already does this and is bit-identical to one card.
- **Partial products are joined by all-gather, not all-reduce.** A gather copies; it never adds. So the step stays
  **bit for bit** with one card. An all-reduce of `K`-split partials would reorder the sums. Batch-1 decode's
  bit-for-bit target and the numerics rules (preserve stated accumulation order) rule that out.
- **Attention is split by head.** Card `c` holds its heads' `q`, `k` and `v` columns, attends to them, and keeps their
  half of the KV cache. Each card's cache halves.
- **Small ops are replicated.** RMSNorm, the residual add and RoPE run on both cards on the replicated row.

### Gathers in one Llama layer

| After | Each card sends | Bytes a half (1.7B, F32) |
|---|---|--:|
| attention's context (its heads) | `[1, 1024]` | 4 KiB |
| `o` (its columns) | `[1, 1024]` | 4 KiB |
| `silu(gate) * up` (its columns of the 8,192) | `[1, 4096]` | 16 KiB |
| `down` (its columns) | `[1, 1024]` | 4 KiB |

That is four gathers a layer, ~96 a step, plus the head: each card scores its vocabulary half, and the argmax
compares two candidates.

### Cost model (to be measured)

- **Weights:** the floor halves, 14.3 -> ~7.2 ms at F32 (and 7.2 -> ~3.6 ms with BF16 storage).
- **Gathers:** 96 × (hop latency + ~0.1-0.3 us at 50 GB/s). At 2-5 us a hop, 0.2-0.5 ms a step, on the critical
  path. The next product's weight prefetch can run under each wait.
- **Small ops:** unchanged, ~4.5 ms, because both cards run them.
- **Estimate:** a 1.7B F32 step falls from ~16.8 ms to ~10-12 ms. After BF16, it is the largest batch-1 lever left,
  and the two combine.
- **Capacity:** a model whose weights fill one card's GDDR fits across two.

### Why it is not scheduled

- It reopens a decision. The roadmap (Track 3, design decisions 2026-10-14) keeps tensor parallelism an eventual
  goal, deliberately not pursued now. This proposal does not change that decision; it names what would.
- **Prerequisites,** all on the roadmap's foundation list:
  - per-card sessions on every tile;
  - card-to-card transfers over Ethernet with no host;
  - tensor placement that can say *replicated* and *split along `N`*;
  - cross-card send and wait entries inside each card's trace.
- **The first step is a measurement.** It decides the whole cost model: one card-to-card Ethernet transfer of
  4-16 KiB, from L1 to the other card's L1, driven by the E1 movers with no host. Report its latency, median over
  warm runs on cards 0 and 1, in the scoreboard. Above ~20 us a hop, 96 gathers cost what the halved stream saves,
  and the idea is dead.

## For batched decode: pipelined micro-batches

- **The scheme.** Layers 0-11 on card 0 and 12-23 on card 1, with at least two micro-batches in flight.
  - Each card streams half the weights per micro-batch step, with no collectives. Only `[B, hidden]` rows cross the
    link.
  - This is Track 3's "Pipelining (layers on different cards) over card-to-card `to_device`; micro-batching later",
    which the post confirms.
- **Trade-off against tensor parallelism:**
  - It does not shorten a batch-1 token: card 1 waits for card 0.
  - It doubles throughput under load.
  - It needs no change to the bit-for-bit argument: each sequence runs the single-card program.
- **How they relate:**
  - [batched-decode.md](batched-decode.md) adds rows that share one weight stream;
  - pipelining adds cards that each stream less;
  - tensor parallelism shortens a token.

## References

- Cerebras, "How we deploy trillion-parameter frontier models on Cerebras", 2026-10-22,
  https://www.cerebras.ai/blog/how-we-deploy-trillion-parameter-frontier-models-on-cerebras
- [batched-decode.md](batched-decode.md): KV capacity and the per-user-speed metric, added from this review.
- [decode-to-bandwidth.md](../plans/decode-to-bandwidth.md), [planned-l1-residency.md](../plans/planned-l1-residency.md):
  the single-card floor and the L1 budget used above.
