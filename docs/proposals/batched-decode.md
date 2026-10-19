# Batched decode, laid out on Blackhole's tiles

Status: **proposal, unscheduled** (2026-10-18). Batch 1 comes first: prefill and decode speed
([decode-to-bandwidth.md](../plans/decode-to-bandwidth.md)) is the current track. This note fixes the goals and the
shape of a batched decoder, so that batch-1 work does not close doors. It also records which batch-1 pieces already
serve it. Roadmap: Track 1, "Batched decode".

## Why it is the largest knob

A decode step reads every weight once whatever the batch. A 1.7B step moves 6.95 GB of F32 weights, ~14.5 ms at the
card's ~480 GB/s ceiling.
- **At batch 1:** the step makes one token.
- **At batch `B`:** the same bytes make `B` tokens, for the cost of the arithmetic and the per-sequence work (each
  sequence's attention and cache write).
- **Against the roofline** ([scoreboard](../learnings/performance-scoreboard.md#roofline-smollm2-17b-f32-one-card)):
  132-863 tokens/s "realistic" at batch 4-32, against ~30 tokens/s at batch 1 today.

## Why tiles decide the shape

A matmul is 32x32 tiles. Each `MVMUL` multiplies 8 rows of `SrcB` (one half of a 16-row face of `A`) by a 16x16 face
of `B` into 8 rows of `Dst`.
- **The rows a product computes come in steps of 8** ([`LiveRows`](../../crates/tt-kernels/src/matmul.rs),
  `step203`):
  - 1-8 rows cost 16 `MVMUL`s a tile pair at HiFi4;
  - 9-16 rows cost 32;
  - 17-32 rows cost 64 (both faces of `A`'s tile row).
- **The weights' bytes are the same for 1 row or 32:** a weight tile is read once and multiplied into every row of
  the tile row.
- **The per-pair compute at 32 rows:** measured at ~0.23 us a pair (profile `1792298755`; all 64 `MVMUL`s at HiFi4).
  A unit reads a 4,160-byte weight tile in ~0.6 us at its GDDR share (~7 GB/s, 64 units saturating).
  - So **a batch of up to 32, one tile row, is still bound by the weight stream at HiFi4.**
  - Two tile rows (33-64) are about at the balance point.
  - Beyond that, compute binds. Lower fidelity (BF16 weights at HiFi2 or LoFi) moves the balance point up.
- **Batch sizes are therefore 8, 16 and 32** (then multiples of 32). A batch of 5 costs a batch of 8's arithmetic. The
  scheduler rounds the batch up to a row group and masks the empty rows. It never pads to a whole tile row unless it
  is past 16.
- **`Dst` holds 512 FP32 rows,** so eight 32x32 FP32 output tiles at once. A batch-32 product can accumulate up to
  eight output columns in `Dst` across `K` chunks. This is the same structure the batch-1 "compute under the reads"
  work needs (one column at a time there).

## Goals

1. **Bit for bit with batch 1, per sequence.**
   - Each row's products keep their `K` order, and each sequence's attention is its own.
   - A sequence decoded in a batch gives the logits it would alone.
   - This is the main gate, as `step185`/`step187` are for batch 1.
2. **One trace per (batch bucket, attended-length bucket).**
   - Shapes are fixed, as `TracedGenerator`'s are today: batch buckets 8, 16, 32 (and 64).
   - Sequences join and leave by slot, with the empty slots masked. No re-capture when a sequence ends.
3. **The weight stream is shared.** Every product reads each weight tile once a step for the whole batch. No product
   path may scale its `B` bytes with the batch.
4. **Per-sequence work scales with the sequence, not the batch's maximum.**
   - Attention reads each sequence's own attended positions.
   - A sequence at position 40 in a batch with one at 500 does not pay for 500 positions.
5. **No host round trip a step** beyond the token ids in and the sampled ids out: sampling (argmax, or top-k later)
   runs on the card.

## Design

### Activations: sequences are rows of one tile row
- **Layout:** a step's activations are `[B, hidden]`, row `b` sequence `b`. Every element-wise op, norm and product
  is today's op at `m = B`. The batch is just `m`.
- **Batch 1 already serves this:**
  - row-limited SFPU programs (`sfpu::with_rows`, row groups of 4, Phase 3);
  - RMSNorm's row-group statistic (S3);
  - thin `A` reads (P1);
  - `A` read once a unit (B1).
  All of these are parameterised by `m`, not specialised to 1.
- **Products:** `LiveRows` trims a product's last tile row to its row group.
- **What batch 1 does that does not carry over:**
  - **The deal of thin products over 64 units** stays: the weights are the same, and only `A` grows to `B` rows.
  - **A packed thin `A`** (a `[1, K]` row laid out as `[K/32, 32]` so a unit reads it in `K/1024` tiles; considered
    for batch 1) only pays at `m = 1`. At `B >= 8`, `A`'s thin read is already dense.

### The cache: a pool of position-major pages
- **The page:** the batch-1 cache is position-major, `[L, kv d]` a layer side (decode-attention A-layout). A batched
  cache keeps that row form and cuts it into **pages of 32 positions**: one tile row of `[32, kv d]`, whole tiles
  wide when `d` is a multiple of 32.
- **The pool:** pages come from one pool a layer side. Each sequence has a page table, its pages in order.
  - A sequence's attended positions are its pages, so attention reads only them (goal 4).
  - Prefill fills whole pages. A step writes one row of its current page.
- **The write:** a step writes `B` rows, row `b` to (page of sequence `b`, its position mod 32). That is one
  `SCATTER_ROWS` with `B` indices. The record already takes a batch of indices, and its last-index rule handles
  repeats. The indices are a `[1, B]` row built on the card from the page tables and the positions.
- **Pages make the tile shape and the cache shape the same:** a page is exactly the tile row `qk`/`pv` read. A head's
  keys in a page are a tile-aligned column block, read in place, transposed by the unpacker (A0b).

### Attention: per sequence, batched over sequences and heads
- **The shape:** for sequence `b` and key/value head `h`, `q` is `[groups, d]` (GQA folded into rows, as today).
  - `K` and `V` are the sequence's pages' column blocks, `qk = q K^T` over its pages, then the softmax, then `P V`.
  - The softmax runs on rows `groups` wide, row-limited.
- **The batch:** the batch-1 batched product (`matmul_dram_batched`) takes a list of `(A block, B block)` items.
  Batching adds items: one per (sequence, head, page run).
  - Its deal over units is already by item.
  - A per-item `K` length (pages) is the new piece: today every item has the same `[m, k, n]`.
- **Lengths:** varying lengths make a ragged batch. Pages keep each item whole tiles; only the last page of each
  sequence is masked. The mask is a `[B, 32]` row of the last page's valid positions.
- **Order change:** splitting a long sequence's positions across units (flash-decode) changes the softmax's sum order.
  That is a decision point, as for batch 1 (decode-attention A5), and is not part of this design.

### Positions, RoPE, the embedding and sampling
- **Positions:** a `[B]` tensor. RoPE selects `B` rows of its table (`select_rows` takes a list today), one per
  sequence.
- **The embedding:** `B` rows selected from the table in one op (`select_taps_jobs` already deals rows over units).
- **Sampling:** argmax over each row of `[B, vocab]` as a row reduction with the index (a new reduce kind). Only `B`
  ids leave the card a step.

### Scheduling (the host side)
- **Slots:** a `BatchedGenerator` holds `B` slots. A request is prefilled into free pages (batch-1 chunks today, or
  chunked prefill packed into rows later), then joins a free slot.
- **The step:** each step replays the trace for (bucket of live slots, bucket of the longest live page count).
  Empty slots carry a zero mask and write a dummy page.
- **Prefill and decode in one step** (chunked prefill rows beside decode rows in the same tile rows): a later
  extension. It needs per-row causal masks, which the row-limited mask op already takes.

## What batch 1 must keep in mind now
- **New kernels take `m` (or a row count), never assume one row.** `LiveRows` is the pattern: a trim chosen by rows,
  bit for bit for the rows it computes.
- **A product's `K`-chunked compute** (decode-to-bandwidth, compute under the reads, as built) passes each output
  tile's FP32 partial through L1 between chunks, one column a job. It takes any `m`. A batch-32 product could instead
  hold several output columns in `Dst` (up to 8 FP32 tiles) across chunks, if the per-chunk reload and pack (~3 us)
  become the bound.
- **The packed `A`** (`tensor::pack_row`) and the weight prefetch are `m = 1` and capture-time choices. A batch keeps
  `A` dense in its tile rows. The prefetch on idle units carries over unchanged: it is about the weights.
- **The cache API** (`FixedCache`, `FixedWrite`) stays layout-agnostic for decoders: `write` takes head-major keys, and
  `attended` gives a head-major view. A paged cache is a third layout behind the same two calls, plus a page table.
- **Trace state donation and in-place writes** (in-place-cache-write) carry over: the pool is the donated state.

## Phases (when scheduled)
1. **B0, the gate first.** A synthetic Llama decoded at batch 8, its sequences at different positions, must give
   each sequence's batch-1 logits bit for bit. It fails until the rest lands.
2. **B1:** `m = B` through the decoder with one shared position: everything but attention and the cache.
3. **B2:** the paged cache, and the batched write with `B` indices.
4. **B3:** per-sequence attention items with per-item `K` (pages).
5. **B4:** on-card argmax and the slotted generator with buckets.
6. **B5:** measure tokens/s at batch 8, 16 and 32 on the 1.7B, against the roofline's batch rows.
