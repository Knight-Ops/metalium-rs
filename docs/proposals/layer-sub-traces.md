# Layer sub-traces: one stored layer, called once a layer

Status: **proposal; its cheaper first step done, the body deferred** (2026-10-21). The next reader section's first
chunk is read ahead on its own transaction ID ([planned-l1-residency.md](../plans/planned-l1-residency.md) Phase 5a): a
1.7B step's fetch 0.83 -> 0.23 ms, but the step is unchanged, since under busy prefetch what the small ops give up the
products wait for. A stored body would take the rest of the fetch and some control the same way, so it waits until the
weight stream, not the small ops, sets the step. (2026-10-20: proposal, unscheduled.) Asked for on 2026-10-19: trace storage and L1 are running out, so
what could rearchitecting free? This note is the trace-storage half. The L1 half is in
[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8: the space audit, and the packed-`A` ring reclaim,
which is done. Roadmap: Track 1, "Decode at GDDR's bandwidth", and Track 2, "Trace segmentation".

## What a trace is today

- **Storage.** Each unit's stream is stored in GDDR as its own region: the stream, plus the reader and writer section
  of every `PAIR_CALL` (`Session::store_stream`, `trace::chunked`).
- **Replay.** A replay is one `CALL` a unit (`dm::op::CALL`):
  - the mover reads the stream into `TRACE_CHUNK` 64 entries at a time;
  - it waits for each chunk (`call_chunks`: `READ`, then `noc::wait(TXN)`) and runs it;
  - it fetches each `PAIR_CALL`'s sections the same way.
- **Relative values.** Generations and barrier targets are stored relative, and `CALL` adds the replay's bases.

Each layer of a decoder is captured again in full, so a trace holds 24 copies of one layer's entries.

## What was measured (1.7B, every tile, 2026-10-20, `session::probe_space_report`)

Card 0, runs `1792466481` and `1792467415`. The capture has the decode steps for 64, 128, 256 and 512 attended
positions, plus one 64-token chunk.

| Trace | Ops | Distinct ops | Stream entries, all units |
|---|--:|--:|--:|
| `Step(64)` … `Step(256)` | 404 | 32 | ~635,000 (~20 MB) |
| `Step(512)` | 400 | 30 | ~635,000 |
| `Chunk(64)` | 789 | 60 | ~3,315,000 (~106 MB) |

- **"Distinct"** means the op's label, its first list and its entries on each unit.
- **The period.** The ops repeat every 16 (one layer): 365 of 388 ops equal the op 16 later.
- **What differs between layers.** Of the 4.65 M words those 365 ops hold, **827 K (17.8%) differ** from the next layer's:

  | What differs | Words | Share |
  |---|--:|--:|
  | tensor references of records (`GATHER` 276 K, `READ_RUN` 225 K, `WRITE_RUN` 94 K, `SCATTER` 92 K, `SCATTER_ROWS` 24 K) | 711 K | 86% |
  | plain `READ`/`WRITE` GDDR addresses (word 3) | 44 K | 5% |
  | `BARRIER` targets (word 1) | 31 K | 4% |
  | `LAUNCH` / `KERNEL_WAIT` generations (word 1) | 40 K | 5% |
  | `KERNEL` | 1.4 K | — |

- **The rest is the same layer to layer:** every L1 address, every program and every count. So a layer is one body,
  plus the tensors it touches and two counters that advance by a constant per layer.
- **Per unit,** a `Step` trace runs ~5,300 entries (`PAIR_CALL` sections included), about 220 a layer. That is ~83
  chunk fetches, plus a section fetch per `PAIR_CALL`, each of them waited on in turn.

## Why it matters beyond storage

The Phase 8 probe ([performance log](../learnings/performance-log.md#the-phase-8-feasibility-probe-the-step-without-its-weight-stream-and-under-one-2026-10-20-cards-0-and-1))
shows that a step's own work costs 1.6x under a ~300 GB/s weight stream. Most of that is latency: draining entries
waiting on reads.

A trace adds serial reads of its own:
- every 64 entries, a chunk fetch the unit waits for;
- every op, a section fetch.

Under the stream, each of these waits longer. A stored layer small enough to stay in L1 (~220 entries, ~7 KB a
unit) takes them off the critical path. It also makes room in GDDR and in the trace's own budget, which bounds
trace segmentation.

## Design

1. **The body.** At `store_trace`, the capture finds a run of ops repeating at period `p`.
   - The test is generic, not model code: equal signatures, then word-for-word equality outside the argument words.
   - One unit's body is that unit's entries for the `p` ops, sections included.
   - The other copies become calls.
2. **Arguments.**
   - **Tensor references:** a ref in a body is replaced by an argument marker, `[ARG, k, 0, …]` in place of the
     ref's first entry, with the second entry unused. The mover takes call argument `k`'s two entries instead, and
     checks them exactly as it checks a ref today (`TensorRef::decode`, the channels the chip has).
   - **Plain addresses:** a `READ` or `WRITE` with a GDDR address becomes an argument too.
     - Either a word marker, or the host emits these moves as `READ_RUN`/`WRITE_RUN` records with a ref.
     - Decide by mover image cost.
3. **Counters.** A call carries its `generation_base` and `barrier_base` as `CALL` does. The host checks that each
   layer's counters equal the body's plus a constant before choosing a body.
4. **The call entry.** A new `CALL_BODY` names:
   - the body: channel, offset and count, or an L1-resident body's address;
   - the two bases;
   - the argument table: where the layer's refs are, in the unit's stream or in L1.

   The top-level stream becomes a head, 24 `CALL_BODY`s with their tables, and a tail. A body calls nothing: one
   level, as `CALL`'s "no call in a call" rule has it.
5. **Residency (second step).** A body kept in L1 for the replay is fetched once per replay. A body kept across
   replays and traces is fetched once.
   - It needs a region: ~7 KB a unit a body.
   - Its candidates are the space audit's: `MAILBOXES` (128 KiB, use unmeasured), `PROGRAMS` (4 x 32 KiB), or `DATA`
     now that the products' plans end at 553 KB.
6. **The host's self-check, before storing.** Expand every call with its arguments and compare the result with the
   captured stream, word for word. A mismatch stores the plain stream and says why. Correctness then rests on an
   exact host comparison as well as on hardware gates.

## Expected

- **Storage:**
  - `Step` traces ~635 K -> ~35-40 K entries: one body, 24 calls, each table ~20-40 refs a unit;
  - `Chunk(64)` 3.3 M -> ~0.2 M.
  - That is ~16x in GDDR, and fewer slots for trace segmentation to manage.
- **Fetches:**
  - with a body resident, a unit's fetches fall from ~83 chunks plus ~400 sections a step to the head, the tail and
    the tables;
  - the probe's tax suggests that is worth a share of the ~1.3 ms the small ops already lose to today's prefetch, and
    more under Phase 8's stream. To be measured before building residency.
- **Programs:** unchanged. Programs are shared by hash already; the program cache is bounded by per-length variants
  (attention), not by repetition.

## Cheaper first steps

- **A chunk fetched ahead.** Fetch chunk `n + 1` while chunk `n` runs, on a transaction ID of its own.
  - This is the same mechanism as Phase 8 Step 3's prefetch ID: `noc::wait(TXN)` would no longer drain it.
  - It halves the serial fetches with no format change.
  - The same goes for a `PAIR_CALL`'s next section (decode-small-ops S5, "the next reader section fetched ahead").
- **Measure the fetch's share first.** Use the profile mode's default detail (`OPS`): B stamps a trace's fetches
  (`dm::trace_mode::FETCH`) only there, and `TT_PROFILE_DETAIL=full` puts them in `other`. Measured 2026-10-21: 0.83 ms of
  a 1.7B step's 5.00 ms of small ops (`1792609145`), 0.23 with the section read ahead (`1792620146`).

## With planned L1 residency

[planned-l1-residency.md](../plans/planned-l1-residency.md) plans a captured graph's activations into L1 a layer at a time, with
the same addresses in every repeat. With it, a body's L1 words are the same in every layer, and its arguments are only
GDDR references: the layer's weights, its cache, and what is left in GDDR. Its argument tables shrink too, since 86% of
the words that differ between layers today are tensor references, many of them activations.

## Risks and rules

- **Encoded reads.** A fast image's `READ_ENCODED` binds its checksum to a per-start key, the tile, the chip and the op
  (`encoded::Context`). A body shared by 24 layers cannot carry 24 checksums. Bodies hold no encoded entries: they
  stay in the call's own stream, or sub-traces start on the safe image only.
- **Ownership.** `check_pair` runs on every binding: each layer's sections expanded with its arguments, not just the
  body.
- **The profile mode** counts ops by index in a trace. A call stands for `p` ops, so windows count through calls.
  `BARRIER_STANDS_FOR` already does that for skipped barriers.
- **Mover image:** ~18 KiB free in B's 42 KiB slot. `ARG` decoding and `CALL_BODY` are small. Residency is a region,
  not code.
- **Generality.** Detection is by period and word equality, never by model. ViT's 12 blocks and a training step's
  repeated layers fit the same rule. Anything that does not repeat keeps the plain stream.

## Phases and gates

1. **Measure:** the fetch's share, from the full-detail profile with and without the probe's background.
2. **Store-time detection and the self-check, host only.** Report what each trace would become. Gate: the expansion
   equals the stream for every trace of `step187` and `step188`, and a corrupted argument is caught.
3. **`ARG` and `CALL_BODY` in the mover**, ttsim first. Gates:
   - a replay equals the plain trace bit for bit (`step185`, `step187`, `step188`, `step193`);
   - `step172` and `step173` in the safe, `fast-reads` and `fast-reads` + `encoded-guard` images;
   - firmware clippy plain, with `encoded-guard`, and with `profile`;
   - negative control: a call given another layer's table diverges.
4. **Residency,** if phase 1 says the fetches matter.
