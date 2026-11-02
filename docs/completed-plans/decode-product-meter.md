# A decode product's pipeline, metered from inside (roadmap Track 2 follow-ups, item 2)

Opened 2026-11-02 on `traced-generation`, after item 1a (`5a1640f`). The design reference's H11 is open on this
measurement ([dataflow-compiler.md](../learnings/dataflow-compiler.md) §13, H11; §6, X10).

## Why

- In their own spans, the 1.7B's decode products read 233-306 GB/s (F27, `1793130462`).
- The same deal reads ~499 GB/s on 96 units with nothing else running (H11's bound, `1793463838`). So the read
  stream's structure is not the limit.
- Neither half binds alone: reads ~5x cheaper (`fast-reads`) bought 0.2 ms a step, and with the weights free a step
  is 6.26 ms.
- What is left is each unit's reads and kernel together, through its slot ring. Every decode hypothesis since F27
  (H4, H5, H10, H11) was bounded from outside a product, and no meter shows that pipeline over time.

This item builds that meter, measures the 1.7B's step with it, and names the next transform by the rules below. It
transforms nothing itself.

## What a product runs on a unit (2026-11-02)

- `Session::matmul_dram` (`m = 1`):
  - `gate|up`, `qkv` and `o` are `K`-whole `Schedule::Blocks`.
  - `down` (`K` = 8192) is `Schedule::Stream`, at least four `K` chunks. Each continuation reloads its partials
    (`matmul_items_continuing`).
- A unit's blocks run as one `op::PAIR` (`streaming_segments`):
  - B, per batch: `[RELEASED]`, `CB Reserve`, the gather, `CB Push`. `Push` first waits for the batch's reads to land.
  - NC: `CB Wait`, the scatter, `CB Pop`.
  - The roles: T0 waits for input, unpacks and pops; T2 reserves output, packs and pushes. All three retire each
    batch before the next.
- The ring holds **one batch for `Blocks`, two for `Stream`** (`capacity`, hard-wired).

## The meter

The profile mode's third detail, `Detail::Pipeline` (`TT_PROFILE_DETAIL=pipeline`). It is `OPS` for the movers, so
every image without the `profile` feature, and the default detail, are unchanged.

- **B's side is already stamped.** `OPS` stamps every `CB` entry, and a `PAIR`'s reader runs its input credits as
  Reserve, Push, Reserve, Push. So per batch:
  - `reserve -> granted` is the Reserve entry: B waits for a slot, which T0's pop of an earlier batch frees;
  - `granted -> issued` is the gap to the Push entry: the gather. Each of its reads waits for room
    under the stream's in-flight cap (`STREAM_CAP`, 2-3), so this is most of the batch's reads in
    flight, not only their issue;
  - `issued -> landed` is the Push entry: B waits for the batch's last reads to land.
- **The roles' side is new.** Under `mailbox::trace_word::BATCHES`, T0 and T2 each record three clock reads a batch
  into their mailbox's batch table (`mailbox::BATCH_TIMES`, at most `BATCH_TIMES_MAX` batches):
  - T0 entering its input wait, its wait returning, and its pop;
  - T2 entering its output reserve, the reserve returning, and its push.

  B copies them into its batch log after the kernel's `ROLES_DONE` (`dm::BATCH_LOG`, plain L1 stores) and stamps one
  `BATCH_BASE` naming the record. The host reads and empties the log at each drain and splices the record into the
  stream (`profile_mode::splice_batch_log`). The roles' clock is the timestamper's own counter, low word, so each time
  is placed exactly against the base's.
  - No role stamps live, since role and mover stamps arriving together lose events.
  - The first form stamped each time (`b1cb441`). On the 1.7B those bursts came out repeated and lost, so ~13k pairs a
    trace were mis-joined ([operating notes](../learnings/silicon-operating-notes.md#the-timestamper-repeats-and-drops-events-in-a-burst-2026-11-02-cards-0-and-1)).
  - A join that breaks the credit protocol's order is dropped by name (`in_protocol_order`), as is every other pair the
    host cannot tell, and the summary counts them.
- **Attribution** (`profile_mode::unit_pipes`). For each op, unit and batch:
  - ring wait, issue and landing (B);
  - starve (T0 waiting for data), unpack, kernel (T0 ready to T2 push), and output wait (T2 waiting for NC);
  - head (op start to the first reserve) and tail (the last push to the unit's arrival at the barrier).

  Two derived figures:
  - **Overlap:** the time a batch's reads are in flight while a kernel runs.
  - **The what-if:** the same batches perfectly overlapped, `r0 + Σ max(r_i, k_(i-1)) + k_last`, against the span from
    the first grant to the last push. This bounds a deeper ring before anything is built.
- **Outputs:**
  - `summary.md`'s pipeline table: each label's slowest unit, and its units' mean.
  - `pipeline.tsv`: each op's batches, one sample, every unit.

## Decision rules (fixed 2026-11-02, before any measurement)

The rules apply to the slowest units of `gate|up`, `down`, `qkv` and `o` in the 1.7B's `Step(64)`, BF16 weights,
card 1, each figure a share of the slowest unit's work. They are checked in this order, and the first that holds
names the next item.

1. **Ring-bound** (X10's ring depth). The what-if saves ≥ 15% of the product's span, *and* B waits for a slot (ring
   wait) ≥ 25% of the unit's work, with the next batch's reads not yet issued. Next: bound a two-batch ring for
   `Blocks` on the card before building it. Its L1 must leave the busy units' prefetch room: A5b lost it with
   four-column rings.
2. **Read-bound.** T0's starve ≥ 40% of the work while the ring wait is < 10%: the unit's reads (B's gather
   and landing) set its pace, the kernel idle. Next: X10's entries and reads in flight for that product, and item 5's reserve
   (X9's large entries, bank-affine placement), which is released only on this verdict.
3. **Reload-bound.** `down`'s continuation batches' kernels exceed its first batches' by ≥ 15% of its span, from
   `pipeline.tsv`'s batch index. Next: the `K`-chunk policy (fewer, longer chunks).
4. **Tail-bound.** The tail (the last push to the barrier: the kernels' wait, NC's writes) is ≥ 15% of the work.
   Next: the kernel tail and NC's scatter.
5. **None holds.** Record the split, close H11 as "no single stage of the pipeline binds", and go on to item 3.

A meter whose verdict two runs disagree on is reported as not reproducible and is not tuned.

## The gates (2026-11-02)

- Unit tests (`profile_mode`):
  - `a_pairs_credits_join_the_roles_batch_tables`, `a_batch_table_that_does_not_join_is_dropped` and
    `the_pipeline_detail_adds_a_base_to_each_wait`.
  - The first was watched failing with a batch's grant and landing swapped, and with the tables' base dropped.
- `step266_batch_tables` (ttsim, and card 0 with `--feature profile`):
  - T0 and T2 keep one entry a batch, in the protocol's order: a `K`-streamed `[1, 8192] @ [8192, 32]` and a
    `K`-whole `[1, 1024] @ [1024, 256]`, four batches each.
  - Its control, a later product with the tables off, leaves them word for word. It was watched failing with the
    roles keeping them unasked.
- `silicon_profile_mode::a_streamed_pairs_batches_keep_the_credit_protocols_order` (card 0, `1793635431`):
  - Every batch is told, none dropped: `down`'s `[1, 8192] @ [8192, 2048]` (64 units, 4 batches each, a ring of
    two) and `gate|up`'s `[1, 2048] @ [2048, 8192]` (64 units, 2 each, a ring of one).
  - Each batch keeps the protocol's order within the stamps' resolution (64 cycles across cores).
  - The meter tells the rings apart: 192 of 256 batches were granted before the batch ahead was popped in the
    ring of two, 0 in the ring of one.
  - The detail costs 160.9 against 160.6 us and 171.8 against 170.0 us a product under `Ops`.
  - Under `Ops`, no batches are told.
- What one product alone shows (the gate's table, F32 `B`, nothing else running): the slowest unit's gather is
  85-90% of its work and T0 starves 84-86%. The kernel is 8-10%, and the ring wait 0-1%: read-bound in
  isolation, at ~410-440 GB/s card-wide.

## Work

- [x] `tt-isa`: `trace_word::BATCHES`, `BATCH_TIMES` and its region in `l1::FIXED`, the events `BATCH_BASE` and
  `BATCH_TIME`.
- [x] Firmware: the roles' batch tables (`corpus.rs` `run_stream`); B's re-stamp after `ROLES_DONE` (`mover.rs`
  `await_roles`), which then clears the tables.
- [x] Host:
  - `Detail::Pipeline`, with its stamps in each op's window budget;
  - `Event::BatchBase` and `Event::BatchTime`;
  - the tables cleared when the mode starts;
  - `unit_pipes`, the summary table and `pipeline.tsv`;
  - `TT_PROFILE_DETAIL=pipeline` in `burn-tt`.
- [x] Gates (above):
  - unit tests of the attribution, each watched failing once;
  - ttsim, the roles' tables alone (ttsim refuses the event stream, divergence row 54; its clock is modelled);
  - card 0, `silicon_profile_mode`'s pipeline test. It checks that the meter shows the credit protocol's own order:
    a batch's landing before T0 sees it, and with a ring of `c` batches, batch `i + c`'s grant after batch `i`'s pop.
- [x] Measure on card 1, apply the rules, and record (2026-11-02, below):
  - the performance log;
  - H11 and X10 in the design reference;
  - roadmap item 2, with its next item named.

## The result (2026-11-02, card 1)

[Log](../learnings/performance-log.md#a-decode-products-pipeline-metered-from-inside-read-bound-by-its-reads-in-flight-2026-11-02-cards-0-and-1).

- Runs: `1793637811` and `1793638678` (safe image), which agree within 0.5 us a product; `1793639446` (`fast-reads`).
- Coverage: every op of every trace is covered, and on `Step(64)` 330,836 pairs are told and 1,800 dropped (0.5%).

| `Step(64)` product | slowest unit, us | ring | gather | T0 starves | kernel | tail | what-if saves |
|---|--:|--:|--:|--:|--:|--:|--:|
| `qkv` | 51.5 | 1% | 75% | 76% | 13% | 8% | 3% |
| `o` | 16.4 | 2% | 49% | 52% | 18% | 21% | 0% |
| `gate\|up` | 123.0 | 1% | 83% | 86% | 11% | 1% | 11% |
| `down` | 117.1 | 1% | 88% | 82% | 11% | 3% | 1% |

**Rule 2, read-bound, names the next item.**
- Rule 1 fails: the ring wait is 1-2% and the what-if at most 11%.
- `fast-reads` leaves the split within a point, so the entries' issue cost is not the bound. A unit's ~4-5 GB/s at
  three 2 KiB reads in flight is ~1.3 us a read under load: its bytes in flight.
- Next: roadmap item 2b, the `m = 1` product over a relaid weight with larger entries (H11's "then" step). The reserve
  of item 5 is released.
