# Layer sub-traces: one stored layer, called once a layer

Status (2026-10-23): **Phases 0-4 done; opt-in (`TT_SUB_TRACES=1`, `Session::set_sub_traces`), not the default.**
A trace whose ops repeat stores one layer body per unit and a `CALL_BODY` a layer. Replays are bit for bit on ttsim
and on card 0, and every stored layout is emulated on the host and compared with the capture before it is written.
- **Storage:** the 1.7B decode trace's layers take 5.6x fewer entries and prefill's 8.4x (not the ~16x estimated).
  Each call's argument table is most of what is left. In GDDR, the 1.7B's trace streams go 675 -> 384 MB: `Step`
  19.3 -> 4.9 MB, `Chunk(64)` 107 -> 21, `Chunk(128)` 175 -> 28; `Chunk(256)` (316 MB) makes no body.
- **Speed:** **a 1.7B step is 0.15 ms slower** (16.77 / 18.19 -> 16.94 / 18.33 ms, runs `1792734860`, `1792735010`),
  and the 135M 0.05-0.08 ms; ResNet-18 and ViT-Tiny are unchanged. A body read from GDDR is fetched as often as the
  layer was, and each call adds its table's read and a re-read of its chunk.
- **Residency (Phase 5) is not built.** The fetches it would take off the critical path were measured at 0.23 ms
  a step (planned-l1-residency 5a), so it is worth at most that, which this plan's first measurement
  already put in doubt.

History: proposal 2026-10-20; its cheaper first step (the next reader section read ahead,
[planned-l1-residency.md](planned-l1-residency.md) 5a) done 2026-10-21; built 2026-10-23 (roadmap Track 1, "Decode at
GDDR's bandwidth", and Track 2, "Trace segmentation").

## What a trace is

- **Storage.** Each unit's stream is stored in GDDR as its own region: the stream, plus the reader and writer section
  of every `PAIR_CALL` (`Session::store_stream`, `trace::chunked`).
- **Replay.** A replay is one `CALL` a unit (`dm::op::CALL`): the mover reads the stream into `OUTER_CHUNK` 64 entries
  at a time, waits for each chunk (`call_chunks`), runs it, and fetches each `PAIR_CALL`'s sections the same way
  (the next reader section's first chunk read ahead, 5a).
- **Relative values.** Generations and barrier targets are stored relative, and `CALL` adds the replay's bases. A
  reader section takes the generation base and no barrier base; NC runs a writer section with neither.

Each layer of a decoder is captured again in full, so a trace held 24 copies of one layer's entries.

## What was measured before building (1.7B, every tile, `session::probe_space_report`)

2026-10-20 (runs `1792466481`, `1792467415`): a `Step` trace ~635 K entries, `Chunk(64)` ~3.3 M; the ops repeat every
16, 365 of 388 ops equal to the op 16 later; of their words 17.8% differ, 86% of those tensor references, then plain
GDDR addresses (5%), barrier targets (4%) and generations (5%). Every L1 address, program and count is the same layer
to layer.

2026-10-23, the space report's `SUBTRACE` lines (`subtrace::report`, card 0, `1792727515`), what each unit would store:

| Trace (planned capture) | Layer entries | Bodies | Tables | Kept as captured | Ratio | Largest table |
|---|--:|--:|--:|--:|--:|--:|
| decode `Step(64)` | 528,218 | 22,966 | 52,888 | 18,846 | **5.6x** | 34 entries |
| prefill `Chunk(64)` | 2,979,236 | 129,532 | 224,204 | 0 | **8.4x** | 90 entries |

- **Who makes a body:** 104 of 120 units; the other 16 run none of the repeated ops (their layers are empty).
- **How many layers are called:** a unit calls 22 or 23 of the 23 repeats; layer 0 differs (first descriptor pokes,
  no prefetch before it).
- **The tables:** a tensor reference costs two entries, so the tables are most of what is stored. Slots that hold the
  same tensor in every layer share one argument: 2.8x -> 5.6x on the decode trace.
- **The plan pass** (the first, unplanned capture) gives 2.8x and 1.1x: its tables are larger (up to 116 and 123
  entries, the table holds 126).
- **The self-check:** 0 mismatches.

## Design, as built

- **Detection** (`tt_kernels::subtrace`, host only, generic): the period from `residency::period` over
  `residency::signatures`; each unit's repeat `i` is its stream from op `start + i·p`'s first entry to the next
  repeat's (gaps included, so a busy unit's moved prefetch list is the layer's it sits in), with its pairs' sections.
  - **The body** is the first layer equal to the next but for tensor references (two entries each, `record::references`),
    plain moves' channel, port and offset, and the generations (stream and reader sections) and barrier targets (stream)
    moving by one offset each.
  - **Any other differing word** keeps that layer as captured. So does a layer with no entries (`NoBody::Empty`).
  - **Arguments:** each slot that differs in a called layer becomes `[ARG_REF, k, 0..]` (its second entry zero) or a
    move's word 6 `ARG_WORD | k` (channel `u32::MAX`, refused if left). Slots taking the same values in every called
    layer share `k`.
- **Storage** (`Session::store_sub_stream`): one region a unit:
  - the unit's own stream: what precedes the layers, a `CALL_BODY` for each called layer and each other layer as
    captured, then what follows;
  - its pairs' sections;
  - the body's stream and its sections;
  - each call's table.

  Each part starts on a 64-byte boundary. `chunked` puts each `CALL_BODY` at an odd entry (a `WAIT` before it if
  needed), so the rest of its chunk is read again from a 64-byte-aligned offset.
- **The checks before storing:**
  - `subtrace::check`: every call expands to its layer, word for word.
  - `subtrace::emulate`: the mover's walk of the stored bytes (chunks on the stream's grid, the body and the re-read,
    bases, arguments in the stream and in both sections), with chunk padding dropped. It must equal the captured
    stream with its sections inlined, or the unit is stored as captured with the reason on stderr.
  - The emulation refuses what the mover refuses (a body of no entries, a table past `ARG_TABLE_ENTRIES`). It found
    the one silicon bug: below.
- **The mover** (`op::CALL_BODY` = 0x4B, `dm::ARG_REF` = 0x4C, `dm::ARG_WORD`):
  - **The entry:** `[CALL_BODY, channel, offset, count, generation_offset, barrier_offset, table_offset, table_entries]`,
    only in a `CALL`'s own stream, never in a body, a section or a list. It is taken by `call_chunks`' patch loop,
    which runs the chunk up to it.
  - **The run:** `call_body` reads the table into `dataflow::ARG_TABLE` (0x12040, 126 entries, the free L1 between NC's
    stub and `OUTER_CHUNK`) and runs the body's chunks as a call's, with its offsets added to the call's bases, then
    the chunk is read again from the entry after.
  - **Arguments:** taken in the patch loop (`take_arguments`, cold, outside `exec` and `Entry::decode`). B's reader
    sections take them too. NC takes them in the writer section from the same table: `pair_call` passes
    `ARG_WORD | entries` in NC's `L1_ADDR`.
  - **Refusals:** past the table, `RANGE`; an unsubstituted reference, `TensorRef::decode`'s `RANGE`; a `CALL_BODY` of
    no entries or with too large a table, `LENGTH`; one anywhere else, `OP`.
  - **Image cost:** B 33,668 -> 35,284 bytes (safe image), NC 18,324 -> 19,020.
- **Fixed on the way:** the encoded reads' trace domain was a toggle (`flip_trace_domain`), so a call inside a call (a
  replay's reader section) ran in the list domain. Now set and restored (`enter_trace_domain`); harmless before, since
  no trace holds an encoded read (`upload_form` encodes after capture).
- **Unchanged:** `OpRecord::entries` are the captured stream's, and the profile mode's per-op stamps, unit barriers and
  bytes are counted on it before storing. The firmware runs the same entries, so windows count through calls
  (`silicon_profile_mode` passes with sub-traces on).

## Corrections to the proposal

- B has ~7-9 KiB free in its 42 KiB slot (now ~1.6 KiB less), not ~18 KiB.
- `MAILBOXES` is 40 KiB with at most 3 KiB free in one piece: no room for a body there.
- Traces hold no `READ_ENCODED`: the "24 checksums" risk does not arise.
- Plain addresses cannot become `READ_RUN`/`WRITE_RUN` (tile slots only): a word marker instead.
- 16x was the bodies alone: the tables of tensor references bring it to 5.6x (decode) and 8.4x (prefill).

## The silicon bug the gates missed

The first card run (`step187` SmolLM2, 30 layers, every tile) stalled at a release-mode barrier on every unit.
- **Bisected** by capture, then by unit, then dumped (`TT_STALL_DUMP=1`, `DataMover::dump_stall`): unit 119 runs none
  of the repeated ops.
- **The cause:** its layers were empty, so it made an empty body with 29 calls. The mover refuses a `CALL_BODY` of no
  entries (`LENGTH`), so the unit never reached its tail's barrier arrival.
- **Why the checks missed it:** the self-check and the first emulation accepted an empty body. Both now refuse it
  (`NoBody::Empty`; the emulation refuses what the mover does). The ttsim gates use one or two tiles, where every unit
  takes part.

## Phases and gates

- [x] **0. Measure** (`subtrace::report` in the space report; table above).
- [x] **1. Host: detection, body and tables, expansion, self-check.** `tt-kernels` unit tests `subtrace::tests`:
  - a body and calls round trip;
  - a call given another layer's table, a corrupted argument and a short table are caught;
  - counters that do not move together and other words keep a layer;
  - the body is the first layer equal to the next;
  - writer sections take no generation;
  - empty layers make no body.
- [x] **2. The mover:** `CALL_BODY`, arguments on B and NC, the domain fix. Gate `step223_sub_traces` (ttsim and card 0;
  `SMOKE`):
  - a chain of six distinct layers on two tiles, every unit calling 5 of 6 layers, GDDR 16,640 B against 24,960
    plain, three replays bit for bit against the ops run fresh;
  - **negative control:** each call given the next call's table (`probe_sub_trace_tables(ProbeTables::Rotate)`)
    differs on 3 of 3 replays;
  - a table one entry short is refused with `RANGE` (on ttsim first).

  `step172`, `step173`, `step215`, `step39`, `step174`, `step223` on ttsim in the safe, `fast-reads` and `fast-reads`
  + `encoded-guard` images. Firmware clippy plain, `encoded-guard`, `profile`, `fast-reads`, `ttinsn-experiment`.
- [x] **3. Store and replay (opt-in):**
  - ttsim with `TT_SUB_TRACES=1`: `step174` (invariants now count a body and its tables), `step187`, `step193`,
    `step205`, `step216`, `step219`, `step222`, `step39`, `step60`;
  - card 0 with `TT_SUB_TRACES=1`: 64 tests, `step185`, `step187`, `step188` (1.7B, traced against eager, bit for bit),
    `step191`, `step193`, `step205`, `step216`, `step219`, `step222`, `step223` and the mover gates (`1792732559`,
    with the probes' fix `1792733080`);
  - `silicon_profile_mode` and `step23` with the `profile` feature (`1792733236`);
  - `--smoke` with the defaults, 1,195 of 1,195 (`1792733253`).
- [x] **4. Measured** (card 0, every tile, same build): see the status and
  [the performance log](../learnings/performance-log.md#layer-sub-traces-stored-bodies-and-calls-2026-10-23-card-0).
  **Not the default:** the step is slower.
- [ ] **5. The body resident in L1** (deferred). It would remove a layer's chunk and section fetches (~0.23 ms a step,
  5a's measure), less the table's read. It needs:
  - ~7 KB a unit reserved from the capture planner (`CAPTURE_ONLY` or `PROGRAMS`);
  - copy-then-patch, since bases and arguments differ per layer.

  Worth building only with what else would make it pay: tables in L1 too, or arguments read by the records rather
  than patched (no per-call table read), or once trace storage, not the step, binds (trace segmentation, prefill
  buckets past 256).
- [ ] **`Chunk(256)`'s bodies:** no unit made one (316 MB stored as captured); report why (most likely tables past
  126 entries) and decide whether the table, or the chunk, should grow.
- [ ] **Smaller tables** (if storage matters): a reference argument as its bases entry alone where its first entry is
  the same in every layer (about half the tables).
