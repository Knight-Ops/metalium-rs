# The fixed cost of a traced replay

Roadmap Track 2, item 6, next step 1 ("Profile the fixed ~33 ms a replay"). Written 2026-10-15 against `b22a245`.
Status: **P0 and P2 done (2026-10-15): the traced decode step is at burn-flex's time, 26.46 ms against 26.42 in the same run (was 37.2). P1 waits (its share is ~1.5 ms).**

## Why this item

A traced SmolLM2-135M decode step replays in **37.1 ms against burn-flex's 26.6** (card 1, every tile, default spread,
run `1792029144`). A traced prefill chunk costs about **33 ms + 0.18 ms a token** (`1792031467`). The fixed ~33 ms is
shared by the decode step and every prefill chunk, and it is what keeps decode behind burn-flex. A replay has no host op
building, so this is device time (plus the input write and the logits read). Nothing has broken it down: none of the
existing instruments reaches inside a replay (below).

Targets: decode at or below burn-flex's 26.6 ms on every tile, and a lower intercept for the chunk curve. Every replay
stays bit for bit the eager step (`step185_fixed_decode`).

## What a replay is made of (read from the code, not measured)

- **Op count.** About 58 server jobs a layer (7 of them matmuls), so about 1,800 a step with the head, the prologue and
  the 60 `copy_into`s of the cache write-back (`crates/tt-models/src/llama.rs`, `Layer::forward`;
  `crates/burn-tt/src/trace.rs`, `capture_with_state`). 33 ms over 1,800 ops is about 18 us an op.
- **Barriers.** Every captured op ends in a barrier that **all units** reach, whatever the op ran on
  (`Session::enqueue_work_inner`, `arriving`): about 120 on every tile, each polling one counter on unit 0's tile
  (`crates/tt-firmware/src/mover.rs`, `barrier`). Eager execution has only the units an op ran on arrive, folded into
  their last list (`step176_barrier_participants`, 2026-10-14). A decode op mostly runs on 6 to 48 units: an m = 1 matmul
  is one job an output tile.
- **The m = 1 matmuls.** 211 a step; each job runs its K tiles as one serial chain (18 tiles, 48 for `down`).
- **The cache write.** Per layer, `expand` copies of the position mask and of the new key and value, then two whole-cache
  `mask_where`s: op count rather than bytes (a cache tensor is 48 KiB at 64 positions).

## What the instruments see in a replay

| Instrument | In a replay |
|---|---|
| `Session::host_times`, `dataflow_stats` | nothing: counted when ops are enqueued, so only at capture |
| `report::part`, `server_report` | the whole replay is one job "(outside an op)" |
| `OpRecord` (`Session::trace_ops`) | per-unit stream ranges of each captured op, but no Burn op name |
| timestamper (`profile_start`, `TT_PROFILE`) | stamps a replay, but its buffer holds 1,024 events (`tt_isa::mailbox::TRACE_BUFFER_BYTES`), about 40 ops |
| `silicon_probe_trace_replay::replay_breakdown` | unit 0's mover intervals by entry class, on toy traces |
| `TracedInference::run_timed` | write / replay / read, but the SmolLM2 benches call `run` |

## Phases

### P0. Measure (no behaviour change)

- [x] **Host split and counts.** The SmolLM2 benches report `run_timed`'s write / replay / read, and the trace's shape:
  ops, barriers, and units an op runs on, from `TracedInference::ops` (the captured `OpRecord`s).
- [x] **Name the ops.** `OpRecord::label` is the Burn op (or part) whose server job enqueued it
  (`Session::set_op_label`, set by `server::timed_job`). The 30 layers are identical, so an op's ordinal within its
  layer gives the stage.
- [x] **Per-op cost against unit count** (synthetic, silicon): captured chains of tiny ops on 1 and on every unit, at 1,
  8, 32 and 120 open tiles; replay time over the op count.
- [x] **Device breakdown of the real replay** on unit 0, mover stamps only, mapped back to labelled `OpRecord`s:
  device time by op class and by layer.

### P1. Barriers in a capture on the units an op ran on (waits: P0 put its share at ~1.5 ms a step)

- [ ] A captured op's barrier counts only the units it ran on; a unit that sat out ops waits (a wait-only `BARRIER`
  entry in its stream) before its next op, as eager execution's catch-up does. A replay advances the barrier counter by
  the arrivals the capture counted, and ends with every unit's arrival so `sync` still means every unit is done.
- [ ] Gate `step186_trace_barrier_participants` (ttsim, then card 0), with a negative control on ttsim only.

### P2. What the P0 table showed, largest first

- [x] **A matmul refills padding only along `K`** (`tensor::matmul_padding_off_k`, `Session::matmul_dram`). Padding past
  `M` (of `op(A)`) or `N` (of `op(B)`) reaches only the output's padding, so an operand whose `K` is whole tiles is used as
  it is and the output's padding is declared undefined when the operand's was. Gate `step24_padding`,
  `padding_off_k_is_used_as_it_is` (dirty `A` past `M`, dirty `B` past `N`, both; no refill queued; Flex bit for bit; the
  output then refilled by a matmul over it as `K`). Watched failing: the rule off (a refill queued), and the rule applied
  along `K` too (`dirty_padding_never_reaches_an_accumulation` and the view test fail).
- [x] **One zero row per width, kept** (`Session::zero_row`): a row gather's tail rows read it, and it was a `metadata`
  write each time (attention's two a layer).
- [x] **A buffer read transposed as a matmul's `B` is transposed once** (`burn-tt` `DramRef::plain_for_matmul`): on its
  second such use a block copy makes it plain and it is kept beside the buffer, unless the buffer is written in place or a
  training step is being captured. Gate `step186_kept_transpose` (the same bits on every use and against an explicit
  plain `B`; the copy queued once; a trace taking the weight as input replays new weights; watched failing with the
  written-in-place check removed). The output head: 2,329 -> ~200 us.
- [x] **The model's own cache write and RoPE** (`tt-models` `llama.rs`, the arithmetic unchanged): the position mask is
  expanded to the cache's shape once a step, not once a layer; `sin` holds `rotate_half`'s sign, so an application is
  `cat(x2, x1) * [-s, s]` instead of `cat(-x2, x1) * [s, s]` (the same bits), without its `neg`. `step185` (ttsim, card 0
  at 30 layers, run `1792036037`) and the HuggingFace host tests pass.
- [x] **A tensor mostly of padding downloads untilized by the card** (`Session::dma_download`): rows filling at most a
  quarter of their tile rows come back as rows, on a unit per 96 tiles. Gate `step58_tile_layout` (now with `[2, 3100]`;
  card 0, run `1792036250`). The logits read: 0.95 -> 0.47 ms.
- [ ] Left, by share (tracked as [roadmap](roadmap.md) Track 2 item 6, next step 6): the cache write's `k` and `v` expands and two `mask_where`s (~70 us a layer; a native row assign at
  a resident index would replace them), the RMSNorm's broadcast `div` (27 us, twice a layer), the replay's start (~170 us:
  the mover takes a `CALL` only as a list of its own, so each unit's catch-up is a second enqueue), and each op's ~3.5 us
  of control.

## Results

### P0 (2026-10-15, card 1 for the model, card 0 for the probe; release, every tile, default spread)

**Where the 37.2 ms goes** (`smollm2_decode_traced`, run `1792034179`, now with `run_timed` and the trace's shape):
input write 0.07 ms, **replay 36.0 ms**, logits read 1.0 ms. The trace is **2,418 ops**, each ending in a barrier every
one of the 120 units reaches; an op runs on a median of 6 units (p10 1, p90 18).

**The barrier is not the cost** (`silicon_probe_trace_replay::replay_per_op_cost`, run `1792034180`). A captured chain
of 16 and 64 ops, replay time over op count:

| Op | 1 tile | 8 tiles | 32 tiles | 120 tiles |
|---|--:|--:|--:|--:|
| `relu` of `[32, 64]` on one unit | 6.2 us | 6.7 | 6.7 | 6.8 |
| `relu`, a tile on every unit | 5.3 | 6.1 | 6.3 | 7.8 |
| `[1, 576] @ [576, 576]` (18 jobs) | 123.3 | 51.0 | 19.2 | 21.0 |

An op on one unit of 120 costs 0.6 us more than on a one-tile session: the barrier with 119 idle units arriving.
P1 (barriers on the units an op ran on) would save at most about that a op, ~1.5 ms a step, so it waits.

**Op by op** (`smollm2_replay_profile`, one layer, run `1792034647`: unit 0's mover stamps in a 128 KiB buffer at the top of
the program cache, `Session::set_profile_buffer`; each op's span runs from one of unit 0's `BARRIER` ends to the next).
The spans sum to 3,846 us: **1,123 us a layer**, the output head 2,329 us, the replay's start 174 us (the first op's barrier
waiting for the 120 `CALL`s to be enqueued), the final norm 98 us. 30 layers of that is 36.4 ms against the 36.0 ms
measured. A layer, by stage:

| Stage | us | What dominates |
|---|--:|---|
| 7 projections | 295 | a `fill-pad list` before each, 16 us: the `[1, 576]` input's 31 padding rows refilled every time |
| attention | 285 | two `resident metadata` writes of a zero row, 34 us each; `qk`/`pv` gathers and fill-pads |
| 2 RMSNorms | 173 | 7 ops each; the broadcast `div`, 27 us |
| cache write | 149 | 5 ops to expand the position mask, 3 to expand `k` and `v`, 2 `mask_where` |
| RoPE | 114 | 7 ops an application, two of them a `neg` of a slice (a repack and the negation) |
| MLP rest, residuals, `repeat_kv` | 107 | |

The head is one matmul whose `B`, the tied embedding table read transposed, goes through the mover's `READ_TRANSPOSED`:
each tile into its scratch slot, then transposed in software, about 10 us a tile (against 0.6 us for an untransposed one).

A tiny op costs 5.7 us on unit 0 (`mul_scalar`: 0.5 barrier, 0.8 kernel, 0.9 data; the other 3.5 us is the op's
control: descriptor pokes, the `PAIR_CALL`'s launch and credits).

### P2 (2026-10-15, card 1, release, every tile, default spread, burn-flex in the same run)

Each row adds to the one before.

| Change | Ops a step | Replay | Logits read | Step | burn-flex | Run |
|---|--:|--:|--:|--:|--:|---|
| P0 baseline | 2,418 | 36.0 ms | 1.00 | 37.2 ms | 26.6 | `1792034179` |
| matmul padding off `K` | 2,207 | 32.6 | 0.96 | 33.8 | 25.9 | `1792035075` |
| kept zero row, kept transpose | 2,147 | 28.7 | 1.19 | 30.0 | 26.4 | `1792035466` |
| mask expanded once a step, RoPE's sign in its table | 1,911 | 25.8 | 0.99 | 27.0 | 26.3 | `1792035579` |
| thin downloads untilized by the card, a unit per 96 tiles | 1,911 | 25.8 | **0.47** | **26.46** | 26.42 | `1792036613` |

The one-layer profile after the first three (`1792035570`): 1,643 us, the head about 200 us. The download's unit count,
swept (card 1, the same step): 4 units 0.66 ms, 8 0.55, 16 0.46, 24 0.49, 32 0.51, 48 0.60, 120 0.95; with the tiles
downloaded and detilized by the host instead, 4 units 0.68, 16 0.77, 120 0.99 (`1792036386`-`1792036493`).
