# Writing a traced cache in place

Status: **planned, not started** (scheduled 2026-10-16; was `docs/proposals/in-place-cache-write.md`). It follows
the cache write of one row a head (roadmap Track 2 item 6, step 6;
[performance log](../learnings/performance-log.md#a-cache-write-of-one-row-a-head-2026-10-16-cards-0-and-1)). The
phases and checklist are at the end ("Plan"); the design above them is the proposal's, unchanged. Read the current
code before treating a name here as current.

**Baseline now** (after the decode matmul work, `dbe71bb`/`6fcb943`): SmolLM2-1.7B decode **47.90 / 50.37 / 53.53
ms** (run `1792161623`, fused projections). The cache write is **5.02 ms a step** in profile `1792162257`
(`Step(64)`, 47.46 ms), unchanged by that work:

| Part of the 48 writes a step | ms |
|---|--:|
| the row scatter (`resident row scatter`) | 1.29 |
| the scatter's tile copy of the cache (`copy list`) | 1.12 |
| the trace's write-back (`copy_into list`) | 1.13 |
| the value's repack (`native repack`, once a layer) | 0.92 |
| a block copy (`block copy list`) | 0.56 |

## What it would remove

A traced decode step writes each layer's keys and values with `scatter_nd(.., Assign)`, which burn-tt runs as
`Session::scatter_rows`: a tile copy of the cache, then the row scatter into the copy. The trace then copies the
result back over the state buffer (`TracedInference::capture_with_state`, one `copy_into` a state tensor). So each
write moves the whole cache twice to change one row a head.

SmolLM2-1.7B, `Step(64)`, run `1792110380` (56.8 ms a replay):

| Part of the 48 writes a step | ms |
|---|--:|
| the scatter's tile copy of the cache | 1.13 |
| the trace's write-back (`copy_into`) | 1.13 |
| the row scatter itself | 1.30 |

Writing the rows into the state buffer itself would remove the first two: **~2.3 ms a 1.7B step (4%)**, and more
for a longer cache or more key/value heads, since both copies cost the cache's size. The value's repack and block
copy after RoPE (1.47 ms) are a separate matter (its layout), not part of this.

## Why it is not free

Burn's tensors are values: `scatter_nd` returns a new tensor and leaves its input as it was. Writing in place breaks
that for every handle that still names the old cache. Today nothing reads one after the write (`Layer::forward`
writes, then attention reads only `attended(new)`, and the step returns the new tensors as state). The hazards are
what a future model, or burn-tt's own bookkeeping, could do:

1. **A model that reads the old cache after writing it** (attending to the old cache and the new token separately,
   or keeping the old cache to roll back, as speculative decoding does). It would read a row from the future,
   silently. `FixedCache` is meant for any decoder, so this cannot be ruled out by reading today's models.
2. **What burn-tt derives from a buffer and keeps.** `DramRef::plain_for_matmul` keeps the plain transpose of a
   buffer read transposed as a matmul's `B`, and `TtTensor::derived` keeps relayouts. Both are dropped only for a
   buffer marked `server::written_in_place`. Today a state buffer is marked when the write-back is recorded, and
   each step's cache is a fresh buffer, so nothing stale can be reused. Written in place mid-step, a copy kept
   from before (the eager warm-up, say) would be read as if it were current.
3. **A view taken before the write.** A slice or reshape of the old cache shares its buffer, so it would see the
   new rows.
4. **The negative controls.** `step185`'s `stateful = false` and `step187`'s private prefill rely on state changing
   only through the write-back. They survive this design (below), but each must be rechecked, not assumed.

What is **not** a hazard: other traces over the same state (`SharedStateTraces`: the prefill chunks and the step
lengths). They read the state buffer at their own replay, after this one finished, exactly as now. A handle to
the state buffer already sees new contents after every replay today; only reads **between** the write and the
end of the same step are new exposure.

## How to make it safe

The rule: **in place only for state a capture has been told it owns, and any read of the superseded contents is an
error by name, never a wrong answer.** Eager execution is unchanged.

### 1. Donation, declared before the step runs

`capture_with_state` learns its `(new, state)` pairs only from what `f` returns, after the step's ops are queued:
too late to write in place. The state must be named first.

- `TracedInference::capture_with_state` (and `SharedStateTraces::capture`, which already holds the state from
  `SharedStateTraces::new`) take the state tensors before calling `f`, and mark each buffer **donated** for the
  capture. A capture with no state (`step185`'s `stateful = false`) donates nothing.
- A donated buffer must own its slots, untransposed and not a row view: the checks `capture_with_state` already
  makes of state, made up front.
- Donation ends with the capture, by a guard (`Capturing` is the model), so an error inside `f` cannot leave a
  buffer donated.

### 2. Kept copies dropped at donation, not at the write-back

- `server::written_in_place(id)` is called for each donated buffer **when it is donated**, as `capture_with_state`
  already does for its inputs. `plain_for_matmul` and `derived` check that flag on every use, so a transpose or
  relayout kept from the warm-up is dropped before the step's first op (hazard 2).

### 3. Generations: a superseded handle fails by name

A handle cannot be found and updated: the old cache is shared by `FixedCache.state`, by `cache_k.clone()` in the
layer, and by views made from either (each a new `TtTensor` cell over the same buffer). So the buffer, not the
cell, carries the check:

- Each buffer (`tensor::Buffer`) has a **generation**, and every `TtTensor` over it records the generation it was
  made at.
- An in-place write bumps the buffer's generation; the tensor it returns records the new one. A view made later
  (the `reshape` back to `[1, kv, L, d]`, `attended`'s slice) records the new generation too.
- Reading a tensor (`dram()`, `to_dram()`) whose generation is older than its buffer's is an error naming the op
  and the buffer: "read of a cache written in place in this capture". Hazards 1 and 3 become a failed capture, not
  a wrong token.
- **At the end of the capture**, each donated buffer's state handle is brought to the current generation. The state
  handles denote "the state as the last replay left it", which is what they mean today after each `copy_into`.

### 4. The write itself

- burn-tt's `scatter_rows_native` writes in place when its table is a donated buffer at its current generation,
  owned and untransposed: `Session::scatter_rows_into` (already there, gated by `step189`'s in-place test) and a
  result over the same buffer, generation bumped. Anything else, and every eager call, keeps the copy.
- Two writes to one state buffer in a step chain: the second's table is the first's result, current generation.
- The write-back: when `new` and `state` are one buffer, `capture_with_state` records no `copy_into`. It still
  marks the buffer written (2).

### What does not change

- Prefill chunks still write through `mask_where` (a padded chunk may run past the cache), so they produce a new
  tensor and keep their write-back. A step and a chunk trace can share one state either way.
- The arithmetic: a row scatter moves raw words, in place or into a copy. Every traced result stays bit for bit the
  eager fixed step and the growing cache.

## Gates

On ttsim, then card 0; each watched failing.

- **Equal results:** `step185`, `step187` (SmolLM2, the synthetic Llamas, the GPT-2 decoder, one driver serving many
  prompts) and `step188` on silicon, bit for bit as now.
- **The copies are gone:** a donated step's trace has no `copy_into` for its state and no tile copy before its
  scatters (counted from `TracedInference::ops` labels). Watched failing with donation off.
- **A read of the old cache fails by name:** a capture whose step reads its pre-write cache after the write is
  refused with the error above, and so is one that reads a view of the cache taken before the write. Watched
  failing (a wrong token instead) with the generation check removed.
- **No stale kept copy:** a state buffer read transposed as a matmul's `B` twice in an eager warm-up (so a plain
  transpose is kept), then a donated capture that writes it and reads it transposed: the result is the eager
  step's. Watched failing with the donation-time `written_in_place` removed.
- **Eager is unchanged:** `step189`'s claim that `scatter_nd` never writes its input holds outside a capture.
- **Errors end donation:** a capture whose `f` fails leaves no buffer donated; a later eager scatter on it copies.
- **The negative controls still diverge:** `step185` with `stateful = false` (nothing donated), `step187`'s private
  prefill (its chunks keep the functional path).

## Cost to measure

`TT_GEN_MODEL=smollm2-1.7b-instruct` `traced_generation` on card 1, with and without `--profile`. The baseline is
decode 47.90 / 50.37 / 53.53 ms (run `1792161623`) and profile `1792162257`; the proposal was written against 57.21 /
59.64 / 62.81 (`1792109477`, profile `1792110380`). The 135M bench checks for no regression (22.91 / 23.35 / 23.98,
`1792162031`). The expected gain is the two copies' ~2.25 ms a 1.7B step; anything well short of it means a copy
survived.

## Plan

Code facts to start from (checked 2026-10-16):
- **The copy:** `Session::scatter_rows` (`crates/tt-kernels/src/session.rs`, ~4955) is `self.copy(table)` then
  `scatter_rows_into(&out, ..)`. `scatter_rows_into` is the in-place primitive, already gated by `step189`'s in-place
  test, which also refuses a view.
- **The routing:** `burn-tt/src/ops/index.rs` `scatter_nd` -> `scatter_rows_native` -> `server::scatter_rows` ->
  `DramBuffers::scatter_rows` (`server.rs`, ~1274).
- **The write-back:** `TracedInference::capture_with_state` (`crates/burn-tt/src/trace.rs`, ~221-332) runs `f` under
  the `Capturing` guard. Only after `f` returns does it check each `(new, state)` pair and record
  `server::copy_into(device, new, state)` (`server.rs`, ~3632; it also marks `dst` `written_in_place`).
- **Shared state:** `SharedStateTraces::capture` (~501-546) pairs the returned new state with its own state, so it
  learns the state only after `f`. That is why donation must move up front.
- **The callers:** `FixedCache`'s step write (`crates/tt-models/src/kv_cache.rs`, `FixedWrite::write`, `Masked`)
  does `reshape -> scatter_nd(Assign) -> reshape`. Chunks keep `mask_where`.

Phases, each on ttsim then card 0, each gate watched failing:

- [ ] **D1. Donation up front.** `capture_with_state` and `SharedStateTraces::capture` take the state before `f`. They
  check it (owned, untransposed, not a row view), mark it donated under a guard that ends with the capture (also on
  error), and call `written_in_place` for each donated buffer at donation.
  - **Gates:** errors end donation; no stale kept transpose (the proposal's gate list).
  - **No behaviour change yet:** the write still copies.
- [ ] **D2. Buffer generations.** `tensor::Buffer` gets a generation, each `TtTensor` records its own, and `dram()` /
  `to_dram()` refuse a stale one by name. Views made after a write take the new generation, and capture end brings
  each state handle current.
  - **Gate:** a read of the old cache, and of a view taken before the write, fails by name. Watched failing (a wrong
    token) with the check removed.
  - Costs one compare a read; measure that the 135M step does not move.
- [ ] **D3. The write in place.** `scatter_rows_native` calls `Session::scatter_rows_into` when the table is donated,
  at its current generation, owned and untransposed; every other call copies, as now. When `new` and `state` are one
  buffer, `capture_with_state` records no `copy_into`.
  - **Gates:**
    - equal results (`step185`, `step187`, `step188` bit for bit);
    - the copies are gone (no `copy list` before a donated scatter and no state `copy_into`, from
      `TracedInference::ops` labels);
    - eager unchanged (`step189`);
    - two writes to one buffer in a step chain;
    - the negative controls still diverge (`step185` `stateful = false`, `step187` private prefill).
- [ ] **D4. Measure** as above. Rows go into the performance log and scoreboard.
- [ ] **D5 (separate, optional): the value's layout** (0.92 + 0.56 ms). **Now the key's too** (2026-10-16): since
  RoPE runs in the projection's layout ([decode-small-ops.md](decode-small-ops.md) S1), the key reaches the write
  as a view and repacks there as well (+0.9 ms a 1.7B step, profile `1792171171`). This is decode-small-ops S1d.
  - **Why it repacks:** `v` is a view of parts of rows (a column block of the fused `qkv` product, then
    `reshape`/`swap_dims` into heads, `Layer::forward`), so the write's `new.reshape([kv, d])` and `scatter_nd`'s
    restack (`ops/index.rs` `restack`) fall to an element repack (`mapped_native`).
  - **The block copy:** which op makes the 48 block copies is not pinned down: `plain_dram`'s `copy_blocks`
    (`ops.rs`) or a region view's materialisation. Read the `TracedInference::ops` labels first.
  - **The fix to try:** a `SCATTER_ROWS` that reads its value rows from a strided view (the record already names a
    tensor), or `FixedWrite` taking the value before the head split.

Documentation on completion: the roadmap (Track 2 item 6, step 6), the performance log and scoreboard,
`docs/README.md`, and this file moved to `completed-plans/`.
