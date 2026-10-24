# Order-changing decode: split-K and flash-decode

Status (2026-10-23): **flash-decode probed and stopped at P0; split-K parked.** The user chose flash-decode variant A
(the default past a measured gain). Its probe, each head's positions as two halves on 64 units with no exchange, saves
**19.7 us a layer at `Step(512)`** in a 1.7B step (18.19 -> 17.71 ms, `1792788575` / `1792788725`) and costs 0.27 ms at
`Step(64)`/`Step(128)`. That is before any exchange, below the ≥ 20 us stop rule below
([log](../learnings/performance-log.md#flash-decodes-probe-a-heads-positions-in-two-halves-2026-10-23-card-0)). Found
while planning: the split can be **bit for bit** by handing run 1's pre-fold sum (`sfpu::reduce::continuation_roles`)
and partial `pv` chain (`matmul::matmul_items_reloading`) to run 2, at five one-way exchanges instead of three, with a
new in-op peer wait (none exists: every cross-unit order today is op, barrier, op). Revisit for contexts past 512
positions, where today's job does not fit L1 and attention runs as the composition; halves fit to 1024.

Original status: **a proposal for a decision, nothing built.** Roadmap Track 1, "Decode at GDDR's bandwidth";
[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Rules: "split-K and split-position attention change
accumulation order. They come back for a decision with measured numbers and are not built unasked." This is that
document. It replaces the one-line decision stubs in
[decode-matmul-bandwidth.md](../plans/decode-matmul-bandwidth.md) P4 (split-K) and
[decode-attention.md](../plans/decode-attention.md) A5 (flash-decode).

**Recommendation:** build **flash-decode, variant A** (below), opt-in and used only past an attended length chosen by
measurement. It is worth ~0.7-1.0 ms of a 1.7B step at a 256-512-position context, and grows with the context.
**Park split-K:** the evidence puts it at ~0.2 ms of the step, at the cost of `down`'s bits.

## Why now

Phase 8's bit-for-bit levers are spent (decode-to-bandwidth, "A product's longest unit"). A 1.7B F32 step is
**16.81 / 18.21 ms** at 32 / 256-token prompts (card 0, `1792763914`), its weights at ~430 GB/s over a replay (92% of
the 469 GB/s measured ceiling). Room for `B` (R2), the stored layer (sub-traces) and the balanced deal were all neutral
or slower. Every remaining lever either halves the bytes (BF16 storage, a separate item) or changes an order.

## What changes, and what does not

- **What an order change gives up:** today's decode, bit for bit. `step185`, `step187`, `step188`, `step191` and
  `step193` compare against the eager schedule, and those claims survive (eager and traced would run the same
  split op), but the results would no longer equal today's.
- **What must not change:** determinism. The split must be a function of the shapes alone (heads, attended length,
  `K`), never of which units are idle in a capture. Otherwise the bits would depend on the prefetch's choices, the
  anchor and the deal. The same split on every card and run gives the same bits.
- **How it is checked:** as the numerics rules require, not by tolerance guessing. A host oracle predicts the device's
  result exactly in the new order (the measured Matrix Unit rule for the products, the SFPU interpreter for `exp`
  and the reductions), and the gate requires equality with it. A bound derived beside the gate covers only the
  difference from the unsplit result and burn-flex.

## Where the time is

`Step(64)`, `Step(128)` and `Step(512)` traces of one profile (1.7B, card 0, `1792762439`), per layer, medians.

| | `Step(64)` | `Step(128)` | `Step(512)` |
|---|--:|--:|--:|
| A replay | 16.25 ms | 16.38 | 17.72 |
| Attention (`attention/rows`, 32 units) | **28.2 us**, 1.33 MB | 36.8 us, 2.66 MB | **99.3 us**, 10.65 MB |
| `down` (64 units) | 172.0 us | ~172 | ~172 |

- **The context's cost is attention.** `Step(64)` to `Step(512)` adds 1.47 ms a replay: attention 1.70 ms more, the
  products 0.19 ms less. A 256-token prompt decodes at positions 256-320, which runs `Step(512)`: that is the 1.4 ms between the
  two timed prompts.
- **Attention at 512 positions is a slow read:** 10.65 MB a layer on 32 units (one head a unit, 333 KB each) in 99 us,
  ~3.4 GB/s a unit and 107 GB/s together. It runs while the idle units read `o`'s 17 MB ahead.
- **`down`:** 46.9 MB in its span over 172 us; at 469 GB/s that is 100 us. Its 24 busy units each read a whole 1.06 MB
  column (decode-to-bandwidth, "`down` prefetched").

## Flash-decode: attention's positions split across units

### Today

`tt_kernels::attention` ([`attention.rs`](../../crates/tt-kernels/src/attention.rs)): one job a key/value head on
one unit, seven kernels in L1, bit for bit burn-tt's composition:
1. `qk = Q K^T`;
2. the scale and mask;
3. the row maximum (clamped);
4. `SUB_EXP`;
5. the row sum (clamped);
6. `DIV`;
7. `pv = P V`.

The softmax normalizes **before** `pv`.

### The split

Each head's attended positions are cut into `s` runs on whole tiles (32 positions), one run a unit. A run's keys and
values are `1/s` of the head's. A combine then joins the runs. Two variants:

- **Variant A: exact maximum, normalize first (recommended).**
  1. Each run computes `qk`, the scale and mask, and its run's maximum. The maxima are exchanged: a maximum is exact
     in any order, so the head's maximum is today's.
  2. Each run computes `SUB_EXP` against the head's maximum: **every `exp` is today's, bit for bit**. Each run then
     computes its run's sum, and the sums are exchanged.
  3. Each run divides by the head's sum, computes its run's `pv`, and the partial outputs are summed.

  **Order changes, only two:**
  - the softmax denominator `l = sum of the runs' sums`: positive terms, reordered;
  - the output `o = sum of the runs' partial pv`: the Matrix Unit's `K` chain over positions is cut into `s` chains
    and added once.

  `P / l` differs from today's only through `l`. The groups of eight inside each `MVMUL` are unchanged, because runs
  are whole tiles. **Cost:** two exchanges between a head's units inside one op, plus the final add.
- **Variant B: one pass (online softmax, the standard flash-decode).** Each run keeps its own maximum `m_i`, `exp(s -
  m_i)`, its sum `l_i` and an unnormalized `o_i`. The combine rescales by `exp(m_i - m)` and divides at the end.
  - **Cost:** one exchange.
  - **Changes:** every `exp` (taken against `m_i`, then multiplied by a second `exp`), the place of the division (after
    `pv`, not before), and both sums. Its bound has to compose the SFPU `exp`'s approximation error, and `SFPMAD` is not
    an IEEE fused multiply-add.
  - **When:** only if A's second exchange measures too dear.

### Which units

- **Not the prefetching units.** During attention the idle units (64-103) read `o`'s 17 MB ahead, and taking them
  would move that read into `o`'s span.
- **Two runs a head on units 0-63** (32 heads x 2) keeps them free, and doubles attention's readers.
- **`s` is a function of the attended length only,** e.g. 1 below a threshold and 2 above, set by measurement; at
  `Step(64)` attention is mostly fixed cost and is not worth splitting.

### What it is worth

These are estimates from the numbers above, to be replaced by the probe's.
- **The floor:** at 512 positions, attention's 10.65 MB and `o`'s 17 MB read ahead share the window between
  RoPE and `o`. 27.6 MB at 469 GB/s takes **~59 us** a layer, against attention's 99 us today.
- **Two runs a head:** ~166 KB a unit at ~5 GB/s is ~33 us. Add ~20 us of fixed cost (attention at `Step(64)` reads 1/8
  of the bytes and still takes 28 us) and two exchanges at ~8-10 us each (a `[1, 2048]` row through a holder's L1,
  planned-l1-residency 2f) for variant A. That is **~55-65 us** a layer.
- **The step:** ~35-45 us a layer x 24 = **~0.8-1.1 ms at `Step(512)`**, ~0.2-0.3 ms at `Step(128)`, ~0 at `Step(64)`.
  It grows with the context (a 2,048-position cache reads 4x the bytes a layer), and it is the same work batched decode
  needs for a long sequence ([batched-decode.md](batched-decode.md), "Order change").

### What to build, in order

1. **A probe, no order change:** today's job with `pv` and `qk` restricted to a run (`attention::probe_stages`-style).
   It times a run's reads and kernels on units 0-63, and one exchange through L1 (NC's peer writes,
   `step218`). If the run plus two exchanges does not beat 99 us by ≥ 20 us a layer, stop and record it.
2. **The host oracle:** the split order, exactly. The measured Matrix Unit rule (silicon operating notes, "the Matrix
   Unit's group adder": groups of eight aligned to `2^(max - 10)`, half-up, then `Dst` adds to nearest with ties toward
   +infinity) and the SFPU interpreter for the maximum, `SUB_EXP`, the sums and `DIV`.
3. **The derived bound** beside the gate, against the unsplit result:
   - `l` is a sum of non-negative terms in two orders;
   - `o` is a `K` chain cut and joined by one `Dst` add.
   Both come from the rules above, not from IEEE round-to-nearest-even.
4. **The op:** `Session::attention_rows` taking runs; the exchanges as L1 writes into a head's other units, as the
   residency path writes; and a capture-time choice of `s` from the attended length.
5. **The gates** ([new-op-definition-of-done.md](../plans/new-op-definition-of-done.md)):
   - device equal to the oracle bit for bit, on ttsim and card 0;
   - against the unsplit op within the derived bound;
   - against burn-flex at `head_dim` 32, 64 and 128, MHA and GQA, attended lengths on and off multiples of 32 (as
     decode-attention's gates);
   - specials: masked positions (`-inf`), **a run entirely masked** (its maximum `-inf`, its sum 0: no `-inf - -inf`
     NaN may reach the combine), NaN in a key, scores large enough that `exp` would overflow without the maximum;
   - **negative controls, watched failing:** a run dropped from the combine; a run's own maximum used in variant A's
     `SUB_EXP`;
   - `step185`, `step187`, `step188` still traced against eager, bit for bit, with the split on.
6. **Measure:** `traced_generation_stream_ceiling` and `traced_generation` (1.7B, 135M), on and off, then `--profile`.
   It stays opt-in until the user decides on the default.

## Split-K: a product's `K` cut across units

### What it would do

A product with fewer output columns than units runs each column's `K` as `s` partial chains on `s` units, then adds
the partials. In a 1.7B step only `down` qualifies: 64 columns, `K` = 8192, 24 units reading whole columns. `qkv` and
`o` are read ahead whole, and `gate|up` and the head have more columns than units.

### The evidence against it now

- **The bit-for-bit version was built and measured:** the handoff (decode-to-bandwidth, "A column's `K` chain handed
  between two units", `1792686180`, profile `1792686773`). It is the same parallel read, with the partial chained
  instead of added.
  - `down`'s first op 179 -> 155 us, plus a second op of 17 us: 172 in all, and the step slower (16.88 -> 17.07 ms).
  - The halves took 93-137 us each, not ~88, because the other halves' reads ran beside them. Inside `down`'s span
    the bound is GDDR's aggregate rate.
  - `qkv` started later (51 -> 72 us a layer), because its units were the spare ones.
- **Split-K would remove only the second op's chain:** ~155 us plus one add of two `[1, 2048]` partials (~9 us, a
  `float_add`), ~164 against 172. That is **~8 us a layer, ~0.2 ms a step**, before `qkv`'s later start takes some of
  it back.
- **The balanced deal** (`1792764360`) showed the same: more readers make each unit slower, and the span's aggregate
  rate does not move.

### What it would cost

`down`'s output bits. Today each column is one `Dst` chain over 256 `K` tiles in order. Split, it is `s` chains, each
from zero, joined by an F32 add. The bound is the reordering of those per-tile sums under the `Dst` rule (to nearest,
ties toward +infinity) plus the joining add's rounding. The groups of eight are unchanged, because the cut is at whole
tiles.

### When to revisit

- **A model whose `K` does not fit a `K`-whole block:** an 8B's `hidden` of 4096. Such products stream with no weight
  prefetch (decode-to-bandwidth Phase 4, "A product's block from what it runs"), and two `K`-whole halves could both
  be read ahead.
- **After BF16 weights:** half the bytes a column shifts the balance between a unit's read and its compute.
- **If a probe shows GDDR idle in `down`'s span** while its whole-column units read.

## Decisions asked of the user

1. **May decode change order at all?** Options:
   - opt-in only, never the default;
   - the default past a measured gain;
   - not at all, which leaves BF16 storage as the next lever.
2. **Flash-decode variant:** A (two exchanges, every `exp` today's) or B (one exchange, more changed).
   Recommended: A, then B only if A's exchanges measure too dear.
3. **Split-K:** park (recommended), or build `down`'s for the ~0.2 ms.
4. **Order relative to BF16 weight storage** (half the bytes, F32's floor ~14.3 ms -> ~7 ms of weights). BF16 is the
   larger lever and changes the weights' values rather than an order. Flash-decode composes with it.
