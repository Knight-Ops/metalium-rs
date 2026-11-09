# Failed, killed and retired approaches

This is the register of approaches that were tried and dropped. It exists so they are not tried again by accident.
Each row says:
- what was tried and the measured result;
- why it was dropped;
- **retry only if**: the condition that would change the verdict;
- where the evidence is.

Most rows were measured under the bit-exact *reference* numeric contract (ADR 0005), on SmolLM2-1.7B batch-1 decode,
on one card. The 2026-11-09 review ([tt-metal-parity.md](../plans/tt-metal-parity.md)) changes several of those
conditions:
- a *performance* contract for inference (BF16 activations, BFP8 weights, HiFi2/LoFi; [ADR 0006](../adr/0006-two-numeric-contracts-reference-and-performance.md));
- batched decode;
- an 8B-class parity model.

Rows whose verdict may flip under those conditions are marked **(revisit)**. A revisit is a new hypothesis with its
own prediction and kill criterion in [dataflow-compiler.md](dataflow-compiler.md) §13, not a reopening of the old one.

Abbreviations:
- **PL**: [performance-log.md](performance-log.md)
- **DC**: [dataflow-compiler.md](dataflow-compiler.md)
- **R@2026-11-08**: the roadmap as it stood before the 2026-11-09 reset (`git show 3e4cefd:docs/plans/roadmap.md`)

How to add a row:
- When a hypothesis is killed or a knob retired, add the row here in the same commit.
- Delete the opt-in code unless the row says why it stays.
- An opt-in knob that lost is not kept alive "in case": the row here, plus the commit that removed it, is the record.

## Pattern: what the failures have in common

- **Schedule changes on a batch-1, bandwidth-bound step under a fixed numeric contract ran out.**
  - From 2026-10-28 to 11-08 about 100 commits moved the 1.7B decode step 11.18 -> 9.90 ms. About 1.55 ms of that came from three changes: weights laid by column, the next weights read before a unit's barrier arrival, and the even deal.
  - In the same window H4, H5, H10, H16, H17, H18 and both X9 forms were killed.
  - With the weights made free the step is 5.33 ms, and the BF16 weight floor is ~7.2 ms. The remaining headroom is under 1.4x and is spread across many small terms.
- **The work was not wasted.** On the same card and model, our bit-exact 9.90 ms step beats tt-metal at batch 1 (10.55 ms in its BFP4/BFP8 performance mode, 11.3 all-BF16; [performance-log.md](performance-log.md#tt-metal-on-the-same-card-smollm2-17b-and-its-products-2026-11-09-card-1)). It ran out of headroom, not of value. The gaps that remain are batching, prefill and the numeric contract.
- **The large wins changed what was moved, not how it was scheduled.**
  - BF16 weight storage: 16.80 -> 12.95 ms.
  - The position-major cache, packed `A` and prefetch: -20%.
  - The row-scatter cache write: -22%.
  - Weights laid by column: -8.5%.
- **"More units" stopped helping once reads were at the card's rate.** Balanced and even deals, more readers, and bank-affine placement all hit the same wall: GDDR's ~470-505 GB/s plus channels' unfair service.
- **Fusing a small op saves only its ~2.5-10 us of fixed cost.** On a 10 ms step a fusion pays only when it removes bytes, not just an op (H10, `TT_ADD_RMS_NORM`).

## Decode: weight layout, deal and placement

| Approach | Result | Why dropped | Retry only if | Evidence |
|---|---|---|---|---|
| X9 two-copy weight layout (`TT_WEIGHT_LAYOUT`, a relaid twin beside the stored weights) | 1.7B step 11.08 -> 9.90 ms, bit for bit, but +3.4 GB of GDDR on the 1.7B, growing with the model | Breaks "a default holds a weight's bytes once". Default for part of 2026-11-03, opt-in after, retired 2026-11-06 (`61a71b2`, B -1,140 bytes) | Never as a default. As a disaggregated-serving option (prefill and decode on different cards), rebuild from `61604b9` | PL [X9 bit for bit](performance-log.md#x9s-weight-layout-bit-for-bit-a-17b-decode-step-1108---990-ms-2026-11-03-cards-0-and-1), [back to opt-in](performance-log.md#x9s-weight-layout-back-to-opt-in-a-default-must-hold-the-weights-once-2026-11-03-cards-0-and-1) |
| X9 in one copy (C0: products of more rows read the relaid weight) | Eager products 1.34-3.38x slower, prefill 4-13% slower | Kill fired. C1 (stored weights laid by column) kept the decode gain in one copy | Not needed: C1 is the default | PL [one copy bounded](performance-log.md#one-copy-of-a-relaid-weight-bounded-the-kill-fires-2026-11-03-cards-0-and-1) |
| Weights stored transposed `[n, k]` | Every product slower (qkv 147.9 -> 157.0 us, head 993 -> 1,022 us) | Slower everywhere | No | PL [weights stored transposed](performance-log.md#packed-twins-a-released-barrier-and-weights-stored-transposed-2026-10-19-cards-0-and-1) |
| Deal rotated over channels | 27.90 -> 27.84 ms | The barrier skew was not channel locality | No | [decode-to-bandwidth.md](../completed-plans/decode-to-bandwidth.md) |
| Balanced deal (`TT_BALANCED_DEAL`) | 16.81 -> 16.86 ms; the head 830 -> 925 us | A unit's rate falls as readers on its channels grow. Retired in F4a.3 | No | PL [a balanced deal](performance-log.md#a-products-units-metered-and-a-balanced-deal-2026-10-23-card-0) |
| H18 / X10, a deal sized to each unit's measured rate; position weights | Slowest unit 1.094x against a 1.10x bar; position weights 1.02x, step slower | Killed: a unit's rate follows its load. (The plain even deal was then made the default by the owner: 10.14 -> 9.90 ms) | No | PL [X10 killed](performance-log.md#x10-a-deal-sized-to-each-units-rate-bounded-killed-h18-2026-11-07-cards-0-and-1) |
| H17, bank-affine placement at 16 KiB (2-3 readers a channel) | `down` 485 -> 504 GB/s (1.04x), `o` 1.06x; < 0.1 ms a step | Below the 1.15x / 0.2 ms kill | **(revisit)** for a batched step whose products are compute-heavier. tt-metal's DRAM-sharded matmul gains 1.6x from 2 readers a bank at M=32, but its compute sits on the reader cores (8-24 of them), which ours does not | PL [firmware diet, H17](performance-log.md#the-firmware-diet-bs-room-180---10356-bytes-at-the-same-speed-h16-and-h17-killed-2026-11-06-cards-0-and-1) |
| F27, bank-local reads | Even with reads 5x cheaper, step 10.82 -> 10.60 ms | A unit's read rate does not bind a decode product | As H17 | PL [F27 probed](performance-log.md#f27-probed-bank-local-reads-and-what-does-not-bind-a-decode-product-2026-10-27-card-0) |
| More requests at tile-sized entries (BF16 stream cap 3-8) | 120-unit products lose up to 23% | No gain | No | R@2026-11-08 2b |
| BFP4 weights for batch-1 decode, predicted >= 1.5x (F1) | 1.09x (11.25 -> 10.31 ms) | The bound was bytes in flight and unpack/math, not bytes | **(revisit)** under the performance contract: BF16 activations, LoFi and BFP `Src` change the unpack and math side that bound it | PL [one matmul for every format pair](performance-log.md#one-matmul-for-every-format-pair-and-bfp-weights-through-the-model-2026-10-28-cards-0-and-1) |
| More BF16 columns a job (`TT_BF16_PLANS=ring/wide`) | No shorter product | Opt-in, no gain | Delete the knob | R@2026-11-08 |

## Decode: prefetch, streams and overlap

| Approach | Result | Why dropped | Retry only if | Evidence |
|---|---|---|---|---|
| K-chain handoff between two units (`TT_STREAM_HANDOFF`) | 16.88 -> 17.07 ms | GDDR aggregate-bound. Retired in F4a.3 | No | PL [K chain handed](performance-log.md#a-columns-k-chain-handed-between-two-units-2026-10-22-card-0) |
| A second job's `B` in a second ring (`TT_PREFETCH_SECOND`, R2) and the trace arena (P1-P3) | 17.18 against 16.77 ms; arena unchanged at 16.78 | Room for `B` does not bind. Retired | No | PL [room for a second job's B](performance-log.md#room-for-a-second-jobs-b-the-trace-arena-and-second-jobs-2026-10-23-card-0) |
| Idle units' lists moved early (F4a.2 first form) and pumped early (`TT_IDLE_AHEAD_EARLY`) | 0.6 ms, then 0.24 / 0.17 ms slower | Reverted, then opt-in. Delete the knob | No | PL [F4a.1-F4a.3](performance-log.md#f4a1-f4a3-the-idle-lists-pumped-edge-credits-and-the-three-schedules-2026-10-30-card-1) |
| H4 / X6, edge credits replacing barriers (`TT_EDGE_CREDITS`) | +0.03 / +0.06 ms on the 1.7B; Qwen3.5 prefill 1.4% faster | H4's kill fired (< 0.3 ms) | **(revisit)** for batched decode and prefill, where barrier skew across 110+ units grows with work per op | PL [F4a.1-F4a.3](performance-log.md#f4a1-f4a3-the-idle-lists-pumped-edge-credits-and-the-three-schedules-2026-10-30-card-1), [edge credits on Qwen3.5](performance-log.md#the-merged-tree-and-edge-credits-on-qwen35-2026-10-31-card-1) |
| H5, spatial pipelining (F6.4) | Step 0.17-0.23 ms slower | Reads beside the slowest units slow them | No, at batch 1 | DC §13 H5; PL [F6.0](performance-log.md#f60-register-fusion-and-spatial-pipelining-bounded-on-the-card-2026-10-31-card-1) |
| Layer sub-traces (`TT_SUB_TRACES`) | Streams 5.6x smaller, step +0.15 ms | The body is fetched as often as the layer. Retired 2026-11-06 (`61604b9`, B -1,624, NC -704 bytes) | Only if trace size (not speed) blocks a model. Long-context 1.A is the better route | PL [layer sub-traces](performance-log.md#layer-sub-traces-stored-bodies-and-calls-2026-10-23-card-0); [layer-sub-traces.md](../completed-plans/layer-sub-traces.md) |
| Busy-unit prefetch, first form | Products 15.2 -> 11.6 ms, but the step only 0.2-0.4 ms | Small ops queued behind it. **Fixed and adopted** once NC stopped draining B's prefetch ID | — (kept: a failure that became a win) | PL [busy prefetch fixed](performance-log.md#busy-prefetch-held-back-by-a-shared-transaction-count-and-fixed-2026-10-21-card-0) |
| A2 sparse waits / A3 one launch | Bounded at 0.03-0.07 / ~0.12 ms | Not built | No | R@2026-11-08 |
| Splitting overflowed profile windows | No change | Reverted | No | R@2026-11-08 |

## Decode: small ops and fusion

| Approach | Result | Why dropped | Retry only if | Evidence |
|---|---|---|---|---|
| H10, register epilogues (a residual add into `Dst`, B13c) | Bound 0.10-0.18 ms against a 0.3 ms kill | Killed on the 1.7B | Under ADR 0001 the kill does not carry to other architectures: re-bound per model family, and in training, where the epilogue also removes a backward pass's bytes | PL [F6.0](performance-log.md#f60-register-fusion-and-spatial-pipelining-bounded-on-the-card-2026-10-31-card-1); DC §13 H10 |
| Residual add + RMSNorm as one op (`TT_ADD_RMS_NORM`) | 11.18 -> 11.26 ms | Slower: every unit needs the whole sum; a removed small op is worth ~2.5 us | Delete the knob | PL [RMSNorm's scale in its op](performance-log.md#the-tt-mnist-regression-found-and-rmsnorms-scale-in-its-op-2026-10-27-cards-0-and-1) |
| Following an operand's runs (`TT_FOLLOW_RUNS`) | 11.63 ms | The scale on half the units cost what the barriers saved | Delete the knob | R@2026-11-08 |
| RMS statistic reading rows thin | Read 19.15 -> 21.01 us; step unchanged | A row costs per read, not per byte | No | R@2026-11-08 |
| RMSNorm that also divided (first form) | 640 us a norm | Too slow. Replaced by the one-op RMSNorm | No | R@2026-11-08 |
| Residency rows multicast to every reader | 43.9 ms a step | Contended multicast serializes. Rows held by 2 readers instead | **(revisit)** as operand multicast for products, which is a different pattern (one sender, a rectangle of receivers, credit-gated, tt-metal's in0/in1 mcast). This row killed *every producer multicasting to all readers*, not that | PL [planned residency applied](performance-log.md#planned-residency-applied-replicated-rows-by-multicast-then-sharded-only-2026-10-20-card-0) |
| Softmax as one op a tile row (`TT_SOFTMAX_ROWS=1`) | Long prefill 15.24 -> 19.82 s | Slower | Delete the knob | R@2026-11-08 |

## Attention: order-changing levers (parked, not failed)

| Approach | Result | Why parked | Retry only if | Evidence |
|---|---|---|---|---|
| Flash-decode probe (a head's positions in two halves) | 19.7 us a layer at 512 positions; +0.27 ms at short steps | Below the stop rule, and it changes the sum's order | **(revisit)** under the performance contract and at 8K+ contexts and batch 32, where tt-metal's SDPA decode uses it by default | PL [flash-decode's probe](performance-log.md#flash-decodes-probe-a-heads-positions-in-two-halves-2026-10-23-card-0); [split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md) |
| Split-K | ~0.2 ms estimated | Changes order; parked | **(revisit)** under the performance contract, for products with K >> N (8B `down`, K = 14,336) | [split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md) |

## Prefill: operand reuse, blocks and fidelity

| Approach | Result | Why dropped | Retry only if | Evidence |
|---|---|---|---|---|
| D5: `B` pulled too (both operands from holders) | Reads 205-273 MB, but 15.18 TFLOPS against `A`-only pull's 25.98 | The holder's L1 fan-out binds | **(revisit)** as true 2-D multicast (one sender writes a rectangle; no fan-out of pulls) | PL [B pulled too](performance-log.md#f4b1-b-pulled-too-the-reads-clause-met-and-the-rate-lost-2026-10-29-cards-0-and-1) |
| A holder's credits as one run (`CREDIT_RUN`) | Within noise | Retired from firmware | No | PL [credits as one run](performance-log.md#f4b1-a-holders-credits-as-one-run-no-gain-2026-10-29-cards-0-and-1) |
| Pull in batched attention products | 0.4-2.0% (kill < 3%) | Kill fired | No | PL [F4b's leftovers](performance-log.md#f4bs-leftovers-the-batched-pull-bounded-the-eager-slowdown-located-2026-10-31-card-0) |
| Measured blocks for eager products | Eager training 1.5-2.9x slower | Eager runs `Auto`'s block | No | DC §13 |
| Static FP32 pull | ResNet-18 9.7% slower | Never pull unmeasured | No | DC §13 |
| Re-weighed planner model | Lost 14-22% on some shapes | Reverted; replaced by measured blocks | No | PL [larger GDDR blocks](performance-log.md#f4b1-larger-gddr-blocks-the-planners-block-not-reads-binds-a-packed-b-2026-10-29-card-1) |
| Direct `UnpackToDst` reload beside running `MVMUL`s (F5.B1) | Wrong results on silicon | Hardware hazard | Never | PL [F5](performance-log.md#f5-dst-subblocks-and-half-sync-2026-10-30-card-0); `c52094e` |
| P7, opt-in LoFi / HiFi2 for prefill | With an F32 `B`, LoFi no faster (14.57 against 14.95 TFLOPS); with a BF16 `B`, 1.18-1.35x | Not built: unpack of F32 `B` bound it, and it gave up bits | **(revisit, first)** under the performance contract: BF16/BFP operands remove the F32 unpack bound. tt-metal reaches 240 TFLOPS BF16-HiFi2 and 470 BFP8-LoFi on p150 | [prefill-operand-reuse.md](../plans/prefill-operand-reuse.md) P7 |
| Derived fidelity (fewest phases keeping every edge bit) | HiFi4's bits until a product overflows FP32 | Opt-in contract | Fold into the performance contract's fidelity rule | DC §4; `TT_DERIVED_FIDELITY` |
| Packer L1 accumulation as a default | — | Reorders the sum (G3) | **(revisit)** under the performance contract (tt-metal's default) | DC §11 |

## Firmware

| Approach | Result | Why dropped | Retry only if | Evidence |
|---|---|---|---|---|
| `opt-level = 2` for the movers (V-series 2026-10-13 said "free") | 2026-11-06 (H16b): ResNet-18 1.9% and prefill 1.2% slower | October's equal speed no longer holds | No | PL [firmware diet](performance-log.md#the-firmware-diet-bs-room-180---10356-bytes-at-the-same-speed-h16-and-h17-killed-2026-11-06-cards-0-and-1) |
| `opt-level = s / z` | Mover 1.5-1.7x slower per entry, roles' push ~7x | Too slow | No | [firmware-architecture-review.md](../proposals/firmware-architecture-review.md) |
| H16a, a hot window in B (`SELECT_ROWS` and padding fill out of `run_record`) | Within noise, 632 bytes larger | Reverted | No | PL firmware diet |
| V1a, refusing the other direction's ops before decode | -4 bytes on B, +96 on NC | LLVM already shared the arms | No | [firmware-architecture-review.md](../proposals/firmware-architecture-review.md) |
| Writing only the NIU registers that change | +21 cycles an entry | The comparison costs more than the writes | No | [performance-scoreboard.md](performance-scoreboard.md) "Per-request cost" |
| X6, the mover's NIU fast path | 3-4% an entry | Entry cost is software, not MMIO. Opt-in | Delete if unused | PL X6 |
| `.ttinsn` fused push, parked 2026-10-10 to 10-25 | 2.77x on push-bound kernels | Parked until its trigger fired, then **shipped** | — (kept: a deferral, not a failure) | [ttinsn-push.md](../completed-plans/ttinsn-push.md) |

## Model-scale ideas rejected on arithmetic

| Approach | Result | Why dropped | Retry only if | Evidence |
|---|---|---|---|---|
| Weights resident in L1 (Cerebras-style) | ~63 cards in F32 for the 1.7B | Rejected on arithmetic | Only for models under ~150 MB across a mesh | [sram-residency-and-multi-card-decode.md](../proposals/sram-residency-and-multi-card-decode.md) |
| Prefill and decode on different cards using X9 | — | Parked use withdrawn with X9 | With disaggregated serving (after batched decode) | R@2026-11-08 2b |

## Knobs to delete

These opt-in switches belong to rows above and are still read by the code as of 2026-11-09. Each should be removed
with its code path ([tt-metal-parity.md](../plans/tt-metal-parity.md) W6, the sunset rule):
- `TT_ADD_RMS_NORM`
- `TT_FOLLOW_RUNS`
- `TT_SOFTMAX_ROWS`
- `TT_BF16_PLANS`
- `TT_IDLE_AHEAD_EARLY`
- `TT_STREAM_HANDOFF` and `TT_PREFETCH_SECOND`, if any reader of them remains
- `TT_TRACE_ARENA`
- any `TT_PROBE_*` hook whose gate is gone

Check each with `grep -rn` before deleting. `TT_EDGE_CREDITS` and `TT_OPERAND_REUSE=pull` stay until their revisits
are decided.
