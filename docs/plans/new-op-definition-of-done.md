# New op: definition of done

What a new kernel or op must have before it is done. This is the standing contract that
`AGENTS.md` points at ("Follow the coverage definition of done"). It was extracted from the
Phase 10 tracker, now archived as [hardware-coverage.md](../completed-plans/hardware-coverage.md);
item ids cited below and in source comments (X6, S6, F0, ...) refer to that file. Open work is
in [roadmap.md](roadmap.md); the generated Burn status table is
[burn-op-coverage.md](burn-op-coverage.md), with per-method notes in
[burn-op-notes.md](burn-op-notes.md).

How to get from a failing Burn conformance test to an op that meets this contract, step by step, is
[conformance-to-feature.md](conformance-to-feature.md).

The two-gate rule, applied per op. An item is `[x]` only when every line below holds for
it, and a line that does not apply says why in the item.

1. **A `tt-isa` helper**, typed so that the hazards are in the API (the Cross-cutting rules in [the Phase 0-10 checklist](../completed-plans/phases-0-10-checklist.md#cross-cutting)),
   with unit tests of its encoding.
2. **An oracle that is not an epsilon** (the [Tolerance policy](../completed-plans/phases-0-10-checklist.md#tolerance-policy)):
   - *bit-exact* wherever the hardware's arithmetic is documented: `numerics::fma_bh` for
     anything `SFPMAD`-shaped, and a port of the page's functional model for `SFPLUT`,
     `SFPLUTFP32`, `SFPARECIP`, `SFPEXEXP`/`SFPSETEXP`/`SFPEXMAN`, `SFPSTOCHRND`,
     `SFPCAST` -- each port checked against the page's pseudocode the way `fma_oracle`
     checks `fma_bh` against `fma.c`;
   - for an *approximation* (a polynomial, a Newton step, a range reduction), the device
     result is still predicted bit for bit by running the same instruction sequence through
     the ported models, **and** its distance from the true function is held to a bound
     *derived* in a comment next to the gate (polynomial remainder, Newton's quadratic
     convergence, the reduction's error), never a guessed number.
3. **A ttsim gate** in `crates/tt-tests/tests/stepNN_*.rs`, inside `fork_scope`,
   watched failing at least once (an empty kernel, a wrong constant, a swapped operand).
4. **A silicon gate**, the same test through `cargo xtask silicon`. Ordinary
   validation uses one card (card 0): the same test on the other card gives no new
   information. Mesh or multi-card ops, and models too large for one card, use both.
5. **Burn routing**: the method overridden in `burn-tt/src/ops.rs` and listed in
   `OVERRIDDEN` (then `cargo xtask gen-burn-ops` and its `--check` mode), agreeing with `burn-flex` bit for
   bit or within the item's derived bound; a residency check that the op downloads
   nothing (`tensor_traffic`, `TT_TRACE_FALLBACK=1` silent); and an entry in the silicon
   smoke tier (`xtask/src/silicon.rs`, `SMOKE`), where burn-tt is compared with
   burn-flex on the card.
6. **Findings logged**: every ttsim refusal or ttsim/silicon disagreement met on the way
   is a row in [`ttsim-divergence.md`](../learnings/ttsim-divergence.md), and every measured fact a ["Measured, not quoted"](../learnings/ttsim-divergence.md#measured-not-quoted)
   entry.
7. **Padding declared** (F0 in [the archived tracker](../completed-plans/hardware-coverage.md), from [`tt-metal-concepts-review.md`](../learnings/tt-metal-concepts-review.md) G1): the op states the pad
   it needs from each input and the pad it leaves (`OpPadding`), and a ragged-shape gate
   chains it into an accumulation.

An op that works only on silicon (ttsim refuses an instruction it needs) can be ticked
with the simulator line `[-]` and the divergence row cited, as `DOTPV` is today.
