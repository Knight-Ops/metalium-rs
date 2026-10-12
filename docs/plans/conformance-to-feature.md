# From a failing conformance test to a documented, passing feature

The standing procedure for closing an entry of
[`expected-failures.toml`](../../crates/burn-tt-conformance/expected-failures.toml). Burn's own backend
tests say what a Burn backend must do; each failure is either a gap in `burn-tt`, a limit we chose, or
a defect in the test. This document says how to tell which, how to fix a gap so that it stays fixed,
and what to write down so the next person does not rediscover it.

It extends [new-op-definition-of-done.md](new-op-definition-of-done.md): that file says what an op
must have to count as done; this one says how to get from a red test to there, in order. The first round
(2026-10-11: seven findings closed, after a dtype fix that every test needed; see the table at the end) is where
every step below was exercised, and the pitfalls are the ones met.

**The one rule behind all of it:** a test going green is the *start* of the evidence, not the end. The
conformance test is upstream's, written against a different backend; it says the behaviour is right, not
that *this implementation* is, on the card, without the host, for the shapes users will bring. Every fix
therefore gets a gate of ours that proves the claim more strongly than the test does, and that is watched
failing before it is trusted.

## The loop

```
 run ──► read ──► cluster ──► decide ──► reproduce ──► root cause ──► design
  ▲                                                                       │
  │                                                                       ▼
 shrink the list ◄── document ◄── silicon ◄── re-run suite ◄── gate (watched failing) ◄── fix
```

Each arrow below is a step with a command and a done-when.

## 1. Run and read

```bash
cargo xtask conformance --release            # the whole suite, ~4.5 minutes on 12 cores
cargo xtask conformance --release --filter softmax --filter ctc    # a slice, seconds
```

Release is the canonical profile (the list is recorded against it; the simulator runs the same
programs faster). A run prints, per category, how many failed as expected, then every **unexpected**
result. Read all four outcomes, not only red:

| Outcome | Means | Do |
|---|---|---|
| unexpected failure | a regression, or a test the list has never seen | triage it (step 2) |
| **unexpected pass** | a fix (yours, or a neighbour's) closed it | remove it from the list (step 8); look at *why* it passes |
| stale entry | the test is gone | delete the entry |
| `todo` | listed but never explained | explain it, or fix it |

The outputs of every test, with burn-tt's `TT_REPORT=1` residency table (which ops ran on the device,
which on the host, what was downloaded), are in `target/conformance/<stamp>/out/`.

**Done when:** you can name every failing test's group.

## 2. Cluster, then order

Tests fail in clusters: one missing capability breaks dozens. Group by the first line of the panic
(`unsupported operation <op>` names the op; `Tensors are not approx eq` is numerics; a bare timeout
is a hang or slowness). Then **order by dependency**, because fixes expose what was behind them: making
`softmax` take any rank turned 10 failing tests into 6 passing and 4 failing for a *different* reason (an
int/bool `reshape` the CTC loss also needed), which would have been invisible while the first one was
open. Fix the one other failures wait on; re-run; look again.

Then **order by value**: what the model ladder needs ([roadmap.md](roadmap.md), Track 1) before what only
a test uses.

## 3. Decide what kind of failure it is

| Kind | Looks like | Verdict |
|---|---|---|
| **Gap** | `unsupported operation X`, for a case Burn defines and a model can bring (a rank, a broadcast, an empty tensor) | Fix it (steps 4-9) |
| **Sharp edge** | an opaque message, a panic on the server thread, a failure that surfaces only at read-back | Fix it, *and* make the failure explicit: a typed error or a refusal that names the operation and the input ([sharp-edges rule](../../AGENTS.md)) |
| **By design** | refused for a technical reason or a recorded decision (F64: not a Tensix dtype; a PyTorch-only limit Burn does not impose is *not* one) | Keep it listed with the reason and the doc that records the decision. Do not "fix" a decision in passing; but a *deferral* ("not planned until a model needs it") is not a design: it is an open item, and a missing capability like `conv3d` (or `interpolate`, before round two closed it) belongs under *Gap* |
| **Precision** | `not approx eq` | Find the mechanism first (step 4). Never loosen the vendored tolerance; if the device cannot meet it, list it with the mechanism and whether a bound was derived |
| **Slow** | timeout in release | Run it once with a long limit on silicon; if it passes there, it is a `timeout` entry ("ttsim does not finish"), not a bug |
| **Test defect** | the failure is in the test or its macro, not in burn-tt | List it with the diagnosis. Never patch the vendored tree (its digest is pinned) |

A decision that contradicts an earlier one needs a test: `conv_transpose`'s output-padding limit was a
deliberate PyTorch rule, so it was relaxed only after the relaxation was shown (against Flex, forward and
all three gradients, one and two dimensions) to be right, and the old gate still passes.

**Done when:** each group is *fix*, *keep (with reason)*, or *defer (roadmap item)*.

## 4. Reproduce small, find the root cause

- Run the single test: build once, then run the binary directly from the crate directory, e.g.
  `target/release/deps/tensor-<hash> <test path> --exact --nocapture` with `TT_REPORT=1`.
  A test's printed name is `<binary>::<test path>`; the binary is run with the part after the first `::`.
- A **hang** is a stack, not a guess. `ptrace` is restricted here, so run the binary *under* gdb
  (`gdb -batch -ex run ...`), send `SIGINT` to the inferior, and `thread apply all bt`. This is how the
  `#[might_panic]` hang was found to be upstream's macro self-deadlocking on its own lock, not burn-tt.
- A failure that **surfaces late** (a sticky error read at the end) is read with `try_into_data`, which
  returns it; `into_data` panics with a message that hides which op failed.
- **Probe the neighbours.** A throwaway test (delete it after) answers "is this part of my bug?".
  Writing the broadcast gate showed `powf(inf, subnormal)` is NaN on the device and `inf` in Flex; a
  same-shape probe proved it was not about broadcasting, so it became a separate finding and the gate
  stayed about the claim it makes.
- For a **numeric** miss, reproduce the mechanism by hand before naming it. `gelu`'s autodiff failure
  (1.4375 against 1.46281) is exactly the operand 40.4628 truncated to TF32, 40.4375: the unpacker
  truncates to ten mantissa bits, a step of 2^-5 between 32 and 64. Say so with the numbers, or say it is
  attributed and not decomposed.

**Done when:** you can state the cause in one sentence that a test could contradict.

## 5. Design the fix

Choose in this order; each is cheaper to maintain than the next.

1. **Compose existing device ops.** `softmax` for any rank moves the axis last, flattens to
   `[rows, len]`, runs the *same* composition `step32` already derived a bound for, and puts it back; no
   new kernel, and the old bound carries over with `n` the axis length.
2. **Widen a guard that was too narrow** after checking the operation really is indifferent to what the
   guard excludes. `int_reshape` and `bool_reshape` needed `reshaped_strided` to accept I32 and Bool: it
   moves stored words and never reads them.
3. **Reuse the sibling's path.** `float_mask_where` and `float_powf` broadcast by the device copies
   (`expanded`) the integer `mask_where` already used.
4. **Decline what the device cannot hold, early and by name.** An empty tensor is built on the host
   (`native_zeros` declines a zero extent, as `native_random` always did) instead of asking the device for
   a buffer with a zero dimension.
5. **A new error where a panic stood.** A matmul block too long for a circular buffer's page counters
   is "too large", not "malformed": `PlanError::TooManyPages`, mapped so the chunk search shrinks K. A
   panic on the server thread kills every later op on that device.
6. **A new kernel** only when nothing composes, and then it follows the full
   [definition of done](new-op-definition-of-done.md) (an ISA helper, an oracle, a simulator gate, a
   silicon gate).

Rules that hold throughout (from [AGENTS.md](../../AGENTS.md)):

- **No host fallback.** An unsupported case fails explicitly; it never computes on the host and returns.
  A *read* is allowed where the result's shape depends on the data, and it must be the smallest one:
  `argwhere` reads back one 4-byte count and nothing else, and the gate pins that.
- **No new knob** without a measured need. The design assumed an "eager upload" policy was needed so the
  suite would exercise the device; a measurement (`TT_REPORT=1`) showed it already did, and the knob was
  not built.
- **Preserve what is declared**: padding (`OpPadding`), residency, L1 requirements through
  `tt_kernels::l1::Requirements`.
- Prefer the **smallest diff that is correct**: every fix of the first round was 5 to 60 lines. A large
  one is a sign the design step was skipped.

**Done when:** the change fits in a sentence, and you can say what it deliberately does *not* handle.

## 6. The gate

A new file `crates/tt-tests/tests/stepNNN_<what>.rs` (next free number; the doc comment at the top says
what Burn's test found, the claim, the oracle and the negative control, in that order). It contains:

1. **An oracle that is not an epsilon.**
   - Data movement (`reshape`, `cat`, `mask_where`, `argwhere`): **bit for bit** equal to Flex's, over
     values that expose a misread word (`i32::MIN`, `i32::MAX`, `-1`; `-0.0`, NaN, infinities, a
     subnormal).
   - An approximation (`softmax`, `powf`): within a bound **derived** beside the gate from the kernels'
     own constants (`EXP_BOUND`, `LOG_BOUND`, `pow_bound`), never a guessed number.
   - Arithmetic with exact answers (a transposed convolution of small integers): exact.
2. **Residency, with numbers.** `tensor_traffic()` before and after: assert the exact uploads and
   downloads the design allows (zero; or one 4-byte count and two radix rows), and `computed_on_device()`
   on the result, which a cached host copy cannot fake.
3. **Shapes that find the edges**: ragged against the 32 x 32 tile *and* whole tiles; ranks 1 to 4 (or
   5); a unit extent; views (transposed, permuted, sliced) so the strided path and the copying path both
   run; and the *limit* itself, with the next size past it refused **by the operation's name**.
4. **Padding.** If the result feeds a reduction or a matmul, chain it into one over a ragged shape:
   the old tile's padding must not appear in the new place.
5. **A negative control**: the same comparison against a *wrong* answer (the wrong axis, a transposed
   mask, a reversed order) must fail, so the check can tell. A gate whose control cannot fail proves
   nothing.
6. **Watch it fail first.** Before trusting the gate, remove the fix (or mutate it) and run:
   it must fail, for the reason you expect. Then restore. Recipe:

   ```bash
   cp crates/burn-tt/src/ops.rs /tmp/ops.rs.bak
   # edit: disable the new path, or flip one operand
   cargo test -p tt-tests --test stepNNN_x --release     # must FAIL
   cp /tmp/ops.rs.bak crates/burn-tt/src/ops.rs
   cargo test -p tt-tests --test stepNNN_x --release     # must pass; check `git diff --stat`
   ```

   **Check that the mutation applied.** A scripted `str.replace` on code the formatter has since
   reflowed silently does nothing, and the gate "passes" with the fix removed. Assert the pattern is
   found, or look at the diff, every time.
7. **State only what is asserted.** A header that promises "nothing is downloaded" over a test that
   reads the results back is a false claim; delete the sentence or assert it.

Run it on the simulator (`cargo test -p tt-tests --test stepNNN_x --release`, inside `with_device`,
which forks), then on card 0 (step 8).

If the fix is in a library crate, add a **unit test beside it** for the unit's own contract (the
`TooManyPages` boundary: `MAX_CB_PAGES` pages is legal and merely has to fit, one more is its own
error), watched failing the same way.

**Done when:** the gate passes, fails without the fix, and its control rejects a wrong answer.

## 7. Re-run the suite

```bash
cargo xtask conformance --release --filter <the module> --filter <its neighbours>
```

Expect **unexpected passes**: the target tests, and usually others (the CTC losses and
`det_4d_rectangular_batch` came free with the `reshape` fix). Each one is evidence the fix is general.
Expect also **no new failures**; if there are any, the fix has a side effect, and that is a bug in the fix.

**Changing a declared contract** (`dtype_usage`, a limit, a refusal) needs more than your own gate: other
gates may pin the old value on purpose. Search for it before the broad run (`grep` the function and the
value), and run the broad run with `cargo test --workspace --exclude tt-tests --release --no-fail-fast`
(plain `cargo test` stops at the first failing binary and hides the rest) and the whole simulator suite
(`cargo test -p tt-tests --features e2e --release`). The `I32` fix passed `step144` and was still caught
by a second gate, `crates/burn-tt/tests/essentials.rs`, whose premise ("integer compute is missing") had
gone stale when the integer ops landed: update the premise and the name, not just the number.

**Numerical failures have a second profile.** `cargo xtask conformance --exact` runs the suite with matmuls
accumulating in exact F32 (`TT_EXACT_MATMUL=1`, `burn_tt::MatmulMode::Exact`) against
`expected-failures-exact.toml`. A `precision` entry that passes there is the Matrix Unit matmul, not a defect of
the op under test; note it in the entry's reason, and do not change the default's arithmetic for it. An entry
that still fails in the exact profile is a real finding. The exact profile is slower on ttsim, so some tests
become `timeout` entries there: give each a silicon verdict in exact mode (`TT_EXACT_MATMUL=1 cargo xtask silicon
--keep-going --timeout-secs 1500 --filter <module>`; the env reaches the spawned test processes).

Run the checks the change touches ([AGENTS.md](../../AGENTS.md)): `cargo fmt --all --check`,
`cargo clippy --workspace --all-targets -- -D warnings`, `cargo xtask check-no-sim-in-ship`,
`cargo xtask check-no-flex-in-backend`, `cargo xtask gen-burn-ops --check`,
`cargo xtask burn-coverage --check`; and, for routing or arithmetic, the training regression
(`cargo test -p tt-tests --features e2e --test step12_mnist`) and the model gates.

## 8. Silicon, on card 0

```bash
cargo xtask silicon --release --device 0 --filter stepNNN_x:: --filter <the conformance module>
```

One card: the same single-card test on the other adds nothing. Anything the simulator cannot host is
silicon-only, with a [`ttsim-divergence.md`](../learnings/ttsim-divergence.md) row. A test too slow for
the simulator may finish in seconds on the card (the 261,120-long matmul: 13 s on card 0, no end in sight
on ttsim): run it there once, and record the run and the verdict in the entry.

Then add the gate and the conformance modules that now pass to `SMOKE` (`xtask/src/silicon.rs`); a filter
that matches nothing fails the run, so a renamed test cannot go quiet.

## 9. Document, then shrink the list

In the same change:

| Where | What |
|---|---|
| `crates/burn-tt-conformance/expected-failures.toml` | `cargo xtask conformance --release --update` removes entries that now pass; **re-read every surviving reason**, because a group whose cause changed is now wrong (the `log_softmax` group became "too slow", the `reshape` group became "slow" and "precision") |
| [burn-op-notes.md](burn-op-notes.md) | the op's row: what is native, what is refused, the gate and the card-0 run |
| [burn-op-coverage.md](burn-op-coverage.md) | `cargo xtask burn-coverage` (generated). A hand-written method that always refuses is counted native: give it a `[-]` disposition in `xtask/src/burn_coverage.rs`, or implement it |
| [roadmap.md](roadmap.md) | tick the finding; add what the fix exposed |
| [burn-backend-parity.md](burn-backend-parity.md) | the B-item the finding belongs to |
| [`learnings/`](../learnings/) | a measured hardware fact in `silicon-operating-notes.md`; a simulator quirk in `ttsim-divergence.md`; a timing in `performance-log.md` |
| the code | a doc comment saying what the function does for which inputs and what it refuses, in the terms of the gate |

And the commit message says: the failing tests, the cause, the fix, the gate, the card-0 run.

## Special cases

**A precision failure the device cannot meet.** List it as `precision` with the *mechanism* and one of
"reproduced by hand" or "attributed, not decomposed". The way to close it is a bound derived for that
composition (as `step85` does for convolution) and, if the vendored tolerance is tighter than the
hardware's arithmetic, it stays listed: the tolerance is Burn's, and is never loosened here.

**A simulator disagreement.** The test passes on silicon and fails on ttsim: silicon is the arbiter. Keep
the gate with the simulator line `[-]` and a divergence row, as the definition of done allows.

**A test that needs more than the device offers** (a mask longer than the sort's 1,024). Refuse it by the
operation's name with the limit, gate the refusal, and list the limit in the notes; do not let a lower
layer's failure leak out of it.

**An upstream defect.** `burn-tensor-testgen`'s `#[might_panic]` deadlocks if the panic text differs from
its expectation, so a correct refusal is reported as a timeout. List the test with that diagnosis; the
suite stays pinned.

**Several tests, one cause, found while fixing another.** Open a roadmap item with the evidence and keep
the current change to its claim, as with the `powf(inf, subnormal)` NaN.

## Checklist

- [ ] failing tests grouped; the fix is the one others wait on
- [ ] kind decided (gap, sharp edge, by design, precision, slow, test defect)
- [ ] cause stated in a sentence a test could contradict
- [ ] design: composed or widened before new; no host fallback; no new knob; padding and residency kept
- [ ] gate `stepNNN`: exact oracle or derived bound; exact traffic; ragged and whole tiles; views; the
      limit refused by name; padding chained; negative control
- [ ] gate **watched failing** with the fix removed (and the mutation checked to have applied)
- [ ] unit test beside a library change, also watched failing
- [ ] conformance: target tests pass; no new failures; unexpected passes removed from the list
- [ ] fmt, clippy, boundary checks, generator `--check`s; training regression if routing or arithmetic moved
- [ ] silicon on card 0; gate and modules in `SMOKE`
- [ ] `expected-failures.toml` shrunk and every surviving reason re-read
- [ ] notes, coverage, roadmap, parity doc, learnings; commit message with the evidence

## Worked examples: round one

| Finding (tests) | Cause | Fix | Gate | Card 0 |
|---|---|---|---|---|
| `softmax`/`log_softmax` of rank 1, 3, 4, any axis, transposed (10) | the composition took a plain rank-2 matrix | move the axis last, flatten, same composition, restore | `step149_softmax_nd` | pass |
| I32/Bool `reshape` that changes the stored matrix (18, and the CTC losses behind softmax) | `reshaped_strided` took float dtypes only | accept I32 and Bool; `int_reshape`/`bool_reshape` call it | `step150_int_bool_reshape` | pass |
| `mask_where`, `powf` with a broadcast operand (8) | equal-shape kernels | expand operands by device copies, as the integer variant did | `step151_broadcast_mask_pow` | pass |
| `cat` with an empty tensor (5) | `native_zeros` asked the device for a zero-extent buffer; the failure surfaced only at read-back | decline empty shapes; the host builds them | `step152_empty_tensors` | pass |
| `[408, 261120] @ [261120, 408]` (1) | a block over 0x7FFF circular-buffer pages panicked the planner and the server thread | `PlanError::TooManyPages`, mapped to "does not fit", so the chunk search shrinks K | unit tests in `l1.rs`, `matmul.rs` | 13 s |
| `bool_argwhere`, `nonzero` (10) | a stub that always refused, listed as native | stable argsort of the negated mask, coordinates by `I32` div/rem; one 4-byte count read | `step153_argwhere` | pass |
| `conv_transpose` with `padding_out >= stride` (2) | PyTorch's limit asserted; Burn's reference has none | relax, after proving forward and all gradients against Flex | `step154_conv_transpose_padding_out` | pass |
| `set_default_dtypes(.., I32)` (every test: found before the first one ran) | `dtype_usage(I32)` was storage-only; int arithmetic is native | `Storage \| Arithmetic` | `step144` (updated) | pass |

## Worked example: round two, `interpolate`

The first planning question for a "missing op" is whether it needs a kernel. This one did not: each output
is a weighted sum of a few inputs, with indices and weights fixed by the shapes, so it composes from three
device primitives (a word-copy gather, an SFPU multiply, the in-order SFPU row sum) with the host building only
the tables. Written up here because it exercised steps 4 to 6 in ways round one did not.

| Finding (tests) | Cause | Fix | Gate | Card 0 |
|---|---|---|---|---|
| `interpolate`, nearest/bilinear/bicubic/Lanczos3, forward (18) and nearest backward under two autodiff binaries (4) | generated stubs that refused by name, filed under "not planned until a model needs it" (a deferral, so a gap) | `ops/interpolate.rs`: shape-only tap tables, one gather, SFPU multiplies, in-order row sum; the gradient is the transpose of the forward map. Lanczos3 backward too, though Flex has none | `step155_interpolate` (14 simulator tests, 15 on silicon with BF16) | pass |
| a `metadata` table of many *distinct* values (found by the Lanczos3 weights; no Burn test) | `tensor::metadata` batched 200 words, but a word that differs from its predecessor costs two list entries, so a tile with more than ~126 distinct values overran the packet limit and **panicked the server thread** | batch by entries as well (`TRANSFER_SIDE_MAX`); batching is unchanged where it already fit | unit test beside it (`tensor.rs`), watched failing | via step155 |

Lessons that generalize:

- **Compare against the arithmetic, not only against a bound.** The card does Flex's f32 products and sums in
  Flex's order for nearest and bilinear, so those are held to *equality*, forward and backward; the derived
  bound is for bicubic and Lanczos3, where Flex accumulates in f64. Measuring first (how many elements differ,
  and the worst difference over the bound) showed the equality was real and the bound was neither violated
  nor slack (worst 0.70).
- **An oracle that is not Flex needs its own validity conditions, asserted.** `image` clamps `f32` images to
  [0, 1] and widens its filter when shrinking, so the gate restricts itself to upsampling and
  `align_corners = false`, keeps the data inside (0, 1), asserts nothing was clamped, and asserts the one
  assumption its error bound rests on (the filter divisor stays above `SIGMA_MIN`). A first draft assumed the
  weight magnitudes sum to at most 1.5; the gate measured 2.23, so the bound was re-derived from facts instead.
- **A control can be inapplicable.** A single input pixel makes every normalized filter agree, and nearest
  ignores `align_corners`; the control for such a case is a different wrong answer (a reversed row), and the
  gate counts how many controls actually fired so it cannot go quiet.
- **A gate must be watched failing at every level.** Four mutations (nearest's rounding rule, Lanczos3's
  renormalization, `align_corners`, the gradient's summation order) each failed the right tests; the order
  mutation passed the unit tests until a multi-tap case pinned it, because the adjoint identity does not see
  order.
- **A shared helper's limit is found by a new kind of input.** `metadata` had only ever carried constants with
  few distinct values. Read the helper, fix it where it lives, and keep the unit test beside it.
- A BF16 narrowing gate is silicon-only (PACR `0x105`, [`ttsim-divergence.md`](../learnings/ttsim-divergence.md)).

Still open gaps (deferred, not designed away): `deform_conv2d` (closed in round five, below).
Closed by a technical limit: F64 casts. Not closed: the precision failures,
listed with their mechanisms, and the tests the simulator is too slow for, each with its silicon verdict.

## Worked example: round three, `scatter_nd` with Mul, Min and Max

The reason on file was "Burn leaves duplicate indices undefined for the others", which is a statement about
Burn's contract, not a limit of the card. The `Assign` path already applied the index tuples one at a time in
index order, so the same loop gives a defined order for the rest: Flex's. A 40-line change, no new primitive.

| Finding (tests) | Cause | Fix | Gate | Card 0 |
|---|---|---|---|---|
| `scatter_nd` Mul/Min/Max: float and int forward (6), gradients under two autodiff binaries (20) | refused by name, filed as a design ("duplicates undefined") though only a deferral | `ops/index.rs` `scatter_rows`: per tuple, `Mul` selects `current * value`; `Min`/`Max` select `value` where `value < current` (`>`) holds, ANDed with the row mask. The gradients needed nothing: Burn composes them from `gather_nd`, comparisons and masks | `step156_scatter_nd_reduce` (3 tests) | pass |

Lessons that generalize:

- **Read a "not supported" reason for whose limit it is.** "Undefined in Burn" means any order is *allowed*, and
  Flex's is a good one; it does not mean the device cannot.
- **A control can be unable to fire on the data you have.** Min and Max do not depend on the order of untied
  values, so "first writer only" and "reverse order", the controls that worked for `Add` and `Assign`, differ from
  the right answer only by luck; the first run failed on exactly that. What can differ for them is a *tie*
  (`-0.0` against `+0.0`), a NaN and a later duplicate that improves on the first, so `min_max_ties_nans_and_zeros_follow_the_comparison`
  builds those by hand and asserts each control differs before it runs the device. Random data cannot tell a `<` from a
  `<=`: with `<=` mutated in, the random-data tests still passed and only the ties test failed.
- **Controls that need the data to move belong on the data that moves.** On the special-laden pass a row can hold
  only infinities and NaNs, which nothing replaces; there Flex is the oracle, and the "something changed" controls
  are asserted on the normals pass.

## Worked example: round four, `conv3d` and `conv_transpose3d`

Another "missing op" that needed no kernel: `ops/conv.rs` already built its im2col and overlap-sum tables from
shapes, so `Geometry` became generic over the number of spatial dimensions and the eight 3-D methods are
one-line calls into the same generic functions. The 2-D path kept its exact source order, which the 2-D gates
(`step85`, `step154`) and the MNIST golden confirm.

| Finding (tests) | Cause | Fix | Gate | Card 0 |
|---|---|---|---|---|
| `conv3d` (7 of 9) and `conv_transpose3d` (11) | generated stubs that refused by name, filed under "not planned until a model needs it" (a deferral, so a gap) | `Geometry<const D: usize>`, generic `conv`, `conv_x_backward`, `conv_weight_backward`, `conv_transpose*`; the `padding_out` crop loops over `D` | `step157_conv3d` (3 simulator tests, plus a BF16 one on silicon) | pass |
| `conv3d` forward, `test_conv3d_groups` and `_complex` (2) | **not the new op, and not TF32 truncation**: the Matrix Unit rounds each product of a group of eight to a quantum set by the group's largest (operands exact in TF32) | none; listed as `precision` with the mechanism | `step158_matmul_accumulation` (the model equals the device bit for bit) | same failure as ttsim; the gate passes on silicon |

Lessons that generalize:

- **Check an attribution before copying it.** These two failures look like the existing "TF32 truncation" ones
  and the list's habit is to say so. The operands were `arange` integers below 2048, so they are exact, and the
  claim would have been wrong. Computing the exact sum on the host, then running a plain matmul on the same
  kind of integers, showed the cause is the shared matmul, found with no device change.
- **Decompose a numerical mismatch by fitting a model to the device, one rule at a time.** Isolate one effect per
  experiment (one big term and one small one; the same pair in every column; the second operand varied), fit the
  rule, then score variants of it against the device on random data, counting mismatches: the right rule is the
  one at zero, and its neighbours are off by hundreds. Three rules were found this way (the exponent is the sum
  of the operands', a zero slice still sets it, `Dst` rounds ties upward), each only visible on data the earlier
  tests did not have. Keep controls that turn each rule and show the gate fails.
- **A filter that lists three tests may run one.** `xtask silicon --filter module::conv3d::` listed three tests
  and ran only the first, which failed; the other two ran when named in full. Check the run count, not the list.
- **A second implementation can decompose a whole group of failures at once.** Fitting the Matrix Unit's
  accumulation explained two `conv3d` tests, but the cheap way to learn what the other 19 `precision` entries
  were was to run the suite with an exact matmul (`--exact`): all 21 passed, so none was a separate defect.
  Build the reference path first when failures share a suspect, and let the suite say which ones it explains.
- **The exact profile exposed no regression, because the default path did not move.** The hook returns before
  acting unless the mode is exact, so the default profile (1,789 pass, 49 listed), the MNIST golden and every
  older gate were unchanged; confirm that, not just the new gate, when adding an opt-in mode.

## Worked example: round five, `deform_conv2d`

The last missing op in the suite (17 tests: nine forward, four autodiff cases under two binaries). Unlike
`interpolate` the taps depend on the data, so the host cannot tabulate them; it builds what the shapes fix and the
card does the rest (`ops/deform_conv.rs`, module documentation). Still no new kernel.

| Finding (tests) | Cause | Fix | Gate | Card 0 |
|---|---|---|---|---|
| `deform_conv2d` forward (9) and backward under two autodiff binaries (8) | refused by name, filed under "not planned until a model needs it" (a deferral) | sample positions, bilinear fractions and corner validity on the device; four resident gathers of channel rows of `x` (an invalid corner reads an appended zero row); Flex's term order; the columns feed `conv`'s own product and weight gradient (`conv_products`, `weight_gradient`, extracted from `conv` and `conv_weight_backward_geometry`); the x gradient is a one-hot matrix times the column gradient | `step160_deform_conv2d` (4 simulator tests, a BF16 one on silicon) | pass |

Lessons that generalize:

- **Passing the vendored tests proved little here.** They passed on the first run (17 of 17, 12 s), but the autodiff
  ones allow `relative(0.5)` on every gradient. The evidence is the gate: data chosen so that every product and sum is
  exact (offsets in quarters, masks in quarters, small integers elsewhere, magnitudes below the group adder's quantum),
  which makes "equal to Flex" a fair demand in a different summation order, forward and all five gradients.
- **Check what a primitive costs before the plan leans on it.** The plan named `select` as the data-dependent gather;
  reading it showed `select_native` is `gather_native`, which stages every row of the table for every element
  returned. The design stands for these sizes, so it got an explicit limit (2^24 staged addresses, refused by name,
  as `interpolate`'s) and a roadmap item, instead of an unmeasured claim. The same read ruled out `select_add` for the
  x gradient: it launches several kernels per index, thousands of launches on the simulator; the one-hot matmul is a
  fixed number.
- **Clear the inputs of an invalid case, not only its result.** Flex returns zero for a sample on or past the border
  without evaluating its neighbours; a sample at infinity has NaN fractions, and `0 * NaN` is NaN. The fractions of an
  invalid sample are set to zero before any product, so nothing needs a final mask, and the one-hot weights stay finite.
- **A NaN test must say whose arithmetic it tests.** With a NaN in `x` the output had Flex's poisoned cells exactly but
  held infinities where Flex has NaN. That is the Matrix Unit (the unpacker turns a NaN into an infinity and `inf * 0` into
  zero; silicon-operating-notes, the 17-pair corpus), not the op. The gate now runs the case in exact matmul mode,
  where the products are the SFPU's and the result must equal Flex NaN for NaN, and asserts only what the fast mode
  guarantees (every cell Flex leaves finite is equal).
- **A control that fails for a bookkeeping reason is not a control.** "Mask dropped" first compared whole results, and
  a run without a mask has no mask gradient, so it differed trivially. The controls now compare numbers.
- **Mutate the implementation, not only the input.** Five source mutations (corners 1 and 2 swapped, the sample-validity
  term dropped, the one-hot weights reversed, the offset gradient's axes swapped, the sampling axes swapped) each fail
  the gate; the border test alone misses the last one because its constant offsets are symmetric in h and w, which
  is why it is not the only test.
- **The simulator is slow where silicon is not:** the main test takes 77 s on ttsim and 1.5 s on card 0, so the
  smoke tier is where a regression is cheap to find.

## Worked example: round six, the suite on `Fusion<TtBackend>` (B13a)

The first run of the suite on `burn_tt::Tt` (`cargo xtask conformance --fusion`; the harness existed, the lists did not).
1,800 of 1,838 passed: the 32 failures listed for the default run recurred, and six were new, all in a layer no earlier
round had exercised: the fuser and burn-fusion itself. After the fix 1,803 pass and 35 are listed.

| Finding (tests) | Cause | Fix | Gate | Card 0 |
|---|---|---|---|---|
| `thresholded_relu` (2) and `mask_fill_swap_dims` (1) returned wrong values | burn-fusion hands a fuser every scalar as `ScalarIr::UInt(id)`, its value bound only at execution, and `ElementWiseFuser` took every id for zero: any `lower_equal_elem` + `mask_fill` pair ran as `relu` | the match records the scalar ids it assumed to be `+0.0` (`TtOptimization::{AddRelu, Relu}.zeros`); `execute` reads them from `Context::scalars` and, if one differs, runs the original operations (`OrderedExecution::operation_within_optimization`) | `step161_fusion_scalar_guard`: a real `relu` and `add` + `relu` stay fused (counters), other thresholds and fill values equal the host oracle after a `relu` of the same shape; fails against the old fuser | pass (log `1791772236`) |
| `bool_equal_elem`, `bool_not_equal_elem`, bool `mask_fill` (3) | burn-fusion 0.21.0 `ScalarIr::to_relative` is `todo!()` for a Bool scalar; the fusion server thread dies | not ours: listed as `bug` with the upstream location in `expected-failures-fusion*.toml` | none | n/a |

Lessons that generalize:

- **A fuser sees a graph, not its values.** burn-fusion reuses a cached plan for the same graph with other scalars, so a
  match on "the scalar is zero" cannot be decided when the fuser runs. Record what the match assumed and check it when the
  plan executes, with the unfused operations as the fallback. The first version looked right because the only scalar in
  the training graphs is zero.
- **The conformance suite is also the test of the layers around the ops.** Nothing in the default run reaches the fuser,
  and the MNIST gates only ever feed it a real ReLU. The wrong results were upstream's own `thresholded_relu` and
  `mask_fill` tests.
- **Run a case right after the plan it must not reuse.** The gate runs each other scalar after a real `relu` of the same
  shape; a gate that began with the thresholded case would not meet a cached plan.
- **The exact profile added nothing under fusion**: the same 20 entries as without it, plus the upstream bool ones.

## Worked example: round seven, the F16 suites (D1)

Burn ships the float tests and the whole autodiff suite again with `f16` as the float element (`tensor_f16.rs`,
`autodiff_f16.rs`, 1,187 tests; upstream runs them on its GPU backends). They are entry binaries here too
(`tests/tensor_f16.rs`, `tests/autodiff_f16.rs`) and run in every profile. They could not start: `dtype_usage(F16)` was
`Storage`, so `set_default_dtypes(F16)` in the setup failed before the first test, and the runner could not even list them.
F16 now widens to F32 on the card, computes, and rounds once, as BF16 does (the generator emits `is_half`).
First run: 145 of 1,187 failed. After the fixes: 22 listed (1,165 pass), 20 in the exact profile (1,167 pass), the same under
`Fusion<TtBackend>` (no fusion-specific finding).

| Finding (tests) | Cause | Fix | Gate | Card 0 |
|---|---|---|---|---|
| every F16 test | `dtype_usage(F16)` was `Storage` | `general`, not `Accelerated` (no packed F16 matmul); arithmetic by widening (`float_compute_input`, the generated wrappers use `is_half`) | `step162_f16_arithmetic` | pass |
| `expand`, `meshgrid`, `slice_fill` (about 18) | `expanded` read a half-precision buffer with the row gather, which looks in F32 buffers only: **also broken for BF16**, no BF16 suite had covered `expand` | widen, expand, narrow back (exact) | same, `expand` cases; fails with the branch removed | pass |
| conv with a bias (6) | the generated wrapper widened the required operands and not an optional one (the bias): **also broken for BF16** | widen optional operands too (generator) | same, `conv2d`, `conv3d_groups`; fails with it removed | pass |
| `float_empty`, `float_random` (about 60), pooling, `interpolate`, `gather_nd` | refused F16 by name | F16 accepted: zeros are the same bits; the F32 draw is cast; pooling widens inside `mean_forward` | same | pass |
| 3 `precision`: `should_diff_gelu`, `cov_1`, `det_random_2x2x4x4` | the Matrix Unit matmul, by elimination: all pass in the exact profile | listed | none | fail the same on card 0 |
| 2 `precision`: CTC gradients | per-operation binary16 rounding in Burn's default `ctc_loss`; not the matmul (fail in the exact profile too) | listed; the tolerance stays | `ctc_gradient_in_f16_is_flexs_to_rounding_noise` | fail the same on card 0 |
| 15 `timeout` | ttsim speed | listed | none | all pass (run `1791811392`) |
| 2 F64 | not a Tensix dtype | listed with the F32 ones | none | n/a |

Lessons that generalize:

- **A dtype suite finds the bugs of the dtype next to it.** Two of the three defects (`expand`, the optional bias) were
  BF16's, latent for as long as BF16 existed, because nothing ran BF16 through `expand` or a biased `conv3d`. Fixing them for
  F16 fixed both.
- **Attribute a precision failure with a second implementation, not with the story that fits.** The CTC gradients missed
  by up to 0.235, far more than one binary16 step, so "rounding" was a hypothesis. Running the same composition on burn-flex
  in F16 gave the same shortfall and agreed with burn-tt to one step; that, not the explanation, went into the gate.
- **A test about rounding must say which results need rounding.** The first version of the gate demanded that rounding change
  something in every case, which a selection (`relu`, `max`, `expand`) cannot satisfy; those now assert the opposite, that
  the result is exactly an operand value.
- **Re-run what the change touched, not everything.** F32 code was not edited (every change is gated on `is_half`), so the
  F32 suites were not re-run; the new binaries were run in each profile, and the BF16 gates that the shared fixes reach
  (`step76`, `78`, `83`-`85`, `127`, `155`, `157`, `160`) were re-run.
