# burn-tt-conformance

Burn's own backend test suite, run against `burn-tt`. Dev-only: nothing ships from this crate.

The suite is `crates/burn-backend-tests/tests/` at Burn `v0.21.0`, the release the workspace compiles
against: about 1,850 tests of tensor operations (float, int, bool) and autodiff. It is not published to
crates.io, so `cargo xtask fetch-burn-tests` downloads it into `vendor/burn-backend-tests/` and verifies
it against the content digest pinned in `PINS.toml`. Upstream's tests check themselves against literals
that Burn's CPU backend meets, so a pass needs no second backend to compare with.

## Results (2026-10-12)

Every lane matches its expected-failures list: no unexpected failure and no unexpected pass. "Listed" tests each have a category and a
reason in the list; **none is a missing operation**. The suites are `tensor`, `autodiff`, `autodiff_checkpointing`, `tensor_f16` and
`autodiff_f16` (3,025 tests), plus `tensor_bf16` and `autodiff_bf16` (1,187 tests, silicon only).

| Lane | Tests | Pass | Listed |
|---|---:|---:|---|
| silicon, card 0, `--strict`: F32 and F16 (run `1791833700`) | 3,025 | 3,001 | 24: 19 `precision` (Matrix Unit matmul), 5 F64 |
| silicon, card 0, `--strict`: BF16 (run `1791830816`) | 1,187 | 1,098 | 89: 87 `precision` (63 are BF16's own resolution, 22 the Matrix Unit matmul, 2 CTC), 2 F64 |
| ttsim, default (fast matmul) | 3,025 | 2,973 | 52: 24 `precision`, 23 `timeout`, 5 F64 |
| ttsim, `--accurate` | 3,025 | 2,989 | 36: 31 `timeout`, 5 F64 |
| ttsim, `--exact` | 3,025 | 2,987 | 38: 31 `timeout`, 5 F64, 2 `unsupported-op` |
| ttsim, `--fusion` (`Fusion<TtBackend>`) | 3,025 | 2,970 | 55: 24 `precision`, 23 `timeout`, 5 F64, 3 `bug` |

What the listed tests are, and why they are not gaps in the backend:

- **Matrix Unit matmul precision (24 in the default profile).** The fast matmul truncates operands to TF32 and adds in groups of
  eight, so a few compositions (LU, determinants, the gelu gradient, conv2d and conv3d) miss tolerances that are never loosened. Every one
  passes under `--accurate` (integer slices with shared exponents, 19x to 64x the fast matmul's cost) and under `--exact`.
- **F64 (5 in F32 and F16, 2 more in BF16).** Not a Tensix dtype; `float_cast` to F64 is refused by name. Two end as timeouts because
  upstream's `#[might_panic]` macro deadlocks when the panic message differs.
- **`timeout` (23 default, 31 in the accurate and exact lists).** The simulator is too slow; each passes on silicon, which is why the
  silicon lists have none.
- **BF16 resolution (63).** The tolerance is as fine as one BF16 step (about 4e-3); they also miss under `--accurate`, and burn-flex run in
  BF16 on the host fails them too (`--reference-flex`).
- **3 `bug` under `--fusion`.** burn-fusion 0.21.0 has a `todo!()` for Bool scalars (`ScalarIr::to_relative`); the same tests pass on
  `TtBackend`.

The ttsim rows are the tests minus the list, from the last run of each lane (the lists are updated with the runs). The silicon runs
have a log under `target/silicon/<run id>.log`. `--fusion`, `--exact` and `--accurate` have no silicon lists yet
([roadmap](../../docs/plans/roadmap.md), "Conformance follow-ups").

Only the backend is ours. Upstream's `common/backend.rs` and `common/autodiff.rs` name
`burn_dispatch::Dispatch`, a closed enum that cannot hold `TtBackend`, so each entry binary in `tests/`
replaces them:

| Entry | Backend under test |
|---|---|
| `tests/tensor.rs` | `TtBackend` |
| `tests/autodiff.rs` | `Autodiff<TtBackend>` |
| `tests/autodiff_checkpointing.rs` | `Autodiff<TtBackend, BalancedCheckpointing>` |
| `tests/tensor_f16.rs`, `tests/autodiff_f16.rs` | the float tests and the autodiff suite with `f16` as the float element (upstream's `tensor_f16.rs` and `autodiff_f16.rs`) |
| `tests/tensor_bf16.rs`, `tests/autodiff_bf16.rs` | the same with `bf16` (upstream has no such entries). **Silicon only**: ttsim refuses the BF16 narrowing, so the ttsim lane skips them |

With `--features fusion` the base backend is `Fusion<TtBackend>` (`burn_tt::Tt`) instead.

## Running

```bash
cargo xtask fetch-ttsim && cargo xtask fetch-burn-tests
cargo xtask conformance                 # the whole suite on ttsim
cargo xtask conformance --release       # the same, optimised (the simulator runs faster)
cargo xtask conformance --filter float::ops::matmul --list
```

`cargo xtask conformance` runs every test **in its own process**: the simulator is a once-per-process
singleton, and one failure must not poison the next test. It then compares the outcome with
[`expected-failures.toml`](expected-failures.toml). A run **fails** on:

- a failing test that is not listed (a regression, or a new finding);
- a listed test that passes (the list only shrinks);
- a listed test that no longer exists;
- an entry still in the `todo` category.

`--exact` is a second profile: matmuls accumulate in exact F32 (`burn_tt::MatmulMode::Exact`, set with
`TT_EXACT_MATMUL=1` for every test process) and the outcome is compared with
[`expected-failures-exact.toml`](expected-failures-exact.toml) (`--fusion` has its own lists, `expected-failures-fusion.toml` and
`expected-failures-fusion-exact.toml`). The
default profile keeps the fast Matrix Unit matmul; a test that only the exact one passes stays listed in
`expected-failures.toml` as a `precision` entry whose reason says so.

`--accurate` is a third: matmuls run as integer slices with shared exponents on the Matrix Unit at LoFi
(`burn_tt::MatmulMode::Accurate`, `TT_ACCURATE_MATMUL=1`), within a derived bound of the exact value at a
small multiple of the fast matmul's cost, and the outcome is compared with
[`expected-failures-accurate.toml`](expected-failures-accurate.toml) (and `-fusion-accurate` with `--fusion`). It is not
combined with `--exact`. `--update` rewrites the list: it drops entries that now pass and adds new failures under `todo`, to be
given a category and a reason by hand. `--strict` sets `TT_STRICT=1`, so a download or a staged matmul
fails the test. Output of every test, with burn-tt's `TT_REPORT=1` residency table, is kept in
`target/conformance/<stamp>/out/`.

`cargo test --workspace` does not build or run any of this: every test target needs the `run` feature,
which only the runner enables.

## Closing a failure

[`docs/plans/conformance-to-feature.md`](../../docs/plans/conformance-to-feature.md) is the procedure from a
red test to a fixed, gated, documented feature (triage, root cause, design, a gate that is watched
failing, silicon, documentation, shrinking the list below).

## Expected failures

Each group in `expected-failures.toml` has a category and one reason for all its tests:

| Category | Meaning |
|---|---|
| `unsupported-op` | burn-tt refuses the operation or shape by name |
| `unsupported-dtype` | a dtype Tensix does not have (F64) |
| `precision` | a result outside the vendored tolerance; the reason cites the mechanism, and the tolerance is never loosened |
| `bug` | an unintended failure (an opaque error, a wrong claim in the docs); each is on the roadmap |
| `sim-divergence` | ttsim differs from silicon; each has a `docs/learnings/ttsim-divergence.md` entry |
| `timeout` | too slow for the limit |
| `todo` | not triaged; a run fails until it is |

The goal is an empty file. Tests upstream marks `#[ignore]` are not run.

## On silicon

`cargo xtask silicon --smoke` includes a subset of the passing modules (`SMOKE` in `xtask/src/silicon.rs`)
and runs it on card 0 only, since the tests use `TtDevice::default()`.

The full suite on silicon is `cargo xtask conformance --silicon [--release]`: it builds with
`burn-tt-conformance/silicon`, runs every test in its own process on card 0, one at a time (the card lock
refuses a second process; the timeout defaults to 600 s, the 300 to 500 LU tests take up to 355 s), and
compares the outcome with `expected-failures-silicon.toml` (`-fusion`, `-exact`, `-accurate` as for ttsim). One
file holds every suite, the BF16 ones included. The run ID is the stamp of its fsync'd log,
`target/silicon/<stamp>.log` (START and END lines, as `cargo xtask silicon` writes). It needs no ttsim,
and `--list` touches no card. Check that no other session is using the card first: a second process is
refused with `CardBusy`, and the runner has no lock between agents.

## Telling a test from a defect: `--reference-flex`

`cargo xtask conformance --reference-flex [--suite ..]` runs the same tests on burn-flex, on the host (build
feature `reference-flex`; this crate is dev-only, `burn-tt` itself has no Flex dependency). It compares with
no list and exits 0, printing the failures and writing `target/conformance/<stamp>/summary.txt`. A test that
burn-flex also fails misses its tolerance for the type's own reasons (BF16 has 8 significant bits, and
upstream's half-precision tolerances were set for F16): list it as `precision` with that evidence, and keep
the vendored tolerance. A test that only burn-tt fails is a finding for
[`conformance-to-feature.md`](../../docs/plans/conformance-to-feature.md).

## Layout

- `src/lib.rs`: re-exports `might_panic`, which the vendored tests import as `burn_backend_tests::might_panic`.
- `tests/common/setup.rs`: the base backend, engine registration (ttsim, or burn-tt's default silicon engine) and default dtypes.
- `tests/common/backend.rs`, `tests/common/autodiff_backend.rs`: the replacements for upstream's `common/`.
- The vendored modules are marked `#[rustfmt::skip]` and `#[allow(..)]`: `cargo fmt` follows `#[path]`
  modules, and the pin is a digest of the files as fetched.
