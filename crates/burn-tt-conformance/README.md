# burn-tt-conformance

Burn's own backend test suite, run against `burn-tt`. Dev-only: nothing ships from this crate.

The suite is `crates/burn-backend-tests/tests/` at Burn `v0.21.0`, the release the workspace compiles
against: about 1,850 tests of tensor operations (float, int, bool) and autodiff. It is not published to
crates.io, so `cargo xtask fetch-burn-tests` downloads it into `vendor/burn-backend-tests/` and verifies
it against the content digest pinned in `PINS.toml`. Upstream's tests check themselves against literals
that Burn's CPU backend meets, so a pass needs no second backend to compare with.

Only the backend is ours. Upstream's `common/backend.rs` and `common/autodiff.rs` name
`burn_dispatch::Dispatch`, a closed enum that cannot hold `TtBackend`, so each entry binary in `tests/`
replaces them:

| Entry | Backend under test |
|---|---|
| `tests/tensor.rs` | `TtBackend` |
| `tests/autodiff.rs` | `Autodiff<TtBackend>` |
| `tests/autodiff_checkpointing.rs` | `Autodiff<TtBackend, BalancedCheckpointing>` |
| `tests/tensor_f16.rs`, `tests/autodiff_f16.rs` | the float tests and the autodiff suite with `f16` as the float element (upstream's `tensor_f16.rs` and `autodiff_f16.rs`) |

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
`expected-failures.toml` as a `precision` entry whose reason says so. `--update` rewrites the list: it drops entries that now pass and adds new failures under `todo`, to be
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
and runs it on card 0 only, since the tests use `TtDevice::default()`. The full suite on silicon is not
part of smoke: build with `burn-tt-conformance/silicon` and run the test binaries one test per process.

## Layout

- `src/lib.rs`: re-exports `might_panic`, which the vendored tests import as `burn_backend_tests::might_panic`.
- `tests/common/setup.rs`: the base backend, engine registration (ttsim, or burn-tt's default silicon engine) and default dtypes.
- `tests/common/backend.rs`, `tests/common/autodiff_backend.rs`: the replacements for upstream's `common/`.
- The vendored modules are marked `#[rustfmt::skip]` and `#[allow(..)]`: `cargo fmt` follows `#[path]`
  modules, and the pin is a digest of the files as fetched.
