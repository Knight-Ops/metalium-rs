# xtask

Build tooling, run as `cargo xtask <task>` (alias in `.cargo/config.toml`). Dev-only
and dependency-free: it shells out to `curl`, `sha256sum`, `rustfmt` and `cargo`.
`cargo xtask` with no task prints the list.

## Tasks

| Task | What it does |
|---|---|
| `fetch-ttsim [--force]` | Pinned `libttsim_bh.so`, `_x2.so`, `_x4.so` into `vendor/`, SHA-256 verified. |
| `fetch-spec [--force]` | Pinned tt-isa-documentation tree into `vendor/`, verified by content digest. |
| `fetch-kmd [--force]` | Pinned tt-kmd `ioctl.h` into `vendor/` (the driver ABI `tt-kmd` binds). |
| `fetch-mnist [--force]` | MNIST into `vendor/mnist/`, decompressed and hash-verified. |
| `gen-cfg [--check]` | `tt-isa/src/cfg/generated.rs` from the pinned `cfg_defines.h` (fetched if absent). |
| `gen-isa [--check]` | `tt-isa/src/isa/generated.rs` from `Bits32.lua`, cross-checked against the spec's syntax blocks, plus the measured Blackhole layouts in `src/gen_isa/Bits32_BH.lua`. |
| `gen-burn-ops [--check]` | `burn-tt/src/generated/ops.rs`: native dispatch for `OVERRIDDEN` in `src/gen_burn.rs`, explicit unsupported methods, and inherited composing Burn defaults. |
| `check-no-flex-in-backend` | `burn-tt` must have no Flex dependency in its normal, build, or dev dependency graphs; external comparisons remain allowed. |
| `check-isa-sources` | Parse `Bits32.lua` and report, generating nothing. |
| `check-no-sim-in-ship` | `cargo tree` over every `SHIPPABLE` crate (`src/ship.rs`) must not reach `tt-ttsim` / `tt-ttsim-sys`; every workspace member must be `SHIPPABLE` or `DEV_ONLY`. |
| `silicon [options]` | The silicon suite, one test per process. |
| `bench [options]` | The firmware benchmarks on silicon, collected into `target/silicon/bench/`. |

`--check` fails if the committed file is stale instead of rewriting it. CI runs all
three generators with `--check`. All pins are in `PINS.toml`.

## `cargo xtask silicon`

Builds `tt-tests` with `--features silicon` once, then runs each test binary
directly, one test per process, single-threaded.

| Option | Meaning |
|---|---|
| `--device N\|all` | Card to run on (default 0); `all` runs the selection on each card in turn. Passed as `TT_SILICON_DEVICE`. |
| `--filter S` | Only tests whose `binary::test` name contains `S`; repeatable, run in filter order. |
| `--smoke` | Adds the current `SMOKE` filters in `src/silicon.rs`, including native model regressions and step91–96 PRNG/BFP conversion, packed-product, propagation and mixed-training gates. |
| `--include-ignored` | Also run `#[ignore]` probes. |
| `--keep-going` | Do not stop at the first failure. |
| `--timeout-secs N` | Per-test wall-clock limit (default 120). |
| `--release` | Optimised build. Quote timings only from this. |
| `--list` | Print the selection without touching a card. |

The log, `target/silicon/<timestamp>.log`, is fsync'd after every line and stamped
with `boot_id`; each test's output goes to `target/silicon/out/`. After a host
crash, the last `START` without an `END` names the test that took it down, and a
changed `boot_id` proves a reboot. A run stops at the first failure by default,
because a failed gate may leave the card in a bad state.
Background: [`docs/learnings/silicon-operating-notes.md`](../docs/learnings/silicon-operating-notes.md).

## `cargo xtask bench`

The benchmark preset of `cargo xtask silicon`: always `--release
--include-ignored`, a 900 s per-test limit unless `--timeout-secs` is given, and
by default the `BENCH` selection in `src/silicon.rs`:
- `silicon_bench_memory`: GDDR6 through the data mover, one tile and every tile.
- `silicon_bench_path`: B through T0/T1/T2, hop by hop, and real ops.
- `silicon_eth_clock`: E1's cycle counter, gated alone.
- `silicon_bench_eth`: one link, both links, both directions.
- `silicon_perf::pcie_*`.

Every other `silicon` option applies; `--filter` replaces the default selection.
The ignored BFP release benchmark is selected explicitly with
`cargo xtask bench --device all --filter step94_bfp_storage::benchmark_resident_bfp_conversion_and_packed_product`.
It validates outputs, uses two warmups/nine samples and reports host medians
plus dataflow statistics for pack/unpack and direct packed matmul.
Every `BENCH {json}` line the tests print is collected into
`target/silicon/bench/<stamp>.jsonl` (one record per line, with its test) and
`<stamp>.md` (a table). [`docs/learnings/firmware-performance.md`](../docs/learnings/firmware-performance.md) is built from one such run.
