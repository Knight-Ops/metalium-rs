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
| `gen-burn-delegate [--check]` | `burn-tt/src/generated/delegate.rs`: every burn-backend op forwarded to burn-flex, except `OVERRIDDEN` in `src/gen_burn.rs`. |
| `check-isa-sources` | Parse `Bits32.lua` and report, generating nothing. |
| `check-no-sim-in-ship` | `cargo tree` over every `SHIPPABLE` crate (`src/ship.rs`) must not reach `tt-ttsim` / `tt-ttsim-sys`; every workspace member must be `SHIPPABLE` or `DEV_ONLY`. |
| `silicon [options]` | The silicon suite, one test per process. |

`--check` fails if the committed file is stale instead of rewriting it. CI runs all
three generators with `--check`. All pins are in `PINS.toml`.

## `cargo xtask silicon`

Builds `tt-tests` with `--features silicon` once, then runs each test binary
directly, one test per process, single-threaded.

| Option | Meaning |
|---|---|
| `--device N\|all` | Card to run on (default 0); `all` runs the selection on each card in turn. Passed as `TT_SILICON_DEVICE`. |
| `--filter S` | Only tests whose `binary::test` name contains `S`; repeatable, run in filter order. |
| `--smoke` | Adds the `SMOKE` filters: `step19_eltwise`, `step11_burn`, `step20_many_tiles` element-wise and column sums, and the `step12_mnist` first forward pass and reduced runs (one and four tiles). |
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
Background: "Silicon operating notes" in `docs/implementation-checklist.md`.
