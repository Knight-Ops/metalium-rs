# tt-ttsim

A safe wrapper over `libttsim.so` (via `tt-ttsim-sys`) that implements
`tt_device::Transport`, so every device gate runs bit-exact against the simulator.
**Development and test only**: `cargo xtask check-no-sim-in-ship` keeps it out of
every shippable crate's dependency graph.

## Key items

| Item | What it does |
|---|---|
| `Simulator::open()` / `open_path()` | The one-per-process simulator. Neither `Send` nor `Sync`. `chip_count`, `transports()`, `clock(n)`. |
| `LibTtsim` (`transport`) | The `Transport`. Validates every access against a mirror of ttsim's decode map before calling, returning `TransportError` instead of letting ttsim `_Exit`. |
| `fork_scope(f)` | Runs simulator work in a forked child, so a fatal ttsim error becomes a test failure that names the test. |
| `outside_fork(f)` | Library work in the parent (a host reference, say) that must not overlap a fork. |
| `default_lib_path`, `x2_lib_path`, `x4_lib_path` | Single, dual (P300) and four chip builds in `vendor/`. |

## Environment

| Variable | Overrides |
|---|---|
| `TT_TTSIM_LIB` | The single-chip library (default `vendor/libttsim_bh.so`) |
| `TT_TTSIM_LIB_X2` | The dual-chip library (`vendor/libttsim_bh_x2.so`) |
| `TT_TTSIM_LIB_X4` | The four-chip library (`vendor/libttsim_bh_x4.so`) |

Pointing `TT_TTSIM_LIB_X2` at the single-chip build is how the multi-chip gates are
checked for vacuity.

## Test

```bash
cargo xtask fetch-ttsim
cargo test -p tt-ttsim
```

`tests/fatality.rs` bypasses the wrapper and proves, inside forked children, that
the accesses `validate` refuses really are fatal. If one starts passing, the
matching rule should be deleted, not kept.

## Gotchas

- `init` may be called once per process, and libttsim is not thread safe. Never use
  it outside `Simulator`, and run test bodies in `fork_scope`.
- ttsim is not silicon. It models an unharvested chip and differs in logged ways
  ([`docs/learnings/ttsim-divergence.md`](../../docs/learnings/ttsim-divergence.md)). A simulator pass is evidence about the simulator.
