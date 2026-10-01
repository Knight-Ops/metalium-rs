# tt-tests

Every gate that spans several crates, on ttsim or on silicon, and the harness they
share. Dev-only (`DEV_ONLY` in `xtask/src/ship.rs`): it is the one place allowed to
depend on both `tt-device` and `tt-ttsim`. Never shipped.

## Features: which device the suite runs against

| Feature | Effect |
|---|---|
| (none) | `harness::Dev` is `Device<LibTtsim>`: every gate in-process on ttsim, in a fork. End-to-end gates show as ignored. |
| `e2e` | Turns on the whole-training gates in `tests/step12_mnist.rs`. |
| `silicon` | `harness::Dev` is `Device<Kmd>`: the same gates against `/dev/tenstorrent/N`. Compiles the `#[cfg(feature = "silicon")]` tests, drops the ttsim-only ones. Implies `e2e`. |

## Run

```bash
cargo test -p tt-tests                                        # validation tier
cargo test -p tt-tests --features e2e --test step12_mnist     # end to end on ttsim
cargo xtask silicon --smoke --release                         # burn-tt vs burn-flex on the cards
cargo xtask silicon --release                                 # every gate on hardware
```

Never run the silicon build with plain `cargo test`: use `cargo xtask silicon`,
which runs one test per process with an fsync'd log (see `xtask/README.md`).

Prerequisites: `cargo xtask fetch-ttsim` and `fetch-mnist`; `fetch-spec` for
`fma_oracle` (it skips without the spec tree or a C compiler).

## What is here

| Files | What they gate |
|---|---|
| `step1` to `step7` | Harness, TLB windows, multi-chip, heartbeat, first Tensix instruction, instruction corpus, local RAM, layout. |
| `step8` to `step10` | Element-wise kernel, matmul, three concurrent roles. |
| `step11_burn`, `step12_mnist` | burn-tt against burn-flex; MNIST training (golden in `tests/golden/mnist_reduced.txt`). |
| `step13`, `step14` | Ethernet, sharded matmul across chips. |
| `step15` to `step21` | Phase 9: GDDR, data mover, resident roles, GDDR matmul, GDDR element-wise, many tiles, one launch per op per tile. |
| `probe_*`, `scan` | Exploratory measurements of ttsim or silicon. |
| `silicon_*` | Silicon-only gates, measurements and benchmarks (`silicon_perf`). |
| `fma_oracle` | `tt_isa::numerics::fma_bh` against the spec's `fma.c`, compiled by `build.rs`. |

`src/`: `harness` (the corpus runner and role helpers), `backend` (where a gate's
`Device` comes from, and silicon isolation: grid from the ARC, tile resets),
`burn_device` (a `TtDevice` attached to ttsim or the card), `mnist` (IDX loader),
`topology` (ttsim's measured multi-chip link maps).

## Environment

| Variable | Use |
|---|---|
| `TT_SILICON_DEVICE` | Card for a silicon run (set by `cargo xtask silicon --device`). Default 0. |
| `TT_TOPOLOGY` | Silicon Burn gates: cards, `"0"` or `"0,1"`. |
| `TT_TILES` | Silicon Burn gates: tile count or `all`, unless the gate fixes its own. |
| `TT_BLESS=1` | Rewrite the MNIST golden from the ttsim run of `the_mlp_trains_on_a_reduced_dataset`. |

## Gotchas

- The golden is written by ttsim and must be reproduced bit for bit on silicon and
  on any tile count. Re-bless only when the arithmetic changes deliberately.
- Silicon gates claim tiles through `backend`, which reads the grid from the ARC
  first. Do not hardcode coordinates outside the surviving columns; the gate tile is
  `backend::GATE_TILE` (3, 4).
- Simulator-specific assertions are `#[cfg(not(feature = "silicon"))]`.
- Divergences: `docs/ttsim-divergence.md`. Operating rules: "Silicon operating
  notes" in `docs/implementation-checklist.md`.
