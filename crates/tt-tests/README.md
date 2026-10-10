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
| `step22` to `step47` | Program cache, profiling, padding; `Dst` tiles and the SFPU instruction set; Burn element-wise, division, exp/log, trig, broadcasts, reductions, softmax; batching, rank-N, `MOP`, loops, traces; integer/bool storage, compare/select and activations. |
| `step48` to `step68` | Ethernet clock and in-flight limits, NoC1 ownership, NC mover and pipeline, host DMA, tile layout, streaming and batched blocks, gather, reductions, mesh, general reduce, K-block matmul. |
| `step69` to `step89` | Reduction primitives, norms, integer ALU, rounding, FPU transpose and pooling, BF16 storage and matmul, extremum scans, integer reductions, attention, slice assignment, convolution, resident indices, SrcA transpose, mesh module reference, MNIST CNN. |
| `step90` to `step104` | Matrix element-wise, seeded PRNG, BFP formats and storage, ADC copies and planes, scalar config, L1 movement, XMOV tensor copy, source banks, unpacker handover. |
| `step105` to `step147` | Phase 10 close-out: mutexes and L1 atomics (105-106), restricted MMIO (107-108), `SFPLOADMACRO` (110), matrix diagnostics (111), packer and unpacker modes (112-114), NoC multicast, atomics, completion and the mover fast path (115-118), posted-write fence (119), tag search (120), wait planner (121), int/bool, indexing, scans and remainder (125-134), sort (135-136), random (140-142), FP16 (143-144), `MathMode` (145-146), mesh traces (147). Steps 109, 122-124 and 148-149 are unused; see `docs/completed-plans/hardware-coverage-closeout.md`. |
| `probe_*`, `scan` | Exploratory measurements of ttsim or silicon (`probe_addr_mod_sweep` holds the `AddrMod` evidence and asserts nothing). |
| `silicon_*` | Silicon-only gates, measurements and benchmarks (`silicon_perf`). |
| `fma_oracle` | `tt_isa::numerics::fma_bh` against the spec's `fma.c`, compiled by `build.rs`. |

`src/`: `harness` (the corpus runner and role helpers), `backend` (where a gate's
`Device` comes from, and silicon isolation: grid from the ARC, tile resets),
`burn_device` (a `TtDevice` attached to ttsim or the card), `mnist` (IDX loader),
`topology` (ttsim's measured multi-chip link maps), `data` (deterministic test data: LCG and xorshift streams, `panic_message`).

Shared test-only modules live beside the gates as `tests/<name>_support/mod.rs` (`eltwise_support`, `fp16_support`, `matrix_debug_support`, `noc_support`, `prng_support`) and are pulled in with `mod <name>_support;`.

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
- Divergences: [`docs/learnings/ttsim-divergence.md`](../../docs/learnings/ttsim-divergence.md). Operating rules:
  [`docs/learnings/silicon-operating-notes.md`](../../docs/learnings/silicon-operating-notes.md).
