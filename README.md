# metalium-rs

A native-Rust stack for Tenstorrent Blackhole (p150a cards): bare-metal firmware
for the baby RISC-V cores, host device access, Tensix kernels, and a Burn backend.
No Python, no C++, no TT-Metalium. Developed against the
[ttsim](https://github.com/tenstorrent/ttsim) simulator and run on two p150a cards.

What it does today: Burn MLP and small transformer models train on the card,
with GDDR-resident matmuls, SFPU arithmetic/activations, softmax, embedding and
loss indexing, and full F32 sums/means. Rank-N tensors use strided views;
tile-aligned batched products stay resident. Fresh and traced compute share
B-reader / resident T0–T2 / NC-writer streaming ownership. Host transfers use
the card's DMA, and Burn submits device operations asynchronously.
Unsupported operations/layouts fail explicitly with operation, shape and dtype.
`burn-tt` owns tensor storage and seed state and has no `burn-flex` dependency;
Flex remains an external comparison backend. Core MLP and transformer gates
require native arithmetic and explicit result readback.
See [the native cutover and datatype backlog](docs/plans/burn-native-cutover.md).

| Doc | What it holds |
|---|---|
| [`docs/plans/roadmap.md`](docs/plans/roadmap.md) | **Start here:** what is left, as tracks (real models, performance, multi-chip) |
| [`docs/completed-plans/phases-0-10-roadmap.md`](docs/completed-plans/phases-0-10-roadmap.md) | Archived master roadmap: why each layer is shaped as it is |
| [`docs/completed-plans/phases-0-10-checklist.md`](docs/completed-plans/phases-0-10-checklist.md) | Archived Phase 0-10 tick-list and evidence |
| [`docs/learnings/silicon-operating-notes.md`](docs/learnings/silicon-operating-notes.md) | Silicon operating notes: hardware survival guide |
| [`docs/learnings/ttsim-divergence.md`](docs/learnings/ttsim-divergence.md) | Where ttsim and silicon disagree, by row number (code cites the rows) |
| [`docs/README.md`](docs/README.md) | Full 3-tier documentation index and codebase subsystem matrix |

## Getting started

```bash
cargo xtask fetch-ttsim   # pinned libttsim builds (1, 2 and 4 chip) into vendor/
cargo xtask fetch-spec    # pinned ISA specification tree into vendor/
cargo xtask fetch-mnist   # MNIST into vendor/mnist/; tt-mnist's build.rs needs it
cargo xtask fetch-kmd     # optional: tt-kmd's ioctl.h, for its ABI layout test
cargo test                # builds the riscv32im firmware and runs the validation tier
```

The firmware is built by `crates/tt-firmware-images/build.rs` as part of any
build that depends on it. `rust-toolchain.toml` requests the
`riscv32im-unknown-none-elf` target and the `llvm-tools` component it needs.
`cargo xtask` with no arguments prints every task.

### Test tiers

| Tier | Command | What it is |
|---|---|---|
| Validation | `cargo test` | Host tests and every simulator gate: ISA, kernels, data mover, residency, many tiles. Bit-exact against ttsim. |
| End to end | `cargo test -p tt-tests --features e2e --test step12_mnist` | Whole Burn training runs on ttsim, held to `crates/tt-tests/tests/golden/mnist_reduced.txt`. Run when the arithmetic changes; CI runs it on every push. |
| Smoke | `cargo xtask silicon --smoke --release` | burn-tt against burn-flex on the cards, single ops up to the reduced training runs. |
| Silicon | `cargo xtask silicon --release` | Every gate on hardware (`tt-tests` feature `silicon`, which implies `e2e`). |
| Benchmarks | `cargo xtask bench` | The firmware benchmarks on the cards, device-timed and held against the spec's peaks, collected into `target/silicon/bench/`. The current numbers are in [`docs/learnings/performance-scoreboard.md`](docs/learnings/performance-scoreboard.md); history is in [`docs/learnings/performance-log.md`](docs/learnings/performance-log.md). |

`cargo xtask silicon --help` lists the options (`--device N|all`, `--filter`,
`--keep-going`, `--timeout-secs`, `--include-ignored`, `--list`). See
[`xtask/README.md`](xtask/README.md).

## Layout

| Crate | Ships | What it is |
|---|---|---|
| [`tt-isa`](crates/tt-isa) | yes | `no_std`, no dependencies. Encoders, register maps, coordinates, L1 map, mover protocol. Built for host and `riscv32im`. |
| [`tt-device`](crates/tt-device) | yes | `Transport` trait, `Device`, TLB windows, ARC telemetry, GDDR, Ethernet tiles. |
| [`tt-kmd`](crates/tt-kmd) | yes | The silicon `Transport`: `/dev/tenstorrent/N` via the tt-kmd driver. |
| [`tt-layout`](crates/tt-layout) | yes | Host tilize / detilize. |
| [`tt-kernels`](crates/tt-kernels) | yes | Tensix kernels, `Session`, GDDR tensors, L1 planner, data mover, multi-chip shard. |
| [`tt-firmware-images`](crates/tt-firmware-images) | yes | Builds and embeds the firmware images. |
| [`burn-tt`](crates/burn-tt) | yes | The Burn backend. |
| [`tt-mnist`](crates/tt-mnist) | yes | Self-contained MNIST training binary. |
| [`tt-firmware`](crates/tt-firmware) | (embedded) | Bare-metal `riscv32im` firmware. A separate workspace. |
| [`tt-ttsim-sys`](crates/tt-ttsim-sys), [`tt-ttsim`](crates/tt-ttsim) | no | The simulator binding and its `Transport`. |
| [`tt-tests`](crates/tt-tests) | no | Every cross-crate gate, on ttsim or on silicon. |
| [`xtask`](xtask) | no | Fetching pins, generators, ship check, silicon runner. |

**Dev-only crates never reach a shipped artifact.** `cargo xtask check-no-sim-in-ship`
fails if any crate in `SHIPPABLE` (`xtask/src/ship.rs`) depends on `tt-ttsim` or
`tt-ttsim-sys`, or if a workspace member is in neither list.

## Things worth knowing before reading the code

- **Every ttsim contract violation terminates the process** (`_Exit`, no unwinding).
  `tt_ttsim::transport` validates each access against ttsim's decode map first, and
  every simulator gate runs in `fork_scope`, so a fatal error becomes a failed test.
- **There is no backdoor into tile memory.** Everything goes through PCIe TLB
  windows, and `INSTRN_BUF_BASE` and `Dst` are not NoC-visible, so a baby RISC-V core
  must push Tensix instructions and read results.
- **On silicon, read the Tensix grid from the ARC before touching a tile.**
  Harvested (fused-off) tiles do not reject an access: the NoC hangs, and the
  recovery reset drops the PCIe link. `tt_kernels::session::Session` does this in the
  right order; `Device::tensix_grid` is the query. Run silicon gates one per process
  through `cargo xtask silicon`. Details:
  [`docs/learnings/silicon-operating-notes.md`](docs/learnings/silicon-operating-notes.md).
- **Divergences are logged, not worked around silently**
  ([`docs/learnings/ttsim-divergence.md`](docs/learnings/ttsim-divergence.md)). Tests that cannot pass
  on ttsim are `#[cfg(feature = "silicon")]`, so they still compile.
- **Upstream revisions are pinned and hash-verified in `PINS.toml`**: the
  specification, ttsim, tt-metal's `cfg_defines.h`, tt-kmd's `ioctl.h`, MNIST, Burn.
- **Three files are generated, never hand-edited**, each with a `--check` mode run
  in CI:

  | Command | Output |
  |---|---|
  | `cargo xtask gen-cfg` | `crates/tt-isa/src/cfg/generated.rs`, backend config fields from `cfg_defines.h` |
  | `cargo xtask gen-isa` | `crates/tt-isa/src/isa/generated.rs`, encodings from `Bits32.lua`, cross-checked against the spec's `TT_*(...)` syntax blocks |
  | `cargo xtask gen-burn-ops` | `crates/burn-tt/src/generated/ops.rs`, native dispatch and explicit unsupported Burn operations |

- **Every encoding records which chip it is evidence for** (`isa::Provenance`). A
  Wormhole-only layout is `UNVERIFIED`; layouts measured where the spec draws only Wormhole live in
  `xtask/src/gen_isa/Bits32_BH.lua`. The Wormhole forms are under
  `isa::generated::defs::wormhole`.

## Next implementation work

Native reductions, K-blocked products, resident indexing, BF16 storage/casts,
convolution/pooling/attention and matrix elementwise arithmetic have landed.
BFP8/4/2 per-tensor storage, native conversions/products and mixed-storage
training/traces are delivered; see the
[storage contract and completed checklist](docs/plans/mixed-bfp-storage.md).
General payload-preserving Tensix transpose and further format coverage
(INT8, `a` variants, quantization) remain open. Burn random draws on the device
(seeded uniform, Bernoulli, normal and integer ranges, step140-142). See
[`docs/plans/burn-op-coverage.md`](docs/plans/burn-op-coverage.md) for per-method status and
[`docs/completed-plans/hardware-coverage.md`](docs/completed-plans/hardware-coverage.md) for the archived gate record.
Backend typed errors, initialization/discovery and Burn conformance are in
[`docs/plans/burn-backend-parity.md`](docs/plans/burn-backend-parity.md).

**Burn's own backend tests** run against burn-tt in
[`crates/burn-tt-conformance`](crates/burn-tt-conformance/README.md) (`cargo xtask conformance`, on ttsim; `--silicon` on card 0).
On silicon, with host downloads and staged matmuls made fatal (`--strict`), 3,001 of 3,025 F32 and F16 tests pass, and 1,098 of
1,187 with a BF16 float element. The rest are listed with reasons (the Matrix Unit's matmul rounding, F64 which Tensix lacks,
BF16's own resolution) and none is a missing operation; see
[its results table](crates/burn-tt-conformance/README.md#results-2026-10-12).

Multi-card meshes (`Topology::Cards`, `burn_tt::MeshEngine`) are one Burn device
whose buffers stay in chip 0's GDDR. Rank-two and batched matmuls (hence convolution
and attention products) are split by output columns across the cards, which hold
GDDR tensors in their own sessions and exchange tile slots over Ethernet with no host
staging; every other op runs on chip 0. The two- and four-chip MLP training gates
match the single-chip golden bit for bit, and mesh inference traces replay on
changed inputs (`step147`). Open: more than one compute tile per card, chips
running concurrently (the per-chip steps are issued one after another), mesh
training traces and `copy_into` on a mesh, and data-parallel training with a
gradient all-reduce. Ethernet pipelining and fusion remain open too. X280 dispatch
is proposed and unscheduled. `ttsim-qemu` is deliberately deferred: the
`tt-kmd` transport is already gated on real cards.
