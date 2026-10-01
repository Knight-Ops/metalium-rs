# metal-rs

A native-Rust stack for Tenstorrent Blackhole (p150a cards): bare-metal firmware
for the baby RISC-V cores, host device access, Tensix kernels, and a Burn backend.
No Python, no C++, no TT-Metalium. Developed against the
[ttsim](https://github.com/tenstorrent/ttsim) simulator and run on two p150a cards.

What it does today: a Burn MLP trains on MNIST with every matmul, element-wise op,
ReLU, bias-gradient sum and weight update on the card, and the dataset, weights and
activations resident in GDDR6 (`crates/tt-mnist`). The loss and the rank-1 bias
updates still run on the host: six small tensors cross PCIe per step. A session can spread work over many Tensix
tiles, and matmuls can be sharded over two cabled cards via Ethernet.

| Doc | What it holds |
|---|---|
| [`docs/RUST_IMPL_PLAN.md`](docs/RUST_IMPL_PLAN.md) | The plan: why each layer is shaped as it is |
| [`docs/implementation-checklist.md`](docs/implementation-checklist.md) | What is done, per phase; "Silicon operating notes" |
| [`docs/ttsim-divergence.md`](docs/ttsim-divergence.md) | Where ttsim and silicon disagree, by row number (code cites the rows) |

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
  through `cargo xtask silicon`. Details: "Silicon operating notes" in
  `docs/implementation-checklist.md`.
- **Divergences are logged, not worked around silently**
  ([`docs/ttsim-divergence.md`](docs/ttsim-divergence.md)). Tests that cannot pass
  on ttsim are `#[cfg(feature = "silicon")]`, so they still compile.
- **Upstream revisions are pinned and hash-verified in `PINS.toml`**: the
  specification, ttsim, tt-metal's `cfg_defines.h`, tt-kmd's `ioctl.h`, MNIST, Burn.
- **Three files are generated, never hand-edited**, each with a `--check` mode run
  in CI:

  | Command | Output |
  |---|---|
  | `cargo xtask gen-cfg` | `crates/tt-isa/src/cfg/generated.rs`, backend config fields from `cfg_defines.h` |
  | `cargo xtask gen-isa` | `crates/tt-isa/src/isa/generated.rs`, encodings from `Bits32.lua`, cross-checked against the spec's `TT_*(...)` syntax blocks |
  | `cargo xtask gen-burn-delegate` | `crates/burn-tt/src/generated/delegate.rs`, forwarding of every burn-backend op to burn-flex |

- **Every encoding records which chip it is evidence for** (`isa::Provenance`). A
  Wormhole-only layout is `UNVERIFIED`; layouts measured where the spec draws only Wormhole live in
  `xtask/src/gen_isa/Bits32_BH.lua`. The Wormhole forms are under
  `isa::generated::defs::wormhole`.

## Not done yet

The `ttsim-qemu` path; keeping tensors in GDDR across a multi-card mesh (today
`Topology::Cards` stages from the host per matmul); spreading a multi-card topology
over more than one tile per card; resident programs (Phase 9.7c). The checklist's
"Phase 9 -- next steps" is the current list.
