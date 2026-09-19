# metal-rs

A native-Rust path to Tenstorrent Blackhole, developed against the
[ttsim](https://github.com/tenstorrent/ttsim) simulator.

Progress is tracked in [`docs/implementation-checklist.md`](docs/implementation-checklist.md).

This repository currently contains the **baseline vertical slice**: the smallest
thing that proves the whole stack, end to end, and the abstractions every later
phase builds on. `RUST_IMPL_PLAN.md` describes the larger programme it is the
first part of.

## What works today

```
host ──libttsim──> chip
  └ PCIe TLB window  → a Tensix tile's L1, read and written        (step 2)
  └ firmware image   → L1, RISCV core released from reset          (step 3)
  └ core pushes      → SFPLOADI ×4, SFPMUL, SFPSTORE               (step 4)
  └ core reads Dst   → L1 mailbox
  └ host reads       → 0x40C0_0000   (3.0 × 2.0 = 6.0, bit-exact)
```

`cargo test` runs all of it, deterministically, with no hardware.

## Getting started

```bash
cargo xtask fetch-ttsim   # downloads the pinned libttsim_bh.so into vendor/
cargo test                # builds the riscv32im firmware and runs every gate
```

The firmware is built automatically by `crates/tt-tests/build.rs`; there is no
separate step. `rustup` needs the `riscv32im-unknown-none-elf` target and the
`llvm-tools` component, both of which `rust-toolchain.toml` requests.

## Layout

| Crate | What it is |
|---|---|
| `tt-isa` | `no_std`, no dependencies. Instruction encoders, register maps, coordinate spaces. Compiled for **both** the host and `riscv32im`. |
| `tt-device` | Host-side device access: the `Transport` trait, TLB windows, L1 addressing, core reset and firmware loading. |
| `tt-firmware` | Bare-metal `riscv32im` binaries. A separate workspace; built by `xtask`/`build.rs`, never by the host `cargo build`. |
| `tt-ttsim-sys` | Raw FFI over the ten `libttsim_*` entry points. |
| `tt-ttsim` | Safe singleton wrapper, fork isolation, and the `Transport` implementation. **Dev-only** — `cargo xtask check-no-sim-in-ship` enforces that it stays out of shippable graphs. |
| `tt-tests` | Gates that span several crates, including every simulator gate. |

## Things worth knowing before reading the code

**Every ttsim contract violation terminates the process.** No return code, no
unwinding, no panic hook. Two consequences shape the design: `tt_ttsim::transport`
validates each access against ttsim's decode map *before* making it, and every
simulator test body runs inside `fork_scope`, so a fatal error becomes a failing
assertion instead of a vanished test runner.
`crates/tt-ttsim/tests/fatality.rs` pins five accesses as genuinely fatal, so the
validation layer cannot quietly become unnecessary.

**There is no backdoor into tile memory.** `libttsim_tile_rd_bytes` is
unsupported on Blackhole, so everything goes through PCIe TLB windows. And
`INSTRN_BUF_BASE` and `Dst` are both unmapped to the NoC, so the host can neither
push a Tensix instruction nor read a compute result — a baby RISC-V core has to
sit in the middle. That is why step 3 is a hard prerequisite for step 4.

**Simulator divergences are logged, not worked around silently.** See
[`docs/ttsim-divergence.md`](docs/ttsim-divergence.md). Tests that cannot pass
against ttsim are `#[cfg(feature = "silicon")]` — compiled always so they cannot
rot, run only with `--features silicon` against real hardware.

**Upstream revisions are pinned in `PINS.toml`.** The specification commit, the
ttsim release, and the tt-metal commit for `cfg_defines.h`. They are specification
inputs, not dependencies: bumping one invalidates the gates until they are re-run.

## Not done yet

Silicon gates (no hardware on hand), the `Kmd` transport and the `ttsim-qemu`
path, `tt-isa-gen`, and everything from the tensor layer upwards.
