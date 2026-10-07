# Working in this repository

Native Rust stack for Tenstorrent Blackhole: host device access, bare-metal
firmware, Tensix kernels and a Burn backend. Develop against ttsim; verify on
silicon. Shipped crates must not depend on TT-Metalium FFI or the simulator.
C/C++ specification models and simulator FFI are development-only test oracles.

## Find the right layer

- `tt-isa`: dependency-free `no_std` encodings, registers, numerics and protocols,
  shared by host and firmware.
- `tt-device` / `tt-kmd`: `Transport` abstraction, device discovery/access and
  silicon driver binding.
- `tt-layout`: host tilize/detilize and format conversion.
- `tt-kernels`: program builders, `Session`, GDDR tensors, L1 planning, cache,
  streaming, traces and mesh execution.
- `tt-firmware`: separate RISC-V workspace, excluded from the host workspace.
  `tt-firmware-images/build.rs` automatically builds, validates and embeds it.
- `burn-tt`: native Burn storage, operation routing and device server threads.
- `tt-tests`: cross-crate gates, simulator engines and external Flex comparisons.
  `tt-mnist`: application/model workloads. `xtask`: fetchers, generators and runners.

Crates live under `crates/`. When adding a host workspace crate, classify it in
`xtask/src/ship.rs` as `SHIPPABLE` or `DEV_ONLY`.

## Read details on demand

Documentation follows a **3-tier lifecycle structure** (see [`docs/README.md`](docs/README.md)):

- `docs/learnings/`: Permanent empirical ground truth (hardware operating notes, simulator divergences, performance baselines). **Must be kept continuously up to date**.
- `docs/plans/`: Actionable technical specifications and active execution checklists (`[ ]` / `[x]`).
- `docs/proposals/`: Unscheduled design RFCs.

Docs retain superseded designs and stale checklist entries. Check current code,
crate READMEs and current-contract sections before treating historical prose as
an API or implementing a supposedly missing feature.

Once plans are completed, they should be moved to `docs/completed-plans`

Use documentation in `vendor/tt-isa-documentation/`, `cfg_defines.h`, and `ioctl.h` if you need documentation about the Tenstorrent Blackhole devices.

| Task                                    | Reference                                                                                                                                |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Setup and task options                  | [README.md](README.md), [xtask/README.md](xtask/README.md)                                                                               |
| New kernel/op; completion criteria      | [hardware-coverage.md](docs/plans/hardware-coverage.md), [tensix-next-features.md](docs/plans/tensix-next-features.md)                   |
| Burn backend changes                    | [burn-native-cutover.md](docs/plans/burn-native-cutover.md), [burn-backend-parity.md](docs/plans/burn-backend-parity.md)                 |
| Scheduler, credits, traces or recovery  | [streaming architecture](docs/learnings/streaming-dataflow-architecture.md), [traced execution](docs/plans/traced-execution.md)          |
| Hardware access/bring-up                | [silicon operating notes](docs/learnings/silicon-operating-notes.md), [implementation checklist](docs/plans/implementation-checklist.md) |
| Simulator refusal or numerical mismatch | [ttsim-divergence.md](docs/learnings/ttsim-divergence.md); read refutations as well as original findings                                 |
| Performance changes                     | [firmware-performance.md](docs/learnings/firmware-performance.md)                                                                        |
| Architecture/specification rationale    | [master-roadmap.md](docs/plans/master-roadmap.md), [tt-metal concepts review](docs/learnings/tt-metal-concepts-review.md)                |

### Codebase Subsystem $\leftrightarrow$ Documentation Matrix

| Crate / Layer           | Relevant Learnings (`docs/learnings/`)                                                        | Relevant Plans & Checklists (`docs/plans/`)                                                          |
| :---------------------- | :-------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| `tt-isa`                | `ttsim-divergence.md`, `riscv-guide-review.md`                                                | `master-roadmap.md`, `hardware-coverage.md`                                                          |
| `tt-device` / `tt-kmd`  | `silicon-operating-notes.md`, `ttsim-divergence.md`                                           | `master-roadmap.md`, `implementation-checklist.md`                                                   |
| `tt-layout`             | `firmware-performance.md`                                                                     | `hardware-coverage.md`, `tensix-next-features.md`                                                    |
| `tt-firmware`           | `firmware-performance.md`, `streaming-dataflow-architecture.md`, `silicon-operating-notes.md` | `hardware-coverage.md`, `traced-execution.md`, `tensix-next-features.md`                             |
| `tt-kernels`            | `streaming-dataflow-architecture.md`, `tt-metal-concepts-review.md`                           | `master-roadmap.md`, `traced-execution.md`, `hardware-coverage.md`, `tensix-next-features.md`        |
| `burn-tt`               | `tt-metal-concepts-review.md`                                                                 | `burn-backend-parity.md`, `burn-native-cutover.md`, `traced-execution.md`, `tensix-next-features.md` |
| `tt-tests` / `tt-mnist` | `ttsim-divergence.md`, `firmware-performance.md`                                              | `implementation-checklist.md`, `hardware-coverage.md`                                                |
| `xtask`                 | `silicon-operating-notes.md`                                                                  | `master-roadmap.md`, `burn-backend-parity.md`                                                        |

## Build and check

Run from the repository root. Toolchain/targets are in `rust-toolchain.toml`.
Pinned inputs live in `vendor/`; fetch missing inputs with `cargo xtask fetch-ttsim`,
`fetch-spec`, `fetch-mnist` (needed by `tt-mnist`), or `fetch-kmd`
(driver ABI test). `PINS.toml` owns revisions/hashes; do not bypass digest checks.

Choose checks for the changed layer; CI and `prek.toml` define the full set:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo test --workspace --features tt-tests/silicon --no-run
cargo clippy --workspace --all-targets --features tt-tests/silicon -- -D warnings
cargo xtask check-no-sim-in-ship
cargo xtask check-no-flex-in-backend
```

Firmware changes also need the separate workspace lint:

```bash
cargo clippy --manifest-path crates/tt-firmware/Cargo.toml \
  --target riscv32im-unknown-none-elf --bins --lib -- -D warnings
```

Arithmetic, routing or execution-order changes also need the training regression:

```bash
cargo test -p tt-tests --features e2e --test step12_mnist
```

**Never execute silicon tests with plain `cargo test`.** Use the isolated runner:

```bash
cargo xtask silicon --release --device all --filter <gate>
cargo xtask silicon --smoke --release --device all
```

`--list` inspects selections without touching cards. Logs are under
`target/silicon/`. If hardware is unavailable, report the unrun gate; compilation
and simulator results do not establish silicon correctness.

## Generated code

Never hand-edit outputs. Change the generator/input, regenerate, then run its
`--check` mode:

| `cargo xtask` task | Output                                |
| ------------------ | ------------------------------------- |
| `gen-cfg`          | `crates/tt-isa/src/cfg/generated.rs`  |
| `gen-isa`          | `crates/tt-isa/src/isa/generated.rs`  |
| `gen-burn-ops`     | `crates/burn-tt/src/generated/ops.rs` |

Burn routing is `OVERRIDDEN` in `xtask/src/gen_burn.rs`. Historical
`gen-burn-delegate` instructions are obsolete. Measured Blackhole encodings belong
in `xtask/src/gen_isa/Bits32_BH.lua` with evidence in `measured.rs` and a gate.
Wormhole-only documentation does not establish Blackhole behavior or encoding.

## Preserve these contracts

- Ensure that documentation stays up to date: record empirical hardware discoveries in
  `docs/learnings/silicon-operating-notes.md`, simulator quirks in `docs/learnings/ttsim-divergence.md`,
  and performance numbers in `docs/learnings/firmware-performance.md`. When working on features or
  ops, update the active checklists in `docs/plans/`.
- `burn-tt` has no Flex dependency or host arithmetic fallback. Unsupported ops,
  shapes and dtypes fail explicitly. Preserve Burn defaults that compose native
  primitives. Model gates reject intermediate downloads and staged host compute;
  verify device computation as well as traffic, since cached host data can hide
  a fallback. `TT_EXACT` is retired.
- GDDR execution uses `Session::enable_dram(b, nc)`: B reads on NoC0, T0/T1/T2
  unpack/compute/pack, NC writes on NoC1. Fresh and traced work share this
  scheduler. `TT_PIPELINE=0` and `TT_BATCH=0` retain ownership.
  `TT_EXECUTION`, `TT_SCATTER` and `TT_ELTWISE` are retired.
- Declare buffers/semaphores through `tt_kernels::l1::Requirements`; fixed regions
  are in `tt_isa::l1`. Preserve concurrent lifetimes, cache pins, trace holds and
  deferred frees. Establish program state explicitly and write whole mailbox
  descriptors: silicon retains state between programs and processes.
- Declare padding through `Pad`/`OpPadding`; test ragged producers followed by
  reductions/matmul. Repairing a view must not alter its parent's padding claims.
- Never guess numerical tolerances. Use specification models/SFPU interpreter
  for exact program results and derive approximation bounds beside the gate.
  Cover signed zeros, subnormals, NaNs, infinities and extremes. Preserve stated
  accumulation order; `SFPMAD` is not IEEE fused multiply-add.
- New ops need an independent oracle, a simulator gate with a meaningful
  negative control, silicon validation on both cards, native Burn routing,
  residency/padding checks and a smoke entry. Follow the coverage definition of
  done. Re-bless `crates/tt-tests/tests/golden/mnist_reduced.txt` with
  `TT_BLESS=1` only for an explained, deliberate arithmetic change.

## Avoid hardware failures

Use checked APIs and `Session` bring-up. Discover each chip's grids from ARC
before tile access; never probe surviving coordinates or copy ttsim's full grid
onto silicon. Harvested-tile accesses can hang the NoC and reboot the host.
Keep one `Device` per chip because power policy is chip-wide. Local data RAM
cannot answer NoC accesses while its owning core is in reset.

Preserve posted-write readback fences, NoC endpoint ownership, checked alignment
and in-flight limits. Firmware L1 polls need fences. Preserve the build-time
instruction gate; do not emit compressed instructions, `fence.i`, `pause` via
`spin_loop()`, or `ebreak` for panic. Ethernet customer code runs on E1; leave E0
and firmware-owned L1 intact. Read the operating notes before low-level changes.

ttsim is a non-reentrant process-wide singleton; violations call `_Exit`.
Simulator gates use `tt_ttsim::fork_scope`; parent library work that could hold
locks across a fork uses `outside_fork`. Unsupported simulator features need
silicon-only gates and a divergence entry.

Measure performance on release silicon with output validation, warmups and
medians (`cargo xtask bench`). Concurrent NC timestamp exports can be invalid;
use host timing and `dataflow_stats` for streaming evidence. Update affected
coverage/checklist entries with code, log new divergences, and record benchmark
conditions/run IDs in the performance scoreboard/change log.
