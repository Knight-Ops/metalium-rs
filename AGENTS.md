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
  `burn-tt-conformance`: Burn's own backend tests against `burn-tt`
  (`cargo xtask conformance`; dev-only). `tt-mnist`: application/model workloads.
  `tt-models`: pretrained models (ResNet-18, ViT-Tiny, SmolLM2-135M), strict safetensors loading and
  mixed-precision plans, checked against Flex (`cargo xtask fetch-weights`; dev-only).
  `xtask`: fetchers, generators and runners.

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
| What to work on next (open items)       | [roadmap.md](docs/plans/roadmap.md)                                                                                                      |
| Setup and task options                  | [README.md](README.md), [xtask/README.md](xtask/README.md)                                                                               |
| New kernel/op; completion criteria      | [new-op-definition-of-done.md](docs/plans/new-op-definition-of-done.md), [burn-op-notes.md](docs/plans/burn-op-notes.md)                  |
| Fixing a failing Burn conformance test  | [conformance-to-feature.md](docs/plans/conformance-to-feature.md), then the crate [README](crates/burn-tt-conformance/README.md)           |
| Burn backend changes                    | [burn-native-cutover.md](docs/plans/burn-native-cutover.md), [burn-backend-parity.md](docs/plans/burn-backend-parity.md)                 |
| Scheduler, credits, traces or recovery  | [streaming architecture](docs/learnings/streaming-dataflow-architecture.md), [traced execution](docs/plans/traced-execution.md)          |
| Hardware access/bring-up                | [silicon operating notes](docs/learnings/silicon-operating-notes.md), [implementation checklist](docs/completed-plans/phases-0-10-checklist.md) |
| Simulator refusal or numerical mismatch | [ttsim-divergence.md](docs/learnings/ttsim-divergence.md); read refutations as well as original findings                                 |
| Performance changes                     | [performance-scoreboard.md](docs/learnings/performance-scoreboard.md) (current), [performance-log.md](docs/learnings/performance-log.md) (history) |
| Architecture/specification rationale    | [phases-0-10-roadmap.md](docs/completed-plans/phases-0-10-roadmap.md), [tt-metal concepts review](docs/learnings/tt-metal-concepts-review.md)                |

### Codebase Subsystem $\leftrightarrow$ Documentation Matrix

| Crate / Layer           | Relevant Learnings (`docs/learnings/`)                                                        | Relevant Plans (`docs/plans/`)                                                          |
| :---------------------- | :-------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| `tt-isa`                | `ttsim-divergence.md`, `riscv-guide-review.md`                                                | `roadmap.md`, `new-op-definition-of-done.md`                                                              |
| `tt-device` / `tt-kmd`  | `silicon-operating-notes.md`, `ttsim-divergence.md`                                           | `roadmap.md`                                                                                         |
| `tt-layout`             | `performance-scoreboard.md`                                                              | `roadmap.md`, `new-op-definition-of-done.md`                                                         |
| `tt-firmware`           | `performance-scoreboard.md`, `streaming-dataflow-architecture.md`, `silicon-operating-notes.md` | `roadmap.md`, `traced-execution.md`, `kernel-fusion.md`                                              |
| `tt-kernels`            | `streaming-dataflow-architecture.md`, `tt-metal-concepts-review.md`                           | `roadmap.md`, `traced-execution.md`, `kernel-fusion.md`, `new-op-definition-of-done.md`              |
| `burn-tt`               | `tt-metal-concepts-review.md`                                                                 | `roadmap.md`, `burn-backend-parity.md`, `burn-native-cutover.md`, `traced-execution.md`, `burn-op-notes.md` |
| `tt-tests` / `tt-mnist` | `ttsim-divergence.md`, `performance-scoreboard.md`                                       | `roadmap.md`, `new-op-definition-of-done.md`                                                         |
| `xtask`                 | `silicon-operating-notes.md`                                                                  | `roadmap.md`, `burn-backend-parity.md`                                                               |

## Build and check

Run from the repository root. Toolchain/targets are in `rust-toolchain.toml`.
Pinned inputs live in `vendor/`; fetch missing inputs with `cargo xtask fetch-ttsim`,
`fetch-spec`, `fetch-mnist` (needed by `tt-mnist`), `fetch-burn-tests` (needed by
`cargo xtask conformance`), `fetch-weights` (needed by the `tt-models` tests and the model-ladder
gates, which are ignored without `--features weights`), or `fetch-kmd` (driver ABI test). `PINS.toml` owns revisions/hashes; do not bypass digest checks.

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
cargo clippy --manifest-path crates/tt-firmware/Cargo.toml \
  --target riscv32im-unknown-none-elf --bins --lib --features encoded-guard -- -D warnings
```

The mover has two read modes (`fast-reads`, see below); a change to `mover.rs` or `tt_isa::dm::encoded`
is run through the encoded-read gate and the differential gate in both: `cargo test -p tt-tests --test
step172_encoded_reads --test step173_mover_differential` and the same with `--features tt-tests/fast-reads` (and
`tt-tests/encoded-guard`). An encoded read's checksum is bound to a per-start key, the tile, the chip and the op
(`encoded::Context`); the host's `validate` also runs the independent `guard`; `enqueue_checked` validates every
`READ_ENCODED`, and only the `#[doc(hidden)] enqueue_raw` sends an entry unchecked (for gates that forge one).

Arithmetic, routing or execution-order changes also need the training regression:

```bash
cargo test -p tt-tests --features e2e --test step12_mnist
```

Changes to `burn-tt` operations or routing also need Burn's own backend tests, whose result
must match `crates/burn-tt-conformance/expected-failures.toml` (a fixed test leaves the list):

```bash
cargo xtask conformance --release
```

**Never execute silicon tests with plain `cargo test`.** Use the isolated runner:

```bash
cargo xtask silicon --release --device 0 --filter <gate>
cargo xtask silicon --smoke --release --device 0
```

Validate on **one card** (card 0): the same single-card test on the other card
adds no information. Use `--device all` only for work that needs both cards:
mesh/multi-card ops, Ethernet, and larger models that span the two.

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
| `burn-coverage`    | `docs/plans/burn-op-coverage.md`      |
| `TT_BLESS=1 cargo test -p tt-tests --test probe_ttinsn_push ttinsn_programs_are_current` | `crates/tt-firmware/src/ttinsn_programs.in` |

Burn routing is `OVERRIDDEN` in `xtask/src/gen_burn.rs`; an unsupported method needs a
`[-]` disposition in `xtask/src/burn_coverage.rs`. Historical
`gen-burn-delegate` instructions are obsolete. Measured Blackhole encodings belong
in `xtask/src/gen_isa/Bits32_BH.lua` with evidence in `measured.rs` and a gate.
Wormhole-only documentation does not establish Blackhole behavior or encoding.

## Preserve these contracts

- Ensure that documentation stays up to date: record empirical hardware discoveries in
  `docs/learnings/silicon-operating-notes.md`, simulator quirks in `docs/learnings/ttsim-divergence.md`,
  current performance rows in `docs/learnings/performance-scoreboard.md` and run history in
  `docs/learnings/performance-log.md`. When working on features or ops, update
  `docs/plans/roadmap.md` and the relevant feature plan.
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
- SFPI (Tenstorrent's GCC toolchain, `riscv-tt-elf-g++`, which compiles C++ `vFloat` kernels to SFPU
  instructions) is a development-only cross-check of SFPU encodings and instruction counts, in the
  same class as the C/C++ specification models: never a dependency of a shipped crate, never the only
  oracle (the specification models still predict results), and any gate that runs it must first pin its
  version and hash in `PINS.toml`. No gate uses it yet. Recipe:
  [new-op-definition-of-done.md](docs/plans/new-op-definition-of-done.md#cross-check-against-sfpi-development-only).
- New ops need an independent oracle, a simulator gate with a meaningful
  negative control, silicon validation on one card (both only for
  mesh/multi-card ops), native Burn routing,
  residency/padding checks and a smoke entry. Follow
  `docs/plans/new-op-definition-of-done.md`. Re-bless `crates/tt-tests/tests/golden/mnist_reduced.txt` with
  `TT_BLESS=1` only for an explained, deliberate arithmetic change.

## Avoid hardware failures

Use checked APIs and `Session` bring-up. Discover each chip's grids from ARC
before tile access; never probe surviving coordinates or copy ttsim's full grid
onto silicon. Harvested-tile accesses can hang the NoC and reboot the host. The B mover is built in one of two
read modes, chosen at build time by the firmware feature `fast-reads` (off by default; `cargo xtask
silicon --feature fast-reads`). The default **safe** image decodes and checks every read on the card
and refuses `READ_ENCODED` as an unknown op. A **fast** image runs host-encoded reads (`dm::encoded`):
validated in full on the host (`encoded::validate`), integrity-checked on the card (`encoded::verify`,
a checksum), and with `encoded-guard` also semantically guarded on the card (`encoded::guard`). The
image reports its mode in `dm::CAPS`, which the host reads; never queue an encoded entry that did not
pass `validate` (`enqueue_checked` trusts its caller), and never send a hand-made one to silicon whose
refusal you have not checked on ttsim.
Keep one `Device` per chip because power policy is chip-wide. Local data RAM
cannot answer NoC accesses while its owning core is in reset.

Preserve posted-write readback fences, NoC endpoint ownership, checked alignment
and in-flight limits. Firmware L1 polls need fences. Preserve the build-time
instruction gate; do not emit compressed instructions, `fence.i`, `pause` via
`spin_loop()`, or `ebreak` for panic. The one exception is the opt-in `ttinsn_t1`
experiment image (cargo feature `ttinsn-experiment`, off by default, never in a
shippable build), whose `.ttinsn` words the gate validates word by word, and which can also write
`.ttinsn` code to L1 at run time and invalidate its own instruction cache
(`tt_firmware::ttinsn`, which checks every word it emits)
(`docs/plans/ttinsn-push-experiment.md`); the gate refuses data directives in code
for every image. Ethernet customer code runs on E1; leave E0
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

<!-- scryer:start -->
# Code Intelligence (Scryer)

This repository is indexed by **Scryer** (Rust, Python, TypeScript). When exploring or about to edit code, use it instead of grep and whole-file reads, and **load the `scryer` skill (`.agents/skills/scryer/SKILL.md`) for the full playbook**.

- Don't know the name → `search_symbols`. Know it, or a crate path like `tokio::sync::Mutex` → `inspect_symbol`. Know a file and line → `get_enclosing_scope` / `resolve_definition`. Want a file's shape → `get_file_outline`. About to change something → `calculate_blast_radius` + `query_adrs`.
- Grep is still right for string literals, log and error messages, comments, environment variables, non-code files (Markdown, JSON, YAML, TOML, Dockerfiles), or when the index doesn't resolve a symbol.
- A list marked `complete` is exhaustive over the indexed files: don't re-grep it unless a `CAVEAT` applies. A cut list says `complete: false` and the next call. An empty result with a `notes` entry means the name or path didn't match, not that nothing uses it; `INDEXING` or `STALE` means partial, so retry shortly. The index follows your edits. Lines are 1-based, columns 0-based.
- Not seen: code behind `cfg` is indexed whether or not it compiles; trait-object and closure calls resolve by name; build-script output; languages other than Rust, Python and TypeScript. "Who calls X?" is `find_references(X)` (default `view: callers`: each caller once, with its line), not grep plus a lookup per hit. Batch where you can: `get_enclosing_scope(at: ["src/a.rs:12", ..])` takes grep hits across files (or `lines: [..]` in one file), `inspect_symbol(symbols: [..])`.
- Antigravity / Gemini CLI: call tools through the gateway, `call_mcp_tool(ServerName: "Scryer", ToolName: "inspect_symbol", Arguments: { "symbol": "MySymbol" })`. No MCP tools at all? Use the CLI: `scryer --help`, `scryer inspect-symbol MySymbol`, `scryer get-file-outline src/main.rs`.
<!-- scryer:end -->
