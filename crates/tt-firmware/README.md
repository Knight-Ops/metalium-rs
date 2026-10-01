# tt-firmware

Bare-metal Rust for the Blackhole baby RISC-V cores (`riscv32im-unknown-none-elf`).
A **separate workspace**, excluded from the host workspace so `cargo build` at the
root does not try to build it for the host. Normally built by
`crates/tt-firmware-images/build.rs`, which embeds the images; you rarely build it
by hand.

## Images (`src/bin/`)

| Binary | Core | Link script | Purpose |
|---|---|---|---|
| `role_t0`, `role_t1`, `role_t2` | T0, T1, T2 | `link.x`, `link_t1.x`, `link_t2.x` | The unpack / math / pack role runners. Each at its core's default reset PC, so all three sit in one tile's L1. Resident in a `Session`. |
| `dm_b` | B | `link_b.x` | The data mover (`tt_isa::dm`): GDDR to and from L1, list entries, op records expanded on the tile, `COMPUTE` on its FP32 unit, `KERNEL` entries that point the resident roles at their programs and run them. |
| `eth_e1` | E1 (Ethernet) | `link_e1.x` | The chip-to-chip data mover (`tt_isa::eth::mover`). |
| `corpus` / `corpus_t0` | T1 / T0 | `link.x` | Generic single-thread program runner (T1 for ttsim, T0 for silicon). |

The program runners push from their fixed program slot, or, when
`mailbox::PROGRAM_ADDR` is non-zero, from that address, which must lie in the
program cache region. The host writes a runner's whole descriptor
(`tt_isa::mailbox::Descriptor`) before every run: L1 survives between processes on
silicon, so a field left unwritten is the previous process's.
| `heartbeat`, `sfpu_mul` | T0, or the corpus core | `link.x` | The step 3 and step 4 bring-up gates. |

`src/lib.rs` is the runtime (entry, `.bss`, status/heartbeat/panic words in the L1
mailbox); `src/corpus.rs` is the shared body of the program runners. `build.rs`
picks each binary's link script and passes the mailbox base from `tt-isa` as a
`--defsym`.

## Build by hand

```bash
cargo build --release --manifest-path crates/tt-firmware/Cargo.toml
cargo clippy --manifest-path crates/tt-firmware/Cargo.toml \
  --target riscv32im-unknown-none-elf --bins --lib -- -D warnings
```

`.cargo/config.toml` sets the target and `-C target-feature=-c`.

## Gotchas

- **No compressed instructions.** Their encoding space is reused by `.ttinsn`: a
  compressed instruction is a Tensix push, not a smaller instruction.
- **No `ebreak` on panic.** A panic publishes `status::PANICKED` and a code, then
  spins, so the host can say why.
- **No `fence.i`** (Zifencei is unimplemented). Code is written before reset is
  released; leaving reset invalidates the I-cache.
- **No core identity at run time** (`mhartid` reads 0): it comes from the link script.
- `tt-firmware-images/build.rs` disassembles every image and refuses instructions
  Blackhole cannot execute (compressed, `lr`/`sc`, `fdiv`, `fsqrt`, the `fmadd`
  family, `fence.i`, `pause`, unlisted `fence` orderings, undecodable words).
  Executing an unimplemented instruction is undefined behaviour, not a trap.
