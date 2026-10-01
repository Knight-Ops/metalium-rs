# tt-isa

Blackhole hardware definitions shared by the host and the firmware: instruction
encoders, register and memory maps, coordinate spaces, and the host/firmware
protocols. `no_std`, `#![forbid(unsafe_code)]`, zero dependencies, because it is
compiled for both the host and `riscv32im-unknown-none-elf`. Anything that cannot
build for both does not belong here.

## Modules

| Module | What it holds |
|---|---|
| `isa` | The Tensix instruction set. `isa::generated` is produced by `cargo xtask gen-isa`; each encoding carries a `Provenance` (Blackhole, shared, Wormhole-only `UNVERIFIED`, or measured). Wormhole forms are in `isa::generated::defs::wormhole`. |
| `cfg` | Backend-configuration fields. `cfg::generated` is produced by `cargo xtask gen-cfg` from `cfg_defines.h`. |
| `backend`, `sfpu`, `matrix`, `sync` | Hand-written layers over the encodings: config writes and unit waits, SFPU hazards, `SrcA`/`SrcB` bank ownership, Tensix semaphores. |
| `noc` | NoC coordinate spaces (typed per NoC), NIU registers, `noc::grid::Tensix` (the surviving columns). |
| `tensix` | Tile memory map, baby RISC-V cores (`Core`), reset PCs, soft reset. |
| `arc` | The ARC telemetry tags: how to ask a chip what it is (harvesting, translation). |
| `dram` | GDDR6 channels (`Dram`, `DramRange`); only a chip's grid can produce a range. |
| `eth` | Ethernet tile register map, firmware-owned L1, and the `eth::mover` protocol. |
| `mailbox` | The L1 contract between host and firmware: status words, role mailboxes, program region. |
| `dm` | The RISCV B data mover protocol: descriptors, list entries (`op::READ`, `WRITE`, `LIST`, `COMPUTE`, `KERNEL`, `WAIT`, ...), element-wise `kind`s. |
| `dm::record` | Op records (`GATHER`, `SCATTER`, `ELTWISE`, `SUM`): a whole op in a few list entries, `expand`ed on the tile and, first, on the host. |
| `l1` | The fixed L1 region map (`REGIONS`: images, mover list, data arena, mailboxes, program slots, trace, program cache), checked at compile time. Kernels never address `l1::DATA` directly; `tt_kernels::l1` places buffers there. |
| `tile` | Tile layout in L1 (`TileDescriptor`, `TileImage`). |
| `numerics` | Bit-exact models of hardware arithmetic (`fma_bh`, a port of the spec's `fma.c`). |

## Test

```bash
cargo test -p tt-isa
```

`tests/llk_crosscheck.rs` checks the generated encodings against tt-metal LLK's
Blackhole `TT_OP_*` macros.

## Gotchas

- Never edit `src/isa/generated.rs` or `src/cfg/generated.rs`. Regenerate with
  `cargo xtask gen-isa` / `gen-cfg` (needs `cargo xtask fetch-spec`); CI runs both
  with `--check`.
- A fact from a Wormhole page is a hypothesis on Blackhole. Several Matrix Unit
  fields sit one bit lower on Blackhole than Wormhole draws them; those are measured
  layouts in `xtask/src/gen_isa/Bits32_BH.lua`.
- Constants cite their specification page. Keep that when adding one.
