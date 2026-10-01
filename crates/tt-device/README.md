# tt-device

Host-side access to one Blackhole chip, written entirely against the `Transport`
trait so the simulator (`tt-ttsim`) and silicon (`tt-kmd`) are the same code path.
No `#[cfg]` distinguishes them. `#![forbid(unsafe_code)]`; depends only on `tt-isa`.

## Key types and modules

| Item | What it does |
|---|---|
| `Transport` (`transport`) | BAR and config-space access, TLB programming, `tick`, `is_simulated`. Implemented by `tt_ttsim::LibTtsim` and `tt_kmd::Kmd`. |
| `Device<T>` (`device`) | Owns a transport: TLB window allocator, split reads/writes, `l1_read`/`l1_write`, `traffic()`. `Device::open` verifies it is a Blackhole. |
| `PowerPolicy` | `Busy` (default): `open` sends the ARC `GO_BUSY` and `Drop` returns the chip to idle. `Manual`: caller must call `set_busy` before compute. |
| `tlb` | The 210 PCIe TLB windows (`WindowKind`). Window 201 belongs to the driver and is never handed out. |
| `telemetry` | ARC telemetry table: `Device::chip_telemetry`, `Device::tensix_grid` (harvesting). |
| `arc_msg` | ARC message queue (busy/idle and other messages). |
| `core_control` | Starting, stopping and loading the baby RISC-V cores. |
| `dram` | GDDR6 channel grid from the chip, host reads/writes by `DramRange`. |
| `ethernet` | Ethernet tiles, their L1 (firmware-owned ranges refused), TT-link writes, RISCV E1. |
| `trace` | The tile's debug timestamper buffer. |

## Test

```bash
cargo test -p tt-device
```

Unit tests only. The device gates live in `tt-tests`, which can depend on the
simulator; this crate must not.

## Gotchas

- **Read the grid before touching a Tensix tile on silicon.** Use
  `Device::tensix_grid`. A fused-off tile does not answer, the NoC hangs, and the
  recovery reset drops the PCIe link. `probe_tile` / `discover_tiles` read tile
  registers and are not safe on unknown silicon coordinates.
- **One `Device` per chip.** Busy/idle is chip-wide and not reference-counted.
- **At idle the Matrix Unit's `Src` reads are unreliable** (divergence row 48), so
  `PowerPolicy::Manual` callers must raise the chip themselves.
- Plain reads and writes refuse the local data RAM aperture (`0xFFB1_4000..0xFFB1_E000`):
  touching it with its core in reset hangs the NoC. Use `Device::local_ram_read` /
  `local_ram_write`, which check the core first.
- See "Silicon operating notes" in `docs/implementation-checklist.md`.
