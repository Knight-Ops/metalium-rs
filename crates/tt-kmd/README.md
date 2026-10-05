# tt-kmd

The silicon `Transport`: one Blackhole reached through `/dev/tenstorrent/N` and
Tenstorrent's kernel driver (`tenstorrent` module, ioctl API version 2). The
production counterpart of `tt-ttsim`. Shippable; depends on `tt-isa`, `tt-device`
and `libc`.

## Key items

| Item | What it does |
|---|---|
| `Kmd::open(index)`, `Kmd::enumerate()` | Open a card and check the loaded module's API version against `PINNED_API_VERSION`. List cards under `DEV_DIR`. |
| `Kmd::allocate_tlb` / `free_tlb` | TLB windows arbitrated by the driver across processes. |
| `Kmd::set_cleanup_write` | A NoC write the driver performs when the descriptor closes, however the process ends (used to hold a tile's cores in reset). |
| `CleanupWrite::register` | The same, on a descriptor of its own that maps no BAR (cheap; one per extra tile). |
| `Kmd::lock` / `unlock` | Driver locks; the soft-reset read-modify-write has no other cross-process protection. |
| `Kmd::telemetry` | sysfs telemetry attributes. |
| `auto_reset_timeout()` | Reads the driver's ARC watchdog parameter (`AUTO_RESET_TIMEOUT_PARAM`). |
| `abi`, `ioctl`, `mapping` | Hand-written ioctl structures, the calls, and BAR mappings. |

## Test

```bash
cargo xtask fetch-kmd     # vendor/ioctl.h at the pinned tag
cargo test -p tt-kmd
```

`tests/abi_layout.rs` compiles the real `vendor/ioctl.h` with `cc` and compares
every `sizeof`/`offsetof` with `abi.rs`. It skips (and says so) if the header or a C
compiler is missing. Everything that talks to a card runs through
`cargo xtask silicon` in `tt-tests`.

## Gotchas

- **`Transport::tick` is a no-op here.** Poll budgets in simulated cycles are
  meaningless on silicon; use wall-clock budgets.
- **Nothing is isolated.** The chip keeps what the last run left (`Dst` has no
  reset value, L1 is not cleared, cores keep running). `tt_kernels::session` resets
  tiles; ad-hoc code must scrub deliberately.
- **The ARC watchdog.** With `auto_reset_timeout` nonzero, a NoC hang becomes a chip
  reset that drops the PCIe link (on a passed-through VM, the host). The harness only
  reports it; fix the access that hangs rather than gating on it.
- `TENSTORRENT_IOCTL_GET_HARVESTING` is a stub in driver 2.11. Harvesting comes from
  ARC telemetry (`tt_device::Device::tensix_grid`).
- See [`docs/learnings/silicon-operating-notes.md`](../../docs/learnings/silicon-operating-notes.md).
