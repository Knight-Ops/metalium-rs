# tt-ttsim-sys

Raw FFI bindings to Tenstorrent's `libttsim.so`, the Blackhole simulator.
**Development and test only**: listed in `DEV_ONLY` in `xtask/src/ship.rs`, and
`cargo xtask check-no-sim-in-ship` fails if any shippable crate depends on it.

- Declares exactly the ten symbols ttsim's linker version script exports
  (`src/libttsim.map` upstream); signatures from ttsim's `docs/libttsim_api.md`.
  ttsim ships no header. Both are pinned in `PINS.toml` (`[ttsim]`).
- The library is loaded with `dlopen` (`libloading`), not linked, so the path is
  configurable and a missing simulator does not break the link step.

## Use `tt-ttsim` instead

Nothing here is safe to call directly. Every entry point belongs to a process-wide,
single-threaded, non-reentrant singleton with no context handle, and any contract
violation terminates the process via `_Exit` (no unwinding, no destructors, no panic
hook). `tt-ttsim` wraps it with a singleton guard, fork isolation and a validation
layer.

## Getting the library

```bash
cargo xtask fetch-ttsim   # vendor/libttsim_bh.so, _x2.so, _x4.so, SHA-256 verified
```
