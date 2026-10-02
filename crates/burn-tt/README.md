# burn-tt

A [Burn](https://burn.dev) backend (`burn-backend` 0.21.0, pinned exactly) for
Tenstorrent Blackhole. Every op forwards to `burn-flex` on the host except the ones
it runs on the card. Shippable: no simulator dependency (the simulator engine used
by the gates lives in `tt-tests`).

## What runs on the device

With an engine that keeps tensors in GDDR (`KmdEngine`, and the ttsim engine in
`tt-tests`), on 2-D F32 tensors:

| Burn op | On the card as |
|---|---|
| `float_matmul` | `Session::matmul_dram` (operands may be transposed views) |
| `float_add` / `sub` / `mul`, `float_mul_scalar` | mover `COMPUTE` kinds; `add` of a `[1, n]` row broadcasts (`ADD_ROW`) |
| `relu`, `relu_backward` | `RELU`, `RELU_BACKWARD` |
| `float_sum_dim(0)` | `COL_SUM`, in Flex's summation order |
| `float_transpose` / `float_swap_dims` (2-D) | a view: same buffer, read transposed |
| `float_slice` of whole rows on 32-row bounds | a view, no copy |

Element-wise ops go to the device only when an operand is already there. Any F32
`float_matmul` the above does not cover (batched, or an engine without GDDR such as
`Topology::Cards`) still runs on the card, staged from the host (`Engine::matmul`).
Everything else downloads its inputs once and runs on Flex.

The hand-written ops are `OVERRIDDEN` in `xtask/src/gen_burn.rs`; the forwarding
of every other op is generated into `src/generated/delegate.rs` by
`cargo xtask gen-burn-delegate` (never hand-edit; CI runs `--check`).

## Key types

| Item | What it is |
|---|---|
| `TtBackend`, `TtDevice { chip }` | The backend marker and a chip by index. |
| `TtTensor` | A shared cell with a lazily filled host copy (Flex) and device copy (GDDR buffer). |
| `attach(device, factory)` | Starts the device's server thread and runs the `Engine` factory on it; returns an `AttachGuard`. Attaching a chip twice is refused. |
| `Engine`, `kmd_engine`, `kmd_mesh_engine` | What a device can do; the silicon engines (one card on a `Session`, or a `Fabric` of cards). |
| `Topology`, `attach_topology` | `Single { card, tile }` or `Cards { .. }`; the one call training code makes. |
| `Topology::from_env` (`TT_TOPOLOGY`, `"0"` or `"0,1"`), `tiles_from_env` (`TT_TILES`, `n` or `all`) | Environment-driven choice, used by the `tt-tests` silicon harness. |
| `tensor_traffic`, `device_traffic`, `record_transfers` | PCIe traffic accounting. |

## Test

```bash
cargo test -p burn-tt
```

Host only: `tests/delegation.rs` checks every non-device op is Flex's answer byte for
byte, `tests/server.rs` drives the server with a host engine. Device behaviour is
gated in `tt-tests` (`step11_burn`, `step19_eltwise`, `step20_many_tiles`,
`step12_mnist`).

## Gotchas

- **Device errors panic.** Burn ops return tensors, not results, so a device op on
  an unattached device or a failed run panics. It never silently falls back.
- `TT_TRACE_FALLBACK=1` prints a backtrace whenever a device tensor is downloaded
  for a host op: the way to find what is still crossing PCIe.
- `TT_PROFILE=<path>` records a device-side profile of everything an attachment
  runs -- each tile's mover lists, entries and records, and its role runs, by the
  tile's own cycle counter -- and writes it as Chrome trace JSON (Perfetto,
  `chrome://tracing`) on detach; `{chip}` in the path becomes the card. Silicon
  only: ttsim does not model the timestamper's event stream.
- `Topology::Cards` keeps nothing in GDDR yet (matmuls only, host-staged) and
  computes on one tile per card; `on_tiles` refuses more.
- Random ops run on Flex, seeded through `Flex::seed`.
