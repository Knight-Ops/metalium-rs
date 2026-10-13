# Firmware architecture review: B, NC, T0, T1, T2 against tt-metal

2026-10-13. Prompted by `dm_b` having 172 bytes left of its 24 KiB slot. Method: read both firmwares,
size every image by symbol (`llvm-nm -C -S --size-sort`), then build and measure variants on card 0.
What was measured is in [performance-log.md](../learnings/performance-log.md#firmware-image-budget-and-core-assignment-2026-10-13-card-0);
what was only reasoned is marked **paper** below. **No live tt-metal timings exist**: the installed ttnn 0.72 wants
SFPI 7.52.0 and the machine has 7.83.0 (its JIT compile fails), and the `tt-metal-v077` tree's ttnn hangs in
device init. The tt-metal side is from its source and the kernel ELFs in its JIT cache.

## Summary

- **The size problem was a layout problem, and it is fixed.** B's entry point is hardwired to L1 offset 0, so its image
  could only grow to the next image, T0's default reset PC (0x6000). The three role images use ~8 KB each but sat in
  16 KiB slots. They are now packed against NC's slot with reset-PC overrides (`tt_isa::l1::roles`), and B's slot is
  43,008 bytes (18,604 free). No speed change on silicon, 1,030/1,030 smoke tests pass on card 0.
- **`opt-level` 2 is free; `s`/`z` are not.** Per-entry mover cost and role push rate are identical at 2 and 3, and
  `dm_b` is 1,744 bytes smaller. At `s`/`z` the mover is 1.6x slower per entry and the roles' push loop is ~7x slower.
  Not adopted yet (only two benches measured); worth a follow-up with an end-to-end run.
- **Our design is not "tt-metal's firmware with more in it".** tt-metal ships a ~5 KB resident firmware plus per-program
  JIT kernels; we ship fixed interpreters. The ~350 cycles per mover entry are mostly decode and register setup, which
  a specialised kernel would remove, but at a cost in per-launch binary traffic and tooling. Nothing measured says
  this is where our time goes: the models are bound by host submission (220 us to 1.1 ms per op), not by the mover.
- **Not done:** the loadable-kernel prototype (V5), the cold-code-at-`s` split (V2) and a segmented image (V4). V3a
  removed the size pressure that motivated V2/V4; V5 is only worth building to measure specialisation speed, and is
  several days. See "What was not built".

## The five cores, ours against tt-metal (Blackhole)

| | ours | tt-metal |
|---|---|---|
| **B** (RISCV 0) | Reader and orchestrator: GDDR->L1 on NoC0, op records expanded on the tile, fills, tilize, PCIe moves, barriers, `KERNEL`/`LAUNCH` of the roles, packets that start NC. Fixed interpreter, 24,404 B. Never writes GDDR. | BRISC firmware (~4.8 KB `.text`, 8.5 KiB slot): boot, wait for go, set up CBs, call the kernel through a function pointer, notify dispatch. Runs a writer kernel (`WriterDataMovementConfig`: BRISC on NoC1). |
| **NC** (RISCV 1) | Writer: L1->GDDR on NoC1, writing records, credits. Fixed interpreter, 14,252 B, started through a jump stub at its reset PC. | NCRISC firmware (~1.2 KB): waits for BRISC, calls the kernel. Runs a reader kernel (NoC0). No IRAM on Blackhole: all five cores execute from L1. |
| **T0/T1/T2** | Three copies of one runner (`corpus::run_role`): wait for a generation, push a host-built Tensix instruction stream from L1 (`Pusher`), wait for retirement, acknowledge. 8.0-8.4 KB each. No math is compiled in. | ~0.6-1.1 KB firmware each; the compute kernel (one source, built three times with `TRISC_UNPACK/MATH/PACK`) carries the op: matmul 2.4/3.5/3.5 KB, eltwise 1.6/0.9/1.7 KB, softmax ~5/4.8/3.7 KB. |
| Where op logic lives | Resident, selected by descriptor at run time (`Entry::decode`, 40+ op codes and records). | In the kernel, selected at compile time (compile args, defines): one ELF per op/shape/feature combination (53 cached matmul variants). |
| Per-launch traffic | A list of 32-byte entries; programs live in an L1 cache (`PROGRAM_CACHE`, ~240 KiB) and are reused. | Kernel binaries, runtime args and CB config re-sent for every launch into a ~69 KB config ring; traces replay recorded command streams from DRAM. |
| CB synchronisation | L1 credits (`dataflow::buffer`), batch-granular, polled by B/NC/roles. | Hardware stream registers (`tiles_received`/`tiles_acked`), per-tile `cb_reserve/push/wait/pop`. |
| Orchestration | B itself, fed by the host over PCIe (4.25 us round trip). | Separate prefetch/dispatch tiles (`cq_prefetch` 32 KB, `cq_dispatch` 14 KB) multicast binaries, launch message and go signal. Documented floor: single-digit us per op, <6 us traced. |
| Code placement | Fixed slots; B's entry is hardwired to 0. | Kernel text linked after firmware, XIP-relocated into the config ring; no fixed address. |

Ours is the reverse of tt-metal's default reader/writer assignment (B reads, NC writes). The mover costs the same on both cores
(performance-log, "NC costs the same as B per entry"), so the assignment is a convention, not a performance lever.

## Why `dm_b` was big (measured, 24,404 B)

Records on the tile (`run_record` + helpers) ~6.7 KB (28%); `exec` with `Entry::decode` inlined 5.3 KB (22%); NoC issue 2.1 KB;
packets and trace replay 2.2 KB; main loop 1.9 KB; tilize/untilize 1.7 KB (cold); transpose/broadcast/fill 1.2 KB; runtime
(memcpy, panics) ~1 KB. Build: `opt-level` 3, fat LTO, one codegen unit, `panic = "abort"`, no compressed instructions
(the encoding space is `.ttinsn`'s). Every op family is resident because the host drives a fixed interpreter; that is the
design, not waste. Waste that exists: `Descriptor::decode` is inlined and also outlined (236 B), `run_record` has
5 bounds checks and 6 `memcpy` calls. Lessons already in the operating notes: one record arm costs ~760 B, `u64`
arithmetic on rv32im is expensive, LLVM unrolls two-iteration loops.

## Measured variants (card 0, release)

| Variant | `dm_b` | Result |
|---|--:|---|
| V0 shipped layout, `opt-level` 3 | 24,404 (172 free) | WAIT 0.31 us, 64 B read 0.48 us, push 2.91 cycles/word |
| V1a refuse other direction's ops before decode | 24,400 B / NC 14,348 | -4 B on B, +96 on NC. Reverted: LLVM already shared the arms; my symbol-size estimate (240 B / 1.9 KB) was wrong |
| `opt-level` 2 | 22,660 | identical speed; -1,744 B |
| `opt-level` s / z | 21,220 / 21,164 | mover 0.48 / 0.52 us WAIT (1.5-1.7x); roles 21-23 cycles/word (7x) |
| **V3a roles repacked (adopted)** | 24,404 (**18,604 free**) | identical speed; B slot 43,008 B |

## The design space for five cores

Each approach: what it changes, what it should cost or buy, and whether it was measured.

1. **Status quo** (B reads and orchestrates, NC writes, T0-T2 push). Measured: ~350 cycles per ~4 KiB entry,
   of which fetch 30, decode 88, command registers 110, cap check 14, busy poll 9, NIU writes 10, issue 6, return/trace 47
   (scoreboard, stage table): the NIU is nearly free, the software around it is not. Empty kernel B->T->B 0.42 us.
2. **Same roles, size hygiene, `opt-level` 2.** Measured above. Cheap, worth taking once an end-to-end run confirms it.
3. **Relocated/repacked images (V3a).** Measured and adopted. A further step available if B needs more than 42 KiB:
   a 2-instruction jump stub at 0 (as NC has, `dm::nc::stub`) and the image anywhere, at the price of taking the space from
   the data arena (32 KiB is 3.6% of it). **Paper.**
4. **Cold code at size optimisation (V2).** Cold paths are ~14% of the image; at `s` they would shrink by about the whole-image
   13%, i.e. a few hundred bytes. **Paper**, and unnecessary after V3a. Not built.
5. **Loadable mover kernels, tt-metal style (V5).** A resident core (list loop, plain reads, credits, launch) plus op families
   as L1 overlays called through a table; B can invalidate its own icache (T1's was measured at ~330 cycles, B's not yet).
   Buys: specialisation by constant folding. Upper bound from the stage table: decode + register setup is 198 of ~350
   cycles per entry, so specialisation could save at most ~55% of the per-entry cost, and only on the entries where the
   host cannot already batch them (16-64 KiB entries are NoC-bound already: 54-66 GB/s at 64 KiB). Costs: overlay shipping
   and residency, a second build path, a function-table ABI. **Paper; not built.**
6. **Role rebalancing.** (a) Swap to tt-metal's convention (NC reads, B writes): no expected gain, the cores cost the same.
   (b) Move control-plane entries (`KERNEL`, `LAUNCH`, barriers, pairs) to NC, which has 10 KB free, so B is only a reader:
   unmeasured and constrained by NoC endpoint ownership (SYS-1419: each GDDR endpoint is owned by one NoC; NC's reset
   and cache-invalidate limits: B or the host must invalidate NC's icache). **Paper.**
7. **Compute side: fewer words, not smaller firmware.** The roles are push-bound (2.9 cycles/word; `gelu` 38,053 cycles per
   tile at 3.31 cycles/word). The levers are fewer pushed words: MOP/replay-heavy streams, and `.ttinsn` fused pushes
   (measured 2.77x on precise `gelu`, parked) or an SFPU micro-op interpreter on T1 (design only,
   [kernel-fusion-architecture.md](../learnings/kernel-fusion-architecture.md) section 6.3; no such interpreter exists in the firmware today).
   tt-metal compiles the same pushes into TRISC kernels; the image sizes are comparable (matmul kernel ~3.4 KB a thread
   against our 8 KB runner), so firmware size is not what limits compute.
8. **Dispatch off the critical path.** tt-metal's biggest structural difference is not its core roles but its dispatch
   tiles. Our models spend 220 us (SmolLM2) to 1.1 ms (ViT) of host time per op and the device is busy 12.6% of wall time
   with 16 queued small matmuls; this is where tt-metal's design and the X280 proposal
   ([x280-on-card-dispatch.md](x280-on-card-dispatch.md)) act, and no reshuffling of the five cores does.

## Recommendation

1. Keep the fixed-interpreter design and the B-reads/NC-writes split; keep V3a. B has ~18 KiB of room: the packed-BF16 row
   select (B-side branch, ~400 B) and the rank-3+ and other-dimension selects can use it.
2. Take `opt-level` 2 after an end-to-end run (`cargo xtask bench` default selection, MNIST step, one model) shows nothing
   moved. 1.7 KB for free.
3. Do not build overlays (5) for size. Build the V5 prototype only if per-entry mover cost becomes the measured bottleneck;
   today the host is.
4. Use SFPI only as a development oracle for SFPU encodings and instruction counts, not as a build dependency
   ([recipe and a checked example](../plans/new-op-definition-of-done.md#cross-check-against-sfpi-development-only)).
5. Prioritise by what the numbers say: dispatch/host submission (8) > fewer pushed words (7) > mover specialisation (5)
   > core reshuffling (6).

## What was not built, and the evidence gap

- **V5 loadable-kernel prototype, V2, V4:** not built (see above). The "specialisation upper bound" in item 5 is
  arithmetic on the recorded stage table, not a measurement.
- **Live tt-metal numbers:** none. `crates/tt-tests/bench/ttnn_reference.py` (add, matmul, embedding; eager, batched and
  traced) is written and untested end to end; it needs a ttnn whose SFPI matches the machine's.
  The one `ttnn_reference.py --only "add f32 64"` attempt with the venv's ttnn printed the compile failures and crashed
  (card healthy afterward); the v077 build hung after device open and was killed by timeout (card healthy afterward).
- **Only two benches** were run for the `opt-level` and layout variants (mover read shapes, null kernel launch), plus the
  smoke selection for correctness of the layout change.
