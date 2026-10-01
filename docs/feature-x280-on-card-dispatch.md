# Feature request: the X280 cores as an on-card host

Status: **proposed, not scheduled.** A design note for a decision, written 2026-10-01
alongside Phase 10's trace work (`hardware-coverage.md` X4d). Nothing here is
implemented.

## The problem it would solve

Every number Phase 10 measured past the kernels themselves is the host being far away:

| Cost | Measured | Where |
|---|---|---|
| A burn-tt call's round trip to the server thread | 32-49 us | `ttsim-divergence.md` row Z |
| Building and enqueueing one op's lists on the host | ~10-25 us an op, ~0.4 ms of a training step | rows V, Z |
| A small download (the logits, 2.5 KB) past its sync | ~140 us, uncached 4-byte MMIO reads at ~5 MB/s | rows M, Z |
| A small upload (2.5 KB) | ~470 us a call | row Z |
| A host op between device ops (the `[64, 10]` loss, argmax) | a download, Flex, an upload | rows V, X |

Traces (X4d) take the first two out of a replayed stretch, but not what has to run between
replays: picking the next batch, reading the predictions, running the loss or the optimizer
step's bookkeeping, deciding whether to stop. All of that is host code today, and every
piece of it pays PCIe.

## What is on the card

Blackhole has four L2CPU tiles, each a coherent cluster of four SiFive x280 cores: sixteen
64-bit RISC-V cores with the vector extension (`BlackholeA0/L2CPUTile/README.md`). Each
tile has:

- a direct connection to a local GDDR6 tile, cached or uncached, in its own address space
  (D5, D6, D7; `MemoryMap.md`), so tensors in GDDR need no NoC hop to read;
- 256 TLB windows onto the NoC (`TLBWindows.md`), so it can reach every Tensix tile's
  L1 -- the data movers' queues and mailboxes this workspace already drives from the host --
  and the PCIe tile, and through it host memory;
- L1/L2 caches and a configurable L3.

In short, a small host on the far side of PCIe.

## What it could do, in order of payoff

1. **Drive traces without the host** (X4d's natural partner). A loop on an x280 replays a
   captured inference graph for every batch in a GDDR queue: it writes each batch's input
   into the trace's input buffer (a GDDR-to-GDDR copy, or the movers'), replays, and
   reads the output where it lands. The host's part is to fill the queue and drain the
   results; the per-batch PCIe traffic is the inputs and outputs only.
2. **Run the small ops that fall back to the host today**, next to the data: argmax,
   a `[64, 10]` softmax and cross-entropy, a scalar loss reduction, and an optimizer's
   per-step bookkeeping. With RVV, on cached GDDR, without a download.
3. **Be the dispatcher.** `tt-kernels` is `no_std`-friendly Rust in its ISA layer and plain
   Rust above it. Its session -- building lists, placing programs, enqueueing to the
   movers, the barriers -- could run on an x280 against the NoC instead of on the host
   against PCIe, with the host sending op-level requests over a ring in GDDR. That is the
   shape of tt-metal's fast dispatch (a prefetcher and dispatcher on device cores), with a
   general-purpose CPU in the dispatcher's seat.
4. **Train on the card.** With 1-3, a whole training step -- batch slicing, forward,
   loss, backward, optimizer -- runs without the host, which is where training traces
   (X4d's training section) would want to be.

## What makes it hard

From the documentation, before any measurement:

- **Reset happens once.** "The harts within each L2CPU tile can only be brought out of
  reset once. Once running, putting them back into reset requires resetting the entire
  Blackhole ASIC" (`L2CPUTile/README.md`, Reset). Software must be able to take a running
  hart back without reset: the page suggests parking harts in machine mode through RNMIs
  (`RNMIs.md`). A bug in our code there costs a board reset, as the wedged tile did.
- **Clocks.** The harts must leave reset with the L2SYS clock low, which means programming
  PLLs through the ARC tile, then raising it (`tt-bh-linux`'s `clock.py` is the reference).
- **Memory behaviour.** The L1 data cache allows one outstanding miss at a time
  (`RUST_IMPL_PLAN.md`'s L2CPU notes), which makes NoC-backed loads slow; the work has to
  live in the local GDDR tile, cached, and touch Tensix L1 only for doorbells and
  mailboxes. Coherence between the x280's caches and NoC writes into GDDR has to be
  designed, not assumed (`Caches.md`).
- **No simulator.** ttsim models Tensix, Ethernet and DRAM; nothing here says it models the
  L2CPU tiles, so every gate would be silicon-only (to check first).
- **Toolchain and runtime.** Bare-metal `riscv64gcv` Rust (`no_std`), our own trap
  handlers and the park mechanism, a loader from the host, and a way to see what a hart is
  doing when it goes wrong. tt-bh-linux runs Linux there; we would not need to.
- **Safety.** A hart has the whole NoC. The rule this workspace learned on the wedged tile
  and the GDDR posted writes -- verify alone, on a healthy tile, before anything depends
  on it -- applies with more force.

## A plan, if it goes ahead

Each step gated on silicon (no ttsim), each one useful alone:

- **L1 Bring-up and park.** Release one tile's harts at a low clock, run a heartbeat in
  machine mode, and park and re-take them by RNMI repeatedly -- the park mechanism is the
  gate, since without it the first bug is a board reset. Raise the clock.
- **L2 NoC from an x280.** Program a TLB window and write a Tensix L1 word the host reads
  back; read a GDDR range through the local tile and through the NoC. Measure both.
- **L3 Drive a mover.** Enqueue a list on a Tensix tile's data mover (`tt_isa::dm`
  `QUEUE_*`) from an x280 and wait for it -- the session's enqueue path without PCIe.
  Measure the per-list cost against the host's.
- **L4 Replay traces from a queue.** X4d's replay driven from a GDDR batch queue: the
  inference loop with the host only filling the queue. Gate: MNIST inference bit for bit,
  per-batch host traffic the inputs and outputs only, images/s measured.
- **L5 Small ops in RVV.** argmax and the small softmax/loss on the x280, bit-for-bit or
  within a derived bound of Flex.
- **L6 Dispatch.** The session's list building on the x280; the host sends op requests.

## Decisions it needs

- Whether the reset-once risk is acceptable on these cards (both have to be in service).
- Whether training on the card (step 4 above) is a goal, which decides how far L5-L6 go.
- Which L2CPU tile, and so which GDDR tile, is the dispatcher's (D5-D7 hold our tensors too).

## References

- `vendor/tt-isa-documentation/BlackholeA0/L2CPUTile/` (README, MemoryMap, TLBWindows,
  Caches, RNMIs, MSICatcher).
- `docs/RUST_IMPL_PLAN.md`, "L2CPU tiles (optional on-chip host)".
- `docs/tt-metal-concepts-review.md` G8: tt-metal's fast dispatch and traces.
- `docs/hardware-coverage.md`, "Out of scope": the earlier reason for leaving these tiles
  out, which this request revisits.
