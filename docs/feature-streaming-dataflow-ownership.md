# Streaming dataflow ownership

## Status (2026-10-03)

### Consolidation decision

Streaming ownership is the only GDDR compute scheduler. This is an explicit
architectural choice despite known fresh-execution regressions, not a claim that
the original nonregression gate passed. The legacy wave scheduler, the B-only
pipeline and the host-queued NC scatter scheduler are removed. `ExecutionMode`,
`set_execution_mode`, `enable_dataflow` and `set_scatter_mover` are gone;
`Session::enable_dram(b, nc)` takes both firmware images. Burn rejects the retired
`TT_EXECUTION` and `TT_SCATTER` settings with migration guidance. `TT_PIPELINE=0`
keeps NC ownership and serializes slot reuse; `TT_BATCH=0` uses the same queue and
synchronization/recovery path.

B is a pure GDDR reader and NC a pure writer (`dm::Mover::permits`, the one
direction table the host checks every list against and each firmware image
compiles in only its own half of). Standalone transfers (uploads, copies, block
copies, row gathers and writes, padding fills) are therefore reader/writer
packets too, with no kernel in them: B reserves a credit on the **transfer**
channel (`dataflow::Stream::Transfer`, B to NC), reads a batch into L1 and
pushes it; NC waits, writes it out, and pops it once the writes are
acknowledged. Consecutive transfers of one depth share a packet (depth one: the
next reserve already waits for the last pop; depth two: halves of a staging
area alternate and a drain, `RELEASED` kind 2, separates transfers). A packet
has no `LAUNCH` and no `KERNEL_WAIT`; B's slot is reported done when NC's last
write is acknowledged, which is what orders a later list after these writes,
and a multi-unit barrier is its trailer. Captured traces replay them as
`PAIR_CALL`s. Downloads and the barrier remain B-only lists: their writes are to
host memory (`HOST_WRITE`, PCIe), not GDDR. NC's NoC #0 requests share B's
initiator registers, so NC only ever writes on NoC #1 beside a reading B.
Descriptor setup and low-level control kernels remain. GDDR compute that cannot
form a resident ownership region is rejected with an error naming the preceding
list and the reason (a role program over half the effective program cache, or a
kernel not between a gather and a scatter); there is no B-only fallback.

### Ownership

B reads on NoC #0, T0/T1/T2 are resident streaming roles, and NC writes on NoC #1.
A B that reads and an NC that writes is the whole of each image's job (see below).
The host sends one `PAIR` packet and commits only B's queue; B starts NC locally.
On success, B's completion joins NC and the roles (its `KERNEL_WAIT`). On a
failure B aborts the region and waits (bounded, about 60 ms) for NC to stop
before reporting its slot, since NC reads the writer entries out of B's ring; the
host then resets the tile, roles included. Fresh multi-tile barriers are a packet
trailer run after NC completes. NC uses transaction ID 4 (B uses 2) and fetches
traced writer commands on NoC #1, an exception to "reads on NoC #0": on B's NoC #0
the fetch would race B's initiator registers.

Compatible gather/compute/scatter batches share one role generation and three
cached streaming scripts. Arithmetic bodies, loops, operand reuse, fidelity and
traversal order are the per-op builders'; there is no cross-operation fusion. A
credit is one contiguous compute-sized **batch**, not a page. Depth is one (the
full arena) or two (alternating halves). T0 releases inputs after unpack
retirement; T2 publishes outputs after pack retirement; NC releases outputs after
acknowledged writes.

Before a gather reuses a staging slot, B waits for the batch that last used the
slot to be **packed** (T2 has stopped writing L1). It also waits for NC's
**release** of an older batch only if the gather's L1 footprint meets what that
batch's scatter reads, or either footprint is not modelled (anything but plain
or transformed GDDR moves and fills). Otherwise the gather overlaps NC writing
the previous batch out, at depth one too. `dataflow_stats` counts both waits.

The three roles keep a retirement barrier between batches, protecting the shared
configuration and semaphore state; removing it needs a separate proof of
backend-state ownership. The last batch of a region skips it: the region's
`KERNEL_WAIT` joins the roles instead.

Every credit, release, retirement and join wait fences and reads the abort word
on each poll, and runs the full peer check (role panics, mover queue errors and
panics) on the first poll and every 64th.

### Packet rules

The host checks every packet, fresh or captured (`tt_kernels::dm::check_pair`):
each credit names the header's capacity, the reader holds only input credits,
and the writer section holds only GDDR writes and output credits, so NC never
issues on NoC #0. The control words live at a fixed address
(`tt_isa::dataflow::INPUT..END`, in NC's mailbox page past its fields); the
session's plan check only confirms the firmware constants agree with themselves.
Counters wrap modulo 2^16 and reset at every packet; a region holds at most
`GROUP_MAX` (40) batches, so on-card wrap is exercised only by the host tests.

### Traces

Traces keep the reader and writer command streams in DRAM and every script and
arithmetic body in the program cache. `PAIR_CALL` starts both directions from one
B replay; the outer and nested fetch buffers are distinct. `LAUNCH`/`KERNEL_WAIT`
generations are rebased and `LAUNCH` bodies held. Fresh and traced compute use the
same schedule. Recovery resets the tile and invalidates traces captured before.

### Profiling

Role profiles bracket whole regions; nested reader replay events are flattened
into the outer mover list. Exports can still be unbalanced with NC running; the
claim that this predates streaming ownership is not yet backed by a reproduction
on the legacy NC split. Use host timing and `dataflow_stats` as the primary
performance evidence.

### Validation

The oracle is the host, never another device schedule (`step60_streaming`,
`step56_scatter_on_nc`):

- burn-flex bit for bit on small-integer operands, for add, row broadcast,
  transposed matmul at every fidelity and source format, column sums and maxima,
  fresh and traced, one and two tiles, and in the silicon sweep and stress;
- every SFPU kind (typed and row-broadcast operands included) bit for bit to its
  host program model (`sfpu::ops::reference_op`), serialized and replayed. The
  per-kind gates hold those models to burn-flex. At 64x64 nothing pipelines, so
  a 512x512 pass repeats one kind per operand shape and signature and asserts that
  depth two ran.

On silicon (card 0 unless noted; release builds):

- the full suite, 295/295 (run `1791066327`); `step60_streaming` and
  `step56_scatter_on_nc` on both cards, 32/32 (run `1791066284`);
- mixed fresh/traced transposed matmul, addition and reduction against
  burn-flex, 120 s each: 80740 one-tile and 147549 eight-tile rounds (run
  `1791067124`);
- the sweep (run `1791067111`, host end-to-end medians, three warmups then 15
  samples, each output first checked against burn-flex) and the models, against
  the same workloads before this round of changes (run `1791060004`):

| Workload | Before (µs) | Now (µs) | Legacy best (µs) |
|---|--:|--:|--:|
| Add 2048², traced, 1 tile | 2418.7 | 2260.5 | 3271.8 |
| Add 2048², traced, 8 tiles | 343.9 | 325.1 | 457.6 |
| Add 2048², traced, 32 tiles | 239.3 | 228.6 | 281.6 |
| Add 64², fresh, 1 tile | 18.86 | 17.15 | 14.80 |
| Matmul 256³, fresh, 32 tiles | 239.6 | 239.7 | 181.1 |

  MNIST (full epoch, `tt-mnist --host`): 1.4 ms/step on one tile (burn-flex 0.5),
  0.9 on four, 91.96% (golden). Transformer (`tt-mnist --model transformer`):
  8.8 ms/step, burn-flex 4.35.

Fresh small ops still trail the removed executor (64² add 1.16x, 32-tile 256³
matmul 1.32x). On 32 tiles that matmul spends ~240 µs on the host per op, ~130 µs
of it enqueueing (segments ~1 µs, placement ~0.65 µs and list checks and writes
~1.4 µs per unit). The sweep's 32-tile rows also show overlap never engaging:
`tensor::pipelined_runs` wants 48 tiles per unit squared, a threshold tuned for
the B-only pipeline and not yet re-tuned for NC ownership.

```
cargo test --workspace --no-fail-fast
cargo xtask silicon --release --keep-going
cargo xtask silicon --release --include-ignored --filter step60_streaming::streaming_performance_sweep
cargo xtask silicon --release --include-ignored --timeout-secs 600 --filter step60_streaming::streaming_stress
```

## Historical (pre-consolidation)

The legacy executor's measurements against streaming are in
[firmware-performance.md](firmware-performance.md#streaming-ownership-rollout)
(run 1791060004: traced 2048² adds 0.74/0.75/0.85 of legacy on 1/8/32 tiles;
fresh 64² add 1.27, fresh 32-tile 256³ matmul 1.32).

## Original approved plan (historical; consolidation supersedes executor selection)

### Summary

Make the standard device execution model:
```
Host submits one work packet per tile
│
▼
B: read inputs → input circular buffers
│
▼
T0: unpack
T1: compute
T2: pack
│
▼
output circular buffer
│
▼
NC: write outputs
```
B, NC, and the compute roles remain resident. Within a compatible operation, the compute roles run a streaming program rather than being relaunched for every block.
Implement this for fresh execution, non-pipelined execution, and traces. Retain the current execution path as an explicit fallback and correctness reference.
Fixed ownership does not mean fixed kernels: shapes, arithmetic programs, batching, and buffer capacities remain configurable.

### Ownership and Runtime

#### Circular-buffer contract

Extend buffer endpoints to distinguish B and NC; retain the existing mover endpoint for legacy programs.
Implement typed producer/consumer operations: reserve space, publish pages, wait for pages, and release pages. Validate one producer and one consumer per buffer.
Allocate descriptors and counters through the existing L1 planner, including their full concurrent lifetimes. Do not reuse storage merely because stages were previously sequential.
Use wrapping L1 page counters—not the four-bit Tensix synchronization semaphores. Keep existing semaphores for internal unpack/math/pack coordination.
Require contiguous batches whose sizes divide ring capacity. Execute differently shaped remainder batches as separate drained regions.
The ownership boundaries are:

| Buffer | Producer | Consumer | When storage becomes reusable |
|---|---|---|---|
| Input | B | T0 | After the consuming unpack instructions retire |
| Output | T2 | NC | After the DRAM writes are acknowledged |

Publishing inputs requires completed reads and any padding, transpose, or broadcast transformation. Publishing outputs requires completed packing. Instruction submission alone is not completion.

#### One host submission

Replace the session’s independent B and NC enqueues with a shared work packet containing reader commands, writer commands, buffer descriptors, and compute-program references.
Store fresh packets in B’s existing command ring; both movers read the immutable packet directly from shared L1.
The host uploads the packet and commits only B’s queue. B publishes NC’s task locally and starts the compute stream.
NC waits directly on output-buffer availability. B no longer waits for every kernel and signals NC for each completed block.
Keep the packet allocated until the reader, all compute roles, and NC finish. B’s queue completion represents completion of the entire packet.
Account for packet headers and both command sections when splitting work to fit ring capacity; split only at drained region boundaries.
Keep low-level independent mover APIs for diagnostics, but prohibit concurrent host-driven NC submissions while the session owns NC.
Preserve the current NoC routing policy: B’s DRAM reads use NoC0; NC’s output writes use NoC1. Keep the shared mover implementation rather than creating direction-specialized firmware.

### Compute and Trace Migration

#### Streaming compute

Extend the resident runner in crates/tt-firmware/src/corpus.rs:160 with a separately tagged streaming-program format. Leave existing instruction and loop encodings supported.
Streaming programs contain bounded operations for:
Pushing existing Tensix instruction chunks, including their current loops.
Waiting for or reserving circular-buffer batches.
Releasing inputs and publishing outputs after backend retirement.
Repeating a compatible batch sequence and completing the region.
Start all three roles once per compatible region. Their streaming loops progress through batches independently; B does not issue a generation per batch.
Reuse arithmetic bodies, MOP/REPLAY lowering, fidelity settings, and internal synchronization. Initially generate slot-specific program variants for ring positions instead of introducing unrestricted runtime instruction patching.

#### Kernel migration

Port in this order:
Element-wise unary/binary operations, including existing broadcast handling.
Matmul using existing blocked operand reuse and accumulation order.
Reductions using their existing traversal and retained intermediate state.
The first implementation streams existing compute-sized batches, not necessarily individual tiles. This avoids sacrificing matmul reuse just to obtain small FIFOs.
Choose depth two where the existing sizing/reuse checks justify overlap; otherwise use depth one. Do not universally halve the staging arena.
Keep multi-pass operations as sequences of drained streaming regions. Unsupported configurations use the existing path and report the fallback reason. Do not add cross-operation fusion, multicast, Ethernet ownership, or new numerical K-blocking behavior in this change.

#### Traces

Extend the capture/replay machinery in crates/tt-kernels/src/session.rs:2426:
Capture reader/writer commands, streaming descriptors, and all referenced compute programs as one retained work description.
Replay through one B queue entry per participating tile; B starts NC on-device, without a second host enqueue.
Initialize buffer counters only at drained region boundaries. Reserve fresh generations for region starts and preserve queued replay ordering.
Fix legacy captured LAUNCH/KERNEL_WAIT generation rebasing and hold programs referenced by LAUNCH, not only KERNEL.
Preserve active streaming state across command-fetch chunks. Reject nested replay dispatch.
Keep the existing cross-tile barriers, but reach them only after NC has committed that tile’s outputs.
Treat streaming programs and their instruction chunks as program-cache resources: pin them during queued work and hold them throughout trace lifetime.

### Interfaces and Compatibility

Add Session::enable_dataflow(b_image, nc_image) so callers explicitly supply both firmware images without introducing a firmware-image dependency into the kernel crate.
Preserve enable_dram(b_image) as the legacy-compatible B-only entrypoint.
Add ExecutionMode::{Auto, Streaming, Legacy} and a mode setter that synchronizes before switching and rejects changes during capture.
Auto is the standard backend default: streaming for supported configurations, otherwise the existing executor with an observable fallback reason.
Streaming refuses unsupported configurations instead of silently falling back. Legacy preserves the current execution path.
Add TT_EXECUTION=auto|streaming|legacy. Preserve TT_SCATTER=b as an explicit legacy B-only override; reject contradictory forced-streaming configuration.
Preserve TT_PIPELINE=0: use serialized batch credits, released after NC finishes the batch. NC remains the writer even when overlap is disabled.
Add per-stage progress, wait-reason, and error reporting. Every software wait must detect cancellation and peer failure. Host timeout/reset recovery must stop all five cores, invalidate captured state, and reclaim resources only after execution is stopped.
Extend docs/feature-traced-pipelining.md:1 into the unified ownership/streaming design. Preserve the repository’s current unrelated work.

### Validation and Rollout

#### Correctness

Property-test page counters, wraparound, capacity limits, contiguous reservations, and illegal ownership.
Test more than three ring rotations, rings exceeding 15 pages, depth-one execution, tails, ragged padding, broadcasts, and transposes.
Compare migrated operations bit-for-bit against legacy execution across supported fidelity and layout configurations.
Test back-to-back operations, streaming/legacy transitions, and cross-tile producer/consumer dependencies.
Test repeated and queued traces with changed inputs, intervening fresh work, chunk boundaries, cache pressure, release, and reset.
Inject reader, writer, and role failures; verify bounded termination and recovery without premature buffer reuse.

#### Performance

Compare legacy B-only, current optional NC splitting, and streaming ownership on 1, 8, and 32 tiles, both fresh and traced.
Measure host submission time, device time, DRAM traffic, buffer stalls, and launch counts separately. Use host timing and counters as primary evidence until concurrent-core timestamper reliability is established.
Acceptance gates:
One host queue commit per work packet, with no NC host doorbell.
Compute launches scale with compatible regions, not batch count.
No extra operand traffic for equivalent blocked matmuls.
No correctness regressions; pass simulator gates, silicon gates, and sustained mixed-workload stress.
At least one mover-bound benchmark improves by 10%; no default-streaming benchmark regresses more than 5% in median end-to-end time against the best existing path.
Develop behind explicit selection, then enable streaming by default only after these gates pass. A failed performance gate blocks the default switch—it does not justify silently weakening correctness or removing the fallback.
