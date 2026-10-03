Streaming Dataflow Ownership
Summary
Make the standard device execution model:
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
B, NC, and the compute roles remain resident. Within a compatible operation, the compute roles run a streaming program rather than being relaunched for every block.
Implement this for fresh execution, non-pipelined execution, and traces. Retain the current execution path as an explicit fallback and correctness reference.
Fixed ownership does not mean fixed kernels: shapes, arithmetic programs, batching, and buffer capacities remain configurable.
Ownership and Runtime
Circular-buffer contract
Extend buffer endpoints to distinguish B and NC; retain the existing mover endpoint for legacy programs.
Implement typed producer/consumer operations: reserve space, publish pages, wait for pages, and release pages. Validate one producer and one consumer per buffer.
Allocate descriptors and counters through the existing L1 planner, including their full concurrent lifetimes. Do not reuse storage merely because stages were previously sequential.
Use wrapping L1 page counters—not the four-bit Tensix synchronization semaphores. Keep existing semaphores for internal unpack/math/pack coordination.
Require contiguous batches whose sizes divide ring capacity. Execute differently shaped remainder batches as separate drained regions.
The ownership boundaries are:
Buffer
Producer
Consumer
When storage becomes reusable
Input
B
T0
After the consuming unpack instructions retire
Output
T2
NC
After the DRAM writes are acknowledged
Publishing inputs requires completed reads and any padding, transpose, or broadcast transformation. Publishing outputs requires completed packing. Instruction submission alone is not completion.
One host submission
Replace the session’s independent B and NC enqueues with a shared work packet containing reader commands, writer commands, buffer descriptors, and compute-program references.
Store fresh packets in B’s existing command ring; both movers read the immutable packet directly from shared L1.
The host uploads the packet and commits only B’s queue. B publishes NC’s task locally and starts the compute stream.
NC waits directly on output-buffer availability. B no longer waits for every kernel and signals NC for each completed block.
Keep the packet allocated until the reader, all compute roles, and NC finish. B’s queue completion represents completion of the entire packet.
Account for packet headers and both command sections when splitting work to fit ring capacity; split only at drained region boundaries.
Keep low-level independent mover APIs for diagnostics, but prohibit concurrent host-driven NC submissions while the session owns NC.
Preserve the current NoC routing policy: B’s DRAM reads use NoC0; NC’s output writes use NoC1. Keep the shared mover implementation rather than creating direction-specialized firmware.
Compute and Trace Migration
Streaming compute
Extend the resident runner in crates/tt-firmware/src/corpus.rs:160 with a separately tagged streaming-program format. Leave existing instruction and loop encodings supported.
Streaming programs contain bounded operations for:
Pushing existing Tensix instruction chunks, including their current loops.
Waiting for or reserving circular-buffer batches.
Releasing inputs and publishing outputs after backend retirement.
Repeating a compatible batch sequence and completing the region.
Start all three roles once per compatible region. Their streaming loops progress through batches independently; B does not issue a generation per batch.
Reuse arithmetic bodies, MOP/REPLAY lowering, fidelity settings, and internal synchronization. Initially generate slot-specific program variants for ring positions instead of introducing unrestricted runtime instruction patching.
Kernel migration
Port in this order:
Element-wise unary/binary operations, including existing broadcast handling.
Matmul using existing blocked operand reuse and accumulation order.
Reductions using their existing traversal and retained intermediate state.
The first implementation streams existing compute-sized batches, not necessarily individual tiles. This avoids sacrificing matmul reuse just to obtain small FIFOs.
Choose depth two where the existing sizing/reuse checks justify overlap; otherwise use depth one. Do not universally halve the staging arena.
Keep multi-pass operations as sequences of drained streaming regions. Unsupported configurations use the existing path and report the fallback reason. Do not add cross-operation fusion, multicast, Ethernet ownership, or new numerical K-blocking behavior in this change.
Traces
Extend the capture/replay machinery in crates/tt-kernels/src/session.rs:2426:
Capture reader/writer commands, streaming descriptors, and all referenced compute programs as one retained work description.
Replay through one B queue entry per participating tile; B starts NC on-device, without a second host enqueue.
Initialize buffer counters only at drained region boundaries. Reserve fresh generations for region starts and preserve queued replay ordering.
Fix legacy captured LAUNCH/KERNEL_WAIT generation rebasing and hold programs referenced by LAUNCH, not only KERNEL.
Preserve active streaming state across command-fetch chunks. Reject nested replay dispatch.
Keep the existing cross-tile barriers, but reach them only after NC has committed that tile’s outputs.
Treat streaming programs and their instruction chunks as program-cache resources: pin them during queued work and hold them throughout trace lifetime.
Interfaces and Compatibility
Add Session::enable_dataflow(b_image, nc_image) so callers explicitly supply both firmware images without introducing a firmware-image dependency into the kernel crate.
Preserve enable_dram(b_image) as the legacy-compatible B-only entrypoint.
Add ExecutionMode::{Auto, Streaming, Legacy} and a mode setter that synchronizes before switching and rejects changes during capture.
Auto is the standard backend default: streaming for supported configurations, otherwise the existing executor with an observable fallback reason.
Streaming refuses unsupported configurations instead of silently falling back. Legacy preserves the current execution path.
Add TT_EXECUTION=auto|streaming|legacy. Preserve TT_SCATTER=b as an explicit legacy B-only override; reject contradictory forced-streaming configuration.
Preserve TT_PIPELINE=0: use serialized batch credits, released after NC finishes the batch. NC remains the writer even when overlap is disabled.
Add per-stage progress, wait-reason, and error reporting. Every software wait must detect cancellation and peer failure. Host timeout/reset recovery must stop all five cores, invalidate captured state, and reclaim resources only after execution is stopped.
Extend docs/feature-traced-pipelining.md:1 into the unified ownership/streaming design. Preserve the repository’s current unrelated work.
Validation and Rollout
Correctness
Property-test page counters, wraparound, capacity limits, contiguous reservations, and illegal ownership.
Test more than three ring rotations, rings exceeding 15 pages, depth-one execution, tails, ragged padding, broadcasts, and transposes.
Compare migrated operations bit-for-bit against legacy execution across supported fidelity and layout configurations.
Test back-to-back operations, streaming/legacy transitions, and cross-tile producer/consumer dependencies.
Test repeated and queued traces with changed inputs, intervening fresh work, chunk boundaries, cache pressure, release, and reset.
Inject reader, writer, and role failures; verify bounded termination and recovery without premature buffer reuse.
Performance
Compare legacy B-only, current optional NC splitting, and streaming ownership on 1, 8, and 32 tiles, both fresh and traced.
Measure host submission time, device time, DRAM traffic, buffer stalls, and launch counts separately. Use host timing and counters as primary evidence until concurrent-core timestamper reliability is established.
Acceptance gates:
One host queue commit per work packet, with no NC host doorbell.
Compute launches scale with compatible regions, not batch count.
No extra operand traffic for equivalent blocked matmuls.
No correctness regressions; pass simulator gates, silicon gates, and sustained mixed-workload stress.
At least one mover-bound benchmark improves by 10%; no default-streaming benchmark regresses more than 5% in median end-to-end time against the best existing path.
Develop behind explicit selection, then enable streaming by default only after these gates pass. A failed performance gate blocks the default switch—it does not justify silently weakening correctness or removing the fallback.
