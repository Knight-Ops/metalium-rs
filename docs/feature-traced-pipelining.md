# Support traced pipelining

## Implemented (2026-10-03)

Streaming ownership is the only GDDR compute scheduler for fresh and traced work.
`LAUNCH` and `KERNEL_WAIT` generations are rebased, programs stay held, and replay
preserves the captured schedule across command chunks. Record payloads are not
interpreted as command headers when rebasing or retaining programs.

Capture retains
both reader and writer streams in DRAM and pins all streaming scripts and their
arithmetic bodies. One host B replay dispatches NC locally with `PAIR_CALL`.
B's outer stream and nested reader fetch use distinct buffers; NC fetches writer
commands through NoC1 with an independent transaction ID. B completes the region
only after NC acknowledges its writes, before reaching cross-tile barriers.
Queued replay generations are reserved afresh; errors invalidate traces and reset
the roles as well as the movers before fresh execution resumes.

Enable GDDR with `Session::enable_dram(b, nc)`; no executor selection is needed.
`TT_PIPELINE=0` serializes slot reuse but keeps NC as the writer. The legacy wave,
B-only pipeline and two-host-queue NC schedulers are removed. Fresh/small
regressions are accepted as an architectural tradeoff, not a universal speedup.
See [ownership implementation and rollout](feature-streaming-dataflow-ownership.md)
and [measurements](firmware-performance.md#streaming-ownership-rollout).

## Original investigation

For B-only pipelined traces, most of the machinery already exists. The missing work is mainly in capture bookkeeping and validation. NC-backed traces need additional protocol changes.
The firmware’s CALL replay path already recognizes and rebases KERNEL, LAUNCH, and KERNEL_WAIT. It can execute the overlapping schedule. See mover.rs.
For B-only support, these changes are needed:
Capture launch and wait generations correctly.
capture_segment() currently converts only KERNEL generations into offsets relative to the capture. It must also convert LAUNCH and KERNEL_WAIT, preserving each launch/wait pairing. Otherwise replay uses capture-time generations instead of its newly reserved generations.
Keep launched programs alive for the trace.
The same capture code only holds program-cache references from KERNEL entries. It must hold references from LAUNCH entries too, preventing eviction until the trace is released. Both omissions are in capture_segment().
Allow eligible pipelines during capture.
Remove the capture exclusion from the pipeline selection paths, retaining the existing workload eligibility checks and set_pipeline(false) behavior. Capture should record the chosen schedule; replay should execute that schedule without replanning.
Validate replay across trace-chunk boundaries.
Replay fetches commands in 64-entry chunks and drains NoC requests between chunks. A launch and its wait can land in different chunks. The implementation must preserve the active generation and staging-buffer ownership across that boundary. Those drains may also reduce overlap, so this needs performance measurement, not just correctness checks.
The existing descriptor POKEs, setup kernels, resident programs, and alternating staging buffers should be reusable. A new pipeline executor does not appear necessary.
Tests should cover:
Captured pipelined matmul, element-wise operations, and reductions matching plain execution bit-for-bit.
Repeated and queued replays with changed inputs and fresh operations between replays.
Launch/wait pairs crossing chunk boundaries, program-cache eviction pressure, and trace release/reset behavior.
Silicon comparison of plain replay, pipelined replay, and fresh pipelined execution.
NC support is a larger extension. Capture currently records one B command stream per tile. It would need to record and retain the NC stream too, enqueue both streams (shared packets have since done this: `PAIR_CALL`), and track NC lifetime and failures. Simply removing the NC capture guard would leave that information missing.
I’d implement B-only pipelined traces first. That combines trace submission savings with the overlap already enabled by default, and gives a measured baseline for adding NC.
