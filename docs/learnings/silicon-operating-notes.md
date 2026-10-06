# Silicon Operating Notes: Hardware Realities & Survival Guide

**Crates:** `tt-device`, `tt-kmd`, `tt-kernels`, `tt-firmware`, `xtask`
**Classification:** Tier 1 — Living Background & Learnings (Source of Truth)
**Hardware target:** Tenstorrent Blackhole A0 (p150a / p300)

---

Learned the hard way during Phase 1's silicon bring-up. Three host crashes bought these;
they apply to every phase's silicon gates and low-level hardware interactions.

**Ask the chip what it is, before addressing any Tensix tile.** Harvesting is per-ASIC,
it is invisible on the simulator (divergence row 35), and it cannot be discovered by
probing, because probing a fused-off tile *is* the hang. `Device::tensix_grid` reads
`ENABLED_TENSIX_COL` (ARC telemetry tag 34) and returns a `grid::Tensix`. Not tag 4,
`HARVESTING_STATE`, which is published and empty (row 36).

**The ARC is the one tile you can address before you know anything.** It sits at raw
`(8, 0)`, and `NoC/Coordinates.md:28-29` gives `Y = 0` and `Y = 1` as the rows where X
translation is not applied — so that coordinate denotes the same tile whether or not
translation is on. Every other coordinate's meaning depends on the translation state you
are trying to read. That invariance is the whole bootstrap; `tt_isa::arc` has a test
pinning `ARC_Y == 0` so it cannot be refactored away.

**The harvesting mask is read by population count, not bit position.**
`NoC/Coordinates.md:54` says fused columns are remapped to *maximal X*, so in translated
space the survivors are a prefix of `grid::TENSIX_COLUMNS` and the count determines the
set. This matters because the raw firmware bit layout is **not** published — UMD documents
its own `HarvestingMasks` as logical indices and says nothing about the word. Measured:
`0xfff` on both cards, contiguous, 12 of 14 columns, harvested at X 15 and 16.
`Tensix::from_enabled_column_mask` refuses a non-contiguous mask rather than guessing,
because the cost of a wrong guess is a dead host, not a failed test.

**Order of operations in `open()` is load-bearing.** Read the grid, *then* assert the gate
tile survives on this ASIC, *then* register `SET_NOC_CLEANUP`, *then* scrub. The cleanup
write is a NoC write to a Tensix tile; registered against a fused-off one it would fire on
every close from then on, including the close that follows the hang.

**Run silicon gates through `cargo xtask silicon`**, which runs one test per process
with an fsync'd log. All tests share one physical card and window allocation is global
card state, so `window_exhaustion_is_an_error_not_a_panic`'s `assert_eq!(held.len(), 201)`
is only true if nothing else holds a window. The simulator hides this by handing out a
fresh chip per call.

**When a run can take the node down, buy forensics first.** A hard kill loses the
journal's last minutes *and* unflushed file data — a linked test binary came back as 9.2 MB
of zeros, and `journalctl`'s last entry predated the real death by over two minutes, which
makes it useless as a time of death. What worked: a phase log `sync`'d after every write,
carrying `/proc/sys/kernel/random/boot_id` on each line so a reboot is evidence rather than
inference; and running the suite **one test at a time**, which is what identified the
failing access instead of losing the output with the session.

**`auto_reset_timeout=0` is the debugging posture.** It disables the ARC watchdog
(`wormhole.c:489` treats 0 that way), converting "hung NoC escalates to a chip reset that
drops the PCIe link and kills the host" into "card is wedged, recoverable". It removes the
safety net that recovers a hung chip, so it is for bring-up, not for keeping. The harness
and `cargo xtask silicon` report its value; neither refuses to run. The fix for a hang is
the access that caused it, which is what the next note is.

**A baby RISC-V's local data RAM does not answer the NoC while its core is in soft reset.**
The slow-path aperture (`0xFFB1_4000..0xFFB1_DFFF`) is documented as NoC-reachable
(`BabyRISCV/README.md:148`) and the page says nothing about reset. The first silicon probe of
it wrote all five RAMs on the gate tile with every core held -- the harness's resting state --
and never completed; the host died under it (run log `START` with no `END`, boot `497d5859`).
`tt-exalens`, Tenstorrent's own debugger, never makes that access: its
`ensure_private_memory_access` plants `jal x0, 0`, releases the core, halts it, and only
then touches private memory, and its ELF loader stages private sections in L1 rather than
writing them with the core in reset. `tt-device` now encodes the rule: `Device::read`/`write`
refuse the aperture (`TransportError::Hazard`); `local_ram_read`/`local_ram_write` refuse a
core in reset, wait out the post-release zeroing, bound T-cores to their 4 KiB, and move one
dword per access; `park_core` puts a core in the exalens loop. Unit tests watch each refusal,
and the reset check was watched failing with the check removed.

**Do not run `probe_niu.rs` as a grid oracle.** It opens `Simulator` directly, so it is
structurally simulator-only and *cannot* reach a card — which is fortunate, because
finding "the highest addressable byte at each coordinate" is precisely the sweep that
hangs on a fused-off tile. Its 140 is where the bad constant came from.

**An Ethernet link reporting "not Up on both ends" wants a card reset, not a code
change.** Seen 2026-10-01 on the X 3 cable link (X 13 stayed Up) with Ethernet code
unchanged since Phase 8; after a device reset `silicon_eth_link` was 10/10 and the
two-card sharded golden and the smoke tier passed again.

**Host infrastructure, for whoever inherits this VM.** The cards are passed through with
`viommu=virtio`. Switching to the Intel vIOMMU broke the passed-through NVMe — admin queue
DMA never completed (`nvme nvme0: I/O tag 28 QID 0 timeout` → `Identify Controller failed
(-4)`), with `AMD-Vi ... IO_PAGE_FAULT` on the host, across four boots; reverting fixed it
immediately. A DRAM-less controller doing Host Memory Buffer DMA through an emulated
vIOMMU under VFIO is fragile. Unrelated to the cards, but it cost an hour of
misattribution.

**Sandbox device visibility is not hardware discovery.** On 2026-10-05, the
workspace sandbox exposed no `/dev/tenstorrent` directory. The elevated isolated
runner accessed both installed cards and passed all 11 integer/attention gates
(run `1791232718`): ordinary gates on card 0, mesh forward on cards 0 and 1.
When device nodes are missing inside this sandbox, retry the authorized isolated
runner with elevated access before concluding hardware is unavailable.

### BF16 Src transpose control (2026-10-05)

`step73` passed signed BF16 unpack→MOVB2D with zero transposes, one
TRNSPSRCB and its inverse on both cards, run `1791233413` (4/4). The same
zero-transpose control fails in ttsim; see its divergence record. This validates
the probe's Src conversion/permutation, not a bit-preserving tensor transpose.
`step81` also rejected a deliberate Nearest→TowardZero mode mutant
(`1791233262`) and passed after restoration (`1791233389`).

### Bounded pooling and raw multi-source copies (2026-10-05)

Both cards pass reusable packed BF16 average staging and bounded F32 window
batches, including changed-input two-tile traces after consumed partial frees:
`1791233597`, 18/18. GMPOOL flushes signed zero/subnormal inputs to +0 and
compares signed NaNs by exponent/mantissa magnitude (`1791233808`, 2/2), rather
than Burn's first-NaN/index semantics. General max pooling keeps its SFPU route.
Raw multi-source copies preserve BF16/F32 NaN payloads, signed zeros and
subnormals in slice assignment, composed cat/repeat and parent views; I32/Bool
and analytic gradients pass too (`1791233990`, 6/6).

### Long trace watchdog progress (2026-10-05)

Resident embedding/select-add expands transformer training into thousands of
retained descriptors. The previous host watchdog treated one entire replay
`CALL` as one queue completion and timed out in ttsim after 50 million cycles
despite ongoing device work (step40, queued list 4922, last finished 4923).
A dedicated mover `TRACE_PROGRESS` word advances only after a retained chunk
finishes, including nested streams. The host samples it when the ordinary
no-progress budget expires and resets the budget only if it advanced; reads and
kernels that stall cannot refresh it. The previously failing step40 transformer
training replay passes in ttsim (136.32 s) and isolated card-0 silicon
(`1791235705`, 1/1). Queue/error publication and completion semantics are unchanged.

Convolution's reversed-kernel-column mutant and resident selection's shifted
source-column mutant both fail their independent simulator oracles; logs are
`target/silicon/step85-mutant-reversed-kernel-column.log` and
`target/silicon/step86-mutant-shifted-column.log`. Mutants were restored.

### Src transpose and GMPOOL ArgMax packed-output limits (2026-10-05)

Step87 uses flat SrcB unpack of 512 datums, with the 16×16 operand copied into
SrcB rows 16..32; TRNSPSRCB leaves rows 0..16 unchanged. Selecting the wrong half
failed the independent transpose oracle (`1791239607`); the corrected native
block/transposed-TF32-matmul path passes both cards (`1791239869`). Its Src/pack
path normalizes signed zeros and subnormals to positive zero and preserves the
tested signed infinities/NaN payloads. TF32 fractional products match the native
raw-copy-prepared reference and the existing derived phase bound; they are not
claimed to be exact scalar F32 sums. Payload-preserving tensor copies stay raw.

GMPOOL with its ArgMax bit set returns zero packed index bits for every probed
column on both cards: unique finite winners in each first-eight row, ties, lower
eight-row winners, signed zeros, subnormals and NaNs. Max values follow the
existing magnitude oracle. This was observed through ordinary F32 packing and
through an attempted integer SFPU load/store (`1791240373`); the simplified
probe is retained (`1791240702`). These results characterize our packed output
path; they do not prove that the internal Blackhole instruction never tracks an
index. The explicit MaxIndexProbe returns diagnostic raw I32 words, with no index
contract. General Burn max-with-indices remains exact SFPU selection.


### Final resident module and typed-index acceptance (2026-10-06)

Isolated release smoke run `1791250791` passes 288/288 on both cards. Convolution
and BF16 attention changed-input traces exercise one and two Tensix tiles with
multiple output tiles. Step88 compares all F32/BF16 forward/gradient results to
single-card execution and independently checks nonzero Q/K gradients in both
mesh partitions, submitted matmul counts on each card and acknowledged Ethernet
traffic. Resident index updates preserve wrapping I32 and Boolean OR semantics;
device-produced indices replay across face/tile boundaries without downloads.
These results complement the Src/GMPOOL measured limitations above, rather than
establishing a general payload-preserving Src transpose or packed ArgMax index
contract.


### CNN training trace parameter storage (2026-10-06)

Step89's BF16 CNN training capture initially failed with `no device buffer 149`:
`DramBuffers::copy_into` consulted only the 32-bit buffer map for packed BF16
parameters. It now dispatches same-format BF16 updates to raw halfword repacking
into the retained allocation, with zero padding and existing declared scratch
lifetimes. Both cards pass F32/BF16 changed-MNIST-batch trace comparisons and
fresh convolution weight equality (`1791252026`). Trace gates explicitly use
`download_device` to inspect mutable parameter buffers; creation-data caches
retain the initial values and cannot serve as an oracle for in-place updates.

### Matrix-unit elementwise arithmetic (2026-10-06)

M1 exposes typed retaining/releasing ELW consumers and resident F32/packed BF16
paths. Step9's broadcast/assignment/nonzero-Dst probes pass both cards
(`1791253529`); step90's repeated final-consumer bank flips, all RHS broadcasts,
all four multiplication fidelities, ragged shapes, freed-operand traces on
one/two tiles, raw parent padding and downstream reductions/matmul pass
`1791255922` (24/24 including independent instruction mutants, explicit Dst
base setup and BF16 gradients). Full smoke passes `1791255409` (322/322).
Final continuation verification is recorded in hardware-coverage.md.

ELWADD/ELWSUB align each operand to a **shared 10-fraction-bit quantum at the
larger exponent**, with half-up rounding of magnitude in the step90 probe.
This holds for TF32 and BF16 Src; it is not an FP32 adder contract. The independent
41-point probe spans exponent differences -20..20. Example: `1 + 2^-11` returns
`1 + 2^-10`, even though the scalar FP32 sum is exact. Src conversion error plus
two alignment quanta bounds add/subtract in the normal-domain gate. The exact
oracle refuses inputs that lose alignment bits. Four multiplication phases
consume 5/7-bit mantissa slices; SrcA's final TF32 bit is ignored. The finite gate
adds conversion error, omitted phase products and a worst-case FP32 truncation
`gamma_8` budget, and separately excludes exceptional/underflowing domains.
Both cards pass the alignment/bound probes (`1791254333`).

The 17-pair special corpus agrees with ttsim (`1791254100`) for F32 storage with
TF32/BF16 Src. Tested signed zeros and subnormals pack to +0. Signed quiet NaNs
against 1 produce signed infinities. `(+inf,-inf)` produces +0 for add and
multiply, +inf for subtract; `inf*0` is +0. Max finite add/multiply overflow;
minimum normal times one-half underflows to +0. These observations are pinned
in step90; arbitrary NaN payloads or all exceptional operand combinations are
not inferred from this corpus. Packed BF16 specials have their own continuation
coverage in step90.

Programs declare their three L1 buffers/semaphores with Requirements, establish
ADCs, formats, RWC/fidelity, Dst offsets/base and bank-release enables, then pack
F32. Silicon writes `DEST_REGW_BASE_Base=0`; ttsim's existing register-6 refusal
requires omitting that write on simulated transports (divergence row 21).
The target distinction is included in the role cache key. The final consumer
releases both banks once per face, and each tile clears Dst before its eight
output blocks. Ragged lanes are independent and output padding is undefined;
downstream repair operates on the result, leaving parent storage/claims intact.

Release performance with validated outputs and dataflow evidence is recorded in
firmware-performance.md. No mesh route or automatic SFPU replacement is enabled.

### Direct ELW RHS broadcasts (2026-10-06)

The matrix kernel now reads the original RHS tile on B/NoC0. For an output tile
at `(tr,tc)`, select RHS tile `(0,tc)` for row, `(tr,0)` for column, or `(0,0)` for
scalar. A face index `f` selects SrcB face `f % 2` for row, `(f / 2) * 2` for
column, or face 0 for scalar. The math SrcB RWC is zero for row/scalar and the
current eight-row half for column. SrcA/Dst addressing, precision/fidelity,
bank handover, NC scatter and padding contracts are unchanged. Reading a whole
RHS slot keeps checked NoC alignment; only valid broadcast lanes are consumed.
No expanded RHS GDDR allocation or coordinate transfer is submitted.

All eleven step90 gates pass on simulator and both cards (22/22,
`1791256564`), including nonconstant row/column/scalar RHS changes during trace
replay on one/two tiles and F32/packed BF16. The resident arithmetic gate checks
zero staging transfer packets. Expanded poisoned-padding validation for all
RHS geometries also passes both cards (`1791256629`, 2/2). Release comparison
`1791256590` validates all 192 results; detailed medians are in
firmware-performance.md. No new simulator divergence was observed.
