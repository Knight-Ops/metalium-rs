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

### BFP and PRNG characterization (2026-10-06)

Both cards pass step91–93, release isolated run `1791297768` (8/8).
Step92 measures codes 6/7/15 for BFP8/BFP4/BFP2 with matching Src input/output
codes. A 1024-datum image carries 16 header bytes and 64 exponent bytes; image
sizes are 1104/592/336 bytes before slot alignment. Sixteen consecutive datums
share each exponent; sub-byte datums occupy the low bits first. The physical
pack stream matches an independent f64 scaling oracle: BF16 truncation,
maximum raw exponent selection, BFP8 magnitude rounding/clamping, then BFP4/2
truncation. Zero magnitude drops its sign; tested input subnormals pack to zero.
Exponent-255 significands are retained in the tested late conversion, including
signed infinities and NaNs. Src/MOVA2D flushes decoded subnormals. Raw low-exponent
nonzero magnitudes can wrap their decoded exponent as the existing decoder
models. These are measured conversion semantics, not IEEE payload preservation.
The executed missing-exponent-section mutant fails the physical oracle on both
cards and ttsim.

Step93 observes exponent-127 histogram bin counts 8/16/32/64/128 for
4/8/16/32/64 packed Dst rows: one increment per eight datums in this configuration,
rather than the WH model's per-datum update. CLREXPHIST clears this histogram
and packer 0's maximum exponent. The gate uses F32 packing independently of
BFP exponent selection. Saturation and nonempty packer 1–3 histories remain
unmeasured.

The checked `backend::write_prng_seed` emits a complete register write and
WRCFG scheduling NOP. In step91, an identical seed write within one program
reseeds ttsim but silicon returns the next old-stream value. Sixteen extra
NOPs, byte RMWCIB writes and a complementary/requested seed sequence did not
make silicon repeat (`1791296771`, `1791296833`, `1791297091`). Advancement
matches the documented LFSR; disabled lanes neither write nor advance. Initial
lane snapshots are in the logs; independent lanes and seed initialization
are not established. This characterizes this instruction path, not every
RISC-V/configuration seed path. Application seeding remains unresolved; Burn
random construction is unchanged.

A follow-up debug read verifies the seed register contains 0, 1, 0x12345678
and 0xffffffff after the tested writes on both cards (`1791298248`, 2/2).
Thus the observed stream continuation is not a missing seed register value.

### Resident BFP conversion and products (2026-10-06)

`step94_bfp_storage` passes both cards (`1791299101`, conversion/replay;
`1791299490`, conversions, direct products, K reloads). BFP8/4/2 images occupy
1104/592/336 bytes, with GDDR slots aligned to 1152/640/384 bytes. The compact
Src conversion uses eight 128-datum banks: each gathers eight exponent bytes
and packed datums before MOVA2D and F32 packing. NoC reads must preserve the
GDDR/L1 low-six-bit congruence; packed datum chunks first land in declared
scratch and move byte-for-byte into the compact image. Exponents remain live
through the eight conversions. NC writes and subsequent reuse pass changed-input
trace replay and varied physical/decoded corpora. Direct same-format matmul
reads full physical slots into declared F32-sized L1 operand slots, preserving
original exponent groups. Existing matrix kernels consume format codes 6/7/15;
HiFi4 products and F32 K accumulator reloads pass exact signed-unit oracles
for ragged shapes and K=257. Padding is established before compression.

`step95_burn_bfp` passes both cards (`1791299490`): unary/scalar preservation,
higher-precision binary promotion, F32 reductions, decoded F32 rearrangements
and identity cast backward retain residency. Master updates and gradients
remain F32 in the tested scalar training gate. This does not establish MNIST
accuracy or broad training/traced workload acceptance.

The initial PRNG snapshots also show strong lane correlation: lanes either
start identically or adjacent lanes share 30 bits after a two-bit left shift.
`step91_seeded_prng` now asserts this measured structure independently of each
lane's LFSR advance. Absolute initial words vary between runs/cards despite
seed-register readback, so no seed-to-lane initialization formula is promised.

Expanded BFP gates (`1791300035`) pass both cards: special/mixed exponent
groups, executed wrong-exponent and reversed datum/nibble/bit stream controls,
shared views, fusion boundaries, parameter casts, changed-input trace outputs
and four-policy actual MNIST MLP/CNN training. The observed accuracy changes
and conversion cost are recorded in `firmware-performance.md`.

Final BFP control/sweep validation `1791301208` passes 28/28 on both cards.
`step94` compares independent f64 scaling and integer-significand encoding
oracles across discarded-bit thresholds, signs, exponents 1/6/126/127/128/254/255
and mixed exponent groups. The missing-edge control supplies a false Zero
padding claim and is rejected by the physical oracle; the lost-accumulator
control executes only the final K tile and disagrees with the full product.
`step95` covers checkpointed identity backward, packed parameter-copy traces,
batched/attention routing, stale source host caches, and softmax's F32 promotion
through reductions. No intermediate model downloads occur. Full both-card
SMOKE `1791300502` passes 364/364; benchmark `1791300393` validates every sample.

### Direct RISC-V PRNG seed stores (2026-10-06)

The follow-up `prng_seed` firmware runs on T1 and writes the generated global
seed field with a full-width RISC-V `sw`, after draining coprocessor work. A
fence alone does not make the initialized states immediately available: the
first probe (`1791302087`, failed restart control) observed inconsistent
snapshots across identical and complementary seeds. Adding 512 iterations of
a RISC-V NOP loop after the fence produces repeatable restarts on both cards
(`1791302152`, 2/2; finalized step91 suite `1791302214`, 4/4). This is a
validated conservative settling interval, not a measured minimum or a
completion-status protocol. It does not change the measured WRCFG behavior.

The gate uses seeds 0, 1, 0x12345678, 0x80000000, 0x55555555, 0xaaaaaaaa and
0xffffffff. Each sequence is seed/seed/complement/seed; all 32 lanes repeat
on identical seeds, change for the complementary seed, and advance according
to the independent LFSR model. Adjacent lane states share 30 bits after a
two-bit left shift. Seed 0xffffffff produces the absorbing all-ones state,
which never advances to a different value. Thus a working restart path does
not establish application randomness quality or independent streams.

For seed 0, silicon's first two lanes are 0xf173cc27/0xc5cf309e; pinned ttsim
starts at 0xc5cf309e/0x173cc27a. Other tested nonabsorbing seeds show the same
one-lane offset. Absolute initialization is therefore not a common simulator
and silicon contract. Burn random remains unchanged; S7 remains partial for
application quality and stream semantics.

Final suite `1791302401` passes 4/4 with the probe leaving a nonabsorbing
stream for subsequent programs on the tile. Simulator gates, image entry
checks, default/silicon workspace Clippy and firmware Clippy pass. All eight
MNIST regression gates pass with unchanged goldens (268.05 s). The probe is
included by SMOKE's existing step91 filter.

## Matrix register chains (2026-10-06)

Step97 adds loaded-bank-only MOVD2A/MOVD2B/MOVB2A consumers. A destination bank
must already belong to the Matrix Unit: Dst-to-Src moves do not establish
ownership or wait for the destination's DVALID handover. The resident chain
retains both banks through both stages and releases each once per face, using
the existing unpack/math/pack synchronization. It explicitly configures format
overrides, disables implied formats, establishes Dst base/offset/RWCs/release
controls, resets fidelity for each stage and moves before clearing eight Dst
rows. Format selection is restored before ordinary subsequent kernels.

Both cards pass step97 (`1791317039`, 20/20), including one/four-row
conversions, masks, exceptional datums, traces and resident-chain consumers.
The cards are accessible through the elevated isolated runner; sandbox visibility
is not hardware availability. Step9 retains the encoding provenance.

The chain's initial one-row Dst clears reproduced the existing ZEROACC physical
address difference (divergence 51), run `1791313486`. Use sixteen-physical-row
mode for each aligned eight-row F32 half, with block index `dst_row/8`; previous
final halves stay live. Both cards pass changed-input trace/reduction replay
with the correction (`1791313617`, 2/2).

Four-row register moves align their read address, but not the Src write address.
Step97 observes effective Src row 71 writing rows 7..10 on both cards. MOVB2A
reads aligned B rows 4..7 and writes unaligned A rows 7..10. The pinned model
incorrectly aligns both sides; see the step97 simulator-divergence addendum.
Mask probes must initialize actual loaded-bank datums explicitly before checking
untouched lanes across repeated programs; do not rely on zero unpack flags or
ZEROACC to scrub storage observed through a raw SFPU dump.

Full release SMOKE `1791317135` passes 388/388 on both cards, including the
expanded one-row summed-offset/wrapping probe and the unchanged MNIST golden.

Retained step9 register-move address-modifier evidence passes on both cards
(`1791317641`, 2/2), including the third modifier bit. No encoding correction or
generated-output change was needed; `gen-isa --check` passes.

Release chain benchmark `1791317719` passes on both cards with validated outputs,
two warmups/seven host-timed samples and dataflow counts. See the resident chain
record in firmware-performance.md for medians and conditions.

## XY-counter rectangle copies (2026-10-06) — measured, not quoted

Card 0 step98 run `1791320021` validates current-thread INCADCXY/ADDRCRXY
counter updates against an independent indexing oracle. ADDRCRXY advances
selected cursor anchors and replaces the corresponding live values, rather
than adding to live values. An empty coordinate mask is a no-op on this card;
ttsim refuses it (divergence 74). Wormhole provenance remains unchanged; the
unsupported ThreadOverride is neither offered nor probed.

The ADC F32 rectangle path selects complete sixteen-datum rows into Dst and
retains every tested raw bit: signed zeros, subnormals, infinities and quiet or
signaling NaN payloads. Every destination tile is cleared before unpacking and
packed only after unpack retirement. Partial-row probes reveal why full rows
matter: clearing Dst validity does not erase its underlying data; writing part
of a row makes the other columns' old bits visible again. Initial run
`1791319848` incorrectly expected these untouched partial-row columns to be
zero. The corrected gate checks written datums and wholly untouched rows, and
the production kernel only writes complete rows. This is observed on both
ttsim and card 0, not a simulator divergence.

Counter lifetime is also observable across programs. The first full smoke run
`1791321400` failed step89 F32 CNN learning after 149 passes; isolated run
`1791321632` reproduced exactly the same losses. Disabling ADC slice dispatch
made the unchanged CNN gate pass (`1791321696`). The copy left unpacker 0's
live Y counters at its final rectangle row, which subsequent Src programs can
inherit. Retiring the last unpack and clearing XY/ZW live counters and cursor
anchors before handoff restores the baseline. With dispatch enabled, run
`1791321793` passes all four CNN learning/trace gates and nine ADC gates. Step98
also checks a copy followed by matmul on each of one/two units. This is a
kernel state-lifetime defect and correction, not a numerical tolerance change.

## Z/W counters and resident plane copies (2026-10-06)

Card-0 step99 runs `1791323973` (initial 4/4) and `1791324198` (expanded 9/9)
validate current-thread INCADCZW/ADDRCRZW against independent counter/address
models. Cursor-relative updates advance selected anchors and replace selected
live values; zero increments restore anchors. Unselected coordinates and other
threads retain their values. Unpacker 1 and packer target selection are observed
through resulting datums, not just instruction encodings. Empty ADDRCRZW masks
are no-ops on this card and are independently refused by ttsim (divergence 75).

Bounded unpack-to-Dst fixtures distinguish Blackhole Z/W counter width from
input addressing. Adding 256 to both input and output Z (or W) with a four-byte
output stride reads input row zero but writes Dst row sixteen. Adding 8192 wraps
both counters to zero. Thus input addressing sees the low eight bits, while
output observes the full thirteen-bit counter, in agreement with Blackhole
ADCs.md. No generator correction or provenance change is needed.

The resident plane kernel groups source-tile reads and copies logical words
into a zeroed 64-row staging slab. Descriptor ZDim=8 gives input byte strides
Z=64/W=512; explicit output Z/W strides select complete sixteen-datum Dst rows.
INCADCZW steps Z live counters; ADDRCRZW restores Z to its zero anchor while
advancing W anchors. Full-row unpacks preserve exceptional F32 raw bits and
avoid the partial-row validity issue described above. Counter changes follow
unpack retirement; XY/ZW live counters and anchors return to zero before
handoff. Changed-input traces, deferred source frees, ragged outputs followed
by reduction/matmul, borrowed row views and repeated copies pass on card 0.

Final expanded step99 plus subsequent CNN learning/trace gates pass 13/13 on
card 0 (`1791324670`). This includes all increments 0–7, all target combinations
and nonempty masks, and a selection whose global W and Z counts are both nine
(beyond the local 8×8 staging coordinates). Full release SMOKE passes 212/212
(`1791324245`). Fresh copies also pass with `TT_PIPELINE=0` (`1791324866`) and
`TT_BATCH=0` (`1791324883`). Release native-repack comparison `1791324901`
validates all six records; its conditions and medians are in
firmware-performance.md. Ordinary tranche acceptance follows the current
card-0 policy; these results do not claim standalone step99 validation on card 1.


### Scalar/configuration foundation and replay publication (2026-10-07)

`step100_scalar_config` passed nine isolated release gates on card 0
(`1791343719`). Scratch words 209–211 are global across configuration banks;
all three issuing threads see the same values. RDCFG reads the selected bank,
CFGSHIFTMASK implements all eight ALU modes, replace/preserve masks, widths
1/32 and rotations 0/31, and scalar register/immediate arithmetic matches the
pinned independent integer models. GPRs remain thread-local across programs.
Only register MULDMAREG also passes ttsim and gains Confirmed provenance;
other scalar forms retain WormholeOnly status with explicit silicon evidence.

A tight replay descriptor body exposed a publication hazard. With a scalar XOR
immediately followed by WRCFG, changed parameters produced the prior descriptor
value (`1791330370`, aliased scalar result; `1791342637`, separate scalar result
register published the preceding shift value 2128 instead of 2127). The same
body unrolled passed. Final GPR snapshots showed the masked index and comparison
were correct; this does not establish a new encoding or a general arithmetic
failure. An explicit `STALLWAIT(Before::EVERYTHING, CONFIG_BUSY)` between final
scalar arithmetic and WRCFG publication made both bodies agree with the independent
oracle across changed parameters and an intervening configuration program
(`1791343224`, then full acceptance `1791343719`). Keep that publication boundary
in tight replay consumers in addition to the read/mutation helpers' own waits.

Programs explicitly initialize GPRs 24–31 and shared scratch on every invocation.
Their descriptors are diagnostic data, never submitted to a memory engine.
Default tensor dispatch is unchanged. No performance claim is made, and card 1
was not used because there was no device-specific discrepancy to investigate.

Final repeat `1791379895` passed all nine gates, with the descriptor's structural
assertion requiring a REPLAY with Load=0 (execution, beyond recording its body).
Workspace tests, seven simulator step100 gates, both Clippy variants, silicon
no-run, generator/shipping checks and eight MNIST regressions also pass; the
MNIST golden is unchanged.

### L1 scalar movement and output ownership (2026-10-07)

Step101/102 pass twelve isolated release gates on card 0, run `1791389018`.
LOADIND matches the pinned widths (0=16, 1=4, 2=2, 3=1 bytes), including
preserved high bits, both offset halves and byte increments on all three roles.
STOREIND_L1 instead uses **0=16, 1=2, 2=4, 3=1 bytes**. The raw size sweep
`measure_indirect_widths`, run `1791385416`, independently measures each form;
the checked helper maps semantic widths accordingly. Generated field positions
and WormholeOnly provenance remain unchanged. XMOV copy/zero passes both config
banks, repeated programs, role handovers, guards and immediate scalar consumers.
Blackhole completion is C0 for scalar requests, C9 for mover requests, and C12
for configuration setup. DMANOP is diagnostic only.

An empty T2 streaming body publishes its output credit immediately; it does
**not** join T0. The first movement consumer exposed premature NC reads and a
copy→matmul result of 98 instead of 130 (`1791388321`). T0 also needs to wait for
T2's output reservation before writing staging reused by NC. The accepted builder
therefore declares two semaphores: T2 posts ready *after its runner reserves
output*, T0 takes ready before any movement/configuration, T0 drains C0/C9 with
full block masks before posting done, and T2 takes done before its runner
publishes. Both semaphores return to zero each batch. T1 is empty; T2 carries
only this ownership shell, with all memory movement on T0. This requires no
mailbox or firmware change and preserves the existing retirement barriers.
No production firmware/host TDMA-RISC mover issuers were found by code audit;
bring-up finishes before local movement. Software B/NC NoC transfers are distinct.

The explicit Session copy/zero APIs support 32-bit F32/I32/Bool and raw BF16,
whole tile-row views, physically zero output padding, and changed-input traces
on one/two tiles with deferred frees. Source padding claims and payload bits are
preserved. Step102 poisons F32 source padding and inspects physical output slots;
copy→reduction/matmul and packed BF16 matmul pass. T0 buffers and both handoff
semaphores are allocated through Requirements; no arbitrary L1 addresses enter
the public API. Mesh/BFP/in-place/rectangle movement and automatic Burn routing
remain outside this contract.
