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

### Source-bank clear, release and shifting (step103, 2026-10-07)

Card-0 release run `1791397029` passes 9/9 Stage C gates. ZEROSRC's inherited
A/B selection and SingleBankMatrixUnit layouts work: zeroing a current unpacker
bank discards staged rows; zeroing a matrix-owned bank retains ownership.
Separate A/B and both-operand CLEARDVALID with Reset=0 and KeepReadingSameSrc=0
release and advance the matrix pointer. Existing SETRWC releases remain unchanged.

UNPACR_NOP_ZEROSRC needs a measured Blackhole layout: WaitLikeUnpacr is bit 5,
BothBanks bit 4, and the clear-value code occupies bits 2..3 (the generator
retains the historical `NegativeInfSrcA` name). The inherited Wormhole word
`0x43000011` **clears both banks**, rather than safely selecting the current bank.
Checked A uses `0x43000021`, B `0x43800021`: current bank, access wait enabled,
zero clear value. The independent held-bank gate stages/clears bank 1 while
matrix-owned bank 0 remains readable, then uses regular UNPACR handover and
explicit release. This is characterization of hardware ownership, not a relaxation
of the checked lockstep types. Bit 3 (clear code 2) produces physical 1.0 in the
TF32 diagnostic; nonzero values remain outside the checked surface.

The gates seed both banks and observe all 64 rows across six bounded handovers,
including an untouched opposite-bank sentinel and a previously cleared bank.
They cover TF32/BF16 changes, partial staging, exceptional bits, row-63 wrapping,
rotate/zero-fill and repeated SHIFTXB with modifiers 0, 1 and 4. Src readback
predicts 19-bit conversion and exponent-zero flushing; it is not a raw F32 copy.
Safe wrong-row, wrong-selection and wrong-mode mutants fail without removing
waits. Existing matmul/pooling consumers run after the diagnostic's explicit
release within the same program, without resetting source state between chains.

Src contents persist across programs: the diagnostic initializes **every row**
of both banks with current-bank NOP zeros before seeding, rather than assuming
rows beyond a partial input are already zero. Both-bank and nonzero-clear raw
characterization runs keep both banks unpacker-owned and all work retired;
they do not authorize those variants for ordinary kernels. Negative infinity,
wider format-dependent clear values and retained-bank releases stay deferred.
Recovery and automatic dispatch are unchanged; no performance claim is made.

Final Stage C acceptance: full card-0 release SMOKE 242/242 (`1791397621`),
including all nine step103 gates with explicit ADC initialization. Simulator
source gates 6/6; MNIST execution-order regression 8/8 with the golden unchanged.
Workspace tests/Clippy, silicon compilation/Clippy, generators and shipping
checks pass. The [completed tranche](../completed-plans/source-bank-clear-release-shift.md)
records the constrained supported forms and remaining variant deferrals.

### Resumed Stage D reboot evidence (2026-10-07)

The user reports a whole-system reboot during step104. Log 1791402960 has two
unfinished START entries (regular_unpacr_control and explicit_a_healthy_bank),
boot a4e3b1bb-3f6c-4431-8a6e-dbd68cbe174a; the resumed boot ID is
77ee3dd9-2c4e-4700-971b-3cbc2dd1a4e4. The log does not distinguish the causal
probe, and ARC watchdog causation has not been established. The repository's
NVMe ext4 mount is read-only inside the current sandbox; this does not establish
the host mount state or a reboot-induced filesystem change. A step104 debug
executable and a release
burn_server executable fail execution as non-ELF data. Rebuild in a writable
recovery copy before bisecting. No step104 silicon success is recorded.

The user explicitly requires normal auto_reset_timeout settings for this
bisect; do not change that parameter. Use separate setup-only, one-bank regular,
concurrent regular and explicit A/B boundaries from the active Stage D checklist.
A failure stops candidate probing; the existing recovery path stays unchanged.

### Stage D silicon reboot bisect (2026-10-07, latest evidence)

The software trigger is isolated to the **unverified Wormhole handover
publication**, not to the prerequisite Blackhole waits in this diagnostic.
`bisect_single_bank_explicit_a` substitutes `0x43000007` for regular SrcA
UNPACR FlipSrc after a retired partial unpack on a healthy bank. It keeps B's
regular publication, uses serialized roles, reads A/B and explicitly releases
both operands. Run `1791406213` has START without END on boot
`cab5ad1c-88c9-4855-b9a7-030f3d73bcad`; the next boot is
`14d13ced-d7bb-4c81-ba18-505d1dd8c540`, and the user reported another reboot.
This second failure is a single runner boundary, unlike the overlapping control
and A-candidate START entries in the original run `1791402960`.

Surviving release card-0 boundaries, all with normal `auto_reset_timeout=10`:

| Boundary | Run | Result |
| --- | --- | --- |
| Established step103 regular control/data mutant | `1791406158` | PASS |
| Stage D configuration only | `1791406171` | PASS |
| One-bank regular publication, serialized roles/readback/release | `1791406183` | PASS |
| Alternating-bank regular control | `1791406197` | PASS |
| A C1 retirement + C5 ownership waits, retaining regular FlipSrc | `1791407483` | PASS |
| B C2 retirement + C6 ownership waits, retaining regular FlipSrc | `1791407513` | PASS |

The wait-only controls use the same staged data, configuration, serialized
readback and release as the failed single-bank A program. Their only publication
is the surviving regular UNPACR. Both wait-only controls remained on the same
boot for more than 15 seconds after completion (beyond the 10-second watchdog
interval). The concurrent six-round loop, format changes and input special
values are absent from the failed single-bank boundary.

**Limit of the root-cause finding:** the failure belongs to replacing regular
publication with the inherited handover word in this program. There is no
silicon acceptance of that word. An unsupported/respecified Blackhole opcode
is a hypothesis, not a measured replacement encoding. We have not separated
NOP execution from its retirement/first matrix consumer, and have no retained
ARC/PCIe reset trace proving the electrical reset mechanism. The guest's
previous-boot kernel journal has no panic/ARC/AER report; its last recorded
system event is 20:50:05 UTC, eight seconds before the failing START at 20:50:13.
The next guest boot starts 20:52:13 UTC. Journal absence is not proof against
an ARC watchdog reset.

Do not issue `0x43000007` (or guess a replacement) in ordinary Blackhole kernels
or recovery. Explicit candidate and consumer gates are ignored by default;
retain WormholeOnly provenance and Stage D's pending status. Regular UNPACR
FlipSrc remains the accepted publication path. No watchdog setting or recovery
implementation was changed. Separate isolated runner invocations must never
overlap: the runner currently has no interprocess card lock.

**Encoding root cause, independently explained:** Tenstorrent's
[Blackhole `TT_OP_UNPACR_NOP` macro](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h#L901-L910)
(read 2026-10-07) places `Set_Dvalid` in bits 8..11, `Src_ClrVal_Ctrl` in
bits 2..3 and the two-bit `Unpack_Pop` in bits 0..1. Therefore `0x43000007`
decodes as `Set_Dvalid=0`, `Src_ClrVal_Ctrl=1`, `Unpack_Pop=3`, not the
Wormhole page's dedicated handover. This agrees with the simulator's independent
`unpack_pop=3` refusal. The software error is using a Wormhole-specific mode
encoding as a Blackhole handover after successful retirement/ownership waits.
This source explains the wrong word; it does not establish the internal effect
of pop=3 on silicon or prove ARC watchdog causation. No replacement word was
probed or blessed. Do not convert this research source into generated measured
provenance without successful semantic gates and the required pinned-input work.

### Stage D measured Blackhole handover (2026-10-07)

This supersedes the preceding "no replacement word" status. The bounded,
non-clearing handover is **SrcA `0x430001e9`, SrcB `0x438001e9`**. Its
Blackhole fields are `Unpack_Pop=1`, `Src_ClrVal_Ctrl=2`,
`Clr_to1_fmt_Ctrl=3`, `Stall_Clr_Cntrl=1`, `Set_Dvalid=1`; all stream/message
controls are zero. Format selector 3 is repurposed to DVALID-only in the
clear-to-one mode. Selecting clear-to-zero instead clears the staged data.
The source is immutable official TT-Metal revision
`201312fe7960b3711a420e2d655d420a39b0230e`:
[assembly.yaml](https://github.com/tenstorrent/tt-metal/blob/201312fe7960b3711a420e2d655d420a39b0230e/tt_metal/tt-llk/tt_llk_blackhole/instructions/assembly.yaml),
[encoding macro](https://github.com/tenstorrent/tt-metal/blob/201312fe7960b3711a420e2d655d420a39b0230e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h),
and [mode constants](https://github.com/tenstorrent/tt-metal/blob/201312fe7960b3711a420e2d655d420a39b0230e/tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_instr_params.h).
These are research evidence, not a replacement for the pinned vendor tree.

Only documented mode hypotheses were tested, on retired partial data in healthy
banks; no arbitrary bit sweep or wedged-tile probe was used:

| SrcA word | Silicon observation | Run |
|---|---|---|
| `0x43000102` | Delay mode; T0 completed, T1 matrix read timed out | `1791411974` |
| `0x430000c1` | Clear mode without Set_Dvalid; T1 timed out | `1791412394` |
| `0x430001e1` | Completed and released, but staged readback was zero | `1791412473` |
| `0x430000c2` | Delay mode with selector 3; T1 timed out | `1791412550` |
| `0x430001e9` | Non-clearing publication and release passed | `1791412640` |
| `0x438001e9` | Independent SrcB publication and release passed | `1791412708` |

After each timeout, candidate execution stopped and the existing Session
recovery plus validated matmul passed (`1791412203`, `1791412455`,
`1791412626`). Neither recovery nor watchdog policy was changed. No new reboot
occurred: boot `14d13ced-d7bb-4c81-ba18-505d1dd8c540` remained stable.
The wrong legacy word selects stream-pop mode 3; in that mode bit 2 is an
enhanced-overlay control rather than a source-clear value. No stream-overlay
lifecycle was established. That is the isolated encoding error and reboot
trigger; the internal NoC/ARC path from it to host reboot remains unmeasured.

Checked helpers drain C1/C2, wait for current-bank unpacker ownership C5/C6,
then issue this fixed profile, both waits with `Before::EVERYTHING`. Retirement
precedes publication of the declared semaphore to math. Output format and
configuration bank remain explicit builder invariants, not typestate proofs.
Full alternating-bank A (`1791412921`) and B (`1791413005`) pass. Combined A/B,
TF32/BF16 changes, special-value physical conversion, untouched-row sentinels,
multiple partials, repeated programs and changed-data negative controls pass
`1791413233`. Direct explicit-handover matmul/pooling with changed inputs and
SrcB source-row reset to nonzero base pass `1791413291`.

Run `1791413005` also caught a regular-control builder error: dividing ADC input
into two halves does not concatenate Src output on this configuration; the
second UNPACR restarts its output address. Both roles completed and there was no
reboot. The multipart diagnostic now re-stages the same range before the final
row, retaining identical control/explicit data. This does not establish arbitrary
disjoint multipart placement. The nonzero row-reset gate enables documented
SrcB row update: a partial advances 0 to 32, handover resets to base 16, and the
next bank receives changed BF16 data at row 16. SrcA uses the required
`SetOvrdWithAddr=1` address override; the hidden SrcA counter is not observable
on that supported path. No unsupported override mode was probed.

Generator inputs now register only this fixed non-clearing profile as measured;
the old Wormhole encoder remains in its compatibility namespace. Other NOP
modes, production adoption, performance and recovery usage remain outside this
evidence.

Final card-0 release SMOKE passes 258/258 (`1791413478`), including step103
9/9 and step104 16/16. The tightened direct consumer gate passes `1791413841`:
inputs 2.0078125 and 4.015625 retain their low mantissa bit in TF32 matmul and
truncate it in BF16 pooling, with exact binary results. MNIST e2e passes 8/8
against the unchanged golden. No firmware handler/mailbox ABI or production
tensor route changed; no performance claim accompanies diagnostic adoption.

## Close-out findings (2026-10-10, measured on card 0 unless noted)

- **A parked Tensix thread survives the backend reset.** A gate that aborts while a thread waits on a
  semaphore (or in a blocking atomic) leaves it parked; every later gate on that tile then times out
  in the harness preamble (`core did not respond within 1000 ms`). The preamble now uses
  `session::reset_thread_state`, which releases semaphores first. The apparent "SFPLOADMACRO hang"
  during lane H was this wedge.
- **Polling the Sync Unit semaphore window.** Back-to-back raw loads (`PollMode::Full`) hang after the
  first poll. One consumed load per poll (`Light`, the default) runs at about 34.7 million polls/s
  and an L1-word completion poll at about 32.8 million, steady over 3 s; `GuardSpec::for_seconds`
  refuses more than 2^30 polls (about 35 s). Mutexes, atomics and the deadline/host-release path
  all use the guarded runner (steps 105/106).
- **Blocking atomics.** `ATCAS` and `ATINCGETPTR` keep the shared Scalar Unit busy for the whole
  retry loop, so another Tensix thread cannot issue even a `SETDMAREG` until the atomic is freed
  (`blocked_atomic_monopolizes_the_scalar_unit`). Only the host or another core's RISC-V store can
  free them; a blocking atomic must never synchronize Tensix threads with each other.
- **Mutexes** 0, 2, 3 and 4 all work on every thread with round-robin handoff; `ATINCGET` wraps its
  field and preserves upper bits and is atomic across threads; `ATSWAP` group form is exact for all
  256 masks and aligned bases, while the **single-register form places data in lanes that follow no
  consistent rule** (excluded, sweep in `atswap_single_form_sweep_diagnostic`).
- **`SFPLOADMACRO`** (probes s00-s13): Store, MAD, Simple and Round sub-units, `LReg[16]`, operand
  substitution, the chain, pipelined macros, predication, forgetting and `SFPSWAP` in the Simple
  sub-unit all agree with the page-derived schedule model. The macro configuration is persistent
  state: write every register a program reads and restore with `MacroConfig::teardown`.
- **L1 tag search accelerator.** An armed trigger load hangs the baby core (all ten scenarios stop at
  step 0, load issued) and the block stays armed across resets, so the earlier passing minimal probe
  fails afterwards. Not adopted; do not run the (ignored) step120 silicon tests.
- **Packer and unpacker modes.** BF16 `UnpackToDst` widens correctly; packer ReLU (all seven modes)
  and edge masking (row sets, partial columns, -inf fill, edge-then-ReLU order) match raw-bit models;
  unpacker tileize is a payload-preserving strided gather; unpacker transpose swaps rows and columns
  but goes through SrcA, and the plain SrcA path itself normalizes -0, subnormals and low mantissa
  bits (`MEASURE src.plain`). The 16-bit Dst read path preserves normal BF16 exactly but maps -0 and
  subnormals to +0 and every NaN to infinity.
- **FP16 packer** (`T7-MEASURE`): the raw packer truncates and saturates (equal to ttsim); the
  rounding packer is ties-even for normals and overflows to infinity from 65520 but drops NaN
  payloads (`0x7e00`) and flushes subnormals.
- **Matrix diagnostics.** `MOVDBGA2D` one/eight-row moves, all four SrcA format overrides (forced to
  TF32 under Fp32), flush and reading a bank the unpacker still owns agree with the page model;
  the increment-1 `AddrMod` advance cases disagree for the `MOVA2D` control too (increments of 8
  agree), so they are an open shared configuration issue. `GATESRCRST` executes safely but has no
  observable effect against a SrcB rewritten through `MOVD2B`.
- **NoC.** All 21 NoC atomic forms (variable-width increment, CAS, mask and indexed swaps, eight
  Zaamo ops, six accumulate formats) match the page models against a neighbouring tile's L1. The
  `NIU_TRANS_COUNT_RTZ_SOURCE` read follows completion; its stickiness is not established.
  Multicast probes were not run.
- **PRNG.** Silicon lane `i` after a restart is `advance^(98-2i)(seed)` (8 seeds x 32 lanes, both
  cards); the seed directive in the role firmware (RISC-V store, fence, 512 NOPs) reproduces the
  model bit for bit under T0 unpack and T2 pack concurrency.
- **Release baselines** for K-blocking and the norm compositions are in `firmware-performance.md`.
