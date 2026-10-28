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
performance-log.md. No mesh route or automatic SFPU replacement is enabled.

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
performance-log.md. No new simulator divergence was observed.

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
and conversion cost are recorded in `performance-log.md`.

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
record in performance-log.md for medians and conditions.

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
performance-log.md. Ordinary tranche acceptance follows the current
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
  a one-row move naming an odd `AddrMod` entry (instruction bit 14 set, eight-row bit clear) writes a
  four-row aligned block on its own first move, for `MOVA2D` and `MOVDBGA2D` alike and whatever the
  entry's increments; the increments then apply as modelled (entry *k* = SrcA +*k*, Dst +7-*k* in the
  sweep), so a one-row move that must write one row names an even entry (`probe_addr_mod_sweep`). `GATESRCRST` executes safely but has no
  observable effect against a SrcB rewritten through `MOVD2B`.
- **NoC.** All 21 NoC atomic forms (variable-width increment, CAS, mask and indexed swaps, eight
  Zaamo ops, six accumulate formats) match the page models against a neighbouring tile's L1. The
  `NIU_TRANS_COUNT_RTZ_SOURCE` read follows completion and the bit is sticky until cleared. Multicast
  writes reach every recipient of a 1x2 and a larger rectangle and no non-recipient (cards 0 and 1, one
  probe per process, no hang or reboot). Request initiator 1 of NoC #0 keeps its registers between
  requests, so `set_state`/`with_state` style issue works (X6, `step118_mover_fast_path`); it is only 3-4%
  faster than rewriting initiator 0's registers (`performance-log.md`).
- **PRNG.** Silicon lane `i` after a restart is `advance^(98-2i)(seed)` (8 seeds x 32 lanes, both
  cards); the seed directive in the role firmware (RISC-V store, fence, 512 NOPs) reproduces the
  model bit for bit under T0 unpack and T2 pack concurrency.
- **Release baselines** for K-blocking and the norm compositions are in `performance-log.md`.

## `.ttinsn` push fusion (2026-10-10, card 0)

`docs/plans/ttinsn-push-experiment.md`; gates `probe_ttinsn_push`, run only with
`cargo xtask silicon --feature ttinsn-experiment` (a feature no shippable build enables).

- **Fused `.ttinsn` words run on silicon.** On RISC-V T1 pushing to Tensix thread 1 (`INSTRN_BUF_BASE`):
  one fused NOP, four adjacent ones (the instruction cache's fusion case), a 130-word program and the full
  ~11,600-word precise `gelu` retire with no hang, and leave `Dst` identical to the runner's push. The
  push phase is 2.78x faster (`performance-log.md`). Only the T1-to-thread-1 pairing was exercised.
- **The instruction-set gate could not see them, and could not see any `.word` at all.** The assembler marks
  `.word` as data, so `llvm-objdump` prints it as `.word` rather than `<unknown>` or `c.*`, and the old
  loop read its first column as a mnemonic. A stray compressed-space word in any image would have passed.
  No image had any data in a code section, so the gate now refuses all of it, and the experiment image's
  `.ttinsn` words are validated byte by byte from the section dump instead (`check_ttinsn_blocks`).
- **`Dst` survives from one run to the next on silicon.** Comparing two runs of a program in one process
  compares the second against the first's leftovers: a push that wrote nothing matched, and a control that
  halved the loop count passed unchanged. Poison the rows before each run (`SENTINEL` stores in the
  program's prelude) before trusting an equality, and keep a control that can fail.
- The probe's first checksum cancelled to zero on the repeated four-row pattern; a comparison that cannot
  tell results apart validates nothing.
- **`SFPSTORE` writes one column parity.** At address `a` (a multiple of four) it writes the even columns
  of rows `a..a + 4`, at `a + 2` the odd columns. A program that defines only one parity and reads both
  reads whatever earlier runs left (a kernel's unpacker writes whole rows), so its results differ from
  session to session and from ttsim. Define both.
- **Run-time code works, with the documented invalidation.** RISC-V T1 wrote `a0 = 1; ret` into L1,
  called it, rewrote it to `a0 = 2` and called it again *without* invalidating: it returned 1, so the
  instruction cache holds code written at run time and serves it stale. After the documented sequence
  (store, read the last word back and consume it, write the mask to `RISCV_IC_INVALIDATE_InvalidateAll`,
  then 24 more instructions; mask bit 2 is T1, a core's own cache) it returned 2. `InstructionCache.md` is
  the Wormhole page (the Blackhole tree links it); this is the Blackhole confirmation for T1. The whole
  sequence costs about 330 cycles. Only T1 invalidating itself was exercised, not one core invalidating
  another's.
- **Translated code is as fast as built-in code once running**, and a fused push is about 1.19 cycles a
  word against the runner's 3.31, but translating costs 13-22 cycles a word (`performance-log.md`).
- **Instruction gathering is on at boot, and tt-metal turns it off.** Blackhole boots with the baby cores'
  hardware gathering of adjacent `.ttinsn` pushes enabled. tt-metal's firmware disables it on every RISC
  around Tensix at start (`configure_gathering()` in `tt_metal/hw/inc/internal/firmware_common.h`:
  `cfg0` CSR `0x7c0`, set bit 1, set bit 18, clear bit 1, with fences) as a workaround for
  tt-metal#16439, and the LLK brackets every `REPLAY` record window with disable/enable if it is on
  ("gathering is done early in the pipeline": a record window can be corrupted when the instructions it
  records are gathered; tt-metal checks this by inspecting the built ELFs, because it is "a codegen
  lottery" that a kernel passes with gathering on). Measured here, gathering has **no effect on speed**
  (a baked fused push is 13,868 cycles with it on or off), and it only ever fuses adjacent `.ttinsn`
  words (`cfg0.DisTriscCache`, `CSRs.md`), which our runner's `lw`/`sw` pushes are not.
  **Since 2026-10-11 every Tensix image turns it off** at start (T0, T1, T2, B and NC, through the
  runner entry and the movers: `tt_firmware::disable_gathering`, tt-metal's sequence), reads `cfg0`
  back, and publishes `panic_code::GATHERING_STILL_ON` if bit 18 did not stick, so a core that kept it
  on cannot run unnoticed. The Ethernet image is left alone: the CSR is documented for the Tensix baby
  cores. Verified by `probe_ttinsn_push::experiment::gathering_is_off_by_default_and_the_flag_keeps_it_on`
  (T1's own `cfg0` read: bit 18 set by default, clear with the experiment's `GATHERING_ON` flag, so the
  check can fail) on card 0 and card 1 (run `1791685705`) and on ttsim, and by the whole smoke suite on
  both cards with the new firmware. The experiment's `gelu` programs hold no `REPLAY`, so they were never
  exposed; the `short` program (32 `REPLAY`) passed bit-exact with it on. The same CSR carries other
  chicken bits tt-metal sets (bit 3: Blackhole's L1 data cache; bit 0: relaxed load/store ordering); this
  repo has never set any and they are untested here. **`cfg0` can outlive a core reset** (ttsim keeps a
  previous run's bit 18 across one; `CSRs.md` says reset clears every field): the experiment's
  gathering-on mode therefore clears the bit explicitly instead of skipping the write, and the shipped
  images set it unconditionally, so neither depends on what ran before.
- **Under `cargo test`, tests that fork a ttsim hung about one run in five when run concurrently**
  (idle processes, always inside a forked child) and never single-threaded; not diagnosed. The
  experiment's ttsim tests serialize on a mutex.

### Fused code from the program cache (2026-10-25, card 0)

`docs/plans/ttinsn-push.md`; gate `step231_ttinsn_push` (ttsim and card 0, run `1792956507`), opt-in
(`TT_TTINSN=1`).

- **The shipped T1 runner calls code the host wrote into the program cache over the NoC** (fused `.ttinsn`
  code, `tt_firmware::fused`), after invalidating its own instruction cache with the documented sequence,
  once a run. The program cache reuses its addresses, so without the invalidation a run would execute what
  the cache held of the previous program there (the experiment's stale-code control). Verified on silicon
  with a real reuse: GELU's fused code run, the cache forgotten, TANH's fused code placed at the same address
  (`Session::probe_forget_programs`, `probe_last_fused_at`): TANH's bits, not GELU's.
- **The host's NoC writes need no device-side read-back before the invalidation**: the runner reads one
  word of the code (any) as the sequence's read-back; the host's writes landed before the list that names
  the code was queued (the order every resident program already relies on).
- **The pushing measured from the runner** (the profile mode's push totals, `mailbox::PUSH_CYCLES`,
  `PUSH_WORDS`): a push into a full instruction FIFO stalls the core, so cycles spent pushing are not
  push-boundness. A kernel is bound by its pushing when its words go at the runner's ~3.3 cycles each; the
  1.7B's products push at ~40-500 cycles a word (behind the Matrix Unit), its `silu_mul` at 5.6 and its
  RMSNorm at 7.3 (runs `1792956684`).

## Card lock (2026-10-11, both cards)

`Session::open_card` takes the driver lock `SESSION_LOCK` (index 63 of 64, clear of the Ethernet cores'
0-15) on the descriptor it opened, right after `Kmd::open` and before anything is written to the card
(`Kmd::lock_card`, `tt-kmd`). Measured by `silicon_card_lock` on card 0 and card 1:

- A second `Session::open_card` of the same card is refused (`SessionError::CardBusy`, `TtError::CardBusy`)
  while the first lives, **in the same process and from another process**, and succeeds as soon as the
  owner is dropped or its process exits. Two descriptors in one process conflict: the driver ties the lock
  to the open file description (`chardev.c`, `ioctl_lock_ctl`), not to the process.
- The lock is released only by closing the descriptor, so a wedged process that is merely stuck (not dead)
  keeps the card: the next open says `CardBusy` instead of resetting tiles under it. `xtask silicon` kills only
  the test process on a timeout, not a `fork_scope` grandchild, so a timed-out gate can leave later gates in
  a `--keep-going` run refused until the grandchild is gone.
- Raw `Kmd::open` (the harness's `backend::open_card`, `on_card`, `in_device`) takes no lock: a gate that
  uses it neither blocks nor is blocked by a session, which is what the multi-card Ethernet gates rely on.
- **`LOCK_CTL`'s `output_size_bytes` is the size of the `out` half (4 bytes), not of the whole struct.** The
  driver does `clear_user(&arg->out, in.output_size_bytes)` after the operation, so the whole struct's 16
  bytes clear 12 bytes past the argument, on the caller's stack. The first version of `ioctl::lock_ctl` (never
  called before this) did that and the process died with SIGSEGV (exit 139) at its first `open_card`.

## Matmul accumulation: the Matrix Unit's group adder (2026-10-11, card 0 and ttsim)

Found by two of Burn's `conv3d` tests (`test_conv3d_groups`, `test_conv3d_complex`: `arange` operands, outputs of
1.7e5 to 4.8e6) and decomposed to a model that equals the device bit for bit (gate `step158_matmul_accumulation`,
ttsim and card 0, run `1791754387`). **It is not TF32 operand truncation**: the operands there are integers below
2048, exact in TF32, and every sum is exact in F32. A tile matmul at HiFi4 still returns results off by whole
units once the sum is large (`[32, 64] @ [64, 1]` of integers below 250 and 53: 32 of 32 rows inexact, worst 8 at
2.3e5), because the unit does not add its products in F32.

**The rule** (`MVMUL.md` calls its float model "a rough guide"; this is measured):

- One `MVMUL` is a 16-term dot product, run once per fidelity phase (four at HiFi4: SrcB `hi`/`lo` times SrcA
  `hi`/`lo`). Its terms are **two groups of eight** (k = 0..7 and 8..15).
- In a group, a term's exponent is `exp(a) + exp(b)`, the sum of the *operands'* exponents, not the product's (with
  `SrcA = 3` the quantum was 2^(16+1-10) = 128, not the 256 of a product of exponent 18). A low slice's exponent is
  taken from its position (SrcB 7 below the operand's, SrcA 5 below) and **a zero slice still sets it**.
- The group is aligned once to a quantum `2^(max term exponent - 10)` (ten fraction bits, the quantum step90 found
  for `ELWADD`), every product is rounded **half-up on its magnitude** to a multiple of it, and the products are
  summed exactly. Terms are not added in a tree: two 31s beside a 1024 vanish together wherever they sit.
- The two group sums go to the F32 `Dst` as two adds, each to nearest with **ties toward +infinity** (a different
  rule from the group's); the four phases of a face run in order, faces in order.
- Across groups, `MVMUL`s, faces and tiles nothing else is lost: `P` at k = 0 and `Q` at k = 8 or 16 or 40 is `P + Q`.
  In one group it is `P + round(Q)`.

A term that is small beside its group's largest therefore loses its low bits, and rounding up is one-sided, so the
error is mostly high (26 of 27 in `groups`). A group of terms of one size loses nothing, which is why small
integers and `step85`/`step157` data equal exact arithmetic and `arange` data does not.

**Bound** (derived from the rule, exact operands): `|result - exact| <= sum over phases, faces, groups of
(nonzero terms) x quantum / 2, plus one ulp of Dst per add`. The worst measured error over thousands of rows was 0.44
of it. It is not small: a group of one 1024 and seven 0.5s gives 1031 for 1027.5 (error 3.5, `2^-8.2`), with every
operand exact in TF32 and every sum exact in F32; with the 0.5s in the other group it is exact. That is **larger than
the `2^-9 sum|ab|` the `step68` and `step85` bounds charge for TF32 truncation**, and those bounds charge nothing for
this alignment, so they hold on their own (fractional, similar-sized) data but are not a guarantee for an
adversarial exponent spread. Re-derive them from this rule before relying on them elsewhere.

**Consequences.** A multi-pass TF32 operand split of the *bits* cannot repair it: the operands are already exact (but see "Making every term a multiple of the quantum" below, which does). The loss is
a property of the FPU's reduction, so a fix is a different reduction, not more operand bits: that is
`burn_tt::MatmulMode::Exact` (2026-10-11, F32 SFPU products summed in `k` order, opt-in; `step159_exact_matmul`),
which passes every Burn test the Matrix Unit's rounding failed, on silicon too (all 24 LU tests, the 300 to 500 square
and rectangular ones taking 95 to 579 s, run `1791758996`; the determinants, attention and CTC too), at 1,000x to 22,000x the cost. For fractional float data the TF32 truncation (about 2^-11 per operand) is of the same order as
a group's rounding, so the effect shows chiefly with exact data and large sums. Not covered: LoFi to HiFi3, BF16
`Src`, `Dst` as BF16, subnormals, NaN and infinity. ttsim and silicon agree on all of it.

### Making every term a multiple of the quantum (2026-10-12, ttsim and card 0)

The loss happens only where a term is not a multiple of its group's quantum `2^(max operand-exponent sum - 10)`. If the
operands are **integers** whose significands fit the slice a phase takes, the largest exponent sum is at most 6 + 4 = 10 at
`Fidelity::Lo` (one phase: `SrcB`'s 7 significant bits times `SrcA`'s 5, masks `0xfffe_0000` and `0xfff8_0000`), so the
quantum is `2^0 = 1` and every product and every sum of eight is exact. Gate: `step158_matmul_accumulation`,
`lofi_is_exact_on_seven_by_five_bit_integers` (|lhs| <= 127, |rhs| <= 31, mixed magnitudes inside one group, ragged `k`, `k` up to
4096 over many K-blocks, sums under 2^24: equal to exact arithmetic on every row) and `lofi_loses_a_bit_past_the_slice` (an 8-bit
lhs or a 6-bit rhs gives wrong rows: the masks cut the extra bit). Planting one 8-bit operand in the first makes it fail. Both pass on ttsim and on card 0 (run `1791818525`): the LoFi rule is the
same on silicon.

That is the Ozaki scheme: scale each row of the left operand and column of the right one by its own power of two, cut each into
integer slices of 7 and 5 bits (`trunc`, then the remainder times the radix), run a `Fidelity::Lo` matmul per slice pair and add
the partial products from the smallest to the largest. `MatmulMode::Accurate` (`ops/accurate_matmul.rs`, gate
`step163_accurate_matmul`) does this with twelve pairs (total weight `7 (s - 1) + 5 (t - 1)` up to 23, past an F32's 24 bits);
measured against an f64 reference on ttsim its error is about 1e-7 of the largest entry (the fast matmul's is about 1e-3), worst
0.21 of the derived bound (the same gate passes on card 0, run `1791818535`), and it equals exact arithmetic on integer data whose sums stay below 2^24, `arange` data of Burn's
conv3d tests included. **Limits that follow from the construction:** the bound scales with each row's and column's largest
magnitude, not the entry's, so a row of very different magnitudes keeps its small entries only to that scale; operands are clamped
to `|x| <= 2^64`; a subnormal is zero. **Cost on silicon (run `1791818553`): 19x to 64x the fast matmul** (18.7x at 1024 cubed), 50x to 750x
faster than the exact mode; the time is mostly the element-wise scaling, slicing and NaN/infinity classification around the
matmuls, not the LoFi matmuls ([performance-log.md](performance-log.md#accurate-matmul-cost-2026-10-12-card-0)).

## RISCV B's image budget (2026-10-13, ttsim and card 0)

**Update 2026-10-13: B's slot is 42 KiB, not 24 KiB.** The role images moved (below); the measurements and
lessons that follow were taken against the old 24,576-byte slot and still describe how code size moves.

B's entry point is hardwired to L1 offset 0, so its image can only grow upward, up to whatever is placed next. That
used to be T0's *default* reset PC (0x6000). The three role images use about 8 KB each, but each sat in a 16 KiB
slot at its core's default reset PC; T0, T1 and T2 have reset-PC overrides, and `Device::load_held` sets one for
every core but B. The role images are now linked at `tt_isa::l1::roles::BASE` (T0 `0xA800`, T1 `0xD000`, T2
`0xF800`, 10 KiB slots ending at NC's reset PC), so `tt_isa::dm::IMAGE_MAX` is 43,008 bytes (`dm_b`: 24,404, so 18,604
left). Only the production role images moved; probe images (`corpus`, `icache_t*`, ...) keep the default PCs and are
never loaded beside the mover. Silicon is unchanged in speed ([performance-log.md](performance-log.md#firmware-image-budget-and-core-assignment-2026-10-13-card-0)).
Also measured there: `opt-level` 2 is as fast as 3 and 1.7 KB smaller; `s` and `z` are not usable (mover 1.6x
slower, role push loop 7x).

Before the role images moved, RISCV B's mover image (`dm_b`) could not pass 0x6000 = 24,576 bytes: T0's default
reset PC is where the next image starts, and B (entry hardwired at 0) has no reset-PC override. The link fails
("section '.text' will not fit in region 'L1'") rather than clobbering T0. Measured with `size -A` on the ELF
(`.text` + `.rodata` + `.bss`; git `e17f4c9` plus the row select):

| Image | Bytes | Left |
|---|--:|--:|
| `dm_b` before `SELECT_ROWS` | 23,644 | 932 |
| with `SELECT_ROWS` (32-bit datums), first build | 24,532 | 44 |
| the same arm with the half-loop kept rolled (`black_box`), as built | 24,404 | 172 |

Packed BF16, measured against the first build (24,532): a branch that reads the 64-byte pair into a scratch
slot and then runs `COPY_HALFWORDS` of the wanted half linked at 24,932 (**356 over**; a second emit site in
the loop instead of one made it 684 over), and the pair read alone, with no copy, at 24,600 (**24 over**).
These were not re-measured after the rolled loop saved 108 bytes, so the pair read alone should now fit.

What the numbers taught (each from a build, not from reading code):
- **One record arm cost 760 bytes.** On rv32im every `u64` add or multiply is several instructions: a checked
  `u64` tile-number computation in `Cursor::new` cost 164 bytes on its own, so the record checks the product
  once (`bound.div_ceil(32) x ct`) and the cursor adds in `u32`.
- **LLVM unrolled the two-iteration loop over a face-row's halves** and duplicated the emit and address code:
  `for half in 0..core::hint::black_box(2)` kept it a loop and saved 108 bytes. Look for this before cutting a
  check.
- **Sizes move by tens of bytes with unrelated changes** (inlining decisions shift): one build of the same arm
  was 8 bytes worse after a change that removed code. Measure each candidate; do not add the estimates.
- **`cargo clippy` made every image about 900 bytes larger** (`dm_b` 24,540 against 23,644 for the same
  source): `tt-firmware-images/build.rs` stripped `RUSTC_WRAPPER` from the nested firmware `cargo` but not
  `RUSTC_WORKSPACE_WRAPPER`, which clippy sets to `clippy-driver`. A change that fit a build overflowed the
  CI lint. It strips both now, so the firmware is the same bytes whoever builds it; firmware is linted on its own
  (`cargo clippy --manifest-path crates/tt-firmware/Cargo.toml ...`).
- RISCV NC's image (`dm_nc`, 14,240 bytes of its 24,576-byte slot, `dm::nc::IMAGE_MAX`) has about 10 KB free.
- **NC's first index-reading record, `SCATTER_ROWS` (2026-10-15, `size -A`, same path, release):** `dm_nc` 13,908 ->
  15,088 bytes (9.5 KB of its slot free). `dm_b` grew too, 24,608 -> 25,724, although B never runs the record (its
  arm is compiled out of B): giving the new op a length in `record::len` alone moved LLVM's inlining in B's
  `run_record` (23 `record_entry` call sites against 10). Neither the opcode's place (0x12, inside the record range,
  or 0x1A after it) nor where `len` lists it changed that. B has ~17 KB of its 42 KiB slot left. The record is the
  writing half the split BF16 select below needs: NC reads its index slot after `publish()`, one index slot a staging
  half.
- **Writes to one GDDR address through different ports are not ordered**: `write_rows` and `SCATTER_ROWS` rotate
  ports, and each port is its own endpoint (`port_for`). A repeated index is therefore resolved in the record (only
  the last index naming a row is written), and across batches by the writer's credit pop, which waits for its
  writes (`Action::Pop`).

**Open: a packed-BF16 row select** (roadmap, Track 2, "Gather without staging"). A BF16 face-row is 32 bytes at
`TILE_DATA + face x 512 + row x 32`: congruent to `TILE_DATA` mod 64 (the GDDR read rule) only on an even row,
and a read's L1 destination must be congruent to its source, so a wanted row of one parity cannot be read
straight into an output row of the other. Every arrangement that lands the wanted row at one fixed L1 place
fails the same congruence for one of the two source parities. The candidates:
1. The B-side branch above: 400 bytes at the time it was measured, more than B had.
2. **Split it.** B reads the 64-byte pair containing the wanted row into a staged tile (a stride and a mask on the
   F32 arm: about 70 bytes on B, which fits) and NC, which has the room, writes the 32 bytes at `pair + 32 x
   (index & 1)` to the output tile's face-row (a write needs only 16-byte congruence). That needs a writing
   record that reads the index tile, so each half of the staging needs its own index slot (B reads the next
   batch's indices while NC still writes the last), and the output's rows past `n` are not written, so its padding
   is `Undefined`.
3. Make room in B: no longer needed -- B's slot grew by 18 KiB (above), so the B-side branch (1) fits.

## Partial-tile GDDR reads in records: one entry beats fewer bytes (2026-10-16, ttsim and cards 0 and 1)

From [decode-matmul-bandwidth.md](../plans/decode-matmul-bandwidth.md) P1 (log:
[decode matmuls](performance-log.md#decode-matmuls-read-only-the-activation-rows-they-use-2026-10-16-cards-0-and-1)).

- **A read shorter than a slot is legal and lands as asked.** `Descriptor::decode` checks only a nonzero length, the
  range inside its channel and `gddr % 64 == l1 % 64`. A read that starts at the same offset inside a GDDR slot and
  an L1 slot is congruent for any length, since both slots are 64-byte aligned (`TILE_SLOT` and `BF16_TILE_SLOT` are
  multiples of 64). Records expand on the card through `Descriptor::decode` (never `READ_ENCODED`), so the
  fast-reads encoder plays no part. Gates `step190` (both read modes) and `step77` on card 0.
- **The B mover pays per entry as much as per byte.** For an `m = 1` product's `A` tiles (one used row in 32):
  - **Two 64-byte face-row reads a tile** (1/32 of the bytes, twice the entries): a 1.7B decode step 1.5 ms faster,
    but the 135M step ~1 ms *slower* (its products are 18 `A` tiles a job).
  - **One read through row `r` of face 1** (`1024 + 64 r` bytes, 27% for `r = 1`, the same entry count): the 1.7B
    step 6.1 ms faster and the 135M step 0.3 ms.

  After it, a `q` job's 128 entries take ~60 us on each of 64 units (~0.45 us an entry, ~290 GB/s card-wide), so
  these products are bound by B's record-expanded entries, not GDDR. Choose a read shape by entry count first.
- **L1 left unread is the next op's data.** Silicon keeps L1 between programs, so a slot's unread rows hold a
  previous op's values (NaN in `step190`'s priming). They are safe only where they reach nothing but padding, with
  the output's padding claim `Undefined`; `step190`'s negative control shows a false `Pad::Zero` there makes
  `C^T @ C` non-finite.
- **B's image:** 25,724 -> 26,528 bytes of 43,008 (default build); NC unchanged at 15,088.

## A decode product's weights: what the card reads at (2026-10-17, card 0)

From [decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 0 (log:
[decode products against their weight stream](performance-log.md#decode-products-against-their-weight-stream-2026-10-17-card-0)).
The probe is `silicon_probe_decode_stream`, traced, so the times are the card's.

- **The read ceiling is ~488 GB/s, 95% of 512.**
  - **Pattern:** every tile of a weight read once as `READ_RUN`s of consecutive tiles, on 120 units: 485-489 GB/s
    for 17-137 MB.
  - **Method:** whole 4,160-byte slots, record-expanded, in the default (safe) image, so `fast-reads` is not needed
    for it.
  - **Against the scoreboard:** above its 469 GB/s.
- **One channel a column costs ~7%, not more.** Each unit reads whole columns down their `K` tiles, as an `m = 1`
  product deals `B`, so all of a column's tiles are on channel `j % 8`. That pattern reads at 440-458 GB/s, on 64
  units and on 120 alike.
  - **This refutes** decode-matmul-bandwidth P5's leading hypothesis, that channels bind the narrow products.
- **64 units saturate GDDR.** One unit streams ~7 GB/s (a 266 KB column share in ~38 us), so ~64 units reach the
  ceiling.
  - **Past 64, units buy no bandwidth.** An `m = 1` product on 120 units is no faster than on 64:
    - `o` and `down` are the same;
    - `qkv` 180 against 160 us, `gate|up` 383 against 352, the 1.7B head 1,051 against 1,024.
  - **The 135M head is the exception.** At `K` = 576 it prefers 48-120 units.
- **So the products' gap is not the read stream.** A product takes 1.2-1.8x its weights' stream on the same units:
  - `o` 74 against 38 us;
  - `down` 269 against 150;
  - `qkv` 160 against 113;
  - `gate|up` 352 against 300.
- **The profile mode shows the same.** It splits a product into the mover's data phase, a ~15 us kernel tail and
  the barrier. The data phase is `B`'s stream plus `A`'s reads.
- **What the gap is made of:**
  - **`A`'s reads:** one entry a `K` tile on every unit. That is 64 at `K` = 2048 and 256 for `down`'s 8192;
  - **the compute after the last read:** ~15 us an op, not overlapped;
  - **a deal's last round:** uneven on 120 units.
- **The "kernel tail" is compute, and it scales with the job** (profile `1792298755`, read 2026-10-18). The profile's
  `kernels` phase is unit 0's `KERNEL_WAIT`: the last block's compute, ~0.23 us a `K`-tile pair (~305 cycles) plus
  ~0.5 us. Attention's `qk`/`pv` (2 pairs) 1.0 us, `o` and `qkv`'s last job (64 pairs) 15.0, `down`'s last K-block
  (64) 15.3, `gate|up`'s last 2-column job (128) 29.6.
  - **The kernel is already launched before the reads** (`streaming_segments`: `LAUNCH` first). Its start-up is
    hidden; the block's compute waits for the whole gather (`CB Push` drains the reads).
  - **Earlier blocks' compute is serialized too.** Under `Staging::Slots` the ring holds one block, so a unit's next
    gather waits for the block before it to be computed. That time is counted as `data`. `down` (4 K-blocks a unit)
    also round-trips each partial through GDDR.
  - **An m = 1 pair may do ~4x the work it needs** (from the code, not yet measured): `matmul_items_reloading`
    unpacks and multiplies all four faces (64 `MVMUL`s a pair at HiFi4), where row 0 needs 16. Whether a pair is
    bound by its `MVMUL`s or its unpacks is unmeasured.

## A BF16 `B` beside a TF32 `A`: the same product (2026-10-26, card 0 and ttsim)

Gate `step232_mixed_src_formats` (card 0 `1792986764`): `A` FP32 in L1 unpacked to `SrcB` under `OutDataFormat` TF32
(4), `B` BF16 in L1 unpacked to `SrcA` under BF16 (5), is **bit for bit the product of the same values both in FP32
through TF32**, at LoFi, HiFi2 and HiFi4, with signed zeros, BF16 subnormals, `bf16::MAX`, infinities and NaN in `B`.
So the Matrix Unit takes the two `Src` registers in different formats, and a BF16 datum in `SrcA` is the TF32 datum of
the same value (the phases slice by bit position). What it buys: a weight stored BF16 (half the bytes) multiplies as
its F32 widening (`SrcRoute::Tf32WithBf16B`, [bf16-weight-storage.md](../plans/bf16-weight-storage.md)). BF16 in with
TF32 out (code 5 in, 4 out) is not a way to write it: the specification calls a mismatched pair undefined and ttsim
refuses it (`in_data_format=5 mismatches out_data_format=4`); not tried on silicon. Blackhole's `UNPACR_Regular.md` is
a stub; this is measured, not read off Wormhole's `WriteSrcBF16`.

**A trap found on the way (fixed):** unpacker 0's address counters outlive a kernel. A product's face selects
(`SETADCZW`) leave its `Z` counter on a later face; the BF16 widening (`bf16::expand`, through `SrcA` and `MOVA2D`) set
only `X`, so every widening after any product read the wrong datums (on ttsim and card 0 alike: `step233`'s widening,
elements shifted by whole faces). It now clears the counters first, as the narrowing always did. Any kernel that
unpacks after another must set every counter it relies on.

## A whole tile in each `Src`, rows reached through the RWCs (2026-10-26, ttsim and card 1)

Probes `probe_pair_src_rows` (each alone on one unit; card 1 `1793037041`, `1793037048`, `1793037050`,
`1793037053`; ttsim first), and the product gates after them (`step229` `pair_orders_are_the_faces_order_bit_for_bit`,
card 1 `1793037349`): exact against host integer sums and bit for bit the face-at-a-time product, at one to four
phases, `B` transposed by the unpacker or stored BF16, one, sixteen and all of a tile's rows.
- **One `UNPACR` of 1,024 datums fills `SrcA` rows 0-63** (`SETADCXX` `X1` 1023 from `Z` 0, `SrcA` row 0 at
  `Dest_cntx0` 64, `SetOvrdWithAddr` set; Wormhole's `UNPACR_Regular.md` says `Row &= 63` on Blackhole). With
  `Haloize_mode` each 16-row face is transposed in place.
- **Unpacker 1 fills `SrcB` rows 0-63** the same way (512 datums for rows 0-31).
- **`MVMUL` reads `SrcA` at `RWCs.SrcA` 16, 32 and 48 and `SrcB` at 8 to 56**, the RWCs moved only by address
  modifiers: plain increments, the carry form (`SrcACR`/`SrcBCR`: `Cr += Incr`, the RWC to it) and clears, as `RWCs.md`
  `ApplyAddrMod` says. A tile pair is then two unpacks (`matmul::PairOrder::Tiles`), not twelve.
- **The pair is not bound by its unpacks.** Two unpacks a pair instead of twelve barely moves a blocked product
  while it gathers its operands (`silicon_probe_pair_cost::pair_orders`); with the gathers left out
  (`without_gathers`, `1793037314`) the tiles kernel runs **23.7-28.8 TFLOPS against the faces kernel's 15.4-15.9**
  (F32 `B`, HiFi4), at LoFi alike. So the kernel was half the products' time, and its unpack count was not what bound it.
- **Phases outer is slower with an F32 `B` only while the product gathers.** Interleaving each `SrcB` half's phases
  with the other half's (`PairOrder::PhasesOuter`, the same unpacks and the same release points: each bank is handed
  back by the same `MVMUL` of the group as in the faces order): with its gathers, HiFi4 15.3 -> 13.7 and 14.9 -> 13.2
  TFLOPS with an F32 `B`, 17.5 -> 18.1 and 15.3 -> 18.9 with a BF16 `B` (`1793036200`, `1793036219`-`1793036229`,
  `1793038281`). What it is not, and what it is:
  - **Not the fused form.** Every (order, `B` format, fidelity) measured takes T1's fused `.ttinsn` form (host report
    `matmul::order` `fused_forms_of_the_pair_cost_products`: 638-3,840 words pushed, all past `FUSE_MIN_WORDS`), and
    with `TT_TTINSN=0` the gap is the same (13.72 / 13.17 against 15.22 / 14.89, `1793038277`).
  - **Not the kernel.** With the gathers left out (`without_gathers`, `1793038270`) the phases outer are faster for
    both formats: F32 `B` HiFi4 15.96 -> 19.92 and 15.50 -> 21.34, BF16 17.75 -> 23.41 and 15.51 -> 21.72; equal at
    LoFi (the same program), and HiFi2 21.40 -> 21.60, 23.80 -> 25.43.
  - **It is the kernel against its gathers.** With them, an F32 `B` product sits at ~14.5-15.3 TFLOPS whatever the
    kernel, and a faster kernel can be slower: the faces order's own LoFi (14.78 / 14.52) and HiFi2 (14.38 / 14.00) run
    below its HiFi4 (15.26 / 14.90) there, as the phases outer do. A kernel that unpacks sooner contends with the mover
    writing the next chunk into L1 (or returns its banks at worse moments for it); a BF16 `B` halves the bytes both
    move, and the faster kernel wins. Which of the two (L1 contention or bank timing) is not isolated.

- **`REG1_ovrd_default_throttle_mode` shares `Config` word 70 with packer 0** (`THCON_SEC0_REG1`): set from the
  unpack role with `ConfigWords` it would clear the packer's formats, so the probe has the pack role write it.
  Unpacker 1's (word 118) ttsim refuses (divergence row 105). Products exact at x1 and x2 with it set.

## An op's recorded L1 top must cover everything it lays out (2026-10-26, card 0 and 1)

Busy prefetch places a unit's next weight ring above the highest `l1_top` the ops since the last product recorded
(`Session::busy_start`), so an op that lays out L1 itself must note its whole reach (`crate::l1::note_planned_top`), not
leave it to the layouts of the kernels it calls. Decode attention's job did not: at `Step(512)` its `K` and `V` slots
sat above what its inner products and reductions noted, the next weights landed on them, and a 1.7B step chose other
tokens (found in `silicon_bench_models::traced_generation`'s 256-token prompt, which the bit-for-bit gates at shorter
lengths never reached; gate `step209` `a_captured_job_records_the_l1_its_slots_reach`). Symptom to recognise: a traced
run that disagrees with itself between the logits path and the token path, fixed by `TT_BUSY_PREFETCH=0`.

## A matmul's transposed `B` through unpacker 0 (2026-10-17, card 0)

From [decode-attention.md](../plans/decode-attention.md) A0b; gate `step201_unpacker_transposed_b`.
- **The mechanism:** a transposed `B` (its tiles in their slots as they lie in GDDR) is unpacked into `SrcA` with
  `Haloize_mode` set, which transposes within each face. `B`'s face `(k, fj)` is unpacked from the slot's face
  `(fj, k)`. The products are **bit for bit** those of the mover's datum-by-datum transpose (`READ_TRANSPOSED`) and
  of `B^T` uploaded plain: TF32, HiFi4, every special value, whole, ragged, thin, K-blocked and batched.
- **What that shows:** the unpack's per-datum conversion into `SrcA` (normalising -0, subnormals and low mantissa
  bits) commutes with the transpose, as the earlier unpacker-mode probes implied.
- **State:** `Haloize_mode` is persistent. Every later unpack configuration writes the whole `SEC0_REG2` word with it
  clear: `unpack_src_config` by construction, `unpack_config` explicitly. So no other kernel unpacks transposed, and
  none unpacks to `Dst` with it set (undefined behaviour).

## A one-row product's compute, `A` reads and weight stream (2026-10-18, cards 0 and 1)

From [decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 4 and 5, SmolLM2-1.7B `Step(64)` profiles on
card 1; gates on card 0.

- **A tile pair is bound by its `MVMUL`s.**
  - **The change:** computing only the rows an output tile holds (`matmul::LiveRows`, `step203`) cut 64 `MVMUL`s a
    pair at HiFi4 to 16 for one row.
  - **The effect:** the last block's compute fell 15.0 -> 4.8 us (64 pairs), ~0.07 us a pair. The unpacks were not
    the bound.
  - **Measured:** profile `1792341444`; 1.7B `Step(64)` 30.91 -> 29.55 ms.
- **Unpacker 1 takes a base anywhere on 16 bytes.**
  - **The setup:** a packed `A` (`[1, K]` laid out `[K / 32, 32]`) is unpacked from a base 64 bytes times the row
    into a packed tile.
  - **What it reads:** face `(0, k)`'s 16 rows of 64 bytes from there, the face 1,024 bytes on.
  - **Result:** bit for bit on every unit of card 0 (`step204`, `1792342641`). The reads past a slot's end land in the
    next slot and only reach output rows past the product's one.
- **Thin `A` reads are entry-bound, and packing pays.**
  - **The rate:** ~0.27 us an entry.
  - **Packed:** `A` is `K / 1024` whole tiles a unit (2 at `K` = 2048, 8 at 8192), plus one packing op of ~5-7 us,
    one read and two face-row writes a tile spread over the product's units.
  - **Measured** (profile `1792342758`, products before -> after, plus the pack): `qkv` 148 -> 135, `o` 64 -> 53,
    `gate|up` 349 -> 337, `down` 246 -> 190 us. The step went 29.55 -> 27.90 ms.
- **The barrier skew is not channel locality.**
  - **The experiment:** dealing `gate|up`'s rounds rotated, so each unit walked every channel pair rather than
    staying on one.
  - **Result:** its barrier went 40 -> 30 us, but its data phase went 269 -> 283 us. The step was 27.90 -> 27.84 ms
    (`1792343591`), so it was reverted.
- **A K chunk as its own batch costs ~3 us of role work.**
  - **What it is:** the FP32 reload, the pack, and the three roles' lockstep retire.
  - **Measured:** streaming every product's `K` in 4 chunks raised `qkv`'s control phase to 36 us and `gate|up`'s to
    89 (`1792358552`). It pays only where `K` would be K-blocked anyway: `down` 184 -> 173 us, the step 27.77 ->
    27.51 ms (`1792359572`).
- **Prefetching weights under small ops slows those ops.**
  - **The cost:** units 64-119 reading the next product's `B` while the small ops run made the latency-bound ones
    slower.
  - **Where it showed:** the residual adds +12 us, RMSNorm's one-unit statistic +11, `gamma`'s product +4, each a
    layer. One unit reading 64 tiles queues behind ~56 units' streams in the GDDR controllers.
  - **Measured, at 56 units:** the products gained ~3.1 ms and the small ops lost ~1.6. The step went 27.51 ->
    25.96 ms (`1792360408`).
  - **The cap, by units:**
    - 32: 25.60 ms (`1792361675`);
    - 20: 26.74 (`1792362531`);
    - 40: 25.61 / 26.03 / 26.40 / 27.23 at `Step(64)` to `Step(512)` (`1792363365`), the setting
      (`session::PREFETCH_UNITS`).

## The movers' per-entry costs in small moves (2026-10-18, card 1, profiles `1792293993`, `1792296384`)

Small moves are bound by the movers' entries, not by GDDR. What a decode step's moves measured:
- **A `FILL` of a staged tile's other 31 rows: ~10 us a tile.** B stores word by word. A one-row
  `resident row select` of 64 tiles in groups of 24 took 253 us on 3 units, and 67 us dealt one tile a unit over 64
  (`index::select_taps_jobs`). Spread a ragged fill across units, or avoid it.
- **A `SCATTER_ROWS` write entry: ~0.37 us to issue on NC.** 64 face-row writes on a unit took ~24 us; split over 16
  units (`record::scatter_rows_range`), the op took 8.8 us.
- **A repack of 32 one-row reads a unit:** ~38 us with each read into one slot before its copy. It was 31-33 us with
  the reads in flight together, and no faster with each read cut from 4,160 to 1,152 bytes (31.0 us on 74 KB,
  `1792298755`). Neither the reads' latency nor their bytes bounds it: it is ~96 entries a unit (32 reads, 64
  `COPY_WORDS`), ~0.3 us an entry.

## A barrier released by multicast from a mover (2026-10-19, cards 0 and 1)

The B mover now issues NoC multicast writes in normal operation (it was only the probe image before): unit 0's B, after
its barrier counter reaches a target, writes it to every other unit's `dm::BARRIER_RELEASE` with one response-marked,
path-reserved multicast per rectangle on static VC 4 (`noc::multicast::registers_to`), rectangles of session tiles only.
On card 0 (`step208`, 8 units) and on card 1 in every decode bench since (120 units, ~1,000 barriers a step, 835 ops), no
hang. What was learned:
- **Completion:** counted on `NIU_MST_WR_ACK_RECEIVED`, never on the transaction ID's outstanding counter (a broadcast
  adds one and takes one per recipient), which is never cleared: ttsim refuses that write, and nothing waits on the ID.
- **L1 keeps the release words across sessions:** they are zeroed with the counter when unit 0's mover starts, before
  any barrier is queued.
- **The release changes little:** 0.05-0.25 ms of a 24-25 ms step against every unit polling unit 0 over NoC0. The
  barrier's time is the slowest unit's arrival.

## A column broadcast costs the mover ~10 us a tile (2026-10-19, card 1, profiles `1792363365`, `1792372923`)

`READ_BROADCAST_COL` reads a `[rows, 1]` operand's tile into the scratch slot and then copies column 0 into every
column with 1,024 single-word stores (`broadcast_col0_from_scratch`). On silicon that is ~10 us a tile: an op with one
such operand took ~15 us of data phase where the same reads without it take ~4.5 (RMSNorm's `DIV` on 64 units 24.1 us,
the softmax's 2-unit ops 24.7). Broadcasting in the SFPU instead (`sfpu::Program::broadcast_col0`, seven lane-row
rotations with the first lane held, ~100 instructions a tile) took them to 14.7 and 15.0 us. Element-wise programs now
read the column as it lies (`record::READ_RUN_COL_AS_IT_LIES`); `READ_BROADCAST_COL` is left to the integer division's
programs.

## What an SFPU kind costs a tile, and a kernel a block (2026-10-19, card 0, `silicon_probe_attention_rows`)

- **An exp-based kind over whole tiles is ~8.5 us a tile.** `SUB_EXP` of a `[1, 512]` row on one unit, all 32 rows
  computed: 153.3 us for 16 tiles, against `MUL`'s 21.7 (runs `1792424868`, `1792424899`). Row-limited to one row,
  in the attention job's 16 score tiles, its kernel took 15.0 us against 13.5 for a `MUL` in its place (`1792425322`,
  `1792424459`). Build exp-, reciprocal- and divide-based kinds for the rows a tile holds
  (`sfpu::ops::code_for_rows`, `reduce::math_programs_rows`).
- **Row-limited `DIV` is ~0.5 us a tile over `MUL`,** and its time does not depend on the divisor: zero, a denormal,
  NaN, infinity and 1e-38 gave 29.4-29.6 us over 16 tiles.
- **Only the rows from 0 count as limited.** Programs built for rows `0..r0 + m` compute every row group up to the last
  live row: a head at row 31 of a stacked tile ran the whole-tile `SUB_EXP` and `DIV`, ~8 us a score tile. Move a
  few live rows to row 0 first (64-byte face-row reads, as `attention::attention_rows` does).
- **A region costs ~3.5-4 us** (a `PAIR_CALL`'s start and finish, its credits, the roles' retirement), whatever it
  computes. Consecutive blocks of a job join one region only with equal semaphore initialisations: a job mixing matmul,
  SFPU and reduction kernels was a region a kernel (seven in the attention op, ~25 of its 35 us at 64 positions) until
  they shared their union (`tensor::share_semaphore_init`; 25 us).
- **A capture holds every program it places,** so an op whose kernels are many and long cannot be captured beside a model's
  traces: the device sort of a 256-wide row nearly fills the 240 KiB cache by itself, 512 does not capture at all (card 0,
  ttsim; the 1.7B's traces hold 231 KB). Eager runs evict between lists and are not limited so. Make a kernel's program one
  body repeated (`code::Loop`), not one a tile: a row's argmax with a program a tile took ~11 KB a unit and broke the 1.7B's
  captures.
- **Many units reading the same few tiles serialise on a channel:** 64 units each reading one row's 2-tile packed twin made
  a norm 17.5 us against 14.4 on 32 (`1792431978`).

## A channel's rate by readers and entry size (2026-10-27, card 0)

From decode-to-bandwidth.md Phase 10 (F27), `silicon_bench_memory::gddr_readers_by_channel` (runs `1793131300` safe,
`1793131553` `fast-reads`; [log](performance-log.md#f27-probed-bank-local-reads-and-what-does-not-bind-a-decode-product-2026-10-27-card-0)).

- **A tile's read rate is set by its entry size, not by its channel or how few share it.** At one 2,112-byte BF16 slot an
  entry a tile reads ~6.5 GB/s in the safe image (~360 cycles an entry) and ~20 (cap 3) to ~35 (cap 8) with `fast-reads`;
  1, 2, 4 and 8 tiles a channel scale that linearly (8 a channel: 401-425 safe, 420-466 fast).
- **Long entries fill a channel from 2 readers:** 16 KiB or 128 KiB entries give 505-511 GB/s at 2, 4 or 8 tiles a
  channel; one tile a channel reads 63.7 GB/s at 128 KiB (99.6% of 64).
- **Reading only the channel beside a tile buys nothing at decode's entry size:** 8 such readers a channel equal 64 units
  reading an interleaved weight's columns (`m = 1` products' deal) to within 1 GB/s in both images.
- **The NoC #0 endpoint nearest above a tile** (shortest Y leg) is how the bench picks its port; on card 0 the
  nearest tiles to channels 0-3 sit at raw X 1-6 and to 4-7 at raw X 11-16 (`banked_tiles`).
- **The B mover's list loop is that tight:** one more check per entry there (a barrier test, nothing stamped) cost
  tile-sized plain reads ~4% (8 a channel 428.6 -> 411.6 GB/s at cap 8, against HEAD's `profile` image the same session,
  `1793140610`, `1793138241`); counting each request's bytes in a `.local` word costs ~1%. Keep profiling work out of
  the loop and in cold functions ([log](performance-log.md#the-cards-gddr-reads-over-a-decode-step-2026-10-27-card-0)).
- **Past ~32 requests in flight a channel the card reads less** (`fast-reads`, 2026-10-28): 8 readers a channel at
  16 KiB, 500-506 GB/s at 8-24 a channel, 467 at 64, 452 at 128; by count, not bytes. **Reading across the chip costs
  ~15 GB/s more:** 120 tiles each on its own half's channels hold ~505 while every tile reads, on all eight ~485-490. With
  equal shares the finish is unfair (tiles end between ~310 and ~1,000 us). What makes the scoreboard's 469
  ([log](performance-log.md#why-card-wide-reads-plateau-at-469-and-what-it-means-for-decode-2026-10-28-card-0)).
- **`NIU_MST_RD_DATA_WORD_RECEIVED` counts every read landing at the tile,** GDDR and other tiles' L1 alike, in 64-byte
  flits: over a 1.7B decode step 3,847 MB against the 3,568 of GDDR reads. Not a GDDR meter; ttsim refuses it (row 106).

## GDDR reads are one queue: prefetch traffic is paid by the small ops (2026-10-20, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8.
- **A unit's reads are issue-bound.** A whole-tile read lands ~1 us after the mover issues it, and the mover issues
  one in ~0.27 us; with ID 6 at a cap of 8, a 128-tile prefetch stalled 0 times (`step215`, `1792468449`). A read the
  mover issues before an entry has almost always landed by the next entry; only reads it has not issued yet are
  still to come.
- **Transaction IDs throttle and drain separately.** With B's own ID capped at one, a plain gather stalled 127 times
  and the same gather on ID 6 none (`step215`).
- **A small op's reads wait behind every unit's streaming reads.** With ~100 units reading a product's weights ahead
  (8 requests each), the residual add after a product went from 14 to ~100 us, nearly all of it its own reads' wait.
  That held whether the prefetch was issued at once or a read per wait-loop turn (profiles `1792470019`,
  `1792504522`). With weights streaming on 40 / 56 idle units only (~260 / ~300 GB/s), a step's own work cost 1.34x /
  1.61x (`1792465817`, `1792465464`). There is no idle GDDR bandwidth under the small ops to take for free: what a
  window streams, its ops wait for. Throttling the stream (ID 6's cap 1-2) traded the two at best ~0.2-0.4 ms a 1.7B
  step.
- **Correction (2026-10-21, card 0): GDDR is not the whole wait.** With planned residency's Phase 3, the residual add
  reads both operands from its own unit's L1 region (`READ_L1`, no GDDR read at all) and takes 8.0 us a call. With busy
  prefetch on (cap 2) it takes **70.5 us** still reading no GDDR, and RoPE and RMSNorm slow ~1.7x (profiles
  `1792596738` off, `1792597186` on). So a small op on a unit that is reading a product's weights ahead waits on that
  unit's own side, not at the controllers. **Cause (probed 2026-10-21):** the next entry.

## Units reading GDDR together are served unevenly; two in flight evens them (2026-10-22, cards 0 and 1)

[decode-to-bandwidth.md](../plans/decode-to-bandwidth.md) Phase 8, "The stream's rate"; profiles `1792629809`,
`1792632189` (`skew.tsv`: each unit's own time at each op).
- **What:** in a 1.7B decode step's `down` product, 64 units each read one 1.06 MB weight column (256 tiles, all on
  one channel, column `j` on channel `j % 8`), the same bytes a unit. Units 0-15 take 150 us, 16-31 154, 32-47 126,
  48-63 113, the same to 0.1 us in two runs; unit 20 is the slowest in all 24 layers. Grouped by channel (unit modulo
  8) the times are flat (132-139 us). `gate|up`, `o` and `qkv` show the same order.
- **Where:** units are numbered row-major from the grid's top (`tt_isa::noc::grid::Tensix::tiles`), so the slow units
  are the top rows. This is the unfair service of the [aggregate-read
  measurement](performance-log.md#what-holds-card-reads-at-430-gbs-unfair-service-not-distance), now tied to position.
- **What it costs:** a product dealt evenly waits for its slowest units: `down` ~38 us a layer over its median unit,
  `gate|up` ~60-80 us.
- **Card 1 is the same** to ~0.1 us (`1792632927`). **But it is not the rows:** with the tiles opened in reverse order,
  so that the product's 64 units are the bottom rows, the order reverses too (`down`'s units 0-15 106.5 us, 48-63 155.0,
  `1792633435`). A unit's service depends on where it sits among the units reading together.
- **What evens it: fewer reads in flight a unit.** At 2 in flight `down`'s units take 145-151 us by group, where at 8
  they took 113-154, and the product is ~20 us faster a layer; at 1 every unit is too slow. Latency-bound reads (few
  tiles a unit, few units) lose ~3% at 2, so only a weight stream's reads use it (`record::GATHER_STREAM`,
  `dm::STREAM_CAP`, default 2; [log](performance-log.md#the-weight-stream-metered-what-is-read-ahead-the-prefetchs-wait-and-each-units-own-time-2026-10-22-card-0)).

## A transaction ID's outstanding count is the tile's, not the core's (2026-10-21, card 0)

`noc::wait(id)` spins on the NIU's `reqs_outstanding(id)` counter, and the NIU is the tile's: B and NC issuing under the
same ID on the same NoC count into one counter, and either core's wait waits for both.
- **What it cost:** NC ended every `CALL` (a pair's writer section) with `noc::wait(PREFETCH_TXN)`, though it never
  issues a prefetch. Under busy-unit prefetch B keeps ID 6's reads in flight on NoC #0, and while it waits for NC to
  finish a pair (`finish_pair`) it issues another each turn the cap has room. NC's `DONE` came only when the count
  happened to reach zero: the residual add's wait for its writer was 66% of a 70.5 us call (1.7B decode, cap 2, profile
  `1792605585`), against 1.5 us with the prefetch off.
- **How it was found:** the profile mode now stamps B's trace fetches and its waits for NC (`fetch`, `writer`
  columns), and a profile image counts NC's own waits (`tt_isa::dataflow::NC_WAITS`, `TT_PROBE_NC_WAITS=1` on the
  stream-ceiling bench). NC's section fetches (~0.1 ms a step), credit waits, write issues and drains did not change
  with the prefetch, nor did its `CALL`s less their parts; B's pump turns were each at most 0.4 us. What was left was
  NC's list-end drain.
- **Fixed:** only B drains the prefetch ID at a list's end (`finish_prefetch`). The add 70.5 -> 12.8 us under cap 2,
  and busy prefetch pays for the first time: a 1.7B step 19.51 / 20.75 -> **18.40 / 19.92 ms** (`1792608298`,
  `1792608583`).
- **Still shared:** NC's `noc::wait(TXN)` (before a `Pop`, at a section's end) also drains NoC #0's `TXN` count, B's
  ordinary reads. Measured small (NC's drains ~30 us a 1.7B step); a wait that must be one core's alone needs an ID
  only that core issues on.

## NC writes another tile's L1 (2026-10-20, ttsim and card 0)

[planned-l1-residency.md](../plans/planned-l1-residency.md) Phase 2b: `op::WRITE_L1`, one unicast `noc::Command::Write`
from NC's L1 into a peer Tensix tile's L1 (gate `step218_l1_write_records`; card 0, safe image `1792524683`,
`fast-reads` `1792524825`, `fast-reads` + `encoded-guard` `1792524963`).
- **The peer is named by the coordinate the session uses for it, on either NoC.** With translation on, NoC #1 takes
  the same `(x, y)` as NoC #0 (`step50`), so the record carries one coordinate and the NIU NC's writes go out on
  issues it. Writes of 64 B to 16 KiB into two peers, on NoC #1 and NoC #0 and at three 16-byte skews, landed byte for
  byte; the guard below the window and a tile no entry named stayed poisoned. No hang.
- **Completion needs nothing new.** A `Command::Write` is response-marked, so NC's drain at a list's end (`noc::wait`
  on its ID, both NIUs once used) waits for it before NC reports done, and its unit then arrives at the barrier: the
  reader after the barrier sees the bytes.
- **What the card can and cannot refuse.** Its decode refuses a coordinate outside the Tensix rows and columns, a
  length of zero or past one request, a range outside `DATA`, and addresses not congruent mod 16, before any NoC
  access (each sent raw, refused with its code, nothing written). It cannot tell a harvested tile (translated X 15
  and 16 here) from a present one: that refusal is the host's alone (`DataMover::set_l1_writes`: the session's own
  tiles and its window), and no such entry is ever sent past it.
- **Image sizes:** B 30,404 -> 30,612 bytes of 43,008 (the decode arm); NC 15,540 -> 16,704 of 24,576 (7,872 left).
- **B reads a tile's L1 over the NoC, its own included** (`op::READ_L1`, a loopback read of the unit's region; every
  `step219` run and the 1.7B with residency on, `1792526703`): no hang, the bytes as written. **A peer's too**
  (`step218` `a_read_pulls_a_peers_window`, 2026-10-20; card 0 safe `1792531966`, `fast-reads` `1792532114`,
  `fast-reads` + `encoded-guard` `1792532259`): B on the gate tile read 64 B to 16 KiB from two peers' windows and its
  own, at three 16-byte skews, byte for byte, nothing past the reads written. The host refuses a peer not the
  session's, a harvested column's and an address outside the window (the remote side is the one checked against it);
  the tile refuses the decode's cases sent raw and on NC, and writes nothing.
- **Many units multicasting into one rectangle at once serialize.** 64 units each multicasting a row's 64-byte face rows
  into the same 104-tile rectangle took ~250 us an op where its GDDR writes took ~14 (`1792527601`): path-reserved
  multicasts that share a rectangle wait for one another. A row many units read whole is delivered some other way
  (`planned-l1-residency.md` 2f).
- **The multicasts' cost is the NoC's, not NC's waits** (2f (c), `dm::MCAST_RUN_ACKS`, card 0, 120 tiles,
  `silicon_probe_row_delivery` `1792533708`): with a producer's run of multicasts drained once and its
  acknowledgements counted once, a `[1, 2048]` row's 64 producers into 32 / 64 / 104 readers took 130 / 200 / 240 us
  against 140 / 210 / 240 counted an entry, and a `[1, 8192]` row 760 against 780. No hang, the bytes as written
  (`step218`, `step220`).
- **A row held by a few of its readers and pulled from them costs what GDDR does, and less under a stream** (2f (a),
  same run): 64 producers unicasting each face row to `h` holders, each reader one `READ_L1` of the whole row from its
  holder. A `[1, 2048]` row to 32-104 readers: 7.5-11 us with 2 or 4 holders against 9-17 through GDDR; under a GDDR
  stream on the idle units, 11-23 against 23-70. One holder serializes its readers (a `[1, 8192]` row, 64 readers:
  33 us), eight cost each producer more writes (28). The unicasts slow under the stream too (4 -> 11-16 us), but
  less than GDDR's reads do. Delivered by one gatherer's multicast, an op and a barrier more: 16-51 us.
- **NC multicasts through NoC #1 too** (`op::WRITE_L1_MCAST`, `step218` `1792525425`): the first NoC #1 multicast
  here, a 2x2 rectangle in translated coordinates with the corners swapped (`multicast::corners_hi`), landed in all four
  tiles and none beside them; no hang. Completion is NoC #1's `WR_ACK_RECEIVED`, read after NC's earlier writes have
  drained and counted to the rectangle's tiles before the next entry. NoC #1's counter is NC's alone: B, on the same
  NIUs, never writes through NoC #1, where NoC #0's counts B's barrier release too. With the multicast, B 30,996 and NC
  17,872 bytes (6,704 left).

## A region at the top of L1 costs a product only past its plan's floor (2026-10-21, card 0)

`silicon_probe_region_cliff` (`1792546573`; `planned-l1-residency.md` 2h-i). A one-row product's layout depends on
the arena it plans in (`l1::arena`) only through the choices sized from it, and those are exact:
- **`K` whole:** `matmul::plan_in` takes the widest block whose *probe* layout fits, and the probe charges `A`
  unpacked (`kt` slots), where the kernel runs a packed `A` (`kt/32 + 1`). The block stands while the arena ends at or
  above `DATA.base + 4160 * ((kt + 1)(nc + 1) - 1)`. At `K` = 2048 that leaves **110,464 bytes**, which is Phase 2a's
  96 KiB / 128 KiB cliff. At 110,464 the layout and the time are those of no region. 64 bytes past it, `nc` 2 -> 1 and
  `qkv` 148 -> 161 us, `gate|up` 342 -> 376, the head 991 -> 1069 (+8-10%). Qwen2.5-1.5B's `gate|up` (`K` = 1536)
  narrows at 106,304 and is **faster** narrowed (319 -> 293 us): the widest block is not always the best.
- **`K` blocked, `A` packed** (8B `qkv` at 4096, every `down`): the jobs are a column each, whatever `plan_in`'s
  width, so its floor binds nothing. The streamed chunk layout's does: 1.7B `down` 339,264 bytes (Phase 1's budget), Qwen
  `down` (8960) 285,184, 8B `down` (14,336) 222,784, 8B `qkv` 376,704. Past each, the chunks change and the time moves
  under 2% (+1.6%, -1.6%, +0.8%, +0.3%).
- **K = 4096 is never `K`-whole today:** the unpacked probe needs 257 slots of 220, so an 8B's `hidden`-wide products
  stream, and no weight prefetch runs ahead of them.
- **`l1_need` depended on a warm cache.** `plan_in`'s probes were noted only on a cache miss, so a capture after an eager
  run recorded the packed layout's need (684,352), not the probe's (938,112). `plan_in` now notes its floor on every
  call, without its probes (`matmul::plan_floor`).

## Trace sections: 64-byte alignment, not 32 (2026-10-15, ttsim and cards 0 and 1)

- **What the code checks:** a `PAIR_CALL` entry's decode checks only that each section's offset is a multiple of
  `ENTRY_BYTES` (32, `tt_isa::dm` `Entry::decode`).
- **What the mover does:** it reads a section in chunks of 64 entries through a `READ` descriptor (`call`,
  `crates/tt-firmware/src/mover.rs`) into L1 chunk addresses that are multiples of 64. A DRAM-to-L1 read needs source
  and destination congruent mod 64 (`tt_isa::dram::ALIGN`).
- **So:** a section at an offset of 32 mod 64 passes the entry's decode and fails its first chunk read (`ALIGNMENT`).
- **Since the change:** `Session::store_stream` packs a unit's stream and its sections into one region, so it starts
  each section on a 64-byte boundary. Before, every section had a slot-aligned region of its own: at least one
  `TILE_SLOT`, 4,160 bytes, which made a trace 5x to 7x its entries (SmolLM2's four traces 3.75 GB, now 563 MB;
  [decode-length-cost.md](../completed-plans/decode-length-cost.md)). Every trace gate passes in both layouts on card
  0, and `step147` on both cards.

## The Tensix L1 map, and changing it (2026-10-22, ttsim and card 0)

Since 2026-10-22 (`docs/plans/decode-to-bandwidth.md`, "Room for a second job's `B`", M2) a Tensix tile's L1 is, in
order: the five baby RISC-V images and B's mover list (`0x0..0x1A000`), NC's mover image (`..0x20000`), **`DATA`**
(`0x20000..0x100000`, 896 KiB, unchanged), **`CAPTURE_ONLY`** (`0x100000..0x116000`, where the mailboxes were: only a
trace's capture plans there), **`PROGRAMS`** (the four fixed program slots, `0x116000..0x136000`), **`MAILBOXES`**
(`0x136000..0x140000`, 40 KiB: the single-core and role mailboxes with their guard blocks and `Dst` dumps, both movers'
mailboxes, the probes' scratch, NC's list ring), `TRACE` (the timestamper's buffer) and the program cache
(`0x144000..0x180000`). `tt_isa::l1::REGIONS` and `l1::FIXED` (every block of the mailboxes) are checked at compile time.

- **Every L1 address is a constant of `tt-isa`, shared by host and firmware.** A change of the map rebuilds every image
  (`tt-firmware-images` embeds them) and needs no address edited elsewhere: the only literals were the probes' scratch,
  now `mailbox::PROBE_SCRATCH`.
- **After a map change, the first process on a card is a `Session` bring-up.** L1 survives between processes, and a core
  still running an image of the old map polls addresses that are now `DATA` or `PROGRAMS`: an old mover reading a
  garbage queue issues arbitrary NoC reads. A `Session` holds every baby core before it writes anything, and its
  driver cleanup holds them again when the process ends; processes outside a `Session` (the harness's single-core
  runs, probes) are not covered if they crash, so start with a `Session` gate. The first run after M2 was
  `step185`/`step187`/`step188`/`step191`/`step219`/`step221`, then `step221` again in a new process.
- **Growing `DATA` is not free even when nothing moves into it.** The first form of this change gave `DATA` the 88 KiB
  (to `0x116000`); every choice sized from the arena changed with it: a `K` = 1024 product took 6 columns, not 5, and
  left its residency region half the room; a `[512]^3` product's blocks and an element-wise op's run lengths changed
  (36-37 tiles instead of 32, whose layouts then met in one half); a 113-tile `K` no longer needed a continuation; and on
  one tile the longer column-sum runs' programs no longer fit the program cache beside a traced 135M's
  (`step193`, silicon only). Hence the capture-only arena: eager sizes stay as they were, and only a capture, whose
  choices are pinned (P3), sees the room.
- **What guards it:** `step221_l1_map` poisons the capture-only arena (the old mailboxes) and every word of `MAILBOXES`
  no `FIXED` block holds, runs eager ops, a norm, a product and a planned capture with replays, and requires every
  poisoned word intact; on ttsim it also opens a
  second session on the first's L1 and requires the same bits (ttsim starts every process from zeroed L1, so this is
  how a gate there sees state an earlier process left).
- **The free L1 below `OUTER_CHUNK` holds a layer body's argument table** (2026-10-23,
  `docs/plans/layer-sub-traces.md`): `dataflow::ARG_TABLE` at `0x12040`, 126 entries, between NC's 8-byte stub at
  `0x12000` and `OUTER_CHUNK` (`0x13000`). B reads a table there for each `CALL_BODY`, and NC reads it in place
  during the body's pairs. Both fence after the read lands (`publish`), so neither core's L0 data cache serves a
  previous layer's table. B's safe image is 35,284 of 43,008 bytes since (it was 33,668), NC's 19,020 of 24,576.

## A stalled replay: what to read (2026-10-23, card 0)

A list that stops making progress is reported as `descriptor N did not finish`, which names only the tile the host
waited on. With `TT_STALL_DUMP=1` the host prints, for each tile whose wait times out (`DataMover::dump_stall`):
- both movers' descriptor, queue and trace-progress words;
- the pair counters and NC's wait reasons;
- the roles' generation, acknowledgement, status and panic words;
- the barrier counter, poll, return and release words.

How to read it:
- Roles with generation equal to acknowledgement are idle.
- NC with `SEQ == DONE` is not in a pair.
- If every unit's last release is below the coordinator's counter, the units wait at a release-mode barrier. The
  coordinator never releases when an arrival is missing: look for the unit that stopped before arriving. That is how
  the layer sub-traces' one silicon bug was found (a unit with no repeated ops given an empty body, which the mover
  refuses, so it never reached its tail's arrival); bisecting by capture and then by unit, with a temporary knob,
  found the unit.

## A role's addresses from GPRs: settle the Scalar Unit before `WRCFG` (2026-10-23, card 0)

An element-wise kernel's and a reduction's unpack and pack now take each tile's L1 address from a Tensix GPR stepped
by `ADDDMAREG`, so a tile's block is the same words for every tile and the runner repeats it
(`sfpu::kernel::roles_looped`, `sfpu::reduce::roles_looped`; `docs/plans/long-context.md` 1.A). Written as
`ADDDMAREG` then, at once, the `WRCFG` that reads the GPR into `THCON_SEC0_REG3_Base_address`, silicon unpacked from
the GPR's old value where ttsim read the new one: every multi-tile element-wise and reduce gate failed on card 0
(`1792793167`; an `add` returned `A`, `B` read from elsewhere). A `STALLWAIT` blocking the Configuration Unit on C0
(this thread's Scalar Unit work outstanding) between them fixed all 39 (`1792793443`, `1792793456`):
`datapath::gpr_settled`, before every `WRCFG` of a stepped GPR. tt-metal's Blackhole LLK does the same after its own
GPR writes (`TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON)`). The Wormhole page `ScalarUnit.md` says a thread's
next instruction waits for a Scalar Unit instruction to complete; on Blackhole, for a `WRCFG` reading the result, that
is not enough. `SETDMAREG` then `WRCFG` (every `config_program`) has always worked without it. The matmul's
`gpr_step` (`ADDDMAREG`, `ADDDMAREG`, `WRCFG`, `WRCFG`) has none and passes; it is not known to be safe by
construction. ttsim divergence row 104.

## Handling silicon runs

Rules that held through the Phase 10 close-out; follow them for any new probe.

- **Risk order.** Documented behaviour first, then measured, then UNVERIFIED or Wormhole-only
  encodings (always an isolated minimal probe first, after a ttsim pass or a recorded refusal), then
  the NoC-hang class (one probe per session, nothing queued behind it). Keep a known-good gate between
  risky probes.
- **A hang is fixed in the API, never retried.** A probe that wedges a core or the NoC becomes a checked
  refusal or a `[-]` with the evidence, not a retry loop.
- **Run silicon one test per process** through `cargo xtask silicon`; agents that edit code do not run
  it concurrently with another run, because a gate that aborts with a parked thread wedges the tile
  until the next session reset.
- **Do not wait on a process with `until ! pgrep -f <pattern>`.** The loop's own command line matches
  the pattern, so it never ends and leaks shells; wait on the run's log or exit status instead.
- **A worktree needs `vendor/`.** It is git-ignored, so copy it from the main checkout before building
  there (`cargo xtask fetch-ttsim`/`fetch-spec` also work).

## Running tt-metal after this repo's gates (2026-10-13, card 1)

tt-metal (v077, UMD 0.77) will not open a card that this repo's Ethernet gates have touched. The first thing it does is return the
active Ethernet cores to base firmware, and the image `tt-firmware`'s `eth_e1` leaves on E1 persists across processes:
`Device 0: Timed out while waiting for active ethernet core 24-25 to become active again. Try resetting the board. Minimum
tt-firmware version is 18.10.0` (`return_to_base_firmware_and_wait_for_heartbeat`). With both cards visible the same state showed as a
hang after "Opening subset of mmio devices" (the first attempt, killed by its 500 s timeout).

- **Fix:** `tt-smi -r <PCI address>` for the card, with nothing else on it (`tt-smi -ls` lists the addresses; card 1 is `0000:02:00.0`
  here). After it, tt-metal opens the card. Nothing else in the repo needed re-establishing: the gates start from a known state.
- **The other direction, 2026-10-13:** after tt-metal runs on card 1 and many benchmark processes, the two-card Ethernet gate
  `step76_burn_bf16::bf16_mesh_product_uses_resident_ethernet_execution_across_two_cards` failed on both card-0 smoke runs and with
  `--device all` (`resident Ethernet transfer: transfer 1 was not acknowledged (last acknowledged: 0)`); `tt-smi -r 0000:01:00.0
  0000:02:00.0` and it passed on both cards, with the same build. A mesh gate that fails like that after tt-metal has touched the
  cards is link state: reset both boards before suspecting the code.
- **Fence the cards:** `TT_VISIBLE_DEVICES=<PCIe device number>` restricts UMD to that card, so tt-metal's cluster start (which
  touches every card it can see) cannot disturb a job on the other one.
- **First import is slow:** `import ttnn` took about three minutes cold (the libraries come off disk); a C++ program linked against
  `build_Release` does not pay it. The installed ttnn 0.72 needs SFPI 7.52.0 and the machine has 7.83.0 in `/opt/tenstorrent/sfpi` (its
  kernel JIT fails), so the working route is a tree with its own SFPI: `tt-metal-v077`'s `build_Release` (SFPI 7.69), which JIT-compiles
  kernels itself.
- **Fast dispatch reserves cores:** the compute grid is 11x10 = 110 on a card where this repo has 12x10 = 120.
