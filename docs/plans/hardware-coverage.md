# Hardware coverage — Phase 10 tracker

**Close-out (2026-10-10):** the remaining instruction groups, the `[~]` items and the performance items were worked through by lane; see [hardware-coverage-closeout.md](hardware-coverage-closeout.md) "Final state" for each lane's disposition, the open items (NoC multicast and MMIO silicon probes, X6) and the verification record. [Burn operation status](burn-op-coverage.md) is generated.

Current Tensix continuation status (2026-10-05): see
[tensix-next-features.md](tensix-next-features.md#2026-10-05-wrap-up-and-next-starting-point)
for completed packed BF16/K/batched/mesh paths, pooling traces, integer reductions,
extremum scans, hardware precision modes, accuracy results and remaining work.
Checked integer division/remainder and axis mean now pass `step82` in ttsim;
card-0 silicon validation passes (`1791232718`). Native attention, ragged F32
batches and distributed batched products pass `step83` simulator and silicon gates; the full tranche remains open.
See the continuation acceptance checklist in `tensix-next-features.md`.
Ordinary validation uses one card; both cards are reserved for actual mesh tests
or device-specific investigations per the user's instruction. Historical two-card
requirements below are superseded for ordinary validation.


> Native cutover: `burn-tt` no longer delegates to Flex. Unsupported methods fail
> explicitly, and `TT_EXACT` is retired. Historical Flex fallback descriptions
> below are superseded by [the current backend contract](../../crates/burn-tt/README.md)
> and [the cutover backlog](burn-native-cutover.md).

The working tick-list for Phase 10 of `RUST_IMPL_PLAN.md` ("Phase 10 — Hardware
coverage"). The plan says *why*; this file says *which parts of a Blackhole Tensix tile
this stack can drive, which it cannot yet, and in what order the rest arrives*. It is the
progress record: an item is ticked here or nowhere.

Execution order, branches and the per-item workflow: the "Execution" section at the end.

Companion guides: [`tt-metal-concepts-review.md`](../learnings/tt-metal-concepts-review.md) (the
Tenstorrent concepts this stack lacks, G1–G16, and the hardware sharp edges to handle in
code) and [`burn-backend-parity.md`](burn-backend-parity.md) (the `burn-tt` surface and
ergonomics roadmap, B0–B16). Item ids here are cited from both.

Same legend as `implementation-checklist.md`: `[x]` done and gated by a test · `[~]`
partially done, see note · `[ ]` not started · `[-]` deliberately not done, with the
reason given.

---

## Where things stand (2026-10-05, 10.3–10.5 in progress)

Milestones 10.0–10.2 are complete. Since their close-out, rank-N storage and
strided views, tile-aligned batched matmul, last-dimension reductions and some
leading-dimension sums, embedding and loss indexing, asynchronous Burn dispatch,
host DMA, and fresh/traced streaming ownership have landed. B only reads GDDR;
NC only writes it; on-card tensor arithmetic runs on Tensix.

The first 10.3 addition is resident full F32 `sum` and `mean` (R1b): reduce
columns, then rows, dividing a mean by the logical element count. Full-buffer
reshape/swap views are reduced at their source; whole-tile-row slices reduce
only their view. Wide matrices use at most 16 tile columns per chunk, with
device block copies and scalar partial sums; tall matrices reuse the existing
chunked row sum. Both ops are native only: host F32 inputs are uploaded, and
unsupported dtypes, empty/rank-zero inputs, engines without native reduction
support and resident views that cannot be copied on the card fail explicitly.
Mesh engines compute the same reductions on chip 0's SFPU, staging through
L1; both passes must fit L1/program slots, and their scalar result is on the
host. The resident path chunks larger shapes. Exact mode also uses
the native reduction and division order, within derived bounds rather than
guaranteeing Flex's bits. No Flex arithmetic is called by either op.
Gates: `step63_burn_full_reduce` and `step59_burn_transformer`.
The scalar gates cover `[37, 8193]` wide chunks, `[8193, 1]` long row sums,
ragged views, special values, native execution in exact mode, host-input uploads,
autodiff and two-tile trace replay with changed inputs. The transformer gate
replaces a 512-byte mean-input download with the explicit 4-byte loss readback
per step. Exact-mode MNIST still computes its per-example losses on the host,
then uploads 256 bytes for the native mean and reads back 4 bytes. Its updated
32-step golden differs by at most four F32 ulps from the earlier golden, all
within the derived reduction/division bound.

Native-only validation (2026-10-04): eight scalar gates pass on ttsim,
including staged mesh sums/means and explicit capacity rejection. All five
MNIST end-to-end gates pass against the updated golden, including two/four-chip
meshes. Device 1 run `1791079196` passes 10/10: seven resident scalar gates,
the transformer loss gate, and single/four-tile MNIST goldens. This run uses
device 1 exclusively; the staged mesh revision has simulator validation only.
The complete workspace host/simulator tier passes, as do lint checks for the
changed backend, kernels and reduction/model tests. Generated delegation and
the no-simulator-in-shipping-crates checks pass.

Full reductions do not complete 10.3. R1c arbitrary axes/layouts and P2 K
blocking pass simulator and both-card gates (below). Product/Boolean reductions,
scans and norm compositions are also gated. The payload-preserving transpose is the mover's `READ_TRANSPOSED`/repack (M3 `[-]`: no Tensix route preserves payloads).

**Remaining after the 2026-10-10 close-out:** convolution, attention, ND indexing and the integer
operations are native; the open items are listed in
[hardware-coverage-closeout.md](hardware-coverage-closeout.md) ("Still open"). Backend
error/setup/conformance work is tracked separately as B1/B2/B10. X280 dispatch remains proposed and
unscheduled.

### Reduction and ALU extensions (2026-10-04; both-card validated)

`step69_reduction_primitives` adds direct F32 product, typed Boolean AND/OR
reductions, rank-N argmax/argmin repacking, and inclusive F32 sum/product scans.
Product masks padding to one and carries an unfolded accumulator through long
axes before a single fold; full products reduce logical axes in descending order.
Scans traverse logical indices in order and reload the preceding tile's last
prefix only after NC release. Native flip/stepped slicing supports scan backward.
`step70_native_norms` validates Burn's existing LayerNorm and RMSNorm compositions,
including wide/ragged inputs, permuted rank-four views, affine parameters and
analytic input/parameter gradients. No fused norm kernel is added.

`step71_integer_alu` covers wrapping I32 add/sub/mul, tensor/scalar bitwise ops,
signed comparisons and modulo-32 shifts (right shift arithmetic). Multiplication
reconstructs full low-32-bit products from 16-bit limbs and both SFPMUL24 halves;
its 23-bit result alone is insufficient. `step72_round_cast` covers ties-even
round/floor/ceil/trunc and saturating F32-to-I32 (NaN to zero). These use raw-bit
SFPU programs, not SFPSTOCHRND's bounded sign-magnitude modes. Existing I32-to-F32
conversion is preserved. `step73_fpu_transpose` gates the typed, non-flipping
TRNSPSRCB helper on TF32 Src data against a permutation oracle and its inverse.
The signed BF16 zero-transpose control, transpose and inverse also pass both
cards (`1791233413`, 4/4); ttsim fails the BF16 copy control, so it is silicon-only.
Its encoding retains Wormhole provenance. Step87 now gates explicit 16×16 Src
transpose and TF32 matmul preparation; raw tensor materialization stays on copies.

All new gates belong to SMOKE. Wrong product padding, omitted scan carry,
incorrect multiplication high bits and wrong rounding mode were each watched
failing. `step67`–`step73` pass both cards: 58/58, run `1791145571`.
BF16 payload-preserving transpose uses the mover's raw copy; the Src transpose is the explicit normalizing route (step87). See [the implementation record](tensix-next-features.md).

### BF16 and pooling (2026-10-05)

`step74`–`step78` pass both cards (24/24, final run `1791160155`). Raw BF16 storage,
views and repacking use two-byte datums and 2112-byte slots, preserving payloads.
Native conversion rounds ties-even, quiets NaNs and flushes BF16 subnormals to
signed zero. Narrowing is silicon-only because ttsim refuses late pack mode
`0x105`; widening uses SrcA/MOVA2D. Packed rank-two matmul accumulates in F32,
with local ragged masking that preserves the parent's bits. Native Burn adapters
widen other operations to F32 and narrow once per operation boundary. Gates cover
normalization derivatives, mixed F32-loss/BF16-parameter SGD and trace replay.

Measured GMPOOL/GAPOOL encodings keep AddrMod at bit 15 and require bit 19.
Block max/sum/mean kernels retain banks through their phases and release once.
General BF16 mean/adaptive pooling stages 16-lane GAPOOL chunks with F32
continuation and explicit divisors. F32 mean uses SFPU; general max uses SFPU
argmax plus raw-bit OR selection to preserve zeros/NaN payloads and agree with
the backward index. Indices and overlap backwards stay resident.
All-padding windows fail explicitly. Geometry
constants now use replayable metadata descriptors, enabling pooling trace capture. A matmul-then-window-pool gate
was observed failing with stale unpacker ADCs; both unpackers are now reset explicitly.

`tt-mnist --bf16` opts into single-card BF16 storage, with F32 accumulation/loss
and F32 trace boundaries. No speedup is claimed: run `1791158550` measures the
two forward GEMMs at approximately 662/81 us (BF16) versus 94/18 us (TF32).
Packed continuation/batched products, actual two-card BF16 mesh execution and
full model accuracy are now gated; see the current implementation record. BF16 tensor transpose uses native bit copies, not TRNSPSRCB.

### R1c / P2 implementation (2026-10-04; both-card validated)

General F32 `sum_dim`, `mean_dim` and `max_dim` now accept every logical
axis, including ragged rank-N reshape/swap/permute views. Existing matrix
axes and supported leading sums retain their arithmetic order. Other axes
are repacked into rows on the card, reduced over columns, and repacked back
to the logical output shape. Only coordinates are constructed on the host;
B reads aligned source tiles and copies words in L1, NC writes assembled
tiles. The input/parent's padding claims are unchanged; repacked outputs
have undefined padding, which downstream kernels mask or repair.
Long column sums and maxima over either matrix axis carry the full unfolded
accumulator between bounded chunks and fold once at the end. Full sum/mean
retain their earlier scalar chunk order and golden.

Resident ordinary and supported tile-aligned batched matmuls now split K
when necessary. Each output tile stays on one unit, packing its FP32 partial
into GDDR and reloading it before continuing the same `MVMUL` sequence.
First and continuation programs have distinct cache keys. A separate prior
buffer and declared reset/load semaphores protect Dst; dependent gathers
wait for NC release, not merely pack retirement. Split-K runs are serialized;
existing unsplit pipelining remains. `Session::set_matmul_k_block_limit`
selects a reproducible maximum block length; the default plans automatically.

Gates: `step67_general_reduce` and `step68_k_block_matmul`, included in
`SMOKE`. Simulator coverage includes arbitrary axes, raw-bit ragged copies,
long reductions against the interpreter, autodiff, two-unit changed-input
trace replay and deferred operand frees; forced K blocks versus unsplit at
all fidelities, TF32/BF16 Src routes, transposes and specials, `[64,8192] @
[8192,64]`, supported batches and resident Burn large-K against a derived
bound. Negative controls fail when axis mapping, edge masking or accumulator
reload is omitted. Both cards passed in run `1791145571`.
Release-silicon K-block medians with validation: `docs/learnings/firmware-performance.md`, "P2 K-blocking and R3 norm release baselines" (2026-10-10).

### 10.2 close-out (historical measurements)

10.2 (branch `phase10-2-activations`) has its instructions (10.2a): every SFPU
instruction the rest of S2-S4 needs has a typed helper, an interpreter model and a
device gate on ttsim and both cards, and the `SFPLUTFP32` hazard is closed. D3 (int
and bool storage) moved into 10.2, so that comparisons and masks stay on the card, and
is done (10.2b): `I32` and `Bool` tensors are resident through Burn, with views and
the logic ops on the card.
`tt-mnist --activation` now trains with any of seven of Burn's activations. S2 is done
(10.2c): compare, select, sign and three activations on the card, exact; leaky-relu and
hard-sigmoid train at 2.9 and 2.2 ms/step (from 6.8 and 5.4) with ReLU's traffic (row
AJ). S4's algebraic ops are on the card (10.2d: `sqrt`, `log1p`, `pow` by tensor, integer
tensor and scalar), and 10.2d's sweeps found and fixed two of 10.1's range-end bugs:
`recip`/`div` above `2^111` and `exp` at exactly its overflow threshold. The
exponential family is on the card (10.2e: `expm1`, `tanh`, `erf`, `sigmoid`, `gelu`
and both backwards, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `log_sigmoid` and its
backward, `softmin`), so all seven of `tt-mnist`'s activations now move only what
ReLU's step moves -- gelu trains at 2.4 / 1.5 ms/step, from 5.6 / 4.9 (row AK). The
runner repeats blocks (X8), so long programs (`pow`, `gelu`) are one op.
Trigonometry is on the card (10.2f): `sin`, `cos` and `tan` for every finite input, by
an exact Payne-Hanek reduction, and `atan`, `atan2`, `asin`, `acos`, each within a
derived bound of a few ulps and through Burn's autodiff -- so S4 is done, and with it
10.2 (closed: ttsim 635/635, silicon 494/494 on both cards, smoke 54/54, MNIST 91.96% on
both cards at 2.0 / 1.7 ms a step, 1 / 4 tiles). Next: 10.3, reductions over any dim,
device transpose, norms.

### Current compute paths

10.1 added: reciprocal, division, `exp` and `log` on the SFPU
(S3, S4a), lane movement (S8), `sum` and `max` over either dim (R1a), softmax and
log-softmax on the device (R2); the matmul's loops replayed and the MOP Expander gated
(X1, X2); the movers' queues, barriers, batching and traces (X4); and wedged tiles
detected and recovered (X5).

Phases 0–9 built the path to the card. Compute currently uses these units:

| Unit | What runs there today | Where |
|---|---|---|
| **Matrix Unit** | `MVMUL` for matmul (TF32/BF16 `Src`, `Lo`..`HiFi4`), `ZEROACC`, GMPOOL/GAPOOL block pooling and BF16 window averages | `tt_kernels::{matmul,fpu}`, `role_t0..2` |
| **B / NC cores** | no arithmetic: B reads GDDR, transposes/broadcasts inputs and dispatches; NC writes GDDR and padding. Transfers and compute share ownership packets. Firmware image gates refuse F-extension instructions | `dm_b.rs`, `dm_nc.rs` |
| **SFPU** | every element-wise op: `ADD`, `SUB`, `MUL`, `MUL_SCALAR`, `ADD_SCALAR`, `RELU`, `RELU_BACKWARD`, `ADD_ROW` (`tt_kernels::kind`) and `kind_sfpu`'s; the sum over rows in Flex's order (`sfpu::reduce::accumulate_in_order`) | `tt_kernels::sfpu::{ops, kernel, reduce}` |
| **Unpackers / packer** | flat FP32 runs and the matmul's tile path; `UnpackToDst` for 128 datums | `tt_kernels::datapath`, `matmul` |

The instruction table includes units not yet used by tensor kernels (`ELW*`,
`TRNSPSRCB`). The SFPU builder covers the activation and
transcendental families, conditional execution, LUTs, lane movement, comparisons
and Boolean logic. Remaining integer arithmetic, casts and PRNG are tracked below.

Burn's device paths are listed in `OVERRIDDEN` (`xtask/src/gen_burn.rs`) and the
coverage tables below. Unsupported operations fail explicitly. Flex is an external test oracle;
there is no backend fallback or exact mode.

**Milestones** (detail in "Work items"):

| # | Milestone | Items | State |
|--:|---|---|---|
| 10.0 | Device profiler; SFPU foundation; today's element-wise ops move from the B core to the SFPU | X3, F0–F5, X1, S1 | `[x]` (F6, optional, deferred; F2's `SFPCONFIG` prologue and F5's further models arrive with S4) |
| 10.1 | Softmax and cross-entropy on the device; `MOP`; op-list traces | S3, S4 (`exp`, `log`), S8, R1 (`max`, `sum`), R2, X2, X4, X5 | `[x]` S3, S4a, S8, R1a, R2 (softmax, log-softmax), X2, X4, X5; cross-entropy moved to 10.5 with D4 (Burn gathers the target column, `float_gather`) |
| 10.2 | Activation and math breadth; int and bool storage | rest of S2–S4, D3 (from 10.4), F2's `SFPCONFIG` | `[x]` 10.2a (the instructions: helpers, models, oracles, gates), 10.2b (D3: `I32` and `Bool` resident), 10.2c (S2: compare, select, sign), 10.2d (S4: `sqrt`, `log1p`, `pow`; S3 and `exp` fixed at their range ends), 10.2e (the exponential family: `expm1`, `sigmoid`, `tanh`, `erf`, `gelu`, the hyperbolics and their inverses, `log_sigmoid`, `softmin`), 10.2f (trig: `sin`, `cos`, `tan` for every finite input, `atan`, `atan2`, `asin`, `acos`) |
| 10.3 | Reductions over any dim, device transpose, norms | P1, M2, M3, R1, R3 | `[x]` general F32 reductions/scans, norm compositions and pooling pass both cards; the payload-preserving transpose is the mover's (M3 `[-]`, evidence in the M3 item); K-block and norm release baselines recorded 2026-10-10 |
| 10.4 | Formats and integers | D1, S5, S6 (D3 moved to 10.2) | `[x]` I32 ALU, rounding, checked division/remainder, native BF16 and FP16 (exact SFPU casts), BFP formats, integer scans/arg-extremes/masks |
| 10.5 | Indexing, convolution, pooling, attention | D4, D5, P2, D6, R4 | `[x]` native pooling, resident indexing including ND gather/scatter, F32/BF16 convolution/attention and gradients, mesh trace replay; D5 `[-]` (host tilize stays the default) |
| 10.6 | The rest: block float, PRNG, `SFPLOADMACRO`, `ELW*`, `SHIFTXB` | D2, S7, S9, M1, M4 | `[x]` M1, D2, S7 (native seeded random), S9 (`SFPLOADMACRO` probes pass), M4 (`SHIFTXB`; `DOTPV`/`SHIFTXA` excluded) |

Checklist items 9.9 (element-wise on the SFPU) and 9.12 (loss on the device) are tracked
here, as S1 and R2.

---

## Definition of done

The two-gate rule, applied per op. An item is `[x]` only when every line below holds for
it, and a line that does not apply says why in the item.

1. **A `tt-isa` helper**, typed so that the hazards are in the API (cross-cutting rule),
   with unit tests of its encoding.
2. **An oracle that is not an epsilon** (the Tolerance policy):
   - *bit-exact* wherever the hardware's arithmetic is documented: `numerics::fma_bh` for
     anything `SFPMAD`-shaped, and a port of the page's functional model for `SFPLUT`,
     `SFPLUTFP32`, `SFPARECIP`, `SFPEXEXP`/`SFPSETEXP`/`SFPEXMAN`, `SFPSTOCHRND`,
     `SFPCAST` -- each port checked against the page's pseudocode the way `fma_oracle`
     checks `fma_bh` against `fma.c`;
   - for an *approximation* (a polynomial, a Newton step, a range reduction), the device
     result is still predicted bit for bit by running the same instruction sequence through
     the ported models, **and** its distance from the true function is held to a bound
     *derived* in a comment next to the gate (polynomial remainder, Newton's quadratic
     convergence, the reduction's error), never a guessed number.
3. **A ttsim gate** in `crates/tt-tests/tests/stepNN_*.rs`, inside `fork_scope`,
   watched failing at least once (an empty kernel, a wrong constant, a swapped operand).
4. **A silicon gate**, the same test through `cargo xtask silicon`. Ordinary
   validation uses one card under the current policy; mesh or device-specific
   investigations use both.
5. **Burn routing**: the method overridden in `burn-tt/src/ops.rs` and listed in
   `OVERRIDDEN` (then `cargo xtask gen-burn-ops` and its `--check` mode), agreeing with `burn-flex` bit for
   bit or within the item's derived bound; a residency check that the op downloads
   nothing (`tensor_traffic`, `TT_TRACE_FALLBACK=1` silent); and an entry in the silicon
   smoke tier (`xtask/src/silicon.rs`, `SMOKE`), where burn-tt is compared with
   burn-flex on the card.
6. **Findings logged**: every ttsim refusal or ttsim/silicon disagreement met on the way
   is a row in `ttsim-divergence.md`, and every measured fact a "Measured, not quoted"
   entry.
7. **Padding declared** (F0, from `tt-metal-concepts-review.md` G1): the op states the pad
   it needs from each input and the pad it leaves (`OpPadding`), and a ragged-shape gate
   chains it into an accumulation.

An op that works only on silicon (ttsim refuses an instruction it needs) can be ticked
with the simulator line `[-]` and the divergence row cited, as `DOTPV` is today.

---

## Hardware inventory

One row per feature of the Tensix tile and its surroundings. Columns: **Enc** -- in the
generated table · **Helper** -- a typed `tt-isa` layer · **Kernel** -- used by a
`tt-kernels` kernel · **Sim** / **Si** -- gated on ttsim / both cards · **Item** -- the
work item below that takes it the rest of the way. **Spec** gives the page and its trust
kind (BH = real Blackhole page, WH = Wormhole only, so every fact is unverified until
measured).

### SFPU (Vector Unit) -- 48 Blackhole pages

Reference: BH `VectorUnit.md` (32 lanes × 32 bits, five sub-units, lane predication, PRNG),
`LReg.md` → WH `LReg.md` (17 `LReg`s: 0–7 general, 8–10 and 15 constants, 11–14 written
through `SFPCONFIG`, 16 for `SFPLOADMACRO` only), BH `Dst.md`.

| Group | Instructions | Enc | Helper | Kernel | Sim | Si | Item |
|---|---|:-:|:-:|:-:|:-:|:-:|---|
| Load / store | `SFPLOAD`, `SFPSTORE`, `SFPLOADI` | x | x (`Program`) | x | x | x | -- |
| Multiply-add | `SFPMAD`, `SFPMUL`, `SFPADD` | x | x (`Program`) | x | x | x | -- |
| Immediate arithmetic | `SFPADDI`, `SFPMULI`, `SFPDIVP2` | x | x (`SFPDIVP2` 0..128) | | x (`SFPDIVP2` from 128: row 66) | x | S2, S4 |
| Move / abs | `SFPMOV`, `SFPABS` | x | x | `~` `SFPMOV` | x | x | S2 |
| Sign, exponent, mantissa | `SFPSETSGN`, `SFPEXEXP`, `SFPEXMAN`, `SFPSETEXP`, `SFPSETMAN` | x | x | `~` (`exp`, `log`, `recip`) | x | x | -- |
| Compare (BH-only `GT`/`LE`) | `SFPGT`, `SFPLE`, `SFPSETCC`, `SFPLZ` | x | x (flags; `SET_VD` masks raw) | x (`RELU`; S2's IEEE comparisons, clamps, selects) | x | x | -- |
| Conditional execution | `SFPENCC`, `SFPPUSHC`, `SFPPOPC`, `SFPCOMPC` | x | x (scopes) | x | x | x | -- |
| Bitwise | `SFPAND`, `SFPOR`, `SFPXOR`, `SFPNOT` | x | x | `~` (masks) | x | x | S5 |
| Integer arithmetic | `SFPIADD`, `SFPMUL24` (BH-only), `SFPSHFT`, `SFPSHFT2` | x | x (`SFPMUL24` with `VC` zero only; `SFPSHFT2` rotate) | `~` (`exp`, `log`, reductions) | x | x | S5 |
| Lookup and reciprocal | `SFPLUT`, `SFPLUTFP32`, `SFPARECIP` (BH-only) | x | x (`SFPLUTFP32`'s indirect destination designed out) | `~` `SFPARECIP` | `~` (`SFPLUTFP32` only `Mod1` 2, 6: row 70) | x | S4 |
| Casts | `SFPCAST` (`_IntFloat`, `_IntInt`, `_IntAbs`) | x | `~` `_IntFloat` round-to-nearest | | `~` | `~` | S6 |
| Rounding | `SFPSTOCHRND` (`_FloatFloat`, `_FloatInt`, `_IntInt`) | x | x (checked modes) | x (explicit precision reduction) | x (step81; supported forms) | x | S6 |
| Lane movement | `SFPSWAP`, `SFPTRANSP` | x | x (`SFPSWAP` min/max) | `~` (reductions) | x | x | S2 |
| Configuration | `SFPCONFIG` | x | `~` `LReg[11..15]` only (`Program::constant`) | | x | x | F2 |
| Macro | `SFPLOADMACRO` | x | | | `-` row 7 | `~` load half | S9 |
| Misc | `SFPNOP` | x | x | x | x | x | -- |
| PRNG | `SFPMOV`/`SFPCAST`/`SFPSTOCHRND` PRNG modes (`VectorUnit.md`, "PRNG") | x | x (diagnostic seed) | x (reads) | x | x both (step91; reseed differs) | S7 partial |

### Matrix Unit (FPU)

Reference: WH `MatrixUnit.md` (STUB-B), WH `MVMUL.md`, WH `SrcASrcB.md`, WH `RWCs.md`, BH `Dst.md`.

| Feature | Enc | Helper | Kernel | Sim | Si | Item |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `MVMUL`, fidelity phases `Lo`..`HiFi4` | x (measured) | x | x | x | x | done (Phases 6–7) |
| `ZEROACC` | x (measured) | | x | x | x | -- |
| `ZEROSRC`, `CLEARDVALID` | x (WH retained) | x (`Banks`) | x (bounded source sequences) | x (matrix clear/release; unpacker clear refuses) | x (step103 card 0) | Stage C |
| `MOVA2D`, `MOVB2D` | x (measured) | x (`Banks`) | x (FPU transpose and source readback) | x | x | F1 |
| `MOVD2A`, `MOVD2B`, `MOVB2A` | x (step9 measured address modifiers, including entry 4) | x (`Banks`, one/four-row, Loaded destination) | x (explicit Session chains; instruction/traffic audit) | x (step97 four-row conversion and chains) | x (step97 both cards, one/four rows, masks and exceptional data) | F1 done |
| `ELWADD`, `ELWSUB`, `ELWMUL` (with `Src` broadcast) | x (measured BH) | x (`Banks`) | x | x (F32 output) | x both | M1 |
| `GMPOOL`, `GAPOOL` | x (measured BH) | x (`Banks`) | x | x | x both | M2 |
| `TRNSPSRCB` | x (WH) | x (`Banks::transpose_b`) | x | x (TF32 Src) | x both (step87) | M3 partial |
| `SHIFTXB` | x (measured) | x (`Banks::shiftxb`) | x (bounded source sequences) | `-` row 50; host model x | x (step103 card 0) | M4 done |
| `MOVDBGA2D` | x | | | `-` row 50 | `~` encoding | diagnostics |
| `DOTPV`, `SHIFTXA` | x | `-` excluded | | DOTPV: `-` row 50; SHIFTXA: not established | DOTPV: `~` encoding; SHIFTXA: not established | M4 exclusions, 2026-10-07 |

### Unpackers and packer

Reference: WH `UNPACR_Regular.md` (conditionalized, authoritative), WH `Unpackers/*`, BH
`PACR.md` ("basic"), WH `Packers/*` -- the thinnest part of the Blackhole tree.

| Feature | State | Item |
|---|---|---|
| Flat FP32 run, `Src` tile path (TF32/BF16), `UnpackToDst` 128 datums, a datum sub-run of a tile (base moved) | `[x]` | -- |
| `UnpackToDst` of a whole 32×32 tile, and the whole tile packed back | `[x]` FP32 (`step25_dst_tile`) | F1 |
| BF16 into `Dst` (`UnpackToDst` on silicon; ttsim refuses, row 31) | `[x]` widened readback over ~4096 sign/exponent/mantissa patterns on card 0 (step114) | D1 |
| Packer output format conversion (FP32 `Dst` → BF16/FP16 L1) | `[x]` BF16 native ties-even late narrowing (step74); FP16: the raw packer truncates and saturates (matches ttsim), the rounding packer is ties-even for normals and overflows to infinity at 65520 but drops NaN payloads and flushes subnormals (`T7-MEASURE`, card 0), so shipped FP16 casts use exact SFPU programs instead (step143/144); the 16-bit Dst read path preserves normals only (zero/subnormal/NaN collapse, step114) | D1 |
| BFP8/BFP4/BFP2 storage, exponent sharing, `CLREXPHIST` | `[x]` -- step92–96; histogram reset and BFP2 packed matmul are silicon-only where ttsim refuses | D2 (delivered formats) |
| Integer formats (INT32 code 8 measured; INT8/UINT8 not) | `[~]` 32-bit integers and bools stored as raw bits through the FP32-coded path (D3); INT8/UINT8 with D2 | D3, D2 |
| Unpacker transpose / tilize modes, broadcast | tilize `[x]` payload-preserving strided gather (step113, card 0); transpose swaps the nibbles correctly but goes through SrcA and is not payload-preserving (`[-]` for M3); no unpacker broadcast mode exists | M3, D5 |
| Packer ReLU and edge masking, `PACR_SETREG` | ReLU `[x]` (all seven modes against a raw-bit model, card 0) and edge masking `[x]` for the row-set path, partial columns, -inf fill and edge-then-ReLU order (step112); `PACR_SETREG` `[-]`; not routed into Session/Burn ops | S1 (opportunistic), D4 |

### Frontend and tracing

Reference: WH `REPLAY.md`, BH `MOPExpander.md`, WH `MOP.md`/`MOP_CFG.md`, BH
`BabyRISCV/AutoTTSync.md` (the expanders and the Wait Gate), `DebugTimestamper.md`.

| Feature | Enc | Helper | Kernel | Sim | Si | Item |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `REPLAY` (record and replay, 32 entries per thread) | x | x | x (SFPU ops) | x | x | X1 |
| `MOP` / `MOP_CFG` (MOP Expander templates) | x (`CONFIRMED`) | x (`frontend::mop`, mailbox `MOP_CFG`) | (X2b) | x | x | X2 |
| Debug timestamper event stream | -- | x (`tt_device::trace`, `tt_kernels::profile`) | x mover and role events | `-` row 54 | x | X3 |
| Op-list traces (a step's records kept in GDDR, replayed) | -- | x (`tt_kernels::trace`) | x | x | x both | X4 |
| `.ttinsn` fusion (four pushes per cycle) | -- | `[-]` | | | | `.ttinsn` is a firmware-image immediate and the runner pushes L1 data, so fusion needs a run-time code generator outside the build-time instruction gate; SFPU math programs are push-bound (39 of 73 kinds, median push/backend cycle ratio 1.17, max 2.87) and shrink through replay and `SFPLOADMACRO` schedules instead (P9, step121) |
| Hazards as data, the wait planner | -- | `[~]` `tt_isa::hazard` table and checker (instruction block bits compared with `STALLWAIT.md`) | x over 954 builder role programs | | | checker `[x]`: 0 missing waits and 1,724 redundant waits (0.15% of backend words); a planner has nothing to insert `[-]` (P9, step121) |
| Three-thread pipelining, double buffering | -- | | x (resident T0/T1/T2 roles, `LAUNCH`/`KERNEL_WAIT`, `enable_dram`) | x | x both | checklist 9.8 |

### Scalar unit, mover, atomics, NoC

Pulled in only when a kernel needs them; each says which.

| Feature | Spec | State | Wanted by |
|---|---|---|---|
| ThCon `SETDMAREG`, `ADDDMAREG`.., `LOADIND`/`STOREIND`, `FLUSHDMA` | WH `ScalarUnit.md` + pages | `SETDMAREG` stages config; `ADDDMAREG` steps matmul addresses (step38); scalar ALU/config readback done (step100); L1 transfers step101/102 `[x]`; `FLUSHDMA` `[-]`, use STALLWAIT | F3 (per-tile parameters without reprogramming) |
| Tensix atomics `ATCAS`, `ATINCGET`, `ATINCGETPTR`, `ATSWAP` | WH | `[ ]` | 9.8 page FIFO, if counters move into Tensix |
| `XMOV` (Tensix mover, L1 → L1) | WH `XMOV.md` | `[x]` step101/102, explicit copy/zero | D4 (copies without the B core) |
| NoC multicast (NIU broadcast; TLB `strided`, row 4) | BH `NoC/MemoryMap.md` | `[x]` typed rectangle from the ARC-discovered grid, write-only encoder with `PATH_RESERVE`, acknowledgement counting against the known recipients, independent model (`tt_isa::noc::multicast`, step115); ttsim executes it fully (8 gates); on cards 0 and 1 the 1x2, 2x2/3x3 and full-grid probes (run one at a time, tile health checked between) deliver the payload to exactly the rectangle with guards unchanged and the one-column-wider mutant caught (2026-10-10); `NOC_BRCST_EXCLUDE` is written 0 and the VC class (buddy 0) is the choice that worked, not a documented layout | weight broadcast to many tiles (9.6 follow-up) |
| NoC atomics | BH `NoC/Atomics.md` | `[x]` typed L1-only requests (variable-width increment, compare-and-swap, mask and indexed swap, eight Zaamo ops, six accumulate formats; `tt_isa::noc::atomic`) with independent decode-from-bits models; all 21 forms and both model mutants pass on card 0 (step116, neighbour tile, one process each); ttsim executes only the full-width increment (rows 88) | R1 across tiles |
| NoC counters / interrupts | BH `NoC/Counters.md`, `Interrupts.md` | counters `[x]`; completion polling `[x]` (on card 0 the `NIU_TRANS_COUNT_RTZ_SOURCE` bit is set after completion, stays set across a later request without a clear and is removed by a clear; the earlier "not established" came from a mutant that could not be observed by before/after samples, replaced by ignored-clear and read-to-clear mutants; a broadcast sets the bit at its first acknowledgement so it is not a multicast completion; step117; ttsim refuses the registers); interrupt handler `[-]` (ttsim has no PIC and a mis-vectored IRQ on silicon runs arbitrary L1; the counter poll the mover uses is the delivered contract) | -- |
| `L1CacheTagSearchAccel` | BH | `[-]` helpers, page model and B probe kept as evidence (`tt_isa::tag_search`, `tag_search_b`, step120; configured through `Config[212..=219]`, triggered by an L0-missing RISC-V B load). **On card 0 every armed trigger load hangs the baby core** (all ten scenarios stop at step 0, `LOAD_ISSUED`) and the block then stays armed across resets (the earlier passing minimal probe fails afterwards), so the silicon semantics are not established and the block is not adopted; no repo consumer; ttsim refuses the config (row 92). Silicon tests are `#[ignore]`d; a board reset clears the state | checklist Phase 9 |
| Debug timestamper | BH (STUB-C) | `[x]` silicon; ttsim row 54 | -- |

### Out of scope, and why

- `[-]` **L2CPU tiles.** Harts leave reset only once per power cycle (`L2CPUTile/README.md:30`).
  Out of scope for Phase 10, but no longer for the reason first given ("nothing a Burn
  backend needs runs better there"): the measured host round trips say the opposite. A
  proposal to use the x280s as an on-card host -- driving traces, small ops, dispatch -- is
  `feature-x280-on-card-dispatch.md`, awaiting a decision.
- `[-]` **PCIe DMA engines.** No register-level documentation (open question 5);
  residency makes bulk transfer a startup cost.
- `[-]` **A GDB stub over the debug interface.** No Blackhole bit layouts (open question 2).

### Tensix coprocessor instruction implementation checklist

Remaining-work review (2026-10-07):
[remaining-firmware-instructions.md](remaining-firmware-instructions.md)
accounts for every pending mnemonic group, distinguishes existing encoding/probe
evidence from missing helpers and semantic gates, and sequences implementation
with explicit research/defer dispositions for unsupported forms. The latest completed tranche
is [Stage D explicit healthy-bank handover](../completed-plans/explicit-unpacker-handover.md)
(step104), following step100–103.

Mnemonic-level instruction inventory: 92 completed checklist rows (`[x]`),
16 pending rows (`[ ]`), and six deliberately omitted groups (`[-]`,
`RMWCIB0..3`, `DOTPV`, `SHIFTXA`, `SETDVALID`, `REG2FLOP_ADC`, `FLUSHDMA`).
These are grouped mnemonic rows, not a count of generated
encodings: generated tables also contain variants and superseded Wormhole
layouts. The earlier 68/47/4 totals were stale, and SETDMAREG was missing here.

Step100 acceptance (2026-10-07): nine isolated release card-0 gates pass
(run `1791379895`), including every configuration ALU/mask mode, both banks,
all issuing threads, every scratch target/selector, scalar register/immediate
forms, aliasing, independent integer/config models and diagnostic replay.
Simulator refusals have surviving controls; supported register multiply and
configuration read/Add remain semantic gates. Only register MULDMAREG provenance
is promoted. Descriptor publication needs an explicit Configuration Unit barrier
in tight replay; see operating notes. Default tensor dispatch stays unchanged;
Burn routing, gradients and padding are inapplicable to step100. DMANOP is now
accepted in step101; FLUSHDMA remains excluded by the review above.

Step101/102 acceptance: twelve isolated release card-0 gates pass
(`1791389018`). T0 performs all movement; T2 carries a declared two-semaphore
output ownership shell, since an empty T2 cannot publish T0 output safely.
F32/I32/Bool/raw BF16, physical zero padding, whole tile-row views, changed-input
traces, deferred frees and downstream reductions/matmul pass. No automatic Burn
routing, mesh/BFP/in-place/rectangle API or firmware ABI change. See operating
notes for measured STOREIND widths and the output-credit handoff, and the
performance scoreboard for validated release comparisons.
[Completed checklist](../completed-plans/scalar-config-foundation.md).

Encoding, semantic evidence and production adoption are distinct. In particular,
ADDDMAREG already drives matmul address stepping (step38), and the delivered LUT
forms already have helpers/models/device gates (step26); broader LUT adoption
remains S10. Step98 adds INCADCXY/ADDRCRXY helpers and an explicit ADC rectangle-copy
kernel. Burn slices retain original native repack after the ADC performance
comparison; automatic ADC adoption is deferred. See [the tranche checklist](adc-row-window-copy.md).

Step98 acceptance (2026-10-06): simulator 9/9; card-0 full release SMOKE
203/203 (`1791321817`), including nine ADC gates and the CNN state-lifetime
regression. Eight MNIST e2e regressions pass with the golden unchanged.
Final isolated benchmark `1791322068` preserves every output bit but costs
23–79% more than native repack for the measured rectangles; conditions and
dataflow counts are in `docs/learnings/firmware-performance.md`. Ordinary
silicon acceptance uses the requested one-card policy.

Burn performance restoration: the original native slice dispatch and engine
plumbing are restored; explicit ADC copies remain available. Updated step98
passes 9/9 simulator gates and card-0 step98 plus CNN passes 13/13
(`1791323039`). ADC optimization and automatic Burn adoption remain deferred.

Step99 acceptance (2026-10-06): explicit Session F32 plane selection uses bounded
staging and both Z/W instructions, preserving exceptional raw bits and zero
padding for ragged dimensions and row views. Simulator 9/9; card-0 ADC followed
by CNN 13/13 (`1791324670`); full release SMOKE 212/212 (`1791324245`). Fresh
copies pass under both ownership settings; changed-input traces, deferred frees
and downstream reduction/matmul pass. Workspace tests/lints, silicon compilation,
generator/shipping checks and all eight MNIST regressions pass, golden unchanged.
Burn routing is inapplicable to this Session-only API. See
[the Z/W tranche checklist](adc-plane-copy.md) and the performance record for
validated native-repack medians. No speedup or additional instruction adoption
is claimed. At step99 close-out, the pending group counts were 7 matrix/source, 1 SFPU,
6 unpacker/packer, 2 frontend, 2 configuration and 19 DMA/register/atomic rows;
REG2FLOP_ADC was pending and RMWCIB0..3 deliberately omitted. Step100 and the
2026-10-07 exclusion review above supersede these historical counts.

#### Matrix Unit (FPU) & Formats (22 instructions)
- [x] `MVMUL`: Matrix-vector multiply (primary GEMM accumulation engine, `Session::matmul_dram`).
- [x] `GAPOOL`: General average pooling (drives BF16 pooling in `crates/tt-kernels/src/fpu.rs`).
- [x] `MOVA2D`: Move $SrcA \to Dst$ (used in BF16 format widening and datapath staging).
- [x] `ZEROACC`: Zero accumulator registers in $Dst$.
- [x] `SETRWC`: Set matrix read/write/column coordinate counters.
- [x] `INCRWC`: Increment matrix read/write/column coordinate counters.
- [x] `ELWADD`: Matrix-unit elementwise addition ($SrcA + SrcB \to Dst$). Item M1.
- [x] `ELWSUB`: Matrix-unit elementwise subtraction ($SrcA - SrcB \to Dst$). Item M1.
- [x] `ELWMUL`: Matrix-unit elementwise multiplication ($SrcA \times SrcB \to Dst$). Item M1.
- [-] `DOTPV`: redundant with non-broadcast `MVMUL`; the pinned `DOTPV.md` explicitly prefers MVMUL. Retain step9 encoding probes, no production helper or new semantic tranche.
- [x] `TRNSPSRCB`: native SrcB block permutation gated on both cards (step87); M3 payload-preserving tensor transpose remains partial.
- [-] `SHIFTXA`: pinned `SHIFTXA.md` calls it unsupported: its input row depends on a preceding matrix instruction through noncontractual hardware behavior. Use explicit staging/repacking or validated SFPU lane movement; no Blackhole bug measurement is claimed by this exclusion.
- [x] `SHIFTXB`: checked non-flipping Loaded-B rotate/zero-fill; step103 physical permutation, row wrapping, modifiers 0/1/4 and data mutants pass card 0 (`1791397029`), silicon-only under row 50. Item M4.
- [x] `MOVD2A`: typed non-flipping helper, resident matrix chains and production instruction/traffic audits; step97 simulator and both-card semantic gates pass (`1791317039`). Step9 encoding provenance is retained.
- [x] `MOVD2B`: typed non-flipping helper, resident matrix chains and production instruction/traffic audits; step97 simulator and both-card semantic gates pass (`1791317039`). Step9 encoding provenance is retained.
- [x] `MOVB2A`: typed non-flipping helper, resident matrix chains and production instruction/traffic audits; step97 simulator and both-card semantic gates pass (`1791317039`). Step9 encoding provenance is retained.
- [x] `MOVB2D`: typed `Banks::movb2d`, production `fpu.rs` transpose and step73 readback; measured four-row encoding in `probe_src`.
- [x] `MOVDBGA2D`: checked `tt_isa::matrix_debug` helpers (bank ownership, 1/8-row forms, format override) and an independent page model; on card 0 (step111) the eight-row 0->0 and 8->8, one-row 3->5, all four SrcA format overrides (each forced to TF32 under Fp32), flush, data mutants and reading a bank the unpacker still owns all agree with the model; ttsim refuses every form (extends row 50). Open, not specific to this instruction: the increment-1 `AddrMod` advance cases disagree with the model for the `MOVA2D` control too (shared address-modifier configuration; increments of 8 agree); those tests are ignored. 16-bit Dst forms are model-only.
- [x] `ZEROSRC`: checked A/B/both current-unpacker and current-matrix banks; ownership/staged claims, all 64 rows and opposite-bank sentinels pass step103, card 0 (`1791397029`). Matrix-bank forms run on ttsim; unpacker-bank forms refuse. Negative-infinity and both-physical-bank forms remain deferred.
- [x] `CLEARDVALID`: checked separate/both Loaded → Empty releases, Reset=0 and KeepReadingSameSrc=0; step103 simulator/card-0 (`1791397029`) alternating-bank and downstream reuse gates. Reset stays excluded; retained-bank reading needs a separate state model.
- [x] `CLREXPHIST`: typed diagnostic helper and independent histogram/max reset on both cards (step93); exponent selection is gated separately. Item D2.
- [-] `GATESRCRST`: the encoding executes safely on card 0 (isolated probe) but there is no observable oracle: with SrcB rewritten through `MOVD2B`, the no-gate, operand-clear and gate arms all read fresh while the no-rewrite detector control reads stale (step111, verdict `NoObservableOracle`, as the page predicts: the instruction is needed only against invalidation-hardware bugs). Supported alternative: normal SrcB loads through the checked `Banks` helpers. The checked helper stays for diagnostics.

#### Vector Unit (SFPU) (40 instructions)
- [x] `SFPADD`: Lanewise floating-point addition/subtraction.
- [x] `SFPMUL`: Lanewise floating-point multiplication.
- [x] `SFPMAD`: Lanewise multiply-accumulate (primary driver for polynomial transcendental fits).
- [x] `SFPARECIP`: Approximate reciprocal and exponential seed.
- [x] `SFPMOV`: Register move and negation.
- [x] `SFPIADD`: 32-bit two's-complement integer addition/subtraction.
- [x] `SFPMUL24`: 24-bit integer multiplication (used for 32-bit full product reconstruction).
- [x] `SFPABS`: Floating-point and integer absolute value.
- [x] `SFPSETSGN`: Set sign bit or absolute value.
- [x] `SFPGT`: Compare greater-than (signed integer and floating-point total order).
- [x] `SFPLE`: Compare less-than-or-equal.
- [x] `SFPLZ`: Test zero / non-zero or count leading zeros.
- [x] `SFPSETCC`: Set condition codes from sign, zero, or comparison.
- [x] `SFPENCC`: Enable or disable conditional execution scopes.
- [x] `SFPPUSHC`: Push condition flags onto SIMD flag stack.
- [x] `SFPCOMPC`: Complement condition flags for SIMD `else` branches.
- [x] `SFPPOPC`: Pop condition flags from SIMD flag stack.
- [x] `SFPAND`: 32-bit lanewise bitwise AND.
- [x] `SFPOR`: 32-bit lanewise bitwise OR.
- [x] `SFPXOR`: 32-bit lanewise bitwise XOR.
- [x] `SFPNOT`: 32-bit lanewise bitwise NOT.
- [x] `SFPSHFT`: In-lane 32-bit logical and arithmetic shifts.
- [x] `SFPSHFT2`: Cross-lane lane rotation and shuffling (drives row reductions).
- [x] `SFPTRANSP`: Cross-lane $4\times 4$ matrix transpose within columns.
- [x] `SFPLOAD`: Load from $Dst$ into vector registers ($LReg$).
- [x] `SFPSTORE`: Store from vector registers ($LReg$) into $Dst$.
- [x] `SFPLOADI`: Load 16-bit or 32-bit immediates into vector registers.
- [x] `SFPSETEXP`: Set floating-point exponent field.
- [x] `SFPSETMAN`: Set floating-point mantissa field.
- [x] `SFPEXEXP`: Extract floating-point exponent field.
- [x] `SFPEXMAN`: Extract floating-point mantissa field.
- [x] `SFPCONFIG`: Configure SFPU constant registers (`LReg[11..15]`).
- [x] `SFPSTOCHRND`: Hardware nearest, stochastic, and toward-zero rounding (validated in `step81`).
- [x] `SFPNOP`: Vector unit pipeline no-op.
- [x] `SFPLOADMACRO`: checked `tt_isa::sfpu_macro` helpers (SFPCONFIG macro registers, program-key descriptor, teardown) and an independent page-derived schedule model (`tt_kernels::sfpu::macro_sched`); silicon probes s00-s13 pass on card 0 (Store, MAD, Simple, Round, `LReg16`, substituted operands, chain, pipelining, predication, forgetting, `SFPSWAP` in the Simple sub-unit; swapped template/delay mutants differ). ttsim refuses the macro (row 7). Step110. Item S9; performance adoption not claimed.
- [x] `SFPDIVP2`: `Program::scale_by_pow2`, interpreter/device gate step26; wrapping immediates separately gated on both cards (divergence 66).
- [x] `SFPSWAP`: `Program::min_max`, step26 interpreter/device comparisons and production reductions. Argmin/argmax variants are separate scope.
- [x] `SFPADDI`: `Program::addi`, BF16-immediate interpreter/device comparison in step26.
- [x] `SFPMULI`: `Program::muli`, BF16-immediate interpreter/device comparison in step26.
- [x] `SFPLUT` / `SFPLUTFP32`: delivered table forms already have step26 device gates; simulator restrictions are divergence 70. Broader lookup adoption remains Item S10; this completion covers delivered instruction forms, while broader kernel adoption remains open.

#### Unpackers & Packers (16 instructions)
- [x] `UNPACR_Regular`: Streaming unpack from L1 to $SrcA$, $SrcB$, or $Dst$.
- [x] `PACR`: Streaming pack from $Dst$ to L1 with hardware format conversion/narrowing.
- [x] `SETADC`: Set address counter registers for 4D tile coordinates.
- [x] `SETADCXX`: Set address counter XX channel.
- [x] `SETADCXY`: Set address counter XY channel.
- [x] `SETADCZW`: Set address counter ZW channel.
- [x] `INCADCXY`: checked current-thread helper and live-counter stepping in ADC rectangle copies; step98.
- [x] `INCADCZW`: checked current-thread Z/W helper and bounded resident plane traversal; independent state/address models and step99 simulator/card-0 gates.
- [x] `ADDRCRXY`: checked cursor-relative helper and row-anchor restoration in ADC rectangle copies; step98.
- [x] `ADDRCRZW`: checked Z/W cursor restoration/advance helper in resident plane copies; step99 covers all masks/targets, wide addressing, zero restoration and negative controls.
- [-] `SETDVALID`: Blackhole implied-format handover is explicitly unsupported in the pinned page. Use regular UNPACR's final FlipSrc; sequenced `UNPACR_NOP_SETDVALID` remains a separate gated task.
- [-] `REG2FLOP_ADC`: pinned page declares unsupported functionality and weak model confidence. Use checked SETADC/SETADCXX/XY/ZW and descriptor reprogramming; no general GPR-to-ADC API is promised. Reopen only for a concrete runtime-value consumer and independent Blackhole evidence.
- [x] `UNPACR_NOP_SETDVALID`: measured Blackhole non-clearing mode 0x1e9; Filling-only A/B handover with C1/C2 retirement and C5/C6 ownership waits. Alternating banks, TF32/BF16, multiple partials, sentinels and replay pass `1791413233`; nonzero SrcB row reset and direct consumers pass `1791413291`, tightened fractional inputs `1791413841`. Card-0 SMOKE 258/258 (`1791413478`). Legacy Wormhole mode 7 is the isolated reboot trigger; exact ARC mechanism remains unproven. See [completed Stage D](../completed-plans/explicit-unpacker-handover.md). Other modes and production/recovery adoption are deferred.
- [-] `UNPACR_NOP_SETREG`: Wormhole page says UnsupportedFunctionality with weak confidence and it uses TDMA-RISC `SetRegBase` state with no Blackhole page; ttsim refuses it (step108). Supported alternative: checked WRCFG/SETC16/RDCFG and step100/101 helpers.
- [x] `UNPACR_NOP_ZEROSRC`: measured BH wait/bank/clear fields, checked current-unpacker zero (WaitLikeUnpacr=1, BothBanks=0); staged claims discarded and regular UNPACR handover retained. Step103 ttsim/card-0 gates, including held-opposite-bank wait and all-row sentinel checks (`1791397029`). Wider values/both banks remain outside the checked API.
- [-] `PACR_SETREG`: no Blackhole page; `SetRegBase`/`SetRegHiScaler` are TDMA-RISC state set only through Wormhole-documented writes into a block shared with the mover command queue; ttsim refuses it (step108). Supported alternative: `PACR` with `Last`, `STALLWAIT` on packer-busy, and the checked configuration helpers.

#### Frontend, Synchronization & Expanders (10 instructions)
- [x] `REPLAY`: Hardware micro-op execution buffer for unrolled loops (Item X1).
- [x] `MOP` / `MOP_CFG`: Macro-op instruction expander templates (Item X2).
- [x] `SEMWAIT`: Wait on hardware semaphore between threads.
- [x] `SEMPOST`: Post hardware semaphore to signaling unit.
- [x] `SEMINIT`: Initialize hardware semaphores.
- [x] `SEMGET`: Query hardware semaphore counter.
- [x] `STALLWAIT`: Hardware pipeline barrier (blocks until unit execution drains).
- [x] `NOP`: General backend pipeline bubble.
- [-] `STREAMWAIT`: no Blackhole NoC Overlay specification is pinned (`BabyRISCV/README.md` links a non-existent `NoC/Overlay` page; `STREAMWAIT.md` defines its conditions only through `NOC_STREAM_READ_REG` with no address or layout); no controlled producer exists without NoC or overlay traffic; a wrong overlay write already rebooted the host (operating notes, stream-pop mode 3). Raw encoder and ThreadConfig fields (`STREAMWAIT_*`, `STREAM_ID_SYNC_SEC*`) retained. Alternative: L1 semaphores/credits with `SEMWAIT`/`STALLWAIT` through the streaming scheduler.
- [-] `STREAMWRCFG`: same missing register layout; no pinned harmless stream register with a known value to read back; documented reordering bug requires `STALLWAIT` afterwards. Reopen when a Blackhole overlay register map is pinned and an isolated silicon read of an idle stream is measured. Alternative: `LOADREG`+`WRCFG` (the page's own suggestion), checked `WRCFG`/`RDCFG`.

#### Backend Configuration
- [x] `WRCFG`: 32-bit and 128-bit backend configuration writes.
- [x] `SETC16`: Direct 16-bit thread configuration writes.
- [x] `RDCFG`: checked backend read plus full Configuration Unit wait; both banks/all threads (step100).
- [x] `CFGSHIFTMASK`: checked eight ALU modes, mask preservation/replacement, four scratch selectors and restricted mutation targets (step100); full matrix silicon, simulator supports only unrotated full-width preserved Add.
- [-] `RMWCIB0..3`: Read-Modify-Write Configuration Immediate Byte (`libttsim_bh.so` has no handler; whole-word `WRCFG` used instead).

#### DMA Engine, Atomics & Registers
- [x] `ATCAS`: checked 4-bit compare/set (`tt_isa::l1_atomic`); compare already met, blocked-until-the-host-writes-the-word and blocked-until-the-other-role's-RISC-V-core-pokes-it pass on card 0 under a guarded run with a measured-rate deadline (step106). **A Tensix thread cannot be the producer:** a thread parked in `ATCAS`/`ATINCGETPTR` keeps every other thread from issuing any Scalar Unit instruction (confirmed on card 0, `blocked_atomic_monopolizes_the_scalar_unit`), so only the host or a RISC-V core can free it. WormholeOnly encoding; ttsim refuses it (row 77).
- [x] `ATGETM` / `ATRELM`: typed `Mutex` (indices 0, 2, 3, 4 only), scoped acquire/release; uncontended on all indices and threads, contended round-robin handoff for every mutex and holder, the deadline/host-release path and negative controls pass on card 0 (step105). ttsim models only index 0 (row 76).
- [ ] `ATRELM`: Atomic mutex release.
- [x] `ATSWAP`: four-GPR group form, all 256 masks and aligned bases on card 0 (step106); the single-register form is `[-]` (lane placement matches neither the page nor a consistent rule; the sweep diagnostic is the evidence) and is unrepresentable in the API.
- [x] `ATINCGET`: field width 1-32, wrapping, upper bits preserved, atomic across the three threads on card 0 (step106); WormholeOnly encoding, ttsim refuses it.
- [x] `ATINCGETPTR`: checked geometry, independent FIFO model; the FIFO through wraps and the blocking pop (write counter poked) and push (read counter poked) gates pass on card 0 (step106). Same Scalar Unit rule as `ATCAS`; WormholeOnly encoding; ttsim refuses it.
- [x] `SETDMAREG`: checked full-width GPR initialization through `backend::set_gpr`; configuration staging and matmul address stepping.
- [x] `ADDDMAREG`: production register-form matmul address stepping and step38; the immediate form is silicon-only (divergence 67).
- [x] `SUBDMAREG`: checked wrapping subtraction, register/immediate forms (step100); silicon semantics, simulator refusal.
- [x] `MULDMAREG`: checked low-16 unsigned multiplication, register/immediate forms (step100); register encoding confirmed on simulator/card 0, immediate silicon-only with WormholeOnly provenance.
- [x] `CMPDMAREG`: checked unsigned greater/less/equal, exact 0/1 result (step100); silicon semantics, simulator refusal.
- [x] `SHIFTDMAREG`: checked logical left/right, register low-five count and 5-bit immediate (step100); silicon semantics, simulator refusal.
- [x] `BITWOPDMAREG`: checked AND/OR/XOR, register/immediate forms (step100); silicon semantics, simulator refusal.
- [-] `FLUSHDMA`: occupies the shared Scalar Unit while waiting; pinned page prefers `STALLWAIT` with equivalent C0–C3 conditions and all block bits. Excluded from production support in favor of the existing barrier; not a claim that every possible use is strictly worse.
- [x] `LOADIND`: checked widths/offset halves/increments, asynchronous read barriers and raw-bit preservation (step101, card 0).
- [x] `STOREIND_L1`: checked L1 stores; measured Blackhole width mapping and all-thread guard gates (step101).
- [x] `STOREIND_MMIO`: allowlisted `SW_INT_PC[28..31]` PIC words only (`tt_isa::mmio_reg`), STOREIND's shifted offset modelled separately; the isolated probe, offsets/increments/halves/threads gate and the wrong-shift control pass on card 0 (2026-10-10, no reboot); ttsim refuses every form.
- [x] `LOADREG`: allowlisted PIC scratch words only; the isolated probe and the host-staged-values gate pass on card 0 (step107).
- [x] `STOREREG`: allowlisted PIC scratch words only; the isolated probe and the store-then-read-back gate (all targets, all threads) pass on card 0 (step107).
- [x] `XMOV`: checked declared L1 copy/zero, C12 setup/C9 completion and explicit resident APIs (step101/102). Item D4.
- [x] `DMANOP`: diagnostic GPR/config/memory preservation (step101), never a completion wait.

---

## Work items

In dependency order. Each names the Burn methods it unlocks; the Burn table below is the
reverse index.

### X — Frontend expanders and tracing

- [x] **X1 `REPLAY`.** Done for SFPU row loops (`tt_isa::frontend::{record, replay}`,
      `REPLAY_BUFFER`; `Program::for_each_row_group`); `step26_sfpu_isa` runs every
      case replayed and unrolled against one interpreter tile, ttsim and both cards,
      and ttsim models `REPLAY` (no divergence). Was: `tt_isa::frontend::replay`: `record(slot, body, exec)` and
      `replay(slot)`, over a per-thread `ReplaySlots` allocator of the 32-entry buffer that
      refuses overlap, a body over 32 and a nested `REPLAY`. The buffer is per-thread state
      that survives between programs (divergence rows 47, 49), so a program records before
      it replays. F2's row-group loop records its body once and replays it where it fits,
      unrolling (and saying so) where it does not. Gate: replayed programs bit-identical to
      their unrolled form on ttsim and both cards.
- [x] **X2 `MOP` / `MOP_CFG`.** Typed templates behind a builder; reconfiguration only
      after `MOPExpanderDoneCheck` (`ManualTTSync.md:57`); Auto TTSync takes the `MOP`'s
      resource declaration (`AutoTTSync.md:26`). Applied to the matmul inner loop, the
      unpacker face loops and `pack_rows`. Gate: MNIST golden bit for bit; program bytes
      down; silicon time measured.
  - [x] **X2a The expander, configured from the mailbox.** `tt_isa::frontend::mop`:
        `MopConfig` (templates 0 and 1, Blackhole's ten-bit counts) refusing what the page
        marks unsupported -- the count overrides, the start/inner/end shape with the
        iteration-count bug, a `MOP` in a loop slot -- and `expand`, the page's functional
        model, as the oracle. A kernel's `mop: [Option<MopConfig>; 3]` goes into each
        role's mailbox (`mailbox::MOP_CFG_VALID`, `MOP_CFG`, nine words); the runner waits
        on `MOPExpanderDoneCheck` and writes them to `TENSIX_MOP_CFG_BASE` before
        pushing, so nothing but instructions is ever in the stream. The session's
        descriptor comparison covers the words, so a queued kernel's configuration is
        never rewritten under it. Gate `step36_mop`: template 1 (start, last and two end
        ops; alternating loop ops) and template 0 (a mask over both halves, 20
        iterations) as integer adds whose sum counts each slot -- the device's tile bit
        for bit the interpreter's on the model's expansion, and the first sum checked by
        hand; and two runs of one `MOP` under two configurations give their own sums
        (watched failing: the hand count off by one; no configuration loaded, ttsim's
        contract exit). ttsim, then silicon alone on the gate tile, both cards. `MOP` and
        `MOP_CFG` were Wormhole-only drawings: the generator now marks them `CONFIRMED`
        with this gate as evidence (`xtask/src/gen_isa/measured.rs`, `CONFIRMED`: only a
        `WormholeOnly` layout, the gate must exist, every field must be exercised).
  - [x] **X2b Applied**: the matmul's `MVMUL` loop, the unpacker face loops, `pack_rows`;
        MNIST golden bit for bit, program bytes and silicon time measured.
    - [x] **Kernels are push-bound, measured** (`silicon_perf::role_push_rate`): a
          matmul tile's unpack role pushes at the runner's ceiling and its backend
          finishes the moment the last word lands, so fewer words is faster. The
          runner itself now pushes in batches of sixteen (2.8 cycles a word, was
          7.6): every kernel about 2.5x faster to issue (row AB). Full suite 424/424.
    - [x] **The loop planner** (`tt_kernels::loops`): `Item::Repeat` lowered to a
          `MOP` looping a `REPLAY` of the recorded body (the loop that saves the
          most takes the one configuration), plain `REPLAY`, or unrolled, each
          choice recorded with its reason; a program with `REPLAY`s of its own is
          left unrolled. Unit tests: the modelled frontend's output
          (`loops::frontend_stream`, MOP then Replay Expander) is the unrolled
          program word for word. Gate `step37_loops`: lowered and unrolled store
          the same counted sum, and the lowered is under a quarter of the words
          (watched failing: one `MOP` dropped, short by exactly its five
          iterations). ttsim and both cards: a `MOP` looping a `REPLAY` on silicon.
    - [~] **The matmul in loops** (`matmul::matmul_items`, lowered by
          `loops::lower_with`):
      - [x] Faces in the order `fi`, `k`, `fj`: each `Dst` face still takes its
            `k = 0` product before its `k = 1` within a pair, so every datum
            accumulates as before -- the MNIST golden is bit for bit -- and each `A`
            face is unpacked once for the two `MVMUL` groups that read it
            (`mvmul_release_a`, then `_both`). The unpackers' `X` range is set once.
            A pair's face block is the same words for every pair: 24 on the unpack
            role, 32 on the math role at LoFi.
      - [x] Tiles stepped by GPR arithmetic where the pairs are evenly spaced (the
            gather's layout): each operand's base and stride in GPRs 24-27, and per
            pair the same eight words -- wait, `ADDDMAREG` base += stride, `WRCFG`
            both -- so each output's K loop is one `Repeat` of the face block and the
            step, 32 words, replayed (`gpr_step`; uneven pairs keep the explicit
            retarget and share the face block). `ADDDMAREG` gated alone first
            (`step38_gpr_add`, both forms on silicon, the register form on ttsim --
            row 67 -- and now `CONFIRMED`). A body already in the buffer is replayed
            by the next output without recording it again.
      - [x] Measured (row AC): a 1x8x1 tile's unpack 514 -> 116 words, LoFi math
            269 -> 53; MNIST inference 0.61 -> 0.53 ms a batch. Full suite 430/430.
      - [x] Math at every fidelity in one replayed unit per `fi`: each `MVMUL`
            names its `Dst` row within the `fi` (the same for both), the RWCs' `Dst`
            holds the `fi`'s base, and address modifiers do the rest
            (`matmul::MATH_AM_*`): 1 the phase, 2 a half's end (`SrcB` on 8), 3 a
            group's end (`SrcB` back), 4 an `fi`'s end (`Dst` on 32). A unit is 32
            `MVMUL`s at HiFi4, so a pair's math is a reset and two `REPLAY`s; the
            planner now writes out a loop too long to record as items, so the loops
            inside it still replay. MNIST golden bit for bit; HiFi4 math for a
            1x8x1 tile 653 -> 75 words, the kernel now held by the backend, not by
            pushing (row AD). A first version moved the base by `Dst`'s carriage
            return (`DestCR`) and left output face (1, 0) wrong on ttsim, though each
            modifier behaved in isolation (`step9`-style probe): unexplained, so the
            design uses plain increments only. Modifiers persist between programs:
            `step9`'s measurement now sets every entry it relies on.
      - [x] The MOP carried to the roles: `Step::Kernel` and the session's lists
            take each role's `MopConfig` (a list's kernels share one descriptor, so
            a list splits where it changes), and `matmul_kernel` lowers the math
            role's loops under one when asked. Measured (row AE): a `MOP` on the
            unpack or pack role is slower than their replays, and on the math role
            no faster end to end -- the matmul is backend-bound -- so the session
            runs it without, which also keeps the matmul ttsim's path (row 68).
            The expander stays gated (`step36_mop`, `step37_loops`) for a loop that
            is push-bound. Full suite: ttsim, silicon 430/430; MNIST 91.96%, 2.0 /
            1.6 ms a step (1 / 4 tiles), inference 0.45 ms a batch of 64.
- [x] **X3 The debug timestamper as a device profiler** (concepts review G13). The B
      mover brackets each list and each top-level entry or record with timestamper events
      when `dm::TRACE` is set (tokens: `tt_isa::mailbox::trace`, source in bits 8..12,
      the op or record kind as detail); role runners record start/pushed/retired per run
      under `Resident::set_profiling`. `Session::profile_start`/`profile_stop` arm every
      unit, drain each stream after every wave (so the 1024-event buffer bounds a wave,
      not a profile), and return a `DeviceProfile`: spans per unit, checked to nest,
      Chrome trace JSON on the host's time line, the clock measured (row N).
      `TT_PROFILE=<path>` profiles a whole burn-tt attachment. Helper/oracle: unit tests
      of the token layout, pairing, refusal of a non-nesting stream and the export.
      Gates (`step23_profile`): on ttsim, profiling refused with its reason and the
      session usable after; on silicon (both cards), two tiles running an add and a
      ragged matmul -- every entry inside a list, exactly one run of each role inside
      each `KERNEL` entry, no role run outside one -- and 300 lists on one tile, more
      than the buffer holds, none lost. Watched failing with the drain disabled (the
      overflow refusal). Sim `[-]`: row 54. First use: row O, the reduced-MNIST
      breakdown. Burn: not applicable (no op).
- [x] **X4 Dispatch: queue, barriers, batching, traces** (concepts review G8). Per-op
      cost is the host's submission and wait (~100-200 us an op, measurement S), so:
  - [x] **X4b A barrier across movers by NoC atomics.** `tt_isa::noc::niu::Command::
        AtomicIncrement` (`CMD_AT`, `NOC_AT_LEN_BE`'s increment layout from
        `Bits32.lua`), and `dm::op::BARRIER` -- wait for this unit's moves, increment
        the coordinator tile's `dm::BARRIER_COUNTER`, poll it by NoC read until the
        target (`k * n` for the `k`-th barrier of `n` units). Gate `step33_barrier`:
        tile A's read after a barrier sees tile B's GDDR write before it, A's list
        submitted first; three rounds, every arrival counted. Watched failing with A's
        barrier after its read (round 2 reads stale bytes). ttsim models the atomic;
        both cards.
  - [x] **X4a A command queue on the mover.** `tt_isa::dm::QUEUE_*`: sixteen slots
        `(first entry, entries)` over the 512-entry list ring; the mover runs queued
        lists in order and counts them done, and a failed list stops the queue with
        its number and code. `DataMover::{enqueue, wait_for, drain, refresh}`: the host
        writes a list where it fits beside those in flight (`ring_room`, never across
        the end; unit test with a 10k-step soak) and waits only for room or a result;
        a stuck queue times out on no progress. Gate
        `step33_barrier::queued_lists_run_in_order_without_waiting`: forty chained
        copies enqueued without waiting -- past the slots and the ring -- arrive whole.
        Watched failing with every list placed at entry 0. ttsim and both cards.
  - [x] **X4c Batching in the session**: ops queue with their outputs placed; a sync
        point (download, explicit) submits them, barriers between multi-unit ops; then
        burn-tt's ops are asynchronous for free. `Session::{sync, set_batching}`
        (`TT_BATCH`, default on): each segment is enqueued on the mover's queue, a
        barrier list follows a multi-unit op, frees wait for the lists that may read
        them, and downloads, runs and profiles sync first. Two hazards found on
        silicon, both closed (table below): a full program cache never makes room
        while lists are queued (the programs they run stay pinned; the session drains
        and places again), and an upload is visible through every port of its channels
        before `dram_write` returns (divergence row T). Gate `step34_batching`: a
        40-op chain of four SFPU kinds through a cache cut to about two kinds'
        programs, bit for bit to the interpreter, with evictions (watched failing with
        eviction allowed while queued: ttsim's contract-violation exit); a four-unit
        layer forward with several lists per unit per op, eight queued passes bit for
        bit to the unbatched one; and an upload-then-op guard (row T: it does not
        reproduce the race, `tt-mnist` does). ttsim and both cards. MNIST: row U.
  - [x] **X4d Traces** -- opt-in capture and replay, inference first (tt-metal's
        `BeginTraceCapture`/`ReplayTrace` the model, its footguns designed out). As built
        (`tt_kernels::trace`, `Session::{begin_trace, end_trace, replay, release_trace,
        write, trace_ops}`; Burn: `burn_tt::Trace`):
    - **Opt in, from wherever the caller is.** `begin_trace` ... `end_trace` around any
      stretch of a batching session's ops captures and runs it once; `replay(id)` runs it
      again, queued like any op. Between replays `Session::write` overwrites a tensor the
      trace reads (shape-checked, a view refused), and the tensors it wrote hold the
      results. Untraced ops run as before, interleaved freely.
    - **Replay without the host.** At `end_trace` each unit's stream goes to GDDR on its
      own channel, padded with `WAIT`s so no record crosses a 64-entry chunk
      (`trace::chunked`); a replay is one list per unit holding one `CALL`
      (`tt_isa::dm::op::CALL`), which the mover runs a chunk at a time from
      `dm::TRACE_CHUNK`. A `CALL` is a list of its own, and one inside a trace is
      refused. Per-run values: each top-level `KERNEL`'s generation and each `BARRIER`'s
      target are patched in the chunk by the `CALL`'s bases (`Resident::take_generations`,
      the session's barrier count); the roles' descriptors are `POKE` entries
      (`dm::op::POKE`, role-mailbox words only) wherever the stream has not set them yet;
      a semaphore setup the host ran during the capture is a `KERNEL` of its own (thread
      0 its program, held in the program cache; threads 1-2 a zero-length program). A
      replay writes 40 bytes a unit over PCIe; the one-layer capture it repeats, 5.6 KB
      (row AG).
    - **Uncorruptible, or a typed refusal** (each provoked in `step39_traces`):
      - a live trace holds every allocation that existed when its capture ended: a free
        of one is deferred to the trace's release. The rule is exact without tracking
        each op's tensors: frees during the capture are deferred too, so what is
        allocated afterwards lies in what was free then (`tensor::FreeSnapshot`);
      - the programs its kernels name are held in the program cache
        (`ProgramCache::hold`; eviction, and the room-making clear, skip them);
      - the session's epoch moves on at every tile reset and mover start, and a replay
        against an older one is `TraceError::Stale`. A failed list -- a replay's too --
        recovers its units as before, so it makes every trace stale, not only its own;
      - a download, a `write` or a host-run kernel (`sync_run`: `run`, `prepare`, the
        host matmul) during a capture is `TraceError::HostTransfer`; turning batching off
        is `Capturing`. An upload is allowed: a constant the replay finds where it was;
      - the descriptors and semaphores the host remembers are forgotten at a capture's
        start (so it records everything its kernels need) and after a replay (so nothing
        queued later trusts them).
    - **Structured, so it can be optimized later.** Each op's `OpRecord` -- its name, its
      range of every unit's stream, whether a barrier followed -- is kept with the trace
      (`Session::trace_ops`); the placements each op read and wrote are the next field
      it needs. Optimizing over the captured graph is **X4e**.
    - **Burn.** `burn_tt::TracedInference` captures on the input's own buffers and
      returns the capture's output values; `run(inputs)` writes, replays and downloads
      in one round trip. Supports $N$ inputs, $M$ outputs, and arbitrary dtypes (`F32`,
      `BF16`, `I32`, `Bool`). Both return host values, not a tensor: a tensor of the
      output buffer would change under its holder at the next run. A closure that falls
      back to the host is refused (the op panics, as a device error does) and the capture
      is ended, never left open. Single-chip engines only; the mesh engine refuses.
      `tt-mnist --infer --trace` replays per batch.
    - Gates: `step39_traces` -- a layer's forward pass on one tile and on two (barriers)
      replayed over three new inputs, each the same ops run fresh bit for bit, and each
      replay's writes under a twentieth of the capture's; a freed weight deferred while
      an allocation of its size lands elsewhere; every refusal. Watched failing: a `CALL`
      over half its entries (replay 0, element 0 wrong). `step40_burn_trace` -- a Burn
      MLP traced and run on new inputs against the fresh Burn ops bit for bit; multi-input
      inference; whole training step tracing with in-place parameter writeback and loss
      readback; and a host fallback refused with the next capture working. ttsim and both
      cards. MNIST inference traced: 205,301 img/s on 8 tiles; training step traced:
      99.45 ms on 1 tile (outperforming 12-core CPU at 102.03 ms).
    - **Training: implemented (2026-10-05).** Whole-step training is fully replayable via
      `burn_tt::TracedTrainingStep` (`feature-traced-execution.md`). Updated weights are
      written back in place in GDDR using `Session::copy_into` captured into the trace stream,
      and scalar loss is read back after hardware replay completion without aborting the trace.
      Stateless (SGD) and stateful (Adam/AdamW) optimizers are supported across arbitrary Burn
      modules via `ModuleVisitor` parameter reflection. Evaluated on MNIST MLP and
      `TinyTransformer`. Multi-tile/multi-card scaling roadmap documented in
      `feature-traced-execution.md`.
  Was: **X4 Op-list traces** (concepts review G8). `Session::begin_trace`/`end_trace`
      capture each unit's expanded lists into GDDR; `replay` is one descriptor per unit,
      B streaming the list from GDDR; a trace binds its tensors and refuses to replay
      after one is freed. Gate: MNIST golden with steps replayed, steady-state PCIe writes
      per step down to the descriptors.
- [x] **X5 Wedged tiles: detect, then recover** (the hazard table's open wedge row).
  - [x] **X5a Detect at open, never fail opaquely.** A role that does not finish the
        tile reset (`session::reset_thread_state`, a few hundred instructions on an
        idle tile) is `RunError::Wedged { tile, roles }`, whose message names the tile
        and threads and says a board reset (`tt-smi -r`, or a power cycle) clears it.
        `Session::open` skips a wedged tile with a warning when tiles are chosen by
        count (`First`, `Count`, `All`), taking the next healthy one, and fails with
        that error for `Exactly`; too few healthy is `SessionError::TooFewHealthy`
        listing the wedged tiles. Unit tests: the selection (a wedged tile passed over,
        the search stopping once enough are found) and both messages. The signature it
        keys on, every stuck role a timeout, is the one the wedged tile (1,2) gave on
        both cards -- and the deliberate wedge of X5b gives again.
  - [x] **X5b Recover in software.** The cause, from the specification and then
        reproduced: a Matrix Unit instruction that reads `Src` waits by itself until its
        bank's `AllowedClient` is the Matrix Unit (`STALLWAIT.md`, C7/C8), and the
        backend pulse hands every bank to the unpackers (`SoftReset.md`, bits 15-16). One
        caught waiting by the pulse waits for good, and its thread takes nothing more --
        an unpacker caught waiting the other way is released by the same pulse, which is
        why only thread 1 stayed stuck. The recovery (`session::unwedge_tile`): the
        pulse, then `datapath::src_feeder` on thread 0 -- four plain `UNPACR`s, the
        matmul's own encoding, one into each bank of each `Src`, none of which can wait
        on a freshly pulsed tile -- which gives the stuck instruction its banks, then
        the pulse again to take them back. `prepare_unit` tries it once when the
        thread reset or the roles' restart comes back `Wedged`, logs the outcome, and
        reports `Wedged` (message updated) if the tile is still stuck. No `UNVERIFIED`
        encoding anywhere. Gate `step41_unwedge`: a math role of one `MVMUL` with
        nothing to feed it (verified encodings only) wedges tile (2,3) through the pulse
        and thread reset -- `Wedged` is asserted, so the gate is not vacuous -- the
        feeding run finishes, the roles restart, and a matmul is bit for bit the one
        before; the session does the same by itself after a failed kernel; and the
        recovery on a healthy tile leaves it healthy, the isolated gate run first.
        Watched failing with an empty feeder ("the feeding run did not finish"). ttsim:
        the wedge and the release by the feeder; it has no pulse to take the banks back
        (row 69), so there the session reports `Wedged` instead of computing from
        them. Silicon: both cards, four runs each. The wedge that started this (tile
        (1,2), from programs overwritten under a queued list) is gone with the boards'
        reset and prevented since X4c; another cause the feeder does not release still
        ends in `Wedged` and a board reset.

- [x] **X8 Block repeats in the role runner** (10.2, asked for when `pow` and `gelu`
      had to split into several ops). An SFPU row loop longer than the 32-entry replay
      buffer was unrolled 32 times into the 8192-word program slot -- `pow` and `gelu`
      came to ~9200 words, so they ran as chains of 4 and 2 ops. MOP and `REPLAY`
      cannot help: both are bounded by the replay buffer. Now a program may carry a
      loop header (`mailbox::loops`: `LOOPED` set in its length word, then the entry
      count, up to four `(start, len, count)` entries, then the code): metadata stored
      with the program, not instructions and not descriptor words, so it travels
      through the program cache and a `KERNEL` entry and kernels with different loops
      queue back to back under one descriptor. The runner (`tt_firmware::corpus::
      Pusher`) checks the table -- inside the code, any two disjoint or nested, at most
      two deep -- and pushes each span with the same sixteen-word fast path, a program
      without a header exactly as before; the mover's `KERNEL` check masks the flag.
      Host: `crate::code::{Code, Loop}` (`Code::stored` writes the header,
      `Code::expand` the stream every model runs); `Program::for_each_row_group`
      stores a body too long to replay once (`LoopForm::Repeated`, the same
      row-counter stepping as the replayed form), and the SFPU kernel's math role
      stores its per-tile block once and repeats it per tile (`kernel::roles_code`):
      LOG's math program for a run is 124 words whatever its length, from ~3500 a
      tile, and every SFPU op's run reaches 64 tiles (`ops::fit`). Found on the way,
      and fixed: a drain the descriptors needed came after a list's programs were
      placed, and unpinned them (hazard table). Gates: `crate::code` unit tests (nested
      expansion, every refusal), every SFPU device gate now running nested repeats
      against the interpreter's expansion bit for bit; watched failing with the
      runner's repeat count off by one (`step29`). ttsim (31 gates) and the full
      silicon suite, 482/482 on both cards.

### Performance follow-ups (measured, not yet scheduled)

From the training and inference profiles of 2026-10-01 (`ttsim-divergence.md` rows V-Z;
`tt-mnist` and `tt-mnist --infer` print the host-side split, `TT_PROFILE` the device's).
Each names the measurement it must move. The Burn-side ones are in
`burn-backend-parity.md` (B5, B8, B16).

- [ ] **X6 A fast path for the mover's requests.** A GDDR read costs ~0.46 us an entry
      however small (row W after row Y), and a matmul gather is one entry per tile: a
      record should issue its moves straight to the NIU -- validated once per record,
      not re-encoded and decoded per tile -- and write only the NIU registers that
      change between requests (tt-metal's `*_set_state`/`*_with_state` pattern; the
      register persistence to be checked on ttsim and in a gate first). Moves:
      `silicon_perf::mover_read_shapes` 4 KiB entries toward the 16 KiB-entry rate, and
      the gather's share of a step (row V: 0.94 ms of 2.4).
- [x] **X7b Posted-write fence as API** (2026-10-10): `Device::{write_fenced, write32_fenced, l1_write_fenced, eth_write_fenced}` and `FencedWrite` issue the posted writes then exactly one read-back of the dword holding the last byte, refusing misaligned, foreign-window, aperture and non-L1 targets before any write. Gated by `tt-device` unit tests and `step119` (a posted-write transport model: ttsim applies BAR writes synchronously and cannot show the race itself); card 0 `posted_then_fenced_l1_writes_read_back_exactly`; both-card Ethernet gates `silicon_eth_link::{host_driven_tt_link, mover::}` pass with `Mover::stage` and the eth test sites migrated. Hot paths whose observer is started by the host afterwards stay unfenced by design.
- [~] **X7 Host transfers** (DMA and batching, 2026-10-03). Tensors use pinned
      host memory and the card's `HOST_READ`/`HOST_WRITE` DMA; parallel host
      tilize/detilize and run records deliver about 11 GB/s on large Session
      transfers. Gated by `step57_host_dma`; measurements in
      `firmware-performance.md`. The Device-level posted-write fence is now API
      (`write_fenced`, `FencedWrite`; X7 below). The following is the historical motivation:
      **Small host transfers.** A `[64, 10]` upload (2.5 KB, two tiles) costs ~470 us
      a call and a download of the same ~140 us past its sync (rows Z, measurement M:
      uncached 4-byte MMIO reads, and `dram_write`'s per-port read-back on each channel
      a tensor touches). Batch the read-backs per tensor, not per channel write; read
      small tensors with the widest loads the BAR allows. A large write is slow too: a
      traced inference batch's 200 KB input takes 2.1 ms (~95 MB/s, against the WC
      aperture's GB/s; row AG), most of a traced batch. Moves: per-call `upload` and
      `download` in `tt-mnist`'s breakdown. With it, the ordering rule as API (row AA):
      a fenced L1 write -- posted writes, then one read-back -- for every host write
      another agent may race, so a caller cannot forget it.
- [x] **X4d Traces** (above): a replay's host side is one entry a unit; what is left of
      a traced inference batch is the input's write (X7) and the output's download.
- Moved to Burn's roadmap with the numbers: **B8** async calls (a call's server round
  trip is 32-49 us, ~0.2 ms of a 0.61 ms inference batch); **B5/D4** a slice not on a
  tile row (batch 1000 inference: 46 ms a batch, 3 MB re-uploaded each); **B16** the loss
  on the device for a small tensor now that ops do not wait one by one.

### P — Prerequisites pulled in when they block

- [x] **P1 Rank-N tensors** *(close-out 2026-10-10: P1a and P1b are done, ragged/broadcast/strided batches included.)*  (concepts review G2), minimal: a logical shape stored as
      `prod(leading)` stacked tile grids, a batch stride in `TensorRef` (0 = broadcast),
      last-dim-preserving reshapes as views. Blocks R1 over leading dims, R3, D6, R4.
  - [x] **P1a Storage and element-wise.** `burn-tt` stores an F32 tensor of any rank
        as `[product of the leading dims, last dim]` (`tensor::stored_dims`, rank 1 as
        one row); `float_reshape` keeping that matrix is a view; element-wise ops of
        one shape, and broadcasts that are a row or column of the stored matrix and
        give the larger operand's shape by NumPy's rule, run on the device;
        `to_device` uploads any rank. Pulled in because it blocked batching: a
        linear layer's rank-1 bias put four transfers -- and so four syncs -- in
        every MNIST step; the steady step now moves only the logits and their
        gradient (the 9.5 budget, `step12_mnist`). Gate `step35_burn_rank_n`:
        rank 1 and 3, row and column broadcasts, reshape views, against Flex bit
        for bit (a NaN by class) and downloading nothing; `[6, 1, 4] + [1, 6, 1]`,
        a column by the matrices but `[6, 6, 4]` by the rule, still right (watched
        failing without the rule's check). ttsim and both cards.
  - [x] **P1b Batch stride.** *(close-out: ragged, broadcast and strided batches repack on the device through `materialized_batched_matmul`.)*  Batched matmul done (2026-10-03), not by a batch
        stride but by blocks: `tensor::matmul_dram_batched` takes one `(A, B)`
        pair of tile-aligned blocks per batch element -- a `TensorRef` whose
        first tile is the block's, the parent's row stride kept, which GATHER
        already honours -- and writes each product to its own tile rows of one
        output; `tensor::copy_blocks` moves whole tiles (and, with
        `READ_RUN` flag bit 3, transposes them through `READ_TRANSPOSED`)
        into any arrangement. burn-tt keeps rank-N reshapes and dimension
        swaps as strided views of one buffer (`burn-tt/src/views.rs`), so
        attention's head split, `K^T` and their gradients move nothing, and
        the head merge is one block copy. Gates: `step60_batched_blocks`
        (every product bit for bit the 2-D matmul of its block, copies bit
        for bit, ragged and overlapping refusals; watched failing with the
        block offset dropped and with the tiles read untransposed; ttsim and
        both cards), `step59_burn_transformer`. Last-dim sum/max and mean_dim
        compositions, plus certain leading-dim sums, are done. General
        leading-dim reductions are now simulator-gated by R1c. Ragged,
        broadcast and strided F32/BF16 batches are done too: each matrix is
        repacked on the card and run through the 2-D product
        (`materialized_batched_matmul`, `burn-tt/src/ops.rs`; step76, step83).
- [x] **P2 K blocking** *(close-out: release baselines recorded in `firmware-performance.md`; planner default is at or near the best block length.)*  (concepts review G3): native FP32 Dst reload implemented
      for resident ordinary and supported batched matmuls (`step68`). Same
      accumulation order, no block-sum addition or packer L1 accumulation.
      Simulator and both-card gates pass (`1791145571`); release baselines recorded 2026-10-10 in `firmware-performance.md`.
      Host-staged `matmul_chunked` retains its separate arithmetic contract.

### F — SFPU foundation (blocks every S item)

- [x] **F0 Padding is a property of the tensor, not an assumption.** `DramTensor`
      carries `pad: Pad` (`Zero` | `Undefined`; a tensor with no ragged edge is
      always `Zero`), upload sets `Zero`, and every op implements `OpPadding`
      (`requires(input) -> PadNeed`, `produces(inputs) -> Pad`, from the op's
      algebra: `ADD_ROW` and `MUL_SCALAR` by a non-finite scalar leave
      `Undefined`, `RELU_BACKWARD` is `Zero` if either input is). The sum over
      rows reads only a ragged tensor's valid rows (once `COL_SUM` on the mover;
      on the SFPU since 2026-10-02, `sfpu::reduce::accumulate_in_order`) and
      zeroes its result's padding rows, so it needs nothing; the matmul needs `Zero` on both operands, which the session
      supplies by `record::FILL_PAD` over the edge tiles only, in place on a
      tensor that owns its slots and through a bit-exact copy (`Session::copy`,
      `READ_RUN` + `WRITE_RUN`; once `kind::COPY`) for a view, so a parent's padding -- and its views' claims -- are never changed by
      a view's fill. MNIST's tensors need no fill (its ragged ones are uploads and
      matmul outputs), so the golden and the 9.5 budget are unchanged. Unit tests:
      the fill touches each edge tile exactly once with the right valid region,
      over three channel masks and 1/3/8 units; each op's declared pad; the
      masked `SUM` record still expands to its reference builder's entries; the
      compute entry's parameter is refused where a kind takes none or out of
      range. Gates: `step19_eltwise::padding_rows_stay_out_of_a_later_accumulation`
      (un-ignored) and `step24_padding` -- {`ADD_ROW`, `MUL_SCALAR(inf)`, `RELU`}
      into the column sum and into matmuls with a ragged `K` in either operand and
      orientation, dirty on both sides of `K`, at `[37, 70]` and `[50, 40]`,
      against Flex bit for bit; every `pad()` claim checked against the raw tiles
      (`Session::download_padded`); a view of a dirty tensor filled through a copy
      with the parent's tiles untouched. Watched failing with the fills skipped
      (both tests, and step19) and with the sum unmasked. ttsim and both cards.
- [x] **F1 Whole-tile `Dst` round trip** (FP32; BF16 moves to D1, its first user).
      A tile image's 1024 datums unpack as one flat run (`datapath::tile_descriptor`),
      which lays the four faces down sixty-four `Dst` rows in the packer's order;
      `datapath::unpack_tile_to_dst(l1, row)` retargets unpacker 0's base and
      `REG5_Dest_cntx0_address` (`Dst` row = `OutAddr/16 - 4`) between tiles, and
      `pack_tile_from_dst(l1, row)` rewrites the packer's configuration with
      `DEST_TARGET_REG_CFG_PACK_SEC0_Offset` = `row` (`Offset << 4` datums). Gate
      `step25_dst_tile`: two tiles (normals across the exponent range, both zeros,
      both infinities, the extremes) into rows 0 and 64, packed back from 64 first,
      and an SFPU walk -- sixteen row groups, both column halves, `SFPLOAD`/`SFPSTORE`
      -- copying rows 0..64 to 128..192, packed back as the tile: 3072 datums bit for
      bit. Watched failing with the odd half skipped and with the second tile's row
      off by four. ttsim and both cards; no divergence.
- [x] **F2 An SFPU program builder** (`tt_kernels::sfpu::Program`; the typed
      registers in `tt_isa::sfpu`). Gate: `step26_sfpu_isa` (below, with F5).
  - [x] An `LReg` newtype (`tt_isa::sfpu::LReg`): `LReg::general(0..8)` writable,
        `ZERO`/`ONE`/`C0_8373`/`LANE_X2` read-only, `ConfigLReg` 11–14 readable only,
        16 not offered; a write to a non-writable one panics while building.
  - [x] Tile iteration: `Program::for_each_row_group(rows, body)` -- the body written
        once, handed an address offset; replayed through X1 when it fits (row
        counter stepped by address modifier 7 on its last `Dst` access, entry 0 at no
        increment, the counter cleared before and after), unrolled otherwise, and
        `Program::loops` says which.
  - [x] Conditional execution as a scope: `if_`, `if_else` emit `SFPPUSHC`,
        `SFPSETCC`, `SFPCOMPC`, `SFPPOPC` balanced; depth tracked, a ninth level
        refused; only the plain push and pop are ever emitted, so the Tier 2
        `SFPPOPC` case cannot arise (and `SFPPOPC.md` contradicts itself on whether
        Blackhole still has it).
  - [x] `SFPCONFIG` constants (`LReg` 11–14) as a named prologue
        (`Program::constant`, 10.2a): loads `L0` and writes the register, refused
        inside a scope (`SFPCONFIG` takes its value and its predication from lanes
        0..8 alone). The interpreter starts the four unknown, so a program that
        reads one it did not write -- one an earlier program left (G11) -- is
        refused there (`a_constant_is_known_only_to_the_program_that_writes_it`).
  - [x] An `SFPNOP` exactly where `stalls_automatically_after_mad` says stalling
        misses -- after any MAD-sub-unit instruction (`SFPMAD`, `SFPMUL`, `SFPADD`,
        `SFPMULI`, `SFPADDI`, `SFPMUL24`, `SFPLUT`, `SFPLUTFP32`) -- including across
        a replayed body's wrap-around; unit-tested to appear once, in the right
        place.
- [x] **F3 The SFPU tile kernel** (`tt_kernels::sfpu::kernel`). T0 unpacks each tile's
      `A` to `Dst` rows 0..64 and `B` to 64..128 -- or, for a row broadcast, `B`'s row 0
      laid four times per face into rows 64..72 by sub-run unpacks
      (`datapath::unpack_datums_to_dst`, the base moved because an uncompressed unpack
      always starts at datum 0) -- T1 runs the op's program writing rows 128..192, T2
      packs them; three semaphores (`unpacked`, `computed`, `free`), declared through
      `crate::l1` in the order that numbers them as a matmul's are, so the two
      alternate with no setup run; slots planned in the data arena. Unary, binary and
      row-broadcast shapes; binary-with-scalar is a unary program with an immediate.
      Every role carries the state it needs (G11): found on the way, a matmul leaves
      unpacker 0's ADC Z at its last face and the next kernel read the wrong datums --
      the unpack role now clears the ADCs, `unpack_config` always writes descriptor
      words 0 and 1, and every SFPU program starts with address modifier 0 at no
      increment and the `Dst` row counter cleared. Gated through S1 and `step19`.
- [x] **F4 Dispatch without new firmware** for SFPU ops; two new mover records for the
      operands. `tensor::sfpu_eltwise` makes each run of tiles a job: `READ_RUN`
      (operands into consecutive slots, with a row-broadcast flag), `Step::Kernel`,
      `WRITE_RUN` (outputs back) -- the matmul's gather/kernel/scatter shape, since the
      session patches only top-level `KERNEL` entries. Programs memoised by op, scalar
      and run length; the run length is measured from the programs, the longest whose
      role programs fit a slot (`ADD_ROW`'s unrolled loop gets shorter runs -- found by
      MNIST's evaluation batch on one tile, now `step19::the_longest_runs_fit_and_match_flex`,
      watched failing with code 8 without it). Adding an op is a program in
      `sfpu::ops` and a gate. (Retired 2026-10-02: `EltwiseUnit`,
      `tensor::sfpu_is_cheaper` and `TT_ELTWISE` chose between the SFPU and the
      mover's FP32 unit by measurement Q's cost model; the mover does no
      arithmetic now, so every op is the SFPU's, and one with no program is
      refused.)
- [x] **F5 Oracles.** *(close-out: every delivered instruction has an interpreter or page model, including `SFPLOADMACRO` (`macro_sched`).)*  `tt_kernels::sfpu::interp::Vector`: `LReg[17][32]` (a
      register nothing has established is `None`, and reading it is refused), per-lane
      `LaneFlags`, `UseLaneFlagsForLaneEnable` and flag stack, the `Dst` row counter
      and address modifiers, the replay buffer (`REPLAY` expanded by its own model),
      and `Dst`; one functional model per instruction, transcribed from its page, and
      anything without one refused by name. Modelled so far: `SFPLOAD`/`SFPSTORE`
      (FP32, INT32), `SFPLOADI` (every mode), `SFPMAD`/`SFPMUL`/`SFPADD` (through
      `fma_bh`), `SFPMOV`, `SFPABS`, `SFPSETSGN`, `SFPSETCC`, `SFPENCC`,
      `SFPPUSHC`/`SFPPOPC` (plain), `SFPCOMPC`, `SFPGT` (flags, `VD`), `SFPARECIP`, `SFPNOP`, and the `SETRWC`/`SETC16`
      forms the builder emits; since 10.2a also `SFPLE`, `SFPSWAP` (every
      contractual mode), `SFPMULI`, `SFPADDI`, `SFPXOR`, `SFPNOT`, `SFPLZ`,
      `SFPMUL24` (`VC` zero), `SFPCAST` (`_IntFloat`, round to nearest),
      `SFPCONFIG` (`LReg[11..15]`), `SFPLUT` and `SFPLUTFP32` (every table, its
      indirect destination included), and the backdoor-load rule (`VD >= 12`
      refused by name). **Plan change:** each S item adds the models it
      needs, and where a page defines a self-contained C function (`ApproxRecip`,
      `ApproxExp`, the LUT and rounding helpers), the port is differential-tested
      against that C extracted from the pinned page and compiled as `fma.c` is;
      the per-instruction ground truth is `step26_sfpu_isa` on the device.
      Gate `step26_sfpu_isa`: fourteen builder programs (add, sub, mul, mad with a
      two-half immediate, negated mad, a mad into its own operand then read, mov,
      neg, abs, set sign, a BF16 immediate, a relu scope, if-else, three nested
      scopes over every condition) over two tiles of every special (both zeros and
      infinities, NaNs of both signs, denormals, extremes), each replayed and
      unrolled, the device tile equal to the interpreter's bit for bit on ttsim and
      both cards; `LReg[8]` measured (row P). Watched failing with a wrong `SFPABS`
      model (silicon refuses it at the negative-NaN datum).
- [x] **F6 (optional) A Burn coverage generator.** *(delivered 2026-10-09 as `cargo xtask burn-coverage [--check]`, generating `burn-op-coverage.md` and failing on an unsupported method without a disposition.)*  `cargo xtask burn-coverage --check`,
      reading `OVERRIDDEN` and the pinned traits, so the table below cannot rot.

### S — SFPU operations

- [x] **S1 Today's element-wise ops on the SFPU** (was checklist 9.9): `ADD`, `SUB`,
      `MUL`, `MUL_SCALAR`, `RELU`, `RELU_BACKWARD`, `ADD_ROW`, and new `ADD_SCALAR`
      (mover and SFPU) for `float_add_scalar`/`float_sub_scalar` (`x - s` as `x + -s`,
      the same bits). `RELU`'s predicate is the mover's integer test, `+0 < x <= +inf`,
      as two `SFPGT`s in the total order. Oracle: each program equals the mover's
      arithmetic in the interpreter over every special (`sfpu::ops` unit test). Gates:
      `step19_eltwise::every_kind_matches_flex_bit_for_bit` on both units, forced, over
      both zeros, infinities, NaNs, denormal-adjacent values at `[37, 70]`, `[64, 128]`,
      `[784, 128]`; the padding and long-run tests; `step27_burn_eltwise` (every
      overridden element-wise method on resident tensors against Flex, no upload or
      download during the op; in `SMOKE`; watched failing with `sub_scalar`'s sign
      unflipped); the MNIST golden bit for bit at 1 and 4 tiles with the 9.5 budget;
      ttsim and both cards. **Measured** (Q, R): per op the SFPU costs ~26 us + ~3 us a
      tile against the mover's ~9 us + 8-22 us a tile, so small ops spread over many
      units stay on the mover; full MNIST 5.1 -> 3.8 ms/step on one tile, 2.5 -> 2.4 on
      eight, 2.3 on 32, accuracy 91.96%.
- [x] **S2 Compare, select, sign** (10.2c). Twenty-six exact kinds
      (`kind_sfpu::{NEG..PRELU}`), each a program of bit and flag operations on raw
      bits (`Format::Int32` loads and stores), so NaN payloads, both zeros and
      denormals come out as the host has them -- only the products (`LEAKY_RELU`'s
      and `PRELU`'s negative side, `HARD_SIGMOID`) go through `SFPMAD`, two
      roundings as Flex's `alpha * x + beta`. **IEEE comparisons from a
      sign-magnitude order**: `SFPGT`/`SFPLE` rank `-0 < +0` and order NaNs, so
      each operand is first made canonical (a zero `+0`) and a lane with a NaN
      takes the unordered answer (`compare_body`); a scalar's NaN and zero sign
      are settled on the host when the program is built. Flex's choices matched,
      measured where Rust leaves them open: `clamp_min`/`clamp_max` give the
      scalar on equal values (`±0`) and the other side of a NaN at every length;
      `sign` keeps a NaN and gives `+0` for a zero; `clamp` refuses NaN or crossed
      bounds (`f32::clamp` panics; burn-tt hands those to Flex). Two scalars per op
      (`Eltwise::scalar2`, in the memo keys), a ternary operand shape (`MASK_WHERE`:
      `Dst` rows 192..256, `Operands::Ternary`, a third `READ_RUN`;
      `Session::eltwise3`), per-operand element types (`sfpu::ops::Sig`: a mask is
      `Bool`, the comparisons' output too), and padding rules from each op's
      algebra at zero (`Eltwise::zero_at_zero`). `SFPSWAP`'s min/max is gated
      (10.2a) but not used: its order is not IEEE's. Oracle: `sfpu::ops::s2`, the
      programs in the interpreter against the host's semantics over every pairing
      of sixteen specials, a product's denormal operands flushed first (numerics
      row D). Gates: `step43_compare_select` -- each kind against `burn-flex`'s own
      op bit for bit at `[37, 70]` and `[64, 128]` with specials on both sides, row
      and column broadcasts for the comparisons and `mask_fill`, the device equal
      to its program, padding claims against raw tiles; a product's lanes with a
      denormal input are the oracle's (the device decides the branch on the raw
      value and computes on the flushed one, which no Flex run states). Watched
      failing with the zero canonicalisation removed (`-0 == +0` false). ttsim and
      both cards. Burn: `float_{neg, abs, sign, clamp, clamp_min, clamp_max}`, the
      twelve comparisons, `float_is_{nan, inf}`, `float_mask_{fill, where}`,
      `leaky_relu`, `hard_sigmoid`, `prelu` (a row of per-channel slopes; one
      weight falls back), and `float_cast` to the dtype a tensor has (a no-op: Burn's
      `hard_sigmoid` casts to `F32` what is, which downloaded it every step) --
      exact, so on the device whatever the size, exact mode included;
      `step47_burn_activations::compare_select_and_sign_stay_on_the_card` (in
      `SMOKE`; watched failing with `float_sign` routed to `ABS`). MNIST (row AJ):
      leaky-relu 6.8 -> 2.9 ms/step, hard-sigmoid 5.4 -> 2.2, both at ReLU's 3.0 KB
      a step and inference at ReLU's.
- [x] **S3 Reciprocal and division** (`float_remainder{,_scalar}` moves to S6, which
      brings `floor`). `tt_isa::numerics::sfpu::{approx_recip, approx_exp, arecip}` port
      `SFPARECIP.md`'s functional model, the tables copied out of the page by script and
      held to the page's own C -- extracted from the pinned tree and compiled by
      `tt-tests/build.rs` (`sfpu_models_oracle`, every input reaching a table or
      branch; watched failing with one table entry changed). `Program::recip`: the
      `SFPARECIP` seed (`e0 < 0.0056`), two Newton steps in fma form, fix-ups for
      `±0`/denormal (`±inf`) and `±inf` (`±0`) -- within one ulp of the correctly
      rounded reciprocal, the bound derived on the method; division is the product and
      one fma correction on finite non-zero lanes, within one ulp (`ops::divide`).
      SFPU-only kinds (`kind_sfpu::{RECIP, DIV, DIV_SCALAR}`, above the mover's) go to
      the SFPU whatever the unit setting. `sfpu::ops::reference` runs any op's program
      over a whole tensor in the interpreter, the oracle for every later op. Gates:
      unit tests (93% of 3072 results correctly rounded, the rest one ulp off; IEEE
      special cases); `step28_division` -- device equal to the program bit for bit,
      the program within one ulp of Flex, at `[37, 70]` and `[96, 128]` with every
      special; `step27_burn_eltwise::division_through_burn_is_within_one_ulp_and_stays_resident`.
      ttsim and both cards. Burn: `float_recip`, `float_div`, `float_div_scalar`.
      **Fixed in 10.2d** (found by its `log1p` sweep): from `|x| > 2^111` the Newton
      step's product `y (1 - x y)` fell below `2^-126` and flushed, so the reciprocal
      was the seed alone (0.56% off), and from `2^126` the seed itself is zero; a
      division by such a `b` was wrong the same way. Where `|x| > 2^100` the
      reciprocal is now of `x 2^-64`, scaled back by an exact multiply, and a division
      scales both operands by `2^-64` (`scale_large_divisor`; on the host for
      `DIV_SCALAR`'s scalar). Held by `ops::transcendental::{recip_is_within_one_ulp_
      in_every_binade, division_is_within_one_ulp_down_to_the_smallest_quotients}`
      (worst 0.72 and 0.81 ulps) and on the device by
      `step44_algebraic::recip_and_division_hold_in_every_binade` (watched failing
      with the scaling disabled, at `1/7.4e33`).
- [x] **10.2a The instructions the rest of S2-S4 needs** (`tt_isa::numerics::sfpu`,
      `tt_kernels::sfpu::{Program, interp}`). Helpers: `Cond::LessEq` (`SFPLE`),
      `min_max` (`SFPSWAP`), `muli`/`addi` (BF16 immediates, refused otherwise),
      `xor`, `not`, `leading_zeros`, `mul24` (`VC` the zero constant: anything
      else adds the page's non-contractual shift-add), `sm32_to_float`,
      `constant` (F2), `lut`, `lut_fp32` with `LutTable`. **The `SFPLUTFP32`
      hazard designed out**: `FP16_3ENTRY_TABLE` is `Mod1 = 10`, which includes
      `INDIRECT_VD`, so the helper loads `VD`'s index into `L7` first (clobbering
      it) and sets `Mod1Mirror`'s `INDIRECT_VD` to match -- automatic stalling
      reads the mirror, and with it clear it would assume `L7` unread and miss
      the `L7` just written. Oracles: `SignMagIsSmaller`, `Lut8ToFp32`,
      `Lut16ToFp32` held to the pages' own C (now compiled as C++20, since
      `Lut16ToFp32` uses `std::bit_cast`; every LUT code, and pairs across every
      sign and exponent class; watched failing with the FP16 bias off by one);
      `SFPCAST`'s conversion against the host's rounding (400k integers). Gate:
      `step26_sfpu_isa::every_new_instruction_matches_the_interpreter`, 21 cases
      over the specials tiles, replayed and unrolled, device equal to the
      interpreter bit for bit -- among them an `INT32` load and store passing
      denormals and NaN payloads, both tables of `SFPCONFIG` constants, and the
      Tier 2 bug measured: `SFPLUTFP32` at `Mod1 = 10` with `L7 = 5` writes `L5`
      and leaves `VD`. `STEP26_CASE=<name>` runs one case alone, as each new
      instruction was first run on silicon. Watched failing with `SFPMUL24`'s high
      half shifted by 22. ttsim runs 17 of the 21 (row 70: `SFPLUTFP32` only at
      `Mod1` 2 and 6, and no `Mod1Mirror`); silicon all 21, both cards. No program
      depends on `SFPLUTFP32`: polynomials are `SFPMAD`'s, which ttsim runs.
      Simulator line `[-]` for those four (row 70), as the definition of done
      allows; ticked at 10.2's close.
- [x] **S4 Transcendentals.** `exp`, `log` (10.1); the rest in 10.2d-f below. Range
      reduction by `SFPEXEXP`/`SFPSETEXP` and integer exponent arithmetic (`SFPIADD`,
      `SFPSHFT`), polynomials in Horner form by `SFPMAD`. `exp`: magic-number rounding
      of `x log2 e`, Cody-Waite reduction, degree-7 Taylor, `2^n` added to the exponent field;
      bound `ops::EXP_BOUND = 1.3e-7` relative, derived on `exp_program`. `log`: `x =
      2^e m`, `m` in `[sqrt(2)/2, sqrt(2))`, `2 atanh(f/(2+f))` through the corrected
      division, `e ln2` in two parts; bound `ops::LOG_BOUND = 7.12 * 2^-24` relative,
      derived on `log_program`. Oracle: interpreter models of `SFPIADD`, `SFPSHFT`,
      `SFPEXEXP`, `SFPEXMAN`, `SFPSETEXP`, `SFPSETMAN`, `SFPDIVP2`, each held to the
      device in `step26_sfpu_isa`; sweeps of 64k (`exp`, worst 0.86 ulps) and 80k
      (`log`, worst 1.91 ulps, near 1) inputs within the derived bounds; an arity test
      keeping `ops::operands` and `ops::program` in step. Gates: `step29_exp_log`
      (device equal to the program bit for bit; the program within the bound plus
      Flex's ulp of Flex, every special); `step27_burn_eltwise` now also asserts each
      result was *computed on the device* (`TtTensor::computed_on_device` -- a host
      fallback on operands with host copies moves no bytes, so the traffic check
      alone was vacuous; watched failing with `float_exp` forced to the host). ttsim
      and both cards; ttsim refuses `SFPDIVP2` by 128 or more (row 66), so the
      builder does not emit it. Burn: `float_exp`, `float_log`. **`exp` fixed in
      10.2d**: its overflow test was `z > 88.72284`, but that float (`0x42b17218`) is
      the first above `ln f32::MAX`, so at exactly it `n = 128` carried into the
      exponent field and gave `0x7f800002`, a NaN -- met by `pow(f32::MAX, 1)`, whose
      `ln` rounds to it. Now `z >= 88.72284`; the boundary floats on both ends are in
      the sweep (watched failing on the old test).
  - [x] **10.2d `sqrt`, `1/sqrt`, `log1p`, `pow`, and the integer cast.**
        `sqrt_program`: the bit-trick seed, three Newton steps for `1/sqrt` (within
        `RSQRT_BOUND = 4.1u`), the root `x y` and one fma correction (within one ulp
        of the correct rounding; worst 1.44 ulps of the exact root over 85k inputs);
        a negative denormal is NaN as on the host, a positive one flushes.
        `log1p_program`: Kahan's `ln(u) x/(u - 1)`, `x` itself where `fl(1 + x) = 1`
        (bits and all, denormals included), `ln u` alone from `2^24`; within
        `LOG1P_BOUND = 11.12u`, worst 3.0 ulps. `x` is spilled to `Dst` rows 256..
        (`kernel::SPILL_ROW`) across `log_program`, which takes every register.
        **`pow` is one op** (`Session::pow`: `POW`, `POW_S` for a scalar exponent,
        `POW_I` for an `I32` one): `log|x|`, a multiply, `exp`, then `powf`'s special
        values (`pow_program`). First landed as a chain of four ops, since one
        program unrolled to 9187 words, past a role's 8192-word slot; folded back
        once the runner repeats blocks (X8). Within `pow_bound(x, y) = |y ln x| (LOG_BOUND + 2^-24) 1.01 +
        EXP_BOUND`, derived (worst 0.46 of it); every pairing of 22 special bases and
        22 special exponents equal to `powf`'s, signs of zeros and infinities
        included; the integer and odd tests on `y` by the magic-number round. An
        `I32` exponent is converted in the program (`as f32`, exact; `i32::MIN` by
        name; `I32_TO_F32` the same alone); `FILL` writes a constant (Flex's `ones` for `x^0`). Gate
        `step44_algebraic`: each kind bit for bit to their programs,
        the programs within their bounds of Flex, the cast equal to Flex's
        `int_into_float`; ttsim and both cards. Burn: `float_sqrt`, `float_log1p`,
        `float_powf`, `float_powi`, `float_powf_scalar{,_impl}` and
        `float_powi_scalar` with Flex's own dispatch (`0` ones, `1` the tensor, `2`
        a product, `-1`/`-2` reciprocals, else `powf`), `int_into_float` to F32
        (exact); `step47_burn_activations::algebraic_ops_stay_on_the_card_within_
        their_bounds`. `RSQRT` has no Burn method; it waits for R3's norms.
  - [x] **10.2e The exponential family and the activations on it.**
        `expm1_program`: `exp`'s reduction, `p = e^r - 1 = r + r^2 q(r)`, then
        `2 (h p + (h - 1/2))`, `h = 2^(n-1)` -- no cancellation near zero, no
        overflow at `n = 128`; within `EXPM1_BOUND = 4.5u`. `sigmoid_program`:
        Flex's two branches over one `e = e^-|x|`, within `SIGMOID_BOUND = EXP_BOUND
        + 4u`. `tanh_program`: `sign(x) t/(t + 2)`, `t = expm1(2|x|)`, `x` itself
        below `2^-12` and `±1` from 9.01; within `TANH_BOUND = 7.5u`. `erf_program`:
        the Taylor series below `1/2`, then `1 - erfc` with `erfc = e^(-a^2)
        erfcx(a)`, `a^2` split exactly (Dekker) and `erfcx` a Chebyshev fit computed
        in the builder from `libm::erfc` (`ERFC_MID`, deg 16 to 3.92), evaluated by
        Clenshaw; within `ERF_BOUND = 10.5u`, the fit and its evaluation measured
        over every float of the interval. `gelu_program`: `2 Phi` by the branch
        that does not cancel -- on the far negative side `erfc` itself (a second fit,
        `ERFC_TAIL`, to 9.3), so it is relatively accurate where Flex's own `1 +
        erf` is not; within `GELU_BOUND = 12.5u`; `gelu_backward` within
        `gelu_backward_bound(x, g)` (absolute: the derivative crosses zero).
        `sigmoid_backward` is exact, Flex's order of roundings. Found here: `SFPMAD`
        is not fused and drops a denormal-range product (numerics rows E, F), so
        the error-free transforms are Dekker's with 12-bit halves; and `exp`'s NaN
        and overflow edge (10.2d). Gates: `step45_exp_family` (each kind bit for bit
        to its program, the programs within their bounds of Flex), ttsim and both
        cards. Burn: `float_tanh`, `float_erf`, `sigmoid{,_backward}`,
        `gelu{,_backward}` -- `silu` follows, Burn's `x * sigmoid(x)` --
        `step47_burn_activations::exp_family_activations_stay_on_the_card_within_
        their_bounds`, which also takes Burn's autodiff through both backward
        kinds. `sinh_cosh_program`: one `expm1` of `a = |x|`, then `(t + t/e)/2`
        or `(e + 1/e)/2`, `e = t + 1` (no cancellation); from `a = 88` the
        argument halved and the result `w (w/2)`, so nothing overflows before
        the result does (89.4159); `sinh` is `x` itself below `2^-12`. Within
        `SINH_BOUND = COSH_BOUND = 12.5u` (the large side's two `expm1`s;
        worst measured 2.2 ulps). Burn: `float_sinh`, `float_cosh`;
        `step47_burn_activations::hyperbolics_and_log_sigmoid_stay_on_the_card_
        within_their_bounds`, with Burn's autodiff of `sinh` (`g cosh
        x`); watched failing with `float_cosh` routed to `SINH`, and `step45`
        with `sinh`'s sign dropped. `asinh_acosh_program`, `atanh_program`: one
        `log1p` each, of an argument that does not cancel -- `a + a^2/(1 +
        sqrt(1 + a^2))`, `t + sqrt(t (t + 2))` (`t = x - 1`, exact), `2a/(1 - a)`
        -- and from `a = 2^12` `log1p(a - 1) + ln 2`, so `2a` never overflows.
        Within `ASINH_BOUND = ACOSH_BOUND = 16.2u`, `ATANH_BOUND = 14.2u`; worst
        measured 3.1, 3.6, 3.3 ulps. `atanh` beyond 1 is NaN by name: from `|x| ~
        2^126` the reciprocal of `1 - a` flushes and the quotient said `0`
        (found by its sweep). Flex's `atanh` is std's, which loses up to `42x`
        its roundings near `-1` (numerics row G); the gates add that error,
        derived. Burn: `float_{asinh, acosh, atanh}`; `step45` watched failing
        with `ln 2` dropped. `log_sigmoid_program`: Flex's two branches as one,
        `min(x, 0) - log1p(e^-|x|)` (one `exp`, one `log1p`, no cancellation),
        within `LOG_SIGMOID_BOUND = EXP_BOUND + LOG1P_BOUND + u` (14.3u; worst
        measured 3.2 ulps); `log_sigmoid_backward` is Flex's `g sigmoid(-x)` on
        the device's sigmoid, within `SIGMOID_BOUND` and the product -- a
        denormal sigmoid flushes there (numerics row D), where Flex's times `g`
        can be normal. Burn: `log_sigmoid{,_backward}`, with Burn's autodiff
        through it (its `sum` is R1b's, on the host); `step45` watched failing
        with the backward's negation dropped. `softmin` is R2's composition
        on the device's exact `NEG` (Burn's default, the softmax of `-x`), within
        the softmax's bound (`step32_burn_softmax`, both dims, three shapes, and
        on the host below eight tiles; watched failing with the negation
        off). Next: `sin`/`cos` and the rest (10.2f).
  - [x] **10.2f Trigonometry.** `trig_reduce`: Payne and Hanek's reduction
        by `pi/2` in exact fixed point, for every finite `x` -- a 92-bit window
        of `2/pi` cut per lane at the exponent (23-bit limbs, five conditional
        shifts and a funnel by register shifts, `Program::shl_by`), `M W mod
        2^92` from `SFPMUL24`'s halves with integer carries, `q mod 4` and a
        67-bit fraction, converted exactly and multiplied by `pi/2` in Dekker's
        12-bit halves. A double-float sum of float chunks, the first design,
        keeps 48 bits, and the float nearest a multiple of `pi/2` (`16367173
        2^72`, `|g| = 2^-29.86`; every float scanned,
        `transcendental::no_float_reduces_closer_than_the_hardest`) needs
        about 56: so the reduction is integer. `r` within `2^-34`; the table is
        fdlibm's `ipio2` (watched failing at `1.2e35` with bit 145 flipped).
        `sin_cos_program`: both Taylor cores on `|r| <= pi/4` (`r_lo` carried),
        the quadrant as sign bits; within `SIN_BOUND = COS_BOUND = 2.2u`
        (worst measured 0.97 ulps over every binade, a few periods and the
        twelve hardest reductions with their neighbours); `sin` is `x` itself
        below `2^-12`. Gates: `step46_trig` (device bit for bit to the program,
        the program within its bound of Flex, every binade and the hardest
        reductions, the zero-padding claim on the raw tiles; watched failing
        with the quadrant's sign bit taken from bit 0), ttsim and both cards.
        Burn: `float_sin`, `float_cos`;
        `step47_burn_activations::trig_stays_on_the_card_within_their_bounds`,
        with Burn's autodiff through both (each the other's kind); watched
        failing with `float_cos` routed to `SIN`. Cost (row AL): 23 us a tile
        against `exp`'s 9.2 and `gelu`'s 29.5. `tan_program`: the same
        reduction and cores, `S/C` or `-C/S` by `q`'s parity (`recip`,
        `divide`), within `TAN_BOUND = 5.8u` -- relative next to the poles
        too, as `r` is (worst measured 2.15 ulps); `x` itself below `2^-12`.
        Burn: `float_tan`, with Burn's autodiff (`g (tan^2 + 1)`, `tan`
        recomputed); `step46` watched failing with the odd quadrants'
        negation dropped. `atan_program`: `t G(t^2)` on `[0, 1]`, `G =
        atan(sqrt s)/sqrt s` a Chebyshev fit (`ATAN_FIT`, degree 13; fit
        0.11u, evaluation 1.33u over every 64th float of `[0, 1)`), and `pi/2
        - atan(1/a)` beyond; within `ATAN_BOUND = 6.2u` (worst measured 1.82
        ulps). `Piece` now carries its function (`Fit`), so every fit is
        computed in the builder from `libm`, and the erfc pieces are unchanged
        (`every_fit_and_its_evaluation_are_within_their_parts`).
        `atan2_program`: the core of `min/max` of the magnitudes (both scaled
        by `2^-64` above `2^100`, as `DIV`), then `pi/2 - v`, `pi - v` and
        `y`'s sign; IEEE's special values over every pairing of 22 signed
        specials; within `ATAN2_BOUND = 6.2u` (worst measured 2.75u over 60k
        pairs). A denormal operand is a zero of its sign (numerics row D), so
        `atan2` of two denormals is a zero's where the host's is their ratio's.
        Same-shape operands only: a row broadcast is an unrolled program, and
        `atan2`'s body 32 times over would not fit a slot -- and burn-tt
        refuses a broadcast `atan2` (it panics, naming the shapes) rather than
        sending it to Flex, which is to go
        (`step47::trig_stays_on_the_card_within_their_bounds`, watched failing
        with the refusal removed). Burn: `float_atan`, `float_atan2`, with
        Burn's autodiff of both (compositions on ops already on the card);
        `step46` watched failing with `atan`'s `pi/2 - v` dropped.
        `asin_acos_program`: one core of `a` up to 0.7, else of `z =
        sqrt((1 - a)/2)` (`1 - a` exact), as `asin t = t + t^3 K(t^2)` --
        `K` a Chebyshev fit of the correction (`ASIN_FIT`, degree 16: fitting
        `asin t/t` itself cost an ulp of the binade above 1, 0.83u that no
        degree removed) -- then `pi/2 - 2v`, `2v`, `pi - 2v` by branch and
        sign. The switch at 0.7, not 1/2, keeps `2v` below the result it is
        taken from. Within `ASIN_BOUND = 6.8u`, `ACOS_BOUND = 4.7u` (worst
        measured 1.39 and 1.13 ulps); beyond 1 NaN by name. Burn:
        `float_asin`, `float_acos`, with Burn's autodiff (`g/sqrt(1 - x^2)`);
        `step46` watched failing with `acos`'s `pi - 2v` dropped. Cost per
        tile (row AL): `atan` 15.0 us, `asin` 17.8, `acos` 17.0, `tan` 28.0.
- [x] **S5 Integer ALU on INT32** *(close-out: wrapping and checked I32 arithmetic, comparisons, shifts, division/remainder, scans, arg-extremes, masks and abs are all native.)*  (format code 8, measured): `SFPIADD`, `SFPMUL24`,
      `SFPAND`/`SFPOR`/`SFPXOR`/`SFPNOT`, `SFPSHFT`, `SFPLZ`. The first `IntTensorOps` on
      the device: `int_{add,sub,mul}{,_scalar}`, comparisons, `bitwise_*`, shifts.
- [x] **S6 Casts and rounding.** *(close-out: rounding, saturating I32, BF16 and FP16 casts are native; `int_cast` to other widths is `[-]` because the device stores I32 only.)*  Deterministic F32 rounding and saturating
      I32 conversion are simulator-gated by `step72`; I32-to-F32 already runs
      natively. Hardware SFPSTOCHRND modes are gated by step81; BF16 and BFP8/4/2 storage are delivered. FP16 and other deferred storage formats remain open.
      Historical intended instruction coverage: `SFPCAST` int ↔ float (never `SFPCAST_IntAbs`: Tier 2,
      use `SFPABS`); `SFPSTOCHRND` FP32 → BF16/FP16 in round-to-nearest and stochastic
      modes, matching the documented (biased) behaviour rather than "fixing" it. Burn:
      `float_cast`, `float_into_int`, `int_into_float`, `float_round`, `float_floor`,
      `float_ceil`, `float_trunc`.
- [x] **S7 The PRNG** (2026-10-10, steps 140-142, lane T6; cards 0 and 1): a host model of the per-lane stream (`advance^(98-2i)(seed)` on silicon, measured for 8 seeds x 32 lanes on both cards; ttsim's `advance^(96-2i)`), a per-tile seed directive honoured by the role firmware (RISC-V store + fence + 512 NOPs), a decorrelated generator (33 reads per slot plus a bijective ARX mixer), native Bernoulli, uniform, normal (Box-Muller over the S4 programs) and `int_random` (multiply-shift, range at most 2^23) with derived bounds, resident Burn `float_random`/`int_random`, dropout composed with no uploads; random inside a trace is refused (replay would repeat the seeds). Entropy is 32 bits per tile and the mixer is an empirical decorrelator, not a statistical-suite generator. Earlier text follows.
  ** Step91 checks the seed-register write, advance and predication. WRCFG does not restart silicon's stream; direct RISC-V configuration stores with a fence and 512 NOP iterations do restart it on both cards. Simulator lane initialization has a one-lane offset. Seeded application stream semantics and quality remain unresolved. Planned: seeded per tile from `Backend::seed`. The claim is distributional
      (a stated statistical test), not bit-exact against Flex, whose generator is
      different. Burn: `float_random`, dropout.
- [x] **S8 Lane movement.** `Program::rotate_row` (`SFPSHFT2_MOD1_SUBVEC_SHFLROR1`),
      `Program::transpose4` (`SFPTRANSP`), with `Program::and`/`or` (`SFPAND`, `SFPOR`)
      for lane masks; interpreter models of all four held to silicon in
      `step26_sfpu_isa` (a three-step rotation, a full row reduction by rotations, a
      transpose then add). First user: R1's in-tile folds.
- [x] **S10 A fast, approximate mode for the transcendentals** (2026-10-10, step145-146, lane T8): `MathMode::{Precise, Approx}` (`TT_MATH=approx`, `Session::set_math_mode`, `TtDevice::set_math_mode`); Approx twins (kinds 0x1d0-0x1d5) for `exp`, `log`, `recip`, `sigmoid`, `tanh`, `gelu` with derived bounds (exp 8.2e-5, log 5.2e-5, recip 3.2e-5, sigmoid 1.13e-4, tanh 5.2e-5 relative; gelu |x|(2.5e-4) absolute), device programs equal the interpreter bit for bit on card 0, mode is in the program cache key. Measured per tile on card 0: exp 8.08 -> 6.51 us (1.24x), log 10.96 -> 7.97 (1.37x), recip 7.82 -> 0.95 (8.24x), sigmoid 12.18 -> 7.40 (1.65x), tanh 13.09 -> 8.10 (1.62x), gelu 28.46 -> 7.31 (3.89x) (`silicon_perf::approx_per_tile_saving`). `sqrt`/`rsqrt` measured and dropped (the saving is not real); fused softmax/norms and backward ops stay Precise. Original text follows. (asked for during 10.2e;
      concepts review G10's `math_approx_mode`). Today every S3/S4 op is built for a
      derived bound of a few ulps (`EXP_BOUND`, `ERF_BOUND`, ...), and that costs
      instructions on the device -- degree-16 Chebyshev fits evaluated by Clenshaw,
      two Newton steps, exactness fix-ups, every special-value scope (`pow`'s seven).
      (The fits' coefficients are computed once per process and baked in as
      immediates, so the cost is device time, not host time.) Most training does not
      need that. Add `MathMode::{Precise, Approx}` -- named apart from exact mode
      (`burn_tt::set_exact`, Flex's bits), which is a different question -- carried
      in `Eltwise` and the program memo keys, chosen per op or per session
      (`TT_MATH=approx`), `Precise` the default. `Approx` programs: hard-coded
      low-degree minimax polynomials or `SFPLUT`/`SFPLUTFP32` tables (10.2a gated
      them), one Newton step or none (`SFPARECIP`'s seed is 0.56%), a coarser range
      reduction, special values only where an ML input meets them (NaN, ±inf, ±0).
      Each still gets a derived bound (looser, stated: e.g. `exp` to 2^-11
      relative), a sweep and a device gate, and `silicon_perf` measures what each
      saves per tile against `Precise`. Burn: the mode on `TtDevice`/config;
      `accuracy()` gains the mode so exact mode still refuses both.
- [x] **S9 `SFPLOADMACRO`.** Helpers, schedule model and silicon probes s00-s13 pass on card 0 (step110); ttsim cannot execute it (row 7). No SFPU default changes; performance adoption not claimed.
      else here works without it.

### M — Matrix Unit beyond `MVMUL`

- [x] **M1 `ELWADD`/`ELWSUB`/`ELWMUL`** with `Src` row, column and scalar broadcast: binary
      element-wise at matrix-unit throughput, at TF32/BF16 `Src` precision. Opt-in, like
      `Fidelity`; never a silent replacement for the FP32 SFPU path. Typed consumers, explicit Session APIs, immutable single-card Burn opt-in,
      packed BF16 inputs/F32 accumulation, resident geometry, traces, padding,
      instruction-stream audit, independent bounds/mutants and release benchmarks
      are implemented (`step90`). Final both-card targeted gates pass `1791255922`
      (24/24), full smoke passes `1791255409` (322/322), and release medians are
      recorded in `1791255969`. Mesh mode is explicitly refused.
- [-] **M2 `GMPOOL`/`GAPOOL`.** *Closed 2026-10-10: GAPOOL averages route (BF16 NCHW); GMPOOL is diagnostic-only because its packed ArgMax returns no index bits on both cards (runs `1791240702`, `1791240373`); max pooling stays on the exact SFPU selection (step78).* Original text: Block max/sum/mean kernels gated on simulator and
      both cards (`step75`). BF16 NCHW average/adaptive pooling uses GAPOOL;
      F32/general max retains SFPU semantics, including resident indices and
      overlapping backwards (`step78`). General GMPOOL routing and performance
      remain open. Pooling geometry uses replayable metadata descriptors, so pooling
      traces replay (step78, step89).
- [-] **M3 Transpose on the Tensix** *(closed 2026-10-10: the payload-preserving transpose is the mover's `READ_TRANSPOSED`/repack; `TRNSPSRCB` remains the explicit TF32/BF16 Src route that normalizes signed zeros and subnormals (step73, step87); the unpacker transpose mode swaps rows and columns correctly (step113, card 0) but goes through the 19-bit SrcA datum, and the plain SrcA path itself normalizes -0, subnormals and low mantissa bits (`MEASURE src.plain`), so no Tensix route preserves payloads)* (`TRNSPSRCB`, or the unpacker's transpose mode) in
      place of the B core's face transpose (`READ_TRANSPOSED`). Materialised
      transposes remain partial: step87 gates the Src permutation and transposed TF32 products on both cards, but normalizes signed zero/subnormal payloads. `float_permute` creates native strided views
      through dimension swaps, without a new transpose kernel (`step66`).
- [x] **M4 `SHIFTXB`.** Checked rotate/zero-fill, physical row wrapping and
      modifier effects pass step103 card 0 (`1791397029`); silicon-only (row 50),
      with host permutation coverage. No automatic dispatch change. `DOTPV` and
      `SHIFTXA` are deliberately excluded by the 2026-10-07 review above.

### R — Reductions and composites

- [x] **R1 Reductions over any dim.** *(close-out: also Flex-order cummin/cummax, integer scans, I32 arg-extremes and the device sort family.)*  Done (R1a): `sum` and `max` over either dim of
      a matrix (`Session::reduce`, `sfpu::reduce`): a reduce kernel -- many input tiles
      into one output tile, four semaphores numbered compatibly with the matmul's and
      the element-wise kernel's -- that accumulates lanewise in `Dst`, masks a ragged
      edge's padding lanes to the identity by lane index (`LReg[15]`, `SFPAND`,
      `SFPSHFT`), and folds within the tile by rotations (over columns, the result
      replicated across the row: broadcast-ready) or `SFPTRANSP` (over rows); the
      gather reads a tile column in column-major order (`READ_RUN` flag bit 2). Max is
      exact (total order: a positive NaN propagates, a negative one is ordered below
      `-inf`); a sum over columns is in tree order, within `2 (n-1) u sum|x|` of
      Flex's; a sum over rows stays on the SFPU, in Flex's order. Oracle: the same
      programs in the interpreter (`reduce::reference`), exact on integer data for
      every shape. Gates: `step31_reduce` (device equal to the program bit for bit;
      max equal to Flex's, sums within the bound; ragged shapes, lines of up to 32
      tiles), `step32_burn_softmax`. Burn: `float_max_dim`, `float_sum_dim` (both
      dims). Full F32 `sum`/`mean` are now resident (R1b): bounded column
      chunks and the existing chunked row sum remove their one-pass L1 limit;
      means use the logical count, and full-source views require no copy.
      `mean_dim` already composes sum_dim and scaling on supported axes.
      `step63_burn_full_reduce` checks the composed program models bit for bit,
      a derived addition-order/division bound against Flex, ragged padding,
      special values, views, native execution in exact mode, host-input uploads,
      autodiff and trace replay. Unsupported inputs fail explicitly; neither
      full reduction delegates arithmetic to Flex.
      Native `argmax`/`argmin` now support rank-one/two F32, I32 indices and
      reduced axes up to 2^23, preserving first ties and first NaNs (`step65`,
      `step66`). Argmin negates then shares argmax's selection. Boolean-to-F32
      conversion unlocks Burn's default float `any`/`all`, full and along
      supported sum axes; `step66` checks truth reductions without host compute.
      Burn defaults also compose `max`, `max_abs*` and the minimum family;
      minima inherit gather's axis and signed-zero limitations, and flattened
      full maxima retain reshape/layout and reduction-size limits.
      R1c adds all F32 axes and ragged view repacking, plus long column
      sums and maxima over either axis (`step67`; both-card validated).
      Simulator and both-card gates (`step69`) cover direct product, native Boolean
      `any`/`all`, rank-N arg-reductions and inclusive cumsum/cumprod.
      R1d (2026-10-09; step130-132, simulator and card 0 `1791560292`/`296`/`300`):
      `float_cummin`/`float_cummax` run `ScanOp::{MinNan,MaxNan}` with Flex's
      `is_nan() || val < acc` rule (a NaN replaces the accumulator and stays until a
      later NaN; the earlier element is kept on equal values including ±0); the
      total-order `Min`/`Max` remain for kernel users. `int_cumsum`/`int_cumprod` wrap
      modulo 2^32, `int_cummin`/`int_cummax` are signed, and `int_argmax`/`int_argmin`
      return the first index of the signed extreme by exact integer comparison, so
      Burn's `int_{max,min}_dim_with_indices` compose over them. Full maxima compose native reshape and
      max reductions; minimum defaults retain their gather limitations.
- [x] **R2's groundwork: broadcasts.** *(close-out: done, see R2.)*  `sfpu::ops::Broadcast::{None, Row, Col}` for
      `ADD`, `SUB`, `MUL`, `DIV` (`ADD_ROW` is now `ADD` with a row broadcast): a row
      laid into `Dst` by sub-run unpacks, a column made into a whole tile by the mover
      (`tt_isa::dm::op::READ_BROADCAST_COL`, `READ_RUN` flag bit 1, `Transform` on
      the decoded entry). The session reads the broadcast from the shapes
      (`tensor::broadcast_of`) and sends to the SFPU whatever the mover cannot do;
      padding rules know a broadcast lands in the padding along its dimension.
      Burn's element-wise ops take a broadcast operand on either side where the op
      commutes. Gates: `step30_broadcast` (row and column, the four kinds, `[37, 70]`
      and `[64, 96]`; Flex bit for bit, `DIV` within one ulp; device equal to the
      program; padding claims checked against raw tiles) and `step27_burn_eltwise`'s
      broadcast cases; ttsim and both cards.
- [x] **R2 Softmax, log-softmax and resident loss compositions** (D4 indexing and R1b full mean). `softmax`
      and `log_softmax` (either dim of a resident matrix) run Burn's own composition
      -- max, broadcast subtract, `exp`, sum, broadcast divide or `log` and subtract --
      on the device end to end, decided once on the whole input; `softmin`
      (10.2e) the same on an exact `NEG`. Against Flex's fused
      softmax: a bound derived from the parts' (`EXP_BOUND` twice, the sum's order, the
      division's ulp, Flex's own counterparts); measured worst `1.0e-6` relative.
      Burn's `CrossEntropyLoss` now gathers its target column on the card
      (D4), and its final full mean is resident (R1b). The small-tensor
      placement thresholds (`APPROX_MIN_TILES`, `SOFTMAX_DEVICE_MIN_TILES`)
      were removed 2026-10-03: approximate operations follow resident data
      at every size. The old placement measurements remain in
      `firmware-performance.md`'s change log.
      **Exact mode (retired):** `burn_tt::set_exact`, `TT_EXACT` and Flex
      placement no longer exist; every op is native within its derived bound, and
      full `sum`/`mean` always used native arithmetic. Gates: `step32_burn_softmax`
      (every step resident, both dims, three shapes; a two-tile tensor on the host and
      bit-identical); the MNIST golden in exact mode. Was: **R2 Softmax, log-softmax,
      cross-entropy on the device** (was checklist 9.12): max,
      subtract, `exp`, sum, reciprocal. General ops gated against Flex; MNIST's
      per-step logits download goes away as a consequence, not as the goal. Burn:
      `softmax`, `log_softmax`, `softmin`.
- [x] **R3 Norms.** *(close-out: release baselines recorded; fusion stays deliberately unbuilt.)*  Burn LayerNorm and RMSNorm already compose native primitives.
      Dedicated forward/backward, layout and residency gates are in `step70`;
      both-card silicon passed (`1791145571`); BF16 derivatives pass `step76`.
      Release baselines recorded 2026-10-10 in `firmware-performance.md`. Fusion is deferred.
- [x] **R4d Mesh trace capture/replay** (2026-10-10, step147): `MeshEngine` captures one session trace per chip between the host-run Ethernet transfers and replays them in order (sync both chips, re-run the transfer, continue), refusing with `UnheldTransfer` any capture whose transfer endpoints no stored trace holds. Changed-input replays of a distributed product chain and an attention forward equal fresh runs bit for bit with no host uploads; card 0/1 gates pass. Mesh training traces and `copy_into` on a mesh remain unsupported.
- [x] **R4 `ModuleOps::attention`.** *(close-out: mesh trace capture/replay is supported, see R4d.)*  Native QKᵀ, scale, positive softcap,
      bottom-right causal/broadcast masks, bias, NaN-safe softmax and V products.
      F32/BF16 forward, Q/K/V/bias gradients, resident training, large-K permuted
      views and one/two-tile replay pass step83. Actual two-card products and
      all gradients match single-card references (step88, `1791249486`), with
      nonzero Q/K gradients on both partitions. Mesh trace capture and replay: ttsim step147 and card 0/1 gates pass (R4d, below).


### D — Formats and data movement

- [x] **D1 BF16 and FP16 tensors in GDDR.** *Close-out 2026-10-10: FP16 follows BF16's raw two-byte path with exact SFPU widening/narrowing (step143-144, card 0, all 65,536 patterns); BF16 into Dst measured (step114); F64 and F16 arithmetic are explicit `[-]`.* Separate `Bf16Tensor`/2112-byte physical slots,
      raw transfers/views/repack, native ties-even pack and SrcA/MOVA2D widening,
      packed rank-two MMA and native Burn compute adapters. `step74`, `step76`,
      `step77` pass both cards; narrowing is silicon-only (ttsim mode `0x105`
      refusal). BF16 operands use half the bytes; most arithmetic still widens
      to F32. Compact packed gathers, K continuations, batched/view products,
      actual two-card mesh execution and full MNIST accuracy are now gated.
      Mesh transport still widens to F32. Measured MNIST GEMMs are slower, not accelerated
      relative to TF32.
- [x] **D2 Per-tensor block float (BFP8/BFP4/BFP2), delivered formats only.** Step92/94 pin physical encodings with independent f64/integer oracles, exponent groups and sub-byte order; step93 independently measures histogram reset. Resident Session conversions/readback/free, direct packed products and native Burn storage propagation, F32 training state, fusion and traces pass both cards. Full SMOKE 364/364 (`1791300502`); final conversion/control/propagation gates 28/28 (`1791301208`); release benchmarks `1791300393`. BFP2 matmul and histogram-stat probing have silicon-only arms with documented simulator refusals. Transposed/mixed/batched products widen on device through the existing scheduler. This is a backend storage extension: BFP `a` variants, INT8/UINT8, packed mesh transport and portable `QTensorOps` remain deferred. See [mixed BFP storage](mixed-bfp-storage.md).
- [x] **D3 Integer and bool storage** (10.2b; was INT32, INT8, bool as a format).
      `tensor::Elem::{F32, I32, Bool}` on every `DramTensor`. The device moves
      every 32-bit pattern unchanged -- the FP32-coded unpack to `Dst` and pack back
      (`step26`'s `INT32` pass-through case) -- so no unpacker or packer is
      reconfigured: the tag says what an op may compute on. `I32` is the host's
      two's complement as bits (the unpacker's `INT32` is sign-magnitude, but
      nothing converts between formats, `SFPIADD` is two's complement, and
      `i32::MIN` survives); `Bool` is `0`/`1` as an integer, never `1.0`, and an
      upload of anything else is refused. `Session::{upload_bits,
      download_bits}`; an FP32 `download` or `write` of another type is refused.
      Every op computing in FP32 refuses another type by `TensorError::Elem`
      before choosing a unit or filling padding (`sfpu::ops::elems`, in
      `broadcast_of`; the matmul, the sums, the reductions); `COPY` moves any.
      `sfpu::ops::accuracy` (`Exact` / `Approximate`) replaces burn-tt's "not a
      mover kind is an approximation". Logic ops pulled forward from S5:
      `kind_sfpu::BOOL_{NOT, AND, OR, XOR}` on raw bits (`Format::Int32` loads and
      stores, since an FP32 store flushes the denormal `1`), with row and column
      broadcasts and padding rules (`false && b` is false). Gates:
      `step42_int_bool_storage` -- round trips at ragged shapes and through a view
      (every sign, the extremes, bits that are FP32 denormals and NaNs), every
      refusal, the logic ops against the truth tables and their programs with
      every broadcast, padding claims against raw tiles; watched failing with
      `BOOL_NOT` storing as FP32. ttsim and both cards. Burn: `TtTensor`'s cell
      carries its dtype onto the device (`tensor::device_elem`: `F32`, `I32` --
      Burn's `IntElem` here -- and a bool of any store); `{int,bool}_{to_device,
      reshape, slice, swap_dims, transpose}` keep a device copy as `float_`'s do
      (shared helpers `reshaped`, `swapped_view`, `row_view`), `bool_{not, and,
      or, xor}` run on it; other int dtypes stay on the host.
      `step66` adds native Boolean equality (`NOT(XOR)`; scalar true is an
      identity and scalar false a not), exact Boolean-to-F32/I32 conversion,
      and `{int,bool}_expand` over the dtype-generic native copier. Conversions
      allocate typed output buffers: physical element types are checked by the
      engine. Unsupported output dtypes fail with input metadata. The gate
      covers all three Boolean stores, ragged shapes and row/column broadcasts;
      ttsim and both cards, in `SMOKE`. Watched failing with Boolean-to-F32's
      integer conversion omitted (divergence log measurement AN).
      `step47_burn_activations::integers_and_booleans_stay_on_the_card` against
      Flex, nothing downloaded, `computed_on_device` (watched failing with
      `bool_and` routed to the host); in `SMOKE`. MNIST unchanged (labels stay
      host values). Element-wise ops copy matrix transposes natively;
      arbitrary strided views retain the whole-tile materialization limits.
  - [-] **D3b INT8/UINT8 codes.** Deferred to D2: nothing would use an 8-bit device
        format yet (Burn's int is `i32`, bools ride INT32), and the codes are best
        measured beside the block-float ones `QTensorOps` needs.
- [x] **D4 Indexing on the B mover.** *(close-out: ND gather/scatter are native (step127); scatter_nd Mul/Min/Max are `[-]`.)*  Current contract (2026-10-06): resident
      arbitrary-axis multi-index gather/select preserves raw F32/BF16/I32/Bool
      bits; B validates logical indices before accesses. Duplicate scatter and
      select updates retain logical order: F32 accumulation with final BF16
      narrowing, wrapping I32 addition and Boolean OR. Embedding and backward
      compose these primitives without index downloads. Stepped slice assignment,
      cat/repeat, expand and flip are native. Steps84/86 cover payloads, domains,
      repeated indices, device-produced indices and changed-input trace replay;
      both cards pass (`1791249159`). Historical step61/62 one-index/row routes
      are superseded by this general geometry. ND gather/scatter remains open.
- [-] **D5 Tilize and untilize on the device** *(closed 2026-10-10: unpacker tileize is a payload-preserving strided row gather on card 0 (step113) but reads 32-datum rows, so a face-ordered tile needs extra passes and no benefit over host or mover tilize is shown; host tilize stays the default, mover `TILIZE` opt-in)* (overlaps checklist 9.10).
      Mover `TILIZE`/`UNTILIZE`, `Session::set_tilize` and `TT_TILIZE=card`
      are implemented and gated (`step58_tile_layout`), but slower than
      host tilize at the measured sizes. Host tilize remains the default.
      Unpacker/Tensix tilize and direct reads from caller memory remain open.
- [x] **D6 Convolution.** *(close-out: conv3d, transposed conv3d and deformable convolution are `[-]` with an explicit failure.)*  Bounded native im2col/matmul and deterministic col2im
      implement grouped/depthwise conv2d, conv1d, transposed 1D/2D and unfold4d,
      including all gradients and dilation/output padding. F32/BF16 uses F32
      continuation and final narrowing. Step85 covers analytic/Flex oracles,
      large K, ragged/permuted views, resident training and one/two-tile traces;
      step88 matches all distributed gradients to single-card references
      (`1791249486`). Conv3D/deformable convolution remain excluded.

---

## Burn op coverage

The compute methods of the pinned Burn 0.21 op traits, with the item that brings each to
the device. Bookkeeping methods (`*_device`, `*_to_device`, `*_into_data`, `*_from_data`,
`*_reshape`, autodiff flags) are left out. **Device** is `x` when the method has a device
path today, `~` when only some shapes do.

### `FloatTensorOps`

| Methods | Device | Item |
|---|:-:|---|
| `float_matmul` | `~` F32 resident: 2-D; rank-N against an unbatched rhs folded to 2-D; direct aligned batches or native ragged materialization (views/broadcast included); new ragged F32 path simulator-gated | P1b |
| `float_add`, `float_sub`, `float_mul` (incl. row and column broadcasts; any rank, P1a), `float_mul_scalar` | x (SFPU) | S1, P1a |
| `float_sum_dim` | all F32 axes/layouts on SFPU; both-card validated; BF16 via native adapters | R1 |
| `float_mean_dim` | `~` native sum_dim plus scaling on supported axes | R1 |
| `float_sum`, `float_mean` | x native nonempty F32, bounded full reductions; uploads host inputs,  unsupported inputs fail | R1b |
| `float_slice`, `float_flip` | x nonempty resident F32/BF16, arbitrary steps/axes via bit-preserving copies; both-card validated | D4 |
| `float_transpose`, `float_swap_dims`, `float_permute` | native rank-N views and device materialization; BFP shares exponent-preserving views, regrouping returns decoded F32; arbitrary payload-preserving Src transpose remains partial | M3, D2 |
| `float_add_scalar`, `float_sub_scalar` | x (SFPU) | S1 |
| `float_div{,_scalar}`, `float_recip` | x (SFPU, within 1 ulp) | S3 |
| `float_remainder{,_scalar}` | x exact SFPU fmod (Flex's `((a%b)+b)%b` bit for bit; NaN payload canonical); 2362 words, long division on integers; card 0 (step133-134) | S6 (T4) |
| `float_neg`, `float_abs`, `float_sign`, `float_clamp{,_min,_max}` | x (SFPU, exact) | S2 |
| comparisons (`float_equal`.. `float_lower_equal_elem`), `float_mask_where`, `float_mask_fill`, `float_is_nan`, `float_is_inf` | x (SFPU, exact; `Bool` results resident) | S2 |
| `float_cast` | x same dtype or native F32↔BF16; other reduced floats refused | D1, S6 |
| `float_exp`, `float_log` | x (SFPU, derived bounds) | S4 |
| `float_log1p`, `float_sqrt`, `float_powf*`, `float_powi*` | x (SFPU, derived bounds; `pow` one op) | S4 |
| `float_erf`, `float_tanh`, `float_sinh`, `float_cosh`, `float_asinh`, `float_acosh`, `float_atanh` | x (SFPU, derived bounds) | S4 |
| `float_sin`, `float_cos`, `float_tan` | x (SFPU, derived bounds, every finite input) | S4 (10.2f) |
| `float_atan`, `float_asin`, `float_acos`, `float_atan2` | x (SFPU, derived bounds; `atan2` same-shape operands only, a broadcast refused) | S4 (10.2f) |
| `float_round`, `float_floor`, `float_ceil`, `float_trunc`, `float_into_int` | x F32 raw-bit rounding and saturating I32 conversion; both-card validated | S6 |
| `float_random`, `int_random` | x native seeded resident draws (Default, Uniform, Bernoulli, Normal); host construction only for a device with no GDDR; F16/F64 fail explicitly; refused inside a trace; cards 0 and 1 (step142) | S7 |
| `float_max_dim` | all F32 axes/layouts on SFPU; both-card validated | R1 |
| `float_argmax`, `float_argmin` | x resident rank-N F32, I32 output, first tie/NaN, axis up to 2^23; both-card validated | R1 |
| `float_any*`, `float_all*` | `~` Burn defaults over native comparisons, Boolean-to-F32 and supported sum axes/full sums | R1 |
| `float_max`, `float_max_abs*` | `~` Burn defaults over reshape/abs/max_dim; existing layout and size limits | R1 |
| `float_min*` | `~` Burn defaults over argmin/gather; existing gather axes and signed-zero limits | R1 |
| `float_prod{,_dim}` | x direct SFPU products on all resident F32 axes; both-card validated | R1 |
| `float_cumsum`, `float_cumprod` | x inclusive logical-order resident F32 scans on all axes; both-card validated | R1 |
| `float_cummin`, `float_cummax` | x resident Flex-order scans (NaN propagates, earlier equal element kept); both-card gates pending, card 0 passes (step130) | R1 |
| `float_sort*`, `float_argsort`, `float_topk`, `float_argtopk`, and the `int_` equivalents | x resident bitonic sort, F32/I32, any rank and axis up to 1024, stable (ties by original index), F32 in `total_cmp` order, trace-replayable; BF16 values widen, sort and narrow (flushing BF16 subnormals); longer axes, other dtypes and empty inputs refused by name; card 0 (step135-139, runs `1791561979`-`1791567677`) | R1 (T5) |
| `float_gather`, `float_scatter_add` | `~` resident arbitrary-axis multi-index raw gather; deterministic duplicate F32/BF16 additions, step86 | D4 |
| `float_select`, `float_select_add` | `~` arbitrary-axis resident indices and ordered additions, step86 | D4 |
| `float_expand`, `int_expand`, `bool_expand` | x nonempty stored dtypes; native byte-preserving gathers/transposes | D4 |
| `float_slice_assign`, `float_cat`, `float_repeat_dim`, `float_unfold` | `~` native raw copies/compositions, step84/85 | D4, M3 |
| `float_gather_nd`, `float_scatter_nd`, `int_gather_nd`, `int_scatter_nd` | x resident F32/I32 (BF16 gather_nd); every coordinate checked against its own axis (DOMAIN 11); scatter Add folds duplicates in index order, Assign is last writer; Mul/Min/Max `[-]` (Burn leaves duplicates undefined); card 0 (step127, runs `1791567843`-`1791567861`) | D4 (T2) |
| `float_cross` | x composition of slice/mul/sub/cat, device-rounding oracle; card 0 (step128) | D4 (T2) |
| `float_grid_sample_2d` | `[-]` out of scope: not planned until a model needs it | -- |

### `ActivationOps`

| Methods | Device | Item |
|---|:-:|---|
| `relu`, `relu_backward` | x (SFPU) | S1 |
| `leaky_relu`, `prelu`, `hard_sigmoid` | x (SFPU, exact; a one-element weight is expanded on the device by `device_op_ungated`, residency gate step129) | S2 |
| `sigmoid{,_backward}`, `gelu{,_backward}` | x (SFPU, derived bounds; `sigmoid_backward` exact) | S4 |
| `log_sigmoid{,_backward}` | x (SFPU, derived bounds) | S4 |
| `softmax`, `log_softmax` | x (device composition, derived bound; every supported size) | R2 |
| `softmin` | x (device composition, derived bound; every supported size) | R2 |

### `ModuleOps`

| Methods | Device | Item |
|---|:-:|---|
| `linear` and its three backwards | `~` over `float_matmul`; a rank-N input folds its batch into the rows (forward, `x` grad), `linear_{weight,bias}_backward` hand-written likewise | B6 |
| `embedding{,_backward}` | x (Burn's default over `select`/`select_add`, on the mover) | D4 |
| `conv1d`, `conv2d`, `conv_transpose2d`, their backwards, `unfold4d` | `~` native F32/BF16, grouped/ragged/dilated forward/backward, trace/training/mesh gates; acceptance open | D6 |
| `avg_pool*`, `adaptive_avg_pool*`, `max_pool*` and backwards | x NCHW 2-D and Burn's 1-D compositions; all-padding windows refused; BF16 averages use GAPOOL | M2 |
| `layer_norm` | x Burn composition over native primitives; F32 and BF16 numerical/gradient gates | R3 |
| `attention` | `~` native F32/BF16 scale/softcap/masks/bias, zero masked rows, gradients/training, large-K views and replay; both-card forward/all-gradient references (`1791249486`) | R4 |
| `conv3d`, `deform_conv2d`, `interpolate`, `ctc_loss`, `rfft`/`irfft` | | not planned until a model needs them |

### `IntTensorOps`, `BoolTensorOps`, `QTensorOps`

| Methods | Device | Item |
|---|:-:|---|
| storage on the device | x `I32`, `Bool` (any store); other int dtypes host | D3 |
| `{int,bool}_{reshape, slice, swap_dims, transpose}` | `~` views, as `float_`'s | D3 |
| `int_{add,sub,mul}{,_scalar}`, comparisons, `bitwise_*` | x I32 full-width wrapping ALU, signed comparisons, masked shifts; both-card validated | S5 |
| `int_{div,remainder}{,_scalar}`, `int_mean_dim` | `~` checked native I32, simulator/card-0 validated (`1791232718`) | S5 |
| `int_neg` | Burn composition over native wrapping ALU | S5 |
| `int_abs` | x wrapping (`i32::MIN` stays `i32::MIN`, as Flex); card 0 (step126, run `1791560877`) | S5 (T1) |
| `int_clamp*`, `int_sign`, `int_max_abs*` | x Burn defaults over native mask fill/abs, no downloads (step126) | S5 (T1) |
| `int_into_float` | x to F32 (SFPU, exact) | S4 (10.2d) |
| `bool_into_float`, `bool_into_int` | x native exact 0/1, F32/I32 output only | S6 |
| `int_cast` | `~` I32→I32 native; other widths fail by name (`[-]`: the device stores I32 only) (step126) | S6 (T1) |
| `int_sum*`, `int_max*`, `int_prod*`, `int_min*` | x I32 reductions (step80) | R1 |
| `int_matmul` | x exact modulo 2^32 composition (expand, `int_mul`, `int_sum_dim`), size budget 2^22 elements, never an f32 matmul; card 0 (step128) | S5 (T2) |
| `int_cumsum`, `int_cumprod`, `int_cummin`, `int_cummax`, `int_argmax`, `int_argmin` | x wrapping/signed resident scans and first-extreme indices (step131-132, card 0) | R1 |
| `bool_and`, `bool_or`, `bool_xor`, `bool_not`, `bool_equal`, `bool_equal_elem` | x (SFPU, exact) | D3 |
| `bool_any{,_dim}`, `bool_all{,_dim}` | x raw 0/1 OR/AND reductions on all axes; both-card validated | R1 |
| `int_mask_where`, `int_mask_fill`, `bool_mask_where`, `bool_mask_fill` | x raw-bit selects and fills (16777217 and `i32::MIN` exact); card 0 (step126) | S5 (T1) |
| `{int,bool}_{permute, flip, unfold}` | x dtype-generic views and copies; card 0 (step125, run `1791560886`) | D3 (T1) |
| indexing (`*_gather`, `*_select`, `*_cat`, `*_slice*`, `*_scatter*`) | `~` resident arbitrary-axis gather/select, stepped slices/assignment/cat/repeat, I32 wrapping and Boolean OR updates; ND indexing remains open | D4 |
| `QTensorOps` | explicit unsupported | Deferred; D2 is backend tensor storage, not portable quantization |

---

## Hazards and known bugs this phase meets

From the Tier 2 register (`implementation-checklist.md`) and the divergence log, mapped to
the item that must handle each. An item is not done while its hazard here is open.

| Hazard | Source | Item |
|---|---|---|
| `SFPMAD` automatic stalling misses seven cases | `SFPMAD.md:72,75-76`; `stalls_automatically_after_mad` | F2 -- closed: the builder inserts the NOP |
| `SFPPOPC` complex modes with a full flag stack | Tier 2 | F2 -- closed: never emitted |
| `SFPLUTFP32` writes `LReg[LReg[7] & 15]`, not `LReg[VD]` | `SFPLUTFP32.md:15` | S4 -- closed (10.2a): `Program::lut_fp32` points `L7` at `VD` and sets `Mod1Mirror`; measured on both cards (`step26`) |
| `SFPSTOCHRND` biased; round-toward-zero sometimes rounds away | Tier 2 | S6 |
| `SFPCAST_IntAbs` computes absolute value | Tier 2 | S5, S6 |
| `SFPMUL` with `Mod1 > 1` refused by ttsim; `SFPMAD` spelling used | divergence row 17 | S1 |
| `-1.0 * 0.0` gives `+0.0` unless the addend is `-0` | numerics row C | S1 |
| Denormals flush, NaNs canonicalise to `0x7FC0_0000` | numerics row D | every oracle |
| `SFPLOADMACRO` unsupported on ttsim | divergence row 7 | S9 |
| `UnpackToDst` refused for 16-bit and block-float inputs on ttsim | divergence row 31 | F1, D1 |
| `DOTPV`, `SHIFTXB`, `MOVDBGA2D` unimplemented on ttsim | divergence row 50 | M4 |
| `STALLWAIT` must block the *consumer*; units run concurrently on silicon | divergence row 46 | F3 |
| `Config` and per-thread state survive between programs | divergence rows 47, 49 | F3 |
| Overwriting a program a queued list will run corrupts the tile | X4c (found on silicon) | X4c -- closed: no eviction while lists are queued |
| A host GDDR write is not yet visible to a mover reading through another port | divergence row T | X4c -- closed: `dram_write` reads back through every port |
| A host L1 write is not ordered against another agent writing the same L1 (an Ethernet transfer landing, a mover) | divergence row AA | closed in `silicon_eth_link` by a read-back fence; closed as API: the `write_fenced` family (X7, step119) -- `Device::write` is posted, and a write another agent may race needs its read-back (X7) |
| The barrier counter in unit 0's L1 keeps an earlier session's count, so every barrier passes at once and multi-unit ops overlap | X4c (found on silicon, once P1 removed the per-step syncs that hid it) | X4c -- closed: zeroed with the session's barrier number whenever unit 0's mover starts (`step34_batching::barriers_count_from_zero_whatever_an_earlier_session_left`) |
| A drain that a descriptor change needs, taken after a list's programs were placed, unpinned them too, so the next placement could evict them under the queued list (an `SFPPUSHC` stack overflow on ttsim) | 10.2's block repeats (programs ~30x smaller changed what the cache evicted) | X8 -- closed: `enqueue_segment` drains before placing |
| A tile wedged by a corrupt run stays wedged: after the backend pulse, every semaphore released (row 65) and the RISC-V semaphore posts (`mailbox::UNWEDGE`), thread 1 takes no instruction (its runner stalls after 29 pushes, one FIFO). Cause: a math instruction waiting for `Src` banks the pulse gave back to the unpackers (reproduced on purpose, row AH). Trying `UNPACR_NOP_SETDVALID` (UNVERIFIED encoding) on the wedged tile took the host down | silicon, 2026-10-01 | closed -- prevented (X4c), detected at open (X5a), recovered by feeding the banks with plain `UNPACR`s (X5b) |
| A list's `READ_RUN` followed by its `WRITE_RUN` had no `WAIT` between: a record's moves are issued without waiting, so a write could read its staging slot before the read landed -- ordered only by timing (each group's writes start after all its reads are issued), which is why small groups were the risk and no gate saw it; ttsim's reads land at once | found reading the mover for D4 (2026-10-03) | closed: `tensor::copy` and `copy_blocks` put a `WAIT` between, as host DMA and `gather_rows` do |
| The mover's completion wait reads an 8-bit counter (`NIU_MST_REQS_OUTSTANDING_ID`) that wraps at 256 in flight, so a long list or record could report done before its data landed | `NoC/Counters.md`; `../learnings/firmware-performance.md` | closed: `tt_isa::noc::niu::InFlight` caps each ID at `MAX_IN_FLIGHT` (128) in `noc::issue`; stalls counted (`DataMover::throttle`, `Session::throttle`, a `session:` warning); `step49_in_flight`, `silicon_bench_memory::gddr_in_flight`; ttsim cannot show it (row 72) |

New ttsim refusals or disagreements found while doing any of this go in
`ttsim-divergence.md`, numbered after the last row, and are cited from the item.

---

## Execution

**Branches.** One per milestone, stacked: `phase10-0-sfpu-foundation` off `main`, each
later milestone off the previous one (`phase10-1-softmax`, `phase10-2-activations`,
`phase10-3-reductions`, `phase10-4-formats`, `phase10-5-indexing-conv`, `phase10-6-rest`).
A milestone's branch is green on ttsim and both cards before the next one starts.

**Per item.** The `tt-isa` helper and its unit tests; the oracle; the ttsim gate, watched
failing once; `cargo xtask silicon --release --device all --filter <gate>`; the Burn
override with its Flex comparison, residency check and `SMOKE` entry; then the docs, in the
same commit as the code:

1. here: the item ticked, its inventory row's columns, its Burn table rows, its hazard row
   closed, the milestone's state, the date in "Where things stand";
2. `ttsim-divergence.md`: a numbered row per refusal or disagreement, a lettered row per
   measurement, cited from the item;
3. `implementation-checklist.md`: the Tier 2 bug entries and silicon-verification backlog
   entries the item settles, and the milestone line `10.N` when it closes;
4. at a milestone's close, `RUST_IMPL_PLAN.md`'s Phase 10 status and any
   `burn-backend-parity.md` row that cites the item.

**Order inside 10.0.** X3 (so S1's gain is measured on the device), F0, F1, F2 with X1, F5,
F3, F4, S1, F6.

### Static copy and convolution continuation (2026-10-05)

`step84`: raw multi-source slice assignment, native empty initialization and
composed cat/repeat preserve F32/BF16/I32/Bool values and slice gradients in
ttsim and both cards (`1791233990`, 6/6). Resident dynamic indexing is covered by step86 below.
`step85`: grouped/depthwise Conv2D, Conv1D composition, transposed Conv2D,
unfold and gradients pass initial independent simulator and both-card gates
(`1791234267`). Card-0 F32/BF16 traces, resident BF16 training and actual two-card
forward/gradient tests pass (`1791234542`). BF16 boundaries narrow once from F32.
`step86` validates resident arbitrary-axis gather/select, duplicate float additions,
DOMAIN refusal and changed-index traces on both cards (`1791235065`). Broader
numerical, mutant and format/layout acceptance is covered by the later lane gates; large-K is gated (`step68`, `1791145571`; `step83`; `1791249159`).

Additional continuation gates: forced ragged K=3609 (113 tiles, plan asserts K
is split) checks convolution forward and all gradients on both cards
(`1791235857`, 22/22 full step85 selections). A fractional permuted input matches
external Flex under an independently derived operand/phase accumulation bound;
BF16 attention replays changed V across ragged heads/sequences on two tiles. Both
cards pass these additions (`1791235944`, 4/4). Reversed convolution kernel-column
and shifted resident-index mutants were observed failing simulator oracles.

`step87` validates explicit 16×16 Src transpose, special values and changed-input
traces; eligible TF32 matmul preparation matches the native raw-copy reference on
both cards (`1791239869`). M3 remains partial because this route normalizes zero
and subnormal payloads. GMPOOL ArgMax's packed path returns no index bits on both
cards (`1791240702`), so exact max/indices stays SFPU. `step88` directly compares
F32/BF16 module forward/all gradients against a single-card reference while
requiring arithmetic and Ethernet activity on both cards.

Step86 typed updates now include I32 wrapping scatter/select-add and Boolean
scatter/select-OR. I32 overflow/duplicates, canonical Boolean updates and
changed-device-index two-tile replay pass both cards (`1791249159`). Slot/geometry
decode checks have an independent unit gate; dynamic domain checks precede datum
addressing. The same run validates large-K permuted attention and F32/BF16
transposed-Conv1D/unfold backward.


### Final continuation verification (2026-10-06)

The updated isolated release smoke selection passes **288/288** on cards 0 and 1
(run `1791250791`), including step82–88, one/two-tile changed-input traces,
resident training, checked domains and actual distributed forward/backward
products. Direct single-card/mesh comparisons cover F32/BF16 outputs and all
module gradients; the attention case has nonzero Q/K contributions on both
partitions and acknowledged Ethernet traffic.

The full default workspace run including doctests passes
(`target/silicon/tranche-workspace-final.log`). Final default/silicon workspace
Clippy, separate RISC-V firmware Clippy, formatting, silicon no-run compilation,
all three generator checks and both shipping dependency checks pass
(`target/silicon/tranche-checks-final.log`). The five MNIST e2e regressions pass
in 264.20 s with the golden unchanged
(`target/silicon/tranche-mnist-current.log`). Release operation baselines and
conditions are recorded in `docs/learnings/firmware-performance.md`; no speedup
guarantee is made. M3 remains partial for the measured Src payload-conversion
limit; ND indexing and excluded convolution/attention variants are not closed.


Convolutional application coverage (2026-10-06): `tt-mnist --model cnn` shares
its Conv2D/ReLU/AvgPool/Linear model with step89. F32/BF16 MNIST learning,
convolution SGD updates, scalar-only training downloads and changed-batch trace
identity on one/two tiles pass both cards (`1791252026`). Native integer/Boolean
slicing and strided swaps are extended for resident label/view preparation;
packed BF16 parameter copying closes the training-capture storage-format gap.


CNN continuation final verification: card 1 passes 15/15 slicing/index/CNN
selections (`1791252326`), and card 0 passes the same 15/15 (`1791252404`).
Restored step89 simulator learning/replay gates pass; the new permuted I32/Bool
slice oracle passes. All five existing MNIST MLP e2e regressions pass in 267.53 s
with the golden unchanged. Workspace default/silicon Clippy, formatting, Burn
generator and both shipping dependency checks pass. Logs are
`target/silicon/cnn-{simulator-final,slice-simulator,mlp-regression,clippy-final,default-clippy}.log`.

### M1 matrix elementwise continuation (2026-10-06)

- [x] Typed ELW retaining/releasing consumers and SrcB broadcast enum; generator
      provenance updated and regenerated without changing measured bit positions.
- [x] Fixed-size tile streaming kernels, complete ADC/RWC/fidelity setup, F32
      Dst, final-consumer bank release and program keys containing precision,
      fidelity, operation, broadcast, packed storage and transport model.
- [x] Explicit F32/packed BF16 Session APIs; single-card Burn mode is selected
      before attachment and snapshotted without per-operation server round trips.
- [x] Independent exact-domain oracle/refusals, exponent alignment probe, finite
      error budgets and measured special corpus; four safe instruction mutants.
- [x] All operations/broadcasts/fidelities, ragged poisoned padding, parent views,
      reductions/matmul consumers, one/two-tile changed-input traces/deferred frees,
      zero intermediate downloads and analytic gradients; step90 is in SMOKE.
- [x] Both cards: final targeted acceptance `1791255922` (24/24), including explicit
      silicon Dst base setup, BF16 analytic gradients, alignment and finite bounds.
- [x] Release host-timed medians with validation, warmups and dataflow_stats:
      final baseline `1791255969`, both cards, 192 comparisons; conditions and
      results are in firmware-performance.md.
- [x] Workspace tests, default/silicon Clippy, silicon compilation, formatting,
      all generator checks and both shipping dependency checks pass. Full release
      smoke `1791255409` passes 322/322 across both cards; final targeted setup
      passes 24/24. All five MNIST e2e regressions pass with the golden unchanged.
      Logs: `target/silicon/m1-*.log` (workspace, MNIST, simulator, silicon
      compilation and final checks).

SFPU remains the default. Ttsim cannot narrow BF16 results (PACR 0x105) or write
Dst base register 6; these exclusions are documented in ttsim-divergence.md.

### M1 direct RHS broadcast optimization (2026-10-06)

- [x] Remove F32/packed BF16 expanded RHS repacking; B reads original slots with
      tile/face/RWC selection, checked alignment and existing bank/NC ownership.
- [x] Simulator step90 11/11 and both-card step90 `1791256564` 22/22, including
      changed nonconstant RHS trace replay on one/two tiles, all geometries,
      deferred frees, analytic gradients, views and downstream consumers.
- [x] Arithmetic gate requires zero staging transfers. Poisoned RHS padding gate
      expanded to row/column/scalar geometry; both-card `1791256629` passes 2/2.
- [x] Release comparison `1791256590`, both cards, 192 validated results, two
      warmups/seven host-timed samples with dataflow_stats. Ragged row multiply
      ~27× faster F32/~19× BF16 against the initial matrix materialization path;
      no application speedup claimed. Firmware performance scoreboard updated.
- [x] Final expanded padding gate and default/silicon workspace Clippy/format
      checks pass; logs `target/silicon/m1-broadcast-*.log`.
- [x] All five MNIST e2e regressions pass (267.70 s), golden unchanged;
      `target/silicon/m1-broadcast-mnist.log`.

### F1 resident matrix register chains (2026-10-06)

`Session::matrix_eltwise_chain` accepts equal-shape resident F32 operands and
returns F32. `MatrixChainTail::{WithRhs,WithLhs,Square}` selects Dst-to-SrcA,
Dst-to-SrcB, or Dst-to-SrcB followed by SrcB-to-SrcA. Both stages use the selected
TF32/BF16 precision and fidelity. The intermediate is explicitly truncated to
10/7 fraction bits; this API does not promise F32 intermediate precision.
No Burn routing, mesh, autodiff or automatic fusion changes are included.

Step97 covers every first op/tail/precision/fidelity, aligned/ragged shapes,
one/two tiles, raw exceptional move datums, poisoned padding, parent row views,
downstream reductions/matmul, changed-input replay and deferred operand frees.
Independent finite-domain f64 phase/grid models and composed error bounds live
beside the gates; raw physical-format reference ports live in `matrix::moves`.
Safe omitted/swapped/wrong-row/stale-accumulator controls fail comparison on the
device. Builder audits require only final GDDR allocation, two reads and one
write per tile, the expected moves, no SFPU arithmetic and one release per bank
per face. Output padding is undefined; parent claims are preserved.

Four-row semantics/chains pass ttsim (8/8 step97); both-card step97 passes
20/20 (`1791317039`). Unaligned row arguments and lane masks use silicon-only
arms; one-row moves are silicon-only (divergence 37 and the step97 addendum).
Silicon aligns read addresses but leaves Src write addresses unaligned; the
independent reference port includes this measured correction. Production uses
aligned addresses. Semantic gates and production instruction audits establish
the three move completions. Full release silicon SMOKE passes on both cards
(`1791317135`, 388/388), including expanded one-row offsets/wrapping and MNIST.
The release benchmark passes on both cards (`1791317719`): measured cases
improve 1.38–1.48×, with region/batch counts halved. Conditions and medians are
recorded in firmware-performance.md; no application speedup is claimed. Encoding
provenance is unchanged.

Stage C source-bank acceptance (2026-10-07): step103 card-0 release semantic gates
9/9 (`1791397029`), simulator 6/6. See [source-bank tranche](../completed-plans/source-bank-clear-release-shift.md)
for the accepted checklist and restricted variants. Full card-0 release SMOKE
passes 242/242 (`1791397621`); MNIST 8/8, golden unchanged; workspace tests,
format/Clippy, silicon compilation/Clippy, generator and shipping checks pass. All four instruction groups have
checked semantic implementations; recovery, Burn routing and the existing SETRWC
release helpers are unchanged. Stage D is accepted below.

Stage D acceptance (2026-10-07): measured non-clearing healthy-bank publication,
checked transitions, independent physical oracle and bounded diagnostics.
Card-0 step104 passes 16/16 within release SMOKE 258/258 (`1791413478`);
the final fractional format-sensitive consumer passes `1791413841`.
Workspace tests, format/Clippy, silicon no-run/Clippy, generator and shipping
dependency checks pass; MNIST e2e 8/8 retains its golden. Simulator refuses the
measured NOP and B ownership wait; regular controls survive isolated probes.
No production routing, recovery or performance change is claimed. See
[completed Stage D](../completed-plans/explicit-unpacker-handover.md).
