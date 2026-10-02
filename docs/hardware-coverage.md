# Hardware coverage — Phase 10 tracker

The working tick-list for Phase 10 of `RUST_IMPL_PLAN.md` ("Phase 10 — Hardware
coverage"). The plan says *why*; this file says *which parts of a Blackhole Tensix tile
this stack can drive, which it cannot yet, and in what order the rest arrives*. It is the
progress record: an item is ticked here or nowhere.

Execution order, branches and the per-item workflow: the "Execution" section at the end.

Companion guides: [`tt-metal-concepts-review.md`](tt-metal-concepts-review.md) (the
Tenstorrent concepts this stack lacks, G1–G16, and the hardware sharp edges to handle in
code) and [`burn-backend-parity.md`](burn-backend-parity.md) (the `burn-tt` surface and
ergonomics roadmap, B0–B16). Item ids here are cited from both.

Same legend as `implementation-checklist.md`: `[x]` done and gated by a test · `[~]`
partially done, see note · `[ ]` not started · `[-]` deliberately not done, with the
reason given.

---

## Where things stand (2026-10-01, 10.2 in progress)

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
exponential family's core is on the card too (10.2e, part: `expm1`, `tanh`, `erf`,
`sigmoid`, `gelu` and both backwards), so all seven of `tt-mnist`'s activations now
move only what ReLU's step moves -- gelu trains at 2.4 / 1.5 ms/step, from 5.6 / 4.9
(row AK). The runner repeats blocks (X8), so long programs (`pow`, `gelu`) are one op.
Next: the hyperbolics, `log_sigmoid` and `softmin` (rest of 10.2e), then trig (10.2f).

### After 10.1

10.1 added, on top of the table below: reciprocal, division, `exp` and `log` on the SFPU
(S3, S4a), lane movement (S8), `sum` and `max` over either dim (R1a), softmax and
log-softmax on the device (R2); the matmul's loops replayed and the MOP Expander gated
(X1, X2); the movers' queues, barriers, batching and traces (X4); and wedged tiles
detected and recovered (X5). The table is 10.0's.

Phases 0–9 built the path to the card. The compute that actually runs on it is narrow:

| Unit | What runs there today | Where |
|---|---|---|
| **Matrix Unit** | `MVMUL` only, for matmul (TF32/BF16 `Src`, `Lo`..`HiFi4`), plus `ZEROACC` | `tt_kernels::matmul`, `role_t0..2` |
| **B core FP32 unit** | the column sum, small element-wise ops spread over many units, and the reference for every element-wise op; padding fills | `tt_isa::dm::kind`, `dm_b.rs::{compute, col_sum, per_datum, fill_pad}` |
| **SFPU** | every element-wise op where it is cheaper (`Auto`): `ADD`, `SUB`, `MUL`, `MUL_SCALAR`, `ADD_SCALAR`, `RELU`, `RELU_BACKWARD`, `ADD_ROW` | `tt_kernels::sfpu::{ops, kernel}` |
| **Unpackers / packer** | flat FP32 runs and the matmul's tile path; `UnpackToDst` for 128 datums | `tt_kernels::datapath`, `matmul` |

The instruction *table* is far ahead of the kernels: `tt_isa::isa::generated` encodes 161
instructions, every SFPU instruction among them, and `ELW*`, `GMPOOL`/`GAPOOL`, `MOP`,
`REPLAY`, `TRNSPSRCB` and the ThCon set besides. The hand-written `tt_isa::sfpu` layer on
top has `loadi`, `load`, `store`, `mad`, `mul`, `add`, `sub`, `load_f32` and `nop` -- no
conditional execution, LUT, reciprocal, exponent or mantissa ops, casts, integer ops,
swaps, transposes or PRNG.

On the Burn side, 12 compute methods have a device path (`burn-tt/src/ops.rs`, listed in
`OVERRIDDEN`, `xtask/src/gen_burn.rs`); no `ModuleOps`, `IntTensorOps` or
`BoolTensorOps` method does. Under device residency every other op is a download, a
Flex op on the host and an upload, which is why breadth is a performance problem and not
only a feature list.

**Milestones** (detail in "Work items"):

| # | Milestone | Items | State |
|--:|---|---|---|
| 10.0 | Device profiler; SFPU foundation; today's element-wise ops move from the B core to the SFPU | X3, F0–F5, X1, S1 | `[x]` (F6, optional, deferred; F2's `SFPCONFIG` prologue and F5's further models arrive with S4) |
| 10.1 | Softmax and cross-entropy on the device; `MOP`; op-list traces | S3, S4 (`exp`, `log`), S8, R1 (`max`, `sum`), R2, X2, X4, X5 | `[x]` S3, S4a, S8, R1a, R2 (softmax, log-softmax), X2, X4, X5; cross-entropy moved to 10.5 with D4 (Burn gathers the target column, `float_gather`) |
| 10.2 | Activation and math breadth; int and bool storage | rest of S2–S4, D3 (from 10.4), F2's `SFPCONFIG` | `[~]` 10.2a (the instructions: helpers, models, oracles, gates), 10.2b (D3: `I32` and `Bool` resident), 10.2c (S2: compare, select, sign), 10.2d (S4: `sqrt`, `log1p`, `pow`; S3 and `exp` fixed at their range ends) |
| 10.3 | Reductions over any dim, device transpose, norms | P1, M2, M3, R1, R3 | `[ ]` |
| 10.4 | Formats and integers | D1, S5, S6 (D3 moved to 10.2) | `[ ]` |
| 10.5 | Indexing, convolution, pooling, attention | D4, D5, P2, D6, R4 | `[ ]` |
| 10.6 | The rest: block float, PRNG, `SFPLOADMACRO`, `ELW*`, `DOTPV` | D2, S7, S9, M1, M4 | `[ ]` |

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
4. **A silicon gate**, the same test on both cards through `cargo xtask silicon`.
5. **Burn routing**: the method overridden in `burn-tt/src/ops.rs` and listed in
   `OVERRIDDEN` (then `cargo xtask gen-burn-delegate`), agreeing with `burn-flex` bit for
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
| Rounding | `SFPSTOCHRND` (`_FloatFloat`, `_FloatInt`, `_IntInt`) | x | | | | | S6 |
| Lane movement | `SFPSWAP`, `SFPTRANSP` | x | x (`SFPSWAP` min/max) | `~` (reductions) | x | x | S2 |
| Configuration | `SFPCONFIG` | x | `~` `LReg[11..15]` only (`Program::constant`) | | x | x | F2 |
| Macro | `SFPLOADMACRO` | x | | | `-` row 7 | `~` load half | S9 |
| Misc | `SFPNOP` | x | x | x | x | x | -- |
| PRNG | `SFPMOV`/`SFPCAST`/`SFPSTOCHRND` PRNG modes (`VectorUnit.md`, "PRNG") | x | | | | | S7 |

### Matrix Unit (FPU)

Reference: WH `MatrixUnit.md` (STUB-B), WH `MVMUL.md`, WH `SrcASrcB.md`, WH `RWCs.md`, BH `Dst.md`.

| Feature | Enc | Helper | Kernel | Sim | Si | Item |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `MVMUL`, fidelity phases `Lo`..`HiFi4` | x (measured) | x | x | x | x | done (Phases 6–7) |
| `ZEROACC`, `ZEROSRC` | x (`ZEROACC` measured) | | x | x | x | -- |
| `MOVA2D`, `MOVB2D`, `MOVD2A`, `MOVD2B`, `MOVB2A` | x (measured) | `~` (`mova2d`, `movb2d`) | | x | x | F1 |
| `ELWADD`, `ELWSUB`, `ELWMUL` (with `Src` broadcast) | x (measured) | | | `~` encoding only | `~` | M1 |
| `GMPOOL`, `GAPOOL` | x | | | | | M2 |
| `TRNSPSRCB` | x (WH) | | | | | M3 |
| `DOTPV`, `SHIFTXA`, `SHIFTXB`, `MOVDBGA2D` | x | | | `-` row 50 | `~` encoding | M4 |

### Unpackers and packer

Reference: WH `UNPACR_Regular.md` (conditionalized, authoritative), WH `Unpackers/*`, BH
`PACR.md` ("basic"), WH `Packers/*` -- the thinnest part of the Blackhole tree.

| Feature | State | Item |
|---|---|---|
| Flat FP32 run, `Src` tile path (TF32/BF16), `UnpackToDst` 128 datums, a datum sub-run of a tile (base moved) | `[x]` | -- |
| `UnpackToDst` of a whole 32×32 tile, and the whole tile packed back | `[x]` FP32 (`step25_dst_tile`) | F1 |
| BF16 into `Dst` (`UnpackToDst` on silicon; ttsim refuses, row 31) | `[ ]` | D1 |
| Packer output format conversion (FP32 `Dst` → BF16/FP16 L1) | `[ ]` | D1 |
| Block-float formats, exponent sharing, `CLREXPHIST` | `[ ]` -- codes are `None` (`tile.rs`) | D2 |
| Integer formats (INT32 code 8 measured; INT8/UINT8 not) | `[~]` 32-bit integers and bools stored as raw bits through the FP32-coded path (D3); INT8/UINT8 with D2 | D3, D2 |
| Unpacker transpose / tilize modes, broadcast | `[ ]` | M3, D5 |
| Packer ReLU and edge masking, `PACR_SETREG` | `[ ]` | S1 (opportunistic), D4 |

### Frontend and tracing

Reference: WH `REPLAY.md`, BH `MOPExpander.md`, WH `MOP.md`/`MOP_CFG.md`, BH
`BabyRISCV/AutoTTSync.md` (the expanders and the Wait Gate), `DebugTimestamper.md`.

| Feature | Enc | Helper | Kernel | Sim | Si | Item |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `REPLAY` (record and replay, 32 entries per thread) | x | x | x (SFPU ops) | x | x | X1 |
| `MOP` / `MOP_CFG` (MOP Expander templates) | x (`CONFIRMED`) | x (`frontend::mop`, mailbox `MOP_CFG`) | (X2b) | x | x | X2 |
| Debug timestamper event stream | -- | x (`tt_device::trace`, `tt_kernels::profile`) | x mover and role events | `-` row 54 | x | X3 |
| Op-list traces (a step's records kept in GDDR, replayed) | -- | | | | | X4 |
| `.ttinsn` fusion (four pushes per cycle) | -- | | | | | checklist Phase 9 |
| Hazards as data, the wait planner | -- | | | | | `RUST_IMPL_PLAN.md` "Hazards as data"; checklist 9.8 |
| Three-thread pipelining, double buffering | -- | | | | | checklist 9.8 |

### Scalar unit, mover, atomics, NoC

Pulled in only when a kernel needs them; each says which.

| Feature | Spec | State | Wanted by |
|---|---|---|---|
| ThCon `SETDMAREG`, `ADDDMAREG`.., `LOADIND`/`STOREIND`, `FLUSHDMA` | WH `ScalarUnit.md` + pages | `SETDMAREG` used for config staging; rest `[ ]` | F3 (per-tile parameters without reprogramming) |
| Tensix atomics `ATCAS`, `ATINCGET`, `ATINCGETPTR`, `ATSWAP` | WH | `[ ]` | 9.8 page FIFO, if counters move into Tensix |
| `XMOV` (Tensix mover, L1 → L1) | WH `XMOV.md` | `[ ]` | D4 (copies without the B core) |
| NoC multicast (NIU broadcast; TLB `strided`, row 4) | BH `NoC/MemoryMap.md` | `[ ]` | weight broadcast to many tiles (9.6 follow-up) |
| NoC atomics | BH `NoC/Atomics.md` | `[ ]` | R1 across tiles |
| NoC counters / interrupts | BH `NoC/Counters.md`, `Interrupts.md` | counters `[x]`; interrupts `[ ]` | -- |
| `L1CacheTagSearchAccel` | BH | `[ ]` | checklist Phase 9 |
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
    - **Burn.** `burn_tt::Trace::capture(&input, || forward(..))` captures on the
      input's own buffer and returns the capture's output values; `run(values)` writes,
      replays and downloads in one round trip. Both return host values, not a tensor: a
      tensor of the output buffer would change under its holder at the next run. A
      closure that falls back to the host is refused (the op panics, as a device error
      does) and the capture is ended, never left open. Single-chip engines only; the
      mesh engine refuses. `tt-mnist --infer --trace` replays per batch.
    - Gates: `step39_traces` -- a layer's forward pass on one tile and on two (barriers)
      replayed over three new inputs, each the same ops run fresh bit for bit, and each
      replay's writes under a twentieth of the capture's; a freed weight deferred while
      an allocation of its size lands elsewhere; every refusal. Watched failing: a `CALL`
      over half its entries (replay 0, element 0 wrong). `step40_burn_trace` -- a Burn
      MLP traced and run on new inputs against the fresh Burn ops bit for bit, and a host
      fallback refused with the next capture working. ttsim and both cards. MNIST
      inference traced: the same predictions; 2.59 ms a batch, of which 2.1 the input's
      200 KB write from the host (row AG, X7).
    - **Training: later, documented.** A training step is replayable once its weights are
      updated in place (the optimizer writing the same buffers) and the loss stays on the
      device; neither holds today (B16, and Burn's tensors are immutable). Until then a
      training loop traces its forward pass at most. With the x280s as an on-card host
      (`feature-x280-on-card-dispatch.md`), the host-side part of a step moves next to the
      data and whole-step traces become the natural shape.
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
- [ ] **X7 Small host transfers.** A `[64, 10]` upload (2.5 KB, two tiles) costs ~470 us
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

- [~] **P1 Rank-N tensors** (concepts review G2), minimal: a logical shape stored as
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
  - [ ] **P1b Batch stride.** `TensorRef` batch stride (0 = broadcast) for batched
        matmul and reductions over leading dims.
- [ ] **P2 K blocking** (concepts review G3): `Dst` reload or packer L1 accumulation, so
      a matmul's K is not capped by L1. Blocks D6's im2col.

### F — SFPU foundation (blocks every S item)

- [x] **F0 Padding is a property of the tensor, not an assumption.** `DramTensor`
      carries `pad: Pad` (`Zero` | `Undefined`; a tensor with no ragged edge is
      always `Zero`), upload sets `Zero`, and every op implements `OpPadding`
      (`requires(input) -> PadNeed`, `produces(inputs) -> Pad`, from the op's
      algebra: `ADD_ROW` and `MUL_SCALAR` by a non-finite scalar leave
      `Undefined`, `RELU_BACKWARD` is `Zero` if either input is). `COL_SUM` reads
      only a ragged tensor's valid rows (`record::SUM`'s `last_rows`, the compute
      entry's new parameter word) and zeroes its result's padding rows, so it
      needs nothing; the matmul needs `Zero` on both operands, which the session
      supplies by `record::FILL_PAD` over the edge tiles only, in place on a
      tensor that owns its slots and through a bit-exact copy (`kind::COPY`) for a
      view, so a parent's padding -- and its views' claims -- are never changed by
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
      `sfpu::ops` and a gate. `EltwiseUnit::{Auto, Sfpu, Mover}`: `Auto`, the default,
      picks by `tensor::sfpu_is_cheaper`, a linear cost model per op from
      `silicon_perf::eltwise_unit_sweep` (measurement Q); `TT_ELTWISE=auto|sfpu|mover`
      for burn-tt.
- [~] **F5 Oracles.** `tt_kernels::sfpu::interp::Vector`: `LReg[17][32]` (a
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
- [ ] **F6 (optional) A Burn coverage generator.** `cargo xtask burn-coverage --check`,
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
- [~] **10.2a The instructions the rest of S2-S4 needs** (`tt_isa::numerics::sfpu`,
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
- [~] **S4 Transcendentals.** Done: `exp`, `log` (10.1). Range reduction by
      `SFPEXEXP`/`SFPSETEXP` and integer exponent arithmetic (`SFPIADD`, `SFPSHFT`),
      polynomials in Horner form by `SFPMAD`. `exp`: magic-number rounding of `x log2
      e`, Cody-Waite reduction, degree-7 Taylor, `2^n` added to the exponent field;
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
  - [~] **10.2e The exponential family and the activations on it** (part).
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
        `step47_burn_activations::hyperbolics_log_sigmoid_and_softmin_stay_on_
        the_card_within_their_bounds`, with Burn's autodiff of `sinh` (`g cosh
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
        with `ln 2` dropped. Remaining: `log_sigmoid{,_backward}`, `softmin`;
        then `sin`/`cos` and the rest (10.2f).
- [ ] **S5 Integer ALU on INT32** (format code 8, measured): `SFPIADD`, `SFPMUL24`,
      `SFPAND`/`SFPOR`/`SFPXOR`/`SFPNOT`, `SFPSHFT`, `SFPLZ`. The first `IntTensorOps` on
      the device: `int_{add,sub,mul}{,_scalar}`, comparisons, `bitwise_*`, shifts.
- [ ] **S6 Casts and rounding.** `SFPCAST` int ↔ float (never `SFPCAST_IntAbs`: Tier 2,
      use `SFPABS`); `SFPSTOCHRND` FP32 → BF16/FP16 in round-to-nearest and stochastic
      modes, matching the documented (biased) behaviour rather than "fixing" it. Burn:
      `float_cast`, `float_into_int`, `int_into_float`, `float_round`, `float_floor`,
      `float_ceil`, `float_trunc`.
- [ ] **S7 The PRNG.** Seeded per tile from `Backend::seed`. The claim is distributional
      (a stated statistical test), not bit-exact against Flex, whose generator is
      different. Burn: `float_random`, dropout.
- [x] **S8 Lane movement.** `Program::rotate_row` (`SFPSHFT2_MOD1_SUBVEC_SHFLROR1`),
      `Program::transpose4` (`SFPTRANSP`), with `Program::and`/`or` (`SFPAND`, `SFPOR`)
      for lane masks; interpreter models of all four held to silicon in
      `step26_sfpu_isa` (a three-step rotation, a full row reduction by rotations, a
      transpose then add). First user: R1's in-tile folds.
- [ ] **S10 A fast, approximate mode for the transcendentals** (asked for during 10.2e;
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
- [ ] **S9 `SFPLOADMACRO`.** Silicon-only (row 7); a performance item, after everything
      else here works without it.

### M — Matrix Unit beyond `MVMUL`

- [ ] **M1 `ELWADD`/`ELWSUB`/`ELWMUL`** with `Src` row, column and scalar broadcast: binary
      element-wise at matrix-unit throughput, at TF32/BF16 `Src` precision. Opt-in, like
      `Fidelity`; never a silent replacement for the FP32 SFPU path.
- [ ] **M2 `GMPOOL`/`GAPOOL`.** Max and average over rows. Burn: `float_max_dim`,
      `float_mean_dim`, `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d` (with D6's
      windowing).
- [ ] **M3 Transpose on the Tensix** (`TRNSPSRCB`, or the unpacker's transpose mode) in
      place of the B core's face transpose (`READ_TRANSPOSED`). Burn: `float_permute`,
      materialised transposes.
- [ ] **M4 `DOTPV`, `SHIFTXA`/`SHIFTXB`.** Silicon-only (row 50). Only when a kernel
      wants them.

### R — Reductions and composites

- [~] **R1 Reductions over any dim.** Done (R1a): `sum` and `max` over either dim of
      a matrix (`Session::reduce`, `sfpu::reduce`): a reduce kernel -- many input tiles
      into one output tile, four semaphores numbered compatibly with the matmul's and
      the element-wise kernel's -- that accumulates lanewise in `Dst`, masks a ragged
      edge's padding lanes to the identity by lane index (`LReg[15]`, `SFPAND`,
      `SFPSHFT`), and folds within the tile by rotations (over columns, the result
      replicated across the row: broadcast-ready) or `SFPTRANSP` (over rows); the
      gather reads a tile column in column-major order (`READ_RUN` flag bit 2). Max is
      exact (total order: a positive NaN propagates, a negative one is ordered below
      `-inf`); a sum over columns is in tree order, within `2 (n-1) u sum|x|` of
      Flex's; a sum over rows stays on the mover, in Flex's order. Oracle: the same
      programs in the interpreter (`reduce::reference`), exact on integer data for
      every shape. Gates: `step31_reduce` (device equal to the program bit for bit;
      max equal to Flex's, sums within the bound; ragged shapes, lines of up to 32
      tiles), `step32_burn_softmax`. Burn: `float_max_dim`, `float_sum_dim` (both
      dims). Remaining (R1b, 10.3): `mean`, `min`, `prod`, `argmax`/`argmin`,
      `any`/`all`, full reductions, `cum*`, more than ~200 tiles along the reduced
      dimension (one pass's L1 limit; refused with a typed error today).
- [~] **R2's groundwork: broadcasts.** `sfpu::ops::Broadcast::{None, Row, Col}` for
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
- [~] **R2 Softmax, log-softmax on the device**; cross-entropy waits on D4. `softmax`
      and `log_softmax` (either dim of a resident matrix) run Burn's own composition
      -- max, broadcast subtract, `exp`, sum, broadcast divide or `log` and subtract --
      on the device end to end, decided once on the whole input. Against Flex's fused
      softmax: a bound derived from the parts' (`EXP_BOUND` twice, the sum's order, the
      division's ulp, Flex's own counterparts); measured worst `1.0e-6` relative.
      Burn's `CrossEntropyLoss` gathers the target column with an integer index
      tensor (`float_gather`, D4), so MNIST's loss -- and its logits download -- stays
      on the host for now. **Placement** (measurement S): an SFPU kernel op costs
      100-200 us whatever its size, a tile's download ~190 us, so the approximate ops
      (division, `exp`, `log`, the SFPU's reductions) reached through Burn's methods run
      on the host below eight tiles (`burn-tt`'s `APPROX_MIN_TILES`) -- autodiff's own
      `log_softmax` on MNIST's two-tile logits took the step from 3.8 to 7.9 ms/step
      on the device -- and softmax compositions decide on their input's size. Heuristic
      until submission is asynchronous or a lookahead exists (X4, B8, B13, B16).
      **Exact mode** (`burn_tt::set_exact`, `TT_EXACT=1`): only ops that give Flex's
      bits run on the device; the MNIST golden runs so. Gates: `step32_burn_softmax`
      (every step resident, both dims, three shapes; a two-tile tensor on the host and
      bit-identical); the MNIST golden in exact mode. Was: **R2 Softmax, log-softmax,
      cross-entropy on the device** (was checklist 9.12): max,
      subtract, `exp`, sum, reciprocal. General ops gated against Flex; MNIST's
      per-step logits download goes away as a consequence, not as the goal. Burn:
      `softmax`, `log_softmax`, `softmin`.
- [ ] **R3 Norms.** `ModuleOps::layer_norm`, RMS norm as a composite, and their backwards.
- [ ] **R4 `ModuleOps::attention`.** Matmul, scale, mask, softmax, matmul -- after R2 and
      the fusion work in Phase 9.

### D — Formats and data movement

- [ ] **D1 BF16 tensors in GDDR.** Packer FP32 → BF16, unpacker BF16 → `Src` and `Dst`;
      half the bytes for every op. `DramTensor` grows a format.
- [ ] **D2 Block float (BFP8/BFP4).** Measure the codes as divergence rows G and H did,
      then exponent sharing and `CLREXPHIST`. Prerequisite for `QTensorOps` on the device.
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
      `step47_burn_activations::integers_and_booleans_stay_on_the_card` against
      Flex, nothing downloaded, `computed_on_device` (watched failing with
      `bool_and` routed to the host); in `SMOKE`. MNIST unchanged (labels stay
      host values). Element-wise ops still do not read a transposed view (M3).
  - [-] **D3b INT8/UINT8 codes.** Deferred to D2: nothing would use an 8-bit device
        format yet (Burn's int is `i32`, bools ride INT32), and the codes are best
        measured beside the block-float ones `QTensorOps` needs.
- [ ] **D4 Indexing on the B mover.** General `slice` (not only whole tile rows),
      `slice_assign`, `cat`, `gather`, `scatter_add`, `select`, `select_add`, `repeat_dim`,
      `expand`, `flip`, `embedding` and its backward. The mover moves; the SFPU is not
      needed.
- [ ] **D5 Tilize and untilize on the device** (overlaps checklist 9.10).
- [ ] **D6 Convolution.** `conv2d` as im2col on the mover plus the existing matmul, then
      its three backwards, `conv1d`, `conv_transpose2d`, `unfold4d`.

---

## Burn op coverage

The compute methods of the pinned Burn 0.21 op traits, with the item that brings each to
the device. Bookkeeping methods (`*_device`, `*_to_device`, `*_into_data`, `*_from_data`,
`*_reshape`, autodiff flags) are left out. **Device** is `x` when the method has a device
path today, `~` when only some shapes do.

### `FloatTensorOps`

| Methods | Device | Item |
|---|:-:|---|
| `float_matmul` | `~` F32 2-D resident; batched host-staged | -- |
| `float_add`, `float_sub`, `float_mul` (incl. row and column broadcasts; any rank, P1a), `float_mul_scalar` | x (SFPU or mover by size) | S1, P1a |
| `float_sum_dim` | x (dim 0 the mover's, exact; dim 1 the SFPU's, order bound) | R1 |
| `float_slice` | `~` whole tile rows | D4 |
| `float_transpose`, `float_swap_dims` | `~` 2-D view | M3 |
| `float_add_scalar`, `float_sub_scalar` | x (SFPU or mover by size) | S1 |
| `float_div{,_scalar}`, `float_recip` | x (SFPU, within 1 ulp) | S3 |
| `float_remainder{,_scalar}` | | S6 |
| `float_neg`, `float_abs`, `float_sign`, `float_clamp{,_min,_max}` | x (SFPU, exact) | S2 |
| comparisons (`float_equal`.. `float_lower_equal_elem`), `float_mask_where`, `float_mask_fill`, `float_is_nan`, `float_is_inf` | x (SFPU, exact; `Bool` results resident) | S2 |
| `float_cast` | `~` to the tensor's own dtype (a no-op); others S6 | S6 |
| `float_exp`, `float_log` | x (SFPU, derived bounds) | S4 |
| `float_log1p`, `float_sqrt`, `float_powf*`, `float_powi*` | x (SFPU, derived bounds; `pow` one op) | S4 |
| `float_erf`, `float_tanh`, `float_sinh`, `float_cosh`, `float_asinh`, `float_acosh`, `float_atanh` | x (SFPU, derived bounds) | S4 |
| `float_sin`, `float_cos`, `float_tan`, inverse trig, `float_atan2` | | S4 (10.2f) |
| `float_round`, `float_floor`, `float_ceil`, `float_trunc`, `float_cast`, `float_into_int` | | S6 |
| `float_random` | | S7 |
| `float_max_dim` | x (SFPU, exact value) | R1 |
| `float_sum`, `float_mean{,_dim}`, `float_prod{,_dim}`, `float_max`, `float_min*`, `float_argmax`, `float_argmin`, `float_any*`, `float_all*`, `float_max_abs*` | | R1 |
| `float_cumsum`, `float_cumprod`, `float_cummin`, `float_cummax` | | R1 |
| `float_sort*`, `float_argsort`, `float_topk`, `float_argtopk` | | R1 (late) |
| `float_gather`, `float_scatter_add`, `float_select{,_add}`, `float_slice_assign`, `float_cat`, `float_repeat_dim`, `float_expand`, `float_flip`, `float_permute`, `float_gather_nd`, `float_scatter_nd`, `float_unfold` | | D4, M3 |
| `float_cross`, `float_grid_sample_2d` | | not planned until a model needs them |

### `ActivationOps`

| Methods | Device | Item |
|---|:-:|---|
| `relu`, `relu_backward` | x (SFPU or mover by size) | S1 |
| `leaky_relu`, `prelu`, `hard_sigmoid` | x (SFPU, exact; `prelu` with one weight on the host) | S2 |
| `sigmoid{,_backward}`, `gelu{,_backward}` | x (SFPU, derived bounds; `sigmoid_backward` exact) | S4 |
| `log_sigmoid{,_backward}` | | S4 |
| `softmax`, `log_softmax` | x (device composition, derived bound; from 8 tiles) | R2 |
| `softmin` | | R2 |

### `ModuleOps`

| Methods | Device | Item |
|---|:-:|---|
| `linear` and its three backwards | `~` default over `float_matmul` | -- |
| `embedding{,_backward}` | | D4 |
| `conv1d`, `conv2d`, `conv_transpose*`, their backwards, `unfold4d` | | D6 |
| `avg_pool*`, `adaptive_avg_pool*`, `max_pool*` and backwards | | M2 + D6 |
| `layer_norm` | | R3 |
| `attention` | | R4 |
| `conv3d`, `deform_conv2d`, `interpolate`, `ctc_loss`, `rfft`/`irfft` | | not planned until a model needs them |

### `IntTensorOps`, `BoolTensorOps`, `QTensorOps`

| Methods | Device | Item |
|---|:-:|---|
| storage on the device | x `I32`, `Bool` (any store); other int dtypes host | D3 |
| `{int,bool}_{reshape, slice, swap_dims, transpose}` | `~` views, as `float_`'s | D3 |
| `int_{add,sub,mul,div,remainder}{,_scalar}`, `int_neg`, `int_abs`, comparisons, `bitwise_*` | | S5 |
| `int_into_float` | x to F32 (SFPU, exact) | S4 (10.2d) |
| `int_cast`, `bool_into_float`, `bool_into_int` | | S6 |
| `int_sum*`, `int_max*`, `int_argmax`.. | | R1 |
| `bool_and`, `bool_or`, `bool_xor`, `bool_not` | x (SFPU, exact) | D3 |
| `bool_mask_*` | | S5 |
| indexing (`*_gather`, `*_select`, `*_cat`, `*_slice*`, `*_scatter*`) | | D4 |
| `QTensorOps` | | D2 (stays Flex's until then) |

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
| A host L1 write is not ordered against another agent writing the same L1 (an Ethernet transfer landing, a mover) | divergence row AA | closed in `silicon_eth_link` by a read-back fence; open as an API rule -- `Device::write` is posted, and a write another agent may race needs its read-back (X7) |
| The barrier counter in unit 0's L1 keeps an earlier session's count, so every barrier passes at once and multi-unit ops overlap | X4c (found on silicon, once P1 removed the per-step syncs that hid it) | X4c -- closed: zeroed with the session's barrier number whenever unit 0's mover starts (`step34_batching::barriers_count_from_zero_whatever_an_earlier_session_left`) |
| A drain that a descriptor change needs, taken after a list's programs were placed, unpinned them too, so the next placement could evict them under the queued list (an `SFPPUSHC` stack overflow on ttsim) | 10.2's block repeats (programs ~30x smaller changed what the cache evicted) | X8 -- closed: `enqueue_segment` drains before placing |
| A tile wedged by a corrupt run stays wedged: after the backend pulse, every semaphore released (row 65) and the RISC-V semaphore posts (`mailbox::UNWEDGE`), thread 1 takes no instruction (its runner stalls after 29 pushes, one FIFO). Cause: a math instruction waiting for `Src` banks the pulse gave back to the unpackers (reproduced on purpose, row AH). Trying `UNPACR_NOP_SETDVALID` (UNVERIFIED encoding) on the wedged tile took the host down | silicon, 2026-10-01 | closed -- prevented (X4c), detected at open (X5a), recovered by feeding the banks with plain `UNPACR`s (X5b) |

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
