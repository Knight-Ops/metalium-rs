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

## Where things stand (2026-10-01, after 10.0)

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
| 10.1 | Softmax and cross-entropy on the device; `MOP`; op-list traces | S3, S4 (`exp`, `log`), S8, R1 (`max`, `sum`), R2, X2, X4 | `[~]` S3 |
| 10.2 | Activation and math breadth | rest of S2–S4 | `[ ]` |
| 10.3 | Reductions over any dim, device transpose, norms | P1, M2, M3, R1, R3 | `[ ]` |
| 10.4 | Formats and integers | D1, S5, S6, D3 | `[ ]` |
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
| Immediate arithmetic | `SFPADDI`, `SFPMULI`, `SFPDIVP2` | x | | | | | S2, S4 |
| Move / abs | `SFPMOV`, `SFPABS` | x | x | `~` `SFPMOV` | x | x | S2 |
| Sign, exponent, mantissa | `SFPSETSGN`, `SFPEXEXP`, `SFPEXMAN`, `SFPSETEXP`, `SFPSETMAN` | x | `~` `SFPSETSGN` | | `~` `SFPSETSGN` | `~` `SFPSETSGN` | S2, S4, S6 |
| Compare (BH-only `GT`/`LE`) | `SFPGT`, `SFPLE`, `SFPSETCC`, `SFPLZ` | x | `~` `SFPSETCC`, `SFPGT` | `~` `SFPGT` (`RELU`) | `~` | `~` | S2 |
| Conditional execution | `SFPENCC`, `SFPPUSHC`, `SFPPOPC`, `SFPCOMPC` | x | x (scopes) | x | x | x | -- |
| Bitwise | `SFPAND`, `SFPOR`, `SFPXOR`, `SFPNOT` | x | | | | | S5 |
| Integer arithmetic | `SFPIADD`, `SFPMUL24` (BH-only), `SFPSHFT`, `SFPSHFT2` | x | | | | | S5, S8 |
| Lookup and reciprocal | `SFPLUT`, `SFPLUTFP32`, `SFPARECIP` (BH-only) | x | `~` `SFPARECIP` | `~` `SFPARECIP` | `~` `SFPARECIP` | `~` `SFPARECIP` | S4 |
| Casts | `SFPCAST` (`_IntFloat`, `_IntInt`, `_IntAbs`) | x | | | | | S6 |
| Rounding | `SFPSTOCHRND` (`_FloatFloat`, `_FloatInt`, `_IntInt`) | x | | | | | S6 |
| Lane movement | `SFPSWAP`, `SFPTRANSP` | x | | | | | S2, S8 |
| Configuration | `SFPCONFIG` | x | | | | | F2 |
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
| Integer formats (INT32 code 8 measured; INT8/UINT8 not) | `[ ]` | D3 |
| Unpacker transpose / tilize modes, broadcast | `[ ]` | M3, D5 |
| Packer ReLU and edge masking, `PACR_SETREG` | `[ ]` | S1 (opportunistic), D4 |

### Frontend and tracing

Reference: WH `REPLAY.md`, BH `MOPExpander.md`, WH `MOP.md`/`MOP_CFG.md`, BH
`BabyRISCV/AutoTTSync.md` (the expanders and the Wait Gate), `DebugTimestamper.md`.

| Feature | Enc | Helper | Kernel | Sim | Si | Item |
|---|:-:|:-:|:-:|:-:|:-:|---|
| `REPLAY` (record and replay, 32 entries per thread) | x | x | x (SFPU ops) | x | x | X1 |
| `MOP` / `MOP_CFG` (MOP Expander templates) | x | | | | | X2 |
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

- `[-]` **L2CPU tiles.** Harts leave reset only once per power cycle (`L2CPUTile/README.md:30`);
  nothing a Burn backend needs runs better there than on the host.
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
- [ ] **X2 `MOP` / `MOP_CFG`.** Typed templates behind a builder; reconfiguration only
      after `MOPExpanderDoneCheck` (`ManualTTSync.md:57`); Auto TTSync takes the `MOP`'s
      resource declaration (`AutoTTSync.md:26`). Applied to the matmul inner loop, the
      unpacker face loops and `pack_rows`. Gate: MNIST golden bit for bit; program bytes
      down; silicon time measured.
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
- [ ] **X4 Op-list traces** (concepts review G8). `Session::begin_trace`/`end_trace`
      capture each unit's expanded lists into GDDR; `replay` is one descriptor per unit,
      B streaming the list from GDDR; a trace binds its tensors and refuses to replay
      after one is freed. Gate: MNIST golden with steps replayed, steady-state PCIe writes
      per step down to the descriptors.

### P — Prerequisites pulled in when they block

- [ ] **P1 Rank-N tensors** (concepts review G2), minimal: a logical shape stored as
      `prod(leading)` stacked tile grids, a batch stride in `TensorRef` (0 = broadcast),
      last-dim-preserving reshapes as views. Blocks R1 over leading dims, R3, D6, R4.
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
- [~] **F2 An SFPU program builder** (`tt_kernels::sfpu::Program`; the typed
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
  - [ ] `SFPCONFIG` constants (`LReg` 11–14) as a named prologue -- with its first
        user (S4's polynomial constants).
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
      forms the builder emits. **Plan change:** each S item adds the models it
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
- [ ] **S2 Compare, select, sign.** `SFPGT`/`SFPLE`/`SFPSETCC` writing 1.0/0.0, `SFPSWAP`'s
      min/max mode, `SFPABS`, `SFPSETSGN`. Burn: `float_{equal,not_equal,greater,
      greater_equal,lower,lower_equal}{,_elem}`, `float_mask_where`, `float_mask_fill`,
      `float_clamp{,_min,_max}`, `float_abs`, `float_neg`, `float_sign`, `leaky_relu`,
      `hard_sigmoid`, `prelu`. Needs bool tensors on the device (`BoolTensorOps` storage,
      D3).
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
- [ ] **S4 Transcendentals.** Range reduction by `SFPEXEXP`/`SFPSETEXP`/`SFPEXMAN`, then
      `SFPMAD` polynomials or `SFPLUTFP32` (its `LReg[LReg[7] & 15]` destination bug
      handled inside the helper, Tier 2). In order: `exp`, `log`, `sqrt`/`rsqrt`, then
      `log1p`, `powf`, `tanh`, `erf`, `sin`/`cos`. Burn: `float_exp`, `float_log`,
      `float_log1p`, `float_sqrt`, `float_powf{,_scalar}`, `float_powi*`, `float_tanh`,
      `float_erf`, `float_sin`, `float_cos`, then the rest of the trig family;
      `sigmoid`, `gelu`, `log_sigmoid` and their backwards.
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
- [ ] **S8 Lane movement.** `SFPTRANSP`, `SFPSHFT2` for reductions inside a tile (feeds R1).
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

- [ ] **R1 Reductions over any dim.** `sum`, `mean`, `max`, `min`, `argmax`, `argmin`,
      `prod`, full and per dim (today only `sum_dim(0)`, on the B core). Within a tile by
      S8/M2, across tiles by the mover (or NoC atomics). Sum order stated against Flex's,
      as `COL_SUM`'s is.
- [ ] **R2 Softmax, log-softmax, cross-entropy on the device** (was checklist 9.12): max,
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
- [ ] **D3 Integer and bool storage** (INT32, INT8, bool as a format) for `IntTensorOps`,
      `BoolTensorOps`, `QTensorOps`.
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
| `float_add`, `float_sub`, `float_mul` (incl. `[1, n]` row broadcast), `float_mul_scalar` | x (SFPU or mover by size) | S1 |
| `float_sum_dim` | `~` dim 0 only (B core) | R1 |
| `float_slice` | `~` whole tile rows | D4 |
| `float_transpose`, `float_swap_dims` | `~` 2-D view | M3 |
| `float_add_scalar`, `float_sub_scalar` | x (SFPU or mover by size) | S1 |
| `float_div{,_scalar}`, `float_recip` | x (SFPU, within 1 ulp) | S3 |
| `float_remainder{,_scalar}` | | S6 |
| `float_neg`, `float_abs`, `float_sign`, `float_clamp{,_min,_max}` | | S2 |
| comparisons (`float_equal`.. `float_lower_equal_elem`), `float_mask_where`, `float_mask_fill`, `float_is_nan`, `float_is_inf` | | S2 |
| `float_exp`, `float_log`, `float_log1p`, `float_sqrt`, `float_powf*`, `float_powi*`, `float_erf` | | S4 |
| `float_sin`, `float_cos`, `float_tan`, `float_tanh`, hyperbolic and inverse trig, `float_atan2` | | S4 |
| `float_round`, `float_floor`, `float_ceil`, `float_trunc`, `float_cast`, `float_into_int` | | S6 |
| `float_random` | | S7 |
| `float_sum`, `float_mean{,_dim}`, `float_prod{,_dim}`, `float_max*`, `float_min*`, `float_argmax`, `float_argmin`, `float_any*`, `float_all*`, `float_max_abs*` | | R1 |
| `float_cumsum`, `float_cumprod`, `float_cummin`, `float_cummax` | | R1 |
| `float_sort*`, `float_argsort`, `float_topk`, `float_argtopk` | | R1 (late) |
| `float_gather`, `float_scatter_add`, `float_select{,_add}`, `float_slice_assign`, `float_cat`, `float_repeat_dim`, `float_expand`, `float_flip`, `float_permute`, `float_gather_nd`, `float_scatter_nd`, `float_unfold` | | D4, M3 |
| `float_cross`, `float_grid_sample_2d` | | not planned until a model needs them |

### `ActivationOps`

| Methods | Device | Item |
|---|:-:|---|
| `relu`, `relu_backward` | x (SFPU or mover by size) | S1 |
| `leaky_relu`, `prelu`, `hard_sigmoid` | | S2 |
| `sigmoid{,_backward}`, `gelu{,_backward}`, `log_sigmoid{,_backward}` | | S4 |
| `softmax`, `log_softmax`, `softmin` | | R2 |

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
| storage on the device | | D3 |
| `int_{add,sub,mul,div,remainder}{,_scalar}`, `int_neg`, `int_abs`, comparisons, `bitwise_*` | | S5 |
| `int_into_float`, `int_cast`, `bool_into_float`, `bool_into_int` | | S6 |
| `int_sum*`, `int_max*`, `int_argmax`.. | | R1 |
| `bool_and`, `bool_or`, `bool_xor`, `bool_not`, `bool_mask_*` | | S5 |
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
| `SFPLUTFP32` writes `LReg[LReg[7] & 15]`, not `LReg[VD]` | `SFPLUTFP32.md:15` | S4 |
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
