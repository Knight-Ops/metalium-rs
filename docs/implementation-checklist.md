# Implementation checklist

The working tick-list for `RUST_IMPL_PLAN.md`. The plan says *why*; this says *what
is done, what is next, and what must not be forgotten*.

Related: [`hardware-coverage.md`](hardware-coverage.md) (Phase 10 tracker),
[`tt-metal-concepts-review.md`](tt-metal-concepts-review.md) (gaps against tt-metal, G1–G16),
[`burn-backend-parity.md`](burn-backend-parity.md) (Burn backend roadmap, B0–B16).

**Status legend:** `[x]` done and gated by a test · `[~]` partially done, see note ·
`[ ]` not started · `[-]` deliberately not applicable here, with the reason given.

**The two-gate rule.** No item is complete until it passes on the simulator *and* on
silicon. **Silicon now exists here:** two Tenstorrent p150a cards (Blackhole), PCIe-passed
through to this VM, driver `tenstorrent` 2.11.0, firmware bundle 19.14.0.0. Phases 1-8
have passed their silicon gates; the "Where things stand" table says what each still owes.

The first silicon run vindicated the rule in the least comfortable way available. Phase 1
had passed on the simulator for weeks while containing an assumption — 140 Tensix tiles —
that is true of ttsim and of no real Blackhole. On hardware it hung the NoC and took the
**host** down with it, twice. The divergence log's row 35 is that finding. A simulator
gate is evidence about the simulator.

**Before ticking a box, watch the gate fail.** Two gates in this repo were initially
vacuous: the disassembly check matched no lines at all, and a discarded
`.riscv.attributes` section made forbidden instructions decode as `<unknown>`. A
gate you have not seen reject something is not yet evidence.

---

## Where things stand

| Phase | State | Note |
|--:|---|---|
| 0 — Simulator harness | `[x]` | `libttsim` path done; `ttsim-qemu` not needed -- the `tt-kmd` path is gated on real cards |
| 1 — Host addresses the chip | `[x]` | **Gated on simulator and on silicon** (both p150a cards, 7/7). `tt-kmd` done; harvesting read from ARC telemetry |
| 2 — Rust on a baby RISC-V | `[~]` | **Silicon gate passed on both cards** (heartbeat, reset, `pc` snapshot, local RAM + zeroing); I-cache and hot-reload paths open |
| 3 — Encoder + first Tensix round-trip | `[~]` | **SFPU gates, corpus and tracing pass on both cards**; `SFPLOADMACRO` load half pinned on silicon; open question 7 open |
| 4 — Layout | `[x]` | **Silicon gate passed on both cards** |
| 5 — Elementwise binary | `[~]` | **FP32 silicon gate passed on both cards** after three datapath fixes (see Silicon campaign); BF16 silicon-only gate not written. Real element-wise ops are Phase 10 (S1) |
| 6 — Matmul | `[x]` | **Multi-tile matmul, TF32 and BF16, padded shapes, on ttsim and both cards**; three roles concurrent, `Dst` handed over by semaphores; HiFi2-4 at tile level; shapes larger than one run planned and chunked |
| 7 — Burn backend, training | `[x]` | **MNIST MLP trains through `burn-autodiff` with every matmul on a Tensix tile, on ttsim and both cards**; the reduced run's loss curve is bit-identical on all three. `burn-tt` forwards everything else to `burn-flex`, generated from the pinned traits |
| 8 — Multi-chip | `[x]` | **MNIST trains with every matmul sharded across the two cabled cards over Ethernet, reproducing the single-chip golden bit for bit**; on ttsim also round a four-chip ring. Link map from the chips; E1 data mover; throughput is Phase 9 |
| 9 — Performance | `[~]` | **Direction: tensors live in the 32 GiB of GDDR6, loaded at startup.** DRAM, the B data mover and resident role firmware gated on ttsim and both cards; MNIST 224 -> 5.8 ms/step on one card (Flex: 0.5), dataset, weights and activations resident in GDDR; 9.5 asserts the steady-state step's PCIe traffic (6224 B of tensors, 193 524 B written); 9.6 deals GDDR ops over many tiles; 9.7a one host round trip per op per tile, 9.7b op records expanded on the tile, 9.7c resident programs (2.5 ms/step on 8 tiles). Next: 9.8 overlap, with the circular-buffer runtime |
| 10 — Hardware coverage | `[ ]` | **Tracked in [`hardware-coverage.md`](hardware-coverage.md).** Only `MVMUL` runs on the Tensix today; element-wise is on the B core, the SFPU runs no tensor op. Next: 10.0, the SFPU foundation, with today's element-wise ops moved onto it |

---

## Silicon campaign (2026-09-30)

Run with `cargo xtask silicon` (one test per process, fsync'd log). Every bug below
passed on ttsim for weeks; each is now fixed in the code, not worked around.

- [x] **Local RAM hung the NoC** when accessed with its core in reset -- `tt-device`
      refuses it (see Silicon operating notes).
- [x] **A hung program wedged the tile** until the harness learnt to pulse the Tensix
      backend reset; **per-thread state and `Dst` leaked between gates** until it reset
      them too (divergence row 47).
- [x] **Rows 30/35 were our off-by-one**: `REG3_Base_address` pointed one unit before
      the tile header. With it fixed, every datum lands where the specification says,
      on ttsim and silicon.
- [x] **`UNPACR` counts with thread 0's ADCs** (row 45): silicon programs run on T0.
- [x] **`STALLWAIT` waits held the wrong units** (row 46): the consumer is now a
      required argument.
- [x] **The datapath now mirrors LLK's thread split.** `harness::Roles`: unpack on
      T0 / thread 0, math on T1 / thread 1, pack on T2 / thread 2, three role images
      at their cores' default reset PCs (`role_t0..2`), one mailbox each
      (`mailbox::role`), run in order. It works on ttsim too -- math on T1 is the one
      thread ttsim lets read `Dst` -- so the T0/T1 target split is only needed by the
      remaining single-thread gates. `probe_src`, `step9_matmul`, `step8_eltwise`
      and `probe_pack` all run this way, on ttsim and both cards; the latter two
      share `datapath::dst_round_trip_roles`. `probe_unpack` and `step5_corpus`
      stay single-thread on purpose, and say why in their headers.
- [x] **Every datapath encoding checked against LLK** (`tt-isa/tests/llk_crosscheck.rs`).
      Two Wormhole layouts were wrong on Blackhole: `MOVB2D`'s `instr_mod` (the
      whole of row 38) and `MOVA2D`/`MOVB2D`'s `AddrMod` (one bit lower, as
      `MVMUL`). Both are now measured `Bits32_BH.lua` layouts, each licensed by a
      `probe_src` gate that passes on ttsim and both cards. `MOVD2A`/`MOVD2B`/`MOVB2A`
      had the same `AddrMod` shift and are measured layouts too, licensed by
      `step9_matmul::mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole`. So are
      `ELWADD`/`ELWSUB`/`ELWMUL`/`DOTPV`/`MOVDBGA2D`/`SHIFTXB` (`AddrMod`) and
      `ZEROACC` (`AddrMod`, `UseDst32b` at 18, Wormhole `Revert` dropped -- the
      generator now takes a `dropped` field list with reasons). `GMPOOL`/`GAPOOL`
      agree with LLK as drawn. Every Matrix Unit encoding LLK defines now agrees
      with it. Found on the way, on silicon only: one-row `ZEROACC` in FP32 `Dst`
      addresses physical rows (row 51), and with an odd `AddrMod` clears eight
      (row 52).
- [x] **`AddrMod` is three bits on Blackhole** (entries 0..7). All 13 measured
      layouts draw 14..16; `gen-isa` takes a `widened` list, held to the same rules
      as `moved` (listed only if the width really changed). Every `AddrMod` gate also
      runs `addr_mod(4)` -- the third bit alone -- against entry 4, on ttsim and both
      cards; `llk_crosscheck` checks all 13 against LLK's `is_valid(addr_mode, 3)`.
- [x] **`Src` -> `Dst` losses: root-caused -- the chip was never raised to busy.**
      UMD sends the ARC `AICLK_GO_BUSY` whenever it opens a chip; this stack never
      did, so every run computed at the idle operating point (800 MHz, ~0.72 V), where
      the Matrix Unit's `Src` reads drop or misplace datums in chip-specific column
      pairs. Proven with a control: one card sent `GO_BUSY` through UMD passed every
      gate, the idle one failed exactly as before. Fixed in `tt-device`:
      `Device::open` sends `GO_BUSY` and waits for AICLK/VCORE to settle, `Drop` sends
      `GO_LONG_IDLE`, `PowerPolicy::Manual` opts out (divergence row 48). **One
      `Device` per chip**: busy/idle is chip-wide and not reference-counted.
      Everything ruled out on the way (tile defect, burst/throttle, timing, config in
      flight, formats, `LaneConfig`, zero flags, `ZEROSRC`, swizzle, the backend
      pulse, encodings) is recorded in `silicon_measure.rs` m12-m32.
- [x] **Stale `Config` between programs.** tt-metal left `SFPU_Fp32_enabled = 1` on
      a tile and the step 4 gates read the previous run's product. The silicon
      per-thread reset now starts with `backend::reset_config` (the deliberate
      `STATE_RESET_EN` write), and runs on every tile a gate claims, not just the gate
      tile (row 49).
- [x] **Silicon regression clean.** The campaign's last two failures were the
      `MOVB2D` `Move4Rows` twin (row 38, fixed); after the Phase 6 close-out the
      unfiltered run was 129/129 on each card. Later phases record their own counts.
- [ ] **Hazard knowledge as data, for a scheduler** -- see `RUST_IMPL_PLAN.md`,
      "Hazards as data". Today every wait is a full `STALLWAIT` chosen by hand.

---

## Cross-cutting

Set up early; retrofitting is expensive.

- [x] **Pin upstream revisions** — `PINS.toml` records the spec commit, the ttsim
      release, and the tt-metal commit for `cfg_defines.h`. All three are now
      hash-verified: the specification by a digest over the *contents* of the files
      the generators read, since GitHub does not promise source tarballs are
      byte-stable.
- [ ] **Re-sync the spec quarterly.** Recent upstream commits fill in exactly the
      Blackhole `UNPACR`/`PACR` gaps Phase 6 depends on. Check before doing
      silicon-discovery work the docs may have obviated.
- [x] **One newtype per coordinate space** — `NocCoord<Noc0>`, `NocCoord<Noc1>`,
      `Translated`, plus `ChipId` from day one. `ChipId` is now load-bearing rather
      than merely present: it is the bdf device field *and* the stride multiplier
      for a chip's BAR windows, so the day-one decision cost nothing to collect on.
- [x] **Divergence log** — `docs/ttsim-divergence.md`: numbered rows, lettered
      measurements, and numerics notes.
- [x] **Silicon-only suite exists** — `--features silicon`, compiled always so it
      cannot rot.
- [x] **Version control**, so the pinning discipline above is enforceable.
- [x] **CI: simulator suite on every commit.** Deterministic, so no flakes.
      Also fmt, clippy under `-D warnings`, and a type-check of the silicon suite.
- [x] **Test tiers** (2026-10-01). The simulator is for validation; whole Burn
      training runs are the `e2e` feature of `tt-tests` (ignored without it,
      on by default with `silicon`), and the everyday smoke test is
      `cargo xtask silicon --smoke`: burn-tt against burn-flex on the cards.
      Measured: all 59 test binaries ran in 53 s, 80% of it the four training
      runs in `step12_mnist`. Default `cargo test` 54 -> 36 s; the e2e tier
      28 s (CI runs both); smoke 42 s for both cards, where the reduced
      training run takes 1.2 s against ttsim's 25. `xtask silicon` now refuses
      a filter that matches nothing, so a stale smoke entry cannot shrink the
      run unseen. See the README's test tiers.
- [x] **CI: `cargo xtask check-no-sim-in-ship`** wired in.
- [x] **CI: `cargo xtask gen-cfg --check`**, so the committed configuration table
      cannot drift from the pinned header.
- [x] **CI: `cargo xtask gen-isa --check`**, likewise for the instruction table —
      and it re-runs the cross-check between the specification's two descriptions
      of the instruction set, so an exception edited without looking at what it
      explains fails there. It also checks every measured Blackhole override
      (`xtask/src/gen_isa/Bits32_BH.lua`) against the diagram it supersedes and
      against the gate that licenses it.
- [x] **Pre-commit hooks** (`prek.toml`) running fmt, clippy for both workspaces,
      and the generated-table check — the same things CI runs, so a push does not
      fail on something a commit could have caught.
- [ ] **CI: silicon suite nightly, and as a merge gate to main.**
- [x] **Forking while another test thread is inside Burn could hang a child.**
      A forked child keeps every lock as it stood; one held by a thread
      computing a `burn-flex` reference in the parent is held forever. Seen
      once (a training gate's child on a futex for ten minutes; every test
      passed alone). `tt_ttsim::fork_scope` now takes a gate exclusively for
      the fork, and parent-side library work goes through `outside_fork`,
      which shares it (step9, step11, step12). Watched: with the gate removed,
      `a_lock_held_by_parent_work_is_not_inherited_by_the_child` hangs until
      its alarm.
- [x] **`burn-flex` as the second differential oracle.** ttsim is the ISA-level
      oracle; this is the tensor-level one. Needed from Phase 5. **Not
      `burn-ndarray`**, which crates.io now marks `[Deprecated] … use burn-flex,
      burn-cuda, burn-rocm`; `burn-flex 0.21.0` is the supported CPU backend and is
      what `PINS.toml` should pin. Taken as a `tt-tests` dev-dependency at 0.21.

### Hazards to encode in the API, not in comments

This is the concrete answer to "why Rust and not C++"; if it is not done, there is
no answer.

- [x] Manual TTSync load adjacency — emitted as one `asm!` block, never left to LLVM.
- [x] TLB `linked` bit — not expressible; the spec says it is never safe from the host.
- [x] Multicast windows are write-only — a multicast read is fatal.
- [x] RISCV B has no reset-PC override — `set_reset_pc` refuses it.
- [x] `LReg[8..]` unwritable by `SFPLOADI`/`SFPMAD` — refused rather than silently dropped.
- [x] `SFPMUL` requires `VC == 9` — `VC` is implicit, not a parameter.
- [x] **Wormhole's `SFPSTORE` cannot be reached by the Blackhole name.** The two
      encodings differ in where `AddrMod` sits (13..15 against 14..15) and are now
      separate table entries; the superseded one lives under
      `isa::generated::defs::wormhole`, so using it has to be deliberate.
- [x] **`INSTRN1_BUF_BASE`/`INSTRN2_BUF_BASE` from T0/T1/T2 hangs the core.**
- [x] **A dropped `Window` silently leaked its TLB window.** It had no `Drop`, so
      a gate that allocated and did not `free_window` shrank the pool for the rest
      of the process, and the diagnostic was a bare `OutOfBounds { offset: 0,
      len: 0 }` from an unrelated function. It cost the Phase 1 silicon gate a run.
      The obvious fix does not typecheck (`Drop` cannot take `&mut Device`), and a
      window that borrowed its device would make holding several at once a borrow
      conflict. **Fixed with a shared pool:** the free list and shadow table live in
      an `Arc<Mutex<Pool>>`, every `Window` holds a handle, and its `Drop` returns
      the index and drops the shadow. `free_window` is now the explicit spelling of
      the same thing. A window from another `Device` is refused
      (`TransportError::Hazard`) before anything reaches the transport. Unit tests:
      three passes over the whole pool by dropping, and the cross-device refusal;
      the first watched failing with the release removed. On silicon,
      `window_exhaustion_is_an_error_not_a_panic` now *drops* its 201 windows, and
      the scrub that follows on the same device is the check (both cards).
- [x] **A fused-off Tensix tile cannot be named by accident.** The predicate that looks
      obvious is the safe one: `grid::is_tensix_geometry(x, y)` answers "could a Tensix
      tile ever be here" and cannot hang anything, while `grid::Tensix::contains(x, y)`
      answers "does *this chip* have one" and requires a value obtainable only from the
      chip's ARC. A count is not a constant: `Tensix::tile_count()` replaced
      `TENSIX_TILE_COUNT`, so there is no 140 left to iterate. This is the hazard that
      cost two host crashes; it is encoded in two type signatures rather than in a
      comment saying "mind the harvesting".
      `tt_isa::tensix::PushesTo<Th>` is implemented for exactly the six non-hanging
      cells of `PushTensixInstruction.md:5-9`, so `push::<RiscvT0, Thread1>` does not
      compile. Marker types rather than a runtime check because a hang cannot be
      caught: it is an unrecoverable lockup, not a fault. Watched failing — adding
      the `RiscvT0 -> Thread1` impl turns exactly that `compile_fail` doctest red and
      no other. `Core::can_push_tensix` was dead before this and is now tied to the
      type-level table by a test, and RISCV NC implements neither trait.
      The cross-check it forced on the firmware found a live bug: `step4_tensix.rs`
      never wrote `mailbox::THREAD_INDEX`, so the firmware read 0 and configured
      **SEC0** — thread 0's `Dst` mapping — while running on T1. It passed only
      because `fmt = 0` is also the reset default.
- [x] **`STALLWAIT` block and condition masks are named, and paired.**
      `tt_isa::backend::{block, cond}` carry the Blackhole numbering — which the
      page says differs from Wormhole's and tells software to abstract — and
      `wait_for_unpacker0`/`wait_for_packer`/`wait_for_sfpu`/`wait_for_matrix` emit
      the block bit each condition needs. A condition set without its block bit
      waits without stopping anything, which reads as a race rather than a missing
      bit. `ZEROACC` runs on the **Matrix Unit**, so its wait is C4+B6 even though
      everything around it in a `Dst` scrub is SFPU work.
- [x] **A `WRCFG` to `STATE_RESET_EN` is refused.** `BackendConfiguration.md:40`:
      writing *anything* to that word, by any instruction but `RMWCIB`, instantly
      zeroes every `Config` word below `GLOBAL_CFGREG_BASE_ADDR32`. It is a
      whole-configuration reset wearing the shape of an ordinary field write.
      `backend::write_word` rejects it and a test watches the rejection.
- [x] **A misaligned 128-bit `WRCFG` is refused rather than rounded.** The
      instruction masks both indices with `& ~3` instead of faulting, so an
      unaligned request silently writes four words elsewhere.
- [x] **`UnpackToDst` clobbers `SrcA[Bank]`** — `UNPACR_Regular.md:441`.
      `tt_isa::matrix::Banks::unpack_to_dst` requires `SrcA` to be `Empty`, so it
      cannot run between the partial unpacks of one operand; a `compile_fail`
      doctest watches it, and goes red when the rule is removed.
- [ ] `Dst` exclusivity across the three Tensix threads. *(Phase 5–6.)*
- [x] **`SrcA`/`SrcB` bank ownership handshake.** `tt_isa::matrix::Banks<A, B>`
      tracks each operand as `Empty` / `Filling` / `Loaded` under a lockstep
      discipline: every bank the unpacker hands over is handed back by exactly one
      flipping consumer before the unpacker writes again. Four `compile_fail`
      doctests, each watched turning red when its forbidden impl is added. The
      discipline was not hypothetical: the `Src` probes hit the mix-up it forbids
      by accident — a `FlipSrc` unpack, a non-flipping `MOVA2D`, a second unpack
      into bank 1 while the move read bank 0 again, and the first run's data back
      with no diagnostic. `SETDVALID` and `CLEARDVALID(Reset)` are left out of the
      safe surface on ttsim's and tt-metal#22383's say-so.
- [x] `NOC_CMD_WR_INLINE` must never target an L1 address. `niu::Command::MmioInline`
      refuses an L1 destination (Phase 8).
- [x] `NOC_CMD_L1_ACC_AT_EN` must always be `false`: `niu::Command` has no way to
      set it.

### Tolerance policy

- [x] Never a guessed epsilon — assert bit patterns against the documented model.
- [x] Derive expected values from `Miscellaneous/FMA/fma.c` (`fma_model_bh`).
      `tt_isa::numerics::fma_bh` is a port, checked against the C itself — which
      `crates/tt-tests/build.rs` compiles — over 200 000 cases. `fma.c` is now inside
      the digest `PINS.toml` pins, so the oracle cannot change underneath the tests.

---

## Phase 0 — Simulator harness

- [x] `tt-ttsim-sys`: the ten `libttsim_*` entry points, hand-written against
      `src/libttsim.map`. There is no header to bindgen.
- [x] `dlopen` at runtime rather than link-time.
- [x] Singleton enforced in the type system — `OnceLock` handing out one
      `!Send + !Sync` token. `cargo test`'s thread-per-test default would otherwise
      corrupt state in ways that look like simulator bugs.
- [x] DMA callbacks installed *before* `init`, even though nothing DMAs — DMA with
      no callbacks is itself fatal.
- [x] `Transport` trait with `tick`, and a `LibTtsim` implementation.
- [x] Validation layer mirroring ttsim's decode map, applied *before* each call.
- [x] `fork_scope` isolation, so a `_Exit` becomes a failing assertion.
- [x] `fatality.rs` pins five accesses as genuinely fatal, so the validation layer
      cannot quietly become unnecessary.
- [x] `xtask fetch-ttsim` with a pinned tag and SHA-256, now over two assets —
      `libttsim_bh.so` and `libttsim_bh_x2.so` — from the one tag, so the
      single-chip and dual-chip simulators cannot drift apart.
- [x] **Gate (sim):** init succeeds, config offset 0 reads `0xB140_1E52`, the clock
      advances, a bad access is rejected before reaching the library.
- [-] **Gate (silicon):** none — this phase is simulator infrastructure by definition.

- [-] **`ttsim-qemu`** (a QEMU fork exposing ttsim as `/dev/tenstorrent/0`, to
      exercise the real ioctl and mmap paths): not needed. Two p150a cards exist and
      the `tt-kmd` path is gated on them directly.

---

## Phase 1 — Host can address the chip

- [x] TLB window allocator; window 201 reserved for the kernel driver even on the
      simulator, so allocation behaviour matches silicon.
- [x] 96-bit window configuration, **both geometries encoded independently** — the
      4 GiB layout is not a shift of the 2 MiB one.
- [x] Shadow table for the configuration registers, which are write-only.
- [x] Transfers split at window boundaries; a straddling access is fatal.
- [x] `ordering = Strict AXI`, matching `ethdump.c:239`.
- [x] **Gate (sim):** pattern round-trips through two *different* windows, on two
      tiles, across a window boundary; every Tensix tile addressable without aliasing.
      Reads 140 tiles on ttsim and 120 on these cards — the count comes from
      `Tensix::tile_count()`, not from a constant, which is the fix for row 35.
- [x] **Gate (sim) under `bh_x2`** — two chips, two `Device`s, one process.
      `crates/tt-tests/tests/step2_multichip.rs`: the same tile coordinate at the
      same address on both chips holds different data, window index 0 on each does
      not collide, and the step 2 round-trip and full-grid sweep both repeat on
      chip 1. That sweep uses `Tensix::FULL` explicitly and says why: it drives the
      dual-chip simulator directly rather than going through the backend, so on silicon
      it would need each chip's own ARC read — two cards are two ASICs with independent
      fuses, which is the divergence this gate claims to look for. `ChipId` is no longer assumed — `Device::open` takes it from
      `Transport::chip()`, so a device cannot claim a chip it does not address.
      Watched failing three ways: against the single-chip build via
      `TT_TTSIM_LIB_X2`, with the chip dropped from `chip_bar_base`, and with
      `Device::open` ignoring the transport.
- [x] **`tt-kmd` crate** — hand-written against a *pinned* `ioctl.h` rather than
      bindgen'd. The pin is unlike the other three in `PINS.toml`: they pin
      specifications, this pins an interface to a program running on this machine, and
      it is only meaningful if it names the version actually loaded. It does — the
      header at `ttkmd-2.11.0` is byte-identical to `/usr/src/tenstorrent-2.11.0/ioctl.h`
      — and the hash is not the real guard anyway: `Kmd::open` asks the driver via
      `GET_DRIVER_INFO` and refuses a mismatch, because a matching header is no evidence
      about the loaded module. `crates/tt-kmd/tests/abi_layout.rs` checks every struct
      offset against the C header, watched failing.
  - [x] Opened **without** `O_APPEND`, deliberately. tt-kmd reads the open flags as a
        power policy (`ioctl.h:363-380`): without it the driver requests high power
        immediately; with it the state starts at zero and every feature — including
        `TT_POWER_FLAG_TENSIX_ENABLE`, whose zero clock-gates the Tensix array — must be
        asked for explicitly. Opening with `O_APPEND` and forgetting produces a chip on
        which nothing runs and nothing says why.
  - [x] `SET_NOC_CLEANUP` registered, but **only after** the grid is known — see the
        ordering note in Silicon operating notes below.
  - [-] `GET_HARVESTING` (`0xFA01`) is a **dead stub** in 2.11.0: `chardev.c:786-787` is a
        bare `break`, there is no handler, and no struct for it in `ioctl.h`. Harvesting
        comes from ARC telemetry instead.
- [x] **Gate (silicon):** the Phase 1 round-trip on real p150a cards — 7/7 on **both**,
      single-threaded, including the full 120-tile sweep and window exhaustion. Not
      passed on the first attempt: see Silicon operating notes.
  - [x] `QUERY_MAPPINGS` takes `&mappings[0].mapping_size`, not the struct base.
  - [x] `SET_NOC_CLEANUP` re-asserts reset if the process dies — any real runtime
        needs it. Returns `EINVAL` on older `tt-kmd`; ethdump ignores the result.

**Why not bindgen, in the end.** The plan called for it, and the reasoning was sound:
`ioctl.h` is real C with structs, which is what bindgen is good at, and `ethdump.c` only
inlines a partial copy. What decided against it is that the ABI needs *two* guards, and
bindgen provides neither. The header must be the one the loaded module was built from —
pinned by hash in `PINS.toml`, checked byte-for-byte against the DKMS source — and the
running module must agree, which only `GET_DRIVER_INFO` can establish at runtime. A
generated binding would have compiled cleanly against a divergent vendored copy (UMD ships
one) and produced a layout that runs and is wrong. `abi_layout.rs` checks the hand-written
structs against the C header instead, which gives the same protection bindgen would while
leaving the pin and the runtime check as the actual guards. (Separately, and still true:
not bindgen for `cfg_defines.h` — see the note on `parse_cfg_defines`; it silently drops
the two oversized masks and discards the section comments that decide `Config` vs
`ThreadConfig`.)

---

## Silicon operating notes

Learned the hard way during Phase 1's silicon bring-up. Three host crashes bought these;
they apply to every later phase's silicon gate, so they live here rather than in Phase 1.

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

---

## Phase 2 — Rust executing on a baby RISC-V

- [x] Target `riscv32im-unknown-none-elf`. **No custom target JSON** — the plan
      assumed one; the built-in target is RV32IM with no compressed instructions,
      which is exactly right.
- [x] Linker script, `_start`, stack in local data RAM, `.bss` zeroing.
- [x] Reset sequencing: read-modify-write on `SOFT_RESET_0`, reset-PC override.
- [x] Panic strategy: status word plus code to L1, then spin. Not `ebreak`, which
      halts the core and needs an external agent to resume.
- [x] `spin()` avoids `core::hint::spin_loop()`, which emits `pause`
      (Zihintpause) — an extension Blackhole does not list.
- [x] Instruction-set gate: no `c.*`, `lr.*`/`sc.*`, `fdiv`, `fsqrt`, `div`, `rem`,
      `fence.i`, no undecodable instructions, no unrecognised `fence` ordering.
      `fence.i` was a live hole until now: it is a distinct mnemonic, so the
      `fence` operand screen never saw it and `forbidden_reason` had no arm for
      it. The spec calls executing it `NonContractualBehavior` and ttsim refuses
      it outright. Watched rejecting a deliberately planted `asm!("fence.i")`.
- [x] **Gate (sim):** heartbeat climbs monotonically; a core held in reset does
      nothing; releasing one core does not disturb the others.
- [x] **`pc` snapshot cross-check** — silicon only, ttsim does not model the
      registers (divergence row 3). `step3_heartbeat::pc_snapshot_lands_in_the_loaded_image`
      passes on both cards. The snapshot of a `j .` loop samples as the loop address
      *and* loop + 4, which is what "speculative" means in practice.
- [~] **Gate (silicon): the highest-value silicon gate in the plan.** Reset
      sequencing, I-cache invalidation and the local-RAM zeroing window are all
      things a simulator may model loosely, and all three land here.
      **`step3_heartbeat` 7/7 on both cards** (2026-09-30): heartbeat, held-in-reset
      control, reset round trip, `pc` snapshot inside the image, two tiles (the far
      one taken from the grid rather than the fused-off `(16, 11)`), and both
      refusals. Reset sequencing and the zeroing window are closed (below); the
      I-cache invalidation path is what keeps this `[~]`.
- [x] **The slow-path local-RAM aperture took the host down** -- because it was accessed
      with the owning cores held in reset. Encoded in `tt-device` (see Silicon operating
      notes) and **confirmed on both cards**: `silicon_local_ram` 4/4 on each. With cores
      parked, all five RAMs round-trip in full through the aperture with no aliasing
      (B/NC 2048 words, T0/T1/T2 1024); `DISABLE_RESET` reads back; releasing T0 with its
      bit clear zeroes all 1024 words, and with it set keeps all 1024 -- the documented
      behaviour, observed for the first time (open question 6, zeroing half).
- [ ] **I-cache invalidation path.** Avoided by construction today — code is
      written before reset is released, and leaving reset invalidates the cache.
      Needed the moment anything reloads a *running* core: write the 5-bit mask to
      `RISCV_IC_INVALIDATE_InvalidateAll` (bit 0=B, 1=T0, 2=T1, 3=T2, 4=NC). That
      register is **not NoC-accessible and not accessible to RISCV NC**, so NC
      depends on another core. Invalidation does not flush the pipeline.
- [~] **Local-RAM zeroing window.** Still avoided by construction — nothing is
      staged into local RAM over the NoC — but the addresses are no longer absent
      from the code. `Core::local_data_ram_noc_address`, `local_data_ram_noc_window`
      and `disable_reset_bits` are in `tt_isa::tensix`, unit-tested against all
      four of Blackhole's mutually inconsistent per-core orderings, and the
      T-core aperture being twice its backing RAM is encoded rather than noted.
      **What the simulator proves:** that it models none of it — neither the
      aperture nor `DISABLE_RESET`, in either direction — watched with a control
      so the refusal is evidence (divergence rows 25 and 26). **What it cannot:**
      anything behavioural, including whether setting the bit abolishes the
      window as `SoftReset.md:116` says. The staging API and its silicon gates
      are the remaining work.
  - [ ] `Device::load_and_start_staged` with `SuppressZeroing` / `WaitOutZeroing`,
        so a caller cannot get the ordering wrong, plus the `#[cfg(feature =
        "silicon")]` pair: staged data survives release with the bit set, and is
        deterministically wiped without it.
  - [ ] **Open:** what occupies the upper 4 KiB of a T-core's 8 KiB slow-path
        window? The memory map lists two identically-labelled rows per T-core and
        the RAM is only 4 KiB. Refused by `local_data_ram_noc_window` for now;
        resolvable only on silicon.
- [ ] **Multi-core loader mutual exclusion.** `&mut self` is sufficient within one
      process; the soft-reset register has no atomic bit operations, so anything
      else touching the same tile needs coordination at a higher level.
- [ ] **Measure I-cache capacities** (open question 1). Undocumented for Blackhole;
      affects hot-loop sizing and whether T1 is viable for large code.

---

## Phase 3 — Instruction encoder + first Tensix round-trip

- [x] SFPU encoders: `SFPLOADI`, `SFPMAD`/`SFPMUL`, `SFPNOP`, `SFPSTORE`, from
      `Diagrams/Src/Bits32.lua`.
- [x] Blackhole `AddrMod` position (13..15, not Wormhole's 14..15) pinned by test.
- [x] `.ttinsn` encoding (rotate left by two) implemented and tested; the firmware
      uses the equivalent `sw`, which disassembles cleanly.
- [x] Push path from a T-core; Manual TTSync before reading `Dst`.
- [x] `Dst` read path, including the swizzle transform.
- [x] **Gate (sim):** the SFPU returns `0x40C0_0000`; seven host-supplied operand
      pairs match the host's own FP32 multiply bit-for-bit.
- [x] **`tt-isa-gen`** — `cargo xtask gen-cfg` parses `cfg_defines.h` into 820
      typed constants across seven sections. `Config` and `ThreadConfig` are
      distinct types, so the write-instruction convention cannot be got wrong.
      The generated table is committed and checked against the pinned header.
- [x] **Set `Dst` access format deliberately** rather than inheriting the reset
      default, which is the first real use of the generated table — and a test
      proves changing it changes the readback, so the path is not inert.
- [x] **Broad instruction corpus, encoded bit-exact.** `cargo xtask gen-isa`
      parses `Diagrams/Src/Bits32.lua` into 148 instruction encodings (161 since the
      measured Blackhole layouts, `isa/generated.rs`'s header) and 19 datum
      layouts, each cross-checked against the hand-written `TT_*(…)` syntax block
      on the page that embeds its diagram. The two sources agree on names, widths,
      signedness **and slot bit positions** across 167 pairs, with ten documented
      exceptions. Provenance is derived from which tree embeds the diagram, plus
      the measured Blackhole overrides; `isa/mod.rs` asserts the split (39
      Blackhole, 24 shared, 24 superseded, 61 Wormhole-only and `UNVERIFIED`, 13
      measured).
- [~] **Broad instruction corpus, *executed* against ttsim.** Split from the
      above deliberately: encoding is host-side and covers everything, executing
      needs surrounding state most instructions do not have yet. The generic
      `corpus` firmware runs a host-staged program and dumps `Dst`, so adding a
      case is host-side data. Eight gates so far, covering `SFPLOADI` modes,
      `SFPMAD`, `SFPMOV`, `SFPABS`, `SFPSTORE` lane placement, and the numerics
      entries C and D of the divergence log. `UNPACR`/`MVMUL`/`PACR` execution
      waits on *our* readiness, not the simulator's: `libttsim_bh.so` carries
      `tensix_pacr`, `tensix_unpacr` and `tensix_mvmul` execute handlers, a block of
      packer-specific refusal strings, and `fp32 to bf16/bfp8` conversion
      diagnostics. **`UNPACR` and `PACR` now execute.** `probe_unpack.rs` stages a
      tile in L1, configures unpacker 0 for `UnpackToDst`, and checks every datum
      against the functional model; `probe_pack.rs` packs `Dst` back out and
      completes the **L1 -> `Dst` -> L1 round trip**. Both carry controls watched
      failing: a corrupted staged datum moves exactly one element at each stage, and
      `ReadIntfSel` is checked to pack exactly the rows it names and leave a
      sentinel beyond them untouched -- `PACR.md`'s central claim, and the one its
      "basic" self-labelling makes worth testing hardest. `ReadIntfSel == 0` is
      confirmed identical to `0b1111`, which the page's branch-free
      `(1 << RowsRemaining) - 1` advice depends on. `MVMUL` stays open: it needs
      `SrcA`/`SrcB` and is genuinely Phase 6.
- [x] **Stand up tracing** (silicon only; ttsim models the counter and not the
      stream, divergence row 54). `tt_isa::tensix::timestamper` holds the register
      map, `tt_device::trace` configures buffer 0 (`configure_trace`: pulse the
      sticky reset, clear `full`/`overflow`, buffer 1 off), decodes it
      (`read_trace`, refusing an overflowed stream) and reads the counter with the
      documented retry loop (`wall_clock`). The role firmware records `START`,
      `PUSHED` and `RETIRED` as 128-bit events -- whole 16-byte writes, so three
      cores share one stream -- when its mailbox's `TRACE` word is set;
      `Run::traced` sets it and returns `Outcome::trace`. First use,
      `step10_matmul_tile::the_timestamper_shows_the_roles_overlap`, both cards:
      released together, the three roles start within ~800 cycles and overlap
      throughout a ~1300-cycle two-round matmul; released in order, ~57 000 cycles
      of host latency separate them (row 53, measured).
- [~] **`SFPLOADMACRO`.** The simulator's refusal is now *watched* rather than
      quoted: `step5_corpus.rs` pushes one, asserts the child dies, and runs a
      control program of the same shape that survives. A silicon-side test of what
      it actually does is still open.
- [ ] **Probe whether ttsim models the documented hardware bugs** (open question 7)
      or the intended behaviour. Either answer is workable but changes what the
      simulator gate proves.
- [x] **Gate (silicon):** the corpus and SFPU gates pass on both cards; the one
      silicon-only behaviour, `SFPLOADMACRO`'s load half, is pinned (`step5_corpus`).

---

## Phase 4 — Layout before compute

Sequenced before compute deliberately: every op sits on it, it is independently
testable, and it forces the hardest architectural decision — how tiled layout and
block-float formats meet Burn's strided tensor model — while that is still cheap to
change.

**A tile is not a 32×32 square.** `UNPACR_Regular.md:182` treats a tile's datums as
a flat `W`/`Z`/`Y`/`X` array with `X` fastest, dimensioned by a `TileDescriptor` in
backend configuration. The familiar four-16×16-face tile is a *choice of
descriptor*, so `tt_isa::tile` models the descriptor and `Layout::tt_metal_32x32` is
one preset over it. Hardcoding 32×32 would have baked in an assumption the
specification does not make.

- [x] Row-major strided ↔ tiled. `crates/tt-layout`: `TensorView` carries shape and
      **element strides**, so Burn's model is the input type rather than something
      adapted to later. A transposed view is gated.
- [x] Padding for shapes that are not a multiple of the tile extent. `PadValue::Zero`
      is named rather than assumed — zero is right for accumulation and wrong for a
      min-reduction, and only the caller knows which. `detilize` drops it.
- [x] FP32/BF16/FP16 conversion against the documented bit patterns. **Not** the
      `Dst`/`Src` layouts — see the correction below. BF16 has both documented
      rounding modes because the packer offers both. FP16 conversion *refuses*
      rather than approximates: the coprocessor has no NaN encoding and reads
      `Exp == 31` as a finite value, so `fp32_to_fp16` returns
      `Err(NoNanEncoding)` rather than silently substituting a number.
- [x] **Gate (sim), host-side:** `crates/tt-layout/tests/roundtrip.rs`, proptest with
      a deterministic RNG (CI is flake-free on purpose, and proptest seeds from the
      OS by default), plus the named awkward shapes `[13, 47]` and `[1, 1, 1024]`.
- [x] **Gate (sim), against the specification:** `crates/tt-layout/tests/placement.rs`
      re-derives `FirstDatum` and the datum address from `UNPACR_Regular.md:55-212`
      *without* calling `TileImage`, and checks every coordinate of seven
      descriptors. **This gate exists because the round-trip gate is vacuous alone:**
      watched passing unchanged with the face convention transposed, which
      `placement.rs` catches. A round trip only proves tilize and detilize agree.
- [x] **Gate (sim), through the transport:** `crates/tt-tests/tests/step7_layout.rs`
      stages a tiled image in a Tensix tile's L1 through one window and reads it back
      through another, including a grid filling the top of L1. Watched failing: a
      negative control corrupts one datum in L1 and asserts the recovered tensor
      differs, in exactly one element.
- [x] **Numerics pinned to the document, not to an epsilon.** `bfp8_to_bf16` is
      checked **exhaustively** over all 65 536 inputs against the C in
      `FloatBitPatterns.md:117-134`. The documented traps are named tests: FP16
      `Exp == 31` is finite, and BFP8 `Sign = 1, Mag = 0` is −2¹²⁸ rather than −0, so
      `-0.0` does not survive a BFP8 round trip.
- [~] **BFP-ready without BFP.** Datum widths are in **bits** so BFP4/BFP2 need no
      signature change, and `TileImage` sizes and places the shared-exponent section
      for BFP8 — tested — though no BFP encoder exists. The packer's encode direction
      is Wormhole-only and cannot be checked against anything until Phase 6.
- [x] **Gate (silicon):** passed on both cards (two tests).
      **The alignment scope is narrower than the plan first assumed.**
      `WormholeB0/NoC/Alignment.md:19,23` says data travelling *from the host via
      PCIe to an L1 address* has **no alignment restrictions at all**, so the host
      staging path cannot violate anything. The C16 congruence applies when an L1
      address is the *source* — the unpacker and packer driving the NoC — which is
      Phase 6. The silicon test therefore asserts the documented "Any" across eleven
      deliberately misaligned tile bases; a failure is a finding against a
      Wormhole-sourced page Blackhole does not carry.

- [x] **Which `Z` plane is which face of a tile is a convention, not a
      specification**, settled by the tile matmul: the unpacker takes face `z` as
      the `z`-th `XDim * YDim` datums (its ADC `Z`), `tt_layout` puts face
      `(z / 2, z % 2)` there -- row-major faces of row-major datums, as LLK -- and
      the product is right on both targets and wrong with the faces transposed.
      `placement.rs` pins the choice.

---

## Phase 5 — First real kernel: elementwise binary

Eltwise before matmul deliberately: it exercises unpack → SFPU → pack with no
`SrcA`/`SrcB` bank handshake, no fidelity phases, no RWC choreography.

- [x] Unpack → SFPU → pack for a binary op. `crates/tt-tests/tests/step8_eltwise.rs`:
      one `UNPACR` fills `Dst` rows 0..8, the kernel reads operand A from row group 0
      and operand B from row group 4, and one `PACR` writes the result back to L1.
      Multiply, add and subtract. The kernel is two passes because `SFPLOAD` and
      `SFPSTORE` each reach 32 of a four-row group's 64 datums — the even columns or
      the odd ones — and a separate gate pins that the two passes cover every datum
      exactly once, so a half-written tile cannot pass unnoticed.
- [x] **Gate (sim):** differential, with **two oracles making two different claims**.
      `tt_isa::numerics::fma_bh` says what a pair of datums produces, bit for bit;
      `burn-flex` says which pairs there should be. The operands for the Burn gate
      are small integers so the two cannot disagree — where they *would* (denormals,
      NaN, the overflow rule) Burn is the wrong oracle and the named cases are the
      right one. No epsilon anywhere.
  - [x] Include denormals — asserted to flush, and the model predicts the flush, so
        the gate checks the pair rather than the device alone.
  - [x] Include NaN — canonicalisation to `0x7FC0_0000` asserted, along with a check
        that a NaN result actually occurred.
  - [x] Assert exact equality against `fma_model_bh`. `tt_isa::numerics::fma_bh` is
        a port of it; `crates/tt-tests/tests/fma_oracle.rs` compiles the real
        `Miscellaneous/FMA/fma.c` and checks the port against it over 200 000
        deterministic cases and all 2 744 edge-case triples. `PINS.toml` now hashes
        that file, so the oracle cannot drift — watched failing before re-pinning.
  - [-] **BF16 is not reachable on the simulator.** ttsim declines `UnpackToDst` for
        every 16-bit and block-float input format (divergence row 31), so the BF16
        kernel is silicon-only until the packer offers another route into `Dst`.
        Phase 6 found a BF16 route into `Src`, which the Matrix Unit can use but the
        SFPU cannot.
      **Controls watched failing:** expecting the host's `f32` multiply instead of
      the model; an empty kernel; swapped operand row groups; a reversed Burn
      operand.
- [x] **Gate (silicon):** the FP32 suite passes on both cards, after the three
      datapath fixes in the Silicon campaign.

---

## Phase 6 — Matmul

**The schedule risk of this project.** Precisely where the Blackhole tree is
thinnest: `Unpackers/` and `Packers/` do not exist in it at all, `SrcASrcB.md` and
`RWCs.md` are absent, and `SrcASrcB.md` is not even *linked* from Blackhole — so
fidelity phases are documented in exactly one place and it is the wrong tree. BH
`PACR.md` is self-labelled "basic" and admits its `ReadIntfSel` interaction is
undocumented.

- [x] **`UNPACR` into `SrcA`/`SrcB`.** `crates/tt-tests/tests/probe_src.rs`: FP32 in
      L1 → TF32 or BF16 in `Src` on both unpackers, and BF16 in L1 → BF16 in `Src`,
      each observed through `MOVA2D`/`MOVB2D` and asserted bit for bit against the
      documented truncation (`tt_isa::tile::fp32_to_tf32`). FP32 *into* `Src` is
      refused as the `UndefinedBehavior` `UNPACR_Regular.md:580` says it is, and
      that refusal is a gate with a surviving control. Negative control: one
      corrupted staged datum moves exactly one `Src` element. Watched failing with
      the truncation removed and with the hidden base mis-sized.
  - [x] **Configuration surface mapped per field**, at zero and at one — row 28's
        register sweep cannot see value-dependent refusals. Unpacker 1 is mostly
        refused (row 36) but none of the refused fields is read on this path.
  - [x] **The hidden output base is 16 bytes, applies on the `Src` path too, and
        scales with the *input* width** (row 35). Corrected on the data side.
  - [x] **A BF16 route exists** — FP32 or BF16 in L1 into `Src` — which the `Dst`
        path never had (row 31). It serves the Matrix Unit, not the SFPU, so the
        Phase 5 BF16 eltwise gate is still silicon-only.
- [x] **Double-bank handshake** — via `UNPACR` `FlipSrc` and `MVMUL`/`SETRWC` flips,
      not `SETDVALID`/`CLEARDVALID`; see the hazard list above.
- [x] **`MVMUL`, one block.** `crates/tt-tests/tests/step9_matmul.rs`, eight gates:
      identity; small integers against the model *and* `burn-flex` (with a
      transpose control); `+=` accumulation; `DstRow` placement with the untouched
      rows asserted zero; phase 0 alone; all four phases through
      `FIDELITY_BASE_Phase`; all four through the RWC `FidelityIncr`; and the
      Blackhole `AddrMod` position. Watched failing with the operands transposed and
      with the full-precision answer expected from phase 0.
- [x] **Fidelity phases, and a bit-exact oracle for them.** ttsim models the phases
      exactly as `MatrixUnit.md:143-165` describes — each phase alone gives exactly
      its partial product. `tt_isa::numerics::mvmul_reference` ports the model and
      **returns `None` unless every product and sum is exact**, because `MVMUL.md`
      calls its float model "a rough guide" to order; the gates choose operands in
      that regime, and the reference caught one set that was not.
- [x] **Blackhole Matrix Unit encodings.** `MVMUL`'s `AddrMod` is bits 14..16, not
      the Wormhole diagram's 15..16 (row 42), so the generated encoder applied the
      wrong modifier for every non-zero value, silently. **Now fixed through the
      generator, not beside it:** `xtask/src/gen_isa/Bits32_BH.lua` holds measured
      Blackhole layouts in `Bits32.lua`'s own dialect, each with a row in
      `measured.rs` naming what it supersedes, which fields moved, and the gate that
      measured them. `gen-isa` refuses an override that supersedes anything but a
      `WormholeOnly` diagram, moves a field it does not list (or lists one it does
      not move), cites a gate that does not exist, or survives the specification
      documenting the instruction for Blackhole. Each refusal has a mutation test,
      and the missing-gate one was watched end to end. The result is
      `Provenance::Measured`, a third status beside documented and `UNVERIFIED`.
      The sweep of every other Matrix Unit instruction is done: 13 measured
      layouts, `ZEROACC` included (Silicon campaign; rows 40, 42).
- [x] **`PACR` out of a `MVMUL` result**, on ttsim and both cards -- the first
      gate in which all three roles work. `datapath::pack_rows` packs any number
      of `Dst` rows as one `PACR` per aligned group of four: the packer's input
      `Ystride` is one FP32 row (64 bytes), a pack `AddrMod` entry steps ADC Y by
      four between `PACR`s, and only the last carries `Last`, so the output
      address generator keeps appending rather than restarting at `L1_Dest_addr`.
      A final partial group gets `(1 << remaining) - 1`. Gates:
      `step9_matmul::the_packer_writes_the_matmul_result_to_l1` (L1 = `Dst` dump
      = `mvmul_reference`, sentinel intact past row 8) and
      `a_partial_final_group_packs_only_its_rows` (six rows, nothing past them).
      Watched failing with the Y step removed (L1 row 4 repeats row 0) and with an
      empty math body. `PCK0_ADDR_BASE_REG_0_Base` is left at reset zero: it is
      register 16, which ttsim does not model (row 28).
- [x] **Concurrent roles, and the `Dst` hand-off.** `Run::concurrent` runs a
      setup program on thread 0 (the `Dst` clear and `SEMINIT`s) and then releases
      all three roles with **one** write to `SOFT_RESET_0`
      (`Device::load_and_start_together`); `tt_isa::sync` carries the math -> pack
      hand-off (`post_after` = `STALLWAIT` + `SEMPOST`, `take` = `SEMWAIT` +
      `SEMGET`, B1 always blocked as `SEMWAIT.md` recommends, `Semaphore` a
      newtype over `0..8`). Programs now live in fixed slots of
      `mailbox::PROGRAM_REGION` (8192 words each, up from 256 inside the mailbox);
      `step9_matmul::a_program_longer_than_the_old_mailbox_limit_runs_to_the_end`
      pushes 2000 `SFPNOP`s before its `MVMUL`, watched failing with the firmware's
      bound put back. Gates in `step10_matmul_tile.rs`, ttsim and both cards:
      three unpack/`MVMUL` rounds through two banks accumulate and pack correctly
      concurrently; in order, two rounds finish and three deadlock on role 0; and
      without the semaphore the pack races the math (0/128 right) on both targets
      -- which silicon showed only after the release was made simultaneous
      (divergence row 53).
- [x] **One 32×32 tile**, from `tt_layout` tile images to `tt_layout` de-tiling,
      on ttsim and both cards (`step10_matmul_tile::a_32x32_tile_matmul_matches_the_model_and_burn`):
      every datum equals `numerics::matmul_tile_reference` (face-composed
      `mvmul_reference`) and `burn-flex`, which agree first; transpose control.
      `matmul::tile_roles`: faces selected by the unpack thread's ADC `Z`
      (`SETADCZW`, one of the eight backlog ADC instructions -- now exercised on
      both targets) with `X` over all 256 datums; in0 -> `SrcB`, in1 -> `SrcA` as
      LLK; 16 unpack pairs and 32 `MVMUL`s per tile, `SETRWC` choosing the `SrcB`
      half; 64 `Dst` rows packed in 16 `PACR`s. Watched failing with `B`'s faces
      transposed and with the `Z` select removed. Operands now sit at `Src` row 0:
      the row-16/row-8 placement and its staged zero rows were a leftover of the
      header off-by-one, and row 0 passes on both targets.
- [x] **BF16 `Src`**, both routes (FP32 in L1 converted, BF16 in L1 as is), same
      tile, both targets (`a_32x32_tile_matmul_through_bf16_src`). The operands are
      exact in BF16, so this shows the routes work end to end, not that the
      conversion truncates; the truncation itself is `probe_src`'s gate.
- [x] **Multi-tile**, ttsim and both cards: `K` depth 2 and 4 (the unpacker
      re-pointed at each pair's images with `WRCFG` of both `REG3_Base_address`
      words between pairs, drained before and `CONFIG_BUSY`-waited after), `M`×`N`
      output tiles (2×3), and padded shapes (`[13,47]@[47,29]`, `[1,64]@[64,96]`)
      through `tt_layout`'s zero padding and cropping. `matmul::matmul_roles`:
      one output tile in `Dst` at a time, `DST_FREE` (pack -> math, starts at 1)
      guarding the `ZEROACC` and `DST_READY` (math -> pack) the `PACR`s; both
      semaphores end where they started. `pack_rows` now resets the packer's ADC Y
      first. Watched failing three ways: no re-pointing (K depth wrong), no
      `DST_FREE` wait (output tile 0 packed as zeros), no Y reset (tile 1 packed
      from the wrong rows).
- [x] Do unpacker/packer bring-up **entirely in the simulator** — every refusal so
      far has been a specification question, answered and logged before moving on.
- [x] **Gate (sim):** `step10_matmul_tile::a_shape_format_and_depth_sweep` --
      four shapes (depth one to three tiles, three of them padded) by three `Src`
      routes (FP32->TF32, FP32->BF16, BF16->BF16), each bit-exact against the
      integer product; `burn-flex` checks the pairing on the single tile. Fidelity
      phases are gated per block (`step9_matmul`, all four through
      `FIDELITY_BASE` and through the RWC); the tile path runs phase 0 on operands
      phase 0 represents exactly, so multi-phase *tiles* are not yet exercised.
      Noted rather than ticked separately: nothing a Burn backend needs first
      depends on it.
- [x] **Gate (silicon):** the same suite, unreduced, passes on both cards --
      18/18 in `step10_matmul_tile` per card, with no divergence from ttsim. The
      one silicon-only finding on the way was the harness's (row 53): cores
      released one by one ran in sequence.

---

## Phase 7 — Burn backend, training

**This is the milestone.** Everything before it is infrastructure; everything after
is reach or speed.

**Closed on ttsim and both cards** (2026-09-30): a `784-128-10` MLP built from
Burn's own `nn::Linear`, `CrossEntropyLoss` and `Sgd` trains on MNIST through
`burn-autodiff` with every forward and backward matmul on a Tensix tile, and the
reduced run's loss curve is the same 32 `f32`s on ttsim and on silicon.

- [x] **Verify the `Backend` supertrait list against the pinned Burn version**
      (0.21.0, from source): see `RUST_IMPL_PLAN.md`, "The Burn surface, as
      pinned". Associated types live on `BackendTypes`; `Backend` requires only
      `name`, `seed`, `dtype_usage`, `device_count`.
- [x] **`burn-tt` delegates to `burn-flex`, generated rather than written.**
      `cargo xtask gen-burn-delegate` reads the pinned `burn-backend`'s seven op
      traits (418 methods, 206 without a default) and `burn-flex`'s own `impl ... for
      Flex` blocks, and emits `crates/burn-tt/src/generated/delegate.rs`: every
      method Flex implements forwards to Flex; every default Flex does *not*
      override is left to its default, which then composes `burn-tt`'s own ops;
      the ones in `OVERRIDDEN` (device ops, device tags, futures) call
      `burn-tt/src/ops.rs`. The generator is syntactic; what each type converts
      to is three traits in `convert.rs` (`IntoFlex`, `FromFlex`, `HasDevice`),
      which rustc checks. It refuses an unparsable signature, generics, a `where`
      clause, an unknown identifier, an unforwarded future, a result with no
      device to tag, an `OVERRIDDEN` entry that is not a method, and a required
      method it cannot find in Flex; each refusal has a unit test. `--check` is in
      CI and `prek.toml`. Burn pinned exactly (`=0.21.0`, `PINS.toml`).
  - [x] **The default rule was found the hard way.** The first generator
        forwarded defaults too, and `ModuleOps::linear` -- a default over
        `float_matmul` that Flex does not override, and what `nn::Linear` calls --
        ran on Flex's matmul, so no `Linear` layer ever reached the device. Caught
        by `step12_mnist::the_first_forward_pass_is_within_the_derived_bound`,
        which asserts the device ran, and pinned by
        `burn-tt/tests/server.rs::a_linear_layer_reaches_the_engine`.
- [x] **Gate (host): the delegation is inert.** `burn-tt/tests/delegation.rs`: an op
      battery (elementwise, scalar, reductions, shape ops, activations, int and
      bool ops, seeded random) gives Flex's bytes exactly, and an F32 matmul on an
      unattached device panics rather than quietly running on the host. Watched
      failing with `float_add` forwarded to `float_sub`: exactly the elementwise
      battery fails.
- [x] **Devices and the server thread.** `TtDevice { chip }`; its hardware lives
      on a thread started by `burn_tt::attach(device, factory)`, which runs the
      factory *on* that thread, so the `!Send` simulator serves a `Send + Sync`
      backend. Dropping the `AttachGuard` drops the hardware (chip back to idle).
      A device attached twice is refused; a factory error is `attach`'s error and
      leaves the device free; an engine error panics with its message
      (`burn-tt/tests/server.rs`, eight tests, with a host engine). `kmd_engine`
      is the silicon engine; the ttsim one is `tt_tests::burn_device`, so
      `burn-tt` never depends on the simulator (`SHIPPABLE`, checked).
- [x] **Four-phase (HiFi) matmul at tile level.** `matmul::Fidelity { Lo, HiFi2,
      HiFi3, HiFi4 }`: one `MVMUL` per phase per `SrcB` half, `addr_mod(1)`
      stepping `FidelityIncr`, the half's `SETRWC` putting the phase back so
      nothing depends on the counter wrapping; `Lo` keeps the established
      encoding (unit-tested). `step10_matmul_tile::a_32x32_tile_at_hifi4_recovers_the_exact_product`
      (HiFi4 = the exact product, HiFi2 = phases 0 and 1, Lo = phase 0) and
      `hifi4_is_exact_across_k_where_lo_is_not`, on ttsim and both cards; the
      sweep runs HiFi4 too. Watched failing with the `addr_mod` removed.
- [x] **Large matmuls are planned, not refused.** The L1 staging `assert!`s are
      `RunError::DoesNotFit`; `matmul::plan` picks the largest `[mc, kc, nc]`
      chunk that fits the staging and output regions *and* every program slot --
      measured by building the programs, not by a formula -- keeping `K` whole
      whenever it can; `matmul_chunked` runs and reassembles them.
      `a_matmul_larger_than_one_run_is_chunked` (`[64, 784] @ [784, 128]`,
      HiFi4), ttsim and both cards. It found divergence row 55: ttsim kills the
      process on a full instruction FIFO where silicon stalls, now handled by
      `mailbox::PUSH_WINDOW` on the simulator only.
- [x] **The device path is shippable.** `tt-firmware-images` builds and checks
      the firmware (moved out of `tt-tests/build.rs`) and exposes `ROLES`, with
      each image's ELF entry checked against its core's reset PC (watched
      failing). `tt_kernels::session` holds the silicon bring-up order -- ARC
      grid, tile check, cleanup write, backend pulse, per-thread reset -- which
      the harness now calls instead of owning. Full silicon regression after the
      move: **132/132 on each card**.
- [x] **Gate (sim + silicon): matmul against Flex.** `step11_burn`: small-integer
      matmuls (plain, padded, batched, broadcast, transposed view) bit-exact;
      random floats within a bound **derived from the formats** -- TF32 operand
      truncation, `4k + k` truncating `Dst`/host additions, Flex's `k`
      round-to-nearest ones, documented in the file -- at worst 0.08-0.36 of it;
      the control, `Fidelity::Lo`, breaks it on 897 of 1320 elements. An
      `Autodiff<TtBackend>` linear layer's `dL/dx` and `dL/dw` within their own
      products' bounds. Identical ratios on ttsim and both cards.
  - [x] **A generator confined to `[-1, 0)`** made the first version of these
        gates weaker than they read (no sign cancellation) and killed every ReLU
        in step 12 (all-negative weights: no gradient, no training, and a host
        run that looked like a Burn bug until bisected). Both generators now
        assert they cover `[-1, 1)`.
- [x] **Gate (sim): MNIST trains, reduced.** `step12_mnist::the_mlp_trains_on_a_reduced_dataset`:
      512 images, batch 64, 4 epochs, SGD at 0.5. The loss falls from 2.321 to
      0.540 (host: 0.538), past the stated factor of 0.5, which the host run of
      the same setup is held to as well. The first forward pass is within a bound
      carried through the network from step 11's (worst 0.027 of it). Data is
      `cargo xtask fetch-mnist`, pinned by both `.gz` and IDX digests.
  - [-] **Final weights within a bound of the host's: not claimed.** Two runs whose
        first steps differ by rounding follow different trajectories; a bound that
        honestly covered that would say nothing. The first step is bounded, the
        curves are printed side by side, and the next item is the stronger claim.
- [x] **Determinism, and ttsim = silicon.** The reduced run's 32 losses are pinned
      as `f32` bits in `crates/tt-tests/tests/golden/mnist_reduced.txt` (written
      by the ttsim run with `TT_BLESS=1`); ttsim reproduces it run after run, and
      **both p150a cards reproduce it exactly** (divergence row I).
- [x] **Gate (silicon): full MNIST.** `the_mlp_trains_on_full_mnist`, one epoch of
      all 60 000 images, then the 10 000 test images, on both cards: loss 2.321 ->
      0.373 by step 900 (host 0.372), **test accuracy 91.96% on the device against
      91.97% on the host**, identical on the two cards. ~435 ms/step on the device
      against ~49 ms/step for Flex on the host: every chunk re-stages its operands
      and re-runs the tile reset, and the data crosses PCIe for every matmul.
      That is the Phase 9 baseline, not a gate.
- [-] `QTensorOps` stays Flex's, on the host: quantization was out of scope for the
      milestone (Phase 10, D2).
- [x] **Open question 3: `burn-fusion` does compose with a hand-written
      backend.** `Fusion<B: FusionBackend>`, where `FusionBackend` is `BackendIr` +
      a `FusionRuntime` supplying `OperationFuser`s over `OperationIr`. Fused
      unpack -> math -> pack chains are a `burn-tt` fuser (Phase 9), not a CubeCL
      question.
- [x] **Next ops and per-run cost**, carried forward: element-wise, ReLU and the
      bias sum moved to the device and tensors became resident in Phase 9 (9.3-9.4);
      everything else is Phase 10.

---

## Phase 8 — Multi-chip

Widest uncertainty band in the plan, and sequenced last on purpose: building
inter-chip transport before single-chip compute is correct is the most common way
projects of this shape stall.

- [x] **Spike, on `bh_x2` and on the two cabled cards** (2026-09-30). All four
      open unknowns are closed. Probes: `probe_eth.rs` (ttsim) and
      `silicon_eth_survey.rs` (silicon, read-only), in steps that each touch
      strictly more than the last. Findings: divergence rows 56-60, measurements
      J and K.
- [x] Resolved: **Ethernet reset.** `ethdump.c:362-451` has it: E1 is bit
      `0x1000` of `SOFT_RESET_0` at `0xFFB1_21B0`, and its reset PC is at
      `0xFFB1_4008`. Silicon reads `0x47000` (E0 running, E1 held), so changes are
      a read-modify-write of E1's bit, never ethdump's whole-word `0`.
      `step13_ethernet::rust_runs_on_e1` and
      `silicon_eth_link::e1_heartbeat_on_a_{portless,live_link}_tile` pass. On the
      live tile the link stayed Up and training Complete.
- [x] Resolved: **the E0 contract is avoided.** E0 is never reset or loaded:
      `EthCore::E0` has no load path, and every reset change touches one bit.
      Customer code runs on E1, as ethdump does. WH `CallingIntoCustomerCode.md`
      is not used.
- [x] Resolved: **the Ethernet PIC is not needed.** Everything polls.
- [x] Resolved (open question 4): **the NoC Overlay is not needed.** A TT-link
      *L1 write* (`ETH_TXQ_CMD = 2`) writes straight into the partner's L1, with
      sequence numbers and automatic resends. The host can drive one through a
      TLB window with no firmware at all: `silicon_eth_link::host_driven_*`,
      4/4 directions, 4096/4096 bytes, sentinel intact.
- [x] **The Ethernet grid comes from the chip.** It is ARC tag 35
      (`arc::tag::ENABLED_ETH`) giving `eth::Ethernet`, and an `EthTile` can only
      be obtained from one, so a harvested tile cannot be named (the row-35
      lesson). ttsim does not publish the tag, so the simulator uses
      `Ethernet::FULL` and says so.
- [x] **The link map comes from the chip.** `Device::eth_link_state` reads the
      port status and the base firmware's chip-info exchange: `BOOT_RESULTS`
      240..246 is the tile itself and 248..254 its partner. On the cabled pair
      that gives E4 <-> E4 and E7 <-> E7. ttsim does not model the exchange
      (row 58), so its map is measured (`step13_ethernet::BH_X2_LINKS`).
- [x] **Firmware-owned Ethernet L1 is refused, not noted.**
      `eth::FIRMWARE_L1` (measured, K) is checked by `eth_read`/`eth_write`/
      `eth_tt_link_write`. `E1_RESET_PC` is inside the *Tensix* local-RAM aperture
      guard, and is reached by one unchecked write on an Ethernet-by-construction
      tile.
- [x] The runtime's own mailbox is now a link-time symbol (`--defsym` from
      `tt-isa`), so one runtime serves Tensix images (`0x100000`) and the
      Ethernet image (`0x1F000`, inside the 512 KiB Ethernet L1 that `0x100000` overruns). The E1 gate was watched
      failing with the release removed.
- [x] Note: Ethernet tiles lack the Tensix conveniences. Local data RAM is **not**
      NoC-accessible and there are **no `pc` snapshots**. The heartbeat therefore
      lives in L1, and so will everything E1 reports.
- [x] **8.1 The NIU request initiator**, `tt_isa::noc::niu::Command`. It is the
      first device-initiated NoC traffic in this workspace. A command is an L1
      read, an L1 write, or an inline MMIO store, and that is all it can be.
      There is no field for `L1_ACC_AT_EN`, broadcast or linked VCs. An inline
      write to L1 (the Blackhole bug), an L1 copy whose addresses are not
      congruent mod 16, and a length over 16 KiB are all refused. For a write, the
      source sits in `NOC_TARG_ADDR`, as `MemoryMap.md:99-104` has it. Unit tests
      cover each refusal. ttsim refuses a read without `RESP_MARKED` (row 61),
      so every command sets it, which the specification makes harmless.
- [x] **8.2 `tt_device::ethernet`:** `ethernet_grid`, `eth_link_state`,
      `eth_read`/`eth_write` (both refuse firmware L1), `eth_tt_link_write`
      (refuses misalignment and firmware L1 at either end), and
      `load_and_start_e1` / `park_e1` (a read-modify-write of E1's bit only).
- [x] **8.3 The `eth_e1` data mover** (`tt_isa::eth::mover` is the shared
      contract). Tensix L1 -> NoC read -> TT-link -> landing wait -> NoC write
      -> Tensix L1, then an acknowledgement back over the link, so the host
      waits on the *sending* chip only. The receiver will not forward until its
      RX queue has no writes outstanding (`ETH_RXQ_OUTSTANDING_WR_CNT`), because
      nothing documented orders RX-queue L1 commits. It does this on silicon
      only, since ttsim cannot be asked and commits synchronously (row 62). The
      record carries its sequence number in both 16-byte halves. The cross-chip
      gate was watched failing with the receiver's forward removed. The first
      version checksummed the whole buffer on both ends, which held the mover
      to 196 MB/s.
- [x] **8.4 `tt_kernels::link`:** `discover` pairs tiles from both chips'
      chip-info exchange, and requires each side to name the other. `Mover`
      provides `start`, `stage`, `send` (with a host deadline, returning
      `TimedOut` rather than hanging) and `landed`. `bh_x4` is pinned
      (`libttsim_bh_x4.so`, same tag) and mapped: it is a ring 0-1-2-3-0 with
      two links per pair (`tt_tests::topology`).
- [x] **8.5 Sharded matmul and training.** `tt_kernels::shard::Fabric` splits
      `N` in 32-column tiles across chips. Each chip's share runs through
      `matmul::plan`'s chunking with `K` whole, so the result is
      **bit-identical** to single-chip. Operands enter and results leave through
      chip 0 over Ethernet only, relayed through intermediate chips (chip 2 of
      the ring goes through chip 1). Programs and descriptors go to each chip
      over its own PCIe (the control plane). In `burn-tt`, `MeshEngine` puts a
      whole fabric behind the existing `Engine` trait as one Burn device, so no
      `attach_mesh` was needed (`kmd_mesh_engine` on silicon). The shard gate
      was watched failing with `B` delivered 16 bytes off.
- [x] **Gate (sim):** `step13_ethernet` (11 gates: link state, TT-link writes
      both ways, a corrupted byte moves exactly one byte, the wrong tile
      receives nothing, refusals, E1 heartbeat and its held control, the mover
      staged both ways, Tensix -> Tensix across chips, no receiver means a
      timeout). `step14_shard`: the MNIST first layer on `bh_x2`, and a
      four-chip ring matmul with relays, both bit-identical to single-chip.
      `step12_mnist::the_mlp_trains_sharded_*`: the reduced run on 2 and on 4
      chips reproduces `mnist_reduced.txt` bit for bit.
- [x] **Gate (silicon), across the cable:** `silicon_eth_link`, 10/10:
      - host-driven TT-link, 4 directions;
      - E1 on a port-less tile and on a live one;
      - the mover staged both ways at 128 KiB;
      - 128 KiB Tensix -> Tensix across cards, acknowledged in 14 us;
      - no receiver means a timeout;
      - the sharded `[64,784] @ [784,128]` HiFi4 matmul, bit-identical to
        single-chip.

      `step12_mnist::the_mlp_trains_sharded_over_two_chips_matching_the_golden`
      passes on the two cards: **the loss curve is the golden, bit for bit.**
- [x] **Throughput, measured** -- the table is in `RUST_IMPL_PLAN.md`, Phase 8
      "As built": 128 KiB Tensix -> Tensix across cards in 14 us (9.4 GB/s),
      about 500x the PCIe TLB path. Integrity: 200 transfers of
      mixed size, each with fresh data and a fenced sentinel, all correct
      (`silicon_eth_bench::mover_integrity`). With the landing wait disabled, 400
      more were also all correct, so the record's in-order arrival suffices in
      practice. The wait stays, because it is what the documentation licenses.
- [x] **The "128 KiB never arrives" benchmark failure was the benchmark's.**
      Its zero-fill of the destination is posted PCIe writes, and some landed
      *after* the TT-link data, zeroing the transfer's tail. 57 of 480 failed
      unfenced, 0 of 480 fenced, and the link has no size limit (40 x 4 KiB
      commands fine). It exposed a real hazard: `Device::write` does not order
      against agents other than the host. `Mover::stage` now reads back, and
      the silicon gates fence what they stage.
- [ ] Double-buffered `TX_STAGE`/`RX_LAND`, and both links at once.
- [ ] **Data-parallel training** (a gradient all-reduce over the links) is not
      done. It reorders sums, so it needs a weaker claim than the golden, and
      `N`-sharding already makes Ethernet load-bearing.

---

## Phase 9 — Performance (silicon-only)

**Direction (2026-09-30):** keep everything on the card. A p150a has 32 GiB of
GDDR6; the dataset and weights are loaded into it once at startup, and a
steady-state training step should move almost nothing over PCIe. The headline
metric is therefore **PCIe bytes per step** (`Device::traffic`), next to ms/step.

**Rules for every slice.** ttsim is the correctness gate (bit for bit against the
golden and `burn-flex`), because pipelined kernels are where correctness is hardest;
every performance number comes from silicon, since ttsim is not cycle-accurate; and
the comparison is against the specification's peak figures, not a competitor.

**Full MNIST, release, card 0** (accuracy 91.96% throughout, against 91.97% on the
host; the reduced golden bit for bit on ttsim and both cards after every slice):

| After | ms/step, 1 tile | ms/step, 8 tiles | Device written per step | What changed |
|---|--:|--:|--:|---|
| 9.0 | 224 | -- | -- | baseline; Flex on the host 0.5 |
| 9.3b | 38.5 | -- | -- | resident role firmware, no per-run reset |
| 9.4b | 16.8 | -- | -- | tensors in GDDR; tensor traffic 675/475 KB -> 216/35 KB up/down |
| profiling | 9.9 | -- | -- | unrolled element-wise, face-wise transposes, program slots compared by word |
| 9.4 views, sums | 5.8 | -- | 193 524 B | dataset preloaded (2.7 s, mostly host tilizing), batches as views, bias sum on device; ~13 KB up, 3 KB down |
| 9.6 | 5.8 | 4.1 | 349 372 B (4 tiles) | many tiles, waves (5.0 on 120) |
| 9.7a | 5.7 | 3.8 | 250 write calls | one launch per op per tile |
| 9.7b | 5.8 | 3.7 | 155 KB | op records expanded on the tile |
| 9.7c | 5.1 | **2.5** | 35 KB | resident program cache |

- [x] **9.0 Release baseline.** `cargo xtask silicon --release` (and the log
      records the profile). Every earlier number was a dev build. Full MNIST,
      release, both cards: **224 ms/step on the device, 0.5 ms/step for Flex on
      the host**, test accuracy 91.96% against 91.97% as before. PCIe is bus-bound
      and unchanged by the build: 19 MB/s write, 5 MB/s read.
- [x] **`Device::traffic`**: bytes and calls in each direction, and TLB
      retargets, counted at the three places a `Device` touches a BAR. Unit-tested
      against the fake transport's own log, and watched failing with the read
      counter removed.
- [x] **`runtime::Profile`**: every `run` reports each phase (stage, setup,
      programs, launch, wait, read-back) with its host time and traffic.
- [x] **9.1 DRAM, from the chip.** `tt_isa::dram::Dram` is built from ARC tags
      36 and 22 only (`Dram::FULL` on ttsim), and a `DramChannel` cannot be named
      unless it is enabled, trained and BIST-clean; a fused-off channel is
      refused outright, since it changes the translated numbering. Endpoints are
      UMD's translated coordinates. `DramRange` stops at `0xFF00_0000` (row 63).
      `Device::{dram_grid, dram_read, dram_write}`. Gate `step15_dram` (a single
      read, patterns in every channel, the three-endpoint aliasing, refusals
      before the transport) on ttsim and both cards; the round-trip gate was
      watched failing. Measurement L.
- [x] **The bulk path is memory-only by construction.** `Transport::bar_*_bulk`
      is reached only from `Device::{l1_write, l1_read}` -- Tensix geometry and
      inside L1, so no register -- and the DRAM accessors. Unit-tested refusals.
      `tt-kmd` maps a second, WC view of BAR0/BAR4 for it (`RESOURCE*_WC`
      transcribed and checked against the header, watched failing) and copies in
      32-byte non-temporal stores and stream loads with an `sfence`.
- [~] **Fast startup upload.** 226 MB/s into GDDR, 152 MB/s into L1, 38 MB/s
      reading: 12x and 8x the dword path, but not WC -- under this KVM
      passthrough the BAR is effectively uncached (measurement M). Enough to load
      MNIST in under a second; real WC needs the host to map the BAR WC.
- [x] **`niu::Command::{ReadDram, WriteDram}`**: take a `DramRange` and a port,
      name the translated endpoint, and refuse a DRAM -> L1 read not congruent
      mod 32 (L1 -> DRAM: mod 16), a fourth port, and more than one request.
- [x] **9.3a The data mover on RISCV B** (`dm_b`, contract `tt_isa::dm`,
      host `tt_kernels::dm::DataMover`): resident, one descriptor at a time,
      DRAM -> L1 and L1 -> DRAM in 16 KiB NIU requests. Every refusal is in
      `Descriptor::decode`, run by the host first and by the firmware again,
      which answers rather than hangs. Gate `step16_dm` (every channel and port,
      multi-request transfers, both refusal paths) on ttsim and both cards; per
      move only the descriptor crosses PCIe, asserted with `Device::traffic`. It
      found the DRAM read rule: **C64, not Wormhole's C32** (row 64).
- [x] **9.3b Resident role firmware.** A non-zero `mailbox::GENERATION` makes
      the runner acknowledge (`ACK`) and wait for the next generation instead of
      stopping. `runtime::Resident` loads the three images once; `Session` resets
      and starts it at open, and `Session::matmul` runs every chunk on it
      (`matmul::matmul_with`). Both burn engines use `Session`, so the golden now
      covers this path. Gate `step17_resident`: MNIST's shapes and fidelities,
      twice round, bit-identical to reset-per-run `matmul_on`; a different kernel
      in between changes nothing; a failed kernel is an error and the session
      recovers (silicon). Watched failing with the math program left stale.
      Recovery found row 65: a `SEMWAIT` survives the backend pulse, so the
      tile reset now releases every semaphore first.
- [x] **9.3c** Programs and chunk plans memoised per process
      (`matmul::programs`, `plan_in`); a resident program slot already holding
      the program is not rewritten.
- [x] **9.4a Tensors in GDDR** (`tt_kernels::tensor`). A `DramTensor` is
      FP32 tiles in 4160-byte slots (zero header, datums, padding to 64, so any
      slot copies to any other under C64), interleaved over the channels, by a
      coalescing per-channel allocator. Upload and download are one bulk
      transfer per channel. The mover takes descriptor **lists**, with a
      transposed tile read (the B core transposes the tile in L1) and FP32
      **compute** entries. Gates: `step18_dram_matmul` (round trips; MNIST's
      forward and backward products, transposed operands included,
      bit-identical to the host-staged matmul; a tile header's contents are
      never read) and `step19_eltwise` (add/sub/mul/mul-scalar/relu/relu-backward/
      broadcast row add against `burn-flex` bit for bit over +-0, +-inf, NaN,
      huge and tiny values; the denormal flush recorded), ttsim and both cards.
- [x] **Element-wise on the baby RISC-V's FP32 unit**, not the SFPU:
      `fadd.s`/`fsub.s`/`fmul.s` round to nearest even with denormals flushed
      (`InstructionSet.md:18-22`), IEEE for every normal case -- `fma_bh`
      agreed with the host on 10^6 random `mul`, `add` and SGD updates each.
      Reached through inline asm (the images stay `riscv32im`); the instruction
      gate now decodes `F` and refuses `fmadd`/`fmsub`/`fnmadd`/`fnmsub` (watched
      refusing a planted one). The SFPU path is the faster follow-up.
- [x] **9.4b `burn-tt` keeps tensors on the device.** `TtTensor` is a shared
      cell with lazily filled host and device copies (clones share both, so what
      the forward pass uploads the backward pass finds); a 2-D transpose of a
      device tensor is a view. On device: `float_matmul`, `float_add` (with the
      `[1, n]` bias broadcast), `float_sub`, `float_mul`, `float_mul_scalar`,
      `relu`, `relu_backward` -- whenever an operand is already there. Anything
      else downloads once, counted by `burn_tt::tensor_traffic`
      (`TT_TRACE_FALLBACK=1` says which op). Both engines keep tensors in GDDR.
- [x] **Profiled and cut, 16.8 -> 9.9 ms/step** (single card, both cards
      alike, golden unchanged): element-wise in unrolled `flw`/`f*`/`fsw` loops
      (7.4 -> 2.6 ms/step); the resident setup run skipped when a kernel
      declares it restores its semaphores (`Kernel::restores_semaphores`, which
      the matmul does by construction); face-wise transposes; and resident
      program slots compared by encoded word -- comparing `Instruction`s
      compared their definitions and cost 14 us per descriptor write on silicon,
      more than the rewrite it saved (matmul 4.6 -> 2.7 ms/step). Per step now:
      element-wise 2.6, matmul 2.7, upload 2.3, download 2.1, host 0.2 ms.
- [x] **One switch for the topology**: `burn_tt::Topology` /
      `attach_topology`, or `TT_TOPOLOGY=0` / `0,1` for the silicon harness, so a
      benchmark runs on one card or both unchanged. Two cards are Phase 8's
      mesh: host-staged, per-chunk resets, chips in turn -- 233 ms/step.
- [x] **Padding rows are not kept zero** -- fixed by `hardware-coverage.md` F0
      (a typed pad state per tensor, a masked column sum, edge-tile refills
      before a matmul); `step19_eltwise::padding_rows_stay_out_of_a_later_accumulation`
      un-ignored and green on ttsim and both cards. Was (reproduced on ttsim, 50 rows give `-133` for Flex's `-105`, 14 padding rows times `b`, and the
      `(x + b)^T @ (y + d)` weight-gradient shape is wrong too):
      `ADD_ROW` adds the bias into a ragged tile's padding rows and `COL_SUM`
      sums all 32 rows (`dm_b.rs`), so `(x + b).sum_dim(0)` on a row count that
      is not a multiple of 32 should be wrong; the crop on download hides the
      rows themselves. MNIST's batches are whole tiles. `tt-metal-concepts-review.md`
      G1; the fix (a typed pad state per tensor) is `hardware-coverage.md` F0.
- [x] **`silicon_eth_link::host_driven_tt_link_*` are flaky** (found by Phase 10's
      full-suite run, 2026-10-01): in the full suite and in their own group, one to
      three of the four `host_driven_tt_link_card{0,1}_to_card{1,0}_x{3,13}` fail per
      pass, each after ~2.6 s, a different set each time, and each passes run alone.
      **Cause: the test, not the link.** The missing bytes were always the payload's
      tail, holding the receiver's `0xEE` sentinel: the host's sentinel write is posted,
      and now and then landed after the Ethernet frames and overwrote them. A read-back
      of both buffers' last words before the transfer starts fixes it: 3 failures in 40
      runs before, 6 in 60 with only the sender's read-back, 0 in 60 with both
      (`ttsim-divergence.md` row AA). The test now prints each mismatched range too.
- [ ] **A `BufferId` can outlive its engine** (found by review): `DramBuffers`
      numbers from 1 per attach (`burn-tt/src/server.rs`), so a tensor kept across
      a detach and re-attach reads another tensor's buffer, and dropping it frees
      one. Reproduced: `burn-tt/tests/stale_buffer.rs`, both ignored until fixed.
      `burn-backend-parity.md` sec. 4.1, roadmap B3.
- [ ] **The two-card full run's accuracy is 0.9195, one card's 0.9196.** The
      reduced sharded run matches the golden bit for bit, so something past 32
      steps diverges on the mesh. Phase 8 code; not yet investigated.
- [ ] **The mesh is not device-resident**: GDDR tensors, per-chip resident
      roles, and chips concurrent rather than in turn.
- [x] **The dataset is preloaded; a batch is a view.** `float_to_device` makes
      an F32 matrix resident (`Tensor::to_device` is the caller saying so), and
      `float_slice` of whole tile rows of a resident matrix is a view of the
      same slots (`DramTensor::rows_view`), keeping its parent alive. The MNIST
      gates now upload the images once and slice each batch.
- [x] **The bias gradient's sum is on the device.** `float_sum_dim(·, 0)` of a
      resident matrix is `tensor::sum_rows`: `COL_SUM` adds each column's rows
      in order from `+0.0`, which is `burn-flex`'s `sum_dim(0)` order exactly
      (`ops/reduce.rs:959-989`) -- bit for bit on edge values and a
      7000-row column spanning several mover lists, ttsim and both cards;
      watched failing with the rows summed in reverse.
- [x] **Small tensors move only what they occupy**: an upload writes only the
      slots its tiles fill, and a small download reads only the faces and face
      rows its data reaches (a `[1, n]` row: 128 bytes a tile, not 33 KB of
      whole regions).
- [x] **The 5.8 ms/step profile** (one tile): element-wise 2.4, matmul 2.4,
      column sums 0.3, downloads 0.3 (logits, two bias gradients), uploads 0.2
      (`g2`, two biases), host 0.2 ms. SGD on the rank-1 biases stays on the
      host (see 9.5).
- [x] **9.5 The residency gate.** `step12_mnist::the_mlp_trains_on_a_reduced_dataset`
      samples `burn_tt::tensor_traffic`, the new `burn_tt::device_traffic` (the
      engine's `Device::traffic`, queued behind every job) and the new opt-in
      transfer log (`record_transfers` / `take_transfers`, direction and shape)
      after every step. After the first step, **every step moves exactly six
      tensors**, each listed in the test with why: up `b1` `[1,128]` and `b2`
      `[1,10]` after the host's SGD step (rank-1), and `dL/dlogits` `[64,10]`
      from the host's loss; down the logits `[64,10]` and the two bias
      gradients -- 6224 B in all. **The device is written exactly 193 524 B per
      step** (428 writes, 12 retargets) on ttsim and both cards alike;
      reads (~19 KB on silicon, ~29 KB on ttsim) are mostly completion polls and
      are printed, not asserted. The golden still holds bit for bit on all
      three. Watched failing with ReLU forced to the host: step 1 lists two
      `NOT EXPECTED` transfers by shape. The figure that matters for 9.7: the
      device writes are **31x the tensor bytes** -- mover lists, kernel
      descriptors and programs, one host round trip per chunk.
- [x] **`tt-mnist`: the milestone as one binary.** A shippable crate whose
      binary trains the MNIST MLP through Burn on the card, with MNIST
      (deflated at build time, `miniz_oxide`) and the firmware embedded, and
      optionally the same run on the host for comparison. Builds static for
      `x86_64-unknown-linux-musl` (15 MB stripped; `tt-kmd`'s ioctl request type
      follows the libc). Full epoch on one card: 91.96% test accuracy, 6.8
      ms/step static, 5.8 with glibc. See `crates/tt-mnist/README.md`.

### Phase 9 -- next steps, in order

At 5.8 ms/step nearly everything left was compute on **one** of 120 Tensix
tiles, done in turn. The slices from there:

- [x] **9.6 Many tiles.** `TileChoice::Count(n)` / `All` opens a `Session`
      over `n` tiles, each a unit with its own resident roles and B mover
      sharing one TLB window (so 120 tiles fit the 201-window pool). A GDDR op
      is now a set of independent `tensor::Job`s -- matmul output blocks
      (`tensor::blocks` shrinks the one-tile plan's block until every unit has
      one, `K` whole), element-wise tile runs and column-sum column runs
      (`tensor::runs`, balanced) -- dealt round-robin and run in **waves**:
      one step started on every unit (`Resident::submit`,
      `DataMover::submit_list`), then every unit waited for. On ttsim one wait
      ticks every tile (row 23), on silicon they are separate tiles. Each extra
      tile's crash-cleanup write is held by a `tt_kmd::CleanupWrite`, a driver
      descriptor that maps no BAR: `Kmd::open` costs 247 ms on silicon and made
      a 120-tile open take 29 s, `CleanupWrite::register` 0.74 ms (0.38 s).
      `TT_TILES=n|all` (and `tt-mnist --tiles`) pick the count; the default
      stays one tile. Gates: `step20_many_tiles` (MNIST's and larger
      products, transposed and ragged, at 1, 2, 3 and 8 tiles bit-identical to
      the host-staged path; element-wise and column sums against `burn-flex`;
      every unit did a share; a stuck kernel on one tile recovers on silicon),
      and `step12_mnist::the_mlp_trains_on_four_tiles_matching_the_golden`
      (the golden bit for bit, and the 9.5 steady-state budget: 6224 B of
      tensors, 349 372 B written per step against one tile's 193 524 --
      per-tile descriptors and programs). Watched failing with the last job
      dropped and with every unit but the first left unwaited (wrong data, not
      just a missing count). ttsim and both cards: 38/38. `tensor::runs` and
      `blocks` unit-tested.
  - **Measured** (`silicon_perf::many_tiles_sweep`, card 0, release):
    `[512,512]@[512,512]` HiFi4 9.99 -> 5.2 (8 tiles) -> 2.2 ms (64) -> 3.65
    ms (120); an add over 2048 tiles 15.8 -> 3.6 -> 1.96 -> 2.06 ms; a column
    sum over 32 columns 45.6 -> 6.4 -> 2.3 -> 2.3 ms. Everything floors at
    about 2 ms, the host's per-wave round trips, and the matmul gets *slower*
    past 64 tiles: smaller blocks, more of them, each with its own program
    writes. **Full MNIST, one card: 5.8 -> 4.1 ms/step on 8 tiles** (5.0 on
    all 120), accuracy 91.96% unchanged; the gain is element-wise (2.4 ->
    0.8 ms), while MNIST's small matmuls stay at 2.4 ms -- bound by the host's
    round trips per block, which is 9.7.
- [x] **9.7a One launch per op per tile.** The B mover runs the resident
      roles itself: a `dm::op::KERNEL` list entry waits for the moves before
      it, posts the next generation to the three role mailboxes in local L1
      and polls their `ACK`s (through a fence: the L0 cache is not coherent,
      Tier 1 bug #7), reporting `error::ROLE` if one panics; `op::WAIT` keeps
      the barrier that separate lists used to give. The host stages a kernel's
      programs once and reserves its generations (`Resident::reserve` /
      `reserved_done`), and `session::segments` turns a tile's steps into as
      few lists as fit: one per op per tile unless it passes `LIST_MAX`
      entries or the tile's programs change. Gates: `step21_one_launch`
      (MNIST's first layer, four blocks on one tile, was 12 host round trips
      and is 1, on one tile and on three; two element-wise runs share a list;
      a 7000-row column sum takes only the lists its entries need), the
      golden at one and four tiles, `segments` unit-tested. Watched failing
      with one list per step (12, 2 and 4 round trips) and with B not waiting
      for the `ACK`s (wrong products). ttsim and both cards: 58/58 with the
      smoke tier. Per MNIST step, one tile: 428 -> 250 PCIe write calls.
  - **Measured** (card 0): full MNIST 3.8 ms/step on 8 tiles (from 4.1), 5.7
    on one; `[512,512]@[512,512]` 1.43 ms on 64 tiles (from 2.2), 2.30 on 120
    (from 3.65). **The floor is now the lists' bytes, not the round trips**:
    the add over 2048 tiles writes 263 KB of descriptors whatever the tile
    count, 1.75 ms at the uncached bulk path's ~150 MB/s (measurement M),
    which is its 2.0 ms; the matmul writes 150-715 KB. A steady MNIST step
    writes 193 KB, almost all of it lists.
- [x] **9.7b Op records.** The host sends an op, not its entries:
      `tt_isa::dm::record` defines `GATHER`/`SCATTER` (a matmul block's
      operands and outputs), `ELTWISE` (a run of tiles) and `SUM` (a run of
      column sums, with `WAIT`s where its old lists ended), each a header plus
      a 64-byte `TensorRef` per tensor (channels, per-channel bases, first
      tile, width), and `record::expand` -- `no_std`, allocation-free, no
      `divu` (`div_rem`: the instruction gate refused the first build) --
      produces the entries on the tile, each run through `Entry::decode` as a
      sent one is. The host expands every record first, so a bad one is still
      refused before PCIe; `segments` never splits one. The 9.7a builders are
      kept in `tensor::reference` as the specification: every record expands
      to exactly their entries, job for job, over three channel masks, 1/3/8
      tiles, row views and every op kind -- watched failing with the sum's
      boundary `WAIT` dropped and with one port changed. Golden bit for bit;
      ttsim and both cards 58/58 with the smoke tier.
  - **Measured** (card 0): an add over 2048 tiles on 64 tiles 1.99 -> 0.58 ms
    (263 KB -> 15 KB written); the column sum 2.29 -> 1.54 ms; the 512^3
    matmul 1.43 -> 1.18 ms. A 7000-row column sum is one list of five entries.
    Full MNIST 3.7 ms/step on 8 tiles, 5.8 on one; its ops are too small for
    list bytes to have been their cost. Steady MNIST step: 193 -> 155 KB
    written (one tile). **What is left is role programs**: the 512^3 matmul
    still writes 605 KB on 1-8 tiles, rewriting each role's single program
    slot whenever consecutive blocks differ in shape, and a program change
    also ends a list (`xt@g1`, ragged at its edges, writes 74 KB).
- [x] **L1 planning, step 1** (design: `RUST_IMPL_PLAN.md`, "L1 planning and
      circular buffers"). `tt_isa::l1` maps every fixed region of a tile's L1,
      ordered and disjoint at compile time, leaving the data arena.
      `tt_kernels::l1` plans `Requirements` -- scratch and circular buffers
      with producer and consumer endpoints, live ranges over a kernel's
      stages, semaphores -- by liveness into the arena, with an independent
      `check`, handles that refuse a foreign plan, and `Requirements::fuse`,
      which joins one kernel's output ring to the next's input. Unit and
      property tests (500 random kernel pairs and their fusions), and the
      checker watched refusing an aliasing plan and the property test
      catching a planner that ignores liveness. Ported: the GDDR matmul's
      layout (A and B as rings from the mover to the unpacker, the outputs as
      a ring from the packer to the mover) and the element-wise and column-sum
      staging. Golden bit for bit; ttsim and both cards 84/84 with the smoke
      tier. The host-staged path keeps its fixed layout.
- [x] **Semaphores are planned, not named.** Every semaphore is a
      `Requirements::semaphore(name, initial, live)` declaration, and a plan's
      `semaphore_init()` is the whole of what a concurrent run initialises, so
      `runtime`, `session` and the mover-driven `Step::Kernel` carry a generic
      init list and know nothing of a matmul. The matmul's `DST_READY`,
      `DST_FREE` and `TILE_SEMAPHORES` constants are gone; `MatmulSemaphores`
      is only the matmul's names for its two (`ready` from 0, `free` from 1),
      and the programs memo is keyed by them. Found on the way: two semaphores
      of different stages may share a number only if they start at the same
      value, or the second inherits the first's -- now a planner rule and a
      `check` refusal, both tested. Gates: the roles' programs differ exactly
      at the semaphore instructions when the pair is swapped; golden bit for
      bit; ttsim and both cards 78/78 with the smoke tier; the two-card
      sharded MNIST reproduces the golden.
- [x] **9.7c Resident programs.** Each tile keeps the kernels it runs in
      `tt_isa::l1::PROGRAM_CACHE` (252 KB), mirrored on the host by
      `tt_kernels::program_cache::ProgramCache`: keyed by the program's words,
      first fit with coalescing, least recently used evicted first, anything a
      list in flight names pinned, programs over half the region bypassed to
      the fixed slots, everything forgotten on a tile reset, and hit, miss,
      upload, eviction and bypass counters (`Session::program_cache_stats`).
      A role runner pushes from `mailbox::PROGRAM_ADDR` when it is non-zero
      (refused outside the region), and a `KERNEL` entry names each role's
      `(address, words)`, written by B before the generation -- so a list may
      run kernels of any number of shapes, and `segments` splits only on a
      change of semaphore setup or past the cache's capacity. Gates:
      `step22_program_cache` (a two-shape `[512,512]@[512,512]` is one list,
      and its second run uploads nothing; 300 KB of kernels through the 252 KB
      region evicts, and every evicted kernel is right when it returns), cache
      and `segments` unit tests including a random placement soak, the golden
      at one and four tiles. Watched failing with B ignoring the addresses
      (wrong products). ttsim and both cards 134/134 with the smoke tier.
  - **Found on silicon only:** a stale `PROGRAM_ADDR` from the previous
    process pointed a tile reset's runner into the cache, and the reset hung
    -- L1 survives between processes, and the plain `runtime::run` path wrote
    the descriptor field by field and had no reason to know the new word.
    `probe_cfgreg` had the same gap (no `TRACE`, `PUSH_WINDOW`). Every runner
    descriptor is now a `mailbox::Descriptor`, written whole, with a test
    that it covers every word the runner reads.
  - **Measured** (card 0): full MNIST **2.5 ms/step on 8 tiles** (from 3.7),
    5.1 on one (from 5.8), accuracy 91.96%; its matmuls 2.08 -> 0.89 ms per
    step on 8 tiles. `[512,512]@[512,512]` 9.65 -> 6.89 ms on one tile (605
    KB -> 15 KB written) and 4.46 -> 1.32 ms on eight. Steady MNIST step on
    one tile 155 -> 35 KB written. Past 8 tiles the matmul slows again (2.0 ms
    on 120, 3600 PCIe calls): what is left is per-tile submission, a
    descriptor write and a list per tile, which is the next floor.
  - The validation tier grew from 36 to 59 s, about 20 of it
    `step22_program_cache`.
- [ ] **9.8 Overlap.** Double-buffer the L1 staging so the mover gathers the
      next chunk while the roles compute this one, and scatters the previous
      one (the `Src`/`Dst` double buffering and the hazards-as-data wait
      planner from the plan belong here).
- [-] **9.9 Element-wise on the SFPU** -- moved to Phase 10, and done there (S1, milestone 10.0).
- [ ] **9.10 Faster start-up.** The preload (2.7 s for 60 000 images) is mostly
      host tilizing: tilize in parallel, or upload row-major and let the movers
      tilize on the device.
- [ ] **9.11 The mesh, device-resident.** Per-chip `Session`s with GDDR and
      resident roles, chips running concurrently, the Ethernet movers moving
      tiles between GDDR rather than host-staged operands; data-parallel
      training over the two cards. First find why the full two-card run's
      accuracy is 0.9195 against one card's 0.9196.
- [-] **9.12 Loss on the device** -- moved to Phase 10 (R2, milestone 10.1).

- [ ] A `Device` write fence as API, rather than read-backs at call sites
      (Phase 8: posted writes race other agents).
- [-] `MOP`/`REPLAY` expansion -- moved to Phase 10 (`hardware-coverage.md` X1, X2).
- [ ] NoC multicast for operands many tiles share (matmul `in0`/`in1`).
- [ ] `.ttinsn` fusion — up to four adjacent pushes per cycle. Deferred from the
      baseline because fused words disassemble as garbage and the instruction-set
      gate would have to stop rejecting undecodable instructions.
- [ ] `L1CacheTagSearchAccel` — Blackhole-only, RISCV B only.

---

## Phase 10 — Hardware coverage

**Tracked in [`hardware-coverage.md`](hardware-coverage.md)**, not here: the inventory of
every Tensix unit and what drives it, the work items (F foundation, S SFPU, M Matrix Unit,
R reductions, D formats and data movement), milestones 10.0–10.6, and the Burn op coverage
table. Ticks happen there. The rationale is `RUST_IMPL_PLAN.md`, "Phase 10".

- [x] **10.0** Device profiler; SFPU foundation; today's element-wise ops on the SFPU (was 9.9). Branch `phase10-0-sfpu-foundation`; full silicon suite 381/386 on both cards, the five being two since-fixed `step23` assertions and the pre-existing Ethernet flake above.
- [ ] **10.1** Softmax and cross-entropy on the device (was 9.12); `MOP`; op-list traces.
- [ ] **10.2** Activation and math breadth.
- [ ] **10.3** Reductions over any dim, pooling, device transpose, norms.
- [ ] **10.4** Formats and integers.
- [ ] **10.5** Indexing and convolution.
- [ ] **10.6** Block float, PRNG, `SFPLOADMACRO`, `ELW*`, `DOTPV`.

---

## Silicon-verification backlog

Areas where Blackhole has **no documentation anywhere** and the Wormhole page is
**not** `TTArchitecture`-conditionalized. Every item needs empirical validation
against ttsim and silicon before it can be trusted. Each finding becomes a
permanent test case — that suite is the project's real asset, because it is what
lets you trust Wormhole-sourced behaviour at all.

**Packers** — [ ] all 10 sub-pages · [ ] `PACR_SETREG` · [ ] `CLREXPHIST`

**Unpackers** — [ ] `README` · [ ] `FormatConversion` · [ ] `FlushCache` ·
[ ] `IncrementContextCounter` · [ ] most `UNPACR_NOP_*`

**State** — [ ] `RWCs` (three Blackhole pages link to it) ·
[ ] `SrcASrcB` **including fidelity phases** (not even linked from Blackhole) ·
[ ] `ZEROACC` · [ ] `ZEROSRC` · [ ] `SETRWC`/`INCRWC`/`GATESRCRST` ·
[ ] all 8 ADC-manipulation instructions

**Numerics** — [ ] `FloatBitPatterns` (highest-value missing numerics doc)

**Compute** — [ ] `SHIFTXA`/`SHIFTXB`/`TRNSPSRCB` · [ ] `GAPOOL`/`DOTPV`

**Frontend** — [ ] `WaitGate` (a silent gap; seven BH pages reason about its
semantics) · [x] `REPLAY` (`step26_sfpu_isa`: SFPU row loops replayed match their unrolled form on ttsim and both cards) · [x] `MOP`/`MOP_CFG` (`step36_mop`: both templates against the page's model, configuration from the role mailbox, ttsim and both cards; generator marks them `CONFIRMED`)

**Units** — [ ] the entire Scalar Unit (ThCon) · [ ] Mover (`XMOV`) ·
[ ] Miscellaneous Unit

**Closed by Phase 1's silicon gate** — [x] NoC coordinate translation is **on**
(`NOC_TRANSLATION` = 1, both cards), so host-facing TLB coordinates are translated space,
which the 120-tile sweep confirms end to end · [x] the translated Tensix map of
`NoC/Coordinates.md:28-46` matches hardware for X and Y, including the gap at X 8 (CPUs)
and X 9 (DRAM) · [x] harvested columns really are remapped to maximal X as
`NoC/Coordinates.md:54` claims — the derivation depends on it and `0xfff` confirms it ·
[x] 202 2 MiB windows with the last reserved for the driver (`blackhole.c:22,42`), 201
allocatable · [x] a Tensix tile's L1 ends at 1536 KiB (`BabyRISCV/README.md:102`): the
last dword is writable and reads back.

---

## Hardware bug register — Tier 1

Documented, not speculative. These bite in Phases 2–4.

- [x] **#2 Manual TTSync load adjacency.** Handled: emitted as one `asm!` block
      with a consuming ALU instruction.
- [x] **#3 Auto TTSync does not cover push-then-load.** Sidestepped: the `Dst` read
      path uses Manual TTSync, which Auto TTSync does not cover either way.
- [ ] **#1 Interrupt handlers cannot write CSRs.** Kills the usual
      save/restore-`fcsr` ISR pattern. Interrupts exist only on B and NC.
- [ ] **#4 `pmacfg0`/`pmacfg1` range selection is broken** — always selects the
      entire address space. Only usable as a global "every load/store behaves as if
      preceded by `fence`" debug switch.
- [ ] **#5 `intp_restore_pc` is only a copy** — writing it does not change the
      `mret` target. With #1, you cannot redirect an interrupt return.
- [ ] **#6 Operand-forwarding starvation hang.** If two unretired instructions both
      write the same register, forwarding happens from neither, and a specific
      sequence hangs forever. Replacing an `addi` with a `nop` flips a working
      example into a permanent hang.
- [ ] **#7 L0 data cache is not coherent.** Nothing written by the NoC, unpackers,
      packers, or another baby invalidates it. Polling loops over L1 **must**
      contain a `fence` or use an atomic. Hits also carry a ~0.8% random
      full-flush chance unless `cfg0.DisLowCachePeriodicFlush` is set.
- [ ] **#8 Store-then-load with non-overlapping byte ranges** has no ordering
      guarantee in either direction.
- [ ] **#9 T0/T1/T2 storing to `INSTRN1_BUF_BASE`/`INSTRN2_BUF_BASE` hangs the
      RISCV.** A stray pointer is an unrecoverable lockup, not a fault. *(See the
      hazard-encoding list above.)*
- [ ] **#10 Configuration Unit cross-thread starvation.** Ordering is enforced
      across all threads regardless of issuer; heavy `WRCFG` use by one thread
      starves RISC-V read/write requests from others.

### Tier 2 — once the Tensix units are driven

- [x] `SFPMAD` — automatic stalling misses seven documented cases, and
      `Instruction::stalls_automatically_after_mad` now decides all of them rather
      than returning `true` for everything. Four depend on the consuming
      instruction's `Mod1`, which it can read because an `Instruction` carries its
      definition. Untested against silicon, and ttsim models no timing, so the
      answers are from the documentation.
- [ ] `SFPLUTFP32` writes to `LReg[LReg[7] & 15]` instead of `LReg[VD]`. *(Phase 10: S4.)*
- [x] `SFPPOPC` — complex modes must not be used with a full conditional-execution stack.
      Handled by construction (`hardware-coverage.md` F2): `tt_kernels::sfpu::Program`
      emits only the plain push and pop, balanced by scope, and refuses a ninth level.
      Blackhole's `SFPPOPC.md` says in a tip that Blackhole fixed the bug and in its
      summary that the complex modes still must not be used on a full stack; the
      builder follows the summary.
- [ ] `SFPSTOCHRND` — stochastic rounding is biased toward increasing magnitude,
      and the new-in-Blackhole round-toward-zero mode sometimes rounds *away* from
      zero. **The functional models in the docs faithfully reproduce the buggy
      behaviour — match them, don't "fix" them.** *(Phase 10: S6.)*
- [ ] `SFPCAST_IntAbs` — the bug makes it compute absolute value; use `SFPABS`. *(Phase 10: S5, S6.)*
- [ ] `CLEARDVALID` — reset is unsafe and drops `SrcA`/`SrcB` banks
      ([tt-metal#22383](https://github.com/tenstorrent/tt-metal/issues/22383)).
- [ ] `STREAMWRCFG` — later Configuration Unit instructions from the same thread
      can reorder ahead of a pending one.
- [ ] `SETC16` — scheduling restrictions on `CFG_STATE_ID_StateID` work around Auto
      TTSync bugs.

---

## Open questions

- [ ] **1.** Blackhole I-cache capacities are undocumented. Measure in Phase 2.
- [ ] **2.** Blackhole debug-interface parity is unverified — the four `RISC_DBG_*`
      registers are named at Wormhole's base but no Blackhole bit layouts exist.
      Prototype against silicon before committing to a GDB-stub architecture.
- [x] **3.** Does `burn-fusion` compose with a hand-written backend? **Yes** --
      see Phase 7 and `RUST_IMPL_PLAN.md`, "The Burn surface, as pinned".
- [x] **4.** Is the NoC Overlay required for multi-chip? **No.** TT-link L1 writes carry the data (Phase 8).
- [ ] **5.** PCIe DMA engines have no register-level documentation anywhere in the
      repo. Plan on TLB-window MMIO for bulk transfer; revisit only if bandwidth
      demands it, and expect driver-source reverse engineering.
- [~] **6.** How faithfully does ttsim model reset sequencing, I-cache invalidation,
      and the local-RAM zeroing window? **Zeroing: not at all** (rows 25, 26), while
      silicon behaves as documented (`silicon_local_ram`). Reset sequencing is closed
      on silicon (Phase 2); ttsim models only the RISC-V reset bits (rows 16, 18).
      I-cache invalidation is still avoided by construction (Phase 2).
- [ ] **7.** Does ttsim model the documented hardware bugs, or the intended
      behaviour? Probe with targeted tests in Phase 3.
- [x] **9.** *(Resolved for the formats this phase needs.)* **The 4-bit
      `InDataFormat`/`OutDataFormat` codes are undocumented** — absent from the
      specification tree *and* from `cfg_defines.h`. Now **measured** rather than
      transcribed: `probe_unpack::survey_the_data_format_codes` unpacks known FP32
      datums once per `(in, out)` pair, and exactly three of 256 run, all returning
      the staged bits unchanged. Combined with two independent documented
      constraints — `Packers/InputAddressGenerator.md`'s `In_data_format & 3` size
      switch and `UNPACR_Regular.md:95`'s list of the four-byte formats — that gives
      `0 = FP32`, `4 = TF32`, `8 = INT32`. Recorded on `tt_isa::tile::L1Format::code`
      as `MEASURED`, a deliberately different marker from `UNVERIFIED`: the latter
      is a hypothesis from a document, this is a measurement where no document
      exists. **`1 = FP16` and `5 = BF16` since measured the same way through the
      `Src` path** (divergence row H), which ttsim does not decline. **Still
      open:** the block-float and 8-bit codes. Divergence rows G and H; re-derive
      on silicon.

- [x] **8.** The Tensix grid topology was measured against ttsim (140 tiles) and
      was wrong for silicon (120, harvested; row 35). It now comes from the chip
      (ARC tag 34, `grid::Tensix`), confirmed on both cards by Phase 1's gate.
