# Implementation checklist

The working tick-list for `RUST_IMPL_PLAN.md`. The plan says *why*; this says *what
is done, what is next, and what must not be forgotten*.

**Status legend:** `[x]` done and gated by a test · `[~]` partially done, see note ·
`[ ]` not started · `[-]` deliberately not applicable here, with the reason given.

**The two-gate rule.** No item is complete until it passes on the simulator *and* on
silicon. **Silicon now exists here:** two Tenstorrent p150a cards (Blackhole), PCIe-passed
through to this VM, driver `tenstorrent` 2.11.0, firmware bundle 19.14.0.0. Phase 1's
silicon gate is closed; every later phase's is still open.

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
| 0 — Simulator harness | `[~]` | `libttsim` path done; `ttsim-qemu` not started |
| 1 — Host addresses the chip | `[x]` | **Gated on simulator and on silicon** (both p150a cards, 7/7). `tt-kmd` done; harvesting read from ARC telemetry |
| 2 — Rust on a baby RISC-V | `[~]` | Heartbeat runs; silicon gate and hot-reload path open |
| 3 — Encoder + first Tensix round-trip | `[~]` | SFPU round-trip, `tt-isa-gen` and the encoding corpus done; tracing and the silicon diff open |
| 4 — Layout | `[~]` | Host-side tilization and the L1 image done; silicon gate open |
| 5 — Elementwise binary | `[~]` | Simulator gate done, FP32 only; BF16 blocked on ttsim, silicon gate open |
| 6 — Matmul | `[ ]` | The schedule risk |
| 7 — Burn backend, training | `[ ]` | The milestone |
| 8 — Multi-chip | `[ ]` | Spike first, then re-estimate |
| 9 — Performance | `[ ]` | Silicon-only |

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
- [x] **Divergence log** — `docs/ttsim-divergence.md`, 26 entries and counting.
- [x] **Silicon-only suite exists** — `--features silicon`, compiled always so it
      cannot rot.
- [x] **Version control**, so the pinning discipline above is enforceable.
- [x] **CI: simulator suite on every commit.** Deterministic, so no flakes.
      Also fmt, clippy under `-D warnings`, and a type-check of the silicon suite.
- [x] **CI: `cargo xtask check-no-sim-in-ship`** wired in.
- [x] **CI: `cargo xtask gen-cfg --check`**, so the committed configuration table
      cannot drift from the pinned header.
- [x] **CI: `cargo xtask gen-isa --check`**, likewise for the instruction table —
      and it re-runs the cross-check between the specification's two descriptions
      of the instruction set, so an exception edited without looking at what it
      explains fails there.
- [x] **Pre-commit hooks** (`prek.toml`) running fmt, clippy for both workspaces,
      and the generated-table check — the same things CI runs, so a push does not
      fail on something a commit could have caught.
- [ ] **CI: silicon suite nightly, and as a merge gate to main.**
- [x] **`burn-flex` as the second differential oracle.** ttsim is the ISA-level
      oracle; this is the tensor-level one. Needed from Phase 5. **Not
      `burn-ndarray`**, which crates.io now marks `[Deprecated] … use burn-flex,
      burn-cuda, burn-rocm`; `burn-flex 0.21.0` is the supported CPU backend and is
      what `PINS.toml` should pin. Taken as a `tt-tests` dev-dependency at 0.21.
      Note for Phase 7: in 0.21 the associated `Device` lives on `BackendTypes`, not
      on `Backend`, so the supertrait list in `RUST_IMPL_PLAN.md` is already stale —
      the item that says to re-verify it against the pinned version is load-bearing.

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
- [ ] **A dropped `Window` silently leaks its TLB window.** `Window` has no `Drop`
      that returns the index to `Device`'s free list, so a gate that allocates and
      does not `free_window` shrinks the pool for the rest of the process. Nothing
      in the type system says so; it is currently a comment on `scrub`, which is
      exactly the failure this section is about. It cost the Phase 1 silicon gate a
      run (see Silicon operating notes) and the diagnostic was a bare
      `OutOfBounds { offset: 0, len: 0 }` from an unrelated function.
      **Why it is still open:** the obvious fix does not typecheck. `Drop` cannot
      take `&mut Device`, so returning the index needs either a borrow of the
      device in `Window` (which makes holding several windows at once — what the
      exhaustion gate and every multi-window transfer do — a borrow conflict), or
      shared interior mutability for the free list, or a `Device::scope`-style
      closure owning the allocation. Pick one deliberately in Phase 2, when the
      firmware path starts holding windows across calls and the cost of getting it
      wrong goes up. Until then the exhaustion gate frees explicitly and `scrub`
      names the cause.
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
- [ ] **`UnpackToDst` clobbers `SrcA[Bank]`** — `UNPACR_Regular.md:441`, rows not
      characterised. Not load-bearing yet (nothing reads `SrcA`), but it makes the
      two unpack modes mutually exclusive and Phase 6 must encode that.
- [ ] `Dst` exclusivity across the three Tensix threads. *(Phase 5–6.)*
- [ ] `SrcA`/`SrcB` bank ownership handshake. *(Phase 6.)*
- [ ] `NOC_CMD_WR_INLINE` must never target an L1 address. *(Needed once a core
      drives an NIU; the host path does not.)*
- [ ] `NOC_CMD_L1_ACC_AT_EN` must always be `false`.

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

### Deferred to step 5 of the baseline plan

- [ ] Build `ttsim-qemu` (a single-patch QEMU fork) from source.
- [ ] Provision a Linux guest image; build and `insmod` `tt-kmd` inside it.
- [ ] Launch with the Blackhole-specific `bar4-size=32G`.
- [ ] Run the Phase 1 and 2 gates through `/dev/tenstorrent/0` in the guest.

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

**Run silicon gates with `--test-threads=1`.** All tests share one physical card. Window
allocation is global card state, so `window_exhaustion_is_an_error_not_a_panic`'s
`assert_eq!(held.len(), 201)` is only true if nothing else holds a window. The simulator
hides this by handing out a fresh chip per call.

**`Window` has no `Drop` that reaches the free list.** A gate that allocates must
`free_window`; dropping leaks it. This surfaced only on silicon, because silicon's
`in_device` scrubs the gate tile *after* the body and needs a window to do it, while the
simulator's never scrubs at all. The one gate whose job is to exhaust windows was the one
that starved the cleanup path. Tracked as an open item under Hazards to encode in the API,
with the three candidate designs and why the obvious one does not typecheck — not left as
"worth fixing at some point", which is the phrasing that produced the 140 in the first
place.

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
safety net that recovers a hung chip, so it is for bring-up, not for keeping.

**Do not run `probe_niu.rs` as a grid oracle.** It opens `Simulator` directly, so it is
structurally simulator-only and *cannot* reach a card — which is fortunate, because
finding "the highest addressable byte at each coordinate" is precisely the sweep that
hangs on a fused-off tile. Its 140 is where the bad constant came from.

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
- [ ] **`pc` snapshot cross-check** — silicon only, ttsim does not model the
      registers. The test is written and `#[cfg(feature = "silicon")]`.
- [ ] **Gate (silicon): the highest-value silicon gate in the plan.** Reset
      sequencing, I-cache invalidation and the local-RAM zeroing window are all
      things a simulator may model loosely, and all three land here.
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
      parses `Diagrams/Src/Bits32.lua` into 148 instruction encodings and 19 datum
      layouts, each cross-checked against the hand-written `TT_*(…)` syntax block
      on the page that embeds its diagram. The two sources agree on names, widths,
      signedness **and slot bit positions** across 167 pairs, with ten documented
      exceptions. Provenance is derived from which tree embeds the diagram: 39
      Blackhole, 24 shared, 11 superseded, 74 Wormhole-only and marked
      `UNVERIFIED`.
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
- [ ] **Stand up tracing.** `DebugTimestamper` gives a tile-wide 64-bit counter at
      `0xFFB1_21F0` plus a hardware event-trace primitive: one store to
      `RISCV_DEBUG_REG_TIMESTAMP` appends `{29-bit token, 64-bit counter}` to an L1
      ring buffer. Strictly better than per-core `mcycle` for correlating events
      across the five babies, and it pays for itself from Phase 5 onward. Use the
      documented retry loop for concurrent readers.
- [~] **`SFPLOADMACRO`.** The simulator's refusal is now *watched* rather than
      quoted: `step5_corpus.rs` pushes one, asserts the child dies, and runs a
      control program of the same shape that survives. A silicon-side test of what
      it actually does is still open.
- [ ] **Probe whether ttsim models the documented hardware bugs** (open question 7)
      or the intended behaviour. Either answer is workable but changes what the
      simulator gate proves.
- [ ] **Gate (silicon):** the corpus diffed against the simulator run. Any mismatch
      is a real finding, since ttsim targets bit-exactness.

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
- [ ] **Gate (silicon):** written and `#[cfg(feature = "silicon")]`, two tests.
      **The alignment scope is narrower than this plan assumed.**
      `WormholeB0/NoC/Alignment.md:19,23` says data travelling *from the host via
      PCIe to an L1 address* has **no alignment restrictions at all**, so the host
      staging path cannot violate anything. The C16 congruence applies when an L1
      address is the *source* — the unpacker and packer driving the NoC — which is
      Phase 6. The silicon test therefore asserts the documented "Any" across eleven
      deliberately misaligned tile bases; a failure is a finding against a
      Wormhole-sourced page Blackhole does not carry.

### Two corrections to `RUST_IMPL_PLAN.md`

1. **The conversion reference.** The plan says Phase 4 converts "using the documented
   `Dst`/`Src` bit layouts". Those describe the **register files** — 19-bit `Src`
   datums, swizzled 16/32-bit `Dst` ones — and are already generated into
   `tt_isa::isa::generated::datum`. Host↔**L1** conversion is governed by
   `FloatBitPatterns.md` and `Packers/FormatConversion.md:85-104`. Both matter, for
   different directions: the `Dst` layouts are what the corpus firmware's readback
   path uses. Implementing against the register layout here would have been wrong.
2. **The alignment gate**, as above: the host→L1 path is documented as unrestricted.

### Open, and deliberately so

- [ ] **Which `Z` plane is which face of a tile is a convention, not a
      specification.** The address generator fixes the *order* datums are visited;
      nothing says which 2-D patch a `Z` plane corresponds to.
      `Layout::tt_metal_32x32` picks row-major faces of row-major datums, and
      `placement.rs` **pins** that choice with a test whose failure means the
      convention changed — it does not claim the choice is correct. Settle it in
      Phase 6 against what the unpacker's ADC walk wants; it is a one-line change.

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
      **Controls watched failing:** expecting the host's `f32` multiply instead of
      the model; an empty kernel; swapped operand row groups; a reversed Burn
      operand.
- [ ] **Gate (silicon):** same suite. A mismatch here is a high-value bug report —
      it means the golden reference and the hardware disagree.

---

## Phase 6 — Matmul

**The schedule risk of this project.** Precisely where the Blackhole tree is
thinnest: `Unpackers/` and `Packers/` do not exist in it at all, `SrcASrcB.md` and
`RWCs.md` are absent, and `SrcASrcB.md` is not even *linked* from Blackhole — so
fidelity phases are documented in exactly one place and it is the wrong tree. BH
`PACR.md` is self-labelled "basic" and admits its `ReadIntfSel` interaction is
undocumented.

- [ ] `UNPACR` into `SrcA`/`SrcB`.
- [ ] Double-bank `SETDVALID`/`CLEARDVALID` handshake.
- [ ] `MVMUL`, fidelity phases.
- [ ] `PACR` out.
- [ ] Single 32×32 tile → blocked → multi-tile.
- [ ] Budget a standing percentage of the phase for empirical discovery rather
      than implementation.
- [ ] Do unpacker/packer bring-up **entirely in the simulator** — ttsim is
      intentionally more restrictive than silicon and raises `UndefinedBehavior` on
      misconfigured state, turning the largest documentation gap in the project
      into loud, deterministic failures instead of silent wrong data. Treat each
      raise as a specification question to answer before moving on.
- [ ] **Gate (sim):** matches `burn-ndarray` across shapes, dtypes, fidelity
      phases, accumulation depths.
- [ ] **Gate (silicon):** same suite. Expect divergence here more than anywhere
      else — this is where Wormhole-sourced assumptions will be wrong.

---

## Phase 7 — Burn backend, training

**This is the milestone.** Everything before it is infrastructure; everything after
is reach or speed.

- [ ] Verify the `Backend` supertrait list against the **pinned** Burn version —
      it moves quickly, and `QTensorOps`/`TransactionOps`/`BackendTypes` are recent
      additions.
- [ ] Implement a narrow core: matmul, add/sub/mul, relu, reshape, transpose,
      reduce sum/mean, broadcast. Let Burn's defaults compose the rest — slow but
      correct. Replace defaults by profiling, not by guess.
- [ ] `QTensorOps` may start unsupported if quantization is out of scope.
- [ ] **Resolve open question 3 before designing kernel dispatch:** does
      `burn-fusion` compose with a hand-written backend? CubeCL-based backends
      compose with autodiff *and* fusion; external ones likely with autodiff only.
      Tensix strongly wants fused unpack→math→pack chains, so fusion must either
      live inside `burn-tt` or be revisited as a CubeCL-target question. **This
      shapes Phase 9.**
- [ ] **Gate (sim):** MNIST MLP trains through `burn-autodiff` on a reduced
      dataset — loss descends, final weights match ndarray within tolerance,
      optimizer step correct. Determinism makes this a regression test too.
- [ ] **Gate (silicon):** full MNIST run matching the simulator's loss curve.

---

## Phase 8 — Multi-chip

Widest uncertainty band in the plan, and sequenced last on purpose: building
inter-chip transport before single-chip compute is correct is the most common way
projects of this shape stall.

- [ ] **Run a 2-week timeboxed spike against `bh_x2` first, then re-estimate.**
      Far cheaper than originally scoped — no second card, no link training, and
      cheaper again now that the spike no longer starts from enumeration: opening
      `bh_x2` and getting one `Device` per chip is done and gated (Phase 1). What
      remains is inter-chip *transport*.
- [ ] Resolve: **Ethernet-tile reset sequencing is undocumented for Blackhole.**
      There is no `BlackholeA0/EthernetTile/SoftReset.md`.
- [ ] Resolve: **RISCV E0 is cooperatively shared with Tenstorrent firmware** — it
      owns link training and calls into customer code. You cannot simply take it;
      `ethdump` uses E1 exclusively. The contract is documented only in the
      Wormhole tree.
- [ ] Resolve: **the Blackhole Ethernet PIC is effectively undocumented.**
- [ ] Resolve (open question 4): **is the NoC Overlay required?** There is no
      `BlackholeA0/NoC/Overlay/` directory at all, yet five Blackhole pages link
      into it. If it is needed, this is Wormhole-docs-plus-silicon work and the
      estimate grows substantially.
- [ ] Note: Ethernet tiles lack the Tensix conveniences — local data RAM is **not**
      NoC-accessible and there are **no `pc` snapshots**.
- [ ] **Gate (sim):** transfers between two chips under `bh_x2`; a model shards
      across `bh_x4` and matches the single-chip run.
- [ ] **Gate (silicon):** the same across two physical cards. Confirm Ethernet
      reset and the E0 firmware contract on silicon specifically — exactly the
      areas a simulator is most likely to model loosely, and ttsim's own README
      flags multichip as less mature than single-chip.

---

## Phase 9 — Performance (silicon-only)

- [ ] `MOP`/`REPLAY` expansion.
- [ ] Three-thread pipelining: unpack on T0, math on T1, pack on T2.
- [ ] Double buffering; multi-tile distribution with NoC multicast.
- [ ] `.ttinsn` fusion — up to four adjacent pushes per cycle. Deferred from the
      baseline because fused words disassemble as garbage and the instruction-set
      gate would have to stop rejecting undecodable instructions.
- [ ] `L1CacheTagSearchAccel` — Blackhole-only, RISCV B only.
- [ ] **Keep using the simulator as the correctness gate** for each optimization —
      pipelined kernels are where correctness is hardest — then measure on silicon.
- [ ] **Every performance number comes from hardware.** ttsim is not cycle-accurate
      and makes no timing claim.
- [ ] Measure against the theoretical peak figures in the spec, not a competitor.

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
semantics) · [ ] `REPLAY` · [ ] `MOP`/`MOP_CFG`

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
- [ ] `SFPLUTFP32` writes to `LReg[LReg[7] & 15]` instead of `LReg[VD]`.
- [ ] `SFPPOPC` — complex modes must not be used with a full conditional-execution stack.
- [ ] `SFPSTOCHRND` — stochastic rounding is biased toward increasing magnitude,
      and the new-in-Blackhole round-toward-zero mode sometimes rounds *away* from
      zero. **The functional models in the docs faithfully reproduce the buggy
      behaviour — match them, don't "fix" them.**
- [ ] `SFPCAST_IntAbs` — the bug makes it compute absolute value; use `SFPABS`.
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
- [ ] **3.** Does `burn-fusion` compose with a hand-written backend? Decide before
      designing kernel dispatch (Phase 7); it shapes Phase 9.
- [ ] **4.** Is the NoC Overlay required for multi-chip? Resolve in the Phase 8 spike.
- [ ] **5.** PCIe DMA engines have no register-level documentation anywhere in the
      repo. Plan on TLB-window MMIO for bulk transfer; revisit only if bandwidth
      demands it, and expect driver-source reverse engineering.
- [ ] **6.** How faithfully does ttsim model reset sequencing, I-cache invalidation,
      and the local-RAM zeroing window? All three land in Phase 2 and are the areas
      most likely to be modelled loosely. Resolve at the Phase 2 silicon gate.
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
      exists. **Still open:** every 16-bit and block-float code, because ttsim
      declines `UnpackToDst` for them entirely (divergence row 31); they must be
      pinned through the packer instead. Divergence row G; re-derive on silicon.

- [ ] **8.** *(New.)* The Tensix grid topology in `tt_isa::noc::grid` was **measured
      against ttsim**, not quoted — `NOC_ENDPOINT_ID` is unimplemented there, so the
      documented discovery probe is unavailable. Three independent figures agree,
      but re-derive it at the first silicon gate and treat a mismatch as a finding.
