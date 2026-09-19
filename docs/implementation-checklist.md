# Implementation checklist

The working tick-list for `RUST_IMPL_PLAN.md`. The plan says *why*; this says *what
is done, what is next, and what must not be forgotten*.

**Status legend:** `[x]` done and gated by a test · `[~]` partially done, see note ·
`[ ]` not started · `[-]` deliberately not applicable here, with the reason given.

**The two-gate rule.** No item is complete until it passes on the simulator *and* on
silicon. There is no silicon on this machine, so every silicon gate below is open.
That is a known, tracked gap — not an oversight — and the divergence log
(`ttsim-divergence.md`) exists to make the eventual comparison cheap.

**Before ticking a box, watch the gate fail.** Two gates in this repo were initially
vacuous: the disassembly check matched no lines at all, and a discarded
`.riscv.attributes` section made forbidden instructions decode as `<unknown>`. A
gate you have not seen reject something is not yet evidence.

---

## Where things stand

| Phase | State | Note |
|--:|---|---|
| 0 — Simulator harness | `[~]` | `libttsim` path done; `ttsim-qemu` not started |
| 1 — Host addresses the chip | `[~]` | TLB/L1 done against the simulator; `tt-kmd` not started |
| 2 — Rust on a baby RISC-V | `[~]` | Heartbeat runs; silicon gate and hot-reload path open |
| 3 — Encoder + first Tensix round-trip | `[~]` | SFPU round-trip and `tt-isa-gen` done; corpus and tracing open |
| 4 — Layout | `[ ]` | |
| 5 — Elementwise binary | `[ ]` | |
| 6 — Matmul | `[ ]` | The schedule risk |
| 7 — Burn backend, training | `[ ]` | The milestone |
| 8 — Multi-chip | `[ ]` | Spike first, then re-estimate |
| 9 — Performance | `[ ]` | Silicon-only |

---

## Cross-cutting

Set up early; retrofitting is expensive.

- [x] **Pin upstream revisions** — `PINS.toml` records the spec commit, the ttsim
      release, and the tt-metal commit for `cfg_defines.h`.
- [ ] **Re-sync the spec quarterly.** Recent upstream commits fill in exactly the
      Blackhole `UNPACR`/`PACR` gaps Phase 6 depends on. Check before doing
      silicon-discovery work the docs may have obviated.
- [x] **One newtype per coordinate space** — `NocCoord<Noc0>`, `NocCoord<Noc1>`,
      `Translated`, plus `ChipId` from day one.
- [x] **Divergence log** — `docs/ttsim-divergence.md`, 21 entries and counting.
- [x] **Silicon-only suite exists** — `--features silicon`, compiled always so it
      cannot rot.
- [x] **Version control**, so the pinning discipline above is enforceable.
- [x] **CI: simulator suite on every commit.** Deterministic, so no flakes.
      Also fmt, clippy under `-D warnings`, and a type-check of the silicon suite.
- [x] **CI: `cargo xtask check-no-sim-in-ship`** wired in.
- [x] **CI: `cargo xtask gen-cfg --check`**, so the committed configuration table
      cannot drift from the pinned header.
- [x] **Pre-commit hooks** (`prek.toml`) running fmt, clippy for both workspaces,
      and the generated-table check — the same things CI runs, so a push does not
      fail on something a commit could have caught.
- [ ] **CI: silicon suite nightly, and as a merge gate to main.**
- [ ] **`burn-ndarray` as the second differential oracle.** ttsim is the ISA-level
      oracle; this is the tensor-level one. Needed from Phase 5.

### Hazards to encode in the API, not in comments

This is the concrete answer to "why Rust and not C++"; if it is not done, there is
no answer.

- [x] Manual TTSync load adjacency — emitted as one `asm!` block, never left to LLVM.
- [x] TLB `linked` bit — not expressible; the spec says it is never safe from the host.
- [x] Multicast windows are write-only — a multicast read is fatal.
- [x] RISCV B has no reset-PC override — `set_reset_pc` refuses it.
- [x] `LReg[8..]` unwritable by `SFPLOADI`/`SFPMAD` — refused rather than silently dropped.
- [x] `SFPMUL` requires `VC == 9` — `VC` is implicit, not a parameter.
- [ ] **`INSTRN1_BUF_BASE`/`INSTRN2_BUF_BASE` from T0/T1/T2 hangs the core.**
      Currently only documented. Make the buffer address a type parameterised by
      core role so the bad combination does not compile. *(Phase 3 leftover.)*
- [ ] `Dst` exclusivity across the three Tensix threads. *(Phase 5–6.)*
- [ ] `SrcA`/`SrcB` bank ownership handshake. *(Phase 6.)*
- [ ] `NOC_CMD_WR_INLINE` must never target an L1 address. *(Needed once a core
      drives an NIU; the host path does not.)*
- [ ] `NOC_CMD_L1_ACC_AT_EN` must always be `false`.

### Tolerance policy

- [x] Never a guessed epsilon — assert bit patterns against the documented model.
- [ ] Derive expected values from `Miscellaneous/FMA/fma.c` (`fma_model_bh`) once
      arithmetic beyond a single multiply is in scope.

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
- [x] `xtask fetch-ttsim` with a pinned tag and SHA-256.
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
      tiles, across a window boundary; all 140 Tensix tiles addressable without
      aliasing.
- [ ] **Gate (sim) under `bh_x2`** — chip indexing is typed but never exercised.
      Needs `libttsim_bh_x2.so` and a second `ChipId`.
- [ ] **`tt-kmd` crate** — open `/dev/tenstorrent/N`, mmap BAR0/2/4, wrap
      `ALLOCATE_TLB` (`0xFA0B`) / `CONFIGURE_TLB` / `FREE_TLB`,
      `GET_DEVICE_INFO` (`0xFA00`), `QUERY_MAPPINGS` (`0xFA02`),
      `SET_NOC_CLEANUP` (`0xFA0E`).
  - [ ] `QUERY_MAPPINGS` takes `&mappings[0].mapping_size`, not the struct base.
  - [ ] `SET_NOC_CLEANUP` re-asserts reset if the process dies — any real runtime
        needs it. Returns `EINVAL` on older `tt-kmd`; ethdump ignores the result.
- [ ] **Gate (silicon):** the same round-trip on a real p150, and across two cards
      if available.

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
      no undecodable instructions, no unrecognised `fence` ordering.
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
- [ ] **Local-RAM zeroing window.** Also avoided by construction — nothing is
      staged into local RAM over the NoC. Needed if that changes: either set the
      `RISCV_DEBUG_REG_DISABLE_RESET` bit first or wait out 2048 cycles.
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
- [ ] **Broad instruction corpus, asserted bit-exact.** Currently five
      instructions. `Bits32.lua` is machine-readable and yields field layouts for
      the whole ISA, so a generator is likely less work than hand-writing more.
- [ ] **Stand up tracing.** `DebugTimestamper` gives a tile-wide 64-bit counter at
      `0xFFB1_21F0` plus a hardware event-trace primitive: one store to
      `RISCV_DEBUG_REG_TIMESTAMP` appends `{29-bit token, 64-bit counter}` to an L1
      ring buffer. Strictly better than per-core `mcycle` for correlating events
      across the five babies, and it pays for itself from Phase 5 onward. Use the
      documented retry loop for concurrent readers.
- [ ] **Seed the silicon-only suite with `SFPLOADMACRO`.** The suite exists but
      holds the `pc`-snapshot and held-backend tests, not this.
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

- [ ] Row-major strided ↔ 32×32 tiled.
- [ ] Padding for non-multiple-of-32 shapes.
- [ ] FP32/BF16/FP16 conversion using the documented `Dst`/`Src` bit layouts, not
      IEEE assumptions.
- [ ] **Gate (sim):** property-tested round-trip, all dtypes, awkward shapes
      (`[13, 47]`, `[1, 1, 1024]`). Cheap in the simulator, expensive on hardware.
- [ ] **Gate (silicon):** representative sample plus every shape that exercises an
      alignment boundary. `BlackholeA0/NoC/Alignment.md` does not exist and
      violations are `UndefinedBehavior`, so this gate is where those get validated.

---

## Phase 5 — First real kernel: elementwise binary

Eltwise before matmul deliberately: it exercises unpack → SFPU → pack with no
`SrcA`/`SrcB` bank handshake, no fidelity phases, no RWC choreography.

- [ ] Unpack → SFPU → pack for a binary op.
- [ ] **Gate (sim):** differential vs `burn-ndarray`, with tolerances derived from
      the documented FMA divergence rather than guessed.
  - [ ] Include denormals — the SFPU flushes them.
  - [ ] Include NaN — Blackhole canonicalises differently from IEEE *and* from
        Wormhole. ttsim replaces every NaN with `0x7FC0_0000`.
  - [ ] Assert exact equality against `fma_model_bh` wherever the operation is a
        pure FMA.
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
      Far cheaper than originally scoped — no second card, no link training.
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

- [ ] `SFPMAD` — automatic stalling misses a handful of cases; see
      `Instruction::stalls_automatically_after_mad` for where to record them.
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
- [ ] **8.** *(New.)* The Tensix grid topology in `tt_isa::noc::grid` was **measured
      against ttsim**, not quoted — `NOC_ENDPOINT_ID` is unimplemented there, so the
      documented discovery probe is unavailable. Three independent figures agree,
      but re-derive it at the first silicon gate and treat a mismatch as a finding.
