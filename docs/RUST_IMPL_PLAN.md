# Rust-Native Software Stack for Tenstorrent Blackhole → Burn Backend

**Status:** Phases 0–8 closed on ttsim and silicon; Phase 9 (performance) and Phase 10
(hardware coverage) in progress. Current state per item is in the checklist.
**Target hardware:** Blackhole A0 (p100 / p150)
**Spec source:** `tt-isa-documentation` @ `f848eb6` (2026-09-18)
**Working checklist:** [`docs/implementation-checklist.md`](implementation-checklist.md)
— per-phase tick-list, current state, and the verification backlog as actual checkboxes.
**Hardware coverage tracker:** [`docs/hardware-coverage.md`](hardware-coverage.md)
— Phase 10: which Tensix units and Burn ops run on the device, and the order the rest arrives.
**Implementation guides:** [`docs/tt-metal-concepts-review.md`](tt-metal-concepts-review.md)
— Tenstorrent-system concepts we lack or do differently (G1–G16), and the hardware sharp
edges to handle in code; [`docs/burn-backend-parity.md`](burn-backend-parity.md) — what
`burn-tt` needs to be a CUDA-grade Burn backend (roadmap B0–B16).

---

## Context

### Why this project exists

Tenstorrent's Blackhole ASIC is programmed today through a C++ stack (TT-Metalium → TT-NN →
TT-Forge). There is no Rust-native path to the hardware, which means Rust ML frameworks —
specifically [Burn](https://github.com/tracel-ai/burn) — cannot target Tenstorrent accelerators
without dragging in a C++ toolchain and its abstractions.

This project builds that path from scratch: a Rust stack spanning bare-metal device firmware,
a host-side runtime, and a Burn `Backend` implementation, with **no C++ dependency at any
layer**.

### What makes this tractable

Tenstorrent publishes the ISA. The `tt-isa-documentation` repository specifies the Tensix
coprocessor instruction set, the baby RISC-V cores, the NoC, and the host interface in enough
detail to implement against — including per-instruction pseudocode and bit-exact numerics
models. A golden-reference simulator ([ttsim](https://github.com/tenstorrent/ttsim)) exists for
differential testing. This is a well-specified target, not a reverse-engineering effort.

### What makes this hard

1. The Tensix coprocessor has **no instruction fetch**. Its instruction stream is pushed word
   by word by the baby RISC-V cores. Writing a "kernel" means emitting a choreographed
   instruction stream across three concurrent hardware threads that share most of their
   backend state.
2. The Blackhole documentation tree is **incomplete** — roughly 55 internal links point at
   pages not yet written, concentrated in exactly the packer/unpacker/scalar-unit area that
   matmul requires. Those must be read from the Wormhole tree and empirically verified.
3. Tensix uses a **32×32 tiled data layout** with block-float formats that have no equivalent
   in Burn's strided row-major tensor model.

### Scope decisions (confirmed with requester)

| Decision | Choice |
|---|---|
| Strategy | **Native Rust only** — no FFI to TT-Metalium at any point |
| Development target | **Simulator first, silicon as verification gate** |
| First milestone | **Training** — MNIST MLP, forward + backward + optimizer step |
| Team | **One engineer**, sequential execution |
| Multi-chip | **In scope** — Ethernet tiles and inter-chip transfer are deliverables |

### Simulator-first development

Every phase is developed against [ttsim](https://github.com/tenstorrent/ttsim) and only then
verified on silicon. This is viable — and cheap — because ttsim is a **full-system** simulator,
not an ISA-level one. It presents a virtual Blackhole device including L1, NoC, and the PCIe
host interface.

Two integration paths, both usable:

1. **`libttsim.so` directly.** A flat C API with **no tt-metal dependency** (system C libraries
   only), so a small Rust `-sys` crate binds it:
   ```c
   void     libttsim_init(void);
   void     libttsim_exit(void);
   uint32_t libttsim_pci_config_rd32(uint32_t bdf, uint32_t offset);
   void     libttsim_pci_config_wr32(uint32_t bdf, uint32_t offset, uint32_t data);
   void     libttsim_pci_mem_rd_bytes(uint64_t paddr, void *dst, uint32_t size);
   void     libttsim_pci_mem_wr_bytes(uint64_t paddr, const void *src, uint32_t size);
   void     libttsim_clock(uint32_t n_clocks);
   void     libttsim_set_pci_dma_mem_callbacks(...);
   ```
2. **`ttsim-qemu`**, which exposes `libttsim.so` to a guest VM as `/dev/tenstorrent/0`.
   **Not used:** two p150a cards arrived at Phase 1, and the real ioctl and mmap paths are
   gated on them directly.

Path (1) is the validation tier: deterministic, in-process, run on every commit.

**Why this is the right call here:**
- ttsim's stated goal is **bit-exact numerical results relative to silicon** — so it is a valid
  oracle for numerics, not merely a functional approximation.
- It is **intentionally more restrictive than silicon**, raising `UndefinedBehavior` where
  hardware would silently proceed. Given how saturated this ISA is with UB cases, that turns a
  whole class of latent corruption bugs into loud failures.
- Time only advances inside `libttsim_clock`, so tests are **fully deterministic** — no
  flakiness, no hardware contention, and race conditions reproduce exactly.
- Multichip `bh_x2` (P300), `bh_x4`, and `bh_x32` (BH Galaxy) configurations exist "with
  significant numbers of Ethernet, multidevice, and fabric tests passing" — so even the
  multi-chip phase is developed before touching a second physical card.

**Note on the "native Rust only" constraint:** binding `libttsim.so` is a *development and test
dependency*, not a runtime one. Nothing ships with it. This does not compromise the native-Rust
goal the way an FFI dependency on TT-NN would.

**Simulator limitations to track** (each becomes a silicon-only verification item):
- **`SFPLOADMACRO` is not supported in the SFPU.** Any kernel using it cannot be
  simulator-validated.
- The library is **single-threaded, non-reentrant, and a process-wide singleton** with no handle
  or context object. The Rust binding must enforce singleton semantics.
- `bh_x32` is x86_64-only at present.
- Timing is **not** cycle-accurate. Performance work (Phase 9) is silicon-only.

### Intended outcome

A Rust workspace in which `cargo test` runs a differential suite against ttsim and silicon, and
in which a Burn model trains on one or more Blackhole cards using a backend written entirely in
Rust.

---

## Sizing and structure

The original estimate for native-only, training-capable, multi-chip, one engineer was 12–16
months; simulator-first development ran Phases 0–8 far faster than that. What the estimate
argued for still holds, and shapes everything below: independently demoable milestones;
multi-chip sequenced after single-chip training, on chip-indexed abstractions from Phase 1;
two binary gates per phase, simulator then silicon; and an escape hatch (narrow the
dtype/shape surface) that is never FFI.

---

## Workspace layout (as built)

```
crates/
  tt-isa/              # no_std — generated instruction and config tables, typed helpers
                       #   (sfpu, matrix, backend, sync, numerics, tile, dm, l1, noc, eth, arc)
  tt-ttsim-sys/        # dev-only — raw dlopen bindings to libttsim
  tt-ttsim/            # dev-only — singleton-safe Simulator, LibTtsim transport, fork_scope
  tt-kmd/              # host — /dev/tenstorrent ioctls, BAR mmap, TLB windows
  tt-device/           # host — Device over a Transport: tiles, NoC, DRAM, loading, telemetry
  tt-layout/           # tilization, padding, dtype conversion
  tt-firmware/         # no_std riscv32im bins (role_t0..2, dm_b, eth_e1, corpus, ...);
                       #   its own workspace, built by xtask
  tt-firmware-images/  # builds and checks the firmware, embeds the images
  tt-kernels/          # kernels and the runtime: matmul, tensor, session, dm, l1 planner,
                       #   program cache, shard, link
  burn-tt/             # the Burn backend
  tt-tests/            # ttsim and silicon gates (stepNN_*, probe_*, silicon_*)
  tt-mnist/            # shippable binary: MNIST training on the card
xtask/                 # gen-cfg, gen-isa, gen-burn-delegate, silicon, fetch-*, check-no-sim-in-ship
vendor/                # pinned specification checkout (PINS.toml)
```

`tt-isa` is the spine: `no_std`, zero host dependencies, compiled for **both** the host (to
generate instruction streams) and the device (to push them). Everything else depends on it.

`tt-ttsim` is a dev-dependency only and must never appear in the dependency graph of a shipped
artifact; `cargo xtask check-no-sim-in-ship` enforces it in CI.

**A transport trait is the key abstraction.** `tt-device` is generic over how it reaches the
chip — `LibTtsim` and `Kmd` (real `/dev/tenstorrent/N`) — so every phase gets simulator and
silicon execution from the same code, with no conditional compilation scattered through it.

## Documentation reference map

All paths are relative to the `tt-isa-documentation` checkout. **Status** is the single most
important column: `REAL` = substantive Blackhole page; `STUB` = one-line redirect to Wormhole;
`ABSENT` = linked from Blackhole docs but the file does not exist.

Where a page is `STUB` or `ABSENT`, the engineer must read the Wormhole equivalent **and treat
every fact as unverified until confirmed against ttsim or silicon.** The top-level README
states outright that Wormhole behaviour does not transfer automatically.

### The single most important structural fact

**`BlackholeA0/` is not a complete tree. It is a delta tree over `WormholeB0/`.**

| | BlackholeA0 | WormholeB0 |
|---|--:|--:|
| `TensixTile/TensixCoprocessor/*.md` | 73 | 142 |
| `.../Unpackers/` | **absent** | 2 files |
| `.../Packers/` | **absent** | 10 files |
| `.../README.md` (backend index) | **absent** | present |
| Dangling internal links | ~45 | 0 |

Every Blackhole page is one of four kinds, and **the stub trailer sentence encodes how much you
can trust the Wormhole page.** This is the trust model the engineer must internalize:

| Kind | Count | Trailer text | What it means |
|---|--:|---|---|
| **REAL** | 43 | — | Blackhole-specific page. Authoritative. |
| **STUB-A** | — | *"…conditionalized inline using `TTArchitecture`"* | **The Wormhole page covers Blackhole.** Look for `if (TTArchitecture == ...)` branches. Safe to implement from. |
| **STUB-B** | — | *"similar, but **not identical**"* | Wormhole conditionalizes *some* cases and warns you. Implement, then verify. |
| **STUB-C** | — | *"behavior is **identical**"* | Read Wormhole verbatim. Safe. |
| **DISPATCH** | 2 | — | Mode index, not a stub (`SFPCAST.md`, `SFPSTOCHRND.md`) |
| **ABSENT** | ~45 links | — | **No file anywhere in the Blackhole tree.** Read Wormhole, then verify on silicon. |

**Read `Glossary.md` (repo root) first.** It defines `TTArchitecture`, `NonContractualBehavior`,
`UndefinedBehavior`, `UnpredictableValue`, and `UnsupportedFunctionality`. These terms are
contractual and the compute docs are saturated with them. Treating `UndefinedBehavior` as
"probably fine" is how you get silent data corruption.

**Wormhole pages that are `TTArchitecture`-conditionalized** (therefore authoritative for
Blackhole — this is the good-news list):

```
ADCs.md  CLEARDVALID.md  ELWADD.md  ELWMUL.md  GMPOOL.md  MatrixUnit.md
MOPExpander.md  MOVA2D.md  MOVB2A.md  MOVB2D.md  MOVD2A.md  MOVD2B.md
MVMUL.md  Packers/OutputAddressGenerator.md  REG2FLOP_Configuration.md
SETDVALID.md  UNPACR_NOP_SETDVALID.md  UNPACR_NOP_ZEROSRC.md  UNPACR_Regular.md
```
(all under `WormholeB0/TensixTile/TensixCoprocessor/`)

Anything **not** on that list and **not** in `BlackholeA0/` is pure-Wormhole text with no
recorded Blackhole delta → empirical verification required.

### Start here — the one working reference implementation

`BlackholeA0/EthernetTile/Samples/ethdump/ethdump.c` (1393 lines, self-contained C) is the
closest thing in the repo to a reference host-side runtime, and it should be read in full
before writing a line of `tt-kmd`. It demonstrates, end to end:

- tt-kmd usage: `GET_DEVICE_INFO` (`0xFA00`), `QUERY_MAPPINGS` (`0xFA02`),
  `ALLOCATE_DMA_BUF` (`0xFA03`), `PIN_PAGES` (`0xFA07`), `ALLOCATE_TLB` (`0xFA0B`),
  `SET_NOC_CLEANUP` (`0xFA0E`) — with the numeric codes spelled out
- mmap of a BAR0 TLB window and direct userspace configuration (it never calls `CONFIGURE_TLB`)
- host memory pinning for device access (`NOC_DMA`, `NOC_TOP_DOWN` flags)
- crash-safety via `SET_NOC_CLEANUP` — any real runtime needs this
- topology discovery printing NoC #0 *and* logical/translated coordinates side by side
- injecting ~400 bytes of RISC-V machine code into Ethernet L1 and running it on RISCV E1
- a real producer/consumer ring across PCIe, including a worked example of using
  `NOC_CMD_VC_STATIC` to order a metadata write behind a data write

Companion writeup: `BlackholeA0/EthernetTile/Samples/ethdump/README.md`.

### Host ↔ device

| Path | Status | Gives you |
|---|---|---|
| `BlackholeA0/PCIExpressTile/README.md` | REAL | BAR layout (BAR0 512 MiB, BAR2 1 MiB DBI, BAR4 32 GiB), datapath, device→host 64-bit NoC map |
| `BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md` | REAL | 210 TLB windows; the 96-bit config layout; cites the tt-kmd ioctls |

Key facts: TLB config array lives at BAR0 `0x1FC0_0000` as
`struct { u32 low32, mid32, high32; } windows[210]; u32 strided[32];`. TLBs 0–31 are 2 MiB with
non-rectangular multicast; 32–200 are 2 MiB; 201 is **reserved for the kernel driver**;
202–209 are 4 GiB in BAR4. `linked` is documented as *never safe to set from the host*.
Keep/skip/exclude coordinates are **raw NoC coordinates even when translation is enabled** —
a genuine type-safety trap.

**Documentation gap:** the PCIe DMA engines are mentioned but have **no register-level
programming guide anywhere in the repo**. Treat DMA as requiring driver-source reverse
engineering; plan to use TLB-window MMIO for bulk transfer initially.

### NoC

| Path | Status | Gives you |
|---|---|---|
| `BlackholeA0/NoC/README.md` | REAL | Packet model (1 header + ≤256 flits × 64 B = 16 KiB max payload), VC layout, deadlock-liable features |
| `BlackholeA0/NoC/Coordinates.md` | REAL | Coordinate systems + the exact translation algorithm (lines 84–129) |
| `BlackholeA0/NoC/MemoryMap.md` | REAL | **The single most important page for the runtime** — NIU registers, 4 request initiators, every command bitfield |
| `BlackholeA0/NoC/RoutingPaths.md` | REAL | NoC0 = X-then-Y, NoC1 = Y-then-X; broadcast trees; needed to reason about `VC_STATIC` |
| `BlackholeA0/NoC/Atomics.md` | REAL | Six atomic forms encoded in `NOC_AT_LEN_BE`, with pseudocode |
| `BlackholeA0/NoC/Counters.md` | REAL | The completion-observation mechanism: 62-entry array at `NIU_BASE + 0x0200` |
| `BlackholeA0/NoC/Interrupts.md` | REAL | Blackhole-only alternative to polling counters |
| `BlackholeA0/NoC/Ordering.md` | **ABSENT** | → `WormholeB0/NoC/Ordering.md` |
| `BlackholeA0/NoC/Alignment.md` | **ABSENT** | → `WormholeB0/NoC/Alignment.md`; violations are UndefinedBehavior |

**Command sequence:** `NIU_BASE` = `0xFFB2_0000` (NoC0) / `0xFFB3_0000` (NoC1); four request
initiators at `+0x0000/0x0800/0x1000/0x1800`. Program `NOC_TARG_ADDR_{LO,MID,HI}`,
`NOC_RET_ADDR_{LO,MID,HI}`, `NOC_PACKET_TAG`, `NOC_CTRL`, `NOC_AT_LEN_BE`, `NOC_AT_DATA`, then
fire by writing 1 to `NOC_CMD_CTRL` (+0x40).

**Semantics trap:** for a non-inline write, `NOC_TARG_ADDR` is the **source** and `NOC_RET_ADDR`
the **destination** — the reverse of a read. The authoritative source/dest/ack matrix is
`MemoryMap.md:99-104`.

**Completion:** pick a 4-bit transaction ID, issue with `NOC_CMD_RESP_MARKED`, poll
`NIU_MST_REQS_OUTSTANDING_ID(i)` to zero. Counters are 8-bit, so a single hardware-split
transfer is capped just under 2 MiB.

### Ordering and alignment — the highest-risk documentation gap

Both Blackhole pages are absent. From the Wormhole equivalents, the constraints that must be
re-verified on silicon:

1. Within one packet, L1 accesses decompose into atomic aligned 16-byte units in arbitrary
   order. Use counters for "done"; never infer partial ordering.
2. Between packets, same source+dest+NoC ⇒ same route, but no-reorder requires the same VC at
   every hop — via `VC_STATIC`, `VC_LINKED`, or `NOC_CMD_CTRL` write order. **Responses and
   acks can always reorder.**
3. At the receiving NIU, up to 12 request VCs process concurrently ⇒ arbitrary cross-VC
   interleave. An MMIO write can race ahead of an earlier L1 write.
4. Alignment violations are UndefinedBehavior, with different congruence rules for MMIO vs L1
   vs other, and atomics requiring a 16-byte-aligned L1 target.
5. Blackhole's TLB `ordering` field is a 2-bit enum (0 Default / 1 Strict AXI / 2 Posted Writes
   / 3 Counted Writes) where Wormhole had separate bits — so this page is *close but not
   identical*. Do not copy assumptions.

### Hardware traps to encode in the API, not comments

- `NOC_CMD_WR_INLINE` must **never** target an L1 address (MMIO only) — `MemoryMap.md:70`
- `NOC_CMD_L1_ACC_AT_EN` is unusable; always `false` — `MemoryMap.md:84`
- After writing `NOC_CMD_CTRL`, **read it back** before reading any counter, or RISC-V
  reordering yields a stale value — `Counters.md:42-43`
- Broadcast + `RESP_MARKED` breaks counter semantics (one increment, N decrements); clear via
  `NIU_BASE + 0x0060` — `Counters.md:152-166`

### Coordinate spaces — the newtype discipline

Four distinct hardware spaces plus one software label:

1. **NoC #0 raw** — origin top-left, increments right/down
2. **NoC #1 raw** — origin bottom-right, increments left/up; mirrored, *not* interchangeable
3. **Translated** — used when `NIU_CFG_0` bit 14 is set; Blackhole's mapping is a combined X/Y
   table (a Y value selects which X table applies), unlike Wormhole's separable per-axis tables
4. **Raw-but-untranslated escape hatches** — PCIe TLB `strided` fields and L2CPU
   `x_exclude_coord` take raw coordinates *even when translation is on*
5. **"Logical"** — not hardware; the translated coordinate firmware writes into `NOC_ID_LOGICAL`

For broadcasts, software must **swap StartX↔EndX and StartY↔EndY between NoC0 and NoC1**.
Unlike Wormhole, Blackhole does **not** write translated coordinates back into MMIO registers,
so read-back code must not assume it does.

### L1

`BlackholeA0/TensixTile/L1.md` is a **STUB** → `WormholeB0/TensixTile/L1.md`. The Wormhole page
describes 16 banks; **Blackhole has 32** (`BabyRISCV/MemoryOrdering.md:36`). Firm Blackhole
facts: 1536 KiB per Tensix tile, 512 KiB per Ethernet tile, 32 banks, L1 load latency ≥ 8
cycles on L0 miss and ≥ 12 for atomics. Treat the Wormhole port/bank topology and per-client
bandwidth tables as shape-guidance only.

### NoC Overlay — entirely absent on Blackhole

There is **no `BlackholeA0/NoC/Overlay/` directory at all**, yet five Blackhole pages link into
it and Blackhole docs reference overlay behaviour normatively (`NOC_PACKET_TAG` overlay bits,
`VC_LINKED` mutual exclusion, eight `tt_cfg_sstatus*` CSRs aliasing overlay stream registers).

Wormhole provides a full subdirectory: `WormholeB0/NoC/Overlay/` — `README.md` (message model,
64 streams per Tensix tile, stream state machine, register reference), `TransferBetweenTiles.md`,
`ReceiveFromSoftware.md`, `TransmitToSoftware.md`, `Gather.md`, `AsGeneralUse.md`,
`LoadConfigurationFromL1.md`, `TransmitToDRAMBuffer.md`, `TransmitToNowhere.md`.

**Planning implication:** overlay use is optional if the runtime uses only NIU request
initiators. Phase 8 (multi-chip) should evaluate whether overlay streams are needed; if so,
that is Wormhole-docs-plus-silicon-validation work and the phase estimate should grow.

### Ethernet — multi-chip transport (Phase 8)

| Path | Status | Gives you |
|---|---|---|
| `BlackholeA0/EthernetTile/README.md` | REAL | 512 KiB L1, 2 RISCVs, 400 GbE as 3 TX + 3 RX queues |
| `BlackholeA0/EthernetTile/EthernetTxRx.md` | REAL | **The multi-chip transport reference** — raw vs TT-link mode, 18-byte header, sequence numbers, auto-retransmit, TX/RX register maps |
| `BlackholeA0/EthernetTile/EthernetRxClassifier.md` | REAL | Seven-stage ingress pipeline, TCAM → flow table |
| `BlackholeA0/EthernetTile/BabyRISCV/README.md` | REAL | RISCV E0/E1; **E0 is shared with Tenstorrent firmware** — customer code uses E1 |
| `BlackholeA0/EthernetTile/L1.md` | **ABSENT** | → `WormholeB0/EthernetTile/L1.md` |
| `BlackholeA0/EthernetTile/BabyRISCV/CallingIntoCustomerCode.md` | **ABSENT** | → `WormholeB0/EthernetTile/BabyRISCV/CallingIntoCustomerCode.md` |

### L2CPU tiles (optional on-chip host — all pages REAL)

`BlackholeA0/L2CPUTile/` — `README.md`, `MemoryMap.md`, `TLBWindows.md`, `Caches.md`,
`RNMIs.md`, `MSICatcher.md`. Four SiFive x280 harts per tile × 4 tiles = 16 cores, each tile
with a hardwired local DRAM tile and 256 NoC TLB windows.

**Reset caveat (critical if used):** due to a hardware bug, L2CPU harts can be taken out of
reset **only once**; restoring reset requires a full ASIC reset (`tt-smi -r`). Mitigation is a
seize-and-park mechanism in machine mode via RNMI. Harts must be released with the L2SYS clock
low, raised afterwards. Reset is a 0→1 transition of bits 4–7 in the ARC tile's `L2CPU_RESET`
at NoC address `0x80030014`.

Also note: L1D permits **only one outstanding miss at a time** — brutal for NoC-backed lines.

### Not present on Blackhole at all

No `BlackholeA0/DRAMTile/` or `BlackholeA0/ARCTile/` directories exist (Wormhole has both).
Blackhole DRAM facts are scattered across `BlackholeA0/README.md`, `NoC/README.md:15`, and
`NoC/Coordinates.md`.

### Baby RISC-V — bring-up (Phases 2–3)

All under `BlackholeA0/TensixTile/BabyRISCV/`. Nine of ten files are REAL Blackhole pages —
this is the best-documented area of the tree.

| Path | Status | Gives you |
|---|---|---|
| `README.md` | REAL | **The master bring-up page.** Pipeline, full MMIO map with per-core columns, load-latency table, L0 cache, local data RAM, `pc` snapshot registers |
| `InstructionSet.md` | REAL | Authoritative extension list + caveats — the input to the Rust target definition |
| `CSRs.md` | REAL | CSR table with conformance notes, `cfg0` chicken bits, `tt_cfg_qstatus`/`bstatus` |
| `MemoryOrdering.md` | REAL | **The single most important correctness document.** Store-queue/retire-queue model, instruction-pairing tables, "Enforcing stronger ordering" recipes |
| `PushTensixInstruction.md` | REAL | `.ttinsn` encoding, routing table, FIFO depths, debug-bus push |
| `AutoTTSync.md` | REAL | **Blackhole-only feature, no Wormhole equivalent.** Changes frontend semantics vs Wormhole |
| `ManualTTSync.md` | REAL | `CoprocessorDoneCheck` / `MOPExpanderDoneCheck` |
| `PCBufs.md` | REAL | B→T*i* FIFOs |
| `L1CacheTagSearchAccel.md` | REAL | Blackhole-only HW tag-search block, RISCV B only. **Phase 9 optimization, not bring-up** |
| `Mailboxes.md` | STUB-C | "identical" — read Wormhole verbatim |
| `InstructionCache.md` | **ABSENT** | → `WormholeB0/.../InstructionCache.md`. **Critical**, see below |
| `InstructionRAM.md` | **ABSENT** | Wormhole-only feature; Blackhole executes only from L1. Stale link |
| `DebugInterface.md` | **ABSENT** | → `WormholeB0/.../DebugInterface.md` (31 KB) — largest untranslated asset |

**Toolchain-determining facts** (from `InstructionSet.md` / `CSRs.md`):

- Implemented: RV32I, M, Zicsr, Zaamo, Zba, Zbb, `pack`/`brev8` (Zbkb subset), `grevi`
  (Bitmanip 0.94-draft), `.ttinsn`, partial Zicntr (`cycle`/`instret`, **no `time`**),
  partial F, partial Zfh, `mret` (**B and NC only**)
- **NOT implemented: `C`** — and the encoding space is *actively reused* by `.ttinsn`.
  Build with `-C target-feature=-c`; the assembler must never relax to compressed forms.
- **NOT implemented: Zalrsc** — no `lr.w`/`sc.w`. AMOs only, and **only against local L1**
  (not MMIO, not remote tiles). `.aq`/`.rl` ignored.
- **NOT implemented: `fdiv.s`, `fsqrt.s`, Zbs, Zbc, Zbkc, Zbkx, `packh`, `zip`, `unzip`**
- **NOT implemented: Zifencei** — see the code-loading note below
- RISCV T2 only: partial RVV 1.0, VLEN=128, ELEN=32; no integer div/rem, no FP div/sqrt/recip.
  Documented perf bugs: false destination-register dependency on most vector instructions;
  fractional LMUL behaves as if LMUL > 1.
- **`misa` reads `0x40201123` and lies** (claims RV32IMABFV; each of A/B/F/V is partial).
  Never probe it.
- **`mhartid` is always zero on every core.** Core identity must be established at build time
  or by address, never by CSR.
- Ethernet cores E0/E1 have no `V` — so one non-vector target covers B, NC, T0, T1, E0, E1,
  and T2 needs a second target if RVV is used.

**Code loading — there is no `fence.i`.** Zifencei is not implemented; executing `fence.i` is a
`nop` (`NonContractualBehavior`). Invalidation is via `RISCV_IC_INVALIDATE_InvalidateAll` in
Tensix backend config, with the Blackhole bit mask documented in
`BlackholeA0/TensixTile/TensixCoprocessor/BackendConfiguration.md:42` (bit 0=B, 1=T0, 2=T1,
3=T2, 4=NC). This register is **not accessible over the NoC and not accessible to RISCV NC** —
so NC cannot invalidate its own cache and depends on another core. Invalidation does **not**
flush the pipeline; up to ~20 further instructions may execute before stale contents clear.
**Blackhole I-cache capacities are documented nowhere in this repo** (Wormhole's are
2/2/0.5/2/0.5 KiB) — treat as unknown.

### Compute path (Phases 5–6)

**The largest gaps in the entire tree are here.** `BlackholeA0/.../Unpackers/` and
`.../Packers/` **do not exist at all**.

| Topic | Blackhole | Fall back to | Status |
|---|---|---|---|
| Backend index (the architectural map) | — | `WormholeB0/.../README.md` | ABSENT — use the Wormhole index as your map |
| `UNPACR` main instruction | `UNPACR_Regular.md` (3 ln) | `WormholeB0/.../UNPACR_Regular.md` (**688 ln — largest doc in repo**) | STUB-A — `TTArchitecture`-conditionalized, **authoritative** |
| Unpacker overview / format conversion / flush / context counter / all `UNPACR_NOP_*` | — | `WormholeB0/.../Unpackers/*` | ABSENT (two `UNPACR_NOP_*` pages are conditionalized) |
| `PACR` | `PACR.md` (74 ln) | — | **REAL**, but self-labelled *"basic"*; line 70 admits *"its interaction with `ReadIntfSel` is not yet documented"* |
| All 10 packer sub-pages, `PACR_SETREG`, `CLREXPHIST` | — | `WormholeB0/.../Packers/*` | ABSENT. Only `OutputAddressGenerator.md` is conditionalized |
| Matrix Unit overview + latency table | `MatrixUnit.md` (3 ln) | `WormholeB0/.../MatrixUnit.md` (265 ln) | STUB-B — perf figures are explicitly Wormhole-only |
| `MVMUL` | — | `WormholeB0/.../MVMUL.md` | ABSENT, but conditionalized |
| **`SrcA`/`SrcB` + fidelity phases** | — | `WormholeB0/.../SrcASrcB.md` | **ABSENT and not even linked from Blackhole — a silent gap.** Fidelity phases are documented in exactly one place, and it is not in the Blackhole tree |
| **`RWCs`** | — | `WormholeB0/.../RWCs.md` | ABSENT, yet **three** Blackhole pages link to it |
| `Dst` incl. RISC-V access + swizzle | `Dst.md` (**341 ln — largest BH page**) | — | **REAL and Blackhole-exclusive.** The swizzle content exists nowhere in Wormhole |
| `SETDVALID`/`CLEARDVALID`/`ZEROACC`/`ZEROSRC`/all `MOV*`/`ELW*`/`GMPOOL`/`GAPOOL`/`DOTPV`/`SHIFTX*`/`SETRWC`/`INCRWC` | — | `WormholeB0/.../*` | ABSENT (many conditionalized — check the list above) |
| Vector Unit | `VectorUnit.md` (175 ln) | — | **REAL**, independently written |
| `LReg` | `LReg.md` (3 ln) | `WormholeB0/.../LReg.md` | STUB-C |
| **`FloatBitPatterns`** | — | `WormholeB0/.../FloatBitPatterns.md` | **ABSENT — highest-value missing numerics doc.** All BFP/FP formats + conversions |

**SFPU instruction pages:** 48 in Blackhole vs 41 in Wormhole. Seven are Blackhole-only
(`SFPARECIP`, `SFPGT`, `SFPLE`, `SFPMUL24`, `SFPCAST_IntInt`, `SFPCAST_IntAbs`,
`SFPCAST_IntFloat`). ~20 are STUB-C ("identical"). ~21 are REAL Blackhole rewrites.

**Encoding:** Blackhole-specific encodings have `_BH`-suffixed SVGs under `Diagrams/Out/`
(`Bits32_SFPLOAD_BH.svg`, `Bits32_SFPSTORE_BH.svg`, `Bits32_PACR_BH.svg`,
`Bits32_STALLWAIT_BH.svg`, and ~10 more). **Diff `_BH` against non-`_BH` when writing the
encoder.** The field layouts are more readably extracted from `Diagrams/Src/Bits32.lua`, which
is the source these SVGs are generated from.

### Configuration and frontend

| Topic | Blackhole | Status |
|---|---|---|
| `BackendConfiguration.md` | 66 ln | **REAL** — `Config`/`ThreadConfig` model, special cases, debug regs |
| `ConfigurationUnit.md` | 35 ln | **REAL**, richer than Wormhole |
| `SETC16` / `RDCFG` / `RMWCIB` | REAL | Note: BH `RMWCIB` says `CfgAddress` where WH says `CfgIndex` — a real encoding difference |
| `CFGSHIFTMASK` / `STREAMWRCFG` / `STREAMWAIT` | REAL | **Blackhole-only instructions**, no Wormhole equivalent |
| `WRCFG` | STUB-C | identical |
| `STALLWAIT` | 213 ln REAL | BH-specific block + condition masks |
| `SEMWAIT` / `ATGETM` / `ATRELM` / `SyncUnit` | REAL | |
| `SEMINIT`/`SEMPOST`/`SEMGET` | STUB-C | identical |
| `MOPExpander` | STUB-B | `TTArchitecture`-conditionalized |
| `MOP` / `MOP_CFG` / `REPLAY` / `WaitGate` | **ABSENT** | `WaitGate` is a silent gap — seven BH pages reason about its semantics |
| Entire Scalar Unit (ThCon), Mover (`XMOV`), Miscellaneous Unit | **ABSENT** | All ThCon instructions are Wormhole-only pages |

**Critical convention** (stated in every config page): the `// Registers for THREAD` section of
`cfg_defines.h` indexes `ThreadConfig` (write with `SETC16`); all other sections index `Config`
(write with `WRCFG`). RISC-V can read both but can only write `Config`, and only with `sw`.

### Suggested reading order for the implementing engineer

1. `Glossary.md` — contractual terminology. Do not skip.
2. `WormholeB0/TensixTile/TensixCoprocessor/README.md` — the only backend index that exists.
3. Diagrams: `TensixFrontend_BH.svg`, `TensixBackend.svg`, `TensixBackendMisc.svg`,
   `UnpackerPipeline.svg`, `PackerPipeline.svg`.
4. Frontend: BH `PushTensixInstruction.md`, `AutoTTSync.md`, `ManualTTSync.md`, `CSRs.md`;
   then WH `MOPExpander.md`, `REPLAY.md`, `WaitGate.md`.
5. State: WH `SrcASrcB.md`, `RWCs.md`, `ADCs.md`; then **BH** `Dst.md` (BH-authoritative).
6. Numerics: WH `FloatBitPatterns.md`, then `Miscellaneous/FMA/README.md`.
7. Data path: WH `Unpackers/README.md` → WH `UNPACR_Regular.md` → WH `Packers/README.md` →
   **BH** `PACR.md`.
8. Compute: WH `MatrixUnit.md` + `MVMUL.md`; **BH** `VectorUnit.md` + SFP pages.
9. Config: BH `BackendConfiguration.md`, `ConfigurationUnit.md`, and the config instructions.

### Silicon-verification backlog

Areas where Blackhole has no documentation and the Wormhole page is not
`TTArchitecture`-conditionalized, so every fact needs validation against ttsim and silicon.
The list, with what is closed, is the checklist's "Silicon-verification backlog".

### External resources

| Resource | URL | Use |
|---|---|---|
| ttsim | https://github.com/tenstorrent/ttsim | **Golden reference simulator and primary development target.** Full-system (L1, NoC, PCIe), bit-exact by design, Blackhole single- and multi-chip. See `docs/libttsim_api.md` for the C API |
| tt-kmd | https://github.com/tenstorrent/tt-kmd | Kernel driver; ioctl definitions |
| tt-metal `cfg_defines.h` | https://github.com/tenstorrent/tt-metal/blob/81989dcdb8f9b340c932ae7a71a346f4f08703eb/tt_metal/hw/inc/blackhole/cfg_defines.h | `Field_ADDR32` / `_MASK` / `_SHAMT` for every config field — the input to `cargo xtask gen-cfg`. Cited directly by `BackendConfiguration.md` |
| TT-LLK | https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/tt-llk | C++ implementation of the layer being reimplemented — reference for *what works*, not for design |
| tt-bh-linux | https://github.com/tenstorrent/tt-bh-linux | L2CPU bring-up, incl. `clock.py` PLL manipulation |
| Burn | https://github.com/tracel-ai/burn | Target framework |

## Implementation phases

Each phase states its deliverable, the specification pages to work from, **two acceptance gates
(simulator then silicon)**, and the hazards known in advance. Durations assume one engineer at
sustained focus.

**The two-gate rule:** no phase is complete until it passes on both. The simulator gate is the
development loop — fast, deterministic, in CI on every commit. The silicon gate is the
correctness contract, and any divergence between the two is a finding worth writing up, because
it means either the simulator, the docs, or your understanding is wrong.

### Phase 0 — Simulator harness (~1–2 weeks)

**Deliverable:** `tt-ttsim` — Rust bindings over the `libttsim_*` entry points, wrapped in
a singleton-enforcing safe API; plus the `Transport` trait in `tt-device` with a `LibTtsim`
implementation.

**References:** ttsim `docs/libttsim_api.md`, ttsim `README.md`.

**Gate (simulator):** a Rust test calls `libttsim_init`, reads PCI config space, confirms the
expected Blackhole device ID, and advances time with `libttsim_clock` without crashing.

**Gate (silicon):** none — this phase is simulator infrastructure by definition.

**Why first:** it is the development loop for every subsequent phase. A week spent here removes
hardware contention, nondeterminism, and slow debug cycles from the following twelve months.

**Hazards:** the library is a process-wide, single-threaded, non-reentrant singleton with no
context handle. Enforce that in the Rust type system (a `OnceLock` guard handing out a single
`!Sync` token) rather than by convention — `cargo test`'s default thread-per-test model will
otherwise corrupt state in ways that look like simulator bugs.

### Phase 1 — Host can address the chip (~1 week)

**Deliverable:** `tt-kmd`. Open `/dev/tenstorrent/N`, mmap BAR0/BAR2/BAR4, wrap the
`TENSTORRENT_IOCTL_ALLOCATE_TLB` / `CONFIGURE_TLB` / `FREE_TLB` ioctls, expose a `TlbWindow`
RAII type bound to a NoC coordinate. Chip-indexed from the start (multi-chip is in scope).

**Gate (simulator):** write a pattern to a Tensix tile's L1 through one 2 MiB
window, read it back through a *different* window, bytes match. Repeat against a second tile to
validate coordinate handling, and against `bh_x2` to validate chip indexing.

**Gate (silicon):** the same test against a real p150, and across two cards if available.
**Closed** — 7/7 on both p150a cards. See `implementation-checklist.md`, and the operating
notes below it, which every later silicon gate depends on.

**Hazards:** multiple coordinate systems exist and are mutually incompatible. Establish the
newtype discipline here (see Cross-cutting), not later.

**What Phase 1 actually cost, and why it is worth reading before Phase 2's silicon gate.**
The estimate above was for the addressing work, which was accurate. What it missed is that
the first contact with real hardware is also the first test of every assumption the
simulator was silently satisfying. One of them — that a Blackhole has 140 Tensix tiles —
was measured on ttsim, recorded as a fact with a comment saying "re-derive it at the first
silicon gate", and then not re-derived. Both cards here have 120: two columns are fused off
for yield, and translation puts them at maximal X, exactly where the gate was writing.

A NoC access to a fused-off tile is not an error. Nothing answers, the read never
completes, the NoC hangs, and the chip is reset to recover it — which drops the PCIe link
and, with the card passed through to a VM, kills the host. Budget for the fact that
hardware bring-up failures on a passed-through accelerator are *host* failures, and that
the evidence dies with the machine unless you arrange otherwise beforehand.

The durable lesson is narrower than "test on hardware": a measured constant needs the
mechanism that would notice it changing, not a comment asking a future reader to check.
`grid::Tensix` now carries the chip's own answer and there is no 140 left in the tree to
iterate.

### Phase 2 — Rust executing on a baby RISC-V (~2–3 weeks)

**Deliverable:** `tt-firmware` skeleton + the loader half of `tt-device`. Custom target JSON,
linker scripts per core role, `_start`, stack in local data RAM, reset sequencing, I-cache
invalidation.

**References:** `BabyRISCV/README.md` (memory map, local data RAM, `pc` snapshots),
`BabyRISCV/InstructionSet.md` + `CSRs.md` (target definition), `TensixTile/SoftReset.md`,
`TensixCoprocessor/BackendConfiguration.md:42` (I-cache invalidate mask),
`WormholeB0/.../BabyRISCV/InstructionCache.md` (invalidation protocol).

**Gate (simulator):** heartbeat — core increments a counter in L1, host polls and observes it
climb, with time advanced explicitly via `libttsim_clock`. Cross-check the `pc` snapshot
registers (`0xFFB1_3138`+). Because the simulator raises `UndefinedBehavior` where silicon
proceeds silently, this gate also catches bad instruction encodings and illegal accesses that
hardware would simply execute.

**Gate (silicon):** same heartbeat on a real part. **This is the highest-value silicon gate in
the plan** — it is where simulator/hardware divergence in reset sequencing, I-cache
invalidation, and the local-data-RAM zeroing window will surface, and all three are things a
simulator may model loosely.

**Hazards — all five are Phase 2 blockers:**
- **No `fence.i`.** Zifencei is not implemented. After writing code to L1 you must write the
  5-bit mask to `RISCV_IC_INVALIDATE_InvalidateAll` via backend config. That register is **not
  NoC-accessible and not accessible to RISCV NC**, so NC depends on another core to invalidate
  its cache. Invalidation does not flush the pipeline.
- **RISCV B has no reset-PC override.** B's entry point is hardwired to L1 offset `0x00000`.
  This is a firm linker-script constraint. T0/T1/T2/NC default to `0x06000`/`0x0A000`/`0x0E000`/
  `0x12000` and *do* have override registers.
- **`ebreak` halts the core** and requires an external agent to resume — it cannot back Rust's
  panic/abort path. Decide the panic strategy now (suggested: write a status word to a known L1
  address, then spin).
- **Soft-reset registers have no atomic bit operations.** Every change is a read-modify-write
  and *software must provide its own mutual exclusion*. Day-one constraint for a multi-core
  loader.
- **Local data RAM self-zeroes for up to 2048 cycles** after reset release. The core's own
  accesses stall automatically; **NoC accesses do not**. If the loader pre-stages `.data` or
  stack over the NoC it must either set the relevant `RISCV_DEBUG_REG_DISABLE_RESET` bit first
  or wait out the window — otherwise initialized data is silently zeroed.

**Also validate here, while the system is trivial:** disassemble and confirm no compressed
instructions, no `lr.w`/`sc.w`, no `fdiv.s`/`fsqrt.s`. Establish core identity at build time —
`mhartid` is zero everywhere.

### Phase 3 — Instruction encoder + first Tensix round-trip (~3 weeks)

**Deliverable:** `cargo xtask gen-cfg` (parse tt-metal's `cfg_defines.h` into Rust consts —
generate, never transcribe; there are thousands of fields) and `gen-isa` (the instruction
table from `Bits32.lua`), then `tt-isa` encoders. Differential harness
against ttsim stood up **now**, not later.

**Gate (simulator):** an SFPU program (`SFPLOADI` ×2 → `SFPMUL` → `SFPNOP` → `SFPSTORE`) runs
and the host reads back the expected FP32 bit pattern (`0x40C0_0000` for 3.0 × 2.0). Then the
corpus test: encode a broad instruction corpus and assert bit-exact results.

**Gate (silicon):** the same corpus on hardware, diffed against the simulator run. **This
establishes the differential harness that every later phase depends on** — and because ttsim
targets bit-exactness, any mismatch is a real finding, not expected noise.

**Note:** `SFPLOADMACRO` is unsupported in ttsim's SFPU, so it must be excluded from the
simulator corpus and tested on silicon only. Track it in a `#[cfg]`-gated silicon-only suite
from the start rather than discovering the hole later.

**References:** `BabyRISCV/PushTensixInstruction.md`, `ManualTTSync.md`, `AutoTTSync.md`,
`TensixCoprocessor/BackendConfiguration.md`, BH `VectorUnit.md` + the SFP instruction pages,
`Diagrams/Src/Bits32.lua` (field layouts in readable source form).

**Why this is the real milestone:** it proves push, the frontend, scheduling hazards, Manual
TTSync, and the config path simultaneously.

**Stand up tracing here too.** `TensixTile/DebugTimestamper.md` (STUB-C, identical to Wormhole)
gives a tile-wide 64-bit cycle counter at `0xFFB1_21F0` plus a **hardware event-trace
primitive**: one store to `RISCV_DEBUG_REG_TIMESTAMP` appends `{29-bit token, 64-bit counter}`
to an L1 ring buffer. One store per event, no software timestamp read. This is strictly better
than per-core `mcycle` for correlating events across the five babies, and it will pay for
itself from Phase 5 onward. Use the documented retry loop when multiple agents read the
counter concurrently.

### Phase 4 — Layout before compute (~4 weeks)

**Deliverable:** `tt-layout`. Row-major strided ↔ tiled. Padding for shapes that are not a
multiple of the tile extent. FP32/BF16/FP16 conversion against the documented bit patterns,
not IEEE assumptions.

**Corrected during implementation.** Two things this paragraph originally said are wrong, and
the checklist records why. (1) The conversion reference is `FloatBitPatterns.md` and
`Packers/FormatConversion.md`, **not** the `Dst`/`Src` bit layouts — those describe the
register files and govern the `Dst` readback path instead. (2) A tile is not a 32×32 square:
`UNPACR_Regular.md:182` dimensions it from a `TileDescriptor`, and 32×32-of-four-16×16-faces
is one choice among many.

**Gate (simulator):** round-trip arbitrary tensors host→device→host unchanged, for every dtype
in scope, including awkward shapes (`[13, 47]`, `[1, 1, 1024]`). Property-test with a
shape/dtype generator — this is cheap in the simulator and expensive on hardware.

**Gate (silicon):** the same suite, reduced to a representative sample. **Narrower than
originally scoped:** `WormholeB0/NoC/Alignment.md:19,23` says the host-via-PCIe to L1 path has
no alignment restrictions at all, so the staging path cannot violate them. The C16 congruence
applies when an L1 address is the *source*, which is the unpacker and packer in Phase 6. The
silicon test asserts the documented "Any" across deliberately misaligned tile bases, since
`Alignment.md` is absent from the Blackhole tree and a failure would be a real finding.

**Sequencing rationale:** this is the layer every op sits on, it is independently testable, and
it forces the hardest architectural decision — how tiled layout and block-float formats meet
Burn's strided tensor model — while it is still cheap to change.

### Phase 5 — First real kernel: elementwise binary (~4 weeks)

Eltwise before matmul, deliberately: it exercises unpack → SFPU → pack with no `SrcA`/`SrcB`
bank handshake, no fidelity phases, no RWC choreography. Smallest thing that is a genuine
kernel.

**Gate (simulator):** differential vs `burn-flex` on random inputs, with tolerances derived
from the documented FMA divergence — not a guessed epsilon. Must include denormals (SFPU
flushes them) and NaN (Blackhole canonicalization differs from IEEE *and* from Wormhole). Since
ttsim targets bit-exactness *including NaN bit patterns*, assert exact equality against
`fma_model_bh` rather than a tolerance wherever the operation is a pure FMA.

**Gate (silicon):** same suite. A simulator/silicon mismatch here is a high-value bug report to
Tenstorrent — it means the golden reference and the hardware disagree.

### Phase 6 — Matmul (~8–12 weeks)

`UNPACR` into `SrcA`/`SrcB`, the double-bank `SETDVALID`/`CLEARDVALID` handshake, `MVMUL`,
fidelity phases, `PACR` out. Single 32×32 tile → blocked → multi-tile.

**References:** WH `UNPACR_Regular.md` (688 lines, conditionalized — authoritative),
WH `Unpackers/*`, WH `Packers/*`, BH `PACR.md`, WH `MatrixUnit.md`, WH `MVMUL.md`,
WH `SrcASrcB.md` (fidelity phases), WH `RWCs.md`, BH `Dst.md`, WH `FloatBitPatterns.md`.

**Hazard — the schedule risk of this project:** this is precisely where the Blackhole tree is
thinnest. `Unpackers/` and `Packers/` do not exist in the Blackhole tree at all. `SrcASrcB.md`
and `RWCs.md` are absent — and `SrcASrcB.md` is not even *linked* from Blackhole, so fidelity
phases are documented in exactly one place and it is the wrong tree. BH `PACR.md` is
self-labelled *"basic"* and explicitly admits its `ReadIntfSel` interaction is undocumented.

Work the **Silicon-verification backlog** above as an explicit checklist. Treat every
Wormhole-sourced fact as unverified until ttsim *and* silicon agree. Budget a standing
percentage of this phase for empirical discovery rather than implementation.

**The simulator is the main mitigation for this phase's risk.** Because ttsim is *intentionally
more restrictive than silicon* and raises `UndefinedBehavior` on misconfigured unpacker/packer
state, it converts the single largest documentation gap in the project into loud, immediate,
deterministic failures instead of silent wrong data. Do the unpacker/packer bring-up entirely in
the simulator; treat each `UndefinedBehavior` it raises as a specification question to answer
from the Wormhole docs before moving on.

**Gate (simulator):** differential vs `burn-flex` with tolerance derived from documented
numerics, across shapes, dtypes, fidelity phases, and accumulation depths.

**Gate (silicon):** same suite. Expect divergence here more than anywhere else — this is where
Wormhole-sourced assumptions about packers, `SrcASrcB`, and `RWCs` will be wrong.

### Phase 7 — Burn backend surface, training (~6–8 weeks)

**Deliverable:** `burn-tt` implementing `Backend` and its supertraits.

The original strategy -- implement a narrow core and let Burn's defaults compose the rest --
and the assumption that `burn-fusion` applies only to CubeCL backends were both wrong for the
pinned release; the corrected surface follows.

#### The Burn surface, as pinned (0.21.0, read from source 2026-09-30)

Verified against `burn-backend 0.21.0`, `burn-ir 0.21.0` and `burn-fusion 0.21.0`.

* **`Backend: BackendTypes + FloatTensorOps<Self> + BoolTensorOps<Self> +
  IntTensorOps<Self> + ModuleOps<Self> + ActivationOps<Self> + QTensorOps<Self> +
  TransactionOps<Self> + Clone + Default + Sized + Send + Sync + Debug + 'static`.**
  `BackendTypes` carries every associated type: `Device: DeviceOps`, the float,
  int, bool and quantized tensor primitives (each `TensorMetadata + 'static`), and
  `FloatElem`, `IntElem`, `BoolElem`. `Backend` itself requires only `name`, `seed`,
  `dtype_usage` and `device_count`; `sync`, `memory_cleanup`,
  `memory_persistent_allocations`, `ad_enabled` and the rest have defaults.
* **The "narrow core" does not exist.** The op traits have roughly two hundred
  methods with no default: `FloatTensorOps` 80 of 124, `IntTensorOps` 67 of 104,
  `BoolTensorOps` 28 of 39, `ModuleOps` 18 of 73, `QTensorOps` 13 of 80;
  `ActivationOps` and `TransactionOps` are all defaults. Burn does not compose the
  rest from a handful of primitives, so "implement matmul, add, relu, reshape and let
  the defaults do the rest" cannot typecheck. **Consequence for Phase 7:** `burn-tt`
  starts as a *delegating* backend -- its tensor primitive wraps a `burn-flex`
  tensor, every required op delegates to `burn-flex` on the host, and the ops worth
  running on Tensix (matmul first, via `tt_kernels::matmul::matmul`) are routed to
  the device. Correct from day one, and each op moved to the device is a change
  behind an unchanged interface, gated against the delegate it replaces.
* **Open question 3 is answered: `burn-fusion` composes with a hand-written
  backend.** `Fusion<B>` is generic over `B: FusionBackend`, which is `BackendIr`
  (conversions between the backend's primitives and a `Handle`) plus a
  `FusionRuntime`: an `Optimization` type, a handle and device type, and
  `fusers(device) -> Vec<Box<dyn OperationFuser<Optimization>>>`. A fuser is shown
  each `OperationIr` in the stream (`fuse`), reports whether it can take more
  (`status`), and produces an `Optimization` whose `execute` runs against the
  handles. Nothing in it is CubeCL: the CubeCL backends are one implementation.
  So Tensix's natural unit -- one unpack -> math -> pack chain, `matmul` + bias +
  activation in one pass through `Dst` -- is a `burn-tt` fuser, not a question of
  retargeting CubeCL. It is still Phase 9 work; Phase 7 needs only that the door
  is open, and it is.

#### As built (2026-09-30)

* **Delegation is generated, and defaults are left alone.** `cargo xtask
  gen-burn-delegate` forwards every op method *Flex implements* to Flex and leaves
  every default Flex does not override to its default, which then composes
  `burn-tt`'s own ops. The distinction is load-bearing: `ModuleOps::linear` is a
  default over `float_matmul`, `nn::Linear` calls it, and forwarding it kept every
  layer on the host. Type knowledge is three conversion traits in `burn-tt`, which
  rustc checks; the generator itself is syntactic and refuses what it cannot read.
* **Hardware lives on a server thread per device** (`burn_tt::attach`), created by a
  factory that runs on that thread. That is what lets the simulator, a `!Send`
  process-wide singleton, back a `Send + Sync` backend; the silicon engine is
  `burn_tt::kmd_engine` over `tt_kernels::session::Session`.
* **Precision.** HiFi4 from TF32 `Src` by default. Agreement with Flex is asserted
  bit-exact where the operands allow and otherwise against a bound derived from
  TF32 truncation and truncating accumulation (`step11_burn`), never an epsilon.
  Training is claimed to *work* (a stated loss factor, met by the host run of the
  same setup) and to be *deterministic and target-independent* (a pinned bit-exact
  loss curve that ttsim wrote and both cards reproduce); it is deliberately not
  claimed to stay within a bound of the host's trajectory.
* **At Phase 7's close only matmul ran on the device**, with tensors on the host
  between ops. Phase 9 made tensors resident and moved element-wise, ReLU and the
  bias sum to the device; Phase 10 covers the rest.

**Gate (simulator):** an MNIST MLP trains through `burn-autodiff` — loss descends, final weights
match the `burn-flex` backend within tolerance, optimizer step is correct. Use a reduced dataset;
the simulator is "slower than silicon but still fast enough" and a full training run is not the
point. Determinism means this doubles as a regression test.

**Gate (silicon):** full MNIST training run, matching the simulator's loss curve.

**This is the milestone.** Everything before it is infrastructure; everything after is reach
(multi-chip) or speed (performance).

### Phase 8 — Multi-chip (~6–10 weeks, **widest uncertainty band in the plan**)

Ethernet tiles, inter-chip transfer, and distribution of a model across cards.

**References:** `EthernetTile/EthernetTxRx.md` (26 KB, REAL — the transport reference; raw vs
TT-link mode, 18-byte header, sequence numbers, auto-retransmit),
`EthernetTile/EthernetRxClassifier.md` (37 KB, REAL), `EthernetTile/BabyRISCV/README.md`,
and `ethdump.c` as a working example of driving the RX path.

**Sequencing rationale:** deliberately last. Inter-chip transport built before single-chip
compute is correct is the most common way projects of this shape stall. Phase 1's chip-indexed
abstractions make this an extension rather than a rewrite.

**The simulator materially de-risks this phase.** ttsim ships `bh_x2` (P300), `bh_x4` (two P300
cards), and `bh_x32` (BH Galaxy) configurations "with significant numbers of Ethernet,
multidevice, and fabric tests passing." So inter-chip development happens before a second
physical card is ever needed, and the multi-chip topology is exercised in CI. This is why the
estimate above is lower than a hardware-first plan would justify. Two caveats: `bh_x32` is
x86_64-only, and ttsim's own README flags multichip testing as less mature than single-chip —
so silicon divergence risk is higher here than elsewhere.

**Four unknowns were open going in:** (1) Ethernet-tile reset sequencing, undocumented for
Blackhole; (2) RISCV E0's cooperative sharing with Tenstorrent firmware, whose contract exists
only in WH `CallingIntoCustomerCode.md`; (3) the undocumented Ethernet PIC; (4) the NoC
Overlay, absent from the Blackhole tree. A spike on `bh_x2` closed all four; see "Spike
outcome". Ethernet tiles also lack the Tensix conveniences: local data RAM is **not
NoC-accessible** and there are **no `pc` snapshots**.

**Gate (simulator):** tensor transfers between two chips under `bh_x2`; a model shards across
`bh_x4` and produces results matching the single-chip run.

**Gate (silicon):** the same across two physical cards.

#### Spike outcome (2026-09-30)

The spike ran on `bh_x2` and on the two p150a cards, which are cabled through
one QSFP-DD port. All four unknowns are closed, and none of them grew the
estimate. Details are in the checklist's Phase 8 section and divergence rows
56-60.

1. **Reset is in `ethdump.c`**, not undocumented. E1 is bit `0x1000` of
   `SOFT_RESET_0` and its reset PC is at `0xFFB1_4008`. Rust ran on E1 next to a
   live link, and the link stayed trained.
2. **E0 is never touched.** Customer code runs on E1. The E0 contract is not
   needed.
3. **No PIC.** Everything polls.
4. **No Overlay.** A TT-link L1 write puts bytes into the partner's L1 with
   resends done in hardware. The host can issue one through a TLB window with
   no firmware of ours. That gives a working inter-chip path before any data
   mover exists.

Both the grid and the link map come from the chips: ARC tag 35 for enabled
tiles, and base firmware's chip-info exchange for who is cabled to whom. ttsim
models neither, so its map is measured.

**Re-estimate for the rest: 3-5 weeks.**
- The E1 data mover with NoC hops: about 1-2 weeks. The NIU request initiators
  are the new surface.
- The link API and `bh_x4`: about 1 week.
- `N`-split sharding with the golden loss curve as the oracle: about 1-2 weeks.

That is down from 6-10 weeks, because the Overlay and E0 branches were not
taken.

#### As built (2026-09-30)

* **No firmware is needed for the first byte.** Ethernet TX-queue registers are
  NoC-visible, so the host can drive a TT-link L1 write through a TLB window.
  Every link was proven this way before any E1 code existed
  (`silicon_eth_link::host_driven_*`).
* **E1 runs a data mover** (`eth_e1`, contract in `tt_isa::eth::mover`):
  1. It pulls from a Tensix tile with the NIU request initiator. That is the
     first device-initiated NoC traffic in this workspace, built by
     `tt_isa::noc::niu::Command`, which cannot express the NoC hazards.
  2. It TT-link-writes the data and then a record to the partner.
  3. On seeing the record, the partner waits for its RX queue's outstanding
     writes to reach zero, then NoC-writes the data into its own Tensix tile.
     TT-link delivers a queue's packets in order, but nothing documented orders
     their L1 commits, so the record alone is not proof the data has landed.
     (The first version checksummed the whole buffer on both ends instead, which
     cost ~3.75 us/KiB.)
  4. The partner acknowledges back over the link, so the host waits on the
     sending chip only.
* **Sharding is along `N` with `K` whole** (`tt_kernels::shard::Fabric`), so the
  sharded product is bit-identical to one chip's. The reduced MNIST run therefore
  reproduces the Phase 7 golden bit for bit, on 2 and 4 simulated chips and on
  the two cabled cards.
  * The data plane is Ethernet only. Operands enter and results leave through
    chip 0, relayed through intermediate chips on the `bh_x4` ring.
  * The control plane is each chip's own PCIe: programs, starts, mover
    descriptors.
* **One Burn device.** `burn_tt::MeshEngine` puts a whole fabric behind the
  existing `Engine` trait. The plan's `attach_mesh` was unnecessary: `attach`
  already runs its factory on the server thread, which can own every chip.
* **What E1 buys today, measured** (`silicon_eth_bench`, throwaway; medians of
  20, card 0 to card 1, 128 KiB):

  | Path | Time | Throughput |
  |---|---|---|
  | Host writes chip 1's Tensix L1 over PCIe (TLB window) | 7.2 ms | 18 MB/s (reads: 5 MB/s) |
  | Mover, first version (whole-buffer checksum on both ends) | 669 us | 196 MB/s |
  | Host drives TT-link itself, 4 KiB commands | 168 us | 0.78 GB/s |
  | Host drives TT-link itself, one 128 KiB command | 14.4 us | 9.1 GB/s |
  | Mover, landing wait instead of checksum, staged -> landed | 11.5 us | 11.4 GB/s |
  | Mover, Tensix -> Tensix across cards | 14.0 us | 9.4 GB/s |

  So the Ethernet path is already *faster* than PCIe for moving tensor data,
  by about 500x. That reverses what this section said before measuring. The
  fixed ~9 us floor is the host polling the ack over PCIe. The host-driven
  rows are bound by the host's per-command register writes; the mover issues
  its 4 KiB commands locally, so command size no longer matters to it. The
  PCIe figure is the bigger story: the TLB-window path is uncached MMIO, and
  it is very probably most of Phase 7's ~435 ms per training step. The
  mover's other payoffs still stand: chips reachable only over Ethernet
  (Galaxy-style), device-resident pipelines where one chip's output feeds
  another's next op with no host round trip (Phase 9), and link bandwidth
  that does not compete with PCIe.
* **Posted writes race other agents** (found by the benchmark, not the link).
  `Device::write` returns while its writes are still in flight, so anything
  other than the host -- E1, the RX queue of a link -- can see stale bytes in
  L1 the host just wrote. A zero-fill racing an incoming transfer lost the
  transfer's tail 57 times in 480, and 0 in 480 once the fill was read back.
  `Mover::stage` now reads its last word back before returning, and the
  silicon gates fence what they stage. A general `Device` fence is open.
* **Not done, deliberately.**
  * Data-parallel training: an all-reduce reorders sums, so it needs a weaker
    claim than the golden.
  * Faster PCIe staging and double-buffered, multi-link Ethernet: Phase 9.

### Phase 9 — Performance (open-ended, **silicon-only**)

`MOP`/`REPLAY` expansion, three-thread pipelining (unpack on T0, math on T1, pack on T2),
double buffering, multi-tile distribution with NoC multicast. Measure against the theoretical
peak figures in the spec, not against a competitor.

**Direction, decided 2026-09-30: device residency first.** The release baseline is 224 ms per
MNIST step on the device against 0.5 ms for Flex on the host, and almost all of it is the host
path: every matmul re-sends its operands over PCIe and repeats per-session work (a reset, seven
firmware loads, program rebuilds). The answer is not a faster PCIe path but not using PCIe at all:
a p150a carries 32 GiB of GDDR6, so the dataset and weights are loaded once at startup and stay
there, and the metric is PCIe bytes per training step. The device-side items above follow once
per-op cost is the device's. Progress and measurements are in the checklist's Phase 9 section;
two findings shape it: the GDDR is reachable and gated on both cards (divergence measurement L),
and under this VM the host's MMIO is uncached whatever the guest maps (measurement M), so bulk
upload tops out at 226 MB/s -- ample for startup, and a reason to keep PCIe off the step.

**Where it stands (2026-10-01).** Full MNIST trains at **2.5 ms/step on eight Tensix tiles**
(5.1 on one; 224 at the start of the phase), bit for bit on the Phase 7 golden, with the
dataset, weights, activations and gradients resident in GDDR. Per steady-state step exactly six
tensors cross PCIe, 6224 B (`dL/dlogits`, the two biases and their gradients, and the logits),
asserted by the 9.5 gate; the loss stays on the host. Done: residency (9.3–9.5), many tiles
(9.6), one launch per op, op records expanded on the tile, and a resident program cache (9.7a–c),
which took the device writes per step from 193 KB to 35 KB. Next: overlap of data movement and
compute with the circular-buffer runtime (9.8), on-device tilizing for the 2.7 s preload (9.10),
and a device-resident, concurrent two-card mesh (9.11). Element-wise on the SFPU and the loss on
the device moved to Phase 10. The checklist's Phase 9 section has the per-slice history and
measurements.

**ttsim does not model cycle-accurate timing.** It remains useful here for *correctness* of the
more aggressive pipelined kernels — which is where correctness is hardest — but every
performance number must come from hardware. Keep using the simulator as the correctness gate
for each optimization, then measure on silicon.

### Phase 10 — Hardware coverage (open-ended, **ttsim + silicon**)

**Deliverable:** the rest of the Tensix tile, driven. Phases 0–9 proved the path to the card
with one compute unit doing real work: the Matrix Unit, for `MVMUL`. Element-wise ops and the
column sum run on the B core's scalar FP32 unit (`dm_b.rs`); the SFPU -- 32 lanes, 48
Blackhole instruction pages, the unit built for exactly this -- runs no tensor op at all, and
`ELW*`, `GMPOOL`/`GAPOOL`, `TRNSPSRCB`, block-float and integer formats are encoded and
unused. On the Burn side, 12 compute methods have a device path; no `ModuleOps`,
`IntTensorOps` or `BoolTensorOps` method does.

**Why now.** `burn-tt` is a general Burn backend, and under device residency (Phase 9) every
op without a device path is a download, a host op and an upload. Breadth is therefore a
performance item as much as a feature list, and the next model -- anything with a softmax, a
norm, a convolution -- falls back on its first unsupported op.

**Tracking.** The inventory, the work items (F, S, M, R, D), the milestones 10.0–10.6 and a
Burn op coverage table live in [`hardware-coverage.md`](hardware-coverage.md), which is the
progress record. Checklist items 9.9 and 9.12 moved there.

**Where it stands (2026-10-01).** Milestone 10.0 is done: a device profiler through the
timestamper (X3); padding as a typed property of every tensor (F0, which fixed the ragged
`ADD_ROW` -> sum bug); whole tiles through `Dst` (F1); the SFPU program builder with
typed registers, scoped conditional execution, automatic `SFPNOP`s and `REPLAY`-driven row
loops (F2, X1) and an interpreter that predicts any program bit for bit, held to ttsim and
both cards instruction by instruction (F5); the SFPU tile kernel and its dispatch (F3, F4);
and every element-wise op on the SFPU, bit-identical to the mover and to Flex, chosen
per op by a measured cost model (S1). MNIST: 3.8 ms/step on one tile (from 5.1), 2.4 on
eight. Milestone 10.1 is done too: reciprocal, division, `exp` and `log` on the SFPU;
`sum` and `max` over either dim; softmax and log-softmax on the device end to end; the
matmul's loops replayed (the MOP Expander gated, and measured no faster on a matmul that
is now backend-bound); the movers' queues, barriers and batching, and op-list traces that
refuse tt-metal's corruptions instead of suffering them; wedged tiles detected at open and
recovered in software. MNIST: 2.0 ms/step on one tile, 1.6 on four; inference 0.45 ms a
batch of 64. Cross-entropy moved to 10.5 with D4. Next: 10.2, activation and math
breadth.

**Ordering: SFPU foundation first.** The foundation -- whole-tile `UnpackToDst` and pack, an
SFPU program builder with typed `LReg`s and scoped conditional execution, one three-role SFPU
tile kernel, a `record::SFPU` dispatched through the existing mover, and ported functional
models as oracles -- is paid for once. After it, an SFPU op is a program builder and a gate,
with no firmware change. The first use moves today's B-core ops onto the SFPU with the MNIST
golden as the regression; the second is softmax/cross-entropy, which needs `exp`, reciprocal
and reductions and so exercises most of the foundation.

**References:** BH `VectorUnit.md`, the 48 BH `SFP*.md` pages (seven Blackhole-only:
`SFPARECIP`, `SFPGT`, `SFPLE`, `SFPMUL24`, three `SFPCAST` modes), WH `LReg.md`, BH `Dst.md`,
WH `GMPOOL.md`/`GAPOOL.md`/`ELWADD.md`/`TRNSPSRCB.md`, WH `Packers/*` and `Unpackers/*` for
format conversion, `Miscellaneous/FMA/fma.c`.

**Oracle policy for approximations.** Transcendentals on the SFPU are instruction sequences
(range reduction, polynomial or LUT, Newton steps), so the Tolerance policy applies twice:
the device result is predicted **bit for bit** by running the same sequence through the
ported per-instruction models, and its distance from the true function is held to a bound
**derived** next to the gate from the approximation's own error terms. Neither is an epsilon.

**Gate (simulator):** per op, bit-exact against the ported models (or within the derived
bound against the true function), and `burn-tt` against `burn-flex`; the op downloads
nothing.

**Gate (silicon):** the same on both cards, and a smoke-tier entry comparing `burn-tt` with
`burn-flex` on the card.

**Hazards:** the Tier 2 SFPU bugs (`SFPLUTFP32`'s destination, `SFPPOPC` on a full stack,
`SFPSTOCHRND`'s bias, `SFPCAST_IntAbs`, `SFPMAD` stalling) all land here; the tracker maps
each to the item that must handle it. ttsim declines `SFPLOADMACRO` (row 7), BF16
`UnpackToDst` (row 31) and `DOTPV`/`SHIFTXB` (row 50), so those parts are silicon-only.

## Cross-cutting concerns

Set these up in Phases 0–3. Retrofitting them later is expensive.

**Differential testing is the backbone.** Two oracles: **ttsim** (the official golden reference,
bit-exact by design, full-system) and **`burn-flex`** at the tensor level (`burn-ndarray` is
deprecated). Stand the harness up
in Phase 3, not Phase 6 — it is what makes every later phase debuggable.

**CI runs on ttsim; silicon is a scheduled gate.** Hardware in CI is a reliability tax, and
because ttsim is a full-system simulator the *entire* stack runs in CI, not just ISA-level
tests. Run the simulator suite on every commit (deterministic, so no flakes); run the silicon
suite nightly and as a merge gate to main.

**Keep a simulator/silicon divergence log.** Every mismatch is a finding: either the simulator,
the documentation, or your understanding is wrong, and all three are worth knowing about. Report
simulator bugs upstream — ttsim is actively developed and the project benefits from the fix.
Maintain a `#[cfg]`-gated silicon-only test suite for things ttsim cannot cover
(`SFPLOADMACRO`, all timing, any feature the README lists as unsupported).

**One newtype per coordinate space.** NoC0 raw, NoC1 raw, translated, and the raw-coordinate
escape hatches in TLB `strided`/exclude fields are all `(u8, u8)` and all different. Type-level
separation pays for itself within a month. Add a `ChipId` from day one — multi-chip is in scope.

**Encode hazards in the API, not in comments.** This is the concrete answer to "why Rust and
not C++", and if it isn't done the answer is "no reason". Minimum set:
- `SFPNOP`-after-`SFPMUL`/`SFPMAD` scheduling requirement
- `Dst` exclusivity across the three Tensix threads
- `SrcA`/`SrcB` bank ownership handshake
- `NOC_CMD_WR_INLINE` may not target L1; `NOC_CMD_L1_ACC_AT_EN` must be false
- Manual TTSync's load-adjacency constraint (must be one inline-asm block, never left to LLVM)

**Generate, never transcribe.** `cfg_defines.h` has thousands of fields. `cargo xtask gen-cfg` parses it;
hand-copying is a guaranteed bug source. Pin the same commit SHA the docs cite.

**Pin the spec, re-sync quarterly.** The docs are actively maintained (357 commits since
2025-05-09; the most recent fill in exactly the Blackhole `UNPACR`/`PACR` gaps this plan depends
on). Record the pinned hash in the repo. Gaps may close mid-project — check before doing
silicon-discovery work that the docs may have obviated.

---

### L1 planning and circular buffers (design, 2026-10-01)

L1 was a set of fixed constants that overlapped by convention -- the matmul's
staging and output areas, element-wise and column-sum slots reusing them, two
hand-picked semaphores -- and kernel fusion cannot work that way: two kernels
that each assume they own `0x2_0000` collide the moment they are one kernel.
The design, of which step 1 is built:

1. **One L1 map** (`tt_isa::l1`): every fixed region (images, the mover's
   list, mailboxes, program slots, trace, and the program cache to come) in
   one table, checked at compile time to be ordered and disjoint. What is left
   is the data arena, `DATA`.
2. **Kernels declare, never address** (`tt_kernels::l1::Requirements`): each
   buffer's size, alignment (not necessarily a power of two: a tile slot is
   4160 bytes), kind -- scratch, or a circular buffer of `pages` pages with one
   producer and one consumer endpoint (mover, unpack, math, pack) -- and the
   stages of the kernel it is live in; and its semaphores, likewise.
3. **A planner places by liveness**: first fit in declaration order, sharing
   bytes (and semaphores, of the tile's eight) between things never live in a
   common stage. Deterministic, so programs built from a plan memoise by
   shape. An independent `check` verifies any plan; a property test plans
   500 random kernel pairs and their fusions through it.
4. **Handles that cannot be misused.** A `Buf` or `Sem` carries its
   requirements' identity in private fields, so it cannot be forged, and a
   plan refuses another kernel's handle. A lifetime brand would catch that at
   compile time but could not be stored with a memoised program or survive a
   merge, so the check is at run time, one comparison. Device-time lifetime is
   a property of the kernel's dataflow, not of a host scope, so RAII freeing
   is not the model; planning is.
5. **Fusion is a merge of requirements**: `Requirements::fuse` concatenates
   two kernels' stages and unifies a circular buffer of the first (its
   producer kept) with one of the second (its consumer kept), dropping the two
   legs through GDDR, and the union plans as one. A fused kernel that does not
   fit is a planning error at build time, never a silent overlap.
6. **The runtime protocol is a page FIFO** (decided): tt-metal's
   push/wait/pop over page counters in L1, polled through fences (the L0 cache
   is not coherent), as mover list entries and role-side waits. It arrives with
   9.8's double buffering, which is its first user.

Order: map, planner and the GDDR kernels ported (done); 9.7c's program cache
in its region of the map; the page-FIFO runtime with 9.8; fusion on the merge.

### Hazards as data (design item, from the silicon campaign)

Every ordering requirement found on silicon -- a consumer must not start before a
producer's result is visible -- has so far been met with a hand-chosen `STALLWAIT`,
and once with a fixed run of `nop`s that was both wrong and expensive. A kernel
compiler needs these as *facts* it can schedule around, not as barriers:

- **Per-instruction effects**, generated alongside the encoding table: which unit
  executes it (unpacker 0/1, packer, Matrix Unit, SFPU, Configuration, Scalar,
  Mover, Sync), and which state it reads and writes (`Src` A/B bank and rows,
  `Dst` rows, `Config` words, `ThreadConfig` entries, GPRs, ADCs, RWCs).
- **Visibility rules** between them, each with its source: `WRCFG` lands 2 cycles
  later (`ConfigurationUnit.md`); a `Dst` write is unreadable for 4 cycles and only
  Matrix Unit and `PACR` readers are stalled for it (`Dst.md:101`); an unpack into
  `Src` is consumable once the bank is handed over (`FlipSrc`), which the Matrix
  Unit waits for by itself; `SFPMAD` results and the cases automatic stalling misses
  (`Instruction::stalls_automatically_after_mad`, already data).
- **A wait planner** that, for each producer->consumer pair, emits the minimal
  `STALLWAIT` -- the producer's condition, the *consumer's* block bits
  (`backend::Before`) -- so everything independent keeps flowing, and places
  independent instructions into fixed-latency windows instead of `nop`s.

Correctness of the planner is checked on ttsim, which models the dependencies but
not the timing; the windows' lengths and the throughput gained are silicon-only
(Phase 9).

## Hardware bug and caveat register

Every item is documented in the spec. **These are not speculative.** Tier 1 will bite in
Phases 2–4.

### Tier 1 — early phases

| # | Hazard | Source |
|--:|---|---|
| 1 | **Interrupt handlers cannot write CSRs.** They may read but not write. Kills the usual save/restore-`fcsr` ISR pattern. Interrupts exist only on B and NC. | `TensixTile/PIC.md:69-70` |
| 2 | **Manual TTSync load adjacency.** After *starting* a load from `CoprocessorDoneCheck`/`MOPExpanderDoneCheck`, the core must not start another load against `PC_BUF_BASE..+0xFFFF` (includes PCBufs **and all 8 Tensix semaphores**) or `TENSIX_MAILBOX0_BASE..+0x3FFF` until the first finishes. Violation → wrong results **or a hung core**. Fix: consuming ALU instruction immediately after, or ≥7 intervening instructions. | `BabyRISCV/ManualTTSync.md:18` |
| 3 | **Auto TTSync does not cover push-then-load.** The "store to push Tensix instruction, then load from config/GPRs/TDMA-RISC" pairing is *not* automatic; requires the push to drain the store queue first. Insert a `fence`. The other three pairings are covered. | `BabyRISCV/AutoTTSync.md:46-47` |
| 4 | **`pmacfg0`/`pmacfg1` range selection is broken** — always selects the entire address space. Only usable semantics: "low bit set ⇒ every load/store behaves as if preceded by `fence`". Useful as a global debug switch, useless as a targeted PMA. | `BabyRISCV/CSRs.md:86` |
| 5 | **`intp_restore_pc` is only a copy** — writing it does not change the `mret` target. Combined with #1, you cannot redirect an interrupt return. | `BabyRISCV/CSRs.md:41` |
| 6 | **Operand-forwarding starvation hang.** If two unretired instructions both write the same register, forwarding happens from *neither*, and a specific sequence **hangs forever**. Replacing an `addi` with a `nop` flips a working example into a permanent hang. | `BabyRISCV/MemoryOrdering.md:149` |
| 7 | **L0 data cache is not coherent.** Nothing written by the NoC, unpackers, packers, or another baby invalidates it. Polling loops over L1 **must** contain a `fence` or use an atomic. Hits also carry a ~0.8% random full-flush chance unless `cfg0.DisLowCachePeriodicFlush` is set. | `BabyRISCV/MemoryOrdering.md:59,61` |
| 8 | **Store-then-load with non-overlapping byte ranges has no ordering guarantee in either direction.** | `BabyRISCV/MemoryOrdering.md:54` |
| 9 | **T0/T1/T2 storing to `INSTRN1_BUF_BASE`/`INSTRN2_BUF_BASE` hangs the RISCV.** A stray pointer here is an unrecoverable lockup, not a fault. | `BabyRISCV/PushTensixInstruction.md:8-9` |
| 10 | **Configuration Unit cross-thread starvation.** Ordering is enforced across all threads regardless of issuer; heavy `WRCFG` use by one thread starves RISC-V read/write requests from others. | `TensixCoprocessor/ConfigurationUnit.md:33` |

### Tier 2 — once the Tensix units are driven

- `SFPMAD` — automatic stalling logic fails to detect a handful of cases (`SFPMAD.md:72,75-76`)
- `SFPLUTFP32` — writes to `LReg[LReg[7] & 15]` instead of `LReg[VD]`; workaround is to stuff
  `VD`'s low 4 bits into `LReg[7]` (`SFPLUTFP32.md:15`)
- `SFPPOPC` — complex modes must not be used with a full conditional-execution stack
- `SFPSTOCHRND` — stochastic rounding is biased toward increasing magnitude; the new-in-Blackhole
  round-toward-zero mode sometimes rounds *away* from zero. The functional models in the docs
  faithfully reproduce the buggy behaviour — match them, don't "fix" them
- `SFPCAST_IntAbs` — this encoding was meant to do something else; the bug makes it compute
  absolute value. Use `SFPABS`
- `CLEARDVALID` — reset is unsafe and drops `SrcA`/`SrcB` banks
  ([tt-metal#22383](https://github.com/tenstorrent/tt-metal/issues/22383))
- `STREAMWRCFG` — later Configuration Unit instructions from the same thread can reorder ahead
  of a pending one
- `SETC16` — scheduling restrictions on `CFG_STATE_ID_StateID` exist specifically to work around
  Auto TTSync bugs

### Non-bug hazards that shape the design

- **Soft-reset registers have no atomic bit operations.** RMW plus caller-provided mutual
  exclusion (`TensixTile/SoftReset.md:3-4`)
- **Invalid/unsupported RISC-V instructions are `UndefinedBehavior`** — per `Glossary.md` that
  includes possible physical damage. There is no illegal-instruction trap to rely on
- **Auto TTSync does not help with** MOP Expander config, Tensix `Dst`, Tensix semaphores,
  `sfpu_cc`, Tensix↔Tensix dependencies, or anything done by B or NC (`AutoTTSync.md:56-65`)
- **`DISABLE_RISC_BP_Disable_bmp_clear_*` must stay false** — the cores are not designed to run
  with a partially disabled branch predictor
- **`pc` snapshots are speculative** (taken as instructions leave the frontend). Fine for
  sampling profilers, useless as a precise fault PC
- **L2CPU harts can be taken out of reset only once** per ASIC power cycle
  (`L2CPUTile/README.md:30`)

---

## Verification

**Per-phase gates.** Both must pass. Simulator gates run in CI on every commit; silicon gates
run nightly and block merges to main.

| Phase | Simulator gate | Silicon gate |
|--:|---|---|
| 0 | `libttsim_init` succeeds, PCI config reports the expected device ID, `libttsim_clock` advances | — (infrastructure) |
| 1 | Pattern round-trips through two different TLB windows, on two tiles and under `bh_x2` | Same on a real p150, and across two cards if available |
| 2 | Heartbeat climbs; `pc` snapshot confirms the core; no `c.*`/`lr.w`/`sc.w`/`fdiv.s`/`fsqrt.s` in the disassembly | Same on hardware — **highest-value silicon gate**; reset, I-cache invalidation and the RAM-zeroing window surface here |
| 3 | SFPU program returns `0x40C0_0000`; instruction corpus is bit-exact | Corpus diffed against the simulator run; `SFPLOADMACRO` tested here only |
| 4 | Property-tested tensor round-trip, all dtypes, awkward shapes | Representative sample + every alignment-boundary shape |
| 5 | Eltwise bit-exact vs `fma_model_bh`; denormals and NaN included | Same suite; any mismatch is an upstream bug report |
| 6 | Matmul matches `burn-flex` across shapes, dtypes, fidelity phases, accumulation depths | Same; expect the most divergence here |
| 7 | MNIST MLP trains on a reduced dataset; loss descends | Full training run matching the simulator's loss curve |
| 8 | Two-chip transfer under `bh_x2`; model shards across `bh_x4` | Same across two physical cards |
| 9 | Correctness of pipelined kernels only | **All performance numbers** — ttsim is not cycle-accurate |
| 10 | Each op bit-exact against the ported instruction models, or within a derived bound of the true function; `burn-tt` against `burn-flex`; no download | Same on both cards; `burn-tt` against `burn-flex` in the smoke tier |

**Tolerance policy.** Never use a guessed epsilon. Derive expected values from the bit-exact
models in `Miscellaneous/FMA/fma.c` (`fma_model_bh`) and the documented per-instruction
"IEEE754 conformance / divergence" sections. The SFPU flushes denormals, canonicalizes NaN
differently from IEEE *and* from Wormhole, and the Matrix Unit diverges further. A test that
passes with a loose epsilon is not evidence of correctness here.

**Standing regression suite.** Every silicon-discovery finding from the verification backlog
becomes a test case. That suite is the project's real asset — it is what lets you trust
Wormhole-sourced behaviour.

---

## Open questions

Tracked, with what has been answered, in the checklist's "Open questions". Still open as of
2026-10-01: Blackhole I-cache capacities (1), debug-interface parity (2), PCIe DMA (5), how
faithfully ttsim models I-cache invalidation (6, in part), and whether ttsim models the
documented hardware bugs (7).
