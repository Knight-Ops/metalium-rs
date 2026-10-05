# Tenstorrent RISC-V guide: review against the repo

**Source.** <https://docs.tenstorrent.com/tt-vscode-toolkit/riscv-guide/>, "Exploring Tenstorrent as
a RISC-V Assembly Programming Platform", document version 1.0, last updated 2025-12-16, target
"Wormhole (n150/n300), Blackhole". Read in full on 2026-10-02. The guide is one page with
Parts 1–12 and Appendices A–B. It has no sub-pages of its own. The sidebar sibling pages were
also read for hardware claims: CS Fundamentals modules 1–6 and 8, Exploring TT-Metalium, Tensix
Grid Playground, Hardware Detection, ttsim Twenty-and-Ten, ttsim QEMU Bridge, TT-Lang intro
and FAQ. Where a fact comes from a sibling page, the page is named.

**Verdict.** The guide is an introductory text. It describes **Wormhole** under tt-metal
conventions. Its Appendix A memory map is titled "(Wormhole)". Almost every number in it is
either Wormhole-only or wrong for both chips. It adds **no Blackhole hardware fact** that the
vendored ISA docs do not already state more precisely. Its value to us is:

1. The tt-metal software conventions it shows: the B/NC reader/writer split, NoC per core, the
   CB API and barrier batching.
2. The list of popular misconceptions below, so nobody imports them.

Legend: **C** = confirms the repo, **A** = adds to it, **X** = contradicts it. "Spec" means
`vendor/tt-isa-documentation/`. "tt-metal" means tt-metal `main` source fetched on 2026-10-02.
tt-metal is cited only to settle a point the guide raises, and is not part of the guide.

---

## 1. Contradictions

### 1a. Guide vs repo (the repo is right in every case, backed by the Spec)

| # | Guide says | Repo / Spec says | Note |
|---|---|---|---|
| X1 | "All five processors implement the **RV32IM** instruction set" (Part 1, "RISC-V ISA"); toolchain `-march=rv32im` (Part 2) | BH: RV32IM + Zicsr + Zaamo + Zba + Zbb + `pack`/`brev8`/`grevi` + partial Zicntr/F/Zfh. B and NC also have `mret`. T2 has partial V (`BabyRISCV/InstructionSet.md:3-35`). The repo compiles rv32im + F (`tt-firmware-images/build.rs:150`). | The guide's ISA is the lowest common denominator, which is tt-metal's compile target. It is not the hardware. |
| X2 | "**No caches** — Explicit DMA operations for memory access" (Part 1); "no cache" (Exploring TT-Metalium) | Each baby core has an L0 **instruction** cache and a 64 B L0 **data** cache (4 × 16 B lines). The data cache is not coherent and is flushed by `fence` or atomics. It also flushes at random about 0.8% of hits unless `DisLowCachePeriodicFlush` is set (`BabyRISCV/README.md:39,138-142`). The repo relies on this: `publish()` fences every poll (`tt-metal-concepts-review.md:619`). | **Dangerous if believed.** A bare poll of an L1 word can spin on a stale L0 line. |
| X3 | "**No interrupts**: Polling-based synchronization only" (Part 12) | BH Tensix has a PIC: 32 software IRQs and 4 hardware IRQs, delivered to **B and NC only**, with a handler `pc` per IRQ (`TensixTile/PIC.md:3-6`). NIU interrupts on transaction-ID completion are new in BH, with NoC 0 → `HW_INT_PC[1]` and NoC 1 → `HW_INT_PC[2]` (`NoC/Interrupts.md:1-12`). | True of tt-metal's firmware, false of the hardware. |
| X4 | Local data RAM at **different** addresses per core: B `0xFFB0_0000`, NC `0xFFB0_1000`, TRISCs `0xFFB0_2000..0x37FF` (Part 1; Appendix A) | Every core sees **its own** RAM at the same `MEM_LOCAL_BASE = 0xFFB0_0000`. This holds on WH (`WormholeB0/.../BabyRISCV/README.md:82-83`) and on BH (`BlackholeA0/.../BabyRISCV/README.md:103-104`). On BH only, the RAMs are also reachable over the NoC (slow path) at `0xFFB1_4000..0xFFB1_DFFF`, in hardware order B, NC, T0, T1, T2 (`README.md:109-116`). Repo: `tensix.rs:196, 121-125, 312-318`. | The guide's addresses do not exist. A firmware image linked to `0xFFB0_1000` for NC would fault or hit unmapped space. |
| X5 | Local RAM sizes: B and NC 4 KB, each TRISC 2 KB (Part 1) | BH: B and NC **8 KiB**, T0/T1/T2 **4 KiB** (`BabyRISCV/README.md:146,157`; `tensix.rs:100-104`). The guide's figures are WH's. | Wormhole only. |
| X6 | NCRISC has a 16 KB IRAM at `0xFFC0_0000` for kernels. The linker places NCRISC `.text` at `MEM_NCRISC_KERNEL_BASE /* IRAM! */` (Part 1; Part 2; Appendix A) | **Blackhole has no IRAM.** "Instructions can only be fetched and executed from L1; instructions cannot be executed out of any other memory regions" (`BlackholeA0/.../BabyRISCV/README.md:39`). The BH memory map has no `0xFFC0_0000` row (`:101-136`). tt-metal's BH `dev_mem_map.h` agrees: "Blackhole Architecture - No IRAM constraints", with `MEM_NCRISC_KERNEL_SIZE = MEM_MAX_KERNEL_SIZE` in L1 (1497 KiB). The WH header has `NCRISC_HAS_IRAM 1` instead. The guide itself labels the IRAM block "(Wormhole architecture feature)" in Part 1, but Part 2's linker script and Appendix A carry no such label. | Settles open question 3 in `tt-metal-concepts-review.md:673`. See §3.2. |
| X7 | L1 = "1464 KB (1.5 MB)" at `0x0000_0000..0x0016_FFFF` (Part 1; Appendix A) | BH: 1536 KiB, `0x0000_0000..0x0017_FFFF` (`BabyRISCV/README.md:102`; `tensix.rs:193`). WH: 1464 KiB, `..0x0016_DFFF`. The guide's end address `0x16_FFFF` (1472 KiB) matches **neither** chip, nor its own 1464 KB. | Internally inconsistent. |
| X8 | NoC address = `Y<<48 | X<<40 | local[39:0]` (Part 4). The CS Fundamentals module 4 page says `Y<<48 | X<<32 | local[31:0]` instead. | BH NIU: a 64-bit local address in `TARG_ADDR_LO/MID`, with coordinates in `TARG_ADDR_HI` as `x | y<<6` (`noc.rs:240-247`, tests at `noc.rs:723-726`). BH widened addresses from 36 to 64 bits (`NoC/README.md:39`). | The guide's two pages disagree with each other. Both are tt-metal *software* packings (`get_noc_addr`) and not a hardware format. Don't copy either one. |
| X9 | Clock "~1 GHz" (Part 8) | BH baby cores and NoC run at **1.35 GHz** (`BabyRISCV/README.md:3`; `NoC/README.md:39`). WH runs at 1 GHz. | Wormhole only. |
| X10 | "880 RISC-V cores (5 per Tensix × 176)" (Part 8, FAQ) | Neither chip has 176 Tensix tiles. BH p150 has 140 (TT-Lang intro: "Blackhole chip: 140 Tensix"; ttsim `blackhole_140_arch.yaml`) and the repo's grid comes from ARC. WH n150 has 72 (Hardware Detection page) or 80 (TT-Lang intro). | Marketing arithmetic. |
| X11 | DRAM "1 GB per chip" (Part 1). Module 2 says "12 GB (1 GB × 12 channels)". | BH: each group of three DRAM tiles shares 4 GiB of GDDR6, and each DRAM tile has its own RISC-V core and L1 (`NoC/README.md:13`). | Wormhole-ish at best. |
| X12 | `crt0`: "Tensix coordinates: Passed via `s1` register (**set by hardware**)" (Part 2) | On leaving soft reset all GPRs are zeroed (`TensixTile/SoftReset.md:116`). `mhartid` reads 0 on every core and `misa` misreports, so identity cannot be probed (`tensix.rs:11-14`). | Whatever `s1` holds comes from software. The guide is wrong here. |
| X13 | Exploring TT-Metalium: "NoC 0: Reads, NoC 1: Writes" as a property of the tile | Both NoCs carry every request type. NoC 0 flows right and down, NoC 1 flows left and up, and each is a torus (`NoC/RoutingPaths.md`; `NoC/README.md:5-9`). The read/write split is a tt-metal kernel convention (§4). | It is a software default, not hardware. |
| X14 | CS Fundamentals module 4: a "2D mesh" with "5-port routers" | Each NoC is a **2D torus**, with wraparound on both axes (`NoC/README.md:5`). The Playground page does say "2D torus". | The guide's pages disagree with each other. |
| X15 | Module 4: "latency 1 cycle/hop", "~5 cycles for 1 hop", "bandwidth 32 bytes/cycle" | BH: router-to-router is **9 cycles** per hop and NIU↔router about 5. One 512-bit (64 B) flit per cycle per axis (`NoC/README.md:62-66`). | The guide's figures are illustrative. Use the Spec. |
| X16 | Hardware Detection: "p150 - Dual chip" | The p150 is a single Blackhole chip (the repo's own target, `../plans/master-roadmap.md`). The guide also calls p300c "single chip" while saying QuietBox 2 holds "two dual-ASIC p300c boards". | Board taxonomy only. Hardware is unaffected. |

### 1b. Found while cross-checking (not from the guide, but worth fixing in the repo)

| # | Repo says | Evidence | Action |
|---|---|---|---|
| R1 | "B's **2 KiB** instruction cache" (`firmware-performance.md:135`; `crates/tt-firmware/src/lib.rs:577`) | The 2 KiB / 2 / ½ / 2 / ½ KiB table is **Wormhole's** (`WormholeB0/.../InstructionCache.md:5-8`). BH `BabyRISCV/README.md:39` links an `InstructionCache.md` that exists neither in `vendor/` nor upstream (404 on `main`, 2026-10-02). **The BH icache sizes are undocumented.** | **Measured (2026-10-03, `probe_icache`): ~4 KiB on every baby core (B, NC, T0, T1, T2)**, a miss ~5.5 cycles per 32 bytes; the repo's "2 KiB" statements are corrected. |
| R2 | `SoftReset.md:114` (BH) mentions "core-local instruction RAM" | That is a WH leftover link. The BH memory map has no IRAM. | None. Don't read it as evidence that BH has IRAM. |

---

## 2. Hardware facts by topic

The guide adds nothing to Blackhole hardware. The rows below are the guide's claims, each
marked against the Spec and the repo, and labelled WH or BH.

### 2.1 Cores

| Fact | Chip | Guide | Mark | Repo / Spec |
|---|---|---|---|---|
| Five cores per Tensix: B, NC, T0 (unpack), T1 (math), T2 (pack) | both | Part 1 | C | `tensix.rs:16-28` |
| B = "RISCV_0 / Data Movement 0", NC = "RISCV_1 / Data Movement 1" | tt-metal naming | Part 1 | C | concepts-review G6(a), `:303-304` |
| Single-issue in-order, no hardware threads | both | Part 1 | C | BH: in-order frontend and EX1, with reordering after EX1 resolved by the retire unit (`BabyRISCV/README.md:22`) |
| No FPU in the RISC-V cores | WH only | Part 1, Part 12 | X | BH has partial F/Zfh (RNE only, DAZ/FTZ, no `fdiv`/`fsqrt`) (`InstructionSet.md:18-26`) |
| No `malloc`, no libc, no OS, no virtual memory, physical addresses | both | Part 12 | C | `no_std` firmware |
| DRAM is not directly addressable from the baby cores, only through the NoC | both | Part 1 | C | the BH memory map has no DRAM window |
| The NC data-RAM stack holds at least 256 B | tt-metal | Part 12 ("256 bytes minimum") | A (software) | tt-metal BH `MEM_{B,NC}RISC_STACK_MIN_SIZE 256`, TRISC0/1 192, TRISC2 256. The stack and globals share local RAM. |
| T0..T2 kernels compile from `trisc.cc` with the role chosen at build time | tt-metal | Part 1 | A (software) | our role runners are one image per core |
| Register state is exposed "via mailbox" for a debugger to read the PC | tt-metal | Part 7 | A | BH has a hardware `pc` snapshot at `0xFFB1_3138..0x314B`, in order B, NC, T0, T1, T2, readable over the NoC. It is speculative: the instruction may never execute (`BabyRISCV/README.md:159-173`; `tensix.rs:93`) |

**What NC can and cannot reach on BH** (`BabyRISCV/README.md:101-136`). The guide is silent
on this. It is listed here because it bounds the NC mover design.

| Region | B | T0–T2 | NC |
|---|---|---|---|
| L1, own local RAM, debug/soft-reset regs, PIC, `pc` snapshot, the other cores' local RAM (slow path) | yes | yes | yes |
| NoC 0 and NoC 1 NIU registers, NoC overlay | yes | yes | **yes** |
| TDMA-RISC (`0xFFB1_1000`) | yes | yes | **no** (WH allowed it) |
| Tensix push, GPRs, PCBufs, Manual TTSync, Tensix semaphores, mailboxes, `TENSIX_CFG` | yes / partial | yes | **no** |
| `RISCV_IC_INVALIDATE` (in `TENSIX_CFG`) | yes | yes | **no**, so NC cannot invalidate its own icache (`tensix.rs:25-26`, C) |
| `mret` / PIC interrupt delivery | yes | no | **yes** |

### 2.2 NoC

| Fact | Chip | Guide | Mark | Repo / Spec |
|---|---|---|---|---|
| Two NoCs, each Tensix has an interface to both | both | Part 1 diagram | C | `NIU_BASE` NoC0 `0xFFB2_0000`, NoC1 `0xFFB3_0000` (`noc.rs:213-215`) |
| The NoCs are physically separate and flow in opposite directions. "No contention for opposite-direction traffic" | both | Playground | C | NoC0 right/down, NoC1 left/up (`RoutingPaths.md`; `RUST_IMPL_PLAN.md:261`) |
| "Row-first routing — horizontal first, then vertical" | NoC 0 only | Playground | C, partial | On NoC 1, unicast is Y-then-X (`RUST_IMPL_PLAN.md:261`). The guide omits it. |
| Multicast: one send, hardware replicates, Tensix receivers only | both | Part 9; module 4 | C | rectangle broadcast. Only Tensix tiles receive, and other tiles opt out (`RoutingPaths.md`). Swap start and end between NoCs (`RUST_IMPL_PLAN.md:320`). Untested here (open question 4). |
| NoC transfers are asynchronous; completion needs a barrier | both | Part 4 | C | `NIU_MST_REQS_OUTSTANDING_ID(i)` per 4-bit transaction ID, 8-bit (`Counters.md:23`) |
| "Batch reads, single barrier" | both | Part 11 | C | already the mover's design (cap per transaction ID, `tt-firmware/src/lib.rs:572-580`) |
| The NoC also reaches PCIe, ARC, DRAM and Ethernet | both | Part 4 | C | `noc.rs` `TileType` |
| A bad NoC address "will fail" | both | Part 12 | **understated** | A harvested tile or a bad coordinate **hangs the NoC** (`tt-metal-concepts-review.md:617`) |

The guide says **nothing** about request initiators or command buffers, transaction IDs,
virtual channels, posted versus non-posted writes, ordering, alignment, atomics, or
`NOC_CMD_WR_INLINE`. The repo's coverage of these (`RUST_IMPL_PLAN.md:268-312`) stands
unchallenged.

### 2.3 Memory and L1 layout

| Fact | Chip | Guide | Mark | Repo |
|---|---|---|---|---|
| L1 is shared by the five cores and reachable from other tiles over the NoC | both | Part 1 | C | |
| L1 holds kernel code and CBs | both | Part 1 | C | `l1.rs` regions |
| tt-metal mailbox at L1 offset 16, 12 768 B (`dev_msgs_t`, runtime args, sync flags, NCRISC halt SP at +4) | WH tt-metal | Part 1 | A (software) | tt-metal BH: ARC FW scratch at 16, **inline-write staging at 32..95** (16 B per NoC per mover), mailbox at 96, 13 440 B. Ours: `MAILBOX_BASE = 0x10_0000` (`mailbox.rs:12`), images from 0 to `0x14000` (`l1.rs:35-41`). We are not bound by theirs. |
| Firmware and kernel space: B 6 KB fw + 48 KB kernel; NC 2 KB fw + 16 KB IRAM; TRISC 1.5 KB fw + 24 KB kernel | WH tt-metal | Part 1 | X for BH | BH tt-metal: firmware 6 KiB + 2.5 KiB for B and 2.5 KiB for each other core, kernels up to 1497 KiB from L1. Ours: fixed slots between reset PCs (`tensix.rs:50-57`): B 24 KiB, T0..T2 16 KiB each, NC 8 KiB. |

### 2.4 Reset, boot and caches

| Fact | Chip | Guide | Mark | Repo / Spec |
|---|---|---|---|---|
| `crt0` sets `gp` (with `norelax`) and `sp` from `__stack_top`, then calls `main` with no libc init | both | Part 2 | C | our `tt-firmware` entry does the same |
| "No explicit reset mechanism" described | — | — | gap | reset PCs: B `0x0`, T0 `0x6000`, T1 `0xA000`, T2 `0xE000`, NC `0x12000`. B's is hardwired, and the others can be overridden by register (`SoftReset.md:116-123`; `tensix.rs:50-57`). Soft-reset bits: B 11, T0–T2 12–14, NC 18 (`tensix.rs:34-41`). ttsim does not model the NC override (`ttsim-divergence.md` row 43). |
| Caches | — | "none" | X | see X2. `fence.i` is not implemented and is rejected by our gate (`ttsim-divergence.md` row 24). The icache is invalidated on leaving reset or by `RISCV_IC_INVALIDATE` (mask bit 4 = NC), written by **another** core. |

### 2.5 Ethernet (E0/E1)

The guide is silent apart from "Ethernet cores (for multi-chip)". Per the Spec, for
completeness:
- Two cores per Ethernet tile at 1.35 GHz, with 512 KiB L1 and 8 KiB local RAM each.
- E0 is shared with Tenstorrent link-training code.
- Local RAM and `pc` snapshots are **not** NoC-visible on Ethernet tiles.
- The PIC differs from the Tensix one.

Source: `EthernetTile/BabyRISCV/README.md:3-14`. Nothing contradicts the repo.

---

## 3. Original NC bring-up design (implemented; 2026-10-02 review)

### 3.1 Facts that bound the design

| Question | Answer (BH) | Source |
|---|---|---|
| Does NC have IRAM? | **No.** It fetches only from L1, through its L0 icache. | `BabyRISCV/README.md:39`; tt-metal BH `dev_mem_map.h` ("No IRAM constraints") |
| NC icache size | **Undocumented on BH.** WH was 512 B, a quarter of B's. If BH kept that ratio, NC's hot loop must be far smaller than B's ~3 KB (`firmware-performance.md:135`). | WH `InstructionCache.md:5-8`; R1 |
| Can NC flush its own icache? | **No.** `TENSIX_CFG` is unmapped for NC, so B or the host must write `RISCV_IC_INVALIDATE` bit 4, or reset NC. | `BabyRISCV/README.md:135`; `BackendConfiguration.md:42` |
| Can NC reach NoC 0 and NoC 1? | **Yes, both.** NC can also take NIU interrupts and has `mret`. | `BabyRISCV/README.md:117-118`; `PIC.md:5` |
| Can NC push Tensix, use PCBufs, Tensix semaphores or mailboxes? | **No.** It must sync with the TRISCs through L1 words, such as CB counters (G5). | `BabyRISCV/README.md:120-135` |
| NC local RAM | 8 KiB at `0xFFB0_0000` (fast) and at `0xFFB1_6000` (slow, NoC-visible). It zeroes for 2048 cycles after reset. Don't touch it over the NoC during that window. | `README.md:110,146-152` |
| NC image slot | `0x12000`, fixed on ttsim, because the override register is unmodelled | `ttsim-divergence.md` row 43; `l1.rs:35-41` |
| Prefetcher bound for NC | `Config.NOC_RISC_END_PC_PC` and `RISC_PREFETCH_CTRL_Enable_NocRisc` | `BackendConfiguration.md:48-49` |

### 3.2 Answer to `tt-metal-concepts-review.md:673` (open question 3)

**No.** Blackhole removed NC's IRAM. NC fetches from L1 like the other four cores, so none of
Wormhole's IRAM rules apply:
- no mover copy into IRAM,
- no branch-predictor-pollution sequence,
- no 16 KiB kernel ceiling.

Two constraints remain:
- the small, undocumented L0 icache (R1);
- NC's inability to invalidate that icache itself, so any code reload needs B or the host, or
  a reset.

The guide's "16 KB IRAM for kernels" applies to Wormhole only.

---

## 4. Tenstorrent software conventions worth adopting

These are tt-metal's choices, which bind nobody. Each is marked with how it would map onto us.

| Convention | Seen in | Adopt? | How it maps here |
|---|---|---|---|
| **Reader on B over NoC 0, writer on NC over NoC 1** (`NOC::RISCV_0_default`; BRISC "reading", NCRISC "writing"; "Fetch from DRAM via NoC 0 … Store to DRAM via NoC 1") | Guide Parts 1 and 3; Exploring TT-Metalium | **Yes**, already the G6 plan | B keeps gathers and `KERNEL`; NC takes `SCATTER` and CB-pop on NoC 1 (`concepts-review.md:312-315`). Opposite flow directions mean reads and writes do not share links. |
| **Per-core command-buffer split.** In the default mode each mover owns one NoC and uses all four initiators on it: 0 = large writes, 1 = reads, 2 = small or register writes, 3 = atomics. In "dynamic NoC" mode, where both movers may use both NoCs, B gets initiators 0/1 and NC gets 2/3 on each NIU. | tt-metal `blackhole/noc_nonblocking_api.h:53-72` (not the guide) | **Yes** | Encode the owner in the type, for example `Initiator<Core, NocId>`. Then an NC command cannot be issued on a B-owned initiator of a shared NIU. Initiators are per-NIU state with a "don't touch while `CMD_CTRL` reads 1" rule (`noc.rs:250-252`), so two cores sharing an initiator would race. Start with NC on NoC 1 alone, using all four. If NC ever issues on NoC 0, apply the 2/3 split. |
| **Transaction IDs per role**, with a barrier on one ID instead of a global barrier | Guide Part 11 (batching); tt-metal | Yes, already done | The counters are per NIU, so NC on NoC 1 has its own 16 IDs and needs no coordination with B. |
| **Inline writes emulated as a 16 B L1 staging slot per NoC per mover**, then a normal write | tt-metal BH `dev_mem_map.h` (`MEM_L1_INLINE_BASE`) | Yes, for any L1 target | The repo already refuses `WR_INLINE` to L1 (`noc.rs:419-422, 461`). Give NC its own staging slot for NoC 1 rather than sharing B's. |
| **CB protocol**: `cb_reserve_back` / `get_write_ptr` / `cb_push_back` and `cb_wait_front` / `get_read_ptr` / `cb_pop_front` | Guide Part 11 | Already G5 | NC cannot use Tensix semaphores, so CB counters between the packer (T2) and NC must be L1 words. Every NC poll must fence (X2). |
| **The NC stack minimum (256 B) checked at link time** | tt-metal (`MEM_*_STACK_MIN_SIZE`) | Yes, cheap | Add a link-time assert to the NC image that stack plus globals fit in 8 KiB with at least 256 B left. |
| Device profiler and DPRINT through an L1 ring | Guide Part 7 | Already have it | Our trace buffer and timestamper. |

---

## 5. Questions from the guide review

Status reconciled 2026-10-04. Items 1 and 3 were answered by local probes;
the remaining questions still need evidence. The original source review is
preserved above.

1. **Answered: BH instruction caches.** `probe_icache` measured about
   4 KiB on each baby core, including B and NC (2026-10-03). See
   `firmware-performance.md`, "Instruction caches". This does not establish
   an X280 cache size.
2. **NoC atomics and inline writes into BH L1 may hang.** tt-metal's BH `dev_mem_map.h` says:
   "issuing inline writes **and atomics** requires all 4 memory ports to accept the transaction
   at the same time … the transaction will hang". The Spec states only the inline-write
   regression (`NoC/README.md:39`) and says nothing about atomics. This bears directly on
   G7's "NoC atomics and L1 semaphores" (`concepts-review.md:363`). Treat NoC atomics to L1
   as an unverified encoding: gate them on ttsim and in an isolated silicon gate before
   any use.
3. **Answered: NC on ttsim and silicon.** NC firmware and NoC1 traffic are
   gated. The current executor always uses NC as the GDDR writer and captures
   both mover streams (`feature-streaming-dataflow-ownership.md`).
4. NoC ordering and alignment on BH: `Ordering.md` and `Alignment.md` are absent for BH
   (`RUST_IMPL_PLAN.md:265-266`). The guide offers nothing.
5. Whether tt-metal's dynamic-NoC split (B 0/1, NC 2/3) is a hardware need or a software
   choice. The header carries "TODO … currently using wormhole b0 copy". Nothing in the Spec
   requires it. It is only a way to avoid sharing initiators.
