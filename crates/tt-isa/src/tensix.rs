//! Tensix tile memory map and baby RISC-V control.
//!
//! Addresses are as seen from inside the tile — which is also what the host writes
//! into a TLB window's `local_offset`, so the same constants serve both sides.
//!
//! Spec: `BlackholeA0/TensixTile/BabyRISCV/README.md:101-136` for the map,
//! `BlackholeA0/TensixTile/SoftReset.md` for reset control.

/// Which of the five baby RISC-V cores in a Tensix tile.
///
/// `mhartid` reads zero on every one of them, and `misa` reads `0x40201123` and
/// lies (it claims RV32IMABFV when A, B, F and V are each only partial). Core
/// identity therefore cannot be discovered at run time — it is fixed at build time
/// or inferred from an address, never probed.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Core {
    /// RISCV B. Can push Tensix instructions to any of the three threads.
    B,
    /// RISCV T0, feeding Tensix thread 0.
    T0,
    /// RISCV T1.
    T1,
    /// RISCV T2. The only core with RVV.
    T2,
    /// RISCV NC. Cannot push Tensix instructions at all, and cannot invalidate its
    /// own instruction cache.
    NC,
}

impl Core {
    pub const ALL: [Core; 5] = [Core::B, Core::T0, Core::T1, Core::T2, Core::NC];

    /// Bit of [`SOFT_RESET_0`] holding this core in reset (`SoftReset.md:10-36`).
    pub const fn soft_reset_bit(self) -> u32 {
        match self {
            Core::B => 11,
            Core::T0 => 12,
            Core::T1 => 13,
            Core::T2 => 14,
            Core::NC => 18,
        }
    }

    pub const fn soft_reset_mask(self) -> u32 {
        1 << self.soft_reset_bit()
    }

    /// Address this core begins executing from after reset is released
    /// (`SoftReset.md:118-123`).
    pub const fn default_reset_pc(self) -> u32 {
        match self {
            Core::B => 0x0_0000,
            Core::T0 => 0x0_6000,
            Core::T1 => 0x0_A000,
            Core::T2 => 0x0_E000,
            Core::NC => 0x1_2000,
        }
    }

    /// Register holding this core's reset PC, if it has one.
    ///
    /// **RISCV B has none**: its entry point is hardwired to L1 offset `0`. That is
    /// a linker-script constraint, not a preference.
    pub const fn reset_pc_register(self) -> Option<u64> {
        match self {
            Core::B => None,
            Core::T0 => Some(TRISC0_RESET_PC),
            Core::T1 => Some(TRISC1_RESET_PC),
            Core::T2 => Some(TRISC2_RESET_PC),
            Core::NC => Some(NCRISC_RESET_PC),
        }
    }

    /// Register and bit that enable this core's reset-PC override.
    pub const fn reset_pc_override(self) -> Option<(u64, u32)> {
        match self {
            Core::B => None,
            Core::T0 => Some((TRISC_RESET_PC_OVERRIDE, 1 << 0)),
            Core::T1 => Some((TRISC_RESET_PC_OVERRIDE, 1 << 1)),
            Core::T2 => Some((TRISC_RESET_PC_OVERRIDE, 1 << 2)),
            Core::NC => Some((NCRISC_RESET_PC_OVERRIDE, 1 << 0)),
        }
    }

    /// Address of this core's `pc` snapshot (`BabyRISCV/README.md:163-169`).
    ///
    /// The snapshot is taken as instructions leave the frontend, so it is
    /// speculative: it may name an instruction that never executes. Good for
    /// liveness and sampling, useless as a precise fault PC.
    pub const fn pc_snapshot(self) -> u64 {
        match self {
            Core::B => 0xFFB1_3138,
            Core::NC => 0xFFB1_313C,
            Core::T0 => 0xFFB1_3140,
            Core::T1 => 0xFFB1_3144,
            Core::T2 => 0xFFB1_3148,
        }
    }

    /// Size of this core's local data RAM (`BabyRISCV/README.md:146`).
    pub const fn local_data_ram_size(self) -> u32 {
        match self {
            Core::B | Core::NC => 8 * 1024,
            Core::T0 | Core::T1 | Core::T2 => 4 * 1024,
        }
    }

    /// Base of this core's local data RAM in the NoC-visible *slow access path*
    /// (`BabyRISCV/README.md:109-116`).
    ///
    /// Each local data RAM appears twice. At `MEM_LOCAL_BASE` it is reachable
    /// only by its own core, with two-cycle loads and no contention. Here it is
    /// reachable from any core **and over the NoC**, at eight cycles or worse —
    /// new in Blackhole, and explicitly intended so that "the host [can] more
    /// easily initialize the local data RAM, and [debuggers can] more easily
    /// inspect it" (`README.md:157`).
    ///
    /// For RISC-V memory ordering the two mappings are *separate memory
    /// regions* (`README.md:154`), so a core must reach its own RAM through
    /// `MEM_LOCAL_BASE` and use this address only for another core's.
    pub const fn local_data_ram_noc_address(self) -> u64 {
        // A uniform 0x2000 stride from LOCAL_DATA_RAM_NOC_BASE, in hardware core
        // order -- the same order as `pc_snapshot`, and deliberately derived from
        // the same index so the two cannot drift apart.
        LOCAL_DATA_RAM_NOC_BASE + (self.hardware_index() as u64) * LOCAL_DATA_RAM_NOC_STRIDE
    }

    /// Address space the slow access path reserves for this core: always 8 KiB.
    ///
    /// Kept distinct from [`local_data_ram_size`](Self::local_data_ram_size)
    /// because for T0/T1/T2 they differ: the memory map gives each T-core *two*
    /// 4 KiB rows with the same label while the RAM is only 4 KiB, and the
    /// specification does not say what the upper half is. Anything staging into
    /// this aperture must bound itself by the size, not by the window.
    pub const fn local_data_ram_noc_window(self) -> u64 {
        LOCAL_DATA_RAM_NOC_STRIDE
    }

    /// Position of this core in the hardware ordering B, NC, T0, T1, T2.
    ///
    /// Private because it is not meaningful on its own — it exists so that the
    /// `pc` snapshot block and the local-RAM aperture, which share this order,
    /// are generated from one place. Note that neither the soft-reset bits nor
    /// the `DISABLE_RESET` bits follow it.
    const fn hardware_index(self) -> u32 {
        match self {
            Core::B => 0,
            Core::NC => 1,
            Core::T0 => 2,
            Core::T1 => 3,
            Core::T2 => 4,
        }
    }

    /// Bits of [`DISABLE_RESET`] for this core: the local-data-RAM bit, then the
    /// debug `DR` register bit (`SoftReset.md:42-54`).
    ///
    /// With the local-data-RAM bit set, leaving soft reset does not start the
    /// zeroing described at [`LOCAL_RAM_ZEROING_CYCLES`] at all — the window is
    /// abolished rather than shortened. **A fourth core ordering**: T0 = 0,
    /// T1 = 2, T2 = 4, B = 6, NC = 8, matching neither the soft-reset bits, nor
    /// the `pc` snapshot order, nor the I-cache invalidate mask.
    pub const fn disable_reset_bits(self) -> (u32, u32) {
        match self {
            Core::T0 => (0, 1),
            Core::T1 => (2, 3),
            Core::T2 => (4, 5),
            Core::B => (6, 7),
            Core::NC => (8, 9),
        }
    }

    /// Can this core push Tensix instructions? (`PushTensixInstruction.md:3`)
    pub const fn can_push_tensix(self) -> bool {
        !matches!(self, Core::NC)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Core::B => "B",
            Core::T0 => "T0",
            Core::T1 => "T1",
            Core::T2 => "T2",
            Core::NC => "NC",
        }
    }
}

/// L1 scratchpad RAM, shared by all five cores and reachable over the NoC.
/// Instructions can *only* be fetched from here (`BabyRISCV/README.md:39`).
pub const L1_BASE: u64 = 0x0000_0000;
/// 1536 KiB per Tensix tile (`BabyRISCV/README.md:102`).
pub const L1_SIZE: u64 = 1536 * 1024;

/// A core's own local data RAM, fast path. **Not reachable over the NoC.**
pub const LOCAL_DATA_RAM_BASE: u64 = 0xFFB0_0000;

/// Tile control / debug / status registers. This block *is* NoC-visible
/// (`BabyRISCV/README.md:106`), which is what lets the host start a core at all.
pub const DEBUG_REGS_BASE: u64 = 0xFFB1_2000;

/// `RISCV_DEBUG_REG_CFGREG_RD_CNTL` — the address to read from backend
/// configuration.
///
/// `BackendConfiguration.md:66`: `Config` and `ThreadConfig` are **not** mapped
/// into the NoC's address space, but writing `x` here makes hardware perform
/// `CFGREG_RDDATA = ((uint32_t*)TENSIX_CFG_BASE)[x & 0x7ff]` a few cycles later.
/// Since this block *is* NoC-visible, that is the host's only way to read backend
/// configuration back — a read-only path, and the only cross-check on a
/// configuration write that does not go through the firmware that performed it.
///
/// **`UNVERIFIED`.** The Blackhole `TileControlDebugStatus.md` names the register
/// but gives no address table; `0xFFB1_2058` is from
/// `WormholeB0/EthernetTile/TileControlDebugStatus.md:25`, where an Ethernet tile
/// repurposes the same block offset for general use. **Confirmed against ttsim** by
/// `crates/tt-tests/tests/probe_cfgreg.rs`, which reads back a word it has just
/// written; re-derive on silicon.
pub const CFGREG_RD_CNTL: u64 = 0xFFB1_2058;

/// `RISCV_DEBUG_REG_CFGREG_RDDATA` — the word [`CFGREG_RD_CNTL`] asked for.
///
/// **`MEASURED`, not documented.** No page in either tree gives this address, so it
/// was found by sweeping the debug block against ttsim
/// (`probe_cfgreg::search_for_the_cfgreg_debug_registers`) and is pinned by
/// `probe_cfgreg::the_host_can_read_backend_configuration_back`.
///
/// It is **eight words** after `CFGREG_RD_CNTL`, not one: `0xFFB1_205C` is the
/// debug daisychain's `DBG_RD_DATA`, which answers a read with
/// `debug_bus_rd_data: DBG_BUS_CTRL must be enabled` rather than with configuration.
/// Guessing the adjacent word would have looked like a working address returning a
/// wrong answer. Re-derive on silicon.
pub const CFGREG_RDDATA: u64 = 0xFFB1_2078;

/// `RISCV_DEBUG_REG_SOFT_RESET_0` (`SoftReset.md:6-8`).
///
/// There are no atomic bit operations on this register: every change is a
/// read-modify-write, and software must provide its own mutual exclusion
/// (`SoftReset.md:3-4`).
pub const SOFT_RESET_0: u64 = 0xFFB1_21B0;

/// `RISCV_DEBUG_REG_DISABLE_RESET` (`SoftReset.md:42-54`).
///
/// Bits 0/2/4 suppress the local-data-RAM zeroing for T0/T1/T2, 6/7 for B, 8/9 for
/// NC. Only needed when staging data into local RAM over the NoC, which the
/// baseline does not do.
pub const DISABLE_RESET: u64 = 0xFFB1_2224;

pub const TRISC0_RESET_PC: u64 = 0xFFB1_2228;
pub const TRISC1_RESET_PC: u64 = 0xFFB1_222C;
pub const TRISC2_RESET_PC: u64 = 0xFFB1_2230;
/// Bits 0/1/2 enable the override for T0/T1/T2.
pub const TRISC_RESET_PC_OVERRIDE: u64 = 0xFFB1_2234;
pub const NCRISC_RESET_PC: u64 = 0xFFB1_2238;
/// Bit 0 enables the override.
pub const NCRISC_RESET_PC_OVERRIDE: u64 = 0xFFB1_223C;

/// `SemaphoreAccess[i]` at `PC_BUF_BASE + 4 * PC_BUF_SEMAPHORE_BASE`, in the
/// address space of RISC-V T0, T1 and T2 only (`SyncUnit.md`, "RISCV access to
/// semaphores"; `BabyRISCV/README.md:128`): a load reads semaphore `i`'s value,
/// a store of an even value posts it (`SEMPOST`), of an odd one gets it
/// (`SEMGET`) -- atomically, and without queueing behind the Tensix thread.
pub const SEMAPHORE_ACCESS: u64 = 0xFFE8_0020;

/// The tile's debug timestamper (`TensixTile/DebugTimestamper.md`, identical to
/// Wormhole's): a 64-bit cycle counter, and an event stream that appends
/// `{token, counter}` records to a buffer in L1, one store per event.
///
/// Reachable from every baby RISC-V and over the NoC.
pub mod timestamper {
    /// `RISCV_DEBUG_REG_WALL_CLOCK_L`: the counter's low half. Reading it also
    /// latches the high half for [`WALL_CLOCK_H`].
    pub const WALL_CLOCK_L: u64 = 0xFFB1_21F0;
    /// The counter's live high half, for the multi-reader retry loop.
    pub const WALL_CLOCK_L_PLUS_4: u64 = 0xFFB1_21F4;
    /// The high half latched by the last [`WALL_CLOCK_L`] read.
    pub const WALL_CLOCK_H: u64 = 0xFFB1_21F8;
    /// `RISCV_DEBUG_REG_TIMESTAMP`: a store appends an event; the low three bits
    /// choose its size.
    pub const TIMESTAMP: u64 = 0xFFB1_21FC;
    /// Buffer enables (bits 0, 1) and the sticky stream reset (bit 31).
    pub const CNTL: u64 = 0xFFB1_2200;
    /// `RISCV_DEBUG_REG_TIMESTAMP_STATUS`: sticky full/overflow bits and buffer
    /// 0's write position (bits 14..32, in 16-byte units).
    pub const STATUS: u64 = 0xFFB1_2204;
    /// Buffer 0's first and last 16-byte unit.
    pub const BUF0_START: u64 = 0xFFB1_2208;
    pub const BUF0_END: u64 = 0xFFB1_220C;

    /// Low three bits of a 128-bit event, `{header, counter_lo, counter_hi, 0}`:
    /// the size that needs no write-accumulation buffer, so events from
    /// several cores cannot interleave within one another.
    pub const APPEND_128B: u32 = 0;
    /// Bytes one 128-bit event occupies in L1.
    pub const EVENT_BYTES: u64 = 16;

    /// The header for an event carrying `token`: the 29 bits above the size.
    pub const fn event_128(token: u32) -> u32 {
        (token << 3) | APPEND_128B
    }

    /// Buffer 0's write position from a [`STATUS`] read, in events.
    pub const fn buf0_position(status: u32) -> u32 {
        status >> 14
    }

    /// [`STATUS`] bit 4: buffer 0 overflowed; events were dropped.
    pub const fn buf0_overflowed(status: u32) -> bool {
        status & (1 << 4) != 0
    }
}

/// First byte of the NoC-visible local-data-RAM aperture
/// (`BabyRISCV/README.md:109`).
pub const LOCAL_DATA_RAM_NOC_BASE: u64 = 0xFFB1_4000;
/// One past its last byte (`README.md:116`).
pub const LOCAL_DATA_RAM_NOC_END: u64 = 0xFFB1_E000;
/// Address space each core is given inside it.
pub const LOCAL_DATA_RAM_NOC_STRIDE: u64 = 0x2000;

/// How long the local data RAM spends zeroing itself after reset is released
/// (`BabyRISCV/README.md:152`).
///
/// The core's own accesses stall automatically for the duration. **NoC accesses do
/// not**: staging `.data` or a stack over the NoC inside this window is silently
/// discarded. The baseline sidesteps it by keeping everything in L1 and letting the
/// core set up its own stack.
pub const LOCAL_RAM_ZEROING_CYCLES: u32 = 2048;

/// Pushing a Tensix instruction: a `sw` of the instruction word to this address
/// (`PushTensixInstruction.md:5-9`).
///
/// From T0/T1/T2 this feeds that core's own Tensix thread, before the MOP expander.
/// From B it feeds thread 0 *after* the expander.
pub const INSTRN_BUF_BASE: u64 = 0xFFE4_0000;
/// B → Tensix thread 1. **Storing here from T0/T1/T2 hangs the RISCV**
/// unrecoverably (`PushTensixInstruction.md:8`).
pub const INSTRN1_BUF_BASE: u64 = 0xFFE5_0000;
/// B → Tensix thread 2. Same hazard as [`INSTRN1_BUF_BASE`].
pub const INSTRN2_BUF_BASE: u64 = 0xFFE6_0000;

/// Which Tensix thread an instruction is being pushed to.
///
/// A marker type rather than a number because the legal `(core, thread)` pairs are
/// a sparse table, not a range — see [`PushesTo`].
pub trait TensixThread {
    /// 0, 1 or 2. Needed where a thread index is data rather than a choice, such as
    /// the mailbox word the host and the image cross-check against each other.
    const INDEX: u32;

    /// `RISC_DEST_ACCESS_CTRL_SEC[INDEX].fmt` — the shape of this thread's RISC-V
    /// view of `Dst` (`Dst.md`, "RISCV access to Dst").
    ///
    /// Carried here because the field is per-thread and picking it by matching on a
    /// runtime index is how the wrong one gets chosen. `SEC` numbering follows the
    /// thread, which is why the association is exact rather than conventional.
    const DST_ACCESS_FMT: crate::cfg::ConfigField;
}

/// Tensix thread 0.
pub struct Thread0;
/// Tensix thread 1.
pub struct Thread1;
/// Tensix thread 2.
pub struct Thread2;

impl TensixThread for Thread0 {
    const INDEX: u32 = 0;
    const DST_ACCESS_FMT: crate::cfg::ConfigField =
        crate::cfg::generated::alu::RISC_DEST_ACCESS_CTRL_SEC0_fmt;
}
impl TensixThread for Thread1 {
    const INDEX: u32 = 1;
    const DST_ACCESS_FMT: crate::cfg::ConfigField =
        crate::cfg::generated::alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt;
}
impl TensixThread for Thread2 {
    const INDEX: u32 = 2;
    const DST_ACCESS_FMT: crate::cfg::ConfigField =
        crate::cfg::generated::alu::RISC_DEST_ACCESS_CTRL_SEC2_fmt;
}

/// A baby RISC-V core, as a type.
///
/// Core identity cannot be discovered at run time -- `mhartid` reads zero on every
/// core and `misa` lies (see [`Core`]) -- so which core an image runs on is a
/// build-time fact. That is exactly what makes it expressible in the type system,
/// and [`PushesTo`] is why it is worth expressing: the `(core, buffer)` table has
/// two entries that hang the core unrecoverably.
///
/// RISCV NC deliberately does not implement this trait: it cannot push at all
/// (`PushTensixInstruction.md:3`).
///
/// ```
/// # use tt_isa::tensix::{PushCore, RiscvT1};
/// fn needs_push_core<C: PushCore>() -> &'static str { C::CORE.name() }
/// assert_eq!(needs_push_core::<RiscvT1>(), "T1");
/// ```
///
/// ```compile_fail
/// # use tt_isa::tensix::{PushCore, RiscvNc};
/// fn needs_push_core<C: PushCore>() {}
/// needs_push_core::<RiscvNc>();
/// ```
pub trait PushCore {
    /// The runtime spelling of the same fact, so the two cannot drift.
    const CORE: Core;
}

/// RISCV B. The only core that can reach all three Tensix threads.
pub struct RiscvB;
/// RISCV T0.
pub struct RiscvT0;
/// RISCV T1.
pub struct RiscvT1;
/// RISCV T2.
pub struct RiscvT2;
/// RISCV NC. Implements neither [`PushCore`] nor [`PushesTo`], by construction.
pub struct RiscvNc;

impl PushCore for RiscvB {
    const CORE: Core = Core::B;
}
impl PushCore for RiscvT0 {
    const CORE: Core = Core::T0;
}
impl PushCore for RiscvT1 {
    const CORE: Core = Core::T1;
}
impl PushCore for RiscvT2 {
    const CORE: Core = Core::T2;
}

/// Which buffer address reaches thread `Th` from this core.
///
/// **This trait exists to make a hang unrepresentable.** The table in
/// `PushTensixInstruction.md:5-9` is:
///
/// | Store address | B | T0 | T1 | T2 |
/// |---|---|---|---|---|
/// | [`INSTRN_BUF_BASE`] | thread 0 | thread 0 | thread 1 | thread 2 |
/// | [`INSTRN1_BUF_BASE`] | thread 1 | *hangs* | *hangs* | *hangs* |
/// | [`INSTRN2_BUF_BASE`] | thread 2 | *hangs* | *hangs* | *hangs* |
///
/// "Hangs" is not a fault that can be caught: it is an unrecoverable lockup of the
/// RISC-V, so a stray store is unrecoverable rather than merely wrong. The impls
/// below are exactly the non-hanging cells, so the bad combinations do not compile
/// instead of being documented and hoped about.
///
/// Note that T0/T1/T2 each reach only their *own* thread, and all three do it
/// through [`INSTRN_BUF_BASE`] -- the buffer address does not identify the thread,
/// the pair does.
///
/// A legal pair resolves:
///
/// ```
/// # use tt_isa::tensix::{PushesTo, RiscvB, Thread2, INSTRN2_BUF_BASE};
/// assert_eq!(<RiscvB as PushesTo<Thread2>>::INSTRN_BUF, INSTRN2_BUF_BASE);
/// ```
///
/// T0 reaching thread 1 would store to `INSTRN1_BUF_BASE` and hang the core. It
/// does not compile:
///
/// ```compile_fail
/// # use tt_isa::tensix::{PushesTo, RiscvT0, Thread1};
/// let _ = <RiscvT0 as PushesTo<Thread1>>::INSTRN_BUF;
/// ```
///
/// Nor does T2 reaching thread 0:
///
/// ```compile_fail
/// # use tt_isa::tensix::{PushesTo, RiscvT2, Thread0};
/// let _ = <RiscvT2 as PushesTo<Thread0>>::INSTRN_BUF;
/// ```
///
/// Nor NC reaching anything at all:
///
/// ```compile_fail
/// # use tt_isa::tensix::{PushesTo, RiscvNc, Thread0};
/// let _ = <RiscvNc as PushesTo<Thread0>>::INSTRN_BUF;
/// ```
pub trait PushesTo<Th: TensixThread>: PushCore {
    /// Address to `sw` the instruction word to.
    const INSTRN_BUF: u64;
}

impl PushesTo<Thread0> for RiscvB {
    const INSTRN_BUF: u64 = INSTRN_BUF_BASE;
}
impl PushesTo<Thread1> for RiscvB {
    const INSTRN_BUF: u64 = INSTRN1_BUF_BASE;
}
impl PushesTo<Thread2> for RiscvB {
    const INSTRN_BUF: u64 = INSTRN2_BUF_BASE;
}
impl PushesTo<Thread0> for RiscvT0 {
    const INSTRN_BUF: u64 = INSTRN_BUF_BASE;
}
impl PushesTo<Thread1> for RiscvT1 {
    const INSTRN_BUF: u64 = INSTRN_BUF_BASE;
}
impl PushesTo<Thread2> for RiscvT2 {
    const INSTRN_BUF: u64 = INSTRN_BUF_BASE;
}

/// Base of the PCBuf / Manual TTSync / Tensix semaphore region
/// (`BabyRISCV/README.md:126-128`).
pub const PC_BUF_BASE: u64 = 0xFFE8_0000;

/// Loading from here blocks until the Tensix coprocessor has finished every
/// instruction from this thread (`ManualTTSync.md`).
///
/// This is the only way for a core to wait for its own pushed instructions before
/// reading `Dst`: a push counts as processed once it reaches a FIFO, not once it
/// executes, and Auto TTSync explicitly does not cover `Dst`
/// (`AutoTTSync.md:56-65`).
pub const COPROCESSOR_DONE_CHECK: u64 = PC_BUF_BASE + 0x04;

/// Loading from here blocks until it is safe to change this thread's MOP Expander
/// configuration (`ManualTTSync.md`).
pub const MOP_EXPANDER_DONE_CHECK: u64 = PC_BUF_BASE + 0x08;

/// `TENSIX_MOP_CFG_BASE`: the issuing core's thread's nine `MopCfg` words
/// (`BabyRISCV/README.md:120`, `MOPExpander.md` "Configuration"), in RISC-V
/// T0, T1 and T2 only. **Write-only**: a load is `UndefinedBehavior`. Write
/// only once [`MOP_EXPANDER_DONE_CHECK`] says no expansion is in progress.
pub const MOP_CFG_BASE: u64 = 0xFFB8_0000;

/// Upper bound of the range subject to the Manual TTSync load-adjacency hazard.
///
/// After *starting* a load from [`COPROCESSOR_DONE_CHECK`] or
/// [`MOP_EXPANDER_DONE_CHECK`], a core must not start another load anywhere in
/// `PC_BUF_BASE ..= PC_BUF_BASE + 0xFFFF` — which includes all eight Tensix
/// semaphores — or in `TENSIX_MAILBOX0_BASE ..= +0x3FFF`, until the first has
/// finished. Violating this returns wrong values or hangs the core outright.
pub const PC_BUF_HAZARD_END: u64 = PC_BUF_BASE + 0xFFFF;

/// Bits of [`SOFT_RESET_0`] that hold the Tensix backend — as opposed to the five
/// baby RISC-V cores — in reset.
///
/// Releasing these is a precondition for any compute: while bit 10 is set, "new
/// Vector Unit (SFPU) instructions will not start (they might or might not be
/// silently discarded)" (`SoftReset.md`). Silently discarded is the dangerous
/// half — a program that never releases it produces no error, just a stale `Dst`.
///
/// Bits 0, 1 and 7 are the unpacker, and `SoftReset.md` requires them to move
/// together. Bits at or above 24 have no effect.
pub const BACKEND_RESET_MASK: u32 = {
    let mut mask = 0u32;
    let mut bit = 0;
    while bit < 24 {
        let is_riscv = bit == 11 || bit == 12 || bit == 13 || bit == 14 || bit == 18;
        if !is_riscv {
            mask |= 1 << bit;
        }
        bit += 1;
    }
    mask
};

// The Manual TTSync registers must sit inside the load-adjacency hazard range,
// since that range is what callers use to decide whether a load is affected.
const _: () = assert!(COPROCESSOR_DONE_CHECK > PC_BUF_BASE);
const _: () = assert!(COPROCESSOR_DONE_CHECK < PC_BUF_HAZARD_END);
const _: () = assert!(MOP_EXPANDER_DONE_CHECK < PC_BUF_HAZARD_END);

/// Tensix backend configuration — `Config`, `ConfigDualWrite` and `ThreadConfig`,
/// mapped one after the other (`BabyRISCV/README.md:135`).
///
/// See [`crate::cfg`] for the field table and the address arithmetic.
pub const TENSIX_CFG_BASE: u64 = 0xFFEF_0000;

/// Tensix `Dst`, mapped into T0/T1/T2's address space (`Dst.md:105`).
/// Unmapped for B and NC, and **not reachable over the NoC** — which is why reading
/// a compute result requires a core to copy it into L1 first.
pub const DST_BASE: u64 = 0xFFBD_8000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_ram_apertures_tile_the_documented_range() {
        for (i, core) in Core::ALL.iter().enumerate() {
            let base = core.local_data_ram_noc_address();
            let end = base + core.local_data_ram_noc_window();
            assert!(
                base >= LOCAL_DATA_RAM_NOC_BASE && end <= LOCAL_DATA_RAM_NOC_END,
                "{} sits outside the documented aperture",
                core.name()
            );
            for other in &Core::ALL[..i] {
                let o_base = other.local_data_ram_noc_address();
                let o_end = o_base + other.local_data_ram_noc_window();
                assert!(
                    end <= o_base || base >= o_end,
                    "{} overlaps {}",
                    core.name(),
                    other.name()
                );
            }
        }
        // Five cores at 0x2000 apiece exactly fill 0xFFB1_4000..0xFFB1_E000.
        assert_eq!(
            LOCAL_DATA_RAM_NOC_END - LOCAL_DATA_RAM_NOC_BASE,
            Core::ALL.len() as u64 * LOCAL_DATA_RAM_NOC_STRIDE
        );
        // The memory map's first row, spelled out rather than derived.
        assert_eq!(Core::B.local_data_ram_noc_address(), 0xFFB1_4000);
        assert_eq!(Core::NC.local_data_ram_noc_address(), 0xFFB1_6000);
        assert_eq!(Core::T0.local_data_ram_noc_address(), 0xFFB1_8000);
        assert_eq!(Core::T1.local_data_ram_noc_address(), 0xFFB1_A000);
        assert_eq!(Core::T2.local_data_ram_noc_address(), 0xFFB1_C000);
    }

    #[test]
    fn local_ram_never_fills_more_than_its_aperture() {
        for core in Core::ALL {
            let size = core.local_data_ram_size() as u64;
            assert!(size <= core.local_data_ram_noc_window());
            // B and NC fill theirs; the T-cores are given twice the window they
            // back, and what occupies the upper half is undocumented. Anything
            // staging here must bound itself by the size.
            let exact = matches!(core, Core::B | Core::NC);
            assert_eq!(size == core.local_data_ram_noc_window(), exact);
        }
    }

    #[test]
    fn disable_reset_bits_are_ten_distinct_bits_below_ten() {
        let mut mask = 0u32;
        for core in Core::ALL {
            let (ram, dr) = core.disable_reset_bits();
            assert_eq!(
                dr,
                ram + 1,
                "{}'s DR bit must follow its RAM bit",
                core.name()
            );
            for bit in [ram, dr] {
                // SoftReset.md:54: bits >= 10 have no effect.
                assert!(bit < 10, "{} names bit {bit}", core.name());
                assert_eq!(mask & (1 << bit), 0, "{} reuses bit {bit}", core.name());
                mask |= 1 << bit;
            }
        }
        assert_eq!(mask, 0x3FF, "the ten documented bits must all be claimed");
    }

    #[test]
    fn the_per_core_orderings_really_are_different() {
        // Blackhole numbers the five cores four different ways. Each of these
        // tables is transcribed from a different page, and the failure mode of
        // copying one into another is a hung core rather than a compile error --
        // so the difference is asserted rather than left as a comment.
        let mut soft_reset = [0u32; 5];
        let mut disable_reset = [0u32; 5];
        let mut hardware = [0u32; 5];
        for (i, core) in Core::ALL.iter().enumerate() {
            soft_reset[i] = core.soft_reset_bit();
            disable_reset[i] = core.disable_reset_bits().0;
            hardware[i] = core.hardware_index();
        }

        assert_eq!(soft_reset, [11, 12, 13, 14, 18]);
        assert_eq!(disable_reset, [6, 0, 2, 4, 8]);
        assert_eq!(hardware, [0, 2, 3, 4, 1]);
        assert_ne!(soft_reset, disable_reset);
        assert_ne!(disable_reset, hardware);

        // The `pc` snapshot block and the local-RAM aperture share the hardware
        // ordering, which is why both are generated from `hardware_index`.
        for core in Core::ALL {
            let i = core.hardware_index() as u64;
            assert_eq!(core.pc_snapshot(), 0xFFB1_3138 + i * 4);
            assert_eq!(
                core.local_data_ram_noc_address(),
                LOCAL_DATA_RAM_NOC_BASE + i * LOCAL_DATA_RAM_NOC_STRIDE
            );
        }
    }

    #[test]
    fn reset_bits_are_distinct_and_match_the_named_masks() {
        let mut mask = 0u32;
        for core in Core::ALL {
            let m = core.soft_reset_mask();
            assert_eq!(mask & m, 0, "{} reuses a reset bit", core.name());
            mask |= m;
        }
        // SoftReset.md:110 names RISCV_SOFT_RESET_0_BRISC / _TRISCS / _NCRISC.
        assert_eq!(Core::B.soft_reset_mask(), 1 << 11);
        let triscs =
            Core::T0.soft_reset_mask() | Core::T1.soft_reset_mask() | Core::T2.soft_reset_mask();
        assert_eq!(triscs, 0b111 << 12);
        assert_eq!(Core::NC.soft_reset_mask(), 1 << 18);
    }

    #[test]
    fn default_reset_pcs_are_in_l1_and_do_not_overlap() {
        let mut pcs = Core::ALL.map(|c| c.default_reset_pc());
        pcs.sort_unstable();
        assert_eq!(pcs, [0x0_0000, 0x0_6000, 0x0_A000, 0x0_E000, 0x1_2000]);
        assert!(pcs.iter().all(|&pc| (pc as u64) < L1_SIZE));
    }

    #[test]
    fn risc_b_has_no_reset_pc_override() {
        // Its entry point is hardwired to L1 offset 0, which every linker script
        // for B has to honour.
        assert_eq!(Core::B.reset_pc_register(), None);
        assert_eq!(Core::B.reset_pc_override(), None);
        assert_eq!(Core::B.default_reset_pc(), 0);
        for core in [Core::T0, Core::T1, Core::T2, Core::NC] {
            assert!(
                core.reset_pc_register().is_some(),
                "{} should be overridable",
                core.name()
            );
        }
    }

    #[test]
    fn trisc_override_bits_are_distinct_within_one_register() {
        let (reg0, b0) = Core::T0.reset_pc_override().unwrap();
        let (reg1, b1) = Core::T1.reset_pc_override().unwrap();
        let (reg2, b2) = Core::T2.reset_pc_override().unwrap();
        assert_eq!((reg0, reg1), (reg2, reg2));
        assert_eq!(b0 | b1 | b2, 0b111);
        // NC uses a different register, so its bit 0 does not collide with T0's.
        let (nc_reg, nc_bit) = Core::NC.reset_pc_override().unwrap();
        assert_ne!(nc_reg, reg0);
        assert_eq!(nc_bit, 1);
    }

    #[test]
    fn pc_snapshots_are_four_bytes_apart_in_core_order() {
        assert_eq!(Core::B.pc_snapshot(), 0xFFB1_3138);
        assert_eq!(Core::NC.pc_snapshot(), Core::B.pc_snapshot() + 4);
        assert_eq!(Core::T0.pc_snapshot(), Core::B.pc_snapshot() + 8);
        assert_eq!(Core::T2.pc_snapshot(), Core::B.pc_snapshot() + 16);
    }

    #[test]
    fn backend_reset_mask_excludes_every_riscv_core() {
        for core in Core::ALL {
            assert_eq!(
                BACKEND_RESET_MASK & core.soft_reset_mask(),
                0,
                "the backend mask would also release {}",
                core.name()
            );
        }
        // It must cover the unit that matters most here: bit 10, the Matrix Unit
        // and Vector Unit.
        assert_ne!(BACKEND_RESET_MASK & (1 << 10), 0);
        // The three unpacker bits move together (SoftReset.md).
        assert_eq!(BACKEND_RESET_MASK & 0b1000_0011, 0b1000_0011);
        // Nothing above bit 23, which has no effect.
        assert_eq!(BACKEND_RESET_MASK >> 24, 0);
    }

    #[test]
    fn manual_ttsync_registers_are_adjacent() {
        // Their placement relative to the hazard range is asserted at compile
        // time; this pins the layout `ManualTTSync.md` describes.
        assert_eq!(
            COPROCESSOR_DONE_CHECK - PC_BUF_BASE,
            4,
            "after one padding word"
        );
        assert_eq!(MOP_EXPANDER_DONE_CHECK - COPROCESSOR_DONE_CHECK, 4);
    }

    #[test]
    fn only_nc_cannot_push_tensix_instructions() {
        assert!(!Core::NC.can_push_tensix());
        assert!(Core::ALL.iter().filter(|c| c.can_push_tensix()).count() == 4);
    }

    /// Every type that can push names a core that [`Core::can_push_tensix`] agrees
    /// can push.
    ///
    /// This is what keeps the runtime enum and the type-level table from drifting.
    /// `RiscvNc` cannot appear here: it implements neither trait. That absence is
    /// gated by the `compile_fail` example on [`PushCore`], which rustdoc runs --
    /// this module is `cfg(test)`, so an example placed here would never run at all.
    #[test]
    fn every_pushing_core_type_agrees_with_the_enum() {
        fn check<C: PushCore>() {
            assert!(
                C::CORE.can_push_tensix(),
                "{} implements PushCore but the enum says it cannot push",
                C::CORE.name()
            );
        }
        check::<RiscvB>();
        check::<RiscvT0>();
        check::<RiscvT1>();
        check::<RiscvT2>();
    }

    /// The buffer address for each legal `(core, thread)` pair, against the table in
    /// `PushTensixInstruction.md:5-9`.
    ///
    /// The illegal pairs are asserted by absence -- `PushesTo` has six impls and the
    /// table has six non-hanging cells -- and by the `compile_fail` examples on
    /// [`PushesTo`] itself, which are the gate actually watched failing: it is the
    /// hang that matters, and a hang cannot be caught at run time.
    #[test]
    fn the_legal_push_pairs_use_the_documented_buffers() {
        // RISCV B reaches all three threads, one buffer each.
        assert_eq!(
            <RiscvB as PushesTo<Thread0>>::INSTRN_BUF,
            INSTRN_BUF_BASE,
            "B -> Tensix thread 0"
        );
        assert_eq!(<RiscvB as PushesTo<Thread1>>::INSTRN_BUF, INSTRN1_BUF_BASE);
        assert_eq!(<RiscvB as PushesTo<Thread2>>::INSTRN_BUF, INSTRN2_BUF_BASE);

        // T0/T1/T2 reach only their own thread, and all three do it through the
        // *same* address. The buffer does not identify the thread; the pair does.
        assert_eq!(<RiscvT0 as PushesTo<Thread0>>::INSTRN_BUF, INSTRN_BUF_BASE);
        assert_eq!(<RiscvT1 as PushesTo<Thread1>>::INSTRN_BUF, INSTRN_BUF_BASE);
        assert_eq!(<RiscvT2 as PushesTo<Thread2>>::INSTRN_BUF, INSTRN_BUF_BASE);
    }

    /// The per-thread `Dst` access format fields are the ones the specification
    /// names, and they are distinct.
    ///
    /// Distinctness is the point: the previous spelling picked between them by
    /// matching on a runtime thread index, where a missing arm is a silent
    /// misconfiguration of a *different* thread rather than an error.
    #[test]
    fn each_thread_carries_its_own_dst_access_format_field() {
        use crate::cfg::generated::alu;

        assert_eq!(Thread0::DST_ACCESS_FMT, alu::RISC_DEST_ACCESS_CTRL_SEC0_fmt);
        assert_eq!(Thread1::DST_ACCESS_FMT, alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt);
        assert_eq!(Thread2::DST_ACCESS_FMT, alu::RISC_DEST_ACCESS_CTRL_SEC2_fmt);

        let shamts = [
            Thread0::DST_ACCESS_FMT.shamt(),
            Thread1::DST_ACCESS_FMT.shamt(),
            Thread2::DST_ACCESS_FMT.shamt(),
        ];
        assert_eq!(
            shamts.len(),
            3,
            "three threads, three fields -- see the distinctness check below"
        );
        assert!(
            shamts[0] != shamts[1] && shamts[1] != shamts[2] && shamts[0] != shamts[2],
            "the three fields must be distinct: {shamts:?}"
        );

        assert_eq!([Thread0::INDEX, Thread1::INDEX, Thread2::INDEX], [0, 1, 2]);
    }
}
