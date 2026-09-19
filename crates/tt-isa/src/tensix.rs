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

/// Tensix `Dst`, mapped into T0/T1/T2's address space (`Dst.md:105`).
/// Unmapped for B and NC, and **not reachable over the NoC** — which is why reading
/// a compute result requires a core to copy it into L1 first.
pub const DST_BASE: u64 = 0xFFBD_8000;

#[cfg(test)]
mod tests {
    use super::*;

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
}
