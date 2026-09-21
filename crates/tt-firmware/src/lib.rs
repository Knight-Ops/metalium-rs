//! Bare-metal runtime for the Blackhole baby RISC-V cores.
//!
//! # What this runtime does not have
//!
//! * **No `ebreak` on panic.** `ebreak` halts the core and requires an external
//!   agent to resume it, so it cannot back Rust's abort path — a panicking core
//!   would simply stop, indistinguishable from one that never started. Instead
//!   [`panic`] publishes [`status::PANICKED`] and a code to the L1 mailbox, then
//!   spins, so the host can report *why* rather than timing out.
//! * **No `fence.i`.** Zifencei is not implemented and `fence.i` executes as a
//!   `nop`. This runtime never needs it: code is written to L1 before reset is
//!   released, and leaving reset invalidates the instruction cache
//!   (`SoftReset.md:116`). Self-modifying code, or reloading a running core, would
//!   need `RISCV_IC_INVALIDATE_InvalidateAll` in Tensix backend config.
//! * **No core identity at run time.** `mhartid` reads zero on every core and
//!   `misa` lies, so which core this is comes from the link script.

#![no_std]

use core::panic::PanicInfo;
use core::ptr::{read_volatile, write_volatile};

use tt_isa::mailbox::{self, panic_code, status};

// Entry point.
//
// Sets up the stack, zeroes `.bss`, then calls `main`. Written in assembly
// because none of it is expressible in Rust: there is no stack yet.
core::arch::global_asm!(
    r#"
    .section .text.start, "ax"
    .global _start
    .type _start, @function
_start:
    // Stack lives in local data RAM. The core's own accesses stall automatically
    // while that RAM zeroes itself after reset, so no delay is needed here.
    la      sp, _stack_top

    // Zero .bss. The linker aligns both symbols to 4, so a word loop is exact.
    la      t0, __bss_start
    la      t1, __bss_end
1:
    bgeu    t0, t1, 2f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b
2:
    call    {main}

    // `main` is diverging, so this is unreachable. Spin rather than fall into
    // whatever follows in L1.
3:
    j       3b
    .size _start, . - _start
"#,
    main = sym crate::main_trampoline,
);

extern "Rust" {
    /// Provided by the binary. Diverging: there is nothing to return to.
    fn firmware_main() -> !;
}

/// Bridges the assembly prologue to the binary's `firmware_main`.
///
/// # Safety
///
/// Called exactly once, from `_start`, with a valid stack.
#[no_mangle]
unsafe extern "C" fn main_trampoline() -> ! {
    set_status(status::RUNNING);
    unsafe { firmware_main() }
}

/// Write a word to this tile's L1.
///
/// # Safety
///
/// `offset` must be within L1 and 4-byte aligned.
#[inline]
pub unsafe fn l1_write32(offset: u64, value: u32) {
    unsafe { write_volatile(offset as *mut u32, value) }
}

/// Read a word from this tile's L1.
///
/// # Safety
///
/// As [`l1_write32`].
#[inline]
pub unsafe fn l1_read32(offset: u64) -> u32 {
    unsafe { read_volatile(offset as *const u32) }
}

/// Publish a status word, ensuring it is visible to the host.
pub fn set_status(value: u32) {
    // SAFETY: the mailbox is a fixed, aligned location inside L1.
    unsafe { l1_write32(mailbox::STATUS, value) };
    publish();
}

/// Make prior stores visible to observers outside this core.
///
/// The L0 data cache is not coherent — nothing written by the NoC, the unpackers,
/// the packers, or another baby invalidates it (`MemoryOrdering.md:59`) — and
/// store-then-load of non-overlapping ranges has no ordering guarantee in either
/// direction (`MemoryOrdering.md:54`). A `fence` is therefore not decoration: it is
/// what makes a heartbeat the host can see.
#[inline]
pub fn publish() {
    // SAFETY: `fence` has no operands and no memory-safety consequences.
    unsafe { core::arch::asm!("fence", options(nostack, preserves_flags)) };
}

/// Publish a result and mark the firmware finished.
pub fn finish(result: u32) {
    // SAFETY: fixed aligned mailbox locations.
    unsafe { l1_write32(mailbox::RESULT, result) };
    set_status(status::DONE);
}

/// Stop, having told the host why.
pub fn fail(code: u32) -> ! {
    // SAFETY: fixed aligned mailbox locations.
    unsafe {
        l1_write32(mailbox::PANIC_CODE, code);
        l1_write32(mailbox::STATUS, status::PANICKED);
    }
    publish();
    spin()
}

/// Halt without halting the core.
///
/// A real halt would need `ebreak`, which cannot be resumed without an external
/// agent. Spinning keeps the core in a state the host can still inspect.
///
/// Deliberately *not* `core::hint::spin_loop()`. On RISC-V that emits `pause`,
/// which is Zihintpause — an extension `InstructionSet.md` does not list among
/// those Blackhole implements, putting it in `UndefinedBehavior` territory with no
/// illegal-instruction trap to catch it. ttsim rejects it outright
/// (`rv32_fence: fence_mode=0x10`), which is how this was found. An empty
/// assembly block keeps the loop from being optimised away without emitting
/// anything.
pub fn spin() -> ! {
    loop {
        // SAFETY: empty, with no operands and no memory effects.
        unsafe { core::arch::asm!("", options(nomem, nostack, preserves_flags)) };
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // The message is discarded: formatting it needs an allocator-free `Write`
    // implementation and several KiB of code, on a core with 4 KiB of stack. The
    // panic *location* is what debugging actually needs, and the code below stands
    // in for it until there is a reason to carry more.
    let code = if info.location().is_some() { panic_code::EXPLICIT } else { panic_code::ARITHMETIC };
    fail(code)
}

/// Pushing Tensix instructions from a baby RISC-V core.
pub mod tensix {
    use tt_isa::sfpu::Instruction;
    use tt_isa::tensix::{self, PushesTo, TensixThread};

    /// Push one Tensix instruction into this core's Tensix thread.
    ///
    /// A push is a plain `sw` of the instruction word to `INSTRN_BUF_BASE`
    /// (`PushTensixInstruction.md:5`). If the FIFO is full the core stalls
    /// automatically, so there is nothing to check.
    ///
    /// # Why `sw` rather than `.ttinsn`
    ///
    /// `.ttinsn IMM32` is the same push expressed as an instruction in the stream —
    /// the word is `IMM32` rotated left by two — which lets the T0/T1/T2
    /// instruction caches fuse up to four adjacent pushes into a single cycle.
    /// That is a throughput optimisation, and it costs something here: the fused
    /// words live in the compressed-instruction encoding space, so they disassemble
    /// as garbage, and the instruction-set gate in `crates/tt-tests/build.rs` would
    /// have to stop rejecting undecodable instructions to let them through. The
    /// encoding is implemented and tested in `tt_isa::sfpu`; using it belongs with
    /// the rest of the performance work, when there is a measurement to justify it.
    ///
    /// # Which core, and which thread
    ///
    /// `C` is the core this image runs on and `Th` is the Tensix thread to reach.
    /// Both are type parameters rather than arguments because the buffer address
    /// depends on the *pair*, and two of the nine combinations hang the RISC-V
    /// unrecoverably (`PushTensixInstruction.md:5-9`). `tt_isa::tensix::PushesTo`
    /// is implemented for exactly the six that do not, so a hanging push is a
    /// compile error rather than a lockup. `C` cannot be inferred -- core identity
    /// is not discoverable at run time -- so every call site names it, which is the
    /// point.
    ///
    /// # Safety
    ///
    /// The caller is responsible for the Tensix-side state the instruction assumes
    /// — in particular that the backend is out of soft reset, since instructions
    /// issued while it is held "might or might not be silently discarded" — and for
    /// `C` actually being the core executing this code. Nothing can check the
    /// latter: `mhartid` reads zero everywhere.
    #[inline]
    pub unsafe fn push<C, Th>(instruction: Instruction)
    where
        Th: TensixThread,
        C: PushesTo<Th>,
    {
        unsafe {
            core::ptr::write_volatile(
                <C as PushesTo<Th>>::INSTRN_BUF as *mut u32,
                instruction.word(),
            )
        }
    }

    /// Push a raw instruction word.
    ///
    /// The corpus firmware needs this: its program arrives as words in L1, encoded
    /// by the host, so there is no [`Instruction`] to push. Everything the word
    /// means was decided on the host side, where the generated table is.
    ///
    /// # Safety
    ///
    /// As [`push`], and additionally the word must be a valid encoding — nothing
    /// here checks it.
    #[inline]
    pub unsafe fn push_word<C, Th>(word: u32)
    where
        Th: TensixThread,
        C: PushesTo<Th>,
    {
        unsafe { core::ptr::write_volatile(<C as PushesTo<Th>>::INSTRN_BUF as *mut u32, word) }
    }

    /// Block until the Tensix coprocessor has retired every instruction this
    /// thread pushed.
    ///
    /// Required before reading `Dst`, and not optional: a push counts as processed
    /// once it reaches a FIFO rather than once it executes
    /// (`PushTensixInstruction.md:27`), and Auto TTSync does not cover `Dst`
    /// (`AutoTTSync.md:56-65`). Without this, a read of `Dst` races the store that
    /// was supposed to fill it.
    ///
    /// # Why this is one assembly block
    ///
    /// Two separate constraints, both from `ManualTTSync.md`, and neither
    /// expressible in Rust:
    ///
    /// * The store before the load is what orders the earlier instruction pushes
    ///   against the load. Its value is discarded; ordering is its only effect.
    /// * The load blocks *inside the memory subsystem*, which does not by itself
    ///   stop the core. The `andi` consuming the result is what actually waits.
    ///   It is also what satisfies the load-adjacency hazard: after starting this
    ///   load, the core must not start another load anywhere in
    ///   `PC_BUF_BASE ..= +0xFFFF` — all eight Tensix semaphores included — until
    ///   it finishes, on pain of wrong values or a hung core. The alternative to a
    ///   consuming ALU instruction is seven intervening instructions.
    ///
    /// Splitting these across statements would let the compiler schedule something
    /// between them, so the sequence is emitted as one block.
    #[inline]
    pub fn wait_for_coprocessor() {
        // SAFETY: the address is fixed; the block has no side effects beyond the
        // documented synchronisation, and clobbers only the scratch registers it
        // declares.
        unsafe {
            core::arch::asm!(
                "sw   zero, 0({addr})",
                "lw   {tmp}, 0({addr})",
                "andi {tmp}, {tmp}, 0",
                addr = in(reg) tensix::COPROCESSOR_DONE_CHECK as u32,
                tmp = out(reg) _,
                options(nostack),
            )
        }
    }

    /// Read a 32-bit `Dst` element through the RISCV mapping.
    ///
    /// Valid only when `RISC_DEST_ACCESS_CTRL_SEC[thread].fmt` selects one of the
    /// 32-bit shapes (0 or 1), which is the reset default. For the 16-bit shapes a
    /// word access misbehaves and `lhu` is required.
    ///
    /// # Safety
    ///
    /// `address` must come from [`tt_isa::sfpu::dst32_address`], and the
    /// coprocessor must have retired the instruction that wrote it — see
    /// [`wait_for_coprocessor`].
    #[inline]
    pub unsafe fn read_dst32(address: u64) -> u32 {
        unsafe { core::ptr::read_volatile(address as *const u32) }
    }
}

/// Reading and writing Tensix backend configuration from a baby RISC-V core.
pub mod cfg {
    use tt_isa::cfg::{ConfigBank, ConfigField, ThreadConfigField};

    /// Read the word containing `field`.
    ///
    /// # Safety
    ///
    /// The Tensix backend must be out of soft reset.
    #[inline]
    pub unsafe fn read_config_word(field: ConfigField, bank: ConfigBank) -> u32 {
        unsafe { core::ptr::read_volatile(field.riscv_address(bank) as *const u32) }
    }

    /// Set `field` to `value`, leaving the rest of its word alone.
    ///
    /// # Why this is a read-modify-write, and what that costs
    ///
    /// RISC-V can write `Config` **only with `sw`** — a whole 32-bit store — so a
    /// bitfield narrower than a word has to be merged in by hand. There is no
    /// atomic form, and the Configuration Unit enforces ordering across all three
    /// Tensix threads regardless of which issued a request, so heavy use from one
    /// thread starves the others. Neither matters for one-time setup before a
    /// kernel runs; both would matter in a loop.
    ///
    /// `WRCFG` and `RMWCIB` are the Tensix-side alternatives, and `RMWCIB` does
    /// the merge in hardware. They belong in a kernel's instruction stream rather
    /// than in a core's prologue, which is why this exists.
    ///
    /// # Safety
    ///
    /// The Tensix backend must be out of soft reset, and no Tensix instruction
    /// that reads this field may be in flight.
    #[inline]
    pub unsafe fn write_config_field(field: ConfigField, bank: ConfigBank, value: u32) {
        let address = field.riscv_address(bank) as *mut u32;
        // SAFETY: the caller guarantees the backend is out of reset; the address
        // comes from the generated field table.
        unsafe {
            let word = core::ptr::read_volatile(address);
            core::ptr::write_volatile(address, field.insert(word, value));
        }
    }

    /// Read a `ThreadConfig` field for `thread`.
    ///
    /// There is no writing counterpart: RISC-V cannot write `ThreadConfig` at all.
    /// `SETC16` is the only way, and it is a Tensix instruction.
    ///
    /// # Safety
    ///
    /// `thread` must be 0, 1 or 2, and the backend must be out of soft reset.
    #[inline]
    pub unsafe fn read_thread_config(field: ThreadConfigField, thread: u32) -> u16 {
        // The entry is 32 bits wide with the value in the low half; read the word
        // and narrow, rather than issuing a 16-bit load against an MMIO region.
        let word =
            unsafe { core::ptr::read_volatile(field.riscv_read_address(thread) as *const u32) };
        field.extract(word as u16)
    }

    /// Which `Config` bank the given Tensix thread is currently using.
    ///
    /// Selected by `ThreadConfig[thread].CFG_STATE_ID_StateID`. Reading it rather
    /// than assuming bank 0 costs one load and means configuration written here
    /// lands where the coprocessor will actually look for it.
    ///
    /// **Not usable against ttsim.** It maps the whole configuration aperture to a
    /// flat `Config` array with the bank hardcoded to zero, and models no
    /// `ThreadConfig` region, so this read is fatal there. Code that must run on
    /// the simulator has to assume bank 0 and say why.
    ///
    /// # Safety
    ///
    /// As [`read_thread_config`].
    #[inline]
    pub unsafe fn active_bank(thread: u32) -> ConfigBank {
        use tt_isa::cfg::generated::thread::CFG_STATE_ID_StateID;
        // SAFETY: delegated to the caller.
        if unsafe { read_thread_config(CFG_STATE_ID_StateID, thread) } == 0 {
            ConfigBank::Bank0
        } else {
            ConfigBank::Bank1
        }
    }
}
