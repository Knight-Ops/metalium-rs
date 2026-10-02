//! Phase 10 gate (X2b): `ADDDMAREG`, GPR arithmetic in the Scalar Unit.
//!
//! A looped kernel steps through tiles by adding a stride to an address held
//! in a Tensix GPR and writing the GPR to the unpacker's base address -- the
//! same words every iteration, which a `MOP` can repeat. `ADDDMAREG` is drawn
//! only on a Wormhole page (`UNVERIFIED` in the table), so before any kernel
//! uses it this gate does, alone: GPRs loaded by `SETDMAREG`, added by both
//! forms of `ADDDMAREG` -- register + register, register + six-bit immediate
//! -- and in place, as a loop's increment would be, the result written by
//! `WRCFG` into `THCON_SEC0_REG3_Base_address` (a whole 32-bit word ttsim
//! models, since kernels write it) and read back by the host through the
//! debug pair (`probe_cfgreg`). Every field takes several values, so a passing
//! run on ttsim and both cards confirms the layout.

use tt_device::{core_control::WaitError, tlb::WindowKind};
use tt_isa::backend;
use tt_isa::cfg::generated::thcon;
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::mailbox::{self, status};
use tt_isa::tensix;
use tt_tests::firmware;
use tt_tests::harness::{self, advance, in_device, Dev, BUDGET, DST_FMT_FP32};

const WORD: u16 = thcon::THCON_SEC0_REG3_Base_address.addr32();

/// Run `program` on the gate tile's single core, then read `WORD` back.
fn run_and_read(dev: &mut Dev<'_>, program: &[Instruction]) -> u32 {
    let tile = harness::tensix_tile();
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    dev.release_tensix_backend(&w, tile).unwrap();
    dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
    let d = mailbox::Descriptor {
        thread_index: harness::CORE_THREAD,
        dst_access_fmt: DST_FMT_FP32,
        program_len: program.len() as u32,
        dump_row_count: 0,
        push_window: mailbox::SIM_PUSH_WINDOW,
        ..Default::default()
    };
    for (at, v) in d.writes(mailbox::role::Mailbox::single_core()) {
        dev.write32(&w, tile, at, v).unwrap();
    }
    for (i, insn) in program.iter().enumerate() {
        dev.write32(&w, tile, mailbox::PROGRAM + (i as u64) * 4, insn.word())
            .unwrap();
    }
    dev.load_and_start(
        &w,
        tile,
        harness::CORE,
        firmware::CORPUS,
        firmware::LOAD_ADDRESS,
    )
    .unwrap();
    match dev
        .wait_for_status(&w, tile, BUDGET, |s| s == status::DONE)
        .unwrap()
    {
        Ok(_) => {}
        Err(WaitError::Panicked { code }) => panic!("firmware panicked, code {code}"),
        Err(e) => panic!("{e}"),
    }
    dev.write32(&w, tile, tensix::CFGREG_RD_CNTL, WORD as u32)
        .unwrap();
    // "a few cycles later" (`BackendConfiguration.md:66`).
    advance(dev, 64);
    dev.read32(&w, tile, tensix::CFGREG_RDDATA).unwrap()
}

fn set(gpr: u32, v: u32) -> [Instruction; 2] {
    backend::set_gpr(gpr, v).unwrap()
}

fn add(result: u32, right: u32, left: u32) -> Instruction {
    encode::adddmareg(result, right, left).unwrap()
}

#[cfg(feature = "silicon")]
fn add_imm(result: u32, imm: u32, left: u32) -> Instruction {
    encode::adddmare_gi(result, imm, left).unwrap()
}

/// `program`, then `gpr` written to `WORD` and the `NOP` `WRCFG.md` asks for.
fn then_write(mut program: Vec<Instruction>, gpr: u32) -> Vec<Instruction> {
    program.push(backend::write_word(gpr, WORD).unwrap());
    program.push(tt_isa::sfpu::nop());
    program
}

fn check(cases: Vec<(&str, Vec<Instruction>, u32, u32)>) {
    in_device(|dev| {
        for (name, program, gpr, want) in cases {
            let got = run_and_read(dev, &then_write(program, gpr));
            assert_eq!(got, want, "{name}: {got:#x}, want {want:#x}");
        }
    });
}

/// The register form: what a kernel's tile stepping uses, the stride held in
/// a GPR. ttsim and silicon.
#[test]
fn adddmareg_adds_two_registers() {
    check(vec![
        (
            "register + register, across GPR groups",
            [
                set(8, 0x0012_3400).to_vec(),
                set(21, 0x0000_0567).to_vec(),
                vec![add(30, 21, 8)],
            ]
            .concat(),
            30,
            0x0012_3967,
        ),
        (
            "a carry out of the low half",
            [
                set(12, 0x0000_FFF0).to_vec(),
                set(14, 0x0000_003F).to_vec(),
                vec![add(13, 14, 12)],
            ]
            .concat(),
            13,
            0x0001_002F,
        ),
        (
            // A loop's increment: the same word four times, in place.
            "in place, repeated",
            [
                set(16, 0x0002_0000).to_vec(),
                set(17, 4160 / 16).to_vec(),
                vec![add(16, 17, 16); 4],
            ]
            .concat(),
            16,
            0x0002_0000 + 4 * (4160 / 16),
        ),
    ]);
}

/// The six-bit immediate form, on silicon only: ttsim does not implement it
/// (`tensix_addgpr: op_b_is_const=1`, divergence row 67).
#[cfg(feature = "silicon")]
#[test]
fn adddmareg_adds_an_immediate() {
    check(vec![
        (
            "register + immediate",
            [set(9, 0x00AB_0000).to_vec(), vec![add_imm(10, 42, 9)]].concat(),
            10,
            0x00AB_002A,
        ),
        (
            "a carry out of the low half",
            [set(12, 0x0000_FFF0).to_vec(), vec![add_imm(13, 63, 12)]].concat(),
            13,
            0x0001_002F,
        ),
    ]);
}
