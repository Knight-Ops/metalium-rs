//! Can a Tensix instruction stream write backend configuration, and can the host
//! see that it did?
//!
//! Two questions, and the second is what makes the first answerable.
//!
//! 1. **Does ttsim execute `SETDMAREG` and `WRCFG`?** `tt_isa::backend` stages
//!    configuration as pushed instructions rather than as RISC-V stores, because
//!    `ThreadConfig` has no store path at all. That plan rests on `WRCFG`, which
//!    sources from a Tensix GPR, which only `SETDMAREG` can load. `SETDMAREG` is
//!    `Provenance::WormholeOnly` in our own table, and a `strings` sweep of
//!    `libttsim_bh.so` finds `tensix_wrcfg` and `tensix_setgpr` but no `setdmareg`
//!    decode symbol. That is suggestive, not conclusive — ttsim names its handlers
//!    after what they do, not after the specification's mnemonic — so this measures
//!    it instead of inferring it.
//!
//! 2. **Are `CFGREG_RD_CNTL` / `CFGREG_RDDATA` where we think they are?**
//!    `BackendConfiguration.md:66` says backend configuration is not NoC-visible and
//!    this register pair is the read-only way in. The Blackhole tree names the pair
//!    but gives no addresses; ours come from a Wormhole *Ethernet* page. ttsim
//!    carries `TENSIX_CREG_READ` and `TENSIX_CREG_RDDATA` handlers, so a wrong
//!    address falls off its decode switch and kills the child rather than returning
//!    a plausible number.
//!
//! Every case forks, so a refusal is a recorded result rather than a dead runner.

use tt_device::{core_control::WaitError, tlb::WindowKind, Device};
use tt_isa::backend;
use tt_isa::cfg::generated::alu;
use tt_isa::cfg::ConfigField;
use tt_isa::isa::Instruction;
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{grid, Noc0, NocCoord};
use tt_isa::sfpu::{self, store_format};
use tt_isa::tensix::{self, Core};
use tt_tests::firmware;
use tt_ttsim::{fork_scope, Simulator};

type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

const CORE: Core = Core::T1;
const CORE_THREAD: u32 = 1;
const DST_FMT_FP32: u32 = 0;
const BUDGET: u64 = 400_000;
/// A scratch GPR well clear of anything the firmware touches.
const SCRATCH_GPR: u32 = 8;

fn tile() -> NocCoord<Noc0> {
    assert!(grid::is_tensix(3, 4));
    NocCoord::new(3, 4).unwrap()
}

/// Stage and run `program`, returning `Dst` rows 0..4 plus the device, so a caller
/// can go on to read tile registers through the same window.
fn run_with<R>(
    dev: &mut Dev<'_>,
    program: &[Instruction],
    after: impl FnOnce(&mut Dev<'_>, &tt_device::Window, NocCoord<Noc0>) -> R,
) -> R {
    assert!(program.len() as u32 <= mailbox::PROGRAM_MAX);
    let tile = tile();
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    dev.release_tensix_backend(&w, tile).unwrap();

    dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
    dev.write32(&w, tile, mailbox::THREAD_INDEX, CORE_THREAD)
        .unwrap();
    dev.write32(&w, tile, mailbox::DST_ACCESS_FMT, DST_FMT_FP32)
        .unwrap();
    dev.write32(&w, tile, mailbox::PROGRAM_LEN, program.len() as u32)
        .unwrap();
    dev.write32(&w, tile, mailbox::DUMP_ROW_FIRST, 0).unwrap();
    dev.write32(&w, tile, mailbox::DUMP_ROW_COUNT, 4).unwrap();
    for (i, insn) in program.iter().enumerate() {
        dev.write32(&w, tile, mailbox::PROGRAM + (i as u64) * 4, insn.word())
            .unwrap();
    }

    dev.load_and_start(&w, tile, CORE, firmware::CORPUS, firmware::LOAD_ADDRESS)
        .unwrap();
    match dev
        .wait_for_status(&w, tile, BUDGET, |s| s == status::DONE)
        .unwrap()
    {
        Ok(_) => {}
        Err(WaitError::Panicked { code }) => panic!("firmware panicked, code {code}"),
        Err(e) => panic!("{e}"),
    }
    after(dev, &w, tile)
}

/// Does this program run to completion under the simulator?
fn survives(program: &[Instruction]) -> bool {
    fork_scope(|| {
        let mut sim = Simulator::open().unwrap();
        let mut dev = Device::open(sim.transport()).unwrap();
        run_with(&mut dev, program, |_, _, _| ());
    })
    .is_ok()
}

/// A program that computes something, so a run that survives has also done work.
fn control_tail() -> Vec<Instruction> {
    let mut p = Vec::new();
    p.extend(sfpu::load_f32(0, 1.5f32.to_bits()).unwrap());
    p.push(sfpu::store(0, store_format::FP32, 0, 0).unwrap());
    p
}

#[test]
fn ttsim_executes_setdmareg_and_wrcfg() {
    // The control first: the tail on its own must run, so a failure below is the
    // configuration instructions and not the harness.
    assert!(
        survives(&control_tail()),
        "the control program must run; if this fails nothing else here means anything"
    );

    // `RISC_DEST_ACCESS_CTRL_SEC1_fmt` rather than something more neutral: ttsim
    // models `Config` as a switch over specific registers rather than as an array
    // (`docs/ttsim-divergence.md` row 21), so a register no workload has exercised
    // is fatal. This one is exercised -- the corpus firmware writes it through the
    // RISC-V path on every run -- so a refusal here is about the *instructions*,
    // which is the question. Writing the reset default keeps the tail's answer
    // unchanged.
    let mut words = backend::ConfigWords::new();
    words
        .set(alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt, DST_FMT_FP32)
        .unwrap();
    let mut staged = [sfpu::nop(); 8];
    let n = words.program(SCRATCH_GPR, &mut staged).unwrap();

    let mut p: Vec<Instruction> = staged[..n].to_vec();
    p.extend(control_tail());
    assert!(
        survives(&p),
        "ttsim declined SETDMAREG or WRCFG. tt_isa::backend stages configuration as \
         instructions and cannot work on the simulator if so; the fallback is RISC-V \
         `sw` for Config plus SETC16 for ThreadConfig"
    );
}

#[test]
fn ttsim_executes_setc16() {
    let entry = backend::ThreadConfigEntry::zeroed(
        tt_isa::cfg::generated::thread::CFG_STATE_ID_StateID.addr32(),
    )
    .set(tt_isa::cfg::generated::thread::CFG_STATE_ID_StateID, 0)
    .unwrap();

    let mut p = vec![entry.encode().unwrap()];
    p.extend(control_tail());
    assert!(
        survives(&p),
        "SETC16 is the only way to write ThreadConfig; the unpacker path needs it"
    );
}

/// Read one `Config` word back over the NoC, through the debug register pair.
fn read_config_word(
    dev: &mut Dev<'_>,
    w: &tt_device::Window,
    tile: NocCoord<Noc0>,
    field: ConfigField,
) -> u32 {
    dev.write32(w, tile, tensix::CFGREG_RD_CNTL, field.addr32() as u32)
        .unwrap();
    // "a few cycles later" (`BackendConfiguration.md:66`).
    dev.tick(64);
    dev.read32(w, tile, tensix::CFGREG_RDDATA).unwrap()
}

/// Stage `value` into `RISC_DEST_ACCESS_CTRL_SEC1_fmt` through the instruction
/// path, then read the containing `Config` word back over the NoC.
fn stage_then_read_back(dev: &mut Dev<'_>, value: u32) -> u32 {
    let mut words = backend::ConfigWords::new();
    words
        .set(alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt, value)
        .unwrap();
    let mut staged = [sfpu::nop(); 8];
    let n = words.program(SCRATCH_GPR, &mut staged).unwrap();
    let mut p: Vec<Instruction> = staged[..n].to_vec();
    p.extend(control_tail());
    let word = run_with(dev, &p, |dev, w, tile| {
        read_config_word(dev, w, tile, alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt)
    });
    alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt.extract(word)
}

#[test]
fn the_host_can_read_backend_configuration_back() {
    // What this proves: the two debug-register addresses are right, ttsim models
    // the path, and a configuration write staged as instructions is observable from
    // the host without trusting the firmware that applied it.
    //
    // The negative control is inside the test rather than beside it, because the
    // claim is comparative. A readback that returned a constant -- which is what a
    // wrong `CFGREG_RDDATA` would do if it happened to land on a readable register
    // rather than a fatal one -- satisfies any single-value assertion. Two
    // different staged values must produce two different readbacks.
    const FP32: u32 = 0;
    const BF16: u32 = 3;

    let result = fork_scope(|| {
        let mut sim = Simulator::open().unwrap_or_else(|e| panic!("{e}"));
        let mut dev = Device::open(sim.transport()).unwrap_or_else(|e| panic!("{e}"));

        let fp32 = stage_then_read_back(&mut dev, FP32);
        let bf16 = stage_then_read_back(&mut dev, BF16);

        assert_eq!(fp32, FP32, "staged {FP32}, read back {fp32}");
        assert_eq!(bf16, BF16, "staged {BF16}, read back {bf16}");
        assert_ne!(
            fp32, bf16,
            "the readback does not depend on what was staged, so it is measuring \
             something other than the configuration write"
        );
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}

/// Where in the debug block do `CFGREG_RD_CNTL` and `CFGREG_RDDATA` actually live?
///
/// Neither tree gives a Blackhole address table. Run with
/// `cargo test -p tt-tests --test probe_cfgreg -- --ignored --nocapture`.
/// Ignored because it is a search, not a claim: it forks once per candidate and
/// prints what it found, and the finding is then pinned by the test above.
///
/// Two passes rather than a cross product: first hold the control register at the
/// Wormhole-sourced `0xFFB1_2058` and look for a data register, then, if nothing
/// answers, walk the control register with the data register one word after it.
#[test]
#[ignore]
fn search_for_the_cfgreg_debug_registers() {
    const VALUE: u32 = 3;

    /// Stage a known `Config` word, then write `cntl` and read `data`.
    /// Exits 42 from the child on a hit so the parent can see it through
    /// `fork_scope`'s status.
    fn probe(cntl: u64, data: u64) -> bool {
        let r = fork_scope(|| {
            let mut sim = Simulator::open().unwrap();
            let mut dev = Device::open(sim.transport()).unwrap();
            let mut words = backend::ConfigWords::new();
            words
                .set(alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt, VALUE)
                .unwrap();
            let mut staged = [sfpu::nop(); 8];
            let n = words.program(SCRATCH_GPR, &mut staged).unwrap();
            let mut p: Vec<Instruction> = staged[..n].to_vec();
            p.extend(control_tail());
            let seen = run_with(&mut dev, &p, |dev, w, tile| {
                dev.write32(
                    w,
                    tile,
                    cntl,
                    alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt.addr32() as u32,
                )
                .unwrap();
                dev.tick(64);
                dev.read32(w, tile, data).unwrap()
            });
            if seen != 0 && alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt.extract(seen) == VALUE {
                println!("  HIT cntl={cntl:#010X} data={data:#010X} word={seen:#010x}");
            }
        });
        r.is_ok()
    }

    println!("pass 1: cntl fixed at 0xFFB1_2058, walking data");
    for data in (0xFFB1_2000u64..0xFFB1_2400).step_by(4) {
        if data != 0xFFB1_2058 {
            probe(0xFFB1_2058, data);
        }
    }

    println!("pass 2: walking cntl, data one word later");
    for cntl in (0xFFB1_2000u64..0xFFB1_2400).step_by(4) {
        probe(cntl, cntl + 4);
    }
}
