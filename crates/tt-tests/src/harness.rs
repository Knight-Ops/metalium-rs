//! Running a host-built Tensix program through the `corpus` firmware.
//!
//! Every gate from step 4 onwards has the same shape: stage some bytes in L1,
//! stage an instruction stream, start a core, wait for it to report `DONE`, then
//! read `Dst` and L1 back. This is that shape, once, so a new gate is a program
//! and a set of assertions rather than another copy of the staging sequence.
//!
//! # Why everything runs inside a fork
//!
//! ttsim `_Exit`s the process on any contract violation, so a refusal would
//! otherwise take the test runner with it. [`in_device`] forks first, which turns
//! a refusal into a failing assertion, and [`survives`] turns it into a *value* —
//! which is what makes the probes in this crate discovery tools rather than just
//! tests.

use tt_device::core_control::WaitError;
use tt_device::tlb::WindowKind;
use tt_device::{Device, Window};
use tt_isa::isa::Instruction;
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{grid, Noc0, NocCoord};
use tt_isa::tensix::Core;
use tt_ttsim::{fork_scope, Simulator};

/// A device backed by the simulator, for the lifetime of one `fork_scope`.
pub type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

/// T1, not T0: ttsim implements the RISC-V view of `Dst` only for `pipe == 1`
/// (`docs/ttsim-divergence.md` row 12). A simulator constraint, not a hardware one.
pub const CORE: Core = Core::T1;
/// The Tensix thread `CORE` pushes to, and the value the firmware cross-checks
/// against `mailbox::THREAD_INDEX`.
pub const CORE_THREAD: u32 = 1;
/// `RISC_DEST_ACCESS_CTRL_SEC*.fmt` 0 is `float Dst32b[512][16]`.
pub const DST_FMT_FP32: u32 = 0;
/// Simulator cycles a run may take before it is called hung.
pub const BUDGET: u64 = 400_000;
/// Pre-filled into every dumped `Dst` datum, so a datum the firmware never wrote
/// is distinguishable from a computed zero.
pub const SENTINEL: u32 = 0xDEAD_BEEF;

/// The Tensix tile the gates use.
pub fn tensix_tile() -> NocCoord<Noc0> {
    assert!(
        grid::is_tensix(3, 4),
        "the gates' tile must be a Tensix tile"
    );
    NocCoord::new(3, 4).unwrap()
}

/// What to stage before a run and what to read after it.
pub struct Run<'a> {
    /// The instruction stream, pushed verbatim by the firmware.
    pub program: &'a [Instruction],
    /// Bytes to write into L1 first, as `(address, data)`.
    pub stage: &'a [(u64, &'a [u8])],
    /// How many `Dst` rows the firmware should copy back.
    pub dump_rows: u32,
    /// L1 ranges to read after the run, as `(address, length)`.
    pub read_back: &'a [(u64, usize)],
    /// `RISC_DEST_ACCESS_CTRL_SEC*.fmt`.
    pub dst_fmt: u32,
}

impl<'a> Run<'a> {
    /// A run that stages nothing and reads back only `Dst`.
    pub fn new(program: &'a [Instruction]) -> Self {
        Run {
            program,
            stage: &[],
            dump_rows: 4,
            read_back: &[],
            dst_fmt: DST_FMT_FP32,
        }
    }

    pub fn stage(mut self, stage: &'a [(u64, &'a [u8])]) -> Self {
        self.stage = stage;
        self
    }

    pub fn dump_rows(mut self, rows: u32) -> Self {
        self.dump_rows = rows;
        self
    }

    pub fn read_back(mut self, ranges: &'a [(u64, usize)]) -> Self {
        self.read_back = ranges;
        self
    }
}

/// What a run produced.
pub struct Outcome {
    /// `Dst`, flattened as `row * mailbox::DUMP_ROW_WORDS + column`.
    pub dst: Vec<u32>,
    /// One buffer per [`Run::read_back`] range, in order.
    pub l1: Vec<Vec<u8>>,
}

impl Outcome {
    /// `Dst[row][column]`, by the firmware's dump indexing.
    pub fn dst_at(&self, row: usize, column: usize) -> u32 {
        self.dst[row * mailbox::DUMP_ROW_WORDS as usize + column]
    }

    /// Every `Dst` datum the firmware wrote that is neither zero nor the sentinel,
    /// as `(flat index, value)`.
    pub fn dst_nonzero(&self) -> Vec<(usize, u32)> {
        self.dst
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0 && v != SENTINEL)
            .map(|(i, &v)| (i, v))
            .collect()
    }
}

/// Stage, run, and read back.
///
/// Panics rather than returning an error: every failure here is a broken gate, not
/// a condition a caller could handle.
pub fn run(dev: &mut Dev<'_>, spec: &Run<'_>) -> Outcome {
    assert!(
        spec.program.len() as u32 <= mailbox::PROGRAM_MAX,
        "program is {} instructions; the mailbox holds {}",
        spec.program.len(),
        mailbox::PROGRAM_MAX
    );
    assert!(spec.dump_rows <= mailbox::DUMP_MAX_ROWS);

    let tile = tensix_tile();
    let w: Window = dev.alloc_window(WindowKind::TwoMib).unwrap();

    // The backend has to be out of reset before the coprocessor executes anything,
    // and before the core can start pushing into it.
    dev.release_tensix_backend(&w, tile).unwrap();

    for (addr, data) in spec.stage {
        dev.write(&w, tile, *addr, data).unwrap();
    }

    dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
    dev.write32(&w, tile, mailbox::THREAD_INDEX, CORE_THREAD)
        .unwrap();
    dev.write32(&w, tile, mailbox::DST_ACCESS_FMT, spec.dst_fmt)
        .unwrap();
    dev.write32(&w, tile, mailbox::PROGRAM_LEN, spec.program.len() as u32)
        .unwrap();
    dev.write32(&w, tile, mailbox::DUMP_ROW_FIRST, 0).unwrap();
    dev.write32(&w, tile, mailbox::DUMP_ROW_COUNT, spec.dump_rows)
        .unwrap();
    for (i, insn) in spec.program.iter().enumerate() {
        dev.write32(&w, tile, mailbox::PROGRAM + (i as u64) * 4, insn.word())
            .unwrap();
    }
    for row in 0..spec.dump_rows {
        for col in 0..mailbox::DUMP_ROW_WORDS {
            dev.write32(&w, tile, mailbox::dump_offset(row, col), SENTINEL)
                .unwrap();
        }
    }

    dev.load_and_start(
        &w,
        tile,
        CORE,
        crate::firmware::CORPUS,
        crate::firmware::LOAD_ADDRESS,
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

    let mut dst = Vec::new();
    for row in 0..spec.dump_rows {
        for col in 0..mailbox::DUMP_ROW_WORDS {
            dst.push(
                dev.read32(&w, tile, mailbox::dump_offset(row, col))
                    .unwrap(),
            );
        }
    }
    let mut l1 = Vec::new();
    for (addr, len) in spec.read_back {
        let mut buf = vec![0u8; *len];
        dev.read(&w, tile, *addr, &mut buf).unwrap();
        l1.push(buf);
    }
    Outcome { dst, l1 }
}

/// Run `f` against a fresh simulator, inside a fork.
///
/// A fresh simulator per call is not just isolation from `_Exit`: `Dst` has no
/// power-on reset value and nothing scrubs it (`Dst.md:15`), so two runs sharing a
/// simulator leave the second reading the first's leftovers.
#[track_caller]
pub fn in_device(f: impl FnOnce(&mut Dev<'_>)) {
    let result = fork_scope(|| {
        let mut sim = Simulator::open().unwrap_or_else(|e| panic!("could not open simulator: {e}"));
        let mut dev = Device::open(sim.transport()).unwrap_or_else(|e| panic!("{e}"));
        f(&mut dev);
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}

/// Did `f` run to completion, or did ttsim refuse something in it?
///
/// The discovery primitive: a refusal becomes `false` rather than a dead runner, so
/// a probe can assert that the simulator declines a configuration *and* that a
/// control of the same shape survives.
pub fn survives(f: impl FnOnce(&mut Dev<'_>)) -> bool {
    fork_scope(|| {
        let mut sim = Simulator::open().unwrap();
        let mut dev = Device::open(sim.transport()).unwrap();
        f(&mut dev);
    })
    .is_ok()
}
