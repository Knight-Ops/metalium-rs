//! The `.ttinsn` push-fusion experiment (`docs/plans/ttinsn-push-experiment.md`).
//!
//! Three things live here:
//!
//! * the **generator** for `crates/tt-firmware/src/ttinsn_programs.in`, the
//!   programs the experiment image was built with, as fused `.ttinsn` words
//!   (`TT_BLESS=1` rewrites it; without, the test fails if it is stale). The
//!   build-time gate's negative controls (a stray `.word`, a run word with low
//!   bits `11`, a non-Tensix opcode) were run by hand, not by a test here
//!   (`docs/plans/ttinsn-push-experiment.md`, Results, Phase 2);
//! * the **ttsim** decode-or-refuse probe;
//! * the **silicon** rungs and measurement, behind the `ttinsn-experiment`
//!   feature, run only through `cargo xtask silicon` on card 0.
//!
//! The programs are an SFPU kernel body, each preceded by a prelude that fills
//! `Dst` rows 0..64 with a fixed pattern and poisons rows 64..256 with a sentinel (a
//! `Dst` has no reset value and keeps its contents between runs on silicon: without
//! the poison, a push that wrote nothing would compare equal to a correct one).

use std::fmt::Write as _;
use tt_isa::isa::Instruction;
use tt_kernels::code::{Code, Loop};
use tt_kernels::sfpu::kernel::{self, A_ROW};
use tt_kernels::sfpu::ops::{self, kind_sfpu, Broadcast};
use tt_kernels::sfpu::{Format, LReg, LoopPolicy, Program};

/// Where the generated file lives.
const GENERATED: &str = "../tt-firmware/src/ttinsn_programs.in";

/// `0x105`: a short real program that uses `REPLAY`; `kind_sfpu::GELU`: the
/// longest-pushing scalar-free unary kind (Phase 0, `per_kind_push_table`).
const SHORT_KIND: u32 = 0x105;

/// FP32 bit patterns written to the input rows, one per row group.
const PATTERN: [u32; 16] = [
    0x3f80_0000,
    0xbf80_0000,
    0x4000_0000,
    0xc000_0000,
    0x3f00_0000,
    0xbf00_0000,
    0x4040_0000,
    0xc040_0000,
    0x3e80_0000,
    0xbe80_0000,
    0x4080_0000,
    0xc080_0000,
    0x3fc0_0000,
    0xbfc0_0000,
    0x0000_0000,
    0x3dcc_cccd,
];

/// Written over every row the program does not take as input before it runs. A
/// `Dst` has no reset value and keeps its contents between runs on silicon, so
/// without this a push that wrote nothing would leave the *previous* run's
/// results in place and compare equal to a correct one. Raw bits (`Int32`), so a
/// store cannot canonicalise them.
const SENTINEL: u32 = 0xa5a5_a5a5;

/// A store writes four rows of one column parity: at address `a` (a multiple of
/// four) the even columns of rows `a..a + 4`, at `a + 2` their odd columns. Every
/// element a program reads or compares is written with both, because what the
/// other column parity holds is whatever earlier runs left (a kernel's unpacker
/// writes whole rows), and results that depend on it differ from session to
/// session and from ttsim.
fn store_rows(p: &mut Program, r: LReg, fmt: Format, first: u32) {
    p.store(r, fmt, first);
    p.store(r, fmt, first + 2);
}

/// Poison `Dst` rows 64..256 with [`SENTINEL`], then fill rows
/// `A_ROW..A_ROW + 64` with [`PATTERN`], four rows a store pair.
fn prelude() -> Vec<Instruction> {
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    let r = LReg::general(0).unwrap();
    p.loadi_bits(r, SENTINEL);
    for g in 16..64 {
        store_rows(&mut p, r, Format::Int32, 4 * g);
    }
    for (g, bits) in PATTERN.iter().enumerate() {
        p.loadi_bits(r, *bits);
        store_rows(&mut p, r, Format::Fp32, A_ROW + 4 * g as u32);
    }
    p.finish()
}

/// `prelude` then `code`, its block repeats moved past the prelude.
fn with_prelude(code: &Code) -> Code {
    let pre = prelude();
    let shift = pre.len() as u32;
    let mut ins = pre;
    ins.extend_from_slice(&code.ins);
    Code {
        ins,
        loops: code
            .loops
            .iter()
            .map(|l| Loop {
                start: l.start + shift,
                ..*l
            })
            .collect(),
    }
}

fn kind_code(kind: u32) -> Code {
    let (operands, code) = ops::code_for(kind, [0.5, 2.0], Broadcast::None).expect("a kind");
    assert_eq!(format!("{operands:?}"), "Unary", "kind {kind:#x}");
    code
}

/// Every program the tests run: the baked ones, and those only the runner and the
/// translator push.
pub fn all_programs() -> Vec<(String, Code)> {
    let mut v = programs();
    v.push(("short".to_string(), with_prelude(&kind_code(SHORT_KIND))));
    v
}

/// The three role programs of a run of `tiles` tiles of `gelu` (unary), as the
/// SFPU kernel builds them (`kernel::roles_code`), and the layout they were
/// planned in. The math role is static: one tile's semaphore take, program and post
/// inside an outer block repeat of `tiles`.
pub fn kernel_roles(tiles: usize) -> (kernel::Layout, [Vec<Instruction>; 3], [Vec<Loop>; 3]) {
    let layout = kernel::plan_layout(tiles, kernel::Operands::Unary).unwrap();
    let (roles, loops) = kernel::roles_code(
        &layout,
        kernel::Operands::Unary,
        &kind_code(kind_sfpu::GELU),
    );
    (layout, roles, loops)
}

/// Tile counts the kernel-level measurement runs.
pub const KERNEL_TILES: [usize; 2] = [1, 16];

/// The tile counts whose math role is also built into the image (`MODE` 1). One
/// only: the image has to fit the T1 slot's 16 KiB, and the run-time translator
/// (`MODE` 2) covers the rest.
pub const BAKED_KERNEL_TILES: [usize; 1] = [16];

/// The math role of a kernel run of `tiles` tiles, as a program.
pub fn kernel_math(tiles: usize) -> Code {
    let (_, roles, loops) = kernel_roles(tiles);
    Code {
        ins: roles[1].clone(),
        loops: loops[1].clone(),
    }
}

/// The programs built into the image as fused words (`MODE` 1): `(name, program)`.
/// `short` was one until the run-time translator made the image outgrow the T1
/// slot; its baked result is in `performance-log.md`.
pub fn programs() -> Vec<(String, Code)> {
    let nop = tt_isa::sfpu::nop();
    let mut v = vec![
        ("nop1".to_string(), Code::plain(vec![nop])),
        ("nop4".to_string(), Code::plain(vec![nop; 4])),
        (
            "gelu".to_string(),
            with_prelude(&kind_code(kind_sfpu::GELU)),
        ),
    ];
    for tiles in BAKED_KERNEL_TILES {
        v.push((format!("gelu_k{tiles}"), kernel_math(tiles)));
    }
    v
}

/// One run of fused words, bracketed by `ttw_b<N>`/`ttw_e<N>`: the symbols the
/// instruction-set gate (`tt-firmware-images/build.rs`) reads to know which bytes
/// are `.ttinsn` words, so it validates those directly and disassembles
/// everything between them.
fn words(out: &mut String, ins: &[Instruction]) {
    if ins.is_empty() {
        return;
    }
    // Numbered across the whole file by counting what is already in `out`.
    let n = out.matches("\nttw_b").count();
    writeln!(out, "ttw_b{n}:").unwrap();
    // Adjacent `.ttinsn` words are what the instruction cache fuses: up to four
    // per cycle, so a run is kept together and unbroken by anything.
    for i in ins {
        let w = i
            .ttinsn_word()
            .expect("every Tensix word is below 0xC000_0000");
        writeln!(out, "    .word {w:#010x}").unwrap();
    }
    writeln!(out, "ttw_e{n}:").unwrap();
}

/// `ins[lo..hi]` with each block repeat inside it, as `Code::expand` walks it:
/// `li` a counter, the body, `addi`/`bnez`. At most two deep (`code::check`),
/// so `t0` and `t1` are enough, and both are caller-saved in the C ABI the
/// function is called through.
fn emit(out: &mut String, ins: &[Instruction], ls: &[Loop], lo: u32, hi: u32, depth: usize) {
    let mut at = lo;
    loop {
        let next = ls
            .iter()
            .filter(|l| {
                l.start >= at && l.start + l.len <= hi && (l.start, l.start + l.len) != (lo, hi)
            })
            .min_by_key(|l| (l.start, std::cmp::Reverse(l.len)))
            .copied();
        let Some(l) = next else {
            words(out, &ins[at as usize..hi as usize]);
            return;
        };
        words(out, &ins[at as usize..l.start as usize]);
        let reg = ["t0", "t1"][depth];
        let label = depth + 1;
        writeln!(out, "    li {reg}, {}", l.count).unwrap();
        writeln!(out, "{label}:").unwrap();
        emit(out, ins, ls, l.start, l.start + l.len, depth + 1);
        writeln!(out, "    addi {reg}, {reg}, -1").unwrap();
        writeln!(out, "    bnez {reg}, {label}b").unwrap();
        at = l.start + l.len;
    }
}

/// The text of `ttinsn_programs.in`.
pub fn generate() -> String {
    let progs = programs();
    let mut s = String::new();
    s.push_str(
        "// @generated by `TT_BLESS=1 cargo test -p tt-tests --test probe_ttinsn_push \\\n\
         //   ttinsn_programs_are_current`; do not edit.\n\
         //\n\
         // Each program the `.ttinsn` experiment image can push fused\n\
         // (`docs/plans/ttinsn-push-experiment.md`): its staged words (to match\n\
         // against), and the same instructions as `.ttinsn` words in section\n\
         // `.ttinsn`, block repeats as loops. A `.ttinsn` word is the Tensix\n\
         // instruction rotated left two bits; the core rotates it back and stores it\n\
         // to `INSTRN_BUF_BASE`, and adjacent ones fuse (`PushTensixInstruction.md`).\n\n",
    );
    s.push_str("pub struct Program {\n    pub name: &'static str,\n    /// Words in the program, as the runner's `code_len`.\n    pub code_len: u32,\n    /// Block repeats.\n    pub loops: u32,\n    /// The first and last (up to) four staged words, to match against.\n    pub head: &'static [u32],\n    pub tail: &'static [u32],\n    pub run: unsafe extern \"C\" fn(),\n}\n\n");
    s.push_str("core::arch::global_asm!(\n    r#\"\n    .section .ttinsn, \"ax\"\n");
    for (name, code) in &progs {
        writeln!(
            s,
            "    .balign 4\n    .global ttinsn_{name}\nttinsn_{name}:"
        )
        .unwrap();
        emit(&mut s, &code.ins, &code.loops, 0, code.ins.len() as u32, 0);
        s.push_str("    ret\n");
    }
    s.push_str("\"#\n);\n\nextern \"C\" {\n");
    for (name, _) in &progs {
        writeln!(s, "    fn ttinsn_{name}();").unwrap();
    }
    s.push_str("}\n\n");
    for (name, code) in &progs {
        let n = code.ins.len();
        let edge = n.min(4);
        for (what, range) in [("HEAD", 0..edge), ("TAIL", n - edge..n)] {
            let line: Vec<String> = code.ins[range]
                .iter()
                .map(|i| format!("{:#010x}", i.word()))
                .collect();
            writeln!(
                s,
                "static {}_{what}: [u32; {edge}] = [{}];",
                name.to_uppercase(),
                line.join(", ")
            )
            .unwrap();
        }
    }
    s.push('\n');
    // The opcode bytes of every Tensix instruction, one bit each: what the run-time
    // translator checks a word against (the firmware cannot afford the whole table).
    let mut opcodes = [0u32; 8];
    for d in tt_isa::isa::generated::ALL {
        opcodes[(d.opcode() >> 5) as usize] |= 1 << (d.opcode() & 31);
    }
    let words: Vec<String> = opcodes.iter().map(|w| format!("{w:#010x}")).collect();
    writeln!(
        s,
        "pub static OPCODES: [u32; 8] = [{}];\n",
        words.join(", ")
    )
    .unwrap();
    writeln!(s, "pub static PROGRAMS: [Program; {}] = [", progs.len()).unwrap();
    for (name, code) in &progs {
        writeln!(
            s,
            "    Program {{ name: \"{name}\", code_len: {}, loops: {}, head: &{up}_HEAD, tail: &{up}_TAIL, run: ttinsn_{name} }},",
            code.ins.len(),
            code.loops.len(),
            up = name.to_uppercase()
        )
        .unwrap();
    }
    s.push_str("];\n");
    s
}

#[test]
fn ttinsn_programs_are_current() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GENERATED);
    let want = generate();
    if std::env::var_os("TT_BLESS").is_some() {
        std::fs::write(&path, &want).unwrap();
        return;
    }
    let have = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        have == want,
        "{} is stale or missing; regenerate it with `TT_BLESS=1 cargo test -p tt-tests \
         --test probe_ttinsn_push ttinsn_programs_are_current`",
        path.display()
    );
}

/// Every program's block repeats are ones the runner accepts (at most two deep,
/// so two counter registers are enough for the generated loops).
#[test]
fn the_programs_are_runnable_by_the_runner() {
    for (name, code) in all_programs() {
        code.stored()
            .unwrap_or_else(|e| panic!("{name}: the runner would refuse it: {e}"));
    }
}

// ---------------------------------------------------------------------------
// The experiment itself: the same image run two ways, `MODE` 0 (the runner's own
// push) and `MODE` 1 (the matching program pushed as fused `.ttinsn` words).
// ---------------------------------------------------------------------------

#[cfg(feature = "ttinsn-experiment")]
// The silicon measurements are compiled out of a simulator build, and their helpers
// (`median`, the timing fields) with them.
#[cfg_attr(not(feature = "silicon"), allow(dead_code))]
mod experiment {
    use super::*;
    use tt_device::trace::TraceEvent;
    use tt_isa::mailbox::trace as ev;
    use tt_kernels::runtime::{self, Kernel, Schedule};
    use tt_tests::harness::{in_device, tensix_tile, Dev};

    /// `ttinsn_t1.rs`'s scratch block.
    const SCRATCH: u64 = tt_isa::mailbox::TTINSN_SCRATCH;
    const STAMP_VALUE: u32 = 0x7717_0001;

    /// The ttsim tests of this file run one at a time. Run concurrently they hung
    /// intermittently (about one run in five, always inside a forked child; the
    /// cause was not found), and single-threaded or alone they never did.
    #[cfg(not(feature = "silicon"))]
    static SIMULATOR: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[cfg(not(feature = "silicon"))]
    fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
        SIMULATOR.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `MODE`'s values and flags (`ttinsn_t1.rs`).
    pub const BAKED: u32 = 1;
    pub const TRANSLATE: u32 = 2;
    pub const SELFTEST: u32 = 3;
    pub const TIME: u32 = 1 << 8;
    /// Only the simulator tests skip the invalidation (ttsim has no cache).
    #[allow(dead_code)]
    pub const NO_INVALIDATE: u32 = 1 << 9;
    pub const NO_OPCODE_CHECK: u32 = 1 << 10;
    /// Leave instruction gathering on, as the core boots; every image otherwise
    /// turns it off (`tt_firmware::disable_gathering`).
    pub const GATHERING_ON: u32 = 1 << 11;
    /// `cfg0` (CSR `0x7c0`) bits: `DisBp`, and `DisTriscCache` -- gathering is off when it is set.
    pub const CFG0_DIS_BP: u32 = 1 << 1;
    pub const CFG0_DIS_TRISC_CACHE: u32 = 1 << 18;

    pub struct Outcome {
        /// How the image pushed the program: 0 the runner's way, 1 baked fused
        /// words, 2 translated at run time.
        pub fused: u32,
        /// Mode 2: words of fused code written, and the cycles spent translating
        /// and invalidating (with `TIME`, silicon only).
        pub translated: u32,
        pub translate_cycles: u32,
        /// Mode 3: what the invalidation self-test's three calls returned.
        pub selftest: [u32; 3],
        /// T1's `cfg0` as it read it after the program retired.
        pub cfg0: u32,
        /// The checksums of `Dst` rows 0..64, 64..128, 128..192 and 192..256,
        /// taken after the program retired.
        pub bands: [u32; 4],
        /// T1's `START -> PUSHED` and `PUSHED -> RETIRED`, in cycles (silicon,
        /// when traced).
        pub push: u64,
        pub retire: u64,
    }

    /// Run `code` as the math role on the gate tile, `mode` the image's `MODE` word.
    pub fn run_math(dev: &mut Dev<'_>, code: &Code, mode: u32, traced: bool) -> Outcome {
        let images = [
            tt_firmware_images::ROLES[0],
            tt_firmware_images::TTINSN_T1,
            tt_firmware_images::ROLES[2],
        ];
        let mode = mode.to_le_bytes();
        let stage = [(SCRATCH, &mode[..])];
        let read_back = [(SCRATCH, 64usize)];
        let kernel = Kernel {
            stage: &stage,
            read_back: &read_back,
            trace: traced,
            loops: [&[], &code.loops, &[]],
            ..Kernel::new([&[], &code.ins, &[]], Schedule::InOrder)
        };
        let out = runtime::run(dev, tensix_tile(), &images, &kernel, 1 << 30)
            .unwrap_or_else(|e| panic!("{e}"));
        let word = |k: usize| u32::from_le_bytes(out.l1[0][4 * k..4 * k + 4].try_into().unwrap());
        assert_eq!(
            word(2),
            STAMP_VALUE,
            "the experiment image did not run (stale scratch block)"
        );
        let (push, retire) = spans(&out.trace);
        Outcome {
            fused: word(1),
            translated: word(8),
            translate_cycles: word(9),
            selftest: [word(10), word(11), word(12)],
            cfg0: word(13),
            bands: [word(4), word(5), word(6), word(7)],
            push,
            retire,
        }
    }

    /// T1's two spans from a traced run.
    fn spans(trace: &[TraceEvent]) -> (u64, u64) {
        let mut at = [0u64; 3];
        for e in trace {
            let (src, what) = ev::split(e.token);
            if src == 1 && (1..=3).contains(&what) {
                at[what as usize - 1] = e.cycles;
            }
        }
        (at[1].saturating_sub(at[0]), at[2].saturating_sub(at[1]))
    }

    pub fn named(name: &str) -> Code {
        all_programs()
            .into_iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no program {name}"))
            .1
    }

    fn median(mut v: Vec<u64>) -> u64 {
        v.sort();
        v[v.len() / 2]
    }

    /// Phase 3: does ttsim decode a `.ttinsn` word, or refuse it? Either is the
    /// recorded answer. The control (the same image, the runner's own push of the
    /// same program) must run, and the fused run's failure is classified: a
    /// panicking assertion of this test is a bug in the experiment, anything else
    /// (ttsim's `_Exit`, a signal) is the simulator's answer.
    /// Every image turns instruction gathering off, and the experiment's flag is
    /// the only way to leave it on: T1's own read of `cfg0` after a run shows
    /// `DisTriscCache` set by default and clear with `GATHERING_ON`, so a bit that
    /// is always set (or never) cannot pass. T0 and T2 run the shipped role
    /// images, which read the bit back themselves and publish
    /// `GATHERING_STILL_ON` if it did not stick, so the run reaching its result
    /// at all says they are off too.
    fn assert_gathering_follows_the_flag(dev: &mut Dev<'_>) {
        let nop = named("nop1");
        let default = run_math(dev, &nop, 0, false);
        assert_ne!(
            default.cfg0 & CFG0_DIS_TRISC_CACHE,
            0,
            "gathering must be off by default: cfg0 = {:#x}",
            default.cfg0
        );
        assert_eq!(
            default.cfg0 & CFG0_DIS_BP,
            0,
            "the branch predictor is left on, as booted: cfg0 = {:#x}",
            default.cfg0
        );
        let kept = run_math(dev, &nop, GATHERING_ON, false);
        assert_eq!(
            kept.cfg0 & CFG0_DIS_TRISC_CACHE,
            0,
            "control: the flag must leave gathering on: cfg0 = {:#x}",
            kept.cfg0
        );
    }

    #[cfg(not(feature = "silicon"))]
    #[test]
    fn ttsim_gathering_is_off_by_default_and_the_flag_keeps_it_on() {
        let _one = one_at_a_time();
        tt_ttsim::fork_scope(|| {
            let mut sim = tt_ttsim::Simulator::open().unwrap();
            let mut dev = tt_device::Device::open(sim.transport()).unwrap();
            assert_gathering_follows_the_flag(&mut dev);
        })
        .unwrap_or_else(|e| panic!("{e}"));
    }

    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, gathering"]
    fn gathering_is_off_by_default_and_the_flag_keeps_it_on() {
        tt_tests::harness::assert_on_silicon();
        in_device(assert_gathering_follows_the_flag);
    }

    #[cfg(not(feature = "silicon"))]
    #[test]
    fn ttsim_decode_or_refuse() {
        let _one = one_at_a_time();
        use tt_ttsim::{fork_scope, ForkError, Simulator};
        let nop1 = named("nop1");
        let attempt = |mode: u32| {
            fork_scope(|| {
                let mut sim = Simulator::open().unwrap();
                let mut dev = tt_device::Device::open(sim.transport()).unwrap();
                let r = run_math(&mut dev, &nop1, mode, false);
                assert_eq!(r.fused, mode, "the image did not take the path asked for");
            })
        };
        attempt(0).expect("control: the image's own push of the same program must run on ttsim");
        let verdict = match attempt(1) {
            Ok(()) => "RAN: ttsim executed the fused `.ttinsn` NOP".to_string(),
            // 101 is `fork_scope`'s code for a panicking closure.
            Err(ForkError::Exited(101)) => {
                panic!("the experiment's own assertion failed, not a ttsim refusal")
            }
            Err(e) => format!("was REFUSED by ttsim: {e}"),
        };
        println!("MEASURE ttsim_ttinsn: a fused `.ttinsn` NOP {verdict}");
    }

    /// ttsim executes fused `.ttinsn` pushes, and a fused push of each program
    /// leaves `Dst` exactly as the runner's own push of it does, band by band.
    ///
    /// Negative controls, so a pass means something: the checksum tells the two
    /// programs' results apart, and one loop count changed in the runner's own
    /// push changes the result. (The first checksum xor-rotated and was zero for
    /// every program; a comparison that cannot fail proves nothing.)
    #[cfg(not(feature = "silicon"))]
    #[test]
    fn ttsim_fused_output_matches_the_runners() {
        let _one = one_at_a_time();
        use tt_ttsim::{fork_scope, Simulator};
        let bands = |code: &Code, mode: u32| -> [u32; 4] {
            let path = std::env::temp_dir().join(format!(
                "ttinsn_{mode}_{:p}_{}",
                code,
                std::process::id()
            ));
            fork_scope(|| {
                let mut sim = Simulator::open().unwrap();
                let mut dev = tt_device::Device::open(sim.transport()).unwrap();
                let r = run_math(&mut dev, code, mode, false);
                assert_eq!(
                    r.fused,
                    mode & 0xff,
                    "the image did not take the path asked for"
                );
                let bytes: Vec<u8> = r.bands.iter().flat_map(|b| b.to_le_bytes()).collect();
                std::fs::write(&path, bytes).unwrap();
            })
            .unwrap_or_else(|e| panic!("mode {mode}: {e}"));
            let bytes = std::fs::read(&path).unwrap();
            let _ = std::fs::remove_file(&path);
            std::array::from_fn(|k| u32::from_le_bytes(bytes[4 * k..4 * k + 4].try_into().unwrap()))
        };
        let (short, gelu) = (named("short"), named("gelu"));
        let (short_runner, gelu_runner) = (bands(&short, 0), bands(&gelu, 0));
        // `short` is not built into the image; the translator covers it (below).
        assert_eq!(
            bands(&short, TRANSLATE | NO_INVALIDATE),
            short_runner,
            "short: translated differs from the runner's push"
        );
        assert_eq!(
            bands(&gelu, 1),
            gelu_runner,
            "gelu: fused differs from the runner's push"
        );
        assert_ne!(
            short_runner[2], gelu_runner[2],
            "control: the checksum must tell two programs' results apart"
        );
        assert_eq!(
            short_runner[0], gelu_runner[0],
            "the prelude gives both the same input"
        );
        // The row loop run half as often (one fewer is not enough: it steps two
        // rows an iteration over four-row groups, so the last iteration only
        // revisits rows an earlier one did): the runner's own push of it must give
        // a different result, or the comparison above could not have failed.
        let mut fewer = gelu.clone();
        let l = fewer
            .loops
            .iter_mut()
            .find(|l| l.count > 1)
            .expect("gelu has block repeats");
        l.count /= 2;
        assert_ne!(
            bands(&fewer, 0)[2],
            gelu_runner[2],
            "control: half the row-loop iterations must change the result"
        );
        println!(
            "MEASURE ttsim_ttinsn: fused == runner for short and gelu; controls differ; bands short {short_runner:#010x?} gelu {gelu_runner:#010x?}"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 4 and 5: silicon, card 0, tile 0 of the gate (`GATE_TILE`), one test per
    // process, each only through
    //   cargo xtask silicon --release --device 0 --feature ttinsn-experiment \
    //     --include-ignored --filter probe_ttinsn_push::experiment::<test>
    // never run together and never with other silicon work. A hang is a refusal
    // to record (`silicon-operating-notes.md`), not something to retry.
    // -----------------------------------------------------------------------

    /// Rung 1: one fused `.ttinsn` NOP retires, and nothing hangs.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, Phase 4"]
    fn rung1_one_fused_nop() {
        tt_tests::harness::assert_on_silicon();
        in_device(|dev| {
            let r = run_math(dev, &named("nop1"), 1, false);
            assert_eq!(r.fused, 1, "the fused path was not taken");
            println!("MEASURE ttinsn rung 1: one fused NOP retired");
        });
    }

    /// Rung 2: four adjacent fused words, the case the instruction cache fuses
    /// into one push: all four retire, nothing hangs.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, Phase 4"]
    fn rung2_four_fused_nops() {
        tt_tests::harness::assert_on_silicon();
        in_device(|dev| {
            let r = run_math(dev, &named("nop4"), 1, false);
            assert_eq!(r.fused, 1, "the fused path was not taken");
            println!("MEASURE ttinsn rung 2: four adjacent fused NOPs retired");
        });
    }

    /// The hash of a band the program never wrote: 1024 words of [`SENTINEL`], as
    /// the image hashes them (FNV-1a over the words in order).
    fn sentinel_band_hash() -> u32 {
        (0..1024).fold(0x811c_9dc5u32, |h, _| {
            (h ^ SENTINEL).wrapping_mul(0x0100_0193)
        })
    }

    /// The runner's own push and the fused push of `name` leave every `Dst` band
    /// the same, and the controls can tell programs and loop counts apart.
    #[cfg(feature = "silicon")]
    fn matches_runner(name: &str, mode: u32) {
        tt_tests::harness::assert_on_silicon();
        in_device(|dev| {
            let code = named(name);
            let runner = run_math(dev, &code, 0, false);
            let fused = run_math(dev, &code, mode, false);
            assert_eq!((runner.fused, fused.fused), (0, mode & 0xff));
            assert_eq!(
                fused.bands, runner.bands,
                "{name}: the fused push left Dst differently from the runner's"
            );
            let unwritten = sentinel_band_hash();
            assert_ne!(
                runner.bands[2], unwritten,
                "control: the result band is the sentinel's hash, so nothing was computed"
            );
            let mut fewer = code.clone();
            if let Some(l) = fewer.loops.iter_mut().find(|l| l.count > 1) {
                l.count /= 2;
                let changed = run_math(dev, &fewer, 0, false);
                assert_ne!(
                    changed.bands[2], runner.bands[2],
                    "control: half the row-loop iterations must change the result"
                );
            }
            println!(
                "MEASURE ttinsn {name}: fused == runner, bands {:#010x?}",
                fused.bands
            );
        });
    }

    /// Rung 4: the full precise `gelu` (11,495 pushed words), bit-exact.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, Phase 4"]
    fn rung4_gelu_matches() {
        matches_runner("gelu", BAKED);
    }

    /// Phase 5: T1's `START -> PUSHED` and `PUSHED -> RETIRED` for the runner's own
    /// push (variant A) and the fused push (variant C) of the same program in the
    /// same image, ABAB, after warm-ups, the median of `REPS`, with every run's
    /// result checked bit-exact against the other's. The fixed start (about 171-183
    /// cycles) and retire (about 10) costs are in the spans and separate from the push
    /// phase by construction.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, Phase 5"]
    fn measure_push_phase() {
        tt_tests::harness::assert_on_silicon();
        const REPS: usize = 9;
        in_device(|dev| {
            for name in ["gelu"] {
                let code = named(name);
                let words = code.expand().len();
                for _ in 0..3 {
                    run_math(dev, &code, 0, true);
                    run_math(dev, &code, 1, true);
                }
                let (mut a_push, mut a_ret, mut c_push, mut c_ret) =
                    (vec![], vec![], vec![], vec![]);
                for _ in 0..REPS {
                    let a = run_math(dev, &code, 0, true);
                    let c = run_math(dev, &code, 1, true);
                    assert_eq!((a.fused, c.fused), (0, 1));
                    assert_eq!(
                        a.bands, c.bands,
                        "{name}: fused result differs from the runner's"
                    );
                    a_push.push(a.push);
                    a_ret.push(a.retire);
                    c_push.push(c.push);
                    c_ret.push(c.retire);
                }
                let (ap, ar, cp, cr) =
                    (median(a_push), median(a_ret), median(c_push), median(c_ret));
                println!(
                    "MEASURE ttinsn {name} ({words} pushed words): runner push {ap} cycles ({:.2}/word) retire +{ar}; \
                     fused push {cp} cycles ({:.2}/word) retire +{cr}; push {:.2}x, push+retire {:.2}x",
                    ap as f64 / words as f64,
                    cp as f64 / words as f64,
                    ap as f64 / cp.max(1) as f64,
                    (ap + ar) as f64 / (cp + cr).max(1) as f64,
                );
            }
        });
    }

    // -----------------------------------------------------------------------
    // The kernel level: the three real roles of an SFPU kernel over N tiles of
    // `gelu` (`kernel::roles_code`), the math role pushed by the runner or fused.
    // -----------------------------------------------------------------------

    use tt_isa::tile::{L1Format, TileImage};
    use tt_kernels::datapath;
    use tt_kernels::sfpu::kernel::OUT_ROW;

    /// Poison the result rows, so a kernel whose math push did nothing packs the
    /// sentinel rather than the previous run's results (`SENTINEL`).
    fn poison_out() -> Vec<Instruction> {
        let mut p = Program::with_policy(LoopPolicy::Unrolled);
        let r = LReg::general(0).unwrap();
        p.loadi_bits(r, SENTINEL);
        for g in 0..16 {
            store_rows(&mut p, r, Format::Int32, OUT_ROW + 4 * g);
        }
        p.finish()
    }

    /// The translation fields are read by the simulator tests' printing only.
    #[allow(dead_code)]
    pub struct KernelOutcome {
        pub fused: u32,
        pub translated: u32,
        pub translate_cycles: u32,
        /// Every output slot's bytes.
        pub out: Vec<u8>,
        /// T1's `START -> PUSHED`.
        pub push: u64,
        /// First role `START` to last role `RETIRED`.
        pub kernel: u64,
    }

    /// Run `math` (a role-1 program built for `tiles` tiles) in a kernel over
    /// `tiles` tiles, after a run that poisons the result rows.
    pub fn run_kernel(
        dev: &mut Dev<'_>,
        tiles: usize,
        math: &Code,
        mode: u32,
        traced: bool,
    ) -> KernelOutcome {
        let images = [
            tt_firmware_images::ROLES[0],
            tt_firmware_images::TTINSN_T1,
            tt_firmware_images::ROLES[2],
        ];
        let tile = tensix_tile();
        let mode_bytes = 0u32.to_le_bytes();
        let poison = poison_out();
        let stage = [(SCRATCH, &mode_bytes[..])];
        let setup = Kernel {
            stage: &stage,
            ..Kernel::new([&[], &poison, &[]], Schedule::InOrder)
        };
        runtime::run(dev, tile, &images, &setup, 1 << 30).unwrap_or_else(|e| panic!("{e}"));

        let (layout, roles, loops) = kernel_roles(tiles);
        let image = TileImage::new(datapath::tile_descriptor(), L1Format::Fp32).unwrap();
        let inputs: Vec<Vec<u8>> = (0..tiles)
            .map(|n| {
                let mut bytes = vec![0u8; image.total_bytes()];
                for i in 0..1024usize {
                    let v = ((i * 37 + n * 101) % 1000) as f32 / 125.0 - 4.0;
                    let at = image.datum_bit_offset(i) / 8;
                    bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
                }
                bytes
            })
            .collect();
        let mode = mode.to_le_bytes();
        let mut stage: Vec<(u64, &[u8])> = vec![(SCRATCH, &mode[..])];
        for (n, bytes) in inputs.iter().enumerate() {
            stage.push((layout.a_at + n as u64 * tt_isa::dm::TILE_SLOT, bytes));
        }
        let out_len = tiles * tt_isa::dm::TILE_SLOT as usize;
        let read_back = [(SCRATCH, 64usize), (layout.out_at, out_len)];
        let kernel = Kernel {
            stage: &stage,
            read_back: &read_back,
            trace: traced,
            loops: [&loops[0], &math.loops, &loops[2]],
            ..Kernel::new(
                [&roles[0], &math.ins, &roles[2]],
                Schedule::Concurrent(&layout.init),
            )
        };
        let out =
            runtime::run(dev, tile, &images, &kernel, 1 << 30).unwrap_or_else(|e| panic!("{e}"));
        let word = |k: usize| u32::from_le_bytes(out.l1[0][4 * k..4 * k + 4].try_into().unwrap());
        assert_eq!(word(2), STAMP_VALUE, "the experiment image did not run");
        let (push, _) = spans(&out.trace);
        let starts = out
            .trace
            .iter()
            .filter(|e| ev::split(e.token).1 == 1)
            .map(|e| e.cycles);
        let ends = out
            .trace
            .iter()
            .filter(|e| ev::split(e.token).1 == 3)
            .map(|e| e.cycles);
        let kernel_cycles = ends
            .max()
            .unwrap_or(0)
            .saturating_sub(starts.min().unwrap_or(0));
        KernelOutcome {
            fused: word(1),
            translated: word(8),
            translate_cycles: word(9),
            out: out.l1[1].clone(),
            push,
            kernel: kernel_cycles,
        }
    }

    /// The fused math push gives the kernel the same output bytes as the runner's,
    /// with the same controls: a math program with half its row-loop iterations
    /// must give different bytes, and the output is not the sentinel.
    pub fn kernel_matches(dev: &mut Dev<'_>, tiles: usize, mode: u32) {
        let math = kernel_math(tiles);
        let runner = run_kernel(dev, tiles, &math, 0, false);
        let fused = run_kernel(dev, tiles, &math, mode, false);
        assert_eq!((runner.fused, fused.fused), (0, mode & 0xff));
        assert_eq!(
            fused.out, runner.out,
            "{tiles} tiles: the kernel's output differs under the fused push"
        );
        let sentinel = SENTINEL.to_le_bytes();
        assert!(
            !runner.out.chunks(4).all(|w| w == sentinel),
            "control: the output is all sentinel, so nothing was computed"
        );
        let mut fewer = math.clone();
        // The row loop is the inner one (the outer repeats the tile).
        let inner = fewer
            .loops
            .iter_mut()
            .filter(|l| l.len < 1000)
            .max_by_key(|l| l.count)
            .expect("an inner row loop");
        inner.count /= 2;
        let changed = run_kernel(dev, tiles, &fewer, 0, false);
        assert_ne!(
            changed.out, runner.out,
            "control: half the row-loop iterations must change the output"
        );
    }

    /// ttsim: the kernel-level result under the fused push equals the runner's.
    #[cfg(not(feature = "silicon"))]
    #[test]
    fn ttsim_kernel_level_matches() {
        let _one = one_at_a_time();
        for tiles in BAKED_KERNEL_TILES {
            in_device(|dev| kernel_matches(dev, tiles, BAKED));
        }
        println!(
            "MEASURE ttsim_ttinsn: kernel-level fused == runner for {BAKED_KERNEL_TILES:?} tiles"
        );
    }

    /// Phase 5, second part: the kernel level on silicon, ABAB, the median of
    /// `REPS` after warm-ups, every run's output compared byte for byte.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, Phase 5"]
    fn measure_kernel_level() {
        tt_tests::harness::assert_on_silicon();
        const REPS: usize = 9;
        in_device(|dev| {
            for tiles in BAKED_KERNEL_TILES {
                let math = kernel_math(tiles);
                kernel_matches(dev, tiles, BAKED);
                for _ in 0..2 {
                    run_kernel(dev, tiles, &math, 0, true);
                    run_kernel(dev, tiles, &math, 1, true);
                }
                let (mut ap, mut cp, mut ak, mut ck) = (vec![], vec![], vec![], vec![]);
                for _ in 0..REPS {
                    let a = run_kernel(dev, tiles, &math, 0, true);
                    let c = run_kernel(dev, tiles, &math, 1, true);
                    assert_eq!(a.out, c.out, "{tiles} tiles: fused output differs");
                    ap.push(a.push);
                    cp.push(c.push);
                    ak.push(a.kernel);
                    ck.push(c.kernel);
                }
                let (ap, cp, ak, ck) = (median(ap), median(cp), median(ak), median(ck));
                println!(
                    "MEASURE ttinsn kernel {tiles} tiles gelu: T1 push runner {ap} fused {cp} cycles ({:.2}x); \
                     whole kernel runner {ak} fused {ck} cycles ({:.2}x), {:.0} -> {:.0} cycles/tile",
                    ap as f64 / cp.max(1) as f64,
                    ak as f64 / ck.max(1) as f64,
                    ak as f64 / tiles as f64,
                    ck as f64 / tiles as f64,
                );
            }
        });
    }

    // -----------------------------------------------------------------------
    // The run-time translator (`MODE` 2): the runner turns the staged program
    // into fused `.ttinsn` code in L1, invalidates T1's instruction cache, and
    // calls it. No program is built into the image for this.
    // -----------------------------------------------------------------------

    /// ttsim: the translated code, run by ttsim's core, gives the runner's result
    /// for every program, with the invalidation skipped (ttsim has no instruction
    /// cache). This checks the emitted RISC-V (counters, loops, branch offsets,
    /// return) and the translation of every word.
    #[cfg(not(feature = "silicon"))]
    #[test]
    fn ttsim_translator_matches() {
        let _one = one_at_a_time();
        for name in ["nop4", "short", "gelu"] {
            in_device(|dev| {
                let code = named(name);
                let runner = run_math(dev, &code, 0, false);
                let t = run_math(dev, &code, TRANSLATE | NO_INVALIDATE, false);
                assert_eq!((runner.fused, t.fused), (0, 2), "{name}: not translated");
                assert!(
                    t.translated as usize > code.ins.len().min(4),
                    "{name}: {}",
                    t.translated
                );
                assert_eq!(
                    t.bands, runner.bands,
                    "{name}: translated differs from the runner's"
                );
            });
        }
        for tiles in KERNEL_TILES {
            in_device(|dev| kernel_matches(dev, tiles, TRANSLATE | NO_INVALIDATE));
        }
        println!("MEASURE ttsim_ttinsn: translated == runner for the programs and {KERNEL_TILES:?} tiles");
    }

    /// ttsim and the invalidate register: does a write of the mask run, or is it
    /// refused? (Divergence log: ttsim is not known to model it.) Classified as
    /// the decode probe is: a panic in the child is this experiment's own
    /// assertion, anything else is the simulator's answer.
    #[cfg(not(feature = "silicon"))]
    #[test]
    fn ttsim_icache_invalidate_register() {
        let _one = one_at_a_time();
        use tt_ttsim::{fork_scope, ForkError, Simulator};
        let nop1 = named("nop1");
        let verdict = match fork_scope(|| {
            let mut sim = Simulator::open().unwrap();
            let mut dev = tt_device::Device::open(sim.transport()).unwrap();
            let r = run_math(&mut dev, &nop1, SELFTEST, false);
            assert_eq!(r.selftest[0], 1);
            assert_eq!(r.selftest[2], 2);
        }) {
            Ok(()) => "RAN".to_string(),
            Err(ForkError::Exited(101)) => panic!("the experiment's own assertion failed"),
            Err(e) => format!("was REFUSED: {e}"),
        };
        println!("MEASURE ttsim_ttinsn: a write to RISCV_IC_INVALIDATE {verdict}");
    }

    /// Rung 0: the mechanism, with no `.ttinsn` at all. Code `a0 = 1; ret` is
    /// called, rewritten to `a0 = 2`, called with no invalidation (the stale
    /// control: 1 if the cache held the old code), then invalidated and called
    /// (must return 2).
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, run-time translator"]
    fn rung0_icache_invalidation() {
        tt_tests::harness::assert_on_silicon();
        in_device(|dev| {
            let r = run_math(dev, &named("nop1"), SELFTEST, false);
            let [first, stale, fresh] = r.selftest;
            println!("MEASURE ttinsn rung 0: first call {first}, rewritten without invalidation {stale}, after invalidation {fresh}");
            assert_eq!(
                first, 1,
                "the buffer's first call did not return its code's value"
            );
            assert_eq!(
                fresh, 2,
                "after the invalidation the call must run the rewritten code"
            );
        });
    }

    /// Rung 5: translate four NOPs and run them.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, run-time translator"]
    fn rung5_translate_nops() {
        tt_tests::harness::assert_on_silicon();
        in_device(|dev| {
            let r = run_math(dev, &named("nop4"), TRANSLATE | TIME, false);
            assert_eq!(r.fused, 2, "the translated path was not taken");
            println!(
                "MEASURE ttinsn rung 5: four NOPs translated to {} words in {} cycles and retired",
                r.translated, r.translate_cycles
            );
        });
    }

    /// Rung 6: a short real program, translated, bit-exact.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, run-time translator"]
    fn rung6_translated_short_program_matches() {
        matches_runner("short", TRANSLATE | TIME);
    }

    /// Rung 7: the full precise `gelu`, translated, bit-exact.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, run-time translator"]
    fn rung7_translated_gelu_matches() {
        matches_runner("gelu", TRANSLATE | TIME);
    }

    /// Rung 8 and the measurement: every way of pushing, ABAB, median of `REPS`.
    /// The translated push's span includes the translation and the invalidation
    /// (the runner's `START` is before them), which is what the kernel pays.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, run-time translator"]
    fn measure_translator() {
        tt_tests::harness::assert_on_silicon();
        const REPS: usize = 9;
        in_device(|dev| {
            for name in ["short", "gelu"] {
                let code = named(name);
                let words = code.expand().len();
                let baked = programs().iter().any(|(n, _)| n == name);
                let mut modes = vec![
                    ("runner", 0),
                    ("translated", TRANSLATE | TIME),
                    (
                        "translated-no-opcode-check",
                        TRANSLATE | TIME | NO_OPCODE_CHECK,
                    ),
                ];
                if baked {
                    modes.insert(1, ("baked", BAKED));
                }
                for _ in 0..3 {
                    for (_, m) in &modes {
                        run_math(dev, &code, *m, true);
                    }
                }
                let mut push = vec![vec![]; modes.len()];
                let mut translate: Vec<Vec<u64>> = vec![vec![]; modes.len()];
                let mut bands = None;
                for _ in 0..REPS {
                    for (k, (_, m)) in modes.iter().enumerate() {
                        let r = run_math(dev, &code, *m, true);
                        assert_eq!(
                            *bands.get_or_insert(r.bands),
                            r.bands,
                            "{name}: modes differ"
                        );
                        push[k].push(r.push);
                        translate[k].push(r.translate_cycles as u64);
                    }
                }
                let med: Vec<u64> = push.into_iter().map(median).collect();
                let line: Vec<String> = modes
                    .iter()
                    .zip(&med)
                    .zip(translate)
                    .map(|(((n, m), c), t)| {
                        if m & 0xff == TRANSLATE {
                            format!("{n} {c} (translating {})", median(t))
                        } else {
                            format!("{n} {c}")
                        }
                    })
                    .collect();
                println!(
                    "MEASURE ttinsn translator {name} ({words} pushed words, {} stored): push cycles: {}",
                    code.ins.len(),
                    line.join(", ")
                );
            }
            for tiles in KERNEL_TILES {
                let math = kernel_math(tiles);
                kernel_matches(dev, tiles, TRANSLATE | TIME);
                let modes: Vec<(&str, u32)> = if BAKED_KERNEL_TILES.contains(&tiles) {
                    vec![
                        ("runner", 0),
                        ("baked", BAKED),
                        ("translated", TRANSLATE | TIME),
                    ]
                } else {
                    vec![("runner", 0), ("translated", TRANSLATE | TIME)]
                };
                for _ in 0..2 {
                    for (_, m) in &modes {
                        run_kernel(dev, tiles, &math, *m, true);
                    }
                }
                let mut kernel = vec![vec![]; modes.len()];
                let mut outs: Option<Vec<u8>> = None;
                for _ in 0..REPS {
                    for (k, (_, m)) in modes.iter().enumerate() {
                        let r = run_kernel(dev, tiles, &math, *m, true);
                        assert_eq!(
                            outs.get_or_insert_with(|| r.out.clone()),
                            &r.out,
                            "{tiles} tiles: modes differ"
                        );
                        kernel[k].push(r.kernel);
                    }
                }
                let line: Vec<String> = modes
                    .iter()
                    .zip(kernel)
                    .map(|((n, _), v)| format!("{n} {}", median(v)))
                    .collect();
                println!(
                    "MEASURE ttinsn translator kernel {tiles} tiles: whole kernel cycles: {}",
                    line.join(", ")
                );
            }
        });
    }

    // -----------------------------------------------------------------------
    // Instruction gathering. tt-metal's firmware disables it on every RISC
    // around Tensix (`configure_gathering`, tt-metal#16439: a `REPLAY` record
    // window can be corrupted when the instructions it records are gathered);
    // Blackhole boots with it enabled, and this repo's firmware never touched it,
    // so everything above ran with it ON.
    // -----------------------------------------------------------------------

    /// How many `REPLAY` instructions each program holds (the records the hazard
    /// is about), counted from the instruction words.
    #[test]
    fn replay_records_in_the_programs() {
        for (name, code) in all_programs() {
            let n = code
                .ins
                .iter()
                .filter(|i| i.def().mnemonic() == "REPLAY")
                .count();
            println!(
                "MEASURE ttinsn replays {name}: {n} REPLAY of {} words",
                code.ins.len()
            );
        }
        let (_, roles, _) = kernel_roles(16);
        let n = roles[1]
            .iter()
            .filter(|i| i.def().mnemonic() == "REPLAY")
            .count();
        println!(
            "MEASURE ttinsn replays gelu_k16 math role: {n} REPLAY of {} words",
            roles[1].len()
        );
    }

    /// Gathering on (the boot default) against off (tt-metal's configuration), for
    /// the runner's push, the baked fused push and the translated push: every way
    /// must still give the same result, and the cycles say whether the fused push's
    /// speed depends on gathering at all.
    #[cfg(feature = "silicon")]
    #[test]
    #[ignore = "experiment: docs/plans/ttinsn-push-experiment.md, gathering"]
    fn measure_gathering() {
        tt_tests::harness::assert_on_silicon();
        const REPS: usize = 9;
        in_device(|dev| {
            for name in ["gelu", "short"] {
                let code = named(name);
                let baked = programs().iter().any(|(n, _)| n == name);
                let mut modes: Vec<(String, u32)> = vec![];
                for (g, flag) in [("gathering on", GATHERING_ON), ("gathering off", 0)] {
                    modes.push((format!("runner, {g}"), flag));
                    if baked {
                        modes.push((format!("baked, {g}"), BAKED | flag));
                    }
                    modes.push((
                        format!("translated, {g}"),
                        TRANSLATE | NO_OPCODE_CHECK | TIME | flag,
                    ));
                }
                for _ in 0..3 {
                    for (_, m) in &modes {
                        run_math(dev, &code, *m, true);
                    }
                }
                let mut push = vec![vec![]; modes.len()];
                let mut bands = None;
                for _ in 0..REPS {
                    for (k, (_, m)) in modes.iter().enumerate() {
                        let r = run_math(dev, &code, *m, true);
                        assert_eq!(
                            *bands.get_or_insert(r.bands),
                            r.bands,
                            "{name}: modes differ"
                        );
                        push[k].push(r.push);
                    }
                }
                let line: Vec<String> = modes
                    .iter()
                    .zip(push)
                    .map(|((n, _), v)| format!("{n} {}", median(v)))
                    .collect();
                println!(
                    "MEASURE ttinsn gathering {name}: push cycles: {}",
                    line.join("; ")
                );
            }
            let math = kernel_math(16);
            let modes = [
                ("runner, gathering on", GATHERING_ON),
                ("runner, gathering off", 0),
                ("baked, gathering on", BAKED | GATHERING_ON),
                ("baked, gathering off", BAKED),
            ];
            for _ in 0..2 {
                for (_, m) in &modes {
                    run_kernel(dev, 16, &math, *m, true);
                }
            }
            let mut kernel = vec![vec![]; modes.len()];
            let mut outs: Option<Vec<u8>> = None;
            for _ in 0..REPS {
                for (k, (_, m)) in modes.iter().enumerate() {
                    let r = run_kernel(dev, 16, &math, *m, true);
                    assert_eq!(
                        outs.get_or_insert_with(|| r.out.clone()),
                        &r.out,
                        "16 tiles: modes differ"
                    );
                    kernel[k].push(r.kernel);
                }
            }
            let line: Vec<String> = modes
                .iter()
                .zip(kernel)
                .map(|((n, _), v)| format!("{n} {}", median(v)))
                .collect();
            println!(
                "MEASURE ttinsn gathering kernel 16 tiles: whole kernel cycles: {}",
                line.join("; ")
            );
        });
    }
}
