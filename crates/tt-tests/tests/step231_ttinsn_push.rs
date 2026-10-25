//! Fused `.ttinsn` push (`Session::set_ttinsn`, on by default, `docs/plans/ttinsn-push.md`): the math role runs
//! its long programs as RISC-V code in the program cache that pushes each Tensix word by executing
//! it (`tt_isa::ttinsn`, `tt_firmware::fused`), instead of loading and storing each word.
//!
//! On eight units, the claims:
//! - **Bit for bit the pushed programs**, for the SFPU kinds the models run and the long
//!   transcendentals, same-shape and broadcast, over values with signed zeros, subnormals, NaNs,
//!   infinities and extremes; and the fused path ran (`session::fused_placements`).
//! - **In a trace,** ops captured with fused programs and replayed over new inputs are the ops
//!   run fresh without them.
//!
//! Negative control: fused code whose `SFPLOADI` immediates are changed after the host's checks
//! (`session::probe_fuse_mutation`) gives other bits, so the card ran the fused code.

use tt_kernels::session::{fused_placements, probe_fuse_mutation, Session, TileChoice};
use tt_kernels::sfpu::ops::kind_sfpu;
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_ttsim::fork_scope;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
            match i % 53 {
                3 => -0.0,
                7 => 0.0,
                13 => f32::from_bits(0x0000_0042),
                19 => -f32::from_bits(0x0070_0000),
                23 => f32::NAN,
                29 => f32::INFINITY,
                31 => f32::NEG_INFINITY,
                37 => f32::MAX,
                41 => -1.0e30,
                _ => x,
            }
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

const ROWS: usize = 64;
const COLS: usize = 96;

fn op(kind: u32) -> Eltwise {
    Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

/// The second operand's shape for a form: same-shape, a row (`[1, COLS]`) or a column
/// (`[ROWS, 1]`).
#[derive(Clone, Copy, Debug)]
enum Form {
    Unary,
    Same,
    Row,
    Col,
}

/// The ops checked: the models' (`SILU_MUL`, the staged `SUB_EXP` and `DIV`, `SIGMOID`, `EXP`)
/// and the long transcendentals (`GELU`, `TANH`, `ERF`, `LOG`, `SQRT`), in their forms.
const CASES: &[(u32, Form)] = &[
    (kind_sfpu::GELU, Form::Unary),
    (kind_sfpu::SIGMOID, Form::Unary),
    (kind_sfpu::TANH, Form::Unary),
    (kind_sfpu::ERF, Form::Unary),
    (kind_sfpu::EXP, Form::Unary),
    (kind_sfpu::LOG, Form::Unary),
    (kind_sfpu::SQRT, Form::Unary),
    (kind_sfpu::SILU, Form::Unary),
    (kind_sfpu::SILU_MUL, Form::Same),
    (kind_sfpu::DIV, Form::Same),
    (kind_sfpu::DIV, Form::Row),
    (kind_sfpu::DIV, Form::Col),
    (kind_sfpu::SUB_EXP, Form::Col),
    (kind_sfpu::SUB_EXP, Form::Row),
];

/// `kind` over fixed operands in `form`, with fused programs on or off: its bits.
fn run<T: tt_device::Transport>(
    s: &mut Session<T>,
    kind: u32,
    form: Form,
    fused: bool,
    seed: u64,
) -> Vec<u32> {
    s.set_ttinsn(fused);
    let a = s.upload(&values(seed, ROWS * COLS), ROWS, COLS).unwrap();
    let b = match form {
        Form::Unary => None,
        Form::Same => Some(
            s.upload(&values(seed + 1, ROWS * COLS), ROWS, COLS)
                .unwrap(),
        ),
        Form::Row => Some(s.upload(&values(seed + 1, COLS), 1, COLS).unwrap()),
        Form::Col => Some(s.upload(&values(seed + 1, ROWS), ROWS, 1).unwrap()),
    };
    let y = s.eltwise(op(kind), &a, b.as_ref()).unwrap();
    let out = bits(&s.download(&y).unwrap());
    for t in [Some(a), b, Some(y)].into_iter().flatten() {
        s.free(t).unwrap();
    }
    s.set_ttinsn(false);
    out
}

#[test]
fn fused_programs_are_the_pushed_programs_bit_for_bit() {
    with_session(|s| {
        for &(kind, form) in CASES {
            let want = run(s, kind, form, false, 3);
            let before = fused_placements();
            let got = run(s, kind, form, true, 3);
            let placed = fused_placements() - before;
            println!("kind {kind:#x} {form:?}: {placed} kernels placed fused");
            assert_eq!(got, want, "kind {kind:#x} {form:?}: fused against pushed");
            if kind == kind_sfpu::GELU {
                assert!(placed > 0, "GELU's programs were not fused");
            }
        }
    });
}

#[test]
fn fused_code_changed_after_its_checks_gives_other_bits() {
    with_session(|s| {
        let want = run(s, kind_sfpu::GELU, Form::Unary, false, 5);
        probe_fuse_mutation(true);
        let before = fused_placements();
        let got = run(s, kind_sfpu::GELU, Form::Unary, true, 5);
        probe_fuse_mutation(false);
        assert!(
            fused_placements() > before,
            "the mutated programs were not placed"
        );
        let differing = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        println!(
            "mutated fused GELU: {differing} of {} outputs differ",
            want.len()
        );
        assert!(differing > 0, "the card did not run the fused code");
        // And the unmutated fused code is the pushed program again.
        assert_eq!(run(s, kind_sfpu::GELU, Form::Unary, true, 5), want);
    });
}

/// `gelu(x)` then `silu(gelu(x)) * x`, captured with fused programs and replayed over new `x`,
/// against the two run fresh without them.
#[test]
fn a_fused_capture_replays_as_the_ops_run_fresh() {
    with_session(|s| {
        let x = s.upload(&values(10, ROWS * COLS), ROWS, COLS).unwrap();
        let step = |s: &mut Session<_>, x: &DramTensor| {
            let g = s.eltwise(op(kind_sfpu::GELU), x, None).unwrap();
            let y = s.eltwise(op(kind_sfpu::SILU_MUL), &g, Some(x)).unwrap();
            s.free(g).unwrap();
            y
        };
        // The fused path is the default: the fresh runs are the pushed programs.
        s.set_ttinsn(false);
        let mut fresh = Vec::new();
        for seed in [10u64, 20, 21] {
            s.write(&x, &values(seed, ROWS * COLS)).unwrap();
            let y = step(s, &x);
            fresh.push(bits(&s.download(&y).unwrap()));
            s.free(y).unwrap();
        }
        s.set_ttinsn(true);
        s.write(&x, &values(10, ROWS * COLS)).unwrap();
        let before = fused_placements();
        s.begin_trace().unwrap();
        let y = step(s, &x);
        let id = s.end_trace().unwrap();
        s.set_ttinsn(false);
        assert!(
            fused_placements() > before,
            "the capture's programs were not fused"
        );
        let mut got = vec![bits(&s.download(&y).unwrap())];
        for seed in [20u64, 21] {
            s.write(&x, &values(seed, ROWS * COLS)).unwrap();
            s.replay(id).unwrap();
            s.sync().unwrap();
            got.push(bits(&s.download(&y).unwrap()));
        }
        assert_eq!(got, fresh, "the fused capture and its replays");
        s.release_trace(id).unwrap();
        s.free(y).unwrap();
        s.free(x).unwrap();
    });
}

/// Two fused programs run one after another from the same address of the program cache (the
/// cache forgotten between them, `Session::probe_forget_programs`): the second is its own program
/// bit for bit. Without the runner's instruction-cache invalidation, silicon would run what its
/// cache held of the first (the `.ttinsn` experiment's stale-code control,
/// `docs/learnings/silicon-operating-notes.md`); ttsim has no instruction cache, so on it this
/// checks only the placement.
#[test]
fn a_fused_program_at_another_s_address_runs_as_itself() {
    use tt_kernels::session::probe_last_fused_at;
    with_session(|s| {
        let want = run(s, kind_sfpu::TANH, Form::Unary, false, 7);
        s.probe_forget_programs().unwrap();
        let gelu = run(s, kind_sfpu::GELU, Form::Unary, true, 7);
        let at = probe_last_fused_at();
        s.probe_forget_programs().unwrap();
        let got = run(s, kind_sfpu::TANH, Form::Unary, true, 7);
        assert_eq!(
            probe_last_fused_at(),
            at,
            "TANH's fused program was not placed where GELU's ran"
        );
        assert_ne!(got, gelu, "TANH computed GELU");
        assert_eq!(got, want, "TANH's fused program at GELU's address");
    });
}
