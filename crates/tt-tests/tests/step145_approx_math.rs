//! Lane T8 gate (S10): `MathMode::{Precise, Approx}` on the device.
//!
//! The Approx programs (`tt_kernels::sfpu::approx`) are approximations with
//! their own derived bounds, so held twice, as `step29_exp_log` holds the
//! Precise ones: the device **bit for bit** to its program run by the
//! interpreter, and the program to an **f64 oracle** (`libm`, independent of
//! the programs) within the bound derived beside it. Every special value class
//! is exact where the contract says so. The mode is the session's
//! (`Session::set_math_mode`, `TT_MATH=approx`); Precise stays the default
//! and bit-identical to today.
//!
//! Negative controls (watched to fail, recorded in the lane report):
//! * `Approx` must break the Precise bound somewhere (`approx_breaks_...`:
//!   mutate `MathMode::lower` to return the kind unchanged);
//! * alternating the modes in one session must give different bits (mutate the
//!   mode out of the kind: `Eltwise::in_mode` returning `self`);
//! * a mutated polynomial coefficient fails the sweep (`EXP_COEFFS[3]`);
//! * a padding claim that is false is caught (`pad_is_honest` on an `exp`).

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::approx::{
    self, gelu_approx_bound, kind, MathMode, EXP_APPROX_BOUND, LOG_APPROX_BOUND,
    RECIP_APPROX_BOUND, RECIP_RANGE_BITS, RECIP_SEED_BOUND, SIGMOID_APPROX_BOUND,
    TANH_APPROX_BOUND,
};
use tt_kernels::sfpu::ops::{kind_sfpu, reference};
use tt_kernels::tensor::{DramTensor, Eltwise, Pad};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn open() -> Session<tt_ttsim::LibTtsim<'static>> {
    // The simulator outlives the closure that opened it: leaked, as the fork
    // that holds it is short-lived (one process per gate).
    let sim = Box::leak(Box::new(tt_ttsim::Simulator::open().unwrap()));
    let dev = tt_device::Device::open(sim.transport()).unwrap();
    Session::open(
        dev,
        tt_firmware_images::ROLES,
        TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        |_, _| Ok(None),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

#[cfg(feature = "silicon")]
fn open() -> Session<tt_kmd::Kmd> {
    Session::open_card(
        tt_tests::backend::device_index(),
        tt_firmware_images::ROLES,
        TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// A session with GDDR on, in a fork (ttsim is a once-per-process singleton).
fn with_session<T: tt_device::Transport>(
    open: impl FnOnce() -> Session<T>,
    f: impl FnOnce(&mut Session<T>),
) {
    if let Err(e) = fork_scope(|| {
        let mut s = open();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// One op with an Approx twin.
struct Op {
    name: &'static str,
    precise: u32,
    approx: u32,
    /// Inputs of every kind the op's range holds, then its specials.
    range: (f64, f64),
    positive: bool,
}

const OPS: [Op; 6] = [
    Op {
        name: "exp",
        precise: kind_sfpu::EXP,
        approx: kind::EXP,
        range: (-90.0, 90.0),
        positive: false,
    },
    Op {
        name: "log",
        precise: kind_sfpu::LOG,
        approx: kind::LOG,
        range: (0.0, 0.0),
        positive: true,
    },
    Op {
        name: "recip",
        precise: kind_sfpu::RECIP,
        approx: kind::RECIP,
        range: (0.0, 0.0),
        positive: true,
    },
    Op {
        name: "sigmoid",
        precise: kind_sfpu::SIGMOID,
        approx: kind::SIGMOID,
        range: (-90.0, 90.0),
        positive: false,
    },
    Op {
        name: "tanh",
        precise: kind_sfpu::TANH,
        approx: kind::TANH,
        range: (-10.0, 10.0),
        positive: false,
    },
    Op {
        name: "gelu",
        precise: kind_sfpu::GELU,
        approx: kind::GELU,
        range: (-12.0, 12.0),
        positive: false,
    },
];

/// `n` inputs for `op`: its range, then every special value class at fixed
/// positions (so each shape meets them).
fn inputs(op: &Op, seed: u64, n: usize) -> Vec<f32> {
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0000),
        f32::from_bits(0xffc0_0001),
        f32::from_bits(0x0000_1234),
        f32::from_bits(0x8000_1234),
        1.0,
        -1.0,
        88.0,
        -87.0,
        -87.5,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN_POSITIVE,
        9.0,
        0.5,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 11 == 0 {
                return specials[(i / 11) % specials.len()];
            }
            let u = (s >> 40) as f64 / (1u64 << 24) as f64;
            if op.positive {
                // Every binade, and for `recip` both signs.
                let bits = 0x0080_0000 + ((s >> 34) as u32 % 0x7e80_0000);
                let v = f32::from_bits(bits);
                if op.name == "recip" && i % 2 == 0 {
                    -v
                } else {
                    v
                }
            } else {
                (op.range.0 + u * (op.range.1 - op.range.0)) as f32
            }
        })
        .collect()
}

/// What the contract says `op` gives for `x`.
enum Expect {
    Nan,
    /// Bit for bit.
    Exact(f32),
    /// Within `bound` of `want`, relative.
    Rel(f64, f64),
    /// Within `bound` of `want`, absolute.
    Abs(f64, f64),
}

fn expect(op: &Op, x: f32) -> Expect {
    let xd = x as f64;
    if x.is_nan() {
        return Expect::Nan;
    }
    let tiny = x.abs() < f32::MIN_POSITIVE;
    match op.name {
        "exp" if x < -87.0 => Expect::Exact(0.0),
        "exp" if x >= 88.0 => Expect::Exact(f32::INFINITY),
        "exp" => Expect::Rel(xd.exp(), EXP_APPROX_BOUND),
        "log" if tiny => Expect::Exact(f32::NEG_INFINITY),
        "log" if x < 0.0 => Expect::Nan,
        "log" if x == f32::INFINITY => Expect::Exact(x),
        // `f = 0` exactly: ln 1 is +0 (a relative bound would divide by it).
        "log" if x == 1.0 => Expect::Exact(0.0),
        "log" => Expect::Rel(xd.ln(), LOG_APPROX_BOUND),
        "recip" if tiny => Expect::Exact(f32::INFINITY.copysign(x)),
        "recip" if x.is_infinite() || x.abs() >= f32::from_bits(0x7e80_0000) => {
            Expect::Exact(0.0f32.copysign(x))
        }
        "recip" if x.abs() > f32::from_bits(RECIP_RANGE_BITS) => {
            Expect::Rel(1.0 / xd, RECIP_SEED_BOUND)
        }
        "recip" => Expect::Rel(1.0 / xd, RECIP_APPROX_BOUND),
        "sigmoid" if x < -87.0 => Expect::Exact(0.0),
        "sigmoid" => Expect::Rel(1.0 / (1.0 + (-xd).exp()), SIGMOID_APPROX_BOUND),
        "tanh" if tiny => Expect::Exact(0.0f32.copysign(x)),
        "tanh" if x.abs() >= 9.0 => Expect::Exact(1.0f32.copysign(x)),
        "tanh" => Expect::Rel(xd.tanh(), TANH_APPROX_BOUND),
        "gelu" if x == f32::NEG_INFINITY => Expect::Nan,
        "gelu" if x == f32::INFINITY => Expect::Exact(x),
        "gelu" => {
            let want = 0.5 * xd * libm::erfc(-xd / std::f64::consts::SQRT_2);
            // The result may flush (a denormal's half): one `f32::MIN_POSITIVE`.
            Expect::Abs(want, gelu_approx_bound(xd, want) + f32::MIN_POSITIVE as f64)
        }
        _ => unreachable!(),
    }
}

/// `got` as `want` says, or the message of why not.
fn holds(want: &Expect, got: f32) -> Result<(), String> {
    match *want {
        Expect::Nan if got.is_nan() => Ok(()),
        Expect::Nan => Err(format!("{got:e} is not NaN")),
        Expect::Exact(e) if got.to_bits() == e.to_bits() => Ok(()),
        Expect::Exact(e) => Err(format!("{got:e} is not {e:e}")),
        // At the bottom of the normal range a result within a few ulps of
        // `2^-126` may land below it, and the store flushes it to zero.
        Expect::Rel(w, _) if got == 0.0 && w.abs() < 4.0 * f32::MIN_POSITIVE as f64 => Ok(()),
        Expect::Rel(w, b) => {
            let e = (got as f64 - w).abs() / w.abs();
            if e <= b {
                Ok(())
            } else {
                Err(format!("{got:e} vs {w:e}: {e:e} relative > {b:e}"))
            }
        }
        Expect::Abs(w, b) => {
            let e = (got as f64 - w).abs();
            if e <= b {
                Ok(())
            } else {
                Err(format!("{got:e} vs {w:e}: {e:e} absolute > {b:e}"))
            }
        }
    }
}

fn eltwise(kind: u32) -> Eltwise {
    Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The device in `mode` is the interpreter's program for the kind that mode
/// runs, bit for bit, and (Approx) the program is within its derived bound of
/// the f64 oracle, specials included.
fn hold_mode<T: tt_device::Transport>(s: &mut Session<T>, mode: MathMode, shapes: &[[usize; 2]]) {
    s.set_math_mode(mode);
    assert_eq!(s.math_mode(), mode);
    for op in &OPS {
        for &[r, c] in shapes {
            let av = inputs(op, (r * 7 + c) as u64, r * c);
            let a = s.upload(&av, r, c).unwrap();
            // The op is named by its Precise kind; the session's mode picks.
            let out = s
                .eltwise(eltwise(op.precise), &a, None)
                .unwrap_or_else(|e| panic!("{} {mode:?}: {e}", op.name));
            let got = s.download(&out).unwrap();
            let run_kind = if mode == MathMode::Approx {
                op.approx
            } else {
                op.precise
            };
            let model = reference(run_kind, 0.0, &av, None, r, c);
            for (i, (g, m)) in got.iter().zip(&model).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    m.to_bits(),
                    "[{r}, {c}] {} {mode:?}: {:e} (bits {:#010x}): device {g:e}, program {m:e}",
                    op.name,
                    av[i],
                    av[i].to_bits()
                );
            }
            if mode == MathMode::Approx {
                for (i, g) in got.iter().enumerate() {
                    if let Err(e) = holds(&expect(op, av[i]), *g) {
                        panic!(
                            "[{r}, {c}] {} approx({:e} = {:#010x}): {e}",
                            op.name,
                            av[i],
                            av[i].to_bits()
                        );
                    }
                }
            }
            s.free(out).unwrap();
            s.free(a).unwrap();
        }
    }
}

#[test]
fn approx_programs_are_the_interpreter_and_within_their_bounds() {
    with_session(open, |s| {
        hold_mode(s, MathMode::Approx, &[[37, 70], [64, 128]])
    });
}

/// Precise is the default, and bit-identical to the Precise programs.
#[test]
fn precise_is_the_default_and_unchanged() {
    with_session(open, |s| {
        assert_eq!(s.math_mode(), MathMode::Precise);
        hold_mode(s, MathMode::Precise, &[[37, 70]]);
    });
}

/// Alternating the modes in one session gives each its own program's bits:
/// the cache key carries the mode (negative control: an `Eltwise::in_mode`
/// that returned `self` makes the Approx rounds equal the Precise ones).
#[test]
fn alternating_modes_give_different_bits() {
    with_session(open, |s| {
        let (r, c) = (37, 70);
        for op in &OPS {
            let av = inputs(op, 5, r * c);
            let a = s.upload(&av, r, c).unwrap();
            let mut seen: Vec<(MathMode, Vec<u32>)> = Vec::new();
            for mode in [
                MathMode::Precise,
                MathMode::Approx,
                MathMode::Precise,
                MathMode::Approx,
            ] {
                s.set_math_mode(mode);
                let out = s.eltwise(eltwise(op.precise), &a, None).unwrap();
                let got = bits_of(&s.download(&out).unwrap());
                s.free(out).unwrap();
                let run_kind = if mode == MathMode::Approx {
                    op.approx
                } else {
                    op.precise
                };
                let want = bits_of(&reference(run_kind, 0.0, &av, None, r, c));
                assert_eq!(got, want, "{} {mode:?}", op.name);
                if let Some((_, earlier)) = seen.iter().find(|(m, _)| *m == mode) {
                    assert_eq!(&got, earlier, "{} {mode:?}: not repeatable", op.name);
                }
                seen.push((mode, got));
            }
            assert_ne!(
                seen[0].1, seen[1].1,
                "{}: Precise and Approx gave the same bits",
                op.name
            );
            s.free(a).unwrap();
        }
    });
}

/// The two modes' errors are the contract: Approx exceeds the Precise bound
/// on some input (so the mode really switched), Precise does not.
#[test]
fn approx_breaks_the_precise_bound_where_precise_holds_it() {
    use tt_kernels::sfpu::ops::{EXP_BOUND, LOG_BOUND, SIGMOID_BOUND, TANH_BOUND};
    with_session(open, |s| {
        let (r, c) = (32, 64);
        let cases: [(&Op, f64); 4] = [
            (&OPS[0], EXP_BOUND),
            (&OPS[1], LOG_BOUND),
            (&OPS[3], SIGMOID_BOUND),
            (&OPS[4], TANH_BOUND),
        ];
        for (op, bound) in cases {
            let av: Vec<f32> = (0..r * c)
                .map(|i| {
                    let x = -3.0 + 6.0 * i as f32 / (r * c) as f32;
                    if op.positive {
                        x.abs() + 0.01
                    } else {
                        x
                    }
                })
                .collect();
            let a = s.upload(&av, r, c).unwrap();
            let want = |x: f32| -> f64 {
                let x = x as f64;
                match op.name {
                    "exp" => x.exp(),
                    "log" => x.ln(),
                    "sigmoid" => 1.0 / (1.0 + (-x).exp()),
                    _ => x.tanh(),
                }
            };
            let mut worst = [0.0f64; 2];
            for (k, mode) in [MathMode::Precise, MathMode::Approx]
                .into_iter()
                .enumerate()
            {
                s.set_math_mode(mode);
                let out = s.eltwise(eltwise(op.precise), &a, None).unwrap();
                let got = s.download(&out).unwrap();
                s.free(out).unwrap();
                for (x, g) in av.iter().zip(&got) {
                    let w = want(*x);
                    if w != 0.0 {
                        worst[k] = worst[k].max((*g as f64 - w).abs() / w.abs());
                    }
                }
            }
            assert!(worst[0] <= bound, "{} precise {:e}", op.name, worst[0]);
            assert!(
                worst[1] > bound,
                "{}: approx stayed within Precise's bound {bound:e}: {:e}",
                op.name,
                worst[1]
            );
            s.free(a).unwrap();
        }
    });
}

/// `TT_MATH=approx` chooses the mode at open; a value that names none is
/// refused at open, naming the variable.
#[test]
fn tt_math_selects_the_mode_at_open() {
    with_session(
        || {
            std::env::set_var("TT_MATH", "approx");
            open()
        },
        |s| {
            assert_eq!(s.math_mode(), MathMode::Approx);
            let av = vec![0.5f32; 64];
            let a = s.upload(&av, 2, 32).unwrap();
            // No `set_math_mode`: the environment alone ran the Approx program.
            let out = s.eltwise(eltwise(kind_sfpu::EXP), &a, None).unwrap();
            let got = bits_of(&s.download(&out).unwrap());
            assert_eq!(
                got,
                bits_of(&reference(kind::EXP, 0.0, &av, None, 2, 32)),
                "TT_MATH=approx ran the Precise program"
            );
            s.set_math_mode(MathMode::Precise);
            assert_eq!(s.math_mode(), MathMode::Precise);
        },
    );
    #[cfg(not(feature = "silicon"))]
    if let Err(e) = fork_scope(|| {
        std::env::set_var("TT_MATH", "fast");
        let sim = Box::leak(Box::new(tt_ttsim::Simulator::open().unwrap()));
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let e = match Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
            |_, _| Ok(None),
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("TT_MATH=fast opened a session"),
        };
        assert!(e.contains("TT_MATH=fast"), "{e}");
    }) {
        panic!("{e}");
    }
}

/// The padding datums `t` holds, `(row, col, value)`.
fn padding<T: tt_device::Transport>(
    s: &mut Session<T>,
    t: &DramTensor,
) -> Vec<(usize, usize, f32)> {
    let raw = s.download_padded(t).unwrap();
    let [rt, ct] = t.grid();
    let mut out = Vec::new();
    for r in 0..32 * rt {
        for c in 0..32 * ct {
            if r >= t.rows || c >= t.cols {
                out.push((r, c, raw[r * 32 * ct + c]));
            }
        }
    }
    out
}

/// `t`'s `pad()` is what its tiles hold: a claim of zero is checked datum by
/// datum (either sign).
fn pad_is_honest<T: tt_device::Transport>(s: &mut Session<T>, t: &DramTensor, what: &str) {
    if t.pad() == Pad::Zero {
        for (r, c, v) in padding(s, t) {
            assert!(
                v == 0.0,
                "{what}: claims zero padding, holds {v} at ({r}, {c})"
            );
        }
    }
}

/// Ragged producers (`[37, 70]`: neither dimension a multiple of 32) in
/// Approx mode into a reduction and a matmul: the padding claim of every
/// Approx op is honest (`tanh`, `gelu` and `sqrt`-like ops keep zero; the
/// rest declare it undefined), and nothing the padding holds reaches a real
/// datum. The consumers' inputs are the downloaded device values themselves,
/// so the oracle is the arithmetic, not the program.
#[test]
fn ragged_approx_outputs_into_reduction_and_matmul() {
    use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
    use tt_kernels::{matmul::Fidelity, matmul::SrcRoute};
    with_session(open, |s| {
        s.set_math_mode(MathMode::Approx);
        let (r, c, n) = (37usize, 70usize, 24usize);
        // Small integer weights: exactly TF32.
        let wv: Vec<f32> = (0..c * n).map(|i| ((i * 7 + 3) % 5) as f32 - 2.0).collect();
        let w = s.upload(&wv, c, n).unwrap();
        for op in &OPS {
            // Moderate inputs, so no sum overflows.
            let av: Vec<f32> = inputs(op, 3, r * c)
                .into_iter()
                .map(|x| {
                    if x.is_finite() && x.abs() < 8.0 && x.abs() > 1e-6 {
                        if op.positive {
                            x.abs()
                        } else {
                            x
                        }
                    } else {
                        0.5
                    }
                })
                .collect();
            let a = s.upload(&av, r, c).unwrap();
            let y = s.eltwise(eltwise(op.precise), &a, None).unwrap();
            pad_is_honest(s, &y, op.name);
            let host = s.download(&y).unwrap();
            // The column sums, per row: the reduction masks the padding.
            let sums = s.reduce(&y, ReduceOp::Sum, Axis::Cols).unwrap();
            let got = s.download(&sums).unwrap();
            for i in 0..r {
                let row = &host[i * c..(i + 1) * c];
                let exact: f64 = row.iter().map(|&v| v as f64).sum();
                let abs: f64 = row.iter().map(|&v| (v as f64).abs()).sum();
                // FP32 accumulation of `c` terms in any order: `c u` of `sum |a|`.
                let tol = c as f64 * (1.0 / 16_777_216.0) * abs + 1e-30;
                assert!(
                    (got[i] as f64 - exact).abs() <= tol,
                    "{} row {i}: sum {} vs {exact}",
                    op.name,
                    got[i]
                );
            }
            // `y @ w`. The exact oracle: the same values uploaded clean (zero
            // padding), whose product no padding can have touched -- bit for
            // bit. A coarse f64 check beside it: TF32 operands cut to 10
            // mantissa bits are each within 2^-10 of themselves even if
            // truncated, so a product within 2^-9, plus the FP32
            // accumulation's c u, of sum |y w|.
            let m = s
                .matmul_dram(
                    &y,
                    false,
                    &w,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    tt_tests::harness::BUDGET,
                )
                .unwrap();
            let prod = s.download(&m).unwrap();
            let clean = s.upload(&host, r, c).unwrap();
            let m2 = s
                .matmul_dram(
                    &clean,
                    false,
                    &w,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    tt_tests::harness::BUDGET,
                )
                .unwrap();
            assert_eq!(
                bits_of(&prod),
                bits_of(&s.download(&m2).unwrap()),
                "{}: the ragged producer's padding reached the product",
                op.name
            );
            s.free(m2).unwrap();
            s.free(clean).unwrap();
            for i in 0..r {
                for j in 0..n {
                    let (mut exact, mut abs) = (0.0f64, 0.0f64);
                    for k in 0..c {
                        let t = host[i * c + k] as f64 * wv[k * n + j] as f64;
                        exact += t;
                        abs += t.abs();
                    }
                    let tol = (2.0 / 1024.0 + c as f64 / 16_777_216.0) * abs + 1e-30;
                    assert!(
                        (prod[i * n + j] as f64 - exact).abs() <= tol,
                        "{} [{i}, {j}]: {} vs {exact}",
                        op.name,
                        prod[i * n + j]
                    );
                }
            }
            for t in [m, sums, y, a] {
                s.free(t).unwrap();
            }
        }
        // Negative control: a false zero claim is caught. `exp(0)` is 1, so
        // the padding of an Approx `exp` is not zero.
        let a = s.upload(&vec![0.25f32; r * c], r, c).unwrap();
        let y = s.eltwise(eltwise(kind_sfpu::EXP), &a, None).unwrap();
        assert_eq!(y.pad(), Pad::Undefined, "exp(0) != 0");
        y.set_pad(Pad::Zero);
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pad_is_honest(s, &y, "forged exp")
        }));
        assert!(caught.is_err(), "a false zero-padding claim went unnoticed");
        y.set_pad(Pad::Undefined);
    });
}

/// The table the lane reports: instructions per tile, Precise against
/// Approx, from the programs the device runs.
#[test]
fn instruction_counts_per_tile() {
    use tt_kernels::sfpu::ops::program;
    println!(
        "{:<8} {:>9} {:>9} {:>7}",
        "op", "precise", "approx", "ratio"
    );
    let count = |k: u32| approx::instructions_per_tile(&program(k, 0.0).expect("a program").1);
    for op in &OPS {
        let (p, a) = (count(op.precise), count(op.approx));
        println!("{:<8} {p:>9} {a:>9} {:>6.2}x", op.name, p as f64 / a as f64);
        assert!(a < p, "{}: approx {a} >= precise {p}", op.name);
    }
}
