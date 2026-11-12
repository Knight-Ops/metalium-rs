//! A row's argmax as one native op (`Session::argmax_row`, `tt_kernels::sfpu::select`;
//! decode-to-bandwidth Phase 7): a decode step's next token chosen on the card, its logits never
//! downloaded. burn-tt's `float_argmax` and `float_argmin` take it for one row along its last
//! dimension; the composition of predicates, masks and reductions took ~9 ms for 49,152 logits
//! on card 0 (`silicon_probe_sampling`).
//!
//! The claims, on eight units:
//! - **The composition's choice, and the host's** (`select::host`, `torch.argmax`'s rule): the
//!   first maximum, any NaN above every number (the first NaN), `-0` and `+0` equal. Over
//!   full-mantissa rows, ties, NaN beside infinity, two NaNs, signed zeros, subnormal maxima,
//!   rows of equal values, and widths that are not whole tiles; for argmax and argmin, rank one
//!   and rank two.
//! - **It is one op, natively:** the report charges the call's jobs to `argmax/row` and nothing is
//!   downloaded or staged.
//!
//! Negative control: the same rows with the first maximum moved by one give the other index.

use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::{TileChoice, TtBackend};
use tt_kernels::sfpu::select;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn row(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 40.0 - 20.0
        })
        .collect()
}

fn cases(n: usize) -> Vec<(String, Vec<f32>)> {
    let mut out = vec![("random".to_string(), row(n as u64, n))];
    let mut tie = row(7, n);
    for &i in &[n / 3, n / 2, n - 1] {
        tie[i] = 25.0;
    }
    out.push(("three equal maxima".into(), tie));
    let mut nan = row(9, n);
    nan[n / 4] = f32::INFINITY;
    nan[n / 2] = -f32::NAN;
    nan[n - 1] = f32::NAN;
    out.push(("NaN beside infinity, two NaNs".into(), nan));
    let mut zeros = vec![-1.0f32; n];
    zeros[n / 3] = -0.0;
    zeros[n / 2] = 0.0;
    out.push(("signed zeros".into(), zeros));
    out.push(("all equal".into(), vec![-3.5; n]));
    let mut small = vec![-f32::MAX; n];
    small[n / 5] = f32::from_bits(1);
    small[n / 7] = f32::from_bits(0x8000_0001);
    out.push(("a subnormal maximum".into(), small));
    out
}

fn chosen(t: Tensor<TtBackend, 2, Int>) -> i64 {
    t.into_data().to_vec::<i32>().unwrap()[0] as i64
}

fn config() -> Config {
    Config {
        tiles: Some(TileChoice::Count(8)),
        ..Config::default()
    }
}

#[test]
fn a_row_s_argmax_is_the_composition_s_and_the_host_s() {
    with_device(config(), |d| {
        for n in [32usize, 33, 100, 1024, 3000, 4096] {
            for (what, v) in cases(n) {
                let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [1, n]), &d);
                let neg: Vec<f32> = v.iter().map(|x| -x).collect();
                for (op, want) in [("argmax", select::host(&v)), ("argmin", select::host(&neg))] {
                    let run = |native: bool| {
                        burn_tt::set_argmax_row(native);
                        let call = || match op {
                            "argmax" => x.clone().argmax(1),
                            _ => x.clone().argmin(1),
                        };
                        let (got, report) = burn_tt::with_report(|| chosen(call()));
                        burn_tt::set_argmax_row(true);
                        (got, report)
                    };
                    let (native, report) = run(true);
                    assert_native_model(&report);
                    assert!(
                        report.op("argmax/row").map_or(0, |s| s.jobs) > 0,
                        "[1, {n}] {what}: {op} ran as the native op"
                    );
                    let (composed, _) = run(false);
                    assert_eq!(native as usize, want, "[1, {n}] {what}: {op}, the host's");
                    assert_eq!(native, composed, "[1, {n}] {what}: {op}, the composition's");
                }
            }
            // Rank one.
            let v = row(11, n);
            let x = Tensor::<TtBackend, 1>::from_data(TensorData::new(v.clone(), [n]), &d);
            let got = x.argmax(0).into_data().to_vec::<i32>().unwrap()[0] as usize;
            assert_eq!(got, select::host(&v), "[{n}]: rank one");
        }
    });
}

#[test]
fn the_maximum_moved_is_the_other_index() {
    with_device(config(), |d| {
        let n = 4096;
        let mut v = row(13, n);
        let first = select::host(&v);
        v[first] -= 100.0;
        let second = select::host(&v);
        assert_ne!(first, second);
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(v, [1, n]), &d);
        assert_eq!(chosen(x.argmax(1)) as usize, second);
    });
}
