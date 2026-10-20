//! Rotary position embedding as one op (`burn_tt::rope`, `Session::rope`; decode-to-bandwidth
//! Phase 7): `x cos + rotate_half(x) sin` in one job a run of tiles, `rotate_half` read as `x`'s
//! own tiles across each head's halves, where the composition was a `cat` copy, a product and
//! `MUL_ADD` (burn-fusion's block ends at the `cat`, so no fuser saw the three together).
//!
//! The claims, on `TtBackend` and on `Tt` (burn-fusion, where it runs as a custom operation):
//! - **Bit for bit the composition** (`x * cos + cat(x2, x1) * sin`, `Rope::apply`'s), for `q` and
//!   `k` read as views of a projection row as `Llama` makes them, a plain `[1, heads, t, d]`
//!   tensor, `head_dim` 64 and 128, one position and a chunk of four, nine heads and 32, over
//!   full-mantissa values, signed zeros, subnormals and large values. `head_dim` 32 (halves of 16
//!   columns) composes, the same bits.
//! - **One op where `q` and `k` are a projection's views** (`burn_tt::rope_count`, which also
//!   sees `Tt`'s fusion stream, run off the caller's thread); a plain `[1, heads, t, d]` (heads as
//!   rows, whose tables would need a row broadcast) composes, the same bits.
//! - **Natively:** nothing staged or computed on the host (`TtBackend`'s report).
//!
//! Negative control: the tables swapped give other bits.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::{Tt, TtBackend};
use tt_models::rope::Rope;
use tt_tests::burn_device::{assert_native_model, with_device, Config};

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match i % 53 {
                3 => -0.0,
                7 => f32::from_bits(0x0000_0042),
                11 => 3.0e30,
                _ => ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0,
            }
        })
        .collect()
}

fn bits<B: Backend>(t: Tensor<B, 4>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// `x * cos + cat(x2, x1) * sin`, as Burn ops.
fn composed<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let [n, h, t, d] = x.dims();
    let x1 = x.clone().slice([0..n, 0..h, 0..t, 0..d / 2]);
    let x2 = x.clone().slice([0..n, 0..h, 0..t, d / 2..d]);
    x * cos + Tensor::cat(vec![x2, x1], 3) * sin
}

/// The rope tables at `t` positions from `start`, as `Llama` selects them.
fn tables<B: Backend>(t: usize, d: usize, device: &B::Device) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let table = Rope::<B>::table(64, d, 10_000.0, burn::tensor::DType::F32, device);
    let at = Tensor::<B, 1, Int>::from_data(
        TensorData::new((5..5 + t as i32).collect::<Vec<_>>(), [t]),
        device,
    );
    table.at(at).tables()
}

/// `heads` heads of `d` from a `[1, t, width]` projection, as `Llama`'s `heads` views them.
fn heads<B: Backend>(row: &Tensor<B, 3>, first: usize, heads: usize, d: usize) -> Tensor<B, 4> {
    let [n, t, _] = row.dims();
    row.clone()
        .slice([0..n, 0..t, first..first + heads * d])
        .reshape([n, t, heads, d])
        .swap_dims(1, 2)
}

fn check<B: Backend>(
    what: &str,
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    native: bool,
) {
    let want = bits(composed(x.clone(), cos.clone(), sin.clone()));
    let before = burn_tt::rope_count();
    let got = bits(burn_tt::rope(&x, &cos, &sin).expect("rope"));
    let ran = burn_tt::rope_count() > before;
    assert_eq!(got, want, "{what}: the composition's bits");
    assert_eq!(ran, native, "{what}: the native op ran");
    let other = bits(burn_tt::rope(&x, &sin, &cos).unwrap());
    assert_ne!(other, want, "{what}: the tables swapped");
}

fn cases<B: Backend>(device: &B::Device) {
    for (d, native) in [(64usize, true), (128, true), (32, false)] {
        for t in [1usize, 4] {
            for (h, kv) in [(9usize, 3usize), (32, 32)] {
                let width = (h + 2 * kv) * d;
                let row = Tensor::<B, 3>::from_data(
                    TensorData::new(values((d * 7 + t * 3 + h) as u64, t * width), [1, t, width]),
                    device,
                );
                let _ = (row.clone() * 1.0).sum().into_data();
                let (cos, sin) = tables::<B>(t, d, device);
                let what =
                    |x: &str| format!("{x}, head_dim {d}, {t} positions, {h} heads over {kv}");
                check(
                    &what("q of a projection"),
                    heads(&row, 0, h, d),
                    cos.clone(),
                    sin.clone(),
                    native,
                );
                check(
                    &what("k of a projection"),
                    heads(&row, h * d, kv, d),
                    cos.clone(),
                    sin.clone(),
                    native,
                );
                let plain = Tensor::<B, 4>::from_data(
                    TensorData::new(values(99, h * t * d), [1, h, t, d]),
                    device,
                );
                // Heads as rows of their own: the tables would need a row broadcast, composed.
                check(&what("a plain tensor"), plain, cos, sin, false);
            }
        }
    }
}

#[test]
fn rope_is_the_composition_bit_for_bit_on_tt_backend() {
    with_device(Config::default(), |d| {
        let ((), report) = burn_tt::with_report(|| cases::<TtBackend>(&d));
        assert_native_model(&report);
    });
}

#[test]
fn rope_is_the_composition_bit_for_bit_on_tt() {
    with_device(Config::default(), |d| cases::<Tt>(&d));
}
