//! The chain fuser (`burn_tt::fusion::chain`): Burn's default `sigmoid`, `silu` and
//! `silu(gate) * up` on `Tt` (burn-fusion over `TtBackend`). Plans: `docs/plans/kernel-fusion.md`
//! Phase 2, `docs/plans/decode-small-ops.md` F0 and S2b.
//!
//! burn-fusion 0.21 does not forward `ActivationOps` to the backend, so `sigmoid` on `Tt` reaches
//! the stream as Burn's composition: a cast to F32, `x -1`, `exp`, `+ 1`, `log`, `x -1`, `exp`
//! and a cast back. The fuser recognises it and runs `TtBackend`'s own sigmoid. The claims:
//! - **Parity:** `sigmoid`, `silu` and `silu(gate) * up` on `Tt` equal the same on `TtBackend`
//!   bit for bit, over signed zeros, subnormals, NaN payloads, infinities, `±f32::MAX` and a
//!   ragged shape, in F32 (and BF16 on silicon), and the fused path ran (`burn_tt::fused_count`).
//! - **One op:** on plain F32 operands `silu(gate) * up` (and `silu`) is one SFPU pass
//!   (`SILU_MUL`, `SILU`) and runs no `sigmoid` or `float_mul`, also where `gate` and `up` are
//!   the column halves of one fused projection read in place (ragged rows); BF16 runs the
//!   separate ops, which keep their intermediate roundings.
//! - **Negative controls**, each also asserting it did not run fused, with the oracle the same
//!   composition run op by op on `TtBackend`:
//!   - the composition with `+ 2` for `+ 1`, after the plan for `+ 1` exists (the scalar is bound
//!     only at execution, so the cached plan must check it);
//!   - the composition with `exp(-x)` kept and read afterwards (fusing would drop its handle);
//!   - the same functions written through the tensor API, with no casts;
//!   - `up * silu(gate)`, whose operand order is not Burn's: only its `silu` fuses.
//!
//!   The controls first require the composition's bits to differ from the native sigmoid's on
//!   these values, so a fused control could not pass.

use burn::tensor::activation::{sigmoid, silu};
use burn::tensor::backend::Backend;
use burn::tensor::{DType, FloatDType, Tensor, TensorData, TensorPrimitive};
use burn_tt::{Tt, TtBackend, TtDevice};
use std::collections::HashMap;
use tt_tests::burn_device::{with_device, Config};

/// Two rows past one tile and a ragged width.
const SHAPE: [usize; 2] = [33, 37];

/// Every special value, then a spread of ordinary ones, in a varying order.
fn values(seed: u32) -> Vec<f32> {
    const SPECIAL: [u32; 16] = [
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x0000_0001, // smallest subnormal
        0x807f_ffff, // largest negative subnormal
        0x0080_0000, // smallest normal
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_0000, // quiet NaN
        0x7fc1_2345, // NaN with a payload
        0xffc0_0001, // negative NaN with a payload
        0x7f7f_ffff, // f32::MAX
        0xff7f_ffff, // -f32::MAX
        0x42b0_0000, // 88: exp's overflow edge
        0xc2b0_0000, // -88
        0x41a0_0000, // 20
        0xc1a0_0000, // -20
    ];
    let n = SHAPE[0] * SHAPE[1];
    (0..n as u32)
        .map(|i| {
            let k = i.wrapping_mul(2_654_435_761).wrapping_add(seed);
            if i % 5 == 0 {
                f32::from_bits(SPECIAL[(k >> 7) as usize % SPECIAL.len()])
            } else {
                // Uniform in [-12, 12): sigmoid's interesting range.
                ((k >> 8) as f32 / (1u32 << 24) as f32) * 24.0 - 12.0
            }
        })
        .collect()
}

fn tensor<B: Backend<Device = TtDevice>>(v: &[f32], dtype: DType, d: &TtDevice) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(TensorData::new(v.to_vec(), SHAPE), d).cast(dtype)
}

fn bits<B: Backend>(t: Tensor<B, 2>) -> Vec<u32> {
    t.into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// A cast through the backend, as Burn's default sigmoid casts (`Tensor::cast` skips a cast to
/// the same dtype; the backend's `float_cast` does not).
fn cast<B: Backend>(t: Tensor<B, 2>, dtype: FloatDType) -> Tensor<B, 2> {
    Tensor::from_primitive(TensorPrimitive::Float(B::float_cast(
        t.into_primitive().tensor(),
        dtype,
    )))
}

/// Burn's default sigmoid (`burn-backend` `ActivationOps::sigmoid`) with `one` for its `1`, and
/// `exp(-x)` returned too when `keep`.
fn composed<B: Backend>(
    x: Tensor<B, 2>,
    one: f32,
    keep: bool,
) -> (Tensor<B, 2>, Option<Tensor<B, 2>>) {
    let dtype: FloatDType = x.dtype().into();
    let e = cast(x, FloatDType::F32).neg().exp();
    let kept = keep.then(|| e.clone());
    let s = (e + one).log().neg().exp();
    (cast(s, dtype), kept)
}

/// The same functions through the tensor API: no casts.
fn written<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    (x.neg().exp() + 1.0).log().neg().exp()
}

fn assert_bits(what: &str, got: &[u32], want: &[u32]) {
    let differ = got.iter().zip(want).filter(|(a, b)| a != b).count();
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what}: {differ} of {} differ; first at {i}: {:#010x} for {:#010x}",
            got.len(),
            got[i],
            want[i]
        );
    }
    assert_eq!(got.len(), want.len(), "{what}: lengths");
}

#[test]
fn the_compositions_run_as_tt_backends_own_ops() {
    with_device(Config::default(), |d| parity(DType::F32, &d));
}

/// Silicon-only, like every BF16 narrowing gate: the result's F32 -> BF16 pack is PACR mode
/// `0x105`, which ttsim refuses (`ttsim-divergence.md`, "BF16 late narrowing").
#[cfg(feature = "silicon")]
#[test]
fn bf16_compositions_run_as_tt_backends_own_ops() {
    with_device(Config::default(), |d| parity(DType::BF16, &d));
}

fn parity(dtype: DType, d: &TtDevice) {
    let (x, y) = (values(1), values(7));
    burn_tt::reset_fusion_counters();

    let want = bits(sigmoid(tensor::<TtBackend>(&x, dtype, d)));
    assert_bits(
        &format!("{dtype:?} sigmoid"),
        &bits(sigmoid(tensor::<Tt>(&x, dtype, d))),
        &want,
    );
    // Burn's composition written out with its casts is the same graph: it fuses too.
    assert_bits(
        &format!("{dtype:?} composed sigmoid"),
        &bits(composed(tensor::<Tt>(&x, dtype, d), 1.0, false).0),
        &want,
    );
    assert_eq!(
        burn_tt::fused_count("tt_sigmoid"),
        2,
        "{dtype:?}: sigmoid fused"
    );

    assert_bits(
        &format!("{dtype:?} silu"),
        &bits(silu(tensor::<Tt>(&x, dtype, d))),
        &bits(silu(tensor::<TtBackend>(&x, dtype, d))),
    );
    assert_eq!(burn_tt::fused_count("tt_silu"), 1, "{dtype:?}: silu fused");

    let silu_mul = |g: Tensor<Tt, 2>, u: Tensor<Tt, 2>| silu(g) * u;
    let (got, ops) = reported(|| bits(silu_mul(tensor(&x, dtype, d), tensor(&y, dtype, d))));
    assert_bits(
        &format!("{dtype:?} silu(gate) * up"),
        &got,
        &bits(silu(tensor::<TtBackend>(&x, dtype, d)) * tensor::<TtBackend>(&y, dtype, d)),
    );
    assert_eq!(
        burn_tt::fused_count("tt_silu_mul"),
        1,
        "{dtype:?}: silu * up fused"
    );
    let calls = |op: &str| ops.get(op).map_or(0, |s| s.0);
    if dtype == DType::F32 {
        assert_eq!(calls("tt_silu_mul"), 1, "one SFPU pass");
        assert_eq!(
            (calls("sigmoid"), calls("float_mul")),
            (0, 0),
            "and no other op"
        );
    } else {
        assert_eq!(
            (calls("sigmoid"), calls("float_mul")),
            (1, 2),
            "{dtype:?}: the ops"
        );
    }
}

/// `(calls, jobs)` of each Burn op `f` ran, on every thread (burn-fusion runs its stream on the
/// thread that drains it; the table is this gate's alone, as it runs in a fork).
fn reported<R>(f: impl FnOnce() -> R) -> (R, HashMap<&'static str, (u64, u64)>) {
    burn_tt::report_reset();
    let r = f();
    let ops = burn_tt::report()
        .0
        .iter()
        .map(|s| (s.op, (s.calls, s.jobs)))
        .collect();
    (r, ops)
}

/// Llama's MLP on a fused `gate|up` product: `gate` and `up` are column blocks of one buffer
/// (`Llama::fused`), which the fused pass reads where they lie, as `TtBackend`'s ops do.
#[test]
fn silu_mul_reads_a_fused_projections_halves_in_place() {
    with_device(Config::default(), |d| {
        const ROWS: usize = 33;
        const FF: usize = 64;
        let v: Vec<f32> = values(5).into_iter().cycle().take(ROWS * 2 * FF).collect();
        fn halves<B: Backend>(h: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
            let gate = h.clone().slice([0..ROWS, 0..FF]);
            let up = h.slice([0..ROWS, FF..2 * FF]);
            (gate, up)
        }
        let data = || TensorData::new(v.clone(), [ROWS, 2 * FF]);
        let h = Tensor::<TtBackend, 2>::from_data(data(), &d);
        let (native, ops) = reported(|| {
            let (gate, up) = halves(h);
            bits(silu(gate) * up)
        });
        eprintln!("TtBackend: {ops:?}");
        burn_tt::reset_fusion_counters();
        let (gate, up) = halves(Tensor::<Tt, 2>::from_data(data(), &d));
        let (got, ops) = reported(|| bits(silu(gate) * up));
        eprintln!("Tt: {ops:?}");
        assert_bits("silu(gate) * up on a fused projection", &got, &native);
        assert_eq!(burn_tt::fused_count("tt_silu_mul"), 1);
        let calls = |op: &str| ops.get(op).map_or(0, |s| s.0);
        assert_eq!(
            (calls("tt_silu_mul"), calls("sigmoid"), calls("float_mul")),
            (1, 0, 0)
        );
    });
}

#[test]
fn near_misses_run_as_their_operations() {
    with_device(Config::default(), |d| {
        let x = values(3);
        let y = values(11);
        let native = bits(sigmoid(tensor::<TtBackend>(&x, DType::F32, &d)));
        let composition = bits(composed(tensor::<TtBackend>(&x, DType::F32, &d), 1.0, false).0);
        assert_ne!(
            native, composition,
            "control: the composition's bits must differ from the native sigmoid's"
        );
        burn_tt::reset_fusion_counters();

        // The plan for `+ 1`, then the same graph with `+ 2`, then `+ 1` again.
        let on_tt = |one: f32| bits(composed(tensor::<Tt>(&x, DType::F32, &d), one, false).0);
        assert_bits("+ 1", &on_tt(1.0), &native);
        let two = bits(composed(tensor::<TtBackend>(&x, DType::F32, &d), 2.0, false).0);
        assert_bits("+ 2", &on_tt(2.0), &two);
        assert_bits("+ 1 again", &on_tt(1.0), &native);
        assert_eq!(
            burn_tt::fused_count("tt_sigmoid"),
            2,
            "only the two `+ 1` chains fuse"
        );

        // `exp(-x)` kept and read afterwards.
        let (s, e) = composed(tensor::<Tt>(&x, DType::F32, &d), 1.0, true);
        let e_want = composed(tensor::<TtBackend>(&x, DType::F32, &d), 1.0, true).1;
        assert_bits("kept exp(-x)", &bits(e.unwrap()), &bits(e_want.unwrap()));
        assert_bits("sigmoid beside a kept exp(-x)", &bits(s), &composition);

        // No casts.
        assert_bits(
            "written without casts",
            &bits(written(tensor::<Tt>(&x, DType::F32, &d))),
            &bits(written(tensor::<TtBackend>(&x, DType::F32, &d))),
        );
        assert_eq!(
            burn_tt::fused_count("tt_sigmoid"),
            2,
            "no control fuses a sigmoid"
        );

        // `up * silu(gate)`: the silu fuses, the product by `up` runs as itself.
        let swapped = |g: Tensor<Tt, 2>, u: Tensor<Tt, 2>| u * silu(g);
        assert_bits(
            "up * silu(gate)",
            &bits(swapped(
                tensor(&x, DType::F32, &d),
                tensor(&y, DType::F32, &d),
            )),
            &bits(
                tensor::<TtBackend>(&y, DType::F32, &d)
                    * silu(tensor::<TtBackend>(&x, DType::F32, &d)),
            ),
        );
        assert_eq!(burn_tt::fused_count("tt_silu"), 1);
        assert_eq!(burn_tt::fused_count("tt_silu_mul"), 0);
    });
}
