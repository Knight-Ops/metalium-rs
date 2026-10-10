//! Validated release Burn operation baselines, including final result readback.
#![cfg(feature = "silicon")]
use burn::tensor::{
    module,
    ops::{AttentionModuleOptions, ConvOptions},
    Int, Tensor, TensorData,
};
use burn_tt::TtBackend;
use std::time::Instant;
use tt_tests::{
    bench::{report, Stats},
    burn_device::{with_device, Config},
};

fn measure(key: &str, mut run: impl FnMut() -> TensorData, expected: &[f32]) {
    let mut samples = Vec::new();
    for repetition in 0..9 {
        let start = Instant::now();
        let output = run();
        let elapsed = start.elapsed().as_secs_f64() * 1e6;
        assert_eq!(output.convert::<f32>().to_vec::<f32>().unwrap(), expected);
        if repetition >= 2 {
            samples.push(elapsed);
        }
    }
    report(key, "us", "host", Stats::of(samples));
}

#[test]
#[ignore = "benchmark"]
fn resident_operation_baselines() {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            println!("BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"tiles\":2,\"warmups\":2,\"samples\":7,\"inputs\":\"resident\",\"timing\":\"host dispatch through final output readback; validation outside timing\",\"dtype\":\"F32 unless key says I32\"}}", tt_tests::bench::git_sha(), !cfg!(debug_assertions));
            let ints = Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(vec![12345; 37 * 70], [37, 70]),
                &d,
            )
            .add_scalar(0);
            measure(
                "I32 division [37,70] / 3",
                || ints.clone().div_scalar(3).into_data(),
                &vec![4115.0; 37 * 70],
            );
            let x = Tensor::<TtBackend, 4>::ones([1, 1, 9, 10], &d).to_device(&d);
            measure(
                "average pool [1,1,9,10] kernel4x8",
                || module::avg_pool2d(x.clone(), [4, 8], [1, 1], [0, 0], false, false).into_data(),
                &[1.0; 18],
            );
            let w = Tensor::<TtBackend, 4>::ones([2, 1, 2, 2], &d).to_device(&d);
            measure(
                "conv2d [1,1,9,10] weight[2,1,2,2]",
                || {
                    module::conv2d(
                        x.clone(),
                        w.clone(),
                        None,
                        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                    )
                    .into_data()
                },
                &[4.0; 144],
            );
            let q = Tensor::<TtBackend, 4>::zeros([1, 1, 3, 32], &d).to_device(&d);
            let k = Tensor::<TtBackend, 4>::zeros([1, 1, 64, 32], &d).to_device(&d);
            let v = Tensor::<TtBackend, 4>::ones([1, 1, 64, 32], &d).to_device(&d);
            measure(
                "attention Q[1,1,3,32] KVseq64",
                || {
                    module::attention(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        None,
                        None,
                        AttentionModuleOptions {
                            scale: None,
                            softcap: None,
                            is_causal: false,
                        },
                    )
                    .into_data()
                },
                &[1.0; 96],
            );
        },
    );
}

#[test]
#[ignore = "benchmark"]
fn bf16_and_mesh_operation_baselines() {
    use burn::tensor::FloatDType;
    let run = |d: burn_tt::TtDevice, mesh: bool| {
        println!("BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"cards\":{},\"warmups\":2,\"samples\":7,\"timing\":\"host dispatch through final readback; validation outside timing\"}}",tt_tests::bench::git_sha(),!cfg!(debug_assertions),if mesh {2} else {1});
        for dtype in [FloatDType::F32, FloatDType::BF16] {
            if !mesh && dtype == FloatDType::F32 {
                continue;
            }
            let x = Tensor::<TtBackend, 4>::ones([1, 64, 2, 3], &d)
                .cast(dtype)
                .to_device(&d);
            let w = Tensor::<TtBackend, 4>::ones([64, 64, 1, 1], &d)
                .cast(dtype)
                .to_device(&d);
            let execution = burn_tt::mesh_execution(d);
            measure(
                &format!("{dtype:?} mesh={mesh} conv2d x[1,64,2,3] w[64,64,1,1]"),
                || {
                    module::conv2d(
                        x.clone(),
                        w.clone(),
                        None,
                        ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
                    )
                    .into_data()
                },
                &[64.0; 384],
            );
            let q = Tensor::<TtBackend, 4>::zeros([1, 1, 3, 64], &d)
                .cast(dtype)
                .to_device(&d);
            let k = Tensor::<TtBackend, 4>::zeros([1, 1, 64, 64], &d)
                .cast(dtype)
                .to_device(&d);
            let v = Tensor::<TtBackend, 4>::ones([1, 1, 64, 64], &d)
                .cast(dtype)
                .to_device(&d);
            measure(
                &format!("{dtype:?} mesh={mesh} attention Q[1,1,3,64] KVseq64"),
                || {
                    module::attention(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        None,
                        None,
                        AttentionModuleOptions {
                            scale: None,
                            softcap: None,
                            is_causal: false,
                        },
                    )
                    .into_data()
                },
                &[1.0; 192],
            );
            if let Some(before) = execution {
                let after = burn_tt::mesh_execution(d).unwrap();
                for card in 0..2 {
                    assert!(after.completed_matmuls[card] > before.completed_matmuls[card]);
                }
                assert!(after.acknowledged_ethernet_bytes > before.acknowledged_ethernet_bytes);
            }
        }
    };
    with_device(Config::default(), |d| run(d, false));
    tt_tests::burn_device::with_mesh_device(Config::default(), 2, |d| run(d, true));
}

// ---------------------------------------------------------------------------
// P2: K-blocked matmul, forced block lengths against the planner's own choice.
// ---------------------------------------------------------------------------

fn k_block_session(f: impl FnOnce(&mut tt_kernels::session::Session<tt_kmd::Kmd>)) {
    if let Err(e) = tt_ttsim::fork_scope(|| {
        let mut s = tt_kernels::session::Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            tt_kernels::session::TileChoice::Count(2),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn same_bits(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}: element {i}: {a} vs {b}");
    }
}

/// `matmul/k_block/<shape>/<limit>`: `[64,8192] @ [8192,64]` and a ragged dense
/// `[37,4097] @ [4097,35]`, TF32 Src, F32 storage, HiFi4, two tiles.
///
/// Every sample is validated outside the timing: bit for bit against the
/// planner-chosen ("unsplit") product of the same operands, and for the
/// `[64,8192]` shape also against an independent exact prediction (B is
/// sparse identity-like, so every product and sum is exact and A's leading
/// columns are the answer in any accumulation order). Forced limits are K
/// tiles per block; `unsplit` is `set_matmul_k_block_limit(None)`.
#[test]
#[ignore = "benchmark"]
fn k_block_matmul_baselines() {
    use std::num::NonZeroUsize;
    use tt_kernels::matmul::{Fidelity, SrcRoute};
    use tt_tests::bench::Conditions;
    k_block_session(|s| {
        let tile = s.tile();
        Conditions::measure(s.device(), tt_tests::backend::device_index(), tile).print();
        println!("BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"tiles\":2,\"warmups\":2,\"samples\":7,\"route\":\"Tf32FromFp32\",\"fidelity\":\"HiFi4\",\"inputs\":\"resident GDDR tensors\",\"timing\":\"host dispatch through sync; validation outside timing\"}}", tt_tests::bench::git_sha(), !cfg!(debug_assertions));
        for (m, k, n, identity) in [(64usize, 8192usize, 64usize, true), (37, 4097, 35, false)] {
            let av: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 - 6.0).collect();
            let bv: Vec<f32> = if identity {
                (0..k * n)
                    .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
                    .collect()
            } else {
                (0..k * n)
                    .map(|i| ((i * 73 + 11) % 257) as f32 / 131.0 - 1.0)
                    .collect()
            };
            let a = s.upload(&av, m, k).unwrap();
            let b = s.upload(&bv, k, n).unwrap();
            let product = |s: &mut tt_kernels::session::Session<tt_kmd::Kmd>| {
                s.matmul_dram(
                    &a,
                    false,
                    &b,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    tt_tests::harness::BUDGET * 80,
                )
                .unwrap()
            };
            s.set_matmul_k_block_limit(None);
            let baseline = product(s);
            let want = s.download(&baseline).unwrap();
            s.free(baseline).unwrap();
            if identity {
                let exact: Vec<f32> = (0..m)
                    .flat_map(|r| av[r * k..r * k + n].iter().copied())
                    .collect();
                same_bits(&want, &exact, "unsplit vs exact");
            }
            let k_tiles = k.div_ceil(32);
            for limit in [None, Some(1usize), Some(4), Some(16), Some(64)] {
                if limit.is_some_and(|l| l >= k_tiles) {
                    continue;
                }
                s.set_matmul_k_block_limit(limit.and_then(NonZeroUsize::new));
                let label = limit.map_or("unsplit".to_string(), |l| format!("k_block={l}"));
                let before = s.dataflow_stats().clone();
                let mut samples = Vec::new();
                for repetition in 0..9 {
                    s.sync().unwrap();
                    let start = Instant::now();
                    let out = product(s);
                    s.sync().unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1e6;
                    same_bits(&s.download(&out).unwrap(), &want, &label);
                    s.free(out).unwrap();
                    if repetition >= 2 {
                        samples.push(elapsed);
                    }
                }
                let after = s.dataflow_stats();
                let key = format!("matmul/k_block/[{m},{k}]@[{k},{n}]/{label}");
                report(&key, "us", "host", Stats::of(samples));
                println!(
                    "MEASURE {key}: regions={} batches={} release_waits={} transfer_packets={} (9 runs)",
                    after.regions - before.regions,
                    after.batches - before.batches,
                    after.release_waits - before.release_waits,
                    after.transfer_packets - before.transfer_packets
                );
            }
            s.set_matmul_k_block_limit(None);
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}

// ---------------------------------------------------------------------------
// R3: Burn LayerNorm / RMSNorm compositions, forward and backward.
// ---------------------------------------------------------------------------

mod norm_bench {
    use super::*;
    use burn::{
        backend::Autodiff,
        module::Param,
        nn::{LayerNormConfig, RmsNormConfig},
        tensor::{
            backend::{AutodiffBackend, Backend},
            DType,
        },
    };
    use burn_flex::{Flex, FlexDevice};
    use burn_tt::tensor_traffic;
    use tt_kernels::{
        kind,
        sfpu::{
            ops::{kind_sfpu, pow_bound, reference},
            reduce::{reference as reduce_reference, Axis, ReduceOp},
        },
    };
    use tt_tests::burn_device::assert_native_model;

    type Ad = Autodiff<TtBackend>;
    type FlexAd = Autodiff<Flex>;
    const EPS: f32 = 1.0;
    const U32: f64 = 1.0 / 16_777_216.0;
    const UBF: f64 = 1.0 / 256.0;

    fn read(data: TensorData) -> Vec<f32> {
        data.convert::<f32>().to_vec::<f32>().unwrap()
    }

    /// `x`: row 0 constant, the rest multiples of 1/16 in [-1, 1] (exact in
    /// BF16); `g`: multiples of 1/8 in [-1, 1].
    fn inputs(rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
        let x = (0..rows * cols)
            .map(|i| {
                if i / cols == 0 {
                    1.0
                } else {
                    ((i * 13) % 31) as f32 / 16.0 - 1.0
                }
            })
            .collect();
        let g = (0..rows * cols)
            .map(|i| ((i * 7) % 17) as f32 / 8.0 - 1.0)
            .collect();
        (x, g)
    }

    /// Independent exact oracle: f64, no rounding. Returns y, dx, dgamma, dbeta for
    /// `loss = sum(g * norm(x))` with gamma = 1, beta = 0:
    /// `dx = (h - mean h)/d - c mean(h c)/d^3` (LayerNorm; RMS has `c = x` and no
    /// `mean h` term), `dgamma_j = sum_r g x_hat`, `dbeta_j = sum_r g`.
    fn exact(x: &[f32], g: &[f32], rows: usize, cols: usize, rms: bool) -> [Vec<f64>; 4] {
        let (mut y, mut dx) = (vec![0.0; rows * cols], vec![0.0; rows * cols]);
        let (mut dgamma, mut dbeta) = (vec![0.0; cols], vec![0.0; cols]);
        let n = cols as f64;
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let h = &g[r * cols..(r + 1) * cols];
            let mean = if rms {
                0.0
            } else {
                row.iter().map(|v| *v as f64).sum::<f64>() / n
            };
            let c: Vec<f64> = row.iter().map(|v| *v as f64 - mean).collect();
            let var = c.iter().map(|c| c * c).sum::<f64>() / n;
            let d = (var + EPS as f64).sqrt();
            let mean_h = if rms {
                0.0
            } else {
                h.iter().map(|v| *v as f64).sum::<f64>() / n
            };
            let mean_hc = h.iter().zip(&c).map(|(h, c)| *h as f64 * c).sum::<f64>() / n;
            for j in 0..cols {
                y[r * cols + j] = c[j] / d;
                dx[r * cols + j] = (h[j] as f64 - mean_h) / d - c[j] * mean_hc / (d * d * d);
                dgamma[j] += h[j] as f64 * c[j] / d;
                dbeta[j] += h[j] as f64;
            }
        }
        [y, dx, dgamma, dbeta]
    }

    /// Relative factor for an error accumulated along a path of `f32_depth`
    /// F32 roundings (any summation order: `gamma_k` of the abs-sum) and, for
    /// BF16 storage, `bf16_depth` BF16 roundings of op results (the adapter
    /// contract: BF16 operands widen to F32, each Burn op rounds its result once,
    /// reductions accumulate in F32).
    fn relative(f32_depth: usize, bf16_depth: usize, bf16: bool) -> f64 {
        let gamma = |k: usize, u: f64| {
            let ku = k as f64 * u;
            assert!(ku < 1.0, "depth {k} too large for the model");
            ku / (1.0 - ku)
        };
        gamma(f32_depth, U32) + if bf16 { gamma(bf16_depth, UBF) } else { 0.0 }
    }

    /// Derived absolute bounds against the exact oracle: `[y, dx, dgamma, dbeta]`.
    ///
    /// Inputs satisfy `|x| <= 1`, `|g| <= 1 = G`, `epsilon = 1`, so `d >= 1`.
    /// For an expression of additions, subtractions, products and divisions
    /// whose every operation rounds with relative error `<= u`, the result is
    /// within `gamma_N * A` of exact, `N` the longest operation chain and `A`
    /// the same expression evaluated on absolute values (so cancellation in
    /// `x - mean` is covered). SFPU `sqrt` and `/` are within 1.5 ulp (<= 3u),
    /// counted as three roundings; `epsilon > 0` keeps `d` away from zero.
    /// Forward chain: sum (n-1), mean (2), sub, square, sum (n-1), mean (2),
    /// +eps, sqrt (3), divide (3), gamma, beta: `2n + 13 <= 2n + 16`. The
    /// quotient's numerator and denominator both carry it: factor 2, with
    /// `A_c = |x| + |mean| <= 2` (1 for RMS). Backward adds the sums over `n`
    /// (d's gradient and the mean's) and over rows, `N <= 4 max(n, rows) + 64`;
    /// the absolute-value gradient `|h|/d + A_c^2 G/d^3 + means` is
    /// `<= 10G` (LayerNorm) / `2G` (RMS). Sqrt's backward evaluates
    /// `pow(w, -0.5)` with `w = var + eps <= 5`: `4 pow_bound` of that term
    /// (step70's allowance). BF16 chains: 16 forward, 64 backward.
    fn bounds(rows: usize, cols: usize, rms: bool, bf16: bool) -> [f64; 4] {
        let a_c = if rms { 1.0 } else { 2.0 };
        let fwd = relative(2 * cols + 16, 16, bf16);
        let back = relative(4 * cols.max(rows) + 64, 64, bf16);
        let a_dx = if rms { 2.0 } else { 10.0 };
        let pow = 4.0 * pow_bound(5.0, -0.5);
        [
            1.01 * 2.0 * fwd * a_c,
            1.01 * (2.0 * back + pow) * a_dx,
            1.01 * 2.0 * back * rows as f64 * a_c,
            1.01 * 2.0 * back * rows as f64,
        ]
    }

    /// Step70's specification model of the forward composition: bit-exact for
    /// F32 (programs run through the SFPU interpreter).
    fn model(x: &[f32], rows: usize, cols: usize, rms: bool) -> Vec<f32> {
        let mean = |x: &[f32]| {
            let sum = reduce_reference(ReduceOp::Sum, Axis::Cols, x, rows, cols);
            reference(kind::MUL_SCALAR, 1.0 / cols as f32, &sum, None, rows, 1)
        };
        let centered = if rms {
            x.to_vec()
        } else {
            reference(kind::SUB, 0.0, x, Some(&mean(x)), rows, cols)
        };
        let squared = reference(kind::MUL, 0.0, &centered, Some(&centered), rows, cols);
        let var = reference(kind::ADD_SCALAR, EPS, &mean(&squared), None, rows, 1);
        let denom = reference(kind_sfpu::SQRT, 0.0, &var, None, rows, 1);
        reference(kind_sfpu::DIV, 0.0, &centered, Some(&denom), rows, cols)
    }

    fn norm<B: Backend>(
        x: Tensor<B, 2>,
        gamma: Tensor<B, 1>,
        beta: Tensor<B, 1>,
        rms: bool,
    ) -> Tensor<B, 2> {
        let cols = x.dims()[1];
        let device = x.device();
        if rms {
            let mut m = RmsNormConfig::new(cols)
                .with_epsilon(EPS as f64)
                .init::<B>(&device);
            m.gamma = Param::from_tensor(gamma);
            m.forward(x)
        } else {
            let mut m = LayerNormConfig::new(cols)
                .with_epsilon(EPS as f64)
                .init::<B>(&device);
            m.gamma = Param::from_tensor(gamma);
            m.beta = Some(Param::from_tensor(beta));
            m.forward(x)
        }
    }

    /// `[dx, dgamma, dbeta]` of `sum(g * norm(x))`.
    fn gradients<B: AutodiffBackend>(
        x: &Tensor<B, 2>,
        gamma: &Tensor<B, 1>,
        beta: &Tensor<B, 1>,
        g: &Tensor<B, 2>,
        rms: bool,
    ) -> Vec<Vec<f32>> {
        let out = norm(x.clone(), gamma.clone(), beta.clone(), rms);
        let grads = (out * g.clone()).sum().backward();
        // RmsNorm has no beta: two gradients.
        let mut out = vec![
            read(x.grad(&grads).unwrap().into_data()),
            read(gamma.grad(&grads).unwrap().into_data()),
        ];
        if !rms {
            out.push(read(beta.grad(&grads).unwrap().into_data()));
        }
        out
    }

    fn within(got: &[f32], want: &[f32], bound: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (*g as f64 - *w as f64).abs() <= bound,
                "{what}[{i}]: {g} vs {w} (bound {bound:e})"
            );
        }
    }

    fn within_exact(got: &[f32], want: &[f64], bound: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (*g as f64 - *w).abs() <= bound,
                "{what}[{i}]: {g} vs exact {w} (bound {bound:e})"
            );
        }
    }

    fn time(key: &str, mut run: impl FnMut() -> Vec<Vec<f32>>, first: &[Vec<f32>]) {
        let mut samples = Vec::new();
        let mut downloads = 0;
        for repetition in 0..9 {
            let before = tensor_traffic();
            let start = Instant::now();
            let out = run();
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            downloads = (tensor_traffic() - before).downloads;
            // The device is deterministic: every repetition repeats the validated bits.
            for (o, f) in out.iter().zip(first) {
                same_bits(o, f, key);
            }
            if repetition >= 2 {
                samples.push(elapsed);
            }
        }
        report(key, "us", "host", Stats::of(samples));
        println!("MEASURE {key}: downloads_per_run={downloads}");
    }

    fn flex_vec(values: &[f32]) -> Tensor<Flex, 1> {
        Tensor::<Flex, 1>::from_data(
            TensorData::new(values.to_vec(), [values.len()]),
            &FlexDevice,
        )
    }

    pub fn run(forward: bool) {
        println!("BENCH {{\"kind\":\"conditions\",\"git\":\"{}\",\"release\":{},\"tiles\":\"default\",\"warmups\":2,\"samples\":7,\"epsilon\":1.0,\"gamma\":1,\"beta\":0,\"timing\":\"host dispatch through final readback; validation outside timing\",\"validation\":\"exact f64 oracle and burn-flex within the derived bound; F32 forward also bit-exact against the SFPU specification model\"}}", tt_tests::bench::git_sha(), !cfg!(debug_assertions));
        with_device(Config::default(), |d| {
            for (rows, cols) in [(37usize, 8193usize), (65, 70)] {
                let (x, g) = inputs(rows, cols);
                let (ones, zeros) = (vec![1.0f32; cols], vec![0.0f32; cols]);
                for rms in [false, true] {
                    // Flex's F32 result, once per shape, is the reference for every dtype.
                    let fx = Tensor::<Flex, 2>::from_data(
                        TensorData::new(x.clone(), [rows, cols]),
                        &FlexDevice,
                    );
                    let flex_y = if forward {
                        read(norm(fx, flex_vec(&ones), flex_vec(&zeros), rms).into_data())
                    } else {
                        Vec::new()
                    };
                    let flex_grads = if forward {
                        Vec::new()
                    } else {
                        let fx = Tensor::<FlexAd, 2>::from_data(
                            TensorData::new(x.clone(), [rows, cols]),
                            &FlexDevice,
                        )
                        .require_grad();
                        let fg = Tensor::<FlexAd, 2>::from_data(
                            TensorData::new(g.clone(), [rows, cols]),
                            &FlexDevice,
                        );
                        let gamma = Tensor::<FlexAd, 1>::from_data(
                            TensorData::new(ones.clone(), [cols]),
                            &FlexDevice,
                        )
                        .require_grad();
                        let beta = Tensor::<FlexAd, 1>::from_data(
                            TensorData::new(zeros.clone(), [cols]),
                            &FlexDevice,
                        )
                        .require_grad();
                        gradients::<FlexAd>(&fx, &gamma, &beta, &fg, rms)
                    };
                    let want = exact(&x, &g, rows, cols, rms);
                    let f32_bounds = bounds(rows, cols, rms, false);
                    if forward {
                        within_exact(&flex_y, &want[0], f32_bounds[0], "flex y vs exact");
                    }
                    for (i, grad) in flex_grads.iter().enumerate() {
                        within_exact(grad, &want[i + 1], f32_bounds[i + 1], "flex grad vs exact");
                    }
                    for dtype in [DType::F32, DType::BF16] {
                        let bf16 = dtype == DType::BF16;
                        let tag = format!(
                            "norm/{}/{dtype:?}/{}/[{rows},{cols}]",
                            if rms { "RmsNorm" } else { "LayerNorm" },
                            if forward {
                                "forward"
                            } else {
                                "forward+backward"
                            }
                        );
                        let tt_bounds = bounds(rows, cols, rms, bf16);
                        let data =
                            |v: &[f32], shape: [usize; 2]| TensorData::new(v.to_vec(), shape);
                        if forward {
                            let tx = Tensor::<TtBackend, 2>::from_data(
                                data(&x, [rows, cols]),
                                (&d, dtype),
                            )
                            .to_device(&d);
                            let gamma = Tensor::<TtBackend, 1>::from_data(
                                TensorData::new(ones.clone(), [cols]),
                                (&d, dtype),
                            )
                            .to_device(&d);
                            let beta = Tensor::<TtBackend, 1>::from_data(
                                TensorData::new(zeros.clone(), [cols]),
                                (&d, dtype),
                            )
                            .to_device(&d);
                            let (out, report_) = burn_tt::with_report(|| {
                                let before = tensor_traffic();
                                let out = norm(tx.clone(), gamma.clone(), beta.clone(), rms);
                                assert_eq!(
                                    tensor_traffic().downloads,
                                    before.downloads,
                                    "{tag}: a download before readback"
                                );
                                out
                            });
                            assert_native_model(&report_);
                            let got = read(out.into_data());
                            within_exact(&got, &want[0], tt_bounds[0], &format!("{tag} vs exact"));
                            within(
                                &got,
                                &flex_y,
                                tt_bounds[0] + f32_bounds[0],
                                &format!("{tag} vs flex"),
                            );
                            if !bf16 {
                                same_bits(
                                    &got,
                                    &model(&x, rows, cols, rms),
                                    &format!("{tag} vs spec model"),
                                );
                            }
                            time(
                                &tag,
                                || {
                                    vec![read(
                                        norm(tx.clone(), gamma.clone(), beta.clone(), rms)
                                            .into_data(),
                                    )]
                                },
                                &[got],
                            );
                        } else {
                            let tx =
                                Tensor::<Ad, 2>::from_data(data(&x, [rows, cols]), (&d, dtype))
                                    .require_grad();
                            let tg =
                                Tensor::<Ad, 2>::from_data(data(&g, [rows, cols]), (&d, dtype));
                            let gamma = Tensor::<Ad, 1>::from_data(
                                TensorData::new(ones.clone(), [cols]),
                                (&d, dtype),
                            )
                            .require_grad();
                            let beta = Tensor::<Ad, 1>::from_data(
                                TensorData::new(zeros.clone(), [cols]),
                                (&d, dtype),
                            )
                            .require_grad();
                            let (got, report_) = burn_tt::with_report(|| {
                                gradients::<Ad>(&tx, &gamma, &beta, &tg, rms)
                            });
                            assert_native_model(&report_);
                            assert_eq!(got.len(), flex_grads.len());
                            for (i, name) in
                                ["dx", "dgamma", "dbeta"].iter().enumerate().take(got.len())
                            {
                                within_exact(
                                    &got[i],
                                    &want[i + 1],
                                    tt_bounds[i + 1],
                                    &format!("{tag} {name} vs exact"),
                                );
                                within(
                                    &got[i],
                                    &flex_grads[i],
                                    tt_bounds[i + 1] + f32_bounds[i + 1],
                                    &format!("{tag} {name} vs flex"),
                                );
                            }
                            time(&tag, || gradients::<Ad>(&tx, &gamma, &beta, &tg, rms), &got);
                        }
                    }
                }
            }
        });
    }
}

/// `norm/<LayerNorm|RmsNorm>/<F32|BF16>/forward/[rows,cols]` on `[37,8193]`
/// (wide, ragged rows) and `[65,70]`; epsilon 1, gamma 1, beta 0.
/// Validated every run (outside timing) against an f64 oracle and burn-flex
/// within a derived bound (see `norm_bench::bounds`), plus the F32 forward bit
/// for bit against step70's SFPU specification model, with no download before
/// the final readback.
#[test]
#[ignore = "benchmark"]
fn norm_forward_baselines() {
    norm_bench::run(true);
}

/// The same shapes through Autodiff: `sum(g * norm(x)).backward()` and the
/// readback of dx, dgamma and dbeta. Validated as above (no spec model: the
/// graph's sqrt backward is an approximation, bounded with `pow_bound`).
#[test]
#[ignore = "benchmark"]
fn norm_backward_baselines() {
    norm_bench::run(false);
}
