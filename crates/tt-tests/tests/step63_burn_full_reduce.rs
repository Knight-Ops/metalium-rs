//! R1b: full F32 sums and means stay resident, including ragged ranks and
//! transposes. The oracle composes the existing SFPU program models; a
//! separate bound checks the different addition order against burn-flex.

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, Trace, TtBackend, TtDevice};
use tt_isa::numerics::add_bh;
use tt_kernels::sfpu::ops::{kind_sfpu, reference_op, Broadcast};
use tt_kernels::sfpu::reduce::{reference, Axis, ReduceOp};
use tt_tests::burn_device::{with_device, Config};

const U: f64 = 1.0 / 16_777_216.0;

fn values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 73 % 257) as f32 - 128.0) / 37.0)
        .collect()
}

fn model(values: &[f32], rows: usize, cols: usize, mean: bool) -> Vec<f32> {
    let mut sum: Option<f32> = None;
    let chunk_cols = 32 * tt_kernels::sfpu::reduce::ROW_CHUNK;
    for first in (0..cols).step_by(chunk_cols) {
        let count = chunk_cols.min(cols - first);
        let chunk: Vec<_> = (0..rows)
            .flat_map(|r| {
                values[r * cols + first..r * cols + first + count]
                    .iter()
                    .copied()
            })
            .collect();
        let column = reference(ReduceOp::Sum, Axis::Cols, &chunk, rows, count);
        let partial = reference(ReduceOp::Sum, Axis::Rows, &column, rows, 1)[0];
        sum = Some(match sum {
            None => partial,
            Some(prior) => f32::from_bits(add_bh(prior.to_bits(), partial.to_bits())),
        });
    }
    let sum = vec![sum.unwrap()];
    if mean {
        reference_op(
            kind_sfpu::DIV_SCALAR,
            [(rows * cols) as f32, 0.0],
            Broadcast::None,
            &[&sum],
            1,
            1,
        )
    } else {
        sum
    }
}

fn resident<const D: usize>(t: &Tensor<TtBackend, D>) -> bool {
    primitive(t.clone()).computed_on_device()
}

fn primitive<const D: usize>(t: Tensor<TtBackend, D>) -> burn_tt::TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("an F32 tensor"),
    }
}

fn floats<B: burn::tensor::backend::Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec().unwrap()
}

fn same(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!(
            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
            "{g:?} vs {w:?}"
        );
    }
}

fn check<const D: usize>(device: &TtDevice, shape: [usize; D], values: Vec<f32>) {
    let n: usize = shape.iter().product();
    let cols = shape[D - 1];
    let rows = n / cols;
    let t = Tensor::<TtBackend, D>::from_data(TensorData::new(values.clone(), shape), device)
        .to_device(device);
    let f = Tensor::<Flex, D>::from_data(TensorData::new(values.clone(), shape), &FlexDevice);
    let before = tensor_traffic();
    let sum = t.clone().sum();
    let mean = t.mean();
    let during = tensor_traffic() - before;
    assert!(resident(&sum) && resident(&mean), "{shape:?}: host result");
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "{shape:?}: {during:?}"
    );
    assert_eq!(sum.dims(), [1]);
    assert_eq!(mean.dims(), [1]);
    let (sum, mean) = (floats(sum), floats(mean));
    same(&sum, &model(&values, rows, cols, false));
    same(&mean, &model(&values, rows, cols, true));

    // Each sum performs at most n-1 rounded additions on any input's
    // path. For normal finite values without overflow, the standard
    // gamma_(n-1) bound is relative to sum|x|, so cancellation is covered.
    // Flex's order contributes the same bound. Division adds its one-ulp
    // error and Flex's half-ulp rounding (conservatively 4u together).
    let gamma = (n - 1) as f64 * U / (1.0 - (n - 1) as f64 * U);
    let abs_sum: f64 = values.iter().map(|x| (*x as f64).abs()).sum();
    let sum_bound = 2.0 * gamma * abs_sum;
    let fs = floats(f.clone().sum())[0];
    let fm = floats(f.mean())[0];
    assert!(
        (sum[0] as f64 - fs as f64).abs() <= sum_bound,
        "{shape:?}: full sum"
    );
    let mean_bound = sum_bound / n as f64 + 4.0 * U * abs_sum / n as f64;
    assert!(
        (mean[0] as f64 - fm as f64).abs() <= mean_bound,
        "{shape:?}: full mean"
    );
}

#[test]
fn full_reductions_are_resident_and_match_the_program_models() {
    with_device(Config::default(), |d| {
        check(&d, [1], vec![3.5]);
        check(&d, [70], values(70));
        check(&d, [37, 70], values(37 * 70));
        check(&d, [64, 10], values(64 * 10));
        check(&d, [3, 5, 70], values(3 * 5 * 70));
        check(&d, [2, 2, 5, 7], values(2 * 2 * 5 * 7));
        // Beyond a one-pass reduction's L1 capacity, in both directions,
        // and across a ragged final column chunk.
        check(&d, [37, 8193], values(37 * 8193));
        check(&d, [8193, 1], values(8193));
    });
}

#[test]
fn padding_and_special_values_follow_the_full_reduction_models() {
    with_device(Config::default(), |d| {
        let [rows, cols] = [37, 70];
        for value in [
            -0.0,
            f32::MIN_POSITIVE / 2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            let v = vec![value; rows * cols];
            let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [rows, cols]), &d)
                .to_device(&d);
            // exp turns the input's zero padding into ones; its output
            // padding must not enter either scalar reduction.
            let dirty = Tensor::<TtBackend, 2>::zeros([rows, cols], &d)
                .to_device(&d)
                .exp();
            assert_eq!(floats(dirty.sum()), vec![(rows * cols) as f32]);
            same(&floats(t.clone().sum()), &model(&v, rows, cols, false));
            same(&floats(t.mean()), &model(&v, rows, cols, true));
        }
    });
}

#[test]
fn full_views_and_row_slices_reduce_on_the_device() {
    with_device(Config::default(), |d| {
        let v = values(37 * 70);
        let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [37, 70]), &d)
            .to_device(&d);
        let before = tensor_traffic();
        let transposed = t.transpose();
        let sum = transposed.clone().sum();
        let mean = transposed.mean();
        assert!(resident(&sum) && resident(&mean));
        assert_eq!((tensor_traffic() - before).downloads, 0);
        same(&floats(sum), &model(&v, 37, 70, false));
        same(&floats(mean), &model(&v, 37, 70, true));

        // Swapping the leading dimensions changes order without changing
        // the tensor's elements. Full reductions read its source directly.
        let v = values(2 * 2 * 32 * 32);
        let t = Tensor::<TtBackend, 4>::from_data(TensorData::new(v.clone(), [2, 2, 32, 32]), &d)
            .to_device(&d);
        let before = tensor_traffic();
        let sum = t.swap_dims(0, 1).sum();
        assert!(resident(&sum));
        assert_eq!((tensor_traffic() - before).downloads, 0);
        same(&floats(sum), &model(&v, 128, 32, false));

        // The loss's `[128, 1] -> [128]` reshape stores a vector logically,
        // but its values still lie down the original buffer's rows.
        let v = values(128);
        let column = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [128, 1]), &d)
            .to_device(&d);
        let before = tensor_traffic();
        let flat: Tensor<TtBackend, 1> = column.reshape([128]);
        let mean = flat.mean();
        assert!(resident(&mean));
        let moved = tensor_traffic() - before;
        assert_eq!((moved.uploads, moved.downloads), (0, 0));
        same(&floats(mean), &model(&v, 128, 1, true));

        let v = values(69 * 70);
        let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [69, 70]), &d)
            .to_device(&d);
        let before = tensor_traffic();
        let sum = t.slice([32..69, 0..70]).sum();
        assert!(resident(&sum));
        assert_eq!((tensor_traffic() - before).downloads, 0);
        same(&floats(sum), &model(&v[32 * 70..], 37, 70, false));
    });
}

#[test]
fn host_f32_inputs_are_uploaded_for_native_full_reductions() {
    with_device(Config::default(), |d| {
        let v = values(37 * 70);
        for mean in [false, true] {
            let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), [37, 70]), &d);
            let before = tensor_traffic();
            let result = if mean { t.mean() } else { t.sum() };
            assert!(resident(&result));
            let moved = tensor_traffic() - before;
            assert_eq!((moved.uploads, moved.downloads), (1, 0));
            assert_eq!(moved.uploaded, (v.len() * 4) as u64);
            same(&floats(result), &model(&v, 37, 70, mean));
        }
    });
}

#[cfg(not(feature = "silicon"))]
#[test]
fn mesh_full_reductions_remain_resident_and_chunk_large_inputs() {
    tt_tests::burn_device::with_mesh_device(Config::default(), 2, |d| {
        for shape in [[37, 70], [64, 10], [1, 8193]] {
            let v = values(shape.iter().product());
            let input = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), shape), &d);
            for mean in [false, true] {
                let (result, report) = burn_tt::with_report(|| {
                    let result = if mean {
                        input.clone().mean()
                    } else {
                        input.clone().sum()
                    };
                    assert!(resident(&result));
                    floats(result)
                });
                same(&result, &model(&v, shape[0], shape[1], mean));
                tt_tests::burn_device::assert_native_model(&report);
            }
        }
    });
}

#[test]
fn full_reduction_autodiff_matches_flex() {
    with_device(Config::default(), |d| {
        let shape = [3, 5, 7];
        let v = values(shape.iter().product());
        for mean in [false, true] {
            let t =
                Tensor::<Autodiff<TtBackend>, 3>::from_data(TensorData::new(v.clone(), shape), &d)
                    .to_device(&d)
                    .require_grad();
            let f = Tensor::<Autodiff<Flex>, 3>::from_data(
                TensorData::new(v.clone(), shape),
                &FlexDevice,
            )
            .require_grad();
            let loss = if mean {
                t.clone().mean()
            } else {
                t.clone().sum()
            };
            assert!(resident(&loss.clone().inner()));
            let want = if mean {
                f.clone().mean()
            } else {
                f.clone().sum()
            };
            let grad = t.grad(&loss.backward()).unwrap();
            let fgrad = f.grad(&want.backward()).unwrap();
            same(&floats(grad), &floats(fgrad));
        }
    });
}

#[test]
fn traced_scalar_reductions_replay_with_changed_inputs() {
    with_device(
        Config {
            tiles: Some(burn_tt::TileChoice::Count(2)),
            ..Config::default()
        },
        |d| {
            for shape in [[37, 70], [37, 8193]] {
                let v = values(shape.iter().product());
                let t = Tensor::<TtBackend, 2>::from_data(TensorData::new(v.clone(), shape), &d)
                    .to_device(&d);
                let input = primitive(t.clone());
                for mean in [false, true] {
                    let (trace, first) = Trace::capture(&input, || {
                        primitive(if mean {
                            t.clone().mean()
                        } else {
                            t.clone().sum()
                        })
                    })
                    .unwrap();
                    assert_eq!(trace.output_dims(), [1, 1]);
                    same(&first, &model(&v, shape[0], shape[1], mean));
                    for offset in [1.0, -2.0, 0.0] {
                        let next: Vec<_> = v.iter().map(|x| x + offset).collect();
                        same(
                            &trace.run(next.clone()).unwrap(),
                            &model(&next, shape[0], shape[1], mean),
                        );
                    }
                    // The last replay restored the capture's input, so fresh work
                    // and the next capture read the same data again.
                    let fresh = if mean {
                        t.clone().mean()
                    } else {
                        t.clone().sum()
                    };
                    same(&floats(fresh), &model(&v, shape[0], shape[1], mean));
                }
            }
        },
    );
}
