//! Full-width wrapping folds and signed extrema, against independent Rust.
use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

#[test]
fn integer_reductions_wrap_and_keep_signed_order_on_ragged_rank_n_views() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (35, 67);
        let values: Vec<i32> = (0..rows * cols)
            .map(|i| {
                let word = (i as u32).wrapping_mul(0x9e3779b9).rotate_left(11) | 1;
                if i % 97 == 0 {
                    i32::MIN
                } else if i % 101 == 0 {
                    i32::MAX
                } else {
                    word as i32
                }
            })
            .collect();
        let x = Tensor::<TtBackend, 2, Int>::from_data(
            TensorData::new(values.clone(), [rows, cols]),
            &d,
        );
        let before = tensor_traffic();
        let ((sums, products, mins, maxs, fullsum, fullprod, fullmin, fullmax), report) =
            burn_tt::with_report(|| {
                (
                    x.clone().sum_dim(1),
                    x.clone().prod_dim(1),
                    x.clone().min_dim(0),
                    x.clone().max_dim(0),
                    x.clone().sum(),
                    x.clone().prod(),
                    x.clone().min(),
                    x.clone().max(),
                )
            });
        assert_native_model(&report);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        for t in [&sums, &products, &mins, &maxs] {
            assert!(t.clone().into_primitive().computed_on_device());
        }
        let sum: Vec<_> = (0..rows)
            .map(|r| {
                values[r * cols..(r + 1) * cols]
                    .iter()
                    .fold(0i32, |a, &b| a.wrapping_add(b))
            })
            .collect();
        let prod: Vec<_> = (0..rows)
            .map(|r| {
                values[r * cols..(r + 1) * cols]
                    .iter()
                    .fold(1i32, |a, &b| a.wrapping_mul(b))
            })
            .collect();
        let min: Vec<_> = (0..cols)
            .map(|c| (0..rows).map(|r| values[r * cols + c]).min().unwrap())
            .collect();
        let max: Vec<_> = (0..cols)
            .map(|c| (0..rows).map(|r| values[r * cols + c]).max().unwrap())
            .collect();
        let read = |t: Tensor<TtBackend, 2, Int>| t.into_data().to_vec::<i32>().unwrap();
        assert_eq!(read(sums), sum);
        assert_eq!(read(products), prod);
        assert_eq!(read(mins), min);
        assert_eq!(read(maxs), max);
        assert!(
            prod.iter().any(|&i| i != 0 && i != 1),
            "zero/identity product negative control"
        );
        assert_eq!(
            fullsum.into_data().to_vec::<i32>().unwrap(),
            vec![values.iter().fold(0i32, |a, &b| a.wrapping_add(b))]
        );
        assert_eq!(
            fullprod.into_data().to_vec::<i32>().unwrap(),
            vec![values.iter().fold(1i32, |a, &b| a.wrapping_mul(b))]
        );
        assert_eq!(fullmin.into_data().to_vec::<i32>().unwrap(), vec![i32::MIN]);
        assert_eq!(fullmax.into_data().to_vec::<i32>().unwrap(), vec![i32::MAX]);
        let odd: Vec<_> = (0..3 * 5 * 7).map(|i| (i * 1234567) | 1).collect();
        let x = Tensor::<TtBackend, 3, Int>::from_data(TensorData::new(odd.clone(), [3, 5, 7]), &d)
            .swap_dims(0, 2);
        let ((sum, prod), report) = burn_tt::with_report(|| (x.clone().sum_dim(1), x.prod_dim(1)));
        assert_native_model(&report);
        assert_eq!(sum.dims(), [7, 1, 3]);
        let mut sums = Vec::new();
        let mut products = Vec::new();
        for c in 0..7 {
            for a in 0..3 {
                sums.push((0..5).fold(0i32, |s, b| s.wrapping_add(odd[(a * 5 + b) * 7 + c])));
                products.push((0..5).fold(1i32, |s, b| s.wrapping_mul(odd[(a * 5 + b) * 7 + c])));
            }
        }
        assert_eq!(sum.into_data().to_vec::<i32>().unwrap(), sums);
        assert_eq!(prod.into_data().to_vec::<i32>().unwrap(), products);
    });
}
