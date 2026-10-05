//! Integer ALU gates use independent Rust wrapping/bitwise arithmetic, including
//! every bit of a product; SFPMUL24's 23-bit truncation is not the public result.
use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{with_device, Config};

#[test]
fn full_width_integer_alu_and_broadcasts_are_native() {
    with_device(Config::default(), |d| {
        let shape = [37, 70];
        let cases = [
            i32::MIN,
            i32::MAX,
            -1,
            0,
            1,
            0x7fffff,
            0xffffff,
            0xffff,
            1 << 24,
            -(1 << 24) - 1,
        ];
        let a: Vec<_> = (0..2590)
            .map(|i| {
                if i < cases.len() {
                    cases[i]
                } else {
                    (i as i32).wrapping_mul(0x61c88647)
                }
            })
            .collect();
        let b: Vec<_> = (0..2590)
            .map(|i| {
                if i < cases.len() {
                    cases[cases.len() - 1 - i]
                } else {
                    (i as i32).wrapping_mul(0x12345679)
                }
            })
            .collect();
        let x = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(a.clone(), shape), &d)
            .to_device(&d);
        let y = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(b.clone(), shape), &d)
            .to_device(&d);
        let before = tensor_traffic();
        let sum = x.clone() + y.clone();
        let sub = x.clone() - y.clone();
        let mul = x.clone() * y.clone();
        for t in [&sum, &sub, &mul] {
            assert!(t.clone().into_primitive().computed_on_device());
        }
        assert_eq!(tensor_traffic().downloads, before.downloads);
        for (got, op) in [
            (sum, i32::wrapping_add as fn(i32, i32) -> i32),
            (sub, i32::wrapping_sub),
            (mul, i32::wrapping_mul),
        ] {
            assert_eq!(
                got.into_data().to_vec::<i32>().unwrap(),
                a.iter()
                    .zip(&b)
                    .map(|(&x, &y)| op(x, y))
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            (x.clone() * -16777217).into_data().to_vec::<i32>().unwrap(),
            a.iter()
                .map(|&v| v.wrapping_mul(-16777217))
                .collect::<Vec<_>>()
        );
        let row: Vec<_> = (0..70i32).map(|i| i.wrapping_mul(0x61c88647)).collect();
        let r = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(row.clone(), [1, 70]), &d);
        assert_eq!(
            (x.clone() * r).into_data().to_vec::<i32>().unwrap(),
            a.iter()
                .enumerate()
                .map(|(i, &v)| v.wrapping_mul(row[i % 70]))
                .collect::<Vec<_>>()
        );
        let col: Vec<_> = (0..37).map(|i| -i - 1).collect();
        let c = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(col.clone(), [37, 1]), &d);
        assert_eq!(
            (x * c).into_data().to_vec::<i32>().unwrap(),
            a.iter()
                .enumerate()
                .map(|(i, &v)| v.wrapping_mul(col[i / 70]))
                .collect::<Vec<_>>()
        );
    });
}

#[test]
fn signed_comparisons_are_exact_beyond_float_integer_precision() {
    with_device(Config::default(), |d| {
        let a = vec![
            i32::MIN,
            i32::MAX,
            -1,
            0,
            16777216,
            16777217,
            -16777217,
            -16777216,
        ];
        let b = vec![
            i32::MAX,
            i32::MIN,
            0,
            -1,
            16777217,
            16777216,
            -16777216,
            -16777217,
        ];
        let x = Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(a.clone(), [8]), &d);
        let y = Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(b.clone(), [8]), &d);
        for (got, want) in [
            (
                x.clone().equal(y.clone()),
                a.iter().zip(&b).map(|(a, b)| a == b).collect::<Vec<_>>(),
            ),
            (
                x.clone().greater(y.clone()),
                a.iter().zip(&b).map(|(a, b)| a > b).collect(),
            ),
            (
                x.clone().greater_equal(y.clone()),
                a.iter().zip(&b).map(|(a, b)| a >= b).collect(),
            ),
            (
                x.clone().lower(y.clone()),
                a.iter().zip(&b).map(|(a, b)| a < b).collect(),
            ),
            (
                x.clone().lower_equal(y),
                a.iter().zip(&b).map(|(a, b)| a <= b).collect(),
            ),
            (
                x.clone().equal_elem(16777217),
                a.iter().map(|&a| a == 16777217).collect(),
            ),
            (x.greater_elem(-1), a.iter().map(|&a| a > -1).collect()),
        ] {
            assert!(got.clone().into_primitive().computed_on_device());
            assert_eq!(got.into_data().to_vec::<bool>().unwrap(), want);
        }
    });
}

#[test]
fn bitwise_and_wrapping_signed_shifts_match_i32() {
    with_device(Config::default(), |d| {
        let a = vec![i32::MIN, i32::MAX, -1, 0, 0x12345678, -12345, 1, -2];
        let b = vec![0, 1, 31, 32, 33, -1, i32::MIN, i32::MAX];
        let x = Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(a.clone(), [8]), &d);
        let y = Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(b.clone(), [8]), &d);
        for (got, want) in [
            (
                x.clone().bitwise_and(y.clone()),
                a.iter().zip(&b).map(|(&a, &b)| a & b).collect::<Vec<_>>(),
            ),
            (
                x.clone().bitwise_or(y.clone()),
                a.iter().zip(&b).map(|(&a, &b)| a | b).collect(),
            ),
            (
                x.clone().bitwise_xor(y.clone()),
                a.iter().zip(&b).map(|(&a, &b)| a ^ b).collect(),
            ),
            (
                x.clone().bitwise_left_shift(y.clone()),
                a.iter()
                    .zip(&b)
                    .map(|(&a, &b)| a.wrapping_shl(b as u32))
                    .collect(),
            ),
            (
                x.clone().bitwise_right_shift(y),
                a.iter()
                    .zip(&b)
                    .map(|(&a, &b)| a.wrapping_shr(b as u32))
                    .collect(),
            ),
            (x.clone().bitwise_not(), a.iter().map(|&a| !a).collect()),
            (
                x.bitwise_right_shift_scalar(33),
                a.iter().map(|&a| a.wrapping_shr(33)).collect(),
            ),
        ] {
            assert_eq!(got.into_data().to_vec::<i32>().unwrap(), want);
        }
    });
}
