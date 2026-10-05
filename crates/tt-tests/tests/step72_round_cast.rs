//! Deterministic rounding is checked against independent host operations, not
//! the SFPSTOCHRND modes whose tie and small-integer behavior differs.
use burn::tensor::{Tensor, TensorData};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{with_device, Config};

#[test]
fn deterministic_rounding_and_saturating_cast_cover_bit_patterns() {
    with_device(Config::default(), |d| {
        let mut bits = vec![
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000, 0xff800000,
            0x7fc00001, 0xffc00001, 0x7f800001, 0x4effffff, 0x4f000000, 0x4f000001, 0xcf000000,
            0xcf000001, 0x3effffff, 0x3f000000, 0x3f000001, 0x3f7ffffe, 0x3f7fffff, 0x3fffffff,
            0x4affffff, 0x4b000000, 0x7f7fffff,
        ];
        for x in [-3.5f32, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5] {
            bits.extend([x.to_bits() - 1, x.to_bits(), x.to_bits() + 1]);
        }
        bits.extend((0..4096u32).map(|i| i.wrapping_mul(0x9e3779b9)));
        let values: Vec<_> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let x =
            Tensor::<TtBackend, 1>::from_data(TensorData::new(values.clone(), [values.len()]), &d)
                .to_device(&d);
        let before = tensor_traffic();
        let outputs = [
            x.clone().round(),
            x.clone().floor(),
            x.clone().ceil(),
            x.clone().trunc(),
        ];
        let cast = x.int();
        assert!(cast.clone().into_primitive().computed_on_device());
        assert_eq!(tensor_traffic().downloads, before.downloads);
        for (got, oracle) in outputs.into_iter().zip([
            f32::round_ties_even as fn(f32) -> f32,
            f32::floor,
            f32::ceil,
            f32::trunc,
        ]) {
            for (&input, got) in values.iter().zip(got.into_data().to_vec::<f32>().unwrap()) {
                let expected = if input.is_nan() { input } else { oracle(input) };
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "input bits {:08x}",
                    input.to_bits()
                );
            }
        }
        assert_eq!(
            cast.into_data().to_vec::<i32>().unwrap(),
            values.iter().map(|&x| x as i32).collect::<Vec<_>>()
        );
    });
}

#[test]
fn i32_to_float_preserves_extremes_and_rounds_large_magnitudes() {
    use burn::tensor::Int;
    with_device(Config::default(), |d| {
        let data = vec![
            i32::MIN,
            i32::MAX,
            0,
            -1,
            16777215,
            16777216,
            16777217,
            -16777217,
        ];
        let x = Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(data.clone(), [8]), &d);
        let got = x.float().into_data().to_vec::<f32>().unwrap();
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            data.iter()
                .map(|&x| (x as f32).to_bits())
                .collect::<Vec<_>>()
        );
    });
}
