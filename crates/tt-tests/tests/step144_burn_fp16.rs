//! Phase 10 gate (D1): FP16 through Burn.
//!
//! `DType::F16` is IEEE binary16 in BF16's two-byte physical slots: a storage and
//! cast dtype. Raw views and uploads keep every bit pattern and move two bytes per
//! element; `float_cast` converts F32, BF16 and F16 on Tensix (`step143` holds the
//! conversions bit for bit); arithmetic on F16 is not claimed and fails naming the
//! operation and dtype; F64 stays unsupported. Everything here runs on ttsim except
//! the BF16 narrowing arm (ttsim refuses mode `0x105`, divergence row 29), which is
//! silicon-only.
//!
//! Residency: every device result is checked `computed_on_device()` and the gates
//! assert `tensor_traffic().downloads` did not move before the explicit readback.
//! Negative controls: the BF16 reading of F16 slots differs from the F16 one, a
//! truncating oracle differs from the device on the corpus's ties, and F16
//! arithmetic must be refused (a silent F32 fallback would not panic).
use burn::tensor::{backend::Backend, DType, FloatDType, Tensor, TensorData, TensorPrimitive};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{assert_native_model, with_device, Config};

mod fp16_support;
use fp16_support::{f32_corpus, oracle_to_f16, oracle_to_f32, truncating_to_f16};

fn data16(bits: Vec<u16>, shape: impl Into<burn::tensor::Shape>, dtype: DType) -> TensorData {
    let mut data = TensorData::new(bits, shape);
    data.dtype = dtype;
    data
}

fn bits16(data: TensorData, dtype: DType) -> Vec<u16> {
    assert_eq!(data.dtype, dtype);
    let mut data = data;
    data.dtype = DType::U16;
    data.to_vec().unwrap()
}

fn resident<const D: usize>(t: &Tensor<TtBackend, D>) {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(t) => assert!(t.computed_on_device()),
        _ => panic!("unexpected quantized tensor"),
    }
}

/// F32 bits of a tensor read back.
fn f32_bits<const D: usize>(t: Tensor<TtBackend, D>) -> Vec<u32> {
    t.into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|f| f.to_bits())
        .collect()
}

#[test]
fn raw_f16_views_preserve_all_payloads_and_use_two_byte_transfers() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (37, 70);
        // Every payload class: NaNs, infinities, subnormals, signed zeros.
        let values: Vec<u16> = (0..rows * cols)
            .map(|i| (i as u16).wrapping_mul(113))
            .chain([0x7c00, 0xfc00, 0x7e01, 0xffff, 0x0001, 0x8000])
            .take(rows * cols)
            .collect();
        let before = tensor_traffic();
        let input = Tensor::<TtBackend, 2>::from_data(
            data16(values.clone(), [rows, cols], DType::F16),
            (&d, DType::F16),
        )
        .to_device(&d);
        assert_eq!(
            (tensor_traffic() - before).uploaded,
            (rows * cols * 2) as u64,
            "two bytes per element"
        );
        let (output, report) = burn_tt::with_report(|| {
            let output = input.clone().transpose().reshape([1, rows * cols]);
            resident(&output);
            let flipped = input.clone().flip([0, 1]);
            resident(&flipped);
            let tail = input.clone().slice([32..37, 0..cols]);
            resident(&tail);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            (output, flipped, tail)
        });
        assert_native_model(&report);
        let want: Vec<_> = (0..cols)
            .flat_map(|c| {
                let values = &values;
                (0..rows).map(move |r| values[r * cols + c])
            })
            .collect();
        assert_eq!(bits16(output.0.into_data(), DType::F16), want);
        assert_eq!(
            bits16(output.1.into_data(), DType::F16),
            values.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            bits16(output.2.into_data(), DType::F16),
            values[32 * cols..]
        );
        assert_eq!(
            bits16(input.into_data(), DType::F16),
            values,
            "views must not modify their parent"
        );
    });
}

#[test]
fn widening_cast_matches_the_oracle_and_stays_resident() {
    with_device(Config::default(), |d| {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let x = Tensor::<TtBackend, 2>::from_data(
            data16(all.clone(), [256, 256], DType::F16),
            (&d, DType::F16),
        )
        .to_device(&d);
        let before = tensor_traffic();
        let (wide, report) = burn_tt::with_report(|| x.clone().cast(FloatDType::F32));
        assert_native_model(&report);
        resident(&wide);
        assert_eq!(
            tensor_traffic().downloads,
            before.downloads,
            "the cast reads nothing back"
        );
        let got = f32_bits(wide);
        for (h, g) in all.iter().zip(&got) {
            assert_eq!(*g, oracle_to_f32(*h), "pattern {h:#06x}");
        }
        // Control: the BF16 reading of the same slots is not the F16 one.
        let as_bf16 = Tensor::<TtBackend, 2>::from_data(
            data16(all.clone(), [256, 256], DType::BF16),
            (&d, DType::BF16),
        )
        .to_device(&d)
        .cast(FloatDType::F32);
        let differing = f32_bits(as_bf16)
            .iter()
            .zip(&all)
            .filter(|(g, h)| **g != oracle_to_f32(**h))
            .count();
        assert!(differing > 60000, "{differing}");
    });
}

/// A ragged producer followed by a reduction: the widened tensor's padding is
/// undefined, and a maximum must not see it.
#[test]
fn widened_ragged_tensor_feeds_a_reduction() {
    with_device(Config::default(), |d| {
        let (rows, cols) = (37, 70);
        let values: Vec<u16> = (0..rows * cols)
            .map(|i| {
                let h = ((i * 40503 + 11) % 65536) as u16;
                if (h >> 10) & 0x1f == 31 {
                    h & 0xbfff
                } else {
                    h
                }
            })
            .collect();
        let x = Tensor::<TtBackend, 2>::from_data(
            data16(values.clone(), [rows, cols], DType::F16),
            (&d, DType::F16),
        )
        .to_device(&d);
        let before = tensor_traffic();
        let ((rowmax, colmax), report) = burn_tt::with_report(|| {
            let wide = x.cast(FloatDType::F32);
            (wide.clone().max_dim(1), wide.max_dim(0))
        });
        assert_native_model(&report);
        resident(&rowmax);
        resident(&colmax);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let value = |r: usize, c: usize| f32::from_bits(oracle_to_f32(values[r * cols + c]));
        let rows_want: Vec<f32> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| value(r, c))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        let cols_want: Vec<f32> = (0..cols)
            .map(|c| {
                (0..rows)
                    .map(|r| value(r, c))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        assert_eq!(rowmax.into_data().to_vec::<f32>().unwrap(), rows_want);
        assert_eq!(colmax.into_data().to_vec::<f32>().unwrap(), cols_want);
    });
}

#[test]
fn narrowing_casts_match_the_oracle_from_f32_int_bool_and_bf16() {
    use burn::tensor::{Bool, Int};
    with_device(Config::default(), |d| {
        let (rows, cols) = (61, 67);
        let bits = f32_corpus(rows * cols);
        let values: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(values, [rows, cols]), &d)
            .to_device(&d);
        let before = tensor_traffic();
        let (narrow, report) = burn_tt::with_report(|| x.cast(FloatDType::F16));
        assert_native_model(&report);
        resident(&narrow);
        assert_eq!(tensor_traffic().downloads, before.downloads);
        let got = bits16(narrow.clone().into_data(), DType::F16);
        for (i, (g, &b)) in got.iter().zip(&bits).enumerate() {
            assert_eq!(*g, oracle_to_f16(b), "{i}: input {b:#010x}");
        }
        assert!(
            bits.iter()
                .zip(&got)
                .any(|(&b, &g)| truncating_to_f16(b) != g),
            "control: truncation would not pass this corpus"
        );
        // The round trip is the oracle's.
        let back = f32_bits(narrow.cast(FloatDType::F32));
        for (i, (g, &b)) in back.iter().zip(&bits).enumerate() {
            assert_eq!(*g, oracle_to_f32(oracle_to_f16(b)), "{i}: {b:#010x}");
        }

        // Integers and booleans reach F16 through the device's own casts.
        let ints = Tensor::<TtBackend, 1, Int>::from_data([3, -2, 0, 2049, 70000, -65520], &d);
        let bools = Tensor::<TtBackend, 1, Bool>::from_data([true, false, true], &d);
        let ((ints, bools), report) =
            burn_tt::with_report(|| (ints.cast(FloatDType::F16), bools.cast(FloatDType::F16)));
        assert_native_model(&report);
        resident(&ints);
        resident(&bools);
        assert_eq!(
            bits16(ints.into_data(), DType::F16),
            [3.0f32, -2.0, 0.0, 2049.0, 70000.0, -65520.0].map(|v| oracle_to_f16(v.to_bits()))
        );
        assert_eq!(bits16(bools.into_data(), DType::F16), [0x3c00, 0, 0x3c00]);

        // BF16 -> F16 widens on Tensix, then narrows: every BF16 pattern.
        let bf: Vec<u16> = (0..=u16::MAX).collect();
        let x = Tensor::<TtBackend, 2>::from_data(
            data16(bf.clone(), [256, 256], DType::BF16),
            (&d, DType::BF16),
        )
        .to_device(&d);
        let out = x.cast(FloatDType::F16);
        resident(&out);
        let got = bits16(out.into_data(), DType::F16);
        for (b, g) in bf.iter().zip(&got) {
            assert_eq!(*g, oracle_to_f16((*b as u32) << 16), "bf16 {b:#06x}");
        }
    });
}

/// Silicon-only: F16 -> BF16 needs the BF16 narrowing pack (mode `0x105`).
#[cfg(feature = "silicon")]
#[test]
fn f16_to_bf16_widens_then_narrows_on_the_device() {
    with_device(Config::default(), |d| {
        let all: Vec<u16> = (0..=u16::MAX)
            .filter(|h| (h >> 10) & 0x1f != 31)
            .take(256 * 255)
            .collect();
        assert_eq!(
            all.len() % 256,
            0,
            "the finite F16 patterns fill whole rows"
        );
        let x = Tensor::<TtBackend, 2>::from_data(
            data16(all.clone(), [all.len() / 256, 256], DType::F16),
            (&d, DType::F16),
        )
        .to_device(&d);
        let out = x.cast(FloatDType::BF16);
        resident(&out);
        let got = bits16(out.into_data(), DType::BF16);
        for (h, g) in all.iter().zip(&got) {
            // BF16 narrowing rounds ties-even and flushes subnormals (row 29's
            // silicon measurement); the F16 values here are exact in BF16 range.
            let value = oracle_to_f32(*h);
            let want = if value & 0x7fff_ffff < 0x0080_0000 {
                (value >> 16) as u16 & 0x8000
            } else {
                let lsb = (value >> 16) & 1;
                (value.wrapping_add(0x7fff + lsb) >> 16) as u16
            };
            assert_eq!(*g, want, "f16 {h:#06x}");
        }
    });
}

#[test]
fn unsupported_cases_fail_explicitly_naming_the_operation_and_dtype() {
    with_device(Config::default(), |d| {
        let message = |f: &mut dyn FnMut()| -> String {
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            std::panic::set_hook(hook);
            match result {
                Ok(()) => panic!("expected an explicit failure, the operation succeeded"),
                Err(payload) => payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default(),
            }
        };
        let f32s = Tensor::<TtBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &d);
        let f16s = Tensor::<TtBackend, 2>::from_data(
            data16(vec![0x3c00; 4], [2, 2], DType::F16),
            (&d, DType::F16),
        );
        // F64 is `[-]`: the cast names the operation and both dtypes.
        let text = message(&mut || {
            let _ = f32s.clone().cast(FloatDType::F64);
        });
        assert!(
            text.contains("float_cast") && text.contains("F64"),
            "{text}"
        );
        let text = message(&mut || {
            let _ = f16s.clone().cast(FloatDType::F64);
        });
        assert!(
            text.contains("float_cast") && text.contains("F16") && text.contains("F64"),
            "{text}"
        );
        // Control for residency: arithmetic on F16 does not silently fall back
        // to the host or to another dtype.
        let text = message(&mut || {
            let _ = f16s.clone() + f16s.clone();
        });
        assert!(
            text.contains("unsupported operation") && text.contains("F16"),
            "{text}"
        );
    });
}

#[test]
fn dtype_usage_reports_f16_as_storage_and_f64_as_unsupported() {
    with_device(Config::default(), |d| {
        let usage = TtBackend::dtype_usage(&d, DType::F16);
        assert!(!usage.is_empty(), "F16 is stored and cast");
        // Storage only, exactly as I32: no arithmetic is claimed.
        assert_eq!(usage, TtBackend::dtype_usage(&d, DType::I32));
        assert!(!TtBackend::supports_dtype(&d, DType::F16));
        assert!(TtBackend::dtype_usage(&d, DType::F64).is_empty());
        assert!(
            TtBackend::supports_dtype(&d, DType::BF16),
            "BF16 is unchanged"
        );
    });
}
