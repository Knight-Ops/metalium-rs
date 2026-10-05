//! Physical BF16 storage, conversion and direct FPU pooling.
//! Pinned ttsim refuses the packer narrowing mode (0x105).
#![cfg(feature = "silicon")]

use tt_device::tlb::WindowKind;
use tt_kernels::session::{Session, TileChoice};
use tt_ttsim::fork_scope;

fn reference(bits: u32) -> u16 {
    if f32::from_bits(bits).is_nan() {
        return ((bits >> 16) as u16) | 0x40;
    }
    if !f32::from_bits(bits).is_finite() {
        return (bits >> 16) as u16;
    }
    let sign = (bits >> 16) as u16 & 0x8000;
    let value = f32::from_bits(bits).abs() as f64;
    let exponent = ((bits >> 23) & 255) as i32 - 127;
    let quantum = 2f64.powi(exponent.max(-126) - 7);
    let rounded = (value / quantum).round_ties_even() * quantum;
    // Blackhole late pack narrowing flushes BF16 subnormals.
    if rounded < 2f64.powi(-126) {
        sign
    } else {
        sign | ((rounded as f32).to_bits() >> 16) as u16
    }
}

#[test]
fn physically_packed_bf16_rounds_even_and_widens_on_tensix() {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let (rows, cols) = (37, 70);
        let specials = [
            0, 0x80000000, 0x00010000, 0x80010000, 1, 0x80000001, 0x007fffff, 0x807fffff,
            0x7f800000, 0xff800000, 0x7f800001, 0xff800001, 0x7fc12345, 0x7f7fffff, 0xff7fffff,
            0x3f808000, 0x3f818000,
        ];
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| {
                f32::from_bits(if i < specials.len() {
                    specials[i]
                } else {
                    ((i as f32 - 1300.0) / 64.0).to_bits() | 0x8000
                })
            })
            .collect();
        let input = s.upload(&values, rows, cols).unwrap();
        let packed = s.bf16_from_f32(&input).unwrap();
        s.sync().unwrap();
        assert_eq!(packed.slot(0).len(), 2112);
        assert!(packed.slot(0).len() < input.slot(0).len());
        let window = s.device().alloc_window(WindowKind::FourGib).unwrap();
        // Inspect every physical datum, including zero padding. This catches
        // pretending BF16 is an F32 allocation or returning cached host data.
        let ct = cols.div_ceil(32);
        for t in 0..packed.tile_count() {
            let range = packed.slot(t);
            let range = range.channel().range(range.offset() + 16, 2048).unwrap();
            let mut bytes = [0u8; 2048];
            s.device().dram_read(&window, range, &mut bytes).unwrap();
            for i in 0..1024 {
                let face = i / 256;
                let r = t / ct * 32 + face / 2 * 16 + i % 256 / 16;
                let c = t % ct * 32 + face % 2 * 16 + i % 16;
                let expected = if r < rows && c < cols {
                    reference(values[r * cols + c].to_bits())
                } else {
                    0
                };
                let got = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                assert_eq!(got, expected, "physical tile {t}, datum {i}, ({r},{c})");
            }
        }
        drop(window);
        let widened = s.bf16_to_f32(&packed).unwrap();
        let got = s.download(&widened).unwrap();
        for (i, (&got, value)) in got.iter().zip(&values).enumerate() {
            assert_eq!(
                got.to_bits(),
                (reference(value.to_bits()) as u32) << 16,
                "widen datum {i}"
            );
        }
        s.free(widened).unwrap();
        s.free_bf16(packed).unwrap();
        s.free(input).unwrap();
        let values: Vec<_> = (0..256).map(|i| ((i * 13) % 31) as f32 - 20.0).collect();
        let input = s.upload(&values, 16, 16).unwrap();
        let packed = s.bf16_from_f32(&input).unwrap();
        for op in [
            tt_kernels::fpu::PoolOp::Max,
            tt_kernels::fpu::PoolOp::Sum,
            tt_kernels::fpu::PoolOp::Mean,
        ] {
            let out = s.bf16_pool_block(&packed, op).unwrap();
            let got = s.download(&out).unwrap();
            let want: Vec<_> = (0..16)
                .map(|c| match op {
                    tt_kernels::fpu::PoolOp::Max => (0..16)
                        .map(|r| values[r * 16 + c])
                        .fold(f32::NEG_INFINITY, f32::max),
                    _ => {
                        (0..16).map(|r| values[r * 16 + c]).sum::<f32>()
                            / if op == tt_kernels::fpu::PoolOp::Mean {
                                16.0
                            } else {
                                1.0
                            }
                    }
                })
                .collect();
            assert_eq!(got, want, "direct BF16 {op:?}");
            s.free(out).unwrap();
        }
        s.free_bf16(packed).unwrap();
        s.free(input).unwrap();
    }) {
        panic!("{e}");
    }
}
