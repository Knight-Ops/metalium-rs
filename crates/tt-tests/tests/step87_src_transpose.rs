//! Explicit Src conversion + transpose; raw copies retain their own contract.
use tt_isa::tile::{bf16_to_fp32, fp32_to_bf16_truncate, fp32_to_tf32};
use tt_kernels::{
    matmul::SrcRoute,
    session::{Session, TileChoice},
};

#[test]
fn src_transpose_face_converts_and_replays_changed_values() {
    tt_ttsim::fork_scope(|| {
        #[cfg(not(feature = "silicon"))]
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        #[cfg(not(feature = "silicon"))]
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        #[cfg(not(feature = "silicon"))]
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
            |_, _| Ok(None),
        )
        .unwrap();
        #[cfg(feature = "silicon")]
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let routes = if cfg!(feature = "silicon") {
            vec![SrcRoute::Tf32FromFp32, SrcRoute::Bf16FromFp32]
        } else {
            vec![SrcRoute::Tf32FromFp32]
        };
        for route in routes {
            let bits: Vec<_> = (0..256)
                .map(|i| ((i as f32 - 128.0) / 7.0).to_bits())
                .collect();
            let values: Vec<_> = bits.iter().copied().map(f32::from_bits).collect();
            let input = s.upload(&values, 16, 16).unwrap();
            s.begin_trace().unwrap();
            let out = s.transpose_src_block(&input, route).unwrap();
            let trace = s.end_trace().unwrap();
            for round in 0..3 {
                let mut values: Vec<_> = (0..256)
                    .map(|i| ((i as f32 - 128.0) / 7.0) + (round as f32 / 3.0))
                    .collect();
                if round == 0 {
                    let specials = [
                        0, 0x80000000, 0x00010000, 0x80010000, 0x7f800000, 0xff800000, 0x7fc10000,
                        0xffc10000,
                    ];
                    for (i, bits) in specials.into_iter().enumerate() {
                        values[i] = f32::from_bits(bits);
                    }
                    s.write(&input, &values).unwrap();
                    s.replay(trace).unwrap();
                }
                if round != 0 {
                    s.write(&input, &values).unwrap();
                    s.replay(trace).unwrap();
                }
                let got: Vec<_> = s
                    .download(&out)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                let want: Vec<_> = (0..256)
                    .map(|i| {
                        let bits = values[i % 16 * 16 + i / 16].to_bits();
                        if bits & 0x7f800000 == 0 {
                            return 0;
                        }
                        match route {
                            SrcRoute::Tf32FromFp32 => fp32_to_tf32(bits),
                            SrcRoute::Bf16FromFp32 => bf16_to_fp32(fp32_to_bf16_truncate(bits)),
                            _ => unreachable!(),
                        }
                    })
                    .collect();
                for (i, (&got, &want)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        got, want,
                        "route {route:?}, round {round}, datum {i}: {got:08x} vs {want:08x}"
                    );
                }
            }
            s.release_trace(trace).unwrap();
            if route == SrcRoute::Tf32FromFp32 {
                let a = s.upload(&vec![1.0; 16 * 16], 16, 16).unwrap();
                let mapping: Vec<_> = (0..256).map(|i| [i % 16, i / 16]).collect();
                let raw = s.repack(&input, &mapping, [16, 16]).unwrap();
                let reference = s
                    .matmul_dram(
                        &a,
                        false,
                        &raw,
                        false,
                        route,
                        tt_kernels::matmul::Fidelity::HiFi4,
                        4_000_000,
                    )
                    .unwrap();
                let reference_values = s.download(&reference).unwrap();
                s.free(reference).unwrap();
                s.free(raw).unwrap();
                s.begin_trace().unwrap();
                let product = s
                    .matmul_dram(
                        &a,
                        false,
                        &input,
                        true,
                        route,
                        tt_kernels::matmul::Fidelity::HiFi4,
                        4_000_000,
                    )
                    .unwrap();
                let product_trace = s.end_trace().unwrap();
                assert!(s
                    .trace_ops(product_trace)
                    .unwrap()
                    .iter()
                    .any(|op| op.what.contains("Src transpose")));
                // Each output column sums one row of B, after Src truncation.
                let current: Vec<_> = (0..256)
                    .map(|i| ((i as f32 - 128.0) / 7.0) + 2.0 / 3.0)
                    .collect();
                let got = s.download(&product).unwrap();
                assert_eq!(
                    got, reference_values,
                    "Src preparation must preserve native matmul results"
                );
                for row in 0..16 {
                    for col in 0..16 {
                        let values = &current[col * 16..col * 16 + 16];
                        let sum: f64 = values.iter().map(|&x| x as f64).sum();
                        let magnitude: f64 = values.iter().map(|&x| (x as f64).abs()).sum();
                        // Step68's TF32 operand and four-phase accumulation
                        // bound also covers fractional/cancelling products.
                        let gamma = 64.0 * 2f64.powi(-23) / (1.0 - 64.0 * 2f64.powi(-23));
                        let bound = (2f64.powi(-9) + (1.0 + 2f64.powi(-9)) * gamma) * magnitude;
                        assert!((got[row * 16 + col] as f64 - sum).abs() <= bound);
                    }
                }
                s.write(&input, &vec![2.0; 256]).unwrap();
                s.replay(product_trace).unwrap();
                assert_eq!(s.download(&product).unwrap(), vec![32.0; 256]);
                s.release_trace(product_trace).unwrap();
                s.free(product).unwrap();
                s.free(a).unwrap();
            }
            s.free(out).unwrap();
            s.free(input).unwrap();
        }
    })
    .unwrap();
}
