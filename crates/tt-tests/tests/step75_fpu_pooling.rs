//! GMPOOL/GAPOOL are opt-in BF16 Src operations, never F32 replacements.
use tt_kernels::{
    fpu::PoolOp,
    session::{Session, TileChoice},
};
use tt_ttsim::fork_scope;

// Independent GMPOOL unit-scaling model: BF16 magnitudes compare as integer
// exponent/mantissa tuples; exponent zero flushes to +0. NaNs participate in
// that ordering. This intentionally differs from Burn's first-NaN selection.
fn gmpool_oracle(column: &[u32]) -> u32 {
    let visit = [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15];
    let mut best = i32::MIN;
    let mut result = 0;
    for r in visit {
        let b = column[r] >> 16;
        let exponent = (b >> 7) & 255;
        let (key, bits) = if exponent == 0 {
            (0, 0)
        } else {
            let magnitude = ((exponent + 127) * 1024 + (b & 127) * 8) as i32;
            (
                if b & 0x8000 == 0 {
                    magnitude
                } else {
                    -magnitude
                },
                b << 16,
            )
        };
        if key >= best {
            best = key;
            result = bits;
        }
    }
    result
}

#[test]
fn matrix_max_pool_specials_follow_magnitude_ordering() {
    if let Err(e) = fork_scope(|| {
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
        let patterns = [
            0, 0x80000000, 0x00010000, 0x80010000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
            0x7fc10000, 0xffc10000, 0x7f810000, 0xff810000,
        ];
        let mut bits = vec![0; 256];
        for r in 0..16 {
            for c in 0..16 {
                bits[r * 16 + c] = match c {
                    0 => 0x80000000,
                    1 => 0x80010000,
                    2 => 0xffc10000,
                    3 => {
                        if r == 12 {
                            0x7fc10000
                        } else {
                            0x7f800000
                        }
                    }
                    4 => {
                        if r == 3 || r == 12 {
                            0x40000000
                        } else {
                            0x3f800000
                        }
                    }
                    _ => patterns[(r + c) % patterns.len()],
                };
            }
        }
        let values: Vec<_> = bits.iter().copied().map(f32::from_bits).collect();
        let input = s.upload(&values, 16, 16).unwrap();
        let out = s.fpu_pool_block(&input, PoolOp::Max).unwrap();
        let got: Vec<_> = s
            .download(&out)
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let want: Vec<_> = (0..16)
            .map(|c| gmpool_oracle(&(0..16).map(|r| bits[r * 16 + c]).collect::<Vec<_>>()))
            .collect();
        assert_eq!(want[0], 0, "negative zero is flushed");
        assert_ne!(want[0], bits[0], "raw-copy semantics must fail this gate");
        assert_eq!(want[3], 0x7fc10000, "positive NaN orders above infinity");
        assert_eq!(got, want);
        s.free(out).unwrap();
        s.free(input).unwrap();
    }) {
        panic!("{e}");
    }
}

#[test]
fn matrix_pooling_uses_explicit_weights_and_releases_banks() {
    if let Err(e) = fork_scope(|| {
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
        for seed in [0usize, 1, 19, 31] {
            // Exact BF16 integers. In all four phases, every intermediate
            // product/sum is exact, giving an independent scalar oracle.
            let values: Vec<_> = (0..256)
                .map(|i| {
                    let value = ((i * 13 + seed) % 31) as f32 - 20.0;
                    if seed == 31 {
                        f32::from_bits((value / 16.0).to_bits() | 0x1234)
                    } else {
                        value
                    }
                })
                .collect();
            let truncated: Vec<_> = values
                .iter()
                .map(|v| f32::from_bits(v.to_bits() & 0xffff0000))
                .collect();
            let input = s.upload(&values, 16, 16).unwrap();
            for op in [PoolOp::Max, PoolOp::Sum, PoolOp::Mean, PoolOp::Max] {
                let out = s.fpu_pool_block(&input, op).unwrap();
                let got = s.download(&out).unwrap();
                let want: Vec<_> = (0..16)
                    .map(|c| match op {
                        PoolOp::Max => (0..16)
                            .map(|r| truncated[r * 16 + c])
                            .fold(f32::NEG_INFINITY, f32::max),
                        _ => {
                            (0..16).map(|r| truncated[r * 16 + c]).sum::<f32>()
                                / if op == PoolOp::Mean { 16.0 } else { 1.0 }
                        }
                    })
                    .collect();
                assert_ne!(
                    want,
                    vec![0.0; 16],
                    "negative control: cleared output cannot pass"
                );
                assert_eq!(got, want, "{op:?}, seed {seed}");
                s.free(out).unwrap();
            }
            s.free(input).unwrap();
        }
        let bad = s.upload(&[1.0; 17 * 16], 17, 16).unwrap();
        assert!(
            s.fpu_pool_block(&bad, PoolOp::Max).is_err(),
            "ragged lanes must not silently contribute"
        );
        s.free(bad).unwrap();
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
#[test]
fn matrix_max_pool_argmax_bit_does_not_publish_indices_on_packed_path() {
    if let Err(e) = fork_scope(|| {
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
        let patterns = [
            0, 0x80000000, 0x00010000, 0x80010000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000,
            0x7fc10000, 0xffc10000, 0x7f810000, 0xff810000,
        ];
        let mut bits = vec![0; 256];
        for r in 0..16 {
            for c in 0..16 {
                bits[r * 16 + c] = match c {
                    0 => 0x80000000,
                    1 => 0x80010000,
                    2 => 0xffc10000,
                    3 => {
                        if r == 12 {
                            0x7fc10000
                        } else {
                            0x7f800000
                        }
                    }
                    4 => {
                        if r == 3 || r == 12 {
                            0x40000000
                        } else {
                            0x3f800000
                        }
                    }
                    5..=12 => {
                        if r == c - 5 {
                            10.0f32.to_bits()
                        } else {
                            (-10.0f32).to_bits()
                        }
                    }
                    13 => {
                        if r == 12 {
                            100.0f32.to_bits()
                        } else {
                            (r as f32).to_bits()
                        }
                    }
                    14 => 1.0f32.to_bits(),
                    _ => patterns[(r + c) % patterns.len()],
                };
            }
        }
        let values: Vec<_> = bits.iter().copied().map(f32::from_bits).collect();
        let input = s.upload(&values, 16, 16).unwrap();
        let out = s.fpu_pool_block(&input, PoolOp::MaxIndexProbe).unwrap();
        let got = s.download_bits(&out).unwrap();
        for c in 0..16 {
            let column: Vec<_> = (0..16).map(|r| bits[r * 16 + c]).collect();
            assert_eq!(
                got[c] & 0xffff0000,
                gmpool_oracle(&column) & 0xffff0000,
                "value column {c}"
            );
            let mut best = i32::MIN;
            let mut winner = 0;
            for r in [4, 5, 6, 7, 0, 1, 2, 3] {
                let bf = column[r] >> 16;
                let exponent = (bf >> 7) & 255;
                let magnitude = ((exponent + 127) * 1024 + (bf & 127) * 8) as i32;
                let key = if exponent == 0 {
                    0
                } else if bf & 0x8000 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
                if key >= best {
                    best = key;
                    winner = r;
                }
            }
            let encoded = [0, 3, 6, 1, 4, 7, 2, 5][winner];
            assert_eq!(
                got[c] & 0xffff,
                0,
                "packed path exposes no index in column {c}"
            );
            println!(
                "GMPOOL encoded column {c}: {:08x}, first-eight row {winner}, documented nonlinear {encoded}",
                got[c]
            );
        }
        s.free(out).unwrap();
        s.free(input).unwrap();
    }) {
        panic!("{e}");
    }
}
