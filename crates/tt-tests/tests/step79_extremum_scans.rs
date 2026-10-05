//! Inclusive raw-total-order scans, with bit-exact independent prefixes.
use tt_kernels::{
    session::{Session, TileChoice},
    sfpu::scan::ScanOp,
};

#[test]
fn min_max_scans_preserve_special_bits_and_continue_across_tiles() {
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
        let (rows, cols) = (67, 35);
        let specials = [
            0x80000000, 0, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321,
            0x7fffffff, 0xffffffff, 0x7f7fffff, 0xff7fffff,
        ];
        let bits: Vec<_> = (0..rows * cols)
            .map(|i| {
                if i / cols < 33 {
                    ((i % 17) as f32 - 8.0).to_bits()
                } else {
                    specials[(i / cols + i % cols) % specials.len()]
                }
            })
            .collect();
        let input = s
            .upload_bits(&bits, rows, cols, tt_kernels::tensor::Elem::F32)
            .unwrap();
        for op in [ScanOp::Min, ScanOp::Max] {
            let mut want = bits.clone();
            for c in 0..cols {
                let mut acc = bits[c];
                for r in 0..rows {
                    let x = bits[r * cols + c];
                    let cmp = f32::from_bits(x).total_cmp(&f32::from_bits(acc));
                    if (op == ScanOp::Min && cmp.is_lt()) || (op == ScanOp::Max && cmp.is_gt()) {
                        acc = x;
                    }
                    want[r * cols + c] = acc;
                }
            }
            assert_ne!(want, bits, "identity-copy negative control");
            let out = s.scan(&input, op).unwrap();
            assert_eq!(s.download_bits(&out).unwrap(), want, "{op:?}");
            s.free(out).unwrap();
        }
        assert_eq!(s.download_bits(&input).unwrap(), bits);
        // Metadata generation crosses the 200-entry continuation boundary and
        // physical faces, preserving all raw words and zero ragged padding.
        let constants = s
            .metadata(&bits, [rows, cols], tt_kernels::tensor::Elem::I32)
            .unwrap();
        assert_eq!(s.download_bits(&constants).unwrap(), bits);
        assert_eq!(constants.pad(), tt_kernels::tensor::Pad::Zero);
        s.free(constants).unwrap();
        assert!(s
            .metadata(&[], [0, 1], tt_kernels::tensor::Elem::I32)
            .is_err());
        s.free(input).unwrap();
    })
    .unwrap();
}
