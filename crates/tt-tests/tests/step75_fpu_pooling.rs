//! GMPOOL/GAPOOL are opt-in BF16 Src operations, never F32 replacements.
use tt_kernels::{
    fpu::PoolOp,
    session::{Session, TileChoice},
};
use tt_ttsim::fork_scope;

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
