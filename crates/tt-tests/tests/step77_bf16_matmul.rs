//! Packed BF16 operands reach the matrix unit directly; no narrowing pack.
use tt_kernels::{
    matmul::Fidelity,
    session::{Session, TileChoice},
};
use tt_ttsim::fork_scope;

#[test]
fn packed_bf16_matmul_masks_local_padding_and_preserves_parent_bits() {
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
        for (m, k, n) in [(3, 7, 5), (37, 65, 35), (32, 784, 128), (3, 4097, 5)] {
            let values = |count, seed| {
                (0..count)
                    .map(|i| ((i * seed) % 7) as f32 - 3.0)
                    .collect::<Vec<_>>()
            };
            let (a, b) = (values(m * k, 3), values(k * n, 5));
            let abits: Vec<_> = a.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let bbits: Vec<_> = b.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let ta = s.upload_bf16(&abits, m, k).unwrap();
            let tb = s.upload_bf16(&bbits, k, n).unwrap();
            let window = s
                .device()
                .alloc_window(tt_device::tlb::WindowKind::FourGib)
                .unwrap();
            // Poison every ragged padding datum. The gather must repair only
            // its L1 copy, and zeroing the parent would fail the bit check.
            let mut poisoned = Vec::new();
            for (t, rows, cols) in [(&ta, m, k), (&tb, k, n)] {
                for tile in 0..t.tile_count() {
                    let slot = t.slot(tile);
                    let mut image = vec![0u8; 2112];
                    s.device().dram_read(&window, slot, &mut image).unwrap();
                    for r in 0..32 {
                        for c in 0..32 {
                            if tile / cols.div_ceil(32) * 32 + r >= rows
                                || tile % cols.div_ceil(32) * 32 + c >= cols
                            {
                                let at = 16 + tt_isa::dm::face_index(r, c) * 2;
                                image[at..at + 2].copy_from_slice(&0x7fc1u16.to_le_bytes());
                            }
                        }
                    }
                    s.device().dram_write(&window, slot, &image).unwrap();
                    poisoned.push((slot, image));
                }
            }
            for limit in [
                None,
                std::num::NonZeroUsize::new(1),
                std::num::NonZeroUsize::new(2),
            ] {
                s.set_matmul_k_block_limit(limit);
                let out = s
                    .matmul_bf16(&ta, &tb, Fidelity::HiFi4, 40_000_000)
                    .unwrap();
                let got = s.download(&out).unwrap();
                let want: Vec<_> = (0..m)
                    .flat_map(|r| {
                        let (a, b) = (&a, &b);
                        (0..n)
                            .map(move |c| (0..k).map(|q| a[r * k + q] * b[q * n + c]).sum::<f32>())
                    })
                    .collect();
                assert_ne!(want, vec![0.0; m * n], "cleared output negative control");
                assert_eq!(got, want, "packed K limit {limit:?}");
                s.free(out).unwrap();
            }
            s.set_matmul_k_block_limit(None);
            for (slot, want) in poisoned {
                let mut got = vec![0u8; 2112];
                s.device().dram_read(&window, slot, &mut got).unwrap();
                assert_eq!(got, want, "local padding repair changed parent storage");
            }
            drop(window);
            assert_eq!(s.download_bf16(&ta).unwrap(), abits);
            assert_eq!(s.download_bf16(&tb).unwrap(), bbits);
            s.free_bf16(ta).unwrap();
            s.free_bf16(tb).unwrap();
        }
        let values: Vec<_> = (0..70).map(|i| (i % 13) as f32 * 0.25).collect();
        let bits: Vec<_> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let input = s.upload_bf16(&bits, 2, 35).unwrap();
        let windows: Vec<Vec<_>> = [1usize, 3, 17, 35]
            .into_iter()
            .enumerate()
            .map(|(i, count)| (0..count).map(|j| [(j + i) / 35, (j + i) % 35]).collect())
            .collect();
        let divisors = [1usize, 4, 32, 64];
        let output = s
            .bf16_pool_windows(&input, &windows, &divisors, [2, 2])
            .unwrap();
        // Each product, sum and power-of-two division is an exact binary
        // rational. The 17/35-datum windows require continuation across chunks.
        let want: Vec<_> = windows
            .iter()
            .zip(divisors)
            .map(|(window, divisor)| {
                window.iter().map(|[r, c]| values[r * 35 + c]).sum::<f32>() / divisor as f32
            })
            .collect();
        assert_eq!(s.download(&output).unwrap(), want);
        assert_eq!(s.download_bf16(&input).unwrap(), bits);
        s.free(output).unwrap();
        s.free_bf16(input).unwrap();
    }) {
        panic!("{e}");
    }
}
