//! Native prediction indices and resident Ethernet mesh compute.
use burn_flex::{Flex, FlexDevice};
use burn_tensor::{Tensor, TensorData};
use burn_tt::TtBackend;
use tt_tests::burn_device::{assert_native_model, with_device, with_mesh_device, Config};

#[test]
fn argmax_preserves_first_ties_nan_priority_and_ragged_edges() {
    with_device(Config::default(), |device| {
        for cols in [1, 10, 37, 65] {
            let mut values: Vec<_> = (0..35 * cols).map(|i| (i % 7) as f32 - 3.0).collect();
            values[..cols].fill(f32::NEG_INFINITY);
            values[cols..2 * cols].fill(f32::NAN);
            if cols > 1 {
                values[2 * cols + 1] = f32::NAN;
                values[3 * cols] = f32::INFINITY;
            }
            let data = TensorData::new(values, [35, cols]);
            for axis in [0, 1] {
                let want = Tensor::<Flex, 2>::from_data(data.clone(), &FlexDevice)
                    .argmax(axis)
                    .into_data()
                    .convert::<i32>();
                let (got, report) = burn_tt::with_report(|| {
                    Tensor::<TtBackend, 2>::from_data(data.clone(), &device)
                        .argmax(axis)
                        .reshape([if axis == 1 { 35 } else { cols }])
                        .into_data()
                });
                assert_eq!(
                    got.to_vec::<i32>().unwrap(),
                    want.to_vec::<i32>().unwrap(),
                    "cols={cols}, axis={axis}"
                );
                assert_native_model(&report);
            }
        }
        let (got, report) = burn_tt::with_report(|| {
            Tensor::<TtBackend, 1>::from_data([1.0, 7.0, 7.0], &device)
                .argmax(0)
                .into_data()
        });
        assert_eq!(got.to_vec::<i32>().unwrap(), [1]);
        assert_native_model(&report);
        let tiny = f32::from_bits(1);
        let data = TensorData::new(
            vec![0.0, tiny, -tiny, 0.0, -0.0, 0.0, 1.0, f32::INFINITY],
            [4, 2],
        );
        let want = Tensor::<Flex, 2>::from_data(data.clone(), &FlexDevice)
            .argmax(1)
            .into_data()
            .convert::<i32>();
        let (got, report) = burn_tt::with_report(|| {
            Tensor::<TtBackend, 2>::from_data(data, &device)
                .argmax(1)
                .reshape([4])
                .into_data()
        });
        assert_eq!(got.to_vec::<i32>().unwrap(), want.to_vec::<i32>().unwrap());
        assert_native_model(&report);
    });
}

fn mesh_products(chips: usize) {
    with_mesh_device(Config::default(), chips, |device| {
        let (m, k, n) = (35, 37, 225);
        let a = TensorData::new(
            (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect::<Vec<_>>(),
            [m, k],
        );
        let b = TensorData::new(
            (0..k * n).map(|i| (i % 7) as f32 - 3.0).collect::<Vec<_>>(),
            [k, n],
        );
        let want = Tensor::<Flex, 2>::from_data(a.clone(), &FlexDevice)
            .matmul(Tensor::from_data(b.clone(), &FlexDevice))
            .into_data();
        let (got, report) = burn_tt::with_report(|| {
            Tensor::<TtBackend, 2>::from_data(a, &device)
                .matmul(Tensor::from_data(b, &device))
                .into_data()
        });
        assert_eq!(got.as_bytes(), want.as_bytes(), "{chips} chips");
        assert_native_model(&report);
        let op = report.op("float_matmul").unwrap();
        assert_eq!((op.downloads, op.staged, op.on_host), (0, 0, 0));
    });
}

#[test]
fn two_chip_products_are_resident_and_exact() {
    mesh_products(2);
}

#[cfg(not(feature = "silicon"))]
#[test]
fn four_chip_products_are_resident_and_exact_through_relays() {
    mesh_products(4);
}

#[cfg(not(feature = "silicon"))]
#[test]
fn ethernet_slot_copies_preserve_bits_across_packets_and_relays() {
    use tt_device::Device;
    use tt_isa::noc::NocCoord;
    use tt_kernels::shard::{Chip, Fabric};
    use tt_kernels::tensor::Elem;
    tt_ttsim::fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open_path(tt_ttsim::x4_lib_path()).unwrap();
        let chips = sim
            .transports()
            .into_iter()
            .map(|t| {
                Chip::new(
                    Device::open(t).unwrap(),
                    NocCoord::new(3, 4).unwrap(),
                    NocCoord::new(6, 7).unwrap(),
                )
                .unwrap()
            })
            .collect();
        let mut fabric = Fabric::new(
            chips,
            &tt_tests::topology::links(&tt_tests::topology::BH_X4),
            tt_firmware_images::ROLES,
            tt_firmware_images::ETH_E1,
        )
        .unwrap();
        fabric
            .enable_resident(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        for shape in [[37, 35], [1025, 257]] {
            let patterns = [
                0,
                0x8000_0000,
                1,
                0x8000_0001,
                0x7fc1_2345,
                0xff80_0000,
                0x7fff_ffff,
            ];
            let bits = (0..shape[0] * shape[1])
                .map(|i| patterns[i % patterns.len()])
                .collect::<Vec<_>>();
            let input = fabric.chips[0]
                .session()
                .upload_bits(&bits, shape[0], shape[1], Elem::F32)
                .unwrap();
            for peer in [1, 2] {
                let remote = fabric.transfer_tensor(0, &input, peer).unwrap();
                let back = fabric.transfer_tensor(peer, &remote, 0).unwrap();
                assert_eq!(
                    fabric.chips[0].session().download_bits(&back).unwrap(),
                    bits
                );
                fabric.chips[0].session().free(back).unwrap();
                fabric.chips[peer].session().free(remote).unwrap();
            }
            fabric.chips[0].session().free(input).unwrap();
        }
    })
    .unwrap();
}
