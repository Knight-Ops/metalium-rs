//! Fixed B-reader / resident-role / NC-writer ownership. The oracle is the
//! host, never another device schedule: burn-flex bit for bit on small-integer
//! operands, where every product and sum is exact in TF32, BF16 at any fidelity
//! and FP32 alike (as `step9_matmul`), and for every SFPU kind the host program
//! model (`sfpu::ops::reference_op`), which the per-kind gates hold to burn-flex.
//! Serialized against overlapped execution is an extra check, not the oracle.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_tiles(count: usize, test: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    fork_scope(|| {
        let mut simulator = tt_ttsim::Simulator::open().unwrap();
        let device = tt_device::Device::open(simulator.transport()).unwrap();
        let mut session = Session::open(
            device,
            tt_firmware_images::ROLES,
            TileChoice::Count(count),
            |_, _| Ok(None),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        session.set_pipeline(true);
        test(&mut session);
    })
    .unwrap();
}

#[cfg(feature = "silicon")]
fn with_tiles(count: usize, test: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    fork_scope(|| {
        let mut session = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(count),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        session.set_pipeline(true);
        test(&mut session);
    })
    .unwrap();
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 17 + seed) % 97) as f32 / 97.0 - 0.5)
        .collect()
}

/// Integers in `-7..=7`: products under 2^6, and the sums of the 256-term
/// products and their 256-row totals under 2^23, so exact everywhere.
fn ints(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * 17 + seed * 31 + index / 5) % 15) as f32 - 7.0)
        .collect()
}

fn flex(values: &[f32], rows: usize, cols: usize) -> Tensor<Flex, 2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [rows, cols]), &FlexDevice)
}

fn flex_values(tensor: Tensor<Flex, 2>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

/// burn-flex's `input^T @ input`, then its column sums (`Axis::Rows`).
fn flex_product_and_sum(input: &[f32], width: usize) -> [Vec<f32>; 2] {
    let a = flex(input, width, width);
    let product = a.clone().transpose().matmul(a);
    [
        flex_values(product.clone()),
        flex_values(product.sum_dim(0)),
    ]
}

fn add<T: tt_device::Transport>(session: &mut Session<T>, input: &DramTensor) -> DramTensor {
    session
        .eltwise(
            Eltwise {
                kind: kind::ADD,
                scalar: 0.0,
                scalar2: 0.0,
            },
            input,
            Some(input),
        )
        .unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Every SFPU kind, serialized, then captured and replayed, bit for bit to the
/// host program model. At 64x64 no op pipelines (`tensor::pipelined_runs`
/// wants 48 tiles per unit squared), so a 512x512 pass repeats one kind per
/// operand shape and element signature, asserting depth two actually ran.
#[test]
fn default_ownership_covers_every_sfpu_kind() {
    use std::collections::HashSet;
    use tt_kernels::sfpu::kernel::Operands;
    use tt_kernels::sfpu::ops::{self, kind_sfpu, reference_op, Broadcast};
    use tt_kernels::tensor::Elem;
    for count in [1, 2] {
        with_tiles(count, |session| {
            for width in [64, 512] {
                let data: Vec<f32> = (0..width * width)
                    .map(|index| 0.125 + (index % 7) as f32 / 16.0)
                    .collect();
                let floats = session.upload(&data, width, width).unwrap();
                let row = session.upload(&data[..width], 1, width).unwrap();
                let booleans = session
                    .upload_bits(&vec![1; width * width], width, width, Elem::Bool)
                    .unwrap();
                let integers = session
                    .upload_bits(&vec![2; width * width], width, width, Elem::I32)
                    .unwrap();
                let host_bool = vec![f32::from_bits(1); width * width];
                let host_int = vec![f32::from_bits(2); width * width];
                let mut classes = HashSet::new();
                let mut overlapped = 0;
                for operation in (1..=kind::LAST).chain(0x100..=kind_sfpu::LAST) {
                    let Some(operands) = ops::operands(operation) else {
                        continue;
                    };
                    let signature = ops::elems(operation);
                    if width > 64 && !classes.insert((operands, signature.inputs)) {
                        continue;
                    }
                    let input = |index: usize| match signature.inputs[index] {
                        Elem::F32 => &floats,
                        Elem::Bool => &booleans,
                        Elem::I32 => &integers,
                    };
                    let host = |index: usize| match signature.inputs[index] {
                        Elem::F32 => &data[..],
                        Elem::Bool => &host_bool[..],
                        Elem::I32 => &host_int[..],
                    };
                    let second = match operands {
                        Operands::Unary => None,
                        Operands::RowBroadcast => Some(&row),
                        _ => Some(input(1)),
                    };
                    let third = (operands == Operands::Ternary).then(|| input(2));
                    // A native row-broadcast kind's own program takes the row
                    // (`Broadcast::None` selects it; `Row` is for kinds that
                    // broadcast a general operand).
                    let mut inputs = match operands {
                        Operands::Unary => vec![host(0)],
                        Operands::RowBroadcast => vec![host(0), &data[..width]],
                        _ => vec![host(0), host(1)],
                    };
                    if operands == Operands::Ternary {
                        inputs.push(host(2));
                    }
                    let expected = bits(&reference_op(
                        operation,
                        [0.125, 0.5],
                        Broadcast::None,
                        &inputs,
                        width,
                        width,
                    ));
                    let op = Eltwise {
                        kind: operation,
                        scalar: 0.125,
                        scalar2: 0.5,
                    };
                    let what = format!("{count} tiles, {width}x{width}, operation {operation:#x}");
                    session.set_pipeline(false);
                    let serial = session.eltwise3(op, input(0), second, third).unwrap();
                    assert_eq!(
                        session.download_bits(&serial).unwrap(),
                        expected,
                        "{what}, serialized"
                    );
                    session.free(serial).unwrap();
                    session.set_pipeline(true);
                    let before = session.pipelined_blocks();
                    session.begin_trace().unwrap();
                    let output = session.eltwise3(op, input(0), second, third).unwrap();
                    let trace = session.end_trace().unwrap();
                    overlapped += session.pipelined_blocks() - before;
                    session.replay(trace).unwrap();
                    assert_eq!(
                        session.download_bits(&output).unwrap(),
                        expected,
                        "{what}, overlapped and replayed"
                    );
                    session.release_trace(trace).unwrap();
                    session.free(output).unwrap();
                }
                if width > 64 {
                    assert!(
                        overlapped > 0,
                        "{count} tiles, {width}x{width}: nothing overlapped"
                    );
                }
                for input in [floats, row, booleans, integers] {
                    session.free(input).unwrap();
                }
            }
            assert!(session.dataflow_stats().regions > 0);
        });
    }
}

#[test]
fn default_ownership_matches_flex_with_and_without_overlap() {
    for count in [1, 2] {
        with_tiles(count, |session| {
            // Enough tiles for several batches at depth one, so slots are
            // reused.
            let data = values(512 * 512, 3);
            let input = session.upload(&data, 512, 512).unwrap();
            let expected = flex_values(flex(&data, 512, 512) * 2.0);
            for overlap in [false, true] {
                session.set_pipeline(overlap);
                let output = add(session, &input);
                assert_eq!(bits(&session.download(&output).unwrap()), bits(&expected));
                session.free(output).unwrap();
            }
            let stats = session.dataflow_stats();
            assert!(stats.regions > 0 && stats.batches > 0);
            // The add's gathers stay off its scatters' slots: slot reuse
            // waits for the pack alone, overlapping NC's writes.
            assert!(
                stats.pack_waits > 0 && stats.release_waits == 0,
                "{stats:?}"
            );
            session.set_pipeline(false);
            let output = add(session, &input);
            assert_eq!(bits(&session.download(&output).unwrap()), bits(&expected));
        });
    }
}

#[test]
fn streaming_matmul_and_reduction_match_flex() {
    for count in [1, 2] {
        with_tiles(count, |session| {
            let data = ints(256 * 256, 5);
            let input = session.upload(&data, 256, 256).unwrap();
            let expected = flex_product_and_sum(&data, 256);
            let run = |session: &mut Session<_>| {
                let product = session
                    .matmul_dram(
                        &input,
                        true,
                        &input,
                        false,
                        SrcRoute::Tf32FromFp32,
                        Fidelity::HiFi4,
                        BUDGET,
                    )
                    .unwrap();
                let sum = session.reduce(&product, ReduceOp::Sum, Axis::Rows).unwrap();
                let outputs = [
                    session.download(&product).unwrap(),
                    session.download(&sum).unwrap(),
                ];
                session.free(product).unwrap();
                session.free(sum).unwrap();
                outputs
            };
            for overlap in [false, true] {
                session.set_pipeline(overlap);
                let actual = run(session);
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert_eq!(bits(actual), bits(expected), "overlap {overlap}");
                }
            }
            session.begin_trace().unwrap();
            let product = session
                .matmul_dram(
                    &input,
                    true,
                    &input,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    BUDGET,
                )
                .unwrap();
            let sum = session.reduce(&product, ReduceOp::Sum, Axis::Rows).unwrap();
            let trace = session.end_trace().unwrap();
            session.replay(trace).unwrap();
            session.replay(trace).unwrap();
            assert_eq!(
                bits(&session.download(&product).unwrap()),
                bits(&expected[0])
            );
            assert_eq!(bits(&session.download(&sum).unwrap()), bits(&expected[1]));
            session.release_trace(trace).unwrap();
        });
    }
}

#[test]
fn streaming_matmul_matches_flex_at_every_fidelity_and_source_format() {
    with_tiles(1, |session| {
        let data = ints(128 * 128, 47);
        let input = session.upload(&data, 128, 128).unwrap();
        let a = flex(&data, 128, 128);
        let expected = bits(&flex_values(a.clone().matmul(a.transpose())));
        for route in [SrcRoute::Tf32FromFp32, SrcRoute::Bf16FromFp32] {
            for fidelity in [
                Fidelity::Lo,
                Fidelity::HiFi2,
                Fidelity::HiFi3,
                Fidelity::HiFi4,
            ] {
                for overlap in [false, true] {
                    session.set_pipeline(overlap);
                    let output = session
                        .matmul_dram(&input, false, &input, true, route, fidelity, BUDGET)
                        .unwrap();
                    assert_eq!(
                        bits(&session.download(&output).unwrap()),
                        expected,
                        "{route:?} {fidelity:?} overlap {overlap}"
                    );
                    session.free(output).unwrap();
                }
            }
        }
    });
}

#[test]
fn streaming_trace_replays_changed_inputs_and_releases_storage() {
    with_tiles(2, |session| {
        session.set_pipeline(true);
        let input = session.upload(&values(1024 * 1024, 7), 1024, 1024).unwrap();
        session.begin_trace().unwrap();
        let output = add(session, &input);
        let trace = session.end_trace().unwrap();
        for seed in [11, 13] {
            let data = values(1024 * 1024, seed);
            session.write(&input, &data).unwrap();
            session.replay(trace).unwrap();
            session.replay(trace).unwrap();
            let expected: Vec<f32> = data.iter().map(|value| value + value).collect();
            assert_eq!(bits(&session.download(&output).unwrap()), bits(&expected));
            let fresh = add(session, &input);
            assert_eq!(bits(&session.download(&fresh).unwrap()), bits(&expected));
            session.free(fresh).unwrap();
        }
        let free_before = session.dram_free_bytes();
        session.release_trace(trace).unwrap();
        assert!(session.dram_free_bytes() > free_before);
    });
}

/// Standalone transfers (uploads, copies, row gathers and writes, fills of
/// padding) run as reader/writer packets with no kernel: B reads a batch in,
/// NC writes it out. Bit for bit the host's rows, fresh and replayed from a
/// trace over changed inputs, on one tile and on two.
#[test]
fn standalone_transfers_are_reader_writer_packets_fresh_and_traced() {
    let (rows, cols) = (100usize, 70usize);
    let picks: Vec<(usize, usize)> = (0..45).map(|i| (0, (i * 7 + 3) % rows)).collect();
    let expected_rows = |data: &[f32]| -> Vec<f32> {
        picks
            .iter()
            .flat_map(|&(_, r)| data[r * cols..(r + 1) * cols].to_vec())
            .collect()
    };
    for tiles in [1, 2] {
        with_tiles(tiles, |session| {
            let data = values(rows * cols, 5);
            let input = session.upload(&data, rows, cols).unwrap();
            let before = session.dataflow_stats().transfer_packets;
            let copy = session.copy(&input).unwrap();
            let gathered = session.gather_rows(&[&input], &picks, cols).unwrap();
            assert!(
                session.dataflow_stats().transfer_packets >= before + tiles as u64,
                "a copy is a transfer packet on every tile"
            );
            assert_eq!(bits(&session.download(&copy).unwrap()), bits(&data));
            assert_eq!(
                bits(&session.download(&gathered).unwrap()),
                bits(&expected_rows(&data))
            );
            session.free(copy).unwrap();
            session.free(gathered).unwrap();
            session.begin_trace().unwrap();
            let traced_copy = session.copy(&input).unwrap();
            let traced_rows = session.gather_rows(&[&input], &picks, cols).unwrap();
            let trace = session.end_trace().unwrap();
            for seed in [11, 13] {
                let next = values(rows * cols, seed);
                session.write(&input, &next).unwrap();
                session.replay(trace).unwrap();
                assert_eq!(bits(&session.download(&traced_copy).unwrap()), bits(&next));
                assert_eq!(
                    bits(&session.download(&traced_rows).unwrap()),
                    bits(&expected_rows(&next))
                );
            }
            session.release_trace(trace).unwrap();
        });
    }
}

#[test]
fn firmware_configuration_is_fixed_and_rejected_during_capture() {
    with_tiles(1, |session| {
        assert!(session
            .enable_dram(tt_firmware_images::DM_B.1, &[])
            .is_err());
        let input = session.upload(&values(64 * 64, 3), 64, 64).unwrap();
        session.begin_trace().unwrap();
        assert!(session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .is_err());
        let output = add(session, &input);
        let trace = session.end_trace().unwrap();
        session.release_trace(trace).unwrap();
        session.free(output).unwrap();
        session.free(input).unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        session.set_pipeline(true);
    });
}

#[test]
fn streaming_ragged_broadcast_and_nonbatched_execution() {
    with_tiles(1, |session| {
        let (data, row) = (values(65 * 97, 19), values(97, 29));
        let input = session.upload(&data, 65, 97).unwrap();
        let bias = session.upload(&row, 1, 97).unwrap();
        let expected = flex_values(flex(&data, 65, 97) + flex(&row, 1, 97));
        let run = |session: &mut Session<_>| {
            session
                .eltwise(
                    Eltwise {
                        kind: kind::ADD,
                        scalar: 0.0,
                        scalar2: 0.0,
                    },
                    &input,
                    Some(&bias),
                )
                .unwrap()
        };
        let batched = run(session);
        assert_eq!(bits(&session.download(&batched).unwrap()), bits(&expected));
        session.free(batched).unwrap();
        session.set_batching(false).unwrap();
        let output = run(session);
        assert_eq!(bits(&session.download(&output).unwrap()), bits(&expected));
        let progress = session.dataflow_progress().unwrap();
        let progress = progress[0].as_ref().unwrap();
        assert_eq!(progress.input[0], progress.input[1]);
        assert_eq!(progress.output[0], progress.output[1]);
        assert_eq!(progress.abort, 0);
        assert_eq!(progress.waits, [0; 5]);
    });
}

#[test]
fn failed_reader_cancels_the_waiting_writer() {
    use tt_device::tlb::WindowKind;
    use tt_isa::dm::{self, op, Mover};
    use tt_kernels::dm::DataMover;
    with_tiles(1, |session| {
        let tile = session.tile();
        let device = session.device();
        let window = device.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = device.dram_grid(&window).unwrap();
        let reader = DataMover::start_on(
            device,
            &window,
            tile,
            &dram,
            Mover::B,
            tt_firmware_images::DM_B.1,
        )
        .unwrap();
        let writer = DataMover::start_on(
            device,
            &window,
            tile,
            &dram,
            Mover::NC,
            tt_firmware_images::DM_NC.1,
        )
        .unwrap();
        let entries = [
            [op::PAIR, 1, 1, 1, 0, 0, 0, 0],
            [0xffff, 0, 0, 0, 0, 0, 0, 0],
            [op::CB, 1, 1, 1, 0, 0, 0, 0],
        ];
        let bytes: Vec<u8> = entries
            .iter()
            .flatten()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        device.l1_write(&window, tile, dm::LIST, &bytes).unwrap();
        device.write32(&window, tile, dm::OP, op::LIST).unwrap();
        device
            .write32(&window, tile, dm::LEN, entries.len() as u32)
            .unwrap();
        device.write32(&window, tile, dm::SEQ, 1).unwrap();
        for _ in 0..10000 {
            device.tick(tt_device::core_control::CYCLES_PER_POLL);
            if device
                .read32(&window, tile, Mover::NC.at(dm::DONE))
                .unwrap()
                == 1
            {
                break;
            }
        }
        assert_eq!(
            device
                .read32(&window, tile, Mover::NC.at(dm::DONE))
                .unwrap(),
            1
        );
        assert_eq!(
            device
                .read32(&window, tile, Mover::NC.at(dm::ERROR))
                .unwrap(),
            dm::error::OP
        );
        assert_eq!(
            device
                .read32(&window, tile, tt_isa::dataflow::ABORT)
                .unwrap(),
            dm::error::OP
        );
        reader.stop(device, &window).unwrap();
        writer.stop(device, &window).unwrap();
    });
}

#[test]
fn failed_writer_cancels_the_waiting_reader() {
    use tt_device::tlb::WindowKind;
    use tt_isa::dm::{self, op, Mover};
    use tt_kernels::dm::DataMover;
    with_tiles(1, |session| {
        let tile = session.tile();
        let device = session.device();
        let window = device.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = device.dram_grid(&window).unwrap();
        let reader = DataMover::start_on(
            device,
            &window,
            tile,
            &dram,
            Mover::B,
            tt_firmware_images::DM_B.1,
        )
        .unwrap();
        let writer = DataMover::start_on(
            device,
            &window,
            tile,
            &dram,
            Mover::NC,
            tt_firmware_images::DM_NC.1,
        )
        .unwrap();
        let entries = [
            [op::PAIR, 1, 1, 1, 0, 0, 0, 0],
            [op::RELEASED, 1, 0, 0, 0, 0, 0, 0],
            [0xffff, 0, 0, 0, 0, 0, 0, 0],
        ];
        let bytes: Vec<u8> = entries
            .iter()
            .flatten()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        device.l1_write(&window, tile, dm::LIST, &bytes).unwrap();
        device.write32(&window, tile, dm::OP, op::LIST).unwrap();
        device
            .write32(&window, tile, dm::LEN, entries.len() as u32)
            .unwrap();
        device.write32(&window, tile, dm::SEQ, 1).unwrap();
        for _ in 0..10000 {
            device.tick(tt_device::core_control::CYCLES_PER_POLL);
            if device.read32(&window, tile, dm::DONE).unwrap() == 1 {
                break;
            }
        }
        assert_eq!(device.read32(&window, tile, dm::DONE).unwrap(), 1);
        assert_eq!(
            device.read32(&window, tile, dm::ERROR).unwrap(),
            dm::error::OP
        );
        assert_eq!(
            device
                .read32(&window, tile, tt_isa::dataflow::ABORT)
                .unwrap(),
            dm::error::OP
        );
        reader.stop(device, &window).unwrap();
        writer.stop(device, &window).unwrap();
    });
}

#[test]
fn large_streaming_pipeline_trace_crosses_chunks() {
    with_tiles(1, |session| {
        let input = session.upload(&values(2048 * 2048, 7), 2048, 2048).unwrap();
        session.begin_trace().unwrap();
        let output = add(session, &input);
        let trace = session.end_trace().unwrap();
        let expected = session.download(&output).unwrap();
        session.replay(trace).unwrap();
        assert_eq!(bits(&session.download(&output).unwrap()), bits(&expected));
        session.release_trace(trace).unwrap();
    });
}

#[test]
fn a_warm_streaming_region_has_one_host_commit() {
    with_tiles(1, |session| {
        let input = session.upload(&values(128 * 128, 31), 128, 128).unwrap();
        session.set_pipeline(true);
        session.set_pipeline(false);
        let warm = add(session, &input);
        session.sync().unwrap();
        session.free(warm).unwrap();
        let before = session.device().traffic();
        let regions = session.dataflow_stats().regions;
        let output = add(session, &input);
        session.sync().unwrap();
        let after = session.device().traffic();
        assert_eq!(after.write_calls - before.write_calls, 3);
        assert_eq!(session.dataflow_stats().regions - regions, 1);
        session.free(output).unwrap();
    });
}

#[test]
fn a_sixteen_page_ring_runs_without_sync_unit_counters() {
    use tt_device::tlb::WindowKind;
    use tt_isa::dataflow::{Action, STREAMED, VERSION};
    use tt_isa::dm::{op, Mover};
    use tt_kernels::dm::DataMover;
    with_tiles(1, |session| {
        let tile = session.tile();
        let device = session.device();
        let window = device.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = device.dram_grid(&window).unwrap();
        let mut reader = DataMover::start_on(
            device,
            &window,
            tile,
            &dram,
            Mover::B,
            tt_firmware_images::DM_B.1,
        )
        .unwrap();
        let writer = DataMover::start_on(
            device,
            &window,
            tile,
            &dram,
            Mover::NC,
            tt_firmware_images::DM_NC.1,
        )
        .unwrap();
        let mut launch = [op::LAUNCH, 0x100, 0, 0, 0, 0, 0, 0];
        for role in 0..3 {
            let address = tt_isa::l1::PROGRAM_CACHE.base + role as u64 * 4096;
            let mut words = vec![VERSION, 49, 16, role as u32];
            words.resize(4 + 49 * 4, 0);
            let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            device.l1_write(&window, tile, address, &bytes).unwrap();
            launch[2 + role * 2] = address as u32;
            launch[3 + role * 2] = words.len() as u32 | STREAMED;
        }
        let mut reads = vec![launch];
        let mut writes = Vec::new();
        for _ in 0..49 {
            reads.push([op::CB, 0, Action::Reserve.word(), 16, 0, 0, 0, 0]);
            reads.push([op::CB, 0, Action::Push.word(), 16, 0, 0, 0, 0]);
            writes.push([op::CB, 1, Action::Wait.word(), 16, 0, 0, 0, 0]);
            writes.push([op::CB, 1, Action::Pop.word(), 16, 0, 0, 0, 0]);
        }
        reads.push([op::KERNEL_WAIT, 0x100, 0, 0, 0, 0, 0, 0]);
        let mut packet = vec![[
            op::PAIR,
            reads.len() as u32,
            writes.len() as u32,
            16,
            0,
            0,
            0,
            0,
        ]];
        packet.extend(reads);
        packet.extend(writes);
        let number = reader.enqueue(device, &window, &packet).unwrap();
        reader.wait_for(device, &window, number).unwrap();
        for address in [
            tt_isa::dataflow::INPUT,
            tt_isa::dataflow::INPUT + 4,
            tt_isa::dataflow::OUTPUT,
            tt_isa::dataflow::OUTPUT + 4,
        ] {
            assert_eq!(device.read32(&window, tile, address).unwrap(), 49);
        }
        reader.stop(device, &window).unwrap();
        writer.stop(device, &window).unwrap();
    });
}

#[test]
fn a_streaming_trace_holds_its_bodies_under_cache_pressure() {
    with_tiles(1, |session| {
        session.limit_program_cache(24 * 1024).unwrap();
        session.set_pipeline(true);
        let input = session.upload(&values(128 * 128, 37), 128, 128).unwrap();
        session.begin_trace().unwrap();
        let output = add(session, &input);
        let trace = session.end_trace().unwrap();
        let expected = session.download(&output).unwrap();
        for scalar in 1..80 {
            let temporary = session
                .eltwise(
                    Eltwise {
                        kind: kind::ADD_SCALAR,
                        scalar: scalar as f32 / 8.0,
                        scalar2: 0.0,
                    },
                    &input,
                    None,
                )
                .unwrap();
            session.sync().unwrap();
            session.free(temporary).unwrap();
        }
        assert!(session
            .program_cache_stats()
            .iter()
            .any(|stats| stats.evictions > 0));
        session.replay(trace).unwrap();
        assert_eq!(bits(&session.download(&output).unwrap()), bits(&expected));
        session.release_trace(trace).unwrap();
    });
}

#[test]
fn a_failed_stream_role_invalidates_traces_and_recovers() {
    use tt_device::tlb::WindowKind;
    with_tiles(1, |session| {
        session.set_pipeline(true);
        let input = session.upload(&values(128 * 128, 41), 128, 128).unwrap();
        session.begin_trace().unwrap();
        let output = add(session, &input);
        let trace = session.end_trace().unwrap();
        let expected = session.download(&output).unwrap();
        let tile = session.tile();
        let device = session.device();
        let window = device.alloc_window(WindowKind::TwoMib).unwrap();
        let address = device
            .read32(
                &window,
                tile,
                tt_isa::mailbox::role::Mailbox::of(0).program_addr(),
            )
            .unwrap();
        device
            .write32(&window, tile, address as u64, tt_isa::dataflow::VERSION + 1)
            .unwrap();
        session.replay(trace).unwrap();
        assert!(session.sync().is_err());
        assert!(session.replay(trace).is_err());
        session.release_trace(trace).unwrap();
        let fresh = add(session, &input);
        assert_eq!(bits(&session.download(&fresh).unwrap()), bits(&expected));
    });
}

#[cfg(feature = "silicon")]
#[test]
#[ignore = "silicon streaming GDDR payload throughput"]
fn streaming_gddr_throughput() {
    use std::time::Instant;
    use tt_tests::bench::{report_rate, Conditions, Stats, REPS};
    let width = 8192;
    let data: Vec<f32> = (0..width * width)
        .map(|index| ((index % 251) as f32 - 125.0) / 128.0)
        .collect();
    let payload = (data.len() * std::mem::size_of::<f32>()) as f64;
    for count in [1, 8, 32, 120] {
        with_tiles(count, |session| {
            let tile = session.tile();
            let conditions =
                Conditions::measure(session.device(), tt_tests::backend::device_index(), tile);
            conditions.print();
            let peak = conditions.gddr_card();
            let input = session.upload(&data, width, width).unwrap();
            let identity = |session: &mut Session<_>| {
                session
                    .eltwise(
                        Eltwise {
                            kind: kind::MUL_SCALAR,
                            scalar: 1.0,
                            scalar2: 0.0,
                        },
                        &input,
                        None,
                    )
                    .unwrap()
            };
            let check = |output: &[f32]| {
                assert_eq!(output.len(), data.len());
                assert!(output
                    .iter()
                    .zip(&data)
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
            };
            for (mode, overlap) in [("serial", false), ("streaming", true)] {
                session.set_pipeline(overlap);
                for warmup in 0..3 {
                    let output = identity(session);
                    session.sync().unwrap();
                    if warmup == 0 {
                        check(&session.download(&output).unwrap());
                    }
                    session.free(output).unwrap();
                }
                let mut fresh = Vec::new();
                for sample in 0..REPS {
                    let start = Instant::now();
                    let output = identity(session);
                    session.sync().unwrap();
                    fresh.push(start.elapsed());
                    if sample + 1 == REPS {
                        check(&session.download(&output).unwrap());
                    }
                    session.free(output).unwrap();
                }
                session.begin_trace().unwrap();
                let output = identity(session);
                let trace = session.end_trace().unwrap();
                for _ in 0..3 {
                    session.replay(trace).unwrap();
                    session.sync().unwrap();
                }
                let mut traced = Vec::new();
                for _ in 0..REPS {
                    let start = Instant::now();
                    session.replay(trace).unwrap();
                    session.sync().unwrap();
                    traced.push(start.elapsed());
                }
                check(&session.download(&output).unwrap());
                for (phase, samples) in [("fresh", fresh), ("trace", traced)] {
                    let timing = Stats::of_durations(samples);
                    let label = format!("gddr identity {mode} {count} tiles {phase} 8192x8192");
                    report_rate(
                        &format!("{label} read payload"),
                        "host",
                        payload,
                        timing,
                        None,
                    );
                    report_rate(
                        &format!("{label} write payload"),
                        "host",
                        payload,
                        timing,
                        None,
                    );
                    report_rate(
                        &format!("{label} read+write payload"),
                        "host",
                        2.0 * payload,
                        timing,
                        peak.as_ref(),
                    );
                }
                session.release_trace(trace).unwrap();
                session.free(output).unwrap();
            }
            session.free(input).unwrap();
        });
    }
}

#[cfg(feature = "silicon")]
#[test]
fn streaming_collects_profile_events_without_affecting_results() {
    for roles in [false, true] {
        with_tiles(1, |session| {
            session.set_pipeline(true);
            let input = session.upload(&values(256 * 256, 53), 256, 256).unwrap();
            session.set_profile_roles(roles);
            session.profile_start().unwrap();
            let fresh = add(session, &input);
            session.sync().unwrap();
            session.begin_trace().unwrap();
            let captured = add(session, &input);
            let trace = session.end_trace().unwrap();
            session.replay(trace).unwrap();
            session.sync().unwrap();
            session.replay(trace).unwrap();
            session.sync().unwrap();
            let profile = session.profile_stop().unwrap();
            assert!(!profile.units[0].events.is_empty());
            match profile.to_chrome_trace() {
                Ok(exported) => assert!(exported.contains("buffer credit")),
                Err(error) => {
                    eprintln!("MEASURE dataflow timestamp_export=false roles={roles} error={error}")
                }
            }
            assert_eq!(
                bits(&session.download(&fresh).unwrap()),
                bits(&session.download(&captured).unwrap())
            );
            session.release_trace(trace).unwrap();
        });
    }
}

#[cfg(feature = "silicon")]
#[test]
#[ignore = "silicon performance sweep"]
fn streaming_performance_sweep() {
    use std::time::Instant;
    for count in [1, 8, 32] {
        with_tiles(count, |session| {
            for (operation, width) in [
                ("add", 64),
                ("add", 512),
                ("add", 2048),
                ("matmul", 256),
                ("matmul", 512),
            ] {
                let data = ints(width * width, 17);
                let input = session.upload(&data, width, width).unwrap();
                let expected = {
                    let a = flex(&data, width, width);
                    bits(&flex_values(if operation == "add" {
                        a * 2.0
                    } else {
                        a.clone().matmul(a)
                    }))
                };
                let mut results = [[0.0f64; 2]; 2];
                for (mode_index, overlap) in [false, true].into_iter().enumerate() {
                    session.set_pipeline(overlap);
                    let run = |session: &mut Session<_>| {
                        if operation == "add" {
                            add(session, &input)
                        } else {
                            session
                                .matmul_dram(
                                    &input,
                                    false,
                                    &input,
                                    false,
                                    SrcRoute::Tf32FromFp32,
                                    Fidelity::HiFi4,
                                    BUDGET,
                                )
                                .unwrap()
                        }
                    };
                    for warmup in 0..3 {
                        let output = run(session);
                        session.sync().unwrap();
                        if warmup == 0 {
                            assert_eq!(
                                bits(&session.download(&output).unwrap()),
                                expected,
                                "{count} tiles, {operation} {width}, overlap {overlap}"
                            );
                        }
                        session.free(output).unwrap();
                    }
                    let mut samples = Vec::new();
                    for _ in 0..15 {
                        let start = Instant::now();
                        let output = run(session);
                        session.sync().unwrap();
                        samples.push(start.elapsed().as_secs_f64() * 1e6);
                        session.free(output).unwrap();
                    }
                    samples.sort_by(f64::total_cmp);
                    results[mode_index][0] = samples[samples.len() / 2];
                    session.begin_trace().unwrap();
                    let output = run(session);
                    let trace = session.end_trace().unwrap();
                    for _ in 0..3 {
                        session.replay(trace).unwrap();
                        session.sync().unwrap();
                    }
                    let mut samples = Vec::new();
                    for _ in 0..15 {
                        let start = Instant::now();
                        session.replay(trace).unwrap();
                        session.sync().unwrap();
                        samples.push(start.elapsed().as_secs_f64() * 1e6);
                    }
                    samples.sort_by(f64::total_cmp);
                    results[mode_index][1] = samples[samples.len() / 2];
                    session.release_trace(trace).unwrap();
                    session.free(output).unwrap();
                }
                for (phase, streaming_time) in results[1].iter().enumerate() {
                    let best = results[0][phase];
                    let ratio = streaming_time / best;
                    eprintln!("MEASURE dataflow tiles={count} op={operation} width={width} phase={} serial_us={:.2} streaming_us={streaming_time:.2} ratio={ratio:.3}", if phase == 0 { "fresh" } else { "trace" }, results[0][phase]);
                }
                session.free(input).unwrap();
            }
        });
    }
}

#[cfg(feature = "silicon")]
#[test]
#[ignore = "sustained silicon validation"]
fn streaming_stress() {
    use std::time::{Duration, Instant};
    let seconds = std::env::var("TT_STREAMING_STRESS_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(120);
    for count in [1, 8] {
        with_tiles(count, |session| {
            let data = ints(256 * 256, 23);
            let input = session.upload(&data, 256, 256).unwrap();
            let run = |session: &mut Session<_>| {
                let product = session
                    .matmul_dram(
                        &input,
                        true,
                        &input,
                        false,
                        SrcRoute::Tf32FromFp32,
                        Fidelity::HiFi4,
                        BUDGET,
                    )
                    .unwrap();
                let doubled = add(session, &product);
                let sum = session.reduce(&doubled, ReduceOp::Sum, Axis::Rows).unwrap();
                [product, doubled, sum]
            };
            let expected: Vec<Vec<u32>> = {
                let a = flex(&data, 256, 256);
                let product = a.clone().transpose().matmul(a);
                let doubled = product.clone() * 2.0;
                [product, doubled.clone(), doubled.sum_dim(0)]
                    .map(|tensor| bits(&flex_values(tensor)))
                    .into()
            };
            session.begin_trace().unwrap();
            let captured = run(session);
            let trace = session.end_trace().unwrap();
            let start = Instant::now();
            let mut rounds = 0;
            while start.elapsed() < Duration::from_secs(seconds) {
                if rounds % 2 == 0 {
                    session.replay(trace).unwrap();
                    for (output, expected) in captured.iter().zip(&expected) {
                        assert_eq!(bits(&session.download(output).unwrap()), *expected);
                    }
                } else {
                    let outputs = run(session);
                    for (output, expected) in outputs.iter().zip(&expected) {
                        assert_eq!(bits(&session.download(output).unwrap()), *expected);
                    }
                    for output in outputs {
                        session.free(output).unwrap();
                    }
                }
                rounds += 1;
            }
            session.release_trace(trace).unwrap();
            for output in captured {
                session.free(output).unwrap();
            }
            eprintln!("MEASURE dataflow stress tiles={count} seconds={seconds} rounds={rounds}");
        });
    }
}
