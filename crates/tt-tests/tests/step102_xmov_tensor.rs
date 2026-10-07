//! Explicit resident movement. ttsim refuses scalar memory and mover setup;
//! instruction probes/models remain in step101 and local_movement unit tests.
#![cfg(feature = "silicon")]
use tt_kernels::{
    session::{Session, TileChoice},
    tensor::{DramTensor, Elem, Pad},
};
use tt_ttsim::fork_scope;
fn with_session(tiles: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(tiles),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    })
    .unwrap();
}
fn bits(n: usize, elem: Elem) -> Vec<u32> {
    let special = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007fffff,
        0x807fffff,
        0x7f800000,
        0xff800000,
        0x7fc12345,
        0xffc54321,
        0x7f812345,
        0xff812345,
        0x7f7fffff,
        0xff7fffff,
        0x80000000,
        0x7fffffff,
        u32::MAX,
    ];
    (0..n)
        .map(|i| {
            if elem == Elem::Bool {
                (i % 2) as u32
            } else if i % 3 == 0 {
                special[i / 3 % special.len()]
            } else {
                (i as u32).wrapping_mul(0x13579bdf)
            }
        })
        .collect()
}
fn physical(
    s: &mut Session<tt_kmd::Kmd>,
    slots: impl Iterator<Item = tt_isa::dram::DramRange>,
    dims: [usize; 2],
    bytes: usize,
) -> Vec<Vec<u8>> {
    s.sync().unwrap();
    let dev = s.device();
    let w = dev
        .alloc_window(tt_device::tlb::WindowKind::TwoMib)
        .unwrap();
    slots
        .enumerate()
        .map(|(tile, range)| {
            let mut data = vec![0; range.len() as usize];
            dev.dram_read(&w, range, &mut data).unwrap();
            for r in 0..32 {
                for c in 0..32 {
                    if tile / dims[1].div_ceil(32) * 32 + r >= dims[0]
                        || tile % dims[1].div_ceil(32) * 32 + c >= dims[1]
                    {
                        let off = 16 + tt_isa::dm::face_index(r, c) * bytes;
                        assert!(
                            data[off..off + bytes].iter().all(|&b| b == 0),
                            "nonzero padding tile {tile} row {r} col {c}"
                        );
                    }
                }
            }
            data
        })
        .collect()
}
fn poison_padding(s: &mut Session<tt_kmd::Kmd>, t: &DramTensor) {
    s.sync().unwrap();
    let dev = s.device();
    let w = dev
        .alloc_window(tt_device::tlb::WindowKind::TwoMib)
        .unwrap();
    for tile in 0..t.placement.tiles() {
        let range = t.placement.slot(tile);
        let mut data = vec![0; range.len() as usize];
        dev.dram_read(&w, range, &mut data).unwrap();
        for r in 0..32 {
            for c in 0..32 {
                if tile / t.cols.div_ceil(32) * 32 + r >= t.rows
                    || tile % t.cols.div_ceil(32) * 32 + c >= t.cols
                {
                    let off = 16 + tt_isa::dm::face_index(r, c) * 4;
                    data[off..off + 4].copy_from_slice(&0x7fc12345u32.to_le_bytes());
                }
            }
        }
        dev.dram_write(&w, range, &data).unwrap();
    }
    t.set_pad(Pad::Undefined);
}
#[test]
fn exceptional_bits_ragged_padding_views_and_resident_traffic() {
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            for dims in [[1, 1], [1, 17], [17, 31], [32, 32], [33, 65], [97, 99]] {
                for elem in [Elem::F32, Elem::I32, Elem::Bool] {
                    let values = bits(dims[0] * dims[1], elem);
                    let source = s.upload_bits(&values, dims[0], dims[1], elem).unwrap();
                    poison_padding(s, &source);
                    let parent_pad = source.pad();
                    let before = s.dataflow_stats().clone();
                    let out = s.copy_xmov(&source).unwrap();
                    assert_eq!(source.pad(), parent_pad);
                    assert_eq!(out.pad(), Pad::Zero);
                    s.sync().unwrap();
                    assert!(
                        s.dataflow_stats().batches > before.batches,
                        "Tensix compute job required"
                    );
                    assert_eq!(
                        s.dataflow_stats().transfer_packets,
                        before.transfer_packets,
                        "no B-only copy packets"
                    );
                    assert_eq!(s.download_bits(&out).unwrap(), values);
                    physical(
                        s,
                        (0..out.placement.tiles()).map(|t| out.placement.slot(t)),
                        dims,
                        4,
                    );
                    let zero = s.zeros_xmov(dims, elem).unwrap();
                    assert_eq!(s.download_bits(&zero).unwrap(), vec![0; values.len()]);
                    physical(
                        s,
                        (0..zero.placement.tiles()).map(|t| zero.placement.slot(t)),
                        dims,
                        4,
                    );
                    s.free(zero).unwrap();
                    s.free(out).unwrap();
                    s.free(source).unwrap();
                }
            }
            let dims = [97, 33];
            let values = bits(dims[0] * dims[1], Elem::F32);
            let parent = s.upload_bits(&values, dims[0], dims[1], Elem::F32).unwrap();
            let view = parent.rows_view(32, 65).unwrap();
            let out = s.copy_xmov(&view).unwrap();
            assert_eq!(s.download_bits(&out).unwrap(), values[32 * 33..]);
            s.free(view).unwrap();
            s.free(parent).unwrap();
            s.free(out).unwrap();
        });
    }
}
#[test]
fn raw_bf16_bits_zeros_and_row_views() {
    with_session(2, |s| {
        let special = [
            0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0xffc5, 0x7f81, 0xff81,
            0x7f7f, 0xff7f,
        ];
        for dims in [[1, 1], [17, 31], [32, 32], [33, 65], [97, 99]] {
            let values: Vec<_> = (0..dims[0] * dims[1])
                .map(|i| {
                    if i % 2 == 0 {
                        special[i / 2 % special.len()]
                    } else {
                        i as u16
                    }
                })
                .collect();
            let source = s.upload_bf16(&values, dims[0], dims[1]).unwrap();
            let out = s.copy_bf16_xmov(&source).unwrap();
            assert_eq!(s.download_bf16(&out).unwrap(), values);
            physical(s, (0..out.tile_count()).map(|t| out.slot(t)), dims, 2);
            let zeros = s.zeros_bf16_xmov(dims).unwrap();
            assert_eq!(s.download_bf16(&zeros).unwrap(), vec![0; values.len()]);
            physical(s, (0..zeros.tile_count()).map(|t| zeros.slot(t)), dims, 2);
            if dims[0] > 32 {
                let view = source.rows_view(32, dims[0] - 32).unwrap();
                let copied = s.copy_bf16_xmov(&view).unwrap();
                assert_eq!(s.download_bf16(&copied).unwrap(), values[32 * dims[1]..]);
                s.free_bf16(copied).unwrap();
                s.free_bf16(view).unwrap();
            }
            s.free_bf16(out).unwrap();
            s.free_bf16(zeros).unwrap();
            s.free_bf16(source).unwrap();
        }
    });
}
#[test]
fn downstream_consumers_changed_input_traces_and_deferred_frees() {
    for tiles in [1, 2] {
        for ownership in [false, true] {
            with_session(tiles, |s| {
                s.set_pipeline(ownership);
                s.set_batching(ownership).unwrap();
                let input = s.upload(&vec![2.0; 33 * 65], 33, 65).unwrap();
                poison_padding(s, &input);
                let a = s.upload(&vec![1.0; 65 * 32], 65, 32).unwrap();
                let copy = s.copy_xmov(&input).unwrap();
                let out = s
                    .matmul_dram(
                        &copy,
                        false,
                        &a,
                        false,
                        tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                        tt_kernels::matmul::Fidelity::HiFi4,
                        4_000_000,
                    )
                    .unwrap();
                assert_eq!(s.download(&out).unwrap(), vec![130.0; 33 * 32]);
                s.free(out).unwrap();
                s.free(a).unwrap();
                s.free(copy).unwrap();
                s.set_batching(true).unwrap();
                s.begin_trace().unwrap();
                let copy = s.copy_xmov(&input).unwrap();
                let zero = s.zeros_xmov([33, 65], Elem::F32).unwrap();
                let sum = s.sum_rows(&copy).unwrap();
                s.free(copy).unwrap();
                s.free(zero).unwrap();
                let trace = s.end_trace().unwrap();
                for v in [2.0, 5.0, -3.0] {
                    s.write(&input, &vec![v; 33 * 65]).unwrap();
                    s.replay(trace).unwrap();
                    assert_eq!(s.download(&sum).unwrap(), vec![v * 33.0; 65]);
                }
                s.free(input).unwrap();
                s.replay(trace).unwrap();
                assert_eq!(s.download(&sum).unwrap(), vec![-99.0; 65]);
                s.release_trace(trace).unwrap();
                s.free(sum).unwrap();
            });
        }
    }
}
#[test]
fn invalid_shapes_and_failed_construction_leave_session_usable() {
    with_session(1, |s| {
        for dims in [[0, 1], [1, 0], [usize::MAX, 2], [usize::MAX, usize::MAX]] {
            assert!(s.zeros_xmov(dims, Elem::F32).is_err());
            assert!(s.zeros_bf16_xmov(dims).is_err());
        }
        let input = s.upload_bits(&[7], 1, 1, Elem::I32).unwrap();
        let bf16 = s.upload_bf16(&[7], 1, 1).unwrap();
        let before = s.dram_free_bytes();
        let mut invalid = input.clone();
        invalid.rows = 65;
        let mut invalid_bf16 = bf16.clone();
        invalid_bf16.rows = 65;
        for _ in 0..3 {
            assert!(s.copy_xmov(&invalid).is_err());
            assert!(s.copy_bf16_xmov(&invalid_bf16).is_err());
            assert_eq!(
                s.dram_free_bytes(),
                before,
                "construction must release allocated outputs"
            );
        }
        s.limit_program_cache(8).unwrap();
        assert!(s.copy_xmov(&input).is_err());
        assert!(s.copy_bf16_xmov(&bf16).is_err());
        assert!(s.zeros_xmov([1, 1], Elem::F32).is_err());
        assert!(s.zeros_bf16_xmov([1, 1]).is_err());
        assert_eq!(
            s.dram_free_bytes(),
            before,
            "submission must release allocated outputs"
        );
        s.limit_program_cache(tt_isa::l1::PROGRAM_CACHE.len())
            .unwrap();
        s.free(input).unwrap();
        s.free_bf16(bf16).unwrap();
        let out = s.zeros_xmov([1, 1], Elem::F32).unwrap();
        assert_eq!(s.download_bits(&out).unwrap(), [0]);
        s.free(out).unwrap();
    });
}
#[test]
#[ignore = "release silicon benchmark"]
fn benchmark_copy_zero_against_native() {
    use std::time::Instant;
    let release = !cfg!(debug_assertions);
    assert!(release, "benchmarks require --release");
    println!("\nMEASURE XMOV end-to-end versus native copy and metadata zero");
    with_session(2, |s| {
        for dims in [[32, 32], [256, 256], [97, 99]] {
            let input = bits(dims[0] * dims[1], Elem::F32);
            let source = s.upload_bits(&input, dims[0], dims[1], Elem::F32).unwrap();
            for mode in ["native_copy", "xmov_copy", "native_zero", "xmov_zero"] {
                let before = s.dataflow_stats().clone();
                let mut samples = vec![];
                for rep in 0..9 {
                    let start = Instant::now();
                    let out = match mode {
                        "native_copy" => s.copy(&source),
                        "xmov_copy" => s.copy_xmov(&source),
                        "native_zero" => s.metadata(&vec![0; dims[0] * dims[1]], dims, Elem::F32),
                        _ => s.zeros_xmov(dims, Elem::F32),
                    }
                    .unwrap();
                    s.sync().unwrap();
                    let elapsed = start.elapsed().as_secs_f64();
                    let got = s.download_bits(&out).unwrap();
                    assert_eq!(
                        got,
                        if mode.ends_with("copy") {
                            input.clone()
                        } else {
                            vec![0; input.len()]
                        }
                    );
                    s.free(out).unwrap();
                    if rep >= 2 {
                        samples.push(elapsed);
                    }
                }
                samples.sort_by(f64::total_cmp);
                println!("BENCH {{\"kind\":\"xmov\",\"key\":\"{mode}_{}x{}\",\"card\":{},\"tiles\":2,\"rows\":{},\"cols\":{},\"warmups\":2,\"samples\":7,\"timed\":\"host dispatch through sync\",\"median\":{},\"unit\":\"us\",\"regions\":{},\"batches\":{},\"transfers\":{}}}",dims[0],dims[1],tt_tests::backend::device_index(),dims[0],dims[1],samples[3]*1e6,s.dataflow_stats().regions-before.regions,s.dataflow_stats().batches-before.batches,s.dataflow_stats().transfer_packets-before.transfer_packets);
            }
            s.free(source).unwrap();
        }
    });
}

#[test]
fn bf16_changed_input_traces_and_packed_matmul() {
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            let a = s.upload_bf16(&vec![0x4000; 33 * 17], 33, 17).unwrap();
            let b = s.upload_bf16(&vec![0x3f80; 17 * 32], 17, 32).unwrap();
            let copy = s.copy_bf16_xmov(&a).unwrap();
            let mm = s
                .matmul_bf16(&copy, &b, tt_kernels::matmul::Fidelity::HiFi4, 4_000_000)
                .unwrap();
            assert_eq!(s.download(&mm).unwrap(), vec![34.0; 33 * 32]);
            s.free(mm).unwrap();
            s.free_bf16(copy).unwrap();
            s.free_bf16(b).unwrap();
            s.begin_trace().unwrap();
            let out = s.copy_bf16_xmov(&a).unwrap();
            let zero = s.zeros_bf16_xmov([33, 17]).unwrap();
            s.free_bf16(zero).unwrap();
            let trace = s.end_trace().unwrap();
            for bits in [0x4000, 0x8001, 0x7f81] {
                let replacement = s.upload_bf16(&vec![bits; 33 * 17], 33, 17).unwrap();
                s.copy_into_bf16(&replacement, &a).unwrap();
                s.free_bf16(replacement).unwrap();
                s.replay(trace).unwrap();
                assert_eq!(s.download_bf16(&out).unwrap(), vec![bits; 33 * 17]);
            }
            s.free_bf16(a).unwrap();
            s.replay(trace).unwrap();
            assert_eq!(s.download_bf16(&out).unwrap(), vec![0x7f81; 33 * 17]);
            s.free_bf16(out).unwrap();
            s.release_trace(trace).unwrap();
        });
    }
}
