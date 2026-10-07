//! XY-counter copies: independent raw-bit indexing oracle and resident execution.
use tt_kernels::{
    matmul::Fidelity,
    session::{Session, TileChoice},
};
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
type Transport<'a> = tt_ttsim::LibTtsim<'a>;
#[cfg(feature = "silicon")]
type Transport<'a> = tt_kmd::Kmd;
fn with_session(tiles: usize, f: impl FnOnce(&mut Session<Transport<'_>>)) {
    if let Err(e) = fork_scope(|| {
        #[cfg(not(feature = "silicon"))]
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        #[cfg(not(feature = "silicon"))]
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        #[cfg(not(feature = "silicon"))]
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(tiles),
            |_, _| Ok(None),
        )
        .unwrap();
        #[cfg(feature = "silicon")]
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(tiles),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn values(rows: usize, cols: usize) -> Vec<f32> {
    let specials = [
        0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000, 0xff800000, 0x7fc12345,
        0xffc54321, 0x7f812345, 0xff812345, 0x7f7fffff, 0xff7fffff,
    ];
    (0..rows * cols)
        .map(|i| {
            f32::from_bits(if i % 11 == 0 {
                specials[i / 11 % specials.len()]
            } else {
                0x3f000000 + i as u32
            })
        })
        .collect()
}
fn expected(a: &[f32], cols: usize, origin: [usize; 2], dims: [usize; 2]) -> Vec<u32> {
    (origin[0]..origin[0] + dims[0])
        .flat_map(|r| (origin[1]..origin[1] + dims[1]).map(move |c| a[r * cols + c].to_bits()))
        .collect()
}
#[test]
fn rectangles_preserve_all_f32_bits_across_faces_and_tiles() {
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            let a = values(97, 99);
            let ta = s.upload(&a, 97, 99).unwrap();
            ta.set_pad(tt_kernels::tensor::Pad::Undefined);
            let parent_pad = ta.pad();
            for (origin, dims) in [
                ([0, 0], [1, 16]),
                ([1, 16], [15, 16]),
                ([7, 16], [37, 48]),
                ([15, 16], [65, 80]),
                ([31, 32], [33, 64]),
                ([63, 80], [34, 16]),
            ] {
                let out = s.copy_rect_adc(&ta, origin, dims).unwrap();
                assert_eq!(out.pad(), tt_kernels::tensor::Pad::Zero);
                let got: Vec<_> = s
                    .download(&out)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(
                    got,
                    expected(&a, 99, origin, dims),
                    "{tiles} tiles {origin:?} {dims:?}"
                );
                assert_eq!(ta.pad(), parent_pad);
                s.free(out).unwrap();
            }
            let view = ta.rows_view(32, 64).unwrap();
            let out = s.copy_rect_adc(&view, [1, 16], [37, 48]).unwrap();
            assert_eq!(
                s.download(&out)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected(&a, 99, [33, 16], [37, 48])
            );
            assert_eq!(ta.pad(), parent_pad);
            s.free(out).unwrap();
            s.free(view).unwrap();
            s.free(ta).unwrap();
        });
    }
}
#[test]
fn invalid_geometry_is_rejected_before_execution() {
    with_session(1, |s| {
        let ta = s.upload(&vec![1.0; 32 * 48], 32, 48).unwrap();
        for (origin, dims) in [
            ([0, 0], [0, 16]),
            ([0, 1], [1, 16]),
            ([0, 0], [1, 17]),
            ([31, 0], [2, 16]),
            ([0, 32], [1, 32]),
            ([usize::MAX, 0], [1, 16]),
        ] {
            assert!(s.copy_rect_adc(&ta, origin, dims).is_err());
        }
        s.free(ta).unwrap();
        let integer = s
            .upload_bits(&vec![1; 32 * 48], 32, 48, tt_kernels::tensor::Elem::I32)
            .unwrap();
        assert!(s.copy_rect_adc(&integer, [0, 0], [16, 16]).is_err());
        s.free(integer).unwrap();
    });
}

/// Independent state model from ADCs.md, INCADCXY/ADDRCRXY.md. It models all
/// targets and masks; an observable unpack checks the resulting live values.
#[derive(Clone, Copy)]
struct Counters {
    live: [u32; 4],
    cursor: [u32; 4],
}
impl Counters {
    fn apply(&mut self, relative: bool, selected: [bool; 4], by: [u32; 4]) {
        for i in 0..4 {
            let mask = if i % 2 == 0 {
                (1 << 18) - 1
            } else {
                (1 << 13) - 1
            };
            if relative {
                if selected[i] {
                    self.cursor[i] = (self.cursor[i] + by[i]) & mask;
                    self.live[i] = self.cursor[i];
                }
            } else {
                self.live[i] = (self.live[i] + by[i]) & mask;
            }
        }
    }
}

#[test]
fn live_and_cursor_counters_have_distinct_masked_semantics() {
    check_xy_masks(1..16);
}

#[test]
#[cfg(feature = "silicon")]
fn empty_cursor_mask_is_a_noop_on_silicon() {
    check_xy_masks(0..1);
}

#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_refuses_empty_cursor_mask() {
    use tt_isa::adc::{self, Coordinates, Targets, Xy};
    use tt_tests::harness::{self, Run};
    assert!(!harness::survives(|dev| {
        let program = [
            tt_kernels::datapath::state_id(),
            adc::advance_cursor(Targets::UNPACKER0, Coordinates::default(), Xy::default()).unwrap(),
        ];
        harness::run(dev, &Run::new(&program).dump_rows(0));
    }));
}

fn check_xy_masks(masks: std::ops::Range<u32>) {
    use tt_isa::{
        adc::{self, Coordinates, Targets, Xy},
        backend::{self, Before, ConfigWords},
        cfg::generated::unpack1,
    };
    use tt_kernels::datapath as dp;
    use tt_tests::harness::{self, Roles, Run};
    let input: Vec<f32> = (0..1024).map(|i| (i + 1) as f32).collect();
    let mut image = vec![0u8; 16];
    for v in &input {
        image.extend(v.to_le_bytes());
    }
    harness::in_device(|dev| {
        for target_bits in 1..8 {
            let mut target = None;
            for (bit, t) in [
                (1, Targets::UNPACKER0),
                (2, Targets::UNPACKER1),
                (4, Targets::PACKERS),
            ] {
                if target_bits & bit != 0 {
                    target = Some(target.map_or(t, |a: Targets| a.union(t)));
                }
            }
            let target = target.unwrap();
            for mask_bits in masks.clone() {
                let selected = [
                    mask_bits & 1 != 0,
                    mask_bits & 2 != 0,
                    mask_bits & 4 != 0,
                    mask_bits & 8 != 0,
                ];
                let mut model = Counters {
                    live: [0, 0, 15, 0],
                    cursor: [0, 0, 15, 0],
                };
                let mut unpack = dp::thread_config();
                let mut words = ConfigWords::new();
                dp::unpack_config(
                    &mut words,
                    dp::flat_descriptor(16).with_y_dim(64),
                    dp::STAGE,
                );
                words
                    .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 64)
                    .unwrap();
                words
                    .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                    .unwrap();
                unpack.extend(dp::config_program(&words));
                unpack.push(tt_isa::isa::generated::encode::zeroacc(3, 0, 0, 0).unwrap());
                unpack.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                unpack.extend(dp::clear_unpacker0_adcs());
                unpack.push(dp::set_adc_x_unpack(0, 15));
                unpack.push(
                    adc::increment(
                        target,
                        Xy {
                            x0: 7,
                            y0: 7,
                            x1: 7,
                            y1: 7,
                        },
                    )
                    .unwrap(),
                );
                unpack.push(
                    adc::advance_cursor(
                        target,
                        Coordinates {
                            x0: selected[0],
                            y0: selected[1],
                            x1: selected[2],
                            y1: selected[3],
                        },
                        Xy {
                            x0: 1,
                            y0: 2,
                            x1: 3,
                            y1: 4,
                        },
                    )
                    .unwrap(),
                );
                if target_bits & 1 != 0 {
                    model.apply(false, [true; 4], [7; 4]);
                    model.apply(true, selected, [1, 2, 3, 4]);
                }
                unpack.push(dp::unpack_instruction());
                unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
                let pack = dp::pack_tile_from_dst(dp::OUT, 0);
                let output = harness::run(
                    dev,
                    &Run::roles(Roles {
                        unpack: &unpack,
                        math: &[],
                        pack: &pack,
                    })
                    .stage(&[(dp::STAGE, &image)])
                    .dump_rows(0)
                    .read_back(&[(dp::OUT, 4096)]),
                );
                let got: Vec<_> = output.l1[0]
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                let [x0, y0, x1, y1] = model.live;
                let mut expected = vec![0u32; 1024];
                let n = (x1 + 1 - x0) as usize;
                let src = (y0 * 16 + x0) as usize;
                let dst = y1 as usize * 16;
                for i in 0..n {
                    expected[dst + i] = input[src + i].to_bits();
                }
                // An unpack marks the whole final Dst row valid; unused
                // columns in a partial row retain old physical bits. Check
                // only written datums and wholly untouched rows. Production
                // uses complete sixteen-datum rows, so this is probe-only.
                for i in 0..1024 {
                    if (dst + n..dst + n.next_multiple_of(16)).contains(&i) {
                        continue;
                    }
                    assert_eq!(
                        got[i], expected[i],
                        "targets={target_bits}, mask={mask_bits}, datum={i}"
                    );
                }
                if target_bits == 1 && mask_bits == 15 {
                    let mut restore = unpack.clone();
                    let cursor = restore
                        .iter()
                        .rposition(|i| i.def().mnemonic() == "ADDRCRXY")
                        .unwrap();
                    restore[cursor] =
                        adc::advance_cursor(target, Coordinates::ALL, Xy::default()).unwrap();
                    let restored = harness::run(
                        dev,
                        &Run::roles(Roles {
                            unpack: &restore,
                            math: &[],
                            pack: &pack,
                        })
                        .stage(&[(dp::STAGE, &image)])
                        .dump_rows(0)
                        .read_back(&[(dp::OUT, 4096)]),
                    );
                    let restored: Vec<_> = restored.l1[0]
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    assert_eq!(
                        &restored[..16],
                        &input[..16].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert!(restored[16..].iter().all(|&v| v == 0));
                    let mut mutant = unpack.clone();
                    let cursor = mutant
                        .iter()
                        .rposition(|i| i.def().mnemonic() == "ADDRCRXY")
                        .unwrap();
                    mutant[cursor] = adc::increment(
                        target,
                        Xy {
                            x0: 1,
                            y0: 2,
                            x1: 3,
                            y1: 4,
                        },
                    )
                    .unwrap();
                    let bad = harness::run(
                        dev,
                        &Run::roles(Roles {
                            unpack: &mutant,
                            math: &[],
                            pack: &pack,
                        })
                        .stage(&[(dp::STAGE, &image)])
                        .dump_rows(0)
                        .read_back(&[(dp::OUT, 4096)]),
                    );
                    let bad: Vec<_> = bad.l1[0]
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    assert_ne!(
                        &bad[dst..dst + n],
                        &expected[dst..dst + n],
                        "live-add mutant must fail the cursor oracle"
                    );
                }
            }
        }
    });
}

#[test]
fn large_live_row_increments_match_documented_addressing() {
    use tt_isa::{
        adc::{self, Targets, Xy},
        backend::{self, Before, ConfigWords},
    };
    use tt_kernels::datapath as dp;
    use tt_tests::harness::{self, Roles, Run};
    let input: Vec<f32> = (0..1024).map(|i| (i + 1) as f32).collect();
    let mut image = vec![0u8; 16];
    for v in &input {
        image.extend(v.to_le_bytes());
    }
    let mut unpack = dp::thread_config();
    let mut words = ConfigWords::new();
    dp::unpack_config(
        &mut words,
        dp::flat_descriptor(16).with_y_dim(64),
        dp::STAGE,
    );
    unpack.extend(dp::config_program(&words));
    unpack.extend(dp::clear_unpacker0_adcs());
    unpack.push(dp::set_adc_x_unpack(0, 15));
    // 1171*7 = 8197. ADCs.md specifies a thirteen-bit Y counter,
    // and the input generator observes only its low eight bits: five.
    // This output gate checks addressing, not the invisible counter width.
    unpack.extend(std::iter::repeat_n(
        adc::increment(
            Targets::UNPACKER0,
            Xy {
                y0: 7,
                ..Xy::default()
            },
        )
        .unwrap(),
        1171,
    ));
    unpack.push(dp::unpack_instruction());
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    let pack = dp::pack_tile_from_dst(dp::OUT, 0);
    harness::in_device(|dev| {
        let output = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &[],
                pack: &pack,
            })
            .stage(&[(dp::STAGE, &image)])
            .dump_rows(0)
            .read_back(&[(dp::OUT, 64)]),
        );
        let got: Vec<_> = output.l1[0]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            got,
            input[80..96]
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    });
}

#[test]
fn changed_input_traces_padding_and_deferred_frees() {
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            let input = s.upload(&vec![2.0; 97 * 99], 97, 99).unwrap();
            let parent_pad = input.pad();
            // Populate counter/configuration state with an ordinary matmul.
            let a = s.upload(&vec![1.0; 32 * 32], 32, 32).unwrap();
            let mm = s
                .matmul_dram(
                    &a,
                    false,
                    &a,
                    false,
                    tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    4_000_000,
                )
                .unwrap();
            s.free(mm).unwrap();
            // A copy must not leave live Y counters for the next Src matmul.
            let fresh = s.copy_rect_adc(&input, [15, 16], [65, 80]).unwrap();
            let mm = s
                .matmul_dram(
                    &a,
                    false,
                    &a,
                    false,
                    tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    4_000_000,
                )
                .unwrap();
            assert_eq!(s.download(&mm).unwrap(), vec![32.0; 32 * 32]);
            s.free(mm).unwrap();
            s.free(fresh).unwrap();
            s.free(a).unwrap();
            s.begin_trace().unwrap();
            let rect = s.copy_rect_adc(&input, [15, 16], [65, 80]).unwrap();
            let sum = s.sum_rows(&rect).unwrap();
            s.free(rect).unwrap();
            let trace = s.end_trace().unwrap();
            for value in [2.0, 5.0, -3.0] {
                s.write(&input, &vec![value; 97 * 99]).unwrap();
                s.replay(trace).unwrap();
                assert_eq!(s.download(&sum).unwrap(), vec![value * 65.0; 80]);
                assert_eq!(input.pad(), parent_pad);
            }
            s.free(input).unwrap();
            // The trace owns the released operand until explicitly released.
            s.replay(trace).unwrap();
            assert_eq!(s.download(&sum).unwrap(), vec![-195.0; 80]);
            s.release_trace(trace).unwrap();
            s.free(sum).unwrap();
        });
    }
}

#[test]
fn burn_slices_keep_native_repack_and_preserve_bits() {
    use burn::{
        backend::Autodiff,
        tensor::{Tensor, TensorData},
    };
    use burn_tt::TtBackend;
    use tt_tests::burn_device::{assert_native_model, with_device, Config};
    with_device(Config::default(), |d| {
        let a = values(97, 99);
        let input = Tensor::<TtBackend, 2>::from_data(TensorData::new(a.clone(), [97, 99]), &d)
            .to_device(&d);
        let (rect, report) = burn_tt::with_report(|| input.clone().slice([15..80, 16..96]));
        assert_native_model(&report);
        assert_eq!(report.moved(), 0);
        assert_eq!(
            rect.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected(&a, 99, [15, 16], [65, 80])
        );
        for (ranges, origin, dims) in [
            ([32..64, 0..99], [32, 0], [32, 99]),
            ([3..39, 1..49], [3, 1], [36, 48]),
        ] {
            let (out, report) = burn_tt::with_report(|| input.clone().slice(ranges));
            assert_native_model(&report);
            assert_eq!(report.moved(), 0);
            assert_eq!(
                out.into_data()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected(&a, 99, origin, dims)
            );
        }
        let (stepped, report) = burn_tt::with_report(|| {
            input.clone().slice([
                burn::tensor::Slice::with_step(15, Some(80), 2),
                burn::tensor::Slice::new(16, Some(96), 1),
            ])
        });
        assert_native_model(&report);
        assert_eq!(report.moved(), 0);
        let want: Vec<_> = (15..80)
            .step_by(2)
            .flat_map(|r| (16..96).map(move |c| (r, c)))
            .map(|(r, c)| a[r * 99 + c].to_bits())
            .collect();
        assert_eq!(
            stepped
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            want
        );
        let transposed = input.clone().transpose();
        let (out, report) = burn_tt::with_report(|| transposed.slice([16..32, 16..48]));
        assert_native_model(&report);
        assert_eq!(report.moved(), 0);
        let want: Vec<_> = (16..32)
            .flat_map(|r| (16..48).map(move |c| (r, c)))
            .map(|(r, c)| a[c * 99 + r].to_bits())
            .collect();
        assert_eq!(
            out.into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            want
        );
        let x = Tensor::<Autodiff<TtBackend>, 2>::from_data(
            TensorData::new(vec![1.0; 41 * 64], [41, 64]),
            &d,
        )
        .require_grad();
        let loss = x.clone().slice([3..40, 16..48]).sum();
        let gradients = loss.backward();
        let got = x
            .grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for r in 0..41 {
            for c in 0..64 {
                assert_eq!(
                    got[r * 64 + c],
                    if (3..40).contains(&r) && (16..48).contains(&c) {
                        1.0
                    } else {
                        0.0
                    }
                );
            }
        }
    });
}

#[test]
#[ignore = "benchmark"]
#[cfg(feature = "silicon")]
fn benchmark_adc_rectangle_vs_native_repack() {
    use std::time::Instant;
    let release = !cfg!(debug_assertions);
    assert!(release, "benchmarks require --release");
    println!("\nMEASURE ADC rectangle versus native repack");
    with_session(2, |s| {
        let input: Vec<_> = (0..128 * 128).map(|i| (i % 37) as f32).collect();
        let a = s.upload(&input, 128, 128).unwrap();
        for (origin, dims) in [
            ([1, 16], [16, 16]),
            ([15, 16], [37, 48]),
            ([15, 16], [65, 80]),
        ] {
            let sources: Vec<_> = (origin[0]..origin[0] + dims[0])
                .flat_map(|r| (origin[1]..origin[1] + dims[1]).map(move |c| [r, c]))
                .collect();
            let want = expected(&input, 128, origin, dims);
            for adc in [false, true] {
                let before = s.dataflow_stats().clone();
                let mut samples = vec![];
                for rep in 0..9 {
                    let start = Instant::now();
                    let out = if adc {
                        s.copy_rect_adc(&a, origin, dims)
                    } else {
                        s.repack(&a, &sources, dims)
                    }
                    .unwrap();
                    s.sync().unwrap();
                    let elapsed = start.elapsed().as_secs_f64();
                    let got: Vec<_> = s
                        .download(&out)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    assert_eq!(got, want);
                    s.free(out).unwrap();
                    if rep >= 2 {
                        samples.push(elapsed);
                    }
                }
                samples.sort_by(f64::total_cmp);
                println!("BENCH {{\"kind\":\"adc_copy\",\"git\":\"{}\",\"card\":{},\"rows\":{},\"cols\":{},\"adc\":{adc},\"tiles\":2,\"warmups\":2,\"samples\":7,\"key\":\"adc_copy_{}x{}_{adc}\",\"timed\":\"host dispatch through sync\",\"median\":{},\"unit\":\"us\",\"p10\":{},\"p90\":{},\"regions\":{},\"batches\":{},\"transfers\":{}}}",tt_tests::bench::git_sha(),tt_tests::backend::device_index(),dims[0],dims[1],dims[0],dims[1],samples[3]*1e6,samples[1]*1e6,samples[5]*1e6,s.dataflow_stats().regions-before.regions,s.dataflow_stats().batches-before.batches,s.dataflow_stats().transfer_packets-before.transfer_packets);
            }
        }
        s.free(a).unwrap();
    });
}

#[test]
fn unpacker1_and_packer_targets_select_their_own_counters() {
    use tt_isa::{
        adc::{self, Coordinates, Targets, Xy},
        backend::{self, Before, ConfigWords},
        isa::generated::encode,
        matrix::Banks,
    };
    use tt_kernels::datapath as dp;
    use tt_tests::harness::{self, Roles, Run};
    let input: Vec<_> = (0..1024).map(|i| (i + 1) as f32).collect();
    let mut image = vec![0u8; 16];
    for v in &input {
        image.extend(v.to_le_bytes());
    }
    harness::in_device(|dev| {
        for target in [
            Targets::UNPACKER0,
            Targets::UNPACKER1,
            Targets::PACKERS,
            Targets::ALL,
        ] {
            let mut unpack = dp::src_thread_config();
            let mut words = ConfigWords::new();
            dp::unpack_src_config(
                &mut words,
                dp::Unpacker::SrcB,
                dp::flat_descriptor(16).with_y_dim(64),
                dp::STAGE,
                4,
            );
            words
                .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                .unwrap();
            unpack.extend(dp::config_program(&words));
            unpack.push(
                encode::Setadcxy::ZERO
                    .u1(1)
                    .x0(1)
                    .x1(1)
                    .y0(1)
                    .y1(1)
                    .encode()
                    .unwrap(),
            );
            unpack.push(
                encode::Setadczw::ZERO
                    .u1(1)
                    .z0(1)
                    .z1(1)
                    .w0(1)
                    .w1(1)
                    .encode()
                    .unwrap(),
            );
            unpack.push(dp::set_adc_x(dp::Unpacker::SrcB, 0, 15));
            unpack.push(
                adc::increment(
                    target,
                    Xy {
                        y0: 7,
                        ..Xy::default()
                    },
                )
                .unwrap(),
            );
            unpack.push(
                adc::advance_cursor(
                    target,
                    Coordinates {
                        y0: true,
                        ..Coordinates::default()
                    },
                    Xy {
                        y0: 2,
                        ..Xy::default()
                    },
                )
                .unwrap(),
            );
            let (i, banks) = Banks::after_reset()
                .unpack_b(encode::UnpacrRegular::ZERO.multi_context_mode(1))
                .unwrap();
            unpack.push(i);
            unpack.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());
            let mut math = tt_kernels::matmul::math_prelude();
            let (i, banks) = banks
                .movb2d(encode::Movb2D::ZERO.src_row(0).dst_row(0))
                .unwrap();
            math.push(i);
            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
            let (i, _) = banks.release_b().unwrap();
            math.push(i);
            let pack = dp::pack_tile_from_dst(dp::OUT, 0);
            let out = harness::run(
                dev,
                &Run::roles(Roles {
                    unpack: &unpack,
                    math: &math,
                    pack: &pack,
                })
                .stage(&[(dp::STAGE, &image)])
                .dump_rows(0)
                .read_back(&[(dp::OUT, 64)]),
            );
            let got: Vec<_> = out.l1[0]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let first = if target == Targets::UNPACKER1 || target == Targets::ALL {
                32
            } else {
                0
            };
            assert_eq!(
                got,
                input[first..first + 16]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );

            // On thread 2 the same helper can select the packer's input row.
            let mut unpack = dp::thread_config();
            let mut words = ConfigWords::new();
            dp::tile_unpack_config(&mut words, dp::STAGE);
            unpack.extend(dp::config_program(&words));
            unpack.extend(dp::clear_unpacker0_adcs());
            unpack.extend(dp::unpack_tile_to_dst(dp::STAGE, 0));
            unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
            let mut pack = vec![dp::state_id()];
            let mut words = ConfigWords::new();
            dp::pack_config(&mut words, dp::OUT);
            pack.extend(dp::config_program(&words));
            pack.extend(dp::pack_rows(4));
            let last = pack.pop().unwrap();
            pack.push(
                adc::increment(
                    target,
                    Xy {
                        y0: 7,
                        ..Xy::default()
                    },
                )
                .unwrap(),
            );
            pack.push(
                adc::advance_cursor(
                    target,
                    Coordinates {
                        y0: true,
                        ..Coordinates::default()
                    },
                    Xy {
                        y0: 4,
                        ..Xy::default()
                    },
                )
                .unwrap(),
            );
            pack.push(last);
            pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
            let out = harness::run(
                dev,
                &Run::roles(Roles {
                    unpack: &unpack,
                    math: &[],
                    pack: &pack,
                })
                .stage(&[(dp::STAGE, &image)])
                .dump_rows(0)
                .read_back(&[(dp::OUT, 256)]),
            );
            let got: Vec<_> = out.l1[0]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let first = if target == Targets::PACKERS || target == Targets::ALL {
                64
            } else {
                0
            };
            assert_eq!(
                got,
                input[first..first + 64]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    });
}

#[test]
fn adc_mutations_are_local_to_the_issuing_thread() {
    use tt_isa::{
        adc::{self, Coordinates, Targets, Xy},
        backend::{self, Before, ConfigWords},
        sync,
    };
    use tt_kernels::{datapath as dp, l1::Requirements};
    use tt_tests::harness::{self, Roles, Run};
    let mut req = Requirements::new(1);
    let ready = req.semaphore("counter initialized", 0, 0..1);
    let mutated = req.semaphore("other thread mutated", 0, 0..1);
    let copied = req.semaphore("copy complete", 0, 0..1);
    let plan = req.plan(tt_isa::l1::DATA).unwrap();
    let (ready, mutated, copied) = (
        plan.semaphore(ready),
        plan.semaphore(mutated),
        plan.semaphore(copied),
    );
    let init = plan.semaphore_init();
    let input: Vec<_> = (0..1024).map(|i| (i + 1) as f32).collect();
    let mut image = vec![0u8; 16];
    for v in &input {
        image.extend(v.to_le_bytes());
    }
    let mut unpack = dp::thread_config();
    let mut words = ConfigWords::new();
    dp::unpack_config(
        &mut words,
        dp::flat_descriptor(16).with_y_dim(64),
        dp::STAGE,
    );
    unpack.extend(dp::config_program(&words));
    unpack.extend(dp::clear_unpacker0_adcs());
    unpack.push(dp::set_adc_x_unpack(0, 15));
    unpack.push(backend::nop());
    unpack.push(sync::post(ready));
    unpack.extend(sync::take(mutated, Before::EVERYTHING));
    unpack.push(dp::unpack_instruction());
    unpack.extend(sync::post_after(sync::Unit::Unpacker0, copied));
    let mut math = vec![dp::state_id()];
    math.extend(sync::take(ready, Before::EVERYTHING));
    math.push(
        adc::increment(
            Targets::UNPACKER0,
            Xy {
                y0: 7,
                ..Xy::default()
            },
        )
        .unwrap(),
    );
    math.push(
        adc::advance_cursor(
            Targets::UNPACKER0,
            Coordinates {
                y0: true,
                ..Coordinates::default()
            },
            Xy {
                y0: 4,
                ..Xy::default()
            },
        )
        .unwrap(),
    );
    math.push(backend::nop());
    math.push(sync::post(mutated));
    let mut pack = vec![dp::state_id()];
    pack.extend(sync::take(copied, Before::EVERYTHING));
    pack.extend(dp::pack_tile_from_dst(dp::OUT, 0));
    harness::in_device(|dev| {
        let output = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &math,
                pack: &pack,
            })
            .concurrent(&init)
            .stage(&[(dp::STAGE, &image)])
            .dump_rows(0)
            .read_back(&[(dp::OUT, 64)]),
        );
        let got: Vec<_> = output.l1[0]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            got,
            input[..16].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    });
}
