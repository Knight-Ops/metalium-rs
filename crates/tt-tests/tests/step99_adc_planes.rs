//! Z/W counters and resident planes: independent state/address and raw-bit oracles.
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

fn oracle(a: &[f32], shape: [usize; 4], origin: [usize; 2], dims: [usize; 2]) -> Vec<u32> {
    let [_, z, y, x] = shape;
    let mut out = Vec::new();
    for w in origin[0]..origin[0] + dims[0] {
        for zz in origin[1]..origin[1] + dims[1] {
            let at = (w * z + zz) * y * x;
            out.extend(a[at..at + y * x].iter().map(|v| v.to_bits()));
        }
    }
    out
}
#[test]
fn planes_preserve_raw_bits_and_ragged_boundaries() {
    for tiles in [1, 2] {
        with_session(tiles, |s| {
            for y in [1, 17, 33] {
                for x in [1, 17, 33] {
                    let shape = [3, 9, y, x];
                    let a = values(27 * y, x);
                    let source = s.upload(&a, 27 * y, x).unwrap();
                    source.set_pad(tt_kernels::tensor::Pad::Undefined);
                    for (origin, dims) in [([0, 0], [1, 1]), ([1, 2], [2, 7]), ([0, 0], [3, 9])] {
                        let out = s.copy_planes_adc(&source, shape, origin, dims).unwrap();
                        assert_eq!(out.pad(), tt_kernels::tensor::Pad::Zero);
                        assert_eq!(
                            s.download(&out)
                                .unwrap()
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            oracle(&a, shape, origin, dims),
                            "{shape:?} {origin:?} {dims:?}"
                        );
                        assert_eq!(source.pad(), tt_kernels::tensor::Pad::Undefined);
                        s.free(out).unwrap();
                    }
                    s.free(source).unwrap();
                }
            }
            // Both selected global plane counts exceed the slab's local 8x8
            // coordinates; selection also skips planes between W groups.
            let shape = [11, 17, 1, 17];
            let a = values(187, 17);
            let source = s.upload(&a, 187, 17).unwrap();
            let out = s.copy_planes_adc(&source, shape, [1, 2], [9, 9]).unwrap();
            assert_eq!(
                s.download(&out)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                oracle(&a, shape, [1, 2], [9, 9])
            );
            s.free(out).unwrap();
            s.free(source).unwrap();
            let a = values(128, 33);
            let source = s.upload(&a, 128, 33).unwrap();
            let view = source.rows_view(32, 96).unwrap();
            let out = s
                .copy_planes_adc(&view, [2, 3, 16, 33], [1, 1], [1, 2])
                .unwrap();
            assert_eq!(
                s.download(&out)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                oracle(&a[32 * 33..], [2, 3, 16, 33], [1, 1], [1, 2])
            );
            s.free(out).unwrap();
            assert_eq!(source.pad(), tt_kernels::tensor::Pad::Zero);
            s.free(view).unwrap();
            s.free(source).unwrap();
        });
    }
}
#[test]
fn changed_input_trace_padding_and_deferred_free() {
    with_session(2, |s| {
        let source = s.upload(&vec![2.0; 6 * 17 * 33], 6 * 17, 33).unwrap();
        let a = s.upload(&vec![1.0; 33 * 17], 33, 17).unwrap();
        for _ in 0..2 {
            let out = s
                .copy_planes_adc(&source, [2, 3, 17, 33], [0, 1], [2, 2])
                .unwrap();
            let mm = s
                .matmul_dram(
                    &out,
                    false,
                    &a,
                    false,
                    tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    4_000_000,
                )
                .unwrap();
            assert_eq!(s.download(&mm).unwrap(), vec![66.0; 68 * 17]);
            s.free(mm).unwrap();
            s.free(out).unwrap();
        }
        s.free(a).unwrap();
        s.begin_trace().unwrap();
        let out = s
            .copy_planes_adc(&source, [2, 3, 17, 33], [0, 1], [2, 2])
            .unwrap();
        let sum = s.sum_rows(&out).unwrap();
        s.free(out).unwrap();
        let trace = s.end_trace().unwrap();
        for v in [2.0, 5.0, -3.0] {
            s.write(&source, &vec![v; 6 * 17 * 33]).unwrap();
            s.replay(trace).unwrap();
            assert_eq!(s.download(&sum).unwrap(), vec![v * 68.0; 33]);
        }
        s.free(source).unwrap();
        s.replay(trace).unwrap();
        assert_eq!(s.download(&sum).unwrap(), vec![-204.0; 33]);
        s.release_trace(trace).unwrap();
        s.free(sum).unwrap();
    });
}
/// Independent state model from ADCs.md, INCADCZW/ADDRCRZW.md. It models all
/// targets and masks; an observable unpack checks the resulting live values.
#[derive(Clone, Copy)]
struct Counters {
    live: [u32; 4],
    cursor: [u32; 4],
}
impl Counters {
    fn apply(&mut self, relative: bool, selected: [bool; 4], by: [u32; 4]) {
        for i in 0..4 {
            let mask = (1 << 13) - 1;
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
    check_zw_masks(1..16);
}

#[test]
#[cfg(feature = "silicon")]
fn empty_cursor_mask_is_a_noop_on_silicon() {
    check_zw_masks(0..1);
}

#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_refuses_empty_cursor_mask() {
    use tt_isa::adc::{self, Targets, Zw, ZwCoordinates};
    use tt_tests::harness::{self, Run};
    assert!(!harness::survives(|dev| {
        let program = [
            tt_kernels::datapath::state_id(),
            adc::advance_cursor_zw(Targets::UNPACKER0, ZwCoordinates::default(), Zw::default())
                .unwrap(),
        ];
        harness::run(dev, &Run::new(&program).dump_rows(0));
    }));
}

fn check_zw_masks(masks: std::ops::Range<u32>) {
    use tt_isa::{
        adc::{self, Targets, Zw, ZwCoordinates},
        backend::{self, Before, ConfigWords},
        cfg::generated::unpack1,
    };
    use tt_kernels::datapath as dp;
    use tt_tests::harness::{self, Roles, Run};
    let input: Vec<f32> = (0..2048).map(|i| (i + 1) as f32).collect();
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
            for inc in 0..8 {
                for mask_bits in masks.clone() {
                    let selected = [
                        mask_bits & 1 != 0,
                        mask_bits & 2 != 0,
                        mask_bits & 4 != 0,
                        mask_bits & 8 != 0,
                    ];
                    let mut model = Counters {
                        live: [0; 4],
                        cursor: [0; 4],
                    };
                    let mut unpack = dp::thread_config();
                    let mut words = ConfigWords::new();
                    dp::unpack_config(&mut words, dp::flat_descriptor(16).with_z_dim(8), dp::STAGE);
                    words
                        .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 64)
                        .unwrap();
                    words
                        .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                        .unwrap();
                    words
                        .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Wstride, 512)
                        .unwrap();
                    unpack.extend(dp::config_program(&words));
                    unpack.push(tt_isa::isa::generated::encode::zeroacc(3, 0, 0, 0).unwrap());
                    unpack.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                    unpack.extend(dp::clear_unpacker0_adcs());
                    unpack.push(dp::set_adc_x_unpack(0, 15));
                    unpack.push(
                        adc::increment_zw(
                            target,
                            Zw {
                                z0: inc,
                                w0: inc,
                                z1: inc,
                                w1: inc,
                            },
                        )
                        .unwrap(),
                    );
                    unpack.push(
                        adc::advance_cursor_zw(
                            target,
                            ZwCoordinates {
                                z0: selected[0],
                                w0: selected[1],
                                z1: selected[2],
                                w1: selected[3],
                            },
                            Zw {
                                z0: 1,
                                w0: 2,
                                z1: 3,
                                w1: 4,
                            },
                        )
                        .unwrap(),
                    );
                    if target_bits & 1 != 0 {
                        model.apply(false, [true; 4], [inc; 4]);
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
                    let [z0, w0, z1, w1] = model.live;
                    let mut expected = vec![0u32; 1024];
                    let n = 16;
                    let src = (w0 * 128 + z0 * 16) as usize;
                    let dst = (w1 * 128 + z1 * 16) as usize;
                    for i in 0..n {
                        expected[dst + i] = input[src + i].to_bits();
                    }
                    for i in 0..1024 {
                        assert_eq!(
                            got[i], expected[i],
                            "targets={target_bits}, mask={mask_bits}, increment={inc}, datum={i}"
                        );
                    }
                    if target_bits == 1 && mask_bits == 15 && inc == 7 {
                        let mut restore = unpack.clone();
                        let cursor = restore
                            .iter()
                            .rposition(|i| i.def().mnemonic() == "ADDRCRZW")
                            .unwrap();
                        restore[cursor] =
                            adc::advance_cursor_zw(target, ZwCoordinates::ALL, Zw::default())
                                .unwrap();
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
                            .rposition(|i| i.def().mnemonic() == "ADDRCRZW")
                            .unwrap();
                        mutant[cursor] = adc::increment_zw(
                            target,
                            Zw {
                                z0: 1,
                                w0: 2,
                                z1: 3,
                                w1: 4,
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
        }
    });
}
#[test]
fn zw_unpacker1_and_packer_targets_select_their_own_counters() {
    use tt_isa::{
        adc::{self, Targets, Zw, ZwCoordinates},
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
            Targets::UNPACKER0.union(Targets::UNPACKER1),
            Targets::UNPACKER0.union(Targets::PACKERS),
            Targets::UNPACKER1.union(Targets::PACKERS),
            Targets::ALL,
        ] {
            let mut unpack = dp::src_thread_config();
            let mut words = ConfigWords::new();
            dp::unpack_src_config(
                &mut words,
                dp::Unpacker::SrcB,
                dp::flat_descriptor(16).with_z_dim(8),
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
                adc::increment_zw(
                    target,
                    Zw {
                        z0: 7,
                        ..Zw::default()
                    },
                )
                .unwrap(),
            );
            unpack.push(
                adc::advance_cursor_zw(
                    target,
                    ZwCoordinates {
                        z0: true,
                        ..ZwCoordinates::default()
                    },
                    Zw {
                        z0: 2,
                        ..Zw::default()
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
            let first = if [
                Targets::UNPACKER1,
                Targets::UNPACKER0.union(Targets::UNPACKER1),
                Targets::UNPACKER1.union(Targets::PACKERS),
                Targets::ALL,
            ]
            .contains(&target)
            {
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
            let mut pack = vec![
                dp::state_id(),
                encode::Setadczw::ZERO
                    .pk(1)
                    .z0(1)
                    .z1(1)
                    .w0(1)
                    .w1(1)
                    .encode()
                    .unwrap(),
            ];
            let mut words = ConfigWords::new();
            dp::pack_config(&mut words, dp::OUT);
            pack.extend(dp::config_program(&words));
            words
                .set(
                    tt_isa::cfg::generated::pack0::PCK0_ADDR_CTRL_ZW_REG_0_Zstride,
                    64,
                )
                .unwrap();
            pack.extend(dp::config_program(&words));
            pack.extend(dp::pack_rows(4));
            let last = pack.pop().unwrap();
            pack.push(
                adc::increment_zw(
                    target,
                    Zw {
                        z0: 7,
                        ..Zw::default()
                    },
                )
                .unwrap(),
            );
            pack.push(
                adc::advance_cursor_zw(
                    target,
                    ZwCoordinates {
                        z0: true,
                        ..ZwCoordinates::default()
                    },
                    Zw {
                        z0: 4,
                        ..Zw::default()
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
            let first = if [
                Targets::PACKERS,
                Targets::UNPACKER0.union(Targets::PACKERS),
                Targets::UNPACKER1.union(Targets::PACKERS),
                Targets::ALL,
            ]
            .contains(&target)
            {
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
fn zw_mutations_are_local_to_the_issuing_thread() {
    use tt_isa::{
        adc::{self, Targets, Zw, ZwCoordinates},
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
    dp::unpack_config(&mut words, dp::flat_descriptor(16).with_z_dim(8), dp::STAGE);
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
        adc::increment_zw(
            Targets::UNPACKER0,
            Zw {
                z0: 7,
                ..Zw::default()
            },
        )
        .unwrap(),
    );
    math.push(
        adc::advance_cursor_zw(
            Targets::UNPACKER0,
            ZwCoordinates {
                z0: true,
                ..ZwCoordinates::default()
            },
            Zw {
                z0: 4,
                ..Zw::default()
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

#[test]
fn invalid_geometry_and_dtype_are_rejected() {
    with_session(1, |s| {
        let source = s.upload(&vec![1.0; 6 * 17 * 33], 6 * 17, 33).unwrap();
        for (shape, origin, dims) in [
            ([2, 3, 17, 33], [0, 0], [0, 1]),
            ([2, 3, 17, 33], [2, 0], [1, 1]),
            ([2, 3, 17, 33], [0, 2], [1, 2]),
            ([2, 3, 17, 32], [0, 0], [1, 1]),
            ([2, 3, 16, 33], [0, 0], [1, 1]),
            ([0, 3, 17, 33], [0, 0], [1, 1]),
            ([usize::MAX, 3, 17, 33], [0, 0], [1, 1]),
            ([2, 3, 17, 33], [usize::MAX, 0], [1, 1]),
            ([2, 3, 17, 33], [0, 0], [usize::MAX, 2]),
        ] {
            assert!(s.copy_planes_adc(&source, shape, origin, dims).is_err());
        }
        s.free(source).unwrap();
        let source = s
            .upload_bits(&vec![1; 1024], 32, 32, tt_kernels::tensor::Elem::I32)
            .unwrap();
        assert!(s
            .copy_planes_adc(&source, [1, 1, 32, 32], [0, 0], [1, 1])
            .is_err());
        s.free(source).unwrap();
    });
}

#[test]
fn input_masks_eight_bits_but_output_uses_thirteen_bits() {
    use tt_isa::{
        adc::{self, Targets, Zw},
        backend::{self, Before, ConfigWords},
        cfg::generated::unpack1,
    };
    use tt_kernels::datapath as dp;
    use tt_tests::harness::{self, Roles, Run};
    let input: Vec<_> = (0..1024).map(|i| (i + 1) as f32).collect();
    let mut image = vec![0; 16];
    for v in &input {
        image.extend(v.to_le_bytes());
    }
    harness::in_device(|dev| {
        for w in [false, true] {
            for increments in [256, 8192] {
                let mut unpack = dp::thread_config();
                let mut words = ConfigWords::new();
                dp::unpack_config(&mut words, dp::flat_descriptor(16), dp::STAGE);
                words
                    .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                    .unwrap();
                words
                    .set(
                        if w {
                            unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Wstride
                        } else {
                            unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride
                        },
                        4,
                    )
                    .unwrap();
                // Explicitly clear the other stride: configuration persists.
                words
                    .set(
                        if w {
                            unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride
                        } else {
                            unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Wstride
                        },
                        0,
                    )
                    .unwrap();
                unpack.extend(dp::config_program(&words));
                unpack.push(tt_isa::isa::generated::encode::zeroacc(3, 0, 0, 0).unwrap());
                unpack.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                unpack.extend(dp::clear_unpacker0_adcs());
                unpack.push(dp::set_adc_x_unpack(0, 15));
                let mut left = increments;
                while left != 0 {
                    let n = left.min(7);
                    let by = if w {
                        Zw {
                            w0: n,
                            w1: n,
                            ..Default::default()
                        }
                    } else {
                        Zw {
                            z0: n,
                            z1: n,
                            ..Default::default()
                        }
                    };
                    unpack.push(adc::increment_zw(Targets::UNPACKER0, by).unwrap());
                    left -= n;
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
                let live = increments & 8191;
                let src = ((live & 255) * 16) as usize;
                let dst = live as usize; // four-byte stride / four-byte datum
                let mut want = vec![0; 1024];
                want[dst..dst + 16].copy_from_slice(
                    &input[src..src + 16]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                );
                assert_eq!(got, want, "W={w} increment={increments}");
            }
        }
    });
}

#[test]
fn traversal_negative_controls_disagree_with_oracle() {
    use tt_isa::{
        adc::{self, Targets, Zw, ZwCoordinates},
        backend::{self, Before, ConfigWords},
        cfg::generated::unpack1,
    };
    use tt_kernels::datapath as dp;
    use tt_tests::harness::{self, Roles, Run};
    let input: Vec<_> = (0..1024).map(|i| (i + 1) as f32).collect();
    let mut image = vec![0; 16];
    for v in &input {
        image.extend(v.to_le_bytes());
    }
    harness::in_device(|dev| {
        for mutant in 0..4 {
            let mut unpack = dp::thread_config();
            let mut words = ConfigWords::new();
            dp::unpack_config(&mut words, dp::flat_descriptor(16).with_z_dim(8), dp::STAGE);
            words
                .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                .unwrap();
            words
                .set(
                    unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride,
                    if mutant == 2 { 512 } else { 64 },
                )
                .unwrap();
            words
                .set(
                    unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Wstride,
                    if mutant == 2 { 64 } else { 512 },
                )
                .unwrap();
            unpack.extend(dp::config_program(&words));
            unpack.push(tt_isa::isa::generated::encode::zeroacc(3, 0, 0, 0).unwrap());
            unpack.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
            unpack.extend(dp::clear_unpacker0_adcs());
            unpack.push(dp::set_adc_x_unpack(0, 15));
            // Fill four complete rows, traversing two Z rows in each W group.
            for w in 0..2 {
                for z in 0..2 {
                    unpack.push(dp::unpack_instruction());
                    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
                    if z == 0 && mutant != 1 {
                        unpack.push(
                            adc::increment_zw(
                                Targets::UNPACKER0,
                                Zw {
                                    z0: 1,
                                    z1: 1,
                                    ..Default::default()
                                },
                            )
                            .unwrap(),
                        );
                    }
                }
                if w == 0 {
                    let by = Zw {
                        w0: 1,
                        w1: 1,
                        ..Default::default()
                    };
                    unpack.push(
                        if mutant == 3 {
                            adc::increment_zw(Targets::UNPACKER0, by)
                        } else {
                            adc::advance_cursor_zw(Targets::UNPACKER0, ZwCoordinates::ALL, by)
                        }
                        .unwrap(),
                    );
                }
            }
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
            let mut want = vec![0; 1024];
            for row in [0, 1, 8, 9] {
                for c in 0..16 {
                    want[row * 16 + c] = input[row * 16 + c].to_bits();
                }
            }
            if mutant == 0 {
                assert_eq!(got, want);
            } else {
                assert_ne!(got, want, "mutant {mutant}");
            }
        }
    });
}
#[test]
#[ignore = "benchmark"]
#[cfg(feature = "silicon")]
fn benchmark_adc_planes_vs_native_repack() {
    use std::time::Instant;
    let release = !cfg!(debug_assertions);
    assert!(release, "benchmarks require --release");
    println!("\nMEASURE ADC planes versus native repack");
    with_session(2, |s| {
        let input: Vec<_> = (0..6 * 33 * 33).map(|i| (i % 37) as f32).collect();
        let a = s.upload(&input, 6 * 33, 33).unwrap();
        for dims in [[1, 1], [1, 2], [2, 2]] {
            let shape = [2, 3, 33, 33];
            let origin = [0, 1];
            let output_dims = [dims[0] * dims[1] * 33, 33];
            let mut sources = Vec::new();
            for w in 0..dims[0] {
                for z in 1..1 + dims[1] {
                    for y in 0..33 {
                        for x in 0..33 {
                            sources.push([(w * 3 + z) * 33 + y, x]);
                        }
                    }
                }
            }
            let want = oracle(&input, shape, origin, dims);
            for adc in [false, true] {
                let before = s.dataflow_stats().clone();
                let mut samples = vec![];
                for rep in 0..9 {
                    let start = Instant::now();
                    let out = if adc {
                        s.copy_planes_adc(&a, shape, origin, dims)
                    } else {
                        s.repack(&a, &sources, output_dims)
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
                println!("BENCH {{\"kind\":\"adc_planes\",\"git\":\"{}\",\"card\":{},\"rows\":{},\"cols\":{},\"adc\":{adc},\"tiles\":2,\"warmups\":2,\"samples\":7,\"key\":\"adc_planes_{}x{}_{adc}\",\"timed\":\"host dispatch through sync\",\"median\":{},\"unit\":\"us\",\"p10\":{},\"p90\":{},\"regions\":{},\"batches\":{},\"transfers\":{}}}",tt_tests::bench::git_sha(),tt_tests::backend::device_index(),output_dims[0],output_dims[1],output_dims[0],output_dims[1],samples[3]*1e6,samples[1]*1e6,samples[5]*1e6,s.dataflow_stats().regions-before.regions,s.dataflow_stats().batches-before.batches,s.dataflow_stats().transfer_packets-before.transfer_packets);
            }
        }
        s.free(a).unwrap();
    });
}
