//! Independent byte/GPR models and isolated L1 instruction acceptance.
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::{thcon, thread},
    isa::{generated::encode, Instruction},
    scalar::{self, OffsetHalf, OffsetIncrement as Inc, TransferWidth as Width},
};
use tt_tests::harness::{self, Roles, Run};
const SRC: u64 = tt_isa::l1::DATA.base;
const DST: u64 = SRC + 8192;
const WIDTHS: [Width; 4] = [Width::Byte, Width::Halfword, Width::Word, Width::Quadword];
const INCS: [Inc; 4] = [Inc::None, Inc::Bytes2, Inc::Bytes4, Inc::Bytes16];
fn bank(id: u16) -> Instruction {
    backend::set_thread_entry(thread::CFG_STATE_ID_StateID.addr32(), id).unwrap()
}
fn run(
    dev: &mut harness::Dev<'_>,
    p: &[Instruction],
    t: usize,
    source: &[u8],
    dest: &[u8],
) -> Vec<u8> {
    let mut roles = [&[][..]; 3];
    roles[t] = p;
    let mut outcome = harness::run(
        dev,
        &Run::roles(Roles {
            unpack: roles[0],
            math: roles[1],
            pack: roles[2],
        })
        .dump_rows(0)
        .stage(&[(SRC, source), (DST, dest)])
        .read_back(&[(DST, dest.len()), (SRC, source.len())]),
    );
    assert_eq!(outcome.l1[1], source, "source/guards unchanged");
    outcome.l1.remove(0)
}
// Capture address/update at issue; merge partial result only at completion.
fn issue(g: &mut [u32; 64], base: usize, half: usize, inc: usize) -> usize {
    let shift = half % 2 * 16;
    let offset = (g[half / 2] >> shift) & 65535;
    let address = g[base] as usize * 16 + offset as usize;
    g[half / 2] =
        (g[half / 2] & !(65535 << shift)) | (((offset as usize + inc) as u32 & 65535) << shift);
    address
}
fn complete_load(g: &mut [u32; 64], memory: &[u8], at: usize, data: usize, bytes: usize) {
    for i in 0..bytes {
        let r = data + i / 4;
        let shift = i % 4 * 8;
        g[r] = (g[r] & !(255 << shift)) | ((memory[at + i] as u32) << shift);
    }
}
#[test]
fn checked_helpers_and_independent_async_alias_model() {
    for width in WIDTHS {
        for inc in INCS {
            for half in [28, 29, 126, 127] {
                for store in [false, true] {
                    let i = if store {
                        scalar::store_indirect_l1(width, OffsetHalf::new(half).unwrap(), inc, 8, 12)
                    } else {
                        scalar::load_indirect(width, OffsetHalf::new(half).unwrap(), inc, 8, 12)
                    }
                    .unwrap();
                    assert_eq!(
                        i.operand("Size"),
                        Some(if store {
                            match width {
                                Width::Quadword => 0,
                                Width::Word => 2,
                                Width::Halfword => 1,
                                Width::Byte => 3,
                            }
                        } else {
                            width as u32
                        })
                    );
                    assert_eq!(i.operand("OffsetHalfReg"), Some(half));
                    assert_eq!(i.operand("OffsetIncrement"), Some(inc as u32));
                }
                let mut g = [0xaabbccdd; 64];
                g[12] = 0;
                g[14] = 0x00200020;
                let old = g[8];
                let at = issue(&mut g, 12, 28 + (half % 2) as usize, inc.bytes() as usize);
                assert_eq!(g[8], old, "load is asynchronous");
                let memory: Vec<_> = (0..128).map(|i| i as u8).collect();
                complete_load(&mut g, &memory, at, 8, width.bytes() as usize);
                let bits = width.bytes().min(4) * 8;
                let mask = u32::MAX >> (32 - bits);
                assert_eq!(g[8] & !mask, old & !mask);
            }
        }
    }
    for width in WIDTHS {
        for data in [0, 63, 64, u32::MAX] {
            let got =
                scalar::load_indirect(width, OffsetHalf::new(28).unwrap(), Inc::None, data, 12);
            assert_eq!(
                got.is_ok(),
                data < 64 && (width != Width::Quadword || data % 4 == 0)
            );
        }
    }
    for (d, b, h) in [
        (8, 8, 28),
        (8, 12, 16),
        (8, 12, 24),
        (9, 12, 28),
        (60, 63, 28),
    ] {
        assert!(scalar::load_indirect(
            Width::Quadword,
            OffsetHalf::new(h).unwrap(),
            Inc::None,
            d,
            b
        )
        .is_err());
    }
    assert!(OffsetHalf::new(128).is_err());
    let mut g = [0u32; 64];
    g[8] = 65535;
    let at = issue(&mut g, 12, 16, 2);
    assert_eq!(at, 65535);
    assert_eq!(g[8], 1, "store source alias sees increment");
    complete_load(&mut g, &[9], 0, 8, 1);
    assert_eq!(g[8], 9, "asynchronous load can overwrite increment");
    for (wait, cond, block) in [
        (
            backend::wait_for_scalar(Before::CONFIG).unwrap(),
            backend::cond::SCALAR_OUTSTANDING,
            backend::block::SCALAR,
        ),
        (
            backend::wait_for_mover(Before::CONFIG).unwrap(),
            backend::cond::MOVER_OUTSTANDING,
            backend::block::MOVER,
        ),
    ] {
        assert_eq!(wait.operand("ConditionMask"), Some(cond));
        assert_eq!(
            wait.operand("BlockMask"),
            Some(block | Before::CONFIG.mask())
        );
    }
}
#[cfg(feature = "silicon")]
fn scalar_program(width: Width, inc: Inc, high: bool, wrong_offset: bool) -> Vec<Instruction> {
    let half = OffsetHalf::new(if high { 29 } else { 28 }).unwrap();
    let mut p = vec![bank(0)];
    for r in 8..12 {
        p.extend(backend::set_gpr(r, 0xaabbccdd + r).unwrap());
    }
    p.extend(backend::set_gpr(12, (SRC / 16) as u32).unwrap());
    p.extend(backend::set_gpr(13, (DST / 16) as u32).unwrap());
    p.extend(
        backend::set_gpr(
            14,
            if high {
                32 << 16 | 0xbeef
            } else {
                0xbeef0000 | 32
            },
        )
        .unwrap(),
    );
    p.extend(backend::set_gpr(15, 64).unwrap());
    p.push(scalar::load_indirect(width, half, inc, 8, 12).unwrap());
    p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
    if wrong_offset {
        p.extend(backend::set_gpr(15, 80).unwrap());
    }
    // Store the complete result group to observe partial-load preservation.
    p.push(
        scalar::store_indirect_l1(
            Width::Quadword,
            OffsetHalf::new(30).unwrap(),
            Inc::None,
            8,
            13,
        )
        .unwrap(),
    );
    p.extend(backend::set_gpr(15, 96).unwrap());
    p.push(
        scalar::store_indirect_l1(Width::Word, OffsetHalf::new(30).unwrap(), Inc::None, 14, 13)
            .unwrap(),
    );
    p.extend(backend::set_gpr(15, 112).unwrap());
    p.push(scalar::store_indirect_l1(width, OffsetHalf::new(30).unwrap(), inc, 8, 13).unwrap());
    p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
    p
}
#[cfg(feature = "silicon")]
fn check_scalar(dev: &mut harness::Dev<'_>, t: usize) {
    let source: Vec<_> = (0..256).map(|i| (i * 37 + 11) as u8).collect();
    let dest = vec![0x5a; 256];
    for width in WIDTHS {
        for inc in INCS {
            for high in [false, true] {
                let p = scalar_program(width, inc, high, false);
                let got = run(dev, &p, t, &source, &dest);
                let mut expected = dest.clone();
                for r in 0..4 {
                    expected[64 + r * 4..68 + r * 4]
                        .copy_from_slice(&(0xaabbcce5u32 + r as u32).to_le_bytes());
                }
                expected[64..64 + width.bytes() as usize]
                    .copy_from_slice(&source[32..32 + width.bytes() as usize]);
                let off = 32 + inc.bytes();
                let updated = if high {
                    off << 16 | 0xbeef
                } else {
                    0xbeef0000 | off
                };
                expected[96..100].copy_from_slice(&updated.to_le_bytes());
                let result = expected[64..64 + width.bytes() as usize].to_vec();
                expected[112..112 + result.len()].copy_from_slice(&result);
                assert_eq!(got, expected, "thread {t} {width:?} {inc:?} high {high}");
            }
        }
    }
    let mutant = run(
        dev,
        &scalar_program(Width::Word, Inc::None, false, true),
        t,
        &source,
        &dest,
    );
    assert_eq!(&mutant[64..68], &dest[64..68]);
    assert_ne!(&mutant[80..84], &dest[80..84]);
}
#[test]
#[cfg(feature = "silicon")]
fn scalar_all_widths_increments_halves_threads_and_guards() {
    harness::in_device(|dev| {
        for t in 0..3 {
            check_scalar(dev, t);
        }
    });
}
fn mover_program(bytes: usize, zero: bool, id: u16, dependent: bool) -> Vec<Instruction> {
    let mut p = vec![bank(id)];
    let mut cfg = ConfigWords::new();
    cfg.set(thcon::THCON_SEC0_REG6_Source_address, (SRC / 16) as u32)
        .unwrap();
    cfg.set(
        thcon::THCON_SEC0_REG6_Destination_address,
        ((DST + 32) / 16) as u32,
    )
    .unwrap();
    cfg.set(thcon::THCON_SEC0_REG6_Buffer_size, (bytes / 16) as u32)
        .unwrap();
    cfg.set(
        thcon::THCON_SEC0_REG6_Transfer_direction,
        if zero { 0 } else { 3 },
    )
    .unwrap();
    p.extend(tt_kernels::datapath::config_program(&cfg));
    p.push(backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY).unwrap());
    p.push(encode::xmov().unwrap());
    p.push(backend::wait_for_mover(Before::EVERYTHING).unwrap());
    if dependent {
        p.extend(backend::set_gpr(12, ((DST + 32) / 16) as u32).unwrap());
        p.extend(backend::set_gpr(13, (DST / 16) as u32).unwrap());
        p.extend(backend::set_gpr(14, 0).unwrap());
        p.extend(backend::set_gpr(15, 16).unwrap());
        p.push(
            scalar::load_indirect(Width::Word, OffsetHalf::new(28).unwrap(), Inc::None, 8, 12)
                .unwrap(),
        );
        p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
        p.push(
            scalar::store_indirect_l1(Width::Word, OffsetHalf::new(30).unwrap(), Inc::None, 8, 13)
                .unwrap(),
        );
        p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
    }
    p
}
#[test]
#[cfg(feature = "silicon")]
fn mover_banks_sizes_role_handovers_and_immediate_consumers() {
    harness::in_device(|dev| {
        let src: Vec<_> = (0..4096).map(|i| (i * 31 + 17) as u8).collect();
        let dest = vec![0xa5; 4160];
        for repeat in 0..2 {
            for t in 0..3 {
                for id in 0..2 {
                    for zero in [false, true] {
                        for bytes in [16, 32, 256, 4096] {
                            let got =
                                run(dev, &mover_program(bytes, zero, id, true), t, &src, &dest);
                            let mut expected = dest.clone();
                            if zero {
                                expected[32..32 + bytes].fill(0);
                            } else {
                                expected[32..32 + bytes].copy_from_slice(&src[..bytes]);
                            }
                            let word = expected[32..36].to_vec();
                            expected[16..20].copy_from_slice(&word);
                            assert_eq!(
                                got, expected,
                                "repeat {repeat} role {t} bank {id} zero {zero} size {bytes}"
                            );
                        }
                    }
                }
            }
        }
    });
}
#[test]
#[cfg(feature = "silicon")]
fn dmanop_preserves_gprs_configuration_and_memory() {
    harness::in_device(|dev| {
        for t in 0..3 {
            let mut p = scalar_program(Width::Quadword, Inc::None, false, false);
            let baseline = run(dev, &p, t, &vec![0x13; 256], &vec![0xa5; 256]);
            let at = p
                .iter()
                .position(|i| i.def().mnemonic() == "STOREIND")
                .unwrap();
            p.splice(at..at, std::iter::repeat_n(scalar::dma_nop(), 17));
            let got = run(dev, &p, t, &vec![0x13; 256], &vec![0xa5; 256]);
            assert_eq!(got, baseline);
        }
    });
}
#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_opcode_probes_with_surviving_control() {
    assert!(harness::survives(|dev| {
        harness::run(dev, &Run::new(&[backend::nop()]).dump_rows(0));
    }));
    for width in WIDTHS {
        for load in [false, true] {
            let mut p = vec![bank(0)];
            p.extend(backend::set_gpr(12, (SRC / 16) as u32).unwrap());
            p.extend(backend::set_gpr(14, 0).unwrap());
            p.extend(backend::set_gpr(8, 0).unwrap());
            p.push(
                if load {
                    scalar::load_indirect(width, OffsetHalf::new(28).unwrap(), Inc::None, 8, 12)
                } else {
                    scalar::store_indirect_l1(width, OffsetHalf::new(28).unwrap(), Inc::None, 8, 12)
                }
                .unwrap(),
            );
            p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
            let survives = harness::survives(|dev| {
                harness::run(
                    dev,
                    &Run::new(&p).dump_rows(0).stage(&[(SRC, &[0x13; 256])]),
                );
            });
            assert!(!survives,"ttsim now supports load={load} width={width:?}; update divergence and enable semantic gate");
        }
    }
    assert!(!harness::survives(|dev| {
        harness::run(dev, &Run::new(&[encode::xmov().unwrap()]).dump_rows(0));
    }));
    for zero in [false, true] {
        assert!(!harness::survives(|dev| {
            run(
                dev,
                &mover_program(16, zero, 0, false),
                0,
                &[0x13; 256],
                &[0xa5; 256],
            );
        }));
    }
    assert!(harness::survives(|dev| {
        run(dev, &[scalar::dma_nop()], 0, &[0x13; 256], &[0xa5; 256]);
    }));
}

#[test]
#[cfg(feature = "silicon")]
fn measure_indirect_widths() {
    harness::in_device(|dev| {
        for size in 0..4 {
            let mut p = vec![bank(0)];
            for r in 8..12 {
                p.extend(backend::set_gpr(r, 0x11223344 + r).unwrap());
            }
            p.extend(backend::set_gpr(13, (DST / 16) as u32).unwrap());
            p.extend(backend::set_gpr(15, 0).unwrap());
            p.push(
                encode::StoreindL1::ZERO
                    .size(size)
                    .offset_half_reg(30)
                    .data_reg(8)
                    .addr_reg(13)
                    .encode()
                    .unwrap(),
            );
            p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
            let got = run(dev, &p, 0, &[0x19; 256], &[0xa5; 256]);
            let bytes = [16, 2, 4, 1][size as usize];
            let mut expected = vec![0xa5; 256];
            let data: Vec<_> = (8..12u32)
                .flat_map(|r| (0x11223344 + r).to_le_bytes())
                .collect();
            expected[..bytes].copy_from_slice(&data[..bytes]);
            assert_eq!(got, expected, "raw STORE size {size}");
        }
    });
}

#[test]
#[cfg(feature = "silicon")]
fn negative_controls_wrong_offset_width_high_bits_and_source() {
    harness::in_device(|dev| {
        let source: Vec<_> = (0..256).map(|i| (i * 29 + 17) as u8).collect();
        let guard = vec![0xa5; 256];
        let correct = run(
            dev,
            &scalar_program(Width::Word, Inc::None, false, false),
            0,
            &source,
            &guard,
        );
        let shifted = run(
            dev,
            &scalar_program(Width::Word, Inc::None, false, true),
            0,
            &source,
            &guard,
        );
        assert_ne!(shifted, correct);
        let narrow = run(
            dev,
            &scalar_program(Width::Halfword, Inc::None, false, false),
            0,
            &source,
            &guard,
        );
        assert_ne!(narrow, correct, "wrong width leaves high bits intact");
        assert_eq!(&narrow[66..68], &0xaabbcce5u32.to_le_bytes()[2..]);
        let mut wrong = mover_program(16, false, 0, false);
        let at = wrong
            .iter()
            .position(|i| i.def().mnemonic() == "XMOV")
            .unwrap();
        let mut cfg = ConfigWords::new();
        cfg.set(
            thcon::THCON_SEC0_REG6_Source_address,
            ((SRC + 16) / 16) as u32,
        )
        .unwrap();
        let mut insert = tt_kernels::datapath::config_program(&cfg);
        insert.push(
            backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY).unwrap(),
        );
        wrong.splice(at..at, insert);
        let got = run(dev, &wrong, 0, &source, &guard);
        assert_eq!(&got[32..48], &source[16..32]);
        assert_ne!(&got[32..48], &source[..16]);
    });
}

#[test]
#[cfg(feature = "silicon")]
fn register_boundaries_and_dmanop_configuration_state() {
    harness::in_device(|dev| {
        for t in 0..3 {
            for (data, base, half) in [(0, 12, 127), (60, 0, 29)] {
                let mut p = vec![bank(0)];
                for r in data..data + 4 {
                    p.extend(backend::set_gpr(r, 0xaabbccdd).unwrap());
                }
                p.extend(backend::set_gpr(base, (SRC / 16) as u32).unwrap());
                p.extend(backend::set_gpr(half / 2, 32 << 16 | 0xbeef).unwrap());
                p.push(
                    scalar::load_indirect(
                        Width::Quadword,
                        OffsetHalf::new(half).unwrap(),
                        Inc::Bytes16,
                        data,
                        base,
                    )
                    .unwrap(),
                );
                p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
                // Publish data through a full, nonaliasing four-register store.
                p.extend(backend::set_gpr(12, (DST / 16) as u32).unwrap());
                p.extend(backend::set_gpr(14, 0).unwrap());
                p.push(
                    scalar::store_indirect_l1(
                        Width::Quadword,
                        OffsetHalf::new(28).unwrap(),
                        Inc::None,
                        data,
                        12,
                    )
                    .unwrap(),
                );
                p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
                // Establish config state before DMANOP and read it back after.
                p.extend(backend::set_gpr(24, 0x12345678).unwrap());
                p.push(
                    backend::write_word(24, thcon::THCON_SEC0_REG3_Base_address.addr32()).unwrap(),
                );
                p.push(
                    backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY)
                        .unwrap(),
                );
                p.extend(std::iter::repeat_n(scalar::dma_nop(), 17));
                p.extend(
                    backend::read_word(24, thcon::THCON_SEC0_REG3_Base_address.addr32()).unwrap(),
                );
                p.extend(backend::set_gpr(14, 64).unwrap());
                p.push(
                    scalar::store_indirect_l1(
                        Width::Word,
                        OffsetHalf::new(28).unwrap(),
                        Inc::None,
                        24,
                        12,
                    )
                    .unwrap(),
                );
                p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
                let source: Vec<_> = (0..256).map(|i| (i * 31 + 17) as u8).collect();
                let guard = vec![0xa5; 256];
                let got = run(dev, &p, t, &source, &guard);
                let mut want = guard;
                want[..16].copy_from_slice(&source[32..48]);
                want[64..68].copy_from_slice(&0x12345678u32.to_le_bytes());
                assert_eq!(got, want);
            }
        }
    });
}

#[test]
#[ignore = "release silicon local movement benchmark"]
#[cfg(feature = "silicon")]
fn benchmark_local_copy_zero() {
    use std::time::Instant;
    use tt_kernels::{
        code::Loop,
        runtime::{Kernel, Schedule},
        session::{Session, TileChoice},
    };
    let release = !cfg!(debug_assertions);
    assert!(release, "benchmarks require --release");
    println!("\nMEASURE resident XMOV local copies and zeros");
    tt_ttsim::fork_scope(|| {
        let mut s=Session::open_card(tt_tests::backend::device_index(),tt_firmware_images::ROLES,TileChoice::First).unwrap();
        for bytes in [16,128,4096] {for zero in [false,true] {
            let program=mover_program(bytes,zero,0,false);
            let start=program.iter().position(|i|i.def().mnemonic()=="XMOV").unwrap();
            let loops=[Loop{start:start as u32,len:2,count:128}];
            let input=tt_tests::bench::pattern(4096,19);
            let guard=vec![0xa5;4160];
            let mut setup=Kernel::new([&[],&[],&[]],Schedule::InOrder);
            let staged=[(SRC,input.as_slice()),(DST,guard.as_slice())];setup.stage=&staged;s.run(&setup,4_000_000).unwrap();
            let mut kernel=Kernel::new([&program,&[],&[]],Schedule::InOrder);
            kernel.loops=[&loops,&[],&[]];
            let mut samples=vec![];
            for rep in 0..9 {let now=Instant::now();s.run(&kernel,4_000_000).unwrap();let elapsed=now.elapsed().as_secs_f64();
                let mut check=Kernel::new([&[],&[],&[]],Schedule::InOrder);check.read_back=&[(DST,4160)];let got=s.run(&check,4_000_000).unwrap().l1.remove(0);let mut want=guard.clone();if zero {want[32..32+bytes].fill(0);}else{want[32..32+bytes].copy_from_slice(&input[..bytes]);}assert_eq!(got,want);
                if rep>=2 {samples.push(elapsed);}
            }
            samples.sort_by(f64::total_cmp);
            println!("BENCH {{\"kind\":\"xmov_local\",\"key\":\"local_{}_{}\",\"card\":{},\"tiles\":1,\"bytes\":{bytes},\"zero\":{zero},\"warmups\":2,\"samples\":7,\"repeats\":128,\"timed\":\"host resident launch and drain, setup once, 128 local issues\",\"median\":{},\"unit\":\"us\"}}",if zero{"zero"}else{"copy"},bytes,tt_tests::backend::device_index(),samples[3]*1e6);
        }}
    }).unwrap();
}

#[test]
#[ignore = "release silicon B-core local movement baseline"]
#[cfg(feature = "silicon")]
fn benchmark_local_b_core_baseline() {
    use std::time::Instant;
    let release = !cfg!(debug_assertions);
    assert!(release, "benchmarks require --release");
    harness::in_device(|dev| {
        let tile = harness::tensix_tile();
        let w = dev
            .alloc_window(tt_device::tlb::WindowKind::TwoMib)
            .unwrap();
        let dram = dev.dram_grid(&w).unwrap();
        let mut mover =
            tt_kernels::dm::DataMover::start(dev, &w, tile, &dram, tt_firmware_images::DM_B.1)
                .unwrap();
        let input = tt_tests::bench::pattern(4096, 19);
        for (bytes, zero) in [(16, false), (128, false), (4096, false), (4096, true)] {
            let dest = if zero { DST + 16 } else { DST + 32 };
            let entry = if zero {
                [
                    tt_isa::dm::op::COPY_WORDS,
                    SRC as u32,
                    dest as u32,
                    1024,
                    0,
                    4,
                    0,
                    0,
                ]
            } else {
                [
                    tt_isa::dm::op::COPY_WORDS,
                    SRC as u32,
                    dest as u32,
                    (bytes / 4) as u32,
                    4,
                    4,
                    0,
                    0,
                ]
            };
            let entries = vec![entry; 128];
            let mut samples = vec![];
            for rep in 0..9 {
                dev.write(&w, tile, SRC, if zero { &[0; 4096] } else { &input })
                    .unwrap();
                dev.write(&w, tile, DST, &vec![0xa5; 4160]).unwrap();
                let now = Instant::now();
                mover.run_list(dev, &w, &entries).unwrap();
                let elapsed = now.elapsed().as_secs_f64();
                let mut got = vec![0; 4160];
                dev.read(&w, tile, DST, &mut got).unwrap();
                let mut want = vec![0xa5; 4160];
                let off = (dest - DST) as usize;
                if zero {
                    want[off..off + bytes].fill(0);
                } else {
                    want[off..off + bytes].copy_from_slice(&input[..bytes]);
                }
                assert_eq!(got, want);
                if rep >= 2 {
                    samples.push(elapsed);
                }
            }
            samples.sort_by(f64::total_cmp);
            println!("BENCH {{\"kind\":\"b_core_local\",\"key\":\"local_b_{}_{}\",\"card\":{},\"tiles\":1,\"bytes\":{bytes},\"zero\":{zero},\"warmups\":2,\"samples\":7,\"repeats\":128,\"timed\":\"host list dispatch and drain, 128 software transfers\",\"median\":{},\"unit\":\"us\"}}",if zero{"zero"}else{"copy"},bytes,tt_tests::backend::device_index(),samples[3]*1e6);
        }
    });
}
