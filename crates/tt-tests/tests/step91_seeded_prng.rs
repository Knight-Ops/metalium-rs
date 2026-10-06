//! Seed-register characterization. This does not route Burn random to hardware.
use tt_isa::{backend, numerics::stochastic};
use tt_kernels::{
    datapath,
    sfpu::{Cond, Format, LReg, Program},
};
use tt_tests::harness::{self, Run};

/// Exercise a direct RISC-V `sw` to Config, with complete coprocessor drains
/// between seed writes. T1's Dst mapping is supported on both targets.
#[test]
fn riscv_seed_store_characterization() {
    use tt_device::tlb::WindowKind;
    use tt_isa::{mailbox, tensix::Core};
    harness::in_device(|dev| {
        let tile = harness::tensix_tile();
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let mut code = datapath::thread_config();
        let mut p = Program::new();
        p.read_prng(LReg::L0);
        p.store(LReg::L0, Format::Int32, 0);
        p.read_prng(LReg::L0);
        p.store(LReg::L0, Format::Int32, 4);
        code.extend(p.finish());
        code.push(backend::wait_for_sfpu(backend::Before::EVERYTHING).unwrap());
        for seed in [
            0,
            1,
            0x12345678,
            0x80000000,
            0x55555555,
            u32::MAX,
            // Finish with a nonabsorbing stream for later programs on this tile.
            0xaaaaaaaa,
        ] {
            dev.release_tensix_backend(&w, tile).unwrap();
            let d = mailbox::Descriptor {
                thread_index: 1,
                program_len: code.len() as u32,
                ..Default::default()
            };
            for (at, value) in d.writes(mailbox::role::Mailbox::single_core()) {
                dev.write32(&w, tile, at, value).unwrap();
            }
            dev.write32(&w, tile, mailbox::OPERAND_A, seed).unwrap();
            dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
            dev.write(&w, tile, mailbox::PROGRAM, &harness::program_bytes(&code))
                .unwrap();
            dev.load_and_start(
                &w,
                tile,
                Core::T1,
                tt_firmware_images::PRNG_SEED,
                tt_firmware_images::LOAD_ADDRESS,
            )
            .unwrap();
            dev.wait_for_status(&w, tile, 400_000, |s| s == mailbox::status::DONE)
                .unwrap()
                .unwrap();
            let mut snapshots = [[0u32; 32]; 4];
            for (phase, snapshot) in snapshots.iter_mut().enumerate() {
                for (lane, first) in snapshot.iter_mut().enumerate() {
                    let row = phase as u32 * 8 + lane as u32 / 8;
                    let col = lane as u32 % 8 * 2;
                    *first = dev
                        .read32(&w, tile, mailbox::dump_offset(row, col))
                        .unwrap();
                    let second = dev
                        .read32(&w, tile, mailbox::dump_offset(row + 4, col))
                        .unwrap();
                    assert_eq!(
                        second,
                        stochastic::advance(*first),
                        "phase {phase} lane {lane}"
                    );
                }
            }
            println!("RISC-V seed={seed:08x} snapshots={snapshots:08x?}");
            assert_eq!(snapshots[0], snapshots[1], "identical seed must restart");
            assert_eq!(snapshots[0], snapshots[3], "restart after a different seed");
            // Omitting all seed stores cannot satisfy both repetition and the
            // complementary-seed difference, including the absorbing state.
            assert_ne!(snapshots[0], snapshots[2], "seed must affect lane states");
            if seed == u32::MAX {
                assert_eq!(snapshots[0], [u32::MAX; 32], "absorbing seed");
            }
            for snapshot in snapshots {
                assert!(snapshot.windows(2).all(|p| p[1] & !3 == p[0] << 2));
            }
        }
    });
}

#[test]
fn seed_register_advance_and_predication() {
    harness::in_device(|dev| {
        for seed in [0, 1, 0x12345678, u32::MAX] {
            let mut code = datapath::thread_config();
            code.extend(backend::write_prng_seed(8, seed).unwrap());
            let mut p = Program::new();
            p.read_prng(LReg::L0);
            p.store(LReg::L0, Format::Int32, 0);
            p.read_prng(LReg::L1);
            p.store(LReg::L1, Format::Int32, 4);
            // L15's lane indices select half the lanes. Disabled lanes must
            // neither overwrite L2 nor advance their generator.
            p.loadi_bits(LReg::L3, 2);
            p.and(LReg::LANE_X2, LReg::L3, LReg::L3);
            p.loadi_bits(LReg::L2, 0x55aa55aa);
            p.if_(Cond::Eq0(LReg::L3), |p| p.read_prng(LReg::L2));
            p.store(LReg::L2, Format::Int32, 8);
            p.read_prng(LReg::L2);
            p.store(LReg::L2, Format::Int32, 12);
            code.extend(p.finish());
            code.push(backend::wait_for_sfpu(backend::Before::EVERYTHING).unwrap());
            code.extend(backend::write_prng_seed(8, seed).unwrap());
            let mut p = Program::new();
            p.read_prng(LReg::L0);
            p.store(LReg::L0, Format::Int32, 2);
            code.extend(p.finish());
            let out = harness::run(dev, &Run::new(&code).dump_rows(16));
            // A silicon debug read distinguishes a missing register write
            // from the measured lack of an immediate stream reset.
            #[cfg(feature = "silicon")]
            {
                let w = dev
                    .alloc_window(tt_device::tlb::WindowKind::TwoMib)
                    .unwrap();
                let tile = harness::tensix_tile();
                dev.write32(
                    &w,
                    tile,
                    tt_isa::tensix::CFGREG_RD_CNTL,
                    tt_isa::cfg::generated::global::PRNG_SEED_Seed_Val.addr32() as u32,
                )
                .unwrap();
                harness::advance(dev, 64);
                assert_eq!(
                    dev.read32(&w, tile, tt_isa::tensix::CFGREG_RDDATA).unwrap(),
                    seed,
                    "seed register write must land"
                );
            }
            let at = |group: usize, lane: usize| out.dst_at(group * 4 + lane / 8, lane % 8 * 2);
            println!(
                "seed={seed:08x} initial={:08x?}",
                (0..32).map(|i| at(0, i)).collect::<Vec<_>>()
            );
            let initial: Vec<_> = (0..32).map(|i| at(0, i)).collect();
            let identical = initial.iter().all(|&v| v == initial[0]);
            let shifted = initial.windows(2).all(|p| p[1] & !3 == p[0] << 2);
            assert!(
                identical || shifted,
                "measured lane initialization correlation changed"
            );
            println!("lane initialization: identical={identical} adjacent_share_30_bits={shifted}");
            for i in 0..32 {
                let first = at(0, i);
                let second = stochastic::advance(first);
                assert_eq!(at(1, i), second, "lane {i} advance");
                assert_eq!(
                    at(2, i),
                    if i % 2 == 0 {
                        stochastic::advance(second)
                    } else {
                        0x55aa55aa
                    }
                );
                assert_eq!(
                    at(3, i),
                    if i % 2 == 0 {
                        stochastic::advance(stochastic::advance(second))
                    } else {
                        stochastic::advance(second)
                    }
                );
                let reseeded = out.dst_at(i / 8, i % 8 * 2 + 1);
                if cfg!(feature = "silicon") {
                    let continued = if i % 2 == 0 {
                        stochastic::advance(stochastic::advance(stochastic::advance(first)))
                    } else {
                        stochastic::advance(stochastic::advance(first))
                    };
                    assert_eq!(
                        reseeded,
                        stochastic::advance(continued),
                        "silicon WRCFG path continues lane {i}"
                    );
                } else {
                    assert_eq!(reseeded, first, "lane {i} reseed");
                }
            }
        }
    });
}
