//! Restricted Scalar Unit MMIO: `LOADREG`, `STOREREG`, `STOREIND` (MMIO).
//!
//! The only addressable words are `SW_INT_PC[28..=31]` of the tile PIC
//! (`tt_isa::mmio_reg`). Host tests pin the allowlist, the address arithmetic
//! and every refusal against literal addresses from `PIC.md` and a separate
//! transcription of the pinned `STOREIND` model. ttsim executes none of these
//! forms (divergence rows 78-79), so the simulator gate records the refusals with
//! surviving controls, and the semantic gates are silicon-only.
use tt_isa::{
    backend::{self, Before},
    isa::generated::encode,
    mmio_reg::{self as mmio, MmioError, Scratch},
    scalar::{OffsetHalf, OffsetIncrement as Inc},
};

/// Literal `SW_INT_PC[i]` addresses from `PIC.md` (`0xFFB1_30A8 + 4 i`).
const PIC_TARGETS: [(Scratch, u32); 4] = [
    (Scratch::Irq28, 0xFFB1_3118),
    (Scratch::Irq29, 0xFFB1_311C),
    (Scratch::Irq30, 0xFFB1_3120),
    (Scratch::Irq31, 0xFFB1_3124),
];

/// The pinned `STOREIND.md` address model, transcribed independently of the
/// module under test: `0xFFB00000 + ((GPR + (Offset >> 4)) & 0x000FFFFC)`.
fn doc_address(gpr: u32, offset: u16) -> u32 {
    doc_address_shift(gpr, offset, 4)
}
/// Same, with the shift as a parameter, for the mutant.
fn doc_address_shift(gpr: u32, offset: u16, shift: u32) -> u32 {
    let addr = gpr.wrapping_add(u32::from(offset >> shift));
    0xFFB0_0000 + (addr & 0x000F_FFFC)
}
fn doc_increment(offset: u16, inc: Inc) -> u16 {
    offset.wrapping_add(match inc as u32 {
        0 => 0,
        1 => 2,
        2 => 4,
        _ => 16,
    })
}

#[test]
fn allowlist_is_exactly_the_four_pinned_words() {
    for (t, a) in PIC_TARGETS {
        assert_eq!(t.address(), a);
        assert_eq!(t.addr_lo() << 2, a - 0xFFB0_0000);
        assert_eq!(t.irq_mask(), 1 << ((a - 0xFFB1_30A8) / 4));
    }
    // Every AddrLo is classified; exactly four resolve, none is truncated.
    let mut accepted = vec![];
    for lo in 0..mmio::ADDR_LO_LIMIT {
        if let Ok(t) = Scratch::from_addr_lo(lo) {
            accepted.push((lo, t));
        }
    }
    assert_eq!(accepted.len(), 4);
    for (lo, t) in accepted {
        assert_eq!(0xFFB0_0000 + (lo << 2), t.address());
    }
    assert!(matches!(
        Scratch::from_addr_lo(mmio::ADDR_LO_LIMIT),
        Err(MmioError::AddrLoTooLarge { .. })
    ));
    // AddrLo has 18 bits: a value that would alias a target modulo 2^18 words is
    // refused, not folded.
    assert!(Scratch::from_addr_lo(Scratch::Irq31.addr_lo() + (1 << 18)).is_err());
}

#[test]
fn every_neighbour_and_forbidden_region_is_refused() {
    // Forbidden below 0xFFB1_1000, including local data RAM and its start.
    for a in (0xFFB0_0000u32..0xFFB1_1000).step_by(0x40) {
        assert_eq!(
            Scratch::from_address(a),
            Err(MmioError::Forbidden { address: a })
        );
    }
    for a in [
        0xFFAF_FFFC,
        0xFFC0_0000,
        0xFFC0_3124,
        0x0000_0000,
        0xFFFF_FFFC,
    ] {
        assert!(matches!(
            Scratch::from_address(a),
            Err(MmioError::Forbidden { .. })
        ));
    }
    // The whole TDMA/debug/PIC/local-RAM-aperture/NoC prefix and every PIC word
    // except the four are NotAllowlisted: SW_INT[i] (raise and read-clear),
    // HW_INT, SW_INT_EN, INT_NO, the `pc` snapshots, SW_INT_PC[0..28], HW_INT_PC.
    let ok: Vec<u32> = PIC_TARGETS.iter().map(|t| t.1).collect();
    for a in (0xFFB1_1000u32..0xFFB8_1000).step_by(4) {
        let r = Scratch::from_address(a);
        if ok.contains(&a) {
            assert!(r.is_ok());
        } else {
            assert_eq!(r, Err(MmioError::NotAllowlisted { address: a }), "{a:#x}");
        }
    }
    // Misaligned addresses never match.
    for (_, a) in PIC_TARGETS {
        for d in 1..4 {
            assert!(Scratch::from_address(a + d).is_err());
        }
    }
}

#[test]
fn irq_enable_precondition_is_checked_per_core() {
    let t = Scratch::Irq30;
    assert_eq!(t.require_irq_disabled(0, 0), Ok(()));
    assert_eq!(t.require_irq_disabled(!(1 << 30), !(1 << 30)), Ok(()));
    for (b, nc) in [(1 << 30, 0), (0, 1 << 30), (u32::MAX, 0)] {
        assert!(matches!(
            t.require_irq_disabled(b, nc),
            Err(MmioError::InterruptEnabled { .. })
        ));
    }
}

#[test]
fn encoded_words_carry_the_checked_operands() {
    for (t, a) in PIC_TARGETS {
        let lo = (a - 0xFFB0_0000) >> 2;
        let l = mmio::load(9, t).unwrap();
        let s = mmio::store(10, t).unwrap();
        assert_eq!(l.def().mnemonic(), "LOADREG");
        assert_eq!(s.def().mnemonic(), "STOREREG");
        // Raw words from the pinned layouts: opcode, 6-bit GPR at 18, 18-bit AddrLo.
        assert_eq!(l.word(), 0x68 << 24 | 9 << 18 | lo);
        assert_eq!(s.word(), 0x67 << 24 | 10 << 18 | lo);
    }
    assert!(mmio::load(64, Scratch::Irq28).is_err());
    assert!(mmio::store(64, Scratch::Irq28).is_err());
    let seq = mmio::store_then_read_back(8, 9, Scratch::Irq29).unwrap();
    let names: Vec<_> = seq.iter().map(|i| i.def().mnemonic()).collect();
    assert_eq!(names, ["STOREREG", "STALLWAIT", "LOADREG", "STALLWAIT"]);
    for w in [seq[1], seq[3]] {
        assert_eq!(
            w.word(),
            backend::wait_for_scalar(Before::EVERYTHING).unwrap().word()
        );
    }
    assert!(mmio::store_then_read_back(8, 8, Scratch::Irq29).is_err());
}

#[test]
fn indirect_model_matches_the_transcribed_pinned_function() {
    let offsets = [
        0u16, 1, 15, 16, 17, 0x30, 0x40, 0xFF, 0x100, 0xFFF, 0x1000, 0xFFF0, 0xFFFF,
    ];
    let mut checked = 0;
    for (t, a) in PIC_TARGETS {
        for off in offsets {
            let plan = mmio::plan_indirect(t, off).unwrap();
            assert_eq!(plan.offset_half, off);
            assert_eq!(doc_address(plan.address_gpr, off), a, "{t:?} {off:#x}");
            assert_eq!(mmio::resolve_indirect(plan.address_gpr, off), Ok(t));
            checked += 1;
        }
    }
    assert_eq!(checked, 52);
    // Cross-check resolve_indirect against the transcription over a grid of
    // arbitrary GPR values: whenever the module accepts, the transcription lands
    // on the target with no dropped bits; whenever the transcription lands on a
    // target only by dropping bits, the module refuses.
    let mut accepted = 0;
    let mut aliased = 0;
    for gpr in (0u32..0x2_0000).chain([0x1_3118, 0x11_3118, 0xFFFF_FFFF, 0x8001_3118]) {
        for off in [0u16, 0x10, 0x40, 0xFFF0, 0x1234] {
            let doc = doc_address(gpr, off);
            let sum = gpr.wrapping_add(u32::from(off >> 4));
            let lossless = sum & !0x000F_FFFC == 0;
            match mmio::resolve_indirect(gpr, off) {
                Ok(t) => {
                    assert!(lossless);
                    assert_eq!(t.address(), doc);
                    accepted += 1;
                }
                Err(MmioError::Truncated { sum: s }) => {
                    assert!(!lossless);
                    assert_eq!(s, sum);
                    if Scratch::from_address(doc).is_ok() {
                        aliased += 1;
                    }
                }
                Err(_) => assert!(lossless || Scratch::from_address(doc).is_err()),
            }
        }
    }
    assert!(accepted > 0);
    assert!(aliased > 0, "the grid must reach an aliasing case");
    // Increments follow the pinned switch, not bytes of MMIO address.
    for inc in [Inc::None, Inc::Bytes2, Inc::Bytes4, Inc::Bytes16] {
        for off in [0u16, 14, 0xFFF0, 0xFFFE, 0xFFFF] {
            assert_eq!(mmio::offset_after(off, inc), doc_increment(off, inc));
        }
    }
    // Sixteen raw increments of 16 advance the sum by sixteen, four words.
    let mut off = 0u16;
    for _ in 0..16 {
        off = mmio::offset_after(off, Inc::Bytes16);
    }
    assert_eq!(off >> 4, 16);
}

#[test]
fn indirect_shift_is_four_and_unlike_the_l1_variant() {
    // The L1 variant adds the offset unshifted to `16 * base`; this one shifts.
    let plan = mmio::plan_indirect(Scratch::Irq28, 0x40).unwrap();
    assert_eq!(plan.address_gpr, Scratch::Irq28.window_offset() - 4);
    assert_eq!(
        mmio::resolve_indirect(plan.address_gpr, 0x40),
        Ok(Scratch::Irq28)
    );
    for shift in [0u32, 1, 2, 3, 5] {
        assert_ne!(
            doc_address_shift(plan.address_gpr, 0x40, shift),
            Scratch::Irq28.address(),
            "a shift of {shift} would land elsewhere"
        );
    }
}

#[test]
fn indirect_encoder_checks_registers_and_fields() {
    let half = OffsetHalf::new(14).unwrap(); // low half of GPR 7
    let i = mmio::store_indirect(half, Inc::Bytes16, 8, 6).unwrap();
    assert_eq!(i.def().key(), "STOREIND_MMIO");
    assert_eq!(i.operand("OffsetHalfReg"), Some(14));
    assert_eq!(i.operand("OffsetIncrement"), Some(3));
    assert_eq!(i.operand("DataReg"), Some(8));
    assert_eq!(i.operand("AddrReg"), Some(6));
    assert_eq!(i.word(), encode::storeind_mmio(14, 3, 8, 6).unwrap().word());
    // Distinct from the L1 variant that shares the opcode.
    let l1 = tt_isa::scalar::store_indirect_l1(
        tt_isa::scalar::TransferWidth::Word,
        half,
        Inc::Bytes16,
        8,
        6,
    )
    .unwrap();
    assert_ne!(i.word(), l1.word());
    for (data, addr, half) in [(8, 8, 14), (6, 6, 14), (7, 6, 14), (8, 7, 14), (7, 6, 15)] {
        let r = mmio::store_indirect(OffsetHalf::new(half).unwrap(), Inc::None, data, addr);
        assert!(r.is_err(), "data {data} addr {addr} half {half} must alias");
    }
    assert!(mmio::store_indirect(half, Inc::None, 64, 6).is_err());
    assert!(mmio::store_indirect(half, Inc::None, 8, 64).is_err());
}

// ---------------------------------------------------------------------------
// Simulator: refusal evidence with surviving controls (divergence rows 78-79).
// ---------------------------------------------------------------------------
#[cfg(not(feature = "silicon"))]
mod simulator {
    use super::*;
    use tt_device::tlb::WindowKind;
    use tt_isa::isa::Instruction;
    use tt_tests::harness::{self, Run};

    fn survives_program(p: &[Instruction]) -> bool {
        harness::survives(|dev| {
            harness::run(dev, &Run::new(p).dump_rows(0));
        })
    }
    fn setup() -> Vec<Instruction> {
        let mut p = vec![];
        p.extend(backend::set_gpr(8, 0x123).unwrap());
        p.extend(backend::set_gpr(6, Scratch::Irq31.window_offset()).unwrap());
        p.extend(backend::set_gpr(7, 0).unwrap());
        p
    }
    fn drained(mut p: Vec<Instruction>, i: Instruction) -> Vec<Instruction> {
        p.push(i);
        p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
        p
    }

    #[test]
    fn simulator_refuses_every_form_and_survives_the_controls() {
        // Control: the identical setup and drain without the instruction.
        let mut control = setup();
        control.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
        assert!(survives_program(&control));

        let t = Scratch::Irq31;
        // `tensix_storereg: disallowed addr=0xffb13124` (UnsupportedFunctionality).
        assert!(!survives_program(&drained(
            setup(),
            mmio::store(8, t).unwrap()
        )));
        // `tensix_decode_loadreg` (UnsupportedFunctionality).
        assert!(!survives_program(&drained(
            setup(),
            mmio::load(9, t).unwrap()
        )));
        // `tensix_decode_storeind` (UnsupportedFunctionality).
        let half = OffsetHalf::new(14).unwrap();
        assert!(!survives_program(&drained(
            setup(),
            mmio::store_indirect(half, Inc::None, 8, 6).unwrap()
        )));
        // The L1 variant is refused by the same decoder (step101), so the
        // refusal above is not evidence about the MMIO field only.
        assert!(!survives_program(&drained(
            setup(),
            tt_isa::scalar::store_indirect_l1(
                tt_isa::scalar::TransferWidth::Word,
                half,
                Inc::None,
                8,
                6
            )
            .unwrap()
        )));
    }

    #[test]
    fn simulator_storereg_refuses_the_whole_window_it_does_not_model() {
        // Only the NoC overlay window (0xFFB4_0000..) decodes at all
        // (`noc_overlay_wr32: offset=0x0`, UnimplementedFunctionality); every
        // other address in the window, including TDMA, debug, PIC and the NoC
        // registers, is `disallowed addr`. Raw encodings: the typed API refuses
        // these addresses, which is the point of the allowlist.
        for a in [
            0xFFB1_1038u32,
            0xFFB1_2224,
            0xFFB1_3118,
            0xFFB1_4000,
            0xFFB2_0000,
            0xFFB8_0000,
        ] {
            assert!(Scratch::from_address(a).is_err() || a == 0xFFB1_3118);
            let mut p = setup();
            p.push(encode::storereg(8, (a - 0xFFB0_0000) >> 2).unwrap());
            p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
            assert!(!survives_program(&p), "{a:#x}");
        }
    }

    #[test]
    fn simulator_cannot_observe_the_pic_from_the_host() {
        // Control: a register the tile MMIO switch does decode.
        assert!(harness::survives(|dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            dev.read32(&w, harness::tensix_tile(), 0xFFB1_21B0).unwrap();
        }));
        // `t_tile_mmio_wr32: addr=0xffb13124` (UnimplementedFunctionality).
        assert!(!harness::survives(|dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            dev.write32(
                &w,
                harness::tensix_tile(),
                Scratch::Irq31.address() as u64,
                1,
            )
            .unwrap();
        }));
        assert!(!harness::survives(|dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            dev.read32(&w, harness::tensix_tile(), Scratch::Irq31.address() as u64)
                .unwrap();
        }));
    }
}

// ---------------------------------------------------------------------------
// Silicon-only arms. Written, compiled, NOT run by the lane that wrote them.
// Run order: the three `silicon_probe_*` tests first, one per session, then the
// gates. Every one is isolated in `in_device`'s fork.
// ---------------------------------------------------------------------------
#[cfg(feature = "silicon")]
mod silicon {
    use super::*;
    use tt_device::tlb::WindowKind;
    use tt_isa::{
        isa::Instruction,
        scalar::{self, TransferWidth},
    };
    use tt_tests::harness::{self, Roles, Run};

    const DST: u64 = tt_isa::l1::DATA.base;
    const SENTINEL: u8 = 0xA5;

    fn run(dev: &mut harness::Dev<'_>, p: &[Instruction], t: usize) -> Vec<u8> {
        let mut roles = [&[][..]; 3];
        roles[t] = p;
        let guard = vec![SENTINEL; 64];
        let mut outcome = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: roles[0],
                math: roles[1],
                pack: roles[2],
            })
            .dump_rows(0)
            .stage(&[(DST, &guard)])
            .read_back(&[(DST, 64)]),
        );
        outcome.l1.remove(0)
    }
    fn word(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }
    /// Store GPR `src` to `DST + 0` and drain.
    fn publish(p: &mut Vec<Instruction>, src: u32) {
        p.extend(backend::set_gpr(13, (DST / 16) as u32).unwrap());
        p.extend(backend::set_gpr(15, 0).unwrap());
        p.push(
            scalar::store_indirect_l1(
                TransferWidth::Word,
                OffsetHalf::new(30).unwrap(),
                Inc::None,
                src,
                13,
            )
            .unwrap(),
        );
        p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
    }

    const WATCHED: [u32; 6] = [
        mmio::GUARD_BELOW,
        0xFFB1_3118,
        0xFFB1_311C,
        0xFFB1_3120,
        0xFFB1_3124,
        mmio::GUARD_ABOVE,
    ];
    fn is_guard(a: u32) -> bool {
        a == mmio::GUARD_BELOW || a == mmio::GUARD_ABOVE
    }

    /// Host access to the PIC words, with the precondition and restore.
    struct Pic<'a> {
        dev: &'a mut harness::Dev<'a>,
        w: tt_device::Window,
        saved: Vec<(u32, u32)>,
    }
    impl<'a> Pic<'a> {
        fn open(dev: &'a mut harness::Dev<'a>) -> Self {
            harness::assert_on_silicon();
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            let tile = harness::tensix_tile();
            let b = dev.read32(&w, tile, mmio::BRISC_SW_INT_EN as u64).unwrap();
            let nc = dev.read32(&w, tile, mmio::NCRISC_SW_INT_EN as u64).unwrap();
            for t in Scratch::ALL {
                t.require_irq_disabled(b, nc).unwrap();
            }
            let mut saved = vec![];
            for a in WATCHED {
                saved.push((a, dev.read32(&w, tile, a as u64).unwrap()));
            }
            Self { dev, w, saved }
        }
        fn read(&mut self, a: u32) -> u32 {
            self.dev
                .read32(&self.w, harness::tensix_tile(), a as u64)
                .unwrap()
        }
        fn write(&mut self, a: u32, v: u32) {
            assert!(WATCHED.contains(&a) && !is_guard(a));
            self.dev
                .write32(&self.w, harness::tensix_tile(), a as u64, v)
                .unwrap();
        }
        fn snapshot(&mut self) -> Vec<(u32, u32)> {
            WATCHED.iter().map(|&a| (a, self.read(a))).collect()
        }
        /// Put the four allowlisted words back; guards are only compared.
        fn restore(&mut self) {
            for (a, v) in self.saved.clone() {
                if !is_guard(a) {
                    self.write(a, v);
                }
            }
            let now = self.snapshot();
            assert_eq!(now, self.saved, "PIC words restored");
        }
    }

    /// Probe 1 (UNVERIFIED STOREREG): one store, one drain, host reads.
    #[test]
    fn silicon_probe_storereg_to_pic_scratch() {
        harness::in_device(|dev| {
            let mut pic = Pic::open(dev);
            let t = Scratch::Irq31;
            let mut p = vec![];
            p.extend(backend::set_gpr(8, 0x0000_0123).unwrap());
            p.push(mmio::store(8, t).unwrap());
            p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
            run(pic.dev, &p, 0);
            let got = pic.read(t.address());
            let rest = pic.snapshot();
            let saved = pic.saved.clone();
            pic.restore();
            assert_eq!(got, 0x123);
            for (a, v) in rest {
                if a != t.address() {
                    let before = saved.iter().find(|s| s.0 == a).unwrap().1;
                    assert_eq!(v, before, "{a:#x} changed");
                }
            }
        });
    }

    /// Probe 2 (UNVERIFIED LOADREG): the host stages a word, one load, publish.
    #[test]
    fn silicon_probe_loadreg_from_pic_scratch() {
        harness::in_device(|dev| {
            let mut pic = Pic::open(dev);
            let t = Scratch::Irq30;
            pic.write(t.address(), 0x0BAD_F00D);
            let mut p = vec![];
            p.push(mmio::load(9, t).unwrap());
            p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
            publish(&mut p, 9);
            let l1 = run(pic.dev, &p, 0);
            let after = pic.read(t.address());
            pic.restore();
            assert_eq!(word(&l1, 0), 0x0BAD_F00D);
            assert_eq!(after, 0x0BAD_F00D, "a load does not modify the word");
        });
    }

    /// Probe 3 (UNVERIFIED STOREIND MMIO): one indirect store, offset zero.
    #[test]
    fn silicon_probe_storeind_mmio_to_pic_scratch() {
        harness::in_device(|dev| {
            let mut pic = Pic::open(dev);
            let t = Scratch::Irq29;
            let plan = mmio::plan_indirect(t, 0).unwrap();
            let mut p = vec![];
            p.extend(backend::set_gpr(8, 0x0000_0456).unwrap());
            p.extend(backend::set_gpr(6, plan.address_gpr).unwrap());
            p.extend(backend::set_gpr(7, 0).unwrap());
            p.push(mmio::store_indirect(OffsetHalf::new(14).unwrap(), Inc::None, 8, 6).unwrap());
            p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
            run(pic.dev, &p, 0);
            let got = pic.read(t.address());
            pic.restore();
            assert_eq!(got, 0x456);
        });
    }

    /// Gate: every target and thread, with the read-back fence and guards.
    #[test]
    fn store_then_read_back_all_targets_and_threads() {
        harness::in_device(|dev| {
            let mut pic = Pic::open(dev);
            for thread in 0..3 {
                for (k, t) in Scratch::ALL.into_iter().enumerate() {
                    let value = 0x0010_0000 * (thread as u32 + 1) + 0x111 * (k as u32 + 1);
                    let mut p = vec![];
                    p.extend(backend::set_gpr(8, value).unwrap());
                    p.extend(backend::set_gpr(9, 0).unwrap());
                    p.extend(mmio::store_then_read_back(8, 9, t).unwrap());
                    publish(&mut p, 9);
                    let before = pic.snapshot();
                    let l1 = run(pic.dev, &p, thread);
                    assert_eq!(
                        word(&l1, 0),
                        value,
                        "read-back fence, thread {thread} {t:?}"
                    );
                    for (a, was) in before {
                        let now = pic.read(a);
                        if a == t.address() {
                            assert_eq!(now, value, "host sees {a:#x}");
                        } else {
                            assert_eq!(now, was, "{a:#x} must not change");
                        }
                    }
                }
            }
            pic.restore();
        });
    }

    /// Gate: loads return what the host staged, on every thread and target.
    #[test]
    fn load_returns_host_staged_values_on_all_threads() {
        harness::in_device(|dev| {
            let mut pic = Pic::open(dev);
            for thread in 0..3 {
                for (k, t) in Scratch::ALL.into_iter().enumerate() {
                    let value = 0xA000_0000 | (thread as u32) << 8 | k as u32;
                    pic.write(t.address(), value);
                    let mut p = vec![];
                    p.extend(backend::set_gpr(9, 0xFFFF_FFFF).unwrap());
                    p.push(mmio::load(9, t).unwrap());
                    p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
                    publish(&mut p, 9);
                    let l1 = run(pic.dev, &p, thread);
                    assert_eq!(word(&l1, 0), value, "thread {thread} {t:?}");
                }
            }
            pic.restore();
        });
    }

    /// Expected PIC words after `n` indirect stores with the given model shift
    /// (4 is the pinned model; other values are the mutants).
    fn indirect_expect(
        start: Scratch,
        offset: u16,
        inc: Inc,
        n: usize,
        shift: u32,
        base: &[(u32, u32)],
    ) -> Vec<(u32, u32)> {
        let plan = mmio::plan_indirect(start, offset).unwrap();
        let mut state = base.to_vec();
        let mut off = offset;
        for k in 0..n {
            let a = doc_address_shift(plan.address_gpr, off, shift);
            let v = 0xC0DE_0000 + k as u32;
            if let Some(e) = state.iter_mut().find(|e| e.0 == a) {
                e.1 = v;
            } else {
                state.push((a, v));
            }
            off = doc_increment(off, inc);
        }
        state
    }
    fn indirect_program(
        start: Scratch,
        offset: u16,
        inc: Inc,
        n: usize,
        half: u32,
    ) -> Vec<Instruction> {
        let plan = mmio::plan_indirect(start, offset).unwrap();
        let mut p = vec![];
        p.extend(backend::set_gpr(6, plan.address_gpr).unwrap());
        let reg = u32::from(offset) << (16 * (half % 2));
        p.extend(backend::set_gpr(half / 2, reg).unwrap());
        for k in 0..n {
            p.extend(backend::set_gpr(8 + k as u32, 0xC0DE_0000 + k as u32).unwrap());
            p.push(
                mmio::store_indirect(OffsetHalf::new(half).unwrap(), inc, 8 + k as u32, 6).unwrap(),
            );
        }
        p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
        // Fence: read a word back through the same path before the program ends.
        p.push(mmio::load(20, start).unwrap());
        p.push(backend::wait_for_scalar(Before::EVERYTHING).unwrap());
        p
    }

    /// Gate: both offset halves, all four increments, accumulation across a
    /// word boundary, every thread; the host compares against the transcription.
    #[test]
    fn storeind_mmio_offsets_increments_halves_and_threads() {
        harness::in_device(|dev| {
            let mut pic = Pic::open(dev);
            for thread in 0..3 {
                for half in [14, 15] {
                    for inc in [Inc::None, Inc::Bytes2, Inc::Bytes4, Inc::Bytes16] {
                        for (start, offset) in [
                            (Scratch::Irq28, 0u16),
                            (Scratch::Irq28, 0x40),
                            (Scratch::Irq29, 0x1F0),
                        ] {
                            for a in WATCHED {
                                if !is_guard(a) {
                                    pic.write(a, 0x5EED_0000 | (a & 0xFFFF));
                                }
                            }
                            let base = pic.snapshot();
                            let want = indirect_expect(start, offset, inc, 5, 4, &base);
                            let p = indirect_program(start, offset, inc, 5, half);
                            run(pic.dev, &p, thread);
                            for (a, v) in want {
                                assert!(WATCHED.contains(&a), "model left the allowlist {a:#x}");
                                assert_eq!(
                                    pic.read(a),
                                    v,
                                    "thread {thread} half {half} {inc:?} {start:?} off {offset:#x} addr {a:#x}"
                                );
                            }
                        }
                    }
                }
            }
            pic.restore();
        });
    }

    /// Negative control for the shift: a model with the L1-style unshifted or
    /// mis-shifted offset must disagree with what the hardware wrote.
    #[test]
    fn wrong_offset_shift_model_disagrees_with_hardware() {
        harness::in_device(|dev| {
            let mut pic = Pic::open(dev);
            for a in WATCHED {
                if !is_guard(a) {
                    pic.write(a, 0x5EED_0000);
                }
            }
            let base = pic.snapshot();
            let p = indirect_program(Scratch::Irq28, 0x40, Inc::Bytes16, 5, 14);
            run(pic.dev, &p, 0);
            let observed = pic.snapshot();
            pic.restore();
            let correct = indirect_expect(Scratch::Irq28, 0x40, Inc::Bytes16, 5, 4, &base);
            for (a, v) in &correct {
                assert_eq!(observed.iter().find(|o| o.0 == *a).unwrap().1, *v);
            }
            for shift in [0u32, 2, 3] {
                let wrong = indirect_expect(Scratch::Irq28, 0x40, Inc::Bytes16, 5, shift, &base);
                let mismatch = wrong
                    .iter()
                    .any(|(a, v)| observed.iter().find(|o| o.0 == *a).map(|o| o.1) != Some(*v));
                assert!(mismatch, "shift {shift} must be distinguishable");
            }
        });
    }
}
