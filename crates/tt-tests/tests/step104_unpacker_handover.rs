//! Healthy-bank explicit publication; no recovery or opcode sweeps.
use tt_isa::{
    backend::{self, Before},
    isa::generated::encode,
    matrix::{Banks, ShiftBMode},
    tile::{L1Format, TileImage},
};
use tt_kernels::source_banks::{self, Clear, Handover, Selection};
use tt_tests::harness::{self, Roles, Run};

// Src has Sign(1), Mantissa(10), Exponent(8), unlike F32. Model the physical
// arrays independently of the checked instruction helpers and kernel builder.
struct Physical {
    data: [[[[u32; 16]; 64]; 2]; 2],
    unpack: [usize; 2],
    matrix: [usize; 2],
    owned: [[bool; 2]; 2],
    staged: [[bool; 2]; 2],
    rwc_b: usize,
    implied: [[u32; 2]; 2],
    row: [usize; 2],
}
impl Physical {
    fn new() -> Self {
        Self {
            data: [[[[0; 16]; 64]; 2]; 2],
            unpack: [0; 2],
            matrix: [0; 2],
            owned: [[false; 2]; 2],
            staged: [[false; 2]; 2],
            rwc_b: 0,
            implied: [[0; 2]; 2],
            row: [0; 2],
        }
    }
    fn fill(&mut self, operand: usize, bits: &[u32], format: u32, flip: bool) {
        let bank = self.unpack[operand];
        assert!(!self.owned[operand][bank]);
        for (i, &bits) in bits.iter().enumerate() {
            let mantissa = (bits >> 13) & if format == 4 { 1023 } else { 1016 };
            self.data[operand][bank][i / 16][i % 16] =
                ((bits >> 31) << 18) | (mantissa << 8) | ((bits >> 23) & 255);
        }
        self.staged[operand][bank] = true;
        if flip {
            self.handover(operand, format, 0);
        }
    }
    // Pinned UNPACR_NOP_SETDVALID functional pseudocode, independent of
    // encodings: publish format, transfer ownership, flip unpack pointer,
    // reset this thread's source row to the configured base (in 16-row units).
    fn handover(&mut self, operand: usize, format: u32, base: usize) {
        let bank = self.unpack[operand];
        assert!(!self.owned[operand][bank]);
        assert!(self.staged[operand][bank]);
        self.owned[operand][bank] = true;
        self.staged[operand][bank] = false;
        self.implied[operand][bank] = format;
        self.unpack[operand] ^= 1;
        self.row[operand] = base << 4;
    }
    fn clear(&mut self, operand: usize, matrix: bool) {
        let bank = if matrix {
            self.matrix[operand]
        } else {
            self.unpack[operand]
        };
        assert_eq!(self.owned[operand][bank], matrix);
        self.data[operand][bank] = [[0; 16]; 64];
        if !matrix {
            self.staged[operand][bank] = false;
        }
    }
    fn shift(&mut self, immediate: usize, increment: usize, mode: ShiftBMode) {
        let bank = self.matrix[1];
        assert!(self.owned[1][bank]);
        let row = &mut self.data[1][bank][(immediate + self.rwc_b) & 63];
        let first = row[0];
        row.copy_within(1..16, 0);
        row[15] = if mode == ShiftBMode::Rotate { first } else { 0 };
        self.rwc_b = (self.rwc_b + increment) & 63;
    }
    fn read(&self, operand: usize, row: usize) -> Vec<u32> {
        let bank = self.matrix[operand];
        assert!(self.owned[operand][bank]);
        self.data[operand][bank][row]
            .iter()
            .map(|&v| {
                if v & 255 == 0 {
                    0
                } else {
                    ((v >> 18) << 31) | ((v & 255) << 23) | (((v >> 8) & 1023) << 13)
                }
            })
            .collect()
    }
    fn release(&mut self, operand: usize) {
        let bank = self.matrix[operand];
        assert!(self.owned[operand][bank]);
        self.owned[operand][bank] = false;
        self.matrix[operand] ^= 1;
        assert_eq!(self.matrix[operand], self.unpack[operand]);
    }
}
fn data() -> Vec<u32> {
    let mut bits: Vec<_> = (0..512).map(|i| (i as f32 + 1.0).to_bits()).collect();
    bits[..16].copy_from_slice(&[
        0, 0x80000000, 1, 0x80010000, 0x7f800000, 0xff800000, 0x7fc12000, 0xffc12000, 0x7f7fffff,
        0xff7fffff, 0x00800000, 0x80800000, 0x3f801fff, 0xbf801fff, 0x3fffffff, 0xbfffffff,
    ]);
    bits
}
fn expected(
    bits: &[u32],
    clear: Clear,
    selection: Selection,
    formats: [u32; 4],
    shift: Option<(ShiftBMode, u32)>,
) -> Vec<u32> {
    let mut p = Physical::new();
    let mut out = vec![];
    for _ in 0..2 {
        for op in 0..2 {
            p.fill(op, bits, 4, true);
            p.release(op);
        }
    }
    for (round, format) in formats.into_iter().enumerate() {
        let clear = if matches!(round, 0 | 3) {
            clear
        } else {
            Clear::None
        };
        let bits_staged = if matches!(round, 1 | 2) {
            &bits[..16]
        } else {
            bits
        };
        for op in 0..2 {
            p.fill(op, bits_staged, format, false);
        }
        let selected = match selection {
            Selection::A => vec![0],
            Selection::B => vec![1],
            Selection::Both => vec![0, 1],
        };
        if matches!(clear, Clear::UnpackerMatrix | Clear::UnpackerNop) {
            for &op in &selected {
                p.clear(op, false);
            }
        }
        for op in 0..2 {
            p.fill(op, &bits[..16], format, true);
        }
        if clear == Clear::Matrix {
            for &op in &selected {
                p.clear(op, true);
            }
        }
        if let Some((mode, _)) = shift {
            p.rwc_b = 1;
            for _ in 0..2 {
                p.shift(63, 1, mode);
            }
            p.rwc_b = 0;
            for _ in 0..2 {
                p.shift(0, 0, mode);
            }
        }
        for op in 0..2 {
            for row in 0..64 {
                out.extend(p.read(op, row));
            }
        }
        for op in 0..2 {
            p.release(op);
        }
    }
    assert!(p.owned.iter().flatten().all(|&v| !v));
    assert_eq!(p.unpack, [0, 0]);
    assert_eq!(p.matrix, [0, 0]);
    out
}
fn run(
    dev: &mut harness::Dev<'_>,
    handover: Handover,
    formats: [u32; 4],
    multipart: bool,
    changed: bool,
) -> Vec<u32> {
    let mut result = vec![];
    for round in 0..4 {
        for operand in [Selection::A, Selection::B] {
            for block in 0..4 {
                let mut p = source_banks::handover_diagnostic(
                    handover, formats, multipart, round, operand, block,
                );
                if !cfg!(feature = "silicon") {
                    // Known simulator refusal (row 36), independent of NOP
                    // semantics. Simulator thread configuration starts at zero.
                    p.unpack.retain(|i| {
                        !(i.def().key() == "SETC16" && i.operand("CfgIndex") == Some(6))
                    });
                }
                let image =
                    TileImage::new(tt_kernels::datapath::flat_descriptor(512), L1Format::Fp32)
                        .unwrap();
                let mut stage = vec![0; image.total_bytes()];
                for (i, bits) in input(changed).into_iter().enumerate() {
                    let offset = image.datum_bit_offset(i) / 8;
                    stage[offset..offset + 4].copy_from_slice(&bits.to_le_bytes());
                }
                let out = harness::run(
                    dev,
                    &Run::roles(Roles {
                        unpack: &p.unpack,
                        math: &p.math,
                        pack: &[],
                    })
                    .concurrent(&p.init)
                    .stage(&[(p.input, &stage)])
                    .dump_rows(16),
                );
                result.extend((0..256).map(|i| out.dst_at(i / 16, i % 16)));
            }
        }
    }
    result
}
fn input(changed: bool) -> Vec<u32> {
    let mut bits = data();
    if changed {
        for b in &mut bits[16..] {
            *b ^= 1 << 31;
        }
    }
    bits
}
fn want(formats: [u32; 4], changed: bool) -> Vec<u32> {
    expected(&input(changed), Clear::None, Selection::Both, formats, None)
}
#[test]
fn handover_encoding_wait_order_and_independent_state_model() {
    for operand in 0..2 {
        let (seq, _) = if operand == 0 {
            let (_, b) = Banks::after_reset()
                .unpack_a_partial(encode::UnpacrRegular::ZERO)
                .unwrap();
            let (seq, _) = b.handover_a().unwrap();
            (seq, ())
        } else {
            let (_, b) = Banks::after_reset()
                .unpack_b_partial(encode::UnpacrRegular::ZERO)
                .unwrap();
            let (seq, _) = b.handover_b().unwrap();
            (seq, ())
        };
        assert_eq!(
            seq[0],
            backend::stallwait(Before::EVERYTHING.mask(), 1 << (operand + 1)).unwrap()
        );
        assert_eq!(
            seq[1],
            backend::stallwait(Before::EVERYTHING.mask(), 1 << (operand + 5)).unwrap()
        );
        assert_eq!(seq[2], encode::unpacr_nop_setdvalid(operand).unwrap());
        assert_eq!(seq[2].operand("WhichUnpacker"), Some(operand));
        assert_eq!(seq[2].word(), 0x430001e9 | (operand << 23));
        assert_eq!((seq[2].word() >> 8) & 15, 1);
        assert_eq!((seq[2].word() >> 6) & 3, 3);
        assert_eq!((seq[2].word() >> 5) & 1, 1);
        assert_eq!((seq[2].word() >> 2) & 3, 2);
        assert_eq!(seq[2].word() & 3, 1);
        // Preserve the WH encoder for host characterization only. It selects
        // stream-pop on BH and must never be used by the checked handover.
        let wh = encode::wormhole::unpacr_nop_setdvalid(operand).unwrap();
        assert_eq!(wh.word(), 0x43000007 | (operand << 23));
        assert_ne!(seq[2].word(), wh.word());
        let mut p = Physical::new();
        for format in [4, 5, 5, 4] {
            p.fill(operand as usize, &data(), format, false);
            p.row[operand as usize] = 48;
            p.handover(operand as usize, format, 1);
            assert_eq!(p.row[operand as usize], 16);
            assert_eq!(
                p.implied[operand as usize][p.matrix[operand as usize]],
                format
            );
            p.release(operand as usize);
        }
        assert_eq!(p.unpack, [0, 0]);
        assert_eq!(p.matrix, [0, 0]);
    }
    assert!(encode::unpacr_nop_setdvalid(2).is_err());
    // Wrong format and wrong row mapping are meaningful oracle mutants.
    assert_ne!(want([4; 4], false), want([5; 4], false));
    let w = want([4; 4], false);
    let mut wrong_row = w.clone();
    wrong_row.rotate_left(16);
    assert_ne!(wrong_row, w);
}
#[test]
fn regular_unpacr_control() {
    harness::in_device(|dev| {
        assert_eq!(
            run(dev, Handover::Regular, [4; 4], false, false),
            want([4; 4], false)
        );
    });
}
#[cfg(feature = "silicon")]
#[test]
fn explicit_a_healthy_bank() {
    harness::in_device(|dev| {
        assert_eq!(
            run(dev, Handover::Explicit(Selection::A), [4; 4], false, false),
            want([4; 4], false)
        );
    });
}
#[cfg(feature = "silicon")]
#[test]
fn explicit_b_healthy_bank() {
    harness::in_device(|dev| {
        assert_eq!(
            run(dev, Handover::Explicit(Selection::B), [4; 4], false, false),
            want([4; 4], false)
        );
    });
}
#[cfg(feature = "silicon")]
#[test]
fn explicit_both_formats_multipart_replay_and_data_mutant() {
    harness::in_device(|dev| {
        for formats in if cfg!(feature = "silicon") {
            vec![[4, 5, 4, 5], [5, 4, 5, 4]]
        } else {
            vec![[4; 4]]
        } {
            for multipart in [false, true] {
                let control = run(dev, Handover::Regular, formats, multipart, false);
                assert_eq!(control, want(formats, false));
                let explicit = run(
                    dev,
                    Handover::Explicit(Selection::Both),
                    formats,
                    multipart,
                    false,
                );
                assert_eq!(explicit, control);
                let changed = run(
                    dev,
                    Handover::Explicit(Selection::Both),
                    formats,
                    multipart,
                    true,
                );
                assert_eq!(changed, want(formats, true));
                assert_ne!(
                    changed, explicit,
                    "changed staged data must fail the original expectation"
                );
            }
        }
    });
}

// Bisect setup independently of bank publication and of the concurrent loop.
// Kept isolated so the runner can stop after any failing boundary.
#[test]
fn bisect_configuration_only() {
    harness::in_device(|dev| {
        let mut p =
            source_banks::handover_diagnostic(Handover::Regular, [4; 4], false, 0, Selection::A, 0);
        let end = p
            .unpack
            .iter()
            .position(|i| i.def().key() == "UNPACR_NOP_ZEROSRC_BH")
            .unwrap();
        p.unpack.truncate(end);
        if !cfg!(feature = "silicon") {
            p.unpack
                .retain(|i| !(i.def().key() == "SETC16" && i.operand("CfgIndex") == Some(6)));
        }
        harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &p.unpack,
                math: &[],
                pack: &[],
            })
            .dump_rows(0),
        );
    });
}

fn bisect_one_bank(handover: Handover) {
    bisect_one_bank_with_waits(handover, None, None);
}

// Independent raw research inputs from official Blackhole LLK revision
// 201312fe7960b3711a420e2d655d420a39b0230e. Mode 0x102 is the failed delay-only
// hypothesis. Mode 0x1e9 is measured non-clearing DVALID publication.
// All stream controls are zero. Do not use the Wormhole skeleton.
fn llk_candidate(
    mut seq: [tt_isa::isa::Instruction; 3],
    mode: u32,
) -> [tt_isa::isa::Instruction; 3] {
    let which = seq[2].operand("WhichUnpacker").unwrap();
    let word = 0x43000000 | mode | (which << 23);
    eprintln!("Blackhole LLK candidate word={word:08x}");
    seq[2] = tt_isa::isa::Instruction::new(word, seq[2].def());
    seq
}

fn bisect_one_bank_with_waits(handover: Handover, wait_only: Option<Selection>, llk: Option<u32>) {
    use tt_isa::{
        backend::ConfigWords,
        cfg::generated::{alu, thread},
    };
    use tt_kernels::{
        datapath::{self, Unpacker},
        l1::Requirements,
    };
    let mut req = Requirements::new(1);
    let image = TileImage::new(datapath::flat_descriptor(512), L1Format::Fp32).unwrap();
    let buffer = req.scratch("bisect source", image.total_bytes() as u64, 16, 0..1);
    let plan = req.plan(tt_isa::l1::DATA).unwrap();
    let input = plan.addr(buffer);
    let mut staged = vec![0; image.total_bytes()];
    for i in 0..16 {
        let offset = image.datum_bit_offset(i) / 8;
        staged[offset..offset + 4].copy_from_slice(&((i + 1) as f32).to_le_bytes());
    }
    let mut up = datapath::src_thread_config();
    if cfg!(feature = "silicon") {
        up.push(datapath::thread_entry(thread::SRCB_SET_Base, 0));
    }
    let mut cfg = ConfigWords::new();
    for operand in [Unpacker::SrcA, Unpacker::SrcB] {
        datapath::unpack_src_config(&mut cfg, operand, datapath::flat_descriptor(512), input, 4);
        up.push(datapath::set_adc_x(operand, 0, 15));
    }
    cfg.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    up.extend(datapath::config_program(&cfg));
    up.push(backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY).unwrap());
    let regular = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    let (i, b) = Banks::after_reset().unpack_a_partial(regular).unwrap();
    up.push(i);
    let (i, b) = b.unpack_b_partial(regular).unwrap();
    up.push(i);
    // The only change from the surviving regular control: the two waits from
    // the checked handover helper, with regular FlipSrc still publishing data.
    // This separates C1/C5 and C2/C6 from the unverified NOP word.
    if let Some(selected) = wait_only {
        let (busy, owner) = match selected {
            Selection::A => (
                backend::cond::UNPACKER0_BUSY,
                backend::cond::SRCA_NOT_UNPACKER,
            ),
            Selection::B => (
                backend::cond::UNPACKER1_BUSY,
                backend::cond::SRCB_NOT_UNPACKER,
            ),
            Selection::Both => unreachable!("one operand per bisect boundary"),
        };
        up.push(backend::stallwait(Before::EVERYTHING.mask(), busy).unwrap());
        up.push(backend::stallwait(Before::EVERYTHING.mask(), owner).unwrap());
    }
    let b = if matches!(handover, Handover::Explicit(Selection::A | Selection::Both)) {
        let (seq, b) = b.handover_a().unwrap();
        up.extend(if let Some(mode) = llk {
            llk_candidate(seq, mode)
        } else {
            seq
        });
        b
    } else {
        let (i, b) = b.unpack_a(regular).unwrap();
        up.push(i);
        b
    };
    let mut b = if matches!(handover, Handover::Explicit(Selection::B | Selection::Both)) {
        let (seq, b) = b.handover_b().unwrap();
        up.extend(if let Some(mode) = llk {
            llk_candidate(seq, mode)
        } else {
            seq
        });
        b
    } else {
        let (i, b) = b.unpack_b(regular).unwrap();
        up.push(i);
        b
    };
    up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    up.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());
    let mut math = tt_kernels::matmul::math_prelude();
    let (i, next) = b.mova2d(encode::Mova2D::ZERO.move8_rows(1)).unwrap();
    math.push(i);
    b = next;
    let (i, b) = b.movb2d(encode::Movb2D::ZERO.dst_row(8)).unwrap();
    math.push(i);
    let (i, _empty) = b.cleardvalid_both().unwrap();
    math.push(i);
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &up,
                math: &math,
                pack: &[],
            })
            .stage(&[(input, &staged)])
            .dump_rows(9),
        );
        for row in [0, 8] {
            for col in 0..16 {
                assert_eq!(out.dst_at(row, col), ((col + 1) as f32).to_bits());
            }
        }
    });
}
#[test]
fn bisect_single_bank_regular_control() {
    bisect_one_bank(Handover::Regular);
}
#[cfg(feature = "silicon")]
#[test]
fn bisect_single_bank_waits_a_regular_publication() {
    bisect_one_bank_with_waits(Handover::Regular, Some(Selection::A), None);
}
#[cfg(feature = "silicon")]
#[test]
fn bisect_single_bank_waits_b_regular_publication() {
    bisect_one_bank_with_waits(Handover::Regular, Some(Selection::B), None);
}
#[test]
#[cfg(feature = "silicon")]
#[ignore = "mode 2 is delay-only; matrix consumer timed out on card 0"]
fn llk_single_bank_explicit_a() {
    bisect_one_bank_with_waits(Handover::Explicit(Selection::A), None, Some(0x102));
}
#[cfg(feature = "silicon")]
#[test]
fn recover_gate_with_existing_session_and_validate_matmul() {
    tt_ttsim::fork_scope(|| {
        let (x, y) = tt_tests::backend::GATE_TILE;
        let mut session = tt_kernels::session::Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            tt_kernels::session::TileChoice::Exactly(x, y),
        )
        .unwrap();
        let values = vec![1.0; 32 * 32];
        let out = session
            .matmul(
                &values,
                &values,
                [32, 32, 32],
                tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                tt_kernels::matmul::Fidelity::HiFi4,
                harness::BUDGET,
            )
            .unwrap();
        assert!(out.iter().all(|v| v.to_bits() == 32.0f32.to_bits()));
    })
    .unwrap();
}
#[cfg(not(feature = "silicon"))]
#[test]
fn simulator_refuses_llk_nop2_with_regular_control() {
    bisect_one_bank(Handover::Regular);
    assert!(std::panic::catch_unwind(|| {
        // The helper's fork catches the simulator's process-wide _Exit. No
        // Simulator or parent lock is held across that fork.
        bisect_one_bank_with_waits(Handover::Explicit(Selection::A), None, Some(0x102));
    })
    .is_err());
}
#[cfg(feature = "silicon")]
#[test]
#[ignore = "mode 2 is delay-only; A matrix consumer timed out on card 0"]
fn llk_single_bank_explicit_b() {
    bisect_one_bank_with_waits(Handover::Explicit(Selection::B), None, Some(0x102));
}
#[cfg(feature = "silicon")]
#[test]
fn bisect_single_bank_explicit_a() {
    bisect_one_bank(Handover::Explicit(Selection::A));
}
#[cfg(feature = "silicon")]
#[test]
fn bisect_single_bank_explicit_b() {
    bisect_one_bank(Handover::Explicit(Selection::B));
}

#[cfg(not(feature = "silicon"))]
#[test]
fn simulator_refuses_explicit_forms_with_surviving_regular_control() {
    // Each probe has a fresh fork and simulator. Never continue inside a
    // simulator process after a refusal.
    bisect_one_bank(Handover::Regular);
    for selected in [Selection::A, Selection::B] {
        assert!(!harness::survives(|dev| {
            run(dev, Handover::Explicit(selected), [4; 4], false, false);
        }));
    }
}

#[test]
fn handover_release_feeds_matmul_pooling_and_changed_input_replay() {
    use tt_isa::{
        backend::{self, Before},
        sync::{self, Unit},
    };
    use tt_kernels::{datapath as d, l1::Requirements, matmul};
    harness::in_device(|dev| {
        for (handover, value) in if cfg!(feature = "silicon") {
            vec![
                (Handover::Regular, 2.0078125),
                (Handover::Explicit(Selection::Both), 2.0078125),
                (Handover::Explicit(Selection::Both), 4.015625),
            ]
        } else {
            vec![
                (Handover::Regular, 2.0078125),
                (Handover::Regular, 4.015625),
            ]
        } {
            for pool in [false, true] {
                let mut p = source_banks::handover_diagnostic(
                    handover,
                    if cfg!(feature = "silicon") {
                        [4, 5, 4, 5]
                    } else {
                        [4; 4]
                    },
                    true,
                    3,
                    Selection::A,
                    0,
                );
                if !cfg!(feature = "silicon") {
                    p.unpack.retain(|i| {
                        !(i.def().key() == "SETC16" && i.operand("CfgIndex") == Some(6))
                    });
                }
                // Reuse a consumed semaphore only after the final release has
                // retired. No reset or new bank-state assertion between chains.
                let ready = p.init[0].0;
                let mut req = Requirements::new(1);
                let b = req.scratch("downstream Src inputs", 2 * 2048 + 32, 16, 0..1);
                let plan = req
                    .plan(tt_isa::l1::Region {
                        name: "downstream source arena",
                        base: p.input + 4096,
                        end: p.input + 16384,
                    })
                    .unwrap();
                let at = plan.addr(b);
                let a_values = vec![[value; 16]; 16];
                let b_values = vec![[1.0; 16]; if pool { 4 } else { 8 }];
                let (a, na) = matmul::stage_operand(0, &a_values);
                let (b, nb) = matmul::stage_operand(0, &b_values);
                p.unpack.extend(matmul::unpack_prelude(matmul::Operands {
                    a_addr: at,
                    na,
                    b_addr: at + 2048,
                    nb,
                    out: if pool { 5 } else { 4 },
                }));
                p.unpack.push(
                    backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY)
                        .unwrap(),
                );
                p.unpack.push(d::set_adc_x(d::Unpacker::SrcA, 0, na - 1));
                p.unpack.push(d::set_adc_x(d::Unpacker::SrcB, 0, nb - 1));
                let unpack = encode::UnpacrRegular::ZERO.multi_context_mode(1);
                let mut banks = if matches!(handover, Handover::Explicit(_)) {
                    let (i, banks) = p.banks.unpack_a_partial(unpack).unwrap();
                    p.unpack.push(i);
                    let (i, banks) = banks.unpack_b_partial(unpack).unwrap();
                    p.unpack.push(i);
                    let (seq, banks) = banks.handover_a().unwrap();
                    p.unpack.extend(seq);
                    let (seq, banks) = banks.handover_b().unwrap();
                    p.unpack.extend(seq);
                    banks
                } else {
                    let (i, banks) = p.banks.unpack_a(unpack).unwrap();
                    p.unpack.push(i);
                    let (i, banks) = banks.unpack_b(unpack).unwrap();
                    p.unpack.push(i);
                    banks
                };
                p.unpack
                    .push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
                p.unpack.extend(sync::post_after(Unit::Unpacker1, ready));
                p.math.extend(sync::take(ready, Before::EVERYTHING));
                p.math.extend(matmul::math_prelude());
                // The diagnostic ended Empty/Empty and both pointers at zero.
                // Consume the measured explicit handover directly, with
                // regular FlipSrc supplying the matching control.
                for _ in 0..4 {
                    let (i, next) = if pool {
                        banks.gapool(matmul::MATH_AM_PHASE, 0).unwrap()
                    } else {
                        banks
                            .mvmul(encode::Mvmul::ZERO.addr_mod(matmul::MATH_AM_PHASE))
                            .unwrap()
                    };
                    p.math.push(i);
                    banks = next;
                }
                p.math
                    .push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                let (i, banks) = banks.release_a().unwrap();
                p.math.push(i);
                let (i, _) = banks.release_b().unwrap();
                p.math.push(i);
                p.math
                    .push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                let image = TileImage::new(d::flat_descriptor(512), L1Format::Fp32).unwrap();
                let mut stage = vec![0; image.total_bytes()];
                for (i, bits) in data().into_iter().enumerate() {
                    let off = image.datum_bit_offset(i) / 8;
                    stage[off..off + 4].copy_from_slice(&bits.to_le_bytes());
                }
                let out = harness::run(
                    dev,
                    &Run::roles(Roles {
                        unpack: &p.unpack,
                        math: &p.math,
                        pack: &[],
                    })
                    .concurrent(&p.init)
                    .stage(&[(p.input, &stage), (at, &a), (at + 2048, &b)])
                    .dump_rows(8),
                );
                for row in 0..if pool { 1 } else { 8 } {
                    for col in 0..16 {
                        // TF32 preserves this low mantissa bit; BF16 drops it.
                        // Multiplication by one and sixteen identical terms
                        // are exact binary operations for these bounded inputs.
                        let converted = if pool {
                            f32::from_bits(value.to_bits() & 0xffff0000)
                        } else {
                            value
                        };
                        assert_eq!(
                            out.dst_at(row, col),
                            (16.0f32 * converted).to_bits(),
                            "{handover:?} value {value} pool {pool} row {row} col {col}"
                        );
                    }
                }
            }
        }
    });
}

// Official Blackhole assembly.yaml describes Clr_to1_fmt_Ctrl=3 as
// DVALID-only with no other side effects in the clear-to-one format control.
// Select clear-to-one (Src_ClrVal_Ctrl=2), its repurposed format selector 3,
// UNP_CLR_SRC=1, Set_Dvalid=1 and WAIT_LIKE_UNPACR=1. Stream controls are zero.
#[test]
#[cfg(feature = "silicon")]
fn llk_nonclearing_dvalid_a() {
    bisect_one_bank_with_waits(Handover::Explicit(Selection::A), None, Some(0x1e9));
}
#[cfg(not(feature = "silicon"))]
#[test]
fn simulator_refuses_nonclearing_dvalid_with_regular_control() {
    bisect_one_bank(Handover::Regular);
    assert!(std::panic::catch_unwind(|| {
        bisect_one_bank_with_waits(Handover::Explicit(Selection::A), None, Some(0x1e9));
    })
    .is_err());
}
#[cfg(feature = "silicon")]
#[test]
fn llk_nonclearing_dvalid_b() {
    bisect_one_bank_with_waits(Handover::Explicit(Selection::B), None, Some(0x1e9));
}

#[cfg(feature = "silicon")]
#[test]
fn explicit_b_resets_source_row_to_nonzero_base() {
    use tt_isa::{
        backend::ConfigWords,
        cfg::generated::{alu, thcon, thread},
        sync::{self, Unit},
    };
    use tt_kernels::{datapath as d, l1::Requirements};
    let mut req = Requirements::new(1);
    let image = TileImage::new(d::flat_descriptor(512), L1Format::Fp32).unwrap();
    let buf = req.scratch("row reset inputs", 2 * image.total_bytes() as u64, 16, 0..1);
    let ready = req.semaphore("row reset loaded", 0, 0..1);
    let released = req.semaphore("row reset released", 0, 0..1);
    let plan = req.plan(tt_isa::l1::DATA).unwrap();
    let input = plan.addr(buf);
    let ready = plan.semaphore(ready);
    let released = plan.semaphore(released);
    let mut stage = vec![0; 2 * image.total_bytes()];
    for n in 0..2 {
        for col in 0..16 {
            let value = (col + 1) as f32 * if n == 0 { 1.0 } else { -1.0 };
            let off = n * image.total_bytes() + image.datum_bit_offset(col) / 8;
            stage[off..off + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    let mut up = d::src_thread_config();
    up.push(d::thread_entry(thread::SRCB_SET_Base, 0));
    let mut math = tt_kernels::matmul::math_prelude();
    let mut banks = Banks::after_reset();
    let unpack = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    for round in 0..4 {
        if round != 0 {
            up.extend(sync::take(released, Before::EVERYTHING));
        }
        if round == 2 {
            up.push(d::thread_entry(thread::SRCB_SET_Base, 1));
        }
        let mut cfg = ConfigWords::new();
        d::unpack_src_config(
            &mut cfg,
            d::Unpacker::SrcB,
            d::flat_descriptor(512),
            input
                + if round == 3 {
                    image.total_bytes() as u64
                } else {
                    0
                },
            if round == 3 { 5 } else { 4 },
        );
        cfg.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
        cfg.set(
            thcon::THCON_SEC1_REG2_Unpack_Src_Reg_Set_Upd,
            u32::from(round >= 2),
        )
        .unwrap();
        up.extend(d::config_program(&cfg));
        up.push(backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY).unwrap());
        up.push(d::set_adc_x(d::Unpacker::SrcB, 0, 15));
        let (i, b) = banks.unpacr_nop_zerosrc_b().unwrap();
        up.push(i);
        let loaded = if round < 2 {
            // Establish both banks and source-row zero with regular FlipSrc.
            let (i, b) = b.unpack_b(unpack).unwrap();
            up.push(i);
            b
        } else {
            // Round 2 advances SrcRow from 0 to 32. Handover must reset
            // it to Base<<4 = 16 before round 3 writes the other bank.
            let (i, b) = b.unpack_b_partial(unpack).unwrap();
            up.push(i);
            let (seq, b) = b.handover_b().unwrap();
            up.extend(seq);
            b
        };
        up.extend(sync::post_after(Unit::Unpacker1, ready));
        math.extend(sync::take(ready, Before::EVERYTHING));
        let mut loaded = loaded;
        if round >= 2 {
            let (i, b) = loaded
                .movb2d(
                    encode::Movb2D::ZERO
                        .src_row(if round == 2 { 0 } else { 16 })
                        .dst_row(round - 2),
                )
                .unwrap();
            math.push(i);
            loaded = b;
        }
        let (i, b) = loaded.release_b().unwrap();
        math.push(i);
        banks = b;
        math.extend(sync::post_after(Unit::Matrix, released));
    }
    up.extend(sync::take(released, Before::EVERYTHING));
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &up,
                math: &math,
                pack: &[],
            })
            .concurrent(&plan.semaphore_init())
            .stage(&[(input, &stage)])
            .dump_rows(2),
        );
        for row in 0..2 {
            for col in 0..16 {
                let value = (col + 1) as f32 * if row == 0 { 1.0 } else { -1.0 };
                assert_eq!(out.dst_at(row, col), value.to_bits(), "row {row} col {col}");
            }
        }
    });
}
