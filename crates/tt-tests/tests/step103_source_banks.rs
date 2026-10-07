//! Physical two-bank oracle, clear/release consumers and isolated refusals.
use tt_isa::{
    isa::generated::encode,
    matrix::{Banks, ShiftBMode},
    tile::{L1Format, TileImage},
};
use tt_kernels::source_banks::{self, Clear, Selection};
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
            self.owned[operand][bank] = true;
            self.staged[operand][bank] = false;
            self.unpack[operand] ^= 1;
        }
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
    clear: Clear,
    selection: Selection,
    formats: [u32; 4],
    shift: Option<(ShiftBMode, u32)>,
    mutant: u8,
) -> Vec<u32> {
    let mut result = vec![];
    for capture_round in 0..4 {
        for capture_operand in [Selection::A, Selection::B] {
            for capture_block in 0..4 {
                let mut p = source_banks::diagnostic(
                    clear,
                    selection,
                    formats,
                    shift,
                    capture_round,
                    capture_operand,
                    capture_block,
                );
                if mutant != 0 {
                    let mut seed_nops = 0;
                    for i in p.math.iter_mut().chain(p.unpack.iter_mut()) {
                        if i.def().key() == "UNPACR_NOP_ZEROSRC_BH" && seed_nops < 4 {
                            seed_nops += 1;
                            continue;
                        }
                        if matches!(mutant, 10 | 11) && i.def().key() == "UNPACR_NOP_ZEROSRC_BH" {
                            let bit = if mutant == 10 { 16 } else { 8 };
                            *i = tt_isa::isa::Instruction::new(i.word() | bit, i.def());
                        }
                        if mutant == 1 && i.def().key() == "ZEROSRC" {
                            let word = i.word() ^ 3;
                            *i = tt_isa::isa::Instruction::new(word, i.def());
                        }
                        if i.def().key() == "SHIFTXB_BH" {
                            if mutant == 2 {
                                *i = encode::shiftxb(
                                    i.operand("AddrMod").unwrap(),
                                    i.operand("ShiftInZero").unwrap(),
                                    2,
                                )
                                .unwrap();
                            }
                            if mutant == 3 {
                                *i = encode::shiftxb(
                                    i.operand("AddrMod").unwrap(),
                                    i.operand("ShiftInZero").unwrap() ^ 1,
                                    i.operand("SrcRow").unwrap(),
                                )
                                .unwrap();
                            }
                        }
                    }
                }
                let bits = data();
                let image =
                    TileImage::new(tt_kernels::datapath::flat_descriptor(512), L1Format::Fp32)
                        .unwrap();
                let mut stage = vec![0; image.total_bytes()];
                for (i, bits) in bits.into_iter().enumerate() {
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
fn equal_bits(got: &[u32], want: &[u32], context: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g,
            w,
            "{context}: round {}, operand {}, row {}, lane {}: {g:08x} vs {w:08x}",
            i / 2048,
            (i / 1024) % 2,
            (i / 16) % 64,
            i % 16
        );
    }
}
#[test]
fn encodings_ranges_and_physical_ownership_model() {
    let (_, b) = Banks::after_reset()
        .unpack_b(encode::UnpacrRegular::ZERO)
        .unwrap();
    for row in [0, 63] {
        for modifier in 0..8 {
            for mode in [ShiftBMode::Rotate, ShiftBMode::ZeroFill] {
                let (_, b) = Banks::after_reset()
                    .unpack_b(encode::UnpacrRegular::ZERO)
                    .unwrap();
                let (i, _) = b.shiftxb(row, modifier, mode).unwrap();
                assert_eq!(
                    i.word() & 0xffffff,
                    (modifier << 14) | ((mode == ShiftBMode::ZeroFill) as u32 * (1 << 10)) | row
                );
            }
        }
    }
    assert!(b.shiftxb(64, 0, ShiftBMode::Rotate).is_err());
    let (_, b) = Banks::after_reset()
        .unpack_b(encode::UnpacrRegular::ZERO)
        .unwrap();
    assert!(b.shiftxb(0, 8, ShiftBMode::Rotate).is_err());
    let (i, b) = Banks::after_reset().zerosrc_unpacker_both().unwrap();
    assert_eq!(i.word(), 0x11000003);
    let (i, b) = b.unpacr_nop_zerosrc_a().unwrap();
    assert_eq!(i.operand("WaitLikeUnpacr"), Some(1));
    assert_eq!(i.operand("BothBanks"), Some(0));
    let (_, b) = b.unpack_a(encode::UnpacrRegular::ZERO).unwrap();
    let (i, _) = b.cleardvalid_a().unwrap();
    assert_eq!(i.operand("Reset"), Some(0));
    assert_eq!(i.operand("KeepReadingSameSrc"), Some(0));
    // Opposite-bank sentinel survives a current-unpacker clear while matrix
    // owns the preceding bank: this hardware state is outside lockstep API.
    let mut p = Physical::new();
    p.fill(1, &data(), 4, true);
    let held = p.data[1][0];
    p.fill(1, &data()[..32], 4, false);
    p.clear(1, false);
    assert_eq!(p.data[1][0], held);
    assert_eq!(p.data[1][1], [[0; 16]; 64]);
    assert!(p.owned[1][0]);
    assert!(!p.staged[1][1]);
    let a = expected(
        &data(),
        Clear::None,
        Selection::A,
        [4; 4],
        Some((ShiftBMode::Rotate, 4)),
    );
    let b = expected(
        &data(),
        Clear::None,
        Selection::A,
        [4; 4],
        Some((ShiftBMode::ZeroFill, 4)),
    );
    assert_ne!(a, b);
}
#[test]
#[cfg(feature = "silicon")]
fn zerosrc_unpacker_a_b_both_partial_staging_and_alternation() {
    harness::in_device(|dev| {
        for selection in [Selection::A, Selection::B, Selection::Both] {
            let formats = if cfg!(feature = "silicon") {
                [4, 5, 4, 5]
            } else {
                [4; 4]
            };
            let got = run(dev, Clear::UnpackerMatrix, selection, formats, None, 0);
            equal_bits(
                &got,
                &expected(&data(), Clear::UnpackerMatrix, selection, formats, None),
                &format!("UnpackerMatrix {selection:?}"),
            );
        }
    });
}
#[test]
fn zerosrc_matrix_a_b_both_and_cleardvalid_reuse() {
    harness::in_device(|dev| {
        for selection in [Selection::A, Selection::B, Selection::Both] {
            let formats = if cfg!(feature = "silicon") {
                [5, 4, 5, 4]
            } else {
                [4; 4]
            };
            equal_bits(
                &run(dev, Clear::Matrix, selection, formats, None, 0),
                &expected(&data(), Clear::Matrix, selection, formats, None),
                &format!("Matrix {selection:?}"),
            );
        }
    });
}
#[test]
fn unpacr_sequenced_zero_a_b_and_regular_handover() {
    harness::in_device(|dev| {
        for selection in [Selection::A, Selection::B, Selection::Both] {
            let formats = if cfg!(feature = "silicon") {
                [4, 5, 5, 4]
            } else {
                [4; 4]
            };
            equal_bits(
                &run(dev, Clear::UnpackerNop, selection, formats, None, 0),
                &expected(&data(), Clear::UnpackerNop, selection, formats, None),
                &format!("UnpackerNop {selection:?}"),
            );
        }
    });
}
#[test]
fn surviving_uncleared_control_and_wrong_selection_mutant() {
    harness::in_device(|dev| {
        let control = run(dev, Clear::None, Selection::A, [4; 4], None, 0);
        assert_eq!(
            control,
            expected(&data(), Clear::None, Selection::A, [4; 4], None)
        );
        let want = expected(&data(), Clear::Matrix, Selection::A, [4; 4], None);
        assert_ne!(run(dev, Clear::Matrix, Selection::A, [4; 4], None, 1), want);
        assert_ne!(control, want);
    });
}
#[test]
#[cfg(feature = "silicon")]
fn shift_modes_wrap_repetition_modifier4_and_safe_mutants() {
    harness::in_device(|dev| {
        for modifier in [1, 4] {
            for mode in [ShiftBMode::Rotate, ShiftBMode::ZeroFill] {
                for clear in [
                    Clear::None,
                    Clear::UnpackerMatrix,
                    Clear::UnpackerNop,
                    Clear::Matrix,
                ] {
                    let want = expected(
                        &data(),
                        clear,
                        Selection::A,
                        [4, 5, 4, 5],
                        Some((mode, modifier)),
                    );
                    equal_bits(
                        &run(
                            dev,
                            clear,
                            Selection::A,
                            [4, 5, 4, 5],
                            Some((mode, modifier)),
                            0,
                        ),
                        &want,
                        &format!("{clear:?} {mode:?} modifier {modifier}"),
                    );
                }
            }
        }
        let shift = Some((ShiftBMode::Rotate, 4));
        let want = expected(&data(), Clear::None, Selection::A, [4; 4], shift);
        for mutant in [2, 3] {
            assert_ne!(
                run(dev, Clear::None, Selection::A, [4; 4], shift, mutant),
                want
            );
        }
    });
}

#[test]
#[cfg(not(feature = "silicon"))]
fn simulator_refuses_each_unpacker_clear_form_with_surviving_controls() {
    for selection in [Selection::A, Selection::B, Selection::Both] {
        assert!(harness::survives(|dev| {
            run(dev, Clear::None, selection, [4; 4], None, 0);
        }));
        assert!(!harness::survives(|dev| {
            run(dev, Clear::UnpackerMatrix, selection, [4; 4], None, 0);
        }));
        assert!(harness::survives(|dev| {
            equal_bits(
                &run(dev, Clear::UnpackerNop, selection, [4; 4], None, 0),
                &expected(&data(), Clear::UnpackerNop, selection, [4; 4], None),
                "BH NOP surviving semantic control",
            );
        }));
    }
}

#[test]
fn released_sources_feed_existing_matmul_and_pooling_sequences() {
    use tt_isa::{
        backend::{self, Before},
        sync::{self, Unit},
    };
    use tt_kernels::{datapath as d, l1::Requirements, matmul};
    harness::in_device(|dev| {
        for clear in if cfg!(feature = "silicon") {
            vec![Clear::UnpackerMatrix, Clear::UnpackerNop, Clear::Matrix]
        } else {
            vec![Clear::Matrix]
        } {
            for pool in [false, true] {
                let mut p = source_banks::diagnostic(
                    clear,
                    Selection::Both,
                    [4, 5, 4, 5],
                    None,
                    3,
                    Selection::A,
                    0,
                );
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
                let a_values = vec![[2.0; 16]; 16];
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
                let (i, banks) = p
                    .banks
                    .unpack_a(encode::UnpacrRegular::ZERO.multi_context_mode(1))
                    .unwrap();
                p.unpack.push(i);
                let (i, mut banks) = banks
                    .unpack_b(encode::UnpacrRegular::ZERO.multi_context_mode(1))
                    .unwrap();
                p.unpack.push(i);
                p.unpack
                    .push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
                p.unpack.extend(sync::post_after(Unit::Unpacker1, ready));
                p.math.extend(sync::take(ready, Before::EVERYTHING));
                p.math.extend(matmul::math_prelude());
                // The diagnostic ended Empty/Empty and both pointers at zero.
                // Use the same existing consumer builders after regular UNPACR.
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
                        assert_eq!(
                            out.dst_at(row, col),
                            32.0f32.to_bits(),
                            "{clear:?} pool {pool} row {row} col {col}"
                        );
                    }
                }
            }
        }
    });
}

#[test]
#[cfg(feature = "silicon")]
fn unpacr_zero_blackhole_bank_and_clear_value_fields() {
    // Unlike ordinary lockstep transitions, these raw characterization cases
    // have both banks unpacker-owned, with all work retired before the NOP.
    // No active matrix reader can race the deliberately both-bank clear.
    harness::in_device(|dev| {
        for selection in [Selection::A, Selection::B] {
            let want = expected(&data(), Clear::UnpackerNop, selection, [4; 4], None);
            equal_bits(
                &run(dev, Clear::UnpackerNop, selection, [4; 4], None, 0),
                &want,
                "BH current-bank zero",
            );
            let both = run(dev, Clear::UnpackerNop, selection, [4; 4], None, 10);
            let operand = if selection == Selection::A { 0 } else { 1 };
            assert_eq!(
                &both[2048 + operand * 1024 + 16..2048 + operand * 1024 + 32],
                &[0; 16]
            );
            assert_ne!(both, want, "bit 4 clears the opposite bank as well");
            let one = run(dev, Clear::UnpackerNop, selection, [4; 4], None, 11);
            assert_eq!(
                &one[operand * 1024 + 16..operand * 1024 + 32],
                &[1.0f32.to_bits(); 16],
                "bit 3 is clear-value code 2, not BothBanks"
            );
            equal_bits(
                &one[2048..4096],
                &want[2048..4096],
                "nonzero current clear preserves opposite bank",
            );
        }
    });
}

#[test]
#[cfg(feature = "silicon")]
fn unpacr_zero_waits_on_current_unpacker_bank_with_matrix_bank_held() {
    use tt_isa::{
        backend::{self, Before, ConfigWords},
        cfg::generated::alu,
        sync::{self, Unit},
    };
    use tt_kernels::{datapath as d, l1::Requirements, matmul};
    harness::in_device(|dev| {
        for u in [d::Unpacker::SrcA, d::Unpacker::SrcB] {
            let mut req = Requirements::new(1);
            let input = req.scratch("two-bank wait probe", 16 + 512 * 4, 16, 0..1);
            let ready = req.semaphore("bank 1 clear retired while bank 0 held", 0, 0..1);
            let released = req.semaphore("matrix released bank 0", 0, 0..1);
            let plan = req.plan(tt_isa::l1::DATA).unwrap();
            let ready = plan.semaphore(ready);
            let released = plan.semaphore(released);
            let at = plan.addr(input);
            let mut cfg = ConfigWords::new();
            d::unpack_src_config(&mut cfg, u, d::flat_descriptor(512), at, 4);
            cfg.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
            let mut up = d::src_thread_config();
            up.extend(d::config_program(&cfg));
            up.push(
                backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY).unwrap(),
            );
            up.push(d::set_adc_x(u, 0, 511));
            up.push(d::unpack_src_instruction(u, true));
            up.push(if u == d::Unpacker::SrcA {
                backend::wait_for_unpacker0(Before::EVERYTHING).unwrap()
            } else {
                backend::wait_for_unpacker1(Before::EVERYTHING).unwrap()
            });
            // Matrix owns bank 0; unpacker owns bank 1. Stage then clear bank 1
            // while T1 waits for publication, so matrix bank 0 stays owned.
            up.push(d::unpack_src_instruction(u, false));
            up.push(encode::unpacr_nop_zerosrc(u as u32, 1, 0, 0).unwrap());
            up.extend(sync::post_after(
                if u == d::Unpacker::SrcA {
                    Unit::Unpacker0
                } else {
                    Unit::Unpacker1
                },
                ready,
            ));
            up.extend(sync::take(released, Before::EVERYTHING));
            up.push(d::set_adc_x(u, 0, 15));
            up.push(d::unpack_src_instruction(u, true));
            up.extend(sync::post_after(
                if u == d::Unpacker::SrcA {
                    Unit::Unpacker0
                } else {
                    Unit::Unpacker1
                },
                ready,
            ));
            up.extend(sync::take(released, Before::EVERYTHING));
            let mut math = matmul::math_prelude();
            math.extend(sync::take(ready, Before::EVERYTHING));
            for dst in [0, 8] {
                math.push(if u == d::Unpacker::SrcA {
                    encode::Mova2D::ZERO
                        .move8_rows(1)
                        .src_row(0)
                        .dst_row(dst)
                        .encode()
                        .unwrap()
                } else {
                    encode::Movb2D::ZERO
                        .move4_rows(1)
                        .src_row(0)
                        .dst_row(dst)
                        .encode()
                        .unwrap()
                });
                math.push(
                    encode::cleardvalid(
                        u32::from(u == d::Unpacker::SrcB),
                        u32::from(u == d::Unpacker::SrcA),
                        0,
                        0,
                    )
                    .unwrap(),
                );
                math.extend(sync::post_after(Unit::Matrix, released));
                if dst == 0 {
                    math.extend(sync::take(ready, Before::EVERYTHING));
                }
            }
            let image = TileImage::new(d::flat_descriptor(512), L1Format::Fp32).unwrap();
            let mut stage = vec![0; image.total_bytes()];
            for i in 0..512 {
                let off = image.datum_bit_offset(i) / 8;
                stage[off..off + 4].copy_from_slice(&((i as f32 + 1.0).to_bits()).to_le_bytes());
            }
            let init = plan.semaphore_init();
            let out = harness::run(
                dev,
                &Run::roles(Roles {
                    unpack: &up,
                    math: &math,
                    pack: &[],
                })
                .concurrent(&init)
                .stage(&[(at, &stage)])
                .dump_rows(16),
            );
            for row in 0..4 {
                for col in 0..16 {
                    assert_eq!(
                        out.dst_at(row, col),
                        ((row * 16 + col + 1) as f32).to_bits(),
                        "held bank 0"
                    );
                    assert_eq!(
                        out.dst_at(8 + row, col),
                        if row == 0 {
                            ((col + 1) as f32).to_bits()
                        } else {
                            0
                        },
                        "cleared bank 1"
                    );
                }
            }
        }
    });
}
