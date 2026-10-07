//! Bounded source-bank diagnostics. No tensor dispatch or recovery changes.
//! T0 retires staged reads before T1 clears unpacker-owned banks; declared
//! semaphores serialize clear, handover, readback and explicit release.
use crate::{
    datapath::{self, Unpacker},
    l1::Requirements,
    runtime::SemaphoreInit,
};
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::{alu, thread},
    isa::{generated::encode, Instruction},
    matrix::{Banks, Empty, ShiftBMode},
    sync::{self, Unit},
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Selection {
    A,
    B,
    Both,
}
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Clear {
    None,
    UnpackerMatrix,
    UnpackerNop,
    Matrix,
}

/// Experimental publication of selected operands; others use final FlipSrc.
/// The candidate NOP encoding awaits isolated Blackhole silicon validation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Handover {
    Regular,
    Explicit(Selection),
}

struct Options {
    shift: Option<(ShiftBMode, u32)>,
    handover: Handover,
    multipart: bool,
    configure_b_base: bool,
}

pub struct Diagnostic {
    pub unpack: Vec<Instruction>,
    pub math: Vec<Instruction>,
    pub init: Vec<SemaphoreInit>,
    pub input: u64,
    pub banks: Banks<Empty, Empty>,
}

/// Seed both banks with TF32 sentinels, then run four alternating-bank rounds
/// with explicit TF32/BF16 setup. Rounds 0 and 3 stage 512 datums and clear;
/// rounds 1 and 2 refill only row 0, observing the untouched opposite bank and
/// the previously cleared bank. Read one aligned 16-row `capture_block` of
/// `capture_operand` in `capture_round`. A uses two eight-row moves, B sixteen
/// single-row moves. Repeating captures observes every physical row and round.
/// `shift` applies two shifts at (63+RWC.SrcB)&63 with modifier 1 or 4
/// advancing SrcB by one, then two repeated shifts of row 0 with modifier 0.
/// All formats and modifier state are initialized even across consecutive runs.
pub fn diagnostic(
    clear: Clear,
    selection: Selection,
    formats: [u32; 4],
    shift: Option<(ShiftBMode, u32)>,
    capture_round: usize,
    capture_operand: Selection,
    capture_block: u32,
) -> Diagnostic {
    diagnostic_impl(
        clear,
        selection,
        formats,
        Options {
            shift,
            handover: Handover::Regular,
            multipart: false,
            configure_b_base: false,
        },
        capture_round,
        capture_operand,
        capture_block,
    )
}

/// Identical staged contents for final regular FlipSrc and explicit handover.
/// No clears are permitted: explicit publication requires a Filling claim.
/// `multipart` re-stages the same range before the final row, testing retirement
/// of multiple regular UNPACRs. It does not concatenate disjoint ranges.
/// Both source bases are established
/// at entry; ADC state and output formats are initialized each round. All
/// operands are released before returning.
/// Explicit mode uses the measured Blackhole non-clearing profile. Output
/// format remains a builder invariant; ownership types cannot inspect it.
pub fn handover_diagnostic(
    handover: Handover,
    formats: [u32; 4],
    multipart: bool,
    capture_round: usize,
    capture_operand: Selection,
    capture_block: u32,
) -> Diagnostic {
    diagnostic_impl(
        Clear::None,
        Selection::Both,
        formats,
        Options {
            shift: None,
            handover,
            multipart,
            configure_b_base: true,
        },
        capture_round,
        capture_operand,
        capture_block,
    )
}

fn diagnostic_impl(
    clear: Clear,
    selection: Selection,
    formats: [u32; 4],
    options: Options,
    capture_round: usize,
    capture_operand: Selection,
    capture_block: u32,
) -> Diagnostic {
    let Options {
        shift,
        handover,
        multipart,
        configure_b_base,
    } = options;
    assert!(capture_round < 4);
    assert!(capture_block < 4);
    assert!(capture_operand != Selection::Both);
    assert!(formats.iter().all(|f| matches!(f, 4 | 5)));
    assert!(shift.is_none_or(|(_, m)| matches!(m, 1 | 4)));
    let mut req = Requirements::new(1);
    let input = req.scratch("source image", 16 + 512 * 4, 16, 0..1);
    let staged = req.semaphore("staged source retired", 0, 0..1);
    let cleared = req.semaphore("matrix clear retired", 0, 0..1);
    let loaded = req.semaphore("source handover retired", 0, 0..1);
    let released = req.semaphore("source release retired", 0, 0..1);
    let plan = req.plan(tt_isa::l1::DATA).unwrap();
    let (staged, cleared, loaded, released) = (
        plan.semaphore(staged),
        plan.semaphore(cleared),
        plan.semaphore(loaded),
        plan.semaphore(released),
    );
    let input = plan.addr(input);
    let mut up = datapath::src_thread_config();
    // src_thread_config establishes A.Base=0 and configuration bank 0.
    if configure_b_base {
        up.push(datapath::thread_entry(thread::SRCB_SET_Base, 0));
    }
    let mut math = crate::matmul::math_prelude();
    // Initialize whole modifier entries, including all clear/carry/fidelity bits.
    for (ab, dst) in [
        (
            thread::ADDR_MOD_AB_SEC0_SrcBIncr,
            thread::ADDR_MOD_DST_SEC0_DestIncr,
        ),
        (
            thread::ADDR_MOD_AB_SEC1_SrcBIncr,
            thread::ADDR_MOD_DST_SEC1_DestIncr,
        ),
        (
            thread::ADDR_MOD_AB_SEC4_SrcBIncr,
            thread::ADDR_MOD_DST_SEC4_DestIncr,
        ),
    ] {
        math.push(datapath::thread_entry(
            ab,
            if ab == thread::ADDR_MOD_AB_SEC0_SrcBIncr {
                0
            } else {
                1
            },
        ));
        math.push(datapath::thread_entry(dst, 0));
    }
    let unpack = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    let mut banks = Banks::after_reset();
    for (round, format) in [4, 4].into_iter().chain(formats).enumerate() {
        let clear = if matches!(round, 2 | 5) {
            clear
        } else {
            Clear::None
        };
        let staged_last = if matches!(round, 3 | 4) { 15 } else { 511 };
        if round != 0 {
            up.extend(sync::take(released, Before::EVERYTHING));
        }
        let mut cfg = ConfigWords::new();
        for u in [Unpacker::SrcA, Unpacker::SrcB] {
            datapath::unpack_src_config(&mut cfg, u, datapath::flat_descriptor(512), input, format);
        }
        cfg.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
        up.extend(datapath::config_program(&cfg));
        up.push(backend::stallwait(Before::EVERYTHING.mask(), backend::cond::CONFIG_BUSY).unwrap());
        up.push(
            encode::Setadcxy::ZERO
                .u0(1)
                .u1(1)
                .x0(1)
                .y0(1)
                .x1(1)
                .y1(1)
                .encode()
                .unwrap(),
        );
        up.push(
            encode::Setadczw::ZERO
                .u0(1)
                .u1(1)
                .z0(1)
                .w0(1)
                .z1(1)
                .w1(1)
                .encode()
                .unwrap(),
        );
        if round < 2 {
            // Src contents survive programs. Establish every row of both
            // physical banks with current-bank NOP zeros before seeding.
            let (i, next) = banks.unpacr_nop_zerosrc_a().unwrap();
            up.push(i);
            let (i, next) = next.unpacr_nop_zerosrc_b().unwrap();
            up.push(i);
            banks = next;
        }
        up.push(datapath::set_adc_x(Unpacker::SrcA, 0, staged_last));
        up.push(datapath::set_adc_x(Unpacker::SrcB, 0, staged_last));
        let (i, b) = banks.unpack_a_partial(unpack).unwrap();
        up.push(i);
        let (i, mut b) = b.unpack_b_partial(unpack).unwrap();
        up.push(i);
        if multipart {
            // Each regular UNPACR restarts its output address on this path.
            // Re-stage the same range to exercise multiple retired partials
            // while retaining identical contents in the regular control.
            up.push(datapath::set_adc_x(Unpacker::SrcA, 0, staged_last));
            up.push(datapath::set_adc_x(Unpacker::SrcB, 0, staged_last));
            let (i, next) = b.unpack_a_partial(unpack).unwrap();
            up.push(i);
            let (i, next) = next.unpack_b_partial(unpack).unwrap();
            up.push(i);
            b = next;
        }
        up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
        up.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());
        up.push(sync::post(staged));
        math.extend(sync::take(staged, Before::EVERYTHING));
        // T1 may clear only after the staged-data publication. T0 waits for
        // its retirement before refilling. The same moved Banks value tracks
        // both roles' instruction streams at construction time.
        up.extend(sync::take(cleared, Before::EVERYTHING));
        up.push(datapath::set_adc_x(Unpacker::SrcA, 0, 15));
        up.push(datapath::set_adc_x(Unpacker::SrcB, 0, 15));
        macro_rules! finish {
            ($b:expr) => {{
                let (i, a) = $b.unpack_a(unpack).unwrap();
                up.push(i);
                let (i, b) = a.unpack_b(unpack).unwrap();
                up.push(i);
                b
            }};
        }
        let mut loaded_banks = if let Handover::Explicit(selected) = handover {
            // Stage the same final row as the regular control, but retain the
            // current bank. The NOP must perform publication without a read.
            let (i, b) = b.unpack_a_partial(unpack).unwrap();
            up.push(i);
            let (i, b) = b.unpack_b_partial(unpack).unwrap();
            up.push(i);
            let a = if matches!(selected, Selection::A | Selection::Both) {
                let (seq, a) = b.handover_a().unwrap();
                up.extend(seq);
                a
            } else {
                let (i, a) = b.unpack_a(unpack).unwrap();
                up.push(i);
                a
            };
            if matches!(selected, Selection::B | Selection::Both) {
                let (seq, b) = a.handover_b().unwrap();
                up.extend(seq);
                b
            } else {
                let (i, b) = a.unpack_b(unpack).unwrap();
                up.push(i);
                b
            }
        } else {
            match (clear, selection) {
                (Clear::UnpackerMatrix, Selection::A) => {
                    let (i, b) = b.zerosrc_unpacker_a().unwrap();
                    math.push(i);
                    finish!(b)
                }
                (Clear::UnpackerMatrix, Selection::B) => {
                    let (i, b) = b.zerosrc_unpacker_b().unwrap();
                    math.push(i);
                    finish!(b)
                }
                (Clear::UnpackerMatrix, Selection::Both) => {
                    let (i, b) = b.zerosrc_unpacker_both().unwrap();
                    math.push(i);
                    finish!(b)
                }
                (Clear::UnpackerNop, Selection::A) => {
                    let (i, b) = b.unpacr_nop_zerosrc_a().unwrap();
                    up.push(i);
                    finish!(b)
                }
                (Clear::UnpackerNop, Selection::B) => {
                    let (i, b) = b.unpacr_nop_zerosrc_b().unwrap();
                    up.push(i);
                    finish!(b)
                }
                (Clear::UnpackerNop, Selection::Both) => {
                    let (i, b) = b.unpacr_nop_zerosrc_a().unwrap();
                    up.push(i);
                    let (i, b) = b.unpacr_nop_zerosrc_b().unwrap();
                    up.push(i);
                    finish!(b)
                }
                _ => finish!(b),
            }
        };
        math.extend(sync::post_after(Unit::Matrix, cleared));
        up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
        up.extend(sync::post_after(Unit::Unpacker1, loaded));
        math.extend(sync::take(loaded, Before::EVERYTHING));
        if clear == Clear::Matrix {
            let (i, next) = match selection {
                Selection::A => loaded_banks.zerosrc_matrix_a().unwrap(),
                Selection::B => loaded_banks.zerosrc_matrix_b().unwrap(),
                Selection::Both => loaded_banks.zerosrc_matrix_both().unwrap(),
            };
            math.push(i);
            loaded_banks = next;
        }
        if let Some((mode, modifier)) = shift.filter(|_| round >= 2) {
            math.push(encode::Setrwc::ZERO.src_b(1).src_b_val(1).encode().unwrap());
            for _ in 0..2 {
                let (i, next) = loaded_banks.shiftxb(63, modifier, mode).unwrap();
                math.push(i);
                loaded_banks = next;
            }
            math.push(encode::Setrwc::ZERO.src_b(1).src_b_val(0).encode().unwrap());
            for _ in 0..2 {
                let (i, next) = loaded_banks.shiftxb(0, 0, mode).unwrap();
                math.push(i);
                loaded_banks = next;
            }
        }
        math.push(
            encode::Setrwc::ZERO
                .src_a(1)
                .src_b(1)
                .dst(1)
                .encode()
                .unwrap(),
        );
        if round == capture_round + 2 {
            if capture_operand == Selection::A {
                // ttsim accepts only eight-row MOVA2D.
                for dst in [0, 8] {
                    let (i, next) = loaded_banks
                        .mova2d(
                            encode::Mova2D::ZERO
                                .move8_rows(1)
                                .src_row(capture_block * 16 + dst)
                                .dst_row(dst),
                        )
                        .unwrap();
                    math.push(i);
                    loaded_banks = next;
                }
            } else {
                for dst in 0..16 {
                    let (i, next) = loaded_banks
                        .movb2d(
                            encode::Movb2D::ZERO
                                .src_row(capture_block * 16 + dst)
                                .dst_row(dst),
                        )
                        .unwrap();
                    math.push(i);
                    loaded_banks = next;
                }
            }
        }
        // Exercise separate A/B releases as well as the both-operands form.
        banks = if round % 2 == 0 {
            let (i, next) = loaded_banks.cleardvalid_both().unwrap();
            math.push(i);
            next
        } else {
            let (i, next) = loaded_banks.cleardvalid_a().unwrap();
            math.push(i);
            let (i, next) = next.cleardvalid_b().unwrap();
            math.push(i);
            next
        };
        math.extend(sync::post_after(Unit::Matrix, released));
    }
    up.extend(sync::take(released, Before::EVERYTHING));
    Diagnostic {
        unpack: up,
        math,
        init: plan.semaphore_init(),
        input,
        banks,
    }
}
