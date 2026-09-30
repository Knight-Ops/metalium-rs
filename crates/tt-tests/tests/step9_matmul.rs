//! Phase 6, first milestone: `MVMUL`, `Dst += SrcB @ SrcA`, one 8x16 by 16x16 block.
//!
//! # Two oracles, two claims -- as in step 8
//!
//! `tt_isa::numerics::mvmul_reference` ports the documented fidelity-phase model
//! (`MVMUL.md`, `MatrixUnit.md:143-165`) and says what each output datum is, bit
//! for bit. It refuses to answer unless every product and sum is exact, because
//! `MVMUL.md` calls its float model "a rough guide" to order and precision; the
//! operands here are chosen so it always answers. `burn-flex` says which datums
//! meet which -- that it is `SrcB @ SrcA` and not a transpose -- for small integers
//! where IEEE and the Matrix Unit cannot disagree. No epsilon anywhere.
//!
//! # Where the operands sit, and why
//!
//! Each operand is staged behind zero rows so that it lands at the row the RWCs
//! point `MVMUL` at. `SrcA` goes at row 16 rather than 8 because ttsim refuses
//! `src_a_row=8` (row 41), which `MVMUL.md`'s `& 0x38` allows.

use tt_isa::backend::{self, Before, ConfigWords, ThreadConfigEntry};
use tt_isa::cfg::generated::{alu, thread};
use tt_isa::cfg::ThreadConfigField;
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::matrix::{Banks, Loaded};
use tt_isa::numerics::mvmul_reference;
use tt_isa::sfpu;
use tt_isa::tile::{fp32_to_tf32, L1Format, TileImage};
use tt_tests::datapath::{
    flat_descriptor, set_adc_x, src_thread_config, unpack_src_config, Unpacker, SCRATCH_GPR, STAGE,
};
use tt_tests::harness::{self, Run};

const ROW: usize = 16;
const SRC_A_ROW: usize = 16;
const SRC_B_ROW: usize = 8;
const TF32_CODE: u32 = 4;

const STAGE_A: u64 = STAGE;
const STAGE_B: u64 = STAGE + 0x2000;

type MatA = [[f32; 16]; 16];
type MatB = [[f32; 16]; 8];
type Body = Box<dyn FnOnce(Banks<Loaded, Loaded>, &mut Vec<Instruction>)>;

const ZERO_DST: MatB = [[0f32; 16]; 8];

/// Stage `rows` as a flat FP32 run that lands at `Src` row `src_row`, column 0.
fn stage_operand(src_row: usize, rows: &[[f32; 16]]) -> (Vec<u8>, u32) {
    let mut datums = vec![0u32; src_row * ROW];
    datums.extend(rows.iter().flatten().map(|v| v.to_bits()));
    let n = datums.len() as u32;
    let image = TileImage::new(flat_descriptor(n), L1Format::Fp32).unwrap();
    let mut staged = vec![0u8; image.total_bytes()];
    for (i, d) in datums.iter().enumerate() {
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + 4].copy_from_slice(&d.to_le_bytes());
    }
    (staged, n)
}

fn thread_entry(field: ThreadConfigField, value: u16) -> Instruction {
    ThreadConfigEntry::zeroed(field.addr32())
        .set(field, value)
        .unwrap()
        .encode()
        .unwrap()
}

fn fidelity_base(phase: u16) -> Instruction {
    thread_entry(thread::FIDELITY_BASE_Phase, phase)
}

/// Configure, stage both operands, zero `Dst`, point the RWCs at the operands,
/// then hand the loaded banks to `body` for the Matrix Unit work.
fn program(na: u32, nb: u32, body: Body) -> (Vec<Instruction>, Vec<Instruction>) {
    // Split the way LLK splits it (`harness::Roles`): the unpacks on thread 0,
    // everything the Matrix Unit does on thread 1. The address modifiers, the
    // fidelity base and the RWCs are the *math* thread's `ThreadConfig` and
    // counters, so they are set there.
    let mut up = src_thread_config();
    let mut words = ConfigWords::new();
    unpack_src_config(
        &mut words,
        Unpacker::SrcA,
        flat_descriptor(na),
        STAGE_A,
        TF32_CODE,
    );
    unpack_src_config(
        &mut words,
        Unpacker::SrcB,
        flat_descriptor(nb),
        STAGE_B,
        TF32_CODE,
    );
    // FP32 `Dst`, and `Zero_Flag_disabled_src` clear: the other setting is the
    // "keep SrcB denormals" mode `MVMUL.md` says must not be used.
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    words
        .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
        .unwrap();
    let mut buf = vec![sfpu::nop(); words.program_len()];
    let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
    up.extend_from_slice(&buf[..k]);

    let unpack = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    let banks = Banks::after_reset();
    up.push(set_adc_x(Unpacker::SrcA, 0, na - 1));
    let (i, banks) = banks.unpack_a(unpack).unwrap();
    up.push(i);
    up.push(set_adc_x(Unpacker::SrcB, 0, nb - 1));
    let (i, banks) = banks.unpack_b(unpack).unwrap();
    up.push(i);
    up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    up.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());

    let mut math = vec![tt_tests::datapath::state_id()];
    // Address modifier 0 moves nothing; 1 advances the fidelity phase only (with
    // the measured Blackhole `AddrMod` position; divergence row 42).
    math.push(thread_entry(thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0));
    math.push(thread_entry(thread::ADDR_MOD_DST_SEC0_DestIncr, 0));
    math.push(thread_entry(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0));
    math.push(thread_entry(thread::ADDR_MOD_DST_SEC1_FidelityIncr, 1));
    math.push(fidelity_base(0));
    // `SrcAVal` is four bits, so 16 takes two steps: set 8, then add the carried
    // 8 (`SETRWC.md`: `if (SrcACr) SrcAVal += RWC.SrcA_Cr`).
    math.push(encode::Setrwc::ZERO.src_a(1).src_a_val(8).encode().unwrap());
    math.push(
        encode::Setrwc::ZERO
            .src_a(1)
            .src_a_cr(1)
            .src_a_val(SRC_A_ROW as u32 - 8)
            .src_b(1)
            .src_b_val(SRC_B_ROW as u32)
            .dst(1)
            .dst_val(0)
            .fidelity(1)
            .encode()
            .unwrap(),
    );
    // All of `Dst`: mode 3 is `CLR_ALL` in LLK's encoding too (row 40).
    math.push(encode::Zeroacc::ZERO.mode(3).encode().unwrap());
    body(banks, &mut math);
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    (up, math)
}

/// One `MVMUL` at `dst_row`, releasing both operands.
fn once(dst_row: u32) -> Body {
    Box::new(move |banks, p| {
        let (i, _) = banks
            .mvmul_release_both(encode::Mvmul::ZERO.dst_row(dst_row))
            .unwrap();
        p.push(i);
    })
}

/// `Dst` rows 0..16 after running `body` over `a` and `b`.
fn run(a: &MatA, b: &MatB, body: Body) -> Vec<u32> {
    let (sa, na) = stage_operand(SRC_A_ROW, a);
    let (sb, nb) = stage_operand(SRC_B_ROW, b);
    let (unpack, math) = program(na, nb, body);
    // `fork_scope` gives the child no return channel.
    let path = std::env::temp_dir().join(format!(
        "ttmvmul-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    harness::in_device(|dev| {
        let roles = harness::Roles {
            unpack: &unpack,
            math: &math,
            pack: &[],
        };
        let out = harness::run(
            dev,
            &Run::roles(roles)
                .stage(&[(STAGE_A, &sa), (STAGE_B, &sb)])
                .dump_rows(16),
        );
        let bytes: Vec<u8> = (0..16 * ROW)
            .flat_map(|f| out.dst_at(f / ROW, f % ROW).to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).unwrap();
    });
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_block(dst: &[u32], first_row: usize, want: &MatB, what: &str) {
    for (i, want_row) in want.iter().enumerate() {
        for (j, w) in want_row.iter().enumerate() {
            let got = dst[(first_row + i) * ROW + j];
            assert_eq!(
                got,
                w.to_bits(),
                "{what}: Dst[{}][{j}] is {got:#010x} = {}, want {w}",
                first_row + i,
                f32::from_bits(got),
            );
        }
    }
}

/// A deterministic generator; CI is flake-free on purpose.
struct Lcg(u64);

impl Lcg {
    /// A uniformly chosen integer in `-bound..=bound`, as an `f32`.
    fn int(&mut self, bound: i32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as i32 % (2 * bound + 1) - bound) as f32
    }
}

fn identity() -> MatA {
    let mut a = [[0f32; 16]; 16];
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    a
}

fn truncated<const R: usize>(m: &[[f32; 16]; R]) -> [[f32; 16]; R] {
    let mut out = *m;
    for v in out.iter_mut().flatten() {
        *v = f32::from_bits(fp32_to_tf32(v.to_bits()));
    }
    out
}

#[test]
fn an_identity_src_a_returns_src_b() {
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    let dst = run(&identity(), &b, once(0));
    assert_block(&dst, 0, &b, "SrcB @ I");
}

/// Small integers, where the device, the documented model and Burn must all agree
/// exactly: phase 0 takes five significant bits of `SrcA` and seven of `SrcB`, so
/// `|A| <= 31` and `|B| <= 127` lose nothing, and every sum stays under 2^24.
#[test]
fn small_integer_products_match_the_model_and_burn() {
    use burn_tensor::{Tensor, TensorData};
    type B = burn_flex::Flex;
    let device = burn_flex::FlexDevice;

    let mut rng = Lcg(0x5eed);
    let mut a = [[0f32; 16]; 16];
    let mut b = [[0f32; 16]; 8];
    a.iter_mut().flatten().for_each(|v| *v = rng.int(31));
    b.iter_mut().flatten().for_each(|v| *v = rng.int(127));

    let model = mvmul_reference(&ZERO_DST, &b, &a, &[0]).expect("small integers are exact");
    let flat = |m: &[[f32; 16]]| m.iter().flatten().copied().collect::<Vec<f32>>();
    let tb = Tensor::<B, 2>::from_data(TensorData::new(flat(&b), [8, 16]), &device);
    let ta = Tensor::<B, 2>::from_data(TensorData::new(flat(&a), [16, 16]), &device);
    let burn: Vec<f32> = tb.clone().matmul(ta.clone()).into_data().to_vec().unwrap();
    let mut burn_block = ZERO_DST;
    for (k, v) in burn.iter().enumerate() {
        burn_block[k / 16][k % 16] = *v;
    }
    assert_eq!(
        model, burn_block,
        "the model and Burn must agree before either judges the device"
    );

    let dst = run(&a, &b, once(0));
    assert_block(&dst, 0, &burn_block, "SrcB @ SrcA");

    // Control: the gate distinguishes `SrcB @ SrcA` from `SrcB @ SrcA^T`.
    let transposed: Vec<f32> = tb.matmul(ta.transpose()).into_data().to_vec().unwrap();
    assert!((0..128).any(|k| dst[k] != transposed[k].to_bits()));
}

/// `MVMUL` is `+=`: without a `ZEROACC` between them, two give twice the product.
#[test]
fn mvmul_accumulates_into_dst() {
    let mut rng = Lcg(7);
    let mut a = [[0f32; 16]; 16];
    let mut b = [[0f32; 16]; 8];
    a.iter_mut().flatten().for_each(|v| *v = rng.int(15));
    b.iter_mut().flatten().for_each(|v| *v = rng.int(63));
    let first = mvmul_reference(&ZERO_DST, &b, &a, &[0]).unwrap();
    let twice = mvmul_reference(&first, &b, &a, &[0]).unwrap();
    assert_ne!(first, twice);
    let dst = run(
        &a,
        &b,
        Box::new(|banks, p| {
            let (i, banks) = banks.mvmul(encode::Mvmul::ZERO).unwrap();
            p.push(i);
            let (i, _) = banks.mvmul_release_both(encode::Mvmul::ZERO).unwrap();
            p.push(i);
        }),
    );
    assert_block(&dst, 0, &twice, "two MVMULs");
}

/// `DstRow` places the eight output rows; the eight it does not name stay zeroed.
#[test]
fn dst_row_places_the_block() {
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = -(k as f32) - 1.0;
    }
    let dst = run(&identity(), &b, once(8));
    assert_block(&dst, 8, &b, "DstRow = 8");
    assert_block(
        &dst,
        0,
        &ZERO_DST,
        "rows 0..8, which DstRow = 8 must not touch",
    );
}

/// Operands whose mantissas use bits in every phase's share, so each phase's
/// contribution is visible.
fn fidelity_operands() -> (MatA, MatB) {
    let mut rng = Lcg(0xf1de);
    let mut a = [[0f32; 16]; 16];
    let mut b = [[0f32; 16]; 8];
    // A: an integer part phase 0 sees, plus 2^-5, which only the odd phases see.
    // B: likewise plus 2^-7, which only phases 2 and 3 see. Kept that narrow so
    // every sum of sixteen products fits FP32's 24 bits -- with a 2^-9 term too
    // the sums need 27, and `mvmul_reference` rightly refuses to answer.
    for v in a.iter_mut().flatten() {
        *v = 1.0 + rng.int(3).abs() + 2f32.powi(-5);
    }
    for v in b.iter_mut().flatten() {
        *v = 1.0 + rng.int(3).abs() + 2f32.powi(-7);
    }
    (truncated(&a), truncated(&b))
}

#[test]
fn phase_zero_alone_drops_exactly_the_bits_the_model_says() {
    let (a, b) = fidelity_operands();
    let model = mvmul_reference(&ZERO_DST, &b, &a, &[0]).unwrap();
    let full = mvmul_reference(&ZERO_DST, &b, &a, &[0, 1, 2, 3]).unwrap();
    assert_ne!(model, full, "the operands must exercise the later phases");
    let dst = run(&a, &b, once(0));
    assert_block(&dst, 0, &model, "phase 0 only");
}

/// All four phases, selected with `FIDELITY_BASE_Phase` between `MVMUL`s.
#[test]
fn four_phases_through_fidelity_base_recover_the_exact_product() {
    let (a, b) = fidelity_operands();
    let full = mvmul_reference(&ZERO_DST, &b, &a, &[0, 1, 2, 3]).unwrap();
    // The four phases are the whole product, computed independently.
    for (i, row) in full.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let exact: f64 = (0..16)
                .map(|k| f64::from(b[i][k]) * f64::from(a[k][j]))
                .sum();
            assert_eq!(f64::from(*v), exact);
        }
    }
    let dst = run(
        &a,
        &b,
        Box::new(|mut banks, p| {
            for phase in 0..3 {
                p.push(fidelity_base(phase));
                let (i, next) = banks.mvmul(encode::Mvmul::ZERO).unwrap();
                p.push(i);
                banks = next;
            }
            p.push(fidelity_base(3));
            let (i, _) = banks.mvmul_release_both(encode::Mvmul::ZERO).unwrap();
            p.push(i);
        }),
    );
    assert_block(&dst, 0, &full, "phases 0..4 via FIDELITY_BASE");
}

/// All four phases the way a kernel steps them: address modifier 1 increments the
/// RWC `FidelityPhase` after each `MVMUL` (`RWCs.md`, `ApplyAddrMod`). This is the
/// gate that found row 42: with the Wormhole encoding of `AddrMod`, it summed phase
/// 0 four times.
#[test]
fn four_phases_through_the_rwc_recover_the_exact_product() {
    let (a, b) = fidelity_operands();
    let full = mvmul_reference(&ZERO_DST, &b, &a, &[0, 1, 2, 3]).unwrap();
    let dst = run(
        &a,
        &b,
        Box::new(|mut banks, p| {
            for _ in 0..3 {
                let (i, next) = banks.mvmul(encode::Mvmul::ZERO.addr_mod(1)).unwrap();
                p.push(i);
                banks = next;
            }
            let (i, _) = banks
                .mvmul_release_both(encode::Mvmul::ZERO.addr_mod(1))
                .unwrap();
            p.push(i);
        }),
    );
    assert_block(&dst, 0, &full, "phases 0..4 via ADDR_MOD FidelityIncr");
}

/// **`MVMUL`'s `AddrMod` is bits 14..15 on Blackhole, not the 15..16 of
/// `Bits32.lua`.** The only diagram in the pinned specification is Wormhole's, and
/// it is wrong here: its `addr_mod(1)` sets bit 15, which Blackhole reads as index
/// **2**, so it applies the wrong modifier and says nothing (divergence row 42).
///
/// This gate is the evidence `xtask/src/gen_isa/Bits32_BH.lua` cites for the
/// measured `MVMUL_BH` layout, which is what `encode::Mvmul` now generates; the
/// Wormhole diagram lives on as `encode::wormhole::Mvmul`. `gen-isa` refuses the
/// layout if this test disappears.
///
/// Measured with a modifier that advances `Dst` by 8 in entry 1 and nothing in
/// entry 2: a second `MVMUL` lands at row 8 only when the first selected entry 1.
#[test]
fn mvmul_addr_mod_sits_one_bit_lower_on_blackhole() {
    let blackhole = encode::Mvmul::ZERO.addr_mod(1).encode().unwrap().word();
    let wormhole = encode::wormhole::Mvmul::ZERO
        .addr_mod(1)
        .encode()
        .unwrap()
        .word();
    assert_eq!(blackhole & 0x0001_c000, 1 << 14, "the measured layout");
    assert_eq!(wormhole & 0x0001_c000, 1 << 15, "the Wormhole diagram");

    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    // The first `MVMUL` carries `addr_mod_bits` on top of the typestate's word.
    let body = |addr_mod_bits: u32| -> Body {
        Box::new(move |banks, p| {
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC1_DestIncr, 8));
            p.push(thread_entry(thread::ADDR_MOD_DST_SEC2_DestIncr, 0));
            let (i, banks) = banks.mvmul(encode::Mvmul::ZERO).unwrap();
            p.push(Instruction::new(i.word() | addr_mod_bits, i.def()));
            let (i, _) = banks.mvmul_release_both(encode::Mvmul::ZERO).unwrap();
            p.push(i);
        })
    };
    let doubled = {
        let mut d = b;
        d.iter_mut().flatten().for_each(|v| *v *= 2.0);
        d
    };

    // The measured encoding: modifier 1, so the second `MVMUL` goes eight rows down.
    let dst = run(&identity(), &b, body(blackhole & 0x0001_c000));
    assert_block(&dst, 0, &b, "first MVMUL, Blackhole addr_mod(1)");
    assert_block(&dst, 8, &b, "second MVMUL after modifier 1");

    // The Wormhole encoding of the same request: modifier 2, which moves nothing,
    // so both land on row 0.
    let dst = run(&identity(), &b, body(wormhole & 0x0001_c000));
    assert_block(&dst, 0, &doubled, "both MVMULs, Wormhole addr_mod(1)");
}

/// The whole identity result, printed next to what it should be. Silicon only:
/// a diagnostic for the open `Src`-path losses, not a claim.
#[cfg(feature = "silicon")]
#[test]
fn dump_the_identity_block() {
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    let dst = run(&identity(), &b, once(0));
    for (r, want_row) in b.iter().enumerate() {
        let got: Vec<String> = (0..16)
            .map(|c| format!("{:5}", f32::from_bits(dst[r * ROW + c])))
            .collect();
        let want: Vec<String> = want_row.iter().map(|w| format!("{w:5}")).collect();
        println!("row {r} got  {}", got.join(" "));
        println!("row {r} want {}", want.join(" "));
    }
}

/// `MOVD2A`'s, `MOVD2B`'s and `MOVB2A`'s `AddrMod` sit at bits 14..15 on
/// Blackhole, one bit lower than the Wormhole diagrams draw them -- as `MVMUL`'s,
/// `MOVA2D`'s and `MOVB2D`'s do (row 42), and as LLK's `addr_mode << 14` has them.
///
/// Each case moves four rows into `Src` twice, both moves carrying `addr_mod(1)`,
/// where entry 1 advances the counter of the rows being *read* by four and entry
/// 2 advances nothing. So the second move reads the next four rows with the
/// measured encoding and the same four again with the Wormhole one (modifier 2).
/// The written `Src` rows are then copied into `Dst` with `MOVA2D`/`MOVB2D`.
/// Four-row moves because ttsim implements no other form of these (as for
/// `MOVA2D`, row 37: `tensix_movd2a: instr_mod=0` is `UnsupportedFunctionality`).
///
/// At the body's start the RWCs are `SrcA` 16, `SrcB` 8, `Dst` 0 (`program`), so
/// `Src` row arguments here are offsets from the staged operands.
#[test]
fn mov_to_src_addr_mod_sits_one_bit_lower_on_blackhole() {
    let words = [
        (
            "MOVD2A",
            encode::Movd2A::ZERO.addr_mod(1).encode().unwrap().word(),
            encode::wormhole::Movd2A::ZERO
                .addr_mod(1)
                .encode()
                .unwrap()
                .word(),
        ),
        (
            "MOVD2B",
            encode::Movd2B::ZERO.addr_mod(1).encode().unwrap().word(),
            encode::wormhole::Movd2B::ZERO
                .addr_mod(1)
                .encode()
                .unwrap()
                .word(),
        ),
        (
            "MOVB2A",
            // (src_a_row, addr_mod, move4_rows, src_b_row)
            encode::movb2_a(0, 1, 0, 0).unwrap().word(),
            encode::wormhole::movb2_a(0, 1, 0, 0).unwrap().word(),
        ),
    ];
    for (name, bh, wh) in words {
        assert_eq!(bh & 0x00ff_ffff, 1 << 14, "{name}: the measured layout");
        assert_eq!(wh & 0x00ff_ffff, 1 << 15, "{name}: the Wormhole diagram");
    }

    // Every datum distinct and exact in TF32, so a row names where it came from.
    let mut a = [[0f32; 16]; 16];
    for (k, v) in a.iter_mut().flatten().enumerate() {
        *v = 1.0 + k as f32;
    }
    let mut b = [[0f32; 16]; 8];
    for (k, v) in b.iter_mut().flatten().enumerate() {
        *v = 1000.0 + k as f32;
    }
    let row_bits = |r: &[f32; 16]| -> Vec<u32> { r.iter().map(|v| v.to_bits()).collect() };
    let dst_row = |dst: &[u32], r: usize| dst[r * ROW..(r + 1) * ROW].to_vec();

    // Entry 1 advances `SrcB` by `b_incr` and `Dst` by `d_incr`; entry 2, nothing.
    let modifiers = |p: &mut Vec<Instruction>, b_incr: u16, d_incr: u16| {
        p.push(thread_entry(thread::ADDR_MOD_AB_SEC1_SrcBIncr, b_incr));
        p.push(thread_entry(thread::ADDR_MOD_DST_SEC1_DestIncr, d_incr));
        p.push(thread_entry(thread::ADDR_MOD_AB_SEC2_SrcBIncr, 0));
        p.push(thread_entry(thread::ADDR_MOD_DST_SEC2_DestIncr, 0));
    };
    let raw = |i: Instruction, bits: u32| Instruction::new(i.word() | bits, i.def());
    let dst_rwc_zero = || encode::Setrwc::ZERO.dst(1).dst_val(0).encode().unwrap();

    for (addr_mod, encoding) in [(words[0].1, "measured"), (words[0].2, "Wormhole")] {
        let bits = addr_mod & 0x00ff_ffff;
        let second = if encoding == "measured" { 4 } else { 0 };

        // MOVD2A: B rows 0..8 into `Dst` 0..8; `Dst` 0..4, then `Dst` RWC..+4,
        // into `SrcA` 24..28 and 28..32; `SrcA` 24..32 back into `Dst` 8..16.
        let dst = run(
            &a,
            &b,
            Box::new(move |_banks, p| {
                modifiers(p, 0, 4);
                for row in [0, 4] {
                    p.push(
                        encode::Movb2D::ZERO
                            .move4_rows(1)
                            .src_row(row)
                            .dst_row(row)
                            .encode()
                            .unwrap(),
                    );
                }
                for src_row in [8, 12] {
                    let i = encode::Movd2A::ZERO
                        .move4_rows(1)
                        .src_row(src_row)
                        .encode()
                        .unwrap();
                    p.push(raw(i, bits));
                }
                p.push(dst_rwc_zero());
                p.push(
                    encode::Mova2D::ZERO
                        .move8_rows(1)
                        .src_row(8)
                        .dst_row(8)
                        .encode()
                        .unwrap(),
                );
            }),
        );
        assert_eq!(
            dst_row(&dst, 8),
            row_bits(&b[0]),
            "MOVD2A {encoding}: first"
        );
        assert_eq!(
            dst_row(&dst, 12),
            row_bits(&b[second]),
            "MOVD2A {encoding}: second"
        );
    }

    for (addr_mod, encoding) in [(words[1].1, "measured"), (words[1].2, "Wormhole")] {
        let bits = addr_mod & 0x00ff_ffff;
        let second = if encoding == "measured" { 4 } else { 0 };

        // MOVD2B: A rows 0..8 into `Dst` 0..8; `Dst` 0..4, then `Dst` RWC..+4,
        // into `SrcB` 16..20 and 20..24; those back into `Dst` 8..16.
        let dst = run(
            &a,
            &b,
            Box::new(move |_banks, p| {
                modifiers(p, 0, 4);
                p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
                for src_row in [8, 12] {
                    let i = encode::Movd2B::ZERO
                        .move4_rows(1)
                        .src_row(src_row)
                        .encode()
                        .unwrap();
                    p.push(raw(i, bits));
                }
                p.push(dst_rwc_zero());
                for row in [8, 12] {
                    p.push(
                        encode::Movb2D::ZERO
                            .move4_rows(1)
                            .src_row(row)
                            .dst_row(row)
                            .encode()
                            .unwrap(),
                    );
                }
            }),
        );
        assert_eq!(
            dst_row(&dst, 8),
            row_bits(&a[0]),
            "MOVD2B {encoding}: first"
        );
        assert_eq!(
            dst_row(&dst, 12),
            row_bits(&a[second]),
            "MOVD2B {encoding}: second"
        );
    }

    for (addr_mod, encoding) in [(words[2].1, "measured"), (words[2].2, "Wormhole")] {
        let bits = addr_mod & 0x00ff_ffff;
        let second = if encoding == "measured" { 4 } else { 0 };

        // MOVB2A: `SrcB` 8..12, then `SrcB` RWC..+4 (B rows 0..4, then 4..8 or
        // 0..4 again), into `SrcA` 24..28 and 28..32; `SrcA` 24..32 back into
        // `Dst` 0..8.
        let dst = run(
            &a,
            &b,
            Box::new(move |_banks, p| {
                modifiers(p, 4, 0);
                for src_a_row in [8, 12] {
                    // (src_a_row, addr_mod, move4_rows, src_b_row)
                    let i = encode::movb2_a(src_a_row, 0, 1, 0).unwrap();
                    p.push(raw(i, bits));
                }
                p.push(
                    encode::Mova2D::ZERO
                        .move8_rows(1)
                        .src_row(8)
                        .encode()
                        .unwrap(),
                );
            }),
        );
        assert_eq!(
            dst_row(&dst, 0),
            row_bits(&b[0]),
            "MOVB2A {encoding}: first"
        );
        assert_eq!(
            dst_row(&dst, 4),
            row_bits(&b[second]),
            "MOVB2A {encoding}: second"
        );
    }
}
