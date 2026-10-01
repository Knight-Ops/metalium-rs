//! Phase 10 gate (F2, F5, X1): SFPU programs, instruction by instruction,
//! against the interpreter.
//!
//! `tt_kernels::sfpu::interp` computes what a program does from the
//! specification's functional models; this is where that is held to the
//! device. Each case is a program from the builder, run over two whole tiles
//! in `Dst` (rows 0 and 64, unpacked as `step25_dst_tile` does) writing a
//! third (rows 128..192, packed back), and the claim is the interpreter's
//! tile **bit for bit** -- every one of 1024 datums, operands chosen to reach
//! every branch the models have: both zeros, both infinities, NaNs of either
//! sign, denormals, the extremes, and ordinary values of either sign.
//!
//! Every case runs twice, its row loop replayed (`REPLAY`, the row counter
//! stepped by an address modifier) and unrolled (immediate addresses), and
//! both must match the same interpreter tile: that is the X1 gate.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::isa::Instruction;
use tt_isa::tile::{L1Format, TileImage};
use tt_kernels::datapath::{
    config_program, pack_tile_from_dst, state_id, thread_config, tile_descriptor,
    tile_unpack_config, unpack_tile_to_dst, OUT, STAGE,
};
use tt_kernels::sfpu::interp::Vector;
use tt_kernels::sfpu::{Cond, Format, LReg, LoopForm, LoopPolicy, Program};
use tt_tests::harness::{self, Roles, Run};

const STAGE_B: u64 = STAGE + tt_isa::dm::TILE_SLOT;

fn operands(seed: u32) -> Vec<u32> {
    let specials: [u32; 14] = [
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_1234,
        0x0000_0001, // denormals
        0x8040_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x4b00_0001,
    ];
    let mut s = seed | 1;
    (0..1024)
        .map(|i| {
            if i % 11 == 0 {
                return specials[(i / 11 + seed as usize) % specials.len()];
            }
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s & 0x8000_0000) | (((s >> 8) % 60 + 100) << 23) | (s & 0x7f_ffff)
        })
        .collect()
}

fn image(datums: &[u32]) -> Vec<u8> {
    let img = TileImage::new(tile_descriptor(), L1Format::Fp32).unwrap();
    let mut b = vec![0u8; img.total_bytes()];
    for (i, d) in datums.iter().enumerate() {
        let at = img.datum_bit_offset(i) / 8;
        b[at..at + 4].copy_from_slice(&d.to_le_bytes());
    }
    b
}

/// Run `math` between unpacking `a` and `b` to rows 0 and 64 and packing
/// rows 128..192, and return the packed tile.
fn on_device(dev: &mut harness::Dev<'_>, math: &[Instruction], a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut unpack = thread_config();
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, STAGE);
    unpack.extend(config_program(&words));
    unpack.extend(unpack_tile_to_dst(STAGE, 0));
    unpack.extend(unpack_tile_to_dst(STAGE_B, 64));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    let mut m = vec![state_id()];
    m.extend_from_slice(math);
    m.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
    let mut pack = vec![state_id()];
    pack.extend(pack_tile_from_dst(OUT, 128));
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    let (ia, ib) = (image(a), image(b));
    let sentinel = vec![0xA5u8; 4096];
    let out = harness::run(
        dev,
        &Run::roles(Roles {
            unpack: &unpack,
            math: &m,
            pack: &pack,
        })
        .stage(&[(STAGE, &ia), (STAGE_B, &ib), (OUT, &sentinel)])
        .dump_rows(0)
        .read_back(&[(OUT, 4096)]),
    );
    out.l1[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A case: a body over one row group's A (in `L0`) and B (in `L1`), leaving
/// its result in `L2`.
type Body = fn(&mut Program);

fn program(policy: LoopPolicy, body: Body) -> Program {
    let mut p = Program::with_policy(policy);
    p.for_each_row_group(64, |p, o| {
        p.load(LReg::L0, Format::Fp32, o);
        p.load(LReg::L1, Format::Fp32, 64 + o);
        body(p);
        p.store(LReg::L2, Format::Fp32, 128 + o);
    });
    p
}

fn cases() -> Vec<(&'static str, Body)> {
    vec![
        ("add", |p| p.add(LReg::L0, LReg::L1, LReg::L2)),
        ("sub", |p| p.sub(LReg::L0, LReg::L1, LReg::L2)),
        ("mul", |p| p.mul(LReg::L0, LReg::L1, LReg::L2)),
        ("mad by an immediate", |p| {
            p.loadi(LReg::L3, 1.1);
            p.mad(LReg::L0, LReg::L3, LReg::L1, LReg::L2);
        }),
        ("negated mad", |p| {
            p.nmad(LReg::L0, LReg::L1, LReg::L0, LReg::L2)
        }),
        ("mad into its own operand, then used", |p| {
            p.mad(LReg::L0, LReg::L1, LReg::L1, LReg::L1);
            p.mul(LReg::L1, LReg::L0, LReg::L2);
        }),
        ("mov", |p| p.mov(LReg::L1, LReg::L2)),
        ("neg", |p| p.neg(LReg::L0, LReg::L2)),
        ("abs", |p| p.abs(LReg::L0, LReg::L2)),
        ("set sign", |p| {
            p.set_sign(LReg::L0, true, LReg::L3);
            p.set_sign(LReg::L3, false, LReg::L2);
            p.add(LReg::L2, LReg::L3, LReg::L2);
        }),
        ("a BF16 immediate", |p| p.loadi(LReg::L2, -2.5)),
        ("relu: zero where negative", |p| {
            p.mov(LReg::L0, LReg::L2);
            p.if_(Cond::Lt0(LReg::L0), |p| p.mov(LReg::ZERO, LReg::L2));
        }),
        ("if-else", |p| {
            p.if_else(
                Cond::Gte0(LReg::L1),
                |p| p.mul(LReg::L0, LReg::L1, LReg::L2),
                |p| p.sub(LReg::L1, LReg::L0, LReg::L2),
            )
        }),
        ("nested scopes, every condition", |p| {
            p.mov(LReg::L1, LReg::L2);
            p.if_(Cond::Ne0(LReg::L0), |p| {
                p.if_else(
                    Cond::Lt0(LReg::L1),
                    |p| {
                        p.if_(Cond::Eq0(LReg::L1), |p| p.loadi(LReg::L2, 7.0));
                        p.neg(LReg::L0, LReg::L2);
                    },
                    |p| p.add(LReg::L0, LReg::L0, LReg::L2),
                );
            });
        }),
    ]
}

#[test]
fn every_case_matches_the_interpreter_replayed_and_unrolled() {
    let (a, b) = (operands(3), operands(5));
    harness::in_device(|dev| {
        for (name, body) in cases() {
            let mut want: Option<Vec<u32>> = None;
            for policy in [LoopPolicy::Replay, LoopPolicy::Unrolled] {
                let p = program(policy, body);
                let form = p.loops()[0];
                if policy == LoopPolicy::Replay {
                    assert!(
                        matches!(form, LoopForm::Replayed { .. }),
                        "{name}: {form:?}"
                    );
                }
                let math = p.finish();
                let mut v = Vector::new();
                v.put_tile(0, &a);
                v.put_tile(64, &b);
                v.run(&math).unwrap_or_else(|e| panic!("{name}: {e}"));
                let model = v.tile(128);
                if let Some(w) = &want {
                    assert_eq!(&model, w, "{name}: the model differs replayed and unrolled");
                }
                let got = on_device(dev, &math, &a, &b);
                for (i, (g, w)) in got.iter().zip(&model).enumerate() {
                    assert_eq!(
                        g, w,
                        "{name} ({form:?}): datum {i}: device {g:#010x}, model {w:#010x} \
                         (a {:#010x}, b {:#010x})",
                        a[i], b[i]
                    );
                }
                want = Some(model);
            }
        }
    });
}

/// `LReg[8]`'s bits: documented only as "0.8373", so measured here.
#[test]
fn the_constant_register_holds_0_8373() {
    let (a, b) = (operands(1), operands(2));
    let mut p = Program::with_policy(LoopPolicy::Unrolled);
    p.for_each_row_group(64, |p, o| {
        p.mov(LReg::C0_8373, LReg::L2);
        p.store(LReg::L2, Format::Fp32, 128 + o);
    });
    let math = p.finish();
    harness::in_device(|dev| {
        let got = on_device(dev, &math, &a, &b);
        assert!(got.iter().all(|&x| x == got[0]), "every lane the same");
        println!("LReg[8] = {:#010x} = {}", got[0], f32::from_bits(got[0]));
        assert!((f32::from_bits(got[0]) - 0.8373).abs() < 1e-3);
    });
}
