//! Inclusive scans in increasing logical row order. A continuation reads the
//! preceding output tile's last row, never a folded partial reduction.
//!
//! Three families share one tile program:
//!
//! - FP32 `Sum`/`Prod` (the Blackhole `SFPMAD` arithmetic) and the raw
//!   total-order `Min`/`Max` of step79;
//! - `MinNan`/`MaxNan`, Flex's `cummin`/`cummax`: `if val.is_nan() || val <
//!   acc { val } else { acc }` (`>` for the maximum) over IEEE comparisons, so
//!   a NaN replaces the accumulator, a NaN accumulator is only replaced by
//!   another NaN (the later NaN's bits win), and `-0 == +0` keeps the earlier
//!   element;
//! - the wrapping 32-bit integer scans `ISum`, `IProd`, `IMin`, `IMax`
//!   (identities 0, 1, `i32::MAX`, `i32::MIN`), on raw two's-complement words.
use super::kernel::{A_ROW, B_ROW, OUT_ROW};
use super::{Cond, Format, LReg, Program};
use tt_isa::isa::Instruction;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScanOp {
    Sum,
    Prod,
    /// Raw FP32 total order: -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN.
    Min,
    Max,
    /// Flex's `cummin`: NaN-propagating IEEE minimum, the earlier element
    /// kept on equal values (`+0` and `-0` included). Identity `+inf`.
    MinNan,
    /// Flex's `cummax`, as [`ScanOp::MinNan`]. Identity `-inf`.
    MaxNan,
    /// Wrapping two's-complement sum of I32 words.
    ISum,
    /// Wrapping low-32-bit product of I32 words.
    IProd,
    /// Signed I32 minimum.
    IMin,
    /// Signed I32 maximum.
    IMax,
}

impl ScanOp {
    /// The physical element type a scan of this kind consumes and produces.
    pub fn elem(self) -> crate::tensor::Elem {
        match self {
            ScanOp::ISum | ScanOp::IProd | ScanOp::IMin | ScanOp::IMax => crate::tensor::Elem::I32,
            _ => crate::tensor::Elem::F32,
        }
    }
}

/// One tile, with an optional preceding output tile in B. The four rows
/// loaded together are transposed into L0..L3, visited in order and then
/// transposed back. L4's row zero survives both transposes (the diagonal).
pub fn program(op: ScanOp, first: bool) -> Vec<Instruction> {
    let mut p = Program::new();
    for (top, bottom) in [(0, 32), (16, 48)] {
        for half in [0, 2] {
            if first {
                p.loadi_bits(LReg::L4, identity(op));
            } else {
                p.load(LReg::L0, Format::Int32, B_ROW + bottom + 12 + half);
                p.transpose4();
                p.mov(LReg::L3, LReg::L4);
            }
            for g in 0..8 {
                let row = if g < 4 {
                    top + 4 * g
                } else {
                    bottom + 4 * (g - 4)
                } + half;
                p.load(LReg::L0, Format::Int32, A_ROW + row);
                p.transpose4();
                for reg in [LReg::L0, LReg::L1, LReg::L2, LReg::L3] {
                    step(&mut p, op, reg);
                    p.mov(LReg::L4, reg);
                }
                p.transpose4();
                p.store(LReg::L0, Format::Int32, OUT_ROW + row);
            }
        }
    }
    p.finish()
}

/// `L4 = op(L4, reg)`. `reg` and `L5..L7` may be clobbered; the other three
/// of `L0..L3` and `L4` are the scan's state. Only lanes 0..8 of `L4` are
/// live: a transpose moves the rest.
fn step(p: &mut Program, op: ScanOp, reg: LReg) {
    const L4: LReg = LReg::L4;
    match op {
        ScanOp::Sum => p.add(L4, reg, L4),
        ScanOp::Prod => p.mul(L4, reg, L4),
        ScanOp::Min => p.if_(Cond::Less(reg, L4), |p| p.mov(reg, L4)),
        ScanOp::Max => p.if_(Cond::Less(L4, reg), |p| p.mov(reg, L4)),
        ScanOp::MinNan | ScanOp::MaxNan => {
            let minimum = op == ScanOp::MinNan;
            let (r, a, t) = (LReg::L5, LReg::L6, LReg::L7);
            // Zeros compare equal in IEEE: both copies become +0 when their
            // magnitude is zero, so the total order of the copies is the IEEE
            // order of their non-NaN values. What is stored is always `reg`'s
            // own bits.
            p.mov(reg, r);
            p.mov(L4, a);
            for x in [r, a] {
                p.shl(x, 1, t);
                p.if_(Cond::Eq0(t), |p| p.mov(LReg::ZERO, x));
            }
            // A `+NaN` replaces the accumulator.
            p.loadi_bits(t, 0x7f80_0000);
            p.if_(Cond::Less(t, r), |p| p.mov(reg, L4));
            if minimum {
                // A non-NaN value replaces a non-NaN accumulator it is below.
                // `a <= +inf` excludes a `+NaN` accumulator; a `-NaN` one is
                // below every non-NaN `reg` in the total order already.
                p.if_(Cond::LessEq(a, t), |p| {
                    p.if_(Cond::Less(r, a), |p| p.mov(reg, L4))
                });
            }
            // A `-NaN` replaces the accumulator.
            p.loadi_bits(t, 0xff80_0000);
            p.if_(Cond::Less(r, t), |p| p.mov(reg, L4));
            if !minimum {
                // `-inf <= a` excludes a `-NaN` accumulator; a `+NaN` one is
                // above every non-NaN `reg` already.
                p.if_(Cond::LessEq(t, a), |p| {
                    p.if_(Cond::Less(a, r), |p| p.mov(reg, L4))
                });
            }
        }
        ScanOp::ISum => p.iadd(reg, L4),
        ScanOp::IProd => {
            // a*b mod 2^32 = alo*blo + ((ahi*blo + alo*bhi) << 16), as
            // `integer::body` computes it, with `a` in `L4` and `b` in `reg`
            // (both consumed). SFPMUL24 returns 23 bits, so the full 16x16
            // low product is rebuilt from its LOW and UPPER modes.
            let (alo, blo, k) = (LReg::L6, LReg::L7, LReg::L5);
            p.loadi_bits(k, 0xffff);
            p.and(L4, k, alo);
            p.and(reg, k, blo);
            p.loadi_bits(k, (-16i32) as u32);
            p.shr_by(k, L4);
            p.shr_by(k, reg);
            p.mul24(L4, blo, false, k);
            p.shl(k, 16, k);
            p.mul24(alo, reg, false, L4);
            p.shl(L4, 16, L4);
            p.iadd(k, L4);
            p.mul24(alo, blo, false, reg);
            p.mul24(alo, blo, true, k);
            p.shl(k, 23, k);
            p.or(k, reg, reg);
            p.iadd(reg, L4);
        }
        ScanOp::IMin | ScanOp::IMax => {
            // Embed signed order in `SFPGT`'s sign-magnitude order by
            // inverting the magnitude of the negative words (`integer::body`).
            let (r, a, m) = (LReg::L5, LReg::L6, LReg::L7);
            p.loadi_bits(m, 0x7fff_ffff);
            p.mov(reg, r);
            p.mov(L4, a);
            p.if_(Cond::Lt0(r), |p| p.xor(m, r));
            p.if_(Cond::Lt0(a), |p| p.xor(m, a));
            let cond = if op == ScanOp::IMin {
                Cond::Less(r, a)
            } else {
                Cond::Less(a, r)
            };
            p.if_(cond, |p| p.mov(reg, L4));
        }
    }
}

fn identity(op: ScanOp) -> u32 {
    match op {
        ScanOp::Sum | ScanOp::ISum => 0,
        ScanOp::Prod => 0x3f800000,
        ScanOp::IProd => 1,
        ScanOp::Min => 0x7fffffff,
        ScanOp::Max => 0xffffffff,
        ScanOp::MinNan => 0x7f80_0000,
        ScanOp::MaxNan => 0xff80_0000,
        ScanOp::IMin => i32::MAX as u32,
        ScanOp::IMax => i32::MIN as u32,
    }
}

/// Independent scalar execution order, using the specified Blackhole arithmetic.
pub fn reference(op: ScanOp, input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let bits: Vec<u32> = input.iter().map(|x| x.to_bits()).collect();
    reference_bits(op, &bits, rows, cols)
        .into_iter()
        .map(f32::from_bits)
        .collect()
}

/// [`reference`] on raw words: FP32 bit patterns for the FP32 operations
/// (NaN payloads and signs are kept as words), two's-complement words for the
/// integer ones.
pub fn reference_bits(op: ScanOp, input: &[u32], rows: usize, cols: usize) -> Vec<u32> {
    let mut out = input.to_vec();
    for c in 0..cols {
        let mut acc = identity(op);
        for r in 0..rows {
            let x = input[r * cols + c];
            acc = match op {
                ScanOp::Sum => tt_isa::numerics::fma_bh(1.0f32.to_bits(), x, acc),
                ScanOp::Prod => tt_isa::numerics::fma_bh(acc, x, 0x80000000),
                ScanOp::Min | ScanOp::Max => {
                    let order = f32::from_bits(x).total_cmp(&f32::from_bits(acc));
                    if (op == ScanOp::Min && order.is_lt()) || (op == ScanOp::Max && order.is_gt())
                    {
                        x
                    } else {
                        acc
                    }
                }
                ScanOp::MinNan | ScanOp::MaxNan => {
                    let (v, a) = (f32::from_bits(x), f32::from_bits(acc));
                    let take = v.is_nan() || if op == ScanOp::MinNan { v < a } else { v > a };
                    if take {
                        x
                    } else {
                        acc
                    }
                }
                ScanOp::ISum => (acc as i32).wrapping_add(x as i32) as u32,
                ScanOp::IProd => (acc as i32).wrapping_mul(x as i32) as u32,
                ScanOp::IMin => (acc as i32).min(x as i32) as u32,
                ScanOp::IMax => (acc as i32).max(x as i32) as u32,
            };
            out[r * cols + c] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfpu::{interp::Vector, kernel};

    /// Word `(r, c)` of a row-major 32x32 tile in face order, as `Dst` and
    /// the interpreter hold it.
    fn face(r: usize, c: usize) -> usize {
        (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16
    }

    fn to_faces(rows: &[u32]) -> Vec<u32> {
        let mut out = vec![0; 1024];
        for r in 0..32 {
            for c in 0..32 {
                out[face(r, c)] = rows[r * 32 + c];
            }
        }
        out
    }

    fn from_faces(faces: &[u32]) -> Vec<u32> {
        (0..1024).map(|i| faces[face(i / 32, i % 32)]).collect()
    }

    /// Run the tile program in the interpreter over a row-major 32x32 tile
    /// (`prev` is the preceding output tile for a continuation).
    fn run(op: ScanOp, tile: &[u32], prev: Option<&[u32]>) -> Vec<u32> {
        let mut v = Vector::new();
        v.put_tile(kernel::A_ROW as usize, &to_faces(tile));
        if let Some(prev) = prev {
            v.put_tile(kernel::B_ROW as usize, &to_faces(prev));
        }
        v.run(&program(op, prev.is_none())).unwrap();
        from_faces(&v.tile(kernel::OUT_ROW as usize))
    }

    fn assert_words(got: &[u32], want: &[u32], what: &str) {
        let bad: Vec<_> = (0..want.len()).filter(|&i| got[i] != want[i]).collect();
        assert!(
            bad.is_empty(),
            "{what}: {} words differ, first at row {} col {}: got {:#010x}, want {:#010x}",
            bad.len(),
            bad[0] / 32,
            bad[0] % 32,
            got[bad[0]],
            want[bad[0]]
        );
    }

    fn operands(op: ScanOp) -> Vec<u32> {
        let special = [
            0u32,
            0x8000_0000,
            1,
            0x8000_0001,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc5_4321,
            0x7fff_ffff,
            0xffff_ffff,
            0x7f7f_ffff,
            0xff7f_ffff,
            1.0f32.to_bits(),
            (-1.0f32).to_bits(),
            2.5f32.to_bits(),
            (-0.25f32).to_bits(),
        ];
        let int = [
            0u32,
            1,
            0xffff_ffff,
            0x8000_0000,
            0x7fff_ffff,
            2,
            0xffff_fffe,
            0x0001_0001,
            0xffff_0000,
            0x0000_ffff,
            3,
            0xffff_fffd,
            0x4000_0000,
            0xc000_0000,
            12345,
            0xffff_cfc7,
        ];
        let table = if op.elem() == crate::tensor::Elem::I32 {
            &int
        } else {
            &special
        };
        (0..1024)
            .map(|i| table[(i / 32 * 5 + i % 32 * 3 + i / 7) % 16])
            .collect()
    }

    /// The program in the interpreter agrees with the scalar reference, a
    /// first tile and a continuation, bit for bit. The sums and products run
    /// on `SFPMAD` rounding that the reference follows (`fma_bh`).
    #[test]
    fn programs_match_the_scalar_references_bit_for_bit() {
        use ScanOp::*;
        for op in [Min, Max, MinNan, MaxNan, ISum, IProd, IMin, IMax] {
            let both = operands(op);
            let first = &both[..1024];
            let want_first = reference_bits(op, first, 32, 32);
            assert_words(
                &run(op, first, None),
                &want_first,
                &format!("{op:?} first tile"),
            );
            let second: Vec<u32> = both.iter().rev().copied().collect();
            // The second tile continues from the first's last row.
            let mut stacked = first.to_vec();
            stacked.extend_from_slice(&second);
            let want = reference_bits(op, &stacked, 64, 32);
            assert_words(
                &run(op, &second, Some(&want[..1024])),
                &want[1024..],
                &format!("{op:?} continuation"),
            );
        }
    }

    /// Many pseudo-random columns of specials (zeros of both signs, NaNs of
    /// both signs and payloads, infinities, extremes) against the reference.
    #[test]
    fn random_special_columns_match_the_scalar_references() {
        use ScanOp::*;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for op in [MinNan, MaxNan, IMin, IMax, ISum, IProd] {
            let table: Vec<u32> = if op.elem() == crate::tensor::Elem::I32 {
                vec![
                    0,
                    1,
                    u32::MAX,
                    i32::MIN as u32,
                    i32::MAX as u32,
                    2,
                    (-2i32) as u32,
                    65535,
                    65536,
                    (-65536i32) as u32,
                    0x1234_5678,
                    0xedcb_a988,
                ]
            } else {
                vec![
                    0,
                    0x8000_0000,
                    1,
                    0x8000_0001,
                    0x007f_ffff,
                    0x807f_ffff,
                    0x0080_0000,
                    0x7f80_0000,
                    0xff80_0000,
                    0x7f80_0001,
                    0x7fc0_0000,
                    0x7fc1_2345,
                    0xffc0_0000,
                    0xffc5_4321,
                    0xff80_0001,
                    0x7fff_ffff,
                    0xffff_ffff,
                    0x7f7f_ffff,
                    0xff7f_ffff,
                    1.0f32.to_bits(),
                    (-1.0f32).to_bits(),
                ]
            };
            for round in 0..24 {
                let tile: Vec<u32> = (0..1024)
                    .map(|_| {
                        let w = next();
                        if round % 3 == 2 && op.elem() == crate::tensor::Elem::I32 {
                            w.rotate_left(w & 31)
                        } else {
                            table[w as usize % table.len()]
                        }
                    })
                    .collect();
                let want = reference_bits(op, &tile, 32, 32);
                assert_words(
                    &run(op, &tile, None),
                    &want,
                    &format!("{op:?} round {round}"),
                );
            }
        }
    }

    /// Flex's rule, evaluated by hand on the cases the total order gets wrong.
    #[test]
    fn flex_semantics_on_zeros_and_nans() {
        let f = f32::to_bits;
        let nan = 0x7fc1_2345u32;
        let neg_nan = 0xffc5_4321u32;
        let col = |op, v: &[u32]| reference_bits(op, v, v.len(), 1);
        // Equal zeros keep the earlier one.
        assert_eq!(col(ScanOp::MinNan, &[f(0.0), f(-0.0)]), [f(0.0), f(0.0)]);
        assert_eq!(col(ScanOp::MinNan, &[f(-0.0), f(0.0)]), [f(-0.0), f(-0.0)]);
        assert_eq!(col(ScanOp::MaxNan, &[f(0.0), f(-0.0)]), [f(0.0), f(0.0)]);
        // A NaN replaces the accumulator and then sticks; a later NaN wins.
        assert_eq!(
            col(ScanOp::MinNan, &[f(1.0), nan, f(0.0), neg_nan, f(-3.0)]),
            [f(1.0), nan, nan, neg_nan, neg_nan]
        );
        assert_eq!(col(ScanOp::MaxNan, &[neg_nan, f(1.0)]), [neg_nan, neg_nan]);
        // The total-order scan differs on exactly these.
        assert_eq!(col(ScanOp::Min, &[f(0.0), f(-0.0)]), [f(0.0), f(-0.0)]);
    }

    /// Every scan program, unrolled over a tile, fits a role's program slot
    /// with room for the unpack and pack roles' own words (`MinNan` is the
    /// longest, about 4000 words of the 8192).
    #[test]
    fn every_program_fits_a_program_slot() {
        use ScanOp::*;
        for op in [Sum, Prod, Min, Max, MinNan, MaxNan, ISum, IProd, IMin, IMax] {
            for first in [true, false] {
                let words = program(op, first).len() as u32;
                assert!(
                    words < tt_isa::mailbox::PROGRAM_MAX / 2 + 256,
                    "{op:?} first={first}: {words} words"
                );
            }
        }
    }
}
