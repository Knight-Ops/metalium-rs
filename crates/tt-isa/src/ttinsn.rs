//! Fused `.ttinsn` code: a role program as RISC-V code that pushes its Tensix words by
//! executing them (`docs/plans/ttinsn-push.md`).
//!
//! A runner pushes a staged program word by word, a load and a store a word (3.31 cycles on
//! silicon). `.ttinsn` is the same push as one instruction, the Tensix word rotated left two
//! bits and executed (`PushTensixInstruction.md:21`, 1.19 cycles a word measured): the core
//! rotates it back and stores it to its push address. A program's block repeats
//! (`crate::mailbox::loops`) become counted loops, so the code is about as long as the staged
//! program, not its expansion.
//!
//! The build-time instruction gate cannot see code written at run time, so [`validate`] is
//! that code's gate, and [`emit`] only makes what it accepts. Fused code is a closed form:
//!
//! * a Tensix word, rotated: low two bits not `0b11` (the compressed space these cores do not
//!   implement), and rotated back, an instruction of the table (`crate::isa`) with its fixed
//!   fields;
//! * a loop: `addi tN, x0, count` (`count` 1..=[`MAX_COUNT`]), its body, `addi tN, tN, -1`,
//!   `bne tN, x0, <the body's first word>`; `t0` for the outer loop and `t1` inside it, at
//!   most [`MAX_DEPTH`] deep. A body past `bne`'s reach ([`MAX_BACK`]) closes with
//!   `beq tN, x0, 8` and `jal x0, <the body's first word>` instead;
//! * `jalr x0, 0(ra)` ([`RET`]) as the last word, and nowhere else.
//!
//! [`execute`] runs fused code as the core would and hands each word it pushes to a closure:
//! the round trip against the staged program's own expansion is the oracle for both.

use crate::isa::generated::ALL;

/// In a program's length word (`crate::mailbox::PROGRAM_LEN`, or a streamed batch's): the words
/// are fused code, which the math role's runner calls after invalidating its instruction cache,
/// instead of pushing them. Never with `crate::mailbox::loops::LOOPED`: the loops are in the
/// code.
pub const FUSED: u32 = 1 << 29;

/// `x0`.
const ZERO: u32 = 0;
/// `t0`: the outer loop's counter. Caller-saved, so the code needs no stack.
pub const T0: u32 = 5;
/// `t1`: the inner loop's counter.
pub const T1: u32 = 6;
/// The counter register at each loop depth.
pub const COUNTERS: [u32; MAX_DEPTH] = [T0, T1];
/// `jalr x0, 0(ra)`.
pub const RET: u32 = 0x0000_8067;
/// Most iterations of one loop: `addi`'s signed 12-bit immediate.
pub const MAX_COUNT: u32 = 2047;
/// Loops nested at most this deep (`crate::mailbox::loops`: two).
pub const MAX_DEPTH: usize = 2;
/// Farthest a `bne` reaches back, in bytes.
pub const MAX_BACK: u32 = 4096;
/// Farthest a `jal` reaches back, in bytes: a far loop's close.
pub const MAX_JUMP: u32 = 1 << 20;

/// `addi rd, rs1, imm`.
pub const fn addi(rd: u32, rs1: u32, imm: i32) -> u32 {
    (((imm as u32) & 0xfff) << 20) | (rs1 << 15) | (rd << 7) | 0x13
}

/// `bne rs, x0, offset`: a B-type branch, `offset` in bytes, even, within +-4 KiB.
pub const fn bne(rs: u32, offset: i32) -> u32 {
    let imm = offset as u32;
    ((imm >> 12 & 1) << 31)
        | ((imm >> 5 & 0x3f) << 25)
        | (rs << 15)
        | (0b001 << 12)
        | ((imm >> 1 & 0xf) << 8)
        | ((imm >> 11 & 1) << 7)
        | 0x63
}

/// `beq rs, x0, offset`.
pub const fn beq(rs: u32, offset: i32) -> u32 {
    bne(rs, offset) & !(0b111 << 12)
}

/// `jal x0, offset`: a J-type jump, `offset` in bytes, even, within +-1 MiB.
pub const fn jump(offset: i32) -> u32 {
    let imm = offset as u32;
    ((imm >> 20 & 1) << 31)
        | ((imm >> 1 & 0x3ff) << 21)
        | ((imm >> 11 & 1) << 20)
        | ((imm >> 12 & 0xff) << 12)
        | 0x6f
}

/// A Tensix word as `.ttinsn` code: rotated left two bits, or `None` for one at or above
/// `0xC000_0000`, which has no `.ttinsn` form.
pub const fn rotate(word: u32) -> Option<u32> {
    if word < 0xC000_0000 {
        Some(word.rotate_left(2))
    } else {
        None
    }
}

/// Whether `word` is an instruction of the table, fixed fields and all.
pub fn is_tensix(word: u32) -> bool {
    ALL.iter().any(|d| d.matches(word))
}

/// Why a program has no fused form, or fused code is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Staged word `index` has no `.ttinsn` form (at or above `0xC000_0000`: a seed directive,
    /// or data).
    NoForm { index: usize, word: u32 },
    /// Word `index` is not an instruction of the table.
    NotTensix { index: usize, word: u32 },
    /// A block repeat or loop nested past [`MAX_DEPTH`].
    Depth { index: usize },
    /// A count of 0 or past [`MAX_COUNT`].
    Count { index: usize, count: u32 },
    /// A loop's close does not reach back to its body's first word, or the body is empty.
    Branch { index: usize },
    /// The output has no room.
    Room,
    /// A block repeat that does not lie inside the program, or overlaps another.
    Loops,
    /// Word `index` is RISC-V, but not the skeleton's word at that place.
    Stray { index: usize, word: u32 },
    /// The code does not end in one [`RET`], with every loop closed.
    Unterminated,
}

/// Write `code`'s fused form to `out`, its block repeats `loops` (`(start, len, count)` as
/// `crate::mailbox::loops::decode` gives them) as counted loops, walked as the runner walks
/// them (`tt_firmware::corpus::Pusher::span`): the earliest repeat first, the longest on a tie.
/// Returns the words written. What it writes [`validate`] accepts.
pub fn emit(code: &[u32], loops: &[(u32, u32, u32)], out: &mut [u32]) -> Result<usize, Refusal> {
    let n = code.len() as u32;
    for (k, a) in loops.iter().enumerate() {
        if a.1 == 0 || a.0.checked_add(a.1).is_none_or(|end| end > n) {
            return Err(Refusal::Loops);
        }
        for b in &loops[..k] {
            let (ae, be) = (a.0 + a.1, b.0 + b.1);
            let nested = (a.0 <= b.0 && be <= ae) || (b.0 <= a.0 && ae <= be);
            if (a.0, a.1) == (b.0, b.1) || !(ae <= b.0 || be <= a.0 || nested) {
                return Err(Refusal::Loops);
            }
        }
    }
    let mut w = Writer { out, len: 0 };
    walk(&mut w, code, loops, (0, n), 0)?;
    w.push(RET)?;
    Ok(w.len)
}

struct Writer<'a> {
    out: &'a mut [u32],
    len: usize,
}

impl Writer<'_> {
    fn push(&mut self, word: u32) -> Result<(), Refusal> {
        *self.out.get_mut(self.len).ok_or(Refusal::Room)? = word;
        self.len += 1;
        Ok(())
    }
}

fn walk(
    w: &mut Writer,
    code: &[u32],
    loops: &[(u32, u32, u32)],
    (lo, hi): (u32, u32),
    depth: usize,
) -> Result<(), Refusal> {
    let mut at = lo;
    loop {
        let mut next: Option<(u32, u32, u32)> = None;
        for &l in loops {
            let inside = l.0 >= at && l.0 + l.1 <= hi && (l.0, l.0 + l.1) != (lo, hi);
            if inside && next.is_none_or(|n| l.0 < n.0 || (l.0 == n.0 && l.1 > n.1)) {
                next = Some(l);
            }
        }
        let Some((start, len, count)) = next else {
            return words(w, code, at, hi);
        };
        words(w, code, at, start)?;
        let index = start as usize;
        if depth >= MAX_DEPTH {
            return Err(Refusal::Depth { index });
        }
        if count == 0 || count > MAX_COUNT {
            return Err(Refusal::Count { index, count });
        }
        let reg = COUNTERS[depth];
        w.push(addi(reg, ZERO, count as i32))?;
        let body = w.len;
        walk(w, code, loops, (start, start + len), depth + 1)?;
        w.push(addi(reg, reg, -1))?;
        // The branch is the next word, back to the body's first; past its reach, a jump over
        // which a branch on zero leaves the loop.
        let back = 4 * (w.len - body) as u32;
        if back <= MAX_BACK {
            w.push(bne(reg, -(back as i32)))?;
        } else if back + 4 <= MAX_JUMP {
            w.push(beq(reg, 8))?;
            w.push(jump(-(back as i32 + 4)))?;
        } else {
            return Err(Refusal::Branch { index });
        }
        at = start + len;
    }
}

fn words(w: &mut Writer, code: &[u32], lo: u32, hi: u32) -> Result<(), Refusal> {
    for (index, &word) in code.iter().enumerate().take(hi as usize).skip(lo as usize) {
        let fused = rotate(word).ok_or(Refusal::NoForm { index, word })?;
        if !is_tensix(word) {
            return Err(Refusal::NotTensix { index, word });
        }
        w.push(fused)?;
    }
    Ok(())
}

/// What one word of fused code is, read on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Word {
    /// A Tensix word, rotated back.
    Push(u32),
    /// `addi t, x0, count`.
    Set {
        reg: u32,
        count: u32,
    },
    /// `addi t, t, -1`.
    Decrement {
        reg: u32,
    },
    /// `bne t, x0, offset`.
    Branch {
        reg: u32,
        offset: i32,
    },
    /// `beq t, x0, 8`: a far loop's exit, over its jump.
    Exit {
        reg: u32,
    },
    /// `jal x0, offset`.
    Jump {
        offset: i32,
    },
    Ret,
    Other,
}

fn read(word: u32) -> Word {
    if word & 3 != 3 {
        return Word::Push(word.rotate_right(2));
    }
    if word == RET {
        return Word::Ret;
    }
    let (opcode, rd, funct3, rs1, rs2) = (
        word & 0x7f,
        word >> 7 & 0x1f,
        word >> 12 & 7,
        word >> 15 & 0x1f,
        word >> 20 & 0x1f,
    );
    let counter = |r: u32| COUNTERS.contains(&r);
    match (opcode, funct3) {
        (0x13, 0) => {
            let imm = (word as i32) >> 20;
            if rs1 == ZERO && counter(rd) && (1..=MAX_COUNT as i32).contains(&imm) {
                Word::Set {
                    reg: rd,
                    count: imm as u32,
                }
            } else if rs1 == rd && counter(rd) && imm == -1 {
                Word::Decrement { reg: rd }
            } else {
                Word::Other
            }
        }
        (0x63, 0 | 1) if rs2 == ZERO && counter(rs1) => {
            let imm = ((word >> 31 & 1) << 12)
                | ((word >> 7 & 1) << 11)
                | ((word >> 25 & 0x3f) << 5)
                | ((word >> 8 & 0xf) << 1);
            // Sign-extend the 13-bit offset.
            let offset = ((imm << 19) as i32) >> 19;
            match funct3 {
                1 => Word::Branch { reg: rs1, offset },
                _ if offset == 8 => Word::Exit { reg: rs1 },
                _ => Word::Other,
            }
        }
        (0x6f, _) if rd == ZERO => {
            let imm = ((word >> 31 & 1) << 20)
                | ((word >> 12 & 0xff) << 12)
                | ((word >> 20 & 1) << 11)
                | ((word >> 21 & 0x3ff) << 1);
            // Sign-extend the 21-bit offset.
            let offset = ((imm << 11) as i32) >> 11;
            Word::Jump { offset }
        }
        _ => Word::Other,
    }
}

/// Check that `fused` is fused code of the closed form the module describes, word by word and
/// independently of [`emit`]: every word is a table instruction or the loop skeleton's word at
/// its place, and the code ends in its one return. Returns the most words it pushes.
pub fn validate(fused: &[u32]) -> Result<u64, Refusal> {
    // Open loops: (counter, the body's first word, its count, words pushed before it).
    let mut open: [(u32, usize, u32, u64); MAX_DEPTH] = [(0, 0, 0, 0); MAX_DEPTH];
    let mut depth = 0;
    let mut pushed: u64 = 0;
    let mut index = 0;
    while index < fused.len() {
        let word = fused[index];
        match read(word) {
            Word::Push(w) => {
                if !is_tensix(w) {
                    return Err(Refusal::NotTensix { index, word: w });
                }
                pushed += 1;
            }
            Word::Set { reg, count } => {
                if depth == MAX_DEPTH {
                    return Err(Refusal::Depth { index });
                }
                if reg != COUNTERS[depth] {
                    return Err(Refusal::Stray { index, word });
                }
                open[depth] = (reg, index + 1, count, pushed);
                pushed = 0;
                depth += 1;
            }
            Word::Decrement { reg } => {
                let Some(d) = depth.checked_sub(1) else {
                    return Err(Refusal::Stray { index, word });
                };
                let (counter, body, count, before) = open[d];
                if reg != counter {
                    return Err(Refusal::Stray { index, word });
                }
                // The branch follows, back to the body's first word: a `bne` where it reaches,
                // else a `beq` over a `jal`.
                let back = 4 * (index + 1 - body) as i64;
                let next = fused.get(index + 1).copied().map(read);
                if body == index {
                    return Err(Refusal::Branch { index });
                }
                if back <= MAX_BACK as i64 {
                    if next
                        != Some(Word::Branch {
                            reg,
                            offset: -back as i32,
                        })
                    {
                        return Err(Refusal::Branch { index: index + 1 });
                    }
                    index += 1;
                } else {
                    let far = fused.get(index + 2).copied().map(read);
                    if next != Some(Word::Exit { reg })
                        || back + 4 > MAX_JUMP as i64
                        || far
                            != Some(Word::Jump {
                                offset: -(back + 4) as i32,
                            })
                    {
                        return Err(Refusal::Branch { index: index + 1 });
                    }
                    index += 2;
                }
                pushed = before + pushed * count as u64;
                depth = d;
            }
            Word::Ret if index + 1 == fused.len() && depth == 0 => return Ok(pushed),
            Word::Ret => return Err(Refusal::Unterminated),
            Word::Branch { .. } | Word::Exit { .. } | Word::Jump { .. } | Word::Other => {
                return Err(Refusal::Stray { index, word })
            }
        }
        index += 1;
    }
    Err(Refusal::Unterminated)
}

/// Run `fused` as the core would (`addi`, the branches, the jump and the return), handing each Tensix word it
/// pushes, rotated back, to `push`, in order. Validates first.
pub fn execute(fused: &[u32], mut push: impl FnMut(u32)) -> Result<(), Refusal> {
    validate(fused)?;
    let mut regs = [0u32; 32];
    let mut pc = 0usize;
    loop {
        let word = fused[pc];
        match read(word) {
            Word::Push(w) => push(w),
            Word::Set { reg, count } => regs[reg as usize] = count,
            Word::Decrement { reg } => regs[reg as usize] = regs[reg as usize].wrapping_sub(1),
            Word::Branch { reg, offset } => {
                if regs[reg as usize] != 0 {
                    pc = (pc as i64 + offset as i64 / 4) as usize;
                    continue;
                }
            }
            Word::Exit { reg } => {
                if regs[reg as usize] == 0 {
                    pc += 2;
                    continue;
                }
            }
            Word::Jump { offset } => {
                pc = (pc as i64 + offset as i64 / 4) as usize;
                continue;
            }
            Word::Ret => return Ok(()),
            Word::Other => unreachable!("validated"),
        }
        pc += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::generated::defs;
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    /// Distinct table words: a NOP and a few `SFPNOP`-like words from the table's own encoders.
    fn tensix(k: u32) -> u32 {
        // Every definition encodes its fixed fields with zero operands; vary by picking defs.
        let below: Vec<u32> = ALL
            .iter()
            .map(|d| d.skeleton())
            .filter(|&w| w < 0xC000_0000)
            .collect();
        below[k as usize % below.len()]
    }

    /// What the runner pushes for `code` with block repeats `loops` (the walk of `emit`, but
    /// expanding instead of looping).
    fn expand(code: &[u32], loops: &[(u32, u32, u32)], (lo, hi): (u32, u32), out: &mut Vec<u32>) {
        let mut at = lo;
        loop {
            let mut next: Option<(u32, u32, u32)> = None;
            for &l in loops {
                let inside = l.0 >= at && l.0 + l.1 <= hi && (l.0, l.0 + l.1) != (lo, hi);
                if inside && next.is_none_or(|n| l.0 < n.0 || (l.0 == n.0 && l.1 > n.1)) {
                    next = Some(l);
                }
            }
            let Some((start, len, count)) = next else {
                out.extend_from_slice(&code[at as usize..hi as usize]);
                return;
            };
            out.extend_from_slice(&code[at as usize..start as usize]);
            for _ in 0..count {
                expand(code, loops, (start, start + len), out);
            }
            at = start + len;
        }
    }

    fn fuse(code: &[u32], loops: &[(u32, u32, u32)]) -> Result<Vec<u32>, Refusal> {
        let mut out = vec![0; code.len() + 4 * loops.len() + 1];
        let n = emit(code, loops, &mut out)?;
        out.truncate(n);
        Ok(out)
    }

    fn pushes(fused: &[u32]) -> Result<Vec<u32>, Refusal> {
        let mut v = Vec::new();
        execute(fused, |w| v.push(w))?;
        Ok(v)
    }

    fn program(n: u32) -> Vec<u32> {
        (0..n).map(tensix).collect()
    }

    #[test]
    fn the_encoders_are_the_assemblers() {
        // `addi t0, x0, 32`, `addi t1, t1, -1`, `bne t0, x0, -8` as GNU as encodes them.
        assert_eq!(addi(T0, ZERO, 32), 0x0200_0293);
        assert_eq!(addi(T1, T1, -1), 0xfff3_0313);
        assert_eq!(bne(T0, -8), 0xfe02_9ce3);
        assert_eq!(
            read(bne(T1, -4096)),
            Word::Branch {
                reg: T1,
                offset: -4096
            }
        );
        assert_eq!(RET, 0x0000_8067);
        // `beq t0, x0, 8` and `jal x0, -8192`.
        assert_eq!(beq(T0, 8), 0x0002_8463);
        assert_eq!(jump(-8192), 0x800f_e06f);
        assert_eq!(read(jump(-8192)), Word::Jump { offset: -8192 });
        let far = -(MAX_JUMP as i32);
        assert_eq!(read(jump(far)), Word::Jump { offset: far });
    }

    #[test]
    fn fused_code_pushes_what_the_runner_pushes() {
        let code = program(40);
        let cases: [&[(u32, u32, u32)]; 5] = [
            &[],
            &[(4, 10, 3)],
            &[(1, 39, 2), (5, 6, 7)],
            &[(2, 3, 5), (10, 20, 4), (12, 4, 9)],
            &[(1, 38, 128), (1, 37, 2)],
        ];
        for loops in cases {
            let fused = fuse(&code, loops).unwrap();
            let mut want = Vec::new();
            expand(&code, loops, (0, 40), &mut want);
            assert_eq!(pushes(&fused).unwrap(), want, "{loops:?}");
            assert_eq!(validate(&fused).unwrap(), want.len() as u64);
            // Stored once, loops a few words each, not expanded.
            assert_eq!(fused.len(), code.len() + 3 * loops.len() + 1);
        }
        // As the runner does, a repeat spanning exactly the range being walked is not a loop of
        // it (`Pusher::span`): the whole program, here, is pushed once.
        let fused = fuse(&code, &[(0, 40, 3)]).unwrap();
        assert_eq!(pushes(&fused).unwrap(), code);
    }

    #[test]
    fn what_has_no_fused_form_is_refused() {
        let mut code = program(8);
        // A seed directive (`crate::dataflow::SEED_DIRECTIVE`) is not a Tensix word.
        code[0] = crate::dataflow::SEED_DIRECTIVE;
        assert!(matches!(
            fuse(&code, &[]),
            Err(Refusal::NoForm { index: 0, .. })
        ));
        let mut code = program(8);
        code[3] = 0x0f00_0000 | 0x00ab_cdef; // an opcode byte no instruction has
        if !is_tensix(code[3]) {
            assert!(matches!(
                fuse(&code, &[]),
                Err(Refusal::NotTensix { index: 3, .. })
            ));
        }
        let code = program(8);
        assert_eq!(
            fuse(&code, &[(0, 7, 2), (1, 5, 2), (2, 2, 2)]),
            Err(Refusal::Depth { index: 2 })
        );
        assert_eq!(
            fuse(&code, &[(0, 4, 0)]),
            Err(Refusal::Count { index: 0, count: 0 })
        );
        assert_eq!(
            fuse(&code, &[(0, 4, 2048)]).unwrap_err(),
            Refusal::Count {
                index: 0,
                count: 2048
            }
        );
        assert_eq!(fuse(&code, &[(6, 4, 2)]), Err(Refusal::Loops));
        assert_eq!(fuse(&code, &[(0, 4, 2), (2, 4, 2)]), Err(Refusal::Loops));
        let long = program(1100);
        // 1,023 words is the longest body a `bne` reaches back over (its decrement included);
        // a longer one closes with a `beq` over a `jal`, one word more.
        for (len, close) in [(1023, 1), (1024, 2), (1099, 2)] {
            let fused = fuse(&long, &[(1, len, 3)]).unwrap();
            assert_eq!(fused.len(), 1100 + 2 + close + 1, "{len}");
            let mut want = Vec::new();
            expand(&long, &[(1, len, 3)], (0, 1100), &mut want);
            assert_eq!(pushes(&fused).unwrap(), want, "{len}");
        }
        let mut out = [0u32; 4];
        assert_eq!(emit(&code, &[], &mut out), Err(Refusal::Room));
    }

    /// Each mutation of good fused code is refused by `validate`: the gate's negative controls.
    #[test]
    fn the_validator_refuses_every_mutation() {
        let code = program(14);
        let good = fuse(&code, &[(1, 12, 3), (3, 4, 5)]).unwrap();
        validate(&good).unwrap();
        // [w, set t0, 2 words, set t1, 4 words, dec t1, bne t1, 6 words, dec t0, bne t0, w, ret]
        let set0 = 1;
        let set1 = 4;
        let dec1 = 9;
        let bne1 = 10;
        let dec0 = 17;
        let bne0 = 18;
        assert_eq!(good.len(), 21);
        assert_eq!(read(good[set1]), Word::Set { reg: T1, count: 5 });
        let refused = |what: &str, f: &dyn Fn(&mut Vec<u32>)| {
            let mut m = good.clone();
            f(&mut m);
            assert!(validate(&m).is_err(), "{what} was accepted");
            assert!(execute(&m, |_| {}).is_err(), "{what} was executed");
        };
        refused("a word with low bits 11 that is no skeleton word", &|m| {
            m[1] = 0x0000_0013
        });
        refused("a stray RISC-V instruction (lw)", &|m| m[2] = 0x0002_a283);
        refused("a non-table Tensix word", &|m| {
            m[5] = rotate(0x3f00_0000).unwrap()
        });
        assert_eq!(good[bne1], bne(T1, -5 * 4));
        assert_eq!(good[bne0], bne(T0, -16 * 4));
        refused("a branch one word short", &|m| m[bne1] = bne(T1, -4 * 4));
        refused("a branch one word long", &|m| m[bne1] = bne(T1, -6 * 4));
        refused("a branch on the wrong counter", &|m| {
            m[bne0] = bne(T1, -16 * 4)
        });
        refused("a foreign counter register", &|m| {
            m[set0] = addi(7, ZERO, 3)
        });
        refused("the inner loop counting on t0", &|m| {
            m[set1] = addi(T0, ZERO, 5)
        });
        refused("a decrement by two", &|m| m[dec1] = addi(T1, T1, -2));
        refused("a count of zero", &|m| m[set0] = addi(T0, ZERO, 0));
        refused("a missing return", &|m| {
            m.pop();
        });
        refused("a return mid-code", &|m| m[6] = RET);
        refused("an unclosed loop", &|m| {
            m.remove(dec0);
            m.remove(dec0);
        });
        refused("a third loop level", &|m| {
            m.insert(dec1 - 1, addi(T0, T0, -1));
        });
        refused("a far close where a branch reaches", &|m| {
            m[bne1] = beq(T1, 8);
            m.insert(bne1 + 1, jump(-6 * 4));
        });
        // Each control did change something the validator saw; the original still passes.
        validate(&good).unwrap();
    }

    #[test]
    fn every_table_instruction_round_trips() {
        for d in ALL {
            let w = d.skeleton();
            if let Some(f) = rotate(w) {
                assert_eq!(read(f), Word::Push(w), "{}", d.key());
                assert!(is_tensix(w), "{}", d.key());
            }
        }
        // `REPLAY` has a fused form (it is pushed like any other word).
        assert!(rotate(defs::REPLAY.skeleton()).is_some());
    }
}

#[cfg(test)]
mod far {
    use super::*;
    extern crate std;
    use std::vec;

    /// A far loop's close: each mutation refused.
    #[test]
    fn a_far_close_is_checked_word_by_word() {
        let nop = rotate(crate::isa::generated::defs::NOP.skeleton()).unwrap();
        let mut good = vec![addi(T0, 0, 2)];
        good.extend(core::iter::repeat_n(nop, 1100));
        let back: i32 = 4 * (1100 + 1);
        good.extend([addi(T0, T0, -1), beq(T0, 8), jump(-(back + 4)), RET]);
        assert_eq!(validate(&good), Ok(2200));
        let mut n = 0;
        execute(&good, |_| n += 1).unwrap();
        assert_eq!(n, 2200);
        let at = good.len() - 3;
        for (what, i, w) in [
            ("a jump one word short", at + 1, jump(-back)),
            ("an exit on the other counter", at, beq(T1, 8)),
            ("an exit past the jump", at, beq(T0, 12)),
            ("a bne where the jump is needed", at, bne(T0, -back)),
        ] {
            let mut m = good.clone();
            m[i] = w;
            assert!(validate(&m).is_err(), "{what} was accepted");
        }
    }
}
