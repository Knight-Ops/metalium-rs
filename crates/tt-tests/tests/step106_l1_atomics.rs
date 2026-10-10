//! L1 atomics (`ATCAS`, `ATSWAP`, `ATINCGET`, `ATINCGETPTR`): independent models
//! of the Wormhole pages, ttsim refusal probes with surviving controls, and
//! silicon-only semantic gates.
//!
//! **UNVERIFIED on Blackhole.** The pinned documentation has Wormhole pages only
//! (`WormholeB0/TensixTile/TensixCoprocessor/AT{CAS,SWAP,INCGET,INCGETPTR}.md`),
//! the generated layouts are `WormholeOnly`, and ttsim refuses all four
//! (divergence row 77), so nothing below is evidence until the silicon gates
//! have run. The silicon request order is in `docs/completed-plans/hardware-coverage-closeout.md`:
//! `ATINCGET`, `ATSWAP`, `ATCAS` with its compare already met, then the blocking
//! forms with a producer on another thread and the deadline active.
//!
//! The models below are written from the page text and share nothing with
//! `tt_isa::scalar::atomic`.
#[cfg(feature = "silicon")]
use tt_isa::scalar::atomic::Region16;
use tt_isa::{backend, isa::Instruction, scalar::atomic};
#[cfg(feature = "silicon")]
use tt_isa::{backend::Before, sync};
use tt_tests::harness;

// ---------------------------------------------------------------------------
// Independent models.
// ---------------------------------------------------------------------------

/// `ATINCGET.md`: `IntMask = (2u << IntWidth) - 1` for the instruction's
/// `IntWidth` field, i.e. a field of `width = IntWidth + 1` bits;
/// `*L1Address = (Incremented & IntMask) | (OriginalValue & ~IntMask)` with
/// `Incremented = OriginalValue + IncrementBy` (wrapping 32-bit); the GPR
/// receives the *whole* original word. Returns `(new word, GPR result)`.
fn incget_model(word: u32, increment: u32, width: u32) -> (u32, u32) {
    assert!((1..=32).contains(&width));
    let mask = ((2u64 << (width - 1)) - 1) as u32;
    let incremented = word.wrapping_add(increment);
    ((incremented & mask) | (word & !mask), word)
}

/// `ATSWAP.md`: the 16-byte block takes, for each set bit `i` of `mask`,
/// halfword `i` of the data; the others are untouched. The data is the four
/// consecutive GPRs at `DataReg & 0x3c`. (The page's single-register form,
/// `SingleDataReg`, is excluded: its lanes on Blackhole follow no rule.)
fn swap_model(block: [u8; 16], mask: u8, data: [u8; 16]) -> [u8; 16] {
    let mut out = block;
    for i in 0..8 {
        if mask >> i & 1 == 1 {
            out[2 * i..2 * i + 2].copy_from_slice(&data[2 * i..2 * i + 2]);
        }
    }
    out
}

fn swap_data_group(group: [u32; 4]) -> [u8; 16] {
    let mut data = [0u8; 16];
    for (i, w) in group.iter().enumerate() {
        data[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    data
}

/// `ATCAS.md`: retry until the whole 32-bit word equals `CmpVal`, then store
/// `SetVal` over the whole word. `None` is "would retry".
fn cas_model(word: u32, compare: u32, set: u32) -> Option<u32> {
    (word == compare).then_some(set)
}

/// `ATINCGETPTR.md`, from the functional model: counters `Rd`, `Wr`; a push is
/// `Ofs` 1, a pop `Ofs` 0; `FIFOSize = Wr - Rd` (32-bit wrapping); empty is
/// size zero; full is `size % capacity == 0 && !empty`, with
/// `capacity = 1 << (IntWidth - 1)`; the counter advances by `1 << IncrLog2`
/// inside its low `IntWidth` bits; the GPR receives the counter before.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct FifoModel {
    width: u32,
    rd: u32,
    wr: u32,
}

impl FifoModel {
    fn capacity(self) -> u32 {
        1 << (self.width - 1)
    }
    fn mask(self) -> u32 {
        (1u32 << self.width) - 1
    }
    fn size(self) -> u32 {
        self.wr.wrapping_sub(self.rd)
    }
    fn empty(self) -> bool {
        self.size() == 0
    }
    fn full(self) -> bool {
        self.size() % self.capacity() == 0 && !self.empty()
    }
    /// What the instruction would do now, or `None` if it would retry.
    fn step(self, push: bool, advance: bool, log2: u32) -> Option<(FifoModel, u32)> {
        if push && self.full() || !push && self.empty() {
            return None;
        }
        let by = if advance { 1u32 << log2 } else { 0 };
        let mask = self.mask();
        let mut next = self;
        let old = if push { self.wr } else { self.rd };
        let new = (old.wrapping_add(by) & mask) | (old & !mask);
        if push {
            next.wr = new;
        } else {
            next.rd = new;
        }
        Some((next, old))
    }
}

#[test]
fn models_agree_with_hand_worked_page_examples() {
    // ATINCGET: a four-bit field wraps inside itself, upper bits preserved, the
    // GPR gets the whole original word.
    assert_eq!(incget_model(0xabcd_ef0f, 3, 4), (0xabcd_ef02, 0xabcd_ef0f));
    assert_eq!(incget_model(0xffff_ffff, 2, 32), (1, 0xffff_ffff));
    assert_eq!(incget_model(1, 1, 1), (0, 1));
    assert_eq!(
        incget_model(1, 2, 1),
        (1, 1),
        "an even increment leaves a one-bit field"
    );
    assert_eq!(incget_model(0x1_0000 - 1, 1, 16), (0, 0xffff));
    assert_eq!(incget_model(0xf000_ffff, 1, 16), (0xf000_0000, 0xf000_ffff));
    // ATSWAP: halfword 0 and 7 only.
    let block: [u8; 16] = core::array::from_fn(|i| i as u8);
    let data = [0xee; 16];
    let got = swap_model(block, 0b1000_0001, data);
    assert_eq!(&got[0..2], &[0xee, 0xee]);
    assert_eq!(&got[2..14], &block[2..14]);
    assert_eq!(&got[14..16], &[0xee, 0xee]);
    assert_eq!(swap_model(block, 0, data), block);
    // ATCAS: equal or retry; the whole word is compared and replaced.
    assert_eq!(cas_model(3, 3, 9), Some(9));
    assert_eq!(cas_model(0x13, 3, 9), None, "upper bits make it unequal");
    // ATINCGETPTR, width 3 (capacity 4, counters 0..8).
    let f = FifoModel {
        width: 3,
        rd: 0,
        wr: 0,
    };
    assert!(f.empty() && !f.full());
    assert_eq!(f.step(false, true, 0), None, "pop of empty retries");
    let (f, old) = f.step(true, true, 0).unwrap();
    assert_eq!((f.wr, old), (1, 0));
    let full = FifoModel {
        width: 3,
        rd: 0,
        wr: 4,
    };
    assert!(full.full());
    assert_eq!(full.step(true, true, 0), None, "push of full retries");
    let wrapped_full = FifoModel {
        width: 3,
        rd: 4,
        wr: 0,
    };
    assert!(
        wrapped_full.full(),
        "0 - 4 wraps to a multiple of the capacity"
    );
    // Occupancy 7 of 4 is not "full" to the instruction (7 % 4 != 0): why the
    // host refuses counters that describe more than the capacity.
    assert!(!FifoModel {
        width: 3,
        rd: 0,
        wr: 7
    }
    .full());
    let (f, old) = FifoModel {
        width: 3,
        rd: 0,
        wr: 7,
    }
    .step(false, true, 0)
    .unwrap();
    assert_eq!((f.rd, old), (1, 0));
    // The counter wraps inside its width, high bits untouched.
    let (f, old) = FifoModel {
        width: 3,
        rd: 7,
        wr: 7,
    }
    .step(true, true, 0)
    .unwrap();
    assert_eq!((f.wr, old), (0, 7));
    assert_eq!(swap_data_group([1, 2, 3, 4])[4..8], [2, 0, 0, 0]);
    // A batch of two.
    let (f, old) = FifoModel {
        width: 4,
        rd: 0,
        wr: 6,
    }
    .step(true, true, 1)
    .unwrap();
    assert_eq!((f.wr, old), (8, 6));
}

#[test]
fn a_wrong_width_or_mask_changes_the_model() {
    // The data mutants the silicon gates use.
    assert_ne!(
        incget_model(0xabcd_ef0f, 3, 4),
        incget_model(0xabcd_ef0f, 3, 5)
    );
    assert_ne!(
        incget_model(0xabcd_ef0f, 3, 4).0,
        incget_model(0xabcd_ef0f, 3, 32).0
    );
    let block = [0x11; 16];
    let data = [0x22; 16];
    assert_ne!(
        swap_model(block, 0b0101_0101, data),
        swap_model(block, 0b0101_0100, data)
    );
}

// ---------------------------------------------------------------------------
// ttsim refusal probes (divergence row 77), with surviving controls.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "silicon"))]
fn atoms() -> Vec<(&'static str, Instruction)> {
    use atomic::{FieldWidth, FifoAction, FifoGeometry, FifoSide, HalfwordMask, Nibble, Word};
    vec![
        (
            "ATCAS",
            atomic::compare_and_set(
                Nibble::new(1).unwrap(),
                Nibble::new(0).unwrap(),
                Word::W0,
                12,
            )
            .unwrap(),
        ),
        (
            "ATSWAP group",
            atomic::masked_store(HalfwordMask::ALL, 8, 12).unwrap(),
        ),
        (
            "ATINCGET",
            atomic::increment_and_get(FieldWidth::new(8).unwrap(), Word::W0, 8, 12).unwrap(),
        ),
        (
            "ATINCGETPTR wait",
            atomic::fifo(
                FifoGeometry::new(3, 0).unwrap(),
                FifoSide::Push,
                FifoAction::Wait,
                8,
                12,
            )
            .unwrap(),
        ),
        (
            "ATINCGETPTR advance",
            atomic::fifo(
                FifoGeometry::new(3, 0).unwrap(),
                FifoSide::Pop,
                FifoAction::Advance,
                8,
                12,
            )
            .unwrap(),
        ),
    ]
}

/// The same program with and without the atomic: ttsim runs the control (the
/// operand set-up and the C0 drain) and refuses the atomic itself at decode.
/// If a future ttsim runs one, this fails and the semantic gate for it must
/// move onto the simulator.
#[cfg(not(feature = "silicon"))]
#[test]
fn simulator_refuses_every_atomic_with_a_surviving_control() {
    use tt_tests::harness::Run;
    let base = tt_isa::l1::DATA.base;
    for (name, atom) in atoms() {
        let setup = || {
            let mut p = Vec::new();
            p.extend(backend::set_gpr(12, (base / 16) as u32).unwrap());
            for r in 8..12 {
                p.extend(backend::set_gpr(r, 0).unwrap());
            }
            p
        };
        let mut control = setup();
        control.push(atomic::consume());
        let mut with = setup();
        with.push(atom);
        with.push(atomic::consume());
        let run = |p: Vec<Instruction>| {
            harness::survives(|dev| {
                harness::run(dev, &Run::new(&p).dump_rows(0).stage(&[(base, &[0u8; 64])]));
            })
        };
        assert!(run(control), "{name}: control");
        assert!(
            !run(with),
            "ttsim now runs {name}: update divergence row 77 and gate it on the simulator"
        );
    }
}

// ---------------------------------------------------------------------------
// Silicon-only semantic gates.
// ---------------------------------------------------------------------------

#[cfg(feature = "silicon")]
mod silicon_gates {
    use super::*;
    use atomic::{
        FieldWidth, FifoAction, FifoCounters, FifoGeometry, FifoSide, HalfwordMask, Nibble, Word,
    };
    use tt_isa::scalar::{self, OffsetHalf, OffsetIncrement as Inc, TransferWidth as Width};
    use tt_kernels::{
        atomics::{self, Budget, GuardSpec, GuardedProgram, RoleProgram, RoleState, Spec},
        l1::{Plan, Requirements},
    };
    use tt_tests::harness::{Roles, Run};

    const GUARD: u8 = 0xa5;
    const ADDR: u32 = 12;
    const RESULT_ADDR: u32 = 13;
    const ZERO: u32 = 14;

    struct Mem {
        plan: Plan,
        /// `guard | target | guard`, 48 bytes.
        arena: u64,
        /// One 16-byte block per recorded result.
        results: u64,
        go: sync::Semaphore,
        done: [sync::Semaphore; 3],
        /// Posted by a producer once it is past its `go` wait.
        reached: sync::Semaphore,
    }

    const RESULT_BLOCKS: usize = 40;

    impl Mem {
        fn new() -> Self {
            let mut req = Requirements::new(1);
            let arena = req.scratch("atomic arena", 48, 16, 0..1);
            let results = req.scratch("results", 16 * RESULT_BLOCKS as u64, 16, 0..1);
            let go = req.semaphore("go", 0, 0..1);
            let done = [
                req.semaphore("T0 complete", 0, 0..1),
                req.semaphore("T1 complete", 0, 0..1),
                req.semaphore("T2 complete", 0, 0..1),
            ];
            let reached = req.semaphore("producer reached", 0, 0..1);
            let plan = req.plan(tt_isa::l1::DATA).unwrap();
            Mem {
                arena: plan.addr(arena),
                results: plan.addr(results),
                go: plan.semaphore(go),
                done: done.map(|s| plan.semaphore(s)),
                reached: plan.semaphore(reached),
                plan,
            }
        }
        fn target(&self) -> Region16 {
            Region16::new(self.arena + 16).unwrap()
        }
        fn arena_image(&self, target: [u8; 16]) -> Vec<u8> {
            let mut v = vec![GUARD; 16];
            v.extend_from_slice(&target);
            v.extend_from_slice(&[GUARD; 16]);
            v
        }
        /// The block recording result `k`.
        fn result(&self, k: usize) -> Region16 {
            assert!(k < RESULT_BLOCKS);
            Region16::new(self.results + 16 * k as u64).unwrap()
        }
    }

    fn word_bytes(words: [u32; 4]) -> [u8; 16] {
        let mut b = [0u8; 16];
        for (i, w) in words.iter().enumerate() {
            b[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        b
    }

    fn le(bytes: &[u8]) -> u32 {
        u32::from_le_bytes(bytes[..4].try_into().unwrap())
    }

    fn point(reg: u32, region: Region16) -> Vec<Instruction> {
        backend::set_gpr(reg, region.gpr_value()).unwrap().to_vec()
    }

    /// Record GPR group `data` (one word, or four for a quadword) in `to`,
    /// then drain C0 so the host can read it.
    fn record(data: u32, quad: bool, to: Region16) -> Vec<Instruction> {
        let mut p = point(RESULT_ADDR, to);
        p.extend(backend::set_gpr(ZERO, 0).unwrap());
        p.push(
            scalar::store_indirect_l1(
                if quad { Width::Quadword } else { Width::Word },
                OffsetHalf::new(ZERO * 2).unwrap(),
                Inc::None,
                data,
                RESULT_ADDR,
            )
            .unwrap(),
        );
        p.push(atomic::consume());
        p
    }

    fn thread_roles<'a>(thread: usize, p: &'a [Instruction]) -> Roles<'a> {
        let mut roles = [&[][..]; 3];
        roles[thread] = p;
        Roles {
            unpack: roles[0],
            math: roles[1],
            pack: roles[2],
        }
    }

    /// Run `program` on `thread`, with `arena` and the result blocks staged,
    /// and return the arena (48 bytes) and the results (`n` blocks).
    fn run(
        dev: &mut harness::Dev<'_>,
        m: &Mem,
        thread: usize,
        program: &[Instruction],
        target: [u8; 16],
        results: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let image = m.arena_image(target);
        let blank = vec![0x5au8; 16 * RESULT_BLOCKS];
        let out = harness::run(
            dev,
            &Run::roles(thread_roles(thread, program))
                .dump_rows(0)
                .stage(&[(m.arena, &image), (m.results, &blank)])
                .read_back(&[(m.arena, 48), (m.results, 16 * results)]),
        );
        let mut l1 = out.l1.into_iter();
        (l1.next().unwrap(), l1.next().unwrap())
    }

    fn check_guards(arena: &[u8]) {
        assert!(
            arena[..16].iter().all(|b| *b == GUARD),
            "guard before the block"
        );
        assert!(
            arena[32..].iter().all(|b| *b == GUARD),
            "guard after the block"
        );
    }

    // ---- ATINCGET -------------------------------------------------------

    fn incget_program(m: &Mem, width: u32, word: u32, increment: u32) -> Vec<Instruction> {
        let mut p = point(ADDR, m.target());
        p.extend(backend::set_gpr(8, increment).unwrap());
        p.push(
            atomic::increment_and_get(
                FieldWidth::new(width).unwrap(),
                Word::new(word).unwrap(),
                8,
                ADDR,
            )
            .unwrap(),
        );
        p.push(atomic::consume());
        p.extend(record(8, false, m.result(0)));
        p
    }

    /// Wrapping in every field width, on every word, from three threads: new
    /// word, preserved upper bits, the returned original, and the neighbouring
    /// words and guards untouched.
    #[test]
    fn atincget_wraps_the_field_preserves_upper_bits_and_returns_the_original() {
        harness::in_device(|dev| {
            let m = Mem::new();
            let patterns = [0xfedc_ba98u32, 0xffff_ffff, 0x0000_0001];
            let increments = [1u32, 3, 0x8000_0001, 0xffff_ffff];
            for thread in 0..3 {
                for width in [1u32, 2, 4, 8, 16, 31, 32] {
                    for word in 0..4u32 {
                        for (k, &pattern) in patterns.iter().enumerate() {
                            let increment = increments[(k + word as usize) % 4];
                            let mut block = [0x33u32, 0x44, 0x55, 0x66];
                            block[word as usize] = pattern;
                            let (new, returned) = incget_model(pattern, increment, width);
                            let (arena, results) = run(
                                dev,
                                &m,
                                thread,
                                &incget_program(&m, width, word, increment),
                                word_bytes(block),
                                1,
                            );
                            check_guards(&arena);
                            let mut want = block;
                            want[word as usize] = new;
                            assert_eq!(
                                &arena[16..32],
                                &word_bytes(want),
                                "thread {thread} width {width} word {word} inc {increment:#x} from {pattern:#x}"
                            );
                            assert_eq!(le(&results), returned, "returned original");
                        }
                    }
                }
            }
        });
    }

    /// Data mutants: a width one too wide or too narrow produces a different
    /// word than the model, so the gate above could not pass with either.
    #[test]
    fn control_atincget_with_the_wrong_width_differs_from_the_model() {
        harness::in_device(|dev| {
            let m = Mem::new();
            let (pattern, increment) = (0xabcd_ef0fu32, 3);
            let block = word_bytes([pattern, 0, 0, 0]);
            let (right, _) = incget_model(pattern, increment, 4);
            for wrong in [3u32, 5] {
                let (arena, _) = run(
                    dev,
                    &m,
                    0,
                    &incget_program(&m, wrong, 0, increment),
                    block,
                    1,
                );
                assert_ne!(
                    le(&arena[16..]),
                    right,
                    "width {wrong} gave the width-4 answer"
                );
                assert_eq!(le(&arena[16..]), incget_model(pattern, increment, wrong).0);
            }
        });
    }

    /// Three threads incrementing one counter at once get three distinct
    /// original values: the instruction is atomic across threads.
    #[test]
    fn atincget_is_atomic_across_the_three_threads() {
        harness::in_device(|dev| {
            let m = Mem::new();
            const EACH: usize = 8;
            let programs: Vec<Vec<Instruction>> = (0..3)
                .map(|t| {
                    let mut p = point(ADDR, m.target());
                    for k in 0..EACH {
                        p.extend(backend::set_gpr(8, 1).unwrap());
                        p.push(
                            atomic::increment_and_get(
                                FieldWidth::new(32).unwrap(),
                                Word::W0,
                                8,
                                ADDR,
                            )
                            .unwrap(),
                        );
                        p.push(atomic::consume());
                        p.extend(record(8, false, m.result(t * EACH + k)));
                    }
                    p
                })
                .collect();
            let image = m.arena_image([0; 16]);
            let blank = vec![0x5au8; 16 * RESULT_BLOCKS];
            let mut launch = atomics::start(
                dev,
                harness::tensix_tile(),
                &tt_tests::firmware::ROLES,
                &Spec {
                    roles: [
                        RoleProgram::Plain(&programs[0]),
                        RoleProgram::Plain(&programs[1]),
                        RoleProgram::Plain(&programs[2]),
                    ],
                    semaphores: &[],
                    stage: &[(m.arena, &image), (m.results, &blank)],
                },
                budget(),
            )
            .unwrap();
            launch.read_back(&[(m.arena, 48), (m.results, 16 * 3 * EACH)]);
            let out = launch.finish(dev, budget()).unwrap();
            check_guards(&out[0]);
            assert_eq!(le(&out[0][16..]), 3 * EACH as u32, "no increment was lost");
            let mut seen: Vec<u32> = (0..3 * EACH).map(|k| le(&out[1][16 * k..])).collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..3 * EACH as u32).collect::<Vec<_>>());
        });
    }

    fn budget() -> Budget {
        Budget::new(60_000_000, std::time::Duration::from_secs(30))
    }

    fn spec() -> GuardSpec {
        GuardSpec::for_seconds(3.0, 30.0, atomics::SILICON_POLLS_PER_SECOND).unwrap()
    }

    // ---- ATSWAP ---------------------------------------------------------

    fn swap_program(m: &Mem, mask: u8, group: u32, values: [u32; 4]) -> Vec<Instruction> {
        let mut p = point(ADDR, m.target());
        for (i, v) in values.iter().enumerate() {
            p.extend(backend::set_gpr(8 + i as u32, *v).unwrap());
        }
        p.push(atomic::masked_store(HalfwordMask(mask), group, ADDR).unwrap());
        p.push(atomic::consume());
        // Nothing comes back: record the data registers afterwards.
        p.extend(record(8, true, m.result(0)));
        p
    }

    const VALUES: [u32; 4] = [0x1122_3344, 0x5566_7788, 0x99aa_bbcc, 0xddee_ff01];
    const BLOCK: [u8; 16] = [
        0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe,
        0xff,
    ];

    /// Every one of the 256 masks, four-GPR form: exactly the selected
    /// halfwords change, and the data registers hold what they held (no old
    /// value comes back).
    #[test]
    fn atswap_group_form_writes_exactly_the_masked_halfwords() {
        harness::in_device(|dev| {
            let m = Mem::new();
            let data = swap_data_group(VALUES);
            for mask in 0..=255u8 {
                let (arena, results) =
                    run(dev, &m, 0, &swap_program(&m, mask, 8, VALUES), BLOCK, 1);
                check_guards(&arena);
                assert_eq!(
                    &arena[16..32],
                    &swap_model(BLOCK, mask, data),
                    "mask {mask:#04x}"
                );
                assert_eq!(
                    &results[..16],
                    &word_bytes(VALUES),
                    "ATSWAP returned something"
                );
            }
        });
    }

    /// Data mutant: a mask differing in one bit differs in exactly the
    /// halfword that bit names.
    #[test]
    fn control_atswap_with_a_wrong_mask_differs_in_that_halfword() {
        harness::in_device(|dev| {
            let m = Mem::new();
            let (right, wrong) = (0b0101_0101u8, 0b0101_0100u8);
            let (a, _) = run(dev, &m, 0, &swap_program(&m, right, 8, VALUES), BLOCK, 1);
            let (b, _) = run(dev, &m, 0, &swap_program(&m, wrong, 8, VALUES), BLOCK, 1);
            assert_ne!(a, b);
            assert_eq!(&a[16..18], &swap_data_group(VALUES)[..2]);
            assert_eq!(&b[16..18], &BLOCK[..2]);
        });
    }

    /// A word whose four bytes, and so both halfwords, are unique to `gpr`
    /// and nonzero, so any halfword the hardware stores names its source.
    fn seed_value(gpr: u32) -> u32 {
        ((0xa0 + gpr) << 24) | ((0x80 + gpr) << 16) | ((0x40 + gpr) << 8) | (gpr + 1)
    }

    /// Registers seeded by the diagnostic: everything but the three the
    /// harness uses (address 12, result address 13, zero offset 14).
    fn seeded() -> Vec<u32> {
        (0..24).filter(|r| ![12, 13, 14].contains(r)).collect()
    }

    /// Which (register, half) of the seeded set a stored halfword equals.
    fn decode_half(half: [u8; 2]) -> String {
        for r in seeded() {
            let v = seed_value(r).to_le_bytes();
            if v[..2] == half {
                return format!("GPR{r}.lo");
            }
            if v[2..] == half {
                return format!("GPR{r}.hi");
            }
        }
        if half == [0, 0] {
            return "zero".into();
        }
        "unknown".into()
    }

    /// DIAGNOSTIC, asserts nothing. **This is the recorded evidence for the
    /// `[-]` exclusion of `ATSWAP`'s single-register form**: the checked API
    /// (`tt_isa::scalar::atomic::masked_store`) offers only the four-GPR form, and
    /// the single form is built here through the raw generated encoder
    /// (`SingleDataReg` = 1) alone.
    ///
    /// Prints, for thread 0, what both forms leave in the block for every
    /// register 8..11 (and 16..19, a different aligned group) and every
    /// one-halfword mask, with the data registers seeded with distinct nonzero
    /// words. Each changed halfword is decoded to the seeded register half it
    /// equals. On Blackhole card 0 the single form's destination lanes follow
    /// no consistent rule (register 8 with mask 0x01 stored zeros where the
    /// Wormhole page puts `GPR8.lo`; registers 9 and 10 touched halfwords 4 and
    /// 5 only; 11 halfwords 6 and 7; 16 only halfword 1; 17 and 18 halfwords 0
    /// and 1; 19 halfword 0 plus 2 and 3), while the four-GPR form matched the
    /// page for all 256 masks. See the run ids in the close-out record.
    #[test]
    fn atswap_single_form_sweep_diagnostic() {
        harness::in_device(|dev| {
            let m = Mem::new();
            println!("\nDIAGNOSTIC ATSWAP single-form sweep, thread 0");
            println!("block before: {BLOCK:02x?}");
            for r in seeded() {
                println!("  GPR{r} = {:#010x}", seed_value(r));
            }
            // (label, SingleDataReg, register)
            let forms: Vec<(String, u32, u32)> = [8u32, 9, 10, 11, 16, 17, 18, 19]
                .iter()
                .map(|r| (format!("single reg {r}"), 1, *r))
                .chain(
                    [8u32, 16]
                        .iter()
                        .map(|r| (format!("group  reg {r}"), 0, *r)),
                )
                .collect();
            for (label, single, reg) in forms {
                for bit in 0..8 {
                    let mask = 1u8 << bit;
                    let mut p = point(ADDR, m.target());
                    for r in seeded() {
                        p.extend(backend::set_gpr(r, seed_value(r)).unwrap());
                    }
                    p.push(
                        tt_isa::isa::generated::encode::atswap(single, mask as u32, reg, ADDR)
                            .unwrap(),
                    );
                    p.push(atomic::consume());
                    let (arena, _) = run(dev, &m, 0, &p, BLOCK, 1);
                    let after = &arena[16..32];
                    let changed: Vec<String> = (0..8)
                        .filter(|h| after[2 * h..2 * h + 2] != BLOCK[2 * h..2 * h + 2])
                        .map(|h| {
                            format!("half{h}={}", decode_half([after[2 * h], after[2 * h + 1]]))
                        })
                        .collect();
                    println!("{label} mask {mask:#04x}: {after:02x?} changed {changed:?}");
                }
            }
        });
    }

    // ---- ATCAS (compare already met) ------------------------------------

    fn cas_program(m: &Mem, set: u32, compare: u32, word: u32) -> Vec<Instruction> {
        let mut p = point(ADDR, m.target());
        p.push(
            atomic::compare_and_set(
                Nibble::new(set).unwrap(),
                Nibble::new(compare).unwrap(),
                Word::new(word).unwrap(),
                ADDR,
            )
            .unwrap(),
        );
        p.push(atomic::consume());
        p
    }

    /// With the word already equal to the compare value the instruction does
    /// not wait: the whole word becomes the set value, nothing else changes.
    #[test]
    fn atcas_with_the_compare_already_met_replaces_the_whole_word() {
        harness::in_device(|dev| {
            let m = Mem::new();
            for thread in 0..3 {
                for word in 0..4u32 {
                    for (set, compare) in [(0u32, 0u32), (1, 0), (0xf, 0x5), (0xa, 0xf), (7, 7)] {
                        let mut block = [0x77u32, 0x88, 0x99, 0xaa];
                        block[word as usize] = compare;
                        let (arena, _) = run(
                            dev,
                            &m,
                            thread,
                            &cas_program(&m, set, compare, word),
                            word_bytes(block),
                            1,
                        );
                        check_guards(&arena);
                        let mut want = block;
                        want[word as usize] = cas_model(compare, compare, set).unwrap();
                        assert_eq!(
                            &arena[16..32],
                            &word_bytes(want),
                            "thread {thread} word {word} set {set} compare {compare}"
                        );
                    }
                }
            }
        });
    }

    // ---- ATINCGETPTR (never blocking here) ------------------------------

    fn fifo_geometry() -> FifoGeometry {
        FifoGeometry::new(3, 0).unwrap()
    }

    /// A deterministic mix of pushes and pops, each issued only when the
    /// model says it will not retry, recording what comes back each time.
    #[test]
    fn atincgetptr_follows_the_fifo_model_through_wraps() {
        harness::in_device(|dev| {
            let m = Mem::new();
            for (geometry, thread) in [
                (FifoGeometry::new(3, 0).unwrap(), 0usize),
                (FifoGeometry::new(4, 1).unwrap(), 1),
                (FifoGeometry::new(2, 0).unwrap(), 2),
                // The same block, a different shape: staging rewrites it whole.
                (FifoGeometry::new(5, 2).unwrap(), 0),
            ] {
                let mut model = FifoModel {
                    width: geometry.width(),
                    rd: 0,
                    wr: 0,
                };
                // Start part-full and wrapped, whole batches only.
                let start = geometry.capacity() / 2;
                model.rd = (geometry.counter_mask() - start + 1) & geometry.counter_mask();
                model.wr = model.rd.wrapping_add(start) & geometry.counter_mask();
                let counters = FifoCounters::new(geometry, model.rd, model.wr).unwrap();
                let mut program = point(ADDR, m.target());
                let mut expected = Vec::new();
                let mut seed = 0x2545_f491u32;
                let mut k = 0;
                while k < 24 {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let (push, advance) = ((seed >> 20) & 1 == 1, (seed >> 21) % 4 != 0);
                    let Some((next, old)) = model.step(push, advance, geometry.batch_log2()) else {
                        continue;
                    };
                    model = next;
                    program.push(
                        atomic::fifo(
                            geometry,
                            if push { FifoSide::Push } else { FifoSide::Pop },
                            if advance {
                                FifoAction::Advance
                            } else {
                                FifoAction::Wait
                            },
                            8,
                            ADDR,
                        )
                        .unwrap(),
                    );
                    program.push(atomic::consume());
                    program.extend(record(8, false, m.result(k)));
                    expected.push(old);
                    k += 1;
                }
                let (arena, results) =
                    run(dev, &m, thread, &program, word_bytes(counters.words()), k);
                check_guards(&arena);
                let got: Vec<u32> = (0..k).map(|i| le(&results[16 * i..])).collect();
                assert_eq!(got, expected, "returned counters, {geometry:?}");
                assert_eq!(
                    &arena[16..32],
                    &word_bytes([model.rd, model.wr, 0, 0]),
                    "final counters and untouched padding, {geometry:?}"
                );
            }
        });
    }

    // ---- Blocking forms: producer on another thread, deadline active ----

    fn launch_two<'a>(
        dev: &mut harness::Dev<'_>,
        m: &Mem,
        consumer: &'a GuardedProgram,
        producer: &'a GuardedProgram,
        block: [u8; 16],
    ) -> atomics::Launch<tt_isa::noc::Noc0> {
        let init = m.plan.semaphore_init();
        let image = m.arena_image(block);
        let blank = vec![0x5au8; 16 * RESULT_BLOCKS];
        let mut launch = atomics::start(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            &Spec {
                roles: [
                    RoleProgram::Guarded(consumer),
                    RoleProgram::Guarded(producer),
                    RoleProgram::Idle,
                ],
                semaphores: &init,
                stage: &[(m.arena, &image), (m.results, &blank)],
            },
            budget(),
        )
        .unwrap();
        launch.read_back(&[(m.arena, 48), (m.results, 64)]);
        launch
    }

    fn settle(dev: &mut harness::Dev<'_>) {
        for _ in 0..200 {
            harness::advance(dev, 100_000);
        }
    }

    /// [`harness::in_device`] whose failure names the gate. `in_device` resets
    /// the tile and the gate thread's Tensix state (the session's reset, which first
    /// releases a thread an earlier run left parked) before the body, so a failure
    /// there is named by the label as the gate that was starting.
    fn labelled(label: &str, f: impl FnOnce(&mut harness::Dev<'_>)) {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| harness::in_device(f)));
        if let Err(payload) = result {
            let text = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "(non-text panic)".into());
            panic!("[{label}] {text}");
        }
    }

    /// Run a blocking gate's body so that **no failure leaves a Tensix thread
    /// parked**: on a panic it prints each role's breadcrumbs, applies `relief`
    /// (writing the L1 state the blocked instruction is polling for, which no
    /// semaphore release can do), holds the cores and runs the repository's
    /// recovery (`Launch::abort_and_recover`), then re-raises. A thread left
    /// parked would make every later gate's reset fail.
    fn rescued(
        dev: &mut harness::Dev<'_>,
        launch: &atomics::Launch<tt_isa::noc::Noc0>,
        relief: impl FnOnce(&mut harness::Dev<'_>, &atomics::Launch<tt_isa::noc::Noc0>),
        body: impl FnOnce(&mut harness::Dev<'_>, &atomics::Launch<tt_isa::noc::Noc0>),
    ) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(dev, launch)));
        if let Err(payload) = result {
            for role in 0..2 {
                match launch.breadcrumbs(dev, role) {
                    Ok(text) => eprintln!(
                        "on failure: {text}; semaphores {:?}",
                        launch.semaphores(dev, role)
                    ),
                    Err(e) => eprintln!("on failure: role {role} unreadable: {e}"),
                }
            }
            relief(dev, launch);
            if let Err(e) = launch.abort_and_recover(dev, &tt_tests::firmware::ROLES) {
                eprintln!("recovery failed: {e}");
            }
            std::panic::resume_unwind(payload);
        }
    }

    /// `ATCAS` blocks on an unmet compare, the role reports BLOCKED, and the
    /// host making the word equal frees it. A word set to the wrong value
    /// leaves it blocked (the control).
    #[test]
    fn atcas_blocks_until_the_host_makes_the_word_equal() {
        labelled("atcas blocks until the host makes the word equal", |dev| {
            let m = Mem::new();
            let mut body = cas_program(&m, 9, 3, 1);
            body.pop(); // the guard appends its own drain
            let consumer = GuardedProgram::new(0, &body, spec(), m.done[0]).unwrap();
            let idle = GuardedProgram::new(1, &[], spec(), m.done[1]).unwrap();
            let launch = launch_two(dev, &m, &consumer, &idle, word_bytes([0x11, 2, 0x33, 0x44]));
            let word1 = m.target().word(Word::W1);
            rescued(
                dev,
                &launch,
                // An ATCAS retries until the word equals: only this frees it.
                |dev, launch| {
                    let _ = launch.write32(dev, word1, 3);
                },
                |dev, launch| {
                    launch
                        .wait(dev, budget(), "ATCAS to report BLOCKED", |dev, l| {
                            Ok(l.state(dev, 0)? == RoleState::Blocked)
                        })
                        .unwrap();
                    // The wrong value: still blocked afterwards, word untouched by it.
                    launch.write32(dev, word1, 4).unwrap();
                    settle(dev);
                    assert_eq!(launch.state(dev, 0).unwrap(), RoleState::Blocked);
                    assert_eq!(launch.read32(dev, word1).unwrap(), 4);
                    // The right one.
                    launch.write32(dev, word1, 3).unwrap();
                    launch
                        .wait(dev, budget(), "ATCAS to complete", |dev, l| {
                            Ok(l.state(dev, 0)? == RoleState::Done)
                        })
                        .unwrap();
                    let out = launch.finish(dev, budget()).unwrap();
                    check_guards(&out[0]);
                    assert_eq!(
                        &out[0][16..32],
                        &word_bytes([0x11, cas_model(3, 3, 9).unwrap(), 0x33, 0x44])
                    );
                },
            );
        });
    }

    /// Wait until both guarded roles are parked (the atomic's role reports
    /// BLOCKED at its deadline, the other at its `go` wait).
    fn both_blocked(launch: &atomics::Launch<tt_isa::noc::Noc0>, dev: &mut harness::Dev<'_>) {
        launch
            .wait(dev, budget(), "both roles to report BLOCKED", |dev, l| {
                Ok(
                    l.state(dev, 0)? == RoleState::Blocked
                        && l.state(dev, 1)? == RoleState::Blocked,
                )
            })
            .unwrap();
    }

    /// HARDWARE FINDING, asserted. A Tensix thread parked in `ATCAS` (or
    /// `ATINCGETPTR`) keeps the Scalar Unit busy: `ScalarUnit.md` says the unit
    /// "is executing at most one instruction at a time" for all three threads,
    /// so until the atomic completes *no instruction from any other thread can
    /// enter it* -- `SETDMAREG` and `STOREIND` included. A producer on another
    /// Tensix thread that must run any scalar instruction before it can feed
    /// the atomic therefore cannot (the first silicon runs of the producer-thread
    /// gates: the producer stayed BLOCKED for the whole grace period).
    ///
    /// Thread 1 here is released past its `go` wait, posts `reached` (a Sync
    /// Unit instruction), then needs one `SETDMAREG`. While thread 0's `ATCAS`
    /// retries, `reached` is posted and the thread never completes; once the
    /// host makes the word equal, both finish. If the producer completes with
    /// the `ATCAS` still parked, this fails and the finding is refuted.
    #[test]
    fn blocked_atomic_monopolizes_the_scalar_unit() {
        labelled("a blocked atcas monopolizes the scalar unit", |dev| {
            let m = Mem::new();
            let mut cas = cas_program(&m, 9, 3, 0);
            cas.pop();
            let consumer = GuardedProgram::new(0, &cas, spec(), m.done[0]).unwrap();
            let mut produce = vec![
                sync::wait_nonzero(m.go, Before::EVERYTHING),
                sync::post(m.reached),
            ];
            produce.extend(backend::set_gpr(8, 1).unwrap());
            let producer = GuardedProgram::new(1, &produce, spec(), m.done[1]).unwrap();
            let launch = launch_two(dev, &m, &consumer, &producer, word_bytes([2, 0, 0, 0]));
            let word0 = m.target().word(Word::W0);
            rescued(
                dev,
                &launch,
                |dev, launch| {
                    let _ = launch.write32(dev, word0, 3);
                },
                |dev, launch| {
                    both_blocked(launch, dev);
                    launch.release(dev, 1, 1 << m.go.index(), budget()).unwrap();
                    let reached = m.reached.index() as usize;
                    launch
                        .wait(
                            dev,
                            budget(),
                            "the producer to pass its go wait",
                            |dev, l| Ok(l.semaphores(dev, 1)?[reached] >= 1),
                        )
                        .unwrap();
                    // Two seconds with the ATCAS still retrying.
                    for _ in 0..20 {
                        harness::advance(dev, 100_000_000);
                    }
                    let sems = launch.semaphores(dev, 1).unwrap();
                    assert_ne!(
                        launch.state(dev, 1).unwrap(),
                        RoleState::Done,
                        "the producer finished while an ATCAS was parked: the Scalar Unit is \
                         NOT monopolized, so the finding is refuted"
                    );
                    assert_eq!(
                        sems[m.done[1].index() as usize],
                        0,
                        "producer completion posted"
                    );
                    println!(
                        "FINDING producer past go (reached={}) but stuck on its SETDMAREG while \
                         thread 0's ATCAS retries: {}",
                        sems[reached],
                        launch.breadcrumbs(dev, 1).unwrap()
                    );
                    // The host makes the word equal: the ATCAS completes, the
                    // Scalar Unit frees, the producer finishes.
                    launch.write32(dev, word0, 3).unwrap();
                    let out = launch.finish(dev, budget()).unwrap();
                    check_guards(&out[0]);
                    assert_eq!(le(&out[0][16..]), 9);
                },
            );
        });
    }

    /// `ATCAS` freed by another agent: role 1's RISC-V core stores the compare
    /// value (`Launch::poke`), not a Tensix thread.
    #[test]
    fn atcas_is_freed_by_the_other_roles_risc_v_core() {
        labelled("atcas freed by a poke", |dev| {
            let m = Mem::new();
            let mut cas = cas_program(&m, 9, 3, 0);
            cas.pop();
            let consumer = GuardedProgram::new(0, &cas, spec(), m.done[0]).unwrap();
            let waiter = vec![sync::wait_nonzero(m.go, Before::EVERYTHING)];
            let producer = GuardedProgram::new(1, &waiter, spec(), m.done[1]).unwrap();
            let launch = launch_two(dev, &m, &consumer, &producer, word_bytes([2, 0, 0, 0]));
            let word0 = m.target().word(Word::W0);
            rescued(
                dev,
                &launch,
                |dev, launch| {
                    let _ = launch.write32(dev, word0, 3);
                },
                |dev, launch| {
                    both_blocked(launch, dev);
                    assert_eq!(launch.read32(dev, word0).unwrap(), 2);
                    launch.poke(dev, 1, word0, 3, budget()).unwrap();
                    launch
                        .wait(dev, budget(), "ATCAS to complete", |dev, l| {
                            Ok(l.state(dev, 0)? == RoleState::Done)
                        })
                        .unwrap();
                    launch.release(dev, 1, 1 << m.go.index(), budget()).unwrap();
                    let out = launch.finish(dev, budget()).unwrap();
                    check_guards(&out[0]);
                    assert_eq!(
                        le(&out[0][16..]),
                        9,
                        "2 became 3 by the poke, the compare held, the set value replaced it"
                    );
                },
            );
        });
    }

    fn fifo_program(
        m: &Mem,
        geometry: FifoGeometry,
        side: FifoSide,
        action: FifoAction,
        record_to: usize,
    ) -> Vec<Instruction> {
        let mut p = point(ADDR, m.target());
        p.push(atomic::fifo(geometry, side, action, 8, ADDR).unwrap());
        p.push(atomic::consume());
        p.extend(record(8, false, m.result(record_to)));
        p
    }

    /// A pop of an empty FIFO blocks; another agent (role 1's RISC-V core)
    /// makes it non-empty by storing the write counter, and the pop gets the
    /// read counter it advanced.
    #[test]
    fn atincgetptr_pop_blocks_until_the_write_counter_is_poked() {
        labelled("atincgetptr pop freed by a poke", |dev| {
            let m = Mem::new();
            let g = fifo_geometry();
            let consumer = GuardedProgram::new(
                0,
                &fifo_program(&m, g, FifoSide::Pop, FifoAction::Advance, 0),
                spec(),
                m.done[0],
            )
            .unwrap();
            let waiter = vec![sync::wait_nonzero(m.go, Before::EVERYTHING)];
            let producer = GuardedProgram::new(1, &waiter, spec(), m.done[1]).unwrap();
            let counters = FifoCounters::new(g, 5, 5).unwrap();
            let launch = launch_two(dev, &m, &consumer, &producer, word_bytes(counters.words()));
            let write_counter = m.target().word(Word::W1);
            rescued(
                dev,
                &launch,
                |dev, launch| {
                    let _ = launch.write32(dev, write_counter, 6);
                },
                |dev, launch| {
                    both_blocked(launch, dev);
                    launch.poke(dev, 1, write_counter, 6, budget()).unwrap();
                    launch
                        .wait(dev, budget(), "the pop to complete", |dev, l| {
                            Ok(l.state(dev, 0)? == RoleState::Done)
                        })
                        .unwrap();
                    launch.release(dev, 1, 1 << m.go.index(), budget()).unwrap();
                    let out = launch.finish(dev, budget()).unwrap();
                    check_guards(&out[0]);
                    let model = FifoModel {
                        width: 3,
                        rd: 5,
                        wr: 6,
                    };
                    let (model, popped) = model.step(false, true, 0).unwrap();
                    assert_eq!(le(&out[1]), popped, "the consumer's counter");
                    assert_eq!(&out[0][16..32], &word_bytes([model.rd, model.wr, 0, 0]));
                },
            );
        });
    }

    /// A push onto a full FIFO blocks; the other agent frees it by storing the
    /// read counter.
    #[test]
    fn atincgetptr_push_blocks_until_the_read_counter_is_poked() {
        labelled("atincgetptr push freed by a poke", |dev| {
            let m = Mem::new();
            let g = fifo_geometry();
            let pusher = GuardedProgram::new(
                0,
                &fifo_program(&m, g, FifoSide::Push, FifoAction::Advance, 0),
                spec(),
                m.done[0],
            )
            .unwrap();
            let waiter = vec![sync::wait_nonzero(m.go, Before::EVERYTHING)];
            let popper = GuardedProgram::new(1, &waiter, spec(), m.done[1]).unwrap();
            // Full, with the write counter wrapped past the read counter.
            let counters = FifoCounters::new(g, 6, 2).unwrap();
            let launch = launch_two(dev, &m, &pusher, &popper, word_bytes(counters.words()));
            let read_counter = m.target().word(Word::W0);
            rescued(
                dev,
                &launch,
                |dev, launch| {
                    let _ = launch.write32(dev, read_counter, 7);
                },
                |dev, launch| {
                    both_blocked(launch, dev);
                    launch.poke(dev, 1, read_counter, 7, budget()).unwrap();
                    launch
                        .wait(dev, budget(), "the push to complete", |dev, l| {
                            Ok(l.state(dev, 0)? == RoleState::Done)
                        })
                        .unwrap();
                    launch.release(dev, 1, 1 << m.go.index(), budget()).unwrap();
                    let out = launch.finish(dev, budget()).unwrap();
                    check_guards(&out[0]);
                    let model = FifoModel {
                        width: 3,
                        rd: 7,
                        wr: 2,
                    };
                    assert!(!model.full());
                    let (model, pushed) = model.step(true, true, 0).unwrap();
                    assert_eq!(le(&out[1]), pushed, "the pusher's counter");
                    assert_eq!(&out[0][16..32], &word_bytes([model.rd, model.wr, 0, 0]));
                },
            );
        });
    }

    // ---- Isolated minimal probes, one instruction each -------------------

    /// The coordinator runs these first, one per instruction, each in a child
    /// process: they establish only that the opcode executes and returns.
    #[test]
    fn probe_atincget_isolated() {
        let m = Mem::new();
        assert!(harness::survives(|dev| {
            run(dev, &m, 0, &incget_program(&m, 32, 0, 1), [0; 16], 1);
        }));
    }

    #[test]
    fn probe_atswap_isolated() {
        let m = Mem::new();
        assert!(harness::survives(|dev| {
            run(dev, &m, 0, &swap_program(&m, 0xff, 8, VALUES), BLOCK, 1);
        }));
    }

    #[test]
    fn probe_atcas_with_the_compare_met_isolated() {
        let m = Mem::new();
        assert!(harness::survives(|dev| {
            run(dev, &m, 0, &cas_program(&m, 1, 0, 0), [0; 16], 1);
        }));
    }

    #[test]
    fn probe_atincgetptr_nonblocking_isolated() {
        let m = Mem::new();
        let g = fifo_geometry();
        assert!(harness::survives(|dev| {
            // An empty FIFO's push does not wait.
            run(
                dev,
                &m,
                0,
                &fifo_program(&m, g, FifoSide::Push, FifoAction::Advance, 0),
                word_bytes([0; 4]),
                1,
            );
        }));
    }
}
