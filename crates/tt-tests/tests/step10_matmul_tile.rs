//! Phase 6, beyond one block: the three roles running at once, and the
//! hand-offs between them.
//!
//! `step9_matmul` runs its roles in order, each to completion, which is one
//! valid schedule for a kernel whose unpacker never gets ahead of the Matrix
//! Unit by more than the two `Src` banks. A tile does not fit that: it streams
//! more operands through each `Src` than there are banks, so the unpack role
//! has to wait for the math role to hand banks back, and the math role has to
//! tell the pack role when `Dst` is ready. `harness::Run::concurrent` releases
//! all three together, and `tt_isa::sync` carries the `Dst` hand-off.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::matrix::Banks;
use tt_isa::numerics::mvmul_reference;
use tt_isa::sync::{self, Semaphore, Unit};
use tt_tests::datapath::{self, config_program, pack_config, set_adc_x, Unpacker, OUT, STAGE};
use tt_tests::harness::{self, Roles, Run, SemaphoreInit};
use tt_tests::matmul::{self, stage_operand, ROW, SRC_A_ROW, SRC_B_ROW, TF32_CODE};

const STAGE_A: u64 = STAGE;
const STAGE_B: u64 = STAGE + 0x2000;

type MatA = [[f32; 16]; 16];
type MatB = [[f32; 16]; 8];
const ZERO_DST: MatB = [[0f32; 16]; 8];

/// Math -> pack: "`Dst` holds a finished result".
const DST_READY: Semaphore = match Semaphore::new(0) {
    Some(s) => s,
    None => unreachable!(),
};

const L1_SENTINEL: u32 = 0xA5A5_5A5A;
const PACK_READBACK: usize = 16 * ROW;

/// A deterministic generator; CI is flake-free on purpose.
struct Lcg(u64);

impl Lcg {
    fn int(&mut self, bound: i32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as i32 % (2 * bound + 1) - bound) as f32
    }
}

fn small_integer_operands(seed: u64) -> (MatA, MatB) {
    let mut rng = Lcg(seed);
    let mut a = [[0f32; 16]; 16];
    let mut b = [[0f32; 16]; 8];
    a.iter_mut().flatten().for_each(|v| *v = rng.int(15));
    b.iter_mut().flatten().for_each(|v| *v = rng.int(63));
    (a, b)
}

/// How the pack role learns `Dst` is ready.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Handoff {
    /// `SEMPOST` after the math role's last `MVMUL` has finished; `SEMWAIT`
    /// then `SEMGET` ahead of the pack role's first `PACR`.
    Semaphore,
    /// None: the pack role packs whenever it gets there. The control.
    Unsynchronised,
}

/// `rounds` unpack/`MVMUL` rounds over the same operands, accumulating, then
/// eight rows packed to [`OUT`]. All the unpacks are issued before any of the
/// `MVMUL`s in program order -- the unpack role does not wait for anything --
/// so with more than two rounds it depends on the math role running
/// alongside it to hand `Src` banks back.
fn roles(na: u32, nb: u32, rounds: usize, handoff: Handoff) -> [Vec<Instruction>; 3] {
    let mut unpack = matmul::unpack_prelude(matmul::Operands {
        a_addr: STAGE_A,
        na,
        b_addr: STAGE_B,
        nb,
        out: TF32_CODE,
    });
    let mut math = matmul::math_prelude();

    // One typestate threads through both programs, so the lockstep rule --
    // every bank handed over is handed back before it is written again -- is
    // checked across the split exactly as it is within one program.
    let base = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    let mut banks = Banks::after_reset();
    for _ in 0..rounds {
        unpack.push(set_adc_x(Unpacker::SrcA, 0, na - 1));
        let (i, loaded_a) = banks.unpack_a(base).unwrap();
        unpack.push(i);
        unpack.push(set_adc_x(Unpacker::SrcB, 0, nb - 1));
        let (i, loaded) = loaded_a.unpack_b(base).unwrap();
        unpack.push(i);
        let (i, empty) = loaded
            .mvmul_release_both(encode::Mvmul::ZERO.dst_row(0))
            .unwrap();
        math.push(i);
        banks = empty;
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    unpack.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());

    let mut pack = Vec::new();
    let mut words = ConfigWords::new();
    pack_config(&mut words, OUT);
    pack.extend(config_program(&words));
    match handoff {
        Handoff::Semaphore => {
            math.extend(sync::post_after(Unit::Matrix, DST_READY));
            pack.extend(sync::take(DST_READY, Before::PACKER));
        }
        Handoff::Unsynchronised => {
            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
        }
    }
    pack.extend(datapath::pack_rows(8));
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, math, pack]
}

const SEMAPHORES: [SemaphoreInit; 1] = [(DST_READY, 0, sync::MAX_VALUE)];

/// Everything a run of [`roles`] needs staged: both operands, and sentinel over
/// the pack output.
struct Staged {
    sa: Vec<u8>,
    sb: Vec<u8>,
    sentinel: Vec<u8>,
    programs: [Vec<Instruction>; 3],
}

impl Staged {
    fn new(a: &MatA, b: &MatB, rounds: usize, handoff: Handoff) -> Staged {
        let (sa, na) = stage_operand(SRC_A_ROW, a);
        let (sb, nb) = stage_operand(SRC_B_ROW, b);
        let sentinel = L1_SENTINEL
            .to_le_bytes()
            .iter()
            .copied()
            .cycle()
            .take(PACK_READBACK * 4)
            .collect();
        Staged {
            sa,
            sb,
            sentinel,
            programs: roles(na, nb, rounds, handoff),
        }
    }

    /// Run it, concurrently or in order.
    fn run(&self, dev: &mut harness::Dev<'_>, concurrent: bool) -> harness::Outcome {
        let [unpack, math, pack] = &self.programs;
        let stage = [
            (STAGE_A, self.sa.as_slice()),
            (STAGE_B, self.sb.as_slice()),
            (OUT, self.sentinel.as_slice()),
        ];
        let read_back = [(OUT, PACK_READBACK * 4)];
        let mut spec = Run::roles(Roles { unpack, math, pack })
            .stage(&stage)
            .dump_rows(16)
            .read_back(&read_back);
        if concurrent {
            spec = spec.concurrent(&SEMAPHORES);
        }
        harness::run(dev, &spec)
    }
}

/// Run [`roles`] concurrently and return `(Dst rows 0..16, the packed words)`.
fn run(a: &MatA, b: &MatB, rounds: usize, handoff: Handoff) -> (Vec<u32>, Vec<u32>) {
    let staged = Staged::new(a, b, rounds, handoff);
    let path = std::env::temp_dir().join(format!(
        "ttconc-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    harness::in_device(|dev| {
        let out = staged.run(dev, true);
        let mut bytes: Vec<u8> = out.dst.iter().flat_map(|w| w.to_le_bytes()).collect();
        bytes.extend_from_slice(&out.l1[0]);
        std::fs::write(&path, bytes).unwrap();
    });
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (dst, l1) = words.split_at(16 * ROW);
    (dst.to_vec(), l1.to_vec())
}

fn assert_block(words: &[u32], want: &MatB, what: &str) {
    for (i, row) in want.iter().enumerate() {
        for (j, w) in row.iter().enumerate() {
            let got = words[i * ROW + j];
            assert_eq!(
                got,
                w.to_bits(),
                "{what}: [{i}][{j}] is {got:#010x} = {}, want {w}",
                f32::from_bits(got)
            );
        }
    }
}

fn accumulated(a: &MatA, b: &MatB, rounds: usize) -> MatB {
    let mut acc = ZERO_DST;
    for _ in 0..rounds {
        acc = mvmul_reference(&acc, b, a, &[0]).expect("small integers are exact");
    }
    acc
}

/// Three rounds through two banks: the unpack role's third `UNPACR` into each
/// `Src` can only proceed once the math role's first `MVMUL` has handed a bank
/// back, which the in-order schedule never lets happen. Concurrently the three
/// products accumulate and the pack role, released by the semaphore, packs
/// their sum.
#[test]
fn concurrent_roles_stream_more_operands_than_there_are_banks() {
    let (a, b) = small_integer_operands(0xc0c0);
    let want = accumulated(&a, &b, 3);
    assert_ne!(want, accumulated(&a, &b, 2));
    let (dst, l1) = run(&a, &b, 3, Handoff::Semaphore);
    assert_block(&dst, &want, "Dst");
    assert_block(&l1, &want, "L1");
    assert!(
        l1[8 * ROW..].iter().all(|&w| w == L1_SENTINEL),
        "the packer wrote past the eight rows"
    );
}

/// The other half of the gate above: the same programs *in order* finish with
/// two rounds, which fit in the two banks, and cannot finish with three -- the
/// unpack role waits for a bank the math role, not yet started, would free.
/// So the concurrent run is not merely one schedule among several that work.
#[test]
fn in_order_the_third_round_waits_forever() {
    let (a, b) = small_integer_operands(0xc0c0);
    let two = Staged::new(&a, &b, 2, Handoff::Unsynchronised);
    assert!(
        harness::survives(|dev| {
            two.run(dev, false);
        }),
        "two rounds fit in two banks and must finish in order"
    );
    let three = Staged::new(&a, &b, 3, Handoff::Unsynchronised);
    assert!(
        !harness::survives(|dev| {
            three.run(dev, false);
        }),
        "a third round cannot get a bank until the math role runs"
    );
}

/// The semaphore is what orders the pack after the math: without it the pack
/// role reaches `PACR` long before the math role has three products, and packs
/// a `Dst` that is not yet the answer.
///
/// On both targets, and it was a finding on both. ttsim interleaves the three
/// threads and showed the race at once (0 of 128 datums right). Silicon first
/// showed nothing -- all 128 right -- because the harness released the roles
/// one at a time, each after loading its image, so the unpack and math roles
/// had finished before the pack role started. Released together by one write
/// to the soft-reset register (`Device::load_and_start_together`), silicon
/// gives 0 of 128 as well.
#[test]
fn without_the_semaphore_the_pack_races_the_math() {
    let (a, b) = small_integer_operands(0xc0c0);
    let want = accumulated(&a, &b, 3);
    let (_, l1) = run(&a, &b, 3, Handoff::Unsynchronised);
    let right = (0..8 * ROW)
        .filter(|&k| l1[k] == want[k / ROW][k % ROW].to_bits())
        .count();
    println!(
        "unsynchronised pack: {right} of {} datums are the answer",
        8 * ROW
    );
    assert!(
        right < 8 * ROW,
        "the pack must race the math without the semaphore, or the gate above proves nothing"
    );
}
