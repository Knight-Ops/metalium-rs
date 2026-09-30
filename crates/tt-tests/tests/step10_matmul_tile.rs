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
use tt_isa::tile::L1Format;
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

// --- One 32x32 tile -----------------------------------------------------------

type Tile = [[f32; 32]; 32];

fn tile_operands(seed: u64) -> (Tile, Tile) {
    // `A` becomes `SrcB` and keeps seven significant bits in phase 0, `B`
    // becomes `SrcA` and keeps five: `|A| <= 127`, `|B| <= 31` lose nothing,
    // and 32-term sums stay far under 2^24.
    let mut rng = Lcg(seed);
    let mut a = [[0f32; 32]; 32];
    let mut b = [[0f32; 32]; 32];
    a.iter_mut().flatten().for_each(|v| *v = rng.int(127));
    b.iter_mut().flatten().for_each(|v| *v = rng.int(31));
    (a, b)
}

fn flat(m: &Tile) -> Vec<f32> {
    m.iter().flatten().copied().collect()
}

/// `C = A @ B` for 32x32 tiles staged by `tt_layout` in `in_fmt`, converted to
/// `out_fmt` in `Src`, through [`matmul::tile_roles`], de-tiled on the host.
fn run_tile(a: &Tile, b: &Tile, in_fmt: L1Format, out_fmt: u32) -> Vec<f32> {
    let (ta, _) = matmul::tilize_f32(&flat(a), 32, 32, in_fmt);
    let (tb, _) = matmul::tilize_f32(&flat(b), 32, 32, in_fmt);
    const A_AT: u64 = STAGE;
    const B_AT: u64 = STAGE + 0x2000;
    let [unpack, math, pack] = matmul::tile_roles(&[(A_AT, B_AT)], in_fmt, out_fmt, OUT);
    let path = std::env::temp_dir().join(format!(
        "tttile-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &math,
                pack: &pack,
            })
            .concurrent(&matmul::TILE_SEMAPHORES)
            .stage(&[(A_AT, &ta), (B_AT, &tb)])
            .dump_rows(0)
            .read_back(&[(OUT, 1024 * 4)]),
        );
        std::fs::write(&path, &out.l1[0]).unwrap();
    });
    let packed = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    matmul::detilize_packed(&packed, 32, 32)
}

/// A whole 32x32 tile, from `tt_layout`'s tile images to `tt_layout`'s
/// de-tiling: every datum matches the face-composed model and Burn, which must
/// agree with each other first.
///
/// This is also where the `Z`-plane-to-face convention stops being a choice:
/// the unpacker takes face `z` as the `z`-th 256 datums of the image (its `Z`
/// ADC, `UNPACR_Regular.md:182`), and `tt_layout` puts face `(z / 2, z % 2)`
/// there. If either were transposed the product would be `A @ B` with two of
/// its quadrants exchanged, which the transpose control below and the
/// quadrant-distinct operands both rule out.
#[test]
fn a_32x32_tile_matmul_matches_the_model_and_burn() {
    use burn_tensor::{Tensor, TensorData};
    type B = burn_flex::Flex;
    let device = burn_flex::FlexDevice;

    let (a, b) = tile_operands(0x7117);
    let model = tt_isa::numerics::matmul_tile_reference(&[[0f32; 32]; 32], &a, &b, &[0])
        .expect("small integers are exact");
    let ta = Tensor::<B, 2>::from_data(TensorData::new(flat(&a), [32, 32]), &device);
    let tb = Tensor::<B, 2>::from_data(TensorData::new(flat(&b), [32, 32]), &device);
    let burn: Vec<f32> = ta.clone().matmul(tb.clone()).into_data().to_vec().unwrap();
    assert_eq!(flat(&model), burn, "the model and Burn must agree first");

    let got = run_tile(&a, &b, L1Format::Fp32, TF32_CODE);
    for (k, (g, w)) in got.iter().zip(&burn).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "C[{}][{}] is {g}, want {w}",
            k / 32,
            k % 32
        );
    }
    // Control: the gate distinguishes `A @ B` from `A @ B^T`.
    let transposed: Vec<f32> = ta.matmul(tb.transpose()).into_data().to_vec().unwrap();
    assert!(got.iter().zip(&transposed).any(|(g, t)| g != t));
}

/// The same tile through BF16 `Src`, by both routes into it (divergence row
/// H): FP32 in L1 converted by the unpacker, and BF16 in L1 unconverted. The
/// operands are small integers, exact in BF16, so the answer is the TF32
/// one's, bit for bit; the Matrix Unit reads BF16 `Src` with the BF16 style
/// (`MVMUL.md`), and phase 0 keeps as many bits of each as TF32 does.
#[test]
fn a_32x32_tile_matmul_through_bf16_src() {
    let (a, b) = tile_operands(0xbf16);
    let model = tt_isa::numerics::matmul_tile_reference(&[[0f32; 32]; 32], &a, &b, &[0])
        .expect("small integers are exact");
    for (what, in_fmt) in [
        ("FP32 in L1", L1Format::Fp32),
        ("BF16 in L1", L1Format::Bf16),
    ] {
        let got = run_tile(&a, &b, in_fmt, matmul::BF16_CODE);
        for (k, (g, w)) in got.iter().zip(flat(&model).iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what}: C[{}][{}] is {g}, want {w}",
                k / 32,
                k % 32
            );
        }
    }
}

// --- Multi-tile ----------------------------------------------------------------

/// `C = A @ B` on the device for row-major `a` (`m` x `k`) and `b` (`k` x `n`),
/// through `tt_layout`'s padding, [`matmul::matmul_roles`] and de-tiling.
fn device_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    in_fmt: L1Format,
    out_fmt: u32,
) -> Vec<f32> {
    let staged = matmul::stage_matmul(a, b, m, k, n, in_fmt);
    let [unpack, math, pack] = matmul::matmul_roles(&staged.outputs, in_fmt, out_fmt);
    let path = std::env::temp_dir().join(format!(
        "ttmm-{}-{:?}.bin",
        std::process::id(),
        std::thread::current().id()
    ));
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &math,
                pack: &pack,
            })
            .concurrent(&matmul::TILE_SEMAPHORES)
            .stage(&[(matmul::MATMUL_STAGE, &staged.a), (staged.b_at, &staged.b)])
            .dump_rows(0)
            .read_back(&[(matmul::MATMUL_OUT, staged.out_bytes())]),
        );
        std::fs::write(&path, &out.l1[0]).unwrap();
    });
    let packed = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    matmul::detilize_packed(&packed, staged.m, staged.n)
}

/// Small integers, `|A| <= 127` and `|B| <= 31` as for one tile: every
/// product and every sum up to `k = 128` is an exact FP32 integer, so the
/// device must match the integer product bit for bit, in any order.
fn int_matrix(rng: &mut Lcg, rows: usize, cols: usize, bound: i32) -> Vec<f32> {
    (0..rows * cols).map(|_| rng.int(bound)).collect()
}

fn exact_product(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let s: i64 = (0..k)
                .map(|kk| a[i * k + kk] as i64 * b[kk * n + j] as i64)
                .sum();
            assert!(s.unsigned_abs() < 1 << 24, "the product must stay exact");
            c[i * n + j] = s as f32;
        }
    }
    c
}

fn assert_matmul(m: usize, k: usize, n: usize, in_fmt: L1Format, out_fmt: u32, seed: u64) {
    let mut rng = Lcg(seed);
    let a = int_matrix(&mut rng, m, k, 127);
    let b = int_matrix(&mut rng, k, n, 31);
    let want = exact_product(&a, &b, m, k, n);
    let got = device_matmul(&a, &b, m, k, n, in_fmt, out_fmt);
    assert_eq!(got.len(), m * n);
    for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "[{m}x{k}] @ [{k}x{n}] {in_fmt:?}: C[{}][{}] is {g}, want {w}",
            idx / n,
            idx % n
        );
    }
}

/// `K` depth: one output tile accumulating two and four tile pairs, the
/// unpacker re-pointed at each pair's images between them and `Dst` never
/// cleared in between.
#[test]
fn k_depth_accumulates_across_tiles() {
    assert_matmul(32, 64, 32, L1Format::Fp32, TF32_CODE, 1);
    assert_matmul(32, 128, 32, L1Format::Fp32, TF32_CODE, 2);
}

/// `M` x `N` output tiles, each cleared only once the packer has finished the
/// last (`matmul::DST_FREE`) and packed to its own place.
#[test]
fn m_by_n_output_tiles() {
    assert_matmul(64, 32, 96, L1Format::Fp32, TF32_CODE, 3);
}

/// Shapes that are not multiples of the tile, padded with zeros by `tt_layout`
/// on the way in and cropped on the way out.
#[test]
fn awkward_shapes_are_padded_and_cropped() {
    assert_matmul(13, 47, 29, L1Format::Fp32, TF32_CODE, 4);
    assert_matmul(1, 64, 96, L1Format::Fp32, TF32_CODE, 5);
}

/// The sweep: shapes, both `Src` formats by both routes, depths one to three,
/// against the exact product. Deterministic; the simulator gate.
#[test]
fn a_shape_format_and_depth_sweep() {
    let shapes = [(32, 32, 32), (40, 70, 33), (64, 96, 32), (17, 32, 64)];
    let formats = [
        (L1Format::Fp32, TF32_CODE),
        (L1Format::Fp32, matmul::BF16_CODE),
        (L1Format::Bf16, matmul::BF16_CODE),
    ];
    for (s, &(m, k, n)) in shapes.iter().enumerate() {
        for (f, &(in_fmt, out_fmt)) in formats.iter().enumerate() {
            assert_matmul(m, k, n, in_fmt, out_fmt, 100 + (s * 3 + f) as u64);
        }
    }
}
