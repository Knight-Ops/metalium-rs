//! decode-to-bandwidth.md, compute under the reads (its first step): a product's last tile row
//! computes only the rows it holds (`tt_kernels::matmul::LiveRows`). An `m = 1` decode product
//! multiplied all four faces of every tile pair (64 `MVMUL`s at HiFi4) for one row: rows 0-7
//! need `A`'s faces `(0, k)` and the first half of `SrcB`, 16; rows 0-15 need both halves.
//!
//! The claims, each against the same product with the trim off (`set_trim_rows(false)`, the
//! kernel every gate before this one held to its oracles):
//! - **Same bits.** Every valid output is the untrimmed product's, bit for bit, across `m` of 1,
//!   3, 8 (rows 0-7), 9, 15, 16 (rows 0-15), 17 and 33 (every row), `K` whole, ragged and
//!   K-blocked (an accumulator reloaded), ragged `N`, `B` transposed, at HiFi4 and LoFi, and for
//!   packed BF16 operands.
//! - **The rows left out are zero.** `A`'s padding rows in GDDR hold NaN and `A` is read as whole
//!   tiles, so the untrimmed product carries NaN into its rows past `m` (the control: those rows
//!   were computed). Trimmed, rows past the computed eight or sixteen are `+0.0`, which every
//!   padding claim allows, and the claim is unchanged.

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::dm::{face_index, TILE_DATA, TILE_SLOT};
use tt_kernels::matmul::{set_trim_rows, Fidelity, LiveRows, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Pad};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        // The other paths' gate: products of several tile rows would otherwise run blocked
        // (`docs/plans/prefill-operand-reuse.md`; `step229` gates those).
        s.set_block_stream(tt_kernels::tensor::BlockStream::Off);
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        // The other paths' gate: products of several tile rows would otherwise run blocked
        // (`docs/plans/prefill-operand-reuse.md`; `step229` gates those).
        s.set_block_stream(tt_kernels::tensor::BlockStream::Off);
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// The datums of rows `from..32` of `t`'s last tile row, as stored in GDDR.
fn rows_from<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &Window,
    t: &DramTensor,
    from: usize,
) -> Vec<f32> {
    s.sync().unwrap();
    let [rt, ct] = t.grid();
    let mut out = Vec::new();
    for j in 0..ct {
        let mut image = vec![0u8; TILE_SLOT as usize];
        s.device()
            .dram_read(w, t.tile(rt - 1, j), &mut image)
            .unwrap();
        for r in from..32 {
            for c in 0..32 {
                let at = TILE_DATA as usize + 4 * face_index(r, c);
                out.push(f32::from_le_bytes(image[at..at + 4].try_into().unwrap()));
            }
        }
    }
    out
}

/// NaN in `t`'s padding rows in GDDR, and the claim saying so.
fn poison_padding_rows<T: tt_device::Transport>(s: &mut Session<T>, w: &Window, t: &DramTensor) {
    s.sync().unwrap();
    let [rt, ct] = t.grid();
    for j in 0..ct {
        let slot = t.tile(rt - 1, j);
        let mut image = vec![0u8; TILE_SLOT as usize];
        s.device().dram_read(w, slot, &mut image).unwrap();
        for r in (t.rows - (rt - 1) * 32)..32 {
            for c in 0..32 {
                let at = TILE_DATA as usize + 4 * face_index(r, c);
                image[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
        }
        s.device().dram_write(w, slot, &image).unwrap();
    }
    t.set_pad(Pad::Undefined);
}

/// The first row of the last tile row a trimmed kernel leaves out, if it leaves any: only a job
/// of that tile row alone trims, and these shapes put a product of more than one tile row in one
/// job (`m` = 33 computes every row).
fn first_left_out(m: usize) -> Option<usize> {
    if m > 32 {
        return None;
    }
    match LiveRows::for_job(m, 0, 1) {
        LiveRows::Eight => Some(8),
        LiveRows::Sixteen => Some(16),
        LiveRows::All => None,
    }
}

/// `[m, k] @ op(B)` for `B` stored `[n, k]` when `b_t`, else `[k, n]`.
struct Case {
    m: usize,
    k: usize,
    n: usize,
    b_t: bool,
    k_block: Option<usize>,
    fidelity: Fidelity,
}

const fn case(m: usize, k: usize, n: usize, b_t: bool, k_block: Option<usize>) -> Case {
    Case {
        m,
        k,
        n,
        b_t,
        k_block,
        fidelity: Fidelity::HiFi4,
    }
}

const CASES: &[Case] = &[
    case(1, 64, 64, false, None),
    case(3, 96, 45, false, None),
    case(8, 64, 64, true, None),
    case(9, 64, 40, false, None),
    case(15, 70, 64, false, None),
    case(16, 128, 32, true, None),
    case(17, 64, 64, false, None),
    case(33, 64, 64, false, None),
    case(1, 192, 64, false, Some(2)),
    case(1, 192, 40, true, Some(2)),
    Case {
        fidelity: Fidelity::Lo,
        ..case(1, 64, 64, false, None)
    },
    Case {
        fidelity: Fidelity::Lo,
        ..case(12, 64, 64, false, None)
    },
];

/// Both claims for F32 operands, case by case.
#[test]
fn a_last_tile_row_computes_only_its_rows() {
    with_session(|s| {
        let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
        // Whole tiles, so `A`'s NaN padding rows reach the untrimmed product's padding rows.
        s.set_thin_matmul_reads(false);
        for (i, c) in CASES.iter().enumerate() {
            let &Case {
                m,
                k,
                n,
                b_t,
                k_block,
                fidelity,
            } = c;
            let label =
                format!("[{m}, {k}] @ [{k}, {n}] b_t {b_t} k_block {k_block:?} {fidelity:?}");
            let (rows, cols) = if b_t { (n, k) } else { (k, n) };
            let a = s.upload(&floats(10 + i as u64, m * k), m, k).unwrap();
            let b = s
                .upload(&floats(20 + i as u64, rows * cols), rows, cols)
                .unwrap();
            poison_padding_rows(s, &w, &a);
            s.set_matmul_k_block_limit(k_block.and_then(std::num::NonZeroUsize::new));
            let product = |s: &mut Session<_>, trim: bool| {
                set_trim_rows(trim);
                let out = s.matmul_dram(&a, false, &b, b_t, ROUTE, fidelity, BUDGET);
                set_trim_rows(true);
                out.unwrap_or_else(|e| panic!("{label}: {e}"))
            };
            let whole = product(s, false);
            let want = s.download(&whole).unwrap();
            let trimmed = product(s, true);
            assert_bits(&s.download(&trimmed).unwrap(), &want, &label);
            assert_eq!(trimmed.pad(), whole.pad(), "{label}: the claim");
            if let Some(from) = first_left_out(m) {
                // A ragged `K` refills `A`'s padding before the product, so only a whole one
                // carries the NaN through.
                if k % 32 == 0 {
                    assert!(
                        rows_from(s, &w, &whole, from)
                            .iter()
                            .any(|x| !x.is_finite()),
                        "{label}: control: the untrimmed product computes rows {from} on"
                    );
                }
                assert!(
                    rows_from(s, &w, &trimmed, from)
                        .iter()
                        .all(|x| x.to_bits() == 0),
                    "{label}: rows {from} on computed"
                );
            }
            s.set_matmul_k_block_limit(None);
            for t in [whole, trimmed, a, b] {
                s.free(t).unwrap();
            }
            println!("{label}: bit for bit");
        }
    });
}

/// Same bits for packed BF16 operands (their own job builder), whole `K` and K-blocked.
#[test]
fn a_bf16_product_computes_only_its_rows() {
    with_session(|s| {
        let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
        let bits = |seed: u64, n: usize| -> Vec<u16> {
            floats(seed, n)
                .iter()
                .map(|x| (x.to_bits() >> 16) as u16)
                .collect()
        };
        for (m, k, n, k_block) in [(1, 128, 64, None), (5, 96, 40, None), (1, 192, 64, Some(2))] {
            let label = format!("BF16 [{m}, {k}] @ [{k}, {n}] k_block {k_block:?}");
            let a = s.upload_bf16(&bits(1, m * k), m, k).unwrap();
            let b = s.upload_bf16(&bits(2, k * n), k, n).unwrap();
            s.set_matmul_k_block_limit(k_block.and_then(std::num::NonZeroUsize::new));
            let product = |s: &mut Session<_>, trim: bool| {
                set_trim_rows(trim);
                let out = s.matmul_bf16(&a, &b, Fidelity::HiFi4, BUDGET);
                set_trim_rows(true);
                out.unwrap_or_else(|e| panic!("{label}: {e}"))
            };
            let whole = product(s, false);
            let want = s.download(&whole).unwrap();
            let trimmed = product(s, true);
            assert_bits(&s.download(&trimmed).unwrap(), &want, &label);
            let from = first_left_out(m).expect("a thin product");
            assert!(
                rows_from(s, &w, &trimmed, from)
                    .iter()
                    .all(|x| x.to_bits() == 0),
                "{label}: rows {from} on computed"
            );
            s.set_matmul_k_block_limit(None);
            for t in [whole, trimmed] {
                s.free(t).unwrap();
            }
            println!("{label}: bit for bit");
        }
    });
}
