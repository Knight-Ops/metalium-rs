//! decode-matmul-bandwidth P1: a GDDR matmul reads only the rows of `op(A)`'s
//! last tile row it uses (`tt_isa::dm::record::gather_rows`, at most 16), not
//! whole tiles. A decode step's products are `m = 1`, so each `A` tile was 32
//! rows read for one used.
//!
//! The claims:
//! - **Same bits.** Every valid output equals the whole-tile read's, bit for
//!   bit, across `m` of 1, 2, 15 and 16 (and 17 and 33, where the last tile row
//!   still qualifies or not), `K` whole, ragged and blocked, ragged `N`, and `B`
//!   transposed. Before each thin product, a whole-tile product of a different
//!   `A` whose padding rows hold NaN fills the same L1 slots, so a read of too
//!   few rows, or at the wrong place, lands on another matrix's values.
//! - **Fewer rows read.** `A`'s padding rows in GDDR hold NaN: whole tiles carry
//!   it into the output's padding rows; the thin read, which ends at row `r` of
//!   face 1, never sees rows 16 to 31 (faces 2 and 3).
//! - **An honest claim.** The thin product's padding rows hold whatever L1 had
//!   (here non-finite), and its padding claim is `Undefined`; ops after it
//!   (a sum over rows, a max over columns, a product with it as `A`, and
//!   `C^T @ C`, whose `K` runs along those rows in both operands) equal the same
//!   ops after the whole-tile product. The negative control: claiming that
//!   padding zero skips the refill, and `C^T @ C` turns non-finite. (One operand
//!   with zero padding along `K` would hide it: the Matrix Unit's product with
//!   zero is zero, whatever the other holds.)

use tt_device::tlb::WindowKind;
use tt_device::Window;
use tt_isa::dm::{face_index, TILE_DATA, TILE_SLOT};
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{DramTensor, Pad};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;
const FIDELITY: Fidelity = Fidelity::HiFi4;

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
    with_tiles(TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1), f)
}

#[cfg(not(feature = "silicon"))]
fn with_tiles(tiles: TileChoice, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, tiles, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    with_tiles(TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1), f)
}

#[cfg(feature = "silicon")]
fn with_tiles(tiles: TileChoice, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            tiles,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// The datums of `t`'s last tile row past its `rows`, as stored in GDDR.
fn padding_rows<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &Window,
    t: &DramTensor,
) -> Vec<f32> {
    let from = t.rows - (t.grid()[0] - 1) * 32;
    rows_from(s, w, t, from)
}

/// The datums of rows `from..32` of `t`'s last tile row, as stored in GDDR.
fn rows_from<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &Window,
    t: &DramTensor,
    from: usize,
) -> Vec<f32> {
    // Uploads and products are queued: read what they wrote, not what was there.
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

/// Write NaN into `t`'s padding rows in GDDR (its last tile row, past
/// `t.rows`), and say so in its claim.
fn poison_padding_rows<T: tt_device::Transport>(s: &mut Session<T>, w: &Window, t: &DramTensor) {
    // After the queued upload, or it writes over the poison.
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

/// `[m, k] @ op(B)` for `B` stored `[n, k]` when `b_t`, else `[k, n]`.
struct Case {
    m: usize,
    k: usize,
    n: usize,
    b_t: bool,
    k_block: Option<usize>,
}

const CASES: &[Case] = &[
    Case {
        m: 1,
        k: 64,
        n: 64,
        b_t: false,
        k_block: None,
    },
    Case {
        m: 2,
        k: 96,
        n: 45,
        b_t: false,
        k_block: None,
    },
    Case {
        m: 15,
        k: 64,
        n: 64,
        b_t: true,
        k_block: None,
    },
    Case {
        m: 16,
        k: 128,
        n: 32,
        b_t: false,
        k_block: None,
    },
    Case {
        m: 17,
        k: 64,
        n: 40,
        b_t: false,
        k_block: None,
    },
    Case {
        m: 33,
        k: 64,
        n: 64,
        b_t: false,
        k_block: None,
    },
    Case {
        m: 1,
        k: 192,
        n: 64,
        b_t: false,
        k_block: Some(2),
    },
    Case {
        m: 3,
        k: 70,
        n: 40,
        b_t: false,
        k_block: None,
    },
];

/// Claims one and two, case by case.
#[test]
fn thin_reads_equal_whole_tiles_and_skip_the_padding_rows() {
    with_session(|s| {
        let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
        for (i, c) in CASES.iter().enumerate() {
            let &Case {
                m,
                k,
                n,
                b_t,
                k_block,
            } = c;
            let label = format!("[{m}, {k}] @ [{k}, {n}] b_t {b_t} k_block {k_block:?}");
            let thin = tt_kernels::tensor::thin_rows(m, false);
            let (bs, bshape) = if b_t {
                ([n, k], (n, k))
            } else {
                ([k, n], (k, n))
            };
            let a = s.upload(&floats(100 + i as u64, m * k), m, k).unwrap();
            let other = s.upload(&floats(200 + i as u64, m * k), m, k).unwrap();
            let clean = s.upload(&floats(300 + i as u64, m * k), m, k).unwrap();
            let b = s
                .upload(&floats(400 + i as u64, bs[0] * bs[1]), bshape.0, bshape.1)
                .unwrap();
            poison_padding_rows(s, &w, &a);
            poison_padding_rows(s, &w, &other);
            s.set_matmul_k_block_limit(k_block.and_then(std::num::NonZeroUsize::new));
            let product = |s: &mut Session<_>, x: &DramTensor, thin: bool| {
                s.set_thin_matmul_reads(thin);
                let out = s.matmul_dram(x, false, &b, b_t, ROUTE, FIDELITY, BUDGET);
                s.set_thin_matmul_reads(true);
                out.unwrap_or_else(|e| panic!("{label}: {e}"))
            };
            // Whole tiles: the reference bits, and `A`'s NaN padding reaching
            // the output's padding rows wherever `K` is whole (a ragged `K`
            // refills `A`'s padding first).
            let whole = product(s, &a, false);
            let want = s.download(&whole).unwrap();
            if k % 32 == 0 && m % 32 != 0 {
                // Every row computed (`step203_live_rows` covers the rows left out), so the
                // read's padding rows reach the output's.
                tt_kernels::matmul::set_trim_rows(false);
                let control = product(s, &a, false);
                tt_kernels::matmul::set_trim_rows(true);
                assert!(
                    padding_rows(s, &w, &control).iter().any(|x| !x.is_finite()),
                    "{label}: control: whole tiles carry A's padding rows"
                );
                s.free(control).unwrap();
            }
            // Claim 1: L1 primed with another matrix and NaN rows, then thin.
            let primed = product(s, &other, false);
            let got = product(s, &a, true);
            assert_bits(&s.download(&got).unwrap(), &want, &label);
            assert_eq!(
                got.pad(),
                if thin != 0 {
                    Pad::Undefined
                } else {
                    whole.pad()
                },
                "{label}: claim"
            );
            // Claim 2: L1 primed with a matrix whose padding rows are zero; a thin
            // read leaves rows 16 to 31, so `A`'s NaN never reaches them.
            if thin != 0 && k % 32 == 0 {
                let zeroed = product(s, &clean, false);
                let thin_out = product(s, &a, true);
                assert!(
                    rows_from(s, &w, &thin_out, 16)
                        .iter()
                        .all(|x| x.is_finite()),
                    "{label}: the thin read reached A's rows 16 to 31"
                );
                assert_bits(&s.download(&thin_out).unwrap(), &want, &label);
                for t in [zeroed, thin_out] {
                    s.free(t).unwrap();
                }
            }
            s.set_matmul_k_block_limit(None);
            for t in [whole, primed, got, a, other, clean, b] {
                s.free(t).unwrap();
            }
            println!("{label}: thin rows {thin}: bit for bit");
        }
    });
}

/// Claim three: what follows a thin product with stale, non-finite padding
/// rows sees nothing of them, and claiming them zero is what would let them in.
#[test]
fn ops_after_a_thin_product_ignore_its_padding_rows() {
    with_session(|s| {
        let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
        let (m, k, n) = (15, 64, 64);
        let a = s.upload(&floats(1, m * k), m, k).unwrap();
        let other = s.upload(&floats(2, m * k), m, k).unwrap();
        let b = s.upload(&floats(3, k * n), k, n).unwrap();
        let wt = s.upload(&floats(5, n * 48), n, 48).unwrap();
        poison_padding_rows(s, &w, &other);
        let c = |s: &mut Session<_>, thin: bool| {
            s.set_thin_matmul_reads(false);
            let primed = s.matmul_dram(&other, false, &b, false, ROUTE, FIDELITY, BUDGET);
            s.free(primed.unwrap()).unwrap();
            s.set_thin_matmul_reads(thin);
            let out = s.matmul_dram(&a, false, &b, false, ROUTE, FIDELITY, BUDGET);
            s.set_thin_matmul_reads(true);
            out.unwrap()
        };
        let follow = |s: &mut Session<_>, c: &DramTensor| -> Vec<Vec<f32>> {
            let outs = [
                s.reduce(c, ReduceOp::Sum, Axis::Rows).unwrap(),
                s.reduce(c, ReduceOp::Max, Axis::Cols).unwrap(),
                s.matmul_dram(c, false, &wt, false, ROUTE, FIDELITY, BUDGET)
                    .unwrap(),
                // Last: its refill rewrites `c`'s padding in place.
                s.matmul_dram(c, true, c, false, ROUTE, FIDELITY, BUDGET)
                    .unwrap(),
            ];
            outs.into_iter()
                .map(|t| {
                    let v = s.download(&t).unwrap();
                    s.free(t).unwrap();
                    v
                })
                .collect()
        };
        let whole = c(s, false);
        let want = follow(s, &whole);
        let thin = c(s, true);
        assert_eq!(thin.pad(), Pad::Undefined);
        assert!(
            padding_rows(s, &w, &thin).iter().any(|v| !v.is_finite()),
            "the thin product's padding rows are stale (NaN from the primed L1)"
        );
        for (i, (got, want)) in follow(s, &thin).iter().zip(&want).enumerate() {
            assert_bits(got, want, &format!("follower {i}"));
        }
        // Negative control: the same product with its padding claimed zero
        // skips the refill, and `C^T @ C` sums the stale rows into valid outputs.
        let lie = c(s, true);
        lie.set_pad(Pad::Zero);
        let bad = s
            .matmul_dram(&lie, true, &lie, false, ROUTE, FIDELITY, BUDGET)
            .unwrap();
        assert!(
            s.download(&bad).unwrap().iter().any(|v| !v.is_finite()),
            "control: a false zero claim lets the stale rows into C^T @ C"
        );
        for t in [bad, whole, thin, lie, a, other, b, wt] {
            s.free(t).unwrap();
        }
    });
}

/// decode-matmul-bandwidth P2: a one-tile-row product's columns are dealt so no
/// unit holds more than its share (`tensor::thin_runs`): `[1, 2048] @ [2048,
/// 192]` on two units, whose plan is two columns a job, was three jobs of two
/// (unit 0 four columns) and is two of two and two of one (three each). Each
/// output tile is the same accumulation in any job, so the product on two units
/// equals the one on one unit (`Spread::Fixed(1)`, where the deal is `blocks`'
/// three jobs of two) bit for bit, thin reads on and off, and with `B`
/// transposed. (The host-staged `Session::matmul` blocks `K` its own way, so it is
/// not the same accumulation at this `K`.)
#[test]
fn columns_dealt_evenly_equal_one_units_product() {
    use tt_kernels::matmul::Staging;
    use tt_kernels::session::Spread;
    let (m, k, n) = (1, 2048, 192);
    let plan = tt_kernels::matmul::plan_in([m, k, n], ROUTE, FIDELITY, Staging::Slots).unwrap();
    assert_eq!(
        plan.tiles,
        [1, 64, 2],
        "the plan's width this case is built on"
    );
    let runs = tt_kernels::tensor::thin_runs([1, 64, 6], [64, 2], Staging::Slots, 2).unwrap();
    assert_eq!(runs, [0..2, 2..4, 4..5, 5..6]);
    assert_eq!(tt_kernels::tensor::blocks([1, 6], [1, 2], 2), [1, 2]);
    with_tiles(TileChoice::Count(2), |s| {
        let (av, bv) = (floats(7, m * k), floats(8, k * n));
        let a = s.upload(&av, m, k).unwrap();
        let b = s.upload(&bv, k, n).unwrap();
        let mut bt = vec![0f32; k * n];
        for r in 0..k {
            for c in 0..n {
                bt[c * k + r] = bv[r * n + c];
            }
        }
        let b_t = s.upload(&bt, n, k).unwrap();
        for (b, transposed) in [(&b, false), (&b_t, true)] {
            let product = |s: &mut Session<_>, spread: Spread, thin: bool| {
                s.set_spread(spread);
                s.set_thin_matmul_reads(thin);
                let c = s
                    .matmul_dram(&a, false, b, transposed, ROUTE, FIDELITY, BUDGET)
                    .unwrap();
                let v = s.download(&c).unwrap();
                s.free(c).unwrap();
                v
            };
            let want = product(s, Spread::Fixed(1), false);
            for thin in [true, false] {
                let what = format!("thin {thin}, B transposed {transposed}");
                assert_bits(&product(s, Spread::All, thin), &want, &what);
            }
        }
        s.set_thin_matmul_reads(true);
        for t in [a, b, b_t] {
            s.free(t).unwrap();
        }
    });
}
