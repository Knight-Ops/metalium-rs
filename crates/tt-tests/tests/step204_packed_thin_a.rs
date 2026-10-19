//! decode-to-bandwidth.md, `A` once a product: a one-row product over a long `K` packs its `A`
//! (`tensor::pack_row`, `[1, K]` laid out `[K / 32, 32]`) and every unit reads `A` as `K / 1024`
//! whole tiles, not `K / 32` thin reads an entry each. The kernel unpacks `K` tile `kk`'s faces
//! from row `kk % 16` of the packed faces (`matmul::packed_a_pairs`), so `SrcB`'s row 0 holds the
//! same values.
//!
//! The claims:
//! - **The packing.** `pack_row` of a `[1, K]` row is the host's `[K / 32, 32]` reshape, bit for
//!   bit, at `K` = 1024, 1056 (a last packed tile of one row) and 8192.
//! - **Same bits.** A product with a packed `A` is the same product without (`set_packed_thin_a`
//!   off: each unit reads `A` thin), bit for bit in its one row, for `K` of 1024, 1056, 2048 and
//!   8192, whole and K-blocked (the packed tiles of each block's `K` range), `B` plain and
//!   transposed, `N` ragged (and `K` = 8192 unlimited: two packed blocks of 128 against four thin
//!   blocks of 64, each continuing the last's partial), with finite special values (signed
//!   zeros, subnormals, the smallest normal) in `A`. Before each product, another product of a
//!   different `A` fills the same L1 slots, so a packed row read at the wrong place lands on
//!   another matrix's values.
//! - **Streamed or not.** With `K` streamed in chunks (`set_streamed_k`, the default: each
//!   column's chunks double-buffered, its partial passed through L1), and without (the K
//!   blocks above), the same bits.
//! - **An honest claim.** The output's padding rows hold later packed rows' products: its claim
//!   is `Undefined`.
//!
//! The unit test `a_packed_a_is_read_in_k_over_1024_tiles` (tt-kernels) counts the reads.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{self, Pad};
#[cfg(not(feature = "silicon"))]
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

/// `floats`, with special values at a few places along `K`: finite ones, none so large that it
/// swamps the rest, so every output keeps each chunk's contribution in its bits (an infinity or
/// NaN in `A` would reach every output column whichever chunks were summed).
fn with_specials(seed: u64, n: usize) -> Vec<f32> {
    let mut v = floats(seed, n);
    let specials = [
        -0.0,
        0.0,
        f32::from_bits(1),
        -f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        300.0,
        -300.0,
    ];
    for (i, x) in specials.into_iter().enumerate() {
        v[(i * 997 + 13) % n] = x;
    }
    v
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
            TileChoice::All,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// The packing alone, against the host's reshape.
#[test]
fn a_packed_row_is_the_rows_reshape() {
    with_session(|s| {
        for k in [1024, 1056, 8192] {
            let x = with_specials(k as u64, k);
            let t = s.upload(&x, 1, k).unwrap();
            let packed = s.pack_row(&t).unwrap();
            assert_eq!([packed.rows, packed.cols], [k / 32, 32]);
            // Rows past `K / 32` are not written; a packing with none has no padding to claim.
            if (k / 32) % 32 != 0 {
                assert_eq!(packed.pad(), Pad::Undefined, "K {k}: the claim");
            }
            assert_bits(&s.download(&packed).unwrap(), &x, &format!("K {k}"));
            for t in [t, packed] {
                s.free(t).unwrap();
            }
            println!("K {k}: packed bit for bit");
        }
    });
}

/// `[1, k] @ op(B)` for `B` stored `[n, k]` when `b_t`, else `[k, n]`.
struct Case {
    k: usize,
    n: usize,
    b_t: bool,
    k_block: Option<usize>,
}

const CASES: &[Case] = &[
    Case {
        k: 1024,
        n: 96,
        b_t: false,
        k_block: None,
    },
    Case {
        k: 1056,
        n: 40,
        b_t: true,
        k_block: None,
    },
    Case {
        k: 2048,
        n: 200,
        b_t: false,
        k_block: None,
    },
    Case {
        k: 8192,
        n: 64,
        b_t: false,
        k_block: Some(64),
    },
    Case {
        k: 2048,
        n: 64,
        b_t: true,
        k_block: Some(48),
    },
    // `K` too long for one block: a packed `A` fits two of 128 (`down`'s), a thin one four of 64.
    Case {
        k: 8192,
        n: 40,
        b_t: true,
        k_block: None,
    },
];

/// The products, case by case.
#[test]
fn a_packed_a_gives_the_unpacked_products_bits() {
    with_session(|s| {
        for (i, c) in CASES.iter().enumerate() {
            let &Case { k, n, b_t, k_block } = c;
            let label = format!("[1, {k}] @ [{k}, {n}] b_t {b_t} k_block {k_block:?}");
            let (rows, cols) = if b_t { (n, k) } else { (k, n) };
            let a = s.upload(&with_specials(10 + i as u64, k), 1, k).unwrap();
            let other = s.upload(&floats(20 + i as u64, k), 1, k).unwrap();
            let b = s
                .upload(&floats(30 + i as u64, rows * cols), rows, cols)
                .unwrap();
            s.set_matmul_k_block_limit(k_block.and_then(std::num::NonZeroUsize::new));
            let product = |s: &mut Session<_>, x: &tensor::DramTensor, packed, streamed| {
                s.set_packed_thin_a(packed);
                s.set_streamed_k(streamed);
                let out = s.matmul_dram(x, false, &b, b_t, ROUTE, Fidelity::HiFi4, BUDGET);
                s.set_packed_thin_a(true);
                s.set_streamed_k(true);
                out.unwrap_or_else(|e| panic!("{label}: {e}"))
            };
            let thin = product(s, &a, false, false);
            let want = s.download(&thin).unwrap();
            for streamed in [false, true] {
                // Another `A`'s product through the same slots, then the packed one.
                let primed = product(s, &other, true, streamed);
                let got = product(s, &a, true, streamed);
                let label = format!("{label} streamed {streamed}");
                assert_bits(&s.download(&got).unwrap(), &want, &label);
                assert_eq!(got.pad(), Pad::Undefined, "{label}: the claim");
                for t in [primed, got] {
                    s.free(t).unwrap();
                }
                println!("{label}: bit for bit");
            }
            s.set_matmul_k_block_limit(None);
            for t in [thin, a, other, b] {
                s.free(t).unwrap();
            }
        }
    });
}
