//! A batched matmul whose products are not whole tile rows, over blocks that are not either
//! (`docs/completed-plans/attention-jobs.md`, P1 and P2).
//!
//! `Session::matmul_dram_batched` refused a batch whose `m` was not whole tiles, and burn-tt then
//! ran attention's products one head at a time (two element repacks and a matmul a head, then a row
//! gather): 56 of a SmolLM2 decode step's 64 attention jobs. Now each product's output starts on a
//! tile row of its own and one row gather stacks them, and an operand whose blocks are whole rows of
//! it but do not start on tile rows (heads stacked in a `[h s, d]` buffer) is gathered once into a
//! block per whole tile rows, each block's tail rows zero.
//!
//! The claim: every product is bit for bit `Session::matmul_dram` on its block uploaded alone, for
//! - ragged `m` over blocks read where they lie (ViT's heads: column blocks of one projection);
//! - blocks stacked by rows, as `Q K^T` and `P V` of a decode step (m = 1) and a prefill (m = 5),
//!   over sequence lengths 1, 31, 32, 33 and 70, `K^T` read transposed;
//! - a block that is the product's `K` along its rows (`V`, and a transposed left operand), whose
//!   tail rows must be zero -- the next block's rows lie there in the source;
//! - an untransposed left block's tail rows, which a stacked output leaves out, are not read at
//!   all (`tensor::GATHER_HOLE`): each check first fills the gather's staging with NaN;
//! - one block shared by several products (grouped heads).
//!
//! Watched failing (2026-10-14, ttsim): the stacking gather reading product `i` from row `i m` (all
//! three tests); a padded block placed one tile row off (all three); the tail rows reading the
//! source's next row instead of the zero row (`shared_blocks_and_a_transposed_k_side`, where both
//! operands' `K` tails would be junk). With only `V`'s tail junk, `P`'s zero `K` padding meets it,
//! and the Matrix Unit gave a zero even against an infinity there, so that case alone cannot see it.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::Block;
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

/// The `[r, c]` block at `[r0, c0]` of a row-major `[_, cols]` matrix, transposed if asked.
fn block(v: &[f32], cols: usize, [r0, c0]: [usize; 2], [r, c]: [usize; 2], t: bool) -> Vec<f32> {
    let b: Vec<f32> = (r0..r0 + r)
        .flat_map(|i| v[i * cols + c0..i * cols + c0 + c].iter().copied())
        .collect();
    if !t {
        return b;
    }
    (0..c)
        .flat_map(|j| (0..r).map(move |i| (i, j)))
        .map(|(i, j)| b[i * c + j])
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
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// One operand: a row-major `[rows, cols]` matrix and each product's block of it.
struct Operand<'a> {
    data: &'a [f32],
    dims: [usize; 2],
    blocks: Vec<Block>,
}

/// `op(A_i) @ op(B_i)` for every block pair, batched, against each product computed alone by
/// `matmul_dram` on its blocks uploaded as plain matrices.
fn check<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &Operand,
    b: &Operand,
    [m, k, n]: [usize; 3],
    what: &str,
) {
    // A row gather of NaN first, as wide as each operand: the staging slots a padded block's
    // gather leaves unread (an untransposed left block's tail rows, under a stacked output) hold
    // NaN, which must reach no product's rows.
    for cols in [a.dims[1], b.dims[1]] {
        let nan = s.upload(&vec![f32::NAN; 64 * cols], 64, cols).unwrap();
        let rows: Vec<(usize, usize)> = (0..64).map(|r| (0, r)).collect();
        let primed = s.gather_rows(&[&nan], &rows, cols).unwrap();
        s.free(primed).unwrap();
        s.free(nan).unwrap();
    }
    let da = s.upload(a.data, a.dims[0], a.dims[1]).unwrap();
    let db = s.upload(b.data, b.dims[0], b.dims[1]).unwrap();
    let items: Vec<_> = a
        .blocks
        .iter()
        .copied()
        .zip(b.blocks.iter().copied())
        .collect();
    let out = s
        .matmul_dram_batched(&da, &db, &items, [m, k, n], ROUTE, Fidelity::HiFi4, BUDGET)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!([out.rows, out.cols], [items.len() * m, n], "{what}");
    let got = s.download(&out).unwrap();
    for (i, (ba, bb)) in items.iter().enumerate() {
        // Each block as the plain matrix the product reads.
        let ea = if ba.transposed { [k, m] } else { [m, k] };
        let eb = if bb.transposed { [n, k] } else { [k, n] };
        let pa = s
            .upload(&block(a.data, a.dims[1], ba.at, ea, ba.transposed), m, k)
            .unwrap();
        let pb = s
            .upload(&block(b.data, b.dims[1], bb.at, eb, bb.transposed), k, n)
            .unwrap();
        let want = s
            .matmul_dram(&pa, false, &pb, false, ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap();
        let want = s.download(&want).unwrap();
        assert_bits(
            &got[i * m * n..(i + 1) * m * n],
            &want,
            &format!("{what}: product {i}"),
        );
    }
}

fn plain(at: [usize; 2]) -> Block {
    Block {
        at,
        transposed: false,
    }
}

fn transposed(at: [usize; 2]) -> Block {
    Block {
        at,
        transposed: true,
    }
}

/// ViT's heads: `Q_h K_h^T` with `Q`, `K` column blocks of one `[197, 3 64]` projection (read where
/// they lie, ragged at its edge), then `P_h V_h` with `P` the stacked scores (whole rows, gathered).
#[test]
fn ragged_products_over_blocks_read_in_place() {
    with_session(|s| {
        let (tokens, heads, dk) = (197, 3, 64);
        let width = heads * dk;
        let q = floats(1, tokens * width);
        let k = floats(2, tokens * width);
        let head = |h: usize| [0, h * dk];
        let qa = Operand {
            data: &q,
            dims: [tokens, width],
            blocks: (0..heads).map(|h| plain(head(h))).collect(),
        };
        let kb = Operand {
            data: &k,
            dims: [tokens, width],
            blocks: (0..heads).map(|h| transposed(head(h))).collect(),
        };
        check(s, &qa, &kb, [tokens, dk, tokens], "vit q k^t");

        let p = floats(3, heads * tokens * tokens);
        let pa = Operand {
            data: &p,
            dims: [heads * tokens, tokens],
            blocks: (0..heads).map(|h| plain([h * tokens, 0])).collect(),
        };
        let vb = Operand {
            data: &k,
            dims: [tokens, width],
            blocks: (0..heads).map(|h| plain(head(h))).collect(),
        };
        check(s, &pa, &vb, [tokens, tokens, dk], "vit p v");
    });
}

/// A decode step (m = 1) and a prefill (m = 5): `Q`, `K`, `P` and `V` heads stacked by rows, over
/// cache lengths around a tile.
#[test]
fn stacked_heads_over_ragged_sequences() {
    with_session(|s| {
        let (heads, dk) = (9, 64);
        for m in [1, 5] {
            for t in [1, 31, 32, 33, 70] {
                if m > t {
                    continue;
                }
                let what = format!("m {m}, t {t}");
                let q = floats(10 + t as u64, heads * m * dk);
                let k = floats(20 + t as u64, heads * t * dk);
                let qa = Operand {
                    data: &q,
                    dims: [heads * m, dk],
                    blocks: (0..heads).map(|h| plain([h * m, 0])).collect(),
                };
                let kb = Operand {
                    data: &k,
                    dims: [heads * t, dk],
                    blocks: (0..heads).map(|h| transposed([h * t, 0])).collect(),
                };
                check(s, &qa, &kb, [m, dk, t], &format!("{what}: q k^t"));

                let p = floats(30 + t as u64, heads * m * t);
                let pa = Operand {
                    data: &p,
                    dims: [heads * m, t],
                    blocks: (0..heads).map(|h| plain([h * m, 0])).collect(),
                };
                // `V`'s rows are the product's `K`: its tail rows must be zero, and the source's
                // next head lies there, starting with an infinity (`0 x inf` is NaN; a finite
                // neighbour times `P`'s zero padding would hide a tail that is not zero).
                let mut v = k.clone();
                for h in 0..heads {
                    v[h * t * dk] = f32::INFINITY;
                }
                let vb = Operand {
                    data: &v,
                    dims: [heads * t, dk],
                    blocks: (0..heads).map(|h| plain([h * t, 0])).collect(),
                };
                check(s, &pa, &vb, [m, t, dk], &format!("{what}: p v"));
            }
        }
    });
}

/// Three key heads shared by nine query heads, each block gathered once; and a transposed left
/// operand stacked by rows, whose rows are the product's `K`.
#[test]
fn shared_blocks_and_a_transposed_k_side() {
    with_session(|s| {
        let (heads, groups, dk, t) = (9, 3, 64, 33);
        let q = floats(40, heads * dk);
        let k = floats(41, groups * t * dk);
        let qa = Operand {
            data: &q,
            dims: [heads, dk],
            blocks: (0..heads).map(|h| plain([h, 0])).collect(),
        };
        let kb = Operand {
            data: &k,
            dims: [groups * t, dk],
            blocks: (0..heads)
                .map(|h| transposed([h / (heads / groups) * t, 0]))
                .collect(),
        };
        check(s, &qa, &kb, [1, dk, t], "grouped q k^t");

        // `A_i^T B_i`: `A` stacks `[t, 40]` blocks by rows, read transposed (`[40, t]`).
        let (count, m, n) = (3, 40, 48);
        let a = floats(42, count * t * m);
        let b = floats(43, count * t * n);
        let aa = Operand {
            data: &a,
            dims: [count * t, m],
            blocks: (0..count).map(|i| transposed([i * t, 0])).collect(),
        };
        let bb = Operand {
            data: &b,
            dims: [count * t, n],
            blocks: (0..count).map(|i| plain([i * t, 0])).collect(),
        };
        check(s, &aa, &bb, [m, t, n], "a^t b");
    });
}

/// Padding off `K` is used as it is (`docs/plans/decode-attention.md` A1): a decode step's `Q`
/// (`[heads m, d]`, `d` whole tiles) whose padding rows hold a bias (`x + b` with `b` a row: the
/// padding rows are `b`), against `K^T` blocks, queues the same ops as the clean `Q` -- no fill
/// before the product -- and every product stays bit for bit its block computed alone. With `K`
/// ragged (`d` = 40) the dirty padding is along `K` and is filled, so the products still match.
#[test]
fn padding_off_k_is_used_as_it_is() {
    use tt_kernels::kind;
    use tt_kernels::tensor::{Eltwise, Pad};
    with_session(|s| {
        let add_row = Eltwise {
            kind: kind::ADD_ROW,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let (heads, t) = (4usize, 40usize);
        for dk in [64usize, 40] {
            let q = floats(41, heads * dk);
            let bias = floats(42, dk);
            let kv = floats(43, heads * t * dk);
            let x = s.upload(&q, heads, dk).unwrap();
            let b = s.upload(&bias, 1, dk).unwrap();
            let xb = s.eltwise(add_row, &x, Some(&b)).unwrap();
            assert_eq!(xb.pad(), Pad::Undefined, "the bias is in the padding rows");
            let kk = s.upload(&kv, heads * t, dk).unwrap();
            let items: Vec<_> = (0..heads)
                .map(|h| (plain([h, 0]), transposed([h * t, 0])))
                .collect();
            let run = |s: &mut Session<_>, a: &tt_kernels::tensor::DramTensor| {
                let before = s.host_times().ops;
                let out = s
                    .matmul_dram_batched(a, &kk, &items, [1, dk, t], ROUTE, Fidelity::HiFi4, BUDGET)
                    .unwrap();
                let ops = s.host_times().ops - before;
                let v = s.download(&out).unwrap();
                s.free(out).unwrap();
                (v, ops)
            };
            let host: Vec<f32> = s.download(&xb).unwrap();
            let clean = s.upload(&host, heads, dk).unwrap();
            // A clean product first: it makes what is kept once (the zero row `K^T`'s tails read).
            let _ = run(s, &clean);
            let (_, clean_ops) = run(s, &clean);
            let (got, dirty_ops) = run(s, &xb);
            if dk % 32 == 0 {
                assert_eq!(dirty_ops, clean_ops, "d {dk}: a fill before the product");
            }
            for h in 0..heads {
                let pa = s.upload(&host[h * dk..(h + 1) * dk], 1, dk).unwrap();
                let pb = s
                    .upload(&block(&kv, dk, [h * t, 0], [t, dk], true), dk, t)
                    .unwrap();
                let want = s
                    .matmul_dram(&pa, false, &pb, false, ROUTE, Fidelity::HiFi4, BUDGET)
                    .unwrap();
                let want = s.download(&want).unwrap();
                assert_bits(
                    &got[h * t..(h + 1) * t],
                    &want,
                    &format!("d {dk}: product {h}"),
                );
            }
            for x in [x, b, xb, kk, clean] {
                s.free(x).unwrap();
            }
        }
    });
}
