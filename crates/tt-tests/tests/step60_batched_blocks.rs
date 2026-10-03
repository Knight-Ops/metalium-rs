//! B6 gate: a batched matmul over blocks of resident tensors, and a block
//! copy, both on the card -- attention's head split without a copy.
//!
//! A projection `[b s, h dk]` holds every head's queries: head `(i, j)` is
//! the `[s, dk]` block at `[i s, j dk]`. `Session::matmul_dram_batched`
//! gathers each block where it lies (a record's tensor ref moved to the
//! block's first tile, the row stride kept) and writes each product to its
//! own tile rows of one output. The claim: every product is bit for bit what
//! `Session::matmul_dram` computes on that block uploaded on its own, and
//! `Session::copy_blocks` is bit for bit the host's rearrangement of the
//! blocks, placed anywhere. Watched failing with the block offset dropped from
//! the refs.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{Block, BlockMove};
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

/// The `[r, c]` block at `[r0, c0]` of a row-major `[_, cols]` matrix.
fn block(v: &[f32], cols: usize, [r0, c0]: [usize; 2], [r, c]: [usize; 2]) -> Vec<f32> {
    (r0..r0 + r)
        .flat_map(|i| v[i * cols + c0..i * cols + c0 + c].iter().copied())
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Attention's two products over heads, `Q K^T` and `P V`, each head's
/// operands blocks of one `[b s, h dk]` projection (`P` a stack of `[s, s]`
/// blocks), against each head's product computed alone.
#[test]
fn a_batched_matmul_over_head_blocks_is_each_heads_product() {
    with_session(|s| {
        let (batch, seq, heads, dk) = (2, 64, 2, 32);
        let width = heads * dk;
        let q = floats(1, batch * seq * width);
        let k = floats(2, batch * seq * width);
        let dq = s.upload(&q, batch * seq, width).unwrap();
        let dkk = s.upload(&k, batch * seq, width).unwrap();
        let at = |i: usize, j: usize| [i * seq, j * dk];
        let items: Vec<(Block, Block)> = (0..batch)
            .flat_map(|i| (0..heads).map(move |j| (i, j)))
            .map(|(i, j)| {
                (
                    Block {
                        at: at(i, j),
                        transposed: false,
                    },
                    Block {
                        at: at(i, j),
                        transposed: true,
                    },
                )
            })
            .collect();
        // Scores: Q_h K_h^T, `[s, dk] @ [dk, s]`.
        let scores = s
            .matmul_dram_batched(
                &dq,
                &dkk,
                &items,
                [seq, dk, seq],
                ROUTE,
                Fidelity::HiFi4,
                BUDGET,
            )
            .unwrap();
        assert_eq!([scores.rows, scores.cols], [batch * heads * seq, seq]);
        let got = s.download(&scores).unwrap();
        for (n, (ba, _)) in items.iter().enumerate() {
            let a = s
                .upload(&block(&q, width, ba.at, [seq, dk]), seq, dk)
                .unwrap();
            let b = s
                .upload(&block(&k, width, ba.at, [seq, dk]), seq, dk)
                .unwrap();
            let want = s
                .matmul_dram(&a, false, &b, true, ROUTE, Fidelity::HiFi4, BUDGET)
                .unwrap();
            let want = s.download(&want).unwrap();
            assert_bits(
                &got[n * seq * seq..(n + 1) * seq * seq],
                &want,
                &format!("head {n}'s scores"),
            );
        }

        // Context: P_h V_h, `[s, s] @ [s, dk]`, `P` the stack of scores.
        let p_items: Vec<(Block, Block)> = items
            .iter()
            .enumerate()
            .map(|(n, (ba, _))| {
                (
                    Block {
                        at: [n * seq, 0],
                        transposed: false,
                    },
                    *ba,
                )
            })
            .collect();
        let ctx = s
            .matmul_dram_batched(
                &scores,
                &dkk,
                &p_items,
                [seq, seq, dk],
                ROUTE,
                Fidelity::HiFi4,
                BUDGET,
            )
            .unwrap();
        let got = s.download(&ctx).unwrap();
        let all_scores = s.download(&scores).unwrap();
        for (n, (_, bv)) in p_items.iter().enumerate() {
            let p = s
                .upload(&block(&all_scores, seq, [n * seq, 0], [seq, seq]), seq, seq)
                .unwrap();
            let v = s
                .upload(&block(&k, width, bv.at, [seq, dk]), seq, dk)
                .unwrap();
            let want = s
                .matmul_dram(&p, false, &v, false, ROUTE, Fidelity::HiFi4, BUDGET)
                .unwrap();
            let want = s.download(&want).unwrap();
            assert_bits(
                &got[n * seq * dk..(n + 1) * seq * dk],
                &want,
                &format!("head {n}'s context"),
            );
        }
    });
}

/// One product of a block that ends at its tensor's ragged edge reads the
/// tensor's own zero padding; a ragged block inside a tensor, or a batch
/// whose products do not start on tile rows, is refused.
#[test]
fn ragged_blocks_only_at_the_edge() {
    with_session(|s| {
        let (rows, cols) = (70, 96);
        let a = floats(3, rows * cols);
        let b = floats(4, 40 * 50);
        let da = s.upload(&a, rows, cols).unwrap();
        let db = s.upload(&b, 40, 50).unwrap();
        // `[70, 40] @ [40, 50]`: A's columns 32..72 would be interior and
        // ragged -- refused; columns 56..96 end at the edge but start off a
        // tile -- refused; B whole.
        let item = |c0| {
            [(
                Block {
                    at: [0, c0],
                    transposed: false,
                },
                Block {
                    at: [0, 0],
                    transposed: false,
                },
            )]
        };
        for c0 in [32, 56] {
            let e = s
                .matmul_dram_batched(
                    &da,
                    &db,
                    &item(c0),
                    [rows, 40, 50],
                    ROUTE,
                    Fidelity::HiFi4,
                    BUDGET,
                )
                .unwrap_err();
            assert!(e.to_string().contains("not whole tiles"), "{e}");
        }
        // Two `[35, 32]` products would share a tile row of the output.
        let two = [item(0)[0], item(0)[0]];
        let e = s
            .matmul_dram_batched(&da, &db, &two, [35, 32, 50], ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap_err();
        assert!(e.to_string().contains("whole tile rows"), "{e}");

        // `[70, 64] @ [64, 50]`: A's last 64 columns (ragged rows at the
        // edge), B a `[64, 50]` tensor (ragged columns at its edge).
        let b = floats(5, 64 * 50);
        let db = s.upload(&b, 64, 50).unwrap();
        let got = s
            .matmul_dram_batched(
                &da,
                &db,
                &item(32),
                [rows, 64, 50],
                ROUTE,
                Fidelity::HiFi4,
                BUDGET,
            )
            .unwrap();
        let ha = s
            .upload(&block(&a, cols, [0, 32], [rows, 64]), rows, 64)
            .unwrap();
        let want = s
            .matmul_dram(&ha, false, &db, false, ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap();
        assert_bits(
            &s.download(&got).unwrap(),
            &s.download(&want).unwrap(),
            "a block at the ragged edge",
        );
    });
}

/// The heads of a `[b s, h dk]` projection, stacked into `[b h s, dk]`, and
/// merged back: the reshapes attention's head split and merge make, bit for
/// bit, the merge writing blocks into the output's columns. Overlapping or
/// incomplete moves are refused.
#[test]
fn block_copies_move_heads_into_rows_and_back() {
    with_session(|s| {
        let (batch, seq, heads, dk) = (2, 64, 2, 32);
        let width = heads * dk;
        let x = floats(6, batch * seq * width);
        let dx = s.upload(&x, batch * seq, width).unwrap();
        let heads_at: Vec<[usize; 2]> = (0..batch)
            .flat_map(|i| (0..heads).map(move |j| [i * seq, j * dk]))
            .collect();
        let split_moves: Vec<BlockMove> = heads_at
            .iter()
            .enumerate()
            .map(|(n, &from)| BlockMove {
                from,
                to: [n * seq, 0],
                extent: [seq, dk],
                transposed: false,
            })
            .collect();
        let split = s
            .copy_blocks(&dx, &split_moves, [batch * heads * seq, dk])
            .unwrap();
        let want: Vec<f32> = heads_at
            .iter()
            .flat_map(|&a| block(&x, width, a, [seq, dk]))
            .collect();
        assert_bits(&s.download(&split).unwrap(), &want, "heads into rows");

        // Back: each stacked head to its columns.
        let merge: Vec<BlockMove> = split_moves
            .iter()
            .map(|m| BlockMove {
                from: m.to,
                to: m.from,
                extent: m.extent,
                transposed: false,
            })
            .collect();
        let merged = s.copy_blocks(&split, &merge, [batch * seq, width]).unwrap();
        assert_bits(&s.download(&merged).unwrap(), &x, "heads back into columns");

        // Incomplete, and overlapping.
        let e = s
            .copy_blocks(&split, &merge[..3], [batch * seq, width])
            .unwrap_err();
        assert!(e.to_string().contains("unwritten"), "{e}");
        let mut twice = merge.clone();
        twice[3] = twice[0];
        let e = s
            .copy_blocks(&split, &twice, [batch * seq, width])
            .unwrap_err();
        assert!(e.to_string().contains("twice"), "{e}");
    });
}

/// A block copy that transposes: each head's `[s, dk]` block of a `[b s, h
/// dk]` projection as its `[dk, s]` transpose, stacked -- `K^T` per head, made
/// a matrix -- bit for bit the host's transpose. Watched failing with the
/// tiles read untransposed.
#[test]
fn a_block_copy_transposes_whole_tiles() {
    with_session(|s| {
        let (batch, seq, heads, dk) = (2, 64, 2, 32);
        let width = heads * dk;
        let x = floats(7, batch * seq * width);
        let dx = s.upload(&x, batch * seq, width).unwrap();
        let mut moves = Vec::new();
        let mut want = Vec::new();
        for i in 0..batch {
            for j in 0..heads {
                let from = [i * seq, j * dk];
                moves.push(BlockMove {
                    from,
                    to: [moves.len() * dk, 0],
                    extent: [dk, seq],
                    transposed: true,
                });
                let b = block(&x, width, from, [seq, dk]);
                want.extend(
                    (0..dk)
                        .flat_map(|c| (0..seq).map(move |r| (r, c)))
                        .map(|(r, c)| b[r * dk + c]),
                );
            }
        }
        let t = s
            .copy_blocks(&dx, &moves, [batch * heads * dk, seq])
            .unwrap();
        assert_bits(&s.download(&t).unwrap(), &want, "heads transposed");
    });
}
