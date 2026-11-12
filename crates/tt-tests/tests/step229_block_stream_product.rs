//! A product of several tile rows as 2-D blocks of output tiles, each job streaming `K` in chunks
//! and keeping its partials in L1 (`tensor::BlockStream`, `docs/plans/prefill-operand-reuse.md`),
//! is bit for bit the other paths: every output tile's `K` chain is the same, in the same order.
//! The negative control accumulates the chunks last first and must differ.
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{self, BlockStream};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(units: usize, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
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
fn with_session(units: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Values whose sums round: the order of a chain shows in the bits.
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 73 + seed) % 257) as f32 / 131.0 - 1.0 + ((i * 7 + seed) % 5) as f32 * 1e-3)
        .collect()
}

/// `values` with the special values sprinkled in: signed zeros, subnormals, infinities, a NaN and
/// the extremes, each in a row and a column of its own so most outputs stay finite.
fn specials(n: usize, seed: usize) -> Vec<f32> {
    let mut v = values(n, seed);
    let special = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(0x007f_ffff),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
    ];
    for (i, s) in special.into_iter().enumerate() {
        let at = (i * 977 + seed * 31) % n;
        v[at] = s;
    }
    v
}

fn same(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i}: {x} vs {y}");
    }
}

/// `op(A) @ op(B)` with the session's current mode.
#[allow(clippy::too_many_arguments)]
fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &tt_kernels::tensor::DramTensor,
    at: bool,
    b: &tt_kernels::tensor::DramTensor,
    bt: bool,
    fidelity: Fidelity,
) -> Vec<f32> {
    let out = s
        .matmul_dram(a, at, b, bt, SrcRoute::Tf32FromFp32, fidelity, BUDGET * 40)
        .unwrap();
    let v = s.download(&out).unwrap();
    s.free(out).unwrap();
    v
}

/// The block shapes run, each with what the planner makes of it for the product (so the gate
/// knows the blocked path ran).
fn shapes(tiles: [usize; 3], units: usize) -> Vec<[usize; 3]> {
    let shapes = [[2, 2, 3], [3, 1, 2], [4, 3, 5], [1, 4, 1], [2, 7, 2]];
    for shape in shapes {
        assert!(
            tensor::block_stream_choice(
                BlockStream::Fixed(shape),
                tiles,
                units,
                f64::INFINITY,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
            )
            .is_some(),
            "{shape:?} on {tiles:?} plans no blocked product"
        );
    }
    shapes.to_vec()
}

#[test]
fn blocked_products_are_the_other_paths_bit_for_bit() {
    with_session(2, |s| {
        // Ragged in every dimension: 4 x 7 x 5 tiles.
        let (m, k, n) = (100, 200, 130);
        let units = 2;
        for (at, bt) in [(false, false), (false, true), (true, false)] {
            let av = specials(m * k, 1);
            let bv = specials(k * n, 11);
            let a = match at {
                false => s.upload(&av, m, k),
                true => s.upload(&av, k, m),
            }
            .unwrap();
            let b = match bt {
                false => s.upload(&bv, k, n),
                true => s.upload(&bv, n, k),
            }
            .unwrap();
            for fidelity in [Fidelity::Lo, Fidelity::HiFi4] {
                s.set_block_stream(BlockStream::Off);
                let want = product(s, &a, at, &b, bt, fidelity);
                for shape in shapes([4, 7, 5], units) {
                    s.set_block_stream(BlockStream::Fixed(shape));
                    let got = product(s, &a, at, &b, bt, fidelity);
                    same(
                        &got,
                        &want,
                        &format!("{shape:?}, {fidelity:?}, A^T {at}, B^T {bt}"),
                    );
                }
            }
            s.set_block_stream(BlockStream::Off);
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}

#[test]
fn looped_blocks_are_the_other_paths_bit_for_bit() {
    with_session(2, |s| {
        // Each output's block stored once and repeated by the runner, its addresses in GPRs
        // (`matmul::block_stream_kernel_looped`): rows of one and of several, columns of one and of
        // several, an 8 x 8 block whose unrolled programs do not fit.
        tt_kernels::matmul::set_block_loops(true);
        let (m, k, n) = (260, 200, 270);
        for (at, bt) in [(false, false), (false, true), (true, false)] {
            let av = specials(m * k, 2);
            let bv = specials(k * n, 13);
            let a = match at {
                false => s.upload(&av, m, k),
                true => s.upload(&av, k, m),
            }
            .unwrap();
            let b = match bt {
                false => s.upload(&bv, k, n),
                true => s.upload(&bv, n, k),
            }
            .unwrap();
            for fidelity in [Fidelity::Lo, Fidelity::HiFi4] {
                s.set_block_stream(BlockStream::Off);
                let want = product(s, &a, at, &b, bt, fidelity);
                for shape in [[8, 2, 8], [4, 3, 5], [1, 4, 3], [3, 1, 1], [9, 2, 2]] {
                    assert!(
                        tensor::block_stream_choice(
                            BlockStream::Fixed(shape),
                            [9, 7, 9],
                            2,
                            f64::INFINITY,
                            SrcRoute::Tf32FromFp32,
                            fidelity,
                        )
                        .is_some(),
                        "{shape:?} plans no looped blocked product"
                    );
                    s.set_block_stream(BlockStream::Fixed(shape));
                    let got = product(s, &a, at, &b, bt, fidelity);
                    same(
                        &got,
                        &want,
                        &format!("looped {shape:?}, {fidelity:?}, A^T {at}, B^T {bt}"),
                    );
                }
            }
            s.set_block_stream(BlockStream::Off);
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
        tt_kernels::matmul::set_block_loops(false);
    });
}

#[test]
fn blocked_products_are_exact_where_the_arithmetic_is() {
    with_session(2, |s| {
        // Small integers: every product and sum is exact in TF32 and F32, so the host's sum is the
        // product whatever the order, an oracle independent of the kernel.
        let (m, k, n) = (70, 160, 96);
        let av: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i * 5) % 11) as f32 - 5.0).collect();
        let want: Vec<f32> = (0..m * n)
            .map(|o| {
                let (r, c) = (o / n, o % n);
                (0..k).map(|q| av[r * k + q] * bv[q * n + c]).sum()
            })
            .collect();
        let a = s.upload(&av, m, k).unwrap();
        let b = s.upload(&bv, k, n).unwrap();
        for shape in [[3, 1, 3], [2, 2, 2], [1, 5, 3]] {
            s.set_block_stream(BlockStream::Fixed(shape));
            same(
                &product(s, &a, false, &b, false, Fidelity::HiFi4),
                &want,
                &format!("{shape:?}"),
            );
        }
        // And a product on the other paths after it: the blocked kernels leave the handshakes
        // where they found them.
        s.set_block_stream(BlockStream::Off);
        same(
            &product(s, &a, false, &b, false, Fidelity::HiFi4),
            &want,
            "the other paths after",
        );
        s.free(a).unwrap();
        s.free(b).unwrap();
    });
}

/// Every pair order (`matmul::PairOrder`, `docs/plans/prefill-operand-reuse.md` P4) is the faces
/// order bit for bit: blocked and on the other paths, `A` or `B` transposed, `B` stored BF16, one
/// to four fidelity phases, products of one row and of sixteen (`LiveRows`), the special values;
/// and exact against host integer sums. The host checker (`matmul::order`) holds each order's
/// streams to the canonical accumulation order; this holds the hardware to them.
#[test]
fn pair_orders_are_the_faces_order_bit_for_bit() {
    use tt_kernels::matmul::{set_pair_order, PairOrder};
    with_session(2, |s| {
        let (m, k, n) = (100, 200, 130);
        let av = specials(m * k, 4);
        let bv = specials(k * n, 17);
        let (a, at_) = (s.upload(&av, m, k).unwrap(), s.upload(&av, k, m).unwrap());
        let (b, bt_) = (s.upload(&bv, k, n).unwrap(), s.upload(&bv, n, k).unwrap());
        // Stored BF16 from its bits: ttsim cannot narrow on the card (divergence "BF16 late narrowing").
        let bits: Vec<u16> = bv.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let b16 = s.upload_bf16_weight(&bits, k, n).unwrap();
        let one = s.upload(&av[..k], 1, k).unwrap();
        let sixteen = s.upload(&av[..16 * k], 16, k).unwrap();
        let cases: [(&str, &_, bool, &_, bool); 6] = [
            ("A B", &a, false, &b, false),
            ("A B^T", &a, false, &bt_, true),
            ("A^T B", &at_, true, &b, false),
            ("A B16", &a, false, &b16, false),
            ("one row", &one, false, &b, false),
            ("sixteen rows", &sixteen, false, &b, false),
        ];
        for fidelity in [Fidelity::Lo, Fidelity::HiFi2, Fidelity::HiFi4] {
            for mode in [
                BlockStream::Off,
                BlockStream::Fixed([4, 3, 5]),
                BlockStream::Fixed([1, 4, 3]),
            ] {
                s.set_block_stream(mode);
                for (what, x, xt, y, yt) in cases {
                    if mode != BlockStream::Off && x.rows < 64 {
                        continue;
                    }
                    set_pair_order(PairOrder::Faces);
                    let want = product(s, x, xt, y, yt, fidelity);
                    for order in PairOrder::ALL {
                        set_pair_order(order);
                        same(
                            &product(s, x, xt, y, yt, fidelity),
                            &want,
                            &format!("{order:?}, {what}, {mode:?}, {fidelity:?}"),
                        );
                    }
                }
            }
        }
        s.set_block_stream(BlockStream::Off);
        for t in [a, at_, b, bt_, b16, one, sixteen] {
            s.free(t).unwrap();
        }
        // Exact: small integers, every product and sum exact in TF32 and F32.
        let (m, k, n) = (70, 160, 96);
        let av: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i * 5) % 11) as f32 - 5.0).collect();
        let want: Vec<f32> = (0..m * n)
            .map(|o| (0..k).map(|q| av[o / n * k + q] * bv[q * n + o % n]).sum())
            .collect();
        let a = s.upload(&av, m, k).unwrap();
        let b = s.upload(&bv, k, n).unwrap();
        for order in PairOrder::ALL {
            set_pair_order(order);
            for mode in [BlockStream::Off, BlockStream::Fixed([2, 2, 2])] {
                s.set_block_stream(mode);
                same(
                    &product(s, &a, false, &b, false, Fidelity::HiFi4),
                    &want,
                    &format!("{order:?}, {mode:?}: exact"),
                );
            }
        }
        set_pair_order(PairOrder::Faces);
        s.set_block_stream(BlockStream::Off);
        s.free(a).unwrap();
        s.free(b).unwrap();
    });
}

#[test]
fn the_planners_blocks_over_more_units_are_the_other_paths_bit_for_bit() {
    // `auto`'s own shapes on three and four units (rounds that do not divide evenly, a unit
    // left idle), MNIST's and ragged products among them.
    for units in [3, 4] {
        with_session(units, |s| {
            for (m, k, n) in [
                (64, 784, 128),
                (96, 320, 288),
                (300, 160, 96),
                (130, 512, 200),
            ] {
                let a = s.upload(&values(m * k, m), m, k).unwrap();
                let b = s.upload(&values(k * n, n), k, n).unwrap();
                s.set_block_stream(BlockStream::Off);
                let want = product(s, &a, false, &b, false, Fidelity::HiFi4);
                s.set_block_stream(BlockStream::Auto);
                let got = product(s, &a, false, &b, false, Fidelity::HiFi4);
                same(
                    &got,
                    &want,
                    &format!("[{m}, {k}] @ [{k}, {n}] on {units} units"),
                );
                s.set_block_stream(BlockStream::Off);
                s.free(a).unwrap();
                s.free(b).unwrap();
            }
        });
    }
}

#[test]
fn chunks_accumulated_out_of_order_are_caught() {
    with_session(2, |s| {
        let (m, k, n) = (64, 256, 64);
        let a = s.upload(&values(m * k, 3), m, k).unwrap();
        let b = s.upload(&values(k * n, 5), k, n).unwrap();
        s.set_block_stream(BlockStream::Off);
        let want = product(s, &a, false, &b, false, Fidelity::HiFi4);
        s.set_block_stream(BlockStream::Fixed([2, 2, 2]));
        same(
            &product(s, &a, false, &b, false, Fidelity::HiFi4),
            &want,
            "in order",
        );
        tensor::probe_reverse_chunks(true);
        let reversed = product(s, &a, false, &b, false, Fidelity::HiFi4);
        tensor::probe_reverse_chunks(false);
        let differ = reversed
            .iter()
            .zip(&want)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert!(
            differ > want.len() / 4,
            "the chunks last first changed only {differ} of {} outputs",
            want.len()
        );
        s.set_block_stream(BlockStream::Off);
        s.free(a).unwrap();
        s.free(b).unwrap();
    });
}

/// A batched product (`Session::matmul_dram_batched`: attention's `P V` and `Q K^T`, a head an
/// item) takes the blocked path a head at a time: bit for bit the other paths, `B` transposed
/// (`K^T`) or not, and the chunks last first (the control) change the outputs, so the blocked path
/// is the one that ran.
#[test]
fn batched_products_in_blocks_are_the_other_paths_bit_for_bit() {
    use tt_kernels::tensor::Block;
    with_session(4, |s| {
        let heads = 3;
        // `P V`: each head's `P` `[96, 256]` stacked by rows, its `V` `[256, 64]` side by side.
        let (m, k, n) = (96, 256, 64);
        let p = s.upload(&specials(heads * m * k, 3), heads * m, k).unwrap();
        let v = s
            .upload(&specials(k * heads * n, 17), k, heads * n)
            .unwrap();
        let pv: Vec<_> = (0..heads)
            .map(|h| {
                (
                    Block {
                        at: [h * m, 0],
                        transposed: false,
                    },
                    Block {
                        at: [0, h * n],
                        transposed: false,
                    },
                )
            })
            .collect();
        // `Q K^T`: each head's `Q` `[96, 64]` and `K` `[160, 64]`, stacked by rows.
        let (qm, qk, qn) = (96, 64, 160);
        let q = s
            .upload(&specials(heads * qm * qk, 5), heads * qm, qk)
            .unwrap();
        let kk = s
            .upload(&specials(heads * qn * qk, 7), heads * qn, qk)
            .unwrap();
        let qkt: Vec<_> = (0..heads)
            .map(|h| {
                (
                    Block {
                        at: [h * qm, 0],
                        transposed: false,
                    },
                    Block {
                        at: [h * qn, 0],
                        transposed: true,
                    },
                )
            })
            .collect();
        let run = |s: &mut Session<_>, a, b, items: &[(Block, Block)], mkn| {
            let out = s
                .matmul_dram_batched(
                    a,
                    b,
                    items,
                    mkn,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    BUDGET * 40,
                )
                .unwrap();
            let v = s.download(&out).unwrap();
            s.free(out).unwrap();
            v
        };
        for (what, a, b, items, mkn) in [
            ("P V", &p, &v, &pv, [m, k, n]),
            ("Q K^T", &q, &kk, &qkt, [qm, qk, qn]),
        ] {
            s.set_block_stream(BlockStream::Off);
            let want = run(s, a, b, items, mkn);
            for shape in [[3, 2, 2], [1, 4, 1], [2, 1, 3]] {
                s.set_block_stream(BlockStream::Fixed(shape));
                let got = run(s, a, b, items, mkn);
                same(&got, &want, &format!("{what}, {shape:?}"));
            }
            s.set_block_stream(BlockStream::Fixed([3, 2, 2]));
            tensor::probe_reverse_chunks(true);
            let reversed = run(s, a, b, items, mkn);
            tensor::probe_reverse_chunks(false);
            let differ = reversed
                .iter()
                .zip(&want)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert!(
                differ > want.len() / 4,
                "{what}: the chunks last first changed only {differ} of {} outputs",
                want.len()
            );
        }
        s.set_block_stream(BlockStream::Off);
        for t in [p, v, q, kk] {
            s.free(t).unwrap();
        }
    });
}
