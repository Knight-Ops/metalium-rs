//! `Dst` subblocks and half-sync (`docs/completed-plans/dataflow-compiler.md` F5, H2;
//! `tt_kernels::tile_op::dst`): a blocked product whose outputs are accumulated `w` <= 4 tiles at a
//! time in one half of `Dst` while the packer drains the other, `A`'s tile held across the
//! subblock's tiles of `B` (`matmul::block_stream_kernel_subblocked`), is the one-tile kernel
//! (`block_stream_kernel_looped`) bit for bit: every output takes the same pairs in the same order.
//!
//! The host checker (`matmul::order::interpret_roles`, `tt-kernels` unit tests) holds the three
//! roles' streams to each output's canonical order under an adversarial schedule; this holds the
//! hardware to them: the special values, ragged products, columns no width divides (3, 5, 7, 9
//! tiles: the FLUX.2 VAE's 96 channels are 3), one to four fidelity phases, either operand
//! transposed, `B` stored BF16, and exact against host integer sums. The negative controls clear
//! both halves at every subblock, or overlap the halves, and must change the product.
//!
//! ttsim checks correctness only: it runs each unit to completion (divergence rows 34, 46), so the
//! half-sync's overlap is silicon's to show. It does interleave the three threads (row 53), and
//! both negative controls are caught there as on silicon.
use tt_kernels::matmul::{self, DstMutation, Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{self, BlockStream, DramTensor};
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
        let tiles = match units {
            0 => TileChoice::All,
            n => TileChoice::Count(n),
        };
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

/// Values whose sums round: the order of a chain shows in the bits.
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 73 + seed) % 257) as f32 / 131.0 - 1.0 + ((i * 7 + seed) % 5) as f32 * 1e-3)
        .collect()
}

/// `values` with signed zeros, subnormals, infinities, a NaN and the extremes, each in a row and a
/// column of its own so most outputs stay finite.
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

fn differing(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// `op(A) @ op(B)` with the session's current mode.
fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    at: bool,
    b: &DramTensor,
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

/// `[rows, kq, cols]` blocks: columns four divides and ones no width divides, one column, rows
/// odd and even (a period of the programs is two rows where a row has an odd number of subblocks),
/// one `K` tile a chunk and several.
const SHAPES: [[usize; 3]; 8] = [
    [4, 3, 5],
    [2, 2, 3],
    [3, 1, 1],
    [1, 4, 7],
    [4, 2, 9],
    [2, 3, 4],
    [3, 2, 8],
    [5, 1, 6],
];

/// The product with the subblocked kernel against the one-tile kernel, each block in [`SHAPES`].
fn subblocked_against_one_tile<T: tt_device::Transport>(
    s: &mut Session<T>,
    tiles: [usize; 3],
    cases: &[(&str, &DramTensor, bool, &DramTensor, bool)],
    fidelities: &[Fidelity],
) {
    for &fidelity in fidelities {
        for shape in SHAPES {
            assert!(
                tensor::block_stream_choice(
                    BlockStream::Fixed(shape),
                    tiles,
                    2,
                    f64::INFINITY,
                    SrcRoute::Tf32FromFp32,
                    fidelity,
                )
                .is_some(),
                "{shape:?} plans no blocked product of {tiles:?}"
            );
            s.set_block_stream(BlockStream::Fixed(shape));
            let built = matmul::subblocked_built();
            for &(what, x, xt, y, yt) in cases {
                matmul::set_dst_subblocks(false);
                let want = product(s, x, xt, y, yt, fidelity);
                matmul::set_dst_subblocks(true);
                same(
                    &product(s, x, xt, y, yt, fidelity),
                    &want,
                    &format!("{shape:?}, {fidelity:?}, {what}"),
                );
            }
            assert!(
                matmul::subblocked_built() > built,
                "{shape:?}, {fidelity:?}: the subblocked kernel was not built"
            );
        }
    }
    s.set_block_stream(BlockStream::Off);
}

#[test]
fn subblocks_are_the_one_tile_kernel_bit_for_bit() {
    with_session(2, |s| {
        // Ragged in every dimension: 10 x 7 x 10 tiles.
        let (m, k, n) = (300, 200, 310);
        let av = specials(m * k, 4);
        let bv = specials(k * n, 17);
        let (a, at_) = (s.upload(&av, m, k).unwrap(), s.upload(&av, k, m).unwrap());
        let (b, bt_) = (s.upload(&bv, k, n).unwrap(), s.upload(&bv, n, k).unwrap());
        // Stored BF16 from its bits: ttsim cannot narrow on the card (divergence "BF16 late
        // narrowing").
        let bits: Vec<u16> = bv.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let b16 = s.upload_bf16_weight(&bits, k, n).unwrap();
        let cases: [(&str, &_, bool, &_, bool); 4] = [
            ("A B", &a, false, &b, false),
            ("A B^T", &a, false, &bt_, true),
            ("A^T B", &at_, true, &b, false),
            ("A B16", &a, false, &b16, false),
        ];
        subblocked_against_one_tile(
            s,
            [10, 7, 10],
            &cases,
            &[Fidelity::Lo, Fidelity::HiFi2, Fidelity::HiFi4],
        );
        for t in [a, at_, b, bt_, b16] {
            s.free(t).unwrap();
        }
    });
}

#[test]
fn subblocks_are_exact_where_the_arithmetic_is() {
    with_session(2, |s| {
        // Small integers: every product and sum is exact in TF32 and F32, so the host's sum is the
        // product whatever the order, an oracle independent of either kernel. 96 columns: the
        // FLUX.2 VAE's narrowest convolution, 3 tiles.
        for (m, k, n) in [(70, 160, 96), (130, 96, 224)] {
            let av: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
            let bv: Vec<f32> = (0..k * n).map(|i| ((i * 5) % 11) as f32 - 5.0).collect();
            let want: Vec<f32> = (0..m * n)
                .map(|o| (0..k).map(|q| av[o / n * k + q] * bv[q * n + o % n]).sum())
                .collect();
            let a = s.upload(&av, m, k).unwrap();
            let b = s.upload(&bv, k, n).unwrap();
            matmul::set_dst_subblocks(true);
            for mode in [
                BlockStream::Fixed([3, 1, 3]),
                BlockStream::Fixed([2, 2, 2]),
                BlockStream::Fixed([1, 3, 7]),
                BlockStream::Auto,
            ] {
                s.set_block_stream(mode);
                same(
                    &product(s, &a, false, &b, false, Fidelity::HiFi4),
                    &want,
                    &format!("[{m}, {k}] @ [{k}, {n}], {mode:?}"),
                );
            }
            // A product on the other paths after it: the subblocked kernel leaves the handshakes,
            // the math role's `Dst` offset and the packer's where it found them.
            s.set_block_stream(BlockStream::Off);
            same(
                &product(s, &a, false, &b, false, Fidelity::HiFi4),
                &want,
                &format!("[{m}, {k}] @ [{k}, {n}], the other paths after"),
            );
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}

/// Each negative control's product against the kernel's: how many outputs differ.
fn mutated<T: tt_device::Transport>(s: &mut Session<T>, mutation: DstMutation) -> (usize, usize) {
    let (m, k, n) = (128, 256, 256);
    let a = s.upload(&values(m * k, 3), m, k).unwrap();
    let b = s.upload(&values(k * n, 5), k, n).unwrap();
    // Four columns a subblock, two `K` tiles a chunk: both halves busy, priors reloaded.
    s.set_block_stream(BlockStream::Fixed([4, 2, 8]));
    matmul::set_dst_subblocks(false);
    let want = product(s, &a, false, &b, false, Fidelity::HiFi4);
    matmul::set_dst_subblocks(true);
    same(
        &product(s, &a, false, &b, false, Fidelity::HiFi4),
        &want,
        "the kernel as built",
    );
    matmul::probe_dst_mutation(mutation);
    let got = product(s, &a, false, &b, false, Fidelity::HiFi4);
    matmul::probe_dst_mutation(DstMutation::None);
    s.set_block_stream(BlockStream::Off);
    s.free(a).unwrap();
    s.free(b).unwrap();
    (differing(&got, &want), want.len())
}

/// Clearing both halves at every subblock (`ZEROACC` of all of `Dst`) erases the half the packer
/// is still draining.
#[test]
fn clearing_both_halves_is_caught() {
    with_session(2, |s| {
        let (differ, of) = mutated(s, DstMutation::ClearBoth);
        println!("MEASURE both halves cleared: {differ} of {of} outputs differ");
        assert!(differ > 0, "both halves cleared changed nothing");
    });
}

/// The second half overlapping the first by three tiles: a subblock overwrites outputs the one
/// before it holds, which every interleaving shows.
#[test]
fn overlapping_subblocks_are_caught() {
    with_session(2, |s| {
        let (differ, of) = mutated(s, DstMutation::Overlap);
        println!("MEASURE overlapping halves: {differ} of {of} outputs differ");
        assert!(differ > 0, "overlapping halves changed nothing");
    });
}

/// Model shapes on every tile, the measured block, subblocked against the one-tile kernel and
/// exact against host integer sums on a grid: the MOSS-Image DiT at 1,024 tokens (qkv, out, the
/// FFN), Qwen3.5 / Gemma 4 prefill at 512 (projections and down projections), and the FLUX.2 VAE's
/// patch products with 384, 192 and 96 output channels. Silicon only (their size).
#[cfg(feature = "silicon")]
#[test]
fn model_shapes_are_the_one_tile_kernel_bit_for_bit() {
    with_session(0, |s| {
        for (m, k, n) in [
            (1024, 3072, 9216),
            (1024, 3072, 3072),
            (1024, 8192, 3072),
            (512, 2560, 9216),
            (512, 10240, 2560),
            (512, 2560, 2048),
            (4096, 3456, 384),
            (16384, 1728, 192),
            (16384, 864, 96),
            (1000, 300, 70),
        ] {
            let av: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 5) as f32 - 2.0).collect();
            let bv: Vec<f32> = (0..k * n).map(|i| ((i * 5 + 1) % 7) as f32 - 3.0).collect();
            let a = s.upload(&av, m, k).unwrap();
            let b = s.upload(&bv, k, n).unwrap();
            s.set_block_stream(BlockStream::Measured);
            matmul::set_dst_subblocks(false);
            let want = product(s, &a, false, &b, false, Fidelity::HiFi4);
            matmul::set_dst_subblocks(true);
            let built = matmul::subblocked_built();
            let got = product(s, &a, false, &b, false, Fidelity::HiFi4);
            same(&got, &want, &format!("[{m}, {k}] @ [{k}, {n}]"));
            assert!(
                matmul::subblocked_built() > built,
                "[{m}, {k}] @ [{k}, {n}]: the subblocked kernel was not built"
            );
            for i in (0..m).step_by(97).chain([m - 1]) {
                for j in (0..n).step_by(61).chain([n - 1]) {
                    let exact: f32 = (0..k).map(|q| av[i * k + q] * bv[q * n + j]).sum();
                    assert_eq!(got[i * n + j], exact, "[{m}, {k}] @ [{k}, {n}]: ({i}, {j})");
                }
            }
            s.set_block_stream(BlockStream::Off);
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}

/// Packed weights (BFP8, BFP4; BFP2 on silicon, which ttsim refuses in `Src`) and FP32 against
/// the one-tile kernel, at blocks whose rows take subblocks four, three, two and one tile wide,
/// and exact against host sums of the values each storage decodes to. Small integers: every
/// product and sum is exact, so a term counted in the wrong output shows.
#[test]
fn packed_weights_are_the_one_tile_kernel_bit_for_bit() {
    with_session(2, |s| {
        let (m, k, n) = (192, 256, 384);
        let av: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 1) % 5) as f32 - 2.0).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i * 7 + 3) % 5) as f32 - 2.0).collect();
        let a = s.upload(&av, m, k).unwrap();
        let b = s.upload(&bv, k, n).unwrap();
        let mut formats = vec![
            ("BFP8", tt_kernels::bfp::BfpFormat::Bfp8),
            ("BFP4", tt_kernels::bfp::BfpFormat::Bfp4),
        ];
        if cfg!(feature = "silicon") {
            formats.push(("BFP2", tt_kernels::bfp::BfpFormat::Bfp2));
        }
        let mut weights: Vec<(&str, DramTensor, Vec<f32>)> = Vec::new();
        for (name, f) in formats {
            let p = s.bfp_from_f32(&b, f).unwrap();
            let decoded = s.bfp_to_f32(&p).unwrap();
            let values = s.download(&decoded).unwrap();
            s.free(decoded).unwrap();
            weights.push((name, DramTensor::from_bfp(p), values));
        }
        weights.push(("FP32", b, bv));
        for (name, w, wv) in &weights {
            let exact: Vec<f32> = (0..m * n)
                .map(|o| (0..k).map(|q| av[o / n * k + q] * wv[q * n + o % n]).sum())
                .collect();
            for shape in [
                [6, 4, 12],
                [2, 4, 8],
                [3, 2, 9],
                [4, 2, 6],
                [6, 4, 4],
                [2, 2, 1],
                [4, 4, 8],
            ] {
                s.set_block_stream(BlockStream::Fixed(shape));
                matmul::set_dst_subblocks(false);
                let want = product(s, &a, false, w, false, Fidelity::HiFi4);
                matmul::set_dst_subblocks(true);
                let built = matmul::subblocked_built();
                let got = product(s, &a, false, w, false, Fidelity::HiFi4);
                let subblocked = matmul::subblocked_built() > built;
                println!(
                    "MEASURE B {name} {shape:?}: subblocked {subblocked}, {} of {} outputs \
                     differ from the one-tile kernel",
                    differing(&got, &want),
                    want.len()
                );
                same(
                    &want,
                    &exact,
                    &format!("B {name} {shape:?}, one-tile, exact"),
                );
                same(&got, &want, &format!("B {name} {shape:?}"));
            }
        }
        s.set_block_stream(BlockStream::Off);
        s.free(a).unwrap();
        for (_, w, _) in weights {
            s.free(w).unwrap();
        }
    });
}

/// A probe: one product repeated, each wrong output tile reported (`TT_PROBE_SHAPE` `mb,kq,nb`,
/// `TT_PROBE_REPEATS`).
#[test]
#[ignore = "a probe"]
fn probe_wrong_tiles() {
    with_session(2, |s| {
        let (m, k, n) = (192, 256, 384);
        let av: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 1) % 5) as f32 - 2.0).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i * 7 + 3) % 5) as f32 - 2.0).collect();
        let a = s.upload(&av, m, k).unwrap();
        let b = s.upload(&bv, k, n).unwrap();
        let p = s
            .bfp_from_f32(&b, tt_kernels::bfp::BfpFormat::Bfp8)
            .unwrap();
        let w = DramTensor::from_bfp(p);
        let shape: Vec<usize> = std::env::var("TT_PROBE_SHAPE")
            .unwrap_or("2,4,8".into())
            .split(',')
            .map(|x| x.parse().unwrap())
            .collect();
        let repeats: usize = std::env::var("TT_PROBE_REPEATS").map_or(20, |v| v.parse().unwrap());
        s.set_block_stream(BlockStream::Fixed([shape[0], shape[1], shape[2]]));
        matmul::set_dst_subblocks(false);
        let want = product(s, &a, false, &w, false, Fidelity::HiFi4);
        matmul::set_dst_subblocks(true);
        for r in 0..repeats {
            let got = product(s, &a, false, &w, false, Fidelity::HiFi4);
            let mut tiles = std::collections::BTreeMap::new();
            for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                if x.to_bits() != y.to_bits() {
                    let e = tiles.entry((i / n / 32, i % n / 32)).or_insert((0, 0.0f32));
                    e.0 += 1;
                    e.1 += x - y;
                }
            }
            println!("MEASURE repeat {r}: wrong tiles (row, col): (count, summed error) {tiles:?}");
            // What the error of each wrong tile is: a K tile's terms missing or doubled, or another
            // column tile's.
            for &(ti, tj) in tiles.keys() {
                let kt = k / 32;
                let mut found = Vec::new();
                for q in 0..kt {
                    for cj in 0..n / 32 {
                        for sign in [-1.0f32, 1.0] {
                            let ok = (0..32).all(|r| {
                                (0..32).all(|c| {
                                    let (i, j) = (ti * 32 + r, tj * 32 + c);
                                    let jj = cj * 32 + c;
                                    let contrib: f32 = (q * 32..q * 32 + 32)
                                        .map(|kk| av[i * k + kk] * bv[kk * n + jj])
                                        .sum();
                                    let want_c: f32 = (q * 32..q * 32 + 32)
                                        .map(|kk| av[i * k + kk] * bv[kk * n + j])
                                        .sum();
                                    got[i * n + j] - want[i * n + j]
                                        == sign * (contrib - if sign > 0.0 { want_c } else { 0.0 })
                                })
                            });
                            if ok {
                                found.push((q, cj, sign));
                            }
                        }
                    }
                }
                println!("MEASURE tile ({ti}, {tj}): explained by (K tile, B column tile, sign) {found:?}");
            }
        }
    });
}
