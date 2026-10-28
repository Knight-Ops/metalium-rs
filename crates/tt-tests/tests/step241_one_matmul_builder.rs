//! One matmul builder for every pair of edge formats (F1d of `docs/plans/dataflow-compiler.md`,
//! ADR 0004): `Session::matmul_dram` takes each operand in its own format -- FP32, BF16 or BFP8/4/2,
//! `A` or `B` -- and multiplies the values it decodes to through the block-stream path
//! (`SrcRoute::Edges`), where BF16 x BF16 and same-format BFP products had builders of their own.
//!
//! - **Every pair** equals the FP32 product of the operands' decoded values, bit for bit (the
//!   specification: a packed datum is moved into `Src` as it is, a BF16 or BFP value is a TF32 one
//!   with its low bits clear), on every path a product takes: whole `K`, ragged, `K` blocks, 2-D
//!   blocks with `K` streamed, looped and not.
//! - **H3 for every pair, and where it ends.** With `Session::set_derived_fidelity`, a product asked
//!   at HiFi4 runs only the phases its edge formats need (`SrcRoute::full_precision`): HiFi4's bits
//!   for every pair while no product overflows FP32 -- and not past that (a product of `1e30`s):
//!   the derived phases are another contract, so they stay opt-in.
//! - **The pairs the old builders took** (BF16 x BF16, `bf16::matmul`; BFP x BFP of one format,
//!   `bfp::matmul`) equalled those builders bit for bit, signed zeros and subnormals included
//!   (`3bc1d89`, ttsim and card 0 `1793162345`), before they were deleted; with specials they equal
//!   the decoded product still.
//! - A BF16 operand whose padding is not known to be zero is repaired before it is read
//!   (`Session::meet`), and gives the same bits.
//!
//! Controls: a packed operand's raw FP32 values give another product; a transposed packed `A` is
//! refused.
//!
//! BFP2 runs on silicon only (ttsim refuses its `Src` format); BF16 tiles are made on the host on
//! ttsim (it refuses the packer's BF16 narrowing).
use tt_isa::format::TileFormat;
use tt_kernels::bfp::BfpFormat;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, Spread, TileChoice};
use tt_kernels::tensor::{BlockStream, DramTensor, Pad};
use tt_ttsim::fork_scope;

/// Values in about [-1, 1] with every FP32 mantissa bit live.
fn values(seed: u64, n: usize) -> Vec<f32> {
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

/// `v` rounded to BF16 (ties to even), held in F32.
fn bf16(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
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
            TileChoice::Count(8),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
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
            TileChoice::Count(8),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// The edge formats a gate runs: ttsim refuses BFP2's `Src` format (`step94`).
fn formats() -> Vec<TileFormat> {
    let mut v = vec![
        TileFormat::FP32,
        TileFormat::BF16,
        TileFormat::BFP8,
        TileFormat::BFP4,
    ];
    if cfg!(feature = "silicon") {
        v.push(TileFormat::BFP2);
    }
    v
}

/// `vals` as a `[rows, cols]` operand stored in `format`, and the FP32 tensor of the values it
/// decodes to.
fn operand<T: tt_device::Transport>(
    s: &mut Session<T>,
    format: TileFormat,
    vals: &[f32],
    rows: usize,
    cols: usize,
) -> (DramTensor, DramTensor) {
    if format == TileFormat::FP32 {
        return (
            s.upload(vals, rows, cols).unwrap(),
            s.upload(vals, rows, cols).unwrap(),
        );
    }
    if format == TileFormat::BF16 {
        let rounded: Vec<f32> = vals.iter().map(|&v| bf16(v)).collect();
        let bits: Vec<u16> = rounded.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        return (
            s.upload_bf16_weight(&bits, rows, cols).unwrap(),
            s.upload(&rounded, rows, cols).unwrap(),
        );
    }
    let bfp = BfpFormat::of(format).expect("a BFP format");
    let w = s.upload(vals, rows, cols).unwrap();
    let packed = s.bfp_from_f32(&w, bfp).unwrap();
    let decoded = s.bfp_to_f32(&packed).unwrap();
    s.free(w).unwrap();
    (DramTensor::from_bfp(packed), decoded)
}

fn mm<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
    fidelity: Fidelity,
) -> DramTensor {
    s.matmul_dram(
        a,
        false,
        b,
        false,
        SrcRoute::Tf32FromFp32,
        fidelity,
        1 << 40,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// One product: `(m, k, n)`, the K-block limit, the block-stream mode, and what it takes.
type Case = (
    usize,
    usize,
    usize,
    Option<usize>,
    BlockStream,
    &'static str,
);

const CASES: [Case; 6] = [
    (64, 256, 96, None, BlockStream::Off, "whole K, slots"),
    (
        37,
        100,
        70,
        None,
        BlockStream::Off,
        "ragged in every dimension",
    ),
    (64, 256, 96, Some(2), BlockStream::Off, "K blocks"),
    (1, 1024, 128, None, BlockStream::Off, "one row"),
    (
        96,
        256,
        128,
        None,
        BlockStream::Fixed([2, 2, 2]),
        "2-D blocks, K streamed",
    ),
    (
        70,
        288,
        100,
        None,
        BlockStream::Fixed([2, 3, 2]),
        "2-D blocks, ragged",
    ),
];

/// Every pair of edge formats: the product of the values the operands decode to.
#[test]
fn every_pair_of_edge_formats_multiplies_its_decoded_values() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        for fa in formats() {
            for fb in formats() {
                for (seed, (m, k, n, limit, block, what)) in CASES.into_iter().enumerate() {
                    let what = format!("{:?} x {:?}, {what}", fa.l1(), fb.l1());
                    let seed = seed as u64;
                    let (a, da) = operand(s, fa, &values(100 + seed, m * k), m, k);
                    let (b, db) = operand(s, fb, &values(200 + seed, k * n), k, n);
                    s.set_matmul_k_block_limit(limit.and_then(std::num::NonZeroUsize::new));
                    s.set_block_stream(block);
                    let loops: &[bool] = match block {
                        BlockStream::Off => &[true],
                        _ => &[true, false],
                    };
                    for &looped in loops {
                        tt_kernels::matmul::set_block_loops(looped);
                        let want = mm(s, &da, &db, Fidelity::HiFi4);
                        let got = mm(s, &a, &b, Fidelity::HiFi4);
                        let want_v = s.download(&want).unwrap();
                        assert!(want_v.iter().any(|&v| v != 0.0), "{what}: an empty product");
                        assert_bits(
                            &format!("{what}, looped {looped}"),
                            &s.download(&got).unwrap(),
                            &want_v,
                        );
                        for t in [want, got] {
                            s.free(t).unwrap();
                        }
                    }
                    tt_kernels::matmul::set_block_loops(true);
                    for t in [a, da, b, db] {
                        s.free(t).unwrap();
                    }
                }
                s.set_matmul_k_block_limit(None);
                s.set_block_stream(BlockStream::Auto);
            }
        }
    });
}

/// Values with signed zeros, subnormals and wide binades among ordinary ones, every eighth.
fn specials(seed: u64, n: usize) -> Vec<f32> {
    let extra = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(0x0040_0000),
        1.0e-30,
        -3.0e30,
        65504.0,
        -1.0e-3,
    ];
    values(seed, n)
        .into_iter()
        .enumerate()
        .map(|(i, v)| if i % 8 == 3 { extra[(i / 8) % 8] } else { v })
        .collect()
}

/// The pairs the old builders took -- BF16 x BF16 and BFP x BFP of one format -- with signed zeros,
/// subnormals and wide binades: the decoded product (the old builders' bits, `3bc1d89`).
#[test]
fn the_old_builders_pairs_are_their_bits() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let pairs: Vec<TileFormat> = formats()
            .into_iter()
            .filter(|&f| f != TileFormat::FP32)
            .collect();
        for f in pairs {
            for (seed, (m, k, n, limit, _, what)) in CASES.into_iter().enumerate() {
                let what = format!("{:?}, {what}", f.l1());
                let seed = seed as u64;
                let make = |v: Vec<f32>| -> Vec<f32> {
                    // BFP takes finite values only; its compression handles the rest.
                    if f == TileFormat::BF16 {
                        v
                    } else {
                        v.into_iter().map(|x| x.clamp(-1.0e30, 1.0e30)).collect()
                    }
                };
                let (a, da) = operand(s, f, &make(specials(300 + seed, m * k)), m, k);
                let (b, db) = operand(s, f, &make(specials(400 + seed, k * n)), k, n);
                s.set_matmul_k_block_limit(limit.and_then(std::num::NonZeroUsize::new));
                let got = mm(s, &a, &b, Fidelity::HiFi4);
                let old = mm(s, &da, &db, Fidelity::HiFi4);
                let got_v = s.download(&got).unwrap();
                assert_bits(&what, &got_v, &s.download(&old).unwrap());
                // A BF16 operand whose padding is not known zero is repaired first.
                if f == TileFormat::BF16 {
                    a.set_pad(Pad::Undefined);
                    b.set_pad(Pad::Undefined);
                    let repaired = mm(s, &a, &b, Fidelity::HiFi4);
                    assert_bits(
                        &format!("{what}, padding repaired"),
                        &s.download(&repaired).unwrap(),
                        &got_v,
                    );
                    s.free(repaired).unwrap();
                }
                for t in [a, da, b, db, got, old] {
                    s.free(t).unwrap();
                }
            }
            s.set_matmul_k_block_limit(None);
        }
    });
}

/// The controls: a packed operand's raw FP32 values are not what it holds, and a packed `A` is
/// read as it lies.
#[test]
fn packed_operands_are_their_own_values_and_read_as_they_lie() {
    with_session(|s| {
        let (m, k, n) = (32, 256, 64);
        let (va, vb) = (values(1, m * k), values(2, k * n));
        for f in formats().into_iter().filter(|&f| f != TileFormat::FP32) {
            let (a, da) = operand(s, f, &va, m, k);
            let raw = s.upload(&va, m, k).unwrap();
            let b = s.upload(&vb, k, n).unwrap();
            let (got, want) = (
                mm(s, &a, &b, Fidelity::HiFi4),
                mm(s, &raw, &b, Fidelity::HiFi4),
            );
            assert_ne!(
                s.download(&got).unwrap(),
                s.download(&want).unwrap(),
                "{f:?} storage must round `A`'s low bits away"
            );
            let at = s.upload(&va, k, m).unwrap();
            let (packed_t, dt) = operand(s, f, &va, k, m);
            assert!(
                s.matmul_dram(
                    &packed_t,
                    true,
                    &b,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    1 << 40
                )
                .is_err(),
                "{f:?}: a transposed packed `A`"
            );
            for t in [a, da, raw, b, got, want, at, packed_t, dt] {
                s.free(t).unwrap();
            }
        }
    });
}

/// Hypothesis H3 for every pair of edge formats, and its boundary: the derived phases give HiFi4's
/// bits wherever the products stay in FP32's range, signed zeros and subnormals included, and other
/// bits once products overflow it.
#[test]
fn derived_phases_are_hifi4_until_a_product_overflows() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let (m, k, n) = (64, 256, 96);
        for (huge, what) in [(1.0e15f32, "in range"), (1.0e30, "overflowing")] {
            let vals = |seed, len| -> Vec<f32> {
                specials(seed, len)
                    .into_iter()
                    .map(|x| if x.abs() > 1.0 { x.signum() * huge } else { x })
                    .collect()
            };
            let mut differ = 0;
            for fa in formats() {
                for fb in formats() {
                    let (a, da) = operand(s, fa, &vals(500, m * k), m, k);
                    let (b, db) = operand(s, fb, &vals(600, k * n), k, n);
                    s.set_derived_fidelity(false);
                    let hifi4 = mm(s, &a, &b, Fidelity::HiFi4);
                    s.set_derived_fidelity(true);
                    let derived = mm(s, &a, &b, Fidelity::HiFi4);
                    s.set_derived_fidelity(false);
                    let (h, d) = (s.download(&hifi4).unwrap(), s.download(&derived).unwrap());
                    let fewer =
                        SrcRoute::edges(fa, fb).unwrap().full_precision() != Fidelity::HiFi4;
                    if !fewer || huge < 1.0e20 {
                        assert_bits(&format!("{fa:?} x {fb:?}, {what}"), &d, &h);
                    } else if h.iter().zip(&d).any(|(x, y)| x.to_bits() != y.to_bits()) {
                        differ += 1;
                    }
                    for t in [a, da, b, db, hifi4, derived] {
                        s.free(t).unwrap();
                    }
                }
            }
            if huge > 1.0e20 {
                assert!(
                    differ > 0,
                    "overflowing products: the derived phases' bits differ"
                );
            }
        }
    });
}
