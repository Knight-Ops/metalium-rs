//! A weight stored in any packed format multiplies as its own values in FP32, bit for bit, on
//! every path a product takes (F1c of `docs/plans/dataflow-compiler.md`): the generic form of
//! `step233`, whose BF16 weight is one case of [`Elem::Packed`]. A BFP8/4/2 weight
//! ([`DramTensor::from_bfp`]) is read whole from its own slots (`record::gather_b_format`) and
//! unpacked in its own format into `SrcA` (`SrcRoute::Tf32WithB`), where it converts to BF16
//! losslessly; so its product is the FP32 product of the values it decodes to
//! (`Session::bfp_to_f32`), through every blocking, streaming and packing the block-stream path
//! has, where `bfp::matmul` built one output tile a job.
//!
//! Hypothesis H3 for mixed pairs (`docs/learnings/dataflow-compiler.md` §13): with TF32 in
//! `SrcB` and BFP4/BFP2 in `SrcA`, the derived phase count (three) gives HiFi4's bits.
//!
//! Controls: the weight's own FP32 values, which BFP storage rounds, give a different product;
//! a transposed packed weight and any route but TF32's are refused.
use tt_isa::format::{SrcType, TileFormat};
use tt_kernels::bfp::BfpFormat;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, Spread, TileChoice};
use tt_kernels::tensor::{BlockStream, DramTensor, Elem};
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

/// The packed formats a gate runs: ttsim refuses BFP2's `Src` format (`step94`).
fn formats() -> Vec<BfpFormat> {
    if cfg!(feature = "silicon") {
        vec![BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2]
    } else {
        vec![BfpFormat::Bfp8, BfpFormat::Bfp4]
    }
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

/// `step233`'s paths, each with the weight packed in every format.
#[test]
fn a_packed_weight_multiplies_as_its_decoded_values() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let cases: [Case; 8] = [
            (64, 256, 96, None, BlockStream::Off, "whole K, slots"),
            (
                37,
                100,
                70,
                None,
                BlockStream::Off,
                "ragged in every dimension",
            ),
            (5, 256, 64, None, BlockStream::Off, "a thin last tile row"),
            (64, 256, 96, Some(2), BlockStream::Off, "K blocks"),
            (1, 1024, 256, None, BlockStream::Off, "a packed one-row A"),
            (
                1,
                1024,
                128,
                Some(8),
                BlockStream::Off,
                "a packed A, K streamed",
            ),
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
        for format in formats() {
            let tile = TileFormat::new(format.l1_format());
            let derived =
                Fidelity::full_precision((TileFormat::FP32, SrcType::Tf32), (tile, SrcType::Bf16))
                    .expect("TF32 beside BFP has a full-precision phase count");
            for (seed, (m, k, n, limit, block, what)) in cases.into_iter().enumerate() {
                let what = format!("{format:?}, {what}");
                let seed = seed as u64;
                let a = s.upload(&values(100 + seed, m * k), m, k).unwrap();
                let w = s.upload(&values(200 + seed, k * n), k, n).unwrap();
                let packed = s.bfp_from_f32(&w, format).unwrap();
                // What the packed weight holds, in FP32: the reference's `B`.
                let decoded = s.bfp_to_f32(&packed).unwrap();
                let stored = DramTensor::from_bfp(packed);
                assert_eq!(stored.elem, Elem::Packed(tile));
                assert_eq!(stored.format().slot_bytes(), format.slot_bytes());
                s.set_matmul_k_block_limit(limit.and_then(std::num::NonZeroUsize::new));
                s.set_block_stream(block);
                let loops: &[bool] = match block {
                    BlockStream::Off => &[true],
                    _ => &[true, false],
                };
                for &looped in loops {
                    tt_kernels::matmul::set_block_loops(looped);
                    let want = mm(s, &a, &decoded, Fidelity::HiFi4);
                    let got = mm(s, &a, &stored, Fidelity::HiFi4);
                    let (want_v, got_v) = (s.download(&want).unwrap(), s.download(&got).unwrap());
                    assert!(want_v.iter().any(|&v| v != 0.0), "{what}: an empty product");
                    assert_bits(&format!("{what}, looped {looped}"), &got_v, &want_v);
                    // H3 for the mixed pair: the derived phases give HiFi4's bits.
                    let fewer = mm(s, &a, &stored, derived);
                    assert_bits(
                        &format!("{what}, looped {looped}, {derived:?} against HiFi4"),
                        &s.download(&fewer).unwrap(),
                        &got_v,
                    );
                    for t in [want, got, fewer] {
                        s.free(t).unwrap();
                    }
                }
                tt_kernels::matmul::set_block_loops(true);
                for t in [a, w, stored, decoded] {
                    s.free(t).unwrap();
                }
            }
            s.set_matmul_k_block_limit(None);
            s.set_block_stream(BlockStream::Auto);

            // The control: the weight's own FP32 values are not what its BFP storage holds.
            let (m, k, n) = (32, 256, 64);
            let a = s.upload(&values(1, m * k), m, k).unwrap();
            let w = s.upload(&values(2, k * n), k, n).unwrap();
            let stored = DramTensor::from_bfp(s.bfp_from_f32(&w, format).unwrap());
            let (want, got) = (
                mm(s, &a, &w, Fidelity::HiFi4),
                mm(s, &a, &stored, Fidelity::HiFi4),
            );
            assert_ne!(
                s.download(&got).unwrap(),
                s.download(&want).unwrap(),
                "{format:?} storage must round the weight's low bits away"
            );
            // A packed weight is read as it lies: transposed it is refused, and so is any route
            // but TF32's.
            for (transposed, route) in [
                (true, SrcRoute::Tf32FromFp32),
                (false, SrcRoute::Bf16FromFp32),
            ] {
                assert!(
                    s.matmul_dram(
                        &a,
                        false,
                        &stored,
                        transposed,
                        route,
                        Fidelity::HiFi4,
                        1 << 40
                    )
                    .is_err(),
                    "{format:?}: transposed {transposed}, {route:?}"
                );
            }
            for t in [a, w, stored, want, got] {
                s.free(t).unwrap();
            }
        }
    });
}
