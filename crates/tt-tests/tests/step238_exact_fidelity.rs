//! Hypothesis H3 of `docs/learnings/dataflow-compiler.md`: a product that runs only the
//! fidelity phases its operand formats need ([`Fidelity::full_precision`], from
//! `SrcASrcB.md`'s phase slices) returns the same bits as one that runs all four.
//!
//! Why it might: the skipped phases multiply a slice that is zero for these formats, and
//! adding a zero group sum to an FP32 `Dst` leaves it unchanged. Why it might not: the
//! Matrix Unit aligns each group of eight terms to a quantum taken from its operands'
//! exponents, a zero slice included (`silicon-operating-notes.md`, "Matmul accumulation"),
//! and a zero added under the group's rounding could still move a signed zero. So the
//! products here put terms of very different sizes in one group, carry zeros of both
//! signs, run across K reloads, and are compared bit for bit.
//!
//! Negative control: BFP8 at one phase, below what its `SrcA` operand needs, must differ.
use tt_isa::format::{SrcType, TileFormat};
use tt_kernels::{
    bfp::BfpFormat,
    matmul::Fidelity,
    session::{Session, TileChoice},
};
use tt_ttsim::fork_scope;
#[cfg(not(feature = "silicon"))]
type Transport<'a> = tt_ttsim::LibTtsim<'a>;
#[cfg(feature = "silicon")]
type Transport<'a> = tt_kmd::Kmd;

fn with_session(f: impl FnOnce(&mut Session<Transport<'_>>)) {
    if let Err(e) = fork_scope(|| {
        #[cfg(not(feature = "silicon"))]
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        #[cfg(not(feature = "silicon"))]
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        #[cfg(not(feature = "silicon"))]
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
            |_, _| Ok(None),
        )
        .unwrap();
        #[cfg(feature = "silicon")]
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// A deterministic spread of values: random signs and significands, magnitudes from 2^-6
/// to 2^6, every ninth an exact zero of either sign.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|i| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (x >> 33) as u32;
            if i % 9 == 4 {
                return if r & 1 == 0 { 0.0 } else { -0.0 };
            }
            let sign = if r & 1 == 0 { 1.0 } else { -1.0 };
            let significand = 1.0 + ((r >> 1) & 0x7f_ffff) as f32 / (1u32 << 23) as f32;
            let exponent = ((r >> 24) % 13) as i32 - 6;
            sign * significand * 2f32.powi(exponent)
        })
        .collect()
}

/// `B`'s rows scaled by `2^(k mod 12)`: a group of eight consecutive `k` holds terms
/// whose exponents differ by up to seven, so each group is aligned to its largest.
fn spread_rows(b: &mut [f32], k: usize, n: usize) {
    for q in 0..k {
        let scale = 2f32.powi((q % 12) as i32);
        for v in &mut b[q * n..(q + 1) * n] {
            *v *= scale;
        }
    }
}

fn product(
    s: &mut Session<Transport<'_>>,
    a: &tt_kernels::bfp::BfpTensor,
    b: &tt_kernels::bfp::BfpTensor,
    fidelity: Fidelity,
) -> Vec<u32> {
    let out = s.bfp_matmul(a, false, b, false, fidelity).unwrap();
    let bits = s
        .download(&out)
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    s.free(out).unwrap();
    bits
}

#[test]
fn derived_fidelity_equals_hifi4_bit_for_bit() {
    with_session(|s| {
        for (m, k, n) in [(37, 65, 35), (3, 257, 5), (64, 96, 64)] {
            let a = values(m * k, 0x5eed_0001 ^ (m * k) as u64);
            let mut b = values(k * n, 0x5eed_0002 ^ (k * n) as u64);
            spread_rows(&mut b, k, n);
            let af = s.upload(&a, m, k).unwrap();
            let bf = s.upload(&b, k, n).unwrap();
            for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
                #[cfg(not(feature = "silicon"))]
                if format == BfpFormat::Bfp2 {
                    continue; // ttsim refuses Src format 15 (step94)
                }
                let tile = TileFormat::new(format.l1_format());
                let derived =
                    Fidelity::full_precision((tile, SrcType::Bf16), (tile, SrcType::Bf16))
                        .expect("a BFP pair has a full-precision phase count");
                assert_eq!(
                    derived.phases(),
                    match format {
                        BfpFormat::Bfp8 => 2,
                        BfpFormat::Bfp4 | BfpFormat::Bfp2 => 1,
                    }
                );
                let ap = s.bfp_from_f32(&af, format).unwrap();
                let bp = s.bfp_from_f32(&bf, format).unwrap();
                for limit in [None, std::num::NonZeroUsize::new(1)] {
                    s.set_matmul_k_block_limit(limit);
                    let full = product(s, &ap, &bp, Fidelity::HiFi4);
                    assert!(full.iter().any(|&v| f32::from_bits(v) != 0.0));
                    let fewer = product(s, &ap, &bp, derived);
                    let differ = full.iter().zip(&fewer).filter(|(x, y)| x != y).count();
                    assert_eq!(
                        differ,
                        0,
                        "{format:?} {m}x{k}x{n} {limit:?}: {derived:?} differs from HiFi4 in \
                         {differ} of {} outputs",
                        full.len()
                    );
                    if format == BfpFormat::Bfp8 {
                        // Negative control: one phase drops SrcA's low slice, which BFP8's
                        // six fraction bits reach, so the gate can see a difference.
                        let below = product(s, &ap, &bp, Fidelity::Lo);
                        assert_ne!(below, full, "BFP8 at one phase must lose bits");
                    }
                }
                s.set_matmul_k_block_limit(None);
                s.free_bfp(ap).unwrap();
                s.free_bfp(bp).unwrap();
            }
            s.free(af).unwrap();
            s.free(bf).unwrap();
        }
    });
}
