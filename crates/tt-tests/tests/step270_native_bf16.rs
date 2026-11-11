//! Roadmap W1, native BF16 arithmetic (`docs/plans/tt-metal-parity.md` §5 W1; ADR 0006).
//!
//! A native BF16 op must compute what today's composition computes. That composition widens each
//! BF16 operand on Tensix (`Session::bf16_to_f32`, SrcA -> MOVA2D), runs the F32 op, and narrows
//! its result once (`Session::bf16_from_f32`: SFPU ties-even quantize, then a BF16 pack). Read
//! natively, the op is then a schedule choice under ADR 0005, legal under both contracts.
//!
//! **A product's BF16 edge is its widened FP32 edge, over every pattern**
//! (`a_bf16_edge_is_its_widened_fp32_edge`). Each of the 65,536 BF16 patterns is a row of `A`
//! (`[65536, 1]`, its padding zero), and `B` is `[[1.0]]`. The product read with `A` as a BF16
//! edge (`SrcRoute::Edges`) must equal, bit for bit, the product read from
//! `bf16_to_f32(A)` (the expand) as an FP32 edge. That covers signed zeros, subnormals,
//! infinities, NaN payloads and every normal. The negative control sets bit 13 of every
//! widened datum, which survives the edge's TF32 cut: the products must then differ.
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Pad};
use tt_ttsim::fork_scope;

fn product(s: &mut Session<impl tt_device::Transport>, a: &DramTensor, b: &DramTensor) -> Vec<u32> {
    let out = s
        .matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            400_000_000,
        )
        .unwrap();
    let bits = s.download_bits(&out).unwrap();
    s.free(out).unwrap();
    bits
}

#[test]
fn a_bf16_edge_is_its_widened_fp32_edge() {
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
        let patterns: Vec<u16> = (0..=u16::MAX).collect();
        let rows = patterns.len();
        let a = s.upload_bf16(&patterns, rows, 1).unwrap();
        let widened = s.bf16_to_f32(&a).unwrap();
        let b = s.upload(&[1.0], 1, 1).unwrap();
        let a = DramTensor::from_bf16(a, Pad::Zero);

        let native = product(&mut s, &a, &b);
        let wide = product(&mut s, &widened, &b);
        let differ: Vec<_> = (0..rows).filter(|&i| native[i] != wide[i]).collect();
        assert!(
            differ.is_empty(),
            "{} of {rows} patterns differ, first {:?}",
            differ.len(),
            differ
                .iter()
                .take(8)
                .map(|&i| format!(
                    "{:#06x}: edge {:#010x}, widened {:#010x}",
                    i, native[i], wide[i]
                ))
                .collect::<Vec<_>>()
        );

        // What the expand itself gives each pattern, by class (a reading for the docs).
        let bits = s.download_bits(&widened).unwrap();
        let mut same = 0;
        let mut classes = std::collections::BTreeMap::<&str, usize>::new();
        for (i, &got) in bits.iter().enumerate() {
            if got == (i as u32) << 16 {
                same += 1;
                continue;
            }
            let class = match (i >> 7) & 0xff {
                0 if i & 0x7f == 0 => "zero",
                0 => "subnormal",
                0xff if i & 0x7f == 0 => "infinity",
                0xff => "nan",
                _ => "normal",
            };
            *classes.entry(class).or_default() += 1;
        }
        println!("MEASURE bf16 expand: {same} of {rows} patterns widen to bits << 16; others {classes:?}");

        // Negative control: a datum the edge would read differently.
        let marked: Vec<f32> = bits.iter().map(|&w| f32::from_bits(w | 1 << 13)).collect();
        let marked = s.upload(&marked, rows, 1).unwrap();
        let control = product(&mut s, &marked, &b);
        let changed = (0..rows).filter(|&i| control[i] != wide[i]).count();
        assert!(
            changed > rows / 2,
            "control: only {changed} of {rows} products changed"
        );

        for t in [widened, b, marked, a] {
            s.free(t).unwrap();
        }
    }) {
        panic!("{e}");
    }
}
