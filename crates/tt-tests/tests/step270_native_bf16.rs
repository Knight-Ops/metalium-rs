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
//!
//! **A BF16 tensor lives in its own slots, and data movement keeps it there**
//! (`a_bf16_tensor_is_copied_in_its_own_slots`). `DramTensor::alloc_elem` gives a BF16 tensor
//! 2,112-byte slots, and its reference names them (`TensorRef::slot`), so `READ_RUN` and
//! `WRITE_RUN` step by them: a copy of a `[67, 300]` BF16 tensor (30 tiles over every channel,
//! ragged edges) lands in BF16 slots with every datum's bits. The control names no slot for the
//! source (`tensor::probe_unnamed_slot`, as every reference was before slots were named): the
//! same copy then reads it at FP32 slots and moves other tiles' bytes. A download as 32-bit
//! datums, which would read 4,096 bytes a tile, is refused by name.
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Pad};
use tt_ttsim::fork_scope;

fn open() -> Session<impl tt_device::Transport> {
    #[cfg(not(feature = "silicon"))]
    let mut s = {
        let sim = Box::leak(Box::new(tt_ttsim::Simulator::open().unwrap()));
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
            |_, _| Ok(None),
        )
        .unwrap()
    };
    #[cfg(feature = "silicon")]
    let mut s = Session::open_card(
        tt_tests::backend::device_index(),
        tt_firmware_images::ROLES,
        TileChoice::Count(2),
    )
    .unwrap();
    s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
        .unwrap();
    s
}

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

#[test]
fn a_bf16_tensor_is_copied_in_its_own_slots() {
    if let Err(e) = fork_scope(|| {
        let mut s = open();
        let (rows, cols) = (67, 300);
        // Every pattern class, then noise: a copy moves bits, whatever they are.
        let bits: Vec<u16> = (0..rows * cols)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761).rotate_left(7) as u16)
            .collect();
        let source =
            DramTensor::from_bf16(s.upload_bf16(&bits, rows, cols).unwrap(), Pad::Undefined);
        let copy = s.copy(&source).unwrap();
        assert_eq!(copy.elem, tt_kernels::tensor::Elem::BF16);
        assert_eq!(copy.slot(0).len(), 2112, "a BF16 copy takes BF16 slots");
        let got = s.download_bf16(&copy.as_bf16().unwrap()).unwrap();
        let differ = (0..rows * cols).filter(|&i| got[i] != bits[i]).count();
        assert_eq!(differ, 0, "{differ} of {} datums differ", rows * cols);
        // Its slots are no FP32 tile's: the 32-bit datum paths refuse it by name.
        let refused = s.download_bits(&copy).unwrap_err().to_string();
        assert!(refused.contains("2112-byte slots"), "{refused}");

        // The control: the same copy with the source's slots unnamed reads it in FP32-sized steps.
        tt_kernels::tensor::probe_unnamed_slot(Some(&source));
        let wrong = s.copy(&source);
        tt_kernels::tensor::probe_unnamed_slot(None);
        let wrong = wrong.unwrap();
        let read = s.download_bf16(&wrong.as_bf16().unwrap()).unwrap();
        let moved = (0..rows * cols).filter(|&i| read[i] != bits[i]).count();
        assert!(
            moved > rows * cols / 2,
            "control: only {moved} datums moved"
        );
        s.free(wrong).unwrap();
        s.free(copy).unwrap();
        s.free(source).unwrap();
    }) {
        panic!("{e}");
    }
}
