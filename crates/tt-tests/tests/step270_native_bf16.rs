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
//!
//! **An element-wise op on BF16 operands is its widened op, rounded once**
//! (`an_eltwise_op_on_bf16_is_its_widened_op_rounded`). Each operand is read as a BF16 tile, widened
//! into `Dst` by the expand's own sequence, the F32 op's program runs, and the result is rounded by
//! the compress's quantize before a BF16 pack (`sfpu::kernel::Formats`). Over unary, scalar,
//! binary, column- and block-broadcast, ternary and comparison kinds -- the 1.7B's `silu`, `silu *
//! up` and residual `add` among them -- at whole tiles, ragged edges and a decode step's one row,
//! and over every BF16 class (signed zeros, subnormals, infinities, NaN payloads, extremes):
//! (i) unrounded (`Out::Fp32`), the op's FP32 result is the F32 op's on the widened operands, bit
//! for bit: the prologue is the expand; (ii) rounded and packed as FP32 (`Out::Bf16AsFp32`, the
//! simulator's form), it is that result rounded by the host's ties-even model (NaNs quieted,
//! subnormals flushed), and the rounding changes datums (a control on (i)); (iii) on silicon, the
//! BF16 result is `bf16_from_f32` of the F32 op on the widened operands, bit for bit, and so is the
//! host model. The negative control packs BF16 without the quantize (`Out::Bf16Unrounded`): it
//! must then differ.
//!
//! **A product with a BF16 result is its F32 product rounded once**
//! (`a_product_with_a_bf16_result_is_its_f32_product_rounded`). `Session::matmul_dram_as` with a
//! BF16 result rounds each output tile in `Dst` once its `K` chain is done, in its last `K`
//! chunk's run, and packs it as BF16. Over every schedule -- `K` whole in `Dst` (blocks, the
//! one-row packed path), `K` streamed in a one-row product, and 2-D blocks with `K` in chunks
//! continued in L1 halves (subblocked under half-sync, looped, written out) -- with FP32 and BF16
//! `A`s and BF16 (and on silicon BFP8) `B`s, ragged and whole: unrounded (`Out::Fp32`) the result
//! is the F32 product, bit for bit (the result's format moves no partial); rounded and packed as
//! FP32 it is the F32 product rounded by the host model (a partial rounded early would differ);
//! on silicon the BF16 result is `bf16_from_f32` of the F32 product, and without the quantize it
//! differs. The schedule whose partials pass through the result (`K` blocks) refuses a BF16 one
//! by name.
//!
//! **A BF16 row keeps a BF16 twin** (`a_bf16_row_keeps_a_bf16_twin`, silicon: a BF16 pack). A
//! one-row BF16 result -- an element-wise op's, a product's -- also writes its row packed
//! `[K / 32, 32]` as `pack_row` lays a BF16 row (32-byte face rows in FP32-sized slots), and a
//! product reads it as its packed `A`: the twin holds the row's bits, and the product is the one
//! of a copy of the row with no twin, bit for bit.
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::kernel::Out;
use tt_kernels::sfpu::ops::kind_sfpu;
use tt_kernels::tensor::{self, DramTensor, Eltwise, Pad};
use tt_kernels::tensor::{BlockStream, Elem};
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

/// BF16 ties-even narrowing of FP32 bits, NaNs quieted (sign and high payload kept), BF16
/// subnormals to signed zero: what `Session::bf16_from_f32` documents, as the high half of an
/// FP32 word. The host's model, independent of the SFPU program it checks.
fn round_bf16(x: u32) -> u32 {
    let r = if x & 0x7fff_ffff > 0x7f80_0000 {
        x | 0x0040_0000
    } else {
        x.wrapping_add(0x7fff + ((x >> 16) & 1))
    } & 0xffff_0000;
    if r & 0x7fff_ffff < 0x0080_0000 {
        r & 0x8000_0000
    } else {
        r
    }
}

/// `n` BF16 patterns, `salt` choosing which: every class first (signed zeros, the smallest and
/// largest subnormals and normals, infinities, quiet and signalling NaNs with payloads, ones and
/// ties-prone mantissas), then noise over every exponent.
fn patterns(n: usize, salt: u32) -> Vec<u16> {
    const SPECIAL: [u16; 20] = [
        0x0000, 0x8000, 0x0001, 0x807f, 0x0080, 0x8080, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0,
        0xffc1, 0x7f81, 0xff9f, 0x3f80, 0xbf80, 0x3f81, 0x4049, 0x3c00, 0x4b00,
    ];
    (0..n)
        .map(|i| {
            let h = (i as u32 ^ salt.wrapping_mul(0x9e37_79b9))
                .wrapping_mul(2_654_435_761)
                .rotate_left(13);
            if (i as u32 + salt) % 7 == 0 {
                SPECIAL[(h % SPECIAL.len() as u32) as usize]
            } else {
                // Mostly moderate exponents, so products and sums stay finite and round.
                let sign = (h >> 31) as u16;
                let exp = 0x60 + (h >> 8) % 0x40;
                sign << 15 | (exp as u16) << 7 | (h & 0x7f) as u16
            }
        })
        .collect()
}

#[test]
fn an_eltwise_op_on_bf16_is_its_widened_op_rounded() {
    use tt_kernels::kind;
    let op = |kind, scalar| Eltwise {
        kind,
        scalar,
        scalar2: 0.0,
    };
    // (op, the operands' shapes): the first is the result's.
    let (m, n) = (67, 100);
    let cases: Vec<(Eltwise, Vec<[usize; 2]>)> = vec![
        (op(kind_sfpu::SILU, 0.0), vec![[m, n]]),
        (op(kind_sfpu::EXP, 0.0), vec![[m, n]]),
        (op(kind_sfpu::RSQRT, 0.0), vec![[m, n]]),
        (op(kind::MUL_SCALAR, 0.125), vec![[m, n]]),
        (op(kind::ADD_SCALAR, 1.5), vec![[m, n]]),
        (op(kind::ADD, 0.0), vec![[m, n], [m, n]]),
        (op(kind::SUB, 0.0), vec![[m, n], [m, n]]),
        (op(kind::MUL, 0.0), vec![[m, n], [m, n]]),
        (op(kind_sfpu::SILU_MUL, 0.0), vec![[m, n], [m, n]]),
        (op(kind::MUL, 0.0), vec![[m, n], [m, 1]]),
        (op(kind::ADD, 0.0), vec![[96, n], [32, n]]),
        (op(kind_sfpu::MUL_ADD, 0.0), vec![[m, n], [m, n], [m, n]]),
        (op(kind_sfpu::LT, 0.0), vec![[m, n], [m, n]]),
        // A decode step's one row: a row-limited program, its quantize over that row alone.
        (op(kind::ADD, 0.0), vec![[1, 300], [1, 300]]),
        (op(kind_sfpu::SILU_MUL, 0.0), vec![[1, 300], [1, 300]]),
    ];
    if let Err(e) = fork_scope(|| {
        let mut s = open();
        let mut rounded = 0;
        #[allow(unused_mut)]
        let mut truncated = 0;
        for (case, (op, shapes)) in cases.iter().enumerate() {
            let what = format!("case {case}: {:#x} on {shapes:?}", op.kind);
            let bits: Vec<Vec<u16>> = shapes
                .iter()
                .enumerate()
                .map(|(k, &[r, c])| patterns(r * c, (case * 3 + k) as u32))
                .collect();
            let inputs: Vec<DramTensor> = shapes
                .iter()
                .zip(&bits)
                .map(|(&[r, c], b)| {
                    DramTensor::from_bf16(s.upload_bf16(b, r, c).unwrap(), Pad::Zero)
                })
                .collect();
            let widened: Vec<DramTensor> = inputs
                .iter()
                .map(|t| s.bf16_to_f32(&t.as_bf16().unwrap()).unwrap())
                .collect();
            let run = |s: &mut Session<_>, xs: &[DramTensor]| {
                s.eltwise3(*op, &xs[0], xs.get(1), xs.get(2)).unwrap()
            };
            let wide = run(&mut s, &widened);
            let f32_result = wide.elem == tensor::Elem::F32;
            let wide_bits = s.download_bits(&wide).unwrap();
            let expect: Vec<u32> = if f32_result {
                wide_bits.iter().map(|&w| round_bf16(w)).collect()
            } else {
                wide_bits.clone()
            };

            // (i) Unrounded: the prologue is the expand.
            tensor::probe_bf16_result(Out::Fp32);
            let unrounded = run(&mut s, &inputs);
            tensor::probe_bf16_result(Out::Bf16);
            assert_eq!(unrounded.elem, wide.elem, "{what}");
            let got = s.download_bits(&unrounded).unwrap();
            let differ = (0..got.len()).filter(|&i| got[i] != wide_bits[i]).count();
            assert_eq!(
                differ,
                0,
                "{what}: unrounded, {differ} of {} differ",
                got.len()
            );
            s.free(unrounded).unwrap();
            if !f32_result {
                // A comparison's result is no float: nothing to round.
                let native = run(&mut s, &inputs);
                assert_eq!(native.elem, wide.elem, "{what}");
                assert_eq!(s.download_bits(&native).unwrap(), wide_bits, "{what}");
                s.free(native).unwrap();
            } else {
                // (ii) Rounded, packed as FP32.
                tensor::probe_bf16_result(Out::Bf16AsFp32);
                let as_fp32 = run(&mut s, &inputs);
                tensor::probe_bf16_result(Out::Bf16);
                let got = s.download_bits(&as_fp32).unwrap();
                let differ: Vec<_> = (0..got.len()).filter(|&i| got[i] != expect[i]).collect();
                assert!(
                    differ.is_empty(),
                    "{what}: rounded, {} of {} differ, first {:?}",
                    differ.len(),
                    got.len(),
                    differ
                        .iter()
                        .take(6)
                        .map(|&i| format!(
                            "{i}: {:#010x} for {:#010x} (of {:#010x})",
                            got[i], expect[i], wide_bits[i]
                        ))
                        .collect::<Vec<_>>()
                );
                rounded += (0..got.len()).filter(|&i| got[i] != wide_bits[i]).count();
                s.free(as_fp32).unwrap();

                #[cfg(feature = "silicon")]
                {
                    // (iii) The BF16 result, against the composition it replaces.
                    let native = run(&mut s, &inputs);
                    assert_eq!(native.elem, tensor::Elem::BF16, "{what}");
                    assert_eq!(native.slot(0).len(), 2112, "{what}: BF16 slots");
                    let got = s.download_bf16(&native.as_bf16().unwrap()).unwrap();
                    let composed = s.bf16_from_f32(&wide).unwrap();
                    let want = s.download_bf16(&composed).unwrap();
                    let differ: Vec<_> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
                    assert!(
                        differ.is_empty(),
                        "{what}: BF16, {} of {} differ from the composition, first {:?}",
                        differ.len(),
                        got.len(),
                        differ
                            .iter()
                            .take(6)
                            .map(|&i| format!("{i}: {:#06x} for {:#06x}", got[i], want[i]))
                            .collect::<Vec<_>>()
                    );
                    let model = (0..want.len())
                        .filter(|&i| u32::from(want[i]) << 16 != expect[i])
                        .count();
                    assert_eq!(
                        model, 0,
                        "{what}: the host model disagrees with bf16_from_f32"
                    );
                    // The negative control: no quantize, and the pack truncates.
                    tensor::probe_bf16_result(Out::Bf16Unrounded);
                    let control = run(&mut s, &inputs);
                    tensor::probe_bf16_result(Out::Bf16);
                    let read = s.download_bf16(&control.as_bf16().unwrap()).unwrap();
                    let changed = (0..read.len()).filter(|&i| read[i] != want[i]).count();
                    // Exact kinds (a power-of-two scale) have nothing to round: the run's sum.
                    truncated += changed;
                    // What the packer makes of an unrounded FP32 datum: its high half?
                    let cut = (0..read.len())
                        .filter(|&i| u32::from(read[i]) != wide_bits[i] >> 16)
                        .count();
                    println!(
                        "MEASURE {what}: unrounded BF16 pack differs in {changed} of {}, and \
                         from each datum's high half in {cut}",
                        read.len()
                    );
                    s.free(control).unwrap();
                    s.free(native).unwrap();
                    let _ = s.free(DramTensor::from_bf16(composed, Pad::Undefined));
                }
            }
            s.free(wide).unwrap();
            for t in widened.into_iter().chain(inputs) {
                s.free(t).unwrap();
            }
        }
        // The control on (i): rounding is not the identity on these results.
        println!(
            "MEASURE native bf16 eltwise: {} cases, {rounded} datums changed by rounding",
            cases.len()
        );
        assert!(rounded > 1000, "only {rounded} datums rounded");
        #[cfg(feature = "silicon")]
        assert!(
            truncated > 1000,
            "control: the unrounded pack changed only {truncated} datums"
        );
        let _ = truncated;
    }) {
        panic!("{e}");
    }
}

#[test]
fn a_product_with_a_bf16_result_is_its_f32_product_rounded() {
    use tt_kernels::matmul::{set_block_loops, set_dst_subblocks};
    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Blocks,
        Stream,
        Subblocked,
        Looped,
        WrittenOut,
    }
    // (schedule, [m, k, n], A BF16, B: 0 BF16, 1 FP32, 2 BFP8 (silicon only))
    let cases: Vec<(Kind, [usize; 3], bool, u8)> = vec![
        (Kind::Blocks, [67, 300, 200], false, 1),
        (Kind::Blocks, [67, 300, 200], true, 0),
        (Kind::Blocks, [1, 1024, 384], true, 0),
        (Kind::Blocks, [1, 1024, 384], false, 1),
        (Kind::Stream, [1, 2048, 96], true, 0),
        (Kind::Subblocked, [128, 512, 128], true, 0),
        (Kind::Subblocked, [96, 256, 100], false, 1),
        (Kind::Looped, [128, 512, 128], true, 0),
        (Kind::WrittenOut, [128, 512, 128], true, 0),
        #[cfg(feature = "silicon")]
        (Kind::Blocks, [1, 2048, 512], true, 2),
        #[cfg(feature = "silicon")]
        (Kind::Subblocked, [256, 1024, 256], true, 2),
    ];
    if let Err(e) = fork_scope(|| {
        let mut s = open();
        let floats = |n: usize, salt: u32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = (i as u32 ^ salt.wrapping_mul(0x85eb_ca6b))
                        .wrapping_mul(2_654_435_761)
                        .rotate_left(11);
                    let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
                    sign * (1.0 + (h >> 9) as f32 / (1u32 << 23) as f32)
                        * f32::powi(2.0, (h >> 4 & 7) as i32 - 4)
                })
                .collect()
        };
        let mut rounded = 0;
        #[allow(unused_mut)]
        let mut truncated = 0;
        for (case, &(kind, [m, k, n], a_bf16, b_kind)) in cases.iter().enumerate() {
            let what = format!(
                "case {case}: {kind:?} [{m}, {k}] @ [{k}, {n}], A BF16 {a_bf16}, B {b_kind}"
            );
            s.set_block_stream(match kind {
                Kind::Subblocked | Kind::Looped | Kind::WrittenOut => BlockStream::Fixed([4, 4, 4]),
                _ => BlockStream::Off,
            });
            set_dst_subblocks(matches!(kind, Kind::Subblocked));
            set_block_loops(matches!(kind, Kind::Looped));
            s.set_matmul_k_block_limit(match kind {
                Kind::Stream => std::num::NonZeroUsize::new(16),
                _ => None,
            });
            s.set_streamed_k(matches!(kind, Kind::Stream));
            let a = if a_bf16 {
                let bits: Vec<u16> = floats(m * k, case as u32)
                    .iter()
                    .map(|v| (round_bf16(v.to_bits()) >> 16) as u16)
                    .collect();
                DramTensor::from_bf16(s.upload_bf16(&bits, m, k).unwrap(), Pad::Zero)
            } else {
                s.upload(&floats(m * k, case as u32), m, k).unwrap()
            };
            let wf = floats(k * n, 1000 + case as u32);
            let b = match b_kind {
                0 => {
                    let bits: Vec<u16> = wf
                        .iter()
                        .map(|v| (round_bf16(v.to_bits()) >> 16) as u16)
                        .collect();
                    s.upload_bf16_weight(&bits, k, n).unwrap()
                }
                1 => s.upload(&wf, k, n).unwrap(),
                _ => {
                    let w = s.upload(&wf, k, n).unwrap();
                    let stored = s.store_bfp(&w, tt_kernels::bfp::BfpFormat::Bfp8).unwrap();
                    s.free(w).unwrap();
                    stored
                }
            };
            let product = |s: &mut Session<_>, out: Elem| {
                s.matmul_dram_as(
                    &a,
                    false,
                    &b,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    1 << 40,
                    out,
                )
            };
            let wide = product(&mut s, Elem::F32).unwrap();
            let wide_bits = s.download_bits(&wide).unwrap();
            let expect: Vec<u32> = wide_bits.iter().map(|&w| round_bf16(w)).collect();

            tensor::probe_bf16_result(Out::Fp32);
            let unrounded = product(&mut s, Elem::BF16);
            tensor::probe_bf16_result(Out::Bf16);
            let unrounded = unrounded.unwrap();
            assert_eq!(
                s.download_bits(&unrounded).unwrap(),
                wide_bits,
                "{what}: unrounded"
            );
            s.free(unrounded).unwrap();

            tensor::probe_bf16_result(Out::Bf16AsFp32);
            let as_fp32 = product(&mut s, Elem::BF16);
            tensor::probe_bf16_result(Out::Bf16);
            let as_fp32 = as_fp32.unwrap();
            let got = s.download_bits(&as_fp32).unwrap();
            let differ: Vec<_> = (0..got.len()).filter(|&i| got[i] != expect[i]).collect();
            assert!(
                differ.is_empty(),
                "{what}: rounded, {} of {} differ, first {:?}",
                differ.len(),
                got.len(),
                differ
                    .iter()
                    .take(6)
                    .map(|&i| format!(
                        "{i}: {:#010x} for {:#010x} (of {:#010x})",
                        got[i], expect[i], wide_bits[i]
                    ))
                    .collect::<Vec<_>>()
            );
            rounded += (0..got.len()).filter(|&i| got[i] != wide_bits[i]).count();
            s.free(as_fp32).unwrap();

            #[cfg(feature = "silicon")]
            {
                let native = product(&mut s, Elem::BF16).unwrap();
                assert_eq!(native.elem, Elem::BF16, "{what}");
                assert_eq!(native.slot(0).len(), 2112, "{what}: BF16 slots");
                let got = s.download_bf16(&native.as_bf16().unwrap()).unwrap();
                let composed = s.bf16_from_f32(&wide).unwrap();
                let want = s.download_bf16(&composed).unwrap();
                let differ: Vec<_> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
                assert!(
                    differ.is_empty(),
                    "{what}: BF16, {} of {} differ from the composition, first {:?}",
                    differ.len(),
                    got.len(),
                    differ
                        .iter()
                        .take(6)
                        .map(|&i| format!("{i}: {:#06x} for {:#06x}", got[i], want[i]))
                        .collect::<Vec<_>>()
                );
                tensor::probe_bf16_result(Out::Bf16Unrounded);
                let control = product(&mut s, Elem::BF16);
                tensor::probe_bf16_result(Out::Bf16);
                let control = control.unwrap();
                let read = s.download_bf16(&control.as_bf16().unwrap()).unwrap();
                truncated += (0..read.len()).filter(|&i| read[i] != want[i]).count();
                s.free(control).unwrap();
                s.free(native).unwrap();
                let _ = s.free(DramTensor::from_bf16(composed, Pad::Undefined));
            }
            for t in [wide, a, b] {
                s.free(t).unwrap();
            }
        }
        // `K` in blocks passes its partials through the result: refused a BF16 one, by name.
        s.set_block_stream(BlockStream::Off);
        s.set_streamed_k(false);
        s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(4));
        let a = s.upload(&floats(67 * 300, 7), 67, 300).unwrap();
        let b = s.upload(&floats(300 * 64, 8), 300, 64).unwrap();
        let refused = s
            .matmul_dram_as(
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                1 << 40,
                Elem::BF16,
            )
            .unwrap_err()
            .to_string();
        assert!(refused.contains("K-blocked"), "{refused}");
        set_dst_subblocks(true);
        println!(
            "MEASURE native bf16 products: {} cases, {rounded} datums changed by rounding",
            cases.len()
        );
        assert!(rounded > 1000, "only {rounded} datums rounded");
        #[cfg(feature = "silicon")]
        assert!(
            truncated > 1000,
            "control: the unrounded pack changed only {truncated} datums"
        );
        let _ = truncated;
    }) {
        panic!("{e}");
    }
}

#[test]
#[cfg_attr(
    not(feature = "silicon"),
    ignore = "a BF16 result's pack: silicon only (ttsim refuses PACR mode 0x105)"
)]
fn a_bf16_row_keeps_a_bf16_twin() {
    use tt_kernels::kind;
    if let Err(e) = fork_scope(|| {
        let mut s = open();
        s.set_packed_twins(true);
        let row = |s: &mut Session<_>, salt| {
            let bits: Vec<u16> = patterns(2048, salt)
                .into_iter()
                .map(|p| if p & 0x7f80 == 0x7f80 { 0x3f80 } else { p })
                .collect();
            DramTensor::from_bf16(s.upload_bf16(&bits, 1, 2048).unwrap(), Pad::Zero)
        };
        let (x, y) = (row(&mut s, 1), row(&mut s, 2));
        let add = Eltwise {
            kind: kind::ADD,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let r = s.eltwise(add, &x, Some(&y)).unwrap();
        assert_eq!(r.elem, Elem::BF16);
        let wbits: Vec<u16> = patterns(2048 * 2048, 3)
            .into_iter()
            .map(|p| (p & 0x807f) | 0x3c00)
            .collect();
        let w = s.upload_bf16_weight(&wbits, 2048, 2048).unwrap();
        // The twin holds the row: packed row `kk` is columns `32 kk..32 kk + 32`.
        let holds = |s: &mut Session<_>, t: &DramTensor, what: &str| {
            let twin = s
                .packed_twin(t)
                .unwrap_or_else(|| panic!("{what}: no twin"))
                .clone();
            assert_eq!(twin.elem, Elem::BF16, "{what}: a BF16 twin");
            let packed = s.download_bf16(&twin.as_bf16().unwrap()).unwrap();
            let want = s.download_bf16(&t.as_bf16().unwrap()).unwrap();
            let differ = (0..want.len()).filter(|&i| packed[i] != want[i]).count();
            assert_eq!(
                differ,
                0,
                "{what}: {differ} of {} twin datums differ",
                want.len()
            );
        };
        holds(&mut s, &r, "an element-wise result");
        let product = |s: &mut Session<_>, a: &DramTensor| {
            s.matmul_dram_as(
                a,
                false,
                &w,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                1 << 40,
                Elem::BF16,
            )
            .unwrap()
        };
        let p = product(&mut s, &r);
        holds(&mut s, &p, "a product's result");
        // Read through the twin, and from a copy that has none.
        let copy = s.copy(&r).unwrap();
        assert!(s.packed_twin(&copy).is_none(), "a copy has no twin");
        let q = product(&mut s, &copy);
        let (pb, qb) = (
            s.download_bf16(&p.as_bf16().unwrap()).unwrap(),
            s.download_bf16(&q.as_bf16().unwrap()).unwrap(),
        );
        assert_eq!(
            pb, qb,
            "a product through the twin is the product without it"
        );
        for t in [p, q, copy, r, w, x, y] {
            s.free(t).unwrap();
        }
    }) {
        panic!("{e}");
    }
}
