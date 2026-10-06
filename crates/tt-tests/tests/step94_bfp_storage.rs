//! Resident block-float conversion, physical allocation and replay lifetimes.
use tt_kernels::{
    bfp::BfpFormat,
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

// Independent f64 arithmetic oracle, with the measured late conversion's
// special exponent-255 significands and subnormal flushing.
fn block(row: &[f32], bits: u32) -> (u8, Vec<u8>) {
    let exp = row.iter().map(|x| (x.to_bits() >> 23) as u8).max().unwrap();
    let quantum = 2f64.powi(exp as i32 - 133);
    let data = row
        .iter()
        .map(|x| {
            let xbits = x.to_bits() & 0xffff0000;
            let scaled = if xbits & 0x7f800000 == 0 {
                0.0
            } else if xbits & 0x7f800000 == 0x7f800000 {
                64.0 + ((xbits >> 16) & 127) as f64 / 2.0
            } else {
                f32::from_bits(xbits).abs() as f64 / quantum
            };
            let mag = (scaled.round().min(127.0) as u8) >> (8 - bits);
            if mag == 0 {
                0
            } else {
                mag | (((xbits >> 31) as u8) << (bits - 1))
            }
        })
        .collect();
    (exp, data)
}

// A second encoding oracle uses integer significands and exact rounded shifts;
// it shares no f64 arithmetic with block(), or decode arithmetic with hardware.
fn integer_block(values: &[f32], width: u32) -> (u8, Vec<u8>) {
    let shared = values
        .iter()
        .map(|v| (v.to_bits() >> 23) as u8)
        .max()
        .unwrap();
    let data = values
        .iter()
        .map(|v| {
            let word = v.to_bits();
            let exponent = ((word >> 23) & 255) as u8;
            let shift = (shared - exponent) as u32 + 1;
            let significand = 128 + ((word >> 16) & 127);
            let magnitude = if exponent == 0 || shift >= 9 {
                0
            } else {
                ((significand + (1 << (shift - 1))) >> shift).min(127)
            } >> (8 - width);
            if magnitude == 0 {
                0
            } else {
                magnitude as u8 | ((word >> 31) as u8) << (width - 1)
            }
        })
        .collect();
    (shared, data)
}

fn expected(
    values: &[f32],
    rows: usize,
    cols: usize,
    format: BfpFormat,
) -> (Vec<Vec<u8>>, Vec<u32>) {
    use tt_isa::tile::{bfp2_to_bf16, bfp4_to_bf16, bfp8_to_bf16};
    let image = format.tile_image();
    let bits = format.l1_format().datum_bits();
    let mut raw = Vec::new();
    let mut decoded = vec![0; rows * cols];
    for tr in 0..rows.div_ceil(32) {
        for tc in 0..cols.div_ceil(32) {
            let face_values: Vec<_> = (0..1024)
                .map(|i| {
                    let r = tr * 32 + i / 256 / 2 * 16 + i % 256 / 16;
                    let c = tc * 32 + i / 256 % 2 * 16 + i % 16;
                    if r < rows && c < cols {
                        values[r * cols + c]
                    } else {
                        0.0
                    }
                })
                .collect();
            let mut bytes = vec![0; image.total_bytes()];
            for (g, row) in face_values.chunks_exact(16).enumerate() {
                let (exp, data) = block(row, bits);
                assert_eq!(
                    (exp, data.clone()),
                    integer_block(row, bits),
                    "independent encoding models"
                );
                bytes[16 + g] = exp;
                for (j, datum) in data.into_iter().enumerate() {
                    let i = g * 16 + j;
                    let at = image.datum_bit_offset(i);
                    bytes[at / 8] |= datum << (at % 8);
                    let d = match format {
                        BfpFormat::Bfp8 => bfp8_to_bf16(datum, exp),
                        BfpFormat::Bfp4 => bfp4_to_bf16(datum, exp),
                        BfpFormat::Bfp2 => bfp2_to_bf16(datum, exp),
                    };
                    let r = tr * 32 + i / 256 / 2 * 16 + i % 256 / 16;
                    let c = tc * 32 + i / 256 % 2 * 16 + i % 16;
                    if r < rows && c < cols {
                        decoded[r * cols + c] = if d & 0x7f80 == 0 { 0 } else { (d as u32) << 16 };
                    }
                }
            }
            raw.push(bytes);
        }
    }
    (raw, decoded)
}

#[test]
fn resident_formats_match_physical_and_decoded_oracles() {
    with_session(|s| {
        for (rows, cols) in [(16, 16), (37, 65), (64, 96)] {
            let values: Vec<_> = (0..rows * cols)
                .map(|i| (i as f32 % 73.0 - 36.0) / 8.0)
                .collect();
            let input = s.upload(&values, rows, cols).unwrap();
            for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
                let packed = s.bfp_from_f32(&input, format).unwrap();
                assert_eq!(packed.slot(0).len(), format.slot_bytes());
                assert!(packed.slot(0).len() < input.slot(0).len());
                assert_eq!(packed.pad(), tt_kernels::tensor::Pad::Zero);
                let (raw, decoded) = expected(&values, rows, cols, format);
                let got_raw = s.download_bfp_raw(&packed).unwrap();
                for (tile, (got, want)) in got_raw.iter().zip(raw).enumerate() {
                    assert_eq!(&got[16..], &want[16..], "{format:?} tile {tile}");
                }
                let got = s.download_bfp(&packed).unwrap();
                for (i, (&got, &want)) in got.iter().zip(&decoded).enumerate() {
                    assert_eq!(got.to_bits(), want, "{format:?} {rows}x{cols} datum {i}");
                }
                s.free_bfp(packed).unwrap();
            }
            s.free(input).unwrap();
        }
    });
}

#[test]
fn conversion_replays_changed_input_and_defers_packed_frees() {
    with_session(|s| {
        let input = s.upload(&vec![1.0; 37 * 65], 37, 65).unwrap();
        for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
            s.begin_trace().unwrap();
            let packed = s.bfp_from_f32(&input, format).unwrap();
            let output = s.bfp_to_f32(&packed).unwrap();
            s.free_bfp(packed).unwrap();
            let trace = s.end_trace().unwrap();
            s.write(&input, &vec![2.0; 37 * 65]).unwrap();
            s.replay(trace).unwrap();
            assert_eq!(s.download(&output).unwrap(), vec![2.0; 37 * 65]);
            s.free(output).unwrap();
            s.release_trace(trace).unwrap();
        }
        s.free(input).unwrap();
    });
}

#[test]
fn packed_products_match_decoded_operands_through_k_reloads() {
    with_session(|s| {
        for (m, k, n) in [(3, 7, 5), (37, 65, 35), (3, 257, 5)] {
            let a: Vec<_> = (0..m * k)
                .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
                .collect();
            let b: Vec<_> = (0..k * n)
                .map(|i| if i % 5 == 0 { 1.0 } else { -1.0 })
                .collect();
            let af = s.upload(&a, m, k).unwrap();
            let bf = s.upload(&b, k, n).unwrap();
            for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
                #[cfg(not(feature = "silicon"))]
                if format == BfpFormat::Bfp2 {
                    continue;
                } // ttsim matrix fmt 15 refusal
                let ap = s.bfp_from_f32(&af, format).unwrap();
                let bp = s.bfp_from_f32(&bf, format).unwrap();
                let want: Vec<_> = (0..m)
                    .flat_map(|r| {
                        let (a, b) = (&a, &b);
                        (0..n)
                            .map(move |c| (0..k).map(|q| a[r * k + q] * b[q * n + c]).sum::<f32>())
                    })
                    .collect();
                assert_ne!(want, vec![0.0; m * n]);
                for limit in [
                    None,
                    std::num::NonZeroUsize::new(1),
                    std::num::NonZeroUsize::new(2),
                ] {
                    s.set_matmul_k_block_limit(limit);
                    let out = s
                        .bfp_matmul(&ap, false, &bp, false, tt_kernels::matmul::Fidelity::HiFi4)
                        .unwrap();
                    assert_eq!(
                        s.download(&out).unwrap(),
                        want,
                        "{format:?} {m}x{k}x{n} {limit:?}"
                    );
                    s.free(out).unwrap();
                }
                if k == 257 {
                    // Execute the omitted-prefix accumulator control: only
                    // the final K tile's logical products survive a lost reload.
                    let tail_a: Vec<_> = (0..m).map(|r| a[r * k + k - 1]).collect();
                    let tail_b: Vec<_> = b[(k - 1) * n..].to_vec();
                    let ta = s.upload(&tail_a, m, 1).unwrap();
                    let tb = s.upload(&tail_b, 1, n).unwrap();
                    let pa = s.bfp_from_f32(&ta, format).unwrap();
                    let pb = s.bfp_from_f32(&tb, format).unwrap();
                    let wrong = s
                        .bfp_matmul(&pa, false, &pb, false, tt_kernels::matmul::Fidelity::HiFi4)
                        .unwrap();
                    assert_ne!(
                        s.download(&wrong).unwrap(),
                        want,
                        "missing prior accumulator must fail"
                    );
                    s.free(wrong).unwrap();
                    s.free_bfp(pa).unwrap();
                    s.free_bfp(pb).unwrap();
                    s.free(ta).unwrap();
                    s.free(tb).unwrap();
                }
                s.free_bfp(ap).unwrap();
                s.free_bfp(bp).unwrap();
            }
            s.free(af).unwrap();
            s.free(bf).unwrap();
        }
        s.set_matmul_k_block_limit(None);
    });
}

#[test]
fn special_groups_and_physical_mutants_follow_measured_conversion() {
    with_session(|s| {
        let patterns = [
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
            0x7f800000, 0xff800000, 0x7fc12345, 0xff800001, 0x7f7fffff, 0xff7fffff, 0x3f808000,
            0x3f818000, 0x3ffc0000, 0x3ffe0000, 0x3fff0000, 0x3f810000,
        ];
        let values: Vec<_> = (0..37 * 65)
            .map(|i| f32::from_bits(patterns[(i * 7 + i / 16) % patterns.len()]))
            .collect();
        let input = s.upload(&values, 37, 65).unwrap();
        for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
            let packed = s.bfp_from_f32(&input, format).unwrap();
            let (raw, decoded) = expected(&values, 37, 65, format);
            for (got, want) in s.download_bfp_raw(&packed).unwrap().iter().zip(raw) {
                assert_eq!(&got[16..], &want[16..]);
            }
            assert_eq!(
                s.download_bfp(&packed)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                decoded
            );
            s.free_bfp(packed).unwrap();
        }
        s.free(input).unwrap();
        // Execute corrupted physical streams through the actual unpacker,
        // rather than merely asserting that two host models differ.
        let values: Vec<_> = (0..32 * 32)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let input = s.upload(&values, 32, 32).unwrap();
        for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
            let packed = s.bfp_from_f32(&input, format).unwrap();
            let mut original = s.download_bfp_raw(&packed).unwrap().remove(0);
            original.resize(format.slot_bytes() as usize, 0);
            let window = s
                .device()
                .alloc_window(tt_device::tlb::WindowKind::FourGib)
                .unwrap();
            let mut wrong = original.clone();
            wrong[16..80]
                .iter_mut()
                .for_each(|e| *e = e.wrapping_add(1));
            s.device()
                .dram_write(&window, packed.slot(0), &wrong)
                .unwrap();
            assert_ne!(
                s.download_bfp(&packed).unwrap(),
                values,
                "wrong exponent selection must fail"
            );
            let mut wrong = original.clone();
            let start = format.tile_image().datums_offset();
            match format {
                BfpFormat::Bfp8 => {
                    for p in wrong[start..start + 1024].chunks_exact_mut(2) {
                        p.swap(0, 1);
                    }
                }
                BfpFormat::Bfp4 => {
                    for byte in &mut wrong[start..start + 512] {
                        *byte = byte.rotate_left(4);
                    }
                }
                BfpFormat::Bfp2 => {
                    for byte in &mut wrong[start..start + 256] {
                        *byte = byte.rotate_left(2);
                    }
                }
            }
            s.device()
                .dram_write(&window, packed.slot(0), &wrong)
                .unwrap();
            assert_ne!(
                s.download_bfp(&packed).unwrap(),
                values,
                "wrong sub-byte order must fail"
            );
            drop(window);
            s.free_bfp(packed).unwrap();
        }
        s.free(input).unwrap();
    });
}

#[cfg(feature = "silicon")]
#[test]
#[ignore = "release benchmark"]
fn benchmark_resident_bfp_conversion_and_packed_product() {
    use std::time::Instant;
    use tt_tests::bench::{report, Conditions, Stats, REPS};
    with_session(|s| {
        let tile = s.tile();
        Conditions::measure(s.device(), tt_tests::backend::device_index(), tile).print();
        let (m, k, n) = (37, 65, 35);
        let af = s.upload(&vec![1.0; m * k], m, k).unwrap();
        let bf = s.upload(&vec![1.0; k * n], k, n).unwrap();
        for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
            let mut packs = Vec::new();
            let mut unpacks = Vec::new();
            let mut products = Vec::new();
            let before = s.dataflow_stats().clone();
            for i in 0..REPS + 2 {
                s.sync().unwrap();
                let t = Instant::now();
                let a = s.bfp_from_f32(&af, format).unwrap();
                let b = s.bfp_from_f32(&bf, format).unwrap();
                s.sync().unwrap();
                let pack = t.elapsed();
                let t = Instant::now();
                let aa = s.bfp_to_f32(&a).unwrap();
                let bb = s.bfp_to_f32(&b).unwrap();
                s.sync().unwrap();
                let unpack = t.elapsed();
                assert_eq!(s.download(&aa).unwrap(), vec![1.0; m * k]);
                assert_eq!(s.download(&bb).unwrap(), vec![1.0; k * n]);
                let t = Instant::now();
                let c = s
                    .bfp_matmul(&a, false, &b, false, tt_kernels::matmul::Fidelity::HiFi4)
                    .unwrap();
                s.sync().unwrap();
                let product = t.elapsed();
                assert_eq!(s.download(&c).unwrap(), vec![k as f32; m * n]);
                if i >= 2 {
                    packs.push(pack);
                    unpacks.push(unpack);
                    products.push(product);
                }
                s.free(aa).unwrap();
                s.free(bb).unwrap();
                s.free(c).unwrap();
                s.free_bfp(a).unwrap();
                s.free_bfp(b).unwrap();
            }
            for (name, samples) in [
                ("pack", packs),
                ("unpack", unpacks),
                ("packed_matmul", products),
            ] {
                report(
                    &format!("bfp/{format:?}/{name}/37x65x35/tiles2"),
                    "us",
                    "host",
                    Stats::of_durations(samples),
                );
            }
            println!(
                "MEASURE {format:?}: slot_bytes={} batches={} release_waits={} transfer_packets={}",
                format.slot_bytes(),
                s.dataflow_stats().batches - before.batches,
                s.dataflow_stats().release_waits - before.release_waits,
                s.dataflow_stats().transfer_packets - before.transfer_packets
            );
        }
        s.free(af).unwrap();
        s.free(bf).unwrap();
    });
}

#[test]
fn transposed_mixed_products_separate_decode_and_compression_error() {
    with_session(|s| {
        let (m, k, n) = (7, 65, 5);
        let a: Vec<_> = (0..m * k).map(|i| (i % 73) as f32 / 8.0 - 4.5).collect();
        let b: Vec<_> = (0..k * n).map(|i| (i % 59) as f32 / 8.0 - 3.5).collect();
        for transpose in [false, true] {
            let av = if transpose {
                (0..k)
                    .flat_map(|q| {
                        let a = &a;
                        (0..m).map(move |r| a[r * k + q])
                    })
                    .collect()
            } else {
                a.clone()
            };
            let bv = if transpose {
                (0..n)
                    .flat_map(|c| {
                        let b = &b;
                        (0..k).map(move |q| b[q * n + c])
                    })
                    .collect()
            } else {
                b.clone()
            };
            let (ar, ac, br, bc) = if transpose {
                (k, m, n, k)
            } else {
                (m, k, k, n)
            };
            let af = s.upload(&av, ar, ac).unwrap();
            let bf = s.upload(&bv, br, bc).unwrap();
            for (fa, fb) in [
                (BfpFormat::Bfp8, BfpFormat::Bfp4),
                (BfpFormat::Bfp4, BfpFormat::Bfp2),
                (BfpFormat::Bfp2, BfpFormat::Bfp8),
                (BfpFormat::Bfp8, BfpFormat::Bfp8),
                (BfpFormat::Bfp4, BfpFormat::Bfp4),
                (BfpFormat::Bfp2, BfpFormat::Bfp2),
            ] {
                #[cfg(not(feature = "silicon"))]
                if !transpose && fa == BfpFormat::Bfp2 && fb == BfpFormat::Bfp2 {
                    continue;
                }
                let ap = s.bfp_from_f32(&af, fa).unwrap();
                let bp = s.bfp_from_f32(&bf, fb).unwrap();
                let ad = expected(&av, ar, ac, fa).1;
                let bd = expected(&bv, br, bc, fb).1;
                let ad: Vec<_> = ad.into_iter().map(f32::from_bits).collect();
                let bd: Vec<_> = bd.into_iter().map(f32::from_bits).collect();
                let mut want = Vec::new();
                let mut compression = Vec::new();
                for r in 0..m {
                    for c in 0..n {
                        let mut value = 0.0;
                        let mut baseline = 0.0f64;
                        let mut bound = 0.0f64;
                        for q in 0..k {
                            let (x, y) = if transpose {
                                (ad[q * m + r], bd[c * k + q])
                            } else {
                                (ad[r * k + q], bd[q * n + c])
                            };
                            // This corpus lives on a dyadic grid whose integer sum
                            // fits 24 significand bits: every product/add is exact.
                            value += x * y;
                            let (a, b) = (a[r * k + q] as f64, b[q * n + c] as f64);
                            baseline += a * b;
                            let (dx, dy) = ((x as f64 - a).abs(), (y as f64 - b).abs());
                            // |(a+da)(b+db)-ab| <= |a|db+|b|da+da*db.
                            bound += a.abs() * dy + b.abs() * dx + dx * dy;
                        }
                        want.push(value);
                        compression.push((baseline, bound));
                    }
                }
                s.set_matmul_k_block_limit(std::num::NonZeroUsize::new(1));
                let out = s
                    .bfp_matmul(
                        &ap,
                        transpose,
                        &bp,
                        transpose,
                        tt_kernels::matmul::Fidelity::HiFi4,
                    )
                    .unwrap();
                let got = s.download(&out).unwrap();
                assert_eq!(got, want);
                for (got, (base, bound)) in got.into_iter().zip(compression) {
                    assert!((got as f64 - base).abs() <= bound);
                }
                s.free(out).unwrap();
                s.free_bfp(ap).unwrap();
                s.free_bfp(bp).unwrap();
            }
            s.free(af).unwrap();
            s.free(bf).unwrap();
        }
        s.set_matmul_k_block_limit(None);
    });
}

#[test]
fn dirty_view_padding_is_repaired_without_changing_parent_claims() {
    with_session(|s| {
        let base = s.upload(&vec![1.0; 64 * 35], 64, 35).unwrap();
        let parent = s
            .eltwise(
                tt_kernels::tensor::Eltwise {
                    kind: tt_kernels::kind::ADD_SCALAR,
                    scalar: 3.0,
                    scalar2: 0.0,
                },
                &base,
                None,
            )
            .unwrap();
        assert_eq!(parent.pad(), tt_kernels::tensor::Pad::Undefined);
        let view = parent.rows_view(32, 32).unwrap();
        for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
            // Lie about dirty padding to execute the missing-mask control.
            // The production cast sees the false claim, so it cannot repair it.
            view.set_pad(tt_kernels::tensor::Pad::Zero);
            let wrong = s.bfp_from_f32(&view, format).unwrap();
            let want = expected(&vec![4.0; 32 * 35], 32, 35, format).0;
            let got = s.download_bfp_raw(&wrong).unwrap();
            assert!(
                got.iter().zip(&want).any(|(g, w)| g[16..] != w[16..]),
                "missing edge repair must fail the physical oracle"
            );
            s.free_bfp(wrong).unwrap();
            view.set_pad(tt_kernels::tensor::Pad::Undefined);
            let packed = s.bfp_from_f32(&view, format).unwrap();
            let (raw, decoded) = expected(&vec![4.0; 32 * 35], 32, 35, format);
            for (got, want) in s.download_bfp_raw(&packed).unwrap().iter().zip(raw) {
                assert_eq!(&got[16..], &want[16..]);
            }
            assert_eq!(
                s.download_bfp(&packed)
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                decoded
            );
            assert_eq!(
                parent.pad(),
                tt_kernels::tensor::Pad::Undefined,
                "repair must not certify parent padding"
            );
            s.free_bfp(packed).unwrap();
        }
        s.free(view).unwrap();
        s.free(parent).unwrap();
        s.free(base).unwrap();
    });
}

#[test]
fn conversion_sweeps_significand_ties_clamping_and_exponent_boundaries() {
    with_session(|s| {
        for exponent in [1u32, 6, 126, 127, 128, 254, 255] {
            let values: Vec<_> = (0..4096u32)
                .map(|i| {
                    let low = [0, 0x7fff, 0x8000, 0xffff][(i / 256 % 4) as usize];
                    let shift = (i / 1024).min(exponent - 1);
                    let exp = exponent - shift;
                    f32::from_bits(((i % 2) << 31) | (exp << 23) | ((i / 2 % 128) << 16) | low)
                })
                .collect();
            let input = s.upload(&values, 32, 128).unwrap();
            for format in [BfpFormat::Bfp8, BfpFormat::Bfp4, BfpFormat::Bfp2] {
                let packed = s.bfp_from_f32(&input, format).unwrap();
                let (raw, decoded) = expected(&values, 32, 128, format);
                for (got, want) in s.download_bfp_raw(&packed).unwrap().iter().zip(raw) {
                    assert_eq!(&got[16..], &want[16..], "{format:?} exp{exponent}");
                }
                assert_eq!(
                    s.download_bfp(&packed)
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    decoded
                );
                s.free_bfp(packed).unwrap();
            }
            s.free(input).unwrap();
        }
    });
}
