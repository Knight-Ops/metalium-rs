//! Phase 10 gate (D1): FP16 (IEEE binary16) tensors in GDDR.
//!
//! The stored format is IEEE binary16 in BF16's 2112-byte physical slot. The
//! coprocessor's own FP16 differs from IEEE (no NaN, `Exp == 31` finite,
//! denormals read as zero), so the shipped casts do the IEEE work in the SFPU and
//! never depend on the packer or unpacker FP16 conversion:
//!
//! - **raw storage**: every one of the 65536 bit patterns round-trips through
//!   upload, readback, row views, repacks, zero fills and the Tensix mover, with
//!   two-byte slots (`raw_fp16_*`);
//! - **widening** (`fp16_to_f32`): B expands each halfword into a 32-bit datum and
//!   an SFPU integer program builds the FP32; every pattern, subnormals, infinities
//!   and NaN payloads included, against an independent oracle
//!   (`widening_matches_the_ieee_definition_for_every_pattern`);
//! - **narrowing** (`fp16_from_f32`): an SFPU integer program computes the exact
//!   ties-even binary16 bits, the packer moves them as raw words and B compacts the
//!   slot; ties, carries, the 65520 overflow tie, subnormal boundaries and NaN
//!   payloads against the oracle (`narrowing_matches_the_ieee_definition`);
//! - a ragged producer followed by a reduction (`widened_ragged_tensor_*`).
//!
//! The oracle here is written from the IEEE definition with `f64` arithmetic and
//! `round_ties_even`, independent of both the SFPU programs and
//! `tt_kernels::fp16::host`.
//!
//! The coprocessor's own conversions are measurement variants. ttsim models them
//! (truncating pack, saturation at the exponent-31 row, flushed denormals, the
//! unpacker moving the 5-bit exponent unconverted: divergence row 101) and
//! `ttsim_models_the_native_conversions` pins that; **silicon is UNMEASURED**:
//! `probe_native_fp16_specials` records it and `native_conversions_match_the_
//! measured_model` compares against `measured::SILICON`, pending the probe.
//!
//! Negative controls: a truncating oracle must differ on the corpus's ties, the
//! BF16 widening of the same slots must differ from the FP16 reading, a wrong
//! repack mapping must differ, and (model level, `tt-kernels` unit tests) a
//! truncating SFPU program fails the tie cases.

use tt_kernels::fp16::{Narrowing, Widening};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

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
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
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
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

// ---------------------------------------------------------------- oracles

#[path = "fp16_oracle/mod.rs"]
mod oracle;
use oracle::{f32_corpus, oracle_to_f16, oracle_to_f32, truncating_to_f16};

// ---------------------------------------------------------------- model gates (host)

#[test]
fn oracles_agree_with_each_other_and_the_library_reference() {
    use tt_kernels::fp16::host;
    for h in 0..=u16::MAX {
        assert_eq!(oracle_to_f32(h), host::f16_to_f32(h), "{h:#06x}");
    }
    for x in f32_corpus(8192) {
        assert_eq!(oracle_to_f16(x), host::f32_to_f16(x), "{x:#010x}");
    }
}

/// Negative control: the corpus is sensitive to ties-even versus truncation.
#[test]
fn the_corpus_separates_ties_even_from_truncation() {
    let corpus = f32_corpus(4087);
    let differing = corpus
        .iter()
        .filter(|&&x| truncating_to_f16(x) != oracle_to_f16(x))
        .count();
    assert!(
        differing > 200,
        "{differing} elements distinguish truncation"
    );
    // 1 + 3 * 2^-11 is a tie whose even neighbour is up.
    assert_eq!(oracle_to_f16(0x3f80_3000), 0x3c02);
    assert_eq!(truncating_to_f16(0x3f80_3000), 0x3c01);
}

// ---------------------------------------------------------------- raw storage

#[test]
fn raw_fp16_storage_preserves_every_pattern_through_views_and_movers() {
    with_session(|s| {
        // 256 x 256 holds every pattern once; ragged shapes hold prefixes.
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let (rows, cols) = (37, 70);
        let values: Vec<u16> = (0..rows * cols)
            .map(|i| all[(i * 131 + 7) % 65536])
            .collect();
        let t = s.upload_fp16(&values, rows, cols).unwrap();
        assert_eq!(t.slot(0).len(), 2112, "two-byte slots, not F32's");
        assert_eq!(s.download_fp16(&t).unwrap(), values);

        let full = s.upload_fp16(&all, 256, 256).unwrap();
        assert_eq!(s.download_fp16(&full).unwrap(), all, "all 65536 patterns");
        s.free_fp16(full).unwrap();

        // A row view of whole tile rows, and its parent unchanged.
        let tail = t.rows_view(32, 5).unwrap();
        assert_eq!(s.download_fp16(&tail).unwrap(), values[32 * cols..]);
        assert!(t.rows_view(1, 5).is_err(), "views are whole tile rows");

        // A transposing repack; the control is that a wrong mapping differs.
        let transpose: Vec<[usize; 2]> = (0..cols)
            .flat_map(|c| (0..rows).map(move |r| [r, c]))
            .collect();
        let want: Vec<u16> = transpose
            .iter()
            .map(|&[r, c]| values[r * cols + c])
            .collect();
        let moved = s.repack_fp16(&t, &transpose, [cols, rows]).unwrap();
        assert_eq!(s.download_fp16(&moved).unwrap(), want);
        let wrong: Vec<[usize; 2]> = (0..cols)
            .flat_map(|c| (0..rows).map(move |r| [r, (c + 1) % cols]))
            .collect();
        let miss = s.repack_fp16(&t, &wrong, [cols, rows]).unwrap();
        assert_ne!(
            s.download_fp16(&miss).unwrap(),
            want,
            "control: wrong mapping"
        );

        let zeros = s.zeros_fp16([rows, cols]).unwrap();
        assert!(s.download_fp16(&zeros).unwrap().iter().all(|&b| b == 0));
        for x in [t, moved, miss, zeros] {
            s.free_fp16(x).unwrap();
        }
    });
}

/// Silicon-only: ttsim refuses the XMOV mover setup (as for BF16, `step102`).
#[cfg(feature = "silicon")]
#[test]
fn raw_fp16_xmov_copy_preserves_every_pattern() {
    with_session(|s| {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let (rows, cols) = (37, 70);
        let values: Vec<u16> = (0..rows * cols)
            .map(|i| all[(i * 131 + 7) % 65536])
            .collect();
        let t = s.upload_fp16(&values, rows, cols).unwrap();
        let copy = s.copy_fp16_xmov(&t).unwrap();
        assert_eq!(s.download_fp16(&copy).unwrap(), values);
        s.free_fp16(copy).unwrap();
        s.free_fp16(t).unwrap();
    });
}

// ---------------------------------------------------------------- widening

#[test]
fn widening_matches_the_ieee_definition_for_every_pattern() {
    with_session(|s| {
        let all: Vec<u16> = (0..=u16::MAX).collect();
        let t = s.upload_fp16(&all, 256, 256).unwrap();
        let wide = s.fp16_to_f32(&t).unwrap();
        let got = s.download(&wide).unwrap();
        for (h, g) in all.iter().zip(&got) {
            assert_eq!(g.to_bits(), oracle_to_f32(*h), "pattern {h:#06x}");
        }
        // Control: the BF16 reading of the same slots is not the FP16 one.
        let as_bf16 = s.bf16_to_f32(&t.as_raw()).unwrap();
        let bf = s.download(&as_bf16).unwrap();
        let differing = all
            .iter()
            .zip(&bf)
            .filter(|(h, g)| g.to_bits() != oracle_to_f32(**h))
            .count();
        assert!(differing > 60000, "{differing}");
        s.free(as_bf16).unwrap();
        s.free(wide).unwrap();
        s.free_fp16(t).unwrap();
    });
}

/// A ragged producer followed by a reduction: `fp16_to_f32` declares undefined
/// padding, and a maximum over either axis must not see it.
#[test]
fn widened_ragged_tensor_feeds_reductions_without_padding() {
    with_session(|s| {
        let (rows, cols) = (37, 70);
        // Finite values only (no exponent 31): a maximum of NaNs is not the test.
        let values: Vec<u16> = (0..rows * cols)
            .map(|i| {
                let h = ((i * 40503 + 11) % 65536) as u16;
                if (h >> 10) & 0x1f == 31 {
                    h & 0xbfff
                } else {
                    h
                }
            })
            .collect();
        let t = s.upload_fp16(&values, rows, cols).unwrap();
        let wide = s.fp16_to_f32(&t).unwrap();
        for (axis, n) in [(Axis::Cols, rows), (Axis::Rows, cols)] {
            let out = s.reduce(&wide, ReduceOp::Max, axis).unwrap();
            let got = s.download(&out).unwrap();
            assert_eq!(got.len(), n);
            for (i, g) in got.iter().enumerate() {
                let want = (0..if axis == Axis::Cols { cols } else { rows })
                    .map(|j| {
                        let (r, c) = if axis == Axis::Cols { (i, j) } else { (j, i) };
                        f32::from_bits(oracle_to_f32(values[r * cols + c]))
                    })
                    .fold(f32::NEG_INFINITY, f32::max);
                assert_eq!(*g, want, "{axis:?} {i}");
            }
            s.free(out).unwrap();
        }
        s.free(wide).unwrap();
        s.free_fp16(t).unwrap();
    });
}

// ---------------------------------------------------------------- narrowing

#[test]
fn narrowing_matches_the_ieee_definition() {
    with_session(|s| {
        let (rows, cols) = (61, 67);
        let bits = f32_corpus(rows * cols);
        let values: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let t = s.upload(&values, rows, cols).unwrap();
        let narrow = s.fp16_from_f32(&t).unwrap();
        assert_eq!(narrow.slot(0).len(), 2112);
        let got = s.download_fp16(&narrow).unwrap();
        for (i, (g, &x)) in got.iter().zip(&bits).enumerate() {
            assert_eq!(*g, oracle_to_f16(x), "{i}: input {x:#010x}");
        }
        // The control the corpus separates: truncation would not pass.
        assert!(bits
            .iter()
            .zip(&got)
            .any(|(&x, &g)| truncating_to_f16(x) != g));
        // And back: widening the narrowed values is the oracle's round trip.
        let wide = s.fp16_to_f32(&narrow).unwrap();
        let back = s.download(&wide).unwrap();
        for (i, (g, &x)) in back.iter().zip(&bits).enumerate() {
            assert_eq!(
                g.to_bits(),
                oracle_to_f32(oracle_to_f16(x)),
                "{i}: {x:#010x}"
            );
        }
        s.free(wide).unwrap();
        s.free_fp16(narrow).unwrap();
        s.free(t).unwrap();
    });
}

// ---------------------------------------------------------------- native conversions

/// The coprocessor's FP32-to-FP16 packer conversion, as a model with each
/// unmeasured behaviour a named field.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct PackerModel {
    /// Mantissa narrowing of a value in range.
    mantissa: Mantissa,
    /// `|x| >= 2^17`, past the exponent-31 row.
    beyond: u16,
    /// NaN input.
    nan: u16,
    /// `2^-15 < |x| < 2^-14`, the band the documentation calls mishandled:
    /// `Some(())` keeps the FP32 mantissa's top ten bits under exponent zero.
    mishandled_band: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Mantissa {
    Truncate,
    NearestEven,
}

impl PackerModel {
    /// What pinned ttsim does (`ttsim_models_the_native_conversions`).
    const TTSIM: Self = Self {
        mantissa: Mantissa::Truncate,
        beyond: 0x7fff,
        nan: 0x7fff,
        mishandled_band: true,
    };

    fn pack(&self, bits: u32) -> u16 {
        let sign = ((bits >> 16) & 0x8000) as u16;
        let a = bits & 0x7fff_ffff;
        if a > 0x7f80_0000 {
            return sign | self.nan;
        }
        if a >= 0x4800_0000 {
            return sign | self.beyond;
        }
        if a < 0x3880_0000 {
            // Denormals, and everything up to 2^-15, flush to signed zero.
            return if self.mishandled_band && a > 0x3800_0000 {
                sign | ((a >> 13) & 0x3ff) as u16
            } else {
                sign
            };
        }
        let kept = match self.mantissa {
            Mantissa::Truncate => a >> 13,
            Mantissa::NearestEven => (a + 0xfff + ((a >> 13) & 1)) >> 13,
        };
        sign | (kept - (112 << 10)) as u16
    }
}

/// PENDING MEASUREMENT (T7): the silicon packer's FP32-to-FP16 behaviour. Fill
/// in from `probe_native_fp16_specials` (`target/silicon/` log, lines starting
/// `T7-MEASURE`), then `native_conversions_match_the_measured_model` becomes a
/// gate. Until then it is `None` and that gate refuses to pass.
mod measured {
    use super::PackerModel;
    #[allow(dead_code)]
    pub const SILICON: Option<PackerModel> = None;
}

/// Special values and ties for the native conversions: ties, overflow just above
/// 65504 and 65520, 2^16, 2^17, subnormal boundaries, NaN payloads, signed zeros
/// and infinities.
fn probe_values() -> Vec<u32> {
    vec![
        0x3f80_0000,
        0xbf80_0000,
        0x0000_0000,
        0x8000_0000,
        0x3f80_1000,
        0x3f80_3000,
        0x3f80_0fff,
        0x3f80_1001,
        0xbf80_3000,
        0x477f_e000,
        0x477f_efff,
        0x477f_f000,
        0x477f_f001,
        0x4780_0000,
        0x4788_b800,
        0x47ff_e000,
        0x4800_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7fc1_2345,
        0xffc1_2345,
        0x7f80_0001,
        0x3880_0000,
        0x387f_e000,
        0x387f_dfff,
        0x3800_0001,
        0x3800_0000,
        0x3380_0000,
        0x3300_0000,
        0x33c0_0000,
        0x0000_0001,
        0x0080_0000,
        0x8000_0001,
        0xb880_0000,
        0xb87f_e000,
    ]
}

fn native_narrow<T: tt_device::Transport>(
    s: &mut Session<T>,
    bits: &[u32],
    narrowing: Narrowing,
) -> Vec<u16> {
    let f: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
    let t = s.upload(&f, 1, f.len()).unwrap();
    let h = s.fp16_from_f32_with(&t, narrowing).unwrap();
    let out = s.download_fp16(&h).unwrap();
    s.free_fp16(h).unwrap();
    s.free(t).unwrap();
    out
}

fn native_widen<T: tt_device::Transport>(s: &mut Session<T>, bits: &[u16]) -> Vec<u32> {
    let t = s.upload_fp16(bits, 1, bits.len()).unwrap();
    let w = s.fp16_to_f32_with(&t, Widening::SrcRaw).unwrap();
    let out = s
        .download(&w)
        .unwrap()
        .iter()
        .map(|f| f.to_bits())
        .collect();
    s.free(w).unwrap();
    s.free_fp16(t).unwrap();
    out
}

/// A recorded fact about the simulator, not about silicon (row 101).
#[cfg(not(feature = "silicon"))]
#[test]
fn ttsim_models_the_native_conversions() {
    with_session(|s| {
        let bits = probe_values();
        let raw = native_narrow(s, &bits, Narrowing::RawPacker);
        for (&x, &g) in bits.iter().zip(&raw) {
            assert_eq!(g, PackerModel::TTSIM.pack(x), "raw packer {x:#010x}");
        }
        // The control: ties-even is not what ttsim's packer does.
        let even = PackerModel {
            mantissa: Mantissa::NearestEven,
            ..PackerModel::TTSIM
        };
        assert!(bits.iter().zip(&raw).any(|(&x, &g)| even.pack(x) != g));
        // SFPU rounding first leaves the packer only exact values: IEEE except
        // that subnormal results are flushed.
        let rounded = native_narrow(s, &bits, Narrowing::Rounded);
        for (&x, &g) in bits.iter().zip(&rounded) {
            let ieee = oracle_to_f16(x);
            let expected = if (ieee >> 10) & 0x1f == 0 && ieee & 0x3ff != 0 {
                ieee & 0x8000
            } else if ieee & 0x7fff > 0x7c00 {
                ieee & 0xfe00 // NaN payloads are canonical here
            } else {
                ieee
            };
            assert_eq!(g, expected, "rounded {x:#010x}");
        }
        // ttsim's unpacker moves the five-bit exponent into the eight-bit field
        // unconverted, so the unassisted SrcA route is not a widening on it.
        let patterns = [0x3c00u16, 0xbc00, 0x7bff, 0x7c00, 0x0400, 0x0001, 0x8000];
        for (&h, g) in patterns.iter().zip(native_widen(s, &patterns)) {
            let sign = (h as u32 & 0x8000) << 16;
            let unconverted = sign | (((h as u32 >> 10) & 0x1f) << 23) | ((h as u32 & 0x3ff) << 13);
            assert_eq!(g, unconverted, "SrcRaw {h:#06x}");
            if h & 0x7fff != 0 {
                assert_ne!(g, oracle_to_f32(h), "SrcRaw is not IEEE on ttsim: {h:#06x}");
            }
        }
    });
}

/// Silicon-only measurement. Records, never judges: the packer's FP16 results
/// for the special-value corpus (unassisted, and after SFPU rounding), and the
/// unpacker's reading of raw patterns. Run this FIRST and ALONE
/// (`--filter probe_native_fp16_specials`); read the `T7-MEASURE` lines.
#[cfg(feature = "silicon")]
#[test]
fn probe_native_fp16_specials() {
    with_session(|s| {
        let bits = probe_values();
        for narrowing in [Narrowing::RawPacker, Narrowing::Rounded] {
            let got = native_narrow(s, &bits, narrowing);
            for (&x, &g) in bits.iter().zip(&got) {
                println!(
                    "T7-MEASURE pack {narrowing:?} {x:#010x} -> {g:#06x} ttsim {:#06x} ieee {:#06x}",
                    PackerModel::TTSIM.pack(x),
                    oracle_to_f16(x)
                );
            }
        }
        let patterns: Vec<u16> = vec![
            0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0x7fff, 0xffff, 0x0400, 0x03ff,
            0x0200, 0x0001, 0x8001, 0x8000, 0x0000, 0x3555, 0x5640,
        ];
        for (&h, g) in patterns.iter().zip(native_widen(s, &patterns)) {
            println!(
                "T7-MEASURE unpack SrcRaw {h:#06x} -> {g:#010x} ieee {:#010x}",
                oracle_to_f32(h)
            );
        }
    });
}

/// Silicon: the unassisted packer against the measured model. Refuses to pass
/// until `measured::SILICON` is filled from the probe.
#[cfg(feature = "silicon")]
#[test]
fn native_conversions_match_the_measured_model() {
    let Some(model) = measured::SILICON else {
        panic!("PENDING MEASUREMENT: fill measured::SILICON from probe_native_fp16_specials");
    };
    with_session(|s| {
        let bits = probe_values();
        let raw = native_narrow(s, &bits, Narrowing::RawPacker);
        for (&x, &g) in bits.iter().zip(&raw) {
            assert_eq!(g, model.pack(x), "raw packer {x:#010x}");
        }
        // The control the model must reject: truncation if the packer rounds.
        let flipped = PackerModel {
            mantissa: match model.mantissa {
                Mantissa::Truncate => Mantissa::NearestEven,
                Mantissa::NearestEven => Mantissa::Truncate,
            },
            ..model
        };
        assert!(bits.iter().zip(&raw).any(|(&x, &g)| flipped.pack(x) != g));
    });
}

/// Silicon: physical slots and padding, as `step74` inspects BF16.
#[cfg(feature = "silicon")]
#[test]
fn physical_fp16_slots_hold_two_byte_datums_and_zero_padding() {
    use tt_device::tlb::WindowKind;
    with_session(|s| {
        let (rows, cols) = (37, 70);
        let bits = f32_corpus(rows * cols);
        let values: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        let input = s.upload(&values, rows, cols).unwrap();
        let packed = s.fp16_from_f32(&input).unwrap();
        s.sync().unwrap();
        assert!(packed.slot(0).len() < input.slot(0).len());
        let window = s.device().alloc_window(WindowKind::FourGib).unwrap();
        let ct = cols.div_ceil(32);
        for t in 0..packed.tile_count() {
            let range = packed.slot(t);
            let range = range.channel().range(range.offset() + 16, 2048).unwrap();
            let mut bytes = [0u8; 2048];
            s.device().dram_read(&window, range, &mut bytes).unwrap();
            for i in 0..1024 {
                let face = i / 256;
                let r = t / ct * 32 + face / 2 * 16 + i % 256 / 16;
                let c = t % ct * 32 + face % 2 * 16 + i % 16;
                let expected = if r < rows && c < cols {
                    oracle_to_f16(bits[r * cols + c])
                } else {
                    0
                };
                let got = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                assert_eq!(got, expected, "physical tile {t}, datum {i}, ({r},{c})");
            }
        }
        drop(window);
        s.free_fp16(packed).unwrap();
        s.free(input).unwrap();
    });
}
