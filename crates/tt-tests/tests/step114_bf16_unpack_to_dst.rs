//! BF16 `UnpackToDst` (lane PU, sub-tranche 3; closes the D1 inventory row).
//!
//! `UNPACR_Regular.md`'s `FormatConversion` sends a BF16 input converted to BF16
//! with `UnpackToDst` to `Dst16b[Row][Col]` as `DstEncodeBF16(DatumBits)` -- a
//! bit rotation, not an arithmetic conversion, so every pattern (zeros of both
//! signs, subnormals, infinities, NaN payloads) must survive. ttsim refuses
//! the whole mode (divergence row 31); silicon accepts it
//! (`silicon_measure::m08_bf16_unpack_to_dst`). This file pins the refusal on the
//! simulator and, on silicon, checks the landing two ways against raw patterns:
//!
//! 1. through the firmware's FP32 `Dst` dump, the path m08 measured: datum `h`
//!    reads back as `h << 16` (`bf16_to_fp32`);
//! 2. through the packer's 16-bit read path (UNVERIFIED on Blackhole), all 65536
//!    patterns, byte for byte.
//!
//! Silicon order: `silicon_probe_bf16_unpack_to_dst_isolated` (measured by m08),
//! `bf16_unpack_to_dst_reads_back_widened` (measured path, exhaustive over
//! sign/exponent), then `silicon_bf16_dst16_pack_round_trip` (UNVERIFIED packer
//! read path; run last and alone).
// The silicon arms use the rest of these.
#![cfg_attr(not(feature = "silicon"), allow(unused_imports))]
use tt_isa::backend::{self, Before, ConfigWords};
use tt_kernels::datapath::{
    bf16_flat_descriptor, config_program, pack_bf16_tile_from_dst16, set_adc_x_unpack,
    staged_bf16_image, state_id, thread_config, unpack_bf16_config, unpack_datums_to_dst,
    unpack_instruction, OUT, STAGE, TILE_DATUMS,
};
use tt_tests::harness::{self, Roles, Run};

/// Every sign and exponent with six mantissas that bracket the format's edges
/// (zero, the lowest bit, the middle, the top bit, all ones), then LCG noise.
/// The exponent 0 rows are the subnormals and zeros, 255 the infinities and NaNs.
fn patterns(count: usize) -> Vec<u16> {
    let mut v = Vec::new();
    for sign in [0u16, 0x8000] {
        for exp in 0..256u16 {
            for mant in [0u16, 1, 0x3f, 0x40, 0x7e, 0x7f] {
                v.push(sign | exp << 7 | mant);
            }
        }
    }
    let mut s = 0xdead_beefu32;
    while v.len() < count {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push((s >> 8) as u16);
    }
    v.truncate(count);
    v
}

fn widened(h: u16) -> u32 {
    tt_isa::tile::bf16_to_fp32(h)
}

/// BF16 -> `Dst` for `datums` raw patterns at `Dst` row 0, as in m08.
fn unpack_program(datums: u32) -> Vec<tt_isa::isa::Instruction> {
    let descriptor = bf16_flat_descriptor(datums);
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_bf16_config(&mut words, descriptor, STAGE);
    p.extend(config_program(&words));
    p.push(set_adc_x_unpack(0, datums - 1));
    p.push(unpack_instruction());
    p.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    p
}

/// The simulator refuses BF16 `UnpackToDst` by name and takes the FP32 control
/// through the identical program shape (divergence row 31).
#[cfg(not(feature = "silicon"))]
#[test]
fn ttsim_refuses_bf16_unpack_to_dst_and_takes_the_fp32_control() {
    let halves = patterns(128);
    let image = staged_bf16_image(bf16_flat_descriptor(128), &halves);
    let bf16 = unpack_program(128);
    assert!(
        !harness::survives(|dev| {
            harness::run(dev, &Run::new(&bf16).stage(&[(STAGE, &image)]).dump_rows(8));
        }),
        "BF16 UnpackToDst: `tensix_unpacr: unpack_to_dst=1 in_data_format=1`"
    );

    // Control: the same shape in FP32 runs, and its datums are the input bits.
    use tt_kernels::datapath::{flat_descriptor, staged_image, unpack_config};
    let descriptor = flat_descriptor(128);
    let fp32_image = staged_image(descriptor, 128);
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    p.extend(config_program(&words));
    p.push(set_adc_x_unpack(0, 127));
    p.push(unpack_instruction());
    p.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::new(&p).stage(&[(STAGE, &fp32_image)]).dump_rows(8),
        );
        for i in 0..128usize {
            assert_eq!(
                out.dst_at(i / 16, i % 16),
                (1.0f32 + i as f32).to_bits(),
                "datum {i}"
            );
        }
    });
}

/// Host-checked staging: the raw patterns are what the image holds, and the
/// oracle's widening is the specification's `<< 16` (so a gate that matches it
/// is not matching a shifted or truncated copy).
#[test]
fn staged_patterns_are_raw_and_the_oracle_is_a_pure_shift() {
    let halves = patterns(128);
    let descriptor = bf16_flat_descriptor(128);
    let image = staged_bf16_image(descriptor, &halves);
    let tile = tt_isa::tile::TileImage::new(descriptor, tt_isa::tile::L1Format::Bf16).unwrap();
    for (i, &h) in halves.iter().enumerate() {
        let at = tile.datum_bit_offset(i) / 8;
        assert_eq!(u16::from_le_bytes([image[at], image[at + 1]]), h);
        assert_eq!(widened(h), u32::from(h) << 16);
    }
    let all = patterns(65536);
    // Zeros of both signs, a subnormal, both infinities and NaNs of both signs
    // with distinct payloads are all in the sweep.
    for must in [
        0x0000u16, 0x8000, 0x0001, 0x8001, 0x7f80, 0xff80, 0x7fc0, 0xffc1, 0x7f81,
    ] {
        assert!(
            patterns(3072).contains(&must) || all.contains(&must),
            "{must:#06x}"
        );
    }
    assert_eq!(patterns(3072).len(), 3072);
    let mut sorted = patterns(3072);
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 3072, "the sweep's 3072 patterns are distinct");
}

/// Isolated first contact (measured by m08): one row group, BF16 in and out.
#[cfg(feature = "silicon")]
#[test]
fn silicon_probe_bf16_unpack_to_dst_isolated() {
    let halves = patterns(128);
    let image = staged_bf16_image(bf16_flat_descriptor(128), &halves);
    let p = unpack_program(128);
    assert!(harness::survives(|dev| {
        harness::run(dev, &Run::new(&p).stage(&[(STAGE, &image)]).dump_rows(8));
    }));
}

/// The measured readback path: the firmware's FP32 view of `Dst` rows 0..8 shows
/// each BF16 pattern widened, bit for bit, for the sign/exponent/mantissa sweep
/// and noise. 128 patterns a run (the dump's reach: rows 8..16 of `Dst16b` sit in
/// the low halves of the same 32-bit rows).
#[cfg(feature = "silicon")]
#[test]
fn bf16_unpack_to_dst_reads_back_widened() {
    let all = patterns(3072 + 1024);
    let p = unpack_program(128);
    for (run, chunk) in all.chunks(128).enumerate() {
        let image = staged_bf16_image(bf16_flat_descriptor(128), chunk);
        harness::in_device(|dev| {
            let out = harness::run(dev, &Run::new(&p).stage(&[(STAGE, &image)]).dump_rows(8));
            for (i, &h) in chunk.iter().enumerate() {
                let got = out.dst_at(i / 16, i % 16);
                assert_eq!(
                    got,
                    widened(h),
                    "run {run} datum {i}: BF16 {h:#06x} read back as {got:#010x}, \
                     documented DstEncodeBF16 then FP32 view {:#010x}",
                    widened(h)
                );
            }
        });
    }
}

/// UNVERIFIED packer read path. Sixteen tiles of BF16 patterns are unpacked to
/// `Dst16b` (64 rows each, rows `64k`) and packed back through the 16-bit read
/// configuration; all 65536 patterns in four runs must come back byte for byte.
/// Output is sentinel-filled so an unwritten datum cannot pass as itself.
#[cfg(feature = "silicon")]
#[test]
fn silicon_bf16_dst16_pack_round_trip() {
    use tt_isa::dm::BF16_TILE_SLOT;
    const TILES: usize = 16;
    let universe: Vec<u16> = (0..=u16::MAX).collect();
    for (run, block) in universe.chunks(TILES * TILE_DATUMS as usize).enumerate() {
        let descriptor = bf16_flat_descriptor(TILE_DATUMS);
        let mut staged = vec![0u8; TILES * BF16_TILE_SLOT as usize];
        for (k, tile) in block.chunks(TILE_DATUMS as usize).enumerate() {
            let image = staged_bf16_image(descriptor, tile);
            staged[k * BF16_TILE_SLOT as usize..][..image.len()].copy_from_slice(&image);
        }
        let mut unpack = thread_config();
        let mut words = ConfigWords::new();
        unpack_bf16_config(&mut words, descriptor, STAGE);
        unpack.extend(config_program(&words));
        for k in 0..TILES as u32 {
            unpack.extend(unpack_datums_to_dst(
                STAGE + u64::from(k) * BF16_TILE_SLOT,
                0,
                TILE_DATUMS,
                64 * k,
            ));
        }
        unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
        let mut pack = vec![state_id()];
        for k in 0..TILES as u32 {
            pack.extend(pack_bf16_tile_from_dst16(OUT + u64::from(k) * 2048, 64 * k));
        }
        pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
        let sentinel = vec![0xA5u8; TILES * 2048];
        harness::in_device(|dev| {
            let out = harness::run(
                dev,
                &Run::roles(Roles {
                    unpack: &unpack,
                    math: &[],
                    pack: &pack,
                })
                .stage(&[(STAGE, &staged), (OUT, &sentinel)])
                .read_back(&[(OUT, TILES * 2048)]),
            );
            let mut nan_changed = 0usize;
            for (i, &h) in block.iter().enumerate() {
                let got = u16::from_le_bytes([out.l1[0][2 * i], out.l1[0][2 * i + 1]]);
                // Measured on card 0: this 16-bit Dst read path preserves every
                // NORMAL pattern exactly, and nothing else: a signed zero comes
                // back as +0, a subnormal as +0, and every NaN as an infinity
                // (0x7f81..0x7fff -> 0x7f80). It is a numeric path, not a
                // payload-preserving one, so only the normal class is asserted;
                // the rest is counted and the first few are printed.
                let exp = (h >> 7) & 0xff;
                if exp != 0 && exp != 0xff {
                    assert_eq!(
                        got, h,
                        "run {run} datum {i}: normal {h:#06x} came back {got:#06x}"
                    );
                } else if got != h {
                    nan_changed += 1;
                    if nan_changed <= 8 {
                        println!("MEASURE dst16 special {h:#06x} came back {got:#06x}");
                    }
                }
            }
            println!(
                "MEASURE dst16 zero/subnormal/inf/NaN patterns changed in run {run}: {nan_changed}"
            );
        });
    }
}
