//! Unpacker input modes (PU sub-tranche 4; decides M3 and D5).
//!
//! Two modes are documented for Blackhole (`UNPACR_Regular.md`, conditionalized):
//!
//! * **Tileize** (`Tileize_mode`, the page's `DiscontiguousInputRows`): the
//!   unpacker reads 32 datums (Blackhole's `UnpackRowWidth` for datums wider
//!   than a byte) from each input row and steps `RowStride` bytes (three
//!   nibbles of `Shift_amount_cntx0..2`, 16-byte units) between rows. Into
//!   `Dst` FP32 this is a payload-preserving strided row gather (D5).
//! * **Transpose** (`Haloize_mode`): unpacker 0 into `SrcA` writes each 16x16
//!   block of rows with the row's low four bits and the column swapped (M3).
//!   `UnpackToDst` with it is `UndefinedBehavior`, and `SrcA` holds a 19-bit
//!   datum, so it cannot be a payload-preserving FP32 transpose.
//!
//! There is no unpacker broadcast; `Upsample*` and `ColShift` are
//! `UnsupportedFunctionality` and not exposed. ttsim models both modes, so the
//! gates below run there against raw-integer oracles; the silicon arms
//! (UNVERIFIED, written not run) measure what the Src path does to payloads.
//!
//! Silicon order, one mode value per probe, each isolated first:
//! 1. `silicon_probe_tileize_isolated` (UNVERIFIED), then
//!    `tileize_strips_match_the_row_stride_model`;
//! 2. `silicon_probe_transpose_isolated` (UNVERIFIED), then
//!    `unpacker_transpose_swaps_row_and_column_low_nibbles`;
//! 3. `silicon_transpose_is_a_pure_permutation_of_the_plain_src_path`, last.
#![cfg_attr(not(feature = "silicon"), allow(unused_imports, dead_code))]
use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::cfg::generated::thcon;
use tt_isa::isa::generated::encode;
use tt_isa::tile::{L1Format, TileImage};
use tt_kernels::datapath::{
    config_program, flat_descriptor, pack_tile_from_dst, set_adc_x, set_adc_x_unpack,
    src_thread_config, state_id, thread_config, tile_base_units, unpack_config, unpack_instruction,
    unpack_src_config, unpack_src_instruction, unpack_src_transposed_config, unpack_tileize_config,
    Unpacker, OUT, STAGE, TILE_DATUMS,
};
use tt_tests::harness::{self, Roles, Run};

// ---- tileize -----------------------------------------------------------------

/// Where the row-major matrix is staged: clear of `OUT`, with room for a 4096-byte
/// stride over 32 rows (128 KB).
const MATRIX_AT: u64 = 0x4_0000;

/// Raw FP32 patterns for a matrix: specials (both zeros, subnormals, infinities,
/// NaN payloads of both signs) every few datums, otherwise distinct normals.
fn matrix(rows: usize, cols: usize) -> Vec<u32> {
    const SPECIALS: [u32; 12] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x807f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc1_2345,
        0x7f80_0001,
        0x7f7f_ffff,
        0x0080_0000,
        0xff7f_ffff,
    ];
    (0..rows * cols)
        .map(|i| {
            if i % 5 == 0 {
                SPECIALS[(i / 5) % SPECIALS.len()]
            } else {
                0x3f80_0000 + (i as u32) * 3 + 1
            }
        })
        .collect()
}

fn matrix_image(m: &[u32]) -> Vec<u8> {
    // The unpacker reads from one header past the base (`tile_base_units`).
    let mut bytes = vec![0u8; 16 + 4 * m.len()];
    for (i, w) in m.iter().enumerate() {
        bytes[16 + 4 * i..][..4].copy_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// The page's addressing, from first principles: datum `j` of the unpack reads
/// input row `j / 32` at column `start + j % 32`.
fn tileize_model(m: &[u32], cols: usize, start: usize, datums: usize) -> Vec<u32> {
    (0..datums)
        .map(|j| m[(j / 32) * cols + start + j % 32])
        .collect()
}

/// Gather two 32-column strips of a `rows x cols` row-major matrix into `Dst`
/// rows 0.. and 64.., through the tileize unpack, and pack both back to L1.
/// `plain` runs the same program without the mode (the negative control).
fn run_tileize(
    m: &[u32],
    cols: usize,
    second_start: usize,
    plain: bool,
    check: impl FnOnce(&[u32]),
) {
    let descriptor = flat_descriptor(TILE_DATUMS);
    let stride = 4 * cols as u32;
    let mut unpack = thread_config();
    let mut words = ConfigWords::new();
    if plain {
        unpack_config(&mut words, descriptor, MATRIX_AT);
    } else {
        unpack_tileize_config(&mut words, descriptor, MATRIX_AT, stride).unwrap();
    }
    unpack.extend(config_program(&words));
    unpack.push(set_adc_x_unpack(0, TILE_DATUMS - 1));
    unpack.push(unpack_instruction());
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    // The second strip: the base moves 4 bytes a column, the destination to `Dst`
    // row 64 (`unpack_datums_to_dst`'s own retargeting, for a start that is not
    // a multiple of four datums of a tile image).
    let mut again = ConfigWords::new();
    again
        .set(
            thcon::THCON_SEC0_REG3_Base_address,
            tile_base_units(MATRIX_AT + 4 * second_start as u64),
        )
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG5_Dest_cntx0_address,
            tt_kernels::datapath::dst_address(64),
        )
        .unwrap();
    unpack.push(backend::wait_for_unpacker0(Before::CONFIG).unwrap());
    unpack.extend(config_program(&again));
    unpack.push(
        backend::stallwait(
            backend::block::UNPACKER | backend::block::CONFIG,
            backend::cond::CONFIG_BUSY,
        )
        .unwrap(),
    );
    unpack.push(set_adc_x_unpack(0, TILE_DATUMS - 1));
    unpack.push(unpack_instruction());
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());

    let mut pack = vec![state_id()];
    pack.extend(pack_tile_from_dst(OUT, 0));
    pack.extend(pack_tile_from_dst(OUT + 4096, 64));
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());

    let image = matrix_image(m);
    let sentinel = vec![0xA5u8; 8192];
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &[],
                pack: &pack,
            })
            .stage(&[(MATRIX_AT, &image), (OUT, &sentinel)])
            .read_back(&[(OUT, 8192)]),
        );
        let got: Vec<u32> = out.l1[0]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        check(&got);
    });
}

/// `(rows, cols, second strip's first column)`: a 32-column strip at column 0
/// and another; strides 256, 192 and 4096 bytes exercise each nibble.
const SHAPES: [(usize, usize, usize); 3] = [(32, 64, 32), (32, 48, 16), (32, 1024, 992)];

#[test]
fn tileize_strips_match_the_row_stride_model() {
    for (rows, cols, second) in SHAPES {
        let m = matrix(rows, cols);
        let want0 = tileize_model(&m, cols, 0, 1024);
        let want1 = tileize_model(&m, cols, second, 1024);
        run_tileize(&m, cols, second, false, |got| {
            for (i, (&g, &w)) in got[..1024].iter().zip(&want0).enumerate() {
                assert_eq!(
                    g, w,
                    "{cols} cols, strip 0, datum {i}: {g:#010x} vs {w:#010x}"
                );
            }
            for (i, (&g, &w)) in got[1024..].iter().zip(&want1).enumerate() {
                assert_eq!(
                    g, w,
                    "{cols} cols, strip at {second}, datum {i}: {g:#010x} vs {w:#010x}"
                );
            }
        });
    }
}

/// The mode is what gathers: without it the unpacker reads contiguous datums, a
/// different tile, and the oracle sees the difference (so a pass above is not
/// a contiguous read that happens to match).
#[test]
fn tileize_oracle_is_not_a_contiguous_read() {
    let (rows, cols, second) = SHAPES[0];
    let m = matrix(rows, cols);
    let strided = tileize_model(&m, cols, 0, 1024);
    assert_ne!(strided, m[..1024].to_vec());
    // Exercising the program without the mode really does give the contiguous
    // read on the simulator, and the strided oracle rejects it.
    run_tileize(&m, cols, second, true, |got| {
        assert_eq!(
            &got[..1024],
            &m[..1024],
            "plain unpack reads contiguous datums"
        );
        assert_ne!(&got[..1024], &strided[..]);
    });
}

/// Isolated first contact for silicon: one tileize unpack, nothing checked.
#[cfg(feature = "silicon")]
#[test]
fn silicon_probe_tileize_isolated() {
    let m = matrix(32, 64);
    let image = matrix_image(&m);
    let descriptor = flat_descriptor(TILE_DATUMS);
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_tileize_config(&mut words, descriptor, MATRIX_AT, 256).unwrap();
    p.extend(config_program(&words));
    p.push(set_adc_x_unpack(0, TILE_DATUMS - 1));
    p.push(unpack_instruction());
    p.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    assert!(harness::survives(|dev| {
        harness::run(
            dev,
            &Run::new(&p).stage(&[(MATRIX_AT, &image)]).dump_rows(16),
        );
    }));
}

// ---- transpose ---------------------------------------------------------------

const TF32: u32 = 4;

/// A 16x16 block of values TF32 holds exactly (integers 1..=256 as FP32), so the
/// `Src` path's own conversion is the identity and only the permutation shows.
fn block() -> Vec<u32> {
    (0..256).map(|i| ((i + 1) as f32).to_bits()).collect()
}

fn block_image(bits: &[u32]) -> Vec<u8> {
    let image = TileImage::new(flat_descriptor(256), L1Format::Fp32).unwrap();
    let mut bytes = vec![0u8; image.total_bytes()];
    for (i, d) in bits.iter().enumerate() {
        let at = image.datum_bit_offset(i) / 8;
        bytes[at..at + 4].copy_from_slice(&d.to_le_bytes());
    }
    bytes
}

/// 256 FP32 datums to `SrcA` (TF32), moved to `Dst` rows 0..16 by two eight-row
/// `MOVA2D`s; `transpose` stages `Haloize_mode`. The 16x16 result as raw words.
fn run_src_block_on(dev: &mut harness::Dev<'_>, bits: &[u32], transpose: bool) -> Vec<u32> {
    let descriptor = flat_descriptor(256);
    let mut unpack = src_thread_config();
    let mut words = ConfigWords::new();
    if transpose {
        unpack_src_transposed_config(&mut words, descriptor, STAGE, TF32).unwrap();
    } else {
        unpack_src_config(&mut words, Unpacker::SrcA, descriptor, STAGE, TF32);
    }
    words
        .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
        .unwrap();
    unpack.extend(config_program(&words));
    unpack.push(set_adc_x(Unpacker::SrcA, 0, 255));
    unpack.push(unpack_src_instruction(Unpacker::SrcA, true));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    let mut math = vec![state_id()];
    for half in [0u32, 8] {
        math.push(
            encode::Mova2D::ZERO
                .move8_rows(1)
                .src_row(half)
                .dst_row(half)
                .encode()
                .unwrap(),
        );
    }
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    let image = block_image(bits);
    let out = harness::run(
        dev,
        &Run::roles(Roles {
            unpack: &unpack,
            math: &math,
            pack: &[],
        })
        .stage(&[(STAGE, &image)])
        .dump_rows(16),
    );
    (0..256).map(|i| out.dst_at(i / 16, i % 16)).collect()
}

fn run_src_block(bits: &[u32], transpose: bool, check: impl FnOnce(&[u32])) {
    harness::in_device(|dev| check(&run_src_block_on(dev, bits, transpose)));
}

/// The page: `Row = (Row & ~0xf) | Col`, `Col = RowLowBits` -- datum `(r, c)`
/// lands at row `c`, column `r`.
fn transpose_model(m: &[u32]) -> Vec<u32> {
    (0..256).map(|i| m[(i % 16) * 16 + i / 16]).collect()
}

#[test]
fn unpacker_transpose_swaps_row_and_column_low_nibbles() {
    let b = block();
    let want = transpose_model(&b);
    assert_ne!(
        want, b,
        "negative control: a transpose that does nothing must fail"
    );
    let same = |what: &str, got: &[u32], want: &[u32]| {
        if let Some(i) = (0..256).find(|&i| got[i] != want[i]) {
            panic!(
                "{what}: datum {i} (row {}, col {}): {:#010x} vs {:#010x}",
                i / 16,
                i % 16,
                got[i],
                want[i]
            );
        }
    };
    run_src_block(&b, true, |got| same("transposed", got, &want));
    // The same program without the mode is the identity.
    run_src_block(&b, false, |got| same("plain", got, &b));
}

/// ttsim refuses the page's `UndefinedBehavior` (transpose with `UnpackToDst`)
/// by name, and takes the plain `UnpackToDst` control. The helper makes the
/// refusal unreachable from the checked API; this is the simulator's side.
#[cfg(not(feature = "silicon"))]
#[test]
fn ttsim_refuses_transpose_to_dst() {
    let image = block_image(&block());
    let build = |halo: bool| {
        let mut p = thread_config();
        let mut words = ConfigWords::new();
        unpack_config(&mut words, flat_descriptor(256), STAGE);
        // Raw field write on purpose: `stage_transpose` refuses this combination.
        words
            .set(thcon::THCON_SEC0_REG2_Haloize_mode, u32::from(halo))
            .unwrap();
        p.extend(config_program(&words));
        p.push(set_adc_x_unpack(0, 255));
        p.push(unpack_instruction());
        p.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
        p
    };
    let run = |halo: bool| {
        let p = build(halo);
        harness::survives(|dev| {
            harness::run(dev, &Run::new(&p).stage(&[(STAGE, &image)]).dump_rows(16));
        })
    };
    assert!(run(false), "UnpackToDst without transpose is the control");
    assert!(!run(true), "unpack_to_dst cannot be used with haloize");
    assert_eq!(
        tt_isa::unpacker::stage_transpose(&mut ConfigWords::new(), true),
        Err(tt_isa::unpacker::UnpackModeError::TransposeToDst)
    );
}

/// Isolated first contact for silicon: one transposed `SrcA` unpack.
#[cfg(feature = "silicon")]
#[test]
fn silicon_probe_transpose_isolated() {
    let image = block_image(&block());
    let mut unpack = src_thread_config();
    let mut words = ConfigWords::new();
    unpack_src_transposed_config(&mut words, flat_descriptor(256), STAGE, TF32).unwrap();
    words
        .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
        .unwrap();
    unpack.extend(config_program(&words));
    unpack.push(set_adc_x(Unpacker::SrcA, 0, 255));
    unpack.push(unpack_src_instruction(Unpacker::SrcA, true));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    let math = vec![
        state_id(),
        encode::Mova2D::ZERO
            .move8_rows(1)
            .src_row(0)
            .dst_row(0)
            .encode()
            .unwrap(),
        backend::wait_for_matrix(Before::EVERYTHING).unwrap(),
    ];
    assert!(harness::survives(|dev| {
        harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &math,
                pack: &[],
            })
            .stage(&[(STAGE, &image)])
            .dump_rows(8),
        );
    }));
}

/// M3's measurement: what does the `Src` path do to payloads, and does the
/// unpacker transpose change it? The 256 datums are -0, +0, subnormals (both
/// signs, sizes across the mantissa), infinities, NaN payloads of both signs,
/// extremes, and values with mantissa bits on both sides of the TF32 cut.
/// Transposed output must be the plain `Src` path's output permuted -- the
/// transpose adds no conversion of its own -- and `MEASURE` lines record, per
/// class, whether the plain path preserved the FP32 payload (it cannot for the
/// 13 low mantissa bits: `SrcA` holds 19 bits). Either way the mover's
/// `READ_TRANSPOSED` stays the payload-preserving contract.
#[cfg(feature = "silicon")]
#[ignore = "measurement: on card 0 the unpacker transpose into SrcA is NOT a pure permutation of the plain SrcA path (the plain path itself normalizes signed zeros, subnormals and low mantissa bits); the MEASURE lines are the M3 evidence, see hardware-coverage.md"]
#[test]
fn silicon_transpose_is_a_pure_permutation_of_the_plain_src_path() {
    const SPECIALS: [u32; 20] = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0040_0000,
        0x0080_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc0_0000,
        0x7fc1_2345,
        0xffc1_2345,
        0x7f80_0001,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x3f80_1fff,
        0x3f80_2000,
        0xbf80_3fff,
    ];
    let bits: Vec<u32> = (0..256)
        .map(|i| {
            if i < SPECIALS.len() {
                SPECIALS[i]
            } else {
                ((i + 1) as f32).to_bits() | (i as u32 & 0x1fff)
            }
        })
        .collect();
    harness::in_device(|dev| {
        let plain = run_src_block_on(dev, &bits, false);
        for (i, (&x, &g)) in bits.iter().zip(&plain).enumerate().take(SPECIALS.len()) {
            println!(
                "MEASURE src.plain[{i}] {x:#010x} -> {g:#010x} ({})",
                if x == g { "preserved" } else { "changed" }
            );
        }
        let got = run_src_block_on(dev, &bits, true);
        let want = transpose_model(&plain);
        for (i, ((&g, &w), &x)) in got.iter().zip(&want).zip(&bits).enumerate() {
            if i < SPECIALS.len() {
                println!("MEASURE src.transposed[{i}] (from {x:#010x}) -> {g:#010x}");
            }
            assert_eq!(
                g, w,
                "datum {i}: transposed {g:#010x}, plain-path permuted {w:#010x}"
            );
        }
    });
}
