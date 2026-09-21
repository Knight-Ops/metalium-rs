//! G2 — every datum sits at the byte the unpacker would fetch it from.
//!
//! # What this gate can and cannot establish
//!
//! `UNPACR_Regular.md:55-212` determines the byte offset of a datum **given its
//! `(W, Z, Y, X)` coordinate**. That much is specification, and the first two tests
//! here check it against a transcription of the formula written from the document
//! rather than by calling [`tt_isa::tile::TileImage`].
//!
//! What the specification does *not* determine is which logical `(row, column)` of a
//! tensor a given `(W, Z, Y, X)` corresponds to. That mapping -- which `Z` plane is
//! which face of the tile -- is software convention, settled in Phase 6 against what
//! the unpacker's ADC walk wants. The last test here therefore *pins* the
//! convention rather than validating it: it fails when the convention changes, so
//! the change is deliberate and visible, not silent.
//!
//! This distinction matters. `roundtrip.rs` passes unchanged if the face convention
//! is transposed, because tilize and detilize agree with each other either way.

use tt_isa::tile::{L1Format, TileDescriptor, TileImage};
use tt_layout::{tilize, HostDtype, Layout, PadValue, TensorView};

/// Byte offset of datum `index`, transcribed from the document.
///
/// Deliberately does not call [`TileImage`]: an oracle that shares code with the
/// thing it checks is not an oracle.
///
/// ```text
/// InAddr = (Base + Offset + 1 + DigestSize) * 16      // :113
/// if BFP and not NoBFPExpSection:
///     NumExponents = ceil(XDim*YDim*ZDim*WDim / 16)   // :134-135
///     InAddr += ceil(NumExponents / 16) * 16          // :136
/// InAddr_Datums = InAddr + FirstDatum * DatumSizeBytes // :204
/// ```
fn documented_byte_offset(d: TileDescriptor, format: L1Format, index: usize) -> usize {
    // Relative to the tile base, so the configured `Base + Offset` drops out.
    let mut addr = (1 + d.digest_size() as usize) * 16;

    let is_bfp = matches!(
        format,
        L1Format::Bfp2
            | L1Format::Bfp2a
            | L1Format::Bfp4
            | L1Format::Bfp4a
            | L1Format::Bfp8
            | L1Format::Bfp8a
    );
    if is_bfp && !d.no_bfp_exp_section() {
        let z = if d.z_dim_raw() == 0 { 1 } else { d.z_dim_raw() };
        let w = if d.w_dim_raw() == 0 { 1 } else { d.w_dim_raw() };
        let elements = d.x_dim() as usize * d.y_dim() as usize * z as usize * w as usize;
        let exponents = elements.div_ceil(16);
        addr += exponents.div_ceil(16) * 16;
    }

    // DatumSizeBytes, from the switch at :92-98.
    let datum_bytes = match format {
        L1Format::Fp32 | L1Format::Tf32 | L1Format::Int32 => 4,
        L1Format::Bf16 | L1Format::Fp16 | L1Format::Int16 => 2,
        _ => 1,
    };
    addr + index * datum_bytes
}

/// `FirstDatum = ((W * ZDim + Z) * YDim + Y) * XDim + X` (`:182`), transcribed.
fn documented_first_datum(d: TileDescriptor, w: u32, z: u32, y: u32, x: u32) -> usize {
    let z_dim = if d.z_dim_raw() == 0 { 1 } else { d.z_dim_raw() };
    (((w as usize * z_dim as usize) + z as usize) * d.y_dim() as usize + y as usize)
        * d.x_dim() as usize
        + x as usize
}

fn descriptors() -> Vec<(&'static str, TileDescriptor, L1Format)> {
    let base = TileDescriptor::zeroed().with_is_uncompressed(true);
    vec![
        (
            "32x32 as four 16x16 faces, FP32",
            base.with_x_dim(16).with_y_dim(16).with_z_dim(4),
            L1Format::Fp32,
        ),
        (
            "32x32 as four 16x16 faces, BF16",
            base.with_x_dim(16).with_y_dim(16).with_z_dim(4),
            L1Format::Bf16,
        ),
        (
            "one flat 16x16 face, FP16",
            base.with_x_dim(16).with_y_dim(16),
            L1Format::Fp16,
        ),
        (
            "a header of five 16-byte blocks",
            base.with_x_dim(16).with_y_dim(16).with_digest_size(4),
            L1Format::Fp32,
        ),
        (
            "a W dimension",
            base.with_x_dim(8).with_y_dim(4).with_z_dim(2).with_w_dim(3),
            L1Format::Bf16,
        ),
        (
            "BFP8, whose exponent section displaces the datums",
            base.with_x_dim(16).with_y_dim(16).with_z_dim(4),
            L1Format::Bfp8,
        ),
        (
            "BFP8 with the exponent section suppressed",
            base.with_x_dim(16)
                .with_y_dim(16)
                .with_z_dim(4)
                .with_no_bfp_exp_section(true),
            L1Format::Bfp8,
        ),
    ]
}

/// The image model agrees with the document, coordinate by coordinate.
#[test]
fn every_datum_offset_matches_the_documented_address_generator() {
    for (name, d, format) in descriptors() {
        let image = TileImage::new(d, format).unwrap();

        assert_eq!(
            image.header_bytes(),
            (1 + d.digest_size() as usize) * 16,
            "{name}: header"
        );

        for w in 0..d.w_dim() {
            for z in 0..d.z_dim() {
                for y in 0..d.y_dim() {
                    for x in 0..d.x_dim() {
                        let expected_index = documented_first_datum(d, w, z, y, x);
                        let index = image.datum_index(w, z, y, x);
                        assert_eq!(
                            index, expected_index,
                            "{name}: FirstDatum at ({w}, {z}, {y}, {x})"
                        );

                        let expected = documented_byte_offset(d, format, index);
                        assert_eq!(
                            image.datum_bit_offset(index),
                            expected * 8,
                            "{name}: byte offset of datum {index}"
                        );
                    }
                }
            }
        }

        // The exponent for a datum is one byte per sixteen datums, counted from the
        // start of the exponent section (`:202`).
        if format.has_exponent_section() && !d.no_bfp_exp_section() {
            for index in [0usize, 15, 16, 31, 1023] {
                if index >= image.datum_count() {
                    continue;
                }
                assert_eq!(
                    image.exponent_byte_offset(index),
                    Some(image.header_bytes() + index / 16),
                    "{name}: exponent for datum {index}"
                );
            }
        }
    }
}

/// The oracle rejects a descriptor whose dimensions have been swapped.
///
/// Watching the gate fail: `YDim` and `ZDim` are interchangeable in the *size* of a
/// tile but not in the *order* datums are visited, so a swap must be visible.
#[test]
fn swapping_y_and_z_changes_the_addresses() {
    let base = TileDescriptor::zeroed()
        .with_is_uncompressed(true)
        .with_x_dim(8);
    let normal = base.with_y_dim(4).with_z_dim(2);
    let swapped = base.with_y_dim(2).with_z_dim(4);

    let a = TileImage::new(normal, L1Format::Fp32).unwrap();
    let b = TileImage::new(swapped, L1Format::Fp32).unwrap();
    assert_eq!(a.datum_count(), b.datum_count(), "same size");

    // Same coordinate, different flat index: YDim is the inner of the two.
    assert_eq!(a.datum_index(0, 1, 0, 0), 32);
    assert_eq!(b.datum_index(0, 1, 0, 0), 16);
    assert_ne!(
        documented_first_datum(normal, 0, 1, 0, 0),
        documented_first_datum(swapped, 0, 1, 0, 0),
        "the oracle itself must distinguish them, or it proves nothing"
    );
}

/// Where `tilize` actually put each element, checked at the documented address.
///
/// Values are unique per logical element, so finding the right one at the computed
/// offset is evidence about placement rather than about the value.
#[test]
fn tilize_writes_each_element_where_the_unpacker_would_read_it() {
    let dtype = HostDtype::F32;
    let shape = [1usize, 32, 32];
    let layout = Layout::tt_metal_32x32(L1Format::Fp32, dtype, shape).unwrap();
    let image = layout.image();
    let d = image.descriptor();

    // Element (r, c) carries the value r * 32 + c, exactly representable in FP32.
    let mut src_bytes = Vec::new();
    for r in 0..32u32 {
        for c in 0..32u32 {
            src_bytes.extend_from_slice(&((r * 32 + c) as f32).to_bits().to_le_bytes());
        }
    }
    let src = TensorView::contiguous(&src_bytes, dtype, shape);
    let tiled = tilize(&src, &layout).unwrap();

    for z in 0..d.z_dim() {
        for y in 0..d.y_dim() {
            for x in 0..d.x_dim() {
                let index = documented_first_datum(d, 0, z, y, x);
                let at = documented_byte_offset(d, L1Format::Fp32, index);
                let found =
                    f32::from_bits(u32::from_le_bytes(tiled[at..at + 4].try_into().unwrap()));
                let (row, col) = layout.logical_coord(0, 0, 0, z, y, x);
                let expected = (row * 32 + col) as f32;
                assert_eq!(
                    found, expected,
                    "datum ({z}, {y}, {x}) -> logical ({row}, {col}) at byte {at}"
                );
            }
        }
    }
}

/// The tile-local coordinate map covers the tile's logical patch exactly once.
///
/// Bijectivity is the part of the convention that *is* required: a map that missed a
/// logical element or claimed one twice would corrupt data whatever Phase 6 decides
/// about face ordering.
#[test]
fn the_coordinate_map_is_a_bijection_onto_the_tile() {
    for (faces_down, faces_across, x, y) in
        [(2usize, 2usize, 16u32, 16u32), (1, 4, 8, 4), (4, 1, 4, 8)]
    {
        let d = TileDescriptor::zeroed()
            .with_is_uncompressed(true)
            .with_x_dim(x)
            .with_y_dim(y)
            .with_z_dim((faces_down * faces_across) as u32);
        let height = y as usize * faces_down;
        let width = x as usize * faces_across;
        let layout = Layout::new(
            d,
            L1Format::Fp32,
            HostDtype::F32,
            [1, height, width],
            faces_down,
            faces_across,
            PadValue::Zero,
        )
        .unwrap();

        let mut seen = vec![false; height * width];
        for z in 0..d.z_dim() {
            for yy in 0..d.y_dim() {
                for xx in 0..d.x_dim() {
                    let (r, c) = layout.logical_coord(0, 0, 0, z, yy, xx);
                    assert!(r < height && c < width, "({r}, {c}) outside the tile");
                    let slot = r * width + c;
                    assert!(!seen[slot], "logical ({r}, {c}) claimed twice");
                    seen[slot] = true;
                }
            }
        }
        assert!(
            seen.iter().all(|s| *s),
            "{faces_down}x{faces_across}: some logical element has no datum"
        );
    }
}

/// Pins the face convention, so changing it is deliberate.
///
/// **Not a correctness claim.** Nothing in the ISA documentation says which face is
/// which; this records the choice `Layout::tt_metal_32x32` makes so that a change
/// shows up as a failing test with this comment attached, rather than as silently
/// different bytes. Phase 6 settles it against the unpacker.
#[test]
fn the_face_convention_is_row_major_faces_of_row_major_datums() {
    let layout = Layout::tt_metal_32x32(L1Format::Fp32, HostDtype::F32, [1, 32, 32]).unwrap();

    // Face 0 is the top-left 16x16.
    assert_eq!(layout.logical_coord(0, 0, 0, 0, 0, 0), (0, 0));
    assert_eq!(layout.logical_coord(0, 0, 0, 0, 15, 15), (15, 15));
    // Face 1 is the top-right, i.e. Z steps across before it steps down.
    assert_eq!(
        layout.logical_coord(0, 0, 0, 1, 0, 0),
        (0, 16),
        "face 1 is top-right: Z advances across the tile before down it"
    );
    // Face 2 is the bottom-left, face 3 the bottom-right.
    assert_eq!(layout.logical_coord(0, 0, 0, 2, 0, 0), (16, 0));
    assert_eq!(layout.logical_coord(0, 0, 0, 3, 0, 0), (16, 16));

    // Within a face, X is the column: consecutive X are consecutive columns, and
    // consecutive datums in memory.
    assert_eq!(layout.logical_coord(0, 0, 0, 0, 0, 1), (0, 1));
    assert_eq!(layout.logical_coord(0, 0, 0, 0, 1, 0), (1, 0));
}
