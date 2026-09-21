//! G1 — tilize then detilize is the identity, over a generated corpus of shapes.
//!
//! This gate on its own proves only self-consistency: a layout that placed every
//! datum at a wrong-but-consistent offset would pass it. `placement.rs` is the gate
//! that checks the offsets against the specification. Both are needed; neither
//! replaces the other.

use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use tt_isa::tile::L1Format;
use tt_layout::{detilize, tilize, HostDtype, Layout, TensorView, TensorViewMut};

/// A deterministic runner.
///
/// proptest seeds itself from the OS by default, which would make a green CI run
/// mean something different each time -- and this repo's CI is deterministic on
/// purpose. A fixed algorithm and seed keep a failure reproducible from the output
/// alone.
fn runner() -> TestRunner {
    TestRunner::new_with_rng(
        Config {
            cases: 256,
            failure_persistence: None,
            ..Config::default()
        },
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    )
}

fn dtypes() -> [HostDtype; 3] {
    [HostDtype::F32, HostDtype::Bf16, HostDtype::F16]
}

fn l1_format(dtype: HostDtype) -> L1Format {
    dtype.identical_l1_format()
}

/// Fill a buffer with values every format in scope represents exactly.
///
/// Small integers: exactly representable in FP32, BF16 and FP16 alike, so a round
/// trip that loses anything is a placement bug rather than a rounding one. Values
/// that do *not* survive conversion are the subject of the numerics tests in
/// `tt_isa::tile`, not of this gate.
fn fill(dtype: HostDtype, count: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(count * dtype.bytes());
    for i in 0..count {
        let v = ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) % 64) as i32 - 32;
        let f = v as f32;
        match dtype {
            HostDtype::F32 => out.extend_from_slice(&f.to_bits().to_le_bytes()),
            HostDtype::Bf16 => {
                out.extend_from_slice(&tt_isa::tile::fp32_to_bf16_round(f.to_bits()).to_le_bytes())
            }
            HostDtype::F16 => out.extend_from_slice(
                &tt_isa::tile::fp32_to_fp16(f.to_bits())
                    .unwrap()
                    .to_le_bytes(),
            ),
        }
    }
    out
}

fn round_trips(dtype: HostDtype, shape: [usize; 3]) -> Result<(), String> {
    let [b, r, c] = shape;
    let layout = Layout::tt_metal_32x32(l1_format(dtype), dtype, shape)
        .map_err(|e| format!("layout: {e}"))?;

    let src_bytes = fill(dtype, b * r * c, 7);
    let src = TensorView::contiguous(&src_bytes, dtype, shape);

    let tiled = tilize(&src, &layout).map_err(|e| format!("tilize: {e}"))?;
    if tiled.len() != layout.total_bytes() {
        return Err(format!(
            "tiled length {} != {}",
            tiled.len(),
            layout.total_bytes()
        ));
    }

    let mut back_bytes = vec![0xAAu8; src_bytes.len()];
    let mut back = TensorViewMut::contiguous(&mut back_bytes, dtype, shape);
    detilize(&tiled, &layout, &mut back).map_err(|e| format!("detilize: {e}"))?;

    if back_bytes != src_bytes {
        let at = back_bytes
            .iter()
            .zip(&src_bytes)
            .position(|(a, b)| a != b)
            .unwrap();
        return Err(format!(
            "{dtype:?} {shape:?}: first difference at byte {at}: {:#04x} != {:#04x}",
            back_bytes[at], src_bytes[at]
        ));
    }
    Ok(())
}

/// The awkward shapes the plan names, plus the boundaries worth naming.
#[test]
fn the_named_awkward_shapes_round_trip() {
    let shapes: &[[usize; 3]] = &[
        [1, 13, 47],  // named in RUST_IMPL_PLAN.md
        [1, 1, 1024], // named in RUST_IMPL_PLAN.md
        [1, 1, 1],    // one element in a 32x32 tile: 1023 pads
        [1, 32, 32],  // exactly one tile
        [1, 33, 33],  // one element into the second tile in both directions
        [1, 31, 31],  // one short in both directions
        [3, 32, 32],  // batching
        [2, 47, 65],  // batched and ragged
        [1, 16, 16],  // exactly one face
        [1, 17, 16],  // one row into the second face
        [1, 64, 64],  // 2x2 tiles
        [1, 128, 8],  // tall and narrow
        [1, 8, 128],  // short and wide
    ];
    for dtype in dtypes() {
        for shape in shapes {
            round_trips(dtype, *shape).unwrap_or_else(|e| panic!("{e}"));
        }
    }
}

#[test]
fn generated_shapes_round_trip() {
    let mut runner = runner();
    runner
        .run(
            &(0usize..3, 1usize..100, 1usize..100, 0usize..3),
            |(d, rows, cols, batch)| {
                let dtype = dtypes()[d];
                round_trips(dtype, [batch + 1, rows, cols]).map_err(TestCaseError::fail)?;
                Ok(())
            },
        )
        .unwrap();
}

/// Padding is zero, and detilize does not write outside the tensor.
#[test]
fn padding_is_zero_and_is_dropped_again() {
    let dtype = HostDtype::F32;
    let shape = [1, 3, 5];
    let layout = Layout::tt_metal_32x32(L1Format::Fp32, dtype, shape).unwrap();
    assert_eq!(layout.grid_tiles(), 1, "3x5 fits in one 32x32 tile");

    let src_bytes = fill(dtype, 15, 1);
    let src = TensorView::contiguous(&src_bytes, dtype, shape);
    let tiled = tilize(&src, &layout).unwrap();

    // 1024 datums, of which 15 carry data; the rest are zero.
    let image = layout.image();
    let mut nonzero = 0;
    for i in 0..image.datum_count() {
        let at = image.datum_bit_offset(i) / 8;
        let word = u32::from_le_bytes(tiled[at..at + 4].try_into().unwrap());
        if word != 0 {
            nonzero += 1;
        }
    }
    // The source contains a zero of its own, so at most 15 datums are non-zero.
    assert!(
        nonzero <= 15,
        "{nonzero} non-zero datums, but only 15 elements were supplied"
    );

    // detilize must not touch anything outside the logical shape.
    let mut back_bytes = vec![0xAAu8; src_bytes.len() + 16];
    let mut back = TensorViewMut::contiguous(&mut back_bytes[..src_bytes.len()], dtype, shape);
    detilize(&tiled, &layout, &mut back).unwrap();
    assert_eq!(&back_bytes[..src_bytes.len()], &src_bytes[..]);
    assert!(
        back_bytes[src_bytes.len()..].iter().all(|b| *b == 0xAA),
        "detilize wrote past the end of the destination"
    );
}

/// A non-contiguous source is read through its strides, not assumed contiguous.
#[test]
fn a_transposed_view_tilizes_as_the_transpose() {
    let dtype = HostDtype::F32;
    // A 4x3 buffer read as its 3x4 transpose: swap the row and column strides.
    let data = fill(dtype, 12, 3);
    let normal = TensorView::contiguous(&data, dtype, [1, 4, 3]);
    let transposed = TensorView::strided(&data, dtype, [1, 3, 4], [12, 1, 3], 0);

    let l_normal = Layout::tt_metal_32x32(L1Format::Fp32, dtype, [1, 4, 3]).unwrap();
    let l_transposed = Layout::tt_metal_32x32(L1Format::Fp32, dtype, [1, 3, 4]).unwrap();

    let a = tilize(&normal, &l_normal).unwrap();
    let b = tilize(&transposed, &l_transposed).unwrap();
    assert_ne!(
        a, b,
        "a transpose that changed nothing would mean the strides were ignored"
    );

    // Reading it back through the same strides returns the original buffer.
    let mut out = vec![0u8; data.len()];
    let mut dst = TensorViewMut::strided(&mut out, dtype, [1, 3, 4], [12, 1, 3], 0);
    detilize(&b, &l_transposed, &mut dst).unwrap();
    assert_eq!(out, data);
}

/// A shape or dtype the layout was not built for is refused, not misread.
#[test]
fn a_mismatched_view_is_refused() {
    let dtype = HostDtype::F32;
    let layout = Layout::tt_metal_32x32(L1Format::Fp32, dtype, [1, 8, 8]).unwrap();
    let data = fill(dtype, 64, 0);

    let wrong_shape = TensorView::contiguous(&data, dtype, [1, 4, 16]);
    assert!(tilize(&wrong_shape, &layout).is_err());

    let wrong_dtype = TensorView::contiguous(&data, HostDtype::Bf16, [1, 8, 8]);
    assert!(tilize(&wrong_dtype, &layout).is_err());

    let right = TensorView::contiguous(&data, dtype, [1, 8, 8]);
    let tiled = tilize(&right, &layout).unwrap();
    let mut out = vec![0u8; data.len()];
    let mut dst = TensorViewMut::contiguous(&mut out, dtype, [1, 8, 8]);
    assert!(detilize(&tiled[..tiled.len() - 1], &layout, &mut dst).is_err());
}
