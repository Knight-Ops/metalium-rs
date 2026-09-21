//! Host-side tilization: a strided row-major tensor to the L1 image the unpacker
//! reads, and back.
//!
//! # Where the shape comes from
//!
//! Nothing here decides what a tile looks like. [`tt_isa::tile::TileDescriptor`]
//! does, and this crate walks the logical tensor in whatever `W`/`Z`/`Y`/`X` order
//! that descriptor implies. The familiar 32x32-of-four-16x16-faces tile is one
//! descriptor among many, offered as [`Layout::tt_metal_32x32`].
//!
//! # What is in scope
//!
//! FP32, BF16 and FP16. The image model underneath handles block-float sizing (see
//! `tt_isa::tile`), but no BFP encoder exists: the packer's encode direction is
//! documented only in the Wormhole tree and cannot be checked against anything until
//! the packers run, in Phase 6.
//!
//! # Padding
//!
//! A tensor whose last two dimensions are not multiples of the tile's `X` and `Y`
//! extents is padded, and [`detilize`] drops the padding again. The packer can do
//! this in hardware with edge masking (`Packers/EdgeMasking.md`), which is the
//! Phase 6 answer for the write-back direction; this is the host-side one.

#![forbid(unsafe_code)]

mod convert;
mod layout;
mod view;

pub use convert::{HostDtype, LayoutError};
pub use layout::{Layout, PadValue};
pub use view::{TensorView, TensorViewMut};

use tt_isa::tile::TileImage;

/// Lay a tensor out as a grid of tile images.
///
/// The result is one contiguous buffer: `grid_tiles()` images of
/// `TileImage::total_bytes()` each, in row-major tile order.
pub fn tilize(src: &TensorView<'_>, layout: &Layout) -> Result<Vec<u8>, LayoutError> {
    layout.check_source(src)?;
    let image = layout.image();
    let mut out = vec![0u8; layout.total_bytes()];

    for tile in 0..layout.grid_tiles() {
        let base = tile * image.total_bytes();
        let (tile_row, tile_col) = layout.tile_coord(tile);
        write_one_tile(src, layout, tile, tile_row, tile_col, &mut out[base..])?;
    }
    Ok(out)
}

fn write_one_tile(
    src: &TensorView<'_>,
    layout: &Layout,
    tile: usize,
    tile_row: usize,
    tile_col: usize,
    out: &mut [u8],
) -> Result<(), LayoutError> {
    let image = layout.image();
    let d = image.descriptor();
    let batch = tile / layout.tiles_per_matrix();

    for w in 0..d.w_dim() {
        for z in 0..d.z_dim() {
            for y in 0..d.y_dim() {
                for x in 0..d.x_dim() {
                    let index = image.datum_index(w, z, y, x);
                    let (row, col) = layout.logical_coord(tile_row, tile_col, w, z, y, x);
                    let value = match layout.read_source(src, batch, row, col) {
                        Some(v) => v,
                        None => layout.pad_bits(),
                    };
                    layout.write_datum(out, image, index, value)?;
                }
            }
        }
    }
    Ok(())
}

/// The inverse of [`tilize`]: read a tiled buffer back into a strided tensor,
/// dropping padding.
pub fn detilize(
    tiled: &[u8],
    layout: &Layout,
    dst: &mut TensorViewMut<'_>,
) -> Result<(), LayoutError> {
    layout.check_destination(dst)?;
    if tiled.len() != layout.total_bytes() {
        return Err(LayoutError::BufferLength {
            expected: layout.total_bytes(),
            found: tiled.len(),
        });
    }
    let image: TileImage = layout.image();
    let d = image.descriptor();

    for tile in 0..layout.grid_tiles() {
        let base = tile * image.total_bytes();
        let (tile_row, tile_col) = layout.tile_coord(tile);
        let batch = tile / layout.tiles_per_matrix();
        for w in 0..d.w_dim() {
            for z in 0..d.z_dim() {
                for y in 0..d.y_dim() {
                    for x in 0..d.x_dim() {
                        let index = image.datum_index(w, z, y, x);
                        let (row, col) = layout.logical_coord(tile_row, tile_col, w, z, y, x);
                        // Padding has no logical coordinate; skip it.
                        if !layout.in_bounds(row, col) {
                            continue;
                        }
                        let value = layout.read_datum(&tiled[base..], image, index)?;
                        layout.write_destination(dst, batch, row, col, value);
                    }
                }
            }
        }
    }
    Ok(())
}
