//! How a logical `[batch, rows, cols]` tensor maps onto a grid of tile images.

use tt_isa::tile::{L1Format, TileDescriptor, TileImage};

use crate::convert::{self, HostDtype, LayoutError};
use crate::view::{TensorView, TensorViewMut};

/// What fills the elements a padded tile has but the tensor does not.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PadValue {
    /// All-zero bits, which is `+0.0` in every format here.
    ///
    /// Named rather than assumed: zero is right for accumulation and wrong for a
    /// min-reduction, and the caller is the only one who knows which this is.
    Zero,
}

/// A tensor-to-tiles mapping.
///
/// Holds the descriptor, the face arrangement within a tile, and the grid extent, so
/// none of it is recomputed per element.
#[derive(Copy, Clone, Debug)]
pub struct Layout {
    image: TileImage,
    format: L1Format,
    dtype: HostDtype,
    batch: usize,
    rows: usize,
    cols: usize,
    faces_down: usize,
    faces_across: usize,
    tile_rows: usize,
    tile_cols: usize,
    pad: PadValue,
}

impl Layout {
    /// Build a layout from an explicit descriptor and face arrangement.
    ///
    /// # The face convention
    ///
    /// A tile's `ZDim` faces are laid out `faces_down` by `faces_across`, walked in
    /// row-major order, each face `YDim` rows by `XDim` columns. **Nothing in the
    /// ISA documentation mandates this**: the address generator only says datums run
    /// `W`, `Z`, `Y`, `X` with `X` fastest (`UNPACR_Regular.md:182`), and which
    /// two-dimensional patch a `Z` plane corresponds to is software's choice. It is
    /// a parameter here rather than a constant so that Phase 6 can settle it against
    /// what the unpacker's ADC walk actually wants, without redesigning anything.
    pub fn new(
        descriptor: TileDescriptor,
        format: L1Format,
        dtype: HostDtype,
        shape: [usize; 3],
        faces_down: usize,
        faces_across: usize,
        pad: PadValue,
    ) -> Result<Self, LayoutError> {
        if !convert::is_supported(format) {
            return Err(LayoutError::UnsupportedFormat { format });
        }
        if format.datum_bits() % 8 != 0 {
            return Err(LayoutError::UnsupportedDescriptor {
                why: "sub-byte datums have no host encoder until the packers run",
            });
        }
        if descriptor.w_dim() != 1 {
            return Err(LayoutError::UnsupportedDescriptor {
                why: "WDim > 1 packs several matrices into one tile; not used yet",
            });
        }
        if faces_down == 0 || faces_across == 0 {
            return Err(LayoutError::UnsupportedDescriptor {
                why: "a tile must have at least one face in each direction",
            });
        }
        if descriptor.z_dim() as usize != faces_down * faces_across {
            return Err(LayoutError::FaceGridMismatch {
                z_dim: descriptor.z_dim(),
                faces_down,
                faces_across,
            });
        }
        if !descriptor.reserved_bits_are_zero() {
            return Err(LayoutError::UnsupportedDescriptor {
                why: "the descriptor has bits set in a reserved run",
            });
        }

        let image =
            TileImage::new(descriptor, format).map_err(|_| LayoutError::UnsupportedDescriptor {
                why: "the descriptor does not describe a reachable tile",
            })?;

        let [batch, rows, cols] = shape;
        let tile_height = descriptor.y_dim() as usize * faces_down;
        let tile_width = descriptor.x_dim() as usize * faces_across;

        Ok(Layout {
            image,
            format,
            dtype,
            batch,
            rows,
            cols,
            faces_down,
            faces_across,
            tile_rows: rows.div_ceil(tile_height),
            tile_cols: cols.div_ceil(tile_width),
            pad,
        })
    }

    /// The conventional 32x32 tile: four 16x16 faces in a 2x2 arrangement.
    ///
    /// This is the tt-metal shape. We have no C++ dependency that forces it, and the
    /// binding constraint is what the unpacker expects, so treat it as a default
    /// rather than a requirement -- see [`Layout::new`] on the face convention.
    pub fn tt_metal_32x32(
        format: L1Format,
        dtype: HostDtype,
        shape: [usize; 3],
    ) -> Result<Self, LayoutError> {
        let descriptor = TileDescriptor::zeroed()
            .with_x_dim(16)
            .with_y_dim(16)
            .with_z_dim(4)
            .with_is_uncompressed(true);
        Layout::new(descriptor, format, dtype, shape, 2, 2, PadValue::Zero)
    }

    pub fn image(&self) -> TileImage {
        self.image
    }

    pub fn dtype(&self) -> HostDtype {
        self.dtype
    }

    pub fn format(&self) -> L1Format {
        self.format
    }

    /// Logical elements one tile covers, as `(rows, columns)`.
    pub fn tile_extent(&self) -> (usize, usize) {
        (
            self.image.descriptor().y_dim() as usize * self.faces_down,
            self.image.descriptor().x_dim() as usize * self.faces_across,
        )
    }

    /// Tiles covering one `[rows, cols]` matrix.
    pub fn tiles_per_matrix(&self) -> usize {
        self.tile_rows * self.tile_cols
    }

    /// Tiles in the whole buffer, batch included.
    pub fn grid_tiles(&self) -> usize {
        self.tiles_per_matrix() * self.batch
    }

    /// Size of the whole tiled buffer.
    pub fn total_bytes(&self) -> usize {
        self.grid_tiles() * self.image.total_bytes()
    }

    /// Which tile of its matrix this flat tile index is, as `(row, column)`.
    pub fn tile_coord(&self, tile: usize) -> (usize, usize) {
        let within = tile % self.tiles_per_matrix();
        (within / self.tile_cols, within % self.tile_cols)
    }

    /// The logical `(row, column)` a datum coordinate names.
    ///
    /// Padding lands outside the tensor, which [`Layout::in_bounds`] detects.
    pub fn logical_coord(
        &self,
        tile_row: usize,
        tile_col: usize,
        _w: u32,
        z: u32,
        y: u32,
        x: u32,
    ) -> (usize, usize) {
        let d = self.image.descriptor();
        let face_row = z as usize / self.faces_across;
        let face_col = z as usize % self.faces_across;
        let (tile_height, tile_width) = self.tile_extent();
        (
            tile_row * tile_height + face_row * d.y_dim() as usize + y as usize,
            tile_col * tile_width + face_col * d.x_dim() as usize + x as usize,
        )
    }

    pub fn in_bounds(&self, row: usize, col: usize) -> bool {
        row < self.rows && col < self.cols
    }

    pub(crate) fn pad_bits(&self) -> u32 {
        match self.pad {
            PadValue::Zero => 0,
        }
    }

    pub(crate) fn check_source(&self, src: &TensorView<'_>) -> Result<(), LayoutError> {
        self.check_shape(src.shape(), src.dtype())
    }

    pub(crate) fn check_destination(&self, dst: &TensorViewMut<'_>) -> Result<(), LayoutError> {
        self.check_shape(dst.shape(), dst.dtype())
    }

    fn check_shape(&self, shape: [usize; 3], dtype: HostDtype) -> Result<(), LayoutError> {
        let expected = [self.batch, self.rows, self.cols];
        if shape != expected {
            return Err(LayoutError::ShapeMismatch {
                expected,
                found: shape,
            });
        }
        if dtype != self.dtype {
            return Err(LayoutError::DtypeMismatch {
                expected: self.dtype,
                found: dtype,
            });
        }
        Ok(())
    }

    /// Read a logical element, or `None` where the tile is padding.
    pub(crate) fn read_source(
        &self,
        src: &TensorView<'_>,
        batch: usize,
        row: usize,
        col: usize,
    ) -> Option<u32> {
        if !self.in_bounds(row, col) {
            return None;
        }
        src.element(batch, row, col)
    }

    pub(crate) fn write_destination(
        &self,
        dst: &mut TensorViewMut<'_>,
        batch: usize,
        row: usize,
        col: usize,
        bits: u32,
    ) {
        dst.set_element(batch, row, col, bits);
    }

    /// Place one datum, converting from the host element type.
    pub(crate) fn write_datum(
        &self,
        out: &mut [u8],
        image: TileImage,
        index: usize,
        host_bits: u32,
    ) -> Result<(), LayoutError> {
        let datum = convert::host_to_l1(self.dtype, host_bits, self.format)?;
        let bytes = self.format.datum_bits() as usize / 8;
        let at = image.datum_bit_offset(index) / 8;
        out[at..at + bytes].copy_from_slice(&datum.to_le_bytes()[..bytes]);
        Ok(())
    }

    /// Read one datum back, converting to the host element type.
    pub(crate) fn read_datum(
        &self,
        tiled: &[u8],
        image: TileImage,
        index: usize,
    ) -> Result<u32, LayoutError> {
        let bytes = self.format.datum_bits() as usize / 8;
        let at = image.datum_bit_offset(index) / 8;
        let mut raw = [0u8; 4];
        raw[..bytes].copy_from_slice(&tiled[at..at + bytes]);
        let datum = u32::from_le_bytes(raw);
        convert::l1_to_host(self.format, datum, self.dtype)
    }
}
