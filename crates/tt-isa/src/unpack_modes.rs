//! Checked unpacker configuration for the two input-side modes the Blackhole
//! pages and `UNPACR_Regular.md` document: tileize (strided input rows) and
//! transpose.
//!
//! Source: `UNPACR_Regular.md` (conditionalized: its `TTArchitecture ==
//! Blackhole` branches are authoritative), `cfg_defines.h`. The fields keep
//! their Wormhole names: `THCON_SEC0_REG2_Tileize_mode` is the page's
//! `DiscontiguousInputRows`, `THCON_SEC0_REG2_Haloize_mode` its `Transpose`,
//! and in tileize mode `THCON_SEC0_REG2_Shift_amount_cntx0..2` are the three
//! nibbles of the input row stride. Every mode value here is **UNVERIFIED on
//! Blackhole silicon** until `step113_unpacker_modes`' silicon arms have run;
//! the Wormhole `UNPACR_NOP` mode-7 precedent rebooted the host, so only the
//! values the pages name are expressible and none is a raw integer.
//!
//! Deliberately absent: `Upsample_rate`, `Upsample_and_interleave` and the
//! `ColShift` use of `Shift_amount` (`UnsupportedFunctionality`: "no known
//! usage"), unpacker 1 transpose (silicon ignores it, "not architecturally
//! guaranteed"), and any broadcast: the unpacker has none.

use crate::backend::{ConfigWords, EncodeError};
use crate::cfg::generated::thcon;

/// The largest input row stride: three nibbles of sixteen-byte units.
pub const MAX_ROW_STRIDE_BYTES: u32 = 65_520;

/// Datums read from one input row before the stride is applied on Blackhole
/// (`UnpackRowWidth` for datums wider than a byte).
pub const BLACKHOLE_ROW_DATUMS: u32 = 32;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum UnpackModeError {
    /// The row stride is not a multiple of sixteen bytes (the three fields
    /// count sixteen-byte units).
    StrideNotAligned {
        bytes: u32,
    },
    /// More than [`MAX_ROW_STRIDE_BYTES`], or zero.
    StrideOutOfRange {
        bytes: u32,
    },
    /// Tileize on a block-float format narrower than a byte is `UndefinedBehavior`
    /// (BFP2/BFP4 addressing); the caller names the datum width in bits.
    DatumTooNarrow {
        bits: u32,
    },
    /// Transpose together with `UnpackToDst` is `UndefinedBehavior`.
    TransposeToDst,
    Encode(EncodeError),
}

impl From<EncodeError> for UnpackModeError {
    fn from(e: EncodeError) -> Self {
        UnpackModeError::Encode(e)
    }
}

/// Stage tileize mode on unpacker 0 with `row_stride_bytes` between the starts
/// of consecutive input rows of [`BLACKHOLE_ROW_DATUMS`] datums, for a datum of
/// `datum_bits` bits. Must be combined with an uncompressed descriptor and no
/// upsampling (this crate never stages either).
///
/// `Shift_amount_cntx3` is left zero: tileize reads only the first three, and
/// `ColShift` is zero in tileize mode.
pub fn stage_tileize(
    words: &mut ConfigWords,
    row_stride_bytes: u32,
    datum_bits: u32,
) -> Result<(), UnpackModeError> {
    if datum_bits < 8 {
        return Err(UnpackModeError::DatumTooNarrow { bits: datum_bits });
    }
    if row_stride_bytes == 0 || row_stride_bytes > MAX_ROW_STRIDE_BYTES {
        return Err(UnpackModeError::StrideOutOfRange {
            bytes: row_stride_bytes,
        });
    }
    if row_stride_bytes % 16 != 0 {
        return Err(UnpackModeError::StrideNotAligned {
            bytes: row_stride_bytes,
        });
    }
    let units = row_stride_bytes / 16;
    words
        .set(thcon::THCON_SEC0_REG2_Tileize_mode, 1)?
        .set(thcon::THCON_SEC0_REG2_Shift_amount_cntx0, units & 0xf)?
        .set(
            thcon::THCON_SEC0_REG2_Shift_amount_cntx1,
            (units >> 4) & 0xf,
        )?
        .set(
            thcon::THCON_SEC0_REG2_Shift_amount_cntx2,
            (units >> 8) & 0xf,
        )?
        .set(thcon::THCON_SEC0_REG2_Shift_amount_cntx3, 0)?;
    Ok(())
}

/// Stage transpose (`Haloize_mode`) on unpacker 0 for a `Src` unpack: each
/// 16x16 block of `SrcA` rows is written with the row's low four bits and the
/// column swapped. `unpack_to_dst` is refused: the page makes it
/// `UndefinedBehavior` (and ttsim refuses it by name). The input start must be
/// 16-byte aligned, which every tile base is.
pub fn stage_transpose(
    words: &mut ConfigWords,
    unpack_to_dst: bool,
) -> Result<(), UnpackModeError> {
    if unpack_to_dst {
        return Err(UnpackModeError::TransposeToDst);
    }
    words.set(thcon::THCON_SEC0_REG2_Haloize_mode, 1)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(w: &ConfigWords) -> u32 {
        w.words()
            .iter()
            .find(|&&(a, _)| a == thcon::THCON_SEC0_REG2_Tileize_mode.addr32())
            .unwrap()
            .1
    }

    #[test]
    fn stride_is_three_nibbles_of_sixteen_bytes() {
        let mut w = ConfigWords::new();
        // 0x1230 bytes = 0x123 sixteen-byte units.
        stage_tileize(&mut w, 0x1230, 32).unwrap();
        let v = word(&w);
        assert_eq!(thcon::THCON_SEC0_REG2_Tileize_mode.extract(v), 1);
        assert_eq!(thcon::THCON_SEC0_REG2_Shift_amount_cntx0.extract(v), 0x3);
        assert_eq!(thcon::THCON_SEC0_REG2_Shift_amount_cntx1.extract(v), 0x2);
        assert_eq!(thcon::THCON_SEC0_REG2_Shift_amount_cntx2.extract(v), 0x1);
    }

    #[test]
    fn the_page_formula_reads_back_the_stride() {
        for stride in [16, 128, 256, 4096, 65_520] {
            let mut w = ConfigWords::new();
            stage_tileize(&mut w, stride, 32).unwrap();
            let v = word(&w);
            let page = (thcon::THCON_SEC0_REG2_Shift_amount_cntx0.extract(v) << 4)
                | (thcon::THCON_SEC0_REG2_Shift_amount_cntx1.extract(v) << 8)
                | (thcon::THCON_SEC0_REG2_Shift_amount_cntx2.extract(v) << 12);
            assert_eq!(page, stride);
        }
    }

    #[test]
    fn unsafe_modes_are_typed_errors() {
        let mut w = ConfigWords::new();
        assert_eq!(
            stage_tileize(&mut w, 100, 32),
            Err(UnpackModeError::StrideNotAligned { bytes: 100 })
        );
        assert_eq!(
            stage_tileize(&mut w, 65_536, 32),
            Err(UnpackModeError::StrideOutOfRange { bytes: 65_536 })
        );
        assert_eq!(
            stage_tileize(&mut w, 0, 32),
            Err(UnpackModeError::StrideOutOfRange { bytes: 0 })
        );
        assert_eq!(
            stage_tileize(&mut w, 64, 4),
            Err(UnpackModeError::DatumTooNarrow { bits: 4 })
        );
        assert_eq!(
            stage_transpose(&mut w, true),
            Err(UnpackModeError::TransposeToDst)
        );
        assert!(stage_transpose(&mut w, false).is_ok());
    }
}
