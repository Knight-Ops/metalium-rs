//! Checked packer configuration for the optional pipeline stages that act on
//! datums read from `Dst`: ReLU and edge masking.
//!
//! Source: `WormholeB0/TensixTile/TensixCoprocessor/Packers/{ReLU,EdgeMasking}.md`.
//! Blackhole's own `PACR.md` is "basic" and says nothing about either stage, so
//! everything here is **UNVERIFIED on Blackhole** until a silicon gate has run
//! (`step112_packer_relu_edge`'s silicon arms). The configuration *fields* are
//! Blackhole's (`cfg_defines.h`, `cfg::generated`); only their behaviour is
//! borrowed. Two layout differences from the Wormhole page are worth knowing:
//! `STACC_RELU_ApplyRelu` is four bits wide on Blackhole where the page's model
//! reads `& 3`, so this API only ever writes 0..=3, and `PCK_EDGE_OFFSET_SEC*`
//! share words 24..27 with the mode and row-set-select fields, so edge masking
//! is staged whole, never one field at a time.
//!
//! # Shared words
//!
//! `Config` word 2 holds `STACC_RELU_*` **and** `ALU_ACC_CTRL_Zero_Flag_disabled_*`
//! and the `DISABLE_RISC_BP_*` bits; a `WRCFG` overwrites all 32. So
//! [`PackerRelu::apply`] takes the zero-flag setting explicitly rather than
//! leaving it to whatever the staged word held, and leaves the breakpoint
//! bits at the zero `backend::reset_config` establishes.

use crate::backend::{ConfigWords, EncodeError};
use crate::cfg::generated::{alu, pack0};

/// The format the threshold's sixteen bits are read in
/// (`ReLU.md`: FP16 for FP16/FP8/BFP8a/4a/2a `Dst` formats, BF16 otherwise --
/// and BF16 when `Dst` is FP32, which is the only intermediate format this
/// crate's datapath configures).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ThresholdFormat {
    Bf16,
    Fp16,
}

impl ThresholdFormat {
    fn is_nan(self, bits: u16) -> bool {
        match self {
            ThresholdFormat::Bf16 => (bits & 0x7f80) == 0x7f80 && (bits & 0x007f) != 0,
            ThresholdFormat::Fp16 => (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0,
        }
    }
}

/// Why a mode could not be built.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PackModeError {
    /// `ReLU.md`: a threshold with the sign bit set (`Threshold <= -0`) is
    /// `UndefinedBehavior` for the two threshold modes.
    NegativeThreshold {
        bits: u16,
    },
    /// A NaN threshold makes every comparison false; the page does not define
    /// it as a mode, so it is refused rather than guessed.
    NanThreshold {
        bits: u16,
    },
    /// A column mask set index outside 0..4, a row-set index outside 0..4, or a
    /// mapping entry outside 0..4 (two-bit fields).
    SetOutOfRange {
        what: &'static str,
        value: u32,
    },
    Encode(EncodeError),
}

impl From<EncodeError> for PackModeError {
    fn from(e: EncodeError) -> Self {
        PackModeError::Encode(e)
    }
}

/// `STACC_RELU_ApplyRelu & 3` (`ReLU.md`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReluKind {
    /// `NO_RELU`: pass through.
    Off = 0,
    /// `ZERO_RELU`: `x <= 0 ? 0 : x`.
    Zero = 1,
    /// `MIN_THRESHOLD_RELU`: `x <= Threshold ? 0 : x`.
    MinThreshold = 2,
    /// `MAX_THRESHOLD_RELU`: `x <= 0 ? 0 : x > Threshold ? Threshold : x`.
    MaxThreshold = 3,
}

/// A validated packer ReLU stage.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PackerRelu {
    kind: ReluKind,
    threshold: u16,
}

impl PackerRelu {
    pub const OFF: Self = Self {
        kind: ReluKind::Off,
        threshold: 0,
    };
    pub const ZERO: Self = Self {
        kind: ReluKind::Zero,
        threshold: 0,
    };

    fn thresholded(
        kind: ReluKind,
        threshold: u16,
        format: ThresholdFormat,
    ) -> Result<Self, PackModeError> {
        if threshold & 0x8000 != 0 {
            return Err(PackModeError::NegativeThreshold { bits: threshold });
        }
        if format.is_nan(threshold) {
            return Err(PackModeError::NanThreshold { bits: threshold });
        }
        Ok(Self { kind, threshold })
    }

    /// `x <= threshold ? 0 : x`. `threshold` is the raw 16-bit pattern in `format`.
    pub fn min_threshold(threshold: u16, format: ThresholdFormat) -> Result<Self, PackModeError> {
        Self::thresholded(ReluKind::MinThreshold, threshold, format)
    }

    /// Clamp to `[0, threshold]`, with a non-positive input replaced by +0.
    pub fn max_threshold(threshold: u16, format: ThresholdFormat) -> Result<Self, PackModeError> {
        Self::thresholded(ReluKind::MaxThreshold, threshold, format)
    }

    pub const fn kind(self) -> ReluKind {
        self.kind
    }

    pub const fn threshold_bits(self) -> u16 {
        self.threshold
    }

    /// Stage `STACC_RELU_*` and the other fields of its `Config` word.
    ///
    /// `zero_flag_disabled_src` is `ALU_ACC_CTRL_Zero_Flag_disabled_src`, which
    /// shares the word; `..._dst` is staged zero (ttsim refuses non-zero).
    pub fn apply(
        self,
        words: &mut ConfigWords,
        zero_flag_disabled_src: bool,
    ) -> Result<(), PackModeError> {
        words
            .set(
                alu::ALU_ACC_CTRL_Zero_Flag_disabled_src,
                u32::from(zero_flag_disabled_src),
            )?
            .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_dst, 0)?
            .set(alu::STACC_RELU_ApplyRelu, self.kind as u32)?
            .set(alu::STACC_RELU_ReluThreshold, u32::from(self.threshold))?;
        Ok(())
    }
}

/// What a masked-out datum becomes (`PCK_EDGE_MODE_mode`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EdgeFill {
    Zero = 0,
    NegativeInfinity = 1,
}

/// Rows in one `xy` plane of the packer's tile position
/// (`PACK_COUNTERS_SEC0_pack_reads_per_xy_plane`): the tile row `Y` the edge
/// lookup sees is `row % plane_rows`. `Four` is what `datapath::pack_config`
/// always configured; `Sixteen` is a face of a 32x32 tile, one mapping entry per
/// face row.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PlaneRows {
    Four = 4,
    Sixteen = 16,
}

/// The row-set path of packer edge masking (`EdgeMasking.md` with
/// `PCK_EDGE_TILE_FACE_SET_SELECT_enable == 0`).
///
/// **Blackhole has one packer with four `Dst` read interfaces** (`PACR.md`), not
/// Wormhole's four packers: the page's `PackerIndex` is always 0, so the row set
/// is one field (`row_set`, `PCK_EDGE_TILE_ROW_SET_SELECT_select[1:0]`; the other
/// three two-bit fields select for packers that do not exist) and the tile row
/// `Y` of the `k`'th enabled read interface is `TilePosition + k`, wrapping at
/// the plane. A datum's mask is
/// `column_masks[row_set_mapping[row_set][row % plane_rows]]`; bit `c` of the
/// mask keeps column `c`, a clear bit replaces the datum with [`EdgeFill`]. The
/// plane (`row / plane_rows`, the `Z` of the face-set path) does not enter.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct EdgeMasking {
    pub fill: EdgeFill,
    pub column_masks: [u16; 4],
    pub row_set: u8,
    pub row_set_mapping: [[u8; 16]; 4],
    pub plane_rows: PlaneRows,
}

/// The state `datapath::pack_config` has always written: every datum kept.
impl EdgeMasking {
    pub const OFF: Self = Self {
        fill: EdgeFill::Zero,
        column_masks: [0xffff; 4],
        row_set: 0,
        row_set_mapping: [[0; 16]; 4],
        plane_rows: PlaneRows::Four,
    };

    /// Whether the datum at `Dst` tile row `row` (0..64), column `column` is kept:
    /// the specification's lookup.
    pub fn keeps(&self, row: usize, column: usize) -> bool {
        let y = row % self.plane_rows as usize;
        let c = self.row_set_mapping[self.row_set as usize][y] as usize;
        self.column_masks[c] >> column & 1 == 1
    }

    fn check(&self) -> Result<(), PackModeError> {
        if self.row_set >= 4 {
            return Err(PackModeError::SetOutOfRange {
                what: "row set",
                value: u32::from(self.row_set),
            });
        }
        for map in &self.row_set_mapping {
            for &c in map {
                if c >= 4 {
                    return Err(PackModeError::SetOutOfRange {
                        what: "mask set",
                        value: u32::from(c),
                    });
                }
            }
        }
        Ok(())
    }

    /// Which mask sets the selected mapping's reachable entries name, as a bit set.
    fn referenced_masks(&self) -> u8 {
        self.row_set_mapping[self.row_set as usize][..self.plane_rows as usize]
            .iter()
            .fold(0, |acc, &c| acc | 1 << c)
    }

    /// Stage the plane size, fill mode, row-set select, the selected row set's
    /// mapping word and the mask words its reachable entries name.
    ///
    /// Words it cannot reach are left alone, which is exact when they hold
    /// `Config`'s reset zero (`backend::reset_config`) and is what ttsim needs:
    /// it models only words 20, 21, 24 and 25 of these (divergence row 28), and
    /// refuses a write to 19, 22, 23, 26 or 27 at any value. Use
    /// [`Self::apply_all`] on silicon to leave nothing to a previous program.
    pub fn apply(&self, words: &mut ConfigWords) -> Result<(), PackModeError> {
        self.apply_words(words, false)
    }

    /// [`Self::apply`] writing every word of the stage, including
    /// `PCK_EDGE_TILE_FACE_SET_SELECT_*` (word 19, disabled: the row-set path)
    /// and all four mapping and mask words. ttsim refuses this (silicon only).
    pub fn apply_all(&self, words: &mut ConfigWords) -> Result<(), PackModeError> {
        self.apply_words(words, true)
    }

    fn apply_words(&self, words: &mut ConfigWords, all: bool) -> Result<(), PackModeError> {
        self.check()?;
        let maps = [
            pack0::TILE_ROW_SET_MAPPING_0_row_set_mapping_0,
            pack0::TILE_ROW_SET_MAPPING_1_row_set_mapping_0,
            pack0::TILE_ROW_SET_MAPPING_2_row_set_mapping_0,
            pack0::TILE_ROW_SET_MAPPING_3_row_set_mapping_0,
        ];
        let masks = [
            pack0::PCK_EDGE_OFFSET_SEC0_mask,
            pack0::PCK_EDGE_OFFSET_SEC1_mask,
            pack0::PCK_EDGE_OFFSET_SEC2_mask,
            pack0::PCK_EDGE_OFFSET_SEC3_mask,
        ];
        if all {
            words
                .set(pack0::PCK_EDGE_TILE_FACE_SET_SELECT_enable, 0)?
                .set(pack0::PCK_EDGE_TILE_FACE_SET_SELECT_select, 0)?;
        }
        words
            .set(
                pack0::PACK_COUNTERS_SEC0_pack_reads_per_xy_plane,
                self.plane_rows as u32,
            )?
            .set(pack0::PCK_EDGE_MODE_mode, self.fill as u32)?
            // The other three two-bit selects name packers Blackhole lacks.
            .set(
                pack0::PCK_EDGE_TILE_ROW_SET_SELECT_select,
                u32::from(self.row_set),
            )?;
        let wanted = if all { 0b1111 } else { self.referenced_masks() };
        for (c, (field, &mask)) in masks.iter().zip(&self.column_masks).enumerate() {
            // Mask 0 is always staged: it shares word 24 with the fill mode.
            if wanted >> c & 1 == 1 || c == 0 {
                words.set(*field, u32::from(mask))?;
            }
        }
        for (b, (map, anchor)) in self.row_set_mapping.iter().zip(maps).enumerate() {
            if all || b == self.row_set as usize {
                let packed = map
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (k, &c)| acc | u32::from(c) << (2 * k));
                // All sixteen two-bit entries are one word (`TILE_ROW_SET_MAPPING_b`).
                words.seed(anchor.addr32(), packed)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_threshold_is_checked() {
        assert_eq!(
            PackerRelu::min_threshold(0x8000, ThresholdFormat::Bf16),
            Err(PackModeError::NegativeThreshold { bits: 0x8000 })
        );
        assert_eq!(
            PackerRelu::max_threshold(0x7fc0, ThresholdFormat::Bf16),
            Err(PackModeError::NanThreshold { bits: 0x7fc0 })
        );
        assert_eq!(
            PackerRelu::max_threshold(0x7e01, ThresholdFormat::Fp16),
            Err(PackModeError::NanThreshold { bits: 0x7e01 })
        );
        // +inf is not NaN in either format.
        assert!(PackerRelu::min_threshold(0x7f80, ThresholdFormat::Bf16).is_ok());
        assert!(PackerRelu::min_threshold(0x7c00, ThresholdFormat::Fp16).is_ok());
    }

    #[test]
    fn relu_word_places_fields_and_the_shared_zero_flag() {
        let mut w = ConfigWords::new();
        PackerRelu::min_threshold(0x3f80, ThresholdFormat::Bf16)
            .unwrap()
            .apply(&mut w, true)
            .unwrap();
        let (addr, value) = w.words()[0];
        assert_eq!(addr, 2);
        assert_eq!(value, 1 | (2 << 2) | (0x3f80 << 6));
    }

    #[test]
    fn edge_lookup_follows_the_page() {
        let mut e = EdgeMasking::OFF;
        e.column_masks = [0xffff, 0x00ff, 0, 0xaaaa];
        e.row_set = 1;
        e.row_set_mapping[1] = [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        // Four-row planes: Y = row % 4 picks the mask set.
        assert!(e.keeps(0, 15));
        assert!(e.keeps(1, 7) && !e.keeps(1, 8));
        assert!(!e.keeps(2, 0));
        assert!(e.keeps(3, 1) && !e.keeps(3, 0));
        // Planes repeat: Z does not enter on the row-set path.
        assert_eq!(e.keeps(4 + 1, 8), e.keeps(1, 8));
        // Sixteen-row planes: the whole mapping is reachable, one entry a face row.
        e.plane_rows = PlaneRows::Sixteen;
        e.row_set_mapping[1][5] = 2;
        assert!(!e.keeps(5, 0) && !e.keeps(16 + 5, 0) && e.keeps(4, 0));
    }

    #[test]
    fn edge_words_are_minimal_unless_all_are_asked_for() {
        let mut e = EdgeMasking::OFF;
        e.row_set = 1;
        e.column_masks = [0xffff, 0, 0, 0];
        e.row_set_mapping[1][0] = 1;
        let mut w = ConfigWords::new();
        e.apply(&mut w).unwrap();
        let has = |w: &ConfigWords, a: u16| w.words().iter().any(|&(x, _)| x == a);
        // Mapping 1, masks 0 and 1; nothing ttsim refuses.
        assert!(has(&w, 21) && has(&w, 24) && has(&w, 25));
        assert!(!has(&w, 19) && !has(&w, 22) && !has(&w, 26));
        let mut all = ConfigWords::new();
        e.apply_all(&mut all).unwrap();
        for a in [19, 20, 21, 22, 23, 24, 25, 26, 27] {
            assert!(has(&all, a), "word {a}");
        }
    }

    #[test]
    fn edge_sets_are_range_checked() {
        let mut e = EdgeMasking::OFF;
        e.row_set_mapping[0][3] = 4;
        assert_eq!(
            e.apply(&mut ConfigWords::new()),
            Err(PackModeError::SetOutOfRange {
                what: "mask set",
                value: 4
            })
        );
    }
}
