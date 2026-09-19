//! NoC coordinate spaces.
//!
//! Blackhole has four distinct hardware coordinate spaces plus one software label,
//! all of which are a pair of small integers and none of which are interchangeable.
//! Mixing them silently addresses the wrong tile, so they are separate types rather
//! than a convention.
//!
//! Spec: `BlackholeA0/NoC/Coordinates.md`.

use core::fmt;
use core::marker::PhantomData;

/// Which of the two NoCs a raw coordinate belongs to.
///
/// The two NoCs are mirrored, not merely offset: NoC #0's origin is top-left and
/// increments rightwards and downwards (`Coordinates.md:7`), NoC #1's origin is
/// bottom-right and increments leftwards and upwards (`Coordinates.md:13`). A NoC
/// #0 coordinate reinterpreted as NoC #1 names a different tile.
pub trait NocId: Copy + Clone + fmt::Debug + Eq + PartialEq {
    /// 0 or 1, as written into the TLB `noc_sel` field.
    const INDEX: u8;
    const NAME: &'static str;
}

/// NoC #0: origin top-left, increasing rightwards and downwards.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Noc0;
impl NocId for Noc0 {
    const INDEX: u8 = 0;
    const NAME: &'static str = "NoC0";
}

/// NoC #1: origin bottom-right, increasing leftwards and upwards.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Noc1;
impl NocId for Noc1 {
    const INDEX: u8 = 1;
    const NAME: &'static str = "NoC1";
}

/// The NoC grid is 17 tiles wide and 12 tall (`NoC/MemoryMap.md:183-184`).
pub const GRID_WIDTH: u8 = 17;
/// See [`GRID_WIDTH`].
pub const GRID_HEIGHT: u8 = 12;

/// A raw NoC coordinate, tagged with which NoC it addresses.
///
/// "Raw" means untranslated — what the hardware uses when coordinate translation
/// (`NIU_CFG_0` bit 14) is off, and what the TLB `strided`/exclude fields always
/// take regardless of translation.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct NocCoord<N: NocId> {
    x: u8,
    y: u8,
    _noc: PhantomData<N>,
}

impl<N: NocId> NocCoord<N> {
    /// Both axes are 6-bit fields in every register that carries a coordinate
    /// (`NoC/MemoryMap.md:110-115`), so anything wider is a programming error
    /// rather than an out-of-range tile.
    pub const fn new(x: u8, y: u8) -> Option<Self> {
        if x < 64 && y < 64 {
            Some(NocCoord {
                x,
                y,
                _noc: PhantomData,
            })
        } else {
            None
        }
    }

    pub const fn x(self) -> u8 {
        self.x
    }

    pub const fn y(self) -> u8 {
        self.y
    }

    /// Pack as `x | (y << 6)`, the layout shared by `NOC_*_ADDR_HI` unicast
    /// coordinates (`NoC/MemoryMap.md:110-115`) and the TLB `x_end`/`y_end` pair.
    pub const fn packed(self) -> u16 {
        (self.x as u16) | ((self.y as u16) << 6)
    }
}

impl<N: NocId> fmt::Debug for NocCoord<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}, {})", N::NAME, self.x, self.y)
    }
}

/// A coordinate in the translated space used when `NIU_CFG_0` bit 14 is set.
///
/// Blackhole's translation is a *combined* X/Y table — the Y value selects which X
/// table applies (`Coordinates.md:26`) — unlike Wormhole's separable per-axis
/// tables. So a translated coordinate cannot be decomposed and re-translated
/// axis-by-axis, and it is not interchangeable with either raw space.
///
/// Note also that Blackhole, unlike Wormhole, does **not** write translated
/// coordinates back into MMIO registers (`NoC/MemoryMap.md:158-166`), so read-back
/// code must not expect to recover one.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Translated {
    x: u8,
    y: u8,
}

impl Translated {
    pub const fn new(x: u8, y: u8) -> Option<Self> {
        if x < 64 && y < 64 {
            Some(Translated { x, y })
        } else {
            None
        }
    }

    pub const fn x(self) -> u8 {
        self.x
    }

    pub const fn y(self) -> u8 {
        self.y
    }

    pub const fn packed(self) -> u16 {
        (self.x as u16) | ((self.y as u16) << 6)
    }
}

impl fmt::Debug for Translated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Translated({}, {})", self.x, self.y)
    }
}

/// Which chip a coordinate refers to.
///
/// Present from the first commit even though the baseline is single-chip: the
/// alternative is threading it through every signature later.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct ChipId(pub u16);

/// Tile type, decoded from `NOC_ENDPOINT_ID` bits 8..=23 (`NoC/MemoryMap.md:196-200`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TileType {
    Tensix,
    Ethernet,
    Pcie,
    Arc,
    Dram,
    L2Cpu,
    Security,
    Unknown(u16),
}

impl TileType {
    pub const fn from_endpoint_id(endpoint_id: u32) -> Self {
        match ((endpoint_id >> 8) & 0xFFFF) as u16 {
            0x0100 => TileType::Tensix,
            0x0200 => TileType::Ethernet,
            0x0300 => TileType::Pcie,
            0x0500 => TileType::Arc,
            0x0800 => TileType::Dram,
            0x0901 => TileType::L2Cpu,
            0x0A00 => TileType::Security,
            other => TileType::Unknown(other),
        }
    }
}

/// Tile index within its type, from `NOC_ENDPOINT_ID` bits 0..=7.
pub const fn endpoint_tile_index(endpoint_id: u32) -> u8 {
    (endpoint_id & 0xFF) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coords_reject_out_of_range() {
        assert!(NocCoord::<Noc0>::new(63, 63).is_some());
        assert!(NocCoord::<Noc0>::new(64, 0).is_none());
        assert!(NocCoord::<Noc0>::new(0, 64).is_none());
    }

    #[test]
    fn packing_matches_ethdump() {
        // ethdump.c:440 hardcodes BH_PCIE_XY as `19 + (24 << 6)` for the
        // host-connected PCIe tile in translated space.
        let pcie = Translated::new(19, 24).unwrap();
        assert_eq!(pcie.packed(), 19 + (24 << 6));
    }

    #[test]
    fn endpoint_id_decodes() {
        // Tile type 0x0100 (Tensix), tile index 0x07.
        assert_eq!(TileType::from_endpoint_id(0x0001_0007), TileType::Tensix);
        assert_eq!(endpoint_tile_index(0x0001_0007), 7);
        assert_eq!(TileType::from_endpoint_id(0x0002_0000), TileType::Ethernet);
    }
}

/// NIU register addresses, as seen from a tile's own address space.
///
/// Spec: `BlackholeA0/NoC/MemoryMap.md:5-37`. These are reachable from the host
/// through a TLB window like any other tile address.
pub mod niu {
    /// NIU base for NoC #0 in a Tensix or Ethernet tile (`MemoryMap.md:5-12`).
    pub const NOC0_BASE: u64 = 0xFFB2_0000;
    /// NIU base for NoC #1.
    pub const NOC1_BASE: u64 = 0xFFB3_0000;

    /// Identifies the tile: index in bits 0..=7, type in bits 8..=23, NoC index in
    /// bits 24..=31 (`MemoryMap.md:196-200`).
    pub const NOC_ENDPOINT_ID: u64 = 0x0048;

    /// `MemoryMap.md:228-238`. Bit 12 marks a tile fused off by harvesting; bit 14
    /// enables coordinate translation.
    pub const NIU_CFG_0: u64 = 0x0100;

    /// Bit 12 of `NIU_CFG_0`: this tile is harvested and must not be used.
    pub const NIU_CFG_0_HARVESTED: u32 = 1 << 12;
    /// Bit 14 of `NIU_CFG_0`: coordinate translation is enabled.
    pub const NIU_CFG_0_TRANSLATION_ENABLED: u32 = 1 << 14;

    /// Translated X/Y of this tile, packed `x | (y << 6)` (`MemoryMap.md:219`).
    pub const NOC_ID_LOGICAL: u64 = 0x0148;
}

/// The Blackhole NoC #0 grid: which raw coordinates hold which kind of tile.
///
/// # Provenance
///
/// `NOC_ENDPOINT_ID` is not implemented by ttsim, so the runtime probe
/// `ethdump.c:462` uses is unavailable there and the layout had to be measured.
/// The measurement (`crates/tt-tests/tests/probe_niu.rs`) finds the highest
/// addressable byte at each coordinate and gets three distinct values, each of
/// which matches an independently documented figure:
///
/// | Measured | Documented | Tile |
/// |---|---|---|
/// | `0x0018_0000` (1536 KiB) | `BabyRISCV/README.md:102` | Tensix |
/// | `0x0008_0000` (512 KiB)  | `EthernetTile/README.md` | Ethernet |
/// | `0xFF00_0000`            | DRAM channel size        | DRAM |
///
/// The resulting Tensix population is 14 columns × 10 rows = 140, which is exactly
/// the documented Tensix tile count. Three independent figures agreeing is what
/// makes this a fact rather than a guess.
///
/// It is nonetheless a *measured* fact about one simulator build, not a quoted one.
/// Re-derive it at the first silicon gate, and treat a mismatch as a finding.
pub mod grid {
    use super::{NocCoord, NocId};

    /// L1 per Tensix tile (`BabyRISCV/README.md:102`).
    pub const TENSIX_L1_SIZE: u64 = 1536 * 1024;
    /// L1 per Ethernet tile (`EthernetTile/README.md`).
    pub const ETHERNET_L1_SIZE: u64 = 512 * 1024;

    /// Columns carrying DRAM tiles rather than compute.
    pub const DRAM_COLUMNS: [u8; 2] = [0, 9];
    /// The row of Ethernet tiles. `ethdump.c:462` scans exactly this row.
    pub const ETHERNET_ROW: u8 = 1;
    /// The column that is neither compute nor DRAM, and which faults on an L1
    /// access. L2CPU and Security tiles live here.
    pub const NON_MEMORY_COLUMN: u8 = 8;
    /// Rows holding Tensix tiles, inclusive.
    pub const TENSIX_ROWS: core::ops::RangeInclusive<u8> = 2..=11;

    /// Is this coordinate a Tensix tile?
    pub const fn is_tensix(x: u8, y: u8) -> bool {
        y >= *TENSIX_ROWS.start()
            && y <= *TENSIX_ROWS.end()
            && x != NON_MEMORY_COLUMN
            && x != DRAM_COLUMNS[0]
            && x != DRAM_COLUMNS[1]
            && x < super::GRID_WIDTH
    }

    /// Every Tensix coordinate, in row-major order.
    pub fn tensix_tiles<N: NocId>() -> impl Iterator<Item = NocCoord<N>> {
        TENSIX_ROWS.flat_map(|y| {
            (0..super::GRID_WIDTH)
                .filter(move |&x| is_tensix(x, y))
                .filter_map(move |x| NocCoord::<N>::new(x, y))
        })
    }

    /// Documented number of Tensix tiles on a Blackhole.
    pub const TENSIX_TILE_COUNT: usize = 140;

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::noc::Noc0;

        #[test]
        fn tensix_population_matches_the_documented_count() {
            assert_eq!(tensix_tiles::<Noc0>().count(), TENSIX_TILE_COUNT);
        }

        #[test]
        fn the_excluded_columns_and_rows_are_excluded() {
            assert!(!is_tensix(NON_MEMORY_COLUMN, 5));
            assert!(!is_tensix(DRAM_COLUMNS[0], 5));
            assert!(!is_tensix(DRAM_COLUMNS[1], 5));
            assert!(!is_tensix(3, ETHERNET_ROW));
            assert!(!is_tensix(3, 0));
            assert!(is_tensix(3, 2));
            assert!(is_tensix(16, 11));
        }
    }
}
