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

    /// One of the NIU's four request initiators (`MemoryMap.md`, "NIU Request
    /// Initiators"), as offsets from its base, `NIU_BASE + i * STRIDE`.
    pub mod initiator {
        pub const STRIDE: u64 = 0x800;
        pub const TARG_ADDR_LO: u64 = 0x00;
        pub const TARG_ADDR_MID: u64 = 0x04;
        pub const TARG_ADDR_HI: u64 = 0x08;
        pub const RET_ADDR_LO: u64 = 0x0C;
        pub const RET_ADDR_MID: u64 = 0x10;
        pub const RET_ADDR_HI: u64 = 0x14;
        pub const PACKET_TAG: u64 = 0x18;
        pub const CTRL: u64 = 0x1C;
        pub const AT_LEN_BE: u64 = 0x20;
        pub const AT_DATA: u64 = 0x28;
        /// Write 1 to issue; hardware clears it once the request has a VC.
        /// Software must not touch the initiator while it reads 1.
        pub const CMD_CTRL: u64 = 0x40;
    }

    /// `NIU_MST_REQS_OUTSTANDING_ID(id)` (`Counters.md`): back to zero once every
    /// response-marked request with this transaction ID has completed. Read
    /// `CMD_CTRL` back first (`Counters.md:42-43`).
    pub const fn reqs_outstanding(id: TxnId) -> u64 {
        0x0200 + (16 + id.0 as u64) * 4
    }

    /// `NOC_PACKET_TRANSACTION_ID`, `0..16` (`MemoryMap.md`, `NOC_PACKET_TAG`).
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct TxnId(u8);

    impl TxnId {
        pub const fn new(id: u8) -> Option<Self> {
            if id < 16 {
                Some(TxnId(id))
            } else {
                None
            }
        }
    }

    /// Largest length one request may carry between L1 addresses. Larger ones
    /// are split by hardware, but then one request moves the 8-bit outstanding
    /// counter by more than one, so this refuses them instead.
    pub const MAX_REQUEST_BYTES: u32 = 16384;
    /// In Tensix and Ethernet tiles, L1 is below this and MMIO at or above it
    /// (WH `NoC/Alignment.md`).
    pub const MMIO_START: u32 = 0xFF00_0000;

    /// An address in a tile, with that tile's NoC #0 coordinate as the initiating
    /// NIU would name it.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct Endpoint {
        pub x: u8,
        pub y: u8,
        pub addr: u32,
    }

    impl Endpoint {
        const fn hi(self) -> u32 {
            (self.x as u32 & 0x3F) | ((self.y as u32 & 0x3F) << 6)
        }
    }

    /// A request, in the only shapes this workspace issues.
    ///
    /// There is no field for `NOC_CMD_L1_ACC_AT_EN` (`MemoryMap.md`: unusable,
    /// must be `false`), for broadcast, or for linked VCs, so none of them can be
    /// set by accident. Writes are always response-marked, so completion is
    /// observable on [`reqs_outstanding`].
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum Command {
        /// Read `len` bytes of `from` into this tile's L1 at `to_local`.
        Read {
            from: Endpoint,
            to_local: u32,
            len: u32,
        },
        /// Write `len` bytes of this tile's L1 at `from_local` to `to`.
        Write {
            from_local: u32,
            to: Endpoint,
            len: u32,
        },
        /// Store `data` to an MMIO register. `NOC_CMD_WR_INLINE` to an L1 address
        /// is unsafe on Blackhole (`MemoryMap.md`, `NOC_CTRL` bit 3), so the
        /// destination must be MMIO.
        MmioInline { to: Endpoint, data: u32 },
        /// Read all of `from`, a range of a usable GDDR channel, into this
        /// tile's L1 at `to_local`, through the channel's endpoint `port`.
        ///
        /// DRAM is an "other" address, so a read into L1 needs the two addresses
        /// congruent mod [`crate::dram::ALIGN`] (64), not 16. The offset fits
        /// the low address word: [`crate::dram::CHANNEL_BYTES`] is below 4 GiB.
        ReadDram {
            from: crate::dram::DramRange,
            port: u8,
            to_local: u32,
        },
        /// Write `len` bytes of this tile's L1 at `from_local` over all of `to`
        /// (C16, WH `NoC/Alignment.md:32`).
        WriteDram {
            from_local: u32,
            to: crate::dram::DramRange,
            port: u8,
        },
    }

    /// Why a [`Command`] was refused.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum RequestError {
        /// An L1 copy whose two addresses are not congruent mod 16, or not both L1
        /// (WH `NoC/Alignment.md`: `UndefinedBehavior`).
        Alignment,
        /// Zero, or more than [`MAX_REQUEST_BYTES`].
        Length,
        /// An inline write aimed at L1, or at an unaligned MMIO address.
        InlineToL1,
        /// A DRAM endpoint port beyond the channel's three.
        Port,
    }

    const CMD_WR: u32 = 2;
    const CMD_RD: u32 = 0;
    const WR_INLINE: u32 = 1 << 3;
    const RESP_MARKED: u32 = 1 << 4;

    impl Command {
        /// The initiator registers to write, in order, before `CMD_CTRL`.
        /// `me` is the initiating tile's coordinate, which reads name as their
        /// return address and writes as the source of their data.
        pub fn registers(
            &self,
            me: (u8, u8),
            txn: TxnId,
        ) -> Result<[(u64, u32); 10], RequestError> {
            use initiator::*;
            let local = |addr: u32| Endpoint {
                x: me.0,
                y: me.1,
                addr,
            };
            let (targ, ret, ctrl, len_be, data) = match *self {
                Command::Read {
                    from,
                    to_local,
                    len,
                } => {
                    check_copy(from.addr, to_local, len)?;
                    // `MemoryMap.md` says `RESP_MARKED` is ignored for reads
                    // (they always respond); ttsim refuses a read without it
                    // (divergence row 61), so it is set.
                    (from, local(to_local), CMD_RD | RESP_MARKED, len, 0)
                }
                Command::Write {
                    from_local,
                    to,
                    len,
                } => {
                    check_copy(from_local, to.addr, len)?;
                    (local(from_local), to, CMD_WR | RESP_MARKED, len, 0)
                }
                Command::MmioInline { to, data } => {
                    if to.addr < MMIO_START || to.addr % 4 != 0 {
                        return Err(RequestError::InlineToL1);
                    }
                    (to, local(0), CMD_WR | WR_INLINE | RESP_MARKED, 0, data)
                }
                Command::ReadDram {
                    from,
                    port,
                    to_local,
                } => {
                    let (from, len) = dram_endpoint(from, port)?;
                    check_dram(from.addr, to_local, len, crate::dram::ALIGN as u32)?;
                    (from, local(to_local), CMD_RD | RESP_MARKED, len, 0)
                }
                Command::WriteDram {
                    from_local,
                    to,
                    port,
                } => {
                    let (to, len) = dram_endpoint(to, port)?;
                    check_dram(from_local, to.addr, len, 16)?;
                    (local(from_local), to, CMD_WR | RESP_MARKED, len, 0)
                }
            };
            Ok([
                (TARG_ADDR_LO, targ.addr),
                (TARG_ADDR_MID, 0),
                (TARG_ADDR_HI, targ.hi()),
                (RET_ADDR_LO, ret.addr),
                (RET_ADDR_MID, 0),
                (RET_ADDR_HI, ret.hi()),
                (PACKET_TAG, (txn.0 as u32) << 10),
                (CTRL, ctrl),
                (AT_LEN_BE, len_be),
                (AT_DATA, data),
            ])
        }
    }

    /// The endpoint `port` of a DRAM range's channel, and the range's length.
    fn dram_endpoint(r: crate::dram::DramRange, port: u8) -> Result<(Endpoint, u32), RequestError> {
        let at = r.channel().endpoint(port).ok_or(RequestError::Port)?;
        let len = u32::try_from(r.len()).map_err(|_| RequestError::Length)?;
        // `CHANNEL_BYTES` is below 4 GiB, so the offset fits the low word.
        let e = Endpoint {
            x: at.x(),
            y: at.y(),
            addr: r.offset() as u32,
        };
        Ok((e, len))
    }

    /// A GDDR <-> L1 copy: the L1 side must be L1, the two congruent mod
    /// `modulus`, and the length one request.
    fn check_dram(src: u32, dst: u32, len: u32, modulus: u32) -> Result<(), RequestError> {
        if len == 0 || len > MAX_REQUEST_BYTES {
            return Err(RequestError::Length);
        }
        if src % modulus != dst % modulus || src >= MMIO_START || dst >= MMIO_START {
            return Err(RequestError::Alignment);
        }
        Ok(())
    }

    fn check_copy(src: u32, dst: u32, len: u32) -> Result<(), RequestError> {
        if len == 0 || len > MAX_REQUEST_BYTES {
            return Err(RequestError::Length);
        }
        if src >= MMIO_START || dst >= MMIO_START || src % 16 != dst % 16 {
            return Err(RequestError::Alignment);
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const T: TxnId = match TxnId::new(3) {
            Some(t) => t,
            None => panic!(),
        };
        fn at(x: u8, y: u8, addr: u32) -> Endpoint {
            Endpoint { x, y, addr }
        }
        fn reg(r: &[(u64, u32); 10], off: u64) -> u32 {
            r.iter().find(|(o, _)| *o == off).unwrap().1
        }

        #[test]
        fn a_write_names_the_local_source_as_target() {
            // MemoryMap.md:99-104: for a non-inline write NOC_TARG_ADDR is the
            // *source* and NOC_RET_ADDR the destination -- the reverse of a read.
            let w = Command::Write {
                from_local: 0x2_0000,
                to: at(4, 5, 0x3_0000),
                len: 64,
            };
            let r = w.registers((3, 1), T).unwrap();
            assert_eq!(reg(&r, initiator::TARG_ADDR_LO), 0x2_0000);
            assert_eq!(reg(&r, initiator::TARG_ADDR_HI), 3 | (1 << 6));
            assert_eq!(reg(&r, initiator::RET_ADDR_LO), 0x3_0000);
            assert_eq!(reg(&r, initiator::RET_ADDR_HI), 4 | (5 << 6));
            assert_eq!(reg(&r, initiator::CTRL), CMD_WR | RESP_MARKED);
            assert_eq!(reg(&r, initiator::PACKET_TAG), 3 << 10);
        }

        #[test]
        fn a_read_returns_to_the_initiator() {
            let rd = Command::Read {
                from: at(4, 5, 0x3_0010),
                to_local: 0x2_0010,
                len: 16,
            };
            let r = rd.registers((3, 1), T).unwrap();
            assert_eq!(reg(&r, initiator::TARG_ADDR_HI), 4 | (5 << 6));
            assert_eq!(reg(&r, initiator::RET_ADDR_HI), 3 | (1 << 6));
            assert_eq!(reg(&r, initiator::CTRL), CMD_RD | RESP_MARKED);
        }

        #[test]
        fn a_dram_read_targets_the_translated_endpoint_and_needs_c64() {
            let ch = crate::dram::Dram::FULL.channel(5).unwrap();
            let rd = |off: u64, to: u32| Command::ReadDram {
                from: ch.range(off, 2048).unwrap(),
                port: 1,
                to_local: to,
            };
            let r = rd(0x10_0060, 0x3_0020).registers((3, 4), T).unwrap();
            // Channel 5, port 1: translated (18, 12 + 3 + 1).
            assert_eq!(reg(&r, initiator::TARG_ADDR_HI), 18 | (16 << 6));
            assert_eq!(reg(&r, initiator::TARG_ADDR_LO), 0x10_0060);
            assert_eq!(reg(&r, initiator::TARG_ADDR_MID), 0);
            assert_eq!(reg(&r, initiator::RET_ADDR_LO), 0x3_0020);
            assert_eq!(reg(&r, initiator::AT_LEN_BE), 2048);
            assert_eq!(reg(&r, initiator::CTRL), CMD_RD | RESP_MARKED);
            // Congruent mod 16, or mod 32, but not mod 64: refused (row 64).
            for off in [0x10_0010, 0x10_0040] {
                assert_eq!(
                    rd(off, 0x3_0020).registers((3, 4), T),
                    Err(RequestError::Alignment)
                );
            }
        }

        #[test]
        fn a_dram_write_needs_c16_a_port_and_one_request() {
            let ch = crate::dram::Dram::FULL.channel(0).unwrap();
            let wr = |len: u64, port: u8, from: u32| Command::WriteDram {
                from_local: from,
                to: ch.range(0x40, len).unwrap(),
                port,
            };
            let r = wr(64, 0, 0x2_0000).registers((3, 4), T).unwrap();
            assert_eq!(reg(&r, initiator::TARG_ADDR_LO), 0x2_0000);
            assert_eq!(reg(&r, initiator::RET_ADDR_HI), 17 | (12 << 6));
            assert_eq!(reg(&r, initiator::RET_ADDR_LO), 0x40);
            assert_eq!(
                wr(64, 0, 0x2_0008).registers((3, 4), T),
                Err(RequestError::Alignment)
            );
            assert_eq!(
                wr(64, 3, 0x2_0000).registers((3, 4), T),
                Err(RequestError::Port)
            );
            assert_eq!(
                wr(MAX_REQUEST_BYTES as u64 + 16, 0, 0x2_0000).registers((3, 4), T),
                Err(RequestError::Length)
            );
            assert_eq!(
                wr(0, 0, 0x2_0000).registers((3, 4), T),
                Err(RequestError::Length)
            );
        }

        #[test]
        fn no_request_sets_l1_accumulate() {
            let cmds = [
                Command::Write {
                    from_local: 0,
                    to: at(1, 2, 0),
                    len: 16,
                },
                Command::Read {
                    from: at(1, 2, 0),
                    to_local: 0,
                    len: 16,
                },
                Command::MmioInline {
                    to: at(1, 2, 0xFFB2_0100),
                    data: 1,
                },
            ];
            for c in cmds {
                assert_eq!(
                    reg(&c.registers((0, 0), T).unwrap(), initiator::CTRL) & (1 << 31),
                    0
                );
            }
        }

        #[test]
        fn hazards_are_refused() {
            let inline_l1 = Command::MmioInline {
                to: at(1, 2, 0x1000),
                data: 1,
            };
            assert_eq!(
                inline_l1.registers((0, 0), T),
                Err(RequestError::InlineToL1)
            );
            let skew = Command::Write {
                from_local: 0x10,
                to: at(1, 2, 0x18),
                len: 16,
            };
            assert_eq!(skew.registers((0, 0), T), Err(RequestError::Alignment));
            let big = Command::Read {
                from: at(1, 2, 0),
                to_local: 0,
                len: 16385,
            };
            assert_eq!(big.registers((0, 0), T), Err(RequestError::Length));
            let mmio_copy = Command::Write {
                from_local: 0,
                to: at(1, 2, 0xFFB0_0000),
                len: 16,
            };
            assert_eq!(mmio_copy.registers((0, 0), T), Err(RequestError::Alignment));
            assert!(TxnId::new(16).is_none());
        }
    }
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
/// It is nonetheless a *measured* fact about one simulator build, not a quoted one,
/// and the first silicon gate found the mismatch the warning that used to sit here
/// predicted: ttsim models an *unharvested* chip, so 140 is the population of a
/// full Blackhole and not of any particular one. Tenstorrent fuses off Tensix
/// columns for yield, so a p150a commonly presents 120. Which columns are gone
/// varies per ASIC and cannot be measured by probing, because probing a fused-off
/// tile is the hang described in [`crate::arc`].
///
/// The geometry below is therefore split in two. [`is_tensix_geometry`] answers
/// "could this coordinate ever hold a Tensix tile", which is a property of the
/// part. [`Tensix`] answers "does *this* chip have a Tensix tile there", which is
/// a property of one ASIC and has to come from its ARC.
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

    /// Every column that can hold Tensix tiles, in ascending X.
    ///
    /// Ascending X is not an arbitrary ordering: `NoC/Coordinates.md:54` says that
    /// when columns are fused off, "`X` is remapped to put the fused columns at
    /// maximal `X`". In translated space the surviving columns are therefore a
    /// prefix of this array and the harvested ones are a suffix, whichever
    /// physical columns were actually lost. That is what lets [`Tensix`] be built
    /// from a *count* and spares us having to guess the bit order of a mask.
    pub const TENSIX_COLUMNS: [u8; 14] = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16];

    /// Rows of Tensix tiles. Harvesting on Blackhole is by column, not row
    /// (`NoC/Coordinates.md:54`), unlike Wormhole where it is by row.
    pub const TENSIX_ROW_COUNT: usize = 10;

    /// Tensix tiles on a Blackhole with nothing fused off.
    ///
    /// The population of the *part*, not of any given chip. A real chip has
    /// [`Tensix::tile_count`] of them.
    pub const FULL_TENSIX_TILE_COUNT: usize = TENSIX_COLUMNS.len() * TENSIX_ROW_COUNT;

    /// Could a Tensix tile ever live at this coordinate?
    ///
    /// Geometry alone. A true answer does not mean the tile is present on the chip
    /// in front of you — for that, ask [`Tensix::contains`]. The distinction is
    /// deliberately awkward to ignore: the name of the thing that *looks* like the
    /// obvious predicate is the one that cannot get you a hung NoC, and the one
    /// that can requires a value you could only have obtained from the ARC.
    pub const fn is_tensix_geometry(x: u8, y: u8) -> bool {
        y >= *TENSIX_ROWS.start()
            && y <= *TENSIX_ROWS.end()
            && x != NON_MEMORY_COLUMN
            && x != DRAM_COLUMNS[0]
            && x != DRAM_COLUMNS[1]
            && x < super::GRID_WIDTH
    }

    /// Which Tensix tiles one particular chip actually has.
    ///
    /// Obtained from the chip's ARC telemetry, never assumed. Construct it with
    /// [`Tensix::from_enabled_column_mask`] on silicon; [`Tensix::FULL`] is correct
    /// for the simulator, which models an unharvested chip.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct Tensix {
        /// How many of [`TENSIX_COLUMNS`] survive, counting from the front.
        enabled_columns: u8,
    }

    impl Tensix {
        /// Every column present: a full Blackhole, and what ttsim models.
        pub const FULL: Tensix = Tensix {
            enabled_columns: TENSIX_COLUMNS.len() as u8,
        };

        /// The first `n` of [`TENSIX_COLUMNS`], or `None` if `n` exceeds the part.
        pub const fn from_enabled_column_count(n: u8) -> Option<Tensix> {
            if n as usize > TENSIX_COLUMNS.len() {
                None
            } else {
                Some(Tensix { enabled_columns: n })
            }
        }
        /// Interpret an `ENABLED_TENSIX_COL` telemetry word
        /// ([`crate::arc::tag::ENABLED_TENSIX_COL`]).
        ///
        /// Only the population count is used, for the reason given on
        /// [`TENSIX_COLUMNS`]: in translated space the survivors are a prefix, so
        /// the count determines the set and the bit order does not matter. That
        /// matters because the bit order is *not* published — UMD documents its own
        /// `HarvestingMasks` as logical indices
        /// (`umd/device/soc_descriptor.hpp:151-158`) but says nothing about the raw
        /// firmware word, and a wrong guess about it would put a write on a
        /// fused-off tile, which is precisely the failure this type exists to make
        /// impossible.
        ///
        /// `None` if the set bits are not contiguous from bit 0. That shape would
        /// contradict `NoC/Coordinates.md:54`, and the honest response to a chip
        /// that contradicts the specification is to stop and report the word, not
        /// to pick an interpretation and drive the NoC with it.
        pub const fn from_enabled_column_mask(mask: u32) -> Option<Tensix> {
            let n = mask.count_ones();
            if n as usize > TENSIX_COLUMNS.len() {
                return None;
            }
            // Contiguous from bit 0 iff the mask is 2^n - 1. Shifting by 32 is UB
            // in Rust, so the full-width case is spelled out.
            let expected = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
            if mask != expected {
                return None;
            }
            Some(Tensix {
                enabled_columns: n as u8,
            })
        }

        /// How many Tensix columns this chip has.
        pub const fn enabled_column_count(&self) -> usize {
            self.enabled_columns as usize
        }

        /// How many Tensix tiles this chip has: 120 on a p150a with two columns
        /// fused off, 140 on a full part.
        pub const fn tile_count(&self) -> usize {
            self.enabled_columns as usize * TENSIX_ROW_COUNT
        }

        /// The X of each surviving column, ascending.
        pub fn columns(&self) -> impl Iterator<Item = u8> + '_ {
            TENSIX_COLUMNS[..self.enabled_columns as usize]
                .iter()
                .copied()
        }

        /// The X of each fused-off column, ascending.
        ///
        /// Worth having explicitly: a gate that wants to prove the harvesting mask
        /// is *right* needs to name the tiles it must not touch, and a test that
        /// asserts something about them beats a comment saying to avoid them.
        pub fn harvested_columns(&self) -> impl Iterator<Item = u8> + '_ {
            TENSIX_COLUMNS[self.enabled_columns as usize..]
                .iter()
                .copied()
        }

        /// Does *this* chip have a Tensix tile at this coordinate?
        pub fn contains(&self, x: u8, y: u8) -> bool {
            is_tensix_geometry(x, y) && self.columns().any(|c| c == x)
        }

        /// Every Tensix tile this chip has, in row-major order.
        pub fn tiles<N: NocId>(&self) -> impl Iterator<Item = NocCoord<N>> + '_ {
            TENSIX_ROWS
                .flat_map(move |y| self.columns().filter_map(move |x| NocCoord::<N>::new(x, y)))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::noc::Noc0;

        #[test]
        fn a_full_part_matches_the_documented_count() {
            assert_eq!(FULL_TENSIX_TILE_COUNT, 140);
            assert_eq!(Tensix::FULL.tile_count(), 140);
            assert_eq!(Tensix::FULL.tiles::<Noc0>().count(), 140);
            assert_eq!(Tensix::FULL.harvested_columns().count(), 0);
        }

        #[test]
        fn the_excluded_columns_and_rows_are_excluded() {
            assert!(!is_tensix_geometry(NON_MEMORY_COLUMN, 5));
            assert!(!is_tensix_geometry(DRAM_COLUMNS[0], 5));
            assert!(!is_tensix_geometry(DRAM_COLUMNS[1], 5));
            assert!(!is_tensix_geometry(3, ETHERNET_ROW));
            assert!(!is_tensix_geometry(3, 0));
            assert!(is_tensix_geometry(3, 2));
            assert!(is_tensix_geometry(16, 11));
        }

        #[test]
        fn the_geometry_columns_are_exactly_the_tensix_columns() {
            // TENSIX_COLUMNS must not drift from the predicate it summarises.
            let derived = (0..super::super::GRID_WIDTH).filter(|&x| is_tensix_geometry(x, 2));
            assert!(derived.eq(TENSIX_COLUMNS.iter().copied()));
        }

        #[test]
        fn a_harvested_p150a_has_120_tiles_and_loses_the_top_columns() {
            // Two columns fused off: NoC/Coordinates.md:54 puts them at maximal X,
            // so 15 and 16 are the ones that must never be addressed.
            let t = Tensix::from_enabled_column_count(12).unwrap();
            assert_eq!(t.tile_count(), 120);
            assert_eq!(t.tiles::<Noc0>().count(), 120);
            assert!(t.harvested_columns().eq([15u8, 16].iter().copied()));
            assert!(t.contains(14, 11));
            assert!(!t.contains(15, 11));
            assert!(!t.contains(16, 11));
            // Still Tensix geometry -- which is exactly why the two predicates
            // have to be different functions.
            assert!(is_tensix_geometry(16, 11));
        }

        #[test]
        fn a_mask_is_read_by_population_count() {
            assert_eq!(
                Tensix::from_enabled_column_mask(0b1111_1111_1111).unwrap(),
                Tensix::from_enabled_column_count(12).unwrap()
            );
            assert_eq!(
                Tensix::from_enabled_column_mask(0x3FFF).unwrap(),
                Tensix::FULL
            );
        }

        #[test]
        fn a_mask_that_contradicts_the_specification_is_refused() {
            // Holes in the middle would mean harvested columns are not at maximal
            // X. Refuse rather than guess: the alternative is writing to a
            // fused-off tile.
            assert!(Tensix::from_enabled_column_mask(0b1111_1111_1101).is_none());
            assert!(Tensix::from_enabled_column_mask(0b11_0000_0000_0000).is_none());
            // More columns than the part has.
            assert!(Tensix::from_enabled_column_mask(u32::MAX).is_none());
            // No columns at all: the shape an absent telemetry tag takes.
            assert_eq!(Tensix::from_enabled_column_mask(0).unwrap().tile_count(), 0);
        }
    }
}
