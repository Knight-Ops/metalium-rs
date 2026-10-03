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
    /// The one address bit that tells the two NIUs apart: a core switches NoC by
    /// flipping it in the base.
    pub const NOC_SELECT: u64 = NOC0_BASE ^ NOC1_BASE;
    const _: () = assert!(NOC_SELECT.count_ones() == 1 && NOC0_BASE & NOC_SELECT == 0);

    /// Which of a tile's two NIUs a request leaves through: the run-time
    /// counterpart of [`super::NocId`]. Requests to GDDR are bound to it
    /// ([`crate::dram::DramChannel::owns`]).
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub enum Niu {
        Noc0,
        Noc1,
    }

    impl Niu {
        /// The NIU's register base in the tile's own address space.
        pub const fn base(self) -> u64 {
            match self {
                Niu::Noc0 => NOC0_BASE,
                Niu::Noc1 => NOC1_BASE,
            }
        }

        /// 0 or 1.
        pub const fn index(self) -> usize {
            self as usize
        }
    }

    /// Identifies the tile: index in bits 0..=7, type in bits 8..=23, NoC index in
    /// bits 24..=31 (`MemoryMap.md:196-200`).
    pub const NOC_ENDPOINT_ID: u64 = 0x0048;

    /// This NIU's raw (untranslated) coordinate, `x | y << 6`, with the grid's
    /// size and routing flags above (`MemoryMap.md`, `NOC_NODE_ID`): where the
    /// tile physically sits, whatever translation renumbers it to.
    pub const NOC_NODE_ID: u64 = 0x0044;

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

        /// The ID itself, `0..16`.
        pub const fn index(self) -> usize {
            self.0 as usize
        }
    }

    /// Most response-marked requests one transaction ID may have in flight.
    ///
    /// [`reqs_outstanding`] is 8 bits and wraps silently (`Counters.md`): with
    /// 256 in flight it reads 0, and a wait on it would end while data is still
    /// arriving. 128 leaves the counter half its range, and is still far past
    /// what fills a link: ~11 requests of 16 KiB cover the NoC's bandwidth-delay
    /// product, 128 of them are 2 MiB.
    pub const MAX_IN_FLIGHT: u16 = 128;
    const _: () = assert!(MAX_IN_FLIGHT >= 1 && MAX_IN_FLIGHT < 256);

    /// Keeps one transaction ID's requests in flight at or under a cap, so its
    /// 8-bit [`reqs_outstanding`] counter cannot wrap.
    ///
    /// `room` is how many more requests may be issued before the counter must
    /// be read: the counter is at most `cap - room`, since completions only
    /// take it down. While there is room an issue costs one compare and one
    /// decrement; without, the counter is read -- and spun on until it is
    /// under the cap -- and `room` refilled from it. Nothing needs telling of
    /// completions: a stale `room` is only ever too small, and costs one read.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct InFlight {
        room: u16,
        cap: u16,
    }

    impl Default for InFlight {
        fn default() -> Self {
            Self::new()
        }
    }

    impl InFlight {
        pub const fn new() -> Self {
            InFlight {
                room: MAX_IN_FLIGHT,
                cap: MAX_IN_FLIGHT,
            }
        }

        /// Use `cap` from now on: 0 for [`MAX_IN_FLIGHT`], otherwise clamped
        /// to `1..=MAX_IN_FLIGHT`. Only with nothing in flight -- it refills
        /// the room -- which is why the mover sets it as a list starts, after
        /// the last one's wait. Lower caps are for gates that force the
        /// throttle.
        pub fn set_cap(&mut self, cap: u32) {
            self.cap = if cap == 0 {
                MAX_IN_FLIGHT
            } else {
                cap.min(MAX_IN_FLIGHT as u32) as u16
            };
            self.room = self.cap;
        }

        pub const fn cap(&self) -> u16 {
            self.cap
        }

        /// Will the next [`InFlight::before_issue`] read the counter -- and so
        /// possibly wait?
        pub const fn at_cap(&self) -> bool {
            self.room == 0
        }

        /// Make room for one more request. `outstanding` reads the ID's
        /// counter (its low 8 bits); it is called only when there is no room
        /// left. Returns how many reads found the ID still at its cap: 0 when
        /// there was room without waiting.
        pub fn before_issue(&mut self, mut outstanding: impl FnMut() -> u8) -> u32 {
            if self.room != 0 {
                return 0;
            }
            let mut full = 0u32;
            loop {
                let n = outstanding() as u16;
                if n < self.cap {
                    self.room = self.cap - n;
                    return full;
                }
                full = full.saturating_add(1);
            }
        }

        /// One request was issued.
        pub fn after_issue(&mut self) {
            self.room -= 1;
        }

        /// The counter was seen at zero: every request has completed.
        pub fn drained(&mut self) {
            self.room = self.cap;
        }
    }

    /// Largest length one request may carry between L1 addresses. Larger ones
    /// are split by hardware, but then one request moves the 8-bit outstanding
    /// counter by more than one -- which [`InFlight`], counting one per request,
    /// would not see -- so this refuses them instead.
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
        /// Add `value` to the 32-bit word at `to` -- the L1 of a Tensix or
        /// Ethernet tile, never MMIO or DRAM -- atomically, and write the word's
        /// old value to this tile's L1 at `ret_local` (`NoC/Atomics.md`,
        /// "Atomic increment": `IntWidth` 31 for a full-width add, `Ofs` the
        /// word's place in its 16-byte unit, so the old value is the result).
        AtomicIncrement {
            to: Endpoint,
            value: u32,
            ret_local: u32,
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
        /// A DRAM endpoint port the issuing NIU does not own
        /// (`crate::dram::DramChannel::owns`): both NoCs on one endpoint is
        /// the SYS-1419 hang.
        PortNoc,
    }

    const CMD_WR: u32 = 2;
    const CMD_RD: u32 = 0;
    const CMD_AT: u32 = 1;
    /// `NOC_AT_LEN_BE`'s opcode for an increment (`Bits32.lua`,
    /// `NOC_AT_LEN_BE_Increment`: bits 12..16 = 1).
    const AT_INCREMENT: u32 = 1 << 12;
    const WR_INLINE: u32 = 1 << 3;
    const RESP_MARKED: u32 = 1 << 4;
    /// `NOC_CMD_VC_STATIC` with `NOC_CMD_STATIC_VC` = class `0b00`, buddy 1.
    const STATIC_VC_1: u32 = (1 << 7) | (1 << 13);

    impl Command {
        /// The initiator registers to write, in order, before `CMD_CTRL`, for
        /// a request through `niu`. `me` is the initiating tile's coordinate,
        /// which reads name as their return address and writes as the source
        /// of their data. A GDDR request through an NIU that does not own its
        /// port is refused ([`RequestError::PortNoc`]).
        ///
        /// Every request is on static virtual channel 1 -- class `0b00`, buddy
        /// bit 1 (`MemoryMap.md`, `NOC_CTRL` bits 7, 13-15) -- as tt-metal
        /// issues its DRAM reads and writes (`blackhole/noc_nonblocking_api.h`,
        /// `NOC_CMD_STATIC_VC(1)`), rather than leaving the choice to the NIU.
        // Inlined into the firmware's issue path, which RISCV B's instruction
        // cache must hold whole: out of line it lands among the cold code.
        #[inline(always)]
        pub fn registers(
            &self,
            me: (u8, u8),
            txn: TxnId,
            niu: Niu,
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
                // One request's worth of a `DramMove`, which owns the GDDR
                // encoding and its checks.
                Command::ReadDram {
                    from,
                    port,
                    to_local,
                } => return DramMove::single(from, port, to_local, false, me, txn, niu),
                Command::WriteDram {
                    from_local,
                    to,
                    port,
                } => return DramMove::single(to, port, from_local, true, me, txn, niu),
                Command::AtomicIncrement {
                    to,
                    value,
                    ret_local,
                } => {
                    if to.addr % 4 != 0
                        || ret_local % 4 != 0
                        || to.addr >= MMIO_START
                        || ret_local >= MMIO_START
                    {
                        return Err(RequestError::Alignment);
                    }
                    // `IntWidth` 31: all 32 bits; `Ofs`: the word within its
                    // 16-byte unit, so `L1Address` is `to.addr` itself.
                    let len_be = AT_INCREMENT | (31 << 2) | ((to.addr >> 2) & 3);
                    (to, local(ret_local), CMD_AT | RESP_MARKED, len_be, value)
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
                (CTRL, ctrl | STATIC_VC_1),
                (AT_LEN_BE, len_be),
                (AT_DATA, data),
            ])
        }
    }

    /// A move between one GDDR range and this tile's L1, checked whole once,
    /// whose NIU requests ([`DramMove::requests`]) are then encoded with no
    /// further checks: each is a sub-range of the checked move, at most
    /// [`MAX_REQUEST_BYTES`], with the congruence preserved (every split is a
    /// multiple of [`MAX_REQUEST_BYTES`] from the start).
    ///
    /// The mover's per-entry path builds one per descriptor rather than
    /// checking each request through [`Command::registers`]: on card 0 the
    /// per-request checks were ~110 of a 4 KiB read entry's ~350 cycles
    /// (`docs/firmware-performance.md`, checklist 9.14).
    #[derive(Copy, Clone, Debug)]
    pub struct DramMove {
        /// The DRAM endpoint, as `TARG`/`RET_ADDR_HI` take it.
        dram_hi: u32,
        dram_at: u32,
        /// This tile, as `TARG`/`RET_ADDR_HI` take it.
        me_hi: u32,
        l1: u32,
        len: u32,
        write: bool,
        tag: u32,
    }

    /// One request of a [`DramMove`]: the initiator registers that are not
    /// zero, by name ([`DramRequest::registers`] lays them out).
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct DramRequest {
        pub targ: u32,
        pub targ_hi: u32,
        pub ret: u32,
        pub ret_hi: u32,
        pub tag: u32,
        pub ctrl: u32,
        pub len: u32,
    }

    impl DramRequest {
        /// As [`Command::registers`] lays a request out.
        pub fn registers(self) -> [(u64, u32); 10] {
            use initiator::*;
            [
                (TARG_ADDR_LO, self.targ),
                (TARG_ADDR_MID, 0),
                (TARG_ADDR_HI, self.targ_hi),
                (RET_ADDR_LO, self.ret),
                (RET_ADDR_MID, 0),
                (RET_ADDR_HI, self.ret_hi),
                (PACKET_TAG, self.tag),
                (CTRL, self.ctrl),
                (AT_LEN_BE, self.len),
                (AT_DATA, 0),
            ]
        }
    }

    impl DramMove {
        /// All of `range` to (`write == false`) or from this tile's L1 at
        /// `l1`, through endpoint `port`, issued by the tile at `me` through
        /// `niu` under `txn`. Refused as [`Command::registers`] would refuse
        /// any of its requests: a port beyond the channel's three, one `niu`
        /// does not own (the SYS-1419 hang), zero bytes, L1 not congruent to
        /// the GDDR address (mod [`crate::dram::ALIGN`] for a read, 16 for a
        /// write), or either side reaching [`MMIO_START`].
        pub fn new(
            range: crate::dram::DramRange,
            port: u8,
            l1: u32,
            write: bool,
            me: (u8, u8),
            txn: TxnId,
            niu: Niu,
        ) -> Result<Self, RequestError> {
            if port >= crate::dram::PORTS {
                return Err(RequestError::Port);
            }
            let at = range
                .channel()
                .endpoint(niu, port)
                .ok_or(RequestError::PortNoc)?;
            let len = u32::try_from(range.len()).map_err(|_| RequestError::Length)?;
            // `CHANNEL_BYTES` is below 4 GiB, so the offset fits the low word.
            let dram_at = range.offset() as u32;
            let modulus = if write { 16 } else { crate::dram::ALIGN as u32 };
            if len == 0 {
                return Err(RequestError::Length);
            }
            let end = |a: u32| a.checked_add(len).filter(|&e| e <= MMIO_START);
            if dram_at % modulus != l1 % modulus || end(dram_at).is_none() || end(l1).is_none() {
                return Err(RequestError::Alignment);
            }
            Ok(DramMove {
                dram_hi: Endpoint {
                    x: at.x(),
                    y: at.y(),
                    addr: 0,
                }
                .hi(),
                dram_at,
                me_hi: Endpoint {
                    x: me.0,
                    y: me.1,
                    addr: 0,
                }
                .hi(),
                l1,
                len,
                write,
                tag: (txn.0 as u32) << 10,
            })
        }

        /// [`Command::registers`] for a move of at most one request.
        #[allow(clippy::too_many_arguments)]
        fn single(
            range: crate::dram::DramRange,
            port: u8,
            l1: u32,
            write: bool,
            me: (u8, u8),
            txn: TxnId,
            niu: Niu,
        ) -> Result<[(u64, u32); 10], RequestError> {
            let m = Self::new(range, port, l1, write, me, txn, niu)?;
            if m.len > MAX_REQUEST_BYTES {
                return Err(RequestError::Length);
            }
            Ok(m.request(0, m.len))
        }

        /// The move's NIU requests, in order: `MAX_REQUEST_BYTES` each, the
        /// last short.
        #[inline(always)]
        pub fn requests(self) -> impl Iterator<Item = [(u64, u32); 10]> {
            (0..self.len)
                .step_by(MAX_REQUEST_BYTES as usize)
                .map(move |done| self.request(done, (self.len - done).min(MAX_REQUEST_BYTES)))
        }

        /// The move's NIU requests as [`DramRequest`]s, in order:
        /// `MAX_REQUEST_BYTES` each, the last short. The values the issuing
        /// core writes, without the register array -- which the mover would
        /// otherwise build on its stack and read back for every request.
        #[inline(always)]
        pub fn words(self) -> impl Iterator<Item = DramRequest> {
            (0..self.len)
                .step_by(MAX_REQUEST_BYTES as usize)
                .map(move |done| self.word(done, (self.len - done).min(MAX_REQUEST_BYTES)))
        }

        #[inline(always)]
        fn word(&self, done: u32, n: u32) -> DramRequest {
            let (targ_hi, targ, ret_hi, ret, ctrl) = if self.write {
                (
                    self.me_hi,
                    self.l1 + done,
                    self.dram_hi,
                    self.dram_at + done,
                    CMD_WR,
                )
            } else {
                (
                    self.dram_hi,
                    self.dram_at + done,
                    self.me_hi,
                    self.l1 + done,
                    CMD_RD,
                )
            };
            DramRequest {
                targ,
                targ_hi,
                ret,
                ret_hi,
                tag: self.tag,
                ctrl: ctrl | RESP_MARKED | STATIC_VC_1,
                len: n,
            }
        }

        /// The registers for `n` bytes from `done` into the move.
        #[inline(always)]
        fn request(&self, done: u32, n: u32) -> [(u64, u32); 10] {
            self.word(done, n).registers()
        }
    }

    /// The host-connected PCIe tile on a p150 board (PCIe 0), translated, as
    /// `ethdump.c:440`'s `BH_PCIE_XY` names it: where a tile's NoC reaches host
    /// memory (`PCIExpressTile/README.md`, "NoC to Host").
    pub const PCIE_HOST: (u8, u8) = (19, 24);

    /// One request of a [`HostMove`]: [`DramRequest`]'s registers with the
    /// target's and the return's high address words, which a host address
    /// fills.
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub struct HostRequest {
        pub targ: u32,
        pub targ_mid: u32,
        pub targ_hi: u32,
        pub ret: u32,
        pub ret_mid: u32,
        pub ret_hi: u32,
        pub tag: u32,
        pub ctrl: u32,
        pub len: u32,
    }

    /// A move between this tile's L1 and host memory at the PCIe tile
    /// ([`crate::dm::op::HOST_READ`]): its requests, `MAX_REQUEST_BYTES` each,
    /// on static VC 1 as every other request. The caller checked it
    /// (`crate::dm::Entry::decode`): the host address in a host window, not
    /// crossing a 4 GiB boundary, congruent with `l1`.
    #[derive(Copy, Clone, Debug)]
    pub struct HostMove {
        pub host_lo: u32,
        pub host_hi: u32,
        pub l1: u32,
        pub len: u32,
        pub write: bool,
        pub me: (u8, u8),
        pub txn: TxnId,
    }

    impl HostMove {
        pub fn words(self) -> impl Iterator<Item = HostRequest> {
            let pcie = Endpoint {
                x: PCIE_HOST.0,
                y: PCIE_HOST.1,
                addr: 0,
            }
            .hi();
            let me = Endpoint {
                x: self.me.0,
                y: self.me.1,
                addr: 0,
            }
            .hi();
            (0..self.len)
                .step_by(MAX_REQUEST_BYTES as usize)
                .map(move |done| {
                    let n = (self.len - done).min(MAX_REQUEST_BYTES);
                    let (host, l1) = (self.host_lo + done, self.l1 + done);
                    let (targ, targ_mid, targ_hi, ret, ret_mid, ret_hi, ctrl) = if self.write {
                        (l1, 0, me, host, self.host_hi, pcie, CMD_WR)
                    } else {
                        (host, self.host_hi, pcie, l1, 0, me, CMD_RD)
                    };
                    HostRequest {
                        targ,
                        targ_mid,
                        targ_hi,
                        ret,
                        ret_mid,
                        ret_hi,
                        tag: (self.txn.0 as u32) << 10,
                        ctrl: ctrl | RESP_MARKED | STATIC_VC_1,
                        len: n,
                    }
                })
        }
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
    mod in_flight_tests {
        use super::*;

        /// A transaction ID's counter: up on each issue, down as requests
        /// complete, here in a pseudo-random order and pace.
        struct Niu {
            outstanding: u32,
            peak: u32,
            reads: u32,
            seed: u32,
        }

        impl Niu {
            fn complete_some(&mut self) {
                self.seed ^= self.seed << 13;
                self.seed ^= self.seed >> 17;
                self.seed ^= self.seed << 5;
                let done = self.seed % 4;
                self.outstanding = self.outstanding.saturating_sub(done);
            }
        }

        #[test]
        fn the_counter_never_passes_the_cap_and_is_never_read_under_it() {
            for cap in [1u32, 2, 7, 128, 0] {
                let mut f = InFlight::new();
                f.set_cap(cap);
                let mut niu = Niu {
                    outstanding: 0,
                    peak: 0,
                    reads: 0,
                    seed: 0x1234_5678 | cap,
                };
                let mut stalls = 0;
                for i in 0..100_000u32 {
                    let reads_before = niu.reads;
                    let full = f.before_issue(|| {
                        niu.reads += 1;
                        niu.complete_some();
                        niu.outstanding as u8
                    });
                    stalls += u32::from(full > 0);
                    if niu.reads == reads_before {
                        assert!(niu.outstanding < f.cap() as u32, "issued at the cap unread");
                    }
                    niu.outstanding += 1;
                    f.after_issue();
                    niu.peak = niu.peak.max(niu.outstanding);
                    if i % 1000 == 999 {
                        while niu.outstanding > 0 {
                            niu.complete_some();
                        }
                        f.drained();
                    }
                }
                assert!(
                    niu.peak <= f.cap() as u32,
                    "cap {}: peak {}",
                    f.cap(),
                    niu.peak
                );
                assert!(niu.peak < 256);
                assert!(stalls > 0, "cap {}: the throttle never engaged", f.cap());
            }
        }

        #[test]
        fn under_the_cap_there_is_no_read() {
            let mut f = InFlight::new();
            for _ in 0..MAX_IN_FLIGHT {
                assert_eq!(f.before_issue(|| panic!("read under the cap")), 0);
                f.after_issue();
            }
            assert!(f.at_cap());
            // At the cap the counter is read; room already there is no stall.
            assert_eq!(f.before_issue(|| 3), 0);
            assert!(!f.at_cap());
            f.drained();
            assert!(!f.at_cap());
        }

        #[test]
        fn caps_clamp() {
            let mut f = InFlight::new();
            assert_eq!(f.cap(), MAX_IN_FLIGHT, "new() is the default cap");
            f.set_cap(0);
            assert_eq!(f.cap(), MAX_IN_FLIGHT);
            f.set_cap(1000);
            assert_eq!(f.cap(), MAX_IN_FLIGHT);
            f.set_cap(1);
            assert_eq!(f.cap(), 1);
        }
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
            let r = w.registers((3, 1), T, Niu::Noc0).unwrap();
            assert_eq!(reg(&r, initiator::TARG_ADDR_LO), 0x2_0000);
            assert_eq!(reg(&r, initiator::TARG_ADDR_HI), 3 | (1 << 6));
            assert_eq!(reg(&r, initiator::RET_ADDR_LO), 0x3_0000);
            assert_eq!(reg(&r, initiator::RET_ADDR_HI), 4 | (5 << 6));
            assert_eq!(reg(&r, initiator::CTRL), CMD_WR | RESP_MARKED | STATIC_VC_1);
            assert_eq!(reg(&r, initiator::PACKET_TAG), 3 << 10);
        }

        #[test]
        fn a_read_returns_to_the_initiator() {
            let rd = Command::Read {
                from: at(4, 5, 0x3_0010),
                to_local: 0x2_0010,
                len: 16,
            };
            let r = rd.registers((3, 1), T, Niu::Noc0).unwrap();
            assert_eq!(reg(&r, initiator::TARG_ADDR_HI), 4 | (5 << 6));
            assert_eq!(reg(&r, initiator::RET_ADDR_HI), 3 | (1 << 6));
            assert_eq!(reg(&r, initiator::CTRL), CMD_RD | RESP_MARKED | STATIC_VC_1);
        }

        #[test]
        fn a_dram_read_targets_the_translated_endpoint_and_needs_c64() {
            let ch = crate::dram::Dram::FULL.channel(1).unwrap();
            let rd = |off: u64, to: u32| Command::ReadDram {
                from: ch.range(off, 2048).unwrap(),
                port: 1,
                to_local: to,
            };
            let r = rd(0x10_0060, 0x3_0020)
                .registers((3, 4), T, Niu::Noc1)
                .unwrap();
            // Channel 1, port 1 (NoC #1's): translated (17, 12 + 3 + 1).
            assert_eq!(reg(&r, initiator::TARG_ADDR_HI), 17 | (16 << 6));
            assert_eq!(reg(&r, initiator::TARG_ADDR_LO), 0x10_0060);
            assert_eq!(reg(&r, initiator::TARG_ADDR_MID), 0);
            assert_eq!(reg(&r, initiator::RET_ADDR_LO), 0x3_0020);
            assert_eq!(reg(&r, initiator::AT_LEN_BE), 2048);
            assert_eq!(reg(&r, initiator::CTRL), CMD_RD | RESP_MARKED | STATIC_VC_1);
            // Congruent mod 16, or mod 32, but not mod 64: refused (row 64).
            for off in [0x10_0010, 0x10_0040] {
                assert_eq!(
                    rd(off, 0x3_0020).registers((3, 4), T, Niu::Noc1),
                    Err(RequestError::Alignment)
                );
            }
            // Port 1 is NoC #1's: through NoC #0 it is refused (SYS-1419).
            assert_eq!(
                rd(0x10_0060, 0x3_0020).registers((3, 4), T, Niu::Noc0),
                Err(RequestError::PortNoc)
            );
        }

        #[test]
        fn a_dram_write_needs_c16_a_port_and_one_request() {
            let ch = crate::dram::Dram::FULL.channel(0).unwrap();
            let wr = |len: u64, port: u8, from: u32| Command::WriteDram {
                from_local: from,
                to: ch.range(0x40, len).unwrap(),
                port,
            };
            let r = wr(64, 0, 0x2_0000).registers((3, 4), T, Niu::Noc0).unwrap();
            assert_eq!(reg(&r, initiator::TARG_ADDR_LO), 0x2_0000);
            assert_eq!(reg(&r, initiator::RET_ADDR_HI), 17 | (12 << 6));
            assert_eq!(reg(&r, initiator::RET_ADDR_LO), 0x40);
            assert_eq!(
                wr(64, 0, 0x2_0008).registers((3, 4), T, Niu::Noc0),
                Err(RequestError::Alignment)
            );
            assert_eq!(
                wr(64, 3, 0x2_0000).registers((3, 4), T, Niu::Noc0),
                Err(RequestError::Port)
            );
            // Ports 0 and 2 are NoC #0's: through NoC #1 they are refused.
            for port in [0, 2] {
                assert_eq!(
                    wr(64, port, 0x2_0000).registers((3, 4), T, Niu::Noc1),
                    Err(RequestError::PortNoc)
                );
            }
            assert_eq!(
                wr(MAX_REQUEST_BYTES as u64 + 16, 0, 0x2_0000).registers((3, 4), T, Niu::Noc0),
                Err(RequestError::Length)
            );
            assert_eq!(
                wr(0, 0, 0x2_0000).registers((3, 4), T, Niu::Noc0),
                Err(RequestError::Length)
            );
        }

        /// A move of several requests encodes each exactly as a `Command` of
        /// that one request would, in order, the last short.
        #[test]
        fn a_long_dram_move_is_its_requests() {
            let ch = crate::dram::Dram::FULL.channel(6).unwrap();
            let len = 2 * MAX_REQUEST_BYTES as u64 + 96;
            for write in [false, true] {
                let range = ch.range(0x20_0040, len).unwrap();
                let port = ch.port_for(Niu::Noc0, 0);
                let mv = DramMove::new(range, port, 0x4_0040, write, (3, 4), T, Niu::Noc0).unwrap();
                let got: [[(u64, u32); 10]; 3] = {
                    let mut it = mv.requests();
                    [it.next().unwrap(), it.next().unwrap(), it.next().unwrap()]
                };
                assert!(mv.requests().nth(3).is_none());
                for (k, regs) in got.iter().enumerate() {
                    let done = k as u64 * MAX_REQUEST_BYTES as u64;
                    let n = (len - done).min(MAX_REQUEST_BYTES as u64);
                    let part = ch.range(range.offset() + done, n).unwrap();
                    let l1 = 0x4_0040 + done as u32;
                    let cmd = if write {
                        Command::WriteDram {
                            from_local: l1,
                            to: part,
                            port,
                        }
                    } else {
                        Command::ReadDram {
                            from: part,
                            port,
                            to_local: l1,
                        }
                    };
                    assert_eq!(
                        *regs,
                        cmd.registers((3, 4), T, Niu::Noc0).unwrap(),
                        "{write} {k}"
                    );
                }
            }
            // Refusals are the whole move's.
            let r = ch.range(0x20_0040, len).unwrap();
            assert_eq!(
                DramMove::new(r, ch.cmfw_port(), 0x4_0050, false, (3, 4), T, Niu::Noc0).err(),
                Some(RequestError::Alignment)
            );
            assert_eq!(
                DramMove::new(r, ch.noc1_port(), 0x4_0040, false, (3, 4), T, Niu::Noc0).err(),
                Some(RequestError::PortNoc)
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
                    reg(&c.registers((0, 0), T, Niu::Noc0).unwrap(), initiator::CTRL) & (1 << 31),
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
                inline_l1.registers((0, 0), T, Niu::Noc0),
                Err(RequestError::InlineToL1)
            );
            let skew = Command::Write {
                from_local: 0x10,
                to: at(1, 2, 0x18),
                len: 16,
            };
            assert_eq!(
                skew.registers((0, 0), T, Niu::Noc0),
                Err(RequestError::Alignment)
            );
            let big = Command::Read {
                from: at(1, 2, 0),
                to_local: 0,
                len: 16385,
            };
            assert_eq!(
                big.registers((0, 0), T, Niu::Noc0),
                Err(RequestError::Length)
            );
            let mmio_copy = Command::Write {
                from_local: 0,
                to: at(1, 2, 0xFFB0_0000),
                len: 16,
            };
            assert_eq!(
                mmio_copy.registers((0, 0), T, Niu::Noc0),
                Err(RequestError::Alignment)
            );
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
