//! Ethernet tiles: register map, firmware-owned L1, and the link facts the chip
//! publishes about itself.
//!
//! Sources, in order of authority:
//! * `BlackholeA0/EthernetTile/{README,EthernetTxRx}.md` and
//!   `BabyRISCV/README.md` for the register map.
//! * `BlackholeA0/EthernetTile/Samples/ethdump/ethdump.c` for reset, the reset-PC
//!   override and the boot-results words, which no page documents. Every such
//!   fact is marked `ETHDUMP` and was re-read on both p150a cards and on ttsim
//!   (`crates/tt-tests/tests/{probe_eth,silicon_eth_survey}.rs`).
//! * Measurement, marked `MEASURED`, where neither says anything.
//!
//! # Which core
//!
//! RISCV E0 belongs to Tenstorrent's base firmware: it trains and retrains the
//! link (`BabyRISCV/README.md`). Everything here that starts code does so on E1,
//! as ethdump does, and [`EthCore`] has no way to name E0 as a target for code.

/// L1 per Ethernet tile (`EthernetTile/README.md`).
pub const L1_SIZE: u64 = 512 * 1024;

/// The two RISC-V cores of an Ethernet tile.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EthCore {
    /// Runs Tenstorrent base firmware. Never reset, never loaded.
    E0,
    /// Ours. Free for customer code (`ethdump/README.md`, "Implementation notes").
    E1,
}

impl EthCore {
    /// Its bit in [`SOFT_RESET_0`] (`ethdump.c:450-451`, `ETHDUMP`).
    pub const fn soft_reset_bit(self) -> u32 {
        match self {
            EthCore::E0 => 0x0800,
            EthCore::E1 => 0x1000,
        }
    }
}

/// `RISCV_DEBUG_REG_SOFT_RESET_0` in an Ethernet tile (`ethdump.c:364`, `ETHDUMP`).
///
/// Reads `0x47000` on live tiles of both p150a cards: E1 held, E0 running, and
/// three bits (13, 14, 18) with no documented meaning here. Changes are therefore
/// read-modify-writes of one core's bit, never whole-word writes -- ethdump writes
/// `0` to release E1, which would also clear those three.
pub const SOFT_RESET_0: u64 = 0xFFB1_21B0;
/// Where E1 starts on leaving reset (`ethdump.c:365`, `ETHDUMP`).
pub const E1_RESET_PC: u64 = 0xFFB1_4008;
/// Written by ethdump alongside [`E1_RESET_PC`] with the code's end
/// (`ethdump.c:366,1052`); its effect is undocumented.
pub const E1_END_PC: u64 = 0xFFB1_400C;
/// E1's local data RAM, 8 KiB, not NoC-visible (`BabyRISCV/README.md`).
pub const LOCAL_RAM: u64 = 0xFFB0_0000;
/// Size of each core's local data RAM.
pub const LOCAL_RAM_SIZE: u64 = 8 * 1024;

/// Base-firmware parameter block in L1 (`ethdump.c:362`, `ETHDUMP`).
pub const BOOT_PARAMS: u64 = 0x7_C000;
/// Base-firmware results block in L1 (`ethdump.c:363`, `ETHDUMP`), 256 words.
pub const BOOT_RESULTS: u64 = 0x7_CC00;

/// Word indices into [`BOOT_RESULTS`].
pub mod boot {
    /// Port status; see [`super::PortStatus`] (`ethdump.c:476-477`).
    pub const PORT_STATUS: u64 = 1;
    /// Training status; 2 is "Complete" (`ethdump.c:483-484`).
    pub const TRAIN_STATUS: u64 = 2;
    /// This tile's chip-info block, seven words. `MEASURED`: word 2 of it equals the
    /// chip's ARC telemetry tag 2, words 3..=4 are its MAC (`ethdump.c:515-516`).
    pub const OWN_INFO: u64 = 240;
    /// 2 once the peer's chip info has arrived at [`PEER_INFO`]; `ethdump.c:512`
    /// tests for 1, which on these cards is every tile *without* a live link.
    /// `MEASURED`.
    pub const INFO_EXCHANGE: u64 = 247;
    /// The peer's chip-info block, laid out as [`OWN_INFO`]. `MEASURED`: on both
    /// cards each live tile's `PEER_INFO` equals its cable partner's `OWN_INFO`.
    pub const PEER_INFO: u64 = 248;
    /// Words in a chip-info block.
    pub const INFO_WORDS: u64 = 7;
    /// Offset of the 48-bit MAC within a block: `(hi 24 bits, lo 24 bits)`.
    pub const INFO_MAC: u64 = 3;
    /// [`INFO_EXCHANGE`]'s value once the exchange completed.
    pub const EXCHANGED: u32 = 2;
}

/// `boot::PORT_STATUS` (`ethdump.c:477`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PortStatus {
    Unknown,
    Up,
    Down,
    NoPort,
    Other(u32),
}

impl PortStatus {
    pub const fn from_word(w: u32) -> Self {
        match w {
            0 => PortStatus::Unknown,
            1 => PortStatus::Up,
            2 => PortStatus::Down,
            3 => PortStatus::NoPort,
            o => PortStatus::Other(o),
        }
    }
}

/// A 48-bit MAC address, as the chip-info block carries it.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Mac(pub u64);

impl Mac {
    /// From the block's two words, each holding 24 bits (`ethdump.c:524-526`).
    pub const fn from_words(hi: u32, lo: u32) -> Self {
        Mac(((hi as u64 & 0xFF_FFFF) << 24) | (lo as u64 & 0xFF_FFFF))
    }
}

/// Ethernet L1 that belongs to base firmware on silicon. `MEASURED` by
/// `silicon_eth_survey::ethernet_l1_occupancy`, both cards, both live tiles, two
/// samples three seconds apart: these ranges held data (or changed); all else was
/// zero. Rounded out to 4 KiB pages, and the tail is taken whole from the first
/// occupied page, since it holds [`BOOT_PARAMS`] and [`BOOT_RESULTS`].
pub const FIRMWARE_L1: [core::ops::Range<u64>; 2] = [0x6000..0x7000, 0x7_2000..L1_SIZE];

/// Is `[addr, addr + len)` clear of every firmware-owned range?
pub const fn is_customer_l1(addr: u64, len: u64) -> bool {
    let end = match addr.checked_add(len) {
        Some(e) => e,
        None => return false,
    };
    if end > L1_SIZE {
        return false;
    }
    let mut i = 0;
    while i < FIRMWARE_L1.len() {
        let r = &FIRMWARE_L1[i];
        if addr < r.end && r.start < end {
            return false;
        }
        i += 1;
    }
    true
}

/// Where our E1 image lives and runs from.
pub const E1_IMAGE: u64 = 0x1_0000;
/// Largest E1 image: up to the mailbox.
pub const E1_IMAGE_MAX: u64 = MAILBOX_BASE - E1_IMAGE;
/// The E1 image's mailbox, laid out as [`crate::mailbox::offset`].
pub const MAILBOX_BASE: u64 = 0x1_F000;
/// Our transfer buffers, `0x20000..0x70000`: 320 KiB.
pub const BUFFERS: core::ops::Range<u64> = 0x2_0000..0x7_0000;

/// TX queue `i`'s register block (`EthernetTxRx.md`, "TX Queue Memory Map").
pub const fn txq_base(i: u8) -> u64 {
    0xFFB9_0000 + i as u64 * 0x1000
}
/// RX queue `i`'s register block (`EthernetTxRx.md`, "RX Queue Memory Map").
pub const fn rxq_base(i: u8) -> u64 {
    0xFFB9_4000 + i as u64 * 0x1000
}

/// TX queue register offsets (`EthernetTxRx.md`).
pub mod txq {
    pub const CTRL: u64 = 0x00;
    pub const CMD: u64 = 0x04;
    pub const STATUS: u64 = 0x08;
    pub const MAX_PKT_SIZE_BYTES: u64 = 0x0C;
    pub const TRANSFER_START_ADDR: u64 = 0x14;
    pub const TRANSFER_SIZE_BYTES: u64 = 0x18;
    pub const DEST_ADDR: u64 = 0x1C;
    pub const TRANSFER_CNT: u64 = 0x30;
    pub const TXPKT_CFG_SEL_SW: u64 = 0x80;
    /// `ETH_TXQ_CTRL_KEEPALIVE`: the queue is in TT-link mode.
    pub const CTRL_KEEPALIVE: u32 = 1 << 0;
    /// `ETH_TXQ_STATUS_CMD_ONGOING_BIT`.
    pub const STATUS_CMD_ONGOING: u32 = 1 << 16;
}

/// RX queue register offsets (`EthernetTxRx.md`).
pub mod rxq {
    pub const CTRL: u64 = 0x00;
    /// `ETH_RXQ_OUTSTANDING_WR_CNT`: writes received from the network and not
    /// yet committed to L1. Read-only; not decoded by ttsim (divergence row 62).
    pub const OUTSTANDING_WR_CNT: u64 = 0x50;
    /// `ETH_RXQ_CTRL_PACKET_MODE`: the queue expects TT-link packets.
    pub const CTRL_PACKET_MODE: u32 = 1 << 1;
}

/// A TX queue command (`ETH_TXQ_CMD`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TxCommand {
    Raw,
    L1Write,
    MmioWrite,
}

impl TxCommand {
    /// The word to write, or `None` where the queue cannot send it: queue 2 has
    /// no MMIO-write packets (`EthernetTxRx.md`, TX Queues warning).
    pub const fn word(self, queue: u8) -> Option<u32> {
        match (self, queue) {
            (_, 3..) => None,
            (TxCommand::MmioWrite, 2) => None,
            (TxCommand::Raw, _) => Some(1),
            (TxCommand::L1Write, _) => Some(2),
            (TxCommand::MmioWrite, _) => Some(4),
        }
    }
}

/// The queue this workspace sends on.
///
/// On silicon base firmware puts all three queues in TT-link mode; queue 2 is
/// the one that carries no MMIO or Overlay traffic, so it is the least likely to
/// be shared, and its `TRANSFER_CNT` read 0 before our first send (`MEASURED`).
/// On ttsim only queues 0 and 1 are in TT-link mode, yet an L1 write on queue 2
/// is delivered all the same (divergence row 56).
pub const DATA_QUEUE: u8 = 2;

/// TT-link L1 writes require 16-byte-aligned addresses and a multiple-of-16
/// length (`EthernetTxRx.md`, `ETH_TXQ_TRANSFER_START_ADDR`/`SIZE_BYTES`/`DEST_ADDR`).
pub const TT_LINK_ALIGN: u64 = 16;

/// A tile's position in the Ethernet row, by endpoint index `E0..E13`.
///
/// `MEASURED` on both cards (`silicon_eth_survey::ethernet_niu_identity`) and
/// consistent with `ethdump/README.md`'s table: the index read from each tile's
/// `NOC_ENDPOINT_ID` at raw `(x, 1)`.
pub const ENDPOINT_X: [u8; 14] = [1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10];

/// The Ethernet row of one chip, from the chip's own answer.
///
/// Built from ARC telemetry tag 35, whose bit `i` is set when endpoint `Ei` is
/// enabled (`MEASURED`: `0x3edf` on both cards, and the two clear bits, 5 and 8,
/// are exactly the tiles whose `NIU_CFG_0` carries the harvested bit). Like
/// [`crate::noc::grid::Tensix`], there is no constructor that does not start from
/// a chip.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Ethernet {
    enabled: u16,
}

impl Ethernet {
    /// ttsim's chips: every endpoint present (`probe_eth::survey_ethernet_tiles`:
    /// no tile of either `bh_x2` chip reports harvested).
    pub const FULL: Ethernet = Ethernet { enabled: 0x3FFF };

    /// From ARC telemetry tag 35. Refuses bits beyond `E13`.
    pub const fn from_enabled_mask(mask: u32) -> Option<Self> {
        if mask & !0x3FFF != 0 {
            return None;
        }
        Some(Ethernet {
            enabled: mask as u16,
        })
    }

    /// Raw NoC #0 X of every enabled tile, by endpoint index.
    pub fn tiles(&self) -> impl Iterator<Item = u8> + '_ {
        (0..14)
            .filter(|i| self.enabled & (1 << i) != 0)
            .map(|i| ENDPOINT_X[i])
    }

    /// Is there an enabled Ethernet tile at raw `(x, 1)` on this chip?
    pub fn contains(&self, x: u8) -> bool {
        self.tiles().any(|t| t == x)
    }

    /// The tile at raw `(x, 1)`, if this chip has one there. The only way to
    /// obtain an [`EthTile`], so a harvested tile cannot be named.
    pub fn tile(&self, x: u8) -> Option<EthTile> {
        self.contains(x).then_some(EthTile { x })
    }
}

/// An enabled Ethernet tile of one chip. See [`Ethernet::tile`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct EthTile {
    x: u8,
}

impl EthTile {
    /// Raw NoC #0 X.
    pub const fn x(self) -> u8 {
        self.x
    }

    /// Its NoC #0 coordinate. Row 1 is not X-translated (`NoC/Coordinates.md:28-29`),
    /// so the raw coordinate is also the one a translated NoC accepts; the survey
    /// read every tile this way on both cards.
    pub fn coord(self) -> crate::noc::NocCoord<crate::noc::Noc0> {
        crate::noc::NocCoord::new(self.x, crate::noc::grid::ETHERNET_ROW)
            .expect("row 1 and X <= 16 are in range")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e0_is_never_the_default() {
        assert_eq!(EthCore::E1.soft_reset_bit(), 0x1000);
        assert_ne!(EthCore::E0.soft_reset_bit(), EthCore::E1.soft_reset_bit());
    }

    #[test]
    fn firmware_l1_is_refused() {
        assert!(is_customer_l1(BUFFERS.start, BUFFERS.end - BUFFERS.start));
        assert!(is_customer_l1(E1_IMAGE, E1_IMAGE_MAX + 0x1000));
        assert!(!is_customer_l1(BOOT_RESULTS, 4));
        assert!(!is_customer_l1(0x6FFC, 8));
        assert!(!is_customer_l1(0x7_1FF0, 0x20));
        assert!(!is_customer_l1(L1_SIZE - 4, 8));
        assert!(!is_customer_l1(u64::MAX, 2));
    }

    #[test]
    fn queue_2_has_no_mmio_writes() {
        assert_eq!(TxCommand::L1Write.word(2), Some(2));
        assert_eq!(TxCommand::MmioWrite.word(2), None);
        assert_eq!(TxCommand::MmioWrite.word(0), Some(4));
        assert_eq!(TxCommand::Raw.word(3), None);
    }

    #[test]
    fn endpoint_x_is_a_permutation_of_the_row() {
        let mut xs = ENDPOINT_X;
        xs.sort_unstable();
        assert_eq!(xs, [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn the_measured_mask_drops_the_harvested_tiles() {
        let g = Ethernet::from_enabled_mask(0x3edf).unwrap();
        assert!(!g.contains(5) && !g.contains(14));
        assert_eq!(g.tiles().count(), 12);
        assert!(Ethernet::from_enabled_mask(0x4000).is_none());
    }

    #[test]
    fn mac_packs_like_ethdump() {
        assert_eq!(Mac::from_words(0x208c47, 0x2e120).0, 0x208c_4702_e120);
    }
}

/// The E1 data mover's contract, shared by `tt-firmware`'s `eth_e1` and the
/// host (`tt_kernels::link`), so the layout cannot drift between them.
///
/// One transfer per link direction at a time. Both ends run the same image,
/// and each is a sender and a receiver:
///
/// 1. The host writes a send descriptor into the sending E1's mailbox, `SEQ`
///    last.
/// 2. The sender pulls the data from a Tensix tile into [`TX_STAGE`] (or takes
///    it as the host staged it), TT-link-writes it to the partner's
///    [`RX_LAND`], and then TT-link-writes an [`INBOX`] record: sequence number,
///    length and destination.
/// 3. The receiver sees a new record, waits until its RX queue has no writes
///    outstanding, NoC-writes [`RX_LAND`] into the destination Tensix tile (or
///    leaves it), and TT-link-writes an [`ACK`] back into the sender's mailbox.
///
/// The wait is what makes step 3 safe. TT-link delivers one queue's packets in
/// order -- the receiver discards anything out of sequence -- so every data
/// packet was *accepted* before the record was. Nothing documented says it was
/// *committed to L1* before the record was, but `ETH_RXQ_OUTSTANDING_WR_CNT`
/// counts exactly the accepted-but-uncommitted writes (`EthernetTxRx.md`). Once
/// the record is visible and that count is zero, everything before it is in L1.
/// An earlier version checksummed the whole buffer on both ends instead, which
/// cost ~3.75 us per KiB and held the mover to ~195 MB/s. The host, on the
/// sending chip, learns of delivery from [`ACKED`], and never has to talk to the
/// other chip.
pub mod mover {
    use super::MAILBOX_BASE as M;

    /// This tile's raw NoC #0 coordinate, written by the host before release:
    /// core identity is not discoverable at run time.
    pub const MY_X: u64 = M + 0x10;
    pub const MY_Y: u64 = M + 0x14;
    /// Nonzero: wait for the RX queue's outstanding writes to drain before
    /// forwarding. Set by the host on silicon; zero on ttsim, which commits RX
    /// writes synchronously and does not decode the counter (row 62).
    pub const LANDING_WAIT: u64 = M + 0x18;
    /// Nonzero: record each step of a send and a receive in [`TRACE_RING`],
    /// stamped by E1's own cycle counter (an Ethernet tile has no
    /// timestamper). Zeroed at start with the rest of the mailbox.
    pub const TRACE: u64 = M + 0x1C;
    /// Events recorded so far. The host zeroes it to empty the ring; the
    /// mover stops recording when the ring is full rather than wrapping.
    pub const TRACE_COUNT: u64 = M + 0x20;

    /// Send descriptor. `SEND_SEQ` is written last and is nonzero.
    pub const SEND_SEQ: u64 = M + 0x40;
    pub const SEND_SRC_X: u64 = M + 0x44;
    pub const SEND_SRC_Y: u64 = M + 0x48;
    pub const SEND_SRC_ADDR: u64 = M + 0x4C;
    pub const SEND_LEN: u64 = M + 0x50;
    pub const SEND_DST_X: u64 = M + 0x54;
    pub const SEND_DST_Y: u64 = M + 0x58;
    pub const SEND_DST_ADDR: u64 = M + 0x5C;

    /// The last `SEND_SEQ` whose data and record were handed to the link.
    pub const SENT: u64 = M + 0x80;
    /// The last sequence number the partner acknowledged, after forwarding.
    pub const ACKED: u64 = M + 0x84;
    /// Nonzero if the mover gave up; see [`error`].
    pub const ERROR: u64 = M + 0x88;
    /// The last inbox sequence number this tile forwarded.
    pub const RECEIVED: u64 = M + 0x8C;

    /// Where the partner's record lands: eight words, `seq` in word 0 *and*
    /// word 7, so a record whose two 16-byte halves have not both arrived is
    /// not mistaken for a whole one.
    pub const INBOX: u64 = M + 0x100;
    /// Where the partner's acknowledgement lands: `seq` in word 0.
    pub const ACK: u64 = M + 0x140;
    /// Local staging for the outgoing record and acknowledgement. The TX queue
    /// may re-read them for resends, so each is only rewritten after the next
    /// handshake has proved the last one arrived.
    pub const RECORD_STAGE: u64 = M + 0x180;
    pub const ACK_STAGE: u64 = M + 0x1C0;

    pub const RECORD_BYTES: usize = 32;
    /// Words of a record: seq, len, dst x, dst y, dst addr, 0, 0, seq.
    pub const RECORD_WORDS: usize = 8;

    /// "No tile": in `SEND_SRC_X`, the data is already in [`TX_STAGE`]; in
    /// `SEND_DST_X`, leave it in [`RX_LAND`].
    pub const NO_TILE: u32 = 0xFF;

    /// Outgoing data, and the landing zone for the partner's.
    pub const TX_STAGE: u64 = 0x2_0000;
    pub const RX_LAND: u64 = 0x4_0000;
    /// Largest single transfer.
    pub const MAX_LEN: u32 = 0x2_0000;
    /// Largest single TT-link command the mover issues. Conservative: 4 KiB
    /// is what the host-driven gates measured, and nothing documents a limit.
    pub const TT_LINK_CHUNK: u32 = 0x4000;

    /// The trace ring: [`TRACE_EVENTS`] records of [`TRACE_RECORD_BYTES`],
    /// each `[event, seq, cycles, 0]`: E1's 32-bit
    /// `cycle`, which wraps (every ~3.2 s), so compare with `wrapping_sub`. In the transfer buffers,
    /// past `RX_LAND`.
    pub const TRACE_RING: u64 = 0x6_0000;
    pub const TRACE_EVENTS: u32 = 1024;
    pub const TRACE_RECORD_BYTES: u64 = 16;

    /// What the mover records ([`TRACE`]), in the order a send and its
    /// receive pass through them.
    pub mod event {
        /// The sender saw a new send descriptor.
        pub const SEND_PICKUP: u32 = 1;
        /// The sender's NoC read of the Tensix source into `TX_STAGE` landed
        /// (only for a Tensix source).
        pub const NOC_IN_DONE: u32 = 2;
        /// The data's TT-link commands have all been latched by the TX queue.
        pub const DATA_SENT: u32 = 3;
        /// The record's TT-link command has been latched; `SENT` follows.
        pub const RECORD_SENT: u32 = 4;
        /// The receiver saw a whole new record.
        pub const RECORD_SEEN: u32 = 5;
        /// The receiver's RX queue had no writes outstanding: the data is in
        /// its L1.
        pub const LANDED: u32 = 6;
        /// The receiver's NoC write into the Tensix destination completed
        /// (only for a Tensix destination).
        pub const NOC_OUT_DONE: u32 = 7;
        /// The receiver's acknowledgement has been latched by its TX queue.
        pub const ACK_SENT: u32 = 8;
        /// The sender saw the acknowledgement arrive.
        pub const ACK_SEEN: u32 = 9;
    }

    /// `ERROR` codes.
    pub mod error {
        /// A descriptor with a length that is zero, not a multiple of 16, or
        /// over [`super::MAX_LEN`].
        pub const LENGTH: u32 = 1;
        /// A Tensix address not 16-byte aligned (the NoC copy needs it
        /// congruent with the 16-aligned staging buffers).
        pub const ALIGNMENT: u32 = 2;
        /// The RX queue never reported its writes committed.
        pub const LANDING: u32 = 3;
    }

    // The layout stays inside the mailbox page and the customer buffers, and
    // every TT-link address is 16-byte aligned -- checked at compile time.
    const _: () = {
        assert!(ACK_STAGE + 16 <= super::BUFFERS.start);
        assert!(RX_LAND + MAX_LEN as u64 <= super::BUFFERS.end);
        assert!(TX_STAGE + MAX_LEN as u64 <= RX_LAND);
        assert!(RX_LAND + MAX_LEN as u64 <= TRACE_RING);
        assert!(TRACE_RING + TRACE_EVENTS as u64 * TRACE_RECORD_BYTES <= super::BUFFERS.end);
        assert!(TRACE_COUNT + 4 <= SEND_SEQ);
        assert!(INBOX % 16 == 0 && ACK % 16 == 0 && RECORD_STAGE % 16 == 0);
        assert!(ACK_STAGE % 16 == 0 && TX_STAGE % 16 == 0 && RX_LAND % 16 == 0);
    };
}
