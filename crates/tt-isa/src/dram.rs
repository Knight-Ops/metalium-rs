//! The GDDR6 behind the DRAM tiles: which channels a chip has, and how to name
//! them.
//!
//! A p150 exposes 32 GiB as eight 4 GiB channels, each on three NoC endpoints
//! that alias one memory (`BlackholeA0/README.md:5,21`). Phase 9 keeps tensors
//! there, so this is where the row-35 lesson applies next: a channel that is
//! fused off or failed training must not be nameable, and nothing here can be
//! obtained except from the chip's own answer.

use crate::noc::niu::Niu;
use crate::noc::{Noc0, NocCoord};

/// Channels on a full Blackhole (UMD `blackhole::NUM_DRAM_BANKS`).
pub const CHANNELS: u8 = 8;

/// NoC endpoints per channel (UMD `NUM_NOC_PORTS_PER_DRAM_BANK`). All three
/// reach the same memory: `MEASURED` on ttsim
/// (`probe_dram::which_endpoints_alias_one_channel`), and in UMD's grouping.
pub const PORTS: u8 = 3;

/// Bytes of a channel this workspace addresses.
///
/// The specification and UMD say 4 GiB (`DRAM_BANK_SIZE`). ttsim stops at
/// `0xFF00_0000` and kills the process above it (`probe_niu`, and
/// `probe_dram::a_four_gib_window_reaches_the_whole_channel`: "DRAM write
/// overrun"). The top 16 MiB are therefore given up rather than probed on
/// silicon, where an access the NoC cannot complete takes the host down
/// (divergence row 35). 8 x 4080 MiB is still 31.9 GiB.
pub const CHANNEL_BYTES: u64 = 0xFF00_0000;

/// DRAM-to-L1 NoC reads need source and destination congruent mod 64.
///
/// Wormhole's table says 32 (`WormholeB0/NoC/Alignment.md:18`), and Blackhole's
/// `NoC/MemoryMap.md:106` links an `Alignment.md` that its tree does not have.
/// ttsim refuses a read congruent mod 32 but not 64 as `UndefinedBehavior`
/// (`probe_dram::which_congruence_ttsim_demands_of_a_dram_read`, divergence row
/// 64), which fits Blackhole's 512-bit flits against Wormhole's 256. The
/// stricter rule is followed on both targets. L1-to-DRAM writes need C16.
pub const ALIGN: u64 = 64;

/// One channel's training state, two bits of `GDDR_STATUS` (ARC tag 22).
///
/// Layout from UMD (`firmware_info_provider.cpp:502-551`): bit `2c` is "done",
/// bit `2c + 1` "failed", for channel `c`; firmware 19.7 and later put the
/// channel's BIST result in the same layout at bit 16. `MEASURED`: both p150a
/// cards here (firmware 19.14.0.0) read `0x5555_5555`, every channel trained
/// and BIST-passed.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Training {
    InProgress,
    Passed,
    Failed,
}

impl Training {
    const fn decode(two_bits: u32) -> Self {
        match two_bits & 3 {
            0b01 => Training::Passed,
            0b00 => Training::InProgress,
            _ => Training::Failed,
        }
    }
}

/// Why a chip's DRAM cannot be described.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DramError {
    /// `ENABLED_GDDR` has bits above channel 7.
    UnknownChannels { enabled: u32 },
    /// A channel is fused off. The translated numbering then depends on which
    /// one (`NoC/Coordinates.md:60`, UMD `fill_dram_noc0_translated_mapping`),
    /// and no card here has that shape to measure it against, so it is refused
    /// rather than guessed. p100 boards are the case.
    Harvested { enabled: u32 },
}

/// The DRAM channels of one chip that are present, trained and BIST-clean.
///
/// Built from ARC telemetry: `ENABLED_GDDR` (tag 36, `MEASURED` `0xff` on both
/// cards, bit `c` for channel `c`, as UMD reads it) and `GDDR_STATUS` (tag 22).
/// As with [`crate::noc::grid::Tensix`] and [`crate::eth::Ethernet`], there is no
/// constructor that does not start from a chip, except [`Dram::FULL`] for ttsim.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dram {
    usable: u8,
}

impl Dram {
    /// ttsim's chips: every channel present. ttsim publishes no GDDR telemetry;
    /// all 24 endpoints answer and alias in threes (`probe_dram`).
    pub const FULL: Dram = Dram { usable: 0xFF };

    /// From `ENABLED_GDDR` and `GDDR_STATUS`. A channel that is enabled but has
    /// not passed training (and, where reported, BIST) is left out: it exists,
    /// but nothing may be stored in it.
    pub const fn from_telemetry(enabled: u32, status: u32) -> Result<Self, DramError> {
        if enabled & !0xFF != 0 {
            return Err(DramError::UnknownChannels { enabled });
        }
        if enabled != 0xFF {
            return Err(DramError::Harvested { enabled });
        }
        let mut usable = 0u8;
        let mut c = 0;
        while c < CHANNELS {
            let trained = Training::decode(status >> (2 * c));
            // Zero BIST bits are "not run", which firmware before 19.7 always
            // reports; only a completed-and-failed BIST disqualifies.
            let bist = Training::decode(status >> (16 + 2 * c));
            if matches!(trained, Training::Passed) && !matches!(bist, Training::Failed) {
                usable |= 1 << c;
            }
            c += 1;
        }
        Ok(Dram { usable })
    }

    /// The training state of channel `c` as `GDDR_STATUS` reports it, for a
    /// diagnostic. Not a way to name the channel.
    pub const fn training(status: u32, c: u8) -> (Training, Training) {
        (
            Training::decode(status >> (2 * c as u32)),
            Training::decode(status >> (16 + 2 * c as u32)),
        )
    }

    /// The usable-channel mask, bit `c` for channel `c`: what the host hands a
    /// data mover, which cannot read the ARC itself.
    pub const fn usable_mask(&self) -> u8 {
        self.usable
    }

    /// A grid from a mask a host already derived from the chip
    /// ([`Dram::usable_mask`]). For firmware, which receives it in a mailbox;
    /// host code starts from [`Dram::from_telemetry`].
    pub const fn from_usable_mask(mask: u8) -> Self {
        Dram { usable: mask }
    }

    /// Every usable channel, ascending.
    pub fn channels(&self) -> impl Iterator<Item = DramChannel> + '_ {
        (0..CHANNELS)
            .filter(|c| self.usable & (1 << c) != 0)
            .map(|index| DramChannel { index })
    }

    pub fn channel_count(&self) -> u8 {
        self.usable.count_ones() as u8
    }

    /// Channel `c`, if this chip can store data in it. The only way to obtain a
    /// [`DramChannel`].
    pub fn channel(&self, c: u8) -> Option<DramChannel> {
        (c < CHANNELS && self.usable & (1 << c) != 0).then_some(DramChannel { index: c })
    }
}

/// A usable DRAM channel of one chip. See [`Dram::channel`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct DramChannel {
    index: u8,
}

impl DramChannel {
    pub const fn index(self) -> u8 {
        self.index
    }

    /// The endpoint the chip's management firmware (CMFW) reads DRAM
    /// telemetry through, over NoC #0: port 2 of channels 0 and 4-7, port 0 of
    /// channels 1-3 (tt-metal `soc_descriptors/blackhole_140_arch.yaml:23-75`,
    /// the `worker_endpoint` NoC #0 subchannels, "to avoid SYS-1419").
    pub const fn cmfw_port(self) -> u8 {
        match self.index {
            1..=3 => 0,
            _ => 2,
        }
    }

    /// The one endpoint NoC #1 owns on this channel: port 1 on channels 0-3,
    /// port 0 on channels 4-7.
    ///
    /// The two DRAM columns mirror each other: port `p` of channel `k` and of
    /// channel `k + 4` sit in the same raw row (tt-metal
    /// `soc_descriptors/blackhole_140_arch.yaml:11-21`). NoC #1 carries write
    /// data along the endpoint's row last (Y then X), so with port 1 on every
    /// channel eight channels' writes shared four rows' links. Port 0 of
    /// channels 4-7 is in four other rows, and is never CMFW's there (CMFW's
    /// is port 2), so each channel gets a row of its own. tt-metal uses port 1
    /// on every channel; this departs from it, safely by the rule in
    /// [`DramChannel::owns`].
    pub const fn noc1_port(self) -> u8 {
        if self.index < 4 {
            1
        } else {
            0
        }
    }

    /// Whether requests through `niu` may use this channel's endpoint `port`.
    ///
    /// Each endpoint belongs to one NoC. Blackhole's DRAM endpoint arbiter
    /// drops requests from one NoC when both NoCs issue back to back to the
    /// same endpoint (tt-metal SYS-1419): on card 0 it hung the chip
    /// (`silicon_bench_memory::gddr_aggregate_nocs`, 2026-10-02, tiles on
    /// both NoCs sharing ports). NoC #1 owns [`DramChannel::noc1_port`]; NoC
    /// #0 owns the other two, which include [`DramChannel::cmfw_port`] -- so
    /// CMFW's endpoint never sees NoC #1.
    pub const fn owns(self, niu: Niu, port: u8) -> bool {
        match niu {
            Niu::Noc1 => port == self.noc1_port(),
            Niu::Noc0 => port < PORTS && port != self.noc1_port(),
        }
    }

    /// The endpoint `niu` uses when asked for `port` (0..3), which spreads a
    /// caller's rotation over the endpoints `niu` owns: NoC #1 always its own;
    /// NoC #0 the port itself, or for NoC #1's port, CMFW's endpoint, which
    /// NoC #0 shares with CMFW. Callers name ports as before; the rule in
    /// [`DramChannel::owns`] is applied here, not by them.
    pub const fn port_for(self, niu: Niu, port: u8) -> u8 {
        match niu {
            Niu::Noc1 => self.noc1_port(),
            Niu::Noc0 if port == self.noc1_port() => self.cmfw_port(),
            Niu::Noc0 => port,
        }
    }

    /// The channel's endpoint `port` (0..3), in **translated** coordinates.
    ///
    /// Translated because both targets accept it and silicon may accept
    /// nothing else: with translation on, `X = 0` has no meaning in the Tensix
    /// rows (`NoC/Coordinates.md:35-42`), so half the raw DRAM coordinates name
    /// nothing, and a NoC access that nothing answers is a hung host (row 35).
    /// The numbering is UMD's for an unharvested chip
    /// (`blackhole_coordinate_manager.cpp:303-346`): channels 0-3 at X 17, 4-7
    /// at X 18, three consecutive Y each from 12. `MEASURED` on ttsim against
    /// the raw endpoints (`probe_dram::does_ttsim_answer_at_translated_dram_coordinates`).
    ///
    /// Only for the NIU that owns the port ([`DramChannel::owns`]): `None` for
    /// any other, as for a port past the three.
    pub fn endpoint(self, niu: Niu, port: u8) -> Option<NocCoord<Noc0>> {
        if port >= PORTS || !self.owns(niu, port) {
            return None;
        }
        let x = 17 + self.index / 4;
        let y = 12 + 3 * (self.index % 4) + port;
        NocCoord::new(x, y)
    }

    /// `[offset, offset + len)` of this channel, if it lies inside
    /// [`CHANNEL_BYTES`].
    pub fn range(self, offset: u64, len: u64) -> Option<DramRange> {
        let end = offset.checked_add(len)?;
        (end <= CHANNEL_BYTES).then_some(DramRange {
            channel: self,
            offset,
            len,
        })
    }
}

/// A byte range inside one usable channel. See [`DramChannel::range`].
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct DramRange {
    channel: DramChannel,
    offset: u64,
    len: u64,
}

impl DramRange {
    pub const fn channel(self) -> DramChannel {
        self.channel
    }
    pub const fn offset(self) -> u64 {
        self.offset
    }
    pub const fn len(self) -> u64 {
        self.len
    }
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_cards_measured_status_makes_every_channel_usable() {
        let d = Dram::from_telemetry(0xFF, 0x5555_5555).unwrap();
        assert_eq!(d.channel_count(), 8);
    }

    #[test]
    fn an_untrained_or_failed_channel_cannot_be_named() {
        // Channel 1 in progress, channel 2 failed training, channel 3 failed BIST.
        let status = 0x5555_5555 & !(0b11 << 2) & !(0b11 << 4) | (0b10 << 4);
        let status = status & !(0b11 << 22) | (0b11 << 22);
        let d = Dram::from_telemetry(0xFF, status).unwrap();
        let mut named = [u8::MAX; 8];
        for (slot, c) in named.iter_mut().zip(d.channels()) {
            *slot = c.index();
        }
        assert_eq!(named[..d.channel_count() as usize], [0, 4, 5, 6, 7]);
        assert!(d.channel(1).is_none() && d.channel(2).is_none() && d.channel(3).is_none());
        assert_eq!(
            Dram::training(status, 2),
            (Training::Failed, Training::Passed)
        );
    }

    #[test]
    fn pre_bist_firmware_is_not_mistaken_for_a_failure() {
        let d = Dram::from_telemetry(0xFF, 0x5555).unwrap();
        assert_eq!(d.channel_count(), 8);
    }

    #[test]
    fn harvested_and_unknown_channels_are_refused() {
        assert_eq!(
            Dram::from_telemetry(0x7F, 0x5555_5555),
            Err(DramError::Harvested { enabled: 0x7F })
        );
        assert_eq!(
            Dram::from_telemetry(0x1FF, 0x5555_5555),
            Err(DramError::UnknownChannels { enabled: 0x1FF })
        );
    }

    #[test]
    fn endpoints_are_umd_translated_coordinates() {
        let d = Dram::FULL;
        let e = |c: u8, p: u8| {
            let ch = d.channel(c).unwrap();
            let n = ch
                .endpoint(Niu::Noc0, p)
                .or(ch.endpoint(Niu::Noc1, p))
                .unwrap();
            (n.x(), n.y())
        };
        assert_eq!(e(0, 0), (17, 12));
        assert_eq!(e(3, 2), (17, 23));
        assert_eq!(e(4, 0), (18, 12));
        assert_eq!(e(7, 2), (18, 23));
        let ch0 = d.channel(0).unwrap();
        assert!(ch0.endpoint(Niu::Noc0, 3).is_none());
        let n = ch0.endpoint(Niu::Noc1, 1).unwrap();
        assert_eq!((n.x(), n.y()), (17, 13));
        let n = d.channel(4).unwrap().endpoint(Niu::Noc1, 0).unwrap();
        assert_eq!((n.x(), n.y()), (18, 12));
    }

    #[test]
    fn each_endpoint_belongs_to_one_noc_and_noc1_never_reaches_cmfw() {
        // And NoC #1's eight endpoints sit in eight different raw rows
        // (`DramChannel::noc1_port`): tt-metal's `dram` table, row by port.
        let rows: [[u8; 3]; 4] = [[0, 1, 11], [2, 10, 3], [9, 4, 8], [5, 7, 6]];
        let mut seen = [false; 12];
        for c in Dram::FULL.channels() {
            let row = rows[(c.index() % 4) as usize][c.noc1_port() as usize] as usize;
            assert!(
                !seen[row],
                "channel {} shares NoC #1's row {row}",
                c.index()
            );
            seen[row] = true;
        }
        for c in Dram::FULL.channels() {
            for p in 0..PORTS {
                assert!(c.owns(Niu::Noc0, p) != c.owns(Niu::Noc1, p), "port {p}");
                assert_eq!(c.endpoint(Niu::Noc1, p).is_some(), p == c.noc1_port());
                for niu in [Niu::Noc0, Niu::Noc1] {
                    let q = c.port_for(niu, p);
                    assert!(c.owns(niu, q), "ch {} {niu:?} {p} -> {q}", c.index());
                }
            }
            assert!(c.owns(Niu::Noc0, c.cmfw_port()));
            assert!(!c.owns(Niu::Noc1, c.cmfw_port()));
        }
    }

    #[test]
    fn ranges_stop_below_the_extent_ttsim_models() {
        let c = Dram::FULL.channel(0).unwrap();
        assert!(c.range(CHANNEL_BYTES - 4, 4).is_some());
        assert!(c.range(CHANNEL_BYTES - 4, 8).is_none());
        assert!(c.range(u64::MAX, 2).is_none());
    }
}
