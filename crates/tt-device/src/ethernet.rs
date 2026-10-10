//! Ethernet tiles: what the chip says about them, their L1, TT-link writes, and
//! RISCV E1.
//!
//! Every accessor here takes a [`EthTile`], which only a chip's [`Ethernet`] grid
//! hands out, so a harvested tile cannot be named. Every L1 access is checked
//! against [`eth::FIRMWARE_L1`] -- the ranges base firmware on E0 was measured to
//! occupy -- because writing there corrupts the code that keeps the link trained,
//! and nothing would say so until the link dropped.

use tt_isa::arc;
use tt_isa::eth::{self, EthCore, EthTile, Ethernet, Mac, PortStatus, TxCommand};
use tt_isa::mailbox;

use crate::core_control::WaitError;
use crate::device::Window;
use crate::telemetry::TelemetryTable;
use crate::{Device, Result, Transport, TransportError};

/// What one Ethernet tile's base firmware reports (`eth::BOOT_RESULTS`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LinkState {
    pub port: PortStatus,
    pub train_status: u32,
    /// This tile's MAC, from its own chip-info block.
    pub own_mac: Mac,
    /// The cable partner's MAC, once the chip-info exchange has happened.
    pub peer_mac: Option<Mac>,
}

impl LinkState {
    /// Up, with a known partner.
    pub fn is_up(&self) -> bool {
        self.port == PortStatus::Up
    }
}

fn refuse_firmware_l1(address: u64, len: usize) -> Result<()> {
    if !eth::is_customer_l1(address, len as u64) {
        return Err(TransportError::Hazard {
            address,
            reason: "Ethernet L1 outside the customer range: base firmware on E0 lives \
                     there (tt_isa::eth::FIRMWARE_L1)",
        });
    }
    Ok(())
}

fn refuse_misaligned(address: u64, what: &'static str) -> Result<()> {
    if address % eth::TT_LINK_ALIGN != 0 {
        return Err(TransportError::Hazard {
            address,
            reason: what,
        });
    }
    Ok(())
}

impl<T: Transport> Device<T> {
    /// The chip's Ethernet row, from ARC telemetry tag 35.
    ///
    /// No fallback: a chip that does not publish the tag is refused, as
    /// [`Device::tensix_grid`] refuses a chip without tag 34. ttsim's ARC does not
    /// answer this at all; simulator callers use [`Ethernet::FULL`] and say so.
    pub fn ethernet_grid(&mut self, window: &Window) -> Result<Ethernet> {
        let table = TelemetryTable::read(self, window)?;
        let mask = table
            .read_tag(self, window, arc::tag::ENABLED_ETH)?
            .ok_or_else(|| {
                TransportError::Io(std::io::Error::other(
                    "the chip does not publish ENABLED_ETH (tag 35)",
                ))
            })?;
        Ethernet::from_enabled_mask(mask).ok_or_else(|| {
            TransportError::Io(std::io::Error::other(format!(
                "ENABLED_ETH = {mask:#x} names endpoints beyond E13"
            )))
        })
    }

    /// Read `buf.len()` bytes of the tile's L1 at `address`, refusing firmware L1.
    pub fn eth_read(
        &mut self,
        window: &Window,
        tile: EthTile,
        address: u64,
        buf: &mut [u8],
    ) -> Result<()> {
        refuse_firmware_l1(address, buf.len())?;
        self.read(window, tile.coord(), address, buf)
    }

    /// Write `data` into the tile's L1 at `address`, refusing firmware L1.
    pub fn eth_write(
        &mut self,
        window: &Window,
        tile: EthTile,
        address: u64,
        data: &[u8],
    ) -> Result<()> {
        refuse_firmware_l1(address, data.len())?;
        self.write(window, tile.coord(), address, data)
    }

    /// [`Device::eth_write`], then one read-back of the dword holding the last
    /// byte written: returns once the bytes have landed. For a buffer another
    /// agent -- the E1 mover, a link transfer -- acts on next. Any byte range;
    /// firmware L1 is refused before anything is written, and a failed write
    /// reads nothing back.
    pub fn eth_write_fenced(
        &mut self,
        window: &Window,
        tile: EthTile,
        address: u64,
        data: &[u8],
    ) -> Result<()> {
        refuse_firmware_l1(address, data.len())?;
        self.write_range_fenced(window, tile.coord(), address, data)
    }

    /// What the tile's base firmware says about its link. Reads only
    /// [`eth::BOOT_RESULTS`], which base firmware publishes for exactly this.
    pub fn eth_link_state(&mut self, window: &Window, tile: EthTile) -> Result<LinkState> {
        let mut words = [0u8; 256 * 4];
        self.read(window, tile.coord(), eth::BOOT_RESULTS, &mut words)?;
        let w = |i: u64| {
            let i = i as usize * 4;
            u32::from_le_bytes([words[i], words[i + 1], words[i + 2], words[i + 3]])
        };
        let mac = |block: u64| {
            Mac::from_words(
                w(block + eth::boot::INFO_MAC),
                w(block + eth::boot::INFO_MAC + 1),
            )
        };
        let exchanged = w(eth::boot::INFO_EXCHANGE) == eth::boot::EXCHANGED;
        Ok(LinkState {
            port: PortStatus::from_word(w(eth::boot::PORT_STATUS)),
            train_status: w(eth::boot::TRAIN_STATUS),
            own_mac: mac(eth::boot::OWN_INFO),
            peer_mac: exchanged.then(|| mac(eth::boot::PEER_INFO)),
        })
    }

    /// Send `len` bytes at local `src` to `dest` in the cable partner's L1, as one
    /// TT-link L1-write command on [`eth::DATA_QUEUE`], driven through the NoC.
    ///
    /// Returns once the queue has *latched* the command (`STATUS_CMD_ONGOING`
    /// clear), which is not delivery: `EthernetTxRx.md` says hardware may re-read
    /// `src` for resends, so it must not change until the receiver has the data.
    /// The destination is checked against this chip's firmware map, which is the
    /// partner's too only because both are Blackhole base firmware -- a peer of a
    /// different kind would need its own.
    pub fn eth_tt_link_write(
        &mut self,
        window: &Window,
        tile: EthTile,
        src: u64,
        dest: u64,
        len: usize,
    ) -> Result<std::result::Result<(), WaitError>> {
        refuse_misaligned(src, "TT-link source must be 16-byte aligned")?;
        refuse_misaligned(dest, "TT-link destination must be 16-byte aligned")?;
        refuse_misaligned(len as u64, "TT-link length must be a multiple of 16")?;
        refuse_firmware_l1(src, len)?;
        refuse_firmware_l1(dest, len)?;
        let q = eth::txq_base(eth::DATA_QUEUE);
        let cmd = TxCommand::L1Write
            .word(eth::DATA_QUEUE)
            .expect("L1 writes on every queue");
        let c = tile.coord();
        // Must not write CMD while a previous command is still being latched.
        if let Err(e) = self.wait_for_txq(window, tile)? {
            return Ok(Err(e));
        }
        self.write32(window, c, q + eth::txq::TRANSFER_START_ADDR, src as u32)?;
        self.write32(window, c, q + eth::txq::TRANSFER_SIZE_BYTES, len as u32)?;
        self.write32(window, c, q + eth::txq::DEST_ADDR, dest as u32)?;
        self.write32(window, c, q + eth::txq::CMD, cmd)?;
        // `EthernetTxRx.md`: read CMD back so the STATUS read cannot overtake it.
        let _ = self.read32(window, c, q + eth::txq::CMD)?;
        self.wait_for_txq(window, tile)
    }

    fn wait_for_txq(
        &mut self,
        window: &Window,
        tile: EthTile,
    ) -> Result<std::result::Result<(), WaitError>> {
        let status = eth::txq_base(eth::DATA_QUEUE) + eth::txq::STATUS;
        let r = self.wait_for_mailbox(window, tile.coord(), status, status, 100_000, |s| {
            s & eth::txq::STATUS_CMD_ONGOING == 0
        })?;
        Ok(r.map(|_| ()))
    }

    /// Where E1 will start on leaving reset.
    pub fn e1_reset_pc(&mut self, window: &Window, tile: EthTile) -> Result<u32> {
        let mut b = [0u8; 4];
        // See `load_and_start_e1` for why this is unchecked.
        self.read_unchecked(window, tile.coord(), eth::E1_RESET_PC, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Hold E1 in reset. Touches only E1's bit.
    pub fn park_e1(&mut self, window: &Window, tile: EthTile) -> Result<()> {
        self.set_eth_reset(window, tile, EthCore::E1, true)
    }

    /// Load `image` at [`eth::E1_IMAGE`] and start E1 there.
    ///
    /// E1 is held first, so a running image is never overwritten in place, and
    /// released last; leaving reset invalidates the instruction cache
    /// (WH `EthernetTile/SoftReset.md`), so there is no `fence.i` question. E0 is
    /// never touched: [`Self::set_eth_reset`] changes one bit.
    pub fn load_and_start_e1(
        &mut self,
        window: &Window,
        tile: EthTile,
        image: &[u8],
    ) -> Result<()> {
        if image.len() as u64 > eth::E1_IMAGE_MAX {
            return Err(TransportError::Hazard {
                address: eth::E1_IMAGE,
                reason: "E1 image overruns its region into the E1 mailbox",
            });
        }
        self.park_e1(window, tile)?;
        self.eth_write(window, tile, eth::E1_IMAGE, image)?;
        let zero = [0u8; 16];
        self.eth_write(window, tile, eth::MAILBOX_BASE, &zero)?;
        // `E1_RESET_PC` lies inside the *Tensix* local-RAM aperture that
        // `Device::write` refuses for every tile. An Ethernet tile maps its
        // reset-PC registers there instead (`ethdump.c:365`); its local RAM is not
        // NoC-visible at all. The unchecked path is taken for this one register,
        // on a tile that is Ethernet by construction.
        self.write_unchecked(
            window,
            tile.coord(),
            eth::E1_RESET_PC,
            &(eth::E1_IMAGE as u32).to_le_bytes(),
        )?;
        self.set_eth_reset(window, tile, EthCore::E1, false)
    }

    /// Read-modify-write of one core's bit in [`eth::SOFT_RESET_0`]; see there for
    /// why not a whole-word write. Private, and only ever called with E1.
    fn set_eth_reset(
        &mut self,
        window: &Window,
        tile: EthTile,
        core: EthCore,
        held: bool,
    ) -> Result<()> {
        debug_assert_eq!(core, EthCore::E1, "E0 belongs to base firmware");
        let c = tile.coord();
        let current = self.read32(window, c, eth::SOFT_RESET_0)?;
        let bit = core.soft_reset_bit();
        let updated = if held { current | bit } else { current & !bit };
        if updated != current {
            self.write32(window, c, eth::SOFT_RESET_0, updated)?;
        }
        Ok(())
    }

    /// Is E1 held in reset?
    pub fn is_e1_in_reset(&mut self, window: &Window, tile: EthTile) -> Result<bool> {
        let v = self.read32(window, tile.coord(), eth::SOFT_RESET_0)?;
        Ok(v & EthCore::E1.soft_reset_bit() != 0)
    }

    /// Read a word of the E1 image's mailbox (`tt_isa::mailbox::offset`).
    pub fn e1_mailbox_read(&mut self, window: &Window, tile: EthTile, offset: u64) -> Result<u32> {
        self.read32(window, tile.coord(), eth::MAILBOX_BASE + offset)
    }

    /// [`Device::wait_for_status`] for the E1 image.
    pub fn wait_for_e1_status(
        &mut self,
        window: &Window,
        tile: EthTile,
        budget_cycles: u64,
        predicate: impl FnMut(u32) -> bool,
    ) -> Result<std::result::Result<u32, WaitError>> {
        self.wait_for_mailbox(
            window,
            tile.coord(),
            eth::MAILBOX_BASE + mailbox::offset::STATUS,
            eth::MAILBOX_BASE + mailbox::offset::PANIC_CODE,
            budget_cycles,
            predicate,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn firmware_l1_is_a_hazard() {
        assert!(refuse_firmware_l1(eth::BOOT_RESULTS, 4).is_err());
        assert!(refuse_firmware_l1(0x6000, 4).is_err());
        assert!(refuse_firmware_l1(eth::BUFFERS.start, 4096).is_ok());
        assert!(refuse_firmware_l1(0x7_1FF8, 16).is_err());
    }

    #[test]
    fn tt_link_alignment_is_a_hazard() {
        assert!(refuse_misaligned(0x2_0008, "x").is_err());
        assert!(refuse_misaligned(0x2_0010, "x").is_ok());
    }
}
