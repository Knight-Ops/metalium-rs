//! GDDR6: what the chip says about its channels, and host reads and writes of
//! them.
//!
//! Every accessor takes a [`DramRange`], which only a chip's [`Dram`] grid can
//! produce, so a fused-off or untrained channel cannot be addressed and nothing
//! can reach past the extent both targets agree on ([`tt_isa::dram::CHANNEL_BYTES`]).

use tt_isa::arc;
use tt_isa::dram::{Dram, DramError, DramRange};

use crate::device::Window;
use crate::telemetry::TelemetryTable;
use crate::{Device, Result, Transport, TransportError};

fn io(msg: String) -> TransportError {
    TransportError::Io(std::io::Error::other(msg))
}

impl<T: Transport> Device<T> {
    /// The chip's usable DRAM channels, from ARC telemetry tags 36 and 22.
    ///
    /// On ttsim, which publishes neither, [`Dram::FULL`]. On silicon the NoC must
    /// be translating: [`tt_isa::dram::DramChannel::endpoint`] hands out translated
    /// coordinates, which name nothing otherwise.
    pub fn dram_grid(&mut self, window: &Window) -> Result<Dram> {
        if self.transport().is_simulated() {
            return Ok(Dram::FULL);
        }
        let table = TelemetryTable::read(self, window)?;
        let tag = |dev: &mut Self, id: u16, name: &str| -> Result<u32> {
            table
                .read_tag(dev, window, id)?
                .ok_or_else(|| io(format!("the chip does not publish {name} (tag {id})")))
        };
        if tag(self, arc::tag::NOC_TRANSLATION, "NOC_TRANSLATION")? == 0 {
            return Err(io("NoC coordinate translation is off; the DRAM endpoints \
                           this workspace names are translated coordinates"
                .into()));
        }
        let enabled = tag(self, arc::tag::ENABLED_GDDR, "ENABLED_GDDR")?;
        let status = tag(self, arc::tag::GDDR_STATUS, "GDDR_STATUS")?;
        Dram::from_telemetry(enabled, status).map_err(|e| match e {
            DramError::UnknownChannels { enabled } => io(format!(
                "ENABLED_GDDR = {enabled:#x} names channels beyond 7"
            )),
            DramError::Harvested { enabled } => io(format!(
                "ENABLED_GDDR = {enabled:#x}: a fused-off channel changes the translated \
                 DRAM numbering (NoC/Coordinates.md:60), which is not implemented"
            )),
        })
    }

    /// Write `data` to `range`, through the channel's first endpoint.
    ///
    /// Any window works; a [`crate::tlb::WindowKind::FourGib`] one covers a whole
    /// channel with one retarget.
    ///
    /// Returns once the bytes are visible through every endpoint of the
    /// channel, not merely posted: the last word is read back through each of
    /// its [`tt_isa::dram::PORTS`] endpoints, as UMD's DRAM barrier touches
    /// every port. The movers read through all three (records rotate the port),
    /// and a read through one port is no fence for writes taken by another.
    /// Without it a mover told to read the range -- by a list, which reaches it
    /// over the tile's L1 -- can overtake the writes: on silicon a batched
    /// MNIST's test batches read part stale images (divergence row T).
    pub fn dram_write(&mut self, window: &Window, range: DramRange, data: &[u8]) -> Result<()> {
        check_len(range, data.len())?;
        let at = range.channel().endpoint(0).expect("port 0 exists");
        // GDDR is memory by construction: the bulk path.
        self.write_memory(window, at, range.offset(), data)?;
        if data.len() >= 4 {
            let last = range.offset() + data.len() as u64 - 4;
            let mut word = [0u8; 4];
            for port in 0..tt_isa::dram::PORTS {
                let via = range.channel().endpoint(port).expect("in range");
                self.read_memory(window, via, last, &mut word)?;
            }
        }
        Ok(())
    }

    /// Read `range` into `out`, through the channel's first endpoint.
    pub fn dram_read(&mut self, window: &Window, range: DramRange, out: &mut [u8]) -> Result<()> {
        check_len(range, out.len())?;
        let at = range.channel().endpoint(0).expect("port 0 exists");
        self.read_memory(window, at, range.offset(), out)
    }
}

fn check_len(range: DramRange, len: usize) -> Result<()> {
    if range.len() != len as u64 {
        return Err(TransportError::OutOfBounds {
            bar: crate::Bar::Bar4,
            offset: range.offset(),
            len: len as u64,
        });
    }
    Ok(())
}
