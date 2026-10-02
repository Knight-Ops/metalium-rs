//! Moving data between GDDR and a Tensix tile's L1, on the tile itself.
//!
//! A [`DataMover`] is the `dm_b` image (`tt_isa::dm`) resident on RISCV B of
//! one Tensix tile. With one, a tensor that lives in DRAM reaches the tile's L1
//! -- where the unpackers can read it -- and a result goes back, with only a
//! descriptor crossing PCIe. Phase 9's point: the data never does.

use std::time::{Duration, Instant};

use tt_device::core_control::CYCLES_PER_POLL;
use tt_device::{Device, Transport, TransportError, Window};
use tt_isa::dm::{self, op, record, Descriptor, Entry};
use tt_isa::dram::{Dram, DramRange};
use tt_isa::mailbox::{offset, status};
use tt_isa::noc::{NocCoord, NocId};

/// Why a data-mover operation did not complete.
#[derive(Debug)]
pub enum DmError {
    Transport(TransportError),
    /// A descriptor the mover would refuse, refused here first, with its
    /// `tt_isa::dm::error` code.
    Invalid(u32),
    /// The mover refused the descriptor with this code.
    Mover(u32),
    /// Not finished within the deadline.
    TimedOut {
        seq: u32,
        done: u32,
    },
    /// B did not reach its prologue.
    NotStarted,
}

impl From<TransportError> for DmError {
    fn from(e: TransportError) -> Self {
        DmError::Transport(e)
    }
}

impl std::fmt::Display for DmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DmError::Transport(e) => write!(f, "{e}"),
            DmError::Invalid(c) => write!(f, "descriptor refused before submission (code {c})"),
            DmError::Mover(c) => write!(f, "the data mover refused the descriptor (code {c})"),
            DmError::TimedOut { seq, done } => {
                write!(f, "descriptor {seq} did not finish (last finished: {done})")
            }
            DmError::NotStarted => write!(f, "RISCV B did not start the data mover"),
        }
    }
}

impl std::error::Error for DmError {}

pub type Result<T> = std::result::Result<T, DmError>;

/// The resident mover on one tile.
pub struct DataMover<N: NocId> {
    tile: NocCoord<N>,
    usable: u8,
    seq: u32,
    /// How long a descriptor may take on silicon. On ttsim the budget is
    /// simulated cycles instead, since the simulator's clock only moves when
    /// ticked.
    pub deadline: Duration,
}

impl<N: NocId> DataMover<N> {
    /// Load `image` (`tt_firmware_images::DM_B`'s bytes) on `tile`'s RISCV B and
    /// start it, for the channels of `dram`.
    ///
    /// `tile` must be the coordinate the host addresses the tile by: the mover
    /// uses it as the return address of its reads.
    pub fn start<T: Transport>(
        d: &mut Device<T>,
        w: &Window,
        tile: NocCoord<N>,
        dram: &Dram,
        image: &[u8],
    ) -> Result<Self> {
        d.write32(w, tile, dm::MY_X, tile.x() as u32)?;
        d.write32(w, tile, dm::MY_Y, tile.y() as u32)?;
        d.write32(w, tile, dm::USABLE, dram.usable_mask() as u32)?;
        // L1 survives between processes: a stale `TRACE` from a profiled run
        // would have the mover store to a timestamper ttsim does not model.
        for word in [dm::SEQ, dm::DONE, dm::ERROR, dm::TRACE] {
            d.write32(w, tile, word, 0)?;
        }
        let status_at = dm::MAILBOX_BASE + offset::STATUS;
        d.write32(w, tile, status_at, 0)?;
        d.load_and_start(w, tile, tt_isa::tensix::Core::B, image, dm::IMAGE_BASE)?;
        let started = d.wait_for_mailbox(
            w,
            tile,
            status_at,
            dm::MAILBOX_BASE + offset::PANIC_CODE,
            1_000_000,
            |s| s == status::RUNNING,
        )?;
        started.map_err(|_| DmError::NotStarted)?;
        Ok(DataMover {
            tile,
            usable: dram.usable_mask(),
            seq: 0,
            deadline: Duration::from_secs(1),
        })
    }

    pub fn tile(&self) -> NocCoord<N> {
        self.tile
    }

    /// Copy all of `from` into this tile's L1 at `to_l1`, through the channel's
    /// endpoint `port`. Returns once the data is in L1.
    pub fn read<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        from: DramRange,
        port: u8,
        to_l1: u32,
    ) -> Result<()> {
        self.run(d, w, op::READ, from, port, to_l1)
    }

    /// Copy `to.len()` bytes of this tile's L1 at `from_l1` over all of `to`.
    /// Returns once the DRAM tile has acknowledged every write.
    pub fn write<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        from_l1: u32,
        to: DramRange,
        port: u8,
    ) -> Result<()> {
        self.run(d, w, op::WRITE, to, port, from_l1)
    }

    /// Run a list of entries ([`Entry`]: reads, writes, transposed tile reads)
    /// as one descriptor: the list is written to L1 in one bulk write, and the
    /// mover waits for all of it before reporting done.
    pub fn run_list<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        entries: &[[u32; 8]],
    ) -> Result<()> {
        for chunk in entries.chunks(dm::LIST_MAX as usize) {
            self.submit_list(d, w, chunk)?;
            self.wait(d, w)?;
        }
        Ok(())
    }

    /// The first half of [`DataMover::run_list`] for at most
    /// `tt_isa::dm::LIST_MAX` entries: check them, write them and start the
    /// mover, without waiting. [`DataMover::wait`] is the other half; the host
    /// can start other tiles' work in between.
    pub fn submit_list<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        entries: &[[u32; 8]],
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        if entries.len() > dm::LIST_MAX as usize {
            return Err(DmError::Invalid(dm::error::LENGTH));
        }
        // The mover's own checks, run first, so a bad entry costs no PCIe: a
        // plain entry decoded, a record expanded and every entry it makes
        // decoded, as the mover will.
        let usable = self.usable as u32;
        let mut i = 0;
        while i < entries.len() {
            let n = record::len(entries[i][0]);
            if n == 1 {
                Entry::decode(usable, entries[i]).map_err(DmError::Invalid)?;
            } else {
                let rec = entries
                    .get(i..i + n)
                    .ok_or(DmError::Invalid(dm::error::LENGTH))?;
                record::expand(rec, |e| Entry::decode(usable, e).map(|_| ()))
                    .map_err(DmError::Invalid)?;
            }
            i += n;
        }
        let bytes: Vec<u8> = entries
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        d.l1_write(w, self.tile, dm::LIST, &bytes)?;
        d.write32(w, self.tile, dm::OP, op::LIST)?;
        d.write32(w, self.tile, dm::LEN, entries.len() as u32)?;
        self.seq = self.seq.wrapping_add(1).max(1);
        d.write32(w, self.tile, dm::SEQ, self.seq)?;
        Ok(())
    }

    fn run<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        op: u32,
        range: DramRange,
        port: u8,
        l1: u32,
    ) -> Result<()> {
        let words = [
            (dm::OP, op),
            (dm::CHANNEL, range.channel().index() as u32),
            (dm::PORT, port as u32),
            (dm::DRAM_OFFSET, range.offset() as u32),
            (dm::L1_ADDR, l1),
            (dm::LEN, range.len() as u32),
        ];
        // The mover's own check, run first, so a bad descriptor costs no PCIe.
        let [(_, o), (_, c), (_, p), (_, off), (_, a), (_, n)] = words;
        Descriptor::decode(self.usable as u32, o, c, p, off, a, n).map_err(DmError::Invalid)?;
        for (at, v) in words {
            d.write32(w, self.tile, at, v)?;
        }
        self.seq = self.seq.wrapping_add(1).max(1);
        d.write32(w, self.tile, dm::SEQ, self.seq)?;
        self.wait(d, w)
    }

    /// Wait for the last descriptor submitted to finish, and report its error.
    /// Returns at once if nothing is outstanding.
    pub fn wait<T: Transport>(&self, d: &mut Device<T>, w: &Window) -> Result<()> {
        let started = Instant::now();
        let simulated = d.transport().is_simulated();
        let mut ticks = 0u64;
        loop {
            let done = d.read32(w, self.tile, dm::DONE)?;
            if done == self.seq {
                return match d.read32(w, self.tile, dm::ERROR)? {
                    dm::error::NONE => Ok(()),
                    code => Err(DmError::Mover(code)),
                };
            }
            let expired = if simulated {
                ticks > 50_000_000
            } else {
                started.elapsed() > self.deadline
            };
            if expired {
                return Err(DmError::TimedOut {
                    seq: self.seq,
                    done,
                });
            }
            d.tick(CYCLES_PER_POLL);
            ticks += CYCLES_PER_POLL as u64;
        }
    }

    /// Hold B in reset again.
    pub fn stop<T: Transport>(self, d: &mut Device<T>, w: &Window) -> Result<()> {
        d.set_core_reset(w, self.tile, tt_isa::tensix::Core::B, true)?;
        Ok(())
    }
}
