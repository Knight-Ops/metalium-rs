//! Moving data between GDDR and a Tensix tile's L1, on the tile itself.
//!
//! A [`DataMover`] is the `dm_b` image (`tt_isa::dm`) resident on RISCV B of
//! one Tensix tile. With one, a tensor that lives in DRAM reaches the tile's L1
//! -- where the unpackers can read it -- and a result goes back, with only a
//! descriptor crossing PCIe. Phase 9's point: the data never does.

use std::time::{Duration, Instant};

use tt_device::core_control::CYCLES_PER_POLL;
use tt_device::{Device, Transport, TransportError, Window};
use tt_isa::dataflow::{Channel, Endpoint};
use tt_isa::dm::{self, op, record, Descriptor, Entry, Mover};
use tt_isa::dram::{Dram, DramRange};
use tt_isa::mailbox::{offset, status};
use tt_isa::noc::niu::Niu;
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
    /// The write NoC asked for is one this mover's image does not have: only
    /// NC writes on NoC #1 (`tt_isa::dm::write_noc`); B writes on NoC #0.
    WriteNoc {
        mover: dm::Mover,
        noc: WriteNoc,
    },
    /// Queued list `list` failed with this code (`tt_isa::dm::QUEUE_ERROR`);
    /// the queue has stopped.
    Queued {
        list: u32,
        code: u32,
    },
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
            DmError::WriteNoc { mover, noc } => {
                write!(f, "{:?} writes on NoC #0 only; {noc:?} is NC's", mover.core)
            }
            DmError::Queued { list, code } => {
                write!(
                    f,
                    "queued list {list} failed (code {code}); the queue has stopped"
                )
            }
        }
    }
}

impl std::error::Error for DmError {}

pub type Result<T> = std::result::Result<T, DmError>;

/// Where a list of `n` entries fits in the queue's ring (`tt_isa::dm::LIST`,
/// `LIST_MAX` entries) beside the lists in flight, oldest first, the next
/// free entry being `write_at`: after the newest, or wrapped to the start
/// before the oldest. Never across the end: the mover reads a list as one run.
fn ring_room(
    in_flight: &std::collections::VecDeque<(u32, u32, u32)>,
    write_at: u32,
    n: u32,
) -> Option<u32> {
    let max = dm::LIST_MAX;
    let Some(&(_, oldest, _)) = in_flight.front() else {
        return Some(0);
    };
    if write_at > oldest {
        if write_at + n <= max {
            Some(write_at)
        } else if n <= oldest {
            Some(0)
        } else {
            None
        }
    } else if write_at + n <= oldest {
        Some(write_at)
    } else {
        None
    }
}

/// What a mover's in-flight cap cost it ([`DataMover::throttle`]).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Throttle {
    /// Requests that waited for room under the cap.
    pub stalls: u32,
    /// Tile-counter cycles they waited, in total (wrapping).
    pub cycles: u32,
}

impl std::ops::Add for Throttle {
    type Output = Throttle;
    fn add(self, o: Throttle) -> Throttle {
        Throttle {
            stalls: self.stalls.wrapping_add(o.stalls),
            cycles: self.cycles.wrapping_add(o.cycles),
        }
    }
}

/// Which NIU a mover's GDDR writes go out on ([`DataMover::set_write_noc`],
/// `tt_isa::dm::write_noc`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WriteNoc {
    Noc0,
    Noc1,
    /// Write entries take turns, NoC #0 first in each list.
    Alternate,
}

impl From<Niu> for WriteNoc {
    fn from(n: Niu) -> Self {
        match n {
            Niu::Noc0 => WriteNoc::Noc0,
            Niu::Noc1 => WriteNoc::Noc1,
        }
    }
}

/// The resident mover on one tile.
pub struct DataMover<N: NocId> {
    tile: NocCoord<N>,
    mover: Mover,
    usable: u8,
    seq: u32,
    /// Lists enqueued so far (`tt_isa::dm::QUEUE_HEAD`).
    head: u32,
    /// Enqueued lists not yet seen finished: `(number, first entry,
    /// entries)`, oldest first.
    in_flight: std::collections::VecDeque<(u32, u32, u32)>,
    /// The ring entry the next list is written from.
    write_at: u32,
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
        Self::start_on(d, w, tile, dram, Mover::B, image)
    }

    /// [`DataMover::start`] on `mover`'s core: RISCV B, or RISCV NC with
    /// `tt_firmware_images::DM_NC`'s bytes, loaded at the top of L1 and
    /// reached through the stub at NC's reset PC
    /// (`Device::load_and_start_nc`). Each mover has its own mailbox, ring and
    /// scratch (`tt_isa::dm::Mover`), so B's and NC's run side by side.
    pub fn start_on<T: Transport>(
        d: &mut Device<T>,
        w: &Window,
        tile: NocCoord<N>,
        dram: &Dram,
        mover: Mover,
        image: &[u8],
    ) -> Result<Self> {
        if image.len() as u64 > mover.image_max {
            return Err(DmError::NotStarted);
        }
        d.write32(w, tile, mover.at(dm::MY_X), tile.x() as u32)?;
        d.write32(w, tile, mover.at(dm::MY_Y), tile.y() as u32)?;
        d.write32(w, tile, mover.at(dm::USABLE), dram.usable_mask() as u32)?;
        // L1 survives between processes: a stale `TRACE` from a profiled run
        // would have the mover store to a timestamper ttsim does not model.
        for word in [
            mover.at(dm::SEQ),
            mover.at(dm::DONE),
            mover.at(dm::ERROR),
            mover.at(dm::TRACE),
            mover.at(dm::THROTTLE_STALLS),
            mover.at(dm::THROTTLE_CYCLES),
            mover.at(dm::WRITE_NOC),
            mover.at(dm::READ_FAST),
            mover.at(dm::QUEUE_HEAD),
            mover.at(dm::QUEUE_DONE),
            mover.at(dm::QUEUE_ERROR),
            mover.at(dm::QUEUE_ERROR_AT),
            mover.at(dm::TRACE_PROGRESS),
        ] {
            d.write32(w, tile, word, 0)?;
        }
        d.write32(w, tile, mover.at(dm::IN_FLIGHT_CAP), dm::TILE_IN_FLIGHT_CAP)?;
        let status_at = mover.mailbox + offset::STATUS;
        d.write32(w, tile, status_at, 0)?;
        match mover.core {
            tt_isa::tensix::Core::NC => d.load_and_start_nc(w, tile, image, mover.image_base)?,
            core => d.load_and_start(w, tile, core, image, mover.image_base)?,
        }
        let started = d.wait_for_mailbox(
            w,
            tile,
            status_at,
            mover.mailbox + offset::PANIC_CODE,
            1_000_000,
            |s| s == status::RUNNING,
        )?;
        started.map_err(|_| DmError::NotStarted)?;
        Ok(DataMover {
            tile,
            mover,
            usable: dram.usable_mask(),
            seq: 0,
            head: 0,
            in_flight: Default::default(),
            write_at: 0,
            deadline: Duration::from_secs(1),
        })
    }

    pub fn tile(&self) -> NocCoord<N> {
        self.tile
    }

    /// Which core's mover this is, and where its mailbox and ring are.
    pub fn mover(&self) -> Mover {
        self.mover
    }

    /// What the in-flight cap (`tt_isa::noc::niu::InFlight`) has cost this
    /// mover since it started: the requests that waited for room, and the tile
    /// cycles they waited. Two PCIe reads; not on any hot path.
    pub fn throttle<T: Transport>(&self, d: &mut Device<T>, w: &Window) -> Result<Throttle> {
        Ok(Throttle {
            stalls: d.read32(w, self.tile, self.mover.at(dm::THROTTLE_STALLS))?,
            cycles: d.read32(w, self.tile, self.mover.at(dm::THROTTLE_CYCLES))?,
        })
    }

    /// Cap this mover's requests in flight at `cap` from its next list: 0 for
    /// `tt_isa::noc::niu::MAX_IN_FLIGHT`, otherwise clamped to
    /// `1..=MAX_IN_FLIGHT`. A mover starts at `dm::TILE_IN_FLIGHT_CAP`; this is
    /// for the gates that force the throttle and the benchmarks that compare
    /// caps.
    pub fn set_in_flight_cap<T: Transport>(
        &self,
        d: &mut Device<T>,
        w: &Window,
        cap: u32,
    ) -> Result<()> {
        d.write32(w, self.tile, self.mover.at(dm::IN_FLIGHT_CAP), cap)?;
        Ok(())
    }

    /// Issue this mover's GDDR reads through the fast path from its next list
    /// (`tt_isa::dm::READ_FAST`), or back through the path that writes every
    /// initiator register per request. Off at start. Only RISCV B reads: NC's
    /// image ignores it.
    ///
    /// The fast path is `noc_async_read_set_state` / `_with_state` on request
    /// initiator 1 of NoC #0 (`NoC/MemoryMap.md`, "NIU Request Initiators"): the
    /// words that do not change between reads are written once per list and
    /// each request writes only the address words and what changed. Its output
    /// is gated byte-identical to the other path's on the simulator and on card
    /// 0 (`step118_mover_fast_path`: initiator 1 keeps its registers). It saves
    /// about 17 ns of the ~0.48 us a request costs, 3-4% on every shape
    /// (`silicon_perf::mover_read_fast_path`), so it stays opt-in.
    pub fn set_read_fast<T: Transport>(
        &self,
        d: &mut Device<T>,
        w: &Window,
        fast: bool,
    ) -> Result<()> {
        d.write32(w, self.tile, self.mover.at(dm::READ_FAST), u32::from(fast))?;
        Ok(())
    }

    /// Send this mover's GDDR writes out through `noc` from its next list
    /// (`tt_isa::dm::WRITE_NOC`): NoC #0 with the reads, NoC #1 on port 1 of
    /// every channel, or each in turn. Reads and barriers stay on NoC #0.
    /// Only NC's image has the NoC #1 path: B refuses anything but NoC #0.
    pub fn set_write_noc<T: Transport>(
        &self,
        d: &mut Device<T>,
        w: &Window,
        noc: impl Into<WriteNoc>,
    ) -> Result<()> {
        let noc = noc.into();
        if noc != WriteNoc::Noc0 && self.mover != dm::Mover::NC {
            return Err(DmError::WriteNoc {
                mover: self.mover,
                noc,
            });
        }
        let word = match noc {
            WriteNoc::Noc0 => dm::write_noc::NOC0,
            WriteNoc::Noc1 => dm::write_noc::NOC1,
            WriteNoc::Alternate => dm::write_noc::ALTERNATE,
        };
        d.write32(w, self.tile, self.mover.at(dm::WRITE_NOC), word)?;
        Ok(())
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
        // The mover's own checks, run first, so a bad entry costs no PCIe.
        self.check(entries)?;
        let bytes: Vec<u8> = entries
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        d.l1_write(w, self.tile, self.mover.list, &bytes)?;
        d.write32(w, self.tile, self.mover.at(dm::OP), op::LIST)?;
        d.write32(w, self.tile, self.mover.at(dm::LEN), entries.len() as u32)?;
        self.seq = self.seq.wrapping_add(1).max(1);
        d.write32(w, self.tile, self.mover.at(dm::SEQ), self.seq)?;
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
            (self.mover.at(dm::OP), op),
            (self.mover.at(dm::CHANNEL), range.channel().index() as u32),
            (self.mover.at(dm::PORT), port as u32),
            (self.mover.at(dm::DRAM_OFFSET), range.offset() as u32),
            (self.mover.at(dm::L1_ADDR), l1),
            (self.mover.at(dm::LEN), range.len() as u32),
        ];
        // The mover's own check, run first, so a bad descriptor costs no PCIe.
        let [(_, o), (_, c), (_, p), (_, off), (_, a), (_, n)] = words;
        if !self.mover.permits(o) {
            return Err(DmError::Invalid(dm::error::DIRECTION));
        }
        Descriptor::decode(self.usable as u32, o, c, p, off, a, n).map_err(DmError::Invalid)?;
        for (at, v) in words {
            d.write32(w, self.tile, at, v)?;
        }
        self.seq = self.seq.wrapping_add(1).max(1);
        d.write32(w, self.tile, self.mover.at(dm::SEQ), self.seq)?;
        self.wait(d, w)
    }

    /// Check `entries` as the mover will, so a bad entry costs no PCIe: a
    /// plain entry decoded, a record expanded and every entry it makes decoded.
    fn check(&self, entries: &[[u32; 8]]) -> Result<()> {
        self.check_as(self.mover, entries)
    }

    /// [`DataMover::check`] for `mover`'s direction: a packet's writer
    /// section is NC's, whichever mover it is queued on.
    fn check_as(&self, mover: Mover, entries: &[[u32; 8]]) -> Result<()> {
        if entries.is_empty() || entries.len() > dm::LIST_MAX as usize {
            return Err(DmError::Invalid(dm::error::LENGTH));
        }
        let usable = self.usable as u32;
        if entries[0][0] == op::PAIR {
            if self.mover != Mover::B
                || tt_isa::dataflow::packet_length(entries[0]).map_err(DmError::Invalid)?
                    != entries.len()
            {
                return Err(DmError::Invalid(dm::error::LENGTH));
            }
            self.check_as(Mover::B, &entries[1..1 + entries[0][1] as usize])?;
            let writer_end = 1 + entries[0][1] as usize + entries[0][2] as usize;
            self.check_as(Mover::NC, &entries[1 + entries[0][1] as usize..writer_end])?;
            check_pair(&entries[..writer_end]).map_err(DmError::Invalid)?;
            if entries[0][4] != 0 {
                if entries[writer_end][0] != op::BARRIER {
                    return Err(DmError::Invalid(dm::error::OP));
                }
                self.check_as(Mover::B, &entries[writer_end..])?;
            }
            return Ok(());
        }
        let mut i = 0;
        while i < entries.len() {
            let n = record::len(entries[i][0]);
            if !mover.permits(entries[i][0]) {
                return Err(DmError::Invalid(dm::error::DIRECTION));
            }
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
        Ok(())
    }

    /// Queue a list (`tt_isa::self.mover.at(dm::QUEUE_HEAD)`) and return its number, without
    /// waiting for it -- only, if the ring or the slots are full, for the
    /// oldest lists to finish. [`DataMover::wait_for`] waits for a number.
    /// The single-descriptor path ([`DataMover::submit_list`]) must not be in
    /// use meanwhile.
    pub fn enqueue<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        entries: &[[u32; 8]],
    ) -> Result<u32> {
        self.check(entries)?;
        self.enqueue_checked(d, w, entries)
    }

    /// Check `entries` as [`DataMover::enqueue`] will before it writes them:
    /// the same decode, so a caller that times the check apart from the queue
    /// ([`DataMover::enqueue_checked`]) refuses exactly what `enqueue` would.
    pub fn check_list(&self, entries: &[[u32; 8]]) -> Result<()> {
        self.check(entries)
    }

    /// [`DataMover::enqueue`] for entries [`DataMover::check_list`] passed on
    /// this mover just now. Nothing checks them again: an unchecked list can
    /// hang a card, so a caller passes the very entries it checked.
    pub fn enqueue_checked<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        entries: &[[u32; 8]],
    ) -> Result<u32> {
        let n = entries.len() as u32;
        let mut refreshed = false;
        let first = loop {
            // The host's own record of what is in flight is conservative --
            // a list it has not seen finish holds its slot -- so it only
            // reads the queue's progress (two uncached PCIe reads, most of an
            // enqueue's cost on many tiles, checklist 9.17) when that record
            // says the ring or the slots are full. A failed list is reported
            // at the next wait or sync instead.
            if let Some(at) = self.room(n) {
                if self.in_flight.len() < dm::QUEUE_LEN as usize {
                    break at;
                }
            }
            if !refreshed {
                self.refresh(d, w)?;
                refreshed = true;
                continue;
            }
            // Full: wait for the oldest list, which frees its slot and entries.
            let oldest = self
                .in_flight
                .front()
                .expect("full means something in flight")
                .0;
            self.wait_for(d, w, oldest)?;
        };
        let bytes: Vec<u8> = entries
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        d.l1_write(
            w,
            self.tile,
            self.mover.list + first as u64 * dm::ENTRY_BYTES,
            &bytes,
        )?;
        let slot = self.mover.at(dm::QUEUE_SLOTS) + (self.head % dm::QUEUE_LEN) as u64 * 4;
        d.write32(w, self.tile, slot, dm::queue_slot(first, n))?;
        self.head = self.head.wrapping_add(1);
        d.write32(w, self.tile, self.mover.at(dm::QUEUE_HEAD), self.head)?;
        self.in_flight.push_back((self.head, first, n));
        self.write_at = first + n;
        Ok(self.head)
    }

    fn room(&self, n: u32) -> Option<u32> {
        ring_room(&self.in_flight, self.write_at, n)
    }

    /// Read how far the queue has got, forgetting the lists that finished;
    /// a failed list is reported, with its number.
    pub fn refresh<T: Transport>(&mut self, d: &mut Device<T>, w: &Window) -> Result<u32> {
        if self.in_flight.is_empty() {
            return Ok(self.head);
        }
        let error = d.read32(w, self.tile, self.mover.at(dm::QUEUE_ERROR))?;
        if error != dm::error::NONE {
            let at = d.read32(w, self.tile, self.mover.at(dm::QUEUE_ERROR_AT))?;
            return Err(DmError::Queued {
                list: at,
                code: error,
            });
        }
        let done = d.read32(w, self.tile, self.mover.at(dm::QUEUE_DONE))?;
        while let Some(&(number, _, _)) = self.in_flight.front() {
            if (done.wrapping_sub(number) as i32) >= 0 {
                self.in_flight.pop_front();
            } else {
                break;
            }
        }
        Ok(done)
    }

    /// Wait until list `number` (from [`DataMover::enqueue`]) has finished.
    /// Fails if a list fails, or if the queue makes no progress for the
    /// deadline (on ttsim, for a budget of simulated cycles).
    pub fn wait_for<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        number: u32,
    ) -> Result<()> {
        let simulated = d.transport().is_simulated();
        let (mut last, mut since, mut ticks) = (self.refresh(d, w)?, Instant::now(), 0u64);
        let mut trace_progress = d.read32(w, self.tile, self.mover.at(dm::TRACE_PROGRESS))?;
        loop {
            let done = self.refresh(d, w)?;
            if (done.wrapping_sub(number) as i32) >= 0 {
                return Ok(());
            }
            if done != last {
                (last, since, ticks) = (done, Instant::now(), 0);
            }
            let stuck = if simulated {
                ticks > 50_000_000
            } else {
                since.elapsed() > self.deadline
            };
            if stuck {
                // A replay is one queue entry, but may contain thousands of
                // kernels and copies. Completed chunks establish progress
                // without extending the budget for an actually stalled chunk.
                let progress = d.read32(w, self.tile, self.mover.at(dm::TRACE_PROGRESS))?;
                if progress != trace_progress {
                    trace_progress = progress;
                    (since, ticks) = (Instant::now(), 0);
                    continue;
                }
                return Err(DmError::TimedOut { seq: number, done });
            }
            d.tick(CYCLES_PER_POLL);
            ticks += CYCLES_PER_POLL as u64;
        }
    }

    /// Wait for every list enqueued so far.
    pub fn drain<T: Transport>(&mut self, d: &mut Device<T>, w: &Window) -> Result<()> {
        self.wait_for(d, w, self.head)
    }

    /// Are lists enqueued and not yet seen finished?
    pub fn busy(&self) -> bool {
        !self.in_flight.is_empty()
    }

    /// Wait for the last descriptor submitted to finish, and report its error.
    /// Returns at once if nothing is outstanding.
    pub fn wait<T: Transport>(&self, d: &mut Device<T>, w: &Window) -> Result<()> {
        let started = Instant::now();
        let simulated = d.transport().is_simulated();
        let mut ticks = 0u64;
        loop {
            let done = d.read32(w, self.tile, self.mover.at(dm::DONE))?;
            if done == self.seq {
                return match d.read32(w, self.tile, self.mover.at(dm::ERROR))? {
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

    /// Hold the mover's core in reset again.
    pub fn stop<T: Transport>(self, d: &mut Device<T>, w: &Window) -> Result<()> {
        d.set_core_reset(w, self.tile, self.mover.core, true)?;
        Ok(())
    }
}

/// A shared packet's ownership rules, beyond what each entry's decode checks:
/// every credit names the header's capacity (the role scripts carry it too,
/// from the same header), and the writer section only writes to GDDR, so NC
/// stays on NoC #1 -- an NC read would share B's NoC #0 initiator. `entries`
/// is the header, the reader section and the writer section.
pub fn check_pair(entries: &[[u32; 8]]) -> std::result::Result<(), u32> {
    let header = entries.first().ok_or(dm::error::LENGTH)?;
    let capacity = header[3];
    let readers = header[1] as usize;
    if entries.len() != 1 + readers + header[2] as usize {
        return Err(dm::error::LENGTH);
    }
    let endpoint = |writer: bool| {
        if writer {
            Endpoint::Writer
        } else {
            Endpoint::Reader
        }
    };
    // One pass over entry heads: the sections were each checked entry by
    // entry already (`DataMover::check`), so a record is judged by its kind.
    let each = |section: &[[u32; 8]], writer: bool| -> std::result::Result<(), u32> {
        let mut index = 0;
        while index < section.len() {
            let head = section[index];
            let count = record::len(head[0]);
            if count > 1 {
                // Only the L1 -> GDDR records may be the writer's.
                if writer && record::direction(head[0]) != Some(record::Direction::Write) {
                    return Err(dm::error::OP);
                }
            } else {
                match Entry::decode(u32::MAX, head)? {
                    // The reader holds the producer's credits (input,
                    // transfer) and the writer the consumer's (output,
                    // transfer), every one at the header's capacity.
                    Entry::Buffer {
                        stream,
                        action,
                        capacity: c,
                    } if c as u32 == capacity
                        && Channel::of(stream, c)
                            .is_some_and(|ch| ch.permits(endpoint(writer), action)) => {}
                    Entry::Buffer { .. } => return Err(dm::error::OP),
                    Entry::Move { descriptor, .. } if writer && descriptor.op == op::WRITE => {}
                    Entry::CheckFlags { .. } if writer => {}
                    _ if writer => return Err(dm::error::OP),
                    _ => {}
                }
            }
            index += count;
        }
        Ok(())
    };
    each(&entries[1..1 + readers], false)?;
    each(&entries[1 + readers..], true)
}

#[cfg(test)]
mod ring {
    use super::ring_room;
    use std::collections::VecDeque;
    use tt_isa::dm::LIST_MAX;

    /// Lists go after the newest, wrap to the start once the end is too
    /// near, never overlap a list in flight, and never cross the end.
    #[test]
    fn lists_fit_beside_what_is_in_flight() {
        let mut q = VecDeque::new();
        assert_eq!(ring_room(&q, 77, 10), Some(0), "empty: from the start");
        q.push_back((1, 0, 300));
        assert_eq!(ring_room(&q, 300, 200), Some(300));
        assert_eq!(
            ring_room(&q, 300, 213),
            None,
            "past the end, and nothing at the start"
        );
        q.push_back((2, 300, 200));
        q.pop_front(); // list 1 done: 0..300 free
        assert_eq!(ring_room(&q, 500, 100), Some(0), "wrapped");
        q.push_back((3, 0, 100));
        assert_eq!(ring_room(&q, 100, 200), Some(100), "up to the oldest");
        assert_eq!(ring_room(&q, 100, 201), None, "into the oldest");
        // A random soak: whatever is placed never overlaps what is in flight.
        let mut s = 12345u32;
        let (mut q, mut at, mut number) = (VecDeque::new(), 0u32, 0u32);
        for _ in 0..10_000 {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = 1 + (s >> 8) % 200;
            if s % 3 == 0 && !q.is_empty() {
                q.pop_front();
                continue;
            }
            if let Some(first) = ring_room(&q, at, n) {
                assert!(first + n <= LIST_MAX);
                for &(_, f, len) in &q {
                    assert!(
                        first + n <= f || f + len <= first,
                        "{first}+{n} over {f}+{len}"
                    );
                }
                number += 1;
                q.push_back((number, first, n));
                at = first + n;
            } else {
                q.pop_front();
            }
        }
    }
}
