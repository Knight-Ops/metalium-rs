//! Traces: a stretch of device ops captured once and replayed without the host
//! (`hardware-coverage.md` X4d; tt-metal's `BeginTraceCapture`/`ReplayTrace`
//! the model).
//!
//! `Session::begin_trace` ... `Session::end_trace` around any ops captures what
//! the session enqueued on every unit -- and runs it, so the capture is also
//! the first run. `Session::replay` runs it again: one list per unit, a single
//! `CALL` entry (`tt_isa::dm::op::CALL`), and each data mover streams the
//! captured entries from GDDR. Between replays, `Session::write` puts new
//! values into the tensors the trace reads (an inference's input), and the
//! tensors it wrote hold the results.
//!
//! What a replay cannot change, the capture makes position-independent: each
//! kernel's generation and each barrier's target are offsets the `CALL`
//! supplies; the roles' descriptors are `POKE` entries in the stream, since
//! the host writes none during a replay; and the semaphore setups the host ran
//! between kernels are kernels of their own in it.
//!
//! And what tt-metal leaves to its users is enforced here (each a
//! [`TraceError`], never a silent corruption):
//!
//! * a trace holds every allocation that existed when its capture ended: a
//!   free of one is deferred until the trace is released, so nothing else
//!   lands in slots it reads or writes. (Precise without tracking each op's
//!   tensors: frees during the capture are deferred too, so anything
//!   allocated after it lies in what was free then -- [`FreeSnapshot`].)
//! * the programs its kernels name stay in the program cache while it lives;
//! * a tile reset or a mover restart since the capture makes it
//!   [`TraceError::Stale`];
//! * a download, a [`Session::write`] or a host-run kernel during a capture is
//!   [`TraceError::HostTransfer`]: a replay cannot repeat host work between
//!   device ops. (An upload is allowed: a constant, which the replay reads
//!   where the capture left it.)
//! * the tile state the host keeps track of -- descriptors, semaphores -- is
//!   forgotten at a capture's start, so the capture records everything its
//!   kernels need, and after a replay, so nothing queued later trusts it.
//!
//! The capture keeps each op's record ([`OpRecord`]) beside the words: the
//! seam for optimizing a captured graph later (X4e).
//!
//! [`Session::write`]: crate::session::Session::write
//! [`FreeSnapshot`]: crate::tensor::FreeSnapshot

use tt_isa::dm::{self, op, record};

use tt_isa::mailbox::{self, role::Mailbox, DESCRIPTOR_WORDS};

use crate::tensor::{FreeSnapshot, Placement};

/// Why a trace operation was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceError {
    /// A capture is already open.
    Capturing,
    /// No capture is open.
    NotCapturing,
    /// No trace by this id (never made, or released).
    Unknown(u64),
    /// The tiles have been reset or a mover restarted since the capture: the
    /// programs, descriptors and counters it was made against are gone.
    Stale,
    /// A host transfer during the capture -- `what` -- which a replay could
    /// not repeat.
    HostTransfer(&'static str),
    /// Traces queue on the movers: the session must be batching
    /// (`Session::set_batching`).
    NotBatching,
    /// A kernel whose programs the program cache does not admit: the host
    /// stages those itself, which a replay cannot.
    NotResident,
    /// A capture with nothing in it.
    Empty,
}

impl core::fmt::Display for TraceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TraceError::Capturing => write!(f, "a trace capture is already open"),
            TraceError::NotCapturing => write!(f, "no trace capture is open"),
            TraceError::Unknown(id) => write!(f, "no trace {id}: never made, or released"),
            TraceError::Stale => write!(
                f,
                "the trace is stale: a tile was reset or a data mover restarted since it was \
                 captured, so the programs and counters it names are gone; capture it again"
            ),
            TraceError::HostTransfer(what) => write!(
                f,
                "a {what} during a trace capture: a replay runs device work only and could \
                 not repeat it; end the capture before reading results"
            ),
            TraceError::NotBatching => write!(
                f,
                "traces run from the data movers' queues: turn batching on (Session::set_batching)"
            ),
            TraceError::NotResident => write!(
                f,
                "a kernel too large for the program cache: the host stages it itself, which \
                 a replay cannot do"
            ),
            TraceError::Empty => write!(f, "the capture ran no device work"),
        }
    }
}

/// A trace, by number.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TraceId(pub(crate) u64);

/// What one captured op was: the graph a later pass can optimize over.
#[derive(Clone, Debug)]
pub struct OpRecord {
    /// Its first list's name (`tensor::stats`).
    pub what: &'static str,
    /// Per unit, the range of that unit's stream its lists took.
    pub entries: Vec<core::ops::Range<usize>>,
    /// Whether a barrier followed it on every unit.
    pub barrier: bool,
}

/// One unit's share of a capture in progress.
#[derive(Default)]
pub(crate) struct UnitCapture {
    pub(crate) regions: Vec<Placement>,
    /// The unit's entries, in order, generations and barrier targets relative.
    pub(crate) stream: Vec<[u32; 8]>,
    /// The resident's last generation when the capture began: the stream's
    /// generations are counted from it.
    pub(crate) generation_base: u32,
    /// What the stream has set each role's descriptor to so far.
    pub(crate) descriptors: [Option<[u32; DESCRIPTOR_WORDS]>; 3],
    /// Program-cache holds for the trace, one per hold: the cache's
    /// generation when taken ([`crate::program_cache::ProgramCache::generation`])
    /// and the address.
    pub(crate) held_programs: Vec<(u64, u64)>,
    /// Entries in each reader and writer section stored for the stream's
    /// `PAIR_CALL`s, chunk padding included: reader, then writer, per packet.
    pub(crate) sections: Vec<u32>,
}

impl UnitCapture {
    /// `POKE`s setting role `thread`'s descriptor to `d`, word by word where
    /// the stream has not already: all but the program's address and length,
    /// which each `KERNEL` entry writes itself.
    pub(crate) fn poke_descriptor(&mut self, thread: usize, d: &mailbox::Descriptor) {
        let mb = Mailbox::of(thread as u32);
        let words = d.writes(mb);
        let have = self.descriptors[thread];
        for (k, &(at, v)) in words.iter().enumerate() {
            if at == mb.program_len() || at == mb.program_addr() {
                continue;
            }
            if have.is_none_or(|h| h[k] != v) {
                self.stream.push([op::POKE, at as u32, v, 0, 0, 0, 0, 0]);
            }
        }
        self.descriptors[thread] = Some(words.map(|(_, v)| v));
    }
}

/// A capture in progress.
pub(crate) struct Capture {
    /// The session's epoch ([`TraceError::Stale`]) when it began.
    pub(crate) epoch: u64,
    /// The session's barrier count when it began.
    pub(crate) barriers_base: u32,
    /// Barriers captured.
    pub(crate) barriers: u32,
    pub(crate) units: Vec<UnitCapture>,
    /// Frees during the capture, deferred to the trace's release.
    pub(crate) freed: Vec<Placement>,
    pub(crate) ops: Vec<OpRecord>,
    /// An op failed part-way through being captured.
    pub(crate) failed: bool,
}

/// One unit's share of a finished trace.
pub(crate) struct UnitTrace {
    pub(crate) regions: Vec<Placement>,
    /// Its stream in GDDR, and where that is: the chip's channel index, the
    /// offset, the entry count.
    pub(crate) stream: Placement,
    pub(crate) channel: u32,
    pub(crate) offset: u32,
    pub(crate) count: u32,
    /// Generations a replay takes on the unit.
    pub(crate) generations: u32,
    /// As [`UnitCapture::held_programs`].
    pub(crate) held_programs: Vec<(u64, u64)>,
    /// As [`UnitCapture::sections`].
    pub(crate) sections: Vec<u32>,
}

/// A finished trace: see the module documentation.
pub(crate) struct Trace {
    pub(crate) epoch: u64,
    pub(crate) units: Vec<Option<UnitTrace>>,
    /// Barriers a replay takes.
    pub(crate) barriers: u32,
    /// GDDR's free space when the capture ended: whatever was not free, the
    /// trace holds.
    pub(crate) free_at_end: FreeSnapshot,
    /// Frees of what it holds, deferred to its release.
    pub(crate) freed: Vec<Placement>,
    pub(crate) ops: Vec<OpRecord>,
}

impl Trace {
    /// Must a free of `p` wait for this trace's release?
    pub(crate) fn holds(&self, p: &Placement) -> bool {
        !self.free_at_end.was_free(p)
    }
}

/// Lay a unit's entries out for streaming: a `WAIT` wherever a record would
/// otherwise cross a chunk of [`dm::TRACE_CHUNK_ENTRIES`], since the mover
/// runs a chunk at a time (`dm::op::CALL`).
pub(crate) fn chunked(entries: &[[u32; 8]]) -> Vec<[u32; 8]> {
    let chunk = dm::TRACE_CHUNK_ENTRIES as usize;
    let wait = [op::WAIT, 0, 0, 0, 0, 0, 0, 0];
    let mut out = Vec::with_capacity(entries.len() + entries.len() / chunk + 1);
    let mut i = 0;
    while i < entries.len() {
        let n = record::len(entries[i][0]).max(1);
        if out.len() % chunk + n > chunk {
            while out.len() % chunk != 0 {
                out.push(wait);
            }
        }
        out.extend_from_slice(&entries[i..i + n]);
        i += n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_record_crosses_a_chunk() {
        let chunk = dm::TRACE_CHUNK_ENTRIES as usize;
        // A stream of WAITs with a five-entry record placed to straddle the
        // first chunk's end.
        let mut v = vec![[op::WAIT, 0, 0, 0, 0, 0, 0, 0]; chunk - 2];
        let rec_len = record::len(record::GATHER);
        let mut rec = vec![[0u32; 8]; rec_len];
        rec[0][0] = record::GATHER;
        v.extend_from_slice(&rec);
        let out = chunked(&v);
        let at = out.iter().position(|e| e[0] == record::GATHER).unwrap();
        assert_eq!(at, chunk, "the record moved to the next chunk");
        assert!(out[chunk - 2..chunk].iter().all(|e| e[0] == op::WAIT));
        assert_eq!(out.len(), chunk + rec_len);
    }

    /// A region's `LAUNCH` and the `KERNEL_WAIT` that joins it, with more than a
    /// chunk of mover entries between: they fall in different chunks, in order,
    /// with their pairing (word 1, the generation) and every entry between
    /// untouched. The firmware keeps the active generation across chunks.
    #[test]
    fn a_launch_and_its_wait_can_fall_in_different_chunks() {
        let chunk = dm::TRACE_CHUNK_ENTRIES as usize;
        let launch = [op::LAUNCH, 7, 16, 1, 32, 1, 48, 1];
        let wait = [op::KERNEL_WAIT, 7, 0, 0, 0, 0, 0, 0];
        let rec_len = record::len(record::GATHER);
        let mut stream = vec![launch];
        // Enough five-entry records to fill two chunks and a part.
        for tag in 0..(2 * chunk / rec_len + 3) {
            let mut rec = vec![[0u32; 8]; rec_len];
            rec[0] = [record::GATHER, tag as u32, 0, 0, 0, 0, 0, 0];
            stream.extend_from_slice(&rec);
        }
        stream.push(wait);
        let out = chunked(&stream);
        let at = |entry: [u32; 8]| out.iter().position(|e| *e == entry).unwrap();
        let (launched, waited) = (at(launch), at(wait));
        assert!(launched / chunk < waited / chunk, "{launched} {waited}");
        assert!(waited - launched > chunk);
        // Nothing was dropped or reordered: what is not padding is the stream.
        let kept: Vec<_> = out
            .iter()
            .filter(|e| **e != [op::WAIT, 0, 0, 0, 0, 0, 0, 0])
            .copied()
            .collect();
        assert_eq!(kept, stream);
        // No record starts in one chunk and ends in the next.
        let mut i = 0;
        while i < out.len() {
            let n = record::len(out[i][0]).max(1);
            assert_eq!(i / chunk, (i + n - 1) / chunk, "record at {i}");
            i += n;
        }
    }
}
