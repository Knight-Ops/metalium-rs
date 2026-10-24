//! Resident kernel programs: the host's mirror of one tile's program cache.
//!
//! A role program is a few hundred to eight thousand instruction words, and a
//! tile used to hold exactly one per role, rewritten whenever consecutive
//! kernels differed -- which, for the blocks of one matmul, they do at every
//! ragged edge. [`ProgramCache`] keeps any number of programs resident in
//! `tt_isa::l1::PROGRAM_CACHE`, and a `KERNEL` list entry names which one each
//! role runs (`tt_isa::dm::op::KERNEL`), so a block of a shape the tile has
//! seen costs no upload and no new list.
//!
//! The host wrote every byte of the region, so it knows exactly what is there;
//! the device keeps no bookkeeping. The policy, chosen against the working
//! sets measured in the checklist (an MNIST step's kernels: 117 KB on one
//! tile, 34 KB on eight, in a 252 KB region (240 KB since the trace buffer grew, X3)):
//!
//! * **Keyed by the program's words**, compared exactly, as the fixed slots
//!   already were.
//! * **First fit**, coalescing on free.
//! * **Least recently used** evicted first.
//! * **Pinned** while a list in flight names it: [`ProgramCache::place`] pins,
//!   [`ProgramCache::unpin_all`] releases once the list has finished. Evicting a
//!   program a submitted list still names would run whatever replaced it.
//! * **Held** by the traces that name it, which only orders eviction: an unheld
//!   program goes first, but a held one goes too when nothing else makes room. A
//!   trace keeps its programs' words and puts back what is missing before it
//!   replays ([`ProgramCache::restore`]), so the traces of a generator need not
//!   fit the cache together, only each alone (`docs/plans/long-context.md`).
//! * **Protected** while the capture in progress holds it
//!   ([`ProgramCache::protect`]): a trace's own programs must all be resident at
//!   once, so a capture never evicts one it took.
//! * **Admission**: a program larger than half the region is not cached
//!   ([`Placed::Bypass`]); it runs from the fixed slot, as before, so one huge
//!   kernel cannot flush everything else.
//! * **Invalidated** whenever the tile is reset ([`ProgramCache::clear`]).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use tt_isa::l1::Region;

/// What a cache has done, for a gate or a profile.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes_uploaded: u64,
    pub evictions: u64,
    /// Programs too large to admit.
    pub bypassed: u64,
}

/// Where a program is, and whether the caller must write it there.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Placed {
    /// Already resident at this address.
    Hit(u64),
    /// Given this address; the caller writes the program there before any
    /// list names it.
    Upload(u64),
    /// Not cached: run it from the fixed slot.
    Bypass,
}

/// Why a program could not be placed.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CacheError {
    /// Everything resident is pinned, and what is free is not enough.
    Full { bytes: u64 },
}

struct Resident {
    words: Arc<[u32]>,
    hash: u64,
    at: u64,
    last_use: u64,
    pinned: bool,
    /// Traces that name it (`crate::trace`): evicted after every unheld program.
    held: u32,
    /// Held by the capture in progress: never evicted until it ends.
    protected: bool,
}

/// One tile's program cache: see the module documentation.
pub struct ProgramCache {
    region: Region,
    /// Free byte ranges, `offset -> bytes`, coalesced.
    free: BTreeMap<u64, u64>,
    resident: Vec<Resident>,
    by_hash: HashMap<u64, Vec<usize>>,
    clock: u64,
    stats: CacheStats,
    /// How many times the cache was forgotten ([`ProgramCache::clear`]): a
    /// hold taken before one names a program that is no longer there.
    clears: u64,
    /// Moved on whenever a program may have left or lost its holds (an eviction, a
    /// release, a clear): a trace restored at the same count needs no restoring.
    changes: u64,
}

/// Is a role program of `words` words cached on a tile, rather than run from
/// its fixed slot? The admission rule of a cache over
/// `tt_isa::l1::PROGRAM_CACHE`, known before any cache exists.
pub fn admitted(words: usize) -> bool {
    words > 0 && words as u64 * 4 <= tt_isa::l1::PROGRAM_CACHE.len() / 2
}

/// Programs start on 16-byte boundaries (`tt_isa::dm::Entry::decode` refuses
/// anything else).
const ALIGN: u64 = 16;

/// The key [`ProgramCache::place_hashed`] takes with a program's words.
pub fn hash(words: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    words.hash(&mut h);
    h.finish()
}

impl ProgramCache {
    /// An empty cache over `region` (`tt_isa::l1::PROGRAM_CACHE` on a tile).
    pub fn new(region: Region) -> Self {
        let base = region.base.next_multiple_of(ALIGN);
        ProgramCache {
            region,
            free: BTreeMap::from([(base, region.end - base)]),
            resident: Vec::new(),
            by_hash: HashMap::new(),
            clock: 0,
            stats: CacheStats::default(),
            clears: 0,
            changes: 0,
        }
    }

    /// Will a program of `words` words be cached rather than bypassed?
    pub fn admits(&self, words: usize) -> bool {
        words > 0 && words as u64 * 4 <= self.region.len() / 2
    }

    /// The region the cache covers.
    pub fn region(&self) -> Region {
        self.region
    }

    /// Place `words`, pinning it until [`ProgramCache::unpin_all`].
    pub fn place(&mut self, words: &[u32]) -> Result<Placed, CacheError> {
        self.place_hashed(&Arc::from(words), hash(words))
    }

    /// [`ProgramCache::place`] of words whose [`hash`] the caller kept: a
    /// hit on the same allocation costs no pass over the words (a role
    /// program is up to eight thousand of them, and every kernel of every
    /// list looks its three up).
    pub fn place_hashed(&mut self, words: &Arc<[u32]>, h: u64) -> Result<Placed, CacheError> {
        self.place_hashed_evicting(words, h, false)
    }

    /// [`ProgramCache::place_hashed`], evicting programs traces hold too when `held_too` (and
    /// nothing else makes room): only where nothing in flight can name one, since a queued
    /// replay does not pin its programs.
    pub fn place_hashed_evicting(
        &mut self,
        words: &Arc<[u32]>,
        h: u64,
        held_too: bool,
    ) -> Result<Placed, CacheError> {
        if !self.admits(words.len()) {
            self.stats.bypassed += 1;
            return Ok(Placed::Bypass);
        }
        self.clock += 1;
        if let Some(&i) = self.by_hash.get(&h).and_then(|v| {
            v.iter().find(|&&i| {
                let r = &self.resident[i].words;
                Arc::ptr_eq(r, words) || **r == **words
            })
        }) {
            let r = &mut self.resident[i];
            r.last_use = self.clock;
            r.pinned = true;
            self.stats.hits += 1;
            return Ok(Placed::Hit(r.at));
        }
        let bytes = (words.len() as u64 * 4).next_multiple_of(ALIGN);
        let at = loop {
            if let Some((&at, _)) = self.free.iter().find(|(_, &len)| len >= bytes) {
                break at;
            }
            // Evict the least recently used program nothing in flight names and the
            // capture in progress does not hold, those no trace holds first.
            let victim = self
                .resident
                .iter()
                .enumerate()
                .filter(|(_, r)| !r.pinned && !r.protected && (r.held == 0 || held_too))
                .min_by_key(|(_, r)| (r.held > 0, r.last_use))
                .map(|(i, _)| i)
                .ok_or(CacheError::Full { bytes })?;
            self.evict(victim);
        };
        let len = self.free.remove(&at).expect("found above");
        if len > bytes {
            self.free.insert(at + bytes, len - bytes);
        }
        self.resident.push(Resident {
            words: words.clone(),
            hash: h,
            at,
            last_use: self.clock,
            pinned: true,
            held: 0,
            protected: false,
        });
        self.by_hash
            .entry(h)
            .or_default()
            .push(self.resident.len() - 1);
        self.stats.misses += 1;
        self.stats.bytes_uploaded += words.len() as u64 * 4;
        Ok(Placed::Upload(at))
    }

    fn evict(&mut self, i: usize) {
        let r = self.resident.swap_remove(i);
        let bytes = (r.words.len() as u64 * 4).next_multiple_of(ALIGN);
        release(&mut self.free, r.at, bytes);
        self.stats.evictions += 1;
        self.changes += 1;
        self.reindex();
    }

    fn reindex(&mut self) {
        self.by_hash.clear();
        for (i, r) in self.resident.iter().enumerate() {
            self.by_hash.entry(r.hash).or_default().push(i);
        }
    }

    /// A trace names the program at `at`: it stays resident, whatever is
    /// placed after it, until [`ProgramCache::release`]. Returns whether one
    /// is there.
    pub fn hold(&mut self, at: u64) -> bool {
        match self.resident.iter_mut().find(|r| r.at == at) {
            Some(r) => {
                r.held += 1;
                true
            }
            None => false,
        }
    }

    /// Which life of the cache a hold is taken in: [`ProgramCache::clear`]
    /// starts the next, and a hold from an earlier one is gone with it.
    pub fn generation(&self) -> u64 {
        self.clears
    }

    /// [`ProgramCache::release`] for a hold taken in `generation`: nothing if
    /// the cache was cleared since, when `at` may hold another trace's program.
    pub fn release_held(&mut self, generation: u64, at: u64) {
        if generation == self.clears {
            self.release(at);
        }
    }

    /// One trace no longer names the program at `at`.
    pub fn release(&mut self, at: u64) {
        if let Some(r) = self.resident.iter_mut().find(|r| r.at == at) {
            r.held = r.held.saturating_sub(1);
            self.changes += 1;
        }
    }

    /// The count [`ProgramCache::restore`]s are stamped with: unchanged, every program restored
    /// since is still there and held.
    pub fn changes(&self) -> u64 {
        self.changes
    }

    /// The capture in progress holds the program at `at`: it stays until
    /// [`ProgramCache::unprotect_all`].
    pub fn protect(&mut self, at: u64) {
        if let Some(r) = self.resident.iter_mut().find(|r| r.at == at) {
            r.protected = true;
        }
    }

    /// The capture ended: its programs are held as any trace's are.
    pub fn unprotect_all(&mut self) {
        for r in &mut self.resident {
            r.protected = false;
        }
    }

    /// The words and hash of the program at `at`, if one is resident there.
    pub fn resident_at(&self, at: u64) -> Option<(Arc<[u32]>, u64)> {
        self.resident
            .iter()
            .find(|r| r.at == at)
            .map(|r| (r.words.clone(), r.hash))
    }

    /// Put `words` (of [`hash`] `h`) back at `at`, where a trace names it, pinned and held:
    /// [`Placed::Hit`] if it is there, else [`Placed::Upload`] once whatever lay
    /// across its bytes is evicted, the caller to write it before any list runs it.
    /// [`CacheError::Full`] if a pinned or protected program lies across them, or any does and
    /// not `evict` (a queued replay may run it: the caller drains first).
    pub fn restore(
        &mut self,
        words: &Arc<[u32]>,
        h: u64,
        at: u64,
        evict: bool,
    ) -> Result<Placed, CacheError> {
        let bytes = (words.len() as u64 * 4).next_multiple_of(ALIGN);
        self.clock += 1;
        if let Some(r) = self
            .resident
            .iter_mut()
            .find(|r| r.at == at && r.hash == h && *r.words == **words)
        {
            r.last_use = self.clock;
            r.pinned = true;
            r.held = r.held.max(1);
            self.stats.hits += 1;
            return Ok(Placed::Hit(at));
        }
        assert!(
            at % ALIGN == 0 && self.region.contains(at, bytes),
            "a program restored where none could have been placed"
        );
        let across = |r: &Resident| {
            let len = (r.words.len() as u64 * 4).next_multiple_of(ALIGN);
            r.at < at + bytes && at < r.at + len
        };
        if self
            .resident
            .iter()
            .any(|r| across(r) && (r.pinned || r.protected || !evict))
        {
            return Err(CacheError::Full { bytes });
        }
        while let Some(i) = self.resident.iter().position(across) {
            self.evict(i);
        }
        // Every byte across it is free now, in one coalesced range.
        let (&start, &len) = self
            .free
            .range(..=at)
            .next_back()
            .filter(|(&s, &l)| s + l >= at + bytes)
            .expect("the evicted bytes are free");
        self.free.remove(&start);
        if at > start {
            self.free.insert(start, at - start);
        }
        if start + len > at + bytes {
            self.free.insert(at + bytes, start + len - at - bytes);
        }
        self.resident.push(Resident {
            words: words.clone(),
            hash: h,
            at,
            last_use: self.clock,
            pinned: true,
            held: 1,
            protected: false,
        });
        self.by_hash
            .entry(h)
            .or_default()
            .push(self.resident.len() - 1);
        self.stats.misses += 1;
        self.stats.bytes_uploaded += words.len() as u64 * 4;
        Ok(Placed::Upload(at))
    }

    /// Nothing in flight names any program any more.
    pub fn unpin_all(&mut self) {
        for r in &mut self.resident {
            r.pinned = false;
        }
    }

    /// Evict everything no trace holds: room made at once, as
    /// [`ProgramCache::clear`] would, but for the programs a trace names.
    pub fn clear_unheld(&mut self) {
        while let Some(i) = self.resident.iter().position(|r| r.held == 0) {
            self.evict(i);
        }
    }

    /// Forget everything: the tile was reset, and what is in its L1 is no
    /// longer the host's to vouch for.
    pub fn clear(&mut self) {
        let (stats, clears, changes) = (self.stats, self.clears + 1, self.changes + 1);
        *self = ProgramCache::new(self.region);
        self.stats = stats;
        self.clears = clears;
        self.changes = changes;
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Bytes of programs resident.
    pub fn resident_bytes(&self) -> u64 {
        self.resident.iter().map(|r| r.words.len() as u64 * 4).sum()
    }

    /// Bytes of the program at `at`, if one is resident there.
    pub fn program_bytes(&self, at: u64) -> Option<u64> {
        self.resident
            .iter()
            .find(|r| r.at == at)
            .map(|r| r.words.len() as u64 * 4)
    }

    /// Bytes of programs some trace holds: what no placement may evict.
    pub fn held_bytes(&self) -> u64 {
        self.resident
            .iter()
            .filter(|r| r.held > 0)
            .map(|r| r.words.len() as u64 * 4)
            .sum()
    }

    /// What the cache holds, and the room left once everything the capture in progress does not
    /// protect is evicted: those bytes' largest run (placement is first fit with no compaction, so
    /// protected programs scattered over the region can refuse a program that fits the bytes).
    pub fn room(&self) -> Room {
        let size = |r: &Resident| (r.words.len() as u64 * 4).next_multiple_of(ALIGN);
        let held = self.resident.iter().filter(|r| r.held > 0).map(size).sum();
        let mut kept: Vec<(u64, u64)> = self
            .resident
            .iter()
            .filter(|r| r.protected)
            .map(|r| (r.at, size(r)))
            .collect();
        kept.sort_unstable();
        let (mut at, mut largest, mut protected) = (self.region.base.next_multiple_of(ALIGN), 0, 0);
        for &(start, bytes) in &kept {
            largest = largest.max(start.saturating_sub(at));
            protected += bytes;
            at = at.max(start + bytes);
        }
        largest = largest.max(self.region.end.saturating_sub(at));
        Room {
            region: self.region.len(),
            held,
            protected,
            largest,
        }
    }
}

/// [`ProgramCache::room`]: what the programs traces hold leave of a cache.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Room {
    /// The region's bytes.
    pub region: u64,
    /// Bytes held by traces, as placed (aligned): evictable, restored by a replay.
    pub held: u64,
    /// Bytes the capture in progress protects.
    pub protected: u64,
    /// The largest run of bytes no protected program covers.
    pub largest: u64,
}

fn release(free: &mut BTreeMap<u64, u64>, at: u64, len: u64) {
    let (mut at, mut len) = (at, len);
    if let Some((&p, &plen)) = free.range(..at).next_back() {
        if p + plen == at {
            free.remove(&p);
            at = p;
            len += plen;
        }
    }
    if let Some(&nlen) = free.get(&(at + len)) {
        free.remove(&(at + len));
        len += nlen;
    }
    free.insert(at, len);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(bytes: u64) -> Region {
        Region {
            name: "test",
            base: 0x14_1000,
            end: 0x14_1000 + bytes,
        }
    }

    fn prog(tag: u32, words: usize) -> Vec<u32> {
        (0..words as u32).map(|i| tag << 16 | i).collect()
    }

    #[test]
    fn a_second_placement_is_a_hit_at_the_same_address() {
        let mut c = ProgramCache::new(region(0x1000));
        let p = prog(1, 10);
        let Placed::Upload(at) = c.place(&p).unwrap() else {
            panic!()
        };
        c.unpin_all();
        assert_eq!(c.place(&p).unwrap(), Placed::Hit(at));
        // Different words are a different program, even of the same length.
        assert!(matches!(c.place(&prog(2, 10)).unwrap(), Placed::Upload(b) if b != at));
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.bytes_uploaded), (1, 2, 80));
    }

    #[test]
    fn the_least_recently_used_is_evicted_and_its_bytes_reused() {
        // Room for two 256-byte programs.
        let mut c = ProgramCache::new(region(0x200));
        let (a, b, d) = (prog(1, 64), prog(2, 64), prog(3, 64));
        let Placed::Upload(at_a) = c.place(&a).unwrap() else {
            panic!()
        };
        c.place(&b).unwrap();
        c.unpin_all();
        c.place(&b).unwrap(); // b is now the more recent
        c.unpin_all();
        assert_eq!(c.place(&d).unwrap(), Placed::Upload(at_a), "a was evicted");
        assert_eq!(c.stats().evictions, 1);
        c.unpin_all();
        assert!(
            matches!(c.place(&a).unwrap(), Placed::Upload(_)),
            "a is gone"
        );
    }

    #[test]
    fn a_pinned_program_is_never_evicted() {
        let mut c = ProgramCache::new(region(0x200));
        c.place(&prog(1, 64)).unwrap();
        c.place(&prog(2, 64)).unwrap();
        // Both pinned: a third does not fit, and nothing may make room.
        assert_eq!(c.place(&prog(3, 64)), Err(CacheError::Full { bytes: 256 }));
        c.unpin_all();
        assert!(c.place(&prog(3, 64)).is_ok());
    }

    #[test]
    fn a_program_over_half_the_region_bypasses_the_cache() {
        let mut c = ProgramCache::new(region(0x400));
        assert!(c.admits(128));
        assert!(!c.admits(129));
        assert_eq!(c.place(&prog(1, 129)).unwrap(), Placed::Bypass);
        assert_eq!(c.stats().bypassed, 1);
        assert_eq!(c.resident_bytes(), 0);
    }

    #[test]
    fn freed_space_coalesces_and_clear_forgets_everything() {
        // Three 256-byte programs leave 256 bytes free, at the far end.
        let mut c = ProgramCache::new(region(0x400));
        for t in 0..3 {
            c.place(&prog(t, 64)).unwrap();
        }
        c.unpin_all();
        // A 512-byte program needs the two oldest evicted and merged.
        assert!(matches!(c.place(&prog(9, 128)).unwrap(), Placed::Upload(_)));
        assert_eq!(c.stats().evictions, 2);
        c.clear();
        assert_eq!(c.resident_bytes(), 0);
        assert!(matches!(c.place(&prog(9, 128)).unwrap(), Placed::Upload(_)));
    }

    /// Random placements: nothing resident overlaps, everything is inside the
    /// region and aligned, and a pinned program keeps its address.
    #[test]
    fn random_placements_never_overlap() {
        let mut c = ProgramCache::new(region(0x4000));
        let mut seed = 0x9e37_u64;
        let mut next = |n: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        for step in 0..3000 {
            let p = prog(next(40) as u32, 1 + next(600) as usize);
            match c.place(&p) {
                Ok(Placed::Hit(at)) | Ok(Placed::Upload(at)) => {
                    assert!(at % ALIGN == 0 && c.region.contains(at, p.len() as u64 * 4));
                }
                Ok(Placed::Bypass) | Err(CacheError::Full { .. }) => {}
            }
            let mut spans: Vec<(u64, u64)> = c
                .resident
                .iter()
                .map(|r| (r.at, r.at + r.words.len() as u64 * 4))
                .collect();
            spans.sort_unstable();
            for w in spans.windows(2) {
                assert!(w[0].1 <= w[1].0, "step {step}: {w:?} overlap");
            }
            if next(4) == 0 {
                c.unpin_all();
            }
        }
        assert!(c.stats().hits > 0 && c.stats().evictions > 0);
    }

    #[test]
    fn clearing_for_room_keeps_held_programs() {
        let mut c = ProgramCache::new(tt_isa::l1::PROGRAM_CACHE);
        let Ok(Placed::Upload(a)) = c.place(&[1, 2, 3]) else {
            panic!("a first program is uploaded")
        };
        let Ok(Placed::Upload(b)) = c.place(&[4, 5, 6]) else {
            panic!("a second program is uploaded")
        };
        assert!(c.hold(a));
        c.unpin_all();
        c.clear_unheld();
        assert!(matches!(c.place(&[1, 2, 3]), Ok(Placed::Hit(at)) if at == a));
        assert!(!c.hold(b), "the unheld program was evicted");
    }

    #[test]
    fn a_release_after_a_clear_leaves_the_new_hold_alone() {
        let mut c = ProgramCache::new(tt_isa::l1::PROGRAM_CACHE);
        let Ok(Placed::Upload(old)) = c.place(&[1, 2, 3]) else {
            panic!("a first program is uploaded")
        };
        assert!(c.hold(old));
        let before = c.generation();
        c.clear();
        assert_ne!(c.generation(), before);
        // A later trace's program lands where the first one was.
        let Ok(Placed::Upload(new)) = c.place(&[4, 5, 6]) else {
            panic!("the cleared cache uploads again")
        };
        assert_eq!(new, old);
        assert!(c.hold(new));
        // The first trace is released late: it must not unhold the second.
        c.release_held(before, old);
        c.unpin_all();
        c.clear_unheld();
        assert!(matches!(c.place(&[4, 5, 6]), Ok(Placed::Hit(at)) if at == new));
        c.release_held(c.generation(), new);
        c.clear_unheld();
        assert!(!c.hold(new), "released in its own life, it is evictable");
    }

    #[test]
    fn room_is_what_the_capture_in_progress_protects_and_the_largest_gap_it_leaves() {
        let mut c = ProgramCache::new(region(0x400));
        let Ok(Placed::Upload(a)) = c.place(&prog(1, 64)) else {
            panic!("a is uploaded")
        };
        c.place(&prog(2, 64)).unwrap();
        let Ok(Placed::Upload(b)) = c.place(&prog(3, 64)) else {
            panic!("b is uploaded")
        };
        assert!(c.hold(a) && c.hold(b));
        c.protect(a);
        c.protect(b);
        let room = c.room();
        assert_eq!((room.region, room.held, room.protected), (0x400, 512, 512));
        // Free once the rest goes: the 256 between them and the 256 after.
        assert_eq!(room.largest, 256);
        c.unprotect_all();
        assert_eq!(c.room().largest, 0x400, "a held program is evictable");
    }

    #[test]
    fn a_held_program_is_evicted_after_unheld_ones_and_a_protected_one_never() {
        let region = Region {
            name: "test",
            base: 0,
            end: 1024,
        };
        let mut c = ProgramCache::new(region);
        let a = vec![1u32; 96]; // 384 bytes
        let Placed::Upload(at_a) = c.place(&a).unwrap() else {
            panic!()
        };
        assert!(c.hold(at_a));
        c.unpin_all();
        // Two more of the same size do not fit beside `a`: the unheld one goes first.
        let b = vec![2u32; 96];
        assert!(matches!(c.place(&b), Ok(Placed::Upload(_))));
        c.unpin_all();
        let d = vec![3u32; 96];
        assert!(
            matches!(c.place(&d), Ok(Placed::Upload(_))),
            "b is evicted, a is not"
        );
        c.unpin_all();
        assert_eq!(c.place(&a).unwrap(), Placed::Hit(at_a));
        c.unpin_all();
        // Held but not protected, a held program goes when nothing else makes room: `d` held
        // too, a 480-byte program beside the two takes the least recently used of them.
        let Placed::Hit(at_d) = c.place(&d).unwrap() else {
            panic!("d is resident")
        };
        assert!(c.hold(at_d));
        c.unpin_all();
        assert_eq!(c.place(&a).unwrap(), Placed::Hit(at_a));
        c.unpin_all();
        let e = vec![4u32; 120];
        let ae: Arc<[u32]> = e.clone().into();
        assert_eq!(
            c.place(&e),
            Err(CacheError::Full { bytes: 480 }),
            "not while a queued replay may run a held program"
        );
        assert!(matches!(
            c.place_hashed_evicting(&ae, hash(&ae), true),
            Ok(Placed::Upload(_))
        ));
        let holds = |at: u64, p: &[u32]| c.resident_at(at).is_some_and(|(w, _)| *w == *p);
        assert!(!holds(at_d, &d) && holds(at_a, &a), "d goes, a stays");
        // Protected, they stay.
        c.clear();
        let Placed::Upload(at_a) = c.place(&a).unwrap() else {
            panic!()
        };
        let Placed::Upload(at_e) = c.place(&e).unwrap() else {
            panic!()
        };
        for at in [at_a, at_e] {
            c.hold(at);
            c.protect(at);
        }
        c.unpin_all();
        let f: Arc<[u32]> = vec![5u32; 120].into();
        assert_eq!(
            c.place_hashed_evicting(&f, hash(&f), true),
            Err(CacheError::Full { bytes: 480 })
        );
    }

    /// A program restored where a trace named it: there already, a hit; evicted, put back at
    /// its address over what lies there now; refused across a pinned program.
    #[test]
    fn restore_puts_a_program_back_at_its_address() {
        let mut c = ProgramCache::new(region(0x400));
        let a: Arc<[u32]> = prog(1, 64).into();
        let ha = hash(&a);
        let Ok(Placed::Upload(at)) = c.place_hashed(&a, ha) else {
            panic!("a is uploaded")
        };
        c.unpin_all();
        assert_eq!(c.restore(&a, ha, at, true), Ok(Placed::Hit(at)));
        c.unpin_all();
        c.clear();
        // Two others now span `a`'s address.
        c.place(&prog(2, 32)).unwrap();
        c.place(&prog(3, 64)).unwrap();
        assert_eq!(
            c.restore(&a, ha, at, true),
            Err(CacheError::Full { bytes: 256 }),
            "pinned programs lie across it"
        );
        c.unpin_all();
        assert_eq!(c.restore(&a, ha, at, true), Ok(Placed::Upload(at)));
        assert_eq!(c.resident_at(at).map(|(w, _)| w), Some(a.clone()));
        // The rest of the region is still first fit, nothing overlapping.
        c.unpin_all();
        let Ok(Placed::Upload(other)) = c.place(&prog(4, 64)) else {
            panic!("room after it")
        };
        assert!(other >= at + 256 || other + 256 <= at);
    }
}
