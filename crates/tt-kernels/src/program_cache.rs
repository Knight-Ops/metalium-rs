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
//! tile, 34 KB on eight, in a 252 KB region):
//!
//! * **Keyed by the program's words**, compared exactly, as the fixed slots
//!   already were.
//! * **First fit**, coalescing on free.
//! * **Least recently used** evicted first.
//! * **Pinned** while a list in flight names it: [`ProgramCache::place`] pins,
//!   [`ProgramCache::unpin_all`] releases once the list has finished. Evicting a
//!   program a submitted list still names would run whatever replaced it.
//! * **Admission**: a program larger than half the region is not cached
//!   ([`Placed::Bypass`]); it runs from the fixed slot, as before, so one huge
//!   kernel cannot flush everything else.
//! * **Invalidated** whenever the tile is reset ([`ProgramCache::clear`]).

use std::collections::{BTreeMap, HashMap};

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
    words: Vec<u32>,
    at: u64,
    last_use: u64,
    pinned: bool,
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

fn hash(words: &[u32]) -> u64 {
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
        if !self.admits(words.len()) {
            self.stats.bypassed += 1;
            return Ok(Placed::Bypass);
        }
        self.clock += 1;
        let h = hash(words);
        if let Some(&i) = self
            .by_hash
            .get(&h)
            .and_then(|v| v.iter().find(|&&i| self.resident[i].words == words))
        {
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
            // Evict the least recently used program nothing in flight names.
            let victim = self
                .resident
                .iter()
                .enumerate()
                .filter(|(_, r)| !r.pinned)
                .min_by_key(|(_, r)| r.last_use)
                .map(|(i, _)| i)
                .ok_or(CacheError::Full { bytes })?;
            self.evict(victim);
        };
        let len = self.free.remove(&at).expect("found above");
        if len > bytes {
            self.free.insert(at + bytes, len - bytes);
        }
        self.resident.push(Resident {
            words: words.to_vec(),
            at,
            last_use: self.clock,
            pinned: true,
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
        self.reindex();
    }

    fn reindex(&mut self) {
        self.by_hash.clear();
        for (i, r) in self.resident.iter().enumerate() {
            self.by_hash.entry(hash(&r.words)).or_default().push(i);
        }
    }

    /// Nothing in flight names any program any more.
    pub fn unpin_all(&mut self) {
        for r in &mut self.resident {
            r.pinned = false;
        }
    }

    /// Forget everything: the tile was reset, and what is in its L1 is no
    /// longer the host's to vouch for.
    pub fn clear(&mut self) {
        let stats = self.stats;
        *self = ProgramCache::new(self.region);
        self.stats = stats;
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// Bytes of programs resident.
    pub fn resident_bytes(&self) -> u64 {
        self.resident.iter().map(|r| r.words.len() as u64 * 4).sum()
    }
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
}
