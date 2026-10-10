//! Checked helpers for the L1 Cache Tag Search Accelerator.
//!
//! Source: `BlackholeA0/TensixTile/BabyRISCV/L1CacheTagSearchAccel.md` (pinned).
//! The block is **not** an MMIO register window. It is configured through eight
//! words of the Tensix backend `Config` (indices 212..=219, all above
//! `GLOBAL_CFGREG_BASE_ADDR32`, so one store lands in both banks), reached from
//! RISCV B by `sw` at `TENSIX_CFG_BASE + 4 * index`, and it is *triggered* by an
//! ordinary load that RISCV B (and only B) performs against an L1 address that
//! the configuration names. The load never reaches L1: the hardware answers it.
//!
//! # What the page defines
//!
//! * A latch: whenever software **changes** the value of `Search_Enable`,
//!   `Tag_alloc`, `Tag_inv`, `Tag_inv_all` or `Data_Valid_chk`, all thirteen
//!   fields are copied into a private `LatchedConfig`. Writing only a tag value
//!   or an address while the five trigger bits keep their value does **not**
//!   re-latch. [`Config::program`] therefore disarms the five bits first, writes
//!   everything else, then arms: the last trigger flip sees the final fields.
//! * Three read-triggered operations, each only for a read that misses the L0
//!   data cache (a `fence` flushes it): the tag search (read at `Start_Addr * 16`),
//!   invalidate-all (read at `Valid_bit_section_start_addr * 16`) and the
//!   bit-vector query (read at `Data_Valid_bit_section_start_addr * 16`).
//!
//! # What the page does not define (so this module refuses it)
//!
//! * `End_Addr < Start_Addr` (the search loop `Tags + i != TagsEnd` would not
//!   terminate), or a tag array with more entries than the validity vector has
//!   bits (`Valids[i / 64]` would index past `Valid_bit_section_end_addr`).
//! * Tag arrays, validity vectors or bit vectors outside the 1.5 MiB of L1.
//! * A `Tag_Value` wider than `Tag_Width` (the page truncates with `(T)`; a
//!   silent truncation is a likely bug, so it is a [`Error::TagTooWide`]).
//! * The value returned to a load that is not an aligned word at the trigger
//!   address (the page says "the result of the read" without sizes), and the
//!   source of `rand()` for the all-valid allocation. Only `lw` at the exact
//!   trigger address is offered, and the allocation index is checked as a range,
//!   not a value ([`SearchOutcome::Allocate`]).
//!
//! Silicon status: `UNVERIFIED`. Nothing here has executed on a Blackhole card;
//! ttsim does not implement the block (divergence row 92).

use crate::cfg::generated::global as g;
use crate::cfg::{ConfigBank, ConfigField};
use crate::tensix::L1_SIZE;

/// The layout of the RISCV B probe's staged script and results
/// (`tt-firmware/src/bin/tag_search_b.rs`), shared with the host test so the two
/// cannot disagree. All of it lies in the data arena (`l1::DATA`).
pub mod probe {
    /// Script header: word 0 is the number of steps; word 1 is non-zero for
    /// blind mode (no `Config` loads, a simulator diagnostic); word 2 is non-zero
    /// to skip the disarm before and after the steps.
    pub const SCRIPT: u64 = 0x2_0000;
    /// Most steps one run executes.
    pub const MAX_STEPS: u32 = 16;
    /// First step.
    pub const STEPS: u64 = SCRIPT + 0x10;
    /// Bytes per step: word 0 the L1 address to load, word 1 the number of
    /// [`super::WriteStep`]s, then `(word, mask, value)` triples.
    pub const STEP_STRIDE: u64 = 0x100;
    /// Where the results go; per step, word 0 is the loaded value and words
    /// 1..=8 are `Config[212..=219]` read back after the writes.
    pub const RESULTS: u64 = 0x2_2000;
    /// Bytes per result record.
    pub const RESULT_STRIDE: u64 = 0x40;
    /// Where the host stages tags, validity vectors and bit vectors.
    pub const DATA: u64 = 0x3_0000;
    /// Nops between the last `Config` store and the load it should trigger.
    pub const SETTLE_NOPS: u32 = 64;
}

/// Bytes per address granule: every `*_Addr` field counts 16-byte units.
pub const GRANULE: u64 = 16;
/// Width of every address field, in bits.
pub const ADDR_BITS: u32 = 17;
/// Largest value of a 17-bit address field.
pub const ADDR_MAX: u32 = (1 << ADDR_BITS) - 1;
/// Width of `Data_Valid_offset`, in bits.
pub const OFFSET_BITS: u32 = 24;

/// First `Config` word the block uses.
pub const FIRST_WORD: u16 = 212;
/// Number of consecutive `Config` words the block uses (212..=219).
pub const WORD_COUNT: usize = 8;

/// `sw` target of `Config` word `word` (bank 0; the words are global).
pub const fn config_address(word: u16) -> u64 {
    g::L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable.riscv_address(ConfigBank::Bank0)
        + (word as u64 - FIRST_WORD as u64) * 4
}

const TRIGGERS: [ConfigField; 5] = [
    g::L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_alloc,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv_all,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_chk,
];

/// Every field of the block, in the order of the page's `LatchedConfig`.
const FIELDS: [ConfigField; 14] = [
    g::L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_alloc,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv_all,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Width,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_low,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Value_high,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Start_Addr,
    g::L1_CACHE_TAG_SEARCH_ACCEL_End_Addr,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_start_addr,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_end_addr,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_chk,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_bit_section_start_addr,
    g::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_offset,
];

/// Why a configuration was refused.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    /// An address does not fit its 17-bit field.
    AddrTooWide { what: &'static str, value: u32 },
    /// `end < start` in a span.
    EndBeforeStart { what: &'static str },
    /// A span's bytes do not lie inside L1.
    OutsideL1 { what: &'static str },
    /// The tag value does not fit the chosen tag width.
    TagTooWide { value: u64, width: TagWidth },
    /// More tags than the validity vector has bits.
    TagsOverrunValidBits { tags: u64, valid_bits: u64 },
    /// `Data_Valid_offset` does not fit 24 bits.
    OffsetTooWide { value: u32 },
    /// The queried bit lies outside L1.
    BitOutsideL1,
    /// The tag array overlaps the validity vector it is paired with.
    Overlap,
}

/// `Tag_Width`: the tag element size.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TagWidth {
    /// `uint8_t` tags.
    W8 = 0,
    /// `uint16_t` tags.
    W16 = 1,
    /// `uint32_t` tags.
    W32 = 2,
    /// `uint64_t` tags.
    W64 = 3,
}

impl TagWidth {
    /// Bytes per tag.
    pub const fn bytes(self) -> u64 {
        1 << (self as u32)
    }

    /// Largest tag value of this width.
    pub const fn max_value(self) -> u64 {
        match self {
            TagWidth::W8 => 0xFF,
            TagWidth::W16 => 0xFFFF,
            TagWidth::W32 => 0xFFFF_FFFF,
            TagWidth::W64 => u64::MAX,
        }
    }

    /// Decode the 2-bit field.
    pub const fn from_bits(bits: u32) -> TagWidth {
        match bits & 3 {
            0 => TagWidth::W8,
            1 => TagWidth::W16,
            2 => TagWidth::W32,
            _ => TagWidth::W64,
        }
    }
}

/// An inclusive span of 16-byte granules in L1, as the `*_Start_Addr` /
/// `*_End_Addr` pairs name them: it covers bytes `first * 16 .. (last + 1) * 16`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Span {
    first: u32,
    last: u32,
}

impl Span {
    /// A span, checked for field width, order and containment in L1.
    pub const fn new(what: &'static str, first: u32, last: u32) -> Result<Span, Error> {
        if first > ADDR_MAX || last > ADDR_MAX {
            return Err(Error::AddrTooWide {
                what,
                value: if first > ADDR_MAX { first } else { last },
            });
        }
        if last < first {
            return Err(Error::EndBeforeStart { what });
        }
        if (last as u64 + 1) * GRANULE > L1_SIZE {
            return Err(Error::OutsideL1 { what });
        }
        Ok(Span { first, last })
    }

    /// First granule.
    pub const fn first(self) -> u32 {
        self.first
    }

    /// Last granule (inclusive).
    pub const fn last(self) -> u32 {
        self.last
    }

    /// First byte address.
    pub const fn start_byte(self) -> u64 {
        self.first as u64 * GRANULE
    }

    /// One past the last byte address.
    pub const fn end_byte(self) -> u64 {
        (self.last as u64 + 1) * GRANULE
    }

    /// Bytes covered.
    pub const fn bytes(self) -> u64 {
        self.end_byte() - self.start_byte()
    }

    const fn overlaps(self, other: Span) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// The thirteen fields of the page's `LatchedConfig`, unchecked: what the
/// registers say, not what is safe to ask for. [`Config`] is the checked form.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Fields {
    pub search_enable: bool,
    pub tag_alloc: bool,
    pub tag_inv: bool,
    pub tag_inv_all: bool,
    pub tag_width: u32,
    pub tag_value: u64,
    pub start_addr: u32,
    pub end_addr: u32,
    pub valid_start_addr: u32,
    pub valid_end_addr: u32,
    pub data_valid_chk: bool,
    pub data_valid_start_addr: u32,
    pub data_valid_offset: u32,
}

/// The eight `Config` words 212..=219 that hold the block's fields.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Words(pub [u32; WORD_COUNT]);

impl Words {
    fn get(&self, f: ConfigField) -> u32 {
        f.extract(self.0[(f.addr32() - FIRST_WORD) as usize])
    }

    fn set(&mut self, f: ConfigField, v: u32) {
        let i = (f.addr32() - FIRST_WORD) as usize;
        self.0[i] = f.insert(self.0[i], v);
    }

    /// Decode every field.
    pub fn fields(&self) -> Fields {
        Fields {
            search_enable: self.get(FIELDS[0]) != 0,
            tag_alloc: self.get(FIELDS[1]) != 0,
            tag_inv: self.get(FIELDS[2]) != 0,
            tag_inv_all: self.get(FIELDS[3]) != 0,
            tag_width: self.get(FIELDS[4]),
            tag_value: u64::from(self.get(FIELDS[5])) | (u64::from(self.get(FIELDS[6])) << 32),
            start_addr: self.get(FIELDS[7]),
            end_addr: self.get(FIELDS[8]),
            valid_start_addr: self.get(FIELDS[9]),
            valid_end_addr: self.get(FIELDS[10]),
            data_valid_chk: self.get(FIELDS[11]) != 0,
            data_valid_start_addr: self.get(FIELDS[12]),
            data_valid_offset: self.get(FIELDS[13]),
        }
    }

    /// Encode `f`, truncating to field widths (callers use [`Config`], which checks).
    pub fn from_fields(f: &Fields) -> Words {
        let mut w = Words::default();
        w.set(FIELDS[0], f.search_enable as u32);
        w.set(FIELDS[1], f.tag_alloc as u32);
        w.set(FIELDS[2], f.tag_inv as u32);
        w.set(FIELDS[3], f.tag_inv_all as u32);
        w.set(FIELDS[4], f.tag_width);
        w.set(FIELDS[5], f.tag_value as u32);
        w.set(FIELDS[6], (f.tag_value >> 32) as u32);
        w.set(FIELDS[7], f.start_addr);
        w.set(FIELDS[8], f.end_addr);
        w.set(FIELDS[9], f.valid_start_addr);
        w.set(FIELDS[10], f.valid_end_addr);
        w.set(FIELDS[11], f.data_valid_chk as u32);
        w.set(FIELDS[12], f.data_valid_start_addr);
        w.set(FIELDS[13], f.data_valid_offset);
        w
    }

    /// Would moving the registers from `self` to `new` re-latch? True exactly when
    /// one of the five trigger bits changes value.
    pub fn relatches(&self, new: &Words) -> bool {
        TRIGGERS.iter().any(|&t| self.get(t) != new.get(t))
    }
}

/// One read-modify-write of a `Config` word: the bits in `mask` become `value`
/// and every other bit (fields the pinned header does not name) is preserved.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct WriteStep {
    /// `Config` word index (212..=219).
    pub word: u16,
    /// Bits of the word this step owns.
    pub mask: u32,
    /// New value of those bits.
    pub value: u32,
}

/// Number of writes in [`Config::program`]: disarm (3), data (5), arm (3).
pub const PROGRAM_STEPS: usize = 11;

/// Mask of the bits of `word` that belong to a field of the block.
pub const fn word_mask(word: u16) -> u32 {
    let mut m = 0;
    let mut i = 0;
    while i < FIELDS.len() {
        if FIELDS[i].addr32() == word {
            m |= FIELDS[i].mask();
        }
        i += 1;
    }
    m
}

const fn trigger_mask(word: u16) -> u32 {
    let mut m = 0;
    let mut i = 0;
    while i < TRIGGERS.len() {
        if TRIGGERS[i].addr32() == word {
            m |= TRIGGERS[i].mask();
        }
        i += 1;
    }
    m
}

/// Writes that clear the five trigger bits and nothing else: the latch then holds
/// "everything off", and no read is intercepted. Run before and after a probe.
pub const DISARM: [WriteStep; 3] = [
    WriteStep {
        word: 212,
        mask: trigger_mask(212),
        value: 0,
    },
    WriteStep {
        word: 218,
        mask: trigger_mask(218),
        value: 0,
    },
    WriteStep {
        word: 219,
        mask: trigger_mask(219),
        value: 0,
    },
];

/// Which of the three operations a load triggers.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Op {
    /// Tag search (with optional invalidate-on-hit and allocate-on-miss).
    Search,
    /// Clear every bit of the validity vector.
    InvalidateAll,
    /// Read one bit of the data-valid bit vector.
    BitQuery,
}

/// A checked configuration of one operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Config {
    op: Op,
    fields: Fields,
}

impl Config {
    /// A tag search over `tags` (entries of `width`), validity bits at `valids`.
    ///
    /// `invalidate` is `Tag_inv` (clear the hit's valid bit); `allocate` is
    /// `Tag_alloc` (on a miss, return the first clear valid bit, or a random
    /// index when every bit is set).
    pub const fn search(
        tags: Span,
        width: TagWidth,
        tag: u64,
        valids: Span,
        invalidate: bool,
        allocate: bool,
    ) -> Result<Config, Error> {
        if tag > width.max_value() {
            return Err(Error::TagTooWide { value: tag, width });
        }
        let tag_count = tags.bytes() / width.bytes();
        let valid_bits = valids.bytes() * 8;
        if tag_count > valid_bits {
            return Err(Error::TagsOverrunValidBits {
                tags: tag_count,
                valid_bits,
            });
        }
        if tags.overlaps(valids) {
            return Err(Error::Overlap);
        }
        Ok(Config {
            op: Op::Search,
            fields: Fields {
                search_enable: true,
                tag_alloc: allocate,
                tag_inv: invalidate,
                tag_inv_all: false,
                tag_width: width as u32,
                tag_value: tag,
                start_addr: tags.first(),
                end_addr: tags.last(),
                valid_start_addr: valids.first(),
                valid_end_addr: valids.last(),
                data_valid_chk: false,
                data_valid_start_addr: 0,
                data_valid_offset: 0,
            },
        })
    }

    /// Invalidate-all over the validity vector `valids`.
    pub const fn invalidate_all(valids: Span) -> Config {
        Config {
            op: Op::InvalidateAll,
            fields: Fields {
                search_enable: false,
                tag_alloc: false,
                tag_inv: false,
                tag_inv_all: true,
                tag_width: 0,
                tag_value: 0,
                start_addr: 0,
                end_addr: 0,
                valid_start_addr: valids.first(),
                valid_end_addr: valids.last(),
                data_valid_chk: false,
                data_valid_start_addr: 0,
                data_valid_offset: 0,
            },
        }
    }

    /// Query bit `offset` of the bit vector at granule `start`.
    pub const fn bit_query(start: u32, offset: u32) -> Result<Config, Error> {
        if start > ADDR_MAX {
            return Err(Error::AddrTooWide {
                what: "Data_Valid_bit_section_start_addr",
                value: start,
            });
        }
        if offset >= (1 << OFFSET_BITS) {
            return Err(Error::OffsetTooWide { value: offset });
        }
        // The page reads the 64-bit word `offset / 64` of the vector.
        if start as u64 * GRANULE + (offset as u64 / 64 + 1) * 8 > L1_SIZE {
            return Err(Error::BitOutsideL1);
        }
        Ok(Config {
            op: Op::BitQuery,
            fields: Fields {
                search_enable: false,
                tag_alloc: false,
                tag_inv: false,
                tag_inv_all: false,
                tag_width: 0,
                tag_value: 0,
                start_addr: 0,
                end_addr: 0,
                valid_start_addr: 0,
                valid_end_addr: 0,
                data_valid_chk: true,
                data_valid_start_addr: start,
                data_valid_offset: offset,
            },
        })
    }

    /// The operation this configuration triggers.
    pub const fn op(&self) -> Op {
        self.op
    }

    /// The fields as registers hold them.
    pub const fn fields(&self) -> &Fields {
        &self.fields
    }

    /// The L1 byte address whose (L0-missing) load triggers [`Config::op`].
    pub const fn trigger_address(&self) -> u64 {
        match self.op {
            Op::Search => self.fields.start_addr as u64 * GRANULE,
            Op::InvalidateAll => self.fields.valid_start_addr as u64 * GRANULE,
            Op::BitQuery => self.fields.data_valid_start_addr as u64 * GRANULE,
        }
    }

    /// The eight words this configuration leaves in `Config[212..=219]`.
    pub fn words(&self) -> Words {
        Words::from_fields(&self.fields)
    }

    /// The writes that make the hardware latch exactly this configuration:
    /// disarm the five trigger bits (so the next flip is a change whatever the
    /// previous state), write every other field, then arm. Every step is a
    /// masked read-modify-write so unnamed bits of the words survive.
    pub fn program(&self) -> [WriteStep; PROGRAM_STEPS] {
        let w = self.words();
        let pick = |word: u16, mask: u32| WriteStep {
            word,
            mask,
            value: w.0[(word - FIRST_WORD) as usize] & mask,
        };
        // Bits of a word that are not trigger bits, and the trigger bits alone.
        let data = |word: u16| pick(word, word_mask(word) & !trigger_mask(word));
        let arm = |word: u16| pick(word, trigger_mask(word));
        // Step 1 owns the whole word: the new non-trigger bits, triggers cleared.
        let disarm = |word: u16| WriteStep {
            word,
            mask: word_mask(word),
            value: data(word).value,
        };
        [
            disarm(212),
            disarm(218),
            disarm(219),
            data(213),
            data(214),
            data(215),
            data(216),
            data(217),
            arm(218),
            arm(219),
            arm(212),
        ]
    }
}

/// What a tag-search load returned.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SearchOutcome {
    /// `0`: no valid match, and `Tag_alloc` was off.
    Miss,
    /// `1 + i`: tag `i` matched and its valid bit was set.
    Hit { index: u32 },
    /// `0x8000_0001 + n`: `Tag_alloc` chose valid-bit `n`.
    Allocate { index: u32 },
}

/// Decode the word a search load returned.
pub const fn decode_search(word: u32) -> SearchOutcome {
    if word == 0 {
        SearchOutcome::Miss
    } else if word & 0x8000_0000 != 0 {
        SearchOutcome::Allocate {
            index: word - 0x8000_0001,
        }
    } else {
        SearchOutcome::Hit { index: word - 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_words_are_212_to_219_and_global() {
        assert_eq!(
            config_address(212),
            g::L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable.riscv_address(ConfigBank::Bank0)
        );
        assert_eq!(config_address(219) - config_address(212), 28);
        for f in FIELDS {
            assert!((FIRST_WORD..FIRST_WORD + WORD_COUNT as u16).contains(&f.addr32()));
        }
    }

    #[test]
    fn field_round_trip_and_masks() {
        let f = Fields {
            search_enable: true,
            tag_alloc: true,
            tag_inv: false,
            tag_inv_all: true,
            tag_width: 3,
            tag_value: 0x0123_4567_89AB_CDEF,
            start_addr: 0x1_2345,
            end_addr: 0x1_FFFF,
            valid_start_addr: 0x0_BEEF,
            valid_end_addr: 0x1_0001,
            data_valid_chk: true,
            data_valid_start_addr: 0x1_5555,
            data_valid_offset: 0xAB_CDEF,
        };
        assert_eq!(Words::from_fields(&f).fields(), f);
    }

    #[test]
    fn trigger_flip_relatches_but_data_change_does_not() {
        let cfg = Config::search(
            Span::new("tags", 0x2000, 0x2003).unwrap(),
            TagWidth::W32,
            7,
            Span::new("valids", 0x2100, 0x2100).unwrap(),
            false,
            false,
        )
        .unwrap();
        let a = cfg.words();
        let mut b = a;
        b.set(FIELDS[5], 8); // only the tag value
        assert!(!a.relatches(&b));
        b.set(FIELDS[2], 1); // Tag_inv
        assert!(a.relatches(&b));
    }

    #[test]
    fn program_disarms_first_and_arms_last() {
        let cfg = Config::invalidate_all(Span::new("valids", 0x2100, 0x2101).unwrap());
        let p = cfg.program();
        assert!(p[..3].iter().all(|s| s.value & trigger_mask(s.word) == 0));
        assert_eq!(p[10].word, 212);
        assert!(p[8..].iter().all(|s| s.mask == trigger_mask(s.word)));
    }

    #[test]
    fn refusals() {
        assert!(matches!(
            Span::new("x", 5, 4),
            Err(Error::EndBeforeStart { .. })
        ));
        assert!(matches!(
            Span::new("x", 0, ADDR_MAX + 1),
            Err(Error::AddrTooWide { .. })
        ));
        assert!(matches!(
            Span::new("x", 0x17FFF, 0x18000),
            Err(Error::OutsideL1 { .. })
        ));
        let tags = Span::new("t", 0x2000, 0x2003).unwrap();
        let v = Span::new("v", 0x2100, 0x2100).unwrap();
        assert!(matches!(
            Config::search(tags, TagWidth::W8, 0x100, v, false, false),
            Err(Error::TagTooWide { .. })
        ));
        // 256 granules of `u8` tags are 4096 tags; one granule of valids has 128 bits.
        let many = Span::new("t", 0x2000, 0x20FF).unwrap();
        assert!(matches!(
            Config::search(many, TagWidth::W8, 1, v, false, false),
            Err(Error::TagsOverrunValidBits { .. })
        ));
        assert!(matches!(
            Config::search(tags, TagWidth::W32, 1, tags, false, false),
            Err(Error::Overlap)
        ));
        assert!(Config::bit_query(0x2000, 1 << 24).is_err());
    }

    #[test]
    fn decode_search_results() {
        assert_eq!(decode_search(0), SearchOutcome::Miss);
        assert_eq!(decode_search(3), SearchOutcome::Hit { index: 2 });
        assert_eq!(
            decode_search(0x8000_0001),
            SearchOutcome::Allocate { index: 0 }
        );
    }
}
