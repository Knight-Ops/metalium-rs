//! L1 Cache Tag Search Accelerator (`BabyRISCV/L1CacheTagSearchAccel.md`).
//!
//! Three layers of evidence:
//!
//! * **Model and encoding (host, always).** [`Model`] is a literal transcription
//!   of the page's pseudocode (`LatchedConfig`, `TagSearch<T>`, invalidate-all,
//!   bit-vector query), written without `tt_isa::tag_search`: its register layout
//!   is a separate table checked against `vendor/cfg_defines.h`, and the scenario
//!   expectations are hand-computed literals. The helpers' register programs are
//!   run *through* the model, which is what shows that `Config::program` leaves the
//!   latch holding exactly what was asked even when only data fields change.
//! * **ttsim (not silicon).** The probe image runs on RISCV B in the simulator
//!   and the observed loads are compared with the page; see
//!   `simulator_does_not_implement_the_block` for what ttsim actually does
//!   (divergence row 92).
//! * **Silicon (`--features silicon`, written, not run).** An isolated minimal
//!   probe inside `survives`, then the semantic gates against the model. UNVERIFIED:
//!   nothing has executed this block on a Blackhole card.
//!
//! Negative control: [`Mutant`] changes the model (tag compare, invalidate-all
//! scope), and the permanent test `mutants_are_caught` requires the expectation
//! table to reject each.

use tt_isa::tag_search::{
    self as ts, probe, Config, Op, Span, TagWidth, WriteStep, PROGRAM_STEPS, WORD_COUNT,
};

const L1: usize = 1536 * 1024;

// ---------------------------------------------------------------------------
// The model: the page, transcribed.
// ---------------------------------------------------------------------------

/// `(Config word, shift, mask)` for each field, transcribed from
/// `vendor/cfg_defines.h` by hand and checked against it below.
struct F(u32, u32, u32);
const SEARCH_ENABLE: F = F(212, 0, 0x1);
const START_ADDR: F = F(212, 1, 0x3fffe);
const END_ADDR: F = F(213, 0, 0x1ffff);
const TAG_VALUE_LOW: F = F(214, 0, 0xffff_ffff);
const TAG_VALUE_HIGH: F = F(215, 0, 0xffff_ffff);
const TAG_WIDTH: F = F(216, 0, 0x3);
const VALID_START: F = F(216, 2, 0x7fffc);
const VALID_END: F = F(217, 0, 0x1ffff);
const DV_START: F = F(218, 0, 0x1ffff);
const DV_CHK: F = F(218, 17, 0x20000);
const DV_OFFSET: F = F(219, 0, 0xff_ffff);
const TAG_INV: F = F(219, 24, 0x100_0000);
const TAG_INV_ALL: F = F(219, 25, 0x200_0000);
const TAG_ALLOC: F = F(219, 26, 0x400_0000);

const ALL_FIELDS: [(&str, &F); 14] = [
    ("Search_Enable", &SEARCH_ENABLE),
    ("Start_Addr", &START_ADDR),
    ("End_Addr", &END_ADDR),
    ("Tag_Value_low", &TAG_VALUE_LOW),
    ("Tag_Value_high", &TAG_VALUE_HIGH),
    ("Tag_Width", &TAG_WIDTH),
    ("Valid_bit_section_start_addr", &VALID_START),
    ("Valid_bit_section_end_addr", &VALID_END),
    ("Data_Valid_bit_section_start_addr", &DV_START),
    ("Data_Valid_chk", &DV_CHK),
    ("Data_Valid_offset", &DV_OFFSET),
    ("Tag_inv", &TAG_INV),
    ("Tag_inv_all", &TAG_INV_ALL),
    ("Tag_alloc", &TAG_ALLOC),
];

fn get(words: &[u32; 8], f: &F) -> u32 {
    (words[(f.0 - 212) as usize] & f.2) >> f.1
}

/// The deliberately wrong models the control runs.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Mutant {
    None,
    /// Compares only the low byte of each tag with the low byte of the value.
    TagCompareLowByte,
    /// Invalidate-all clears the first 64-bit word of the vector only.
    InvalidateAllFirstWord,
}

#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct Latched {
    search_enable: bool,
    tag_alloc: bool,
    tag_inv: bool,
    tag_inv_all: bool,
    tag_width: u32,
    tag_value: u64,
    start_addr: u32,
    end_addr: u32,
    valid_start: u32,
    valid_end: u32,
    data_valid_chk: bool,
    dv_start: u32,
    dv_offset: u32,
}

struct Model {
    l1: Vec<u8>,
    cfg: [u32; 8],
    latched: Latched,
    mutant: Mutant,
    rng: u64,
}

impl Model {
    fn new(mutant: Mutant) -> Self {
        Model {
            l1: vec![0; L1],
            cfg: [0; 8],
            latched: Latched::default(),
            mutant,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn triggers(&self) -> [u32; 5] {
        let w = &self.cfg;
        [
            get(w, &SEARCH_ENABLE),
            get(w, &TAG_ALLOC),
            get(w, &TAG_INV),
            get(w, &TAG_INV_ALL),
            get(w, &DV_CHK),
        ]
    }

    /// A masked store to `Config[word]`. The page: "whenever software _changes_
    /// the value of any of" the five trigger fields, hardware latches everything.
    fn write_cfg(&mut self, word: u32, mask: u32, value: u32) {
        let before = self.triggers();
        let i = (word - 212) as usize;
        self.cfg[i] = (self.cfg[i] & !mask) | (value & mask);
        if self.triggers() != before {
            let w = &self.cfg;
            self.latched = Latched {
                search_enable: get(w, &SEARCH_ENABLE) != 0,
                tag_alloc: get(w, &TAG_ALLOC) != 0,
                tag_inv: get(w, &TAG_INV) != 0,
                tag_inv_all: get(w, &TAG_INV_ALL) != 0,
                tag_width: get(w, &TAG_WIDTH),
                tag_value: u64::from(get(w, &TAG_VALUE_LOW))
                    + (u64::from(get(w, &TAG_VALUE_HIGH)) << 32),
                start_addr: get(w, &START_ADDR),
                end_addr: get(w, &END_ADDR),
                valid_start: get(w, &VALID_START),
                valid_end: get(w, &VALID_END),
                data_valid_chk: get(w, &DV_CHK) != 0,
                dv_start: get(w, &DV_START),
                dv_offset: get(w, &DV_OFFSET),
            };
        }
    }

    fn rd(&self, byte: u64, len: usize) -> u64 {
        let b = byte as usize;
        let mut v = 0u64;
        for k in 0..len {
            v |= u64::from(self.l1[b + k]) << (8 * k);
        }
        v
    }

    fn valid_bit(&self, base: u64, i: u64) -> bool {
        self.rd(base + (i / 64) * 8, 8) >> (i % 64) & 1 != 0
    }

    fn clear_valid_bit(&mut self, base: u64, i: u64) {
        let at = (base + (i / 64) * 8) as usize;
        self.l1[at + ((i % 64) / 8) as usize] &= !(1 << (i % 8));
    }

    /// A load by RISCV B that misses the L0 data cache.
    fn load(&mut self, addr: u64) -> u32 {
        let l = self.latched;
        let granule = addr / 16;
        if l.search_enable
            && !l.tag_inv_all
            && !l.data_valid_chk
            && granule == u64::from(l.start_addr)
        {
            return self.tag_search();
        }
        if l.tag_inv_all && granule == u64::from(l.valid_start) {
            let valids = u64::from(l.valid_start) * 16;
            let valids_end = (u64::from(l.valid_end) + 1) * 16;
            let mut at = valids;
            while at != valids_end {
                if self.mutant == Mutant::InvalidateAllFirstWord && at != valids {
                    break;
                }
                for k in 0..8 {
                    self.l1[(at + k) as usize] = 0;
                }
                at += 8;
            }
            return 0;
        }
        if l.data_valid_chk && !l.tag_inv_all && granule == u64::from(l.dv_start) {
            let v = u64::from(l.dv_start) * 16;
            let off = u64::from(l.dv_offset);
            return (self.rd(v + (off / 64) * 8, 8) >> (off % 64) & 1) as u32;
        }
        self.rd(addr, 4) as u32
    }

    fn tag_search(&mut self) -> u32 {
        let l = self.latched;
        let size = 1u64 << l.tag_width;
        let tags = u64::from(l.start_addr) * 16;
        let tags_end = (u64::from(l.end_addr) + 1) * 16;
        let valids = u64::from(l.valid_start) * 16;
        let n = (tags_end - tags) / size;
        let tag_mask = if size == 8 {
            u64::MAX
        } else {
            (1u64 << (8 * size)) - 1
        };
        for i in 0..n {
            let t = self.rd(tags + i * size, size as usize);
            let equal = if self.mutant == Mutant::TagCompareLowByte {
                (t & 0xFF) == (l.tag_value & 0xFF)
            } else {
                t == (l.tag_value & tag_mask)
            };
            if equal {
                if self.valid_bit(valids, i) {
                    if l.tag_inv {
                        self.clear_valid_bit(valids, i);
                    }
                    return 1 + i as u32;
                }
                break;
            }
        }
        if l.tag_alloc {
            let valids_end = (u64::from(l.valid_end) + 1) * 16;
            let words = (valids_end - valids) / 8;
            for i in 0..words {
                let word = self.rd(valids + i * 8, 8);
                for j in 0..64 {
                    if word >> j & 1 == 0 {
                        return 0x8000_0001 + (i * 64 + j) as u32;
                    }
                }
            }
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            return 0x8000_0001 + (self.rng % (words * 64)) as u32;
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Scenarios: hand-computed expectations.
// ---------------------------------------------------------------------------

/// Where scenarios stage data (`probe::DATA`), as 16-byte granules.
const TAGS: u64 = probe::DATA;
const VALIDS: u64 = probe::DATA + 0x400;
const BITS: u64 = probe::DATA + 0x800;

fn span(name: &'static str, byte: u64, granules: u32) -> Span {
    let first = (byte / 16) as u32;
    Span::new(name, first, first + granules - 1).unwrap()
}

fn le(bytes: &[u64], size: usize) -> Vec<u8> {
    bytes
        .iter()
        .flat_map(|v| v.to_le_bytes()[..size].to_vec())
        .collect()
}

#[derive(Clone)]
enum Do {
    /// Program a checked configuration, then load at its trigger address.
    Prog(Config),
    /// Raw masked stores, then a load at `load_at`.
    Raw(Vec<WriteStep>, u64),
}

#[derive(Clone, Debug)]
enum Expect {
    Word(u32),
    /// A random allocation: any of `0x8000_0001 ..= 0x8000_0000 + bits`.
    AllocBelow(u32),
}

struct Scenario {
    name: &'static str,
    stage: Vec<(u64, Vec<u8>)>,
    steps: Vec<(Do, Expect)>,
    /// Final contents of ranges, hand-computed.
    memory: Vec<(u64, Vec<u8>)>,
}

fn search(tag: u64, inv: bool, alloc: bool) -> Config {
    Config::search(
        span("tags", TAGS, 4),
        TagWidth::W32,
        tag,
        span("valids", VALIDS, 1),
        inv,
        alloc,
    )
    .unwrap()
}

/// Sixteen `u32` tags and a validity word `0b1_0111`: tags 0, 1, 2 and 4 valid, 3 not.
fn tags32() -> Vec<(u64, Vec<u8>)> {
    let tags: [u64; 16] = [5, 9, 0x1234, 77, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    vec![
        (TAGS, le(&tags, 4)),
        (VALIDS, le(&[0b1_0111, 0], 8)),
        (VALIDS + 16, vec![0xA5; 16]),
    ]
}

fn scenarios() -> Vec<Scenario> {
    let mut v = vec![];
    // Hit, invalidate-on-hit, hit on an invalid entry aborts (the later valid
    // duplicate is not found), allocation, miss, and an ordinary load that is not
    // at the trigger address passing through.
    v.push(Scenario {
        name: "search32",
        stage: tags32(),
        steps: vec![
            // tags[2] = 0x1234 and valid: returns 1 + 2.
            (Do::Prog(search(0x1234, false, false)), Expect::Word(3)),
            // tags[1] = 9 is the first match and valid: returns 2, clears bit 1.
            (Do::Prog(search(9, true, false)), Expect::Word(2)),
            // Now tags[1] matches but is invalid: abort, even though tags[4] = 9
            // is valid. No allocation: 0.
            (Do::Prog(search(9, false, false)), Expect::Word(0)),
            // The same with Tag_alloc: valids are 0b1_0101, first clear bit is 1.
            (Do::Prog(search(9, false, true)), Expect::Word(0x8000_0002)),
            // tags[3] = 77 is found but invalid: aborts, allocates bit 1.
            (Do::Prog(search(77, false, true)), Expect::Word(0x8000_0002)),
            // Absent tag, no allocation.
            (Do::Prog(search(0x55, false, false)), Expect::Word(0)),
            // Armed for search, but a load one granule on is an ordinary load:
            // tags[4] = 9 is its first word.
            (Do::Raw(vec![], TAGS + 16), Expect::Word(9)),
        ],
        memory: vec![
            (VALIDS, le(&[0b1_0101, 0], 8)),
            (VALIDS + 16, vec![0xA5; 16]),
        ],
    });
    // Allocation when every valid bit is set, and the first clear bit in a later word.
    let all = Scenario {
        name: "alloc_random",
        stage: vec![
            (TAGS, le(&[1, 2, 3, 4], 4)),
            (VALIDS, le(&[u64::MAX, u64::MAX], 8)),
        ],
        steps: vec![(Do::Prog(search(0x55, false, true)), Expect::AllocBelow(128))],
        memory: vec![(VALIDS, le(&[u64::MAX, u64::MAX], 8))],
    };
    v.push(all);
    v.push(Scenario {
        name: "alloc_second_word",
        stage: vec![
            (TAGS, le(&[1, 2, 3, 4], 4)),
            (VALIDS, le(&[u64::MAX, !(1 << 5)], 8)),
        ],
        steps: vec![(
            Do::Prog(search(0x55, false, true)),
            Expect::Word(0x8000_0001 + 64 + 5),
        )],
        memory: vec![(VALIDS, le(&[u64::MAX, !(1 << 5)], 8))],
    });
    // Tag widths. All tags valid so only the compare decides.
    let width = |w: TagWidth, size: usize, tags: &[u64], want: u64| {
        let tag_granules = ((tags.len() * size) as u64).div_ceil(16) as u32;
        (
            vec![
                (TAGS, le(tags, size)),
                (VALIDS, le(&[u64::MAX, u64::MAX], 8)),
            ],
            Do::Prog(
                Config::search(
                    span("tags", TAGS, tag_granules),
                    w,
                    want,
                    span("valids", VALIDS, 1),
                    false,
                    false,
                )
                .unwrap(),
            ),
        )
    };
    let (stage, d) = width(
        TagWidth::W8,
        1,
        &[1, 2, 0xAA, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        0xAA,
    );
    v.push(Scenario {
        name: "width8",
        stage,
        steps: vec![(d, Expect::Word(3))],
        memory: vec![],
    });
    // The low bytes collide (0x55); only a full compare finds index 1.
    let (stage, d) = width(
        TagWidth::W16,
        2,
        &[0x0155, 0x0255, 3, 4, 5, 6, 7, 8],
        0x0255,
    );
    v.push(Scenario {
        name: "width16",
        stage,
        steps: vec![(d, Expect::Word(2))],
        memory: vec![],
    });
    let (stage, d) = width(
        TagWidth::W32,
        4,
        &[0x1_0000, 0x2_0000, 0x3_0000, 4],
        0x2_0000,
    );
    v.push(Scenario {
        name: "width32",
        stage,
        steps: vec![(d, Expect::Word(2))],
        memory: vec![],
    });
    // 64-bit tags: tag 0 equals the value's low half only.
    let (stage, d) = width(
        TagWidth::W64,
        8,
        &[0x0000_0000_0000_0001, 0xDEAD_BEEF_0000_0001],
        0xDEAD_BEEF_0000_0001,
    );
    v.push(Scenario {
        name: "width64",
        stage,
        steps: vec![(d, Expect::Word(2))],
        memory: vec![],
    });
    // Invalidate-all: four words cleared, the words around them untouched.
    let inv = Config::invalidate_all(span("valids", VALIDS, 2));
    v.push(Scenario {
        name: "invalidate_all",
        stage: vec![
            (VALIDS - 16, vec![0xA5; 16]),
            (VALIDS, vec![0xFF; 32]),
            (VALIDS + 32, vec![0xA5; 16]),
        ],
        steps: vec![(Do::Prog(inv), Expect::Word(0))],
        memory: vec![
            (VALIDS - 16, vec![0xA5; 16]),
            (VALIDS, vec![0; 32]),
            (VALIDS + 32, vec![0xA5; 16]),
        ],
    });
    // Bit-vector query: bit 4 of word 0 and bit 63 of word 1 (index 127) set.
    let q = |off: u32| Do::Prog(Config::bit_query((BITS / 16) as u32, off).unwrap());
    v.push(Scenario {
        name: "bit_query",
        stage: vec![(BITS, le(&[1 << 4, 1 << 63], 8))],
        steps: vec![
            (q(4), Expect::Word(1)),
            (q(5), Expect::Word(0)),
            (q(63), Expect::Word(0)),
            (q(64), Expect::Word(0)),
            (q(127), Expect::Word(1)),
        ],
        memory: vec![(BITS, le(&[1 << 4, 1 << 63], 8))],
    });
    // The latch only follows trigger changes. After arming for tag 0x1234, a raw
    // store of tag 9 into Tag_Value_low leaves the latch on 0x1234 (result 3);
    // toggling Tag_inv re-latches, and the search now finds tag 9 and, because
    // Tag_inv is now set, clears its valid bit.
    let base = search(0x1234, false, false);
    let w214 = WriteStep {
        word: 214,
        mask: 0xffff_ffff,
        value: 9,
    };
    let inv_on = WriteStep {
        word: 219,
        mask: 1 << 24,
        value: 1 << 24,
    };
    v.push(Scenario {
        name: "latch_follows_triggers_only",
        stage: tags32(),
        steps: vec![
            (Do::Prog(base), Expect::Word(3)),
            (Do::Raw(vec![w214], TAGS), Expect::Word(3)),
            (Do::Raw(vec![inv_on], TAGS), Expect::Word(2)),
        ],
        memory: vec![
            (VALIDS, le(&[0b1_0101, 0], 8)),
            (VALIDS + 16, vec![0xA5; 16]),
        ],
    });
    v
}

fn steps_of(d: &Do) -> (Vec<WriteStep>, u64) {
    match d {
        Do::Prog(c) => (c.program().to_vec(), c.trigger_address()),
        Do::Raw(w, at) => (w.clone(), *at),
    }
}

/// What one run observed, from the model or from a device.
struct Observed {
    loads: Vec<u32>,
    #[cfg_attr(not(feature = "silicon"), allow(dead_code))]
    readback: Vec<[u32; 8]>,
    memory: Vec<Vec<u8>>,
}

fn run_model(s: &Scenario, mutant: Mutant) -> Observed {
    let mut m = Model::new(mutant);
    for (at, bytes) in &s.stage {
        m.l1[*at as usize..*at as usize + bytes.len()].copy_from_slice(bytes);
    }
    // The probe disarms first (clearing the five trigger bits).
    for st in ts::DISARM {
        m.write_cfg(u32::from(st.word), st.mask, st.value);
    }
    let mut loads = vec![];
    let mut readback = vec![];
    for (d, _) in &s.steps {
        let (writes, at) = steps_of(d);
        for w in writes {
            m.write_cfg(u32::from(w.word), w.mask, w.value);
        }
        readback.push(m.cfg);
        loads.push(m.load(at));
    }
    Observed {
        loads,
        readback,
        memory: s
            .memory
            .iter()
            .map(|(at, b)| m.l1[*at as usize..*at as usize + b.len()].to_vec())
            .collect(),
    }
}

fn check_expectations(s: &Scenario, o: &Observed) -> Result<(), String> {
    for (k, ((_, want), got)) in s.steps.iter().zip(&o.loads).enumerate() {
        let ok = match want {
            Expect::Word(w) => w == got,
            Expect::AllocBelow(n) => (0x8000_0001..=0x8000_0000 + n).contains(got),
        };
        if !ok {
            return Err(format!(
                "{} step {k}: loaded {got:#x}, expected {want:?}",
                s.name
            ));
        }
    }
    for ((at, want), got) in s.memory.iter().zip(&o.memory) {
        if want != got {
            return Err(format!(
                "{}: memory at {at:#x} is {got:02x?}, expected {want:02x?}",
                s.name
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Host gates.
// ---------------------------------------------------------------------------

#[test]
fn model_reproduces_the_hand_computed_expectations() {
    for s in scenarios() {
        check_expectations(&s, &run_model(&s, Mutant::None)).unwrap();
    }
}

/// Negative control: each wrong model must be rejected by the expectations.
#[test]
fn mutants_are_caught() {
    for mutant in [Mutant::TagCompareLowByte, Mutant::InvalidateAllFirstWord] {
        let caught: Vec<_> = scenarios()
            .iter()
            .filter_map(|s| check_expectations(s, &run_model(s, mutant)).err())
            .collect();
        assert!(!caught.is_empty(), "{mutant:?} passed every scenario");
        println!("MUTANT {mutant:?} rejected: {}", caught[0]);
    }
}

#[test]
fn register_layout_matches_the_vendored_header() {
    let header = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vendor/cfg_defines.h"
    ))
    .expect("run `cargo xtask fetch-spec` or copy vendor/");
    let define = |name: &str, what: &str| -> u32 {
        let key = format!("#define L1_CACHE_TAG_SEARCH_ACCEL_{name}_{what} ");
        let line = header
            .lines()
            .find(|l| l.starts_with(&key))
            .unwrap_or_else(|| panic!("{key}"));
        let v = line[key.len()..].trim();
        match v.strip_prefix("0x") {
            Some(h) => u32::from_str_radix(h, 16).unwrap(),
            None => v.parse().unwrap(),
        }
    };
    for (name, f) in ALL_FIELDS {
        assert_eq!(define(name, "ADDR32"), f.0, "{name} word");
        assert_eq!(define(name, "SHAMT"), f.1, "{name} shift");
        assert_eq!(define(name, "MASK"), f.2, "{name} mask");
    }
    // The helpers' generated table agrees with the same header.
    use tt_isa::cfg::generated::global as g;
    let pairs = [
        (g::L1_CACHE_TAG_SEARCH_ACCEL_Search_Enable, &SEARCH_ENABLE),
        (g::L1_CACHE_TAG_SEARCH_ACCEL_Start_Addr, &START_ADDR),
        (g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_Width, &TAG_WIDTH),
        (
            g::L1_CACHE_TAG_SEARCH_ACCEL_Valid_bit_section_start_addr,
            &VALID_START,
        ),
        (g::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_chk, &DV_CHK),
        (g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv, &TAG_INV),
        (g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_inv_all, &TAG_INV_ALL),
        (g::L1_CACHE_TAG_SEARCH_ACCEL_Tag_alloc, &TAG_ALLOC),
        (g::L1_CACHE_TAG_SEARCH_ACCEL_Data_Valid_offset, &DV_OFFSET),
    ];
    for (generated, f) in pairs {
        assert_eq!(
            (
                u32::from(generated.addr32()),
                u32::from(generated.shamt()),
                generated.mask()
            ),
            (f.0, f.1, f.2)
        );
    }
    assert_eq!(WORD_COUNT, 8);
    assert_eq!(ts::config_address(212), 0xFFEF_0000 + 212 * 4);
}

/// The helper's register program, run through the page's latch rule, leaves the
/// latch holding the requested fields, from any previous state -- including when
/// no trigger bit differs from the previous configuration, which is where a
/// single-pass write goes stale.
#[test]
fn program_latches_the_requested_fields_from_any_previous_state() {
    let configs = [
        search(0x1234, false, false),
        search(9, true, true),
        search(0x77, true, true), // same triggers as the previous: only the tag differs
        Config::invalidate_all(span("valids", VALIDS, 2)),
        Config::bit_query((BITS / 16) as u32, 100).unwrap(),
        Config::bit_query((BITS / 16) as u32, 101).unwrap(), // data-only change
        search(1, false, false),
    ];
    let mut m = Model::new(Mutant::None);
    for st in ts::DISARM {
        m.write_cfg(u32::from(st.word), st.mask, st.value);
    }
    for c in configs {
        for w in c.program() {
            m.write_cfg(u32::from(w.word), w.mask, w.value);
        }
        let f = c.fields();
        let l = m.latched;
        assert_eq!(l.search_enable, f.search_enable, "{:?}", c.op());
        assert_eq!(l.tag_alloc, f.tag_alloc);
        assert_eq!(l.tag_inv, f.tag_inv);
        assert_eq!(l.tag_inv_all, f.tag_inv_all);
        assert_eq!(l.data_valid_chk, f.data_valid_chk);
        assert_eq!(l.tag_width, f.tag_width);
        assert_eq!(l.tag_value, f.tag_value);
        assert_eq!(l.start_addr, f.start_addr);
        assert_eq!(l.end_addr, f.end_addr);
        assert_eq!(l.valid_start, f.valid_start_addr);
        assert_eq!(l.valid_end, f.valid_end_addr);
        assert_eq!(l.dv_start, f.data_valid_start_addr);
        assert_eq!(l.dv_offset, f.data_valid_offset);
        assert_eq!(m.latched.tag_inv_all, c.op() == Op::InvalidateAll);
        assert_eq!(PROGRAM_STEPS, 11);
    }
}

/// The pitfall `Config::program` exists to avoid: writing the new fields word by
/// word, triggers included, from a state whose trigger bits already match, never
/// re-latches, so the search keeps the old tag.
#[test]
fn a_single_pass_write_goes_stale_where_program_does_not() {
    let (a, b) = (search(0x1234, false, false), search(9, false, false));
    let mut naive = Model::new(Mutant::None);
    let mut good = Model::new(Mutant::None);
    for m in [&mut naive, &mut good] {
        for w in a.program() {
            m.write_cfg(u32::from(w.word), w.mask, w.value);
        }
        assert_eq!(m.latched.tag_value, 0x1234);
    }
    for (i, word) in b.words().0.iter().enumerate() {
        naive.write_cfg(212 + i as u32, ts::word_mask(212 + i as u16), *word);
    }
    assert_eq!(
        naive.latched.tag_value, 0x1234,
        "no trigger changed, so no re-latch"
    );
    assert_eq!(naive.cfg[2], 9, "the register itself did change");
    for w in b.program() {
        good.write_cfg(u32::from(w.word), w.mask, w.value);
    }
    assert_eq!(good.latched.tag_value, 9);
}

/// What the page says about overlapping conditions: Tag_inv_all blocks both the
/// search and the bit-vector query, so a load at their trigger granule is an
/// ordinary load.
#[test]
fn invalidate_all_blocks_search_and_query() {
    let mut m = Model::new(Mutant::None);
    m.l1[0x30000..0x30004].copy_from_slice(&3u32.to_le_bytes());
    m.write_cfg(212, !0, 1 | (0x3000 << 1)); // Search_Enable, Start_Addr = 0x3000
    m.write_cfg(213, !0, 0x3000);
    m.write_cfg(214, !0, 3);
    m.write_cfg(216, !0, 2 | (0x3040 << 2)); // W32 tags, validity at 0x3040
    m.write_cfg(217, !0, 0x3040);
    m.write_cfg(218, !0, (1 << 17) | 0x3000); // Data_Valid_chk, bit vector at 0x3000
    m.write_cfg(219, !0, 1 << 25); // Tag_inv_all
                                   // Search (tag 3 would hit at index 0, giving 1) and query (bit 0 is set, giving
                                   // 1) are both blocked: the load returns the raw word, 3.
    assert_eq!(m.load(0x30000), 3);
    // The read at the validity start is the invalidate-all trigger.
    m.l1[0x30400] = 0xFF;
    assert_eq!(m.load(0x30400), 0);
    assert_eq!(m.rd(0x30400, 4), 0);
}

#[test]
fn trigger_addresses_name_the_page_registers() {
    let tags = span("tags", TAGS, 4);
    let valids = span("valids", VALIDS, 1);
    let s = Config::search(tags, TagWidth::W32, 1, valids, false, false).unwrap();
    assert_eq!(s.trigger_address(), TAGS);
    assert_eq!(Config::invalidate_all(valids).trigger_address(), VALIDS);
    assert_eq!(
        Config::bit_query((BITS / 16) as u32, 3)
            .unwrap()
            .trigger_address(),
        BITS
    );
    assert_eq!(
        ts::decode_search(0x8000_0002),
        ts::SearchOutcome::Allocate { index: 1 }
    );
}

// ---------------------------------------------------------------------------
// Device runs (ttsim and silicon).
// ---------------------------------------------------------------------------

use tt_device::tlb::WindowKind;
use tt_isa::mailbox::{self, status};
use tt_tests::harness::{self, Dev};

/// How the probe runs: the normal read-modify-write form, or the simulator
/// diagnostics (`probe::SCRIPT`'s header words 1 and 2).
#[derive(Clone, Copy)]
struct Mode {
    blind: bool,
    keep: bool,
}
#[cfg_attr(not(feature = "silicon"), allow(dead_code))]
const NORMAL: Mode = Mode {
    blind: false,
    keep: false,
};

/// Host wait budget for one probe run, in simulated cycles (on silicon the
/// device layer converts it with a one second floor). Finite: a hung probe is an
/// error with a diagnosis, not a stuck test.
const PROBE_BUDGET: u64 = 4_000_000;

/// Read the breadcrumb and the stuck step's readback, as text.
fn diagnosis(
    dev: &mut Dev<'_>,
    w: &tt_device::Window,
    tile: tt_isa::noc::NocCoord<tt_isa::noc::Noc0>,
    why: &str,
) -> String {
    let mut word = |at: u64| dev.read32(w, tile, at).unwrap_or(0xDEAD_DEAD);
    let stage = word(probe::STAGE);
    let (step, phase) = (stage >> 8, stage & 0xFF);
    let rec = probe::RESULTS + u64::from(step) * probe::RESULT_STRIDE;
    let cfg: Vec<u32> = (0..8).map(|i| word(rec + 4 + 4 * i)).collect();
    let panic = word(mailbox::PANIC_CODE);
    format!(
        "{why}: last stage step {step} phase {} ({phase}), panic code {panic}, \
         Config[212..=219] readback of that step {cfg:#x?}, its load result {:#x}",
        probe::phase::name(phase),
        word(rec)
    )
}

/// Stage `s` and its script, run the probe on RISCV B, and read everything back.
///
/// On a hang or panic the error names the last breadcrumb and the `Config`
/// readback, and (unless `mode.keep`) a disarm-only run first resets B and clears
/// the block's trigger bits, so a failed gate does not leave it armed.
fn run_on_device(dev: &mut Dev<'_>, s: &Scenario, mode: Mode) -> Result<Observed, String> {
    let r = run_once(dev, s, mode);
    if r.is_err() && !mode.keep {
        let cleanup = Scenario {
            name: "disarm",
            stage: vec![],
            steps: vec![(Do::Raw(vec![], probe::DATA + 0x2000), Expect::Word(0))],
            memory: vec![],
        };
        let c = run_once(dev, &cleanup, NORMAL);
        return r.map_err(|e| {
            format!(
                "{e}; disarm-only cleanup run: {}",
                c.map_or_else(|e| e, |_| "ok".into())
            )
        });
    }
    r
}

fn run_once(dev: &mut Dev<'_>, s: &Scenario, mode: Mode) -> Result<Observed, String> {
    let tile = harness::tensix_tile();
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    // Config is only reachable with the Tensix backend out of reset.
    dev.release_tensix_backend(&w, tile).unwrap();
    for (at, bytes) in &s.stage {
        dev.write(&w, tile, *at, bytes).unwrap();
    }
    dev.write32(&w, tile, probe::STAGE, 0).unwrap();
    dev.write32(&w, tile, probe::SCRIPT, s.steps.len() as u32)
        .unwrap();
    dev.write32(&w, tile, probe::SCRIPT + 4, u32::from(mode.blind))
        .unwrap();
    dev.write32(&w, tile, probe::SCRIPT + 8, u32::from(mode.keep))
        .unwrap();
    for (k, (d, _)) in s.steps.iter().enumerate() {
        let (writes, at) = steps_of(d);
        let mut words = vec![at as u32, writes.len() as u32];
        for st in &writes {
            words.extend([u32::from(st.word), st.mask, st.value]);
        }
        let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
        dev.write(
            &w,
            tile,
            probe::STEPS + k as u64 * probe::STEP_STRIDE,
            &bytes,
        )
        .unwrap();
    }
    dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
    let (core, image, at) = tt_firmware_images::TAG_SEARCH_B;
    dev.load_and_start(&w, tile, core, image, at).unwrap();
    let done = dev
        .wait_for_status(&w, tile, PROBE_BUDGET, |x| x == status::DONE)
        .unwrap();
    if let Err(e) = done {
        let msg = diagnosis(dev, &w, tile, &format!("{}: {e}", s.name));
        // Hold B (a stuck load does not release on its own) before returning.
        let _ = dev.set_core_reset(&w, tile, core, true);
        return Err(msg);
    }
    let mut loads = vec![];
    let mut readback = vec![];
    for k in 0..s.steps.len() as u64 {
        let mut rec = [0u32; 9];
        for (i, r) in rec.iter_mut().enumerate() {
            *r = dev
                .read32(
                    &w,
                    tile,
                    probe::RESULTS + k * probe::RESULT_STRIDE + i as u64 * 4,
                )
                .unwrap();
        }
        loads.push(rec[0]);
        readback.push(rec[1..].try_into().unwrap());
    }
    let memory = s
        .memory
        .iter()
        .map(|(at, b)| {
            let mut out = vec![0u8; b.len()];
            dev.read(&w, tile, *at, &mut out).unwrap();
            out
        })
        .collect();
    dev.set_core_reset(&w, tile, core, true).unwrap();
    Ok(Observed {
        loads,
        readback,
        memory,
    })
}

/// A device's readback matches the model's `Config` words under the fields' masks.
#[cfg_attr(not(feature = "silicon"), allow(dead_code))]
fn readback_matches(s: &Scenario, device: &Observed) -> Result<(), String> {
    let model = run_model(s, Mutant::None);
    for (k, (d, m)) in device.readback.iter().zip(&model.readback).enumerate() {
        for i in 0..8 {
            let mask = ts::word_mask(212 + i as u16);
            if d[i] & mask != m[i] & mask {
                return Err(format!(
                    "{} step {k} word {}: read {:#x}, model {:#x}",
                    s.name,
                    212 + i,
                    d[i] & mask,
                    m[i] & mask
                ));
            }
        }
    }
    Ok(())
}

/// What ttsim does with the block (divergence row 92, proposed).
#[cfg(not(feature = "silicon"))]
#[test]
fn simulator_does_not_implement_the_block() {
    // One Raw step per `Config` word, no disarm: ttsim refuses every access to the
    // block's eight words, so its semantics cannot run there and the gates are
    // silicon-only. The control accesses no `Config` word and loads one staged word.
    let one = |writes: Vec<WriteStep>| Scenario {
        name: "word",
        stage: vec![(TAGS, le(&[0x1234_5678], 4))],
        steps: vec![(Do::Raw(writes, TAGS), Expect::Word(0x1234_5678))],
        memory: vec![],
    };
    let survives = |writes: Vec<WriteStep>, blind: bool| {
        let s = one(writes);
        harness::survives(|dev| {
            let o = run_on_device(dev, &s, Mode { blind, keep: true }).expect("finished");
            assert_eq!(o.loads, [0x1234_5678]);
        })
    };
    assert!(survives(vec![], true), "the control must survive");
    for word in 212..=219u16 {
        let w = vec![WriteStep {
            word,
            mask: 0,
            value: 0,
        }];
        let store = survives(w.clone(), true);
        let load = survives(w, false);
        // `tensix_cfg_wr32` (UnsupportedFunctionality) and `tensix_cfg_rd32`
        // (UnimplementedFunctionality) both refuse these words.
        assert!(!store, "ttsim now accepts a store to Config[{word}]");
        assert!(!load, "ttsim now accepts a load of Config[{word}]");
    }
}

// ---------------------------------------------------------------------------
// Silicon: written, not run. The minimal probe goes first, inside `survives`.
// ---------------------------------------------------------------------------

/// Risk class: documented (BlackholeA0 page), UNVERIFIED on silicon. The probe
/// only stores to `Config` words 212..=219, loads one L1 word of a staged tag
/// array, and disarms. First run: `silicon_tag_search_minimal_probe`. Every
/// scenario is its own test and its own child process, so a hang in one is
/// isolated and reports the last probe stage reached.
#[cfg(feature = "silicon")]
mod silicon {
    use super::*;

    fn scenario(name: &str) -> Scenario {
        scenarios().into_iter().find(|s| s.name == name).unwrap()
    }

    /// Run `name` in a fresh child; a hang or panic names its last stage.
    fn run_checked(name: &str) {
        let s = scenario(name);
        harness::in_device(|dev| match run_on_device(dev, &s, NORMAL) {
            Ok(o) => {
                println!("SILICON {name} loads {:#x?}", o.loads);
                check_expectations(&s, &o).unwrap();
                readback_matches(&s, &o).unwrap();
            }
            Err(e) => {
                eprintln!("SILICON FAILURE {e}");
                panic!("{e}");
            }
        });
    }

    #[test]
    #[ignore = "hangs the baby core on card 0 (hardware finding, see hardware-coverage.md L1CacheTagSearchAccel) and leaves the block armed; do not run"]
    fn silicon_tag_search_minimal_probe() {
        // One search over four tags, one load, disarm. Hung core or dead NoC
        // fails the child, not the host.
        let s = scenario("width32");
        assert!(harness::survives(|dev| {
            let o = run_on_device(dev, &s, NORMAL).expect("the probe finished");
            println!("SILICON minimal probe loads {:#x?}", o.loads);
        }));
    }

    macro_rules! per_scenario {
        ($($test:ident => $name:literal),* $(,)?) => {
            $( #[test] #[ignore = "hangs the baby core on card 0 (hardware finding, see hardware-coverage.md L1CacheTagSearchAccel) and leaves the block armed; do not run"] fn $test() { run_checked($name); } )*
        };
    }
    per_scenario! {
        silicon_tag_search_search32 => "search32",
        silicon_tag_search_alloc_random => "alloc_random",
        silicon_tag_search_alloc_second_word => "alloc_second_word",
        silicon_tag_search_width8 => "width8",
        silicon_tag_search_width16 => "width16",
        silicon_tag_search_width32 => "width32",
        silicon_tag_search_width64 => "width64",
        silicon_tag_search_invalidate_all => "invalidate_all",
        silicon_tag_search_bit_query => "bit_query",
        silicon_tag_search_latch_follows_triggers_only => "latch_follows_triggers_only",
    }

    /// The model mutant must disagree with silicon in the scenario its
    /// expectation check rejects it in.
    fn mutant_diverges(mutant: Mutant, name: &str) {
        let s = scenario(name);
        harness::in_device(|dev| {
            let o = run_on_device(dev, &s, NORMAL).unwrap_or_else(|e| panic!("{e}"));
            assert!(
                check_expectations(&s, &o).is_ok(),
                "silicon disagrees with the page"
            );
            assert!(
                check_expectations(&s, &run_model(&s, mutant)).is_err(),
                "{mutant:?} agrees with silicon in {name}"
            );
        });
    }

    #[test]
    #[ignore = "hangs the baby core on card 0 (hardware finding, see hardware-coverage.md L1CacheTagSearchAccel) and leaves the block armed; do not run"]
    fn silicon_rejects_the_tag_compare_mutant() {
        mutant_diverges(Mutant::TagCompareLowByte, "width16");
    }

    #[test]
    #[ignore = "hangs the baby core on card 0 (hardware finding, see hardware-coverage.md L1CacheTagSearchAccel) and leaves the block armed; do not run"]
    fn silicon_rejects_the_invalidate_scope_mutant() {
        mutant_diverges(Mutant::InvalidateAllFirstWord, "invalidate_all");
    }
}
