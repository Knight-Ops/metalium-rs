//! Tensix backend configuration fields.
//!
//! The field table in [`generated`] is produced by `cargo xtask gen-cfg` from
//! tt-metal's `cfg_defines.h` at the commit `PINS.toml` records — the same commit
//! `BackendConfiguration.md:17` cites. It is **generated, never transcribed**:
//! there are 820 fields, and hand-copying them is a guaranteed bug source.
//!
//! # The two variables, and why they are different types
//!
//! `BackendConfiguration.md` describes two separate configuration variables,
//! mapped one after the other into the address space of RISCV B / T0 / T1 / T2
//! starting at [`tensix::TENSIX_CFG_BASE`](crate::tensix::TENSIX_CFG_BASE):
//!
//! ```text
//! uint32_t Config[2][CFG_STATE_SIZE * 4];
//! uint32_t ConfigDualWrite[CFG_STATE_SIZE * 4];
//! struct {uint16_t Value, Padding;} ThreadConfig[3][THD_STATE_SIZE];
//! ```
//!
//! Which one a field belongs to decides how it may be written, and the two are
//! not interchangeable:
//!
//! | | [`ConfigField`] | [`ThreadConfigField`] |
//! |---|---|---|
//! | Source section | everything except `// Registers for THREAD` | `// Registers for THREAD` |
//! | Value width | 32 bits | **16 bits** |
//! | Tensix write | `WRCFG`, `STREAMWRCFG`, `RMWCIB`, `CFGSHIFTMASK` | `SETC16` |
//! | RISC-V write | `sw` only | **none** |
//! | RISC-V read | yes | yes |
//!
//! Keeping them as distinct types is the point of this module: the convention is
//! stated in every configuration page in the specification, and getting it wrong
//! means writing a field with an instruction that addresses a different variable
//! entirely. [`ThreadConfigField`] therefore has no write address at all, and
//! `SETC16`/`WRCFG` wrappers can take the type they actually accept.

pub mod generated;

use crate::tensix::TENSIX_CFG_BASE;

pub use generated::{CFG_STATE_SIZE, THD_STATE_SIZE};

/// `u32` words in one bank of `Config`.
pub const CONFIG_WORDS_PER_BANK: u32 = CFG_STATE_SIZE * 4;
/// Number of `Config` banks.
pub const CONFIG_BANKS: u32 = 2;
/// Number of `ThreadConfig` banks, one per Tensix thread.
pub const THREAD_CONFIG_BANKS: u32 = 3;

/// Which bank of `Config` an access refers to.
///
/// Any Tensix thread can access any bank; which one is in use is selected by
/// `ThreadConfig[CurrentThread].CFG_STATE_ID_StateID`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ConfigBank {
    Bank0 = 0,
    Bank1 = 1,
}

/// A bitfield within the thread-agnostic `Config` variable.
///
/// Construct only via [`generated`]; the fields are private so that a
/// hand-written `ConfigField` cannot quietly enter the table.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ConfigField {
    addr32: u16,
    shamt: u8,
    mask: u32,
}

/// A bitfield within the thread-specific `ThreadConfig` variable.
///
/// Values are 16 bits: `ThreadConfig` entries are `struct {uint16_t Value,
/// uint16_t Padding;}`, and `SETC16` — the only instruction that writes them —
/// carries a 16-bit immediate.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ThreadConfigField {
    addr32: u16,
    shamt: u8,
    mask: u16,
}

/// A run of consecutive `Config` words addressed as a unit.
///
/// Two fields in `cfg_defines.h` — the `TileDescriptor` aggregates — have a
/// 128-bit mask and a shift of zero. They are not bitfields within a word but
/// whole four-word structures, so they get their own type rather than being
/// forced into a `u32` mask that cannot hold them.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ConfigSpan {
    addr32: u16,
    words: u16,
}

impl ConfigField {
    /// Called by generated code only.
    #[doc(hidden)]
    pub const fn new(addr32: u16, shamt: u8, mask: u32) -> Self {
        ConfigField {
            addr32,
            shamt,
            mask,
        }
    }

    /// Index of the containing word within a `Config` bank.
    pub const fn addr32(self) -> u16 {
        self.addr32
    }

    pub const fn shamt(self) -> u8 {
        self.shamt
    }

    pub const fn mask(self) -> u32 {
        self.mask
    }

    /// Largest value this field can hold.
    pub const fn max_value(self) -> u32 {
        self.mask >> self.shamt
    }

    /// Read this field out of the word that contains it.
    pub const fn extract(self, word: u32) -> u32 {
        (word & self.mask) >> self.shamt
    }

    /// Place `value` into the word that contains this field, leaving the rest
    /// of the word alone.
    ///
    /// A value too large for the field is truncated by the mask rather than
    /// spilling into neighbouring fields. Use [`Self::fits`] to check first
    /// where that matters.
    pub const fn insert(self, word: u32, value: u32) -> u32 {
        (word & !self.mask) | ((value << self.shamt) & self.mask)
    }

    pub const fn fits(self, value: u32) -> bool {
        value <= self.max_value()
    }

    /// Byte address of the containing word, for a RISC-V `lw` or `sw`.
    ///
    /// RISC-V may both read and write `Config`, but **only with `sw`** — a
    /// narrower store does not work.
    pub const fn riscv_address(self, bank: ConfigBank) -> u64 {
        let index = (bank as u32) * CONFIG_WORDS_PER_BANK + self.addr32 as u32;
        TENSIX_CFG_BASE + (index as u64) * 4
    }

    /// Byte address of the containing word in `ConfigDualWrite`, where a single
    /// store lands in both banks at once.
    ///
    /// Reads from `ConfigDualWrite` are `UnsupportedFunctionality`; in practice
    /// one behaves like a read of bank 0. Write-only by intent here.
    pub const fn riscv_dual_write_address(self) -> u64 {
        let index = CONFIG_BANKS * CONFIG_WORDS_PER_BANK + self.addr32 as u32;
        TENSIX_CFG_BASE + (index as u64) * 4
    }
}

impl ThreadConfigField {
    /// Called by generated code only.
    #[doc(hidden)]
    pub const fn new(addr32: u16, shamt: u8, mask: u16) -> Self {
        ThreadConfigField {
            addr32,
            shamt,
            mask,
        }
    }

    /// Index of the containing entry within one thread's `ThreadConfig` bank.
    ///
    /// This is also the immediate `SETC16` takes.
    pub const fn addr32(self) -> u16 {
        self.addr32
    }

    pub const fn shamt(self) -> u8 {
        self.shamt
    }

    pub const fn mask(self) -> u16 {
        self.mask
    }

    pub const fn max_value(self) -> u16 {
        self.mask >> self.shamt
    }

    pub const fn extract(self, value: u16) -> u16 {
        (value & self.mask) >> self.shamt
    }

    pub const fn insert(self, entry: u16, value: u16) -> u16 {
        (entry & !self.mask) | ((value << self.shamt) & self.mask)
    }

    pub const fn fits(self, value: u16) -> bool {
        value <= self.max_value()
    }

    /// Byte address of the containing entry, for a RISC-V **read**.
    ///
    /// There is deliberately no write counterpart: RISC-V cannot write
    /// `ThreadConfig` at all. `SETC16` is the only way, and it addresses the
    /// entry by [`Self::addr32`] rather than by byte address.
    ///
    /// `thread` must be 0, 1 or 2. The specification indexes this as
    /// `[CurrentThread]`, so a core reading its own configuration passes its own
    /// thread number.
    pub const fn riscv_read_address(self, thread: u32) -> u64 {
        debug_assert!(thread < THREAD_CONFIG_BANKS);
        // ThreadConfig follows both Config banks and ConfigDualWrite, and its
        // entries are four bytes each despite holding 16-bit values.
        let base = (CONFIG_BANKS + 1) * CONFIG_WORDS_PER_BANK;
        let index = base + thread * THD_STATE_SIZE + self.addr32 as u32;
        TENSIX_CFG_BASE + (index as u64) * 4
    }
}

impl ConfigSpan {
    /// Called by generated code only.
    #[doc(hidden)]
    pub const fn new(addr32: u16, words: u16) -> Self {
        ConfigSpan { addr32, words }
    }

    pub const fn addr32(self) -> u16 {
        self.addr32
    }

    /// Length of the span in 32-bit words.
    pub const fn words(self) -> u16 {
        self.words
    }

    /// Byte address of the first word, for a RISC-V `lw`/`sw` sequence.
    pub const fn riscv_address(self, bank: ConfigBank) -> u64 {
        let index = (bank as u32) * CONFIG_WORDS_PER_BANK + self.addr32 as u32;
        TENSIX_CFG_BASE + (index as u64) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::generated::{alu, thcon, thread};
    use super::*;

    #[test]
    fn state_sizes_match_the_header() {
        assert_eq!(CFG_STATE_SIZE, 56);
        assert_eq!(THD_STATE_SIZE, 68);
        assert_eq!(CONFIG_WORDS_PER_BANK, 224);
    }

    /// The decisive check on the parse *and* the section classification.
    ///
    /// `Config` is declared `[CFG_STATE_SIZE * 4]` and `ThreadConfig` is
    /// `[THD_STATE_SIZE]`. If a field were placed in the wrong variable — the
    /// exact mistake this module's types exist to prevent — its index would very
    /// likely fall outside the other variable's bounds.
    #[test]
    fn every_field_is_in_bounds_for_its_variable() {
        for (name, f) in generated::ALL_CONFIG_FIELDS {
            assert!(
                (f.addr32() as u32) < CONFIG_WORDS_PER_BANK,
                "{name}: Config index {} exceeds {CONFIG_WORDS_PER_BANK}",
                f.addr32()
            );
        }
        for (name, f) in generated::ALL_THREAD_CONFIG_FIELDS {
            assert!(
                (f.addr32() as u32) < THD_STATE_SIZE,
                "{name}: ThreadConfig index {} exceeds {THD_STATE_SIZE}",
                f.addr32()
            );
        }
    }

    #[test]
    fn the_table_has_the_expected_shape() {
        // 820 fields in the pinned header, two of which are whole-register
        // aggregates rather than bitfields. (The header also carries seven
        // `<SECTION>_CFGREG_BASE_ADDR32` defines, which are section metadata and
        // are emitted as each module's `CFGREG_BASE` rather than as fields.)
        let total = generated::ALL_CONFIG_FIELDS.len()
            + generated::ALL_THREAD_CONFIG_FIELDS.len()
            + generated::ALL_CONFIG_SPANS.len();
        assert_eq!(total, 820);
        assert_eq!(generated::ALL_THREAD_CONFIG_FIELDS.len(), 223);
        assert_eq!(generated::ALL_CONFIG_SPANS.len(), 2);
    }

    /// The `Config` sections partition the bank exactly, in the order their
    /// `CFGREG_BASE` values give. Every field must fall inside its own section.
    ///
    /// This is a much stronger statement than the bounds check above: a field
    /// attributed to the wrong section would still be in bounds for `Config`,
    /// but would land outside that section's range.
    #[test]
    fn config_sections_partition_the_bank() {
        use super::generated::{global, pack0, thcon, unpack0, unpack1};

        /// Module name, first word index, and that section's fields.
        type Section = (&'static str, u16, &'static [(&'static str, ConfigField)]);

        // In ascending base order.
        let sections: [Section; 6] = [
            ("alu", alu::CFGREG_BASE, generated::ALU_FIELDS),
            ("pack0", pack0::CFGREG_BASE, generated::PACK0_FIELDS),
            ("unpack0", unpack0::CFGREG_BASE, generated::UNPACK0_FIELDS),
            ("unpack1", unpack1::CFGREG_BASE, generated::UNPACK1_FIELDS),
            ("thcon", thcon::CFGREG_BASE, generated::THCON_FIELDS),
            ("global", global::CFGREG_BASE, generated::GLOBAL_FIELDS),
        ];

        let mut previous_end = 0u16;
        for (i, (name, base, fields)) in sections.iter().enumerate() {
            assert_eq!(
                *base, previous_end,
                "{name} does not start where the previous section ends"
            );
            let end = sections
                .get(i + 1)
                .map(|(_, b, _)| *b)
                .unwrap_or(CONFIG_WORDS_PER_BANK as u16);
            for (field, f) in *fields {
                assert!(
                    f.addr32() >= *base && f.addr32() < end,
                    "{field} has index {} but section {name} spans {base}..{end}",
                    f.addr32()
                );
            }
            previous_end = end;
        }
        assert_eq!(
            previous_end, CONFIG_WORDS_PER_BANK as u16,
            "sections leave a gap at the end"
        );
    }

    #[test]
    fn masks_and_shifts_are_consistent() {
        for (name, f) in generated::ALL_CONFIG_FIELDS {
            assert_ne!(f.mask(), 0, "{name} has an empty mask");
            // The mask's lowest set bit must be the shift amount, or extract and
            // insert disagree about where the field starts.
            assert_eq!(
                f.mask().trailing_zeros(),
                f.shamt() as u32,
                "{name}: mask {:#x} does not start at shift {}",
                f.mask(),
                f.shamt()
            );
            // Contiguous: a bitfield with a hole would silently corrupt
            // neighbours on insert.
            let normalised = f.mask() >> f.shamt();
            assert!(
                normalised.count_ones() == normalised.trailing_ones(),
                "{name}: mask {:#x} is not contiguous",
                f.mask()
            );
        }
        for (name, f) in generated::ALL_THREAD_CONFIG_FIELDS {
            assert_ne!(f.mask(), 0, "{name} has an empty mask");
            assert_eq!(f.mask().trailing_zeros(), f.shamt() as u32, "{name}");
        }
    }

    #[test]
    fn extract_and_insert_round_trip() {
        let f = thread::CFG_STATE_ID_StateID;
        for v in 0..=f.max_value() {
            assert_eq!(f.extract(f.insert(0, v)), v);
            // Neighbouring bits must survive: starting from an all-ones entry,
            // every bit outside the field must still be set afterwards.
            let busy = f.insert(u16::MAX, v);
            assert_eq!(busy & !f.mask(), !f.mask());
        }

        let g = alu::ALU_FORMAT_SPEC_REG0_SrcAUnsigned;
        assert_eq!(g.max_value(), 1);
        assert_eq!(g.extract(g.insert(0xDEAD_BEEF, 1)), 1);
        assert_eq!(g.extract(g.insert(0xDEAD_BEEF, 0)), 0);
    }

    #[test]
    fn oversized_values_are_truncated_not_spilled() {
        let f = alu::ALU_FORMAT_SPEC_REG0_SrcAUnsigned;
        // One bit wide, so 3 cannot fit. The excess must not reach neighbours.
        let word = f.insert(0, 3);
        assert_eq!(word & !f.mask(), 0, "insert spilled outside the field");
        assert!(!f.fits(3));
        assert!(f.fits(1));
    }

    /// The field step 4 relies on. Its presence here is what lets the Dst format
    /// be set deliberately rather than inherited from a reset default.
    #[test]
    fn the_dst_access_format_field_is_where_the_docs_say() {
        let f = alu::RISC_DEST_ACCESS_CTRL_SEC1_fmt;
        assert_eq!(f.addr32(), 3);
        assert_eq!(f.shamt(), 21);
        assert_eq!(f.mask(), 0x00e0_0000);
        assert_eq!(f.max_value(), 7, "fmt is three bits");
    }

    #[test]
    fn riscv_addresses_follow_the_declared_layout() {
        // Config[0][0] is the very start of the region.
        let f = ConfigField::new(0, 0, 0xFFFF_FFFF);
        assert_eq!(f.riscv_address(ConfigBank::Bank0), TENSIX_CFG_BASE);
        // Bank 1 begins one whole bank later.
        assert_eq!(
            f.riscv_address(ConfigBank::Bank1),
            TENSIX_CFG_BASE + (CONFIG_WORDS_PER_BANK as u64) * 4
        );
        // ConfigDualWrite follows both banks.
        assert_eq!(
            f.riscv_dual_write_address(),
            TENSIX_CFG_BASE + 2 * (CONFIG_WORDS_PER_BANK as u64) * 4
        );
        // ThreadConfig follows all three, with 4-byte entries.
        let t = ThreadConfigField::new(0, 0, 0xFFFF);
        assert_eq!(
            t.riscv_read_address(0),
            TENSIX_CFG_BASE + 3 * (CONFIG_WORDS_PER_BANK as u64) * 4
        );
        assert_eq!(
            t.riscv_read_address(1) - t.riscv_read_address(0),
            (THD_STATE_SIZE as u64) * 4
        );
    }

    #[test]
    fn the_whole_region_fits_the_documented_aperture() {
        // TENSIX_CFG_BASE spans 0xFFEF_0000..=0xFFEF_FFFF (BabyRISCV/README.md:135).
        let t = ThreadConfigField::new((THD_STATE_SIZE - 1) as u16, 0, 0xFFFF);
        let last = t.riscv_read_address(THREAD_CONFIG_BANKS - 1);
        assert!(
            last <= TENSIX_CFG_BASE + 0xFFFF,
            "configuration overruns its aperture"
        );
    }

    #[test]
    fn tile_descriptors_are_four_word_spans() {
        let d = thcon::THCON_SEC0_REG0_TileDescriptor;
        assert_eq!(d.words(), 4, "a 128-bit aggregate");
        assert_eq!(d.addr32(), 64);
        // And the SEC1 copy sits elsewhere.
        assert_ne!(thcon::THCON_SEC1_REG0_TileDescriptor.addr32(), d.addr32());
    }
}
