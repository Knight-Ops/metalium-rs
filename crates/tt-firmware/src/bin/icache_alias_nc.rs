//! Instruction-cache geometry probe on RISCV NC (`src/icache_alias.rs`), at NC's
//! image base, reached through the stub at its reset PC.

#![no_std]
#![no_main]

/// Where the rows go (`mailbox::ICACHE_ALIAS_RESULTS`), as `icache_alias_b`'s.
const RESULTS: u64 = tt_isa::mailbox::ICACHE_ALIAS_RESULTS;

/// Bytes of callable `ret`s: NC's slot is 24 KiB, so four addresses 4 KiB apart.
const REGION: u32 = 16 * 1024;

include!("../icache_alias.rs");
