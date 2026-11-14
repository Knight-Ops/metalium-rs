//! Instruction-cache geometry probe on RISCV B (`src/icache_alias.rs`).

#![no_std]
#![no_main]

/// Where the rows go (`mailbox::ICACHE_ALIAS_RESULTS`).
const RESULTS: u64 = tt_isa::mailbox::ICACHE_ALIAS_RESULTS;

/// Bytes of callable `ret`s: room for eight addresses 4 KiB apart.
const REGION: u32 = 32 * 1024;

include!("../icache_alias.rs");
