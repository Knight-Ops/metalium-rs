//! Instruction-cache probe on RISCV B (`src/icache_probe.rs`).

#![no_std]
#![no_main]

/// Where the timings go: `[size, straight-line cycles, jump-chain cycles]`
/// per size, in the probes' own L1 (`mailbox::PROBE_SCRATCH`).
const RESULTS: u64 = tt_isa::mailbox::icache_results(0);

/// Each code block's size.
const BLOCK: u32 = 8192;

include!("../icache_probe.rs");
