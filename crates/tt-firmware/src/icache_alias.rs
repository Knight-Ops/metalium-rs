// The instruction cache's geometry, by aliasing: shared by `bin/icache_alias_b.rs` and
// `bin/icache_alias_nc.rs`, each of which defines `RESULTS` (where in L1 the timings go) and
// `REGION` (the bytes of callable code), and includes this.
//
// `probe_icache` (`icache_probe.rs`) measured the capacity, about 4 KiB, but not how the cache is
// organised. That decides how code must be placed: whether two hot functions 4 KiB apart evict
// each other (direct-mapped), and how much a line holds. The region here is `REGION` bytes of
// `ret`, so calling any word of it touches one instruction: one line. A timed loop calls a set of
// such addresses, round after round:
// * ways: `n` addresses `k` bytes apart. Addresses a multiple of the set stride (capacity / ways)
//   apart fall in one set, so the cost per call steps up once `n` passes the ways (with LRU);
//   addresses less than a line apart share one.
// * line: every `step` bytes over a footprint, ascending and descending. Over a footprint larger
//   than the cache every line misses once a round, so a call costs a miss every `line / step`
//   calls: the step at which the cost stops rising is the line. A sequential prefetcher hides
//   ascending misses, not descending ones.
//
// The timing loop's own lines sit at `C` (about 300 bytes, 64-aligned); every set of addresses
// starts at `C + 448` (mod 4 KiB), which is in none of the loop's sets for any set stride from
// 512 bytes to 4 KiB (16 ways or more would see the loop's lines in the 256-byte rows).

use tt_firmware::{finish, l1_write32, publish};

core::arch::global_asm!(
    r#"
    .section .text.probe_alias, "ax"
    .balign 4096
    .global probe_alias_start
probe_alias_start:
    .rept {words}
    ret
    .endr
"#,
    words = const REGION / 4,
);

extern "C" {
    static probe_alias_start: u8;
}

/// Distances between the addresses of the ways test, in bytes.
const KS: [u32; 6] = [256, 512, 1024, 2048, 4096, 8192];
/// Most addresses at one distance.
const MOST: u32 = 8;
/// Steps of the line test, in bytes.
const STEPS: [u32; 7] = [4, 8, 16, 32, 64, 128, 256];
/// Footprints of the line test: one that fits the cache, one twice its measured size.
const FOOTPRINTS: [u32; 2] = [1024, 8192];
/// Timed rounds of the ways test, after one untimed.
const WAY_ROUNDS: u32 = 64;
/// Timed rounds of the line test, after one untimed.
const LINE_ROUNDS: u32 = 8;

fn clock() -> u32 {
    // SAFETY: the tile's wall clock, a read-only debug register every baby
    // core reaches (`BabyRISCV/README.md`, memory map).
    unsafe { core::ptr::read_volatile(tt_isa::tensix::timestamper::WALL_CLOCK_L as *const u32) }
}

/// Cycles for `rounds` rounds of calls at `first + i * step` for `i < n` (descending if `down`),
/// after one untimed round.
#[inline(never)]
fn time(first: usize, step: usize, n: usize, down: bool, rounds: u32) -> u32 {
    let round = || {
        for i in 0..n {
            let i = if down { n - 1 - i } else { i };
            // SAFETY: the address is a word inside the region of `ret`s (checked by the
            // callers), so the call returns at once, touching nothing.
            let f: extern "C" fn() = unsafe { core::mem::transmute(first + i * step) };
            f();
        }
    };
    round();
    let t0 = clock();
    for _ in 0..rounds {
        round();
    }
    clock().wrapping_sub(t0)
}

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    let region = core::ptr::addr_of!(probe_alias_start) as usize;
    let end = region + REGION as usize;
    // The first address of every set: 448 bytes past the timing loop, modulo 4 KiB.
    let loop_at = time as *const () as usize & !63;
    let first = region + ((loop_at + 448) & 4095);
    let mut at = RESULTS;
    let mut put = |a: u32, b: u32, cycles: u32| {
        // SAFETY: `RESULTS` is L1 this binary owns (see its definition), and the
        // rows written stay inside it (`ROWS`).
        unsafe {
            l1_write32(at, a);
            l1_write32(at + 4, b);
            l1_write32(at + 8, cycles);
        }
        at += 12;
    };
    for &k in &KS {
        for n in 1..=MOST {
            let fits = first + (n as usize - 1) * k as usize + 4 <= end;
            let cycles = if fits {
                time(first, k as usize, n as usize, false, WAY_ROUNDS)
            } else {
                0
            };
            put(k, n, cycles);
        }
    }
    for &footprint in &FOOTPRINTS {
        for &step in &STEPS {
            for down in [false, true] {
                let n = (footprint / step) as usize;
                let cycles = if first + footprint as usize <= end {
                    time(first, step as usize, n, down, LINE_ROUNDS)
                } else {
                    0
                };
                put(footprint | (down as u32) << 31, step, cycles);
            }
        }
    }
    publish();
    finish(ROWS);
    tt_firmware::spin()
}

/// Rows written, each `[a, b, cycles]`: the ways test's `[k, n, ..]`, then the line test's
/// `[footprint | descending << 31, step, ..]`.
const ROWS: u32 = KS.len() as u32 * MOST + (FOOTPRINTS.len() * STEPS.len() * 2) as u32;
const _: () = assert!(ROWS as u64 * 12 <= tt_isa::mailbox::ICACHE_ALIAS_BYTES);
