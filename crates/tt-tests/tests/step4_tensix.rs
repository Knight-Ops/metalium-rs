//! Step 4 gate: a Tensix instruction executes and the host reads the result.
//!
//! The whole stack in one test. The host writes an image into L1 through a TLB
//! window, releases the Tensix backend and RISCV T0 from soft reset; T0 pushes an
//! SFPU program into the coprocessor, waits for it to retire, and copies `Dst[0][0]`
//! into the L1 mailbox; the host reads `0x40C0_0000` back out.
//!
//! Both ends of that chain are forced. `INSTRN_BUF_BASE` and `Dst` are unmapped to
//! the NoC, so the host can neither push the instructions nor read the answer — a
//! baby RISC-V core has to sit in the middle.

use tt_device::{core_control::WaitError, tlb::WindowKind, Device};
use tt_isa::mailbox::{self, status};
use tt_isa::noc::{grid, Noc0, NocCoord};
use tt_isa::tensix::Core;
use tt_tests::firmware;
use tt_ttsim::{fork_scope, Simulator};

type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

#[track_caller]
fn in_device(f: impl FnOnce(&mut Dev<'_>)) {
    let result = fork_scope(|| {
        let mut sim = Simulator::open().unwrap_or_else(|e| panic!("could not open simulator: {e}"));
        let mut dev = Device::open(sim.transport()).unwrap_or_else(|e| panic!("{e}"));
        f(&mut dev);
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}

fn tensix_tile(x: u8, y: u8) -> NocCoord<Noc0> {
    assert!(grid::is_tensix(x, y));
    NocCoord::new(x, y).unwrap()
}

const BUDGET: u64 = 400_000;

/// 6.0f32 — the answer the whole baseline exists to produce.
const EXPECTED: u32 = 0x40C0_0000;

/// Which core runs the SFPU program.
///
/// **T1, not T0.** ttsim implements the RISC-V view of `Dst` only for T1
/// (`tensix_dst_rd32` verifies `pipe == 1`), so a `Dst` read from T0 terminates the
/// process. The image is identical either way — `load_and_start` points the core's
/// reset-PC override at it — and the choice is a simulator constraint, not a
/// hardware one. See `docs/ttsim-divergence.md`.
const CORE: Core = Core::T1;

/// The Tensix thread [`CORE`] drives. `mhartid` reads zero on every core, so the
/// firmware cannot work this out for itself.
const CORE_THREAD: u32 = 1;

/// `RISC_DEST_ACCESS_CTRL_SEC*.fmt` values. ttsim implements 0, 2 and 3.
const DST_FMT_FP32: u32 = 0;
const DST_FMT_BF16: u32 = 3;

/// Multiply `a` by `b` on `tile`'s Vector Unit and return the FP32 bit pattern.
fn multiply_on_device(dev: &mut Dev<'_>, tile: NocCoord<Noc0>, a: f32, b: f32) -> u32 {
    multiply_with_fmt(dev, tile, a, b, DST_FMT_FP32)
}

/// As [`multiply_on_device`], but with a chosen `Dst` access format.
fn multiply_with_fmt(dev: &mut Dev<'_>, tile: NocCoord<Noc0>, a: f32, b: f32, fmt: u32) -> u32 {
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();

    // The backend has to come out of reset before the coprocessor will execute
    // anything. Done before the core starts, so there is no window in which the
    // core is pushing into a halted backend.
    dev.release_tensix_backend(&w, tile).unwrap();

    // Operands and a sentinel result, staged while the core is still held.
    dev.write32(&w, tile, mailbox::OPERAND_A, a.to_bits())
        .unwrap();
    dev.write32(&w, tile, mailbox::OPERAND_B, b.to_bits())
        .unwrap();
    dev.write32(&w, tile, mailbox::RESULT, SENTINEL).unwrap();
    dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
    dev.write32(&w, tile, mailbox::THREAD_INDEX, CORE_THREAD)
        .unwrap();
    dev.write32(&w, tile, mailbox::DST_ACCESS_FMT, fmt).unwrap();

    dev.load_and_start(&w, tile, CORE, firmware::SFPU_MUL, firmware::LOAD_ADDRESS)
        .unwrap();

    match dev
        .wait_for_status(&w, tile, BUDGET, |s| s == status::DONE)
        .unwrap()
    {
        Ok(_) => {}
        Err(WaitError::Panicked { code }) => panic!("firmware panicked, code {code}"),
        Err(e) => panic!("{e}"),
    }
    let result = dev.read32(&w, tile, mailbox::RESULT).unwrap();
    assert_ne!(result, SENTINEL, "the firmware never wrote a result");
    result
}

/// Written to the result slot before each run, so a stale or absent write is
/// distinguishable from a computed answer.
const SENTINEL: u32 = 0xDEAD_BEEF;

#[test]
fn sfpu_multiplies_three_by_two() {
    in_device(|dev| {
        let result = multiply_on_device(dev, tensix_tile(3, 4), 3.0, 2.0);
        assert_eq!(
            result,
            EXPECTED,
            "expected 6.0 ({EXPECTED:#010x}), got {result:#010x} ({})",
            f32::from_bits(result)
        );
        assert_eq!(f32::from_bits(result), 6.0);
    });
}

#[test]
fn the_result_is_bit_exact_not_approximate() {
    // ttsim's claim is bit-exactness relative to silicon, so the assertion is on
    // the bit pattern rather than within a tolerance. A test that passed with a
    // loose epsilon would not be evidence of much: the interesting failures in
    // this area are wrong NaN canonicalisation and flushed denormals, both of
    // which survive an epsilon comparison.
    in_device(|dev| {
        let result = multiply_on_device(dev, tensix_tile(5, 6), 3.0, 2.0);
        assert_eq!(result, 3.0f32.mul_add(2.0, 0.0).to_bits());
        // 6.0 == 1.5 x 2^2, so the mantissa field is 0x400000 and the biased
        // exponent is 129. Spelled out field by field because a swizzle slip in
        // the Dst read path would move exponent and mantissa bytes past each
        // other while leaving the value superficially plausible.
        assert_eq!(result >> 31, 0, "sign");
        assert_eq!((result >> 23) & 0xFF, 129, "biased exponent");
        assert_eq!(result & 0x007F_FFFF, 0x0040_0000, "mantissa");
    });
}

#[test]
fn it_works_on_more_than_one_tile() {
    // Guards against the answer coming from somewhere fixed -- a stale mailbox, or
    // a window that never retargeted -- rather than from the tile under test.
    in_device(|dev| {
        for (x, y) in [(1u8, 2u8), (16, 11), (7, 5)] {
            let tile = tensix_tile(x, y);
            assert_eq!(
                multiply_on_device(dev, tile, 3.0, 2.0),
                EXPECTED,
                "tile ({x},{y})"
            );
        }
    });
}

/// The real negative control: the device must be *computing*, not returning a
/// constant.
///
/// Each pair is checked against the host's own FP32 multiply. The operands are
/// chosen to have non-zero low 16 bits, so the `SFPLOADI` UPPER/LOWER split is
/// exercised rather than the BF16-representable happy path that 3.0 and 2.0 take.
#[test]
fn it_computes_rather_than_returning_a_constant() {
    in_device(|dev| {
        let tile = tensix_tile(6, 8);
        let cases: &[(f32, f32)] = &[
            (3.0, 2.0),
            (1.0, 1.0),
            (-4.5, 8.25),
            (0.1, 0.3),
            (1.234_567_9, 9.876_543),
            (65536.0, 65536.0),
            (-1.0, 0.0),
        ];
        let mut seen = std::collections::HashSet::new();
        for &(a, b) in cases {
            let got = multiply_on_device(dev, tile, a, b);
            assert_eq!(
                got,
                (a * b).to_bits(),
                "{a} * {b}: device gave {} ({got:#010x}), host gives {}",
                f32::from_bits(got),
                a * b
            );
            seen.insert(got);
        }
        assert!(
            seen.len() > 4,
            "the device returned suspiciously few distinct values"
        );
    });
}

/// The configuration write is real, not decoration.
///
/// The firmware sets `RISC_DEST_ACCESS_CTRL_SEC1_fmt` from the generated field
/// table before reading `Dst`. If that write did nothing, changing the requested
/// format would change nothing either — so asking for BF16 and getting the same
/// FP32 answer back would mean the whole `tt-isa` configuration path is inert.
///
/// The BF16 value is deliberately not asserted. `SFPSTORE` wrote a 32-bit datum
/// and this reads it back through a 16-bit shape, and the overlay between
/// `Dst32b` and `Dst16b` is not documented well enough to predict. That it
/// *differs* is the whole claim.
#[test]
fn the_dst_format_configuration_write_takes_effect() {
    in_device(|dev| {
        let tile = tensix_tile(4, 4);
        let as_fp32 = multiply_with_fmt(dev, tile, 3.0, 2.0, DST_FMT_FP32);
        assert_eq!(as_fp32, EXPECTED);

        let as_bf16 = multiply_with_fmt(dev, tile, 3.0, 2.0, DST_FMT_BF16);
        assert_ne!(
            as_bf16, as_fp32,
            "changing RISC_DEST_ACCESS_CTRL_SEC1_fmt changed nothing, so the \
             configuration write never reached the hardware"
        );
    });
}

/// ttsim does not model Tensix backend soft reset, so this cannot be a simulator
/// gate.
///
/// `SOFT_RESET_0` comes up at `0x0004_7800` — only the five baby RISC-V bits —
/// and ttsim's handler acts on those alone, rejecting a write that sets any other
/// bit (`TTSIM_VERIFY(!(data & ~0x47800))`). So the backend is permanently
/// released there and cannot be held. On silicon, holding bit 10 should make the
/// Vector Unit's instructions "not start (they might or might not be silently
/// discarded)", and silent discard is exactly the failure this checks for.
#[cfg(feature = "silicon")]
#[test]
fn a_held_backend_does_not_produce_the_answer() {
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix_tile(2, 3);

        dev.write32(&w, tile, mailbox::OPERAND_A, 3.0f32.to_bits())
            .unwrap();
        dev.write32(&w, tile, mailbox::OPERAND_B, 2.0f32.to_bits())
            .unwrap();
        dev.write32(&w, tile, mailbox::RESULT, SENTINEL).unwrap();
        dev.write32(&w, tile, mailbox::THREAD_INDEX, CORE_THREAD)
            .unwrap();
        dev.write32(&w, tile, mailbox::DST_ACCESS_FMT, DST_FMT_FP32)
            .unwrap();

        // Deliberately skip release_tensix_backend.
        dev.load_and_start(&w, tile, CORE, firmware::SFPU_MUL, firmware::LOAD_ADDRESS)
            .unwrap();
        let outcome = dev
            .wait_for_status(&w, tile, BUDGET, |s| s == status::DONE)
            .unwrap();

        match outcome {
            Ok(_) => {
                let result = dev.read32(&w, tile, mailbox::RESULT).unwrap();
                assert_ne!(
                    result, EXPECTED,
                    "the SFPU produced the correct answer while held in soft reset, \
                     so releasing the backend is not what makes the real test pass"
                );
            }
            Err(WaitError::TimedOut { .. }) => {}
            Err(e) => panic!("unexpected failure: {e}"),
        }
    });
}

#[test]
fn the_firmware_reaches_done_not_just_running() {
    // `finish` writes RESULT before STATUS, so observing DONE means RESULT is
    // already valid. Pinned because the ordering is the contract.
    in_device(|dev| {
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix_tile(4, 7);
        dev.release_tensix_backend(&w, tile).unwrap();
        dev.write32(&w, tile, mailbox::OPERAND_A, 3.0f32.to_bits())
            .unwrap();
        dev.write32(&w, tile, mailbox::OPERAND_B, 2.0f32.to_bits())
            .unwrap();
        dev.write32(&w, tile, mailbox::THREAD_INDEX, CORE_THREAD)
            .unwrap();
        dev.write32(&w, tile, mailbox::DST_ACCESS_FMT, DST_FMT_FP32)
            .unwrap();
        dev.load_and_start(&w, tile, CORE, firmware::SFPU_MUL, firmware::LOAD_ADDRESS)
            .unwrap();
        dev.wait_for_status(&w, tile, BUDGET, |s| s == status::DONE)
            .unwrap()
            .unwrap();
        assert_eq!(dev.read_status(&w, tile).unwrap(), status::DONE);
        assert_eq!(dev.read32(&w, tile, mailbox::RESULT).unwrap(), EXPECTED);
    });
}
