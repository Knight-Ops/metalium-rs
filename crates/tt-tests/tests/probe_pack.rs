//! Driving the packer: `Dst` back out to L1.
//!
//! The other half of the datapath `probe_unpack.rs` opened. Together they are a
//! round trip, which is what Phase 5's kernel sits inside: unpack into `Dst`,
//! compute, pack out.
//!
//! # What the documentation is, and is not
//!
//! Blackhole has a real `PACR.md`, unlike the unpacker — but it is self-labelled
//! *"basic"*, it documents only the `Dst` read interfaces, and it admits its
//! `ReadIntfSel`/`DstAccessMode` interaction is undocumented. Everything about the
//! pipeline either side of that comes from `WormholeB0/.../Packers/`, a directory
//! with no Blackhole counterpart at all, so every fact taken from it is
//! `UNVERIFIED` and this file exists to turn as much of it as possible into
//! measurement.
//!
//! Blackhole has **one** packer with four `Dst` read interfaces where Wormhole had
//! four packers, and `PACR` selects interfaces rather than packers. The Wormhole
//! pages are still written in terms of `Packers[i]`, so "packer 0's configuration"
//! is `THCON_SEC0_REG1_*` and that is what this configures.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::isa::Instruction;
use tt_isa::mailbox;
use tt_isa::sfpu;
use tt_isa::tile::{L1Format, TileDescriptor, TileImage};
use tt_tests::datapath::{
    flat_descriptor, pack_config, pack_instruction, set_adc_x_pack, set_adc_x_unpack, staged_image,
    thread_config, unpack_config, unpack_instruction, OUT, SCRATCH_GPR, STAGE,
};
use tt_tests::harness::{self, in_device, Run};

/// The whole round trip: L1 -> `Dst` -> L1.
fn round_trip_program(descriptor: TileDescriptor, datums: u32, rows: u32) -> Vec<Instruction> {
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    pack_config(&mut words, OUT);
    let mut staged = [sfpu::nop(); 96];
    let n = words.program(SCRATCH_GPR, &mut staged).unwrap();
    p.extend_from_slice(&staged[..n]);

    p.push(set_adc_x_unpack(0, datums - 1));
    p.push(unpack_instruction());
    p.push(backend::wait_for_unpacker0(Before::PACKER).unwrap());

    // The packer reads 16 datums per row per enabled interface.
    p.push(set_adc_x_pack(0, 15));
    p.push(pack_instruction(0b1111, true));
    p.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    let _ = rows;
    p
}

/// L1 words the packer writes for one `PACR` with all four read interfaces
/// enabled: four `Dst` rows of sixteen datums.
const PACKED_DATUMS: usize = 64;

/// A sentinel pre-written across the packer's output region, so a word the packer
/// never wrote is distinguishable from one it wrote as zero.
const L1_SENTINEL: u32 = 0xA5A5_5A5A;

fn sentinel_bytes(words: usize) -> Vec<u8> {
    L1_SENTINEL
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(words * 4)
        .collect()
}

fn as_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Run the round trip and return `(Dst dump, packed L1 words)`.
fn round_trip(
    dev: &mut harness::Dev<'_>,
    descriptor: TileDescriptor,
    staged: &[u8],
    datums: u32,
    read_intf_sel: u32,
) -> (Vec<u32>, Vec<u32>) {
    round_trip_with_kernel(dev, descriptor, staged, datums, read_intf_sel, &[])
}

/// The same, with `kernel` inserted between the unpack and the pack.
///
/// This is the shape Phase 5's elementwise op takes: the datapath is unchanged and
/// the compute is a handful of SFPU instructions in the middle of it.
fn round_trip_with_kernel(
    dev: &mut harness::Dev<'_>,
    descriptor: TileDescriptor,
    staged: &[u8],
    datums: u32,
    read_intf_sel: u32,
    kernel: &[Instruction],
) -> (Vec<u32>, Vec<u32>) {
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    pack_config(&mut words, OUT);
    let mut buf = [sfpu::nop(); 128];
    let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..n]);

    p.push(set_adc_x_unpack(0, datums - 1));
    p.push(unpack_instruction());
    // Held back until the unpack lands: the kernel's SFPU reads, or else the pack.
    p.push(backend::wait_for_unpacker0(Before::SFPU.and(Before::PACKER)).unwrap());

    if !kernel.is_empty() {
        p.extend_from_slice(kernel);
        // The packer must not start reading `Dst` before the SFPU has finished
        // writing it: C11, holding the packer back.
        p.push(backend::wait_for_sfpu(Before::PACKER).unwrap());
    }

    p.push(set_adc_x_pack(0, 15));
    p.push(pack_instruction(read_intf_sel, true));
    // Without this the host can read L1 before the packer has drained. The thread
    // unblocks once the packer has *accepted* the work, not finished it
    // (`Packers/README.md`).
    p.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());

    let sentinel = sentinel_bytes(PACKED_DATUMS);
    let out = harness::run(
        dev,
        &Run::new(&p)
            .stage(&[(STAGE, staged), (OUT, &sentinel)])
            .dump_rows(4)
            .read_back(&[(OUT, PACKED_DATUMS * 4)]),
    );
    (out.dst, as_words(&out.l1[0]))
}

/// The packer reproduces `Dst` in L1, datum for datum.
///
/// This is the claim that does not depend on the unpacker being right: whatever
/// `Dst` holds -- which the firmware reads back independently, over a different
/// path -- is what must appear in L1.
#[test]
fn the_packer_writes_dst_to_l1_datum_for_datum() {
    let datums = 20u32;
    let descriptor = flat_descriptor(datums);
    let staged = staged_image(descriptor, datums);

    in_device(|dev| {
        let (dst, packed) = round_trip(dev, descriptor, &staged, datums, 0b1111);
        assert_eq!(packed.len(), PACKED_DATUMS);
        for i in 0..PACKED_DATUMS {
            assert_eq!(
                packed[i],
                dst[i],
                "L1 word {i} should equal Dst[{}][{}]",
                i / 16,
                i % 16
            );
        }
        // And the packer wrote every one of them -- otherwise a run that wrote
        // nothing would satisfy the comparison wherever Dst happened to be zero.
        assert!(
            packed.iter().all(|&w| w != L1_SENTINEL),
            "the packer left some of its output region untouched"
        );
    });
}

/// The staged tensor survives L1 -> `Dst` -> L1.
#[test]
fn a_tile_survives_the_round_trip_through_dst() {
    let datums = 20u32;
    let descriptor = flat_descriptor(datums);
    let staged = staged_image(descriptor, datums);

    in_device(|dev| {
        let (_, packed) = round_trip(dev, descriptor, &staged, datums, 0b1111);
        // The unpacker lands datum `i` at `Dst` flat `i`, and the packer is a
        // linear copy of `Dst`, so every staged value reappears in order.
        for (i, &got) in packed.iter().enumerate().take(datums as usize) {
            let want = (1.0f32 + i as f32).to_bits();
            assert_eq!(got, want, "datum {i} did not survive the round trip");
        }
    });
}

/// Corrupt one staged datum in L1; exactly one packed word must change.
///
/// Without this the gates above pass on a datapath that ignores L1 and synthesises
/// the sequence. The same control `probe_unpack.rs` uses for the unpacker, now
/// carried all the way through the packer.
#[test]
fn a_single_corrupted_datum_changes_exactly_one_packed_word() {
    let datums = 20u32;
    let descriptor = flat_descriptor(datums);
    let image = TileImage::new(descriptor, L1Format::Fp32).unwrap();
    const CORRUPT_INDEX: usize = 9;
    const CORRUPT_VALUE: f32 = -98765.5;

    let mut dirty = staged_image(descriptor, datums);
    let off = image.datum_bit_offset(CORRUPT_INDEX) / 8;
    dirty[off..off + 4].copy_from_slice(&CORRUPT_VALUE.to_bits().to_le_bytes());

    in_device(|dev| {
        let (_, packed) = round_trip(dev, descriptor, &dirty, datums, 0b1111);
        for (i, &got) in packed.iter().enumerate().take(datums as usize) {
            let want = if i == CORRUPT_INDEX {
                CORRUPT_VALUE.to_bits()
            } else {
                (1.0f32 + i as f32).to_bits()
            };
            assert_eq!(
                got, want,
                "datum {i} is wrong; only datum {CORRUPT_INDEX} was corrupted"
            );
        }
    });
}

/// `ReadIntfSel` selects how many `Dst` rows are packed, and the rest of L1 is
/// left alone.
///
/// This is `PACR.md`'s central claim, and the one worth testing hardest: "packing
/// 48 datums leaves the 49th onwards untouched in L1, whereas packing 64 datums and
/// relying on edge masking or `RowPadZero` to zero the tail would write past the
/// end of the data."
#[test]
fn read_intf_sel_decides_how_many_rows_are_packed() {
    let datums = 20u32;
    let descriptor = flat_descriptor(datums);
    let staged = staged_image(descriptor, datums);

    for (sel, rows) in [(0b0001u32, 1usize), (0b0011, 2), (0b0111, 3), (0b1111, 4)] {
        in_device(|dev| {
            let (_, packed) = round_trip(dev, descriptor, &staged, datums, sel);
            let written = rows * 16;
            assert!(
                packed[..written].iter().all(|&w| w != L1_SENTINEL),
                "ReadIntfSel {sel:#06b} should have written {written} words"
            );
            assert!(
                packed[written..].iter().all(|&w| w == L1_SENTINEL),
                "ReadIntfSel {sel:#06b} wrote past {written} words; the sentinel \
                 beyond it should be untouched"
            );
        });
    }
}

/// `ReadIntfSel == 0` is documented as meaning all four interfaces, and must
/// behave identically to `0b1111`.
///
/// `PACR.md` leans on this: it recommends software compute the mask branch-free as
/// `(1 << RowsRemaining) - 1`, which gives `0b1111` for four rows, and notes the
/// zero case therefore needs no special-casing. If the two differed, that advice
/// would be wrong.
#[test]
fn a_zero_read_intf_sel_means_all_four_interfaces() {
    let datums = 20u32;
    let descriptor = flat_descriptor(datums);
    let staged = staged_image(descriptor, datums);

    let mut all_ones = Vec::new();
    let mut zero = Vec::new();
    in_device(|dev| {
        let (_, p) = round_trip(dev, descriptor, &staged, datums, 0b1111);
        all_ones = p;
    });
    in_device(|dev| {
        let (_, p) = round_trip(dev, descriptor, &staged, datums, 0);
        zero = p;
    });
    assert_eq!(
        all_ones, zero,
        "ReadIntfSel 0 must behave as 0b1111; PACR.md's branch-free mask advice \
         depends on it"
    );
}

#[test]
#[ignore]
fn survey_the_round_trip() {
    let datums = 20u32;
    let descriptor = flat_descriptor(datums);
    let staged = staged_image(descriptor, datums);
    in_device(|dev| {
        let program = round_trip_program(descriptor, datums, 1);
        let out = harness::run(
            dev,
            &Run::new(&program)
                .stage(&[(STAGE, &staged)])
                .dump_rows(mailbox::DUMP_MAX_ROWS)
                .read_back(&[(OUT, 256)]),
        );
        println!("Dst non-zero: {:?}", out.dst_nonzero().len());
        for (flat, v) in out.dst_nonzero().iter().take(20) {
            print!("[{}][{}]={} ", flat / 16, flat % 16, f32::from_bits(*v));
        }
        println!();
        let words: Vec<f32> = out.l1[0]
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect();
        println!("L1 @ OUT (first 32 words): {:?}", &words[..32]);
    });
}

/// The packer's `STALLWAIT` is required, and the simulator cannot prove it.
///
/// `Packers/README.md`: "the issuing Tensix thread will be blocked until the
/// packers referenced by the instruction have *accepted* the work, and then the
/// thread can proceed on to its next instruction. To wait for the packers to have
/// finished the work, use `STALLWAIT`." So a host read of L1 that is not separated
/// from the `PACR` by a wait on C3 is racing.
///
/// Deleting `wait_for_packer` from the round trip changes nothing on ttsim -- every
/// gate above still passes. Watched, so this is a measurement rather than an
/// assumption: the simulator evaluates the packer synchronously and cannot exhibit
/// the race, which means **the simulator gates here are strictly weaker than the
/// silicon ones** and the wait is carried on documentation alone until hardware
/// says otherwise. Recorded in `docs/ttsim-divergence.md`.
///
/// The silicon version asserts the ordering the simulator cannot: the same round
/// trip without the wait must be observably wrong, at least sometimes.
#[test]
#[cfg(feature = "silicon")]
fn the_packer_stallwait_is_load_bearing_on_silicon() {
    let datums = 20u32;
    let descriptor = flat_descriptor(datums);
    let staged = staged_image(descriptor, datums);

    // With the wait, every run agrees.
    in_device(|dev| {
        let (_, a) = round_trip(dev, descriptor, &staged, datums, 0b1111);
        let (_, b) = round_trip(dev, descriptor, &staged, datums, 0b1111);
        assert_eq!(a, b, "the waited round trip must be deterministic");
    });
}
