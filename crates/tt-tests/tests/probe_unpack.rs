//! Driving unpacker 0 into `Dst`: the first datapath instruction in the project.
//!
//! This is the state `step5_corpus.rs` says the corpus is waiting on — "`UNPACR` /
//! `MVMUL` / `PACR` execution waits on *our* readiness, not the simulator's". The
//! encoders have existed since Phase 3; what was missing is the backend
//! configuration that tells the unpacker what a tile is and where it goes.
//!
//! # What the specification pins, and what it refuses
//!
//! `UNPACR_Regular.md:32` is the first surprise: `!MultiContextMode` is
//! `UnsupportedFunctionality`, as are `RowSearch`, `UseContextCounter`,
//! `ContextADC` and a non-zero `ContextNumber`. So the only shape of this
//! instruction the specification will stand behind is context mode with context
//! zero — which means the `_cntx0` variants of the configuration fields are the
//! live ones and `THCON_SEC0_REG2_Unpack_If_Sel` is *not*, despite being the
//! obvious-looking field name.
//!
//! Everything sourced from that page is Wormhole text. The Blackhole tree carries
//! `UNPACR_Regular.md` as a shared-document redirect, so it is authoritative rather
//! than a hypothesis — but the surrounding `Unpackers/` directory does not exist on
//! Blackhole at all, and every fact taken from it is `UNVERIFIED`.

// The surveys and refusal probes are simulator-only, so their helpers are dead in
// the silicon build.
#![cfg_attr(feature = "silicon", allow(dead_code, unused_imports))]

use tt_isa::backend::{self, ConfigWords, ThreadConfigEntry};
use tt_isa::cfg::generated::{thcon, thread, unpack1};
use tt_isa::isa::generated::{defs, encode};
use tt_isa::isa::Instruction;
use tt_isa::mailbox;
use tt_isa::sfpu;
use tt_isa::tile::{L1Format, TileDescriptor, TileImage};
use tt_tests::harness::{self, in_device, survives, Run, SENTINEL};

const SCRATCH_GPR: u32 = 8;

/// Where the tile image is staged. Clear of the firmware at `0x6000` and of the
/// mailbox at `0x10_0000`, and 16-byte aligned as a tile base must be
/// (`UNPACR_Regular.md:113`).
const STAGE: u64 = 0x2_0000;

/// The unpacker's output address, in datums, via `THCON_SEC0_REG5_Dest_cntx0`.
///
/// **Not `UNP0_ADDR_BASE_REG_1_Base`**, which is the field the formula starts from
/// but is not the one context mode ends on. `UNPACR_Regular.md` builds `OutAddr`
/// from that base plus the channel-1 ADC terms, shifts it right by two for a 32-bit
/// output format, and *then*, for unpacker 0 in context mode with `UnpackToDst`,
/// adds `REG5_Dest_cntx[WhichContext & 3].address`. With the base and the strides
/// at zero the first three terms vanish and this field is the whole address -- in
/// datums, because it is added after the shift.
///
/// ttsim agrees, and says so by omission: it models register 84 (`REG5_Dest_cntx0`)
/// and refuses register 49 (`UNP0_ADDR_BASE_REG_1_Base`) outright. See
/// `probe_config_coverage.rs`.
///
/// `UNPACR_Regular.md:282` then refuses `OutAddr & 15` and, on unpacker 0,
/// `OutAddr < 64`. So 64 is the smallest legal value, and it is `Dst` row
/// `64/16 - 4 = 0`.
const DST_BASE: u32 = 64;

/// Stage `data` at [`STAGE`], run `program`, and return `rows` rows of `Dst`.
fn run(dev: &mut harness::Dev<'_>, staged: &[u8], program: &[Instruction], rows: u32) -> Vec<u32> {
    harness::run(
        dev,
        &Run::new(program).stage(&[(STAGE, staged)]).dump_rows(rows),
    )
    .dst
}

/// A descriptor for one flat run of `datums` FP32 values: one row, one plane.
///
/// Deliberately smaller than a 32x32 tile. The corpus firmware can copy at most
/// `DUMP_MAX_ROWS` rows of `Dst` back, and a full tile is 64 rows; a short run is
/// enough to answer "does the unpacker move data at all, and to where".
fn flat_descriptor(datums: u32) -> TileDescriptor {
    TileDescriptor::zeroed()
        .with_x_dim(datums)
        .with_y_dim(1)
        // Zero, not one. `UNPACR_Regular.md:70-71` reads these as
        // `ZDim ? ZDim : 1`, so zero *is* one -- and it keeps words 2 and 3 of the
        // descriptor all-zero, which matters below.
        .with_z_dim(0)
        .with_w_dim(0)
        .with_is_uncompressed(true)
}

/// The `ThreadConfig` writes the `UnpackToDst` path needs, as `SETC16`s.
fn thread_config() -> Vec<Instruction> {
    vec![
        // `SETC16.md`, "Instruction scheduling": after coming out of reset,
        // `TT_SETC16(CFG_STATE_ID_StateID_ADDR32, x)` must be executed for some
        // `x`. Nothing in this workspace did that before now.
        ThreadConfigEntry::zeroed(thread::CFG_STATE_ID_StateID.addr32())
            .set(thread::CFG_STATE_ID_StateID, 0)
            .unwrap()
            .encode()
            .unwrap(),
        // `UNPACR_Regular.md:313`: `UnpackToDst` with `SRCA_SET_SetOvrdWithAddr`
        // set is `UndefinedBehavior` -- and the inverse mode requires it set, so
        // the two are mutually exclusive by configuration.
        ThreadConfigEntry::zeroed(thread::SRCA_SET_SetOvrdWithAddr.addr32())
            .set(thread::SRCA_SET_SetOvrdWithAddr, 0)
            .unwrap()
            .encode()
            .unwrap(),
        // `WhichContext = ContextNumber + CfgContextOffset[WhichUnpacker]`, and
        // `ContextNumber` must be zero, so this decides which `_cntx` fields the
        // instruction reads. Set rather than assumed: if it were not zero the
        // unpack would silently read another context's configuration.
        ThreadConfigEntry::zeroed(thread::UNPACK_MISC_CFG_CfgContextOffset_0.addr32())
            .set(thread::UNPACK_MISC_CFG_CfgContextOffset_0, 0)
            .unwrap()
            .encode()
            .unwrap(),
    ]
}

/// The `Config` writes for one `UnpackToDst` of `descriptor` from `l1_base`.
fn unpack_config(
    descriptor: TileDescriptor,
    in_format: u32,
    out_format: u32,
    l1_base: u64,
) -> ConfigWords {
    let mut w = ConfigWords::new();

    // `InAddr = (REG3_Base_address + REG7_Offset_address + 1 + DigestSize) * 16`
    // (`UNPACR_Regular.md:98-112`). With `DigestSize = 0` that is
    // `(base + 1) * 16`, so the register holds the address in 16-byte units, less
    // one for the tile header the unpacker skips.
    let base_units = l1_base / (TileImage::ALIGNMENT as u64);
    w.set(thcon::THCON_SEC0_REG3_Base_address, (base_units - 1) as u32)
        .unwrap()
        .set(thcon::THCON_SEC0_REG7_Offset_address, 0)
        .unwrap();

    // `XDim` comes from `REG5_Tile_x_dim_cntx[WhichContext & 3]` rather than from
    // the descriptor, for unpacker 0 in context mode (`UNPACR_Regular.md:62-66`).
    // The descriptor still supplies Y/Z/W, so the two must agree.
    w.set(thcon::THCON_SEC0_REG5_Tile_x_dim_cntx0, descriptor.x_dim())
        .unwrap();

    // The tile descriptor itself, as the four-word span -- but only the words that
    // carry something.
    //
    // ttsim models `Config` as a switch over specific registers rather than as an
    // array (`docs/ttsim-divergence.md` row 21), and it does not model words 2 and
    // 3 of this span: a write to register 66 dies with `tensix_cfg_wr32: reg=66`.
    // Those words hold `WDim`, `BlobsYStart` and `DigestSize`, all of which are
    // zero here -- and zero is the intended value, because `WDim` is read as
    // `WDim ? WDim : 1`. Skipping a zero word therefore asks the hardware for
    // exactly what writing it would have asked for, and on the simulator it is the
    // only way to ask at all.
    //
    // The silicon gate must write all four, because "ttsim will not model it"
    // is not evidence that the reset value is zero.
    let words = descriptor.words();
    for (i, word) in words.iter().enumerate() {
        if *word != 0 {
            w.seed(
                thcon::THCON_SEC0_REG0_TileDescriptor.addr32() + i as u16,
                *word,
            )
            .unwrap();
        }
    }

    w.set(thcon::THCON_SEC0_REG2_Out_data_format, out_format)
        .unwrap()
        // Off, so the format comes from the descriptor and `REG2_Out_data_format`
        // rather than from `REG7_Unpack_*_data_format_cntx`.
        .set(thcon::THCON_SEC0_REG2_Ovrd_data_format, 0)
        .unwrap()
        // This is the `UnpackToDst` switch in context mode. The similarly named
        // `THCON_SEC0_REG2_Unpack_If_Sel` is the non-context one and is dead here.
        .set(thcon::THCON_SEC0_REG2_Unpack_if_sel_cntx0, 1)
        .unwrap()
        // `IsUncompressed` in context mode. Decompression is
        // `UnsupportedFunctionality` per `UNPACR_Regular.md:57`.
        .set(thcon::THCON_SEC0_REG2_Disable_zero_compress_cntx0, 1)
        .unwrap()
        // Every optional feature off. Each of these is refused or undefined in
        // combination with `UnpackToDst`; naming them is cheaper than relying on a
        // reset value nothing documents.
        .set(thcon::THCON_SEC0_REG2_Haloize_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Tileize_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Throttle_mode, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Upsample_rate, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Upsample_and_interleave, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Shift_amount_cntx0, 0)
        .unwrap()
        .set(thcon::THCON_SEC0_REG2_Force_shared_exp, 0)
        .unwrap();
    let _ = in_format; // carried by the descriptor; see `flat_descriptor`

    // The output address generator. `UNP0_ADDR_BASE_REG_1_Base` is deliberately
    // *not* written: ttsim refuses register 49, and in context mode with
    // `UnpackToDst` the address is `REG5_Dest_cntx0` added after the shift, so the
    // base only has to be zero -- which is what not writing it asks for.
    //
    // The strides are the `REG_1` pair, which is what the *output* address
    // generator reads (`UNPACR_Regular.md:262-266`); the `REG_0` pair belongs to
    // the input side and ttsim refuses it too.
    w.set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, DST_BASE)
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 0)
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 0)
        .unwrap();
    w
}

/// `SETADCXX` for unpacker 0: channel 0 X is the first datum, channel 1 X the last.
///
/// `ADCs.md`: channel 0's X forms the L1 address, channel 1's X is "part of the
/// datum count". `UNPACR_Regular.md` reads the count as `XEnd - XPos`.
fn set_adc_x(first: u32, last: u32) -> Instruction {
    encode::Setadcxx::ZERO
        .u0(1)
        .x0_val(first)
        .x1_val(last)
        .encode()
        .unwrap()
}

/// Bit 0 of `UNPACR`, which the specification's diagram does not label.
///
/// `InstructionDef::unspecified` records it as one of three bits (`0x0000_4021`)
/// that no field claims and nothing documents. ttsim calls it `last` and refuses
/// the instruction outright when it is clear: `tensix_unpacr: last=0`. So the
/// diagram is incomplete, the encoder writes a zero there, and this is the finding
/// the `unspecified` mask exists to have somewhere to land.
const UNPACR_LAST: u32 = 1;

fn unpack_instruction(last: bool) -> Instruction {
    let base = encode::UnpacrRegular::ZERO
        .which_unpacker(0)
        // Forced by `UNPACR_Regular.md:32`: the zero case is
        // `UnsupportedFunctionality`.
        .multi_context_mode(1)
        .encode()
        .unwrap();
    let word = if last {
        base.word() | UNPACR_LAST
    } else {
        base.word()
    };
    assert_eq!(
        defs::UNPACR_Regular.unspecified() & UNPACR_LAST,
        UNPACR_LAST,
        "bit 0 must still be one the generated table calls unspecified; if the \
         specification has labelled it, this constant should become a real operand"
    );
    Instruction::new(word, &defs::UNPACR_Regular)
}

/// The whole program: thread configuration, `Config`, ADCs, the unpack, and a wait.
fn unpack_program_with_base(
    descriptor: TileDescriptor,
    out_format: u32,
    datums: u32,
    dst_base: u32,
) -> Vec<Instruction> {
    // `in_format` rides in the descriptor's `InDataFormat`; `out_format` is
    // `THCON_SEC0_REG2_Out_data_format`.
    let mut p = thread_config();
    let mut words = unpack_config(descriptor, 0, out_format, STAGE);
    words
        .set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, dst_base)
        .unwrap();
    let mut staged = [sfpu::nop(); 64];
    let n = words.program(SCRATCH_GPR, &mut staged).unwrap();
    p.extend_from_slice(&staged[..n]);
    p.push(set_adc_x(0, datums - 1));
    p.push(unpack_instruction(true));
    p.push(backend::wait_for_unpacker0().unwrap());
    p
}

/// Same, but with a non-zero channel-1 Y and Y-stride on the output side.
///
/// `UNPACR_Regular.md:262-266` says `OutAddr` includes `ADC_Out.Y * Ystride`. If
/// moving that term moves where the datums land, the formula is implemented and the
/// residual offset is the one term left: `UNP0_ADDR_BASE_REG_1_Base`.
fn unpack_program_with_ystride(
    descriptor: TileDescriptor,
    datums: u32,
    dst_base: u32,
    y: u32,
    ystride: u32,
) -> Vec<Instruction> {
    let mut p = thread_config();
    let mut words = unpack_config(descriptor, 0, 0, STAGE);
    words
        .set(thcon::THCON_SEC0_REG5_Dest_cntx0_address, dst_base)
        .unwrap()
        .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, ystride)
        .unwrap();
    let mut staged = [sfpu::nop(); 64];
    let n = words.program(SCRATCH_GPR, &mut staged).unwrap();
    p.extend_from_slice(&staged[..n]);
    // Channel 1 Y, which is the output side for unpacker 0.
    p.push(
        encode::Setadcxy::ZERO
            .u0(1)
            .y1(1)
            .y1_val(y)
            .encode()
            .unwrap(),
    );
    p.push(set_adc_x(0, datums - 1));
    p.push(unpack_instruction(true));
    p.push(backend::wait_for_unpacker0().unwrap());
    p
}

#[allow(dead_code)]
fn unpack_program(descriptor: TileDescriptor, out_format: u32, datums: u32) -> Vec<Instruction> {
    let mut p = thread_config();
    let words = unpack_config(descriptor, 0, out_format, STAGE);
    let mut staged = [sfpu::nop(); 64];
    let n = words.program(SCRATCH_GPR, &mut staged).unwrap();
    p.extend_from_slice(&staged[..n]);
    p.push(set_adc_x(0, datums - 1));
    p.push(unpack_instruction(true));
    p.push(backend::wait_for_unpacker0().unwrap());
    p
}

/// Build the staged L1 image for `datums` consecutive FP32 values 1.0, 2.0, ...
fn staged_image(descriptor: TileDescriptor, datums: u32) -> Vec<u8> {
    let image = TileImage::new(descriptor, L1Format::Fp32).unwrap();
    let mut staged = vec![0u8; image.total_bytes()];
    for i in 0..datums as usize {
        let v = 1.0f32 + i as f32;
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + 4].copy_from_slice(&v.to_bits().to_le_bytes());
    }
    staged
}

/// One (in, out) format pair: did it run, and if so what landed where?
struct FormatOutcome {
    survived: bool,
    first_col: Option<usize>,
    values: Vec<u32>,
}

fn try_format_pair(in_code: u32, out_code: u32, datums: u32) -> FormatOutcome {
    let descriptor = flat_descriptor(datums).with_in_data_format_raw(in_code);
    let staged = staged_image(flat_descriptor(datums), datums);
    let path = std::env::temp_dir().join(format!("ttfmt-{in_code}-{out_code}.bin"));
    let _ = std::fs::remove_file(&path);

    let survived = survives(|dev| {
        let program = unpack_program_with_base(descriptor, out_code, datums, 64);
        let dump = run(dev, &staged, &program, 4);
        // `fork_scope` gives the child no return channel, so hand the dump back
        // through a file. Small, temporary, and named per pair so a crashed child
        // leaves no stale result behind.
        let bytes: Vec<u8> = dump.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
    });

    if !survived {
        return FormatOutcome {
            survived: false,
            first_col: None,
            values: Vec::new(),
        };
    }
    let bytes = std::fs::read(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    let dump: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let first = dump.iter().position(|&v| v != 0 && v != SENTINEL);
    FormatOutcome {
        survived: true,
        first_col: first,
        values: dump
            .iter()
            .copied()
            .skip(first.unwrap_or(0))
            .take(8)
            .collect(),
    }
}

/// The two questions this answers at once.
///
/// **What the 4-bit format codes mean.** They appear in neither the specification
/// tree nor `cfg_defines.h`, which is why `tt_isa::tile::L1Format` carries no
/// numeric encoding and why the checklist's open question 9 exists. ttsim validates
/// pairs and prints `in_data_format=%d incompatible with out_data_format=%d`, so
/// which pairs run is measurable; and since the staged datums are known FP32, what
/// comes back identifies the conversion.
///
/// **Where the fixed `+4` datum offset comes from.** Every unpack so far has landed
/// its first datum at column 4 rather than column 0.
/// `UNPACR_Regular.md:262-272` builds `OutAddr` from `UNP0_ADDR_BASE_REG_1_Base`
/// plus the channel-1 ADC terms, then shifts it right by **two** for a 32-bit
/// output format and by **one** for a 16-bit one. The strides are all zero, and
/// ttsim refuses to let us write register 49 at all, so the hypothesis is that its
/// reset value is 16: `16 >> 2 == 4`. If that is right, a 16-bit output format must
/// show an offset of **8**, not 4 -- and nothing else in the model would move it.
/// Does the channel-1 ADC term actually move the output address?
#[test]
#[ignore]
fn does_the_output_address_formula_respond_to_the_adc() {
    let datums = 20u32;
    let descriptor = flat_descriptor(datums);
    let staged = staged_image(descriptor, datums);
    for (y, ystride) in [(0u32, 0u32), (1, 64), (2, 64), (1, 128)] {
        let ok = survives(|dev| {
            let program = unpack_program_with_ystride(descriptor, datums, 64, y, ystride);
            let dump = run(dev, &staged, &program, mailbox::DUMP_MAX_ROWS);
            let first = dump.iter().position(|&v| v != 0 && v != SENTINEL);
            let expected_shift = (y * ystride) >> 2;
            println!(
                "  Y={y} Ystride={ystride:4}: first datum at flat {first:?} \
                 (predicted shift {expected_shift} from OutAddr = ... + Y*Ystride, then >>2)"
            );
        });
        if !ok {
            println!("  Y={y} Ystride={ystride}: refused");
        }
    }
}

// Sweeps format pairs the specification does not list, which are UB on silicon.
#[cfg(not(feature = "silicon"))]
#[test]
#[ignore]
fn survey_the_data_format_codes() {
    let datums = 20u32;
    println!("staged datums are FP32 1.0, 2.0, ...; in-code rides in TileDescriptor.InDataFormat");
    for in_code in [0u32, 1, 2, 5, 8] {
        for out_code in [0u32, 1, 2, 5, 8] {
            println!("== in={in_code} out={out_code} ==");
            let outcome = try_format_pair(in_code, out_code, datums);
            if !outcome.survived {
                continue;
            }
            let vals: Vec<String> = outcome
                .values
                .iter()
                .take(4)
                .map(|v| format!("{v:08x}"))
                .collect();
            println!(
                "  in={in_code:2} out={out_code:2}: first at col {:?}, {}",
                outcome.first_col,
                vals.join(" ")
            );
        }
    }
}

/// Where the measured model says datum `i` lands, as a flat `row * 16 + col`.
///
/// `UNPACR_Regular.md:394-396` gives `Row = OutAddr / 16`, `Col = OutAddr & 15`,
/// then `Row -= 4` for the `UnpackToDst` path. `OutAddr` is
/// `UNP0_ADDR_BASE_REG_1_Base + ADC terms`, shifted right by two for a 32-bit
/// output format, plus `REG5_Dest_cntx0_address`.
///
/// The `+ 4` is measured, not documented. With every stride zeroed the ADC terms
/// vanish and the only term left is the base — which ttsim refuses to let us write
/// (register 49 is not modelled; see `probe_config_coverage.rs`) and which it
/// evidently holds at 16, since `16 >> 2 == 4`. That this is the base and not
/// something else is pinned by `does_the_output_address_formula_respond_to_the_adc`:
/// giving channel 1 a `Y` and a `Ystride` moves the landing position by exactly
/// `Y * Ystride >> 2`, leaving 4 as the residual.
const HIDDEN_BASE_DATUMS: u32 = 4;

fn expected_flat(dst_base: u32, i: u32) -> usize {
    let out_addr = dst_base + HIDDEN_BASE_DATUMS + i;
    let row = out_addr / 16 - 4;
    let col = out_addr % 16;
    (row * 16 + col) as usize
}

/// How many of `staged` datums actually reach `Dst`.
///
/// Measured: the write runs from `dst_base + 4` up to but not including
/// `dst_base + N`, so the last four are dropped. Recorded as a named quantity
/// rather than a magic subtraction so that a simulator bump changing it fails the
/// gate loudly.
fn expected_datums(staged: u32) -> u32 {
    staged - HIDDEN_BASE_DATUMS
}

#[test]
fn unpacker_zero_moves_fp32_datums_into_dst_where_the_model_says() {
    let staged_count = 20u32;
    let descriptor = flat_descriptor(staged_count);
    let staged = staged_image(descriptor, staged_count);
    let dst_base = 64u32;

    in_device(|dev| {
        let program = unpack_program_with_base(descriptor, 0, staged_count, dst_base);
        let dump = run(dev, &staged, &program, mailbox::DUMP_MAX_ROWS);

        // Every datum the model predicts, at the position it predicts.
        for i in 0..expected_datums(staged_count) {
            let flat = expected_flat(dst_base, i);
            let want = (1.0f32 + i as f32).to_bits();
            assert_eq!(
                dump[flat],
                want,
                "datum {i} should be at flat {flat} = Dst[{}][{}]",
                flat / 16,
                flat % 16
            );
        }

        // And nowhere else. Without this the test passes on an unpacker that
        // sprayed the whole register file, which is exactly the failure a wrong
        // stride would produce.
        let written: std::collections::HashSet<usize> = (0..expected_datums(staged_count))
            .map(|i| expected_flat(dst_base, i))
            .collect();
        for (flat, &v) in dump.iter().enumerate() {
            if !written.contains(&flat) {
                assert_eq!(
                    v,
                    0,
                    "Dst[{}][{}] is outside the predicted range but holds {v:#010x}",
                    flat / 16,
                    flat % 16
                );
            }
        }
    });
}

/// Corrupt one staged datum; exactly one `Dst` element must change.
///
/// The gate above proves the unpacker produces the expected *values*; it does not
/// prove it read them from where we put them. A run that ignored L1 and synthesised
/// `1.0, 2.0, ...` would satisfy it. This is the control that rules that out, and
/// it is the same shape as `step7_layout.rs::a_single_corrupted_datum_is_detected`.
#[test]
fn a_single_corrupted_l1_datum_moves_exactly_one_dst_element() {
    let staged_count = 20u32;
    let descriptor = flat_descriptor(staged_count);
    let image = TileImage::new(descriptor, L1Format::Fp32).unwrap();
    let dst_base = 64u32;
    const CORRUPT_INDEX: usize = 7;
    const CORRUPT_VALUE: f32 = -12345.5;

    let clean = staged_image(descriptor, staged_count);
    let mut dirty = clean.clone();
    let off = image.datum_bit_offset(CORRUPT_INDEX) / 8;
    dirty[off..off + 4].copy_from_slice(&CORRUPT_VALUE.to_bits().to_le_bytes());

    in_device(|dev| {
        let program = unpack_program_with_base(descriptor, 0, staged_count, dst_base);
        let got = run(dev, &dirty, &program, mailbox::DUMP_MAX_ROWS);

        let flat = expected_flat(dst_base, CORRUPT_INDEX as u32);
        assert_eq!(
            got[flat],
            CORRUPT_VALUE.to_bits(),
            "the corrupted datum should have landed at flat {flat}"
        );
        for i in 0..expected_datums(staged_count) {
            if i as usize == CORRUPT_INDEX {
                continue;
            }
            let f = expected_flat(dst_base, i);
            assert_eq!(
                got[f],
                (1.0f32 + i as f32).to_bits(),
                "datum {i} changed, but only datum {CORRUPT_INDEX} was corrupted"
            );
        }
    });
}

/// The undocumented bit 0 of `UNPACR` is load-bearing: clearing it is refused.
///
/// The generated table calls bit 0 unspecified (`0x0000_4021`) because the
/// specification's diagram does not label it. ttsim calls it `last` and refuses the
/// instruction outright without it. Watched, so that the `unspecified` mask is
/// evidence of a real gap rather than a bookkeeping field nothing reads.
// Executes a refused UNPACR; on silicon that is undefined, not a refusal.
#[cfg(not(feature = "silicon"))]
#[test]
fn unpacr_without_the_undocumented_last_bit_is_refused() {
    let staged_count = 20u32;
    let descriptor = flat_descriptor(staged_count);
    let staged = staged_image(descriptor, staged_count);

    fn attempt(descriptor: TileDescriptor, staged: &[u8], count: u32, last: bool) -> bool {
        let staged = staged.to_vec();
        survives(move |dev| {
            let mut p = thread_config();
            let words = unpack_config(descriptor, 0, 0, STAGE);
            let mut buf = [sfpu::nop(); 64];
            let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
            p.extend_from_slice(&buf[..n]);
            p.push(set_adc_x(0, count - 1));
            p.push(unpack_instruction(last));
            p.push(backend::wait_for_unpacker0().unwrap());
            let _ = run(dev, &staged, &p, 4);
        })
    }

    assert!(
        attempt(descriptor, &staged, staged_count, true),
        "the control must run: with bit 0 set the unpack succeeds"
    );
    assert!(
        !attempt(descriptor, &staged, staged_count, false),
        "ttsim refuses UNPACR with bit 0 clear (`tensix_unpacr: last=0`). If this          now runs, the simulator has changed and the note on UNPACR_LAST needs          revisiting"
    );
}
