//! Stage 2 of the silicon campaign: measure the ttsim artefacts the datapath gates
//! correct for, before deciding what the silicon corrections are.
//!
//! Every datapath gate from Phase 4 on carries at least one correction for
//! something ttsim does that the specification says it should not:
//!
//! | Row | ttsim | What the gates do about it |
//! |--:|---|---|
//! | 30, 35 | a hidden 16-byte unpacker output base, scaled by the *input* width | expect datum `i` at `hidden + i`, and the last `hidden` dropped |
//! | 32 | `Sub_l1_tile_header_size` stuck at 0, so the packer skips 16 bytes | set `L1_Dest_addr` one unit low |
//! | 37 | `MOVA2D` single-row form refused | move eight rows |
//! | 38 | `MOVB2D Move4Rows` moves one row, silently | move one row at a time |
//! | 39 | FP16 into `SrcB` reads back unlike FP16 into `SrcA` | not on the route; recorded only |
//! | 31 | BF16 `UnpackToDst` refused | FP32-only eltwise |
//!
//! On silicon each of those corrections is expected to be *wrong*, and a gate
//! that fails because of one says nothing about the feature it is testing. So
//! these probes run first. They assert nothing but that the program ran to
//! `DONE`; each prints `MEASURE <key> = <value>` lines, which `cargo xtask
//! silicon` keeps in the per-test output file. The values then go into the gates'
//! per-target corrections and into `docs/ttsim-divergence.md`.
//!
//! Every program here is one the simulator already runs, changed only in the
//! variable being measured, and every configuration is one the specification
//! defines. Nothing here executes a documented `UndefinedBehavior`.

#![cfg(feature = "silicon")]

use tt_isa::backend::{self, ConfigWords};
use tt_isa::cfg::generated::{alu, thcon, unpack0, unpack1};
use tt_isa::cfg::ConfigField;
use tt_isa::isa::generated::encode;
use tt_isa::sfpu;
use tt_isa::tensix;
use tt_isa::tile::{L1Format, TileImage};
use tt_tests::datapath::{
    flat_descriptor, pack_config, pack_instruction, set_adc_x, set_adc_x_pack, set_adc_x_unpack,
    src_thread_config, thread_config, unpack_config, unpack_instruction, unpack_src_config,
    unpack_src_instruction, Unpacker, OUT, SCRATCH_GPR, STAGE,
};
use tt_tests::harness::{self, assert_on_silicon, in_device, Run, SENTINEL};

/// Datums staged per probe: more than one row, fewer than two, so a shifted
/// landing and a truncated tail are both visible.
const N: usize = 20;
const ROW: usize = 16;

fn measure(key: &str, value: impl std::fmt::Display) {
    println!("MEASURE {key} = {value}");
}

/// Stage `datums` of `format` under the descriptor's input code `code`.
fn stage(format: L1Format, code: u32, datums: &[u32]) -> Vec<u8> {
    let descriptor = flat_descriptor(datums.len() as u32).with_in_data_format_raw(code);
    let image = TileImage::new(descriptor, format).unwrap();
    let bytes = (format.datum_bits() / 8) as usize;
    let mut staged = vec![0u8; image.total_bytes()];
    for (i, d) in datums.iter().enumerate() {
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + bytes].copy_from_slice(&d.to_le_bytes()[..bytes]);
    }
    staged
}

/// Where does the run `expected` start in `dump`, and how much of it is there?
///
/// Returns `(first flat index, consecutive datums matching from there)`, or
/// `None` if datum 0 is nowhere.
fn landing(dump: &[u32], expected: &[u32]) -> Option<(usize, usize)> {
    let first = dump.iter().position(|&v| v == expected[0])?;
    let run = dump[first..]
        .iter()
        .zip(expected)
        .take_while(|(a, b)| a == b)
        .count();
    Some((first, run))
}

fn report_landing(key: &str, dump: &[u32], expected: &[u32]) {
    let rows: Vec<String> = dump
        .chunks(ROW)
        .take(4)
        .map(|r| {
            r.iter()
                .map(|v| format!("{v:08x}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    for (i, r) in rows.iter().enumerate() {
        measure(&format!("{key}.row{i}"), r);
    }
    match landing(dump, expected) {
        Some((first, run)) => {
            measure(&format!("{key}.first_flat"), first);
            measure(
                &format!("{key}.landed"),
                format!("{run} of {}", expected.len()),
            );
        }
        None => {
            measure(&format!("{key}.first_flat"), "absent");
            let head: Vec<String> = dump
                .iter()
                .take(2 * ROW)
                .map(|v| format!("{v:08x}"))
                .collect();
            measure(&format!("{key}.dump_head"), head.join(" "));
        }
    }
}

/// Rows 30 and 31: where FP32 `UnpackToDst` lands its first datum.
///
/// ttsim lands datum 0 at flat 4 and drops the last four. The specification puts
/// it at 0 and lands all of them.
#[test]
fn m01_unpack_to_dst_landing() {
    assert_on_silicon();
    let descriptor = flat_descriptor(N as u32);
    let bits: Vec<u32> = (0..N).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, L1Format::Fp32.code().unwrap(), &bits);
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    let mut buf = [sfpu::nop(); 64];
    let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..n]);
    p.push(set_adc_x_unpack(0, N as u32 - 1));
    p.push(unpack_instruction());
    p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
    in_device(|dev| {
        let out = harness::run(dev, &Run::new(&p).stage(&[(STAGE, &staged)]).dump_rows(4));
        report_landing("dst.fp32", &out.dst, &bits);
    });
}

/// Rows 35 and 37-39: the `Src` path, observed through `MOVA2D`/`MOVB2D`.
fn src_probe(
    key: &str,
    unpacker: Unpacker,
    (format, in_code): (L1Format, u32),
    out_code: u32,
    datums: &[u32],
    expected: impl Fn(u32) -> u32,
    single_row_mova2d: bool,
) {
    let staged = stage(format, in_code, datums);
    let mut p = src_thread_config();
    let mut words = ConfigWords::new();
    let descriptor = flat_descriptor(datums.len() as u32).with_in_data_format_raw(in_code);
    unpack_src_config(&mut words, unpacker, descriptor, STAGE, out_code);
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    let mut buf = [sfpu::nop(); 64];
    let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..k]);
    p.push(set_adc_x(unpacker, 0, datums.len() as u32 - 1));
    p.push(unpack_src_instruction(unpacker, true));
    match unpacker {
        Unpacker::SrcA => {
            p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
            if single_row_mova2d {
                for r in 0..2 {
                    p.push(encode::Mova2D::ZERO.src_row(r).dst_row(r).encode().unwrap());
                }
            } else {
                p.push(
                    encode::Mova2D::ZERO
                        .move8_rows(1)
                        .src_row(0)
                        .dst_row(0)
                        .encode()
                        .unwrap(),
                );
            }
        }
        Unpacker::SrcB => {
            p.push(backend::wait_for_unpacker1(backend::Before::EVERYTHING).unwrap());
            for r in 0..8 {
                p.push(encode::Movb2D::ZERO.src_row(r).dst_row(r).encode().unwrap());
            }
        }
    }
    p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
    let want: Vec<u32> = datums.iter().map(|&d| expected(d)).collect();
    in_device(|dev| {
        let out = harness::run(dev, &Run::new(&p).stage(&[(STAGE, &staged)]).dump_rows(8));
        report_landing(key, &out.dst, &want);
    });
}

fn fp32_operands() -> Vec<u32> {
    (0..N)
        .map(|i| (1.0f32 + i as f32).to_bits() | 0x3FFF)
        .collect()
}

fn bf16_operands() -> Vec<u32> {
    (0..N)
        .map(|i| u32::from(tt_isa::tile::fp32_to_bf16_truncate((1.0f32 + i as f32).to_bits()) | 1))
        .collect()
}

#[test]
fn m02_src_landing_fp32_to_tf32() {
    assert_on_silicon();
    for (key, u) in [("srca", Unpacker::SrcA), ("srcb", Unpacker::SrcB)] {
        src_probe(
            &format!("{key}.fp32_tf32"),
            u,
            (L1Format::Fp32, 0),
            4,
            &fp32_operands(),
            tt_isa::tile::fp32_to_tf32,
            false,
        );
    }
}

/// Row 35(b): does the base scale with the input width or the output width?
/// ttsim: FP32 -> BF16 lands four in (input width), BF16 -> BF16 eight in.
#[test]
fn m03_src_landing_bf16() {
    assert_on_silicon();
    let to_bf16 = |d: u32| tt_isa::tile::bf16_to_fp32(tt_isa::tile::fp32_to_bf16_truncate(d));
    let bf16_as_fp32 = |d: u32| tt_isa::tile::bf16_to_fp32(d as u16);
    for (key, u) in [("srca", Unpacker::SrcA), ("srcb", Unpacker::SrcB)] {
        src_probe(
            &format!("{key}.fp32_bf16"),
            u,
            (L1Format::Fp32, 0),
            5,
            &fp32_operands(),
            to_bf16,
            false,
        );
        src_probe(
            &format!("{key}.bf16_bf16"),
            u,
            (L1Format::Bf16, 5),
            5,
            &bf16_operands(),
            bf16_as_fp32,
            false,
        );
    }
}

/// Row 37: `MOVA2D`'s single-row form, which ttsim refuses.
#[test]
fn m04_mova2d_single_row() {
    assert_on_silicon();
    src_probe(
        "srca.mova2d_single_row",
        Unpacker::SrcA,
        (L1Format::Fp32, 0),
        4,
        &fp32_operands(),
        tt_isa::tile::fp32_to_tf32,
        true,
    );
}

/// Row 39: FP16 `0x3C01` into each `Src`, moved to `Dst` under `Fp32_enabled`.
/// ttsim: `0x0780_2000` from `SrcA`, `0x1780_0000` from `SrcB`.
#[test]
fn m05_fp16_into_srca_and_srcb() {
    assert_on_silicon();
    let halves = vec![0x3C01u32; N];
    for (key, u) in [("srca", Unpacker::SrcA), ("srcb", Unpacker::SrcB)] {
        let staged = stage(L1Format::Fp16, 1, &halves);
        let mut p = src_thread_config();
        let mut words = ConfigWords::new();
        let descriptor = flat_descriptor(N as u32).with_in_data_format_raw(1);
        unpack_src_config(&mut words, u, descriptor, STAGE, 1);
        words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
        let mut buf = [sfpu::nop(); 64];
        let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
        p.extend_from_slice(&buf[..k]);
        p.push(set_adc_x(u, 0, N as u32 - 1));
        p.push(unpack_src_instruction(u, true));
        match u {
            Unpacker::SrcA => {
                p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
                p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
            }
            Unpacker::SrcB => {
                p.push(backend::wait_for_unpacker1(backend::Before::EVERYTHING).unwrap());
                p.push(encode::Movb2D::ZERO.encode().unwrap());
                p.push(encode::Movb2D::ZERO.src_row(1).dst_row(1).encode().unwrap());
            }
        }
        p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
        in_device(|dev| {
            let out = harness::run(dev, &Run::new(&p).stage(&[(STAGE, &staged)]).dump_rows(2));
            let distinct: std::collections::BTreeSet<u32> =
                out.dst.iter().copied().filter(|&v| v != SENTINEL).collect();
            let shown: Vec<String> = distinct.iter().map(|v| format!("{v:08x}")).collect();
            measure(&format!("{key}.fp16_3c01.dst_values"), shown.join(" "));
        });
    }
}

/// Row 32: where the packer's output lands relative to `OUT`.
///
/// `pack_config` sets `L1_Dest_addr` one 16-byte unit *low* to cancel ttsim's
/// stuck `Sub_l1_tile_header_size`. If silicon has no such skip, the output lands
/// 16 bytes before `OUT`. Measured by locating the `Dst` dump the firmware read
/// back independently inside a wide L1 window around `OUT`.
#[test]
fn m06_packer_output_offset() {
    assert_on_silicon();
    const MARGIN: usize = 64;
    const PACKED: usize = 64;
    let descriptor = flat_descriptor(N as u32);
    let bits: Vec<u32> = (0..N).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, 0, &bits);
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    pack_config(&mut words, OUT);
    let mut buf = [sfpu::nop(); 128];
    let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..n]);
    p.push(set_adc_x_unpack(0, N as u32 - 1));
    p.push(unpack_instruction());
    p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
    p.push(set_adc_x_pack(0, 15));
    p.push(pack_instruction(0b1111, true));
    p.push(backend::wait_for_packer(backend::Before::EVERYTHING).unwrap());

    let window = OUT - MARGIN as u64;
    let span = 2 * MARGIN + PACKED * 4;
    let sentinel: Vec<u8> = 0xA5A5_5A5Au32
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(span)
        .collect();
    in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::new(&p)
                .stage(&[(STAGE, &staged), (window, &sentinel)])
                .dump_rows(4)
                .read_back(&[(window, span)]),
        );
        let l1: Vec<u32> = out.l1[0]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Find the first staged datum in both, then compare offsets.
        let in_dst = out.dst.iter().position(|&v| v == bits[0]);
        let in_l1 = l1.iter().position(|&v| v == bits[0]);
        match (in_dst, in_l1) {
            (Some(d), Some(l)) => {
                // `l` words into the window; the packer's word 0 is `d` words
                // before datum 0 in `Dst`, so its L1 address is window + 4(l - d).
                let byte = (l as i64 - d as i64) * 4 - MARGIN as i64;
                measure("pack.word0_offset_from_OUT_bytes", byte);
            }
            _ => {
                measure("pack.word0_offset_from_OUT_bytes", "not located");
                let shown: Vec<String> = l1.iter().map(|v| format!("{v:08x}")).collect();
                measure("pack.l1_window", shown.join(" "));
            }
        }
    });
}

/// Row 30: the unpackers' `ADDR_BASE_REG_1_Base` as the chip holds it.
///
/// ttsim refuses to let either be written and holds both at 16. Read back
/// through the `CFGREG` debug pair (row E) after an ordinary run, without writing
/// them: this is what the gates have been running against.
#[test]
fn m07_unpacker_address_base_registers() {
    assert_on_silicon();
    fn read(dev: &mut harness::Dev<'_>, field: ConfigField) -> u32 {
        let tile = harness::tensix_tile();
        let w = dev
            .alloc_window(tt_device::tlb::WindowKind::TwoMib)
            .unwrap();
        dev.write32(&w, tile, tensix::CFGREG_RD_CNTL, field.addr32() as u32)
            .unwrap();
        harness::advance(dev, 64);
        let word = dev.read32(&w, tile, tensix::CFGREG_RDDATA).unwrap();
        dev.free_window(w);
        field.extract(word)
    }
    let p = vec![backend::nop()];
    in_device(|dev| {
        let _ = harness::run(dev, &Run::new(&p));
        measure(
            "cfg.UNP0_ADDR_BASE_REG_1_Base",
            read(dev, unpack0::UNP0_ADDR_BASE_REG_1_Base),
        );
        measure(
            "cfg.UNP1_ADDR_BASE_REG_1_Base",
            read(dev, unpack1::UNP1_ADDR_BASE_REG_1_Base),
        );
        measure(
            "cfg.THCON_SEC0_REG1_Sub_l1_tile_header_size",
            read(dev, thcon::THCON_SEC0_REG1_Sub_l1_tile_header_size),
        );
    });
}

/// Row 31: does a 16-bit input reach `Dst` on silicon?
///
/// BF16 in, BF16 out, `UnpackToDst`: `FormatConversion` defines it
/// (`DstEncodeBF16`), and ttsim refuses it. Dumped twice, once per `Dst` access
/// format the RISC-V view offers for it, because which one reads it back is
/// itself part of the answer.
#[test]
fn m08_bf16_unpack_to_dst() {
    assert_on_silicon();
    let halves = bf16_operands();
    let staged = stage(L1Format::Bf16, 5, &halves);
    let descriptor = flat_descriptor(N as u32).with_in_data_format_raw(5);
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    words
        .set(thcon::THCON_SEC0_REG2_Out_data_format, 5)
        .unwrap();
    let mut buf = [sfpu::nop(); 64];
    let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..n]);
    p.push(set_adc_x_unpack(0, N as u32 - 1));
    p.push(unpack_instruction());
    p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
    let want: Vec<u32> = halves
        .iter()
        .map(|&h| tt_isa::tile::bf16_to_fp32(h as u16))
        .collect();
    for (label, fmt) in [("fmt0", 0u32), ("fmt3", 3)] {
        in_device(|dev| {
            let stage = [(STAGE, staged.as_slice())];
            let mut run = Run::new(&p).stage(&stage).dump_rows(4);
            run.dst_fmt = fmt;
            let out = harness::run(dev, &run);
            report_landing(&format!("dst.bf16.{label}"), &out.dst, &want);
            let head: Vec<String> = out
                .dst
                .iter()
                .take(ROW)
                .map(|v| format!("{v:08x}"))
                .collect();
            measure(&format!("dst.bf16.{label}.row0"), head.join(" "));
        });
    }
}

/// Every non-zero `Config` field, as the chip holds it after a trivial run.
///
/// ttsim starts every run with all of `Config` zero, and the datapath programs
/// write only the fields whose zero is wrong for them -- so on the simulator a
/// field they never write *is* zero. On silicon it is whatever the last program,
/// or the boot firmware, left, and the first Stage 2 run unpacked nothing
/// visible. The backend pulse in the harness zeroes only the THCON block
/// (`SoftReset.md`); this reads all 224 words through the `CFGREG` debug pair
/// (row E, confirmed on silicon) to see what the rest holds.
#[test]
fn m09_non_zero_config_fields_on_silicon() {
    assert_on_silicon();
    let p = vec![backend::nop()];
    in_device(|dev| {
        let _ = harness::run(dev, &Run::new(&p));
        let tile = harness::tensix_tile();
        let w = dev
            .alloc_window(tt_device::tlb::WindowKind::TwoMib)
            .unwrap();
        let mut words = Vec::new();
        for addr32 in 0..tt_isa::cfg::CONFIG_WORDS_PER_BANK {
            dev.write32(&w, tile, tensix::CFGREG_RD_CNTL, addr32)
                .unwrap();
            harness::advance(dev, 64);
            words.push(dev.read32(&w, tile, tensix::CFGREG_RDDATA).unwrap());
        }
        dev.free_window(w);
        let nonzero: Vec<String> = words
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(i, v)| format!("{i}:{v:#x}"))
            .collect();
        measure("cfg.nonzero_words", nonzero.join(" "));
        for (name, field) in tt_isa::cfg::generated::ALL_CONFIG_FIELDS {
            let v = field.extract(words[field.addr32() as usize]);
            if v != 0 {
                measure(&format!("cfg.field.{name}"), format!("{v:#x}"));
            }
        }
    });
}

/// Which `Dst` positions does an FP32 `UnpackToDst` write, on silicon?
///
/// m01 found none of the staged datums in the first four rows. This pre-fills
/// rows 0..16 -- both column parities -- with a marker through the SFPU, whose
/// `Dst` writes read back correctly on silicon (`step5_corpus`), then unpacks
/// and reports every position that no longer holds the marker. Run with
/// `ALU_ACC_CTRL_Fp32_enabled` both ways, since the unpack path never set it and
/// m09 found it left at 1 by an earlier program.
#[test]
fn m10_where_unpack_to_dst_writes() {
    assert_on_silicon();
    const MARK: u32 = 0xC0E0_0000; // -7.0
    let descriptor = flat_descriptor(N as u32);
    let bits: Vec<u32> = (0..N).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, L1Format::Fp32.code().unwrap(), &bits);
    for fp32_enabled in [1u32, 0] {
        let mut p = Vec::new();
        p.extend(sfpu::load_f32(0, MARK).unwrap());
        for group in 0..4u32 {
            for cols in [0, sfpu::DST_ODD_COLUMNS] {
                p.push(sfpu::store(0, sfpu::store_format::FP32, 0, (group * 4) | cols).unwrap());
            }
        }
        p.push(backend::wait_for_sfpu(backend::Before::EVERYTHING).unwrap());
        p.extend(thread_config());
        let mut words = ConfigWords::new();
        unpack_config(&mut words, descriptor, STAGE);
        words
            .set(alu::ALU_ACC_CTRL_Fp32_enabled, fp32_enabled)
            .unwrap();
        let mut buf = [sfpu::nop(); 64];
        let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
        p.extend_from_slice(&buf[..n]);
        p.push(set_adc_x_unpack(0, N as u32 - 1));
        p.push(unpack_instruction());
        p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
        in_device(|dev| {
            let out = harness::run(
                dev,
                &Run::new(&p)
                    .stage(&[(STAGE, &staged)])
                    .dump_rows(tt_isa::mailbox::DUMP_MAX_ROWS),
            );
            let marked = out.dst.iter().filter(|&&v| v == MARK).count();
            let changed: Vec<String> = out
                .dst
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != MARK)
                .map(|(i, v)| format!("[{}][{}]={v:08x}", i / ROW, i % ROW))
                .collect();
            let key = format!("dst_writes.fp32_enabled_{fp32_enabled}");
            measure(
                &format!("{key}.still_marked"),
                format!("{marked} of {}", out.dst.len()),
            );
            measure(&format!("{key}.changed"), changed.join(" "));
        });
    }
}

/// Rows 30 and 35, re-examined: is the four-datum shift an *output* base or an
/// *input* that starts 16 bytes early?
///
/// m10 on silicon: 20 positions written from `[0][0]`, the first four zero, then
/// the staged 1.0.. -- and m09: `UNP0_ADDR_BASE_REG_1_Base` reads 0 on silicon,
/// not ttsim's 16, yet the landing is identical. A zeroed output prefix and a
/// read of the 16 unstaged bytes before `STAGE` look the same. Staging four
/// markers there tells them apart.
#[test]
fn m11_what_the_first_four_positions_hold() {
    assert_on_silicon();
    let descriptor = flat_descriptor(N as u32);
    let bits: Vec<u32> = (0..N).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, L1Format::Fp32.code().unwrap(), &bits);
    let prefix: Vec<u8> = [100.0f32, 101.0, 102.0, 103.0]
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    let mut buf = [sfpu::nop(); 64];
    let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..n]);
    p.push(set_adc_x_unpack(0, N as u32 - 1));
    p.push(unpack_instruction());
    p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
    in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::new(&p)
                .stage(&[(STAGE - 16, &prefix), (STAGE, &staged)])
                .dump_rows(2),
        );
        let head: Vec<String> = out.dst[..24].iter().map(|v| format!("{v:08x}")).collect();
        measure("dst.prefix_probe.first_24", head.join(" "));
    });
}

/// Why does FP32 -> TF32 into `SrcA` corrupt scattered datums on silicon, when
/// `SrcB` is clean and ttsim is clean?
///
/// m02: a few positions per run come back with mantissa bits cleared (6.0 as
/// 4.0), and *which* positions changes run to run -- a race, which ttsim cannot
/// exhibit. `UNPACR_Regular.md:325-346` models "SrcA burst drop cases" for
/// exactly this shape -- unpacker 0, into `SrcA`, more than 16 datums, with
/// Blackhole's default x4 throttle -- and says the model is "mildly simplified".
/// Three variants, three runs each: 20 datums at the default throttle; 16
/// datums, which the check exempts; 20 datums with the throttle forced to x1,
/// which makes a burst one row.
#[test]
fn m12_srca_corruption_versus_burst_size() {
    assert_on_silicon();
    let run_case = |n: usize, force_x1: bool| -> Vec<usize> {
        let datums: Vec<u32> = (0..n)
            .map(|i| (1.0f32 + i as f32).to_bits() | 0x3FFF)
            .collect();
        let staged = stage(L1Format::Fp32, 0, &datums);
        let mut p = src_thread_config();
        let mut words = ConfigWords::new();
        let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
        unpack_src_config(&mut words, Unpacker::SrcA, descriptor, STAGE, 4);
        words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
        if force_x1 {
            words
                .set(thcon::THCON_SEC0_REG1_ovrd_default_throttle_mode, 1)
                .unwrap()
                .set(thcon::THCON_SEC0_REG2_Throttle_mode, 0)
                .unwrap();
        }
        let mut buf = [sfpu::nop(); 64];
        let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
        p.extend_from_slice(&buf[..k]);
        p.push(set_adc_x(Unpacker::SrcA, 0, n as u32 - 1));
        p.push(unpack_src_instruction(Unpacker::SrcA, true));
        p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
        p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
        p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
        let path = std::env::temp_dir().join(format!("m12-{}-{n}-{force_x1}", std::process::id()));
        in_device(|dev| {
            let out = harness::run(dev, &Run::new(&p).stage(&[(STAGE, &staged)]).dump_rows(2));
            let bad: Vec<String> = (0..n)
                .filter(|&i| out.dst[i] != tt_isa::tile::fp32_to_tf32(datums[i]))
                .map(|i| i.to_string())
                .collect();
            std::fs::write(&path, bad.join(",")).unwrap();
        });
        let s = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        s.split(',')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().unwrap())
            .collect()
    };
    for (n, force_x1) in [(20usize, false), (16, false), (20, true)] {
        for attempt in 0..3 {
            let bad = run_case(n, force_x1);
            measure(
                &format!("srca_race.n{n}.x1_{force_x1}.run{attempt}"),
                format!("{} bad of {}: {bad:?}", bad.len(), n),
            );
        }
    }
}

/// m12 follow-up: race, or cells?
///
/// Same 20-datum FP32 -> TF32 `SrcA` path. Variants: as m12; with 64 `NOP`s
/// between the unpacker wait and `MOVA2D`; and with plain operands (no low
/// mantissa bits), to see whether the loss depends on the data. Run on both
/// cards to see whether the positions are the chip's or the design's.
#[test]
fn m13_srca_corruption_race_or_cells() {
    assert_on_silicon();
    let run_case = |plain: bool, pad: usize| -> Vec<(usize, u32, u32)> {
        let n = 20usize;
        let datums: Vec<u32> = (0..n)
            .map(|i| (1.0f32 + i as f32).to_bits() | if plain { 0 } else { 0x3FFF })
            .collect();
        let staged = stage(L1Format::Fp32, 0, &datums);
        let mut p = src_thread_config();
        let mut words = ConfigWords::new();
        let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
        unpack_src_config(&mut words, Unpacker::SrcA, descriptor, STAGE, 4);
        words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
        let mut buf = [sfpu::nop(); 64];
        let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
        p.extend_from_slice(&buf[..k]);
        p.push(set_adc_x(Unpacker::SrcA, 0, n as u32 - 1));
        p.push(unpack_src_instruction(Unpacker::SrcA, true));
        p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
        p.extend(std::iter::repeat_n(backend::nop(), pad));
        p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
        p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
        let path = std::env::temp_dir().join(format!("m13-{}-{plain}-{pad}", std::process::id()));
        in_device(|dev| {
            let out = harness::run(dev, &Run::new(&p).stage(&[(STAGE, &staged)]).dump_rows(2));
            let bad: Vec<String> = (0..n)
                .filter_map(|i| {
                    let want = tt_isa::tile::fp32_to_tf32(datums[i]);
                    let got = out.dst[i];
                    (got != want).then(|| format!("{i}:{want:08x}:{got:08x}"))
                })
                .collect();
            std::fs::write(&path, bad.join(",")).unwrap();
        });
        let s = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        s.split(',')
            .filter(|t| !t.is_empty())
            .map(|t| {
                let f: Vec<&str> = t.split(':').collect();
                (
                    f[0].parse().unwrap(),
                    u32::from_str_radix(f[1], 16).unwrap(),
                    u32::from_str_radix(f[2], 16).unwrap(),
                )
            })
            .collect()
    };
    for (plain, pad) in [(false, 0usize), (false, 64), (true, 0)] {
        for attempt in 0..3 {
            let bad = run_case(plain, pad);
            let shown: Vec<String> = bad
                .iter()
                .map(|(i, w, g)| format!("d{i} want {w:08x} got {g:08x}"))
                .collect();
            measure(
                &format!("srca_race2.plain_{plain}.pad{pad}.run{attempt}"),
                format!("{} bad: {}", bad.len(), shown.join("; ")),
            );
        }
    }
}

/// m13 follow-up: are the bad `SrcA` positions *unwritten*?
///
/// Deterministic per card, different between cards, unaffected by padding, and
/// the wrong values look like `Dst` leftovers rather than damaged datums. So:
/// mark `Dst` rows 0..8 through the SFPU first (as m10), then unpack FP32 -> TF32
/// into `SrcA` and `MOVA2D` it out, and list every position that still holds the
/// marker -- once for `SrcA`, once for the same path through `SrcB`/`MOVB2D`.
#[test]
fn m14_which_positions_the_src_path_leaves_unwritten() {
    assert_on_silicon();
    const MARK: u32 = 0xC0E0_0000;
    let n = 20usize;
    let datums: Vec<u32> = (0..n).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, 0, &datums);
    for unpacker in [Unpacker::SrcA, Unpacker::SrcB] {
        let mut p = Vec::new();
        p.extend(sfpu::load_f32(0, MARK).unwrap());
        for group in 0..2u32 {
            for cols in [0, sfpu::DST_ODD_COLUMNS] {
                p.push(sfpu::store(0, sfpu::store_format::FP32, 0, (group * 4) | cols).unwrap());
            }
        }
        p.push(backend::wait_for_sfpu(backend::Before::EVERYTHING).unwrap());
        p.extend(src_thread_config());
        let mut words = ConfigWords::new();
        let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
        unpack_src_config(&mut words, unpacker, descriptor, STAGE, 4);
        words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
        let mut buf = [sfpu::nop(); 64];
        let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
        p.extend_from_slice(&buf[..k]);
        p.push(set_adc_x(unpacker, 0, n as u32 - 1));
        p.push(unpack_src_instruction(unpacker, true));
        match unpacker {
            Unpacker::SrcA => {
                p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
                p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
            }
            Unpacker::SrcB => {
                p.push(backend::wait_for_unpacker1(backend::Before::EVERYTHING).unwrap());
                for r in 0..2 {
                    p.push(encode::Movb2D::ZERO.src_row(r).dst_row(r).encode().unwrap());
                }
            }
        }
        p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
        in_device(|dev| {
            let out = harness::run(dev, &Run::new(&p).stage(&[(STAGE, &staged)]).dump_rows(2));
            let marked: Vec<usize> = (0..2 * ROW).filter(|&f| out.dst[f] == MARK).collect();
            let row0: Vec<String> = out.dst[..ROW].iter().map(|v| format!("{v:08x}")).collect();
            let key = format!("{unpacker:?}").to_lowercase();
            measure(
                &format!("unwritten.{key}.still_marked_flat"),
                format!("{marked:?}"),
            );
            measure(&format!("unwritten.{key}.row0"), row0.join(" "));
        });
    }
}

/// m14 follow-up: is it the RISC-V `Dst` read racing the Matrix Unit's write?
///
/// `Dst.md:101`: after an instruction writes `Dst`, its 8x16 block cannot be
/// read for four cycles, and hardware stalls only Matrix Unit and `PACR`
/// readers. The firmware reads `Dst` from RISC-V as soon as the coprocessor
/// reports idle, which nothing makes wait for that write-back. Padding before
/// `MOVA2D` changed nothing (m13); this pads *after* it.
#[test]
fn m15_dst_read_after_matrix_write() {
    assert_on_silicon();
    let n = 20usize;
    let datums: Vec<u32> = (0..n).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, 0, &datums);
    for (unpacker, pad) in [
        (Unpacker::SrcA, 0usize),
        (Unpacker::SrcA, 16),
        (Unpacker::SrcA, 64),
        (Unpacker::SrcB, 0),
        (Unpacker::SrcB, 64),
    ] {
        for attempt in 0..3 {
            let mut p = src_thread_config();
            let mut words = ConfigWords::new();
            let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
            unpack_src_config(&mut words, unpacker, descriptor, STAGE, 4);
            words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
            let mut buf = [sfpu::nop(); 64];
            let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
            p.extend_from_slice(&buf[..k]);
            p.push(set_adc_x(unpacker, 0, n as u32 - 1));
            p.push(unpack_src_instruction(unpacker, true));
            match unpacker {
                Unpacker::SrcA => {
                    p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
                    p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
                }
                Unpacker::SrcB => {
                    p.push(backend::wait_for_unpacker1(backend::Before::EVERYTHING).unwrap());
                    for r in 0..2 {
                        p.push(encode::Movb2D::ZERO.src_row(r).dst_row(r).encode().unwrap());
                    }
                }
            }
            p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
            p.extend(std::iter::repeat_n(backend::nop(), pad));
            in_device(|dev| {
                let out = harness::run(dev, &Run::new(&p).stage(&[(STAGE, &staged)]).dump_rows(2));
                let bad: Vec<usize> = (0..n).filter(|&i| out.dst[i] != datums[i]).collect();
                let key = format!("{unpacker:?}").to_lowercase();
                measure(
                    &format!("raw.{key}.pad{pad}.run{attempt}"),
                    format!("bad {bad:?}"),
                );
            });
        }
    }
}

/// With the unpacker properly waited for, `MOVA2D` reads zeros.
///
/// m15 with a full barrier after the `SrcA` unpack moved nothing but zeros,
/// where a B3-only wait (the old `wait_for_unpacker0`) moved nearly everything.
/// So the old near-miss was `MOVA2D` racing the unpacker, and the real question
/// is which bank it reads once it does not. Three waits: B3 only, B3 + B6, and
/// a full barrier; row 0 of `Dst` printed for each.
#[test]
fn m16_srca_move_versus_wait_mask() {
    assert_on_silicon();
    let n = 20usize;
    let datums: Vec<u32> = (0..n).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, 0, &datums);
    for (label, mask, clear) in [
        ("b3", backend::block::UNPACKER, true),
        (
            "b3_b6",
            backend::block::UNPACKER | backend::block::MATRIX,
            true,
        ),
        ("all", backend::Before::EVERYTHING.mask(), true),
        ("b3.no_prelude", backend::block::UNPACKER, false),
        ("all.no_prelude", backend::Before::EVERYTHING.mask(), false),
    ] {
        let mut p = src_thread_config();
        let mut words = ConfigWords::new();
        let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
        unpack_src_config(&mut words, Unpacker::SrcA, descriptor, STAGE, 4);
        words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
        let mut buf = [sfpu::nop(); 64];
        let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
        p.extend_from_slice(&buf[..k]);
        p.push(set_adc_x(Unpacker::SrcA, 0, n as u32 - 1));
        p.push(unpack_src_instruction(Unpacker::SrcA, true));
        p.push(backend::stallwait(mask, backend::cond::UNPACKER0_BUSY).unwrap());
        p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
        p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
        in_device(|dev| {
            let stage = [(STAGE, staged.as_slice())];
            let mut run = Run::new(&p)
                .stage(&stage)
                .dump_rows(tt_isa::mailbox::DUMP_MAX_ROWS);
            run.clear_dst = clear;
            let out = harness::run(dev, &run);
            let nonzero: Vec<String> = out
                .dst_nonzero()
                .iter()
                .map(|(f, v)| format!("[{}][{}]={v:08x}", f / ROW, f % ROW))
                .collect();
            measure(&format!("srca_wait.{label}.nonzero"), nonzero.join(" "));
        });
    }
}

/// Is the `SrcA` truncation a property of the gate tile?
///
/// With the thread state reset and a full barrier, card 0's gate tile still
/// returns columns 9 and 13 of an FP32 -> TF32 `SrcA` unpack with their top
/// mantissa bits cleared, every run, while `SrcB` is clean; card 1 shows other
/// columns. If the columns follow the tile, it is the tile's storage. Same
/// program on several tiles of this chip.
#[test]
fn m17_srca_truncation_across_tiles() {
    assert_on_silicon();
    let n = 32usize;
    let datums: Vec<u32> = (0..n).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, 0, &datums);
    let reset = tt_tests::datapath::thread_state_reset();
    let mut p = src_thread_config();
    let mut words = ConfigWords::new();
    let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
    unpack_src_config(&mut words, Unpacker::SrcA, descriptor, STAGE, 4);
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    let mut buf = [sfpu::nop(); 64];
    let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..k]);
    p.push(set_adc_x(Unpacker::SrcA, 0, n as u32 - 1));
    p.push(unpack_src_instruction(Unpacker::SrcA, true));
    p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
    p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
    p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
    for (x, y) in [(3u8, 4u8), (1, 2), (5, 6), (7, 9), (11, 3), (2, 10)] {
        in_device(|dev| {
            let mut r = Run::new(&reset).dump_rows(0);
            r.tile = Some((x, y));
            let _ = harness::run(dev, &r);
            let stage = [(STAGE, staged.as_slice())];
            let mut r = Run::new(&p).stage(&stage).dump_rows(2);
            r.tile = Some((x, y));
            let out = harness::run(dev, &r);
            let bad: Vec<String> = (0..n)
                .filter(|&i| out.dst[i] != datums[i])
                .map(|i| format!("c{}:{:08x}", i % ROW, out.dst[i]))
                .collect();
            measure(
                &format!("srca_tiles.({x},{y})"),
                format!("{} bad: {}", bad.len(), bad.join(" ")),
            );
        });
    }
}

/// Unpacker's `SrcA` write, or the Matrix Unit's `SrcA` read?
///
/// m17: FP32 -> TF32 into `SrcA` loses fixed columns on every tile of both
/// cards, so it is not a defect but something about how the path is driven.
/// This fills `SrcA` without the unpacker: `UnpackToDst` (clean on silicon)
/// puts the datums in `Dst` rows 0..2, `MOVD2A` copies them into `SrcA` rows
/// 0..2, and `MOVA2D` brings them back to `Dst` rows 8..10. A bank is first
/// handed to the Matrix Unit by an ordinary flip-unpack into `SrcA`, whose own
/// contents `MOVD2A` then overwrites. Clean rows 8..10 put the fault in the
/// unpacker's `SrcA` write; bad ones put it on the Matrix side.
#[test]
fn m18_srca_filled_by_movd2a() {
    assert_on_silicon();
    let n = 32usize;
    let datums: Vec<u32> = (0..n).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, 0, &datums);
    let barrier = |p: &mut Vec<tt_isa::isa::Instruction>| {
        p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
        p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
    };

    // 1. Datums into Dst rows 0..2, through the clean path.
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, flat_descriptor(n as u32), STAGE);
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    let mut buf = [sfpu::nop(); 64];
    let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..k]);
    p.push(set_adc_x_unpack(0, n as u32 - 1));
    p.push(unpack_instruction());
    barrier(&mut p);

    // 2. Hand a SrcA bank to the Matrix Unit.
    p.extend(src_thread_config());
    let mut words = ConfigWords::new();
    let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
    unpack_src_config(&mut words, Unpacker::SrcA, descriptor, STAGE, 4);
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..k]);
    p.push(set_adc_x(Unpacker::SrcA, 0, n as u32 - 1));
    p.push(unpack_src_instruction(Unpacker::SrcA, true));
    barrier(&mut p);

    // 3. Overwrite SrcA rows 0..2 from Dst, then move SrcA back to Dst row 8.
    for r in 0..2u32 {
        p.push(encode::Movd2A::ZERO.src_row(r).dst_row(r).encode().unwrap());
    }
    barrier(&mut p);
    p.push(
        encode::Mova2D::ZERO
            .move8_rows(1)
            .src_row(0)
            .dst_row(8)
            .encode()
            .unwrap(),
    );
    barrier(&mut p);

    in_device(|dev| {
        let stage = [(STAGE, staged.as_slice())];
        let out = harness::run(dev, &Run::new(&p).stage(&stage).dump_rows(16));
        let bad = |base: usize| -> Vec<String> {
            (0..n)
                .filter(|&i| out.dst[base + i] != datums[i])
                .map(|i| format!("c{}:{:08x}", i % ROW, out.dst[base + i]))
                .collect()
        };
        measure("movd2a.dst_rows_0_2.bad", bad(0).join(" "));
        measure("movd2a.back_rows_8_10.bad", bad(8 * ROW).join(" "));
    });
}

/// Does `ConfigWords::program` write the configuration it was given?
///
/// It loads each word into one scratch GPR with a `SETDMAREG` pair (Scalar
/// Unit) and immediately `WRCFG`s it (Configuration Unit), with no fence either
/// side, trusting each word to be consumed before the next overwrites the GPR.
/// LLK never does that: it waits for the Scalar Unit before every `WRCFG` and
/// separates each from what follows. This runs the `SrcA` path's configuration
/// program alone and reads every word back through `CFGREG`, several times.
#[test]
fn m19_config_words_land_as_written() {
    assert_on_silicon();
    let mut words = ConfigWords::new();
    let descriptor = flat_descriptor(20).with_in_data_format_raw(0);
    unpack_src_config(&mut words, Unpacker::SrcA, descriptor, STAGE, 4);
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    let mut buf = vec![sfpu::nop(); words.program_len()];
    let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
    let mut p = src_thread_config();
    p.extend_from_slice(&buf[..k]);
    p.push(
        backend::stallwait(
            backend::Before::EVERYTHING.mask(),
            backend::cond::CONFIG_BUSY,
        )
        .unwrap(),
    );
    let intended: Vec<(u16, u32)> = words.words().to_vec();
    for attempt in 0..4 {
        in_device(|dev| {
            let _ = harness::run(dev, &Run::new(&p).dump_rows(0));
            let tile = harness::tensix_tile();
            let w = dev
                .alloc_window(tt_device::tlb::WindowKind::TwoMib)
                .unwrap();
            let mut wrong = Vec::new();
            for &(addr32, want) in &intended {
                dev.write32(&w, tile, tensix::CFGREG_RD_CNTL, addr32 as u32)
                    .unwrap();
                harness::advance(dev, 64);
                let got = dev.read32(&w, tile, tensix::CFGREG_RDDATA).unwrap();
                if got != want {
                    wrong.push(format!("{addr32}:want {want:#x} got {got:#x}"));
                }
            }
            dev.free_window(w);
            measure(
                &format!("cfg_program.run{attempt}"),
                format!(
                    "{} of {} wrong: {}",
                    wrong.len(),
                    intended.len(),
                    wrong.join("; ")
                ),
            );
        });
    }
}

/// Does declaring the `Src` formats to the Matrix Unit fix the column losses?
///
/// LLK always sets `ALU_FORMAT_SPEC_REG0_SrcA` / `REG1_SrcB` to the format it
/// unpacks; the datapath here never has, because ttsim did not care. The Matrix
/// Unit decodes the 19-bit `Src` storage by that format, and card 0's column-9
/// loss survives `MOVD2A` -> `MOVA2D` without the unpacker (m18), so it is on
/// the Matrix side. Same path as m15, both unpackers, with and without.
#[test]
fn m20_src_formats_declared_to_the_matrix_unit() {
    assert_on_silicon();
    let n = 32usize;
    let datums: Vec<u32> = (0..n).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, 0, &datums);
    for declare in [false, true] {
        for unpacker in [Unpacker::SrcA, Unpacker::SrcB] {
            for attempt in 0..3 {
                let mut p = src_thread_config();
                let mut words = ConfigWords::new();
                let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
                unpack_src_config(&mut words, unpacker, descriptor, STAGE, 4);
                words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
                if declare {
                    words
                        .set(alu::ALU_FORMAT_SPEC_REG0_SrcA, 4)
                        .unwrap()
                        .set(alu::ALU_FORMAT_SPEC_REG1_SrcB, 4)
                        .unwrap();
                }
                let mut buf = vec![sfpu::nop(); words.program_len()];
                let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
                p.extend_from_slice(&buf[..k]);
                p.push(set_adc_x(unpacker, 0, n as u32 - 1));
                p.push(unpack_src_instruction(unpacker, true));
                match unpacker {
                    Unpacker::SrcA => {
                        p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
                        p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
                    }
                    Unpacker::SrcB => {
                        p.push(backend::wait_for_unpacker1(backend::Before::EVERYTHING).unwrap());
                        for r in 0..2 {
                            p.push(encode::Movb2D::ZERO.src_row(r).dst_row(r).encode().unwrap());
                        }
                    }
                }
                p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
                in_device(|dev| {
                    let stage = [(STAGE, staged.as_slice())];
                    let out = harness::run(dev, &Run::new(&p).stage(&stage).dump_rows(2));
                    let bad: Vec<usize> = (0..n).filter(|&i| out.dst[i] != datums[i]).collect();
                    let key = format!("{unpacker:?}").to_lowercase();
                    measure(
                        &format!("declared_{declare}.{key}.run{attempt}"),
                        format!("bad {bad:?}"),
                    );
                });
            }
        }
    }
}

/// Is stale `LaneConfig.BLOCK_DEST_MOV` what drops the columns?
///
/// `MOVA2D`/`MOVB2D`/`MOVD2A` skip column `c` when bit `c & 1` of
/// `LaneConfig[c / 2].BLOCK_DEST_MOV` is set (`SFPCONFIG.md`, `MOVA2D.md:91`), a
/// per-column-pair mechanism that fits losses fixed by column across rows and
/// tiles. `LaneConfig` is Vector Unit state that leaving SFPU soft reset should
/// zero; this zeroes it explicitly -- `SFPCONFIG` with `VD = 15`,
/// `MOD1_IMM16_IS_VALUE`, `Imm16 = 0` -- before the m20 path.
#[test]
fn m21_lane_config_zeroed_before_the_move() {
    assert_on_silicon();
    let n = 32usize;
    let datums: Vec<u32> = (0..n).map(|i| (1.0f32 + i as f32).to_bits()).collect();
    let staged = stage(L1Format::Fp32, 0, &datums);
    for zero_lane_config in [false, true] {
        for unpacker in [Unpacker::SrcA, Unpacker::SrcB] {
            for attempt in 0..3 {
                let mut p = Vec::new();
                if zero_lane_config {
                    p.push(tt_isa::isa::generated::encode::sfpconfig(0, 15, 1).unwrap());
                    p.push(backend::wait_for_sfpu(backend::Before::EVERYTHING).unwrap());
                }
                p.extend(src_thread_config());
                let mut words = ConfigWords::new();
                let descriptor = flat_descriptor(n as u32).with_in_data_format_raw(0);
                unpack_src_config(&mut words, unpacker, descriptor, STAGE, 4);
                words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
                let mut buf = vec![sfpu::nop(); words.program_len()];
                let k = words.program(SCRATCH_GPR, &mut buf).unwrap();
                p.extend_from_slice(&buf[..k]);
                p.push(set_adc_x(unpacker, 0, n as u32 - 1));
                p.push(unpack_src_instruction(unpacker, true));
                match unpacker {
                    Unpacker::SrcA => {
                        p.push(backend::wait_for_unpacker0(backend::Before::EVERYTHING).unwrap());
                        p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
                    }
                    Unpacker::SrcB => {
                        p.push(backend::wait_for_unpacker1(backend::Before::EVERYTHING).unwrap());
                        for r in 0..2 {
                            p.push(encode::Movb2D::ZERO.src_row(r).dst_row(r).encode().unwrap());
                        }
                    }
                }
                p.push(backend::wait_for_matrix(backend::Before::EVERYTHING).unwrap());
                in_device(|dev| {
                    let stage = [(STAGE, staged.as_slice())];
                    let out = harness::run(dev, &Run::new(&p).stage(&stage).dump_rows(2));
                    let bad: Vec<String> = (0..n)
                        .filter(|&i| out.dst[i] != datums[i])
                        .map(|i| format!("{i}:{:08x}", out.dst[i]))
                        .collect();
                    let key = format!("{unpacker:?}").to_lowercase();
                    measure(
                        &format!("lane_config_zeroed_{zero_lane_config}.{key}.run{attempt}"),
                        format!("bad [{}]", bad.join(" ")),
                    );
                });
            }
        }
    }
}
