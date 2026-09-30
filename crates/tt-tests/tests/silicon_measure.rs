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
    p.push(backend::wait_for_unpacker0().unwrap());
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
            p.push(backend::wait_for_unpacker0().unwrap());
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
            p.push(backend::wait_for_unpacker1().unwrap());
            for r in 0..8 {
                p.push(encode::Movb2D::ZERO.src_row(r).dst_row(r).encode().unwrap());
            }
        }
    }
    p.push(backend::wait_for_matrix().unwrap());
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
                p.push(backend::wait_for_unpacker0().unwrap());
                p.push(encode::Mova2D::ZERO.move8_rows(1).encode().unwrap());
            }
            Unpacker::SrcB => {
                p.push(backend::wait_for_unpacker1().unwrap());
                p.push(encode::Movb2D::ZERO.encode().unwrap());
                p.push(encode::Movb2D::ZERO.src_row(1).dst_row(1).encode().unwrap());
            }
        }
        p.push(backend::wait_for_matrix().unwrap());
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
    p.push(backend::wait_for_unpacker0().unwrap());
    p.push(set_adc_x_pack(0, 15));
    p.push(pack_instruction(0b1111, true));
    p.push(backend::wait_for_packer().unwrap());

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
    p.push(backend::wait_for_unpacker0().unwrap());
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
