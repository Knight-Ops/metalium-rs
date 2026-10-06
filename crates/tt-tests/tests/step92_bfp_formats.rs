//! Blackhole format-code and physical-stream characterization. Candidate
//! codes originally came from the WH table; both-card measurements establish
//! the corresponding Blackhole shipping format constants.
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::thcon,
    tile::{L1Format, TileImage},
};
use tt_kernels::{datapath, sfpu::kernel};
use tt_tests::harness::{self, Roles, Run};

// Independent arithmetic oracle: BF16 truncation followed by quantization in
// units of 2^(shared exponent - 127 - 6). Sub-byte formats then truncate the
// BFP8 magnitude. This intentionally uses f64 scaling, not the decoder's shifts.
fn reference(values: &[f32], image: TileImage) -> Vec<u8> {
    let mut result = vec![0u8; image.total_bytes()];
    for (group, row) in values.chunks_exact(16).enumerate() {
        let exponent = row.iter().map(|v| (v.to_bits() >> 23) as u8).max().unwrap();
        result[image.header_bytes() + group] = exponent;
        let quantum = 2.0f64.powi(exponent as i32 - 127 - 6);
        for (col, value) in row.iter().enumerate() {
            let bits = value.to_bits() & 0xffff0000;
            let scaled = if bits & 0x7f800000 == 0 {
                0.0
            } else if bits & 0x7f800000 == 0x7f800000 {
                // Interpret the exponent-255 significand directly; IEEE f64
                // arithmetic cannot represent the packer's NaN magnitude.
                64.0 + ((bits >> 16) & 127) as f64 / 2.0
            } else {
                f32::from_bits(bits).abs() as f64 / quantum
            };
            let magnitude = scaled.round().min(127.0) as u8;
            let width = image.format().datum_bits();
            let magnitude = magnitude >> (8 - width);
            let datum = if magnitude == 0 {
                0
            } else {
                magnitude | (((bits >> 31) as u8) << (width - 1))
            };
            let at = image.datum_bit_offset(group * 16 + col);
            result[at / 8] |= datum << (at % 8);
        }
    }
    result
}

#[test]
fn pack_block_formats() {
    harness::in_device(|dev| {
        let layout = kernel::plan_layout(1, kernel::Operands::Unary).unwrap();
        let input_image = TileImage::new(datapath::tile_descriptor(), L1Format::Fp32).unwrap();
        let mut values: Vec<_> = (0..1024)
            .map(|i| {
                ((i % 16) as f32 - 8.0 + if i % 3 == 0 { 65.0 / 1024.0 } else { 0.0 })
                    * 2.0f32.powi((i / 16 % 9) - 4)
            })
            .collect();
        let specials = [
            0, 0x80000000, 1, 0x80000001, 0x00010000, 0x80010000, 0x007fffff, 0x807fffff,
            0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345,
            0xffc12345,
        ];
        // Isolated groups expose each special's exponent; one mixed group
        // also tests suppression of finite values by exponent-255 peers.
        for (group, bits) in specials.into_iter().enumerate() {
            values[group * 16..group * 16 + 16].fill(f32::from_bits(bits));
        }
        for (value, bits) in values[256..272].iter_mut().zip(specials) {
            *value = f32::from_bits(bits);
        }
        let mut input = vec![0u8; input_image.total_bytes()];
        for (i, v) in values.iter().enumerate() {
            let at = input_image.datum_bit_offset(i) / 8;
            input[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (format, code) in [
            (L1Format::Bfp8, 6),
            (L1Format::Bfp4, 7),
            (L1Format::Bfp2, 15),
        ] {
            let image = TileImage::new(datapath::tile_descriptor(), format).unwrap();
            let mut up = datapath::thread_config();
            up.extend(datapath::clear_unpacker0_adcs());
            let mut words = ConfigWords::new();
            datapath::tile_unpack_config(&mut words, layout.a_at);
            up.extend(datapath::config_program(&words));
            up.extend(datapath::unpack_tile_to_dst(layout.a_at, 0));
            up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
            let mut pack = vec![datapath::state_id()];
            let mut words = ConfigWords::new();
            datapath::pack_config(&mut words, layout.out_at + 16);
            words
                .set(thcon::THCON_SEC0_REG1_Out_data_format, code)
                .unwrap();
            words
                .set(
                    tt_isa::cfg::generated::alu::ALU_ROUNDING_MODE_Packer_srnd_en,
                    0,
                )
                .unwrap();
            words
                .set(
                    thcon::THCON_SEC0_REG1_Exp_section_size,
                    (image.exponent_section_bytes() / 16) as u32,
                )
                .unwrap();
            pack.extend(datapath::config_program(&words));
            pack.extend(datapath::pack_rows(64));
            pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
            let sentinel = vec![0xa5; image.total_bytes() + 64];
            let out = harness::run(
                dev,
                &Run::roles(Roles {
                    unpack: &up,
                    math: &[],
                    pack: &pack,
                })
                .stage(&[(layout.a_at, &input), (layout.out_at, &sentinel)])
                .read_back(&[(layout.out_at, sentinel.len())]),
            );
            let bytes = &out.l1[0];
            println!(
                "{format:?} bytes={} prefix={:02x?}",
                image.total_bytes(),
                &bytes[16..112]
            );
            assert_eq!(
                &bytes[image.total_bytes()..],
                &[0xa5; 64],
                "writes beyond physical image"
            );
            assert_ne!(
                &bytes[16..image.total_bytes()],
                &sentinel[16..image.total_bytes()],
                "packer must write"
            );
            let expected = reference(&values, image);
            for (at, (&got, &want)) in bytes
                .iter()
                .zip(&expected)
                .enumerate()
                .skip(16)
                .take(image.total_bytes() - 16)
            {
                assert_eq!(got, want, "{format:?} byte {at}");
            }
            // BF16 truncation loses < |x|/128. BFP8 rounding/clamping loses
            // at most one BFP8 quantum; the subsequent sub-byte truncation
            // increases that to at most one quantum of the requested format.
            // Src normalization adds < 2^-126 for a flushed subnormal. This
            // is an absolute group bound, never a guessed relative tolerance.
            for (i, &value) in values.iter().enumerate().filter(|(_, x)| x.is_finite()) {
                let exp = bytes[image.exponent_byte_offset(i).unwrap()];
                if exp == 255 {
                    continue;
                } // Special-valued peers are modelled exactly above.
                let at = image.datum_bit_offset(i);
                let width = format.datum_bits();
                let datum = bytes[at / 8] >> (at % 8) & ((1u16 << width) - 1) as u8;
                let bits = match format {
                    L1Format::Bfp8 => tt_isa::tile::bfp8_to_bf16(datum, exp),
                    L1Format::Bfp4 => tt_isa::tile::bfp4_to_bf16(datum, exp),
                    L1Format::Bfp2 => tt_isa::tile::bfp2_to_bf16(datum, exp),
                    _ => unreachable!(),
                };
                let decoded = if bits & 0x7f80 == 0 {
                    0.0
                } else {
                    f32::from_bits((bits as u32) << 16)
                };
                let quantum = 2.0f64.powi(exp as i32 - 127 - (width as i32 - 2));
                let bound = (value as f64).abs() / 128.0 + quantum + 2.0f64.powi(-126);
                assert!(
                    (decoded as f64 - value as f64).abs() <= bound,
                    "{format:?} datum {i}: compression bound {bound}"
                );
            }
            // Execute a safe missing-exponent-section mutant. The physical
            // oracle must reject its displaced datum stream, not merely agree
            // with the same decoding code on both sides.
            words
                .set(thcon::THCON_SEC0_REG1_Exp_section_size, 0)
                .unwrap();
            let mut mutant_pack = vec![datapath::state_id()];
            mutant_pack.extend(datapath::config_program(&words));
            mutant_pack.extend(datapath::pack_rows(64));
            mutant_pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
            let mutant = harness::run(
                dev,
                &Run::roles(Roles {
                    unpack: &up,
                    math: &[],
                    pack: &mutant_pack,
                })
                .stage(&[(layout.a_at, &input), (layout.out_at, &sentinel)])
                .read_back(&[(layout.out_at, sentinel.len())]),
            );
            assert_ne!(
                &mutant.l1[0][16..image.total_bytes()],
                &expected[16..],
                "missing exponent section mutant"
            );
        }
    });
}

#[test]
fn unpack_raw_block_formats() {
    use datapath::Unpacker;
    use tt_isa::{cfg::generated::alu, isa::generated::encode, matrix::Banks, tile};
    harness::in_device(|dev| {
        for (format, code) in [
            (L1Format::Bfp8, 6),
            (L1Format::Bfp4, 7),
            (L1Format::Bfp2, 15),
        ] {
            for (exp, phase) in [0, 1, 6, 127, 254, 255]
                .into_iter()
                .flat_map(|exp| [(exp, 0), (exp, 128)])
            {
                let descriptor = datapath::flat_descriptor(128).with_in_data_format_raw(code);
                let image = TileImage::new(descriptor, format).unwrap();
                let mut bytes = vec![0u8; image.total_bytes()];
                bytes[16..32].fill(exp);
                let mut expected = Vec::new();
                for i in 0..128 {
                    let width = format.datum_bits();
                    let datum = (i as u8 ^ phase) & ((1u16 << width) - 1) as u8;
                    let at = image.datum_bit_offset(i);
                    bytes[at / 8] |= datum << (at % 8);
                    let value = match format {
                        L1Format::Bfp8 => tile::bfp8_to_bf16(datum, exp),
                        L1Format::Bfp4 => tile::bfp4_to_bf16(datum, exp),
                        L1Format::Bfp2 => tile::bfp2_to_bf16(datum, exp),
                        _ => unreachable!(),
                    };
                    // Src-to-Dst normalization flushes subnormals to +0.
                    expected.push(if value & 0x7f80 == 0 {
                        0
                    } else {
                        (value as u32) << 16
                    });
                }
                let mut up = datapath::src_thread_config();
                let mut words = ConfigWords::new();
                datapath::unpack_src_config(
                    &mut words,
                    Unpacker::SrcA,
                    descriptor,
                    datapath::STAGE,
                    code,
                );
                words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
                up.extend(datapath::config_program(&words));
                up.push(datapath::set_adc_x(Unpacker::SrcA, 0, 127));
                let (instruction, banks) = Banks::after_reset()
                    .unpack_a(encode::UnpacrRegular::ZERO.multi_context_mode(1))
                    .unwrap();
                up.push(instruction);
                let mut math = tt_kernels::matmul::math_prelude();
                let (instruction, banks) =
                    banks.mova2d(encode::Mova2D::ZERO.move8_rows(1)).unwrap();
                math.push(instruction);
                math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                let (instruction, _) = banks.release_a().unwrap();
                math.push(instruction);
                let out = harness::run(
                    dev,
                    &Run::roles(Roles {
                        unpack: &up,
                        math: &math,
                        pack: &[],
                    })
                    .stage(&[(datapath::STAGE, &bytes)])
                    .dump_rows(8),
                );
                let got: Vec<_> = (0..128).map(|i| out.dst_at(i / 16, i % 16)).collect();
                assert_eq!(got, expected, "{format:?} shared exponent {exp}");
            }
        }
    });
}
