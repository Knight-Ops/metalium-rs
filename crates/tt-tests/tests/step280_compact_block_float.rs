//! Compact block-float tiles (`docs/plans/compact-block-float.md` C0): a tile read into its L1
//! slot without its 16-byte header leaves whatever the slot held there. The unpacker takes the
//! tile's shape and format from its configuration, not from the header, so a BFP8, BFP4 or BFP2
//! tile unpacks to the same `Dst` rows whatever its header bytes are.
use tt_isa::{
    backend::{self, Before, ConfigWords},
    tile::{L1Format, TileImage},
};
use tt_kernels::datapath;
use tt_tests::harness::{self, Roles, Run};

#[test]
fn a_block_float_tile_unpacks_the_same_under_any_header() {
    use datapath::Unpacker;
    use tt_isa::{cfg::generated::alu, isa::generated::encode, matrix::Banks};
    harness::in_device(|dev| {
        for (format, code) in [
            (L1Format::Bfp8, 6),
            (L1Format::Bfp4, 7),
            (L1Format::Bfp2, 15),
        ] {
            let descriptor = datapath::flat_descriptor(128).with_in_data_format_raw(code);
            let image = TileImage::new(descriptor, format).unwrap();
            assert_eq!(image.header_bytes(), 16);
            let mut body = vec![0u8; image.total_bytes()];
            // Eight groups of 16 datums, each its own exponent.
            for (g, e) in body[16..24].iter_mut().enumerate() {
                *e = 120 + g as u8;
            }
            let width = format.datum_bits();
            for i in 0..128 {
                let datum = ((i * 37 + 11) as u8) & ((1u16 << width) - 1) as u8;
                let at = image.datum_bit_offset(i);
                body[at / 8] |= datum << (at % 8);
            }
            let mut unpack = |bytes: &[u8]| {
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
                    .stage(&[(datapath::STAGE, bytes)])
                    .dump_rows(8),
                );
                (0..128)
                    .map(|i| out.dst_at(i / 16, i % 16))
                    .collect::<Vec<_>>()
            };
            let clean = unpack(&body);
            assert!(
                clean.iter().any(|&v| v != 0),
                "{format:?}: the tile unpacks to zeros"
            );
            for header in [[0xffu8; 16], [0xa5; 16], *b"\x13\x37garbage-header"] {
                let mut bytes = body.clone();
                bytes[..16].copy_from_slice(&header);
                assert_eq!(
                    unpack(&bytes),
                    clean,
                    "{format:?} under header {header:02x?}"
                );
            }
        }
    });
}
