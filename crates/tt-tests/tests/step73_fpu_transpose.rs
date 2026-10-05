//! TRNSPSRCB is a Src permutation, never a bit-preserving F32 transpose.
//! This gate is simulator coverage until the isolated runner passes both cards.
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::alu,
    isa::generated::encode,
    matrix::{transpose_b_reference, Banks},
    tile::{bf16_to_fp32, fp32_to_bf16_truncate, fp32_to_tf32, L1Format, TileImage},
};
use tt_kernels::datapath::{
    config_program, flat_descriptor, set_adc_x, src_thread_config, unpack_src_config, Unpacker,
    STAGE,
};
use tt_tests::harness::{self, Roles, Run};

fn run(datums: &[u32], format: u32, transposes: usize, base: u32, expected: &[u32]) {
    let image = TileImage::new(flat_descriptor(512), L1Format::Fp32).unwrap();
    let mut staged = vec![0u8; image.total_bytes()];
    for (i, &d) in datums.iter().enumerate() {
        let offset = image.datum_bit_offset(i) / 8;
        staged[offset..offset + 4].copy_from_slice(&d.to_le_bytes());
    }
    let mut up = src_thread_config();
    let mut words = ConfigWords::new();
    unpack_src_config(
        &mut words,
        Unpacker::SrcA,
        flat_descriptor(512),
        STAGE,
        format,
    );
    unpack_src_config(
        &mut words,
        Unpacker::SrcB,
        flat_descriptor(512),
        STAGE,
        format,
    );
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    up.extend(config_program(&words));
    up.push(set_adc_x(Unpacker::SrcB, 0, 511));
    let (instruction, mut banks) = Banks::after_reset()
        .unpack_b(encode::UnpacrRegular::ZERO.multi_context_mode(1))
        .unwrap();
    up.push(instruction);
    up.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());
    let mut math = tt_kernels::matmul::math_prelude();
    for _ in 0..transposes {
        let (instruction, next) = banks.transpose_b().unwrap();
        math.push(instruction);
        banks = next;
    }
    for r in 0..16 {
        let (instruction, next) = banks
            .movb2d(encode::Movb2D::ZERO.src_row(base + r).dst_row(r))
            .unwrap();
        math.push(instruction);
        banks = next;
    }
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    let (release, _) = banks.release_b().unwrap();
    math.push(release);
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &up,
                math: &math,
                pack: &[],
            })
            .stage(&[(STAGE, &staged)])
            .dump_rows(16),
        );
        let got: Vec<_> = (0..256).map(|i| out.dst_at(i / 16, i % 16)).collect();
        for (i, (&got, &want)) in got.iter().zip(expected).enumerate() {
            assert_eq!(got, want, "format {format}, base {base}, transposes {transposes}, datum {i}: {got:08x} vs {want:08x}");
        }
    })
}

#[test]
fn src_b_transpose_matches_the_permutation_and_is_its_own_inverse() {
    let data: Vec<u32> = (0..512)
        .map(|i| ((i as f32) - 256.0).to_bits() | 0x1fff)
        .collect();
    for format in [4] {
        let decoded: Vec<_> = data
            .iter()
            .map(|&bits| {
                if format == 4 {
                    fp32_to_tf32(bits)
                } else {
                    bf16_to_fp32(fp32_to_bf16_truncate(bits))
                }
            })
            .collect();
        let mut expected = [[0u32; 16]; 32];
        for (r, row) in expected.iter_mut().enumerate() {
            row.copy_from_slice(&decoded[r * 16..r * 16 + 16]);
        }
        transpose_b_reference(&mut expected);
        let expected: Vec<_> = expected.into_iter().flatten().collect();
        assert_ne!(
            expected, decoded,
            "negative control: omission of transpose must fail"
        );
        for base in [0, 16] {
            let range = base as usize * 16..(base as usize + 16) * 16;
            run(&data, format, 0, base, &decoded[range.clone()]);
            run(&data, format, 1, base, &expected[range.clone()]);
            run(&data, format, 2, base, &decoded[range]);
        }
    }
}
