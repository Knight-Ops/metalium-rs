//! Hardware precision rounding held to the functional model and an observed
//! PRNG predecessor. No reset seed or idealized statistical distribution is assumed.
use tt_isa::{
    backend::{self, Before, ConfigWords},
    numerics::stochastic::{self, Precision, Rounding},
    tile::{L1Format, TileImage},
};
use tt_kernels::{
    datapath::{
        self, pack_tile_from_dst, state_id, thread_config, tile_unpack_config, unpack_tile_to_dst,
    },
    sfpu::{kernel, Format, LReg, LoopPolicy, Program},
};
use tt_tests::harness::{self, Roles, Run};

#[test]
fn hardware_precision_modes_match_documented_bugs_and_actual_prng_state() {
    harness::in_device(|dev| {
        let layout = kernel::plan_layout(1, kernel::Operands::Binary).unwrap();
        let snapshot = layout.b_at.unwrap();
        let image = TileImage::new(datapath::tile_descriptor(), L1Format::Fp32).unwrap();
        let specials = [
            0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345,
            0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x3f808000, 0xbf808000, 0x3f80ffff,
            0x3f801fff, 0x3f800000,
        ];
        let bits: Vec<_> = (0..1024)
            .map(|i| {
                if i % 3 == 0 {
                    specials[(i / 3) % specials.len()]
                } else {
                    (i as u32).wrapping_mul(1664525).wrapping_add(1013904223)
                }
            })
            .collect();
        let mut input = vec![0u8; image.total_bytes()];
        for (i, &word) in bits.iter().enumerate() {
            let at = image.datum_bit_offset(i) / 8;
            input[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        for precision in [Precision::Bf16, Precision::Tf32] {
            for rounding in [
                Rounding::Nearest,
                Rounding::TowardZero,
                Rounding::Stochastic,
            ] {
                if !cfg!(feature = "silicon")
                    && (rounding != Rounding::Nearest || precision == Precision::Tf32)
                {
                    // Pinned ttsim exits on rnd_mode=1/2 and instr_mod1=0;
                    // silicon covers all six modes and their >= defect.
                    continue;
                }
                let mut p = Program::with_policy(LoopPolicy::Unrolled);
                p.for_each_row_group(64, |p, o| {
                    p.load(LReg::L0, Format::Int32, o);
                    p.read_prng(LReg::L1);
                    p.hardware_round(LReg::L0, LReg::L2, precision, rounding);
                    p.store(LReg::L1, Format::Int32, o);
                    p.store(LReg::L2, Format::Int32, kernel::OUT_ROW + o);
                });
                let mut up = thread_config();
                up.extend(datapath::clear_unpacker0_adcs());
                let mut words = ConfigWords::new();
                tile_unpack_config(&mut words, layout.a_at);
                up.extend(datapath::config_program(&words));
                up.extend(unpack_tile_to_dst(layout.a_at, 0));
                up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
                let mut math = vec![state_id()];
                math.extend(p.finish());
                math.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());
                let mut pack = vec![state_id()];
                pack.extend(pack_tile_from_dst(layout.out_at + 16, kernel::OUT_ROW));
                pack.extend(pack_tile_from_dst(snapshot + 16, 0));
                pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
                let sentinel = vec![0xa5u8; 4096];
                let out = harness::run(
                    dev,
                    &Run::roles(Roles {
                        unpack: &up,
                        math: &math,
                        pack: &pack,
                    })
                    .stage(&[
                        (layout.a_at, &input),
                        (layout.out_at + 16, &sentinel),
                        (snapshot + 16, &sentinel),
                    ])
                    .read_back(&[(layout.out_at + 16, 4096), (snapshot + 16, 4096)]),
                );
                let decode = |bytes: &[u8]| {
                    bytes
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>()
                };
                let rounded = decode(&out.l1[0]);
                let previous = decode(&out.l1[1]);
                let want: Vec<_> = bits
                    .iter()
                    .zip(&previous)
                    .map(|(&x, &state)| {
                        stochastic::round(x, stochastic::advance(state), precision, rounding)
                    })
                    .collect();
                assert_ne!(want, bits, "identity instruction negative control");
                assert_eq!(rounded, want, "{precision:?}/{rounding:?}");
            }
        }
    });
}

#[test]
fn interpreter_requires_explicit_prng_state_and_consumes_each_enabled_lane() {
    use tt_kernels::sfpu::interp::Vector;
    let mut p = Program::new();
    p.read_prng(LReg::L1);
    p.hardware_round(LReg::L0, LReg::L2, Precision::Bf16, Rounding::Stochastic);
    let code = p.finish();
    assert!(Vector::new().run(&code).is_err());
    let mut v = Vector::new();
    let seed = std::array::from_fn(|i| (i as u32) * 1234567);
    v.prng = Some(seed);
    v.lreg[0] = Some([0x3f808000; 32]);
    v.run(&code).unwrap();
    assert_eq!(v.lreg[1], Some(seed));
    assert_eq!(
        v.prng,
        Some(seed.map(|s| stochastic::advance(stochastic::advance(s))))
    );
    assert_eq!(
        v.lreg[2],
        Some(seed.map(|s| stochastic::round(
            0x3f808000,
            stochastic::advance(s),
            Precision::Bf16,
            Rounding::Stochastic
        )))
    );
}
