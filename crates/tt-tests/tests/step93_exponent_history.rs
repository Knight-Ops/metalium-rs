//! CLREXPHIST is histogram reset, independently of BFP exponent selection.
//! Pinned ttsim refuses ENABLE_ACC_STATS (SETC16 register 45).
#![cfg(feature = "silicon")]
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::{thcon, thread},
    isa::generated::encode,
    matrix, tensix,
    tile::{L1Format, TileImage},
};
use tt_kernels::{datapath, sfpu::kernel};
use tt_tests::harness::{self, Roles, Run};

#[test]
fn clear_resets_histogram_and_maximum() {
    harness::in_device(|dev| {
        let layout = kernel::plan_layout(1, kernel::Operands::Unary).unwrap();
        let image = TileImage::new(datapath::tile_descriptor(), L1Format::Fp32).unwrap();
        let mut input = vec![0u8; image.total_bytes()];
        for i in 0..1024 {
            let at = image.datum_bit_offset(i) / 8;
            input[at..at + 4].copy_from_slice(&1.0f32.to_le_bytes());
        }
        for rows in [4, 8, 16, 32, 64] {
            let mut up = datapath::thread_config();
            up.extend(datapath::clear_unpacker0_adcs());
            let mut words = ConfigWords::new();
            datapath::tile_unpack_config(&mut words, layout.a_at);
            up.extend(datapath::config_program(&words));
            up.extend(datapath::unpack_tile_to_dst(layout.a_at, 0));
            up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
            let mut pack = vec![
                datapath::state_id(),
                datapath::thread_entry(thread::ENABLE_ACC_STATS_Enable, 1),
                matrix::clear_exponent_history(),
                backend::wait_for_matrix(Before::EVERYTHING).unwrap(),
            ];
            let mut words = ConfigWords::new();
            datapath::pack_config(&mut words, layout.out_at + 16);
            pack.extend(datapath::config_program(&words));
            pack.extend(datapath::pack_rows(rows));
            pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
            // Store state snapshots in unused unpacker configuration after its last
            // read. No subsequent instruction consumes these addresses as geometry.
            let fields = [
                thcon::THCON_SEC0_REG3_Base_address,
                thcon::THCON_SEC1_REG3_Base_address,
                thcon::THCON_SEC0_REG7_Offset_address,
                thcon::THCON_SEC1_REG7_Offset_address,
            ];
            for (n, field) in fields.iter().enumerate() {
                if n == 2 {
                    pack.push(matrix::clear_exponent_history());
                    pack.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
                }
                pack.push(
                    encode::SetdmaregSpecial::ZERO
                        .result_size(1)
                        .input_source(if n % 2 == 0 { 9 } else { 7 })
                        .input_half_reg(if n % 2 == 0 { 0 } else { 6 })
                        .result_half_reg(16)
                        .encode()
                        .unwrap(),
                );
                pack.push(backend::nop());
                pack.push(backend::write_word(8, field.addr32()).unwrap());
                pack.push(backend::nop());
            }
            harness::run(
                dev,
                &Run::roles(Roles {
                    unpack: &up,
                    math: &[],
                    pack: &pack,
                })
                .stage(&[(layout.a_at, &input)]),
            );
            let w = dev
                .alloc_window(tt_device::tlb::WindowKind::TwoMib)
                .unwrap();
            let tile = harness::tensix_tile();
            let got: Vec<_> = fields
                .iter()
                .map(|f| {
                    dev.write32(&w, tile, tensix::CFGREG_RD_CNTL, f.addr32() as u32)
                        .unwrap();
                    harness::advance(dev, 64);
                    dev.read32(&w, tile, tensix::CFGREG_RDDATA).unwrap()
                })
                .collect();
            assert_eq!(
                got,
                [127, (rows * 2) << 24, 0, 0],
                "rows {rows}: nonempty histogram then clear"
            );
        }
    });
}
