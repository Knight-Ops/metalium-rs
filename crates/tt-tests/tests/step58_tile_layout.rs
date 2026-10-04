//! Tilizing on the card (`tt_isa::dm::op::TILIZE`, `UNTILIZE`): a tile's
//! mover turns a row-major block in L1 into a tile slot's datums and back,
//! so the host can keep row-major data -- Burn's -- and the tile layout stays
//! on the card.
//!
//! The claims: a band of blocks side by side, ragged in both directions,
//! tilizes to exactly the host tilizer's datums (`matmul::tilize_f32_fp32`,
//! itself checked against `tt_layout`'s), padding zero; untilized into a band
//! filled with a sentinel, the valid datums come back and nothing outside
//! them is written; and a malformed entry is refused on the host.

use tt_device::tlb::WindowKind;
use tt_isa::dm::{error, fill, op};
use tt_kernels::dm::{DataMover, DmError};
use tt_kernels::matmul;
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile};

/// The band: 32 rows of 3 tiles' columns, the last column tile 20 wide, the
/// rows 23 valid (the band is the tensor's last).
const COLS: usize = 84;
const ROWS: usize = 23;
/// Bytes between rows: past the 84 datums, so the band's slack is checked.
const STRIDE: u32 = 3 * 128 + 64;
const BAND: u32 = 0x2_0000;
const SLOTS: u32 = 0x4_0000;
const BACK: u32 = 0x6_0000;

fn entry(op: u32, block: u32, slot: u32, valid: u32) -> [u32; 8] {
    [op, block, STRIDE, slot, valid, 0, 0, 0]
}

fn valid(j: usize) -> u32 {
    fill::param(ROWS as u32, (COLS - 32 * j).min(32) as u32)
}

#[test]
fn the_mover_tilizes_and_untilizes_a_ragged_band() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let mut m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
        let values: Vec<f32> = (0..ROWS * COLS).map(|i| i as f32 * 0.25 - 100.0).collect();
        // The band as the host would stage it: rows `STRIDE` apart, the slack
        // and the rows past the valid ones holding garbage.
        let mut band = vec![0xA5u8; 32 * STRIDE as usize];
        for r in 0..ROWS {
            for c in 0..COLS {
                let at = r * STRIDE as usize + 4 * c;
                band[at..at + 4].copy_from_slice(&values[r * COLS + c].to_le_bytes());
            }
        }
        d.l1_write(&w, t, BAND as u64, &band).unwrap();
        d.l1_write(&w, t, SLOTS as u64, &vec![0xEEu8; 3 * 4096])
            .unwrap();
        let tilize: Vec<_> = (0..3)
            .map(|j| {
                entry(
                    op::TILIZE,
                    BAND + 128 * j as u32,
                    SLOTS + 4096 * j as u32,
                    valid(j),
                )
            })
            .collect();
        let n = m.enqueue(d, &w, &tilize).unwrap();
        m.wait_for(d, &w, n).unwrap();
        let want = matmul::tilize_f32_fp32(&values, ROWS, COLS);
        for j in 0..3 {
            let mut got = vec![0u8; 4096];
            d.l1_read(&w, t, SLOTS as u64 + 4096 * j as u64, &mut got)
                .unwrap();
            let image = &want[j * matmul::TILE_IMAGE_BYTES + 16..][..4096];
            assert!(got == image, "tile {j}: not the host tilizer's datums");
        }
        // Back, into a band of sentinels: valid datums only.
        d.l1_write(&w, t, BACK as u64, &vec![0x5Au8; 32 * STRIDE as usize])
            .unwrap();
        let untilize: Vec<_> = (0..3)
            .map(|j| {
                entry(
                    op::UNTILIZE,
                    BACK + 128 * j as u32,
                    SLOTS + 4096 * j as u32,
                    valid(j),
                )
            })
            .collect();
        let n = m.enqueue(d, &w, &untilize).unwrap();
        m.wait_for(d, &w, n).unwrap();
        let mut back = vec![0u8; 32 * STRIDE as usize];
        d.l1_read(&w, t, BACK as u64, &mut back).unwrap();
        for r in 0..32 {
            for b in 0..STRIDE as usize {
                let at = r * STRIDE as usize + b;
                let want = if r < ROWS && b < 4 * COLS {
                    band[at]
                } else {
                    0x5A
                };
                assert_eq!(back[at], want, "row {r}, byte {b}");
            }
        }
        // Refused on the host.
        for (bad, code) in [
            (
                [op::TILIZE, BAND + 2, STRIDE, SLOTS, 0, 0, 0, 0],
                error::ALIGNMENT,
            ),
            ([op::TILIZE, BAND, 64, SLOTS, 0, 0, 0, 0], error::ALIGNMENT),
            (
                [op::UNTILIZE, BAND, STRIDE, SLOTS + 8, 0, 0, 0, 0],
                error::ALIGNMENT,
            ),
            (
                [op::TILIZE, BAND, STRIDE, SLOTS, 0x2020, 0, 0, 0],
                error::OP,
            ),
        ] {
            let e = m.enqueue(d, &w, &[bad]).unwrap_err();
            assert!(
                matches!(e, DmError::Invalid(c) if c == code),
                "{bad:x?}: {e}"
            );
        }
        // What a tile costs: 64 tilizes in one list, host-timed.
        let many: Vec<_> = (0..64u32)
            .map(|k| entry(op::TILIZE, BAND + 128 * (k % 3), SLOTS + 4096 * (k % 3), 0))
            .collect();
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            let n = m.enqueue(d, &w, &many).unwrap();
            m.wait_for(d, &w, n).unwrap();
        }
        println!(
            "MEASURE tilize: {:.2} us a tile (64 a list, list overhead included)",
            t0.elapsed().as_secs_f64() * 1e6 / 640.0
        );
        m.stop(d, &w).unwrap();
    });
}

/// The session's two places to tilize (`Session::set_tilize`) give the same
/// tensors: uploaded either way and downloaded either way, every shape's bits
/// round-trip, ragged ones and a view's rows included, on one tile and three.
#[test]
fn tilizing_on_the_card_or_the_host_gives_the_same_tensors() {
    use tt_kernels::session::{Session, TileChoice, Tilize};
    fn check<T: tt_device::Transport>(s: &mut Session<T>, units: usize) {
        for [r, c] in [[1usize, 1usize], [64, 10], [70, 50], [64, 784], [128, 3100]] {
            let v: Vec<f32> = (0..r * c).map(|i| (i as f32).sin() * 1e3).collect();
            for up in [Tilize::Host, Tilize::Card] {
                s.set_tilize(up);
                let t = s.upload(&v, r, c).unwrap();
                for down in [Tilize::Host, Tilize::Card] {
                    s.set_tilize(down);
                    let back = s.download(&t).unwrap();
                    assert!(
                        back.iter().zip(&v).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{units} tiles, [{r}, {c}], up {up:?}, down {down:?}"
                    );
                    if r >= 64 {
                        let view = t.rows_view(32, 32).unwrap();
                        let rows = s.download(&view).unwrap();
                        assert!(
                            rows.iter()
                                .zip(&v[32 * c..64 * c])
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{units} tiles, [{r}, {c}] rows 32..64, down {down:?}"
                        );
                    }
                }
                s.free(t).unwrap();
            }
        }
    }
    for units in [1usize, 3] {
        #[cfg(not(feature = "silicon"))]
        let r = tt_ttsim::fork_scope(|| {
            let mut sim = tt_ttsim::Simulator::open().unwrap();
            let dev = tt_device::Device::open(sim.transport()).unwrap();
            let mut s = Session::open(
                dev,
                tt_firmware_images::ROLES,
                TileChoice::Count(units),
                |_, _| Ok(None),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            check(&mut s, units);
        });
        #[cfg(feature = "silicon")]
        let r = tt_ttsim::fork_scope(|| {
            let mut s = Session::open_card(
                tt_tests::backend::device_index(),
                tt_firmware_images::ROLES,
                TileChoice::Count(units),
            )
            .unwrap_or_else(|e| panic!("{e}"));
            s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
                .unwrap();
            check(&mut s, units);
        });
        if let Err(e) = r {
            panic!("{units} tiles: {e}");
        }
    }
}
