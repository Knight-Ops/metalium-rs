//! RMSNorm's statistic over only the rows a tile row has (`docs/plans/decode-small-ops.md` S3):
//! a tile row of fewer than 32 rows -- a decode step's `[1, d]` -- runs the kernel's programs
//! over only the row groups its rows occupy (`tt_kernels::sfpu::reduce::row_offsets`: one of
//! eight for one row), on the card.
//!
//! The claims:
//! - **Same bits.** Every row's statistic equals the specification model's run of the same
//!   programs (`sfpu::norm::reference`, itself held bit for bit to the composed ops by the unit
//!   gate), at 1, 2, 5, 16, 17, 33 and 48 rows (last tile rows of 1, 2, 5, 16, 17, 1 and 16
//!   rows) and widths 2048, 576 and 70 (ragged).
//! - **Every row used is computed.** Before each statistic, one of a `[32, cols]` tensor of NaN
//!   runs on the same unit: its NaN fills the same L1 slots and every row of the accumulator in
//!   `Dst`, so a row whose group the programs skip shows as NaN. (Watched failing with the last
//!   tile row programmed for a group of rows too few.)
//! - **An honest claim:** the output's padding rows are whatever `Dst` held, claimed
//!   `Pad::Undefined`.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::norm;
use tt_kernels::tensor::Pad;
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let scale = [1.0e-3f32, 1.0, 37.0][i % 3];
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let tiles = TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1);
        let mut s = Session::open(dev, tt_firmware_images::ROLES, tiles, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[test]
fn row_limited_statistic_is_the_models_bit_for_bit_over_primed_l1() {
    with_session(|s| {
        let eps = 1e-5f32;
        for cols in [2048, 576, 70] {
            let nan = s.upload(&vec![f32::NAN; 32 * cols], 32, cols).unwrap();
            for rows in [1, 2, 5, 16, 17, 33, 48] {
                let label = format!("[{rows}, {cols}]");
                let x = floats((rows * 10_000 + cols) as u64, rows * cols);
                let want = norm::reference(&x, rows, cols, eps);
                let a = s.upload(&x, rows, cols).unwrap();
                // The same slots, whole, every datum NaN.
                let primed = s.rms(&nan, eps).unwrap();
                let got = s.rms(&a, eps).unwrap();
                assert_eq!(got.pad(), Pad::Undefined, "{label}: claim");
                let got_v = s.download(&got).unwrap();
                assert_eq!(got_v.len(), rows, "{label}");
                for (r, (g, w)) in got_v.iter().zip(&want).enumerate() {
                    assert_eq!(g.to_bits(), w.to_bits(), "{label}: row {r}: {g} for {w}");
                }
                for t in [primed, got, a] {
                    s.free(t).unwrap();
                }
                println!("{label}: bit for bit");
            }
            s.free(nan).unwrap();
        }
    });
}

/// A decode row's statistic over a long `K` reads the row packed (`Session::set_packed_rms`,
/// `sfpu::norm::math_packed`): `pack_row` spreads the row's tiles over the units, and the
/// statistic's one unit reads `K / 1024` tiles. Bit for bit the model and the unpacked path, at
/// `K` of 1024, 1056 (a packed tile of one row), 2048, 4096 and 8192 (past the unpacked path's one
/// job, which refuses it by name), with signed zeros,
/// denormals, `f32::MAX`, infinities and NaNs of two payloads among ordinary values; the packing
/// is seen to run (a transfer packet) and not to run with the switch off.
#[test]
fn a_decode_row_s_statistic_reads_it_packed() {
    const SPECIAL: [u32; 8] = [
        0x8000_0000,
        0x0000_0001,
        0x807f_ffff,
        0x7f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffa0_0001,
    ];
    with_session(|s| {
        let eps = 1e-5f32;
        for cols in [1024usize, 1056, 2048, 4096, 8192] {
            let plain = floats(cols as u64, cols);
            let mut cases = vec![plain.clone()];
            for k in 0..SPECIAL.len() {
                // One special value a case, the last two NaNs of different payloads together.
                let mut x = plain.clone();
                x[(k * 977 + 31) % cols] = f32::from_bits(SPECIAL[k]);
                if k + 1 == SPECIAL.len() {
                    x[cols / 2] = f32::from_bits(SPECIAL[k - 1]);
                }
                cases.push(x);
            }
            for (i, x) in cases.iter().enumerate() {
                let label = format!("[1, {cols}] case {i}");
                let want = norm::reference(x, 1, cols, eps)[0];
                assert_eq!(
                    norm::reference_packed(x, eps).to_bits(),
                    want.to_bits(),
                    "{label}"
                );
                let a = s.upload(x, 1, cols).unwrap();
                let mut got = [want.to_bits(); 2];
                for (n, packed) in [true, false].into_iter().enumerate() {
                    s.set_packed_rms(packed);
                    if !packed && !norm::fits(cols) {
                        // Past one job's tiles the row is refused by name; packed it is not.
                        let e = s.rms(&a, eps).unwrap_err().to_string();
                        assert!(e.contains("does not fit"), "{label}: {e}");
                        continue;
                    }
                    let before = s.dataflow_stats().transfer_packets;
                    let t = s.rms(&a, eps).unwrap();
                    let packets = s.dataflow_stats().transfer_packets - before;
                    assert_eq!(
                        packets > 0,
                        packed,
                        "{label}: packed {packed}, {packets} packets"
                    );
                    got[n] = s.download(&t).unwrap()[0].to_bits();
                    s.free(t).unwrap();
                }
                s.set_packed_rms(true);
                assert_eq!(got, [want.to_bits(); 2], "{label}: packed, unpacked, model");
                s.free(a).unwrap();
            }
        }
    });
}
