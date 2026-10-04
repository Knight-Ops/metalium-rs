//! Phase 9 gate: a matmul whose operands and result live in GDDR.
//!
//! `Session::matmul_dram` never stages an operand from the host: the RISCV B
//! mover gathers each chunk's tiles from GDDR into L1 -- transposing the ones
//! autodiff asks for transposed -- and writes the output tiles back. The claim
//! is that it computes exactly what the host-staged `Session::matmul` computes
//! on the same (host-transposed) data, bit for bit, and that the data never
//! crossed PCIe.

use tt_device::tlb::WindowKind;
use tt_isa::dm::{TILE_DATA, TILE_SLOT};
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;

fn floats(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = v[r * cols + c];
        }
    }
    t
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
            |_, _| Ok(None),
        )
        .unwrap();
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
fn tensors_round_trip_through_gddr() {
    with_session(|s| {
        for (i, [r, c]) in [[1, 1], [32, 32], [70, 50], [64, 784], [784, 128]]
            .into_iter()
            .enumerate()
        {
            let v = floats(i as u64, r * c);
            let t = s.upload(&v, r, c).unwrap();
            assert_bits(&s.download(&t).unwrap(), &v, &format!("[{r}, {c}]"));
            s.free(t).unwrap();
        }
        // Everything freed coalesces back into one region per channel.
        let before = s.dram_free_bytes();
        let t = s.upload(&floats(9, 64 * 64), 64, 64).unwrap();
        assert!(s.dram_free_bytes() < before);
        s.free(t).unwrap();
        // The upload is queued, so the free waits for the next sync.
        s.sync().unwrap();
        assert_eq!(s.dram_free_bytes(), before);
    });
}

/// MNIST's forward and backward products, each operand in GDDR, transposed
/// where autodiff transposes it, against the host path on host-transposed data.
#[test]
fn gddr_matmuls_are_bit_identical_to_host_staged_ones() {
    with_session(|s| {
        // (label, A stored [r, c], op(A) transposed?, B stored, op(B) transposed?)
        let cases = [
            ("x@W1", [64, 784], false, [784, 128], false),
            ("h@W2", [64, 128], false, [128, 10], false),
            ("g2@W2t", [64, 10], false, [128, 10], true),
            ("ht@g2", [64, 128], true, [64, 10], false),
            ("xt@g1", [64, 784], true, [64, 128], false),
            ("ragged", [37, 45], true, [37, 70], false),
        ];
        for (i, (label, [ar, ac], ta, [br, bc], tb)) in cases.into_iter().enumerate() {
            let (av, bv) = (
                floats(10 + i as u64, ar * ac),
                floats(20 + i as u64, br * bc),
            );
            let (a_op, [m, k]) = if ta {
                (transpose(&av, ar, ac), [ac, ar])
            } else {
                (av.clone(), [ar, ac])
            };
            let (b_op, n) = if tb {
                (transpose(&bv, br, bc), br)
            } else {
                (bv.clone(), bc)
            };
            let want = s
                .matmul(&a_op, &b_op, [m, k, n], ROUTE, Fidelity::HiFi4, BUDGET)
                .unwrap_or_else(|e| panic!("{e}"));

            let a = s.upload(&av, ar, ac).unwrap();
            let b = s.upload(&bv, br, bc).unwrap();
            let before = s.device().traffic();
            let c = s
                .matmul_dram(&a, ta, &b, tb, ROUTE, Fidelity::HiFi4, BUDGET)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let moved = s.device().traffic() - before;
            let got = s.download(&c).unwrap();
            assert_bits(&got, &want, label);
            let operands = ((ar * ac + br * bc) * 4) as u64;
            println!(
                "MEASURE dram matmul {label}: {} B written, {} B read over PCIe ({} B of operands)",
                moved.bytes_written, moved.bytes_read, operands
            );
            for t in [a, b, c] {
                s.free(t).unwrap();
            }
        }
    });
}

/// The header of a stored tile is never read: fill every operand slot's header
/// with the fill word untouched GDDR holds, and nothing changes. What lets the
/// mover write output datums only (`tensor::matmul_dram`).
#[test]
fn a_tile_headers_contents_do_not_matter() {
    with_session(|s| {
        let (av, bv) = (floats(31, 64 * 96), floats(32, 96 * 64));
        let want = s
            .matmul(&av, &bv, [64, 96, 64], ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap();
        let a = s.upload(&av, 64, 96).unwrap();
        let b = s.upload(&bv, 96, 64).unwrap();
        let w4 = s.device().alloc_window(WindowKind::FourGib).unwrap();
        let junk: Vec<u8> = 0xa616_cb96u32.to_le_bytes().repeat(TILE_DATA as usize / 4);
        for t in [&a, &b] {
            for i in 0..t.placement.tiles() {
                let slot = t.placement.slot(i);
                let h = slot.channel().range(slot.offset(), TILE_DATA).unwrap();
                s.device().dram_write(&w4, h, &junk).unwrap();
            }
        }
        assert_eq!(TILE_SLOT % 64, 0);
        let c = s
            .matmul_dram(&a, false, &b, false, ROUTE, Fidelity::HiFi4, BUDGET)
            .unwrap();
        assert_bits(&s.download(&c).unwrap(), &want, "junk headers");
    });
}
