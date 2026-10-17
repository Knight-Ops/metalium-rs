//! B1 of `docs/plans/decode-step-bandwidth.md`: a GDDR matmul's later blocks on a unit do not
//! read `A` again (`tt_isa::dm::record::GATHER_A_RESIDENT`). With `K` whole under one staging,
//! `A` sits at the same slots for every block, so a unit whose previous block of the op read the
//! same `A` rows reads only `B`. A decode step's `gate|up` (272 blocks on 120 units) and its head
//! (816) run several blocks a unit.
//!
//! The claims:
//! - **Same bits.** Every product equals the same product with `A` read for every block
//!   (`Session::set_resident_matmul_a(false)`), bit for bit: `m` of 1 (thin), 17 (whole tiles)
//!   and 33 (two tile rows, whose blocks alternate `A` rows on a unit), ragged `N`, `B`
//!   transposed (the head's form), on one unit and on two. Before each product, a product of a
//!   different `A` whose padding rows hold NaN fills the same `A` slots, so a block marked
//!   resident that is not -- a unit's first, or one after other `A` rows -- computes with
//!   another matrix.
//! - **`A` read once a unit.** Captured, the product's `A` bytes (counted at the trace's store,
//!   `Session::set_count_trace_bytes`) are one read of `A`'s rows a unit that ran it, against
//!   one a block with the flag off; its `B` bytes are unchanged.

use tt_isa::dm::TILE_SLOT;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Pad};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const ROUTE: SrcRoute = SrcRoute::Tf32FromFp32;
const FIDELITY: Fidelity = Fidelity::HiFi4;

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

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

#[cfg(not(feature = "silicon"))]
fn with_tiles(tiles: TileChoice, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, tiles, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_tiles(tiles: TileChoice, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            tiles,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Every datum of `t`'s padding rows NaN, in GDDR: what a block that reads another matrix's `A`
/// slots by mistake carries into valid outputs.
fn poison_padding_rows<T: tt_device::Transport>(s: &mut Session<T>, t: &DramTensor) {
    use tt_device::tlb::WindowKind;
    use tt_isa::dm::{face_index, TILE_DATA};
    s.sync().unwrap();
    let w = s.device().alloc_window(WindowKind::FourGib).unwrap();
    let [rt, ct] = t.grid();
    for j in 0..ct {
        let slot = t.tile(rt - 1, j);
        let mut image = vec![0u8; TILE_SLOT as usize];
        s.device().dram_read(&w, slot, &mut image).unwrap();
        for r in (t.rows - (rt - 1) * 32)..32 {
            for c in 0..32 {
                let at = TILE_DATA as usize + 4 * face_index(r, c);
                image[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
        }
        s.device().dram_write(&w, slot, &image).unwrap();
    }
    t.set_pad(Pad::Undefined);
}

/// `[m, k] @ op(B)`, `B` stored `[n, k]` when `b_t`.
const CASES: &[(usize, usize, usize, bool)] = &[
    (1, 2048, 192, false),
    (1, 2048, 170, false),
    (1, 2048, 192, true),
    (17, 512, 300, false),
    (33, 1024, 160, false),
];

#[test]
fn resident_a_equals_reading_a_every_block() {
    for tiles in [
        TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        TileChoice::Count(2),
    ] {
        with_tiles(tiles, |s| {
            let units = s.tiles().len();
            for (i, &(m, k, n, b_t)) in CASES.iter().enumerate() {
                let label = format!("[{m}, {k}] @ [{k}, {n}] b_t {b_t}, {units} units");
                let a = s.upload(&floats(10 + i as u64, m * k), m, k).unwrap();
                let other = s.upload(&floats(20 + i as u64, m * k), m, k).unwrap();
                poison_padding_rows(s, &other);
                let (r, c) = if b_t { (n, k) } else { (k, n) };
                let b = s.upload(&floats(30 + i as u64, r * c), r, c).unwrap();
                let product = |s: &mut Session<_>, resident: bool| {
                    // The `A` slots primed with another matrix, read as whole tiles.
                    s.set_resident_matmul_a(false);
                    s.set_thin_matmul_reads(false);
                    let primed = s.matmul_dram(&other, false, &b, b_t, ROUTE, FIDELITY, BUDGET);
                    s.free(primed.unwrap()).unwrap();
                    s.set_thin_matmul_reads(true);
                    s.set_resident_matmul_a(resident);
                    let c = s.matmul_dram(&a, false, &b, b_t, ROUTE, FIDELITY, BUDGET);
                    s.set_resident_matmul_a(true);
                    let c = c.unwrap_or_else(|e| panic!("{label}: {e}"));
                    let v = s.download(&c).unwrap();
                    s.free(c).unwrap();
                    v
                };
                let want = product(s, false);
                assert!(want.iter().all(|v| v.is_finite()), "{label}: reference");
                assert_bits(&product(s, true), &want, &label);
                for t in [a, other, b] {
                    s.free(t).unwrap();
                }
                println!("{label}: bit for bit");
            }
        });
    }
}

#[test]
fn a_is_read_once_a_unit() {
    let (m, k, n) = (1, 2048, 192);
    with_tiles(TileChoice::Count(2), |s| {
        s.set_count_trace_bytes(true);
        let a = s.upload(&floats(1, m * k), m, k).unwrap();
        let b = s.upload(&floats(2, k * n), k, n).unwrap();
        let mut counted = Vec::new();
        for resident in [false, true] {
            s.set_resident_matmul_a(resident);
            s.begin_trace().unwrap();
            let c = s.matmul_dram(&a, false, &b, false, ROUTE, FIDELITY, BUDGET);
            let id = s.end_trace().unwrap();
            s.free(c.unwrap()).unwrap();
            let ops = s.trace_ops(id).unwrap();
            assert_eq!(ops.len(), 1);
            counted.push(ops[0].bytes);
            s.release_trace(id).unwrap();
        }
        s.set_resident_matmul_a(true);
        let [off, on] = [counted[0], counted[1]];
        // Two units, columns dealt 2, 2, 1, 1 (`tensor::thin_runs`): four blocks, each unit's
        // second one resident. A thin `A` is one `1024 + 64`-byte read of each of its 64 tiles.
        let a_rows = (k as u64 / 32) * (1024 + 64);
        assert_eq!(off.a, 4 * a_rows, "every block reads A");
        assert_eq!(on.a, 2 * a_rows, "A once a unit");
        assert_eq!(on.b, off.b, "B unchanged");
        assert_eq!(off.b, (k as u64 / 32) * (n as u64 / 32) * TILE_SLOT);
        assert_eq!(off.reads - on.reads, 2 * (k as u64 / 32));
        for t in [a, b] {
            s.free(t).unwrap();
        }
    });
}
