//! Activations laid by row (`docs/plans/a-by-row.md` P3, `tensor::Layout::ByRow`): a chain of ops
//! whose tensors are laid by row computes the bits it computes interleaved.
//!
//! The claims, each at `m` = 64, 128, 200 (ragged) and 256 rows:
//! - **Every op on a tensor laid by row is the interleaved op bit for bit**: an SFPU eltwise
//!   (`READ_RUN` / `WRITE_RUN`), F32 -> BF16 -> F32 casts (BF16 slots laid by row; silicon only, as
//!   a BF16 pack is), a product
//!   whose `A` is laid by row (read a row a request, `BSpan::of_rows`) and whose output is
//!   (`SCATTER`), a maximum over columns and a sum over rows, a second product chained on the
//!   first's output, a product over a row view (`first` inside the grid), and a transposed `A`
//!   (read down its columns). Under the session's rule (only grids of as many tile rows as
//!   channels) and in the gate's mode (every grid of two tile columns,
//!   `Session::set_row_activations_everywhere`).
//! - **The layout is what is read** (the negative control): references that name the channels
//!   rotated by one (`tensor::probe_row_shift`) give other bits.
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{BlockOperand, DramTensor, Elem, Eltwise, Layout};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(units: usize, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
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
fn with_session(units: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Values whose sums round, so a product's terms and order show in its bits.
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 73 + seed) % 257) as f32 / 131.0 - 1.0 + ((i * 7 + seed) % 5) as f32 * 1e-3)
        .collect()
}

/// How the session lays activations for a run of the chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Interleaved,
    Tall,
    Everywhere,
}

/// The chain's results, downloaded, and the layout of its first eltwise's output.
fn chain<T: tt_device::Transport>(
    s: &mut Session<T>,
    m: usize,
    mode: Mode,
) -> (Vec<Vec<f32>>, Layout) {
    s.set_row_activations(mode != Mode::Interleaved);
    s.set_row_activations_everywhere(mode == Mode::Everywhere);
    let (k, n, n2) = (256, 320, 96);
    let abs = Eltwise {
        kind: tt_kernels::sfpu::ops::kind_sfpu::ABS,
        scalar: 0.0,
        scalar2: 0.0,
    };
    let product = |s: &mut Session<T>, a: &DramTensor, at: bool, b: &DramTensor| {
        s.matmul_dram(
            a,
            at,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::Lo,
            BUDGET * 40,
        )
        .unwrap()
    };
    let a = s.upload(&values(m * k, 1), m, k).unwrap();
    let b = s.upload(&values(k * n, 2), k, n).unwrap();
    let b2 = s.upload(&values(n * n2, 3), n, n2).unwrap();
    let at = s.upload(&values(k * m, 4), k, m).unwrap();
    let e = s.eltwise(abs, &a, None).unwrap();
    let layout = e.placement.layout();
    // A BF16 result's pack is silicon's alone (ttsim refuses PACR mode 0x105, step270).
    let casts = (!s.is_simulated()).then(|| {
        let half = s.cast_block(BlockOperand::whole(&e), Elem::BF16).unwrap();
        let back = s.cast_block(BlockOperand::whole(&half), Elem::F32).unwrap();
        (half, back)
    });
    let y = product(s, casts.as_ref().map_or(&e, |(_, back)| back), false, &b);
    let max = s.reduce(&y, ReduceOp::Max, Axis::Cols).unwrap();
    let sum = s.sum_rows(&y).unwrap();
    let z = s.eltwise(abs, &y, None).unwrap();
    let w = product(s, &z, false, &b2);
    let view = e.rows_view(32, (m - 32) / 32 * 32).unwrap();
    let v = product(s, &view, false, &b);
    let et = s.eltwise(abs, &at, None).unwrap();
    let t = product(s, &et, true, &b);
    let mut out: Vec<Vec<f32>> = [&y, &max, &sum, &w, &v, &t]
        .iter()
        .map(|x| s.download(x).unwrap())
        .collect();
    if let Some((half, back)) = casts {
        out.push(s.download(&half).unwrap());
        s.free(half).unwrap();
        s.free(back).unwrap();
    }
    for x in [a, b, b2, at, e, y, max, sum, z, w, v, et, t] {
        s.free(x).unwrap();
    }
    (out, layout)
}

fn same(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
    const NAMES: [&str; 7] = [
        "product",
        "max",
        "sum",
        "second product",
        "row view",
        "transposed",
        "bf16 cast",
    ];
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.len(), y.len(), "{what}: {}", NAMES[i]);
        for (j, (p, q)) in x.iter().zip(y).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "{what}: {} element {j}: {p} vs {q}",
                NAMES[i]
            );
        }
    }
}

#[test]
fn a_chain_over_activations_laid_by_row_is_the_interleaved_chain() {
    with_session(4, |s| {
        let channels = 8;
        for m in [64, 128, 200, 256] {
            let (plain, flat) = chain(s, m, Mode::Interleaved);
            assert_eq!(flat, Layout::Interleaved);
            let (tall, laid) = chain(s, m, Mode::Tall);
            let rows = m.div_ceil(32);
            let want = if rows >= channels {
                Layout::ByRow
            } else {
                Layout::Interleaved
            };
            assert_eq!(laid, want, "m {m}: the session's rule");
            same(&plain, &tall, &format!("m {m}, the session's rule"));
            let (all, laid) = chain(s, m, Mode::Everywhere);
            assert_eq!(laid, Layout::ByRow, "m {m}: the gate's mode");
            same(&plain, &all, &format!("m {m}, every grid"));
        }
        // The control: the same chain read with its channels rotated is not the chain.
        tt_kernels::tensor::probe_row_shift(true);
        let (shifted, _) = chain(s, 256, Mode::Everywhere);
        tt_kernels::tensor::probe_row_shift(false);
        let (plain, _) = chain(s, 256, Mode::Interleaved);
        let differs = plain
            .iter()
            .zip(&shifted)
            .any(|(x, y)| x.iter().zip(y).any(|(p, q)| p.to_bits() != q.to_bits()));
        assert!(differs, "the rotated layout read the same bits");
        s.set_row_activations(false);
        s.set_row_activations_everywhere(false);
    });
}
