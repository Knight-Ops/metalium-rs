//! `Session::set_spread`: how many of a session's units each op runs on. Whatever the spread,
//! an op computes the same bits -- a job runs the same wherever it runs, and each builder gets
//! back the unit count it built its jobs for -- and only the units it chose enqueue anything.
//!
//! The oracle is the host: small integers, so every sum and product here is exact in FP32 and
//! TF32 alike. Ops: a row-broadcast add, a relu, a matmul with each operand transposed and
//! a `K` long enough to be cut into blocks, a reduction over rows and over columns, and a copy.
//! The controls: a spread of one unit and of two leave the other units with no list, and a
//! spread by work puts a small op on one unit and a large one on all four.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, Spread, TileChoice};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

const UNITS: usize = 4;

#[cfg(not(feature = "silicon"))]
fn with_tiles(count: usize, test: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    fork_scope(|| {
        let mut simulator = tt_ttsim::Simulator::open().unwrap();
        let device = tt_device::Device::open(simulator.transport()).unwrap();
        let mut session = Session::open(
            device,
            tt_firmware_images::ROLES,
            TileChoice::Count(count),
            |_, _| Ok(None),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        test(&mut session);
    })
    .unwrap();
}

#[cfg(feature = "silicon")]
fn with_tiles(count: usize, test: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    fork_scope(|| {
        let mut session = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(count),
        )
        .unwrap();
        session
            .enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        test(&mut session);
    })
    .unwrap();
}

/// Integers in `-3..=3`.
fn ints(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 7 + seed * 13 + i / 3) % 7) as f32 - 3.0)
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn ew<T: tt_device::Transport>(
    s: &mut Session<T>,
    kind: u32,
    a: &DramTensor,
    b: Option<&DramTensor>,
) -> DramTensor {
    s.eltwise(
        Eltwise {
            scalar2: 0.0,
            kind,
            scalar: 0.0,
        },
        a,
        b,
    )
    .unwrap()
}

/// The ops, run under the session's spread, against the host; and the lists each unit
/// enqueued for them.
fn run_ops<T: tt_device::Transport>(s: &mut Session<T>) -> Vec<u64> {
    let (m, k, n) = (64, 256, 96);
    let x = ints(m * k, 1);
    let w = ints(k * n, 2);
    let bias = ints(n, 3);
    // `x` is uploaded as its transpose and `w` as its own, so the matmul reads `x`
    // transposed; then `w` transposed, the other way.
    let xt: Vec<f32> = (0..k * m).map(|i| x[(i % m) * k + i / m]).collect();
    let wt: Vec<f32> = (0..n * k).map(|i| w[(i % k) * n + i / k]).collect();
    let dx_t = s.upload(&xt, k, m).unwrap();
    let dw = s.upload(&w, k, n).unwrap();
    let dw_t = s.upload(&wt, n, k).unwrap();
    let db = s.upload(&bias, 1, n).unwrap();
    s.sync().unwrap();
    let before = s.lists_per_tile();

    let mm = |s: &mut Session<T>, a, at, b, bt| {
        s.matmul_dram(
            a,
            at,
            b,
            bt,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap()
    };
    let h1 = mm(s, &dx_t, true, &dw, false);
    let h2 = mm(s, &dx_t, true, &dw_t, true);
    let hb = ew(s, kind::ADD_ROW, &h1, Some(&db));
    let r = ew(s, kind::RELU, &hb, None);
    let col_sums = s.reduce(&r, ReduceOp::Sum, Axis::Rows).unwrap();
    let row_sums = s.reduce(&r, ReduceOp::Sum, Axis::Cols).unwrap();
    let copied = s.copy(&h2).unwrap();
    let after = s.lists_per_tile();

    // The host's.
    let mut h = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            h[i * n + j] = (0..k).map(|kk| x[i * k + kk] * w[kk * n + j]).sum();
        }
    }
    let rr: Vec<f32> = (0..m * n).map(|e| (h[e] + bias[e % n]).max(0.0)).collect();
    let cs: Vec<f32> = (0..n)
        .map(|j| (0..m).map(|i| rr[i * n + j]).sum())
        .collect();
    let rs: Vec<f32> = (0..m)
        .map(|i| (0..n).map(|j| rr[i * n + j]).sum())
        .collect();

    assert_eq!(
        bits(&s.download(&h1).unwrap()),
        bits(&h),
        "matmul, A transposed"
    );
    assert_eq!(
        bits(&s.download(&h2).unwrap()),
        bits(&h),
        "matmul, both transposed"
    );
    assert_eq!(bits(&s.download(&copied).unwrap()), bits(&h), "copy");
    assert_eq!(bits(&s.download(&r).unwrap()), bits(&rr), "add a row, relu");
    assert_eq!(
        bits(&s.download(&col_sums).unwrap()),
        bits(&cs),
        "sum over rows"
    );
    assert_eq!(
        bits(&s.download(&row_sums).unwrap()),
        bits(&rs),
        "sum over columns"
    );
    for t in [
        dx_t, dw, dw_t, db, h1, h2, hb, r, col_sums, row_sums, copied,
    ] {
        s.free(t).unwrap();
    }
    after.iter().zip(&before).map(|(a, b)| a - b).collect()
}

#[test]
fn every_spread_computes_the_hosts_bits_on_the_units_it_chose() {
    with_tiles(UNITS, |s| {
        assert_eq!(s.lists_per_tile().len(), UNITS);
        for spread in [
            Spread::All,
            Spread::Fixed(1),
            Spread::Fixed(2),
            Spread::Work { share: 16, min: 1 },
            Spread::Work {
                share: 1 << 20,
                min: 1,
            },
        ] {
            s.set_spread(spread);
            let lists = run_ops(s);
            let used = lists.iter().filter(|&&l| l > 0).count();
            match spread {
                // The ops here all have work for four units.
                Spread::All => assert_eq!(used, UNITS, "{spread:?}: {lists:?}"),
                Spread::Fixed(k) => {
                    assert!(lists[..k].iter().all(|&l| l > 0), "{spread:?}: {lists:?}");
                    assert!(lists[k..].iter().all(|&l| l == 0), "{spread:?}: {lists:?}");
                }
                // A share larger than any op's work: one unit for every op.
                Spread::Work { share, .. } if share > 1 << 16 => {
                    assert_eq!(used, 1, "{spread:?}: {lists:?}")
                }
                Spread::Work { .. } => {}
            }
        }
    });
}

/// A spread by work: an op small for its share runs on one unit, a large one on all.
#[test]
fn a_spread_by_work_gives_a_small_op_one_unit_and_a_large_one_all() {
    with_tiles(UNITS, |s| {
        s.set_spread(Spread::Work { share: 16, min: 1 });
        let small = s.upload(&ints(32 * 64, 4), 32, 64).unwrap();
        let large = s.upload(&ints(256 * 256, 5), 256, 256).unwrap();
        s.sync().unwrap();
        for (t, want, tiles) in [(&small, 1, 2), (&large, UNITS, 64)] {
            let before = s.lists_per_tile();
            let r = ew(s, kind::RELU, t, None);
            let lists: Vec<u64> = s
                .lists_per_tile()
                .iter()
                .zip(&before)
                .map(|(a, b)| a - b)
                .collect();
            let used = lists.iter().filter(|&&l| l > 0).count();
            assert_eq!(
                used, want,
                "a relu of {tiles} tiles at 16 a unit: {lists:?}"
            );
            let want_bits: Vec<f32> = s.download(t).unwrap().iter().map(|v| v.max(0.0)).collect();
            assert_eq!(bits(&s.download(&r).unwrap()), bits(&want_bits));
            s.free(r).unwrap();
        }
        s.free(small).unwrap();
        s.free(large).unwrap();
    });
}
