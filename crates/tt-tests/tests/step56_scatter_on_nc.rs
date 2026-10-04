//! Fixed reader/writer ownership: B gathers, the resident roles compute,
//! and NC writes outputs under buffer credits. The claims:
//!
//! * a pipelined matmul, element-wise op and reduction give burn-flex's bits,
//!   on one tile and on two, with ops queued back to back (nothing synced
//!   between them, so one op's NC scatters and the next op's B gathers are in
//!   flight together) -- small-integer operands, so every product and sum is
//!   exact and the host is the oracle, not another device schedule;
//! * overlap switches off and on mid-session without changing NC ownership.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

/// Integers in `-7..=7`: the 512-term products stay under 2^15, exact in TF32
/// and FP32 alike.
fn ints(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + seed * 31 + i / 5) % 15) as f32 - 7.0)
        .collect()
}

/// burn-flex's `a @ b`, doubled, and the doubled product's column maxima.
fn flex_chain(a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) -> [Vec<f32>; 3] {
    let tensor = |v: &[f32], r, c| {
        Tensor::<Flex, 2>::from_data(TensorData::new(v.to_vec(), [r, c]), &FlexDevice)
    };
    let product = tensor(a, m, k).matmul(tensor(b, k, n));
    let doubled = product.clone() * 2.0;
    [product, doubled.clone(), doubled.max_dim(0)].map(|t| t.into_data().to_vec::<f32>().unwrap())
}

#[cfg(not(feature = "silicon"))]
fn with_tiles(n: usize, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(n),
            |_, _| Ok(None),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{n} tiles: {e}");
    }
}

#[cfg(feature = "silicon")]
fn with_tiles(n: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(n),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{n} tiles: {e}");
    }
}

/// A matmul, an add of its product and a max over its rows, queued back to
/// back, then downloaded: the three results.
fn chain<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
) -> [Vec<f32>; 3] {
    let c = s
        .matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let d = s
        .eltwise(
            Eltwise {
                kind: kind::ADD,
                scalar: 0.0,
                scalar2: 0.0,
            },
            &c,
            Some(&c),
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let e = s
        .reduce(&d, ReduceOp::Max, Axis::Rows)
        .unwrap_or_else(|e| panic!("{e}"));
    let out = [&c, &d, &e].map(|t| s.download(t).unwrap());
    for t in [c, d, e] {
        s.free(t).unwrap();
    }
    out
}

#[test]
fn fixed_nc_ownership_matches_flex() {
    for units in [1usize, 2] {
        with_tiles(units, |s| {
            // 1024 x 512 @ 512 x 1024: a pipelined matmul on both tile
            // counts, and its 1024-tile product enough for the add and the
            // max to pipeline too.
            let (m, k, n) = (1024, 512, 1024);
            let (av, bv) = (ints(1, m * k), ints(2, k * n));
            let a = s.upload(&av, m, k).unwrap();
            let b = s.upload(&bv, k, n).unwrap();
            let want = flex_chain(&av, &bv, [m, k, n]);
            s.set_pipeline(false);
            assert!(chain(s, &a, &b) == want, "{units} tiles: serialized");
            for round in 0..2 {
                s.set_pipeline(true);
                let before = s.pipelined_blocks();
                let got = chain(s, &a, &b);
                let overlapped = s.pipelined_blocks() - before;
                assert!(overlapped > 0, "{units} tiles: nothing pipelined");
                for (what, (w, g)) in ["product", "sum", "max"].iter().zip(want.iter().zip(&got)) {
                    assert_eq!(w.len(), g.len());
                    for (i, (x, y)) in w.iter().zip(g).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "{units} tiles, round {round}, {what}: element {i}: {x} vs {y}"
                        );
                    }
                }
                println!(
                    "{units} tiles, round {round}: {overlapped} blocks, NC scattering, bits equal"
                );
                s.set_pipeline(false);
                let again = chain(s, &a, &b);
                assert!(
                    again == want,
                    "{units} tiles, round {round}: serialized ownership after overlap"
                );
            }
            s.free(a).unwrap();
            s.free(b).unwrap();
        });
    }
}
