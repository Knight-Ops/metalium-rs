//! Checklist 9.15: element-wise ops and reductions overlap their moves with
//! their kernels, as GDDR matmuls do (`step54_pipeline`).
//!
//! With `Session::set_pipeline` (on by default), an op long enough for at
//! least two runs a unit of `tensor::MIN_PIPELINED_RUN` tiles, and a share a
//! unit of `tensor::PIPELINE_SHARE` tiles per unit, is split into runs in
//! alternating halves of the data arena, and a unit's list runs
//! `LAUNCH k, scatter k-1, gather k+1, KERNEL_WAIT k`. The claims:
//!
//! * the results are the plain path's bits -- unary, binary, row-broadcast and
//!   ternary element-wise ops, and sums and maxima over either axis -- on one
//!   tile and on two;
//! * runs overlapped (`Session::pipelined_blocks`) where
//!   `tensor::pipelined_runs` says they may, and not on an op too short.
//!
//! The plain path's own accuracy is `step19_eltwise`'s and `step32`'s.

use tt_kernels::kind;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::kind_sfpu;
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{DramTensor, Elem, Eltwise};
use tt_ttsim::fork_scope;

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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{n} tiles: {e}");
    }
}

/// `op` run plain and then pipelined: the same bits, and how many runs
/// overlapped.
fn both<T: tt_device::Transport>(
    s: &mut Session<T>,
    what: &str,
    op: impl Fn(&mut Session<T>) -> DramTensor,
) -> u64 {
    let run = |s: &mut Session<T>, on: bool| {
        s.set_pipeline(on);
        let before = s.pipelined_blocks();
        let out = op(s);
        let v = s.download(&out).unwrap();
        s.free(out).unwrap();
        (v, s.pipelined_blocks() - before)
    };
    let (plain, none) = run(s, false);
    assert_eq!(none, 0, "{what}: overlapped with pipelining off");
    let (piped, overlapped) = run(s, true);
    assert_eq!(plain.len(), piped.len(), "{what}");
    for (i, (p, q)) in plain.iter().zip(&piped).enumerate() {
        assert_eq!(p.to_bits(), q.to_bits(), "{what}: element {i}: {p} vs {q}");
    }
    println!("{what}: {overlapped} runs overlapped, bits equal");
    overlapped
}

fn ew(kind: u32) -> Eltwise {
    Eltwise {
        kind,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

#[test]
fn pipelined_element_wise_ops_and_reductions_are_the_plain_paths_bits() {
    for units in [1usize, 2] {
        with_tiles(units, |s| {
            // 16 x 16 tiles: on two tiles, a share of 128 tiles a unit
            // (`tensor::PIPELINE_SHARE` asks 96), two runs or more each.
            let (r, c) = (512, 512);
            let a = s.upload(&floats(1, r * c), r, c).unwrap();
            let b = s.upload(&floats(2, r * c), r, c).unwrap();
            let mask = s
                .upload_bits(
                    &floats(3, r * c)
                        .iter()
                        .map(|&x| (x > 0.0) as u32)
                        .collect::<Vec<_>>(),
                    r,
                    c,
                    Elem::Bool,
                )
                .unwrap();
            let bias = s.upload(&floats(4, c), 1, c).unwrap();
            let tiles = (r / 32) * (c / 32);
            let mut overlapped = 0;
            overlapped += both(s, &format!("{units} tiles, add"), |s| {
                s.eltwise(ew(kind::ADD), &a, Some(&b)).unwrap()
            });
            overlapped += both(s, &format!("{units} tiles, exp"), |s| {
                s.eltwise(ew(kind_sfpu::EXP), &a, None).unwrap()
            });
            overlapped += both(s, &format!("{units} tiles, add a row"), |s| {
                s.eltwise(ew(kind::ADD), &a, Some(&bias)).unwrap()
            });
            overlapped += both(s, &format!("{units} tiles, where"), |s| {
                s.eltwise3(ew(kind_sfpu::MASK_WHERE), &a, Some(&mask), Some(&b))
                    .unwrap()
            });
            // Every element-wise op above is split into at least two runs a
            // unit, each overlapped but a unit's first.
            assert!(
                overlapped >= 4 * units as u64,
                "{units} tiles: {overlapped} runs overlapped over {tiles} tiles x 4 ops"
            );
            // 32 x 32 tiles: on two tiles, 512 a unit
            // (`tensor::REDUCE_PIPELINE_SHARE` asks 384).
            let big = s.upload(&floats(5, 1024 * 1024), 1024, 1024).unwrap();
            let mut reduced = 0;
            for (op, axis) in [
                (ReduceOp::Sum, Axis::Rows),
                (ReduceOp::Sum, Axis::Cols),
                (ReduceOp::Max, Axis::Rows),
                (ReduceOp::Max, Axis::Cols),
            ] {
                reduced += both(s, &format!("{units} tiles, {op:?} over {axis:?}"), |s| {
                    s.reduce(&big, op, axis).unwrap()
                });
            }
            assert!(reduced > 0, "{units} tiles: no reduction overlapped");
            for t in [a, b, mask, bias, big] {
                s.free(t).unwrap();
            }
        });
    }
}

/// An op too short for runs of `MIN_PIPELINED_RUN` tiles runs plain.
#[test]
fn a_short_op_runs_plain() {
    with_tiles(3, |s| {
        let a = s.upload(&floats(1, 64 * 64), 64, 64).unwrap();
        let n = both(s, "add, 4 tiles on 3", |s| {
            s.eltwise(ew(kind::ADD), &a, Some(&a)).unwrap()
        });
        assert_eq!(n, 0);
        s.free(a).unwrap();
    });
}
