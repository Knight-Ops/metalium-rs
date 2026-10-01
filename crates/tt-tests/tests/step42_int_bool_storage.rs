//! Phase 10 gate (D3, milestone 10.2b): integer and boolean tensors in GDDR.
//!
//! The device moves every 32-bit pattern unchanged (the FP32-coded unpack and
//! pack; `step26_sfpu_isa`'s `INT32` pass-through), so `I32` and `Bool` are
//! tags on the tensor that say what an op may compute on. Held here:
//!
//! - round trips of `I32` (two's complement, `i32::MIN` and `-1` included)
//!   and `Bool` (`0`/`1`) at ragged and whole shapes, and through a view;
//! - every op that computes in FP32 refusing them by type
//!   (`TensorError::Elem`), and `COPY`, which moves datums, taking them;
//! - the logic ops (`kind_sfpu::BOOL_*`) against the host's truth table bit for
//!   bit, the device equal to their programs, with row and column broadcasts,
//!   and their padding claims against the raw tiles.

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::sfpu::ops::{kind_sfpu, reference};
use tt_kernels::sfpu::reduce::{Axis, ReduceOp};
use tt_kernels::tensor::{Elem, Eltwise, Pad, TensorError};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
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
        s.enable_dram(tt_firmware_images::DM_B.1).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Integers of every size and sign, the extremes and the ones whose bits are
/// FP32 denormals and NaNs.
fn ints(seed: u32, n: usize) -> Vec<u32> {
    let specials = [
        0i32,
        1,
        -1,
        i32::MIN,
        i32::MAX,
        0x7f80_0001,
        0x007f_ffff,
        -0x0080_0000,
    ];
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            if i % 7 == 0 {
                specials[(i / 7) % specials.len()] as u32
            } else {
                s >> (s % 29)
            }
        })
        .collect()
}

fn bools(seed: u32, n: usize) -> Vec<u32> {
    ints(seed, n).iter().map(|x| x.count_ones() & 1).collect()
}

const SHAPES: [(usize, usize); 4] = [(37, 70), (64, 128), (1, 5), (33, 1)];

#[test]
fn integers_and_booleans_round_trip_bit_for_bit() {
    with_session(|s| {
        for (r, c) in SHAPES {
            for (elem, v) in [(Elem::I32, ints(3, r * c)), (Elem::Bool, bools(5, r * c))] {
                let t = s.upload_bits(&v, r, c, elem).unwrap();
                assert_eq!(t.elem, elem);
                assert_eq!(t.pad(), Pad::Zero);
                assert_eq!(s.download_bits(&t).unwrap(), v, "{elem:?} [{r}, {c}]");
                s.free(t).unwrap();
            }
        }
        // A view keeps its parent's element type and bits.
        let v = ints(9, 96 * 40);
        let t = s.upload_bits(&v, 96, 40, Elem::I32).unwrap();
        let view = t.rows_view(32, 64).unwrap();
        assert_eq!(view.elem, Elem::I32);
        assert_eq!(s.download_bits(&view).unwrap(), v[32 * 40..]);
    });
}

#[test]
fn ops_that_compute_in_fp32_refuse_integers_and_booleans() {
    with_session(|s| {
        let i = s.upload_bits(&ints(1, 37 * 70), 37, 70, Elem::I32).unwrap();
        let b = s
            .upload_bits(&bools(1, 37 * 70), 37, 70, Elem::Bool)
            .unwrap();
        let f = s.upload(&vec![1.5; 37 * 70], 37, 70).unwrap();
        let refused = |r: Result<_, TensorError>, what: &str| match r {
            Err(TensorError::Elem { .. }) => {}
            Err(e) => panic!("{what}: refused, but not by type: {e}"),
            Ok(_) => panic!("{what}: computed on integers"),
        };
        for kind in [tt_isa::dm::kind::ADD, tt_isa::dm::kind::MUL, kind_sfpu::DIV] {
            let op = Eltwise {
                scalar2: 0.0,
                kind,
                scalar: 0.0,
            };
            refused(s.eltwise(op, &i, Some(&i)).map(|_| ()), "an integer add");
            refused(s.eltwise(op, &f, Some(&b)).map(|_| ()), "FP32 with a Bool");
        }
        for kind in [tt_isa::dm::kind::RELU, kind_sfpu::EXP] {
            refused(
                s.eltwise(
                    Eltwise {
                        scalar2: 0.0,
                        kind,
                        scalar: 0.0,
                    },
                    &b,
                    None,
                )
                .map(|_| ()),
                "a Bool's exp",
            );
        }
        let not = Eltwise {
            scalar2: 0.0,
            kind: kind_sfpu::BOOL_NOT,
            scalar: 0.0,
        };
        refused(s.eltwise(not, &f, None).map(|_| ()), "!x of FP32");
        refused(s.eltwise(not, &i, None).map(|_| ()), "!x of an I32");
        let mm = s.matmul_dram(
            &i,
            false,
            &f,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            1 << 40,
        );
        refused(mm.map(|_| ()), "a matmul of integers");
        refused(s.sum_rows(&i).map(|_| ()), "a column sum of integers");
        refused(
            s.reduce(&b, ReduceOp::Max, Axis::Cols).map(|_| ()),
            "a max of booleans",
        );
        refused(s.download(&i).map(|_| ()), "an I32 downloaded as FP32");
        match s.upload_bits(&[0, 1, 2], 1, 3, Elem::Bool) {
            Err(TensorError::Shape(m)) => assert!(m.contains("not 0 or 1"), "{m}"),
            r => panic!("a Bool of 2: {r:?}"),
        }
        // `COPY` moves datums: an integer comes through it whole.
        let copy = Eltwise {
            scalar2: 0.0,
            kind: tt_isa::dm::kind::COPY,
            scalar: 0.0,
        };
        let c = s.eltwise(copy, &i, None).unwrap();
        assert_eq!(c.elem, Elem::I32);
        assert_eq!(s.download_bits(&c).unwrap(), s.download_bits(&i).unwrap());
    });
}

#[test]
fn the_logic_ops_are_the_truth_tables_with_broadcasts() {
    with_session(|s| {
        for (r, c) in [(37, 70), (64, 96)] {
            let (av, bv) = (bools(11, r * c), bools(13, r * c));
            let (row, col) = (bools(17, c), bools(19, r));
            let a = s.upload_bits(&av, r, c, Elem::Bool).unwrap();
            let b = s.upload_bits(&bv, r, c, Elem::Bool).unwrap();
            let rb = s.upload_bits(&row, 1, c, Elem::Bool).unwrap();
            let cb = s.upload_bits(&col, r, 1, Elem::Bool).unwrap();
            let truth = |kind: u32, x: u32, y: u32| match kind {
                kind_sfpu::BOOL_AND => x & y,
                kind_sfpu::BOOL_OR => x | y,
                kind_sfpu::BOOL_XOR => x ^ y,
                _ => 1 - x,
            };
            let f = |v: &[u32]| v.iter().map(|&x| f32::from_bits(x)).collect::<Vec<f32>>();
            let check = |s: &mut Session<_>, kind: u32, b: Option<(&_, &[u32], usize)>| {
                let op = Eltwise {
                    scalar2: 0.0,
                    kind,
                    scalar: 0.0,
                };
                let out = s.eltwise(op, &a, b.map(|(t, _, _)| t)).unwrap();
                assert_eq!(out.elem, Elem::Bool);
                let got = s.download_bits(&out).unwrap();
                let want: Vec<u32> = (0..r * c)
                    .map(|k| {
                        let y = b.map_or(0, |(_, v, how)| match how {
                            0 => v[k],
                            1 => v[k % c],
                            _ => v[k / c],
                        });
                        truth(kind, av[k], y)
                    })
                    .collect();
                assert_eq!(got, want, "{kind:#x} [{r}, {c}] {:?}", b.map(|x| x.2));
                let model = reference(kind, 0.0, &f(&av), b.map(|(_, v, _)| f(v)).as_deref(), r, c);
                let model: Vec<u32> = model.iter().map(|x| x.to_bits()).collect();
                assert_eq!(got, model, "{kind:#x}: the device is its program");
                // What the op says its padding holds, it holds.
                if out.pad() == Pad::Zero {
                    let raw = s.download_padded(&out).unwrap();
                    assert!(
                        raw.iter().enumerate().all(|(k, x)| {
                            let (i, j) = (k / (32 * c.div_ceil(32)), k % (32 * c.div_ceil(32)));
                            i < r && j < c || x.to_bits() == 0
                        }),
                        "{kind:#x}: claimed zero padding"
                    );
                }
            };
            check(s, kind_sfpu::BOOL_NOT, None);
            for kind in [kind_sfpu::BOOL_AND, kind_sfpu::BOOL_OR, kind_sfpu::BOOL_XOR] {
                check(s, kind, Some((&b, &bv, 0)));
                check(s, kind, Some((&rb, &row, 1)));
                check(s, kind, Some((&cb, &col, 2)));
            }
        }
    });
}
