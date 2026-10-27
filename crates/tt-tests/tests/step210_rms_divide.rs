//! RMSNorm up to its scale as one op for a decode row (`Session::rms_norm`, `tensor::rms_divide`;
//! decode-to-bandwidth Phase 7): every unit computes the statistic itself from the row's packed
//! twin and divides its own tiles by it, where the statistic ran on one unit as an op of its own
//! and the 63 units of the `DIV` after it waited on that op's barrier.
//!
//! On eight units, the claims:
//! - **Bit for bit the two ops** (`Session::set_rms_divide(false)`: `Session::rms`, then
//!   `kind_sfpu::DIV` by its column), at `K` 1024, 2048 and 4096, over full-mantissa values,
//!   signed zeros, subnormals, values near `f32::MAX` and a row whose statistic overflows, and
//!   rows holding NaN and infinities. The result's packed twin is the packing of the result.
//! - **It is one op:** one ownership region fewer than the two ops.
//! - **In a trace,** `x + z` and its norm captured, and replayed over new `x`, are the two run
//!   fresh.
//!
//! Negative control: a row with one datum moved (enough to move its statistic) gives every other
//! output other bits.

use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::DramTensor;
use tt_ttsim::fork_scope;

fn values(seed: u64, n: usize, specials: bool) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0;
            if !specials {
                return x;
            }
            match i % 61 {
                3 => -0.0,
                7 => 0.0,
                13 => f32::from_bits(0x0000_0042),
                19 => -f32::from_bits(0x0070_0000),
                29 => 3.0e3,
                _ => x,
            }
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
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
            TileChoice::Count(8),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

const EPS: f32 = 1e-5;

/// A row made by an element-wise op (`x * 1`), as a decode step's residual sum is: it has a
/// packed twin.
fn row<T: tt_device::Transport>(s: &mut Session<T>, v: &[f32]) -> DramTensor {
    let raw = s.upload(v, 1, v.len()).unwrap();
    let one = tt_kernels::tensor::Eltwise {
        kind: tt_kernels::kind::MUL_SCALAR,
        scalar: 1.0,
        scalar2: 0.0,
    };
    let x = s.eltwise(one, &raw, None).unwrap();
    s.free(raw).unwrap();
    x
}

/// `rms_norm(x)` with the fused op on or off, its bits, its twin's bits and the regions it took.
fn norm<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    fused: bool,
) -> (Vec<u32>, Option<Vec<u32>>, u64) {
    s.set_rms_divide(fused);
    let before = s.dataflow_stats().regions;
    let y = s.rms_norm(x, None, EPS).unwrap();
    let regions = s.dataflow_stats().regions - before;
    s.set_rms_divide(true);
    let out = bits(&s.download(&y).unwrap());
    let twin = s
        .packed_twin(&y)
        .cloned()
        .map(|t| bits(&s.download(&t).unwrap()));
    s.free(y).unwrap();
    (out, twin, regions)
}

#[test]
fn a_decode_row_s_norm_is_the_two_ops_bit_for_bit() {
    with_session(|s| {
        for (k, seed, specials) in [
            (1024usize, 3u64, true),
            (2048, 5, true),
            (4096, 7, true),
            (2048, 11, false),
        ] {
            let x = row(s, &values(seed, k, specials));
            let (fused, twin, fused_regions) = norm(s, &x, true);
            let (want, _, regions) = norm(s, &x, false);
            assert_eq!(fused, want, "K {k}: the two ops' bits");
            println!("K {k}: {fused_regions} regions fused, {regions} as two ops");
            assert!(
                fused_regions < regions,
                "K {k}: {fused_regions} regions fused against {regions} as two ops"
            );
            // The twin is the result's packed row (`pack_row`'s layout, `[K / 32, 32]`: row `kk`
            // is tile `kk`'s data row, here the row's columns `32 kk..32 kk + 32`).
            let twin = twin.expect("the result has a packed twin");
            assert_eq!(twin, fused, "K {k}: the twin is the packed result");
            s.free(x).unwrap();
        }
        // Special rows: NaN and infinities, values near `f32::MAX` (the statistic overflows).
        for special in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3.0e38] {
            let mut v = values(13, 2048, false);
            v[100] = special;
            v[1500] = -special;
            let x = row(s, &v);
            let (fused, _, _) = norm(s, &x, true);
            let (want, _, _) = norm(s, &x, false);
            assert_eq!(fused, want, "a row holding {special}");
            s.free(x).unwrap();
        }
    });
}

#[test]
fn another_row_s_statistic_gives_other_bits() {
    with_session(|s| {
        let v = values(17, 2048, false);
        let mut w = v.clone();
        // Enough to move the statistic: one ulp of one square is under the sum's ulp.
        w[7] += 0.25;
        let (x, y) = (row(s, &v), row(s, &w));
        let (fx, _, _) = norm(s, &x, true);
        let (fy, _, _) = norm(s, &y, true);
        let differing = fx
            .iter()
            .zip(&fy)
            .enumerate()
            .filter(|&(i, (a, b))| i != 7 && a != b)
            .count();
        println!("one datum moved: {differing} of 2047 other outputs differ");
        assert!(differing > 0, "the statistic does not reach the outputs");
        s.free(x).unwrap();
        s.free(y).unwrap();
    });
}

/// `x + z` and its norm, captured and replayed over new `x`, against the two run fresh.
#[test]
fn a_captured_norm_replays_as_the_op_run_fresh() {
    with_session(|s| {
        let k = 2048;
        let add = tt_kernels::tensor::Eltwise {
            kind: tt_kernels::kind::ADD,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let dz = s.upload(&values(2, k, true), 1, k).unwrap();
        let x = s.upload(&values(10, k, true), 1, k).unwrap();
        let step = |s: &mut Session<_>| {
            let y = s.eltwise(add, &x, Some(&dz)).unwrap();
            let n = s.rms_norm(&y, None, EPS).unwrap();
            s.free(y).unwrap();
            n
        };
        let mut fresh = Vec::new();
        for seed in [10u64, 20, 21] {
            s.write(&x, &values(seed, k, true)).unwrap();
            let n = step(s);
            fresh.push(bits(&s.download(&n).unwrap()));
            s.free(n).unwrap();
        }
        s.write(&x, &values(10, k, true)).unwrap();
        s.begin_trace().unwrap();
        let n = step(s);
        let id = s.end_trace().unwrap();
        let mut got = vec![bits(&s.download(&n).unwrap())];
        for seed in [20u64, 21] {
            s.write(&x, &values(seed, k, true)).unwrap();
            s.replay(id).unwrap();
            s.sync().unwrap();
            got.push(bits(&s.download(&n).unwrap()));
        }
        assert_eq!(got, fresh, "the capture and its replays");
        s.release_trace(id).unwrap();
        for t in [n, dz, x] {
            s.free(t).unwrap();
        }
    });
}

/// `rms_norm(x, Some(gamma))` with the fused op on or off: its bits, its twin's and the regions.
fn scaled<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    gamma: &DramTensor,
    fused: bool,
) -> (Vec<u32>, Option<Vec<u32>>, u64) {
    s.set_rms_divide(fused);
    let before = s.dataflow_stats().regions;
    let y = s.rms_norm(x, Some(gamma), EPS).unwrap();
    let regions = s.dataflow_stats().regions - before;
    s.set_rms_divide(true);
    let out = bits(&s.download(&y).unwrap());
    let twin = s
        .packed_twin(&y)
        .cloned()
        .map(|t| bits(&s.download(&t).unwrap()));
    s.free(y).unwrap();
    (out, twin, regions)
}

/// The norm times its scale (`gamma`, decode-to-bandwidth Phase 7): bit for bit the norm and then
/// `kind::MUL` by `gamma` as an op of its own, fused or not, over gammas holding signed zeros,
/// subnormals, infinities and NaN; one op where the fused norm is one; the twin the packed
/// result. Negative control: the unscaled norm differs.
#[test]
fn a_scaled_norm_is_the_norm_then_the_product_bit_for_bit() {
    with_session(|s| {
        let mul = tt_kernels::tensor::Eltwise {
            kind: tt_kernels::kind::MUL,
            scalar: 0.0,
            scalar2: 0.0,
        };
        for (k, seed) in [(1024usize, 3u64), (2048, 5), (4096, 7)] {
            let x = row(s, &values(seed, k, true));
            let mut g = values(seed + 100, k, true);
            g[9] = f32::INFINITY;
            g[33] = f32::NAN;
            g[34] = -f32::INFINITY;
            let gamma = s.upload(&g, 1, k).unwrap();
            let n = s.rms_norm(&x, None, EPS).unwrap();
            let want = s.eltwise(mul, &n, Some(&gamma)).unwrap();
            let want_bits = bits(&s.download(&want).unwrap());
            let (_, _, two_regions) = norm(s, &x, true);
            for fused in [true, false] {
                let (got, twin, regions) = scaled(s, &x, &gamma, fused);
                assert_eq!(
                    got, want_bits,
                    "K {k}, fused {fused}: the norm then the product"
                );
                if fused {
                    println!("K {k}: scaled {regions} regions, unscaled {two_regions}");
                    assert_eq!(regions, two_regions, "K {k}: the scale adds no op");
                    assert_eq!(twin.expect("a packed twin"), got, "K {k}: the twin");
                }
            }
            let unscaled = bits(&s.download(&n).unwrap());
            assert_ne!(unscaled, want_bits, "K {k}: the scale reaches the outputs");
            for t in [x, gamma, n, want] {
                s.free(t).unwrap();
            }
        }
    });
}

/// A one-row product's output, `x @ w`, which now keeps a packed twin as an element-wise
/// result's does: the delta a residual sum adds.
fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    x: &DramTensor,
    n: usize,
    seed: u64,
) -> DramTensor {
    let w = s
        .upload(&values(seed, x.cols * n, false), x.cols, n)
        .unwrap();
    let y = s
        .matmul_dram(
            x,
            false,
            &w,
            false,
            tt_kernels::matmul::SrcRoute::Tf32FromFp32,
            tt_kernels::matmul::Fidelity::HiFi4,
            tt_tests::harness::BUDGET,
        )
        .unwrap();
    s.free(w).unwrap();
    y
}

/// `(sum, norm)` bits, the twins' bits and the regions of `add_rms_norm(a, b, gamma)`.
#[allow(clippy::type_complexity)]
fn residual<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
    gamma: Option<&DramTensor>,
) -> ([Vec<u32>; 2], [Option<Vec<u32>>; 2], u64) {
    let before = s.dataflow_stats().regions;
    let (sum, n) = s.add_rms_norm(a, b, gamma, EPS).unwrap();
    let regions = s.dataflow_stats().regions - before;
    let got = [&sum, &n].map(|t| bits(&s.download(t).unwrap()));
    let twins = [&sum, &n].map(|t| {
        s.packed_twin(t)
            .cloned()
            .map(|p| bits(&s.download(&p).unwrap()))
    });
    s.free(sum).unwrap();
    s.free(n).unwrap();
    (got, twins, regions)
}

/// A residual sum and the norm after it as one op (decode-to-bandwidth Phase 7): bit for bit the
/// `ADD` and the norm as two ops, with and without the scale, at `K` 1024, 2048 and 4096 over
/// specials, `b` a one-row product's output (its packed twin the product's); fewer regions than
/// the two ops; the sum's and the norm's twins their packed rows. Controls: an operand with no
/// twin takes the two ops (more regions, the same bits); the norm of `a` alone differs.
#[test]
fn a_residual_sum_and_its_norm_are_the_two_ops_bit_for_bit() {
    with_session(|s| {
        let add = tt_kernels::tensor::Eltwise {
            kind: tt_kernels::kind::ADD,
            scalar: 0.0,
            scalar2: 0.0,
        };
        for (k, seed) in [(1024usize, 3u64), (2048, 5), (4096, 7)] {
            let a = row(s, &values(seed, k, true));
            let x = row(s, &values(seed + 1, 64, false));
            let b = product(s, &x, k, seed + 2);
            assert!(
                s.packed_twin(&b).is_some(),
                "K {k}: the product keeps a twin"
            );
            let mut g = values(seed + 100, k, true);
            g[9] = f32::INFINITY;
            let gamma = s.upload(&g, 1, k).unwrap();
            for scale in [None, Some(&gamma)] {
                let sum = s.eltwise(add, &a, Some(&b)).unwrap();
                let n = s.rms_norm(&sum, scale, EPS).unwrap();
                let want = [&sum, &n].map(|t| bits(&s.download(t).unwrap()));
                s.free(sum).unwrap();
                s.free(n).unwrap();
                let (got, twins, regions) = residual(s, &a, &b, scale);
                let what = format!("K {k}, scaled {}", scale.is_some());
                for (part, (g, w)) in ["sum", "norm"].iter().zip(got.iter().zip(&want)) {
                    if let Some(i) = g.iter().zip(w).position(|(x, y)| x != y) {
                        let n = g.iter().zip(w).filter(|(x, y)| x != y).count();
                        panic!(
                            "{what}: the {part}: {n} differ, first at {i}: {:#010x} for {:#010x}",
                            g[i], w[i]
                        );
                    }
                }
                for (twin, packed) in twins.iter().zip(&got) {
                    assert_eq!(twin.as_ref(), Some(packed), "{what}: a twin");
                }
                // No twin for `b`: the two ops, the same bits.
                let host = s.download(&b).unwrap();
                let plain = s.upload(&host, 1, k).unwrap();
                assert!(s.packed_twin(&plain).is_none());
                let (two, _, two_regions) = residual(s, &a, &plain, scale);
                assert_eq!(two, want, "{what}: as two ops");
                println!("{what}: {regions} regions, {two_regions} as two ops");
                assert!(regions < two_regions, "{what}: one op fewer");
                s.free(plain).unwrap();
            }
            let alone = s.rms_norm(&a, None, EPS).unwrap();
            let (got, _, _) = residual(s, &a, &b, None);
            assert_ne!(
                bits(&s.download(&alone).unwrap()),
                got[1],
                "K {k}: `b` reaches the norm"
            );
            for t in [a, x, b, gamma, alone] {
                s.free(t).unwrap();
            }
        }
    });
}

/// A residual norm captured with the product before it, and replayed over new `x`, equals the
/// ops run fresh.
#[test]
fn a_captured_residual_norm_replays_as_the_ops_run_fresh() {
    with_session(|s| {
        let k = 2048;
        let w = s.upload(&values(4, 64 * k, false), 64, k).unwrap();
        let a = row(s, &values(9, k, true));
        let gamma = s.upload(&values(8, k, false), 1, k).unwrap();
        let x = s.upload(&values(10, 64, false), 1, 64).unwrap();
        let step = |s: &mut Session<_>| {
            let b = s
                .matmul_dram(
                    &x,
                    false,
                    &w,
                    false,
                    tt_kernels::matmul::SrcRoute::Tf32FromFp32,
                    tt_kernels::matmul::Fidelity::HiFi4,
                    tt_tests::harness::BUDGET,
                )
                .unwrap();
            let (sum, n) = s.add_rms_norm(&a, &b, Some(&gamma), EPS).unwrap();
            s.free(b).unwrap();
            (sum, n)
        };
        let mut fresh = Vec::new();
        for seed in [10u64, 20, 21] {
            s.write(&x, &values(seed, 64, false)).unwrap();
            let (sum, n) = step(s);
            fresh.push([&sum, &n].map(|t| bits(&s.download(t).unwrap())));
            s.free(sum).unwrap();
            s.free(n).unwrap();
        }
        s.write(&x, &values(10, 64, false)).unwrap();
        s.begin_trace().unwrap();
        let (sum, n) = step(s);
        let id = s.end_trace().unwrap();
        let mut got = vec![[&sum, &n].map(|t| bits(&s.download(t).unwrap()))];
        for seed in [20u64, 21] {
            s.write(&x, &values(seed, 64, false)).unwrap();
            s.replay(id).unwrap();
            s.sync().unwrap();
            got.push([&sum, &n].map(|t| bits(&s.download(t).unwrap())));
        }
        assert_eq!(got, fresh, "the capture and its replays");
        s.release_trace(id).unwrap();
        for t in [sum, n, w, a, gamma, x] {
            s.free(t).unwrap();
        }
    });
}
