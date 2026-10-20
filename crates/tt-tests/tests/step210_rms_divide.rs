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
    let y = s.rms_norm(x, EPS).unwrap();
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
            let n = s.rms_norm(&y, EPS).unwrap();
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
