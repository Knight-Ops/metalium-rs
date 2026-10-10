//! Lane T6 (S7), step 141: seeded random on the device -- the tile kernel and
//! the distributions built on it, against the host model of `step140`.
//!
//! Oracles (independent of the programs): `tt_kernels::prng`'s pure-Rust model
//! (`tile_words`/`tile_units`, fitted to ttsim's lane initialisation in
//! `step140`), f32/f64 host arithmetic for the affine maps, and an f64
//! Box-Muller for the normal, each with the bound derived beside its gate.
//!
//! On ttsim the draw equals `Target::Ttsim`'s model bit for bit. On silicon it
//! must equal `Target::Silicon`'s, whose lane constant is PENDING MEASUREMENT
//! (`step140::lane_initialisation_matches_the_model`, run first).
//!
//! Negative control (watched to fail, recorded in the lane report): remove the
//! seed directive (`Kernel::seeded` pushing no directive) and a tile that follows
//! another on the same core draws the stale stream, so `every_tile_restarts`
//! fails against the model.

mod prng_support;

use prng_support::{expected, inverse, normal_bound, units_of};

use tt_kernels::prng::*;
use tt_kernels::session::{Session, TileChoice};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn open() -> Session<tt_ttsim::LibTtsim<'static>> {
    let sim = Box::leak(Box::new(tt_ttsim::Simulator::open().unwrap()));
    let dev = tt_device::Device::open(sim.transport()).unwrap();
    Session::open(
        dev,
        tt_firmware_images::ROLES,
        TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        |_, _| Ok(None),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

#[cfg(feature = "silicon")]
fn open() -> Session<tt_kmd::Kmd> {
    Session::open_card(
        tt_tests::backend::device_index(),
        tt_firmware_images::ROLES,
        TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn with_session<T: tt_device::Transport>(
    open: impl FnOnce() -> Session<T>,
    f: impl FnOnce(&mut Session<T>),
) {
    if let Err(e) = fork_scope(|| {
        let mut s = open();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// The minimal silicon probe: one tile through the role firmware's seed
/// directive. Prints the first row so a mismatch is a measurement.
#[test]
fn one_tile_probe() {
    with_session(open, |s| {
        let target = s.prng_target();
        let t = s
            .random_tiles([32, 32], Output::Word, 0x1410, role::INT_WORD)
            .unwrap();
        let got = s.download_bits(&t).unwrap();
        let want = expected([32, 32], 0x1410, role::INT_WORD, target, false);
        println!("T6-MEASURE one tile {target:?} row0 {:08x?}", &got[..32]);
        println!("T6-MEASURE one tile {target:?} want {:08x?}", &want[..32]);
        assert_eq!(got, want);
    });
}

/// Enough tiles to churn the program cache (one distinct math program each).
#[test]
fn many_tiles_equal_the_model() {
    with_session(open, |s| {
        let target = s.prng_target();
        let dims = [256, 224]; // 8 x 7 tiles
        let t = s
            .random_tiles(dims, Output::Unit, 0x1411, role::UNIT)
            .unwrap();
        assert_eq!(
            s.download_bits(&t).unwrap(),
            expected(dims, 0x1411, role::UNIT, target, true)
        );
    });
}

#[test]
fn unit_tiles_equal_the_model_bit_for_bit_padding_included() {
    with_session(open, |s| {
        let target = s.prng_target();
        let dims = [70, 50]; // 3 x 2 tiles, ragged on both edges
        let t = s
            .random_tiles(dims, Output::Unit, 0x141, role::UNIT)
            .unwrap();
        assert_eq!(
            t.pad(),
            tt_kernels::tensor::Pad::Undefined,
            "random padding is declared Undefined"
        );
        let got = s.download_bits(&t).unwrap();
        let want = expected(dims, 0x141, role::UNIT, target, true);
        assert_eq!(got, want, "{target:?} logical elements");
        // The whole padded matrix: every tile is the model's tile.
        let padded: Vec<u32> = s
            .download_padded(&t)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let inv = inverse();
        for r in 0..96 {
            for c in 0..64 {
                let tile = (r / 32) * 2 + c / 32;
                let model = tile_units(tile_seed(0x141, tile as u64, role::UNIT), target);
                assert_eq!(padded[r * 64 + c], model[inv[r % 32][c % 32]], "({r}, {c})");
            }
        }
        // The same draw again is the same bits; another base is not.
        let again = s
            .random_tiles(dims, Output::Unit, 0x141, role::UNIT)
            .unwrap();
        assert_eq!(
            s.download_bits(&again).unwrap(),
            got,
            "same seed, same bits"
        );
        let other = s
            .random_tiles(dims, Output::Unit, 0x142, role::UNIT)
            .unwrap();
        let other_bits = s.download_bits(&other).unwrap();
        assert_ne!(other_bits, got);
        assert_eq!(
            other_bits,
            expected(dims, 0x142, role::UNIT, target, true),
            "and it is the model's for that base"
        );
    });
}

/// Every tile after the first on a core restarts the stream: the device has one
/// core here, so tile `n` runs straight after tile `n - 1`. Without the restart
/// procedure it would continue the previous tile's stream.
#[test]
fn every_tile_restarts_the_stream() {
    with_session(open, |s| {
        let target = s.prng_target();
        let dims = [64, 160]; // 10 tiles through one core
        let t = s
            .random_tiles(dims, Output::Word, 0x1414, role::INT_WORD)
            .unwrap();
        let got = s.download_bits(&t).unwrap();
        let want = expected(dims, 0x1414, role::INT_WORD, target, false);
        let first_bad = got.iter().zip(&want).position(|(a, b)| a != b);
        assert_eq!(
            first_bad, None,
            "element {first_bad:?} differs from the model"
        );
    });
}

#[test]
fn bernoulli_uniform_and_integer_ranges_equal_their_host_emulation() {
    with_session(open, |s| {
        let target = s.prng_target();
        let dims = [40, 70];
        // Bernoulli: `u < p`, as F32 and as I32.
        for p in [0.0f32, 0.3, 0.5, 1.0] {
            let u = units_of(&expected(dims, 7, role::BERNOULLI, target, true));
            let f = s.random_bernoulli(dims, 7, p, false).unwrap();
            let got = s.download(&f).unwrap();
            for (i, (&g, &x)) in got.iter().zip(&u).enumerate() {
                assert_eq!(g, f32::from(u8::from(x < p)), "p = {p}, element {i}");
            }
            let n = s.random_bernoulli(dims, 7, p, true).unwrap();
            let got = s.download_bits(&n).unwrap();
            for (i, (&g, &x)) in got.iter().zip(&u).enumerate() {
                assert_eq!(g, u32::from(x < p), "I32 p = {p}, element {i}");
            }
        }
        // Uniform [lo, hi): the device's order of f32 operations, bit for bit,
        // inside the half-open interval and within `uniform_bound` of the map.
        for (lo, hi) in [(-3.5f32, 7.25), (0.0, 1.0), (100.0, 100.5)] {
            let u = units_of(&expected(dims, 9, role::UNIFORM, target, true));
            let t = s.random_uniform(dims, 9, lo, hi).unwrap();
            let got = s.download(&t).unwrap();
            let bound = uniform_bound(lo, hi);
            for (i, (&g, &x)) in got.iter().zip(&u).enumerate() {
                let host = ((x * (hi - lo)) + lo).min(next_down(hi));
                assert_eq!(g.to_bits(), host.to_bits(), "[{lo}, {hi}) element {i}");
                assert!(g >= lo && g < hi);
                let real = f64::from(lo) + f64::from(x) * (f64::from(hi) - f64::from(lo));
                assert!((f64::from(g) - real).abs() <= bound);
            }
        }
        // Integers [lo, hi): `lo + trunc(u R)`.
        for (lo, hi) in [(0i32, 10), (-7, 1000), (5, 6), (i32::MIN, i32::MIN + 4096)] {
            let u = units_of(&expected(dims, 11, role::INT_RANGE, target, true));
            let t = s.random_int_range(dims, 11, lo, hi).unwrap();
            let got = s.download_bits(&t).unwrap();
            let range = (i64::from(hi) - i64::from(lo)) as f32;
            for (i, (&g, &x)) in got.iter().zip(&u).enumerate() {
                let host = lo.wrapping_add((x * range) as i32);
                assert_eq!(g as i32, host, "[{lo}, {hi}) element {i}");
                assert!(g as i32 >= lo && (g as i32) < hi);
            }
        }
        // The raw 32-bit words are Burn's default `i32` distribution.
        let w = s.random_int_words(dims, 13).unwrap();
        assert_eq!(
            s.download_bits(&w).unwrap(),
            expected(dims, 13, role::INT_WORD, target, false)
        );
        // Unsupported parameters fail with a message, never fall back.
        assert!(s.random_bernoulli(dims, 1, 1.5, false).is_err());
        assert!(s.random_bernoulli(dims, 1, f32::NAN, false).is_err());
        assert!(s.random_uniform(dims, 1, 1.0, 1.0).is_err());
        assert!(s.random_uniform(dims, 1, 0.0, f32::INFINITY).is_err());
        assert!(s.random_int_range(dims, 1, 0, 0).is_err());
        assert!(s.random_int_range(dims, 1, 0, (1 << 23) + 1).is_err());
        assert!(s.random_normal(dims, 1, 0.0, -1.0).is_err());
    });
}

#[test]
fn normal_is_within_the_derived_bound_of_an_f64_box_muller() {
    with_session(open, |s| {
        let target = s.prng_target();
        let dims = [64, 96];
        for (mean, std) in [(0.0f32, 1.0f32), (2.0, 3.0), (-1.5, 0.25)] {
            let u1 = units_of(&expected(dims, 21, role::NORMAL_RADIUS, target, true));
            let u2 = units_of(&expected(dims, 21, role::NORMAL_ANGLE, target, true));
            let t = s.random_normal(dims, 21, mean, std).unwrap();
            let got = s.download(&t).unwrap();
            let mut worst = 0.0f64;
            for (i, ((&g, &a), &b)) in got.iter().zip(&u1).zip(&u2).enumerate() {
                let x = 1.0 - f64::from(a);
                let theta = f64::from(b * TWO_PI); // the F32 the device cosines
                let z_ref = (-2.0 * x.ln()).sqrt() * theta.cos();
                let out_ref = f64::from(mean) + f64::from(std) * z_ref;
                let bound = normal_bound(z_ref, out_ref, f64::from(std));
                let err = (f64::from(g) - out_ref).abs();
                worst = worst.max(err / bound);
                assert!(
                    err <= bound,
                    "mean {mean} std {std} element {i}: u1 {a} u2 {b}: got {g}, want {out_ref}, err {err:e} > {bound:e}"
                );
            }
            println!("T6-STAT normal mean {mean} std {std}: worst error / bound = {worst:.3}");
        }
    });
}
