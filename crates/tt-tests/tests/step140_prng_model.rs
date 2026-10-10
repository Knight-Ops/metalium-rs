//! S7, step 140: the host model of the hardware PRNG stream, and the
//! pinned statistical claims made on it.
//!
//! * `lane_initialisation_matches_the_model` is the only device arm: it restarts
//!   the PRNG with the RISC-V store procedure (`prng_seed` firmware, step91's)
//!   for eight seeds and compares **all 32 lanes** with
//!   `tt_kernels::prng::initial_state`. On ttsim it holds for `Target::Ttsim`
//!   (`advance^(96 - 2 lane)(seed)`). On silicon it is the PENDING MEASUREMENT
//!   of the model's `98 - 2 lane` (established for seed 0, lanes 0 and 1 only):
//!   it prints `T6-MEASURE` lines first, so a mismatch is a measurement, not a
//!   mystery.
//! * everything else is host-only: the distribution gates run on the model's
//!   stream, for both targets (a ttsim device draw equals the `Ttsim` model bit
//!   for bit -- `step141`).
//!
//! Negative controls (each watched to fail, recorded in the close-out record):
//! the raw LFSR words fail the pair tests; a generator with its low bits
//! masked fails the low-bit chi-square; a weaker mixer fails the adjacent-lane
//! pair test.

mod prng_support;

use prng_support::*;
use tt_kernels::prng::*;

const TILES: usize = 1200;
const BASE: u64 = 0x7e57_0140;

/// `TILES` tiles of unit uniforms in tile-image order, for `target`.
fn units(target: Target, role: u32) -> Vec<Vec<f32>> {
    (0..TILES)
        .map(|t| {
            tile_units(tile_seed(BASE, t as u64, role), target)
                .into_iter()
                .map(f32::from_bits)
                .collect()
        })
        .collect()
}

/// Disjoint pairs: adjacent lanes of one slot, adjacent slots of one lane, and
/// lanes sixteen apart.
fn pair_families(tiles: &[Vec<f32>]) -> [Vec<(f32, f32)>; 3] {
    let at = |t: &Vec<f32>, slot, lane| t[image_index(slot, lane)];
    let mut lane = Vec::new();
    let mut slot = Vec::new();
    let mut far = Vec::new();
    for t in tiles {
        for s in 0..GROUPS {
            for l in (0..LANES).step_by(2) {
                lane.push((at(t, s, l), at(t, s, l + 1)));
            }
            for l in 0..16 {
                far.push((at(t, s, l), at(t, s, l + 16)));
            }
        }
        for s in (0..GROUPS).step_by(2) {
            for l in 0..LANES {
                slot.push((at(t, s, l), at(t, s + 1, l)));
            }
        }
    }
    [lane, slot, far]
}

fn flat(tiles: &[Vec<f32>]) -> Vec<f32> {
    tiles.iter().flatten().copied().collect()
}

/// Every p-value of the generator's tests, named.
fn p_values(tiles: &[Vec<f32>]) -> Vec<(String, f64)> {
    let all = flat(tiles);
    let mut out = vec![
        (
            "top 8 bits, 256 bins".to_string(),
            binned_p(&all, 256, false),
        ),
        (
            "low 8 bits, 256 bins".to_string(),
            binned_p(&all, 256, true),
        ),
        ("low 4 bits, 16 bins".to_string(), binned_p(&all, 16, true)),
    ];
    let n = all.len() as f64;
    // The exact moments of k / 2^23, k uniform on [0, 2^23).
    let (mean, var) = (0.5 - 1.0 / 33_554_432.0, (1.0 - 2f64.powi(-46)) / 12.0);
    let (m, v) = mean_var(&all.iter().map(|&x| f64::from(x)).collect::<Vec<_>>());
    out.push((
        "mean z".into(),
        2.0 * (1.0 - normal_cdf(z_score(m, mean, (var / n).sqrt()).abs())),
    ));
    let sd_var = ((1.0 / 80.0 - 1.0 / 144.0) / n).sqrt();
    out.push((
        "variance z".into(),
        2.0 * (1.0 - normal_cdf(z_score(v, var, sd_var).abs())),
    ));
    let mut sample: Vec<f64> = all.iter().map(|&x| f64::from(x)).collect();
    out.push((
        "KS vs U[0,1)".into(),
        ks_p(&mut sample, |x| x.clamp(0.0, 1.0)),
    ));
    for (name, pairs) in ["lane+1", "slot+1", "lane+16"]
        .into_iter()
        .zip(pair_families(tiles))
    {
        out.push((format!("pairs {name}, 16x16"), pair_p(&pairs, 16)));
    }
    out
}

fn assert_passes(what: &str, tiles: &[Vec<f32>]) {
    for (name, p) in p_values(tiles) {
        println!("T6-STAT {what} {name}: p = {p:.3e}");
        assert!(p >= ALPHA, "{what}: {name} fails, p = {p:e} < {ALPHA:e}");
    }
}

#[test]
fn uniform_statistics_pass_on_both_target_streams() {
    for (name, target) in [("ttsim", Target::Ttsim), ("silicon", Target::Silicon)] {
        assert_passes(name, &units(target, role::UNIT));
    }
}

/// The unmixed LFSR words, as a generator: adjacent lanes are shifts of each
/// other (`lane[i + 1] = lane[i] << 2 | ..`), adjacent reads halvings.
fn raw_units(target: Target) -> Vec<Vec<f32>> {
    (0..TILES)
        .map(|t| {
            let raw = raw_words(tile_seed(BASE, t as u64, role::UNIT), target);
            let mut image = vec![0u32; 1024];
            for (slot, row) in raw.iter().enumerate() {
                for (lane, &w) in row.iter().enumerate() {
                    image[image_index(slot, lane)] = unit_bits(w);
                }
            }
            image.into_iter().map(f32::from_bits).collect()
        })
        .collect()
}

#[test]
fn negative_control_the_raw_lfsr_words_fail() {
    let tiles = raw_units(Target::Ttsim);
    let failing: Vec<_> = p_values(&tiles)
        .into_iter()
        .filter(|(_, p)| *p < ALPHA)
        .map(|(n, _)| n)
        .collect();
    println!("T6-STAT raw words fail: {failing:?}");
    assert!(
        failing.iter().any(|n| n.starts_with("pairs lane+1")),
        "the unmixed adjacent-lane shift structure must be visible: {failing:?}"
    );
}

#[test]
fn negative_control_masked_low_bits_fail_the_low_bit_chi_square() {
    // The same stream with the 8 low bits of each 23-bit uniform cleared.
    let masked: Vec<Vec<f32>> = units(Target::Ttsim, role::UNIT)
        .into_iter()
        .map(|t| {
            t.into_iter()
                .map(|x| {
                    let k = (f64::from(x) * 8_388_608.0) as u32 & !0xff;
                    k as f32 / 8_388_608.0
                })
                .collect()
        })
        .collect();
    let all = flat(&masked);
    let low = binned_p(&all, 256, false);
    let low_bits = binned_p(&all, 256, true);
    println!("T6-STAT masked: top p = {low:.3e}, low-bit p = {low_bits:.3e}");
    assert!(low >= ALPHA, "the top bits are untouched");
    assert!(low_bits < ALPHA, "masked low bits must fail the chi-square");
}

/// A weaker mixer (three of the five rounds): still a bijection, but the
/// adjacent-lane structure survives. Shown on the same pair test.
#[test]
fn negative_control_a_weak_mixer_fails_the_adjacent_lane_pairs() {
    let weak = |x: u32| {
        let mut x = x;
        x = x.wrapping_add(x << 10);
        x ^= x >> 6;
        x = x.wrapping_add(x << 3);
        x ^= x >> 11;
        x.wrapping_add(x << 15)
    };
    let target = Target::Ttsim;
    let tiles: Vec<Vec<f32>> = (0..TILES)
        .map(|t| {
            let raw = raw_words(tile_seed(BASE, t as u64, role::UNIT), target);
            let mut image = vec![0u32; 1024];
            for (slot, row) in raw.iter().enumerate() {
                for (lane, &w) in row.iter().enumerate() {
                    image[image_index(slot, lane)] = unit_bits(weak(w));
                }
            }
            image.into_iter().map(f32::from_bits).collect()
        })
        .collect();
    let p = pair_p(&pair_families(&tiles)[0], 16);
    println!("T6-STAT weak mixer adjacent-lane pairs: p = {p:.3e}");
    assert!(p < ALPHA, "the weak mixer must fail: p = {p:e}");
}

// ---------------------------------------------------------------------------
// Distributions built on the unit uniform (the host oracle for `step141`).
// ---------------------------------------------------------------------------

#[test]
fn bernoulli_frequencies_are_ceil_p_over_two_to_the_23() {
    let all = flat(&units(Target::Ttsim, role::BERNOULLI));
    let n = all.len() as f64;
    // Exactly representable and awkward probabilities, and the ends.
    for p in [
        0.0f32,
        1.0,
        0.5,
        0.1,
        0.9,
        1.0 / 3.0,
        1e-3,
        0.999,
        1.0 - 2f32.powi(-24),
    ] {
        let ones = all.iter().filter(|&&u| u < p).count() as f64;
        let exact = (f64::from(p) * 8_388_608.0).ceil() / 8_388_608.0;
        let sd = (n * exact * (1.0 - exact)).sqrt();
        println!(
            "T6-STAT bernoulli p = {p}: ones {ones}, expected {:.1}, sd {sd:.1}",
            n * exact
        );
        if sd == 0.0 {
            assert_eq!(ones, n * exact, "p = {p} is exact");
        } else {
            let z = z_score(ones, n * exact, sd);
            assert!(z.abs() < Z_ALPHA, "p = {p}: z = {z}");
        }
        // The stated bias: the probability of one is within 2^-23 above p.
        assert!(exact >= f64::from(p) && exact - f64::from(p) < 1.0 / 8_388_608.0);
    }
}

#[test]
fn uniform_affine_map_stays_in_the_half_open_interval_within_its_bound() {
    let all = flat(&units(Target::Ttsim, role::UNIFORM));
    for (lo, hi) in [
        (-3.5f32, 7.25),
        (0.0, 1.0),
        (1e6, 1e6 + 64.0),
        (-1e-3, 1e-3),
        (2.0, 2.000_001),
    ] {
        let bound = uniform_bound(lo, hi);
        let scale = hi - lo;
        let mut worst = 0.0f64;
        for &u in &all {
            // The device's order: fl(fl(u s) + lo), then the clamp below hi.
            let r = ((u * scale) + lo).min(next_down(hi));
            assert!(r >= lo && r < hi, "[{lo}, {hi}): {r}");
            let real = f64::from(lo) + f64::from(u) * (f64::from(hi) - f64::from(lo));
            worst = worst.max((f64::from(r) - real).abs());
        }
        println!("T6-STAT uniform [{lo}, {hi}): worst error {worst:e}, bound {bound:e}");
        assert!(worst <= bound, "[{lo}, {hi}): {worst:e} > {bound:e}");
    }
}

/// `z = sqrt(-2 ln(1 - u1)) cos(2 pi u2)` in f64 from the model's two draws.
fn box_muller(u1: f32, u2: f32) -> f64 {
    let x = 1.0 - f64::from(u1);
    (-2.0 * x.ln()).sqrt() * (2.0 * std::f64::consts::PI * f64::from(u2)).cos()
}

#[test]
fn box_muller_over_the_model_stream_is_normal() {
    let u1 = flat(&units(Target::Ttsim, role::NORMAL_RADIUS));
    let u2 = flat(&units(Target::Ttsim, role::NORMAL_ANGLE));
    let mut z: Vec<f64> = u1
        .iter()
        .zip(&u2)
        .map(|(&a, &b)| box_muller(a, b))
        .collect();
    let n = z.len() as f64;
    let (mean, var) = mean_var(&z);
    // N(0, 1) truncated at sqrt(-2 ln 2^-23) = 5.64 sigma: the lost tail mass
    // 1.7e-8 moves the variance by under 1e-6, far below sd(var) = sqrt(2/n).
    let zm = z_score(mean, 0.0, (1.0 / n).sqrt());
    let zv = z_score(var, 1.0, (2.0 / n).sqrt());
    let m4 = z.iter().map(|v| v.powi(4)).sum::<f64>() / n;
    // Kurtosis 3, variance of the sample fourth moment (96 / n for a normal).
    let zk = z_score(m4, 3.0, (96.0 / n).sqrt());
    let p_ks = ks_p(&mut z, normal_cdf);
    println!("T6-STAT normal: mean z {zm:.2}, var z {zv:.2}, m4 z {zk:.2}, KS p {p_ks:.3e}");
    assert!(zm.abs() < Z_ALPHA && zv.abs() < Z_ALPHA && zk.abs() < Z_ALPHA);
    assert!(p_ks >= ALPHA);
    let max = z.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    assert!(max < 5.65, "the tail is cut at 5.64 sigma: {max}");
}

#[test]
fn integer_range_counts_are_floor_or_ceil_of_two_to_the_23_over_r() {
    // The map is deterministic in k = 2^23 u, so the bias is exactly countable:
    // `floor(k R / 2^23)` by exact integer arithmetic splits the 2^23 values of
    // k among R results as `floor` or `ceil` of `2^23 / R`, a relative spread of
    // at most `R / 2^23` (`int_range_bias`). The device computes it in F32
    // (`floor(fl(u R))`); the F32 product can round up across an integer, which
    // is counted here, never assumed away.
    for range in [1u64, 2, 3, 10, 256, 1000, 65_536, 1_000_003, 1 << 23] {
        let mut exact = vec![0u64; range as usize];
        let mut f32s = vec![0u64; range as usize];
        let mut crossings = 0u64;
        let r = range as f32;
        for k in 0..(1u64 << 23) {
            let e = ((k * range) >> 23) as usize;
            let u = k as f32 / 8_388_608.0;
            let v = (u * r) as u64;
            assert!(v < range, "R = {range}: k = {k} gave {v}");
            exact[e] += 1;
            f32s[v as usize] += 1;
            crossings += u64::from(v as usize != e);
        }
        let spread = |c: &[u64]| *c.iter().max().unwrap() - *c.iter().min().unwrap();
        let mean = (1u64 << 23) as f64 / range as f64;
        println!(
            "T6-STAT int range {range}: exact spread {}, f32 spread {}, f32 crossings {crossings}, mean {mean:.3}, bound {:.3e}",
            spread(&exact),
            spread(&f32s),
            int_range_bias(range)
        );
        assert!(spread(&exact) <= 1, "exact counts are floor or ceil");
        assert!(spread(&exact) as f64 / mean <= int_range_bias(range) * (1.0 + 1e-12));
        // Each crossing moves one count by one, up and down.
        assert!(spread(&f32s) <= 1 + 2 * crossings);
    }
}

// ---------------------------------------------------------------------------
// The device arm: the lane initialisation the model rests on.
// ---------------------------------------------------------------------------

/// Restart the PRNG by the RISC-V store procedure for eight seeds and compare
/// all 32 lanes with the model (`step91`'s probe, widened to every lane).
#[test]
fn lane_initialisation_matches_the_model() {
    use tt_device::{tlb::WindowKind, Transport};
    use tt_isa::{backend, mailbox, tensix::Core};
    use tt_kernels::{
        datapath,
        sfpu::{Format, LReg, Program},
    };
    use tt_tests::harness;
    harness::in_device(|dev| {
        let tile = harness::tensix_tile();
        let target = Target::from_simulated(dev.transport().is_simulated());
        let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let mut code = datapath::thread_config();
        let mut p = Program::new();
        p.read_prng(LReg::L0);
        p.store(LReg::L0, Format::Int32, 0);
        code.extend(p.finish());
        code.push(backend::wait_for_sfpu(backend::Before::EVERYTHING).unwrap());
        for seed in [
            0u32,
            1,
            0x1234_5678,
            0x8000_0000,
            0x5555_5555,
            0xdead_beef,
            0x0000_ffff,
            // Finish with a nonabsorbing stream for later programs on this tile.
            0xaaaa_aaaa,
        ] {
            dev.release_tensix_backend(&w, tile).unwrap();
            let d = mailbox::Descriptor {
                thread_index: 1,
                program_len: code.len() as u32,
                ..Default::default()
            };
            for (at, value) in d.writes(mailbox::role::Mailbox::single_core()) {
                dev.write32(&w, tile, at, value).unwrap();
            }
            dev.write32(&w, tile, mailbox::OPERAND_A, seed).unwrap();
            dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
            dev.write(&w, tile, mailbox::PROGRAM, &harness::program_bytes(&code))
                .unwrap();
            dev.load_and_start(
                &w,
                tile,
                Core::T1,
                tt_firmware_images::PRNG_SEED,
                tt_firmware_images::LOAD_ADDRESS,
            )
            .unwrap();
            dev.wait_for_status(&w, tile, 400_000, |s| s == mailbox::status::DONE)
                .unwrap()
                .unwrap();
            // Phase 0 is the first restart: each lane's first read.
            let lanes: Vec<u32> = (0..LANES)
                .map(|lane| {
                    let row = lane as u32 / 8;
                    let col = lane as u32 % 8 * 2;
                    dev.read32(&w, tile, mailbox::dump_offset(row, col))
                        .unwrap()
                })
                .collect();
            let model: Vec<u32> = (0..LANES).map(|l| initial_state(seed, target, l)).collect();
            println!("T6-MEASURE seed={seed:08x} target={target:?} lanes={lanes:08x?}");
            println!("T6-MEASURE seed={seed:08x} model  ={model:08x?}");
            assert_eq!(lanes, model, "seed {seed:#x}: lanes differ from the model");
        }
    });
}
