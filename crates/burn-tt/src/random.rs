//! Independent per-device random streams.
//!
//! A device with resident GDDR draws **on the device** (lane T6, S7): each call
//! takes the next `base` of the device's seeded sequence and runs the seeded
//! tile kernel (`tt_kernels::prng`), so a draw is a pure function of
//! `(seed, call number, tile)` and costs no upload. Only a device with no
//! resident buffers -- unattached, or an engine without GDDR -- builds the
//! tensor on the host from the per-device `StdRng`, exactly like `from_data`:
//! the result is host-resident construction data that the device never
//! computes, so no device arithmetic is ever replaced by the host (a draw on an
//! attached GDDR device that fails is an error, never a host retry).
use crate::TtDevice;
use burn_backend::{Distribution, Shape, TensorData};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use tt_kernels::prng::splitmix64;

struct Stream {
    seed: u64,
    /// Device draws made since the seed.
    draws: u64,
    /// The host-construction generator (devices without GDDR).
    host: StdRng,
}

impl Stream {
    fn new(seed: u64) -> Self {
        Stream {
            seed,
            draws: 0,
            host: StdRng::seed_from_u64(seed),
        }
    }
}

fn streams() -> &'static Mutex<HashMap<TtDevice, Stream>> {
    static STREAMS: OnceLock<Mutex<HashMap<TtDevice, Stream>>> = OnceLock::new();
    STREAMS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn seed(device: TtDevice, seed: u64) {
    streams()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(device, Stream::new(seed));
}

/// The 64-bit base of the device's next draw: `splitmix64` of the seed and the
/// number of draws since it was set.
pub(crate) fn next_base(device: TtDevice) -> u64 {
    let mut streams = streams().lock().unwrap_or_else(|e| e.into_inner());
    let stream = streams.entry(device).or_insert_with(|| Stream::new(0));
    stream.draws += 1;
    splitmix64(stream.seed ^ splitmix64(stream.draws))
}

pub(crate) fn float(device: TtDevice, shape: Shape, distribution: Distribution) -> TensorData {
    let mut streams = streams().lock().unwrap_or_else(|e| e.into_inner());
    let rng = &mut streams.entry(device).or_insert_with(|| Stream::new(0)).host;
    TensorData::random::<f32, _, _>(shape, distribution, rng)
}

pub(crate) fn int(device: TtDevice, shape: Shape, distribution: Distribution) -> TensorData {
    let mut streams = streams().lock().unwrap_or_else(|e| e.into_inner());
    let rng = &mut streams.entry(device).or_insert_with(|| Stream::new(0)).host;
    TensorData::random::<i32, _, _>(shape, distribution, rng)
}

/// What a device draw computes: Burn's four distributions, their parameters
/// as the device takes them (F32, `i32`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Draw {
    /// `Distribution::Default` for floats: `[0, 1)`.
    Unit,
    Uniform(f32, f32),
    Bernoulli {
        p: f32,
        int: bool,
    },
    Normal {
        mean: f32,
        std: f32,
        int: bool,
    },
    /// `Distribution::Uniform` for integers: `[lo, hi)`.
    IntRange(i32, i32),
    /// `Distribution::Default` for integers: every `i32` equally likely.
    IntWords,
}

/// A float draw of `distribution` (F32 parameters; the conversion rounds to
/// nearest, as `f64 as f32`).
pub(crate) fn float_draw(distribution: Distribution) -> Draw {
    match distribution {
        Distribution::Default => Draw::Unit,
        Distribution::Uniform(lo, hi) => Draw::Uniform(lo as f32, hi as f32),
        Distribution::Bernoulli(p) => Draw::Bernoulli {
            p: p as f32,
            int: false,
        },
        Distribution::Normal(mean, std) => Draw::Normal {
            mean: mean as f32,
            std: std as f32,
            int: false,
        },
    }
}

/// An integer draw: Burn converts the parameters with `f64 as i32`
/// (saturating) and truncates a normal toward zero.
pub(crate) fn int_draw(distribution: Distribution) -> Draw {
    match distribution {
        Distribution::Default => Draw::IntWords,
        Distribution::Uniform(lo, hi) => Draw::IntRange(lo as i32, hi as i32),
        Distribution::Bernoulli(p) => Draw::Bernoulli {
            p: p as f32,
            int: true,
        },
        Distribution::Normal(mean, std) => Draw::Normal {
            mean: mean as f32,
            std: std as f32,
            int: true,
        },
    }
}
