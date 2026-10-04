//! Independent per-device random streams for tensor construction.
use crate::TtDevice;
use burn_backend::{Distribution, Shape, TensorData};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn streams() -> &'static Mutex<HashMap<TtDevice, StdRng>> {
    static STREAMS: OnceLock<Mutex<HashMap<TtDevice, StdRng>>> = OnceLock::new();
    STREAMS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn seed(device: TtDevice, seed: u64) {
    streams()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(device, StdRng::seed_from_u64(seed));
}

pub(crate) fn float(device: TtDevice, shape: Shape, distribution: Distribution) -> TensorData {
    let mut streams = streams().lock().unwrap_or_else(|e| e.into_inner());
    let rng = streams
        .entry(device)
        .or_insert_with(|| StdRng::seed_from_u64(0));
    TensorData::random::<f32, _, _>(shape, distribution, rng)
}

pub(crate) fn int(device: TtDevice, shape: Shape, distribution: Distribution) -> TensorData {
    let mut streams = streams().lock().unwrap_or_else(|e| e.into_inner());
    let rng = streams
        .entry(device)
        .or_insert_with(|| StdRng::seed_from_u64(0));
    TensorData::random::<i32, _, _>(shape, distribution, rng)
}
