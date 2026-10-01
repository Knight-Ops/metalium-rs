//! How much tensor data has crossed between host and device, process-wide.
//!
//! The Phase 9 claim is that a steady-state training step moves almost none;
//! this is what a gate asserts it with. Counted where `burn-tt` moves tensor
//! data (uploads and downloads), not at the transport, so it says which tensors
//! fell back to the host rather than how many bytes of descriptors a kernel
//! needed -- `tt_device::Device::traffic` says that.

use std::sync::atomic::{AtomicU64, Ordering};

static UP: AtomicU64 = AtomicU64::new(0);
static DOWN: AtomicU64 = AtomicU64::new(0);
static UPLOADS: AtomicU64 = AtomicU64::new(0);
static DOWNLOADS: AtomicU64 = AtomicU64::new(0);

/// Tensor bytes moved so far, and how many moves.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorTraffic {
    pub uploaded: u64,
    pub downloaded: u64,
    pub uploads: u64,
    pub downloads: u64,
}

impl std::ops::Sub for TensorTraffic {
    type Output = TensorTraffic;
    fn sub(self, o: TensorTraffic) -> TensorTraffic {
        TensorTraffic {
            uploaded: self.uploaded - o.uploaded,
            downloaded: self.downloaded - o.downloaded,
            uploads: self.uploads - o.uploads,
            downloads: self.downloads - o.downloads,
        }
    }
}

/// Everything so far.
pub fn tensor_traffic() -> TensorTraffic {
    TensorTraffic {
        uploaded: UP.load(Ordering::Relaxed),
        downloaded: DOWN.load(Ordering::Relaxed),
        uploads: UPLOADS.load(Ordering::Relaxed),
        downloads: DOWNLOADS.load(Ordering::Relaxed),
    }
}

/// Which way a tensor crossed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    Up,
    Down,
}

/// One tensor crossing PCIe: which way, and its `[rows, cols]` as stored on
/// the device (a transposed view downloads as its buffer).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Transfer {
    pub direction: Direction,
    pub shape: [usize; 2],
}

/// The transfer log, while one is being kept.
static LOG: std::sync::Mutex<Option<Vec<Transfer>>> = std::sync::Mutex::new(None);

/// Start (`true`) or stop keeping a log of every transfer, for a gate that
/// wants to say *which* tensors crossed rather than how many bytes did.
/// Off by default, so a long run does not grow it.
pub fn record_transfers(on: bool) {
    *LOG.lock().unwrap_or_else(|p| p.into_inner()) = on.then(Vec::new);
}

/// The transfers logged since the last call (empty if no log is kept).
pub fn take_transfers() -> Vec<Transfer> {
    LOG.lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_mut()
        .map(std::mem::take)
        .unwrap_or_default()
}

fn log(direction: Direction, rows: usize, cols: usize) {
    if let Some(l) = LOG.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
        l.push(Transfer {
            direction,
            shape: [rows, cols],
        });
    }
}

pub(crate) fn uploaded(rows: usize, cols: usize) {
    UP.fetch_add((rows * cols * 4) as u64, Ordering::Relaxed);
    UPLOADS.fetch_add(1, Ordering::Relaxed);
    log(Direction::Up, rows, cols);
}

pub(crate) fn downloaded(rows: usize, cols: usize) {
    DOWN.fetch_add((rows * cols * 4) as u64, Ordering::Relaxed);
    DOWNLOADS.fetch_add(1, Ordering::Relaxed);
    log(Direction::Down, rows, cols);
}

/// Wall-clock time spent in device calls, by kind: where a training step's
/// time goes. Measured on the calling thread around each server round trip.
pub fn device_time() -> Vec<(&'static str, u64, std::time::Duration)> {
    TIMES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .map(|(k, (n, d))| (*k, *n, *d))
        .collect()
}

static TIMES: std::sync::Mutex<
    std::collections::BTreeMap<&'static str, (u64, std::time::Duration)>,
> = std::sync::Mutex::new(std::collections::BTreeMap::new());

pub(crate) fn timed<R>(kind: &'static str, f: impl FnOnce() -> R) -> R {
    let t0 = std::time::Instant::now();
    let r = f();
    let d = t0.elapsed();
    let mut t = TIMES.lock().unwrap_or_else(|p| p.into_inner());
    let e = t.entry(kind).or_default();
    e.0 += 1;
    e.1 += d;
    r
}
