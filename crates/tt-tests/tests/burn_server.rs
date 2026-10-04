//! The device server, with a host engine standing in for hardware: an F32
//! matmul reaches the engine attached to its device, batches and broadcasts
//! the way Flex does, and attaching follows the one-owner rule.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use burn_flex::{Flex, FlexDevice};
use burn_tensor::{Tensor, TensorData};
use burn_tt::{attach, Engine, EngineError, TtBackend, TtDevice};

/// Computes on the host, in `k` order, and counts its calls.
struct HostEngine {
    calls: Arc<AtomicUsize>,
}

impl Engine for HostEngine {
    fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        [m, k, n]: [usize; 3],
    ) -> Result<Vec<f32>, EngineError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok((0..m * n)
            .map(|x| (0..k).map(|q| a[x / n * k + q] * b[q * n + x % n]).sum())
            .collect())
    }
}

fn host(device: TtDevice) -> (burn_tt::AttachGuard, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let c = calls.clone();
    let guard = attach(device, move |serve| {
        serve.serve(&mut HostEngine { calls: c });
        Ok(())
    })
    .unwrap();
    (guard, calls)
}

/// Small integers, so every summation order agrees and Flex is the answer.
fn ints(shape: &[usize], seed: usize) -> TensorData {
    let n: usize = shape.iter().product();
    TensorData::new(
        (0..n)
            .map(|i| ((i * 7 + seed) % 9) as f32 - 4.0)
            .collect::<Vec<f32>>(),
        shape.to_vec(),
    )
}

fn check<const D: usize>(a: &[usize], b: &[usize], runs: usize, device: TtDevice) {
    let (_guard, calls) = host(device);
    let want = Tensor::<Flex, D>::from_data(ints(a, 1), &FlexDevice)
        .matmul(Tensor::<Flex, D>::from_data(ints(b, 2), &FlexDevice))
        .into_data();
    let got = Tensor::<TtBackend, D>::from_data(ints(a, 1), &device)
        .matmul(Tensor::<TtBackend, D>::from_data(ints(b, 2), &device))
        .into_data();
    assert_eq!(want.shape, got.shape, "{a:?} @ {b:?}");
    assert_eq!(want.as_bytes(), got.as_bytes(), "{a:?} @ {b:?}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        runs,
        "{a:?} @ {b:?}: engine runs"
    );
}

#[test]
fn a_2d_matmul_is_one_engine_run() {
    check::<2>(&[5, 7], &[7, 3], 1, TtDevice::new(100));
}

#[test]
fn full_reductions_reject_an_engine_without_native_reduction_support() {
    let device = TtDevice::new(120);
    let (_guard, _) = host(device);
    let input = Tensor::<TtBackend, 1>::from_data([1.0, 2.0], &device);
    for mean in [false, true] {
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if mean {
                input.clone().mean()
            } else {
                input.clone().sum()
            }
        }))
        .expect_err("a native full reduction needs device storage");
        let message = failure.downcast_ref::<String>().unwrap();
        assert!(message.contains("native full reduction is unsupported by this engine"));
    }
}

/// A real batch on the right is one run per element; a right side with no
/// batch (a `Linear`'s weight, unsqueezed) folds the left's batch into its
/// rows and is one run.
#[test]
fn batches_are_one_run_each_and_broadcast() {
    check::<3>(&[4, 5, 7], &[4, 7, 3], 4, TtDevice::new(101));
    check::<3>(&[4, 5, 7], &[1, 7, 3], 1, TtDevice::new(102));
    check::<4>(&[2, 1, 5, 7], &[1, 3, 7, 2], 6, TtDevice::new(103));
}

/// A transposed operand is a strided view in Flex; the device path must read
/// it in logical order.
#[test]
fn a_transposed_view_is_read_in_logical_order() {
    let device = TtDevice::new(104);
    let (_guard, _) = host(device);
    let b = || ints(&[3, 7], 5);
    let want = Tensor::<Flex, 2>::from_data(ints(&[5, 7], 1), &FlexDevice)
        .matmul(Tensor::<Flex, 2>::from_data(b(), &FlexDevice).transpose())
        .into_data();
    let got = Tensor::<TtBackend, 2>::from_data(ints(&[5, 7], 1), &device)
        .matmul(Tensor::<TtBackend, 2>::from_data(b(), &device).transpose())
        .into_data();
    assert_eq!(want.as_bytes(), got.as_bytes());
}

#[test]
fn a_device_attached_twice_is_refused() {
    let device = TtDevice::new(105);
    let (_guard, _) = host(device);
    let again = attach(device, |serve| {
        serve.serve(&mut HostEngine {
            calls: Arc::new(AtomicUsize::new(0)),
        });
        Ok(())
    });
    assert!(again.is_err());
}

#[test]
fn a_factory_error_is_attachs_error_and_leaves_the_device_free() {
    let device = TtDevice::new(106);
    let err = attach(device, |_serve| Err(EngineError("no such card".into())));
    assert_eq!(err.err(), Some(EngineError("no such card".into())));
    // The failed attach must not leave the device claimed.
    let (_guard, _) = host(device);
}

#[test]
#[should_panic(expected = "is not attached")]
fn a_detached_device_is_not_attached() {
    let device = TtDevice::new(107);
    let (guard, _) = host(device);
    drop(guard);
    let a = Tensor::<TtBackend, 2>::ones([2, 2], &device);
    let _ = a.clone().matmul(a).into_data();
}

#[test]
fn an_engine_error_panics_with_it() {
    struct Failing;
    impl Engine for Failing {
        fn matmul(&mut self, _: &[f32], _: &[f32], _: [usize; 3]) -> Result<Vec<f32>, EngineError> {
            Err(EngineError("the tile hung".into()))
        }
    }
    let device = TtDevice::new(108);
    let _guard = attach(device, |serve| {
        serve.serve(&mut Failing);
        Ok(())
    })
    .unwrap();
    let r = std::panic::catch_unwind(|| {
        let a = Tensor::<TtBackend, 2>::ones([2, 2], &device);
        let _ = a.clone().matmul(a).into_data();
    });
    let msg = r.unwrap_err();
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("the tile hung"), "{msg}");
}

/// `burn::nn::Linear` calls `ModuleOps::linear`, a default over `float_matmul`
/// that generated dispatch leaves to Burn. With no bias, this minimal engine
/// needs only matmul. Biased Linears are covered by the native model gates.
#[test]
fn a_linear_layer_reaches_the_engine() {
    let device = TtDevice::new(109);
    let (_guard, calls) = host(device);
    let x = Tensor::<TtBackend, 2>::from_data(ints(&[6, 5], 1), &device);
    let w = Tensor::<TtBackend, 2>::from_data(ints(&[5, 4], 2), &device);
    let got = burn_tensor::module::linear(x, w, None).into_data();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "linear must run its matmul on the engine"
    );
    let want = burn_tensor::module::linear(
        Tensor::<Flex, 2>::from_data(ints(&[6, 5], 1), &FlexDevice),
        Tensor::<Flex, 2>::from_data(ints(&[5, 4], 2), &FlexDevice),
        None,
    )
    .into_data();
    assert_eq!(want.as_bytes(), got.as_bytes());
}

/// Keeps tensors on the host; every device matmul fails.
#[derive(Default)]
struct FailingMatmul {
    next: burn_tt::BufferId,
    live: std::collections::HashMap<burn_tt::BufferId, Vec<f32>>,
}

impl Engine for FailingMatmul {
    fn matmul(&mut self, _: &[f32], _: &[f32], _: [usize; 3]) -> Result<Vec<f32>, EngineError> {
        Err(EngineError("no host-staged matmul here".into()))
    }
    fn supports_dram(&self) -> bool {
        true
    }
    fn upload(
        &mut self,
        v: &[f32],
        _r: usize,
        _c: usize,
    ) -> Result<burn_tt::BufferId, EngineError> {
        self.next += 1;
        self.live.insert(self.next, v.to_vec());
        Ok(self.next)
    }
    fn download(&mut self, id: burn_tt::BufferId) -> Result<Vec<f32>, EngineError> {
        self.live
            .get(&id)
            .cloned()
            .ok_or_else(|| EngineError(format!("no buffer {id}")))
    }
    fn free(&mut self, id: burn_tt::BufferId) {
        self.live.remove(&id);
    }
    fn matmul_dram(
        &mut self,
        _: burn_tt::BufferId,
        _: bool,
        _: burn_tt::BufferId,
        _: bool,
    ) -> Result<(burn_tt::BufferId, [usize; 2]), EngineError> {
        Err(EngineError("the tile hung".into()))
    }
}

fn panic_text(r: std::thread::Result<()>) -> String {
    let e = r.unwrap_err();
    e.downcast_ref::<String>().cloned().unwrap_or_default()
}

/// Asynchronous dispatch (B8): a device op returns before it runs, so its
/// failure is reported where its result is first waited for -- naming the op
/// and the engine's error.
#[test]
fn an_asynchronous_op_s_failure_is_reported_at_the_wait() {
    let device = TtDevice::new(110);
    let _guard = attach(device, |serve| {
        serve.serve(&mut FailingMatmul::default());
        Ok(())
    })
    .unwrap();
    let a = Tensor::<TtBackend, 2>::ones([2, 2], &device).to_device(&device);
    // Returns: the failure is not known yet.
    let c = a.clone().matmul(a);
    let msg = panic_text(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || {
            let _ = c.into_data();
        },
    )));
    assert!(
        msg.contains("matmul") && msg.contains("the tile hung"),
        "{msg}"
    );
}

/// A failure whose result nobody reads is not lost: the attachment's next
/// wait reports it, whatever that wait reads.
#[test]
fn a_failure_nobody_reads_is_reported_by_the_next_wait() {
    let device = TtDevice::new(111);
    let _guard = attach(device, |serve| {
        serve.serve(&mut FailingMatmul::default());
        Ok(())
    })
    .unwrap();
    let a = Tensor::<TtBackend, 2>::ones([2, 2], &device).to_device(&device);
    drop(a.clone().matmul(a.clone()));
    let b = a.mul_scalar(1.0);
    let msg = panic_text(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || {
            let _ = b.into_data();
        },
    )));
    assert!(
        msg.contains("an earlier device op failed") && msg.contains("the tile hung"),
        "{msg}"
    );
}
