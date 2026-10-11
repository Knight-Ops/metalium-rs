//! The device server, with a host engine standing in for hardware: an F32
//! matmul reaches the engine attached to its device, batches and broadcasts
//! the way Flex does, and attaching follows the one-owner rule.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use burn_flex::{Flex, FlexDevice};
use burn_tensor::backend::Backend;
use burn_tensor::{Tensor, TensorData};
use burn_tt::{
    attach, detach, init, refuse_default_engine, reset_default_engine, set_default_engine, Engine,
    EngineFactory, Serve, TtBackend, TtConfig, TtDevice, TtError,
};

/// Computes on the host, in `k` order, and counts its calls.
struct HostEngine {
    calls: Arc<AtomicUsize>,
}

impl Engine for HostEngine {
    fn matmul(&mut self, a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) -> Result<Vec<f32>, TtError> {
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
    let err = attach(device, |_serve| Err(TtError::Device("no such card".into())));
    assert_eq!(err.err(), Some(TtError::Device("no such card".into())));
    // The failed attach must not leave the device claimed.
    let (_guard, _) = host(device);
}

/// A host-staged op has no `Result` to return in Burn's signatures, so its
/// failure is a panic in the op -- the one case that is not deferred to a wait.
#[test]
fn a_host_staged_engine_error_panics_in_the_op_with_it() {
    struct Failing;
    impl Engine for Failing {
        fn matmul(&mut self, _: &[f32], _: &[f32], _: [usize; 3]) -> Result<Vec<f32>, TtError> {
            Err(TtError::Device("the tile hung".into()))
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
    /// Panic in `matmul_dram` instead of returning an error.
    panics: bool,
    next: burn_tt::BufferId,
    live: std::collections::HashMap<burn_tt::BufferId, Vec<f32>>,
}

impl Engine for FailingMatmul {
    fn matmul(&mut self, _: &[f32], _: &[f32], _: [usize; 3]) -> Result<Vec<f32>, TtError> {
        Err(TtError::Device("no host-staged matmul here".into()))
    }
    fn supports_dram(&self) -> bool {
        true
    }
    fn upload(&mut self, v: &[f32], _r: usize, _c: usize) -> Result<burn_tt::BufferId, TtError> {
        self.next += 1;
        self.live.insert(self.next, v.to_vec());
        Ok(self.next)
    }
    fn download(&mut self, id: burn_tt::BufferId) -> Result<Vec<f32>, TtError> {
        self.live
            .get(&id)
            .cloned()
            .ok_or_else(|| TtError::Device(format!("no buffer {id}")))
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
    ) -> Result<(burn_tt::BufferId, [usize; 2]), TtError> {
        if self.panics {
            panic!("the engine blew up");
        }
        Err(TtError::Device("the tile hung".into()))
    }
}

fn attach_failing(device: TtDevice, panics: bool) -> burn_tt::AttachGuard {
    attach(device, move |serve| {
        serve.serve(&mut FailingMatmul {
            panics,
            ..Default::default()
        });
        Ok(())
    })
    .unwrap()
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

/// B1: an asynchronous failure is the `ExecutionError` of `try_into_data`,
/// which is what Burn documents it for; the caller decides whether to panic.
#[test]
fn an_engine_error_is_try_into_data_s_error() {
    let device = TtDevice::new(112);
    let _guard = attach_failing(device, false);
    let a = Tensor::<TtBackend, 2>::ones([2, 2], &device).to_device(&device);
    let c = a.clone().matmul(a);
    let err = c.try_into_data().expect_err("the matmul failed");
    let text = err.to_string();
    assert!(
        text.contains("matmul") && text.contains("the tile hung"),
        "{text}"
    );
}

/// Strict mode keeps the panic, at the call that reads the failure.
#[test]
fn strict_mode_panics_where_try_into_data_would_return_the_error() {
    let device = TtDevice::new(113);
    let _guard = attach_failing(device, false);
    let a = Tensor::<TtBackend, 2>::ones([2, 2], &device).to_device(&device);
    let c = a.clone().matmul(a);
    let msg = panic_text(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || {
            let _ = burn_tt::strictly(|| c.try_into_data());
        },
    )));
    assert!(msg.contains("the tile hung"), "{msg}");
}

/// `sync` is the barrier Burn documents: the first failure since the device
/// started, if any, and nothing for a device that has nothing to wait for.
#[test]
fn sync_reports_a_failure_nobody_read() {
    let healthy = TtDevice::new(114);
    let (_guard, _) = host(healthy);
    assert!(TtBackend::sync(&healthy).is_ok());
    // No engine at all: nothing to wait for, and no attach.
    assert!(TtBackend::sync(&TtDevice::new(60_000)).is_ok());

    let failing = TtDevice::new(115);
    let _guard = attach_failing(failing, false);
    let a = Tensor::<TtBackend, 2>::ones([2, 2], &failing).to_device(&failing);
    assert!(TtBackend::sync(&failing).is_ok(), "nothing has failed yet");
    drop(a.clone().matmul(a));
    let err = TtBackend::sync(&failing).expect_err("the dropped matmul failed");
    assert!(err.to_string().contains("the tile hung"), "{err}");
}

/// An engine that panics ends its own service, not the process: the panic's
/// message comes back from every later wait instead of "the thread stopped".
#[test]
fn a_panicking_engine_is_reported_by_every_later_wait() {
    let device = TtDevice::new(116);
    let _guard = attach_failing(device, true);
    let a = Tensor::<TtBackend, 2>::ones([2, 2], &device).to_device(&device);
    let c = a.clone().matmul(a.clone());
    for read in [c.clone(), a.clone().mul_scalar(1.0)] {
        let text = read
            .try_into_data()
            .expect_err("the server panicked")
            .to_string();
        assert!(
            text.contains("panicked") && text.contains("the engine blew up"),
            "{text}"
        );
    }
    let err = TtBackend::sync(&device).expect_err("the server panicked");
    assert!(err.to_string().contains("the engine blew up"), "{err}");
}

// --- B2: the registry attaches a device nobody attached -------------------------

/// The registry's default engine is process-wide: tests that set it take turns,
/// and put silicon back when they finish (or panic).
struct Registry(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

fn registry() -> Registry {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    reset_default_engine();
    Registry(turn)
}

impl Drop for Registry {
    fn drop(&mut self) {
        reset_default_engine();
    }
}

/// A registry engine that serves a `HostEngine`, counting how often the
/// registry asked for a factory.
fn host_registry(
    asked: &Arc<AtomicUsize>,
) -> impl Fn(TtDevice, &TtConfig) -> Result<EngineFactory, TtError> + Send + Sync + 'static {
    let asked = asked.clone();
    move |_, _| {
        asked.fetch_add(1, Ordering::SeqCst);
        let factory: EngineFactory = Box::new(|serve: Serve| {
            serve.serve(&mut HostEngine {
                calls: Arc::new(AtomicUsize::new(0)),
            });
            Ok(())
        });
        Ok(factory)
    }
}

fn square_matmul(device: &TtDevice) -> TensorData {
    let a = Tensor::<TtBackend, 2>::from_data(ints(&[3, 3], 1), device);
    a.clone().matmul(a).into_data()
}

fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let e = std::panic::catch_unwind(f).expect_err("expected a panic");
    e.downcast_ref::<String>().cloned().unwrap_or_default()
}

#[test]
fn an_op_on_an_unattached_device_attaches_it_through_the_registry() {
    let _registry = registry();
    let asked = Arc::new(AtomicUsize::new(0));
    set_default_engine(host_registry(&asked));
    let device = TtDevice::new(130);
    // No `attach`: the first op opens the device, and the answer is Flex's.
    let want = Tensor::<Flex, 2>::from_data(ints(&[3, 3], 1), &FlexDevice);
    let want = want.clone().matmul(want).into_data();
    assert_eq!(square_matmul(&device).as_bytes(), want.as_bytes());
    assert_eq!(square_matmul(&device).as_bytes(), want.as_bytes());
    assert_eq!(
        asked.load(Ordering::SeqCst),
        1,
        "one factory call, not one per op"
    );
    assert!(TtDevice::enumerate().contains(&device));
    detach(device).unwrap();
    assert!(!TtDevice::enumerate().contains(&device));
}

#[test]
fn a_failed_attach_fails_the_same_way_again_without_opening_the_card_again() {
    let _registry = registry();
    let asked = Arc::new(AtomicUsize::new(0));
    let counted = asked.clone();
    set_default_engine(move |_, _| {
        counted.fetch_add(1, Ordering::SeqCst);
        Err(TtError::NoSuchCard {
            card: 131,
            present: vec![0, 1],
        })
    });
    let device = TtDevice::new(131);
    let first = panic_message(|| drop(square_matmul(&device)));
    let second = panic_message(|| drop(square_matmul(&device)));
    assert!(
        first.contains("no card 131") && first.contains("cards present: [0, 1]"),
        "{first}"
    );
    assert_eq!(first, second);
    assert_eq!(asked.load(Ordering::SeqCst), 1);
    // `init` is the retry: it clears the failure and asks the registry again.
    let ok = Arc::new(AtomicUsize::new(0));
    set_default_engine(host_registry(&ok));
    init(device, TtConfig::default()).unwrap();
    assert_eq!(ok.load(Ordering::SeqCst), 1);
    assert_eq!(square_matmul(&device).shape.to_vec(), vec![3, 3]);
    detach(device).unwrap();
}

#[test]
fn init_is_idempotent_and_refuses_another_configuration() {
    let _registry = registry();
    let asked = Arc::new(AtomicUsize::new(0));
    set_default_engine(host_registry(&asked));
    let device = TtDevice::new(132);
    init(device, TtConfig::default()).unwrap();
    init(device, TtConfig::default()).unwrap();
    assert_eq!(asked.load(Ordering::SeqCst), 1);
    let other = TtConfig::default().tiles(burn_tt::TileChoice::Count(2));
    assert_eq!(
        init(device, other).unwrap_err(),
        TtError::AlreadyAttached { device, auto: true }
    );
    // An explicit attach of a device the registry attached names `init`/`detach`.
    let err = attach(device, |_| Ok(())).err().expect("refused");
    assert!(err.to_string().contains("burn_tt::detach"), "{err}");
    detach(device).unwrap();
    let (_guard, _) = host(device);
    // And the other way: the registry leaves an explicit attachment alone.
    assert_eq!(
        init(device, TtConfig::default()).unwrap_err(),
        TtError::AlreadyAttached {
            device,
            auto: false
        }
    );
}

#[test]
#[should_panic(expected = "is not attached")]
fn a_registry_that_refuses_leaves_the_device_unattached() {
    let _registry = registry();
    refuse_default_engine();
    let _ = square_matmul(&TtDevice::new(133));
}

#[test]
fn a_device_with_no_card_is_named_and_what_is_present_is_listed() {
    let _registry = registry();
    // The silicon default, on a node that cannot exist: nothing is opened.
    let device = TtDevice::new(0x7FFF);
    let first = panic_message(|| drop(square_matmul(&device)));
    assert!(
        first.contains("/dev/tenstorrent"),
        "the message says what is missing: {first}"
    );
    assert_eq!(first, panic_message(|| drop(square_matmul(&device))));
}

#[test]
fn device_count_counts_nodes_and_attachments_without_opening_anything() {
    let device = TtDevice::new(134);
    assert!(!TtDevice::enumerate().contains(&device));
    assert!(
        burn_tt::device_traffic(device).is_none(),
        "asking does not attach"
    );
    assert!(!TtDevice::enumerate().contains(&device));
    let (_guard, _) = host(device);
    assert!(TtDevice::enumerate().contains(&device));
    // Other tests attach and detach meanwhile, so only a bound is stable.
    assert!(TtBackend::device_count(0) >= 1);
    assert_eq!(TtBackend::device_count(7), 0, "every device is type 0");
}
