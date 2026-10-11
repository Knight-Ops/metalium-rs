//! B2 gate: a device nobody attached is attached by the first op that needs it.
//!
//! Burn code does not call `burn_tt::attach`: `TtDevice::default()` is enough,
//! as `CudaDevice::default()` is. The registry opens whatever
//! `set_default_engine` chose -- here the simulator, which `burn-tt` never
//! depends on -- the first time an op needs the device, and the result is
//! checked against burn-flex and for residency: the product is computed on the
//! device, not staged through the host.
//!
//! Negative controls: a registry that refuses attaches nothing (the same code
//! panics "is not attached", so the passing run really did attach one), and a
//! parent process that attached a *host* engine before forking does not leave
//! the child a dead server thread to wait on.

// The simulator is what this gate registers; silicon has `silicon_lazy_attach`.
#![cfg(not(feature = "silicon"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use burn_flex::{Flex, FlexDevice};
use burn_tensor::{Tensor, TensorData};
use burn_tt::{
    detach, refuse_default_engine, reset_default_engine, set_default_engine, Engine, EngineFactory,
    Serve, TtBackend, TtDevice, TtError,
};
use tt_tests::burn_device::{register_simulator, Config};

/// Small integers, so the TF32 product is exact and Flex is the answer.
fn ints(shape: &[usize], seed: usize) -> TensorData {
    let n: usize = shape.iter().product();
    TensorData::new(
        (0..n)
            .map(|i| ((i * 7 + seed) % 9) as f32 - 4.0)
            .collect::<Vec<f32>>(),
        shape.to_vec(),
    )
}

/// The registry and the attachment table are the process's: one test at a
/// time, so no fork happens while another test is attaching.
fn one_at_a_time() -> MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    let turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    reset_default_engine();
    turn
}

const SHAPES: ([usize; 2], [usize; 2]) = ([32, 64], [64, 32]);

fn flex_product() -> TensorData {
    let a = Tensor::<Flex, 2>::from_data(ints(&SHAPES.0, 1), &FlexDevice);
    let b = Tensor::<Flex, 2>::from_data(ints(&SHAPES.1, 2), &FlexDevice);
    a.matmul(b).into_data()
}

/// `A @ B` on `device`, without any `attach`, and whether it ran on the card.
fn product(device: &TtDevice) -> (TensorData, bool) {
    let a = Tensor::<TtBackend, 2>::from_data(ints(&SHAPES.0, 1), device);
    let b = Tensor::<TtBackend, 2>::from_data(ints(&SHAPES.1, 2), device);
    let c = a.matmul(b);
    let resident = match c.clone().into_primitive() {
        burn_tensor::TensorPrimitive::Float(t) => t.computed_on_device(),
        _ => panic!("unexpected quantized tensor"),
    };
    (c.into_data(), resident)
}

#[test]
fn the_first_op_attaches_the_simulator_and_matches_flex() {
    let _turn = one_at_a_time();
    tt_ttsim::fork_scope(|| {
        register_simulator(Config::default());
        // A device number with no `/dev/tenstorrent` node, so `enumerate` lists
        // it only once something is attached.
        let device = TtDevice::new(77);
        assert!(!TtDevice::enumerate().contains(&device));
        let (got, resident) = product(&device);
        assert!(resident, "the product must be computed on the device");
        assert_eq!(got.as_bytes(), flex_product().as_bytes());
        assert!(TtDevice::enumerate().contains(&device));
        // Re-attaching is the host engine's test (`burn_server`): the simulator is a
        // once-per-process singleton with no re-init path.
        detach(device).unwrap();
        assert!(!TtDevice::enumerate().contains(&device));
    })
    .unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn a_registry_that_refuses_attaches_nothing() {
    let _turn = one_at_a_time();
    tt_ttsim::fork_scope(|| {
        refuse_default_engine();
        let device = TtDevice::new(78);
        let failure = std::panic::catch_unwind(|| product(&device)).expect_err("nothing serves it");
        let message = failure
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_default();
        assert!(message.contains("is not attached"), "{message}");
        assert!(!TtDevice::enumerate().contains(&device));
    })
    .unwrap_or_else(|e| panic!("{e}"));
}

/// Only the matmul, host-staged: enough to attach and compute in the parent.
struct Host(Arc<AtomicUsize>);

impl Engine for Host {
    fn matmul(&mut self, a: &[f32], b: &[f32], [m, k, n]: [usize; 3]) -> Result<Vec<f32>, TtError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok((0..m * n)
            .map(|x| (0..k).map(|q| a[x / n * k + q] * b[q * n + x % n]).sum())
            .collect())
    }
}

#[test]
fn a_child_forked_from_an_attached_parent_attaches_its_own_engine() {
    let _turn = one_at_a_time();
    // The parent attaches a host engine on first use. Its server thread does
    // not exist in the forked child, though the attachment table does.
    let ran = Arc::new(AtomicUsize::new(0));
    let counted = ran.clone();
    set_default_engine(move |_, _| {
        let counted = counted.clone();
        let factory: EngineFactory = Box::new(move |serve: Serve| {
            serve.serve(&mut Host(counted));
            Ok(())
        });
        Ok(factory)
    });
    let device = TtDevice::new(0);
    let a = Tensor::<TtBackend, 2>::from_data(ints(&[2, 2], 1), &device);
    let _ = a.clone().matmul(a).into_data();
    assert_eq!(
        ran.load(Ordering::SeqCst),
        1,
        "the parent attached a host engine"
    );

    tt_ttsim::fork_scope(|| {
        register_simulator(Config::default());
        let (got, resident) = product(&device);
        assert!(resident, "the child computed on its own, simulator engine");
        assert_eq!(got.as_bytes(), flex_product().as_bytes());
    })
    .unwrap_or_else(|e| panic!("{e}"));

    // The child's work did not touch the parent's engine.
    assert_eq!(ran.load(Ordering::SeqCst), 1);
    detach(device).unwrap();
    reset_default_engine();
}
