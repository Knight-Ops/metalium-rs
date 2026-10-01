//! A device tensor must not outlive its attachment silently.
//!
//! An engine numbers its buffers itself, from 1 (`burn_tt::DramBuffers`, which
//! both hardware engines use), and a tensor remembers only its device and that
//! number. Detach the device and attach it again and the new engine hands out
//! the same numbers, so a tensor kept from the first attachment names whatever
//! the second put there: reading it returns another tensor's data, and
//! dropping it frees another tensor's buffer.
//!
//! The claim is the weakest a fix must meet: a stale tensor may be refused
//! (a panic, or later a typed error), but it must never read, or free, a
//! buffer that is not its own.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use burn_tensor::{Tensor, TensorData};
use burn_tt::{attach, AttachGuard, BufferId, Engine, EngineError, TtBackend, TtDevice};
use tt_isa::dm::kind;

/// Keeps tensors on the host, numbered from 1 per engine as `DramBuffers`
/// numbers them; only `MUL_SCALAR` computes.
#[derive(Default)]
struct HostDram {
    next: BufferId,
    live: HashMap<BufferId, (Vec<f32>, usize, usize)>,
}

impl HostDram {
    fn insert(&mut self, v: Vec<f32>, r: usize, c: usize) -> BufferId {
        self.next += 1;
        self.live.insert(self.next, (v, r, c));
        self.next
    }
    fn get(&self, id: BufferId) -> Result<&(Vec<f32>, usize, usize), EngineError> {
        self.live
            .get(&id)
            .ok_or_else(|| EngineError(format!("no buffer {id}")))
    }
}

impl Engine for HostDram {
    fn matmul(&mut self, _: &[f32], _: &[f32], _: [usize; 3]) -> Result<Vec<f32>, EngineError> {
        Err(EngineError("not used".into()))
    }
    fn supports_dram(&self) -> bool {
        true
    }
    fn upload(&mut self, v: &[f32], r: usize, c: usize) -> Result<BufferId, EngineError> {
        Ok(self.insert(v.to_vec(), r, c))
    }
    fn download(&mut self, id: BufferId) -> Result<Vec<f32>, EngineError> {
        Ok(self.get(id)?.0.clone())
    }
    fn free(&mut self, id: BufferId) {
        self.live.remove(&id);
    }
    fn eltwise(
        &mut self,
        k: u32,
        scalar: f32,
        a: BufferId,
        b: Option<BufferId>,
    ) -> Result<(BufferId, [usize; 2]), EngineError> {
        if k != kind::MUL_SCALAR || b.is_some() {
            return Err(EngineError(format!("kind {k} not modelled")));
        }
        let (v, r, c) = self.get(a)?.clone();
        let out = v.iter().map(|x| x * scalar).collect();
        Ok((self.insert(out, r, c), [r, c]))
    }
}

fn host_dram(device: TtDevice) -> AttachGuard {
    attach(device, |serve| {
        serve.serve(&mut HostDram::default());
        Ok(())
    })
    .unwrap()
}

/// A device-only tensor: uploaded, then scaled on the device, so the result
/// has no host copy to hide behind.
fn on_device(values: [f32; 4], scale: f32, device: &TtDevice) -> Tensor<TtBackend, 2> {
    Tensor::<TtBackend, 2>::from_data(TensorData::new(values.to_vec(), [2, 2]), device)
        .to_device(device)
        .mul_scalar(scale)
}

fn values(t: Tensor<TtBackend, 2>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

/// In attachment 1, `stale` is buffer 2. In attachment 2, `other` is buffer 2
/// too. Reading `stale` must not return `other`'s data.
#[test]
#[ignore = "known bug: BufferIds restart per attachment (checklist Phase 9); un-ignore with the fix"]
fn a_tensor_from_a_previous_attachment_does_not_read_another_buffer() {
    let device = TtDevice::new(120);
    let first = host_dram(device);
    let stale = on_device([1.0, 2.0, 3.0, 4.0], 2.0, &device);
    drop(first);

    let _second = host_dram(device);
    let other = on_device([10.0, 20.0, 30.0, 40.0], 3.0, &device);

    let read = catch_unwind(AssertUnwindSafe(|| values(stale)));
    if let Ok(got) = read {
        assert_eq!(
            got,
            [2.0, 4.0, 6.0, 8.0],
            "a tensor from the first attachment read a buffer of the second"
        );
    }
    assert_eq!(values(other), [30.0, 60.0, 90.0, 120.0]);
}

/// Dropping `stale` after the re-attach must not free `other`'s buffer.
#[test]
#[ignore = "known bug: BufferIds restart per attachment (checklist Phase 9); un-ignore with the fix"]
fn dropping_a_stale_tensor_does_not_free_another_buffer() {
    let device = TtDevice::new(121);
    let first = host_dram(device);
    let stale = on_device([1.0, 2.0, 3.0, 4.0], 2.0, &device);
    drop(first);

    let _second = host_dram(device);
    let other = on_device([10.0, 20.0, 30.0, 40.0], 3.0, &device);
    drop(stale);

    let read = catch_unwind(AssertUnwindSafe(|| values(other)));
    assert!(
        matches!(&read, Ok(v) if v == &[30.0, 60.0, 90.0, 120.0]),
        "dropping a tensor from the first attachment freed a buffer of the second: {}",
        match read {
            Ok(v) => format!("read {v:?}"),
            Err(e) => e
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "panicked".into()),
        }
    );
}
