//! The delegation is inert: every op but the device ones gives Flex's answer,
//! byte for byte. Host only -- no device is attached, and none is needed,
//! because nothing here reaches `float_matmul` on F32.
//!
//! Watched failing: with `float_add` in `generated/delegate.rs` forwarded to
//! `float_sub`, `elementwise_and_scalar_ops` fails and nothing else does.

use burn_flex::{Flex, FlexDevice};
use burn_tensor::{activation, backend::Backend, Bool, Distribution, Int, Tensor, TensorData};
use burn_tt::{TtBackend, TtDevice};

fn data<B: Backend, const D: usize>(t: Tensor<B, D>) -> TensorData {
    t.into_data()
}

fn floats() -> TensorData {
    TensorData::new(
        (0..48)
            .map(|i| (i as f32 - 20.5) * 0.37)
            .collect::<Vec<f32>>(),
        [4, 3, 4],
    )
}

/// Both backends, the same op battery, the same bytes.
macro_rules! both {
    ($name:ident, |$t:ident, $d:ident| $body:expr) => {
        #[test]
        fn $name() {
            fn battery<B: Backend>($d: &B::Device) -> Vec<TensorData> {
                let $t = Tensor::<B, 3>::from_data(floats(), $d);
                $body
            }
            let host = battery::<Flex>(&FlexDevice);
            let tt = battery::<TtBackend>(&TtDevice::default());
            assert_eq!(host.len(), tt.len());
            for (i, (h, t)) in host.iter().zip(&tt).enumerate() {
                assert_eq!(h.dtype, t.dtype, "result {i}: dtype");
                assert_eq!(h.shape, t.shape, "result {i}: shape");
                assert_eq!(h.as_bytes(), t.as_bytes(), "result {i}: bytes");
            }
        }
    };
}

both!(elementwise_and_scalar_ops, |t, _d| {
    let u = t.clone().exp().div_scalar(3.0);
    vec![
        data(t.clone().add(u.clone())),
        data(t.clone().sub(u.clone())),
        data(t.clone().mul(u.clone())),
        data(t.clone().div(u.clone())),
        data(t.clone().powf_scalar(2.0)),
        data(t.clone().abs().sqrt()),
        data(t.clone().neg().tanh()),
        data(t.clone().clamp(-1.0, 1.0)),
        data(t.clone().add_scalar(1.5).log()),
    ]
});

both!(reductions_and_shape_ops, |t, _d| {
    vec![
        data(t.clone().sum()),
        data(t.clone().mean_dim(1)),
        data(t.clone().max_dim(2)),
        data(t.clone().reshape([12, 4])),
        data(t.clone().swap_dims(0, 2)),
        data(t.clone().slice([1..3, 0..2, 1..4])),
        data(Tensor::cat(vec![t.clone(), t.clone()], 1)),
        data(t.clone().argmax(2).float()),
        data(t.clone().expand([2, 4, 3, 4])),
    ]
});

both!(activations_and_losses, |t, _d| {
    vec![
        data(activation::relu(t.clone())),
        data(activation::softmax(t.clone(), 2)),
        data(activation::log_softmax(t.clone(), 2)),
        data(activation::sigmoid(t.clone())),
        data(activation::gelu(t.clone())),
    ]
});

both!(int_and_bool_ops, |t, _d| {
    let i: Tensor<_, 3, Int> = t.clone().int();
    let b: Tensor<_, 3, Bool> = t.clone().greater_elem(0.0);
    vec![
        data(i.clone().add_scalar(3).mul(i.clone()).float()),
        data(i.clone().sum_dim(1).float()),
        data(b.clone().int().float()),
        data(b.clone().bool_not().int().float()),
        data(t.clone().mask_fill(b, 7.0)),
        data(i.clone().abs().clamp(0, 4).one_hot::<4>(5).float()),
    ]
});

/// The random ops draw from Flex's generator on both backends, and
/// `TtBackend::seed` seeds it: the same seed, the same numbers.
#[test]
fn seeded_random_matches() {
    Flex::seed(&FlexDevice, 7);
    let h = Tensor::<Flex, 2>::random([8, 8], Distribution::Default, &FlexDevice).into_data();
    TtBackend::seed(&TtDevice::default(), 7);
    let t = Tensor::<TtBackend, 2>::random([8, 8], Distribution::Default, &TtDevice::default())
        .into_data();
    assert_eq!(h.as_bytes(), t.as_bytes());
}

/// Not every op is inert: an F32 matmul goes to the device, and with none
/// attached it says so instead of quietly running on the host.
#[test]
#[should_panic(expected = "is not attached")]
fn an_f32_matmul_needs_an_attached_device() {
    let d = TtDevice::new(9);
    let a = Tensor::<TtBackend, 2>::ones([4, 4], &d);
    let _ = a.clone().matmul(a).into_data();
}
