//! Phase 10 gate (X4d, Burn): `burn_tt::Trace` over a Burn forward pass.
//!
//! A two-layer MLP's forward pass -- `relu(x @ w1 + b1) @ w2 + b2`, as Burn's
//! `Linear` composes it -- captured on its input, then run on new inputs:
//! each run's output is the same Burn ops run fresh on that input, bit for
//! bit. And a capture whose closure falls back to the host is refused without
//! leaving the device capturing: the next capture works.

use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_tt::{Trace, TtBackend};
use tt_tests::burn_device::{with_device, Config};

type T2 = Tensor<TtBackend, 2>;

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn primitive(t: T2) -> burn_tt::TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("a float tensor"),
    }
}

const M: usize = 32;

#[test]
fn a_traced_forward_pass_is_the_fresh_one() {
    with_device(Config::default(), |device| {
        let t = |seed, r, c| -> T2 {
            Tensor::from_data(TensorData::new(values(seed, r * c), [r, c]), &device)
        };
        let (w1, b1, w2, b2) = (t(2, 784, 128), t(3, 1, 128), t(4, 128, 10), t(5, 1, 10));
        let forward = |x: T2| -> T2 {
            burn::tensor::activation::relu(x.matmul(w1.clone()) + b1.clone()).matmul(w2.clone())
                + b2.clone()
        };

        let x = t(1, M, 784);
        let xp = primitive(x.clone());
        let fresh = forward(x.clone()).into_data().to_vec::<f32>().unwrap();
        let (trace, first) = Trace::capture(&xp, || primitive(forward(x.clone()))).unwrap();
        assert_eq!(trace.output_dims(), [M, 10]);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&first), bits(&fresh), "the capture's own run");

        for round in 0..3u64 {
            let input = values(10 + round, M * 784);
            let got = trace.run(input.clone()).unwrap();
            let want = forward(Tensor::from_data(TensorData::new(input, [M, 784]), &device))
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(bits(&got), bits(&want), "run {round}");
        }
    });
}

#[test]
fn a_capture_that_falls_back_to_the_host_is_refused_and_ends() {
    with_device(Config::default(), |device| {
        let x: T2 = Tensor::from_data(TensorData::new(values(1, 32 * 32), [32, 32]), &device);
        let xp = primitive(x.clone());
        // `cumsum` has no device path: a download, which the capture refuses.
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Trace::capture(&xp, || primitive(x.clone().cumsum(1)))
        }));
        assert!(
            refused.is_err() || refused.as_ref().is_ok_and(|r| r.is_err()),
            "a host fallback was captured"
        );
        // Not left capturing: an ordinary capture works.
        let relu = || primitive(burn::tensor::activation::relu(x.clone()));
        let (trace, first) = Trace::capture(&xp, relu).unwrap();
        let again = trace.run(values(1, 32 * 32)).unwrap();
        assert_eq!(first, again);
    });
}
