//! Phase 10 gate (X4d, Burn): `burn_tt::Trace` over a Burn forward pass.
//!
//! A two-layer MLP's forward pass -- `relu(x @ w1 + b1) @ w2 + b2`, as Burn's
//! `Linear` composes it -- captured on its input, then run on new inputs:
//! each run's output is the same Burn ops run fresh on that input, bit for
//! bit. And a capture whose closure downloads to the host is refused without
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

fn primitive<const D: usize>(t: Tensor<TtBackend, D>) -> burn_tt::TtTensor {
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
fn a_capture_that_downloads_to_the_host_is_refused_and_ends() {
    with_device(Config::default(), |device| {
        let x: T2 = Tensor::from_data(TensorData::new(values(1, 32 * 32), [32, 32]), &device);
        let xp = primitive(x.clone());
        // Use an explicit download: cumsum now has a traceable native path.
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Trace::capture(&xp, || {
                let _ = x.clone().into_data();
                primitive(x.clone())
            })
        }));
        assert!(
            refused.is_err() || refused.as_ref().is_ok_and(|r| r.is_err()),
            "a host download was captured"
        );
        // Not left capturing: an ordinary capture works.
        let relu = || primitive(burn::tensor::activation::relu(x.clone()));
        let (trace, first) = Trace::capture(&xp, relu).unwrap();
        let again = trace.run(values(1, 32 * 32)).unwrap();
        assert_eq!(first, again);
    });
}

#[test]
fn generalized_inference_trace_multi_input_multi_output() {
    use burn_tt::{InputPayload, TracedInference};
    with_device(Config::default(), |device| {
        let t = |seed: u64, r: usize, c: usize| -> T2 {
            Tensor::from_data(TensorData::new(values(seed, r * c), [r, c]), &device)
        };
        let x1 = t(100, 32, 64);
        let x2 = t(200, 32, 64);
        let (p1, p2) = (primitive(x1.clone()), primitive(x2.clone()));

        let (trace, first) = TracedInference::capture(&[&p1, &p2], || {
            vec![
                primitive(x1.clone() + x2.clone()),
                primitive(x1.clone() - x2.clone()),
            ]
        })
        .unwrap();

        assert_eq!(first.len(), 2);
        assert_eq!(trace.output_shapes(), vec![[32, 64], [32, 64]]);

        // Replay with new inputs
        let new_x1 = values(300, 32 * 64);
        let new_x2 = values(400, 32 * 64);
        let out = trace
            .run(vec![
                InputPayload::F32(new_x1.clone()),
                InputPayload::F32(new_x2.clone()),
            ])
            .unwrap();

        assert_eq!(out.len(), 2);
        let t_new_x1: T2 = Tensor::from_data(TensorData::new(new_x1, [32, 64]), &device);
        let t_new_x2: T2 = Tensor::from_data(TensorData::new(new_x2, [32, 64]), &device);
        let exp1 = (t_new_x1.clone() + t_new_x2.clone())
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let exp2 = (t_new_x1 - t_new_x2).into_data().to_vec::<f32>().unwrap();

        assert_eq!(out[0].as_f32().unwrap(), exp1.as_slice());
        assert_eq!(out[1].as_f32().unwrap(), exp2.as_slice());
    });
}

#[test]
fn traced_training_step_updates_weights_in_place_and_reduces_loss() {
    use burn::backend::Autodiff;
    use burn::module::Param;
    use burn::nn::Linear;
    use burn::optim::{GradientsParams, Optimizer, SgdConfig};
    use burn_tt::{InputPayload, TracedTrainingStep};

    type AD = Autodiff<TtBackend>;

    with_device(Config::default(), |device| {
        let model = Linear::<AD> {
            weight: Param::from_tensor(Tensor::from_data([[0.5], [1.5]], &device)),
            bias: Some(Param::from_tensor(Tensor::from_data([0.0], &device))),
        };
        let mut optim = SgdConfig::new().init();

        let x_data = vec![1.0f32, 2.0]; // [1, 2]
        let y_target = vec![10.0f32]; // [1, 1] target: want output to approach 10.0

        let x = Tensor::<AD, 2>::from_data(TensorData::new(x_data.clone(), [1, 2]), &device);
        let y = Tensor::<AD, 2>::from_data(TensorData::new(y_target.clone(), [1, 1]), &device);

        let xp = primitive(x.clone().inner());
        let yp = primitive(y.clone().inner());

        let (traced_step, first_loss) = TracedTrainingStep::capture(&[&xp, &yp], || {
            let pred = model.forward(x.clone());
            let loss = (pred - y.clone()).square().mean();
            let grads = GradientsParams::from_grads(loss.clone().backward(), &model);
            let new_model = optim.step(0.05, model.clone(), grads);

            let updates = vec![
                (
                    primitive(new_model.weight.val().inner()),
                    primitive(model.weight.val().inner()),
                ),
                (
                    primitive(new_model.bias.as_ref().unwrap().val().inner()),
                    primitive(model.bias.as_ref().unwrap().val().inner()),
                ),
            ];
            (primitive(loss.inner()), updates)
        })
        .unwrap();

        assert!(first_loss > 0.0, "initial loss should be positive");
        let mut prev_loss = first_loss;

        // Replay several training steps and observe loss decrease
        for step in 1..=5 {
            let timing = traced_step
                .step(vec![
                    InputPayload::F32(x_data.clone()),
                    InputPayload::F32(y_target.clone()),
                ])
                .unwrap();

            assert!(
                timing.loss < prev_loss,
                "step {step}: loss should decrease (prev {prev_loss}, current {})",
                timing.loss
            );
            prev_loss = timing.loss;
        }

        // Verify the weight buffer held by model was actually modified in-place
        let current_w = primitive(model.weight.val().inner())
            .download_device()
            .to_vec::<f32>()
            .unwrap();
        // Since y was 10.0 and initial output was 1*0.5 + 2*1.5 = 3.5, weights should have increased!
        assert!(current_w[0] > 0.5, "weight 0 should have increased");
        assert!(current_w[1] > 1.5, "weight 1 should have increased");
    });
}

#[test]
fn transformer_traced_training_runs_and_loss_decreases() {
    with_device(Config::default(), |device| {
        let weights = tt_mnist::transformer::init(59);
        let run = tt_mnist::transformer::train_traced(&weights, 3, &device)
            .expect("train_traced should succeed on TinyTransformer");
        assert_eq!(run.losses.len(), 3);
        assert!(run.losses[0] > 0.0);
        assert!(
            run.losses[2] < run.losses[0],
            "traced transformer loss should decrease: {:?} -> {:?}",
            run.losses[0],
            run.losses[2]
        );
    });
}
