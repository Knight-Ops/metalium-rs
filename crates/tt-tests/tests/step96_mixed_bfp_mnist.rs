//! Application precision policies retain F32 masters and losses. Accuracy is
//! reported against the same F32 initialization, without changing F32 goldens.
use burn::{
    backend::Autodiff,
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{GradientsParams, Optimizer, SgdConfig},
    tensor::{backend::Backend, Int, Tensor, TensorData, TensorPrimitive},
};
use burn_tt::{
    storage::{NativeStoragePolicy, StorageFormat as S, TensorStorageExt},
    TtBackend,
};
use tt_tests::{
    burn_device::{assert_native_model, with_device, Config},
    mnist,
};
type AD = Autodiff<TtBackend>;
fn f32_policy(_: &str) -> S {
    S::F32
}
fn policy2(name: &str) -> S {
    if name.ends_with("hidden.weight") || name.ends_with("conv.weight") {
        S::Bfp2
    } else if name.ends_with("head.weight") {
        S::Bfp8
    } else if name.ends_with("relu") {
        S::Bfp4
    } else {
        S::F32
    }
}
fn policy4(name: &str) -> S {
    if name.ends_with("hidden.weight") || name.ends_with("conv.weight") {
        S::Bfp4
    } else if name.ends_with("head.weight") {
        S::Bfp8
    } else if name.ends_with("relu") {
        S::Bfp2
    } else {
        S::F32
    }
}
fn policy8(name: &str) -> S {
    if name.ends_with("hidden.weight") || name.ends_with("conv.weight") {
        S::Bfp8
    } else if name.ends_with("head.weight") {
        S::Bfp2
    } else if name.ends_with("relu") {
        S::Bfp4
    } else {
        S::F32
    }
}
fn primitive<const D: usize>(t: Tensor<TtBackend, D>) -> burn_tt::TtTensor {
    let TensorPrimitive::Float(t) = t.into_primitive() else {
        panic!()
    };
    t
}
fn train<M: AutodiffModule<AD>, const D: usize>(
    mut model: M,
    images: Tensor<AD, 2>,
    labels: Tensor<AD, 1, Int>,
    policy: NativeStoragePolicy,
    forward: impl Fn(&M, Tensor<AD, 2>, &NativeStoragePolicy) -> Tensor<AD, 2>,
    weight: impl Fn(&M) -> Tensor<AD, D>,
    name: &str,
) {
    let d = images.device();
    let initial = weight(&model).clone().into_data().to_vec::<f32>().unwrap();
    let loss_fn = CrossEntropyLossConfig::new().init(&d);
    let mut optimizer = SgdConfig::new().init();
    let before = burn_tt::tensor_traffic();
    let ((losses, model_out), report) = burn_tt::with_report(|| {
        let mut losses = Vec::new();
        for _ in 0..4 {
            let logits = forward(&model, images.clone(), &policy);
            // F32 bias already promotes logits; make the sensitive loss
            // boundary explicit in the example regardless of policy changes.
            let loss = loss_fn.forward(logits.with_storage(S::F32), labels.clone());
            let grads = loss.clone().backward();
            let gradient = weight(&model).grad(&grads).unwrap();
            assert_eq!(gradient.storage_format(), S::F32);
            let grad = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(0.05, model, grad);
            assert_eq!(weight(&model).storage_format(), S::F32);
            losses.push(loss);
        }
        assert_eq!(burn_tt::tensor_traffic().downloads, before.downloads);
        (losses, model)
    });
    model = model_out;
    assert_native_model(&report);
    let losses: Vec<_> = losses
        .into_iter()
        .map(|t| t.into_data().to_vec::<f32>().unwrap()[0])
        .collect();
    assert!(losses.iter().all(|v| v.is_finite()));
    let final_weight = weight(&model).into_data().to_vec::<f32>().unwrap();
    assert_ne!(initial, final_weight, "optimizer must update F32 masters");
    let logits = forward(&model, images.clone(), &policy)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let labels_vec = labels.into_data().to_vec::<i32>().unwrap();
    let correct = logits
        .chunks_exact(10)
        .zip(labels_vec)
        .filter(|(row, label)| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0
                == *label as usize
        })
        .count();
    eprintln!("PRECISION {name} loss={losses:?} accuracy={correct}/4");
    let input = primitive(images.clone().inner());
    let (trace, first) = burn_tt::Trace::capture(&input, || {
        primitive(forward(&model, images.clone(), &policy).inner())
    })
    .unwrap();
    assert_eq!(first, logits);
    let mut changed = images.clone().into_data().to_vec::<f32>().unwrap();
    changed.reverse();
    let replay = trace.run(changed.clone()).unwrap();
    let fresh = forward(
        &model,
        Tensor::from_data(TensorData::new(changed, [4, 784]), &d),
        &policy,
    )
    .into_data()
    .to_vec::<f32>()
    .unwrap();
    assert_eq!(replay, fresh, "changed-input precision-policy replay");
}
fn inputs(split: &mnist::Split, d: &burn_tt::TtDevice) -> (Tensor<AD, 2>, Tensor<AD, 1, Int>) {
    (
        Tensor::from_data(
            TensorData::new(split.images[..4 * 784].to_vec(), [4, 784]),
            d,
        )
        .to_device(d),
        Tensor::from_data(
            TensorData::new(
                split.labels[..4]
                    .iter()
                    .map(|&v| v as i32)
                    .collect::<Vec<_>>(),
                [4],
            ),
            d,
        )
        .to_device(d),
    )
}
#[test]
fn mnist_mlp_policies_keep_f32_training_state_and_replay() {
    let split = mnist::load(true);
    with_device(Config::default(), |d| {
        for (name, policy) in [
            ("MLP F32", f32_policy as fn(&str) -> S),
            ("MLP BFP2/4/8", policy2),
            ("MLP BFP4/2/8", policy4),
            ("MLP BFP8/4/2", policy8),
        ] {
            TtBackend::seed(&d, 23);
            let model = tt_mnist::precision::Mlp::<AD>::new(&d);
            let (x, y) = inputs(&split, &d);
            train(
                model,
                x,
                y,
                NativeStoragePolicy(policy),
                |m, x, p| m.forward(x, p),
                |m| m.hidden.weight.val(),
                name,
            );
        }
    });
}
#[test]
fn mnist_cnn_policies_keep_f32_training_state_and_replay() {
    let split = mnist::load(true);
    with_device(Config::default(), |d| {
        for (name, policy) in [
            ("CNN F32", f32_policy as fn(&str) -> S),
            ("CNN BFP2/4/8", policy2),
            ("CNN BFP4/2/8", policy4),
            ("CNN BFP8/4/2", policy8),
        ] {
            let model =
                tt_mnist::cnn::Cnn::<AD>::new(&Default::default(), burn::tensor::DType::F32, &d);
            let (x, y) = inputs(&split, &d);
            train(
                model,
                x,
                y,
                NativeStoragePolicy(policy),
                |m, x, p| m.forward_with_precision(x, p),
                |m| m.conv.weight.val(),
                name,
            );
        }
    });
}
