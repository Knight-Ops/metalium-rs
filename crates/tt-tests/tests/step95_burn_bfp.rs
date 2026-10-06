//! Physical storage propagation and straight-through gradients, independently
//! checked with exactly representable values and no intermediate downloads.
use burn::{
    backend::Autodiff,
    tensor::{Tensor, TensorData, TensorPrimitive},
};
use burn_tt::{
    storage::{StorageFormat as S, TensorStorageExt},
    tensor_traffic, TtBackend,
};
use tt_tests::burn_device::{assert_native_model, with_device, Config};
fn resident<const D: usize>(t: &Tensor<TtBackend, D>) {
    let TensorPrimitive::Float(t) = t.clone().into_primitive() else {
        panic!()
    };
    assert!(t.computed_on_device());
}
#[test]
fn propagation_promotes_binary_and_reductions_without_downloads() {
    with_device(Config::default(), |d| {
        for f in [S::Bfp8, S::Bfp4, S::Bfp2] {
            let source = Tensor::<TtBackend, 2>::from_data(
                TensorData::new(vec![1.0; 37 * 65], [37, 65]),
                &d,
            );
            // Cache the uncompressed source before casting: the cast must not
            // reuse it for readback or arithmetic.
            let _ = source.clone().into_data();
            let before = tensor_traffic();
            let ((packed, scalar, mixed, reduced, flipped), report) = burn_tt::with_report(|| {
                let packed = source.clone().with_storage(f);
                resident(&packed);
                let scalar = packed.clone().mul_scalar(2.0);
                let mixed = packed.clone() + source.clone();
                let reduced = packed.clone().sum_dim(1);
                let flipped = packed.clone().flip([1]);
                assert_eq!(packed.storage_format(), f);
                assert_eq!(scalar.storage_format(), f);
                assert_eq!(mixed.storage_format(), S::F32);
                assert_eq!(reduced.storage_format(), S::F32);
                assert_eq!(flipped.storage_format(), S::F32);
                assert_eq!(tensor_traffic().downloads, before.downloads);
                (packed, scalar, mixed, reduced, flipped)
            });
            assert_native_model(&report);
            assert_eq!(packed.clone().transpose().storage_format(), f);
            assert_eq!(packed.clone().reshape([37, 65]).storage_format(), f);
            assert_eq!(packed.reshape([1, 37 * 65]).storage_format(), S::F32);
            for t in [scalar, mixed] {
                assert_eq!(t.into_data().to_vec::<f32>().unwrap(), vec![2.0; 37 * 65]);
            }
            assert_eq!(reduced.into_data().to_vec::<f32>().unwrap(), vec![65.0; 37]);
            assert_eq!(
                flipped.into_data().to_vec::<f32>().unwrap(),
                vec![1.0; 37 * 65]
            );
        }
        let a =
            Tensor::<TtBackend, 2>::from_data([[1.0, 1.0], [1.0, 1.0]], &d).with_storage(S::Bfp2);
        let b = a.clone().with_storage(S::Bfp4);
        let c = a.clone() + b.clone();
        assert_eq!(c.storage_format(), S::Bfp4);
        assert_eq!(a.matmul(b).storage_format(), S::Bfp4);
    });
}
#[test]
fn cast_backward_is_identity_and_master_update_stays_f32() {
    type AD = Autodiff<TtBackend>;
    with_device(Config::default(), |d| {
        for f in [S::Bfp8, S::Bfp4, S::Bfp2] {
            let master = Tensor::<AD, 2>::from_data([[1.0, 1.0], [1.0, 1.0]], &d).require_grad();
            let before = tensor_traffic();
            let ((loss, gradient, updated), report) = burn_tt::with_report(|| {
                let weight = master.clone().with_storage(f);
                let loss = weight.mul_scalar(2.0).with_storage(S::F32).sum();
                let grads = loss.clone().backward();
                let gradient = master.grad(&grads).unwrap();
                assert_eq!(gradient.storage_format(), S::F32);
                let updated = master.clone().inner() - gradient.clone().mul_scalar(0.25);
                assert_eq!(updated.storage_format(), S::F32);
                assert_eq!(tensor_traffic().downloads, before.downloads);
                (loss, gradient, updated)
            });
            assert_native_model(&report);
            assert_eq!(loss.into_data().to_vec::<f32>().unwrap(), vec![8.0]);
            assert_eq!(gradient.into_data().to_vec::<f32>().unwrap(), vec![2.0; 4]);
            assert_eq!(updated.into_data().to_vec::<f32>().unwrap(), vec![0.5; 4]);
        }
    });
}

#[test]
fn trace_retains_compression_boundaries_and_reads_changed_values() {
    with_device(Config::default(), |d| {
        for f in [S::Bfp8, S::Bfp4, S::Bfp2] {
            let input = Tensor::<TtBackend, 2>::from_data(
                TensorData::new(vec![1.0; 37 * 65], [37, 65]),
                &d,
            )
            .to_device(&d);
            let TensorPrimitive::Float(primitive) = input.clone().into_primitive() else {
                panic!()
            };
            let (trace, first) = burn_tt::Trace::capture(&primitive, || {
                let output = input.clone().with_storage(f).mul_scalar(2.0);
                assert_eq!(output.storage_format(), f);
                let TensorPrimitive::Float(output) = output.into_primitive() else {
                    panic!()
                };
                output
            })
            .unwrap();
            assert_eq!(first, vec![2.0; 37 * 65]);
            assert_eq!(trace.run(vec![2.0; 37 * 65]).unwrap(), vec![4.0; 37 * 65]);
        }
    });
}

#[test]
fn fusion_and_group_preserving_views_keep_rounding_boundaries() {
    use burn::tensor::activation::relu;
    with_device(Config::default(), |d| {
        for f in [S::Bfp8, S::Bfp4, S::Bfp2] {
            for f in [S::Bfp8, S::Bfp4, S::Bfp2] {
                let a = Tensor::<TtBackend, 2>::ones([2, 16], &d).with_storage(f);
                let correct = a.clone().mul_scalar(1.1);
                let uncompressed = a.with_storage(S::F32).mul_scalar(1.1);
                assert_eq!(correct.storage_format(), f);
                assert_ne!(
                    correct.into_data().to_vec::<f32>().unwrap(),
                    uncompressed.into_data().to_vec::<f32>().unwrap(),
                    "missing output compression must fail"
                );
            }
            let a = Tensor::<TtBackend, 2>::from_data(
                TensorData::new(vec![1.0; 64 * 35], [64, 35]),
                &d,
            )
            .with_storage(f);
            let tail = a.clone().slice([32..64, 0..35]);
            assert_eq!(tail.storage_format(), f);
            drop(a);
            assert_eq!(
                tail.into_data().to_vec::<f32>().unwrap(),
                vec![1.0; 32 * 35]
            );
            let x =
                Tensor::<burn_tt::Tt, 2>::from_data([[1.0, 1.0], [-1.0, -1.0]], &d).with_storage(f);
            let y = relu(x.clone() + x.clone());
            assert_eq!(y.storage_format(), f);
            let z = y.clone().with_storage(S::Bfp8).mul_scalar(2.0);
            assert_eq!(z.storage_format(), S::Bfp8);
            assert_eq!(
                y.into_data().to_vec::<f32>().unwrap(),
                vec![2.0, 2.0, 0.0, 0.0]
            );
            assert_eq!(
                z.into_data().to_vec::<f32>().unwrap(),
                vec![4.0, 4.0, 0.0, 0.0]
            );
        }
    });
}

#[test]
fn cached_source_values_batched_products_and_attention_use_decoded_storage() {
    use burn::tensor::{
        activation::{log_softmax, softmax, softmin},
        module::attention,
    };
    use burn_tensor::ops::AttentionModuleOptions;
    with_device(Config::default(), |d| {
        for f in [S::Bfp8, S::Bfp4, S::Bfp2] {
            let x = Tensor::<TtBackend, 2>::from_data([[1.1; 16]; 2], &d);
            let original = x.clone().into_data().to_vec::<f32>().unwrap();
            let packed = x.with_storage(f);
            resident(&packed);
            let decoded = packed.clone().into_data().to_vec::<f32>().unwrap();
            assert_ne!(
                original, decoded,
                "cached uncompressed host data must be invalidated"
            );
            for out in [
                softmax(packed.clone(), 1),
                log_softmax(packed.clone(), 1),
                softmin(packed.clone(), 1),
            ] {
                assert_eq!(
                    out.storage_format(),
                    S::F32,
                    "softmax reductions promote the compound result"
                );
                resident(&out);
            }
            let scalar = packed.mul_scalar(2.0);
            assert_eq!(
                scalar.into_data().to_vec::<f32>().unwrap(),
                decoded.iter().map(|v| v * 2.0).collect::<Vec<_>>()
            );
            let a = Tensor::<TtBackend, 3>::ones([2, 3, 7], &d).with_storage(f);
            let b = Tensor::<TtBackend, 3>::ones([1, 7, 5], &d);
            let out = a.matmul(b);
            assert_eq!(out.storage_format(), S::F32);
            assert_eq!(
                out.into_data().to_vec::<f32>().unwrap(),
                vec![7.0; 2 * 3 * 5]
            );
            let q = Tensor::<TtBackend, 4>::zeros([1, 1, 3, 5], &d).with_storage(f);
            let k = Tensor::<TtBackend, 4>::zeros([1, 1, 2, 5], &d).with_storage(f);
            let v = Tensor::<TtBackend, 4>::ones([1, 1, 2, 3], &d).with_storage(f);
            let before = tensor_traffic();
            let (out, report) = burn_tt::with_report(|| {
                attention(
                    q,
                    k,
                    v,
                    None,
                    None,
                    AttentionModuleOptions {
                        scale: Some(1.0),
                        softcap: None,
                        is_causal: false,
                    },
                )
            });
            assert_native_model(&report);
            assert_eq!(tensor_traffic().downloads, before.downloads);
            assert_eq!(out.storage_format(), S::F32);
            assert_eq!(out.into_data().to_vec::<f32>().unwrap(), vec![1.0; 9]);
        }
    });
}

#[test]
fn checkpointed_cast_backward_and_packed_parameter_copy_preserve_contracts() {
    type AD =
        Autodiff<TtBackend, burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing>;
    with_device(Config::default(), |d| {
        for f in [S::Bfp8, S::Bfp4, S::Bfp2] {
            let x = Tensor::<AD, 2>::from_data([[1.0, 2.0], [1.0, 2.0]], &d).require_grad();
            let a = x.clone().mul_scalar(2.0).with_storage(f);
            let loss = (a.clone() * a).with_storage(S::F32).sum();
            let grads = loss.backward();
            let g = x.grad(&grads).unwrap();
            assert_eq!(g.storage_format(), S::F32);
            assert_eq!(
                g.into_data().to_vec::<f32>().unwrap(),
                if f == S::Bfp2 {
                    vec![0.0, 16.0, 0.0, 16.0]
                } else {
                    vec![8.0, 16.0, 8.0, 16.0]
                }
            );
            let original = Tensor::<TtBackend, 2>::ones([2, 2], &d).with_storage(f);
            let input = Tensor::<TtBackend, 2>::ones([2, 2], &d).to_device(&d);
            let TensorPrimitive::Float(ip) = input.clone().into_primitive() else {
                panic!()
            };
            let TensorPrimitive::Float(op) = original.clone().into_primitive() else {
                panic!()
            };
            let (trace, first) = burn_tt::TracedTrainingStep::capture(&[&ip], || {
                let update = input.clone().mul_scalar(2.0).with_storage(f);
                let loss = update.clone().with_storage(S::F32).sum();
                let TensorPrimitive::Float(loss) = loss.into_primitive() else {
                    panic!()
                };
                let TensorPrimitive::Float(update) = update.into_primitive() else {
                    panic!()
                };
                (loss, vec![(update, op.clone())])
            })
            .unwrap();
            assert_eq!(first, 8.0);
            assert_eq!(
                original.clone().into_data().to_vec::<f32>().unwrap(),
                vec![2.0; 4]
            );
            let _ = trace
                .step(vec![burn_tt::InputPayload::F32(vec![2.0; 4])])
                .unwrap();
            assert_eq!(
                original
                    .clone()
                    .into_primitive()
                    .tensor()
                    .download_device()
                    .to_vec::<f32>()
                    .unwrap(),
                vec![4.0; 4]
            );
        }
    });
}
