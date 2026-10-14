//! ResNet-18 on the host: the pretrained network loads completely and classifies real photographs.
//!
//! This is what makes the later comparison meaningful. If the architecture, the layout
//! conversion or the preprocessing were wrong, burn-tt and burn-flex would still agree with each
//! other on nonsense; here the weights must produce the right ImageNet class on the host first.
//! Needs `cargo xtask fetch-weights`.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_models::image_input::{load_chw, IMAGENET_MEAN, IMAGENET_STD};
use tt_models::resnet::ResNet18;

fn logits(model: &ResNet18<Flex>, image: &str) -> Vec<f32> {
    let x = load_chw(image, 224, IMAGENET_MEAN, IMAGENET_STD).unwrap();
    let x = Tensor::<Flex, 4>::from_data(TensorData::new(x, [1, 3, 224, 224]), &FlexDevice);
    model.forward(x).into_data().to_vec::<f32>().unwrap()
}

fn top(logits: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    idx.truncate(k);
    idx
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn the_pretrained_network_loads_completely() {
    let (_, loaded) = ResNet18::<Flex>::pretrained(&FlexDevice).unwrap();
    // The file holds 122 tensors; 20 are PyTorch's `num_batches_tracked` counters.
    assert_eq!(loaded.ignored.len(), 20, "{:?}", loaded.ignored);
    assert_eq!(loaded.applied, 122 - 20, "{loaded:?}");
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn two_cats_on_a_couch_are_cats() {
    let (model, _) = ResNet18::<Flex>::pretrained(&FlexDevice).unwrap();
    let l = logits(&model, "cats.jpg");
    assert_eq!(l.len(), 1000);
    // ImageNet-1k: 281 tabby, 282 tiger cat, 285 Egyptian cat, 283 Persian... Any feline is right.
    let top5 = top(&l, 5);
    eprintln!("cats top-5 {top5:?}");
    assert!(
        (281..=285).contains(&top5[0]),
        "the top class {} is not a cat: {top5:?}",
        top5[0]
    );
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn control_a_wrong_weight_changes_the_answer() {
    // The same network with fresh random weights must not classify the cats as cats; otherwise
    // the test above would pass without the checkpoint.
    let model = ResNet18::<Flex>::new(&FlexDevice);
    let l = logits(&model, "cats.jpg");
    let top5 = top(&l, 5);
    assert!(
        !top5.iter().all(|c| (281..=285).contains(c)),
        "an untrained network's top-5 {top5:?} is all cats"
    );
}
