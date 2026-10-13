//! ViT-Tiny on the host: the pretrained network loads completely and classifies real photographs.
//! The same reasoning as `resnet18_host`: the host run has to be right before the device run is
//! compared with it. Needs `cargo xtask fetch-weights`.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use tt_models::image_input::{load_chw, HALF_MEAN, HALF_STD};
use tt_models::vit::{VitTiny, FULL_DEPTH};

fn logits(model: &VitTiny<Flex>, image: &str) -> Vec<f32> {
    let x = load_chw(image, 224, HALF_MEAN, HALF_STD).unwrap();
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
    let (_, loaded) = VitTiny::<Flex>::pretrained(FULL_DEPTH, &FlexDevice).unwrap();
    assert!(loaded.ignored.is_empty(), "{:?}", loaded.ignored);
    assert_eq!(loaded.applied, 152, "{loaded:?}");

    // A shallower network consumes the first blocks and says it ignored the rest.
    let (_, loaded) = VitTiny::<Flex>::pretrained(2, &FlexDevice).unwrap();
    assert_eq!(loaded.ignored.len(), 10 * 12, "{:?}", loaded.ignored);
}

#[test]
#[cfg_attr(
    not(feature = "weights"),
    ignore = "needs the weights: cargo xtask fetch-weights, then --features tt-models/weights"
)]
fn two_cats_on_a_couch_are_cats() {
    let (model, _) = VitTiny::<Flex>::pretrained(FULL_DEPTH, &FlexDevice).unwrap();
    let top5 = top(&logits(&model, "cats.jpg"), 5);
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
fn control_an_untrained_network_does_not_see_cats() {
    let model = VitTiny::<Flex>::new(FULL_DEPTH, &FlexDevice);
    let top5 = top(&logits(&model, "cats.jpg"), 5);
    assert!(
        !top5.iter().all(|c| (281..=285).contains(c)),
        "an untrained network's top-5 {top5:?} is all cats"
    );
}
