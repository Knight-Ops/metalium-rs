//! A training run interrupted by a checkpoint continues exactly as an uninterrupted one.
//!
//! B15's checkpoint gate (`burn-backend-parity.md` 3.4). Burn's `Learner` saves a model through a
//! `FileRecorder`; this uses the same recorder (`BinFileRecorder` at full precision) directly. A
//! `TinyTransformer` trains on the device: once for `BEFORE + AFTER` steps straight through, and
//! once for `BEFORE` steps, saved, loaded into a *fresh* model, and trained `AFTER` more on the
//! same batches. The two end with bit-identical weights.
//!
//! The optimiser is plain SGD, which keeps no state, so the weights are the whole checkpoint; a
//! stateful optimiser would need its own record, and this gate does not claim it.
//!
//! What makes the comparison able to fail:
//!
//! * the control resumes from a model that was *not* loaded (a fresh initialisation): its weights
//!   must differ, or the equality above would hold for any model;
//! * the saved file is read back and saved again, and the second file must equal the first byte
//!   for byte (the recorder is lossless at full precision);
//! * the weights of the interrupted run at the checkpoint differ from those it ends with, so a
//!   load that ignored the training would be caught.

use burn::backend::Autodiff;
use burn::module::{Module, ModuleVisitor, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_tt::TtBackend;
use tt_mnist::transformer::{batch, init, load, TinyTransformer, LR};
use tt_tests::burn_device::{with_device, Config};

type B = Autodiff<TtBackend>;

const BEFORE: usize = 1;
const AFTER: usize = 1;

/// Every float parameter's bits, in visiting order.
struct Bits(Vec<Vec<u32>>);

impl<Bk: Backend> ModuleVisitor<Bk> for Bits {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<Bk, D>>) {
        let data = param.val().into_data();
        self.0.push(
            data.to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect(),
        );
    }
}

fn bits<Bk: Backend>(model: &TinyTransformer<Bk>) -> Vec<Vec<u32>> {
    let mut v = Bits(Vec::new());
    model.visit(&mut v);
    v.0
}

/// Steps `from..to` of SGD on the device, from `model`.
fn train(
    mut model: TinyTransformer<B>,
    from: usize,
    to: usize,
    device: &burn_tt::TtDevice,
) -> TinyTransformer<B> {
    let mut optim = SgdConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(device);
    for step in from..to {
        let (x, y) = batch::<B>(step, device);
        let loss = loss_fn.forward(model.forward(x), y);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(LR, model, grads);
    }
    model
}

#[test]
fn a_checkpoint_resumes_bit_identically() {
    let weights = init(170);
    let dir = std::env::temp_dir().join(format!("tt-ckpt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    with_device(Config::default(), |d| {
        // Straight through.
        let straight = train(load::<B>(&weights, &d), 0, BEFORE + AFTER, &d);

        // Interrupted: train, save, load into a fresh model, continue.
        let at_checkpoint = train(load::<B>(&weights, &d), 0, BEFORE, &d);
        let saved = bits(&at_checkpoint);
        at_checkpoint
            .save_file(dir.join("model"), &recorder)
            .expect("save");
        let restored = TinyTransformer::<B>::new(&d)
            .load_file(dir.join("model"), &recorder, &d)
            .expect("load");
        assert_eq!(
            bits(&restored),
            saved,
            "the load did not restore the saved weights"
        );
        let resumed = train(restored, BEFORE, BEFORE + AFTER, &d);

        let end = bits(&straight);
        assert_eq!(
            bits(&resumed),
            end,
            "resuming from a checkpoint diverged from the uninterrupted run"
        );
        assert_ne!(
            saved, end,
            "control: training must change the weights between the two points"
        );

        // Control: the same continuation from a model that was not loaded.
        let unloaded = train(TinyTransformer::<B>::new(&d), BEFORE, BEFORE + AFTER, &d);
        assert_ne!(
            bits(&unloaded),
            end,
            "control: a fresh model reached the same weights, so the equality proves nothing"
        );

        // The recorder is lossless: save what was loaded and the file does not change.
        resumed_roundtrip(&dir, &recorder, &d);
    });
    let _ = std::fs::remove_dir_all(dir);
}

fn resumed_roundtrip(
    dir: &std::path::Path,
    recorder: &BinFileRecorder<FullPrecisionSettings>,
    d: &burn_tt::TtDevice,
) {
    let again = TinyTransformer::<B>::new(d)
        .load_file(dir.join("model"), recorder, d)
        .expect("load again");
    again
        .save_file(dir.join("again"), recorder)
        .expect("save again");
    let first = std::fs::read(dir.join("model.bin")).expect("first file");
    let second = std::fs::read(dir.join("again.bin")).expect("second file");
    assert_eq!(first, second, "save, load, save changed the file");
}
