//! D4 gate, first part: a gather and its backward on the card -- the column
//! a loss picks by its targets.
//!
//! `float_gather` along the last dimension with one index per row (Burn's
//! `CrossEntropyLoss` gathers each row's target this way) masks every other
//! element to `-0` and sums the row, which is the gathered element whatever
//! the order; `float_scatter_add`, its backward, adds where the
//! column is the row's index. Both against `burn-flex` bit for bit, on values
//! with signed zeros, infinities and NaNs -- but for one derived exception:
//! a gathered `-0` is `+0`, the SFPU's sum dropping a zero's sign
//! (`ttsim-divergence.md` row C) -- with the indices coming from the host as
//! targets do, and downloading nothing. Watched failing with the mask
//! inverted.

use burn::tensor::{Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{with_device, Config};

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match i % 17 {
                0 => -0.0,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => f32::NAN,
                _ => ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0,
            }
        })
        .collect()
}

fn targets(seed: u64, rows: usize, classes: usize) -> Vec<i32> {
    (0..rows)
        .map(|i| ((i as u64 * 2654435761 + seed) % classes as u64) as i32)
        .collect()
}

/// Bit for bit, a NaN by class (the device canonicalises NaNs, numerics row
/// D) and, where `zero_sign` is false, a zero by value: the SFPU's sum drops a
/// `-0`'s sign (row C), so a gathered `-0` is `+0`.
fn same(got: &[f32], want: &[f32], zero_sign: bool) -> bool {
    got.len() == want.len()
        && got.iter().zip(want).all(|(g, w)| {
            g.to_bits() == w.to_bits()
                || (g.is_nan() && w.is_nan())
                || (!zero_sign && *g == 0.0 && *w == 0.0)
        })
}

fn on_device<const D: usize>(t: &Tensor<TtBackend, D>) -> bool {
    match t.clone().into_primitive() {
        burn::tensor::TensorPrimitive::Float(p) => p.computed_on_device(),
        _ => false,
    }
}

fn check<const D: usize>(d: &burn_tt::TtDevice, shape: [usize; D]) {
    let n: usize = shape.iter().product();
    let c = shape[D - 1];
    let rows = n / c;
    let mut ishape = shape;
    ishape[D - 1] = 1;
    let x = values(c as u64, n);
    let idx = targets(rows as u64, rows, c);
    let v = values(rows as u64 + 7, rows);

    let tx = Tensor::<TtBackend, D>::from_data(TensorData::new(x.clone(), shape), d).to_device(d);
    let fx = Tensor::<Flex, D>::from_data(TensorData::new(x, shape), &FlexDevice);
    let ti = Tensor::<TtBackend, D, Int>::from_data(TensorData::new(idx.clone(), ishape), d);
    let fi = Tensor::<Flex, D, Int>::from_data(TensorData::new(idx, ishape), &FlexDevice);
    let tv = Tensor::<TtBackend, D>::from_data(TensorData::new(v.clone(), ishape), d);
    let fv = Tensor::<Flex, D>::from_data(TensorData::new(v, ishape), &FlexDevice);

    let before = tensor_traffic();
    let g = tx.clone().gather(D - 1, ti.clone());
    let s = tx.scatter(D - 1, ti, tv, burn::tensor::IndexingUpdateOp::Add);
    let moved = tensor_traffic() - before;
    assert_eq!(moved.downloads, 0, "{shape:?}: {moved:?}");
    assert!(
        on_device(&g) && on_device(&s),
        "{shape:?}: not on the device"
    );

    let want = fx
        .clone()
        .gather(D - 1, fi.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let got = g.into_data().to_vec::<f32>().unwrap();
    assert!(
        same(&got, &want, false),
        "{shape:?} gather:\n{got:?}\n{want:?}"
    );
    let want = fx
        .scatter(D - 1, fi, fv, burn::tensor::IndexingUpdateOp::Add)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let got = s.into_data().to_vec::<f32>().unwrap();
    assert!(same(&got, &want, true), "{shape:?} scatter_add");
}

#[test]
fn a_gather_by_target_and_its_backward_are_flex_s_bits_on_the_card() {
    with_device(Config::default(), |d| {
        check(&d, [64, 10]);
        check(&d, [128, 64]);
        check(&d, [4, 32, 64]);
        check(&d, [37, 70]);
    });
}

/// D4, second part: an embedding on the card. `select` along dimension 0 of
/// a resident table moves each row where it lies (`Session::gather_rows`),
/// bit for bit; `select_add` -- the embedding's gradient -- adds rows by
/// index in Flex's order (`Session::rows_add`), with repeats; and Burn's
/// `nn::Embedding` trains through them under autodiff, matching Flex. Indices
/// come from the host, as token ids do; nothing is downloaded.
#[test]
fn an_embedding_s_lookup_and_gradient_run_on_the_card() {
    use burn::backend::Autodiff;
    use burn::module::Module;
    use burn::nn::{Embedding, EmbeddingConfig};
    with_device(Config::default(), |d| {
        for (vocab, dm, n) in [(64, 64, 128), (100, 70, 37)] {
            let w = values(vocab as u64, vocab * dm);
            let ids: Vec<i32> = (0..n)
                .map(|i| {
                    if i % 5 == 0 {
                        3
                    } else {
                        ((i * 13 + 7) % vocab) as i32
                    }
                })
                .collect();
            let g = values(n as u64, n * dm);
            let tw = Tensor::<TtBackend, 2>::from_data(TensorData::new(w.clone(), [vocab, dm]), &d)
                .to_device(&d);
            let fw = Tensor::<Flex, 2>::from_data(TensorData::new(w, [vocab, dm]), &FlexDevice);
            let ti = Tensor::<TtBackend, 1, Int>::from_data(TensorData::new(ids.clone(), [n]), &d);
            let fi = Tensor::<Flex, 1, Int>::from_data(TensorData::new(ids, [n]), &FlexDevice);
            let tg = Tensor::<TtBackend, 2>::from_data(TensorData::new(g.clone(), [n, dm]), &d)
                .to_device(&d);
            let fg = Tensor::<Flex, 2>::from_data(TensorData::new(g, [n, dm]), &FlexDevice);

            let before = tensor_traffic();
            let sel = tw.clone().select(0, ti.clone());
            let add = tw.select_assign(0, ti, tg, burn::tensor::IndexingUpdateOp::Add);
            let moved = tensor_traffic() - before;
            assert_eq!(moved.downloads, 0, "[{vocab}, {dm}]: {moved:?}");
            assert!(on_device(&sel) && on_device(&add), "[{vocab}, {dm}]");
            let want = fw
                .clone()
                .select(0, fi.clone())
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let got = sel.into_data().to_vec::<f32>().unwrap();
            assert!(same(&got, &want, true), "[{vocab}, {dm}] select");
            let want = fw
                .select_assign(0, fi, fg, burn::tensor::IndexingUpdateOp::Add)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let got = add.into_data().to_vec::<f32>().unwrap();
            // A row summed to zero from signed zeros may differ in the zero's
            // sign (row C), as a gather's may.
            assert!(same(&got, &want, false), "[{vocab}, {dm}] select_add");
        }

        // `nn::Embedding` forward and backward, from Flex's weights.
        let (vocab, dm, b, s) = (64, 64, 4, 32);
        <Flex as burn::tensor::backend::Backend>::seed(&FlexDevice, 61);
        let fe: Embedding<Autodiff<Flex>> = EmbeddingConfig::new(vocab, dm).init(&FlexDevice);
        let w = fe.weight.val().into_data();
        let te: Embedding<Autodiff<TtBackend>> = EmbeddingConfig::new(vocab, dm).init(&d);
        let te = te.map(&mut Load(Some(w)));
        let ids: Vec<i32> = (0..b * s).map(|i| ((i * 7 + 1) % vocab) as i32).collect();
        let fx = Tensor::<Autodiff<Flex>, 2, Int>::from_data(
            TensorData::new(ids.clone(), [b, s]),
            &FlexDevice,
        );
        let tx = Tensor::<Autodiff<TtBackend>, 2, Int>::from_data(TensorData::new(ids, [b, s]), &d);
        let (fo, to) = (fe.forward(fx), te.forward(tx));
        assert!(
            to.clone()
                .inner()
                .into_primitive()
                .tensor()
                .computed_on_device(),
            "nn::Embedding's lookup ran on the host"
        );
        // Through a device op, so the gradient reaching the embedding is
        // made on the card, as a model's is (`sum`'s own is a host tensor).
        let fgr = (fo.clone() * fo.clone()).sum().backward();
        let tgr = (to.clone() * to.clone()).sum().backward();
        let got = to.into_data().to_vec::<f32>().unwrap();
        let want = fo.into_data().to_vec::<f32>().unwrap();
        assert!(same(&got, &want, true), "nn::Embedding forward");
        let tw = te.weight.grad(&tgr).unwrap();
        assert!(
            tw.clone().into_primitive().tensor().computed_on_device(),
            "nn::Embedding's gradient ran on the host"
        );
        let got = tw.into_data().to_vec::<f32>().unwrap();
        let want = fe
            .weight
            .grad(&fgr)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(same(&got, &want, false), "nn::Embedding backward");
    });
}

/// Replace the one float parameter.
struct Load(Option<TensorData>);

impl<B: burn::tensor::backend::Backend> burn::module::ModuleMapper<B> for Load {
    fn map_float<const D: usize>(
        &mut self,
        param: burn::module::Param<Tensor<B, D>>,
    ) -> burn::module::Param<Tensor<B, D>> {
        let (id, t, _) = param.consume();
        let data = self.0.take().expect("one parameter");
        burn::module::Param::initialized(id, Tensor::from_data(data, &t.device()).require_grad())
    }
}
