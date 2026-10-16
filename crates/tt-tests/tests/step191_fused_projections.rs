//! decode-matmul-bandwidth P3: a Llama with each layer's `q|k|v` and `gate|up` projections one
//! matrix each (`Llama::fused`), the products' outputs split back by column views.
//!
//! The claims:
//! - **Bit for bit the unfused model.** Every output column of a fused product keeps its own
//!   accumulation over the hidden dimension, so the logits equal the unfused model's, eager (the
//!   growing cache) and traced (`TracedGenerator`, chunks and steps), on Llama shapes SmolLM2 lacks
//!   (grouped-query 4:1, head widths 32 and 128, untied heads, ragged widths) and on SmolLM2-135M's
//!   checkpoint (`weights` feature).
//! - **Fewer products:** a forward runs `4 layers + head` matmuls, not `7 layers + head`.
//! - **The fused matrix is what runs** (negative control): scaling `v_proj` after fusing changes
//!   the unfused model's generation and leaves the fused one's, whose `v` comes from `qkv`.

mod ladder_support;

use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_tt::{TtBackend, TtDevice};
use ladder_support::{assert_same, prompt, traced_equals_eager};
use tt_models::generate::Shapes;
use tt_models::llama::{Llama, LlamaConfig, Precision};
use tt_tests::burn_device::{with_device, Config};

const MAX_LEN: usize = 16;
const BUCKETS: [usize; 2] = [4, 8];
const LENGTHS: [usize; 4] = [1, 3, 5, 9];
const STEPS: usize = 3;

fn configs() -> Vec<(&'static str, LlamaConfig)> {
    let base = LlamaConfig {
        hidden: 64,
        heads: 2,
        kv_heads: 2,
        head_dim: 32,
        ff: 96,
        vocab: 100,
        layers: 2,
        rope_theta: 10_000.0,
        eps: 1e-5,
        tied_head: true,
    };
    vec![
        ("multi-head, tied, ragged vocabulary", base),
        (
            "grouped-query 4:1, untied",
            LlamaConfig {
                hidden: 128,
                heads: 4,
                kv_heads: 1,
                ff: 160,
                vocab: 64,
                tied_head: false,
                ..base
            },
        ),
        (
            "head width 128, ragged width, untied",
            LlamaConfig {
                hidden: 96,
                heads: 2,
                kv_heads: 1,
                head_dim: 128,
                ff: 200,
                vocab: 77,
                tied_head: false,
                ..base
            },
        ),
    ]
}

/// Scales every matrix whose path names `which` by `factor`.
struct Scale {
    which: &'static str,
    factor: f32,
    path: Vec<String>,
}

impl<B: Backend> ModuleMapper<B> for Scale {
    fn enter_module(&mut self, name: &str, _container: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container: &str) {
        self.path.pop();
    }
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        if D == 2 && self.path.join(".").contains(self.which) {
            let factor = self.factor;
            param.map(|t| t.mul_scalar(factor))
        } else {
            param
        }
    }
}

fn scaled(model: Llama<TtBackend>, which: &'static str) -> Llama<TtBackend> {
    model.map(&mut Scale {
        which,
        factor: 1.001,
        path: Vec::new(),
    })
}

/// Matmuls one forward of `model` over `t` tokens runs.
fn products(model: &Llama<TtBackend>, t: usize, d: &TtDevice) -> u64 {
    let cfg = model.config();
    let ids = Tensor::<TtBackend, 1, Int>::from_ints(&prompt(t, cfg.vocab)[..], d).reshape([1, t]);
    let mut cache = model.cache(MAX_LEN, burn::tensor::DType::F32, d);
    let (_, report) = burn_tt::with_report(|| {
        model
            .forward(ids, &mut cache, burn::tensor::DType::F32)
            .into_data()
    });
    report
        .0
        .iter()
        .filter(|s| s.op == "float_matmul")
        .map(|s| s.calls)
        .sum()
}

fn check(unfused: &Llama<TtBackend>, what: &str, lengths: &[usize], d: &TtDevice) {
    // A forward first: it materializes the lazily initialized parameters, which a clone taken
    // before would initialize again, to other values.
    let layers = unfused.config().layers as u64;
    assert_eq!(
        products(unfused, 3, d),
        7 * layers + 1,
        "{what}: unfused products"
    );
    let fused = unfused.clone().fused();
    assert_eq!(
        products(&fused, 3, d),
        4 * layers + 1,
        "{what}: fused products"
    );
    for &len in lengths {
        let p = prompt(len, unfused.config().vocab);
        let want = unfused.generate(&p, STEPS, Precision::F32, d);
        let got = fused.generate(&p, STEPS, Precision::F32, d);
        assert_same(&got, &want, &format!("{what}, prompt {len}: fused, eager"));
    }
    // Traced, against the fused eager schedules (which equal the unfused ones above).
    let shapes = Shapes::new(MAX_LEN, &BUCKETS);
    traced_equals_eager(
        &fused,
        &shapes,
        lengths,
        STEPS,
        &format!("{what}, fused"),
        d,
    );
}

#[test]
fn fused_llamas_equal_unfused_ones_bit_for_bit() {
    with_device(Config::default(), |d| {
        for (what, config) in configs() {
            let model = Llama::<TtBackend>::new(&config, &d);
            check(&model, what, &LENGTHS, &d);
        }
    });
}

/// The negative control: after fusing, the separate `v_proj` is not what runs.
#[test]
fn a_fused_model_computes_with_the_fused_matrix() {
    with_device(Config::default(), |d| {
        let (_, config) = configs().swap_remove(1);
        let model = Llama::<TtBackend>::new(&config, &d);
        let p = prompt(5, config.vocab);
        // Before the clone, which would otherwise initialize its own parameters.
        let want = model.generate(&p, STEPS, Precision::F32, &d);
        let fused = model.clone().fused();
        assert_same(
            &fused.generate(&p, STEPS, Precision::F32, &d),
            &want,
            "fused",
        );
        let unfused_scaled = scaled(model, "v_proj").generate(&p, STEPS, Precision::F32, &d);
        assert_ne!(
            ladder_support::bits(&unfused_scaled.logits[0]),
            ladder_support::bits(&want.logits[0]),
            "control: scaling v_proj changes the unfused model"
        );
        assert_same(
            &scaled(fused, "v_proj").generate(&p, STEPS, Precision::F32, &d),
            &want,
            "a fused model's v comes from qkv",
        );
    });
}

#[test]
#[cfg_attr(not(feature = "weights"), ignore = "needs `cargo xtask fetch-weights`")]
#[cfg_attr(
    all(feature = "weights", not(feature = "silicon")),
    ignore = "slow on the simulator (minutes): run explicitly with -- --ignored"
)]
fn smollm2_fused_equals_unfused() {
    let (layers, lengths, config) = if cfg!(feature = "silicon") {
        (
            tt_models::llama::SMOLLM2_135M.layers,
            &LENGTHS[..],
            Config {
                tiles: Some(burn_tt::TileChoice::All),
                ..Config::default()
            },
        )
    } else {
        (1, &[3][..], Config::default())
    };
    with_device(config, |d| {
        let (model, _) = Llama::<TtBackend>::smollm2_135m(layers, Precision::F32, &d).unwrap();
        check(&model, "SmolLM2", lengths, &d);
    });
}
