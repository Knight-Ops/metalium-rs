//! Probe: where the host memory of a BF16 Qwen3.5 run on the device goes. `step260_qwen35`'s BF16
//! gate grows the host's resident set by ~8 GB every 5 s late in the run (killed at 44-55 GB) where
//! the F32 gate does not; this loads the model with one layer and prints the resident set after
//! each stage of a forward pass, with the op report, so the stage that allocates is named.

#![cfg(feature = "silicon")]

use burn::tensor::{Int, Tensor, TensorData};
use burn_tt::{TtBackend, TtDevice};
use tt_models::llama::Precision;
use tt_models::qwen35::Qwen35;
use tt_tests::burn_device::{with_device, Config};

fn rss_gb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let kb: f64 = status
        .lines()
        .find(|l| l.starts_with("VmRSS"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap();
    kb / 1048576.0
}

fn stage(name: &str) {
    eprintln!("rss after {name}: {:.2} GB", rss_gb());
}

#[test]
fn where_a_bf16_qwen35_forward_holds_host_memory() {
    for precision in [Precision::Bf16, Precision::F32] {
        with_device(Config::default(), |d: TtDevice| {
            eprintln!("--- {precision:?}");
            stage("the fork");
            let (model, _) = Qwen35::<TtBackend>::qwen3_5_4b(1, precision, &d).unwrap();
            stage("loading one layer and the tables");
            let ((), report) = burn_tt::with_report(|| {
                let dtype = precision.dtype();
                let ids = [248045, 846, 198, 3710, 369];
                let tokens = Tensor::<TtBackend, 2, Int>::from_data(
                    TensorData::new(ids.to_vec(), [1, ids.len()]),
                    &d,
                );
                for pass in 0..4 {
                    let mut state = model.state(8, dtype, &d);
                    let x = model.hidden(model.embed(tokens.clone()), &mut state, dtype);
                    let _ = x.into_data();
                    stage(&format!("layer stack, pass {pass}"));
                }
                let mut state = model.state(16, dtype, &d);
                let _ = model
                    .hidden(model.embed(tokens.clone()), &mut state, dtype)
                    .into_data();
                for step in 0..4 {
                    let one = tokens.clone().slice([0..1, 0..1]);
                    let _ = model
                        .hidden(model.embed(one), &mut state, dtype)
                        .into_data();
                    stage(&format!("decode step {step}"));
                }
            });
            eprintln!("{report}");
        });
    }
}

/// The whole BF16 model through `forward` (the head included): the resident set after loading,
/// after the prompt and after each decode step.
#[test]
fn where_the_whole_bf16_model_holds_host_memory() {
    with_device(Config::default(), |d: TtDevice| {
        stage("the fork");
        let (model, _) = Qwen35::<TtBackend>::qwen3_5_4b(32, Precision::Bf16, &d).unwrap();
        stage("loading 32 layers");
        let dtype = Precision::Bf16.dtype();
        let ids: [i32; 17] = [
            248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 248046, 198, 248045, 74455, 198,
            248068, 198,
        ];
        let tokens = |ids: &[i32]| {
            Tensor::<TtBackend, 2, Int>::from_data(
                TensorData::new(ids.to_vec(), [1, ids.len()]),
                &d,
            )
        };
        let mut state = model.state(32, dtype, &d);
        let hidden = model.hidden(model.embed(tokens(&ids)), &mut state, dtype);
        let _ = hidden.slice([0..1, 0..1, 0..1]).into_data();
        stage("the prompt's layer stack");
        let mut state = model.state(32, dtype, &d);
        let _ = model.forward(tokens(&ids), &mut state, dtype).into_data();
        stage("the prompt through forward (the head included)");
        for step in 0..4 {
            let _ = model
                .forward(tokens(&[90700]), &mut state, dtype)
                .into_data();
            stage(&format!("decode step {step}"));
        }
    });
}

/// Three layers (DeltaNets only) against four (the first full-attention layer added), BF16, over
/// the 17-token prompt: which kind of layer holds the host memory.
#[test]
fn which_layer_kind_holds_host_memory() {
    for layers in [3, 4] {
        with_device(Config::default(), |d: TtDevice| {
            let (model, _) = Qwen35::<TtBackend>::qwen3_5_4b(layers, Precision::Bf16, &d).unwrap();
            stage(&format!("loading {layers} layers"));
            let dtype = Precision::Bf16.dtype();
            let ids: Vec<i32> = (0..17).map(|i| 1000 + 37 * i).collect();
            let tokens = Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(ids, [1, 17]), &d);
            let mut state = model.state(32, dtype, &d);
            let ((), report) = burn_tt::with_report(|| {
                let h = model.hidden(model.embed(tokens), &mut state, dtype);
                let _ = h.slice([0..1, 0..1, 0..1]).into_data();
            });
            stage(&format!("the prompt through {layers} layers"));
            eprintln!("{report}");
        });
    }
}
