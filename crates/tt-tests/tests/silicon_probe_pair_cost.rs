//! What a K-tile pair of a blocked prefill product costs, and why (`docs/plans/prefill-operand-reuse.md`
//! P4): the product's rate on every tile, traced, under variants that each take one suspect away.
//!
//! - **HiFi4 against LoFi:** four `MVMUL` phases a pair against one. If the rate barely moves, the
//!   pair is not bound by the Matrix Unit.
//! - **`B` in FP32 against `B` stored BF16** (`Session::store_bf16`): half the bytes each of `B`'s
//!   eight face unpacks a pair moves. If the rate moves, the unpacker binds. And `B` stored BFP8,
//!   BFP4 and BFP2 (`DramTensor::from_bfp`, `docs/plans/dataflow-compiler.md` F1c), each checked
//!   against the values its storage decodes to: at one-row decode shapes (`TT_PAIR_SHAPES=1,...`) a
//!   product is bound by its weight's bytes, which these hold in ~0.28 / 0.15 / 0.09 of FP32's.
//! - **No drain** (`matmul::probe_no_drain`, its own test, so its own process): each pair's step
//!   without waiting for both unpackers to finish before the next bases are written. The products
//!   may be wrong; only the time is measured.
//! - **The unpackers' speed** (`matmul::probe_throttle`, a test each): `Throttle_mode` x2 or x4 for
//!   unpacker 0 (`B`) and unpacker 1 (`A`), where Wormhole's page says x1 fetches 16 bytes a cycle.
//!
//! Every product but the no-drain ones is checked against exact integer arithmetic on the host (the
//! operands are small integers: every product and sum is exact in FP32 and in TF32 and BF16).
//!
//! Each variant: `TT_PAIR_SHAPES` (`m,k,n;...`, default two of the 1.7B's prefill products at 1,024
//! rows), 2 warm replays, then the median of 7. Silicon only, a probe: `cargo xtask bench --filter
//! silicon_probe_pair_cost`.

#![cfg(feature = "silicon")]

use std::time::Instant;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::DramTensor;

fn shapes() -> Vec<[usize; 3]> {
    std::env::var("TT_PAIR_SHAPES").map_or(vec![[1024, 2048, 8192], [1024, 8192, 2048]], |v| {
        v.split(';')
            .map(|s| {
                let d: Vec<usize> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
                [d[0], d[1], d[2]]
            })
            .collect()
    })
}

/// Integers whose products and sums are exact: the products may be checked against each other.
fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + seed) % 5) as f32 - 2.0).collect()
}

/// The median replay of `a @ b` at `fidelity`, in us, and the product.
fn replay_us(
    s: &mut Session<tt_kmd::Kmd>,
    a: &DramTensor,
    b: &DramTensor,
    fidelity: Fidelity,
) -> (f64, Vec<f32>) {
    s.begin_trace().unwrap();
    let c = s
        .matmul_dram(
            a,
            false,
            b,
            false,
            SrcRoute::Tf32FromFp32,
            fidelity,
            1 << 40,
        )
        .unwrap();
    let id = s.end_trace().unwrap();
    let mut times = Vec::new();
    for i in 0..9 {
        let t0 = Instant::now();
        s.replay(id).unwrap();
        s.sync().unwrap();
        if i >= 2 {
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    s.release_trace(id).unwrap();
    let out = s.download(&c).unwrap();
    s.free(c).unwrap();
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2], out)
}

fn run(no_drain: bool, throttle: [u32; 2]) {
    run_orders(no_drain, throttle, &[None]);
}

/// The gathers a probe leaves out: `A`'s (`tensor::probe_skip_a_reads`) or every one
/// (`tensor::probe_skip_block_gathers`). The products are then wrong and go unchecked.
#[derive(Clone, Copy, PartialEq)]
enum Skip {
    Nothing,
    A,
    All,
}

static SKIP: std::sync::Mutex<Skip> = std::sync::Mutex::new(Skip::Nothing);

/// [`run`] under each pair order (`matmul::PairOrder`; `None` the default), in one session: the
/// program caches key on the order.
fn run_orders(
    no_drain: bool,
    throttle: [u32; 2],
    orders: &[Option<tt_kernels::matmul::PairOrder>],
) {
    tt_kernels::matmul::probe_no_drain(no_drain);
    tt_kernels::matmul::probe_throttle(throttle[0], throttle[1]);
    let skip = *SKIP.lock().unwrap();
    tt_kernels::tensor::probe_skip_a_reads(skip == Skip::A);
    tt_kernels::tensor::probe_skip_block_gathers(skip == Skip::All);
    tt_ttsim::fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::All,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
        for [m, k, n] in shapes() {
            let (av, bv) = (values(m * k, 1), values(k * n, 3));
            let a = s.upload(&av, m, k).unwrap();
            let b = s.upload(&bv, k, n).unwrap();
            let bb = s.store_bf16(&b).unwrap();
            // Each BFP weight with the values its storage decodes to (BFP2's one magnitude bit
            // cannot hold these integers, so the reference is the decoded weight).
            let packed: Vec<(&str, DramTensor, Vec<f32>)> = [
                ("BFP8", tt_kernels::bfp::BfpFormat::Bfp8),
                ("BFP4", tt_kernels::bfp::BfpFormat::Bfp4),
                ("BFP2", tt_kernels::bfp::BfpFormat::Bfp2),
            ]
            .into_iter()
            .map(|(name, f)| {
                let p = s.bfp_from_f32(&b, f).unwrap();
                let decoded = s.bfp_to_f32(&p).unwrap();
                let values = s.download(&decoded).unwrap();
                s.free(decoded).unwrap();
                (name, DramTensor::from_bfp(p), values)
            })
            .collect();
            let flops = 2.0 * (m * k * n) as f64;
            for &order in orders {
            if let Some(order) = order {
                tt_kernels::matmul::set_pair_order(order);
            }
            let drain = format!(
                "{}, throttle {}/{}{}{}",
                if no_drain { "no drain" } else { "drained" },
                throttle[0],
                throttle[1],
                order.map_or(String::new(), |o| format!(", {o:?}")),
                match skip {
                    Skip::Nothing => "",
                    Skip::A => ", no A gathers",
                    Skip::All => ", no gathers",
                }
            );
            // `TT_PAIR_FIDELITIES` (`lo,hifi2,hifi3,hifi4`; default LoFi and HiFi4).
            let fidelities: Vec<(&str, Fidelity)> = std::env::var("TT_PAIR_FIDELITIES")
                .map_or(vec![("HiFi4", Fidelity::HiFi4), ("LoFi", Fidelity::Lo)], |v| {
                    v.split(',')
                        .map(|f| match f.trim() {
                            "lo" => ("LoFi", Fidelity::Lo),
                            "hifi2" => ("HiFi2", Fidelity::HiFi2),
                            "hifi3" => ("HiFi3", Fidelity::HiFi3),
                            "hifi4" => ("HiFi4", Fidelity::HiFi4),
                            other => panic!("TT_PAIR_FIDELITIES: {other}"),
                        })
                        .collect()
                });
            let mut weights: Vec<(&str, &DramTensor, &[f32])> = vec![("FP32", &b, &bv), ("BF16", &bb, &bv)];
            weights.extend(packed.iter().map(|(name, t, v)| (*name, t, v.as_slice())));
            let cases: Vec<(String, &DramTensor, &[f32], Fidelity)> = weights
                .into_iter()
                .flat_map(|(format, b, bv)| {
                    fidelities
                        .iter()
                        .map(move |&(name, f)| (format!("{name}, B {format}"), b, bv, f))
                })
                .collect();
            for (what, b, bv, fidelity) in cases {
                let (us, out) = replay_us(&mut s, &a, b, fidelity);
                let tflops = flops / us / 1e6;
                if !no_drain && skip == Skip::Nothing {
                    for i in (0..m).step_by(97).chain([m - 1]) {
                        for j in (0..n).step_by(101).chain([n - 1]) {
                            let exact: f32 = (0..k).map(|q| av[i * k + q] * bv[q * n + j]).sum();
                            assert_eq!(out[i * n + j], exact, "{what}, {drain}: ({i}, {j})");
                        }
                    }
                }
                println!(
                    "MEASURE pair cost [{m},{k}]@[{k},{n}] {what}, {drain}: {us:.1} us, {tflops:.2} TFLOPS"
                );
                println!(
                    "BENCH {{\"kind\":\"result\",\"key\":\"pair cost [{m},{k}]@[{k},{n}] {what}, {drain}\",\"unit\":\"TFLOPS\",\"timed\":\"host\",\"median\":{tflops},\"p10\":{tflops},\"p90\":{tflops},\"n\":7}}"
                );
            }
            }
            for t in [a, b, bb].into_iter().chain(packed.into_iter().map(|(_, t, _)| t)) {
                s.free(t).unwrap();
            }
        }
    })
    .unwrap();
}

#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_pair_cost"]
fn pair_cost() {
    run(false, [0, 0]);
}

#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_pair_cost"]
fn pair_cost_without_the_drain() {
    run(true, [0, 0]);
}

#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_pair_cost"]
fn pair_cost_unpacker0_x2() {
    run(false, [1, 0]);
}

#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_pair_cost"]
fn pair_cost_unpacker0_x4() {
    run(false, [2, 0]);
}

#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_pair_cost"]
fn pair_cost_both_x2() {
    run(false, [1, 1]);
}

/// Every pair order (`TT_PAIR_ORDERS`, comma-separated `faces,phases,auto,srca,srcb,tiles`; all by
/// default), each product exact.
#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_pair_cost::pair_orders"]
fn pair_orders() {
    use tt_kernels::matmul::PairOrder;
    let orders: Vec<Option<PairOrder>> = match std::env::var("TT_PAIR_ORDERS") {
        Err(_) => PairOrder::ALL.iter().copied().map(Some).collect(),
        Ok(v) => v
            .split(',')
            .map(|o| {
                Some(match o.trim() {
                    "faces" => PairOrder::Faces,
                    "phases" => PairOrder::PhasesOuter,
                    "auto" => PairOrder::Auto,
                    "srca" => PairOrder::SrcAWhole,
                    "srcb" => PairOrder::SrcBWhole,
                    "tiles" => PairOrder::Tiles,
                    other => panic!("TT_PAIR_ORDERS: {other}"),
                })
            })
            .collect(),
    };
    run_orders(false, [0, 0], &orders);
}

/// [`pair_orders`] with `A`'s gathers left out, then every gather (`TT_PAIR_ORDERS` as there):
/// what of a pair is the kernel's, and what the mover's. The products are wrong.
#[test]
#[ignore = "a probe: cargo xtask bench --filter silicon_probe_pair_cost::without_gathers"]
fn without_gathers() {
    for skip in [Skip::A, Skip::All] {
        *SKIP.lock().unwrap() = skip;
        pair_orders();
    }
}
