//! A buffer read transposed as a matmul's `B` again and again is transposed once
//! (`docs/plans/replay-fixed-cost.md`, P2).
//!
//! The mover reads a tile transposed one datum at a time (`READ_TRANSPOSED`: the tile into its
//! scratch slot, then a software transpose), about 10 us a tile on silicon, so SmolLM2's tied output
//! head, `x @ E^T` over a `[49152, 576]` table, spent 2.3 ms of every 36 ms decode step reading `E`
//! transposed. Now (`burn-tt` `DramRef::plain_for_matmul`) the second matmul that reads a buffer
//! transposed as `B` makes its transpose a plain matrix with one block copy and keeps it beside the
//! buffer, and every later one reads that.
//!
//! The claims:
//! - the product is the same bits on every use: read transposed (the first), after the copy (the
//!   second) and from the kept copy (the third), and equal to the product with `B` made plain
//!   explicitly;
//! - the first use queues the matmul alone, the second one more job (the copy), the third the matmul
//!   alone again;
//! - a buffer written in place is never read from a copy kept before the write: a trace that takes
//!   the weight as its input, captured after two eager uses had kept a copy, replays new weights
//!   into its product (negative control: with the written-in-place check removed, the replays read
//!   the stale copy and this fails).

use burn::tensor::{Tensor, TensorData, TensorPrimitive};
use burn_tt::{InputPayload, OutputPayload, TracedInference, TtBackend};
use tt_tests::burn_device::{with_device, Config};

/// `n` finite values from `seed`, varied in sign and magnitude.
fn values(seed: u64, n: usize) -> Vec<f32> {
    // Odd, and a different one for each seed.
    let mut s = seed.wrapping_mul(2).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as i32 - (1 << 23)) as f32 / (1 << 20) as f32
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn on_card(x: &Tensor<TtBackend, 2>) -> burn_tt::TtTensor {
    let TensorPrimitive::Float(p) = x.clone().into_primitive() else {
        unreachable!()
    };
    p.ensure_resident();
    p
}

fn read(x: Tensor<TtBackend, 2>) -> Vec<f32> {
    x.into_data().to_vec::<f32>().unwrap()
}

/// Jobs `f` queued on the server thread, and its result.
fn jobs<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let (r, ops) = burn_tt::with_report(f);
    (r, ops.0.iter().map(|s| s.jobs).sum())
}

// `M` and `N` ragged, `K` whole tiles and not.
const SHAPES: [[usize; 3]; 2] = [[5, 96, 70], [33, 40, 64]];

#[test]
fn a_weight_used_transposed_is_transposed_once() {
    with_device(Config::default(), |d| {
        for [m, k, n] in SHAPES {
            let x =
                Tensor::<TtBackend, 2>::from_data(TensorData::new(values(1, m * k), [m, k]), &d);
            let w =
                Tensor::<TtBackend, 2>::from_data(TensorData::new(values(2, n * k), [n, k]), &d);
            on_card(&x);
            on_card(&w);
            // `B` made plain explicitly, on its own buffer: the product to equal.
            let plain = Tensor::<TtBackend, 2>::from_data(w.to_data(), &d)
                .transpose()
                .mul_scalar(1.0);
            let want = read(x.clone().matmul(plain));
            let mut queued = Vec::new();
            for use_ in 1..=3 {
                let (got, n_jobs) = jobs(|| read(x.clone().matmul(w.clone().transpose())));
                assert_eq!(bits(&got), bits(&want), "[{m}, {k}, {n}]: use {use_}");
                queued.push(n_jobs);
            }
            assert_eq!(
                queued[1],
                queued[0] + 1,
                "[{m}, {k}, {n}]: the second use adds the copy: {queued:?}"
            );
            assert_eq!(
                queued[2], queued[0],
                "[{m}, {k}, {n}]: the third reads the kept copy: {queued:?}"
            );
        }
    });
}

#[test]
fn a_weight_written_in_place_is_never_read_from_a_kept_copy() {
    with_device(Config::default(), |d| {
        let [m, k, n] = SHAPES[0];
        let x = Tensor::<TtBackend, 2>::from_data(TensorData::new(values(1, m * k), [m, k]), &d);
        let w = Tensor::<TtBackend, 2>::from_data(TensorData::new(values(2, n * k), [n, k]), &d);
        let (xp, wp) = (on_card(&x), on_card(&w));
        // Two eager uses: a copy of `w^T` is kept.
        for _ in 0..2 {
            let _ = read(x.clone().matmul(w.clone().transpose()));
        }
        // A trace whose input is `w`: each replay writes it.
        let (trace, _) = TracedInference::capture(&[&xp, &wp], || {
            let TensorPrimitive::Float(y) =
                x.clone().matmul(w.clone().transpose()).into_primitive()
            else {
                unreachable!()
            };
            vec![y]
        })
        .unwrap();
        for seed in [3, 4] {
            let w2 = values(seed, n * k);
            let want = read(
                x.clone().matmul(
                    Tensor::<TtBackend, 2>::from_data(TensorData::new(w2.clone(), [n, k]), &d)
                        .transpose(),
                ),
            );
            let got = trace
                .run(vec![
                    InputPayload::F32(values(1, m * k)),
                    InputPayload::F32(w2),
                ])
                .unwrap();
            let OutputPayload::F32(got) = &got[0] else {
                panic!("an F32 product")
            };
            assert_eq!(bits(got), bits(&want), "a replay with weights {seed}");
        }
    });
}
