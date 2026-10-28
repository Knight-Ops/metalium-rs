//! A weight stored BF16 (`Session::store_bf16`, `tensor::Elem::BF16`: half the bytes) multiplies
//! as the same weight in FP32, bit for bit, on every path a product takes: whole tiles, a thin
//! last tile row, `K` blocks, a packed one-row `A`, a streamed `K`, 2-D blocks with `K` streamed,
//! ragged edges, and a capture whose products read their weights ahead on idle units
//! (`docs/plans/bf16-weight-storage.md` A2). Its `B` is read from BF16 slots into 2,112-byte L1
//! slots (`record::GATHER_B_BF16`, A5b: every layout steps `B` by `SrcRoute::b_slot`, the looped
//! 2-D block kernel by a register of its own, so both block kernels are run) and unpacked as BF16
//! (`SrcRoute::TF32_WITH_BF16_B`, gate `step232`).
//! The control stores a weight with mantissa bits below BF16's, which the storage rounds away.
use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, Spread, TileChoice};
use tt_kernels::tensor::{BlockStream, DramTensor, Eltwise};
use tt_ttsim::fork_scope;

/// Values in about [-1, 1] with every FP32 mantissa bit live.
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// [`values`] rounded to BF16 (ties-even): a BF16 checkpoint's weights widened to FP32.
fn bf16_values(seed: u64, n: usize) -> Vec<f32> {
    values(seed, n)
        .into_iter()
        .map(|v| {
            let b = v.to_bits();
            let r = b + 0x7fff + ((b >> 16) & 1);
            f32::from_bits(r & 0xffff_0000)
        })
        .collect()
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}: element {i}: {g} vs {w}");
    }
}

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(8),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        s.set_batching(true).unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// `w` (BF16-valued) stored BF16: converted on the card on silicon ([`Session::store_bf16`]),
/// and as its bits from the host on ttsim, which cannot run the conversion's BF16 pack.
fn store<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &DramTensor,
    values: &[f32],
) -> DramTensor {
    if cfg!(feature = "silicon") {
        s.store_bf16(w).unwrap()
    } else {
        let bits: Vec<u16> = values.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        s.upload_bf16_weight(&bits, w.rows, w.cols).unwrap()
    }
}

fn mm<T: tt_device::Transport>(s: &mut Session<T>, a: &DramTensor, b: &DramTensor) -> DramTensor {
    s.matmul_dram(
        a,
        false,
        b,
        false,
        SrcRoute::Tf32FromFp32,
        Fidelity::HiFi4,
        1 << 40,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// One product: `(m, k, n)`, the K-block limit, the block-stream mode, and what it takes.
type Case = (
    usize,
    usize,
    usize,
    Option<usize>,
    BlockStream,
    &'static str,
);

/// Each path's product with the weight in FP32 and stored BF16: the same bits.
#[test]
fn a_stored_bf16_weight_multiplies_as_its_fp32_values() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        // (m, k, n, the K-block limit, the block-stream mode, what it takes)
        let cases: [Case; 8] = [
            (64, 256, 96, None, BlockStream::Off, "whole K, slots"),
            (
                37,
                100,
                70,
                None,
                BlockStream::Off,
                "ragged in every dimension",
            ),
            (5, 256, 64, None, BlockStream::Off, "a thin last tile row"),
            (64, 256, 96, Some(2), BlockStream::Off, "K blocks"),
            (1, 1024, 256, None, BlockStream::Off, "a packed one-row A"),
            (
                1,
                1024,
                128,
                Some(8),
                BlockStream::Off,
                "a packed A, K streamed",
            ),
            (
                96,
                256,
                128,
                None,
                BlockStream::Fixed([2, 2, 2]),
                "2-D blocks, K streamed",
            ),
            (
                70,
                288,
                100,
                None,
                BlockStream::Fixed([2, 3, 2]),
                "2-D blocks, ragged",
            ),
        ];
        for (seed, (m, k, n, limit, block, what)) in cases.into_iter().enumerate() {
            let seed = seed as u64;
            let a = s.upload(&values(100 + seed, m * k), m, k).unwrap();
            let wv = bf16_values(200 + seed, k * n);
            let w = s.upload(&wv, k, n).unwrap();
            let stored = store(s, &w, &wv);
            assert_eq!(stored.elem, tt_kernels::tensor::Elem::BF16);
            s.set_matmul_k_block_limit(limit.and_then(std::num::NonZeroUsize::new));
            s.set_block_stream(block);
            // 2-D blocks: the looped kernel and the unrolled one.
            let loops: &[bool] = match block {
                BlockStream::Off => &[true],
                _ => &[true, false],
            };
            for &looped in loops {
                tt_kernels::matmul::set_block_loops(looped);
                let want = mm(s, &a, &w);
                let got = mm(s, &a, &stored);
                let (want_v, got_v) = (s.download(&want).unwrap(), s.download(&got).unwrap());
                assert!(want_v.iter().any(|&v| v != 0.0), "{what}: an empty product");
                assert_bits(&format!("{what}, looped {looped}"), &got_v, &want_v);
                s.free(want).unwrap();
                s.free(got).unwrap();
            }
            tt_kernels::matmul::set_block_loops(true);
            // The stored weight widens back to the values it was stored from, after the
            // products (watched failing: the widening kept a product's unpacker `Z` counter).
            let widened = s.widen_bf16(&stored).unwrap();
            assert_bits(
                &format!("{what}: widened"),
                &s.download(&widened).unwrap(),
                &s.download(&w).unwrap(),
            );
            for t in [a, w, stored, widened] {
                s.free(t).unwrap();
            }
        }
        s.set_matmul_k_block_limit(None);
        s.set_block_stream(BlockStream::Auto);

        // The control: a weight with bits below BF16's is not the weight its storage holds.
        let (m, k, n) = (32, 256, 64);
        let a = s.upload(&values(1, m * k), m, k).unwrap();
        let wv = values(2, k * n);
        let w = s.upload(&wv, k, n).unwrap();
        // On ttsim the host's truncation stands for the card's rounding: either loses the bits.
        let stored = store(s, &w, &wv);
        let (want, got) = (mm(s, &a, &w), mm(s, &a, &stored));
        assert_ne!(
            s.download(&got).unwrap(),
            s.download(&want).unwrap(),
            "BF16 storage must round the low bits away"
        );
        // A stored weight is read as it lies: transposed it is refused, and so is any route
        // but TF32's.
        assert!(s
            .matmul_dram(
                &a,
                false,
                &stored,
                true,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                1 << 40
            )
            .is_err());
        assert!(s
            .matmul_dram(
                &a,
                false,
                &stored,
                false,
                SrcRoute::Bf16FromFp32,
                Fidelity::HiFi4,
                1 << 40
            )
            .is_err());
        for t in [a, w, stored, want, got] {
            s.free(t).unwrap();
        }
    });
}

const K: usize = 1024;

/// `x @ w0`, a scale, `@ w1`, a relu, `@ w2`, the element-wise ops on units 0-3 so units 4-7
/// read the next product's weights ahead (`step205`'s chain).
fn forward<T: tt_device::Transport>(
    s: &mut Session<T>,
    w: &[DramTensor; 3],
    x: &DramTensor,
) -> DramTensor {
    let ew = |s: &mut Session<T>, kind, scalar, a: &DramTensor| {
        s.eltwise(
            Eltwise {
                kind,
                scalar,
                scalar2: 0.0,
            },
            a,
            None,
        )
        .unwrap()
    };
    let h = mm(s, x, &w[0]);
    let hs = ew(s, kind::MUL_SCALAR, 0.5, &h);
    let g = mm(s, &hs, &w[1]);
    let gr = ew(s, kind::RELU, 0.0, &g);
    let y = mm(s, &gr, &w[2]);
    for t in [h, hs, g, gr] {
        s.free(t).unwrap();
    }
    y
}

/// A captured chain of one-row products whose stored BF16 weights are read ahead on idle units
/// replays as the FP32 weights' chain, over new inputs.
#[test]
fn prefetched_bf16_weights_replay_as_fp32_weights() {
    with_session(|s| {
        s.set_spread(Spread::Fixed(4));
        let wv = [(1, K), (2, K), (3, 256)].map(|(seed, n)| bf16_values(seed, K * n));
        let w = [0, 1, 2].map(|i| s.upload(&wv[i], K, wv[i].len() / K).unwrap());
        let stored = [0, 1, 2].map(|i| store(s, &w[i], &wv[i]));
        let x = s.upload(&values(10, K), 1, K).unwrap();
        let mut runs = Vec::new();
        for weights in [&w, &stored] {
            let warm = forward(s, weights, &x);
            s.free(warm).unwrap();
            s.write(&x, &values(10, K)).unwrap();
            s.begin_trace().unwrap();
            let y = forward(s, weights, &x);
            let id = s.end_trace().unwrap();
            let read_ahead = s
                .trace_ops(id)
                .unwrap()
                .iter()
                .filter(|op| op.what.contains("matmul") || op.what == tt_kernels::tensor::PREFETCH)
                .any(|op| op.entries[4..8].iter().any(|r| !r.is_empty()));
            assert!(read_ahead, "the idle units read the weights ahead");
            let mut replays = vec![s.download(&y).unwrap()];
            for round in 0..2u64 {
                s.write(&x, &values(20 + round, K)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                replays.push(s.download(&y).unwrap());
            }
            s.release_trace(id).unwrap();
            s.free(y).unwrap();
            runs.push(replays);
        }
        for (i, (got, want)) in runs[1].iter().zip(&runs[0]).enumerate() {
            assert_bits(&format!("run {i}"), got, want);
        }
        for t in w.into_iter().chain(stored).chain([x]) {
            s.free(t).unwrap();
        }
    });
}
