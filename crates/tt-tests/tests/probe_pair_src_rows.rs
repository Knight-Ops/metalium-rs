//! Probes for a K-tile pair that holds an operand's tile whole in its `Src` (`docs/plans/prefill-operand-reuse.md`
//! P4, B3): each a product run alone on one unit, exact against host integer sums and bit for bit
//! the faces order (`matmul::PairOrder::Faces`), at one to four phases, `B` transposed or stored
//! BF16, one, sixteen and all of a tile's rows, on the other paths and blocked.
//!
//! - **`B` whole in `SrcA`** (`PairOrder::SrcAWhole`): one 1,024-datum unpack writes `SrcA` rows
//!   0-63 (`UNPACR_Regular.md`: `Row &= 63` with `SetOvrdWithAddr` on Blackhole), and `MVMUL` reads
//!   16 of them from `RWCs.SrcA` at 16, 32 and 48, stepped by address modifiers (`SrcACR`).
//! - **`A` whole in `SrcB`** (`PairOrder::SrcBWhole`): unpacker 1 writes `SrcB` rows 0-63, and
//!   `MVMUL` reads 8 of them from `RWCs.SrcB` at 8 to 56 (`SrcBCR`).
//! - **Both** (`PairOrder::Tiles`): two unpacks a pair.
//! - **The throttle override** (`matmul::probe_throttle_override`): `REG1_ovrd_default_throttle_mode`
//!   with unpacker 0 at x1 and x2, under the faces order and both tiles whole.
//!
//! ttsim first (`cargo test -p tt-tests --test probe_pair_src_rows -- --ignored`), then card by
//! card in isolation (`cargo xtask silicon --device N --filter probe_pair_src_rows::<probe>`).
use tt_kernels::matmul::{set_pair_order, Fidelity, PairOrder, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::BlockStream;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(1),
            |_, _| Ok(None),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
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
            TileChoice::Count(1),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn same(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i}: {x} vs {y}");
    }
}

fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &tt_kernels::tensor::DramTensor,
    at: bool,
    b: &tt_kernels::tensor::DramTensor,
    bt: bool,
    fidelity: Fidelity,
) -> Vec<f32> {
    let out = s
        .matmul_dram(a, at, b, bt, SrcRoute::Tf32FromFp32, fidelity, BUDGET * 40)
        .unwrap();
    let v = s.download(&out).unwrap();
    s.free(out).unwrap();
    v
}

/// Integers `[-6, 6]` (exact in every format and sum) or values whose sums round.
fn values(n: usize, seed: usize, exact: bool) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if exact {
                ((i * 7 + seed) % 13) as f32 - 6.0
            } else {
                ((i * 73 + seed) % 257) as f32 / 131.0 - 1.0 + ((i * 7 + seed) % 5) as f32 * 1e-3
            }
        })
        .collect()
}

/// `order` against the host's integer sums and the faces order, over the shapes and modes above.
fn probe<T: tt_device::Transport>(s: &mut Session<T>, order: PairOrder) {
    let (m, k, n) = (96, 160, 96);
    for exact in [true, false] {
        let av = values(m * k, 1, exact);
        let bv = values(k * n, 5, exact);
        let a = s.upload(&av, m, k).unwrap();
        let at_ = s.upload(&av, k, m).unwrap();
        let b = s.upload(&bv, k, n).unwrap();
        let bt_ = s.upload(&bv, n, k).unwrap();
        let bits: Vec<u16> = bv.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let b16 = s.upload_bf16_weight(&bits, k, n).unwrap();
        let one = s.upload(&av[..k], 1, k).unwrap();
        let sixteen = s.upload(&av[..16 * k], 16, k).unwrap();
        let cases: [(&str, &_, bool, &_, bool, usize); 6] = [
            ("A B", &a, false, &b, false, m),
            ("A B^T", &a, false, &bt_, true, m),
            ("A^T B", &at_, true, &b, false, m),
            ("A B16", &a, false, &b16, false, m),
            ("one row", &one, false, &b, false, 1),
            ("sixteen rows", &sixteen, false, &b, false, 16),
        ];
        for fidelity in [
            Fidelity::Lo,
            Fidelity::HiFi2,
            Fidelity::HiFi3,
            Fidelity::HiFi4,
        ] {
            for mode in [BlockStream::Off, BlockStream::Fixed([2, 2, 2])] {
                s.set_block_stream(mode);
                for (what, x, xt, y, yt, rows) in cases {
                    if mode != BlockStream::Off && rows < 64 {
                        continue;
                    }
                    set_pair_order(PairOrder::Faces);
                    let want = product(s, x, xt, y, yt, fidelity);
                    set_pair_order(order);
                    let got = product(s, x, xt, y, yt, fidelity);
                    let what = format!("{order:?}, {what}, {mode:?}, {fidelity:?}");
                    same(&got, &want, &what);
                    if exact && fidelity == Fidelity::HiFi4 && what.contains("A B,") {
                        for i in 0..rows {
                            for j in 0..n {
                                let sum: f32 = (0..k).map(|q| av[i * k + q] * bv[q * n + j]).sum();
                                assert_eq!(got[i * n + j], sum, "{what}: exact at ({i}, {j})");
                            }
                        }
                    }
                }
            }
        }
        s.set_block_stream(BlockStream::Off);
        set_pair_order(PairOrder::Faces);
        for t in [a, at_, b, bt_, b16, one, sixteen] {
            s.free(t).unwrap();
        }
    }
}

#[test]
#[ignore = "a probe: B whole in SrcA rows 0-63"]
fn srca_whole_tile() {
    with_session(|s| probe(s, PairOrder::SrcAWhole));
}

#[test]
#[ignore = "a probe: A whole in SrcB rows 0-63"]
fn srcb_whole_tile() {
    with_session(|s| probe(s, PairOrder::SrcBWhole));
}

#[test]
#[ignore = "a probe: both tiles whole"]
fn both_whole_tiles() {
    with_session(|s| probe(s, PairOrder::Tiles));
}

/// `REG1_ovrd_default_throttle_mode` set for unpacker 0 (`matmul::probe_throttle_override`), at x1
/// then x2, under the faces order and both tiles whole: each setting in a process of its own (the
/// program caches do not key on the probes), exact against the host's integer sums.
#[test]
#[ignore = "a probe: unpacker 0's throttle override"]
fn unpacker0_throttle_override() {
    for throttle in [0, 1] {
        for order in [PairOrder::Faces, PairOrder::Tiles] {
            with_session(|s| {
                tt_kernels::matmul::probe_throttle_override(true);
                tt_kernels::matmul::probe_throttle(throttle, 0);
                set_pair_order(order);
                let (m, k, n) = (96, 160, 96);
                let av = values(m * k, 2, true);
                let bv = values(k * n, 7, true);
                let a = s.upload(&av, m, k).unwrap();
                let b = s.upload(&bv, k, n).unwrap();
                let got = product(s, &a, false, &b, false, Fidelity::HiFi4);
                for i in 0..m {
                    for j in 0..n {
                        let sum: f32 = (0..k).map(|q| av[i * k + q] * bv[q * n + j]).sum();
                        assert_eq!(
                            got[i * n + j],
                            sum,
                            "{order:?}, override, x{}: ({i}, {j})",
                            1 << throttle
                        );
                    }
                }
                s.free(a).unwrap();
                s.free(b).unwrap();
            });
        }
    }
}
