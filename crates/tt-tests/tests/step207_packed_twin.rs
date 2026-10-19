//! A one-row element-wise result written packed by the op that makes it
//! (`docs/plans/decode-to-bandwidth.md` Phase 3, `Session::set_packed_twins`): a `[1, K]` F32 result
//! over a long `K` also writes each output tile's data row into a `[K / 32, 32]` twin from the
//! slots its scatter writes out (`tensor::packed_twin_writes`), so a product's packed `A` and
//! RMSNorm's statistic read it with no packing op.
//!
//! On eight units, the claims:
//! - **The twin is the packing.** It equals `pack_row` of the result, row for row, bit for bit, at
//!   `K` 1024 (a whole packed tile), 1056 (a ragged one) and 2048, for a plain op and an op over
//!   column blocks (`eltwise_blocks`).
//! - **Its readers are the packed paths' bits, with no packing op.** A product of the result and
//!   its statistic equal the same with twins off (each packing first), and run no transfer packet.
//! - **A twin never outlives its row.** After the result is written in place (`write`,
//!   `copy_into`, `write_rows`), its statistic and product are the new row's; freeing the result
//!   gives the twin's GDDR back.
//! - **In a trace,** the capture and its replays over new inputs are the ops run fresh.
//!
//! Watched failing: with `Session::drop_twin` doing nothing (the statistic after a write reads the
//! old row's twin), and with the twin's rows written one face row off.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{BlockOperand, DramTensor, Eltwise};
use tt_ttsim::fork_scope;

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
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
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn add() -> Eltwise {
    Eltwise {
        kind: kind::ADD,
        scalar: 0.0,
        scalar2: 0.0,
    }
}

/// The twin's data rows (`K / 32` of them) as the host reads them.
fn twin_rows<T: tt_device::Transport>(s: &mut Session<T>, y: &DramTensor) -> Vec<u32> {
    let twin = s.packed_twin(y).expect("a twin").clone();
    bits(&s.download(&twin).unwrap())
}

fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    w: &DramTensor,
) -> Vec<u32> {
    let p = s
        .matmul_dram(
            a,
            false,
            w,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            1 << 40,
        )
        .unwrap();
    let v = bits(&s.download(&p).unwrap());
    s.free(p).unwrap();
    v
}

fn statistic<T: tt_device::Transport>(s: &mut Session<T>, a: &DramTensor) -> u32 {
    let r = s.rms(a, 1e-5).unwrap();
    let v = s.download(&r).unwrap()[0].to_bits();
    s.free(r).unwrap();
    v
}

/// Claims one and two.
#[test]
fn the_twin_is_the_packing_and_its_readers_pack_nothing() {
    with_session(|s| {
        for k in [1024usize, 1056, 2048] {
            let (x, z) = (values(k as u64, k), values(k as u64 + 1, k));
            let w = s.upload(&values(7, k * 64), k, 64).unwrap();
            let (dx, dz) = (s.upload(&x, 1, k).unwrap(), s.upload(&z, 1, k).unwrap());
            let wide = s.upload(&values(9, 2 * k), 1, 2 * k).unwrap();
            for blocks in [false, true] {
                let label = format!("K {k}, blocks {blocks}");
                let y = if blocks {
                    // The right half of a `[1, 2K]` row plus `z`: a fused projection's half.
                    s.eltwise_blocks(
                        add(),
                        BlockOperand {
                            tensor: &wide,
                            at: [0, k],
                            dims: [1, k],
                        },
                        Some(BlockOperand::whole(&dz)),
                        None,
                    )
                    .unwrap()
                } else {
                    s.eltwise(add(), &dx, Some(&dz)).unwrap()
                };
                let packed = s.pack_row(&y).unwrap();
                let want = bits(&s.download(&packed).unwrap());
                assert_eq!(twin_rows(s, &y), want, "{label}: the twin");
                s.free(packed).unwrap();
                let before = s.dataflow_stats().transfer_packets;
                let (p, r) = (product(s, &y, &w), statistic(s, &y));
                assert_eq!(
                    s.dataflow_stats().transfer_packets,
                    before,
                    "{label}: the readers packed nothing"
                );
                // The same reads with twins off: each reader packs the row first.
                s.set_packed_twins(false);
                let before = s.dataflow_stats().transfer_packets;
                let (p1, r1) = (product(s, &y, &w), statistic(s, &y));
                assert!(
                    s.dataflow_stats().transfer_packets > before,
                    "{label}: twins off, the readers pack"
                );
                s.set_packed_twins(true);
                assert_eq!((p, r), (p1, r1), "{label}: twin against packing");
                s.free(y).unwrap();
            }
            for t in [w, dx, dz, wide] {
                s.free(t).unwrap();
            }
        }
    });
}

/// Claim three.
#[test]
fn a_twin_never_outlives_its_row() {
    with_session(|s| {
        let k = 2048;
        let free0 = s.dram_free_bytes();
        let w = s.upload(&values(7, k * 64), k, 64).unwrap();
        let dz = s.upload(&values(2, k), 1, k).unwrap();
        let y = s.eltwise(add(), &dz, Some(&dz)).unwrap();
        // Each way of writing `y` in place, then its readers against a fresh copy of the new row.
        for way in ["write", "copy_into", "write_rows"] {
            let new = values(way.len() as u64 + 40, k);
            match way {
                "write" => s.write(&y, &new).unwrap(),
                "copy_into" => {
                    let src = s.upload(&new, 1, k).unwrap();
                    s.copy_into(&src, &y).unwrap();
                    s.free(src).unwrap();
                }
                _ => {
                    let src = s.upload(&new, 1, k).unwrap();
                    s.write_rows(&y, &src, &[(0, 0)]).unwrap();
                    s.free(src).unwrap();
                }
            }
            assert!(s.packed_twin(&y).is_none(), "{way}: the twin dropped");
            let fresh = s.upload(&new, 1, k).unwrap();
            assert_eq!(
                statistic(s, &y),
                statistic(s, &fresh),
                "{way}: the statistic"
            );
            assert_eq!(
                product(s, &y, &w),
                product(s, &fresh, &w),
                "{way}: the product"
            );
            s.free(fresh).unwrap();
        }
        let y2 = s.eltwise(add(), &dz, Some(&dz)).unwrap();
        assert!(s.packed_twin(&y2).is_some());
        for t in [y, y2, w, dz] {
            s.free(t).unwrap();
        }
        s.sync().unwrap();
        assert_eq!(s.dram_free_bytes(), free0, "every twin's GDDR given back");
    });
}

/// Claim four: `x + z`, its statistic and a product of it, captured and replayed over new `x`.
#[test]
fn a_captured_twin_replays_as_the_ops_run_fresh() {
    with_session(|s| {
        let k = 2048;
        let w = s.upload(&values(7, k * 64), k, 64).unwrap();
        let dz = s.upload(&values(2, k), 1, k).unwrap();
        let x = s.upload(&values(10, k), 1, k).unwrap();
        let step = |s: &mut Session<_>| {
            let y = s.eltwise(add(), &x, Some(&dz)).unwrap();
            let r = s.rms(&y, 1e-5).unwrap();
            let p = s
                .matmul_dram(
                    &y,
                    false,
                    &w,
                    false,
                    SrcRoute::Tf32FromFp32,
                    Fidelity::HiFi4,
                    1 << 40,
                )
                .unwrap();
            s.free(y).unwrap();
            (r, p)
        };
        let mut fresh = Vec::new();
        for seed in [10u64, 20, 21] {
            s.write(&x, &values(seed, k)).unwrap();
            let (r, p) = step(s);
            fresh.push((
                bits(&s.download(&r).unwrap()),
                bits(&s.download(&p).unwrap()),
            ));
            s.free(r).unwrap();
            s.free(p).unwrap();
        }
        s.write(&x, &values(10, k)).unwrap();
        s.begin_trace().unwrap();
        let (r, p) = step(s);
        let id = s.end_trace().unwrap();
        let mut got = vec![(
            bits(&s.download(&r).unwrap()),
            bits(&s.download(&p).unwrap()),
        )];
        for seed in [20u64, 21] {
            s.write(&x, &values(seed, k)).unwrap();
            s.replay(id).unwrap();
            s.sync().unwrap();
            got.push((
                bits(&s.download(&r).unwrap()),
                bits(&s.download(&p).unwrap()),
            ));
        }
        assert_eq!(got, fresh, "the capture and its replays");
        s.release_trace(id).unwrap();
        for t in [r, p, w, dz, x] {
            s.free(t).unwrap();
        }
    });
}
