//! The roles' batch tables (`tt_isa::mailbox::BATCH_TIMES`), the half of the profile mode's
//! pipeline detail (`tt_kernels::profile_mode::Detail::Pipeline`) that runs on ttsim: the stamp
//! stream that carries them to the host does not (divergence row 54), but the clock the roles read
//! is modelled. Plan: `docs/completed-plans/decode-product-meter.md`.
//!
//! The claims, for a one-row product with `K` streamed in chunks (`down`'s schedule and `K`, a ring
//! of two batches) and one with `K` whole in several column blocks (`gate|up`'s, a ring of one):
//! - **One entry a batch.** T0's and T2's counts both equal the batches the product's pair ran
//!   (`Session::dataflow_stats`).
//! - **In the order the protocol allows.** In each batch T0 waits, sees the batch, then pops it,
//!   and T2 waits, gets a slot, then pushes; T0 sees a batch before T2 pushes it; and a batch
//!   starts only after the one before it retired, so T0's wait for batch `i + 1` follows its pop
//!   of batch `i`, and T2's its push.
//! - **Only when asked.** With the tables off, a later product leaves them as they were, word for
//!   word (the negative control).
//!
//! Only a `profile` image's roles keep the tables, so the gate needs `tt-tests/profile`.
#![cfg(feature = "profile")]

use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
#[cfg(not(feature = "silicon"))]
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

fn floats(seed: u64, n: usize) -> Vec<f32> {
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

#[cfg(not(feature = "silicon"))]
fn with_session(f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
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

/// One unit of the card, so a product's every batch is that unit's.
#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::First,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// `b` happened no earlier than `a` on the tile's counter (its low word, which wraps).
fn after(a: u32, b: u32) -> bool {
    (b.wrapping_sub(a) as i32) >= 0
}

/// `[1, k] @ [k, n]`, `K` streamed or whole; returns the batches its pair ran.
fn product<T: tt_device::Transport>(s: &mut Session<T>, k: usize, n: usize, streamed: bool) -> u64 {
    let a = s.upload(&floats(k as u64, k), 1, k).unwrap();
    let b = s.upload(&floats(n as u64, k * n), k, n).unwrap();
    s.set_streamed_k(streamed);
    let before = s.dataflow_stats().batches;
    let out = s
        .matmul_dram(
            &a,
            false,
            &b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET,
        )
        .unwrap();
    s.download(&out).unwrap();
    let batches = s.dataflow_stats().batches - before;
    s.set_streamed_k(true);
    for t in [a, b, out] {
        s.free(t).unwrap();
    }
    batches
}

#[test]
fn each_batch_is_recorded_once_in_the_protocols_order() {
    with_session(|s| {
        // Started units first: the tables are asked of the roles a unit has.
        product(s, 1024, 32, true);
        s.probe_keep_batch_tables(true);
        let mut kept = None;
        for (k, n, streamed) in [(8192, 32, true), (1024, 256, false)] {
            let what = format!("[1, {k}] @ [{k}, {n}], K streamed {streamed}");
            let batches = product(s, k, n, streamed);
            let tables = s.probe_batch_tables().unwrap();
            let [(n0, t0), (n2, t2)] = &tables[0];
            println!("{what}: {batches} batches; T0 {n0}, T2 {n2}");
            assert!(batches >= 2, "{what}: a pipeline, not one batch");
            assert_eq!(
                (*n0 as u64, *n2 as u64),
                (batches, batches),
                "{what}: the counts"
            );
            assert_eq!((t0.len(), t2.len()), (batches as usize, batches as usize));
            for (i, (a, c)) in t0.iter().zip(t2).enumerate() {
                let at = format!("{what}: batch {i}");
                assert!(after(a[0], a[1]) && after(a[1], a[2]), "{at}: T0 {a:?}");
                assert!(after(c[0], c[1]) && after(c[1], c[2]), "{at}: T2 {c:?}");
                assert!(after(a[1], c[2]), "{at}: T0 saw it before T2 pushed it");
                if let (Some(a1), Some(c1)) = (t0.get(i + 1), t2.get(i + 1)) {
                    assert!(after(a[2], a1[0]), "{at}: T0 popped before waiting again");
                    assert!(after(c[2], c1[0]), "{at}: T2 pushed before waiting again");
                }
            }
            kept = Some(tables);
        }
        // Off: a product with another count leaves the tables as the last one left them.
        s.probe_keep_batch_tables(false);
        let batches = product(s, 8192, 64, true);
        let tables = s.probe_batch_tables().unwrap();
        assert_ne!(
            tables[0][0].0 as u64, batches,
            "the control must run another count"
        );
        assert_eq!(Some(tables), kept, "tables written though not asked for");
    });
}
