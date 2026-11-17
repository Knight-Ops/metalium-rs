//! RISCV NC's entry log (`tt_isa::dm::nc_log`), the half of the profile mode's sender-side meter
//! that runs on ttsim: the stamp stream does not (divergence row 54), but NC's log is plain stores
//! to L1 and the clock it reads is modelled. Plan: `docs/plans/prefill-operand-reuse.md` (W2's
//! sender-side meter).
//!
//! The claims, on a multicast product over a 2 x 3 grid (`tile_op::multicast`):
//! - **Every strip is logged once, in the multicast's order.** Each sender's NC takes a transfer
//!   batch, waits for its readers' frees, multicasts the strip, counts the run's acknowledgements,
//!   raises its ready word and copies it into its readers, then pops the batch; the ready words
//!   raised over every unit's log are the strips the product multicast (`multicast::strips`), and
//!   only units that send have them.
//! - **The log is whole and in time order**: no record lost, each list's records from its anchor
//!   to its end in clock order, and every unit's output batches taken.
//! - **Only when armed** (the negative control): disarmed, a later product leaves every log empty.
//!
//! On silicon, also the meter it feeds: the profile mode's pipeline detail tells every pair of a
//! traced multicast product, the senders' too (their transfer credits, which once broke the join,
//! named apart by stream), and each sender's NC took as many strips as its B handed it.
//!
//! Only a `profile` image logs, so the gate needs `tt-tests/profile`.
#![cfg(feature = "profile")]

use tt_device::trace::TraceEvent;
use tt_isa::dataflow::{Action, Stream};
use tt_isa::dm::nc_log as l;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::profile::Event;
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{BlockStream, Reuse};
use tt_kernels::tile_op::multicast;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(choice: TileChoice, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(dev, tt_firmware_images::ROLES, choice, |_, _| Ok(None)).unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

#[cfg(feature = "silicon")]
fn with_session(choice: TileChoice, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            choice,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 73 + seed) % 257) as f32 / 131.0 - 1.0)
        .collect()
}

/// `[128, k] @ [k, 288]` multicasting both operands; returns the strips it multicast.
fn product<T: tt_device::Transport>(s: &mut Session<T>, k: usize) -> u64 {
    let a = s.upload(&values(128 * k, 3), 128, k).unwrap();
    let b = s.upload(&values(k * 288, 4), k, 288).unwrap();
    s.set_operand_reuse(Reuse::Multicast { a: true, b: true })
        .unwrap();
    let before = multicast::strips();
    let out = s
        .matmul_dram(
            &a,
            false,
            &b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET * 40,
        )
        .unwrap();
    s.sync().unwrap();
    let strips = multicast::strips() - before;
    for t in [a, b, out] {
        s.free(t).unwrap();
    }
    strips
}

/// A unit's log as `(tag, time)` records, each list's times from its own anchor (the counter's low
/// word: a list is far shorter than its wrap).
fn records(log: &[u32], lost: u32) -> Vec<(u32, u64)> {
    let out = tt_kernels::profile_mode::splice_nc_log(Vec::new(), log, lost, u64::from(u32::MAX));
    out.iter()
        .filter_map(|e: &TraceEvent| match Event::decode(e.token) {
            Event::NcRecord { tag } => Some((tag, e.cycles)),
            _ => None,
        })
        .collect()
}

fn is_cb(tag: u32, stream: Stream, action: Action) -> bool {
    tag == l::tag(l::CB, stream.word() << 2 | action.word())
}

#[test]
fn every_strip_is_logged_once_in_the_multicasts_order() {
    with_session(TileChoice::Rect { cols: 3, rows: 2 }, |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 1]));
        // Started units first: the log is armed through each unit's mover.
        product(s, 256);
        s.probe_nc_log(true).unwrap();
        let strips = product(s, 512);
        assert!(strips > 0, "nothing was multicast");
        let logs = s.probe_take_nc_logs().unwrap();
        let (mut raised, mut senders) = (0u64, 0);
        for (u, (log, lost)) in logs.iter().enumerate() {
            let what = format!("unit {u}");
            assert_eq!(*lost, 0, "{what}: records lost");
            let r = records(log, *lost);
            assert!(!r.is_empty(), "{what}: nothing logged");
            if std::env::var("TT_DUMP_NC_LOG").is_ok() {
                let tags: Vec<String> = r
                    .iter()
                    .map(|&(t, c)| format!("{:x}.{:x}@{}", l::kind(t), l::arg(t), c))
                    .collect();
                eprintln!("{what}: {}", tags.join(" "));
            }
            // Each list from its anchor to its end, in clock order.
            let mut open = false;
            for (i, &(tag, t)) in r.iter().enumerate() {
                match l::kind(tag) {
                    l::ANCHOR if l::arg(tag) == 0 => {
                        assert!(!open, "{what}: record {i}: a list inside a list");
                        open = true;
                    }
                    l::END => {
                        assert!(open, "{what}: record {i}: an end outside a list");
                        open = false;
                    }
                    _ => assert!(open, "{what}: record {i}: outside a list"),
                }
                if i > 0 && l::kind(tag) != l::ANCHOR {
                    assert!(
                        r[i - 1].1 <= t,
                        "{what}: record {i} before the one before it"
                    );
                }
            }
            assert!(!open, "{what}: a list not ended");
            let stats = tt_kernels::profile_mode::NcStats::of(&r, 0);
            assert!(stats.outputs > 0, "{what}: no output batch taken");
            // Each strip: transfer wait, then the frees' waits, the multicast, its acks, the ready
            // word raised and copied, then the pop. A chunk with no readers has nothing between.
            let mut ready = 0u64;
            let mut i = 0;
            while i < r.len() {
                if !is_cb(r[i].0, Stream::Transfer, Action::Wait) {
                    i += 1;
                    continue;
                }
                let end = (i + 1..r.len())
                    .find(|&j| {
                        is_cb(r[j].0, Stream::Transfer, Action::Pop) || l::kind(r[j].0) == l::END
                    })
                    .unwrap_or_else(|| panic!("{what}: a transfer at record {i} never ended"));
                let kinds: Vec<u32> = r[i + 1..end].iter().map(|&(t, _)| l::kind(t)).collect();
                if l::kind(r[end].0) == l::END {
                    // A plain transfer (its tiles written out, the pop the write record's own):
                    // no strip.
                    assert!(
                        kinds.iter().all(|&k| k == l::WRITE),
                        "{what}: record {i}: a transfer logged as {kinds:?}"
                    );
                    i = end + 1;
                    continue;
                }
                let strip = [l::WAIT_CREDIT, l::MCAST, l::SETTLE, l::CREDIT, l::MCAST];
                let waits = kinds.iter().take_while(|&&k| k == l::WAIT_CREDIT).count();
                assert!(
                    kinds.is_empty() || (waits >= 1 && kinds[waits - 1..] == strip),
                    "{what}: record {i}: a strip logged as {kinds:?}"
                );
                ready += kinds.iter().filter(|&&k| k == l::CREDIT).count() as u64;
                i = end + 1;
            }
            if ready > 0 {
                senders += 1;
            }
            assert_eq!(ready > 0, stats.multicast > 0, "{what}: {stats:?}");
            raised += ready;
        }
        assert_eq!(
            raised, strips,
            "ready words raised against strips multicast"
        );
        assert!(senders >= 2, "{senders} senders");

        // The negative control: disarmed, a later product logs nothing.
        s.probe_nc_log(false).unwrap();
        assert!(product(s, 256) > 0);
        let logs = s.probe_take_nc_logs().unwrap();
        assert!(
            logs.iter().all(|(log, lost)| log.is_empty() && *lost == 0),
            "logged though disarmed"
        );
    });
}

/// The meter itself: the pipeline detail over a traced multicast product tells every pair, the
/// senders' too, and each sender's NC took the strips its B handed it.
#[cfg(feature = "silicon")]
#[test]
fn the_pipeline_detail_tells_the_senders() {
    use tt_kernels::profile_mode::{Detail, ProfileOptions};
    with_session(TileChoice::Rect { cols: 3, rows: 2 }, |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 1]));
        s.set_operand_reuse(Reuse::Multicast { a: true, b: true })
            .unwrap();
        let k = 512;
        let a = s.upload(&values(128 * k, 3), 128, k).unwrap();
        let b = s.upload(&values(k * 288, 4), k, 288).unwrap();
        let options = ProfileOptions {
            detail: Detail::Pipeline,
            ..ProfileOptions::default()
        };
        s.profile_mode(Some(options)).unwrap();
        s.begin_trace().unwrap();
        let out = s
            .matmul_dram(
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET * 40,
            )
            .unwrap();
        let id = s.end_trace().unwrap();
        s.profile_mode(None).unwrap();
        s.profile_mode(Some(options)).unwrap();
        for _ in 0..3 {
            s.replay(id).unwrap();
        }
        s.sync().unwrap();
        let report = s.profile_mode(None).unwrap().expect("the mode was on");
        s.release_trace(id).unwrap();
        for t in [a, b, out] {
            s.free(t).unwrap();
        }
        let t = report.traces.values().next().expect("one trace");
        assert_eq!(t.overflowed, 0);
        assert_eq!(t.pipe_dropped, [0; 7], "a pair not told");
        let sample = t
            .pipes
            .iter()
            .find_map(|p| p.first())
            .expect("the product sampled");
        assert!(sample.sender_units >= 2, "{sample:?}");
        let mut senders = 0;
        // The product's units (NC's log also keeps the trace's plain transfers, with no batches).
        for (u, _, _, batches, nc) in t.pipe_batches.iter().flatten().filter(|u| !u.3.is_empty()) {
            let sent = batches.iter().filter(|b| b.sent()).count() as u64;
            assert_eq!(nc.lost, 0, "unit {u}: NC records lost");
            assert_eq!(
                sent, nc.strips,
                "unit {u}: strips B handed NC against NC took"
            );
            if sent > 0 {
                senders += 1;
                for b in batches.iter().filter(|b| b.sent()) {
                    assert!(
                        b.granted <= b.transfer[0] && b.transfer[3] <= b.issued,
                        "unit {u}: a transfer outside its batch {b:?}"
                    );
                }
            }
        }
        assert_eq!(senders, sample.sender_units, "senders told");
        eprintln!("{}", report.summary(&|id| id.to_string()));
    });
}
