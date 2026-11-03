//! X9's weight layout (`Session::set_weight_layout`, `docs/plans/decode-bytes-in-flight.md` Phase
//! B, roadmap Track 2 follow-ups item 2b): a capture copies each unit's share of a one-row
//! product's weight into a region contiguous in one channel, in the order the unit reads it, and
//! the product's records read the region in a few large reads (`record::GATHER` over a one-channel
//! `B`).
//!
//! The claims, for `down`'s schedule (`K` streamed in chunks, a ring of two) and `gate|up`'s (`K`
//! whole), with an FP32 `B` and a stored BF16 one:
//! - **Bit for bit.** A replay of the relaid product equals the eager product and the replay of the
//!   same capture without the layout, bit for bit, every replay.
//! - **The weight is relaid.** The regions hold exactly the weight's slots.
//! - **A write reaches the replay.** After the weight is written in place, a replay equals the
//!   product over the new weight (its regions filled again first).
//! - **The replay reads the regions** (the negative control): with the refill suppressed
//!   (`probe_stale_relaid`), a replay after a write still gives the old product.
//! - **A capture that writes its relaid weight is refused**, where the same capture without the
//!   layout is not.
//! - **A full GDDR leaves the weight where it lies.** Where a region would leave its channel less
//!   than `RELAID_HEADROOM` free, the capture still succeeds (its own streams and programs stored),
//!   relays nothing, and its replays give the same bits; with the room back, the same capture
//!   relays the weight (the control).
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{probe_stale_relaid, Session, TileChoice};
use tt_kernels::tensor::DramTensor;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

/// Values from an odd seed (`seed | 1`: seeds 2k and 2k + 1 give the same values).
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
    use tt_tests::backend::GATE_TILE;
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

/// Every tile of the card: products dealt over many units, each with a region of its own.
#[cfg(feature = "silicon")]
fn with_session(f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::All,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `B` as the product reads it: FP32, or the same values' top halves stored BF16.
fn weight<T: tt_device::Transport>(
    s: &mut Session<T>,
    seed: u64,
    k: usize,
    n: usize,
    bf16: bool,
) -> DramTensor {
    let v = floats(seed, k * n);
    if bf16 {
        let half: Vec<u16> = v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
        s.upload_bf16_weight(&half, k, n).unwrap()
    } else {
        s.upload(&v, k, n).unwrap()
    }
}

fn route(bf16: bool) -> SrcRoute {
    if bf16 {
        SrcRoute::TF32_WITH_BF16_B
    } else {
        SrcRoute::Tf32FromFp32
    }
}

fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
    bf16: bool,
) -> DramTensor {
    s.matmul_dram(a, false, b, false, route(bf16), Fidelity::HiFi4, BUDGET)
        .unwrap()
}

/// The eager product's bits.
fn eager<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
    bf16: bool,
) -> Vec<u32> {
    let out = product(s, a, b, bf16);
    let got = bits(&s.download(&out).unwrap());
    s.free(out).unwrap();
    got
}

/// One product captured (with the layout `on` or off), its output after each of `replays`
/// replays, and the trace and output to release.
fn traced<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
    bf16: bool,
    on: bool,
) -> (tt_kernels::trace::TraceId, DramTensor) {
    s.set_weight_layout(on).unwrap();
    s.begin_trace().unwrap();
    let out = product(s, a, b, bf16);
    let id = s.end_trace().unwrap();
    s.set_weight_layout(false).unwrap();
    (id, out)
}

fn replayed<T: tt_device::Transport>(
    s: &mut Session<T>,
    id: tt_kernels::trace::TraceId,
    out: &DramTensor,
) -> Vec<u32> {
    s.replay(id).unwrap();
    bits(&s.download(out).unwrap())
}

#[test]
fn a_relaid_weight_gives_the_products_bits() {
    with_session(|s| {
        for (k, n, streamed) in [(8192, 64, true), (2048, 256, false)] {
            for bf16 in [false, true] {
                let what = format!("[1, {k}] @ [{k}, {n}], K streamed {streamed}, BF16 {bf16}");
                s.set_streamed_k(streamed);
                let a = s.upload(&floats(k as u64, k), 1, k).unwrap();
                let b = weight(s, n as u64, k, n, bf16);
                let want = eager(s, &a, &b, bf16);
                assert!(want.iter().any(|&w| w != 0), "{what}: a product");
                let (plain, plain_out) = traced(s, &a, &b, bf16, false);
                assert_eq!(
                    replayed(s, plain, &plain_out),
                    want,
                    "{what}: unrelaid replay"
                );
                let before = s.relaid_bytes();
                let (id, out) = traced(s, &a, &b, bf16, true);
                let relaid = s.relaid_bytes() - before;
                let slot = if bf16 { 2112 } else { 4160 };
                println!("{what}: {relaid} bytes relaid");
                assert_eq!(
                    relaid,
                    (k / 32 * (n / 32)) as u64 * slot,
                    "{what}: the weight's slots, once each"
                );
                for r in 0..3 {
                    assert_eq!(replayed(s, id, &out), want, "{what}: replay {r}");
                }
                s.release_trace(id).unwrap();
                s.release_trace(plain).unwrap();
                for t in [a, b, out, plain_out] {
                    s.free(t).unwrap();
                }
                assert_eq!(s.relaid_bytes(), before, "{what}: freed with the weight");
            }
        }
        s.set_streamed_k(true);
    });
}

#[test]
fn a_write_to_a_relaid_weight_reaches_its_replays() {
    with_session(|s| {
        let (k, n) = (8192, 64);
        let a = s.upload(&floats(1, k), 1, k).unwrap();
        let b = weight(s, 21, k, n, false);
        let (id, out) = traced(s, &a, &b, false, true);
        let first = eager(s, &a, &b, false);
        assert_eq!(replayed(s, id, &out), first, "before any write");
        // Written in place: the replay's regions are filled again first.
        s.write(&b, &floats(33, k * n)).unwrap();
        let second = eager(s, &a, &b, false);
        assert_ne!(second, first, "a different weight");
        assert_eq!(replayed(s, id, &out), second, "after a write");
        // The negative control: with the refill suppressed, the replay reads what the regions
        // held, the weight before this write.
        probe_stale_relaid(true);
        s.write(&b, &floats(45, k * n)).unwrap();
        let stale = replayed(s, id, &out);
        probe_stale_relaid(false);
        assert_eq!(
            stale, second,
            "the replay reads the regions, not the weight"
        );
        assert_ne!(eager(s, &a, &b, false), second, "the weight did change");
        s.release_trace(id).unwrap();
        for t in [a, b, out] {
            s.free(t).unwrap();
        }
    });
}

#[test]
fn a_capture_that_writes_its_relaid_weight_is_refused() {
    with_session(|s| {
        let (k, n) = (1024, 64);
        let a = s.upload(&floats(5, k), 1, k).unwrap();
        let b = weight(s, 6, k, n, false);
        let c = weight(s, 9, k, n, false);
        for on in [false, true] {
            s.set_weight_layout(on).unwrap();
            s.begin_trace().unwrap();
            let out = product(s, &a, &b, false);
            s.copy_into(&c, &b).unwrap();
            let ended = s.end_trace();
            s.set_weight_layout(false).unwrap();
            match (on, ended) {
                (false, Ok(id)) => s.release_trace(id).unwrap(),
                (true, Err(e)) => assert!(e.to_string().contains("relaid"), "{e}"),
                (on, r) => panic!("layout {on}: {:?}", r.map(|_| ())),
            }
            s.free(out).unwrap();
        }
        for t in [a, b, c] {
            s.free(t).unwrap();
        }
    });
}

#[test]
fn a_full_gddr_leaves_the_weight_where_it_lies() {
    use tt_kernels::tensor::Elem;
    with_session(|s| {
        let (k, n) = (2048, 256);
        s.set_streamed_k(false);
        let a = s.upload(&floats(7, k), 1, k).unwrap();
        let b = weight(s, 11, k, n, false);
        let want = eager(s, &a, &b, false);
        // Room for the capture's own tensors and stored streams (8 MiB, well under
        // `RELAID_HEADROOM` a channel), given back once GDDR is otherwise full.
        let room = s.empty(1024, 2048, Elem::F32).unwrap();
        let mut fill = Vec::new();
        let (mut rows, mut cols) = (8192, 32768);
        loop {
            while let Ok(t) = s.empty(rows, cols, Elem::F32) {
                fill.push(t);
            }
            if cols > 32 {
                cols /= 2;
            } else if rows > 32 {
                rows /= 2;
            } else {
                break;
            }
        }
        s.free(room).unwrap();
        let (free, total) = s.dram_free_bytes_by_channel().unwrap();
        println!(
            "filled: {} tensors, {free} bytes free on the fullest channel, {total} in all",
            fill.len()
        );
        let before = s.relaid_bytes();
        let (id, out) = traced(s, &a, &b, false, true);
        assert_eq!(s.relaid_bytes(), before, "no room: nothing relaid");
        for r in 0..2 {
            assert_eq!(replayed(s, id, &out), want, "unrelaid replay {r}");
        }
        s.release_trace(id).unwrap();
        s.free(out).unwrap();
        for t in fill {
            s.free(t).unwrap();
        }
        // The control: with the room back, the same capture relays the weight.
        let (id, out) = traced(s, &a, &b, false, true);
        assert!(s.relaid_bytes() > before, "room again: relaid");
        assert_eq!(replayed(s, id, &out), want, "relaid replay");
        s.release_trace(id).unwrap();
        for t in [a, b, out] {
            s.free(t).unwrap();
        }
        s.set_streamed_k(true);
    });
}
