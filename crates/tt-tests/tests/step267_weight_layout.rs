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
//! - **A padding fill in place reaches the regions.** After an op zeroes the weight's padding in
//!   place (`meet`), a replay's padding columns are the eager product's.
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

/// A relaid weight whose padding a later op zeroes in place (`meet`, for a product that needs
/// it zero along its `K`): its regions must hold the weight's bytes, padding included, so a later
/// replay's padding columns are the eager product's. (A relaid product's output claims undefined
/// padding today, since its `A` is packed, so no claim depends on this yet; a product of more rows
/// reading the regions would.)
#[test]
fn a_padding_fill_in_place_reaches_the_regions() {
    use tt_kernels::tensor::{Eltwise, Pad};
    with_session(|s| {
        let (k, n) = (2048usize, 250usize);
        let width = n.div_ceil(32) * 32;
        s.set_streamed_k(false);
        let a = s.upload(&floats(13, k), 1, k).unwrap();
        // `exp` of a zero-padded weight: its padding columns hold `e^0`, not zero.
        let raw = weight(s, 17, k, n, false);
        let exp = Eltwise {
            kind: tt_kernels::sfpu::ops::kind_sfpu::EXP,
            scalar: 0.0,
            scalar2: 0.0,
        };
        let b = s.eltwise(exp, &raw, None).unwrap();
        s.free(raw).unwrap();
        assert_eq!(b.pad(), Pad::Undefined, "the weight's padding is not zero");
        // `B`'s `K` is whole tiles, so the product reads it as it is and relays it, padding and all.
        let before = s.relaid_bytes();
        let (id, out) = traced(s, &a, &b, false, true);
        assert!(s.relaid_bytes() > before, "relaid");
        let row0 = |s: &mut Session<_>, t: &DramTensor| -> Vec<u32> {
            bits(&s.download_padded(t).unwrap()[..width])
        };
        s.replay(id).unwrap();
        let eager_out = product(s, &a, &b, false);
        assert_eq!(row0(s, &out), row0(s, &eager_out), "before the fill");
        assert!(
            row0(s, &out)[n..].iter().any(|&x| x != 0),
            "the padding columns hold the weight's padding's image"
        );
        s.free(eager_out).unwrap();
        // A product over `B`'s ragged columns needs them zero: filled in place.
        let c = weight(s, 19, n, 32, false);
        let fill = product(s, &b, &c, false);
        assert_eq!(b.pad(), Pad::Zero, "filled in place");
        s.replay(id).unwrap();
        let eager_out = product(s, &a, &b, false);
        let (got, want) = (row0(s, &out), row0(s, &eager_out));
        assert!(want[n..].iter().all(|&x| x == 0), "eager: zero padding");
        assert_eq!(
            got[n..],
            want[n..],
            "after the fill: the regions hold the weight's padding"
        );
        assert_eq!(got, want, "after the fill: the product");
        s.release_trace(id).unwrap();
        for t in [a, b, c, fill, out, eager_out] {
            s.free(t).unwrap();
        }
        s.set_streamed_k(true);
    });
}

/// Phase C0's probe (`tensor::probe_relaid_blocks`, the bound on one copy instead of two): once a
/// decode capture has relaid a weight, products of more rows -- eager and captured -- read it from
/// its regions, a gather of a block cut into reads of its runs, and give the products' bits. Every
/// tile they read comes from a region (coverage), and the control (each weight's first region read
/// one slot off, `probe_relaid_shift`) gives other bits.
#[test]
fn products_of_more_rows_read_a_relaid_weight_from_its_regions() {
    use tt_kernels::tensor::{
        probe_relaid_audit, probe_relaid_blocks, probe_relaid_blocks_counts, probe_relaid_shift,
    };
    with_session(|s| {
        for (k, n, streamed) in [(8192, 64, true), (2048, 256, false)] {
            for bf16 in [false, true] {
                let what =
                    format!("[64, {k}] @ [{k}, {n}], decode K streamed {streamed}, BF16 {bf16}");
                s.set_streamed_k(streamed);
                let row = s.upload(&floats(k as u64 + 1, k), 1, k).unwrap();
                let a = s.upload(&floats(k as u64 + 3, 64 * k), 64, k).unwrap();
                let b = weight(s, n as u64 + 5, k, n, bf16);
                let want = eager(s, &a, &b, bf16);
                // The decode capture that relays `B`.
                tt_kernels::tensor::probe_relaid_audit_reset();
                let (decode, decode_out) = traced(s, &row, &b, bf16, true);
                assert!(s.relaid_bytes() > 0, "{what}: relaid");
                probe_relaid_blocks(true);
                assert_eq!(
                    eager(s, &a, &b, bf16),
                    want,
                    "{what}: eager, from the regions"
                );
                let (id, out) = traced(s, &a, &b, bf16, false);
                for r in 0..2 {
                    assert_eq!(replayed(s, id, &out), want, "{what}: replay {r}");
                }
                let [split, records, regions, lies] = probe_relaid_blocks_counts();
                let ([blocks, stream], widths) = probe_relaid_audit();
                println!(
                    "{what}: {split} gathers became {records}; {regions} bytes from regions, {lies} as they lie; \
                     coverage {blocks:?} / {stream:?}, widths {widths:?}"
                );
                assert!(
                    split > 0 && regions > 0,
                    "{what}: the products read the regions"
                );
                assert_eq!(lies, 0, "{what}: every tile from a region");
                let c = if streamed { stream } else { blocks };
                assert_eq!((c[2], c[3]), (c[1], 0), "{what}: every tile in one region");
                // The control: each weight's first region read one slot off.
                probe_relaid_blocks(true);
                probe_relaid_shift(true);
                let shifted = eager(s, &a, &b, bf16);
                probe_relaid_shift(false);
                probe_relaid_blocks(false);
                assert_ne!(
                    shifted, want,
                    "{what}: a region read one slot off is caught"
                );
                s.release_trace(id).unwrap();
                s.release_trace(decode).unwrap();
                for t in [row, a, b, out, decode_out] {
                    s.free(t).unwrap();
                }
            }
        }
        s.set_streamed_k(true);
    });
}
