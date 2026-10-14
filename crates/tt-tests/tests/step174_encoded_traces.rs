//! Host-encoded reads in traces (`DataMover::trace_form`): the fast mode's saving for a replay.
//!
//! A trace's entries are streamed from GDDR by the mover's `CALL`, and a trace is captured once and replayed
//! many times, so the host encodes its plain reads once, at capture, in the trace's checksum domain
//! (`encoded::Domain::Trace`) -- the replay's reads then run in the tight loop at ~90 cycles, not the safe
//! decode's ~360, with no host work at all. This holds, in a `fast-reads` build:
//! - what the capture stores really is encoded (read back from GDDR), in the trace's domain;
//! - a replay is the fresh ops bit for bit (`step39` holds that for every trace; this is the encoded case);
//! - a stored entry damaged after the capture is refused by the card (`error::ENCODED`) and nothing runs
//!   past it, and an entry written in the list's domain is refused in a trace (and the reverse, in
//!   `tt_isa::dm::encoded`'s tests): the two domains do not mix;
//!
//! and in the default build, that a trace stores plain reads exactly as before.

use tt_kernels::kind;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{DramTensor, Eltwise};
use tt_tests::backend::GATE_TILE;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_tiles(choice: TileChoice, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
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
fn with_tiles(choice: TileChoice, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
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

fn gate_tile() -> TileChoice {
    TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1)
}

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// The layer's weights, resident.
struct Layer {
    w1: DramTensor,
    b1: DramTensor,
    w2: DramTensor,
}

impl Layer {
    fn upload<T: tt_device::Transport>(s: &mut Session<T>) -> Self {
        Layer {
            w1: s.upload(&values(2, 784 * 128), 784, 128).unwrap(),
            b1: s.upload(&values(3, 128), 1, 128).unwrap(),
            w2: s.upload(&values(4, 128 * 10), 128, 10).unwrap(),
        }
    }

    /// The forward pass on `x`; every intermediate freed.
    fn forward<T: tt_device::Transport>(&self, s: &mut Session<T>, x: &DramTensor) -> DramTensor {
        let mm = |s: &mut Session<T>, a, b| {
            s.matmul_dram(
                a,
                false,
                b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                1 << 40,
            )
            .unwrap()
        };
        let ew = |s: &mut Session<T>, kind, a, b| {
            s.eltwise(
                Eltwise {
                    scalar2: 0.0,
                    kind,
                    scalar: 0.0,
                },
                a,
                b,
            )
            .unwrap()
        };
        let h = mm(s, x, &self.w1);
        let hb = ew(s, kind::ADD_ROW, &h, Some(&self.b1));
        let r = ew(s, kind::RELU, &hb, None);
        let y = mm(s, &r, &self.w2);
        for t in [h, hb, r] {
            s.free(t).unwrap();
        }
        y
    }
}

const ENCODED: u32 = tt_isa::dm::op::READ_ENCODED;

/// How many stored entries are encoded reads, and how many plain single-request reads, across a unit's
/// stream and streaming regions.
fn census(parts: &[Vec<[u32; 8]>]) -> (usize, usize) {
    let (mut encoded, mut plain) = (0, 0);
    for part in parts {
        let mut i = 0;
        while i < part.len() {
            match part[i][0] {
                ENCODED => encoded += 1,
                tt_isa::dm::op::READ => plain += 1,
                _ => {}
            }
            i += tt_isa::dm::record::len(part[i][0]).max(1);
        }
    }
    (encoded, plain)
}

#[cfg(feature = "fast-reads")]
fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}, element {i}: {g:e} against {w:e}"
        );
    }
}

/// Capture the layer's forward pass on `[64, 784]`: the layer, its input, its output and the trace.
fn capture<T: tt_device::Transport>(
    s: &mut Session<T>,
) -> (Layer, DramTensor, DramTensor, tt_kernels::trace::TraceId) {
    let layer = Layer::upload(s);
    let x = s.upload(&values(1, 64 * 784), 64, 784).unwrap();
    let warm = layer.forward(s, &x);
    s.free(warm).unwrap();
    s.begin_trace().unwrap();
    let y = layer.forward(s, &x);
    let id = s.end_trace().unwrap();
    (layer, x, y, id)
}

/// The default build: a trace stores its entries as the capture produced them, none encoded.
#[cfg(not(feature = "fast-reads"))]
#[test]
fn a_safe_image_stores_no_encoded_reads() {
    with_tiles(gate_tile(), |s| {
        let (_, _, _, id) = capture(s);
        let (encoded, _) = census(&s.trace_entries(id, 0).unwrap());
        assert_eq!(
            encoded, 0,
            "a safe image has no encoded reads, in a trace or out"
        );
        s.release_trace(id).unwrap();
    });
}

#[cfg(feature = "fast-reads")]
mod fast {
    use super::*;
    use tt_isa::dm::{encoded, error, op};

    /// What the capture stores is encoded, in the trace's domain and no other: each stored encoded read
    /// validates as a trace's (for this mover's context) and fails as a list's; the records it replaced
    /// are gone from the reader sections and the stream; and the replay is the fresh ops, bit for bit.
    #[test]
    fn a_capture_stores_its_reads_encoded_in_the_traces_domain() {
        with_tiles(gate_tile(), |s| {
            let (layer, x, y, id) = capture(s);
            let (x, y) = (&x, &y);
            let ctx = s.encode_context(0).expect("a mover");
            let parts = s.trace_entries(id, 0).unwrap();
            let (n, _) = census(&parts);
            assert!(n >= 100, "the capture stored only {n} encoded reads");
            for part in &parts {
                let mut i = 0;
                while i < part.len() {
                    let e = part[i];
                    if e[0] == op::READ_ENCODED {
                        assert_eq!(encoded::validate_for_trace(&ctx, e), Ok(()), "{e:x?}");
                        assert_eq!(
                            encoded::validate(&ctx, e),
                            Err(error::ENCODED),
                            "a trace's entry passed as a list's: {e:x?}"
                        );
                    }
                    // A read record that expands statically does not survive.
                    assert!(
                        !matches!(
                            e[0],
                            tt_isa::dm::record::GATHER | tt_isa::dm::record::READ_RUN
                        ),
                        "a static read record was stored unexpanded: {e:x?}"
                    );
                    i += tt_isa::dm::record::len(e[0]).max(1);
                }
            }
            for round in 0..2u64 {
                s.write(x, &values(10 + round, 64 * 784)).unwrap();
                s.replay(id).unwrap();
                s.sync().unwrap();
                let replayed = s.download(y).unwrap();
                let fresh = layer.forward(s, x);
                assert_bits(
                    &format!("replay {round}"),
                    &replayed,
                    &s.download(&fresh).unwrap(),
                );
                s.free(fresh).unwrap();
            }
            s.release_trace(id).unwrap();
        });
    }

    /// The first stored encoded read and where it is: (part, entry).
    fn first_encoded(parts: &[Vec<[u32; 8]>]) -> (usize, usize) {
        for (p, part) in parts.iter().enumerate() {
            let mut i = 0;
            while i < part.len() {
                if part[i][0] == op::READ_ENCODED {
                    return (p, i);
                }
                i += tt_isa::dm::record::len(part[i][0]).max(1);
            }
        }
        panic!("no encoded read stored");
    }

    fn refused_by_the_card<T: tt_device::Transport>(
        s: &mut Session<T>,
        id: tt_kernels::trace::TraceId,
    ) {
        // The replay is queued; the card's refusal comes back at the sync.
        s.replay(id).unwrap();
        let r = s.sync();
        let text = format!("{:?}", r.expect_err("the card ran a damaged trace"));
        assert!(
            text.contains(&format!("code: {}", error::ENCODED))
                || text.contains(&format!("code {}", error::ENCODED)),
            "refused, but not as an encoded read: {text}"
        );
    }

    /// A stored entry damaged after the capture -- a bit of its checksum -- is refused by the card at the
    /// replay: it runs nothing it should not, and says why. The control is the same trace undamaged.
    #[test]
    fn a_damaged_stored_entry_is_refused_by_the_card() {
        with_tiles(gate_tile(), |s| {
            let (_, x, _, id) = capture(s);
            s.write(&x, &values(20, 64 * 784)).unwrap();
            s.replay(id).unwrap();
            s.sync().unwrap();
            let parts = s.trace_entries(id, 0).unwrap();
            let (p, i) = first_encoded(&parts);
            let sum = parts[p][i][encoded::CHECK_WORD];
            s.poke_trace_entry(id, 0, p, i, encoded::CHECK_WORD, sum ^ 1)
                .unwrap();
            refused_by_the_card(s, id);
        });
    }

    /// An encoded read written into a trace in the *list's* domain -- the checksum a list entry would carry,
    /// otherwise a good request -- is refused in a trace: the card verifies a trace's entries in the
    /// trace's domain, so a list's entry, or a trace's in a list (`tt_isa::dm::encoded`'s tests), does not run.
    #[test]
    fn an_entry_in_the_lists_domain_is_refused_in_a_trace() {
        with_tiles(gate_tile(), |s| {
            let (_, _, _, id) = capture(s);
            let ctx = s.encode_context(0).unwrap();
            let parts = s.trace_entries(id, 0).unwrap();
            let (p, i) = first_encoded(&parts);
            let mut e = parts[p][i];
            e[encoded::CHECK_WORD] = encoded::check_with(&e, ctx.seed(op::READ_ENCODED));
            s.poke_trace_entry(id, 0, p, i, encoded::CHECK_WORD, e[encoded::CHECK_WORD])
                .unwrap();
            refused_by_the_card(s, id);
        });
    }
}
