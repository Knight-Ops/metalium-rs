//! Weights laid by column (`Session::set_column_weights`, `docs/plans/decode-bytes-in-flight.md`
//! Phase C1, roadmap Track 2 follow-ups item 2b, "the scalable form"): a stored weight's column
//! `c` is consecutive slots on channel `c % n`, the weight's only copy. A one-row product takes
//! each of its columns into L1 by column, in large pieces (`record::REF_L1_BY_COLUMN`); a block
//! reads it a tile at a time, as it reads an interleaved weight.
//!
//! The claims, for `down`'s schedule (`K` streamed in chunks) and `gate|up`'s (`K` whole), with a
//! weight stored BF16 and one stored BFP8:
//! - **Bit for bit.** A one-row product over the weight laid by column equals the same product over
//!   the weight interleaved, eager and in every replay of a capture; so does a block's (`m` = 64
//!   and a ragged 40).
//! - **The weight is what was stored.** Read back through its slots it is the interleaved weight's
//!   bits, and widened it is the same F32 tensor.
//! - **The card reads it by its layout** (the negative control): named with its channels rotated
//!   by one (`probe_column_shift`), each column is read from another's channel, and the products
//!   differ.
//! - **Only where it costs no GDDR.** A weight whose columns do not fill every channel's region
//!   (5 columns over 8 channels) is stored interleaved (`Layout::for_weight`), so each weight here
//!   has a multiple of 8 columns.
use tt_kernels::bfp::BfpFormat;
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{probe_column_shift, DramTensor, Layout};
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

/// Every tile of the card: products dealt over many units, prefetching units among them.
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

/// How a weight is stored.
#[derive(Clone, Copy, Debug)]
enum Stored {
    Bf16,
    Bfp8,
}

impl Stored {
    fn route(self) -> SrcRoute {
        match self {
            Stored::Bf16 => SrcRoute::TF32_WITH_BF16_B,
            Stored::Bfp8 => {
                SrcRoute::tf32_with_b(tt_isa::format::TileFormat::new(BfpFormat::Bfp8.l1_format()))
            }
        }
    }
}

/// `[k, n]` from `seed` stored as `how`, laid as `layout`.
fn weight<T: tt_device::Transport>(
    s: &mut Session<T>,
    seed: u64,
    [k, n]: [usize; 2],
    how: Stored,
    layout: Layout,
) -> DramTensor {
    s.set_column_weights(layout == Layout::ByColumn).unwrap();
    let v = floats(seed, k * n);
    let w = match how {
        Stored::Bf16 => {
            let half: Vec<u16> = v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
            s.upload_bf16_weight(&half, k, n).unwrap()
        }
        Stored::Bfp8 => {
            let f = s.upload(&v, k, n).unwrap();
            let w = s.store_bfp(&f, BfpFormat::Bfp8).unwrap();
            s.free(f).unwrap();
            w
        }
    };
    s.set_column_weights(false).unwrap();
    assert_eq!(w.placement.layout(), layout, "stored as asked");
    w
}

fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
    how: Stored,
) -> DramTensor {
    s.matmul_dram(a, false, b, false, how.route(), Fidelity::HiFi4, BUDGET)
        .unwrap()
}

/// The eager product's bits.
fn eager<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    b: &DramTensor,
    how: Stored,
) -> Vec<u32> {
    let out = product(s, a, b, how);
    let got = bits(&s.download(&out).unwrap());
    s.free(out).unwrap();
    got
}

/// The schedules: `down`'s (`K` streamed) and `gate|up`'s (`K` whole).
const SHAPES: [(usize, usize, bool); 2] = [(8192, 256, true), (2048, 256, false)];

fn stores() -> [Stored; 2] {
    [Stored::Bf16, Stored::Bfp8]
}

#[test]
fn a_one_row_product_over_a_weight_laid_by_column_gives_its_bits() {
    with_session(|s| {
        for (k, n, streamed) in SHAPES {
            for how in stores() {
                let what = format!("[1, {k}] @ [{k}, {n}], K streamed {streamed}, {how:?}");
                s.set_streamed_k(streamed);
                let a = s.upload(&floats(k as u64, k), 1, k).unwrap();
                let flat = weight(s, n as u64, [k, n], how, Layout::Interleaved);
                let cols = weight(s, n as u64, [k, n], how, Layout::ByColumn);
                let want = eager(s, &a, &flat, how);
                assert!(want.iter().any(|&w| w != 0), "{what}: a product");
                assert_eq!(eager(s, &a, &cols, how), want, "{what}: eager");
                s.begin_trace().unwrap();
                let out = product(s, &a, &cols, how);
                let id = s.end_trace().unwrap();
                for r in 0..3 {
                    s.replay(id).unwrap();
                    assert_eq!(bits(&s.download(&out).unwrap()), want, "{what}: replay {r}");
                }
                s.release_trace(id).unwrap();
                for t in [a, flat, cols, out] {
                    s.free(t).unwrap();
                }
            }
        }
        s.set_streamed_k(true);
    });
}

#[test]
fn a_block_over_a_weight_laid_by_column_gives_its_bits() {
    with_session(|s| {
        let (k, n) = (2048, 256);
        for how in stores() {
            let flat = weight(s, 7, [k, n], how, Layout::Interleaved);
            let cols = weight(s, 7, [k, n], how, Layout::ByColumn);
            for m in [64, 40] {
                let what = format!("[{m}, {k}] @ [{k}, {n}], {how:?}");
                let a = s.upload(&floats(m as u64, m * k), m, k).unwrap();
                let want = eager(s, &a, &flat, how);
                assert!(want.iter().any(|&w| w != 0), "{what}: a product");
                assert_eq!(eager(s, &a, &cols, how), want, "{what}");
                s.free(a).unwrap();
            }
            s.free(flat).unwrap();
            s.free(cols).unwrap();
        }
    });
}

#[test]
fn a_weight_laid_by_column_holds_what_was_stored() {
    with_session(|s| {
        let (k, n) = (96, 512);
        for how in stores() {
            let what = format!("[{k}, {n}], {how:?}");
            let flat = weight(s, 3, [k, n], how, Layout::Interleaved);
            let cols = weight(s, 3, [k, n], how, Layout::ByColumn);
            let widen = |s: &mut Session<_>, w: &DramTensor| {
                let f = match how {
                    Stored::Bf16 => s.widen_bf16(w).unwrap(),
                    Stored::Bfp8 => s.bfp_to_f32(&w.as_bfp().unwrap()).unwrap(),
                };
                let v = bits(&s.download(&f).unwrap());
                s.free(f).unwrap();
                v
            };
            let want = widen(s, &flat);
            assert_eq!(widen(s, &cols), want, "{what}: widened");
            if let Stored::Bf16 = how {
                let raw = |s: &mut Session<_>, w: &DramTensor| {
                    s.download_bf16(&w.as_bf16().unwrap()).unwrap()
                };
                let want = raw(s, &flat);
                assert_eq!(raw(s, &cols), want, "{what}: its BF16 bits");
            }
            s.free(flat).unwrap();
            s.free(cols).unwrap();
        }
    });
}

/// The negative control: the same weight named with its channels rotated is read from other
/// columns' slots, one-row product and block alike.
#[test]
fn a_weight_named_with_its_channels_rotated_gives_other_bits() {
    with_session(|s| {
        for (k, n, streamed) in SHAPES {
            let what = format!("[{k}, {n}], K streamed {streamed}");
            s.set_streamed_k(streamed);
            let how = Stored::Bf16;
            let cols = weight(s, 11, [k, n], how, Layout::ByColumn);
            // A block of `down`'s depth is slow on the simulator; `gate|up`'s covers blocks.
            let rows: &[usize] = if streamed { &[1] } else { &[1, 64] };
            for &m in rows {
                let a = s.upload(&floats(m as u64 + 1, m * k), m, k).unwrap();
                let want = eager(s, &a, &cols, how);
                probe_column_shift(true);
                let shifted = eager(s, &a, &cols, how);
                probe_column_shift(false);
                assert_ne!(
                    shifted, want,
                    "{what}, m {m}: the rotated names read other tiles"
                );
                assert_eq!(eager(s, &a, &cols, how), want, "{what}, m {m}: restored");
                s.free(a).unwrap();
            }
            s.free(cols).unwrap();
        }
        s.set_streamed_k(true);
    });
}

/// A weight of 5 columns over 8 channels would leave 3 channels' regions a column empty, so it
/// is stored interleaved with the layout on; one of 8 columns is laid by column.
#[test]
fn a_weight_whose_columns_do_not_fill_the_channels_stays_interleaved() {
    with_session(|s| {
        s.set_column_weights(true).unwrap();
        for (n, want) in [(160, Layout::Interleaved), (256, Layout::ByColumn)] {
            let w = s.upload_bf16_weight(&vec![0x3f80; 96 * n], 96, n).unwrap();
            assert_eq!(w.placement.layout(), want, "[96, {n}]");
            s.free(w).unwrap();
        }
        s.set_column_weights(false).unwrap();
    });
}
