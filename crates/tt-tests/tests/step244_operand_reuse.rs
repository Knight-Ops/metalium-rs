//! Operand reuse across cores (`docs/plans/dataflow-compiler.md` F4b, H1; `tile_op::reuse`): the
//! units of a blocked product that read the same strip of `A` share its reads. A holder reads it
//! from GDDR and credits its readers; each reader copies it from the holder's L1 (`op::READ_L1`)
//! into the slots its own read would fill, credits the holder free, and reads only `B`. With `B`
//! shared too, the units that take the same columns share `B`'s strips the same way
//! (`record::GATHER_B_RESIDENT`), and a unit that copies both reads nothing from GDDR.
//!
//! The claims:
//! - **A shared product is the unshared one bit for bit**, ragged in every dimension, either
//!   operand transposed, at HiFi4 and LoFi, over signed zeros, subnormals, infinities and NaNs,
//!   with one holder a group and with two, `A` shared alone or with `B`, and with units left
//!   idle by `B`'s equal groups: the copies move bytes only.
//! - **Each unit leaves its credit words at zero** (`tt_isa::dm::CREDITS`), so a trace of a shared
//!   product replays as the eager product, again and again.
//! - **The negative control:** copies of either operand that read 16 bytes past their strip
//!   change the product.
//! - **A stalled shared product fails and leaves the session usable:** a reader that waits for a
//!   credit its holder never sends stalls the op until its deadline; the session recovers every
//!   unit (each mover restarted, its credit words zeroed), and the next shared product is the
//!   unshared one bit for bit with every credit word at zero. Silicon only: ttsim cannot recover
//!   a tile whose roles stalled inside a kernel (divergence row 16).
//!
//! What ttsim cannot show: its NoC requests complete before the next is issued (divergence row
//! 72), so a copy cannot race its holder's next read there. The pass's own model test
//! (`tile_op::reuse::tests`) interleaves the units at random and catches a holder that does not
//! wait for its copies; silicon runs the same products with real concurrency.
use tt_device::tlb::WindowKind;
use tt_isa::dm::{credit_word, CREDIT_WORDS};
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{BlockStream, DramTensor, Reuse};
use tt_kernels::tile_op::reuse;
use tt_tests::harness::BUDGET;
use tt_ttsim::fork_scope;

#[cfg(not(feature = "silicon"))]
fn with_session(units: usize, f: impl FnOnce(&mut Session<tt_ttsim::LibTtsim<'_>>)) {
    if let Err(e) = fork_scope(|| {
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
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
fn with_session(units: usize, f: impl FnOnce(&mut Session<tt_kmd::Kmd>)) {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(units),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

/// Values whose sums round, so a product's terms and order show in its bits, with the special
/// values sprinkled in.
fn values(n: usize, seed: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n)
        .map(|i| ((i * 73 + seed) % 257) as f32 / 131.0 - 1.0 + ((i * 7 + seed) % 5) as f32 * 1e-3)
        .collect();
    let special = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(0x007f_ffff),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
    ];
    for (i, s) in special.into_iter().enumerate() {
        v[(i * 977 + seed * 31) % n] = s;
    }
    v
}

fn same(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: element {i}: {x} vs {y}");
    }
}

/// `op(A) @ op(B)` under `reuse`, and how many strips it copied.
#[allow(clippy::too_many_arguments)]
fn product<T: tt_device::Transport>(
    s: &mut Session<T>,
    a: &DramTensor,
    at: bool,
    b: &DramTensor,
    bt: bool,
    fidelity: Fidelity,
    reuse_: Reuse,
) -> (Vec<f32>, u64) {
    s.set_operand_reuse(reuse_).unwrap();
    let before = reuse::copies();
    let out = s
        .matmul_dram(a, at, b, bt, SrcRoute::Tf32FromFp32, fidelity, BUDGET * 40)
        .unwrap();
    let v = s.download(&out).unwrap();
    s.free(out).unwrap();
    (v, reuse::copies() - before)
}

/// Every unit's op-level credit words.
fn credit_words<T: tt_device::Transport>(s: &mut Session<T>) -> Vec<[u32; CREDIT_WORDS as usize]> {
    s.sync().unwrap();
    let tiles = s.tiles();
    let d = s.device();
    let w = d.alloc_window(WindowKind::TwoMib).unwrap();
    let words = tiles
        .iter()
        .map(|&t| std::array::from_fn(|i| d.read32(&w, t, credit_word(i as u32) as u64).unwrap()))
        .collect();
    d.free_window(w);
    words
}

/// `[m, k, n]` on `units` in `[mb, kq, nb]` blocks with at most `holders` holders a group (and
/// `b_holders` a column group, `B` shared too), for each transpose and fidelity: shared equals
/// unshared, and something was shared.
fn case(
    units: usize,
    [m, k, n]: [usize; 3],
    block: [usize; 3],
    holders: usize,
    b_holders: Option<usize>,
) {
    with_session(units, |s| {
        s.set_block_stream(BlockStream::Fixed(block));
        for (at, bt, fidelity) in [
            (false, false, Fidelity::HiFi4),
            (true, false, Fidelity::HiFi4),
            (false, true, Fidelity::Lo),
        ] {
            let what = format!("[{m},{k}]@[{k},{n}] {block:?} on {units}, holders {holders}, {b_holders:?}, A^T {at}, B^T {bt}, {fidelity:?}");
            let (ar, ac) = if at { (k, m) } else { (m, k) };
            let (br, bc) = if bt { (n, k) } else { (k, n) };
            let a = s.upload(&values(ar * ac, 1), ar, ac).unwrap();
            let b = s.upload(&values(br * bc, 2), br, bc).unwrap();
            let (plain, none) = product(s, &a, at, &b, bt, fidelity, Reuse::Off);
            assert_eq!(none, 0, "{what}: unshared copies nothing");
            let reuse_ = Reuse::Pull { holders, b_holders };
            let (shared, copies) = product(s, &a, at, &b, bt, fidelity, reuse_);
            assert!(copies > 0, "{what}: nothing was shared");
            same(&plain, &shared, &what);
            assert!(
                credit_words(s).iter().all(|w| w.iter().all(|&x| x == 0)),
                "{what}: a credit word left counted"
            );
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}

/// Two block rows over four units: two groups of two, each a holder and a reader.
#[test]
fn a_shared_product_is_the_unshared_one_bit_for_bit() {
    case(4, [96, 256, 320], [2, 2, 3], 1, None);
}

/// Ragged in every dimension: the last block row of one tile row, a short last column block, and
/// `K`'s last tile partly padding.
#[test]
fn a_ragged_shared_product_is_the_unshared_one_bit_for_bit() {
    case(4, [72, 232, 300], [2, 2, 3], 1, None);
}

/// Two block rows over six units with two holders a group: each holder a reader of its own.
#[test]
fn two_holders_a_group_give_the_same_bits() {
    case(6, [64, 256, 416], [1, 2, 2], 2, None);
}

/// `B` shared too: two block rows over four units, two groups of two, so each unit holds or
/// copies each operand, and one copies both and reads nothing.
#[test]
fn a_product_sharing_both_operands_is_the_unshared_one_bit_for_bit() {
    case(4, [96, 256, 320], [2, 2, 3], 1, Some(1));
}

/// Both shared, ragged in every dimension.
#[test]
fn a_ragged_product_sharing_both_operands_is_the_unshared_one_bit_for_bit() {
    case(4, [72, 232, 300], [2, 2, 3], 1, Some(1));
}

/// Three block rows over seven units: groups of two and one unit idle, with two holders of
/// `B` a column group of three.
#[test]
fn equal_groups_leave_the_remainder_idle_and_give_the_same_bits() {
    case(7, [160, 256, 416], [2, 2, 2], 1, Some(2));
}

/// A trace of a shared product replays as its eager product, every time, and leaves every credit
/// word at zero.
#[test]
fn a_shared_product_replays_as_its_eager_product() {
    replays(Reuse::Pull {
        holders: 1,
        b_holders: None,
    });
}

/// [`a_shared_product_replays_as_its_eager_product`] with `B` shared too.
#[test]
fn a_product_sharing_both_operands_replays_as_its_eager_product() {
    replays(Reuse::Pull {
        holders: 1,
        b_holders: Some(1),
    });
}

fn replays(reuse_: Reuse) {
    with_session(4, |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 3]));
        let a = s.upload(&values(96 * 256, 3), 96, 256).unwrap();
        let b = s.upload(&values(256 * 320, 4), 256, 320).unwrap();
        let (eager, copies) = product(s, &a, false, &b, false, Fidelity::HiFi4, reuse_);
        assert!(copies > 0);
        s.begin_trace().unwrap();
        let c = s
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
        for replay in 0..3 {
            s.replay(id).unwrap();
            s.sync().unwrap();
            same(
                &eager,
                &s.download(&c).unwrap(),
                &format!("replay {replay}"),
            );
            assert!(
                credit_words(s).iter().all(|w| w.iter().all(|&x| x == 0)),
                "replay {replay}: a credit word left counted"
            );
        }
        s.release_trace(id).unwrap();
        s.free(c).unwrap();
    });
}

/// The negative control: copies that read 16 bytes past their strip change the product.
#[test]
fn a_misrouted_copy_changes_the_product() {
    misrouted(None, reuse::probe_misroute);
}

/// The negative control for `B`: only `B`'s copies misrouted, with both operands shared.
#[test]
fn a_misrouted_copy_of_b_changes_the_product() {
    misrouted(Some(1), reuse::probe_misroute_b);
}

fn misrouted(b_holders: Option<usize>, probe: fn(bool)) {
    with_session(4, |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 3]));
        let a = s.upload(&values(96 * 256, 5), 96, 256).unwrap();
        let b = s.upload(&values(256 * 320, 6), 256, 320).unwrap();
        let (plain, _) = product(s, &a, false, &b, false, Fidelity::HiFi4, Reuse::Off);
        probe(true);
        let (shared, copies) = product(
            s,
            &a,
            false,
            &b,
            false,
            Fidelity::HiFi4,
            Reuse::Pull {
                holders: 1,
                b_holders,
            },
        );
        probe(false);
        assert!(copies > 0);
        assert!(
            plain
                .iter()
                .zip(&shared)
                .any(|(x, y)| x.to_bits() != y.to_bits()),
            "a misrouted copy left the product unchanged"
        );
    });
}

/// A shared product whose readers wait for one credit more than their holders send fails by its
/// deadline; then the session runs the same product shared, bit for bit, every credit word at zero.
/// Silicon only: the stalled units' roles wait inside their kernels, and recovering such a tile
/// takes the backend pulse ttsim does not model (`docs/learnings/ttsim-divergence.md` row 16).
#[cfg(feature = "silicon")]
#[test]
fn a_stalled_shared_product_leaves_the_session_usable() {
    with_session(4, |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 3]));
        let a = s.upload(&values(96 * 256, 9), 96, 256).unwrap();
        let b = s.upload(&values(256 * 320, 10), 256, 320).unwrap();
        let (plain, _) = product(s, &a, false, &b, false, Fidelity::HiFi4, Reuse::Off);
        let both = Reuse::Pull {
            holders: 1,
            b_holders: Some(1),
        };
        s.set_operand_reuse(both).unwrap();
        reuse::probe_overwait(true);
        let stalled = s
            .matmul_dram(
                &a,
                false,
                &b,
                false,
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                BUDGET * 40,
            )
            .and_then(|out| s.sync().map(|_| out));
        reuse::probe_overwait(false);
        assert!(stalled.is_err(), "a reader waiting past its credits stalls");
        let (shared, copies) = product(s, &a, false, &b, false, Fidelity::HiFi4, both);
        assert!(copies > 0);
        same(&plain, &shared, "after a stalled shared product");
        assert!(
            credit_words(s).iter().all(|w| w.iter().all(|&x| x == 0)),
            "a credit word left counted after recovery"
        );
    });
}

/// H1's product on every tile, shared, against unshared: the scale the bench runs at.
#[cfg(feature = "silicon")]
#[test]
#[ignore = "a probe: cargo xtask silicon --filter step244_operand_reuse::every_tile"]
fn every_tile_h1_product_is_the_unshared_one() {
    if let Err(e) = fork_scope(|| {
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::All,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        let [m, k, n] = std::env::var("TT_REUSE_SHAPE").map_or([2048, 2048, 16384], |v| {
            let d: Vec<usize> = v.split(',').map(|x| x.trim().parse().unwrap()).collect();
            [d[0], d[1], d[2]]
        });
        let a = s.upload(&values(m * k, 7), m, k).unwrap();
        let b = s.upload(&values(k * n, 8), k, n).unwrap();
        let (plain, _) = product(&mut s, &a, false, &b, false, Fidelity::HiFi4, Reuse::Off);
        let holders = std::env::var("TT_REUSE_HOLDERS").map_or(1, |v| v.parse().unwrap());
        let b_holders = std::env::var("TT_REUSE_B_HOLDERS")
            .ok()
            .map(|v| v.parse().unwrap());
        let t0 = std::time::Instant::now();
        s.set_operand_reuse(Reuse::Pull { holders, b_holders })
            .unwrap();
        let before = reuse::copies();
        let out = s.matmul_dram(
            &a,
            false,
            &b,
            false,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            BUDGET * 40,
        );
        eprintln!(
            "shared: {:?} after {:?}, {} copies",
            out.as_ref().err(),
            t0.elapsed(),
            reuse::copies() - before
        );
        if out.is_err() {
            // Each unit's actors' wait reasons (B, T0-T2, NC) and its credit words, by pattern.
            let tiles = s.tiles();
            let d = s.device();
            let w = d.alloc_window(WindowKind::TwoMib).unwrap();
            let mut seen: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
            for (u, &t) in tiles.iter().enumerate() {
                let reasons: Vec<u32> = (0..5)
                    .map(|a| {
                        d.read32(&w, t, tt_isa::dataflow::WAIT_REASON + 4 * a)
                            .unwrap()
                    })
                    .collect();
                let words: Vec<u32> = (0..CREDIT_WORDS)
                    .map(|i| d.read32(&w, t, credit_word(i) as u64).unwrap())
                    .collect();
                seen.entry(format!("reasons {reasons:?} words {words:?}"))
                    .or_default()
                    .push(u);
            }
            d.free_window(w);
            for (k, units) in seen {
                eprintln!("STALL {k}: units {units:?}");
            }
        }
        let out = out.unwrap();
        same(&plain, &s.download(&out).unwrap(), "every tile");
    }) {
        panic!("{e}");
    }
}
