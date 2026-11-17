//! Operand multicast across a grid of units (roadmap W2; `tile_op::multicast`): the units of a
//! blocked product are a grid of rows (`TileChoice::Rect`), and each grid row's first unit reads
//! the row's strips of `A` from GDDR and multicasts them along the row, each column's unit in grid
//! row 0 reads the column's strips of `B` and multicasts them down the column
//! (`op::WRITE_L1_MCAST` on RISCV NC, credited both ways through the op-level credit words). The
//! readers gather only what they do not receive (`record::GATHER_A_RESIDENT`,
//! `record::GATHER_B_RESIDENT`).
//!
//! The claims:
//! - **A multicast product is the unshared one bit for bit**, ragged in every dimension, either
//!   operand transposed, at HiFi4 and LoFi, over signed zeros, subnormals, infinities and NaNs,
//!   with `A`, `B` or both multicast, with units sitting rounds out (column blocks that do not
//!   divide the grid's width, block rows that do not divide its height) and a unit past the grid:
//!   the multicasts move bytes only.
//! - **Each unit leaves its credit words at zero** (`tt_isa::dm::CREDITS`), so a trace of a
//!   multicast product replays as the eager product, again and again.
//! - **The negative control:** multicasts of either operand that land 16 bytes past their strip
//!   change the product.
//! - **A stalled multicast product fails and leaves the session usable** (silicon only, as
//!   `step244`'s: ttsim cannot recover a tile whose roles stalled inside a kernel, divergence row
//!   16).
//!
//! What ttsim cannot show: its NoC requests complete before the next is issued (divergence row
//! 72), so a multicast cannot race a reader's kernel there. The pass's own model test
//! (`tile_op::multicast::tests`) interleaves every unit's B, kernel and NC at random and catches
//! a sender that does not wait for its readers' frees, a sender's B that gathers into a half its
//! NC may still be multicasting from, and a reader that does not wait for its strip; silicon runs
//! the same products with real concurrency.
use tt_device::tlb::WindowKind;
use tt_isa::dm::{credit_word, CREDIT_WORDS};
use tt_kernels::matmul::{Fidelity, SrcRoute};
use tt_kernels::session::{Session, TileChoice};
use tt_kernels::tensor::{BlockStream, DramTensor, Reuse};
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

/// A `rows x cols` grid of units.
fn rect(rows: usize, cols: usize) -> TileChoice {
    TileChoice::Rect { cols, rows }
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

/// `op(A) @ op(B)` under `reuse`, and how many strips it multicast.
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
    let before = multicast::strips();
    let out = s
        .matmul_dram(a, at, b, bt, SrcRoute::Tf32FromFp32, fidelity, BUDGET * 40)
        .unwrap();
    let v = s.download(&out).unwrap();
    s.free(out).unwrap();
    (v, multicast::strips() - before)
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

fn zeroed<T: tt_device::Transport>(s: &mut Session<T>, what: &str) {
    assert!(
        credit_words(s).iter().all(|w| w.iter().all(|&x| x == 0)),
        "{what}: a credit word left counted"
    );
}

/// `[m, k, n]` on a `rows x cols` grid in `[mb, kq, nb]` blocks, multicasting `A` and/or `B`,
/// for each transpose and fidelity: multicast equals unshared, and something was multicast.
fn case(grid: [usize; 2], [m, k, n]: [usize; 3], block: [usize; 3], a_: bool, b_: bool) {
    let [rows, cols] = grid;
    with_session(rect(rows, cols), |s| {
        s.set_block_stream(BlockStream::Fixed(block));
        for (at, bt, fidelity) in [
            (false, false, Fidelity::HiFi4),
            (true, false, Fidelity::HiFi4),
            (false, true, Fidelity::Lo),
        ] {
            let what = format!(
                "[{m},{k}]@[{k},{n}] {block:?} on {rows}x{cols}, A {a_}, B {b_}, A^T {at}, B^T \
                 {bt}, {fidelity:?}"
            );
            let (ar, ac) = if at { (k, m) } else { (m, k) };
            let (br, bc) = if bt { (n, k) } else { (k, n) };
            let a = s.upload(&values(ar * ac, 1), ar, ac).unwrap();
            let b = s.upload(&values(br * bc, 2), br, bc).unwrap();
            let (plain, none) = product(s, &a, at, &b, bt, fidelity, Reuse::Off);
            assert_eq!(none, 0, "{what}: unshared multicasts nothing");
            let reuse_ = Reuse::Multicast { a: a_, b: b_ };
            let (shared, strips) = product(s, &a, at, &b, bt, fidelity, reuse_);
            assert!(strips > 0, "{what}: nothing was multicast");
            same(&plain, &shared, &what);
            zeroed(s, &what);
            s.free(a).unwrap();
            s.free(b).unwrap();
        }
    });
}

/// Both operands on a 2 x 3 grid: two block rows a grid row, three column blocks a unit.
#[test]
fn a_multicast_product_is_the_unshared_one_bit_for_bit() {
    case([2, 3], [128, 256, 288], [2, 2, 1], true, true);
}

/// A weight laid by column (the stored weights' default): its strips gathered a column a request
/// into L1 by column (`record::BSpan`), multicast as they lie, and the readers' kernels stepping
/// `B` by column over their resident strips. Both operands, `A` alone and `B` alone (the senders
/// of `A` beside a resident weight read their `A` inline), each fidelity.
#[test]
fn a_weight_laid_by_column_multicasts_bit_for_bit() {
    let (m, k, n) = (128, 256, 512);
    with_session(rect(2, 4), |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 2]));
        let bits: Vec<u16> = values(k * n, 12)
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let b = s.upload_bf16_weight(&bits, k, n).unwrap();
        assert_eq!(
            b.placement.layout(),
            tt_kernels::tensor::Layout::ByColumn,
            "the weight by column"
        );
        let a = s.upload(&values(m * k, 11), m, k).unwrap();
        for fidelity in [Fidelity::HiFi4, Fidelity::Lo] {
            // Unblocked, `B` read a tile at a time: the order by column is not on both sides.
            s.set_block_stream(BlockStream::Off);
            let (want, _) = product(s, &a, false, &b, false, fidelity, Reuse::Off);
            s.set_block_stream(BlockStream::Fixed([2, 2, 2]));
            let (plain, _) = product(s, &a, false, &b, false, fidelity, Reuse::Off);
            same(&want, &plain, &format!("by column, blocked, {fidelity:?}"));
            for (a_, b_) in [(true, true), (true, false), (false, true)] {
                let what = format!("by column, A {a_}, B {b_}, {fidelity:?}");
                let reuse_ = Reuse::Multicast { a: a_, b: b_ };
                let (shared, strips) = product(s, &a, false, &b, false, fidelity, reuse_);
                assert!(strips > 0, "{what}: nothing was multicast");
                same(&plain, &shared, &what);
                zeroed(s, &what);
            }
        }
        s.free(a).unwrap();
        s.free(b).unwrap();
    });
}

/// `A` alone, and `B` alone.
#[test]
fn either_operand_alone_gives_the_same_bits() {
    case([2, 3], [128, 256, 288], [2, 2, 1], true, false);
    case([2, 3], [128, 256, 288], [2, 2, 1], false, true);
}

/// Ragged in every dimension, on a 3 x 3 grid whose units sit rounds out: the last block row of
/// one tile row (block rows 2, 1 and 1 a grid row), column blocks 2, 2 and 1 a unit with the last
/// one short, and `K`'s last tile partly padding.
#[test]
fn a_ragged_multicast_product_is_the_unshared_one_bit_for_bit() {
    case([3, 3], [232, 232, 400], [2, 2, 3], true, true);
}

/// More grid rows than block rows: the rows past them run nothing, and `B`'s readers are the rows
/// that run.
#[test]
fn grid_rows_past_the_block_rows_give_the_same_bits() {
    case([3, 2], [64, 256, 256], [1, 2, 2], true, true);
}

/// A trace of a multicast product replays as its eager product, every time, and leaves every
/// credit word at zero.
#[test]
fn a_multicast_product_replays_as_its_eager_product() {
    with_session(rect(2, 3), |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 1]));
        let a = s.upload(&values(128 * 256, 3), 128, 256).unwrap();
        let b = s.upload(&values(256 * 288, 4), 256, 288).unwrap();
        let both = Reuse::Multicast { a: true, b: true };
        let (plain, _) = product(s, &a, false, &b, false, Fidelity::HiFi4, Reuse::Off);
        let (eager, strips) = product(s, &a, false, &b, false, Fidelity::HiFi4, both);
        assert!(strips > 0);
        same(&plain, &eager, "eager");
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
            zeroed(s, &format!("replay {replay}"));
        }
        s.release_trace(id).unwrap();
        s.free(c).unwrap();
    });
}

/// The negative control: multicasts of `A`, then of `B`, that land 16 bytes past their strip
/// change the product.
#[test]
fn a_misrouted_multicast_changes_the_product() {
    with_session(rect(2, 3), |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 1]));
        let a = s.upload(&values(128 * 256, 5), 128, 256).unwrap();
        let b = s.upload(&values(256 * 288, 6), 256, 288).unwrap();
        let (plain, _) = product(s, &a, false, &b, false, Fidelity::HiFi4, Reuse::Off);
        for operand in [0, 1] {
            multicast::probe_misroute(operand, true);
            let (shared, strips) = product(
                s,
                &a,
                false,
                &b,
                false,
                Fidelity::HiFi4,
                Reuse::Multicast { a: true, b: true },
            );
            multicast::probe_misroute(operand, false);
            assert!(strips > 0);
            assert!(
                plain
                    .iter()
                    .zip(&shared)
                    .any(|(x, y)| x.to_bits() != y.to_bits()),
                "a misrouted multicast of operand {operand} left the product unchanged"
            );
        }
    });
}

/// A multicast product whose readers wait for one credit more than their senders send fails by
/// its deadline; then the session runs the same product multicast, bit for bit, every credit word
/// at zero. Silicon only (the module doc).
#[cfg(feature = "silicon")]
#[test]
fn a_stalled_multicast_product_leaves_the_session_usable() {
    with_session(rect(2, 3), |s| {
        s.set_block_stream(BlockStream::Fixed([2, 2, 1]));
        let a = s.upload(&values(128 * 256, 9), 128, 256).unwrap();
        let b = s.upload(&values(256 * 288, 10), 256, 288).unwrap();
        let (plain, _) = product(s, &a, false, &b, false, Fidelity::HiFi4, Reuse::Off);
        let both = Reuse::Multicast { a: true, b: true };
        s.set_operand_reuse(both).unwrap();
        multicast::probe_overwait(true);
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
        multicast::probe_overwait(false);
        assert!(stalled.is_err(), "a reader waiting past its credits stalls");
        let (shared, strips) = product(s, &a, false, &b, false, Fidelity::HiFi4, both);
        assert!(strips > 0);
        same(&plain, &shared, "after a stalled multicast product");
        zeroed(s, "after recovery");
    });
}

/// The probe's grids at H1's shape and the 1.7B's `down`, multicast, against unshared: the scale
/// the bench runs at, with real concurrency, on 8 x 8, 8 x 12 and 10 x 12. Silicon only.
#[cfg(feature = "silicon")]
#[test]
fn the_probe_grids_give_the_same_bits() {
    for (grid, shape, block) in [
        ([8, 8], [2048, 2048, 16384], [8, 2, 8]),
        ([8, 12], [2048, 2048, 16384], [8, 2, 8]),
        ([10, 12], [2048, 2048, 16384], [4, 2, 8]),
        ([8, 8], [2048, 8192, 2048], [8, 2, 8]),
    ] {
        let [rows, cols] = grid;
        let [m, k, n] = shape;
        with_session(rect(rows, cols), |s| {
            s.set_block_stream(BlockStream::Fixed(block));
            let a = s.upload(&values(m * k, 7), m, k).unwrap();
            let b = s.upload(&values(k * n, 8), k, n).unwrap();
            let what = format!("{shape:?} {block:?} on {rows}x{cols}");
            let (plain, _) = product(s, &a, false, &b, false, Fidelity::HiFi4, Reuse::Off);
            let both = Reuse::Multicast { a: true, b: true };
            let (shared, strips) = product(s, &a, false, &b, false, Fidelity::HiFi4, both);
            assert!(strips > 0, "{what}: nothing was multicast");
            same(&plain, &shared, &what);
            zeroed(s, &what);
        });
    }
}
