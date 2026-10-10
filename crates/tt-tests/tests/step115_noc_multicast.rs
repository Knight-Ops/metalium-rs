//! NoC multicast writes to a rectangle of Tensix tiles
//! (`BlackholeA0/NoC/MemoryMap.md`, `RoutingPaths.md`, `Interrupts.md:19`).
//!
//! Three layers of evidence:
//!
//! * **Model (host, always).** [`recipients`] is the page's rectangle rule
//!   (`NOC_TARG_ADDR` "for broadcast request packets": `EndX`/`EndY`/`StartX`/
//!   `StartY`, the wrap rule for `Start > End`, opt-out of every tile that is not
//!   a surviving Tensix tile), applied to the `RET_ADDR_HI` word the encoder
//!   produced. It does not use `Rect::tiles`: the encoder's rectangle and the
//!   page's rule are compared over every valid rectangle of a full and of a
//!   harvested grid, and against hand-written literals.
//! * **ttsim.** ttsim executes the multicast, acknowledging every recipient
//!   (`NIU_MST_WR_ACK_RECEIVED` advances by the recipient count and the ID's
//!   outstanding counter ends at `1 - recipients`, as the page says). The oracle
//!   is the host's model of the expansion: every recipient's payload arrives and
//!   every non-recipient's guard words are unchanged. ttsim refuses the write
//!   that clears the ID's counter (`NIU_BASE + 0x60`), so its runs skip it
//!   (`probe::flag::NO_CLEAR`; divergence row 89, proposed).
//! * **Silicon (`--features silicon`, written, NOT run).** The 1x2 probe first,
//!   inside `survives`, then the rectangle and full-grid gates. Risk class:
//!   documented, UNVERIFIED on silicon, **NoC-hang class** (a broadcast tree).
//!   The grid is the chip's ARC-discovered one.
//!
//! Negative controls: [`Mutant`] changes the model (rectangle off by one column,
//! NoC #1 corners not swapped) and `mutants_are_caught` requires the literals to
//! reject each; the device gate runs a multicast whose rectangle is one column
//! wider than the one checked and requires it to fail.

mod noc_support;

use std::collections::BTreeSet;

/// A rectangle as `(x0, y0, x1, y1)`.
type R4 = (u8, u8, u8, u8);

use tt_isa::noc::grid::Tensix;
use tt_isa::noc::multicast::{full_grid_except, AckCount, AckProgress, MulticastWrite, Rect};
use tt_isa::noc::niu::{initiator, Niu, TxnId};
use tt_isa::noc::probe;

// ---------------------------------------------------------------------------
// The model: the page's rectangle rule.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mutant {
    None,
    /// `EndX` is exclusive: the last column is dropped.
    EndExclusive,
    /// The NoC #1 corners are used as given, not swapped back.
    NoSwapOnNoc1,
}

fn span(start: u32, end: u32, v: u32) -> bool {
    if start <= end {
        start <= v && v <= end
    } else {
        v <= end || start <= v
    }
}

/// The tiles a broadcast with this `RET_ADDR_HI` reaches through `niu`: the
/// product of the X and Y spans, intersected with the tiles that have not opted
/// out (firmware opts out everything but surviving Tensix tiles).
fn recipients(hi: u32, niu: Niu, grid: &Tensix, mutant: Mutant) -> BTreeSet<(u8, u8)> {
    let (mut ex, mut ey, mut sx, mut sy) =
        (hi & 63, (hi >> 6) & 63, (hi >> 12) & 63, (hi >> 18) & 63);
    if niu == Niu::Noc1 && mutant != Mutant::NoSwapOnNoc1 {
        // `Coordinates.md:26`: software swaps Start and End on the mirrored NoC.
        core::mem::swap(&mut ex, &mut sx);
        core::mem::swap(&mut ey, &mut sy);
    }
    if mutant == Mutant::EndExclusive {
        ex = ex.wrapping_sub(1);
    }
    let mut out = BTreeSet::new();
    for y in 0..64u32 {
        for x in 0..64u32 {
            if span(sx, ex, x) && span(sy, ey, y) && grid.contains(x as u8, y as u8) {
                out.insert((x as u8, y as u8));
            }
        }
    }
    out
}

fn set(tiles: &[(u8, u8)]) -> BTreeSet<(u8, u8)> {
    tiles.iter().copied().collect()
}

/// Hand-written expectations: `(rect, NoC, recipients)`.
#[allow(clippy::type_complexity)]
fn literals() -> Vec<(R4, Niu, Vec<(u8, u8)>)> {
    let two = vec![(4, 4), (5, 4)];
    let four = vec![(4, 5), (5, 5), (4, 6), (5, 6)];
    vec![
        ((4, 4, 5, 4), Niu::Noc0, two.clone()),
        ((4, 4, 5, 4), Niu::Noc1, two),
        ((4, 5, 5, 6), Niu::Noc0, four.clone()),
        ((4, 5, 5, 6), Niu::Noc1, four),
        ((7, 2, 7, 3), Niu::Noc0, vec![(7, 2), (7, 3)]),
        ((10, 11, 11, 11), Niu::Noc1, vec![(10, 11), (11, 11)]),
    ]
}

fn expected_of(rect: R4, niu: Niu, mutant: Mutant) -> BTreeSet<(u8, u8)> {
    let g = Tensix::FULL;
    let r = Rect::new(&g, rect.0, rect.1, rect.2, rect.3).unwrap();
    recipients(r.hi(niu), niu, &g, mutant)
}

#[test]
fn model_reproduces_the_hand_written_recipients() {
    for (rect, niu, want) in literals() {
        assert_eq!(
            expected_of(rect, niu, Mutant::None),
            set(&want),
            "{rect:?} on {niu:?}"
        );
    }
}

#[test]
fn mutants_are_caught() {
    for m in [Mutant::EndExclusive, Mutant::NoSwapOnNoc1] {
        let caught = literals()
            .iter()
            .any(|(rect, niu, want)| expected_of(*rect, *niu, m) != set(want));
        assert!(caught, "{m:?} survived every expectation");
    }
}

/// The encoder's rectangle and the page's rule agree over every valid rectangle,
/// on both NoCs, on a full grid and on a 12-column (p150a) grid.
#[test]
fn encoder_hi_word_expands_to_the_rectangle_on_both_nocs() {
    for grid in [Tensix::FULL, Tensix::from_enabled_column_count(12).unwrap()] {
        let mut valid = 0;
        for x0 in 0..17u8 {
            for x1 in x0..17 {
                for y0 in 0..12u8 {
                    for y1 in y0..12 {
                        let Ok(r) = Rect::new(&grid, x0, y0, x1, y1) else {
                            continue;
                        };
                        valid += 1;
                        let want: BTreeSet<_> = r.tiles().collect();
                        assert_eq!(want.len() as u32, r.tile_count());
                        for niu in [Niu::Noc0, Niu::Noc1] {
                            assert_eq!(
                                recipients(r.hi(niu), niu, &grid, Mutant::None),
                                want,
                                "{:?} {niu:?}",
                                (x0, y0, x1, y1)
                            );
                        }
                    }
                }
            }
        }
        assert!(valid > 1000, "{valid}");
    }
}

#[test]
fn acknowledgements_come_from_the_known_recipients() {
    let w = MulticastWrite {
        from_local: 0x2_2000,
        rect: Rect::new(&Tensix::FULL, 4, 5, 6, 6).unwrap(),
        to_addr: 0x2_2400,
        len: 64,
    };
    assert_eq!(w.acks(), 6);
    let c = AckCount::start(100, &w);
    // The ID's outstanding counter says nothing here (it ends at `1 - 6`), so the
    // only progress measure is the ack counter.
    assert_eq!(c.progress(105), AckProgress::Waiting { missing: 1 });
    assert_eq!(c.progress(106), AckProgress::Complete);
    assert_eq!(c.progress(107), AckProgress::Excess { extra: 1 });
    // The registers carry exactly this ID's response-marked write.
    let r = w
        .registers((3, 4), TxnId::new(7).unwrap(), Niu::Noc0)
        .unwrap();
    assert_eq!(r.last().unwrap().0, initiator::BRCST_EXCLUDE);
}

// ---------------------------------------------------------------------------
// Device runs (ttsim and silicon).
// ---------------------------------------------------------------------------

use noc_support::{multicast, read_bytes, run_probe, write_bytes, Res};
use tt_isa::noc::{Noc0, NocCoord};
use tt_tests::harness::{self, Dev};

const SRC: u64 = probe::DATA;
const DEST: u64 = probe::DATA + 0x400;
const LEN: usize = 64;
const GUARD: u8 = 0x77;

/// Whether the probe clears the ID's counter after a multicast. ttsim refuses
/// that write, so on the simulator it does not.
const CLEAR_FLAG: u32 = if cfg!(feature = "silicon") {
    0
} else {
    probe::flag::NO_CLEAR
};

/// A wrong request the control and the refusal arms send.
#[derive(Clone, Copy, PartialEq, Debug)]
#[allow(dead_code)] // the refusal variants are simulator-only
enum Tamper {
    None,
    /// `EndX` one column past the rectangle (NoC #0 only).
    OneColumnWider,
    /// `NOC_CMD_PATH_RESERVE` cleared.
    NoPathReserve,
    /// Start and End swapped on NoC #0: a wrapping rectangle.
    ReversedCorners,
}

fn tamper_regs(regs: &mut [(u64, u32)], t: Tamper) {
    for (off, v) in regs.iter_mut() {
        match (t, *off) {
            (Tamper::OneColumnWider, initiator::RET_ADDR_HI) => *v += 1,
            (Tamper::NoPathReserve, initiator::CTRL) => *v &= !(1 << 8),
            (Tamper::ReversedCorners, initiator::RET_ADDR_HI) => {
                let (ex, ey, sx, sy) = (*v & 63, (*v >> 6) & 63, (*v >> 12) & 63, (*v >> 18) & 63);
                *v = sx | (sy << 6) | (ex << 12) | (ey << 18);
            }
            _ => {}
        }
    }
}

/// Every tile of this chip's grid except `me`, as device coordinates. On silicon
/// the grid is the ARC's; a tile outside it is never constructed.
fn watched(
    dev: &mut Dev<'_>,
    me: NocCoord<Noc0>,
    rects: &[R4],
    scope: Scope,
) -> Vec<NocCoord<Noc0>> {
    let grid = harness::tensix_grid(dev);
    // The neighbourhood: every grid tile within one column or row of a rectangle.
    let near = |t: &NocCoord<Noc0>| {
        rects
            .iter()
            .any(|r| t.x() + 1 >= r.0 && t.x() <= r.2 + 1 && t.y() + 1 >= r.1 && t.y() <= r.3 + 1)
    };
    grid.tiles::<Noc0>()
        .filter(|t| (t.x(), t.y()) != (me.x(), me.y()))
        .filter(|t| scope == Scope::WholeGrid || near(t))
        .collect()
}

/// Which tiles an oracle reads: the first silicon probe touches only the
/// rectangle and a ring around it; every other run, every tile of the grid.
#[derive(Clone, Copy, PartialEq, Debug)]
#[allow(dead_code)] // `Neighbourhood` is the silicon probe's
enum Scope {
    Neighbourhood,
    WholeGrid,
}

fn payload(seed: u8) -> Vec<u8> {
    (0..LEN as u8).map(|i| seed.wrapping_add(i)).collect()
}

/// Send `rects` as multicasts from the gate tile through `niu`, and check the
/// payload reached exactly the tiles of those rectangles on this chip's grid.
fn run_case(dev: &mut Dev<'_>, rects: &[R4], niu: Niu, tamper: Tamper) -> Result<(), String> {
    run_case_in(dev, rects, niu, tamper, Scope::WholeGrid)
}

fn run_case_in(
    dev: &mut Dev<'_>,
    rects: &[R4],
    niu: Niu,
    tamper: Tamper,
    scope: Scope,
) -> Result<(), String> {
    let me = harness::tensix_tile();
    let grid = harness::tensix_grid(dev);
    let watch = watched(dev, me, rects, scope);
    let data = payload(0x40);
    write_bytes(dev, me, SRC, &data);
    for t in &watch {
        write_bytes(dev, *t, DEST, &[GUARD; LEN]);
    }
    let mut expected: BTreeSet<(u8, u8)> = BTreeSet::new();
    let mut reqs = vec![];
    for (k, &(x0, y0, x1, y1)) in rects.iter().enumerate() {
        let rect = Rect::new(&grid, x0, y0, x1, y1).map_err(|e| format!("{e:?}"))?;
        expected.extend(rect.tiles());
        let txn = k as u32 + 1;
        let w = MulticastWrite {
            from_local: SRC as u32,
            rect,
            to_addr: DEST as u32,
            len: LEN as u32,
        };
        let mut regs = w
            .registers((me.x(), me.y()), TxnId::new(txn as u8).unwrap(), niu)
            .map_err(|e| format!("{e:?}"))?;
        tamper_regs(&mut regs, tamper);
        let flags = CLEAR_FLAG
            | if niu == Niu::Noc1 {
                probe::flag::NOC1
            } else {
                0
            };
        reqs.push(multicast(&regs, txn, flags, w.acks()));
    }
    let res: Vec<Res> = run_probe(dev, me, &reqs, &[])?;
    for (k, r) in res.iter().enumerate() {
        let n = Rect::new(&grid, rects[k].0, rects[k].1, rects[k].2, rects[k].3)
            .unwrap()
            .tile_count();
        if r.status != probe::status::OK {
            let why = match r.status {
                probe::status::EXCESS_ACKS => "more acks than recipients",
                probe::status::TIMEOUT => "acks missing (timeout)",
                _ => "probe refused",
            };
            return Err(format!("request {k}: {why}, status {} ({r:?})", r.status));
        }
        if r.ack_after.wrapping_sub(r.ack_before) != n {
            return Err(format!("request {k}: {n} recipients, acks {r:?}"));
        }
        // A broadcast increments the ID's counter once and decrements it once
        // per recipient (`Counters.md`); the probe has not cleared it on ttsim.
        if CLEAR_FLAG != 0 && r.outstanding_after != (1u32.wrapping_sub(n)) & 0xFF {
            return Err(format!("request {k}: outstanding {r:?} for {n} recipients"));
        }
    }
    for t in &watch {
        let got = read_bytes(dev, *t, DEST, LEN);
        let in_rect = expected.contains(&(t.x(), t.y()));
        let want = if in_rect {
            &data[..]
        } else {
            &[GUARD; LEN][..]
        };
        if got != want {
            return Err(format!(
                "tile ({}, {}): {} but {:02x?}",
                t.x(),
                t.y(),
                if in_rect {
                    "a recipient did not get the payload"
                } else {
                    "a non-recipient's guard changed"
                },
                &got[..8]
            ));
        }
    }
    Ok(())
}

fn full_grid_rects(dev: &mut Dev<'_>) -> Vec<R4> {
    let me = harness::tensix_tile();
    let grid = harness::tensix_grid(dev);
    full_grid_except(&grid, me.x(), me.y())
        .iter()
        .flatten()
        .map(|r| (r.x_range().0, r.y_range().0, r.x_range().1, r.y_range().1))
        .collect()
}

#[cfg(not(feature = "silicon"))]
mod simulator {
    use super::*;

    const CASES: [(R4, &str); 5] = [
        ((4, 4, 5, 4), "1x2"),
        ((4, 5, 5, 6), "2x2"),
        ((5, 4, 5, 4), "1x1"),
        ((4, 6, 6, 8), "3x3"),
        ((11, 5, 13, 7), "3x3 in the right block"),
    ];

    #[test]
    fn simulator_multicast_reaches_exactly_the_rectangle() {
        for niu in [Niu::Noc0, Niu::Noc1] {
            for (rect, name) in CASES {
                harness::in_device(|dev| {
                    run_case(dev, &[rect], niu, Tamper::None)
                        .unwrap_or_else(|e| panic!("{name} on {niu:?}: {e}"));
                });
            }
        }
    }

    #[test]
    fn simulator_full_grid_multicast_reaches_every_other_tile() {
        for niu in [Niu::Noc0, Niu::Noc1] {
            harness::in_device(|dev| {
                let rects = full_grid_rects(dev);
                assert_eq!(
                    rects.len(),
                    5,
                    "left block split around (3, 4), right block whole"
                );
                run_case(dev, &rects, niu, Tamper::None).unwrap_or_else(|e| panic!("{niu:?}: {e}"));
            });
        }
    }

    /// The negative control: a multicast one column wider than the rectangle
    /// the oracle checks is caught, by the acks and by the guard words.
    #[test]
    fn a_rectangle_one_column_too_wide_is_caught() {
        harness::in_device(|dev| {
            let e = run_case(dev, &[(4, 4, 5, 4)], Niu::Noc0, Tamper::OneColumnWider)
                .expect_err("the oracle accepted a rectangle one column too wide");
            assert!(e.contains("acks") || e.contains("guard"), "{e}");
        });
    }

    /// What ttsim refuses (divergence row 89, proposed), each in its own fork,
    /// against a control of the same shape that survives.
    #[test]
    fn simulator_refusals_have_surviving_controls() {
        let ran = |t: Tamper, flags_clear: bool| {
            harness::survives(|dev| {
                let me = harness::tensix_tile();
                let grid = harness::tensix_grid(dev);
                let rect = Rect::new(&grid, 4, 4, 5, 4).unwrap();
                write_bytes(dev, me, SRC, &payload(1));
                let w = MulticastWrite {
                    from_local: SRC as u32,
                    rect,
                    to_addr: DEST as u32,
                    len: LEN as u32,
                };
                let mut regs = w
                    .registers((me.x(), me.y()), TxnId::new(1).unwrap(), Niu::Noc0)
                    .unwrap();
                tamper_regs(&mut regs, t);
                let flags = if flags_clear {
                    0
                } else {
                    probe::flag::NO_CLEAR
                };
                let _ = run_probe(dev, me, &[multicast(&regs, 1, flags, w.acks())], &[]);
            })
        };
        assert!(ran(Tamper::None, false), "the control must survive");
        assert!(
            !ran(Tamper::NoPathReserve, false),
            "ttsim accepts a multicast without path reserve"
        );
        assert!(
            !ran(Tamper::ReversedCorners, false),
            "ttsim accepts a wrapping rectangle"
        );
        assert!(
            !ran(Tamper::None, true),
            "ttsim accepts the clear of NIU_BASE+0x60"
        );
    }
}

/// Silicon: written, not run. Order (one session each, nothing queued behind a
/// NoC-hang-class probe): the neighbour atomic of `step116`, then
/// `silicon_noc_multicast_minimal_probe` (1x2), then the rectangle gate, then the
/// full grid. Expected: every recipient's payload, every other tile's guard
/// unchanged, acks equal to the recipient count, outstanding `1 - n` before the
/// clear and 0 after it. A hang names its last breadcrumb in the error.
#[cfg(feature = "silicon")]
mod silicon {
    use super::*;

    /// Run alone. A 1x2 rectangle two tiles from the initiator, inside `survives`.
    #[test]
    fn silicon_noc_multicast_minimal_probe() {
        harness::assert_on_silicon();
        assert!(harness::survives(|dev| {
            // Both recipients are claimed through the grid-checked constructor.
            let _ = harness::tile(dev, 4, 4);
            let _ = harness::tile(dev, 5, 4);
            run_case_in(
                dev,
                &[(4, 4, 5, 4)],
                Niu::Noc0,
                Tamper::None,
                Scope::Neighbourhood,
            )
            .unwrap();
        }));
    }

    #[test]
    fn silicon_noc_multicast_rectangles() {
        harness::assert_on_silicon();
        harness::in_device(|dev| {
            for rect in [(4, 5, 5, 6), (4, 6, 6, 8)] {
                run_case(dev, &[rect], Niu::Noc0, Tamper::None).unwrap();
            }
        });
    }

    #[test]
    fn silicon_noc_multicast_full_grid() {
        harness::assert_on_silicon();
        harness::in_device(|dev| {
            let rects = full_grid_rects(dev);
            run_case(dev, &rects, Niu::Noc0, Tamper::None).unwrap();
        });
    }

    /// The control: a rectangle one column wider than checked must fail the oracle
    /// on silicon too.
    #[test]
    fn silicon_rejects_the_one_column_wider_mutant() {
        harness::assert_on_silicon();
        harness::in_device(|dev| {
            assert!(run_case(dev, &[(4, 4, 5, 4)], Niu::Noc0, Tamper::OneColumnWider).is_err());
        });
    }
}
