//! NoC multicast writes to a rectangle of Tensix tiles (`BlackholeA0/NoC/MemoryMap.md`,
//! "`NOC_TARG_ADDR` and `NOC_RET_ADDR`", `NOC_CTRL` bits 5, 7, 8, 13-16;
//! `RoutingPaths.md`; `Interrupts.md:19`).
//!
//! What is deliberately not expressible:
//!
//! * A rectangle that is not wholly inside the chip's *surviving* Tensix tiles
//!   ([`Rect::new`] takes the ARC-discovered [`Tensix`], so a harvested column, an
//!   Ethernet or DRAM row and the non-compute columns 0, 8 and 9 are all errors
//!   rather than relying on the broadcast opt-out masks firmware configures
//!   (`ROUTER_CFG_1`/`_3`).
//! * `StartX > EndX` or `StartY > EndY`: hardware reads those as a rectangle that
//!   wraps around the grid. The caller's rectangle is always `start <= end`; the
//!   NoC #1 encoding swaps the corners itself (`Coordinates.md:26`: broadcasts
//!   need Start and End swapped on the mirrored NoC).
//! * Reads, atomics, linked virtual channels, `NOC_BRCST_EXCLUDE` (no documented
//!   layout; written zero, since silicon keeps whatever an earlier program left),
//!   the Y-major route and the initiator as a recipient.
//!
//! Completion is **not** `NIU_MST_REQS_OUTSTANDING_ID`: a response-marked
//! broadcast increments it once and decrements it once per recipient
//! (`Counters.md`, "Clear NIU transaction ID counters"). It is [`AckCount`]: the
//! initiator's `NIU_MST_WR_ACK_RECEIVED` must advance by exactly the recipient
//! count the host derived from the rectangle, after which the ID's counter is
//! cleared with `niu::CLEAR_OUTSTANDING`.

use super::grid::{Tensix, TENSIX_ROWS};
use super::niu::{
    check_copy, initiator::*, Endpoint, Niu, RequestError, TxnId, CMD_WR, RESP_MARKED,
};

/// `NOC_CMD_BRCST_PACKET`.
const BRCST_PACKET: u32 = 1 << 5;
/// `NOC_CMD_VC_STATIC`.
const VC_STATIC: u32 = 1 << 7;
/// `NOC_CMD_PATH_RESERVE`: every router on the tree reserves its virtual channel
/// before any data leaves the NIU, so a multicast cannot deadlock against other
/// traffic. Always set (the page: software is otherwise responsible for traffic
/// patterns that cannot deadlock; ttsim refuses a multicast without it).
const PATH_RESERVE: u32 = 1 << 8;
/// `NOC_CMD_STATIC_VC` class bits (14..16) = `0b10`, the only class the page
/// allows a multicast, and buddy bit (13) = 0: virtual channel 4 in the
/// `(class << 1) | buddy` numbering (tt-metal's `NOC_MULTICAST_WRITE_VC`).
/// `NOC_CMD_VC_LINKED` (bit 6) is never set.
const STATIC_VC_MULTICAST: u32 = VC_STATIC | (0b10 << 14);

/// A non-empty rectangle of Tensix tiles that exist on this chip, in translated
/// coordinates, `start <= end` on both axes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Rect {
    x0: u8,
    y0: u8,
    x1: u8,
    y1: u8,
}

/// Why a rectangle was refused.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RectError {
    /// `start > end` on an axis (hardware would wrap around the grid).
    Reversed,
    /// A tile of the rectangle is not a surviving Tensix tile of this chip.
    NotTensix { x: u8, y: u8 },
}

impl Rect {
    /// The rectangle `[x0, x1] x [y0, y1]`, every tile of which must be a Tensix
    /// tile of `grid` (discovered from the chip's ARC, never assumed).
    pub fn new(grid: &Tensix, x0: u8, y0: u8, x1: u8, y1: u8) -> Result<Self, RectError> {
        if x0 > x1 || y0 > y1 {
            return Err(RectError::Reversed);
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                if !grid.contains(x, y) {
                    return Err(RectError::NotTensix { x, y });
                }
            }
        }
        Ok(Rect { x0, y0, x1, y1 })
    }

    pub const fn x_range(&self) -> (u8, u8) {
        (self.x0, self.x1)
    }

    pub const fn y_range(&self) -> (u8, u8) {
        (self.y0, self.y1)
    }

    /// The number of recipients: the number of acknowledgements to count.
    pub const fn tile_count(&self) -> u32 {
        (self.x1 - self.x0 + 1) as u32 * (self.y1 - self.y0 + 1) as u32
    }

    pub const fn contains(&self, x: u8, y: u8) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    /// Every recipient, row-major.
    pub fn tiles(&self) -> impl Iterator<Item = (u8, u8)> + '_ {
        (self.y0..=self.y1).flat_map(move |y| (self.x0..=self.x1).map(move |x| (x, y)))
    }

    /// `NOC_*_ADDR_HI` of a broadcast: `EndX | EndY << 6 | StartX << 12 |
    /// StartY << 18`. On NoC #1 the corners are swapped, because the mirrored
    /// NoC's raw coordinates run the other way while translation does not change
    /// the direction of data flow (`Coordinates.md:26`).
    pub const fn hi(&self, niu: Niu) -> u32 {
        let (s, e) = match niu {
            Niu::Noc0 => ((self.x0, self.y0), (self.x1, self.y1)),
            Niu::Noc1 => ((self.x1, self.y1), (self.x0, self.y0)),
        };
        (e.0 as u32) | ((e.1 as u32) << 6) | ((s.0 as u32) << 12) | ((s.1 as u32) << 18)
    }
}

/// The rectangles that together cover every Tensix tile of `grid`: columns
/// 1..=7 and columns 10.., each over all ten Tensix rows. Two, not one, because
/// columns 8 and 9 are not compute and a single rectangle across them would rely
/// on their opt-out.
pub fn full_grid(grid: &Tensix) -> [Option<Rect>; 2] {
    let (y0, y1) = (*TENSIX_ROWS.start(), *TENSIX_ROWS.end());
    let mut left = None;
    let mut right = None;
    for x in grid.columns() {
        let side = if x < 8 { &mut left } else { &mut right };
        *side = match *side {
            None => Some((x, x)),
            Some((a, _)) => Some((a, x)),
        };
    }
    let mk = |c: Option<(u8, u8)>| c.map(|(a, b)| Rect::new(grid, a, y0, b, y1).unwrap());
    [mk(left), mk(right)]
}

impl Rect {
    /// The rectangle minus one tile, as up to four rectangles (rows above, rows
    /// below, and the cells left and right of the tile in its row). A tile
    /// outside the rectangle returns the rectangle whole. For a multicast that
    /// must reach every tile but the initiator's own.
    pub fn without(&self, x: u8, y: u8) -> [Option<Rect>; 4] {
        if !self.contains(x, y) {
            return [Some(*self), None, None, None];
        }
        let cut = |x0: u8, y0: u8, x1: u8, y1: u8, ok: bool| {
            if ok {
                Some(Rect { x0, y0, x1, y1 })
            } else {
                None
            }
        };
        [
            cut(self.x0, self.y0, self.x1, y.wrapping_sub(1), y > self.y0),
            cut(self.x0, y + 1, self.x1, self.y1, y < self.y1),
            cut(self.x0, y, x.wrapping_sub(1), y, x > self.x0),
            cut(x + 1, y, self.x1, y, x < self.x1),
        ]
    }
}

/// [`full_grid`] without the tile `(x, y)`: at most eight rectangles that
/// together reach every Tensix tile of `grid` but that one, each exactly once.
pub fn full_grid_except(grid: &Tensix, x: u8, y: u8) -> [Option<Rect>; 8] {
    let mut out = [None; 8];
    let mut n = 0;
    for r in full_grid(grid).into_iter().flatten() {
        for piece in r.without(x, y).into_iter().flatten() {
            out[n] = Some(piece);
            n += 1;
        }
    }
    out
}

/// Why a [`MulticastWrite`] was refused.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MulticastError {
    /// Length zero or above one request, or the two L1 addresses are not
    /// congruent mod 16, or one is MMIO.
    Request(RequestError),
    /// The initiating tile is inside the rectangle: the initiator is never a
    /// recipient here (`NOC_CMD_BRCST_SRC_INCLUDE` is not exposed).
    ContainsInitiator,
}

/// Write `len` bytes of this tile's L1 at `from_local` to L1 address `to_addr` of
/// every tile of `rect`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MulticastWrite {
    pub from_local: u32,
    pub rect: Rect,
    pub to_addr: u32,
    pub len: u32,
}

impl MulticastWrite {
    /// The initiator registers to write before `CMD_CTRL`, eleven because
    /// `NOC_BRCST_EXCLUDE` is written too. Response-marked, so every recipient
    /// acknowledges; static virtual channel 4 ([`STATIC_VC_MULTICAST`]); path
    /// reserved; X major.
    pub fn registers(
        &self,
        me: (u8, u8),
        txn: TxnId,
        niu: Niu,
    ) -> Result<[(u64, u32); 11], MulticastError> {
        check_copy(self.from_local, self.to_addr, self.len).map_err(MulticastError::Request)?;
        if self.rect.contains(me.0, me.1) {
            return Err(MulticastError::ContainsInitiator);
        }
        let local = Endpoint {
            x: me.0,
            y: me.1,
            addr: self.from_local,
        };
        Ok([
            (TARG_ADDR_LO, self.from_local),
            (TARG_ADDR_MID, 0),
            (TARG_ADDR_HI, local.hi()),
            (RET_ADDR_LO, self.to_addr),
            (RET_ADDR_MID, 0),
            (RET_ADDR_HI, self.rect.hi(niu)),
            (PACKET_TAG, (txn.index() as u32) << 10),
            (
                CTRL,
                CMD_WR | RESP_MARKED | BRCST_PACKET | PATH_RESERVE | STATIC_VC_MULTICAST,
            ),
            (AT_LEN_BE, self.len),
            (AT_DATA, 0),
            (BRCST_EXCLUDE, 0),
        ])
    }

    /// The acknowledgements to count: one per recipient.
    pub const fn acks(&self) -> u32 {
        self.rect.tile_count()
    }
}

/// Where a multicast is, by the initiator's acknowledgement counter.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AckProgress {
    /// This many recipients have not acknowledged yet.
    Waiting { missing: u32 },
    /// Exactly the expected number arrived.
    Complete,
    /// More acknowledgements than recipients: other traffic shares the counter,
    /// so the count proves nothing about this multicast.
    Excess { extra: u32 },
}

/// Counts a multicast's acknowledgements: read `NIU_MST_WR_ACK_RECEIVED` before
/// writing `CMD_CTRL` (with nothing else response-marked in flight on this NIU),
/// then ask [`AckCount::progress`] of later reads.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AckCount {
    before: u32,
    expected: u32,
}

impl AckCount {
    pub const fn start(before: u32, write: &MulticastWrite) -> Self {
        AckCount {
            before,
            expected: write.acks(),
        }
    }

    pub const fn progress(&self, now: u32) -> AckProgress {
        let got = now.wrapping_sub(self.before);
        if got < self.expected {
            AckProgress::Waiting {
                missing: self.expected - got,
            }
        } else if got == self.expected {
            AckProgress::Complete
        } else {
            AckProgress::Excess {
                extra: got - self.expected,
            }
        }
    }
}

/// The word to write to `niu::CLEAR_OUTSTANDING` once the acks are counted.
pub const fn clear_outstanding(txn: TxnId) -> u32 {
    1 << txn.index()
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: TxnId = match TxnId::new(5) {
        Some(t) => t,
        None => panic!(),
    };

    #[test]
    fn rectangles_come_only_from_the_discovered_grid() {
        let full = Tensix::FULL;
        let harvested = Tensix::from_enabled_column_count(12).unwrap();
        assert!(Rect::new(&full, 3, 4, 4, 4).is_ok());
        // A harvested column (15 on a 12-column chip) is refused, though the same
        // rectangle is fine on a full part.
        assert!(Rect::new(&full, 14, 2, 15, 3).is_ok());
        assert_eq!(
            Rect::new(&harvested, 14, 2, 15, 3),
            Err(RectError::NotTensix { x: 15, y: 2 })
        );
        // Spanning the non-compute columns, the Ethernet row, the DRAM row.
        assert_eq!(
            Rect::new(&full, 7, 2, 10, 2),
            Err(RectError::NotTensix { x: 8, y: 2 })
        );
        assert_eq!(
            Rect::new(&full, 3, 1, 3, 2),
            Err(RectError::NotTensix { x: 3, y: 1 })
        );
        assert_eq!(
            Rect::new(&full, 3, 11, 3, 12),
            Err(RectError::NotTensix { x: 3, y: 12 })
        );
        // Reversed corners would wrap around the grid in hardware.
        assert_eq!(Rect::new(&full, 4, 4, 3, 4), Err(RectError::Reversed));
        assert_eq!(Rect::new(&full, 3, 5, 4, 4), Err(RectError::Reversed));
    }

    #[test]
    fn the_full_grid_is_two_rectangles_covering_every_tile_once() {
        for n in [0u8, 1, 7, 8, 12, 14] {
            let g = Tensix::from_enabled_column_count(n).unwrap();
            let rects = full_grid(&g);
            let mut seen = [[0u8; 12]; 17];
            for r in rects.iter().flatten() {
                for (x, y) in r.tiles() {
                    seen[x as usize][y as usize] += 1;
                }
            }
            let mut total = 0;
            for (x, col) in seen.iter().enumerate() {
                for (y, &n_seen) in col.iter().enumerate() {
                    assert!(n_seen <= 1, "({x},{y}) twice at {n} columns");
                    assert_eq!(
                        n_seen == 1,
                        g.contains(x as u8, y as u8),
                        "({x},{y}) at {n}"
                    );
                    total += n_seen as usize;
                }
            }
            assert_eq!(total, g.tile_count(), "{n} columns");
        }
    }

    #[test]
    fn the_full_grid_without_the_initiator_covers_every_other_tile_once() {
        for n in [1u8, 7, 8, 12, 14] {
            let g = Tensix::from_enabled_column_count(n).unwrap();
            for (mx, my) in [(1, 2), (3, 4), (7, 11), (10, 2), (1, 11)] {
                let mut seen = [[0u8; 12]; 17];
                for r in full_grid_except(&g, mx, my).iter().flatten() {
                    for (x, y) in r.tiles() {
                        seen[x as usize][y as usize] += 1;
                    }
                }
                for x in 0..17u8 {
                    for y in 0..12u8 {
                        let want = u8::from(g.contains(x, y) && (x, y) != (mx, my));
                        assert_eq!(
                            seen[x as usize][y as usize], want,
                            "({x},{y}) minus ({mx},{my}) at {n}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_hi_word_swaps_its_corners_on_noc1() {
        let r = Rect::new(&Tensix::FULL, 3, 4, 5, 6).unwrap();
        // EndX 5, EndY 6, StartX 3, StartY 4.
        assert_eq!(r.hi(Niu::Noc0), 5 | (6 << 6) | (3 << 12) | (4 << 18));
        assert_eq!(r.hi(Niu::Noc1), 3 | (4 << 6) | (5 << 12) | (6 << 18));
    }

    #[test]
    fn a_write_is_a_response_marked_path_reserved_class_2_broadcast() {
        let g = Tensix::FULL;
        let w = MulticastWrite {
            from_local: 0x2_0000,
            rect: Rect::new(&g, 4, 4, 5, 4).unwrap(),
            to_addr: 0x3_0000,
            len: 64,
        };
        let r = w.registers((3, 4), T, Niu::Noc0).unwrap();
        let reg = |o| r.iter().find(|(a, _)| *a == o).unwrap().1;
        let ctrl = reg(CTRL);
        assert_eq!(ctrl & 3, 2, "write");
        for (bit, name) in [
            (4, "RESP_MARKED"),
            (5, "BRCST_PACKET"),
            (7, "VC_STATIC"),
            (8, "PATH_RESERVE"),
        ] {
            assert_ne!(ctrl & (1 << bit), 0, "{name}");
        }
        for (bit, name) in [
            (6, "VC_LINKED"),
            (3, "WR_INLINE"),
            (16, "BRCST_XY"),
            (17, "SRC_INCLUDE"),
            (31, "L1_ACC"),
        ] {
            assert_eq!(ctrl & (1 << bit), 0, "{name}");
        }
        assert_eq!((ctrl >> 14) & 3, 0b10, "multicast class");
        assert_eq!((ctrl >> 13) & 1, 0, "buddy");
        assert_eq!(
            reg(TARG_ADDR_HI),
            3 | (4 << 6),
            "acks return to the initiator"
        );
        assert_eq!(reg(RET_ADDR_HI), w.rect.hi(Niu::Noc0));
        assert_eq!(reg(RET_ADDR_LO), 0x3_0000);
        assert_eq!(
            reg(BRCST_EXCLUDE),
            0,
            "written, not left to the last program"
        );
        assert_eq!(reg(PACKET_TAG), 5 << 10);
        assert_eq!(w.acks(), 2);
    }

    #[test]
    fn a_write_refuses_the_initiator_and_bad_copies() {
        let g = Tensix::FULL;
        let rect = Rect::new(&g, 3, 4, 5, 4).unwrap();
        let w = |from_local, to_addr, len| MulticastWrite {
            from_local,
            rect,
            to_addr,
            len,
        };
        assert_eq!(
            w(0x2_0000, 0x3_0000, 64).registers((4, 4), T, Niu::Noc0),
            Err(MulticastError::ContainsInitiator)
        );
        let me = (3, 5);
        assert_eq!(
            w(0x2_0000, 0x3_0004, 64).registers(me, T, Niu::Noc0),
            Err(MulticastError::Request(RequestError::Alignment))
        );
        assert_eq!(
            w(0x2_0000, 0x3_0000, 0).registers(me, T, Niu::Noc0),
            Err(MulticastError::Request(RequestError::Length))
        );
        assert_eq!(
            w(0x2_0000, 0xFFB2_0000, 16).registers(me, T, Niu::Noc0),
            Err(MulticastError::Request(RequestError::Alignment))
        );
    }

    #[test]
    fn acknowledgements_are_counted_not_outstanding_requests() {
        let g = Tensix::FULL;
        let w = MulticastWrite {
            from_local: 0x2_0000,
            rect: Rect::new(&g, 4, 4, 6, 5).unwrap(),
            to_addr: 0x3_0000,
            len: 16,
        };
        let c = AckCount::start(u32::MAX - 2, &w);
        assert_eq!(
            c.progress(u32::MAX - 2),
            AckProgress::Waiting { missing: 6 }
        );
        assert_eq!(c.progress(1), AckProgress::Waiting { missing: 2 });
        assert_eq!(c.progress(3), AckProgress::Complete);
        assert_eq!(c.progress(4), AckProgress::Excess { extra: 1 });
        assert_eq!(clear_outstanding(T), 1 << 5);
    }
}
