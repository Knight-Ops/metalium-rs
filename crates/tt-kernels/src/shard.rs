//! A matmul split across chips, with data moving between them only over
//! Ethernet.
//!
//! **Data plane.** Operands enter through chip 0 and the result leaves through
//! chip 0. Every other chip receives its operands and returns its result over
//! [`crate::link`]'s movers, relayed through intermediate chips when it is not
//! adjacent to chip 0. **Control plane.** The host loads programs, starts runs
//! and writes mover descriptors on every chip over that chip's own PCIe link:
//! small MMIO writes, none of them tensor data.
//!
//! **Why along `N`.** Each chip computes whole output columns with `K` kept
//! whole, so every output element is accumulated in the same order it would be
//! on one chip. The sharded result is therefore *bit-identical* to the
//! single-chip one, which is what lets the single-chip golden loss curve stay the
//! oracle. Splitting `K` (or data-parallel training with a gradient all-reduce)
//! would reorder the sums and need a weaker claim.

use std::collections::VecDeque;

use tt_device::{Device, Transport, TransportError, Window};
use tt_isa::noc::{Noc0, NocCoord};

use crate::link::{Dest, Dir, Link, LinkError, Mover, Source};
use crate::matmul::{self, Fidelity, SrcRoute, MATMUL_OUT, MATMUL_STAGE};
use crate::runtime::{self, Kernel, RoleImages, RunError, Schedule};
use crate::session::{reset_thread_state, reset_tile};

/// Why a sharded matmul failed.
#[derive(Debug)]
pub enum ShardError {
    Run(RunError),
    Link(LinkError),
    /// A chip with no route to chip 0.
    Unreachable(usize),
}

impl From<RunError> for ShardError {
    fn from(e: RunError) -> Self {
        ShardError::Run(e)
    }
}
impl From<LinkError> for ShardError {
    fn from(e: LinkError) -> Self {
        ShardError::Link(e)
    }
}
impl From<TransportError> for ShardError {
    fn from(e: TransportError) -> Self {
        ShardError::Run(RunError::Transport(e))
    }
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::Run(e) => write!(f, "{e}"),
            ShardError::Link(e) => write!(f, "{e}"),
            ShardError::Unreachable(c) => write!(f, "chip {c} has no Ethernet route to chip 0"),
        }
    }
}

impl std::error::Error for ShardError {}

/// Where a relayed piece (at most `mover::MAX_LEN`) waits on an intermediate
/// chip, in its relay tile; and where returning pieces land on chip 0.
pub const RELAY_AT: u32 = 0x2_0000;

/// One chip of a fabric.
pub struct Chip<T: Transport> {
    pub dev: Device<T>,
    window: Window,
    /// Where this chip computes.
    pub compute: NocCoord<Noc0>,
    /// Scratch for relaying, and on chip 0 the inbox for returning results.
    /// Distinct from `compute`, whose staging region a relay would overwrite.
    pub relay: NocCoord<Noc0>,
}

impl<T: Transport> Chip<T> {
    pub fn new(
        mut dev: Device<T>,
        compute: NocCoord<Noc0>,
        relay: NocCoord<Noc0>,
    ) -> Result<Self, TransportError> {
        assert_ne!(
            compute, relay,
            "the relay tile must not be the compute tile"
        );
        let window = dev.alloc_window(tt_device::tlb::WindowKind::TwoMib)?;
        Ok(Chip {
            dev,
            window,
            compute,
            relay,
        })
    }
}

struct Hop {
    /// Chip at the link's `a` end, and at its `b` end.
    ends: (usize, usize),
    mover: Mover,
}

/// Chips joined by Ethernet movers, with chip 0 as the host's way in and out.
pub struct Fabric<T: Transport> {
    pub chips: Vec<Chip<T>>,
    hops: Vec<Hop>,
    images: RoleImages<'static>,
    /// For each chip, the chips from 0 to it, inclusive.
    routes: Vec<Option<Vec<usize>>>,
}

fn two_mut<X>(v: &mut [X], i: usize, j: usize) -> (&mut X, &mut X) {
    assert_ne!(i, j);
    if i < j {
        let (l, r) = v.split_at_mut(j);
        (&mut l[i], &mut r[0])
    } else {
        let (l, r) = v.split_at_mut(i);
        (&mut r[0], &mut l[j])
    }
}

impl<T: Transport> Fabric<T> {
    /// Start `e1_image` on both ends of every link in `links`, each given as
    /// `(chip at a, chip at b, link)`, one link per adjacent pair.
    pub fn new(
        mut chips: Vec<Chip<T>>,
        links: &[(usize, usize, Link)],
        images: RoleImages<'static>,
        e1_image: &[u8],
    ) -> Result<Self, ShardError> {
        let mut hops = Vec::new();
        for &(p, q, link) in links {
            let (a, b) = two_mut(&mut chips, p, q);
            let mover = Mover::start(&mut a.dev, &a.window, &mut b.dev, &b.window, link, e1_image)?;
            hops.push(Hop {
                ends: (p, q),
                mover,
            });
        }
        let routes = bfs(chips.len(), &hops);
        Ok(Fabric {
            chips,
            hops,
            images,
            routes,
        })
    }

    pub fn len(&self) -> usize {
        self.chips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chips.is_empty()
    }

    /// The chips from 0 to `chip`, as Ethernet carries data between them.
    pub fn route(&self, chip: usize) -> Option<&[usize]> {
        self.routes[chip].as_deref()
    }

    /// One hop of `len <= mover::MAX_LEN` bytes from chip `p` to chip `q`.
    fn hop(
        &mut self,
        p: usize,
        q: usize,
        src: Source,
        dst: Dest,
        len: u32,
    ) -> Result<(), ShardError> {
        let h = self
            .hops
            .iter_mut()
            .find(|h| h.ends == (p, q) || h.ends == (q, p))
            .expect("routes only use existing hops");
        let dir = if h.ends.0 == p { Dir::AToB } else { Dir::BToA };
        let c = &mut self.chips[p];
        if let Source::Staged = src {
            unreachable!("staging goes through `send`")
        }
        h.mover.send(&mut c.dev, &c.window, dir, src, dst, len)?;
        Ok(())
    }

    /// Carry `data` from the host, entering at chip 0, to `addr` in `chip`'s
    /// compute tile, over Ethernet only. `chip` must not be 0.
    fn deliver(&mut self, chip: usize, addr: u32, data: &[u8]) -> Result<(), ShardError> {
        let route = self.routes[chip]
            .clone()
            .ok_or(ShardError::Unreachable(chip))?;
        let max = tt_isa::eth::mover::MAX_LEN as usize;
        for (i, piece) in data.chunks(max).enumerate() {
            let at = addr + (i * max) as u32;
            let len = piece.len() as u32;
            // Enter at chip 0: the host writes the first hop's staging buffer.
            let (p0, q0) = (route[0], route[1]);
            let last = route.len() - 1;
            let dst = |fab: &Self, q: usize, is_last: bool| {
                if is_last {
                    Dest::Tensix(fab.chips[q].compute, at)
                } else {
                    Dest::Tensix(fab.chips[q].relay, RELAY_AT)
                }
            };
            {
                let d = dst(self, q0, last == 1);
                let h = self
                    .hops
                    .iter_mut()
                    .find(|h| h.ends == (p0, q0) || h.ends == (q0, p0))
                    .unwrap();
                let dir = if h.ends.0 == p0 { Dir::AToB } else { Dir::BToA };
                let c = &mut self.chips[p0];
                h.mover.stage(&mut c.dev, &c.window, dir, piece)?;
                h.mover
                    .send(&mut c.dev, &c.window, dir, Source::Staged, d, len)?;
            }
            for k in 1..last {
                let (p, q) = (route[k], route[k + 1]);
                let src = Source::Tensix(self.chips[p].relay, RELAY_AT);
                let d = dst(self, q, k + 1 == last);
                self.hop(p, q, src, d, len)?;
            }
        }
        Ok(())
    }

    /// Carry `len` bytes at `addr` in `chip`'s compute tile back to chip 0's
    /// relay tile at [`RELAY_AT`], over Ethernet only, and read them there.
    fn collect(&mut self, chip: usize, addr: u32, len: usize) -> Result<Vec<u8>, ShardError> {
        let mut route = self.routes[chip]
            .clone()
            .ok_or(ShardError::Unreachable(chip))?;
        route.reverse();
        let max = tt_isa::eth::mover::MAX_LEN as usize;
        let mut out = vec![0u8; len];
        for off in (0..len).step_by(max) {
            let n = max.min(len - off) as u32;
            let last = route.len() - 1;
            for k in 0..last {
                let (p, q) = (route[k], route[k + 1]);
                let src = if k == 0 {
                    Source::Tensix(self.chips[p].compute, addr + off as u32)
                } else {
                    Source::Tensix(self.chips[p].relay, RELAY_AT)
                };
                // Intermediate relays and chip 0's inbox are both relay tiles.
                self.hop(p, q, src, Dest::Tensix(self.chips[q].relay, RELAY_AT), n)?;
            }
            let c = &mut self.chips[0];
            c.dev.read(
                &c.window,
                c.relay,
                RELAY_AT as u64,
                &mut out[off..off + n as usize],
            )?;
        }
        Ok(out)
    }

    /// One chunk (`[rows, inner, cols]`, fitting one run) on `chip`.
    #[allow(clippy::too_many_arguments)]
    fn run_on(
        &mut self,
        chip: usize,
        a: &[f32],
        b: &[f32],
        [m, k, n]: [usize; 3],
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<Vec<f32>, ShardError> {
        let images = self.images;
        let tile = self.chips[chip].compute;
        {
            let dev = &mut self.chips[chip].dev;
            reset_tile(dev, tile)?;
            reset_thread_state(dev, tile, &images)?;
        }
        if chip == 0 {
            let c = &mut self.chips[0];
            return Ok(matmul::matmul(
                &mut c.dev,
                tile,
                &images,
                a,
                b,
                [m, k, n],
                route,
                fidelity,
                budget,
            )?);
        }
        let (in_fmt, out_fmt) = route.formats();
        let staged = matmul::stage_matmul(a, b, m, k, n, in_fmt)?;
        self.deliver(chip, MATMUL_STAGE as u32, &staged.a)?;
        self.deliver(chip, staged.b_at as u32, &staged.b)?;
        let [unpack, math, pack] =
            matmul::matmul_roles(&staged.outputs, staged.sems, in_fmt, out_fmt, fidelity);
        let kernel = Kernel::new([&unpack, &math, &pack], Schedule::Concurrent(&staged.init));
        runtime::run(&mut self.chips[chip].dev, tile, &images, &kernel, budget)?;
        let packed = self.collect(chip, MATMUL_OUT as u32, staged.out_bytes())?;
        Ok(matmul::detilize_packed(&packed, m, n))
    }

    /// `A[m, k] @ B[k, n]`, row-major, with `B`'s columns split across every
    /// reachable chip in 32-column tiles, and each chip's share run in as many
    /// chunks as [`matmul::plan`] says. Bit-identical to the single-chip
    /// [`crate::session::matmul_on`] whenever `plan` keeps `K` whole for both,
    /// which it does for every shape a chip's share has if it does for the whole.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        [m, k, n]: [usize; 3],
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<Vec<f32>, ShardError> {
        assert_eq!(a.len(), m * k, "A is not [m, k]");
        assert_eq!(b.len(), k * n, "B is not [k, n]");
        let chips: Vec<usize> = (0..self.len())
            .filter(|&c| self.routes[c].is_some())
            .collect();
        let tiles = n.div_ceil(32).max(1);
        let per = tiles.div_ceil(chips.len());
        let mut c = vec![0f32; m * n];
        for (s, &chip) in chips.iter().enumerate() {
            let (j0, j1) = ((s * per * 32).min(n), ((s + 1) * per * 32).min(n));
            if j0 == j1 {
                continue;
            }
            let w = j1 - j0;
            let bj: Vec<f32> = (0..k)
                .flat_map(|r| b[r * n + j0..r * n + j1].iter().copied())
                .collect();
            let cj =
                matmul::matmul_chunked(a, &bj, [m, k, w], route, fidelity, |sa, sb, shape| {
                    self.run_on(chip, sa, sb, shape, route, fidelity, budget)
                })?;
            for r in 0..m {
                c[r * n + j0..r * n + j1].copy_from_slice(&cj[r * w..(r + 1) * w]);
            }
        }
        Ok(c)
    }
}

/// Breadth-first routes from chip 0 over the hops, so a relayed transfer takes
/// the fewest links.
fn bfs(n: usize, hops: &[Hop]) -> Vec<Option<Vec<usize>>> {
    let mut routes: Vec<Option<Vec<usize>>> = vec![None; n];
    if n == 0 {
        return routes;
    }
    routes[0] = Some(vec![0]);
    let mut queue = VecDeque::from([0usize]);
    while let Some(p) = queue.pop_front() {
        for h in hops {
            let q = match h.ends {
                (x, y) if x == p => y,
                (x, y) if y == p => x,
                _ => continue,
            };
            if routes[q].is_none() {
                let mut r = routes[p].clone().unwrap();
                r.push(q);
                routes[q] = Some(r);
                queue.push_back(q);
            }
        }
    }
    routes
}

/// Park E1 at both ends of every link: the mover is not left running after
/// the fabric that started it is gone. Best-effort, since `Drop` cannot report.
impl<T: Transport> Drop for Fabric<T> {
    fn drop(&mut self) {
        for h in &self.hops {
            let l = h.mover.link();
            for (chip, tile) in [(h.ends.0, l.a), (h.ends.1, l.b)] {
                let c = &mut self.chips[chip];
                let _ = c.dev.park_e1(&c.window, tile);
            }
        }
    }
}
