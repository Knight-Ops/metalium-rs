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
    /// A staged native operation does not fit the tile's L1 arena.
    Shape(String),
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
            ShardError::Shape(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for ShardError {}

/// Where a relayed piece (at most `mover::MAX_LEN`) waits on an intermediate
/// chip, in its relay tile; and where returning pieces land on chip 0.
pub const RELAY_AT: u32 = 0x2_0000;

/// One chip of a fabric.
pub struct Chip<T: Transport> {
    dev: Option<Device<T>>,
    session: Option<crate::session::Session<T>>,
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
            dev: Some(dev),
            session: None,
            window,
            compute,
            relay,
        })
    }

    pub fn device(&mut self) -> &mut Device<T> {
        self.parts().0
    }

    fn parts(&mut self) -> (&mut Device<T>, &Window) {
        let dev = match &mut self.session {
            Some(session) => session.device(),
            None => self.dev.as_mut().expect("chip owns its device"),
        };
        (dev, &self.window)
    }

    pub fn session(&mut self) -> &mut crate::session::Session<T> {
        self.session.as_mut().expect("resident fabric initialized")
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
            let (ad, aw) = a.parts();
            let (bd, bw) = b.parts();
            let mover = Mover::start(ad, aw, bd, bw, link, e1_image)?;
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

    /// Retain each chip's device in a session. Tensor payloads can then stay
    /// in GDDR while E1 moves whole tile slots between chips.
    pub fn enable_resident(
        &mut self,
        b: &'static [u8],
        nc: &'static [u8],
    ) -> Result<(), ShardError> {
        if self.is_empty() {
            return Err(ShardError::Unreachable(0));
        }
        for chip in &mut self.chips {
            if chip.session.is_none() {
                let dev = chip.dev.take().expect("chip device");
                let mut session = crate::session::Session::open(
                    dev,
                    self.images,
                    crate::session::TileChoice::Exactly(chip.compute.x(), chip.compute.y()),
                    |_, _| Ok(None),
                )
                .map_err(|e| ShardError::Shape(e.to_string()))?;
                session
                    .enable_dram(b, nc)
                    .map_err(|e| ShardError::Shape(e.to_string()))?;
                chip.session = Some(session);
            }
        }
        Ok(())
    }

    /// Copy slots over Ethernet, including intermediate L1 relays. No tensor
    /// data is read or written by the host. Both sessions are synchronized
    /// before E1 accesses their allocations.
    pub fn transfer_tensor(
        &mut self,
        from: usize,
        t: &crate::tensor::DramTensor,
        to: usize,
    ) -> Result<crate::tensor::DramTensor, crate::tensor::TensorError> {
        use crate::tensor::TensorError;
        if from >= self.len() || to >= self.len() {
            return Err(TensorError::Shape(
                "resident transfer chip is outside the fabric".into(),
            ));
        }
        self.chips[from].session().sync()?;
        if from == to {
            return self.chips[to].session().copy(t);
        }
        let peer = if from == 0 {
            to
        } else if to == 0 {
            from
        } else {
            return Err(TensorError::Shape(
                "resident transfer must pass through chip 0".into(),
            ));
        };
        let mut route = self
            .routes
            .get(peer)
            .and_then(Clone::clone)
            .ok_or_else(|| TensorError::Shape(format!("chip {peer} has no route")))?;
        if from != 0 {
            route.reverse();
        }
        self.chips[to].session().sync()?;
        let out = self.chips[to].session().empty(t.rows, t.cols, t.elem)?;
        let result = (|| {
            let [rt, ct] = t.grid();
            // Coalesce consecutive slots on the same source/destination
            // channels, up to one Ethernet packet's payload capacity.
            let mut ranges: Vec<(tt_isa::dram::DramRange, tt_isa::dram::DramRange)> = Vec::new();
            for r in 0..rt {
                for c in 0..ct {
                    let (src, dst) = (t.tile(r, c), out.tile(r, c));
                    if let Some((a, b)) = ranges.iter_mut().rev().find(|(a, b)| {
                        a.channel() == src.channel()
                            && b.channel() == dst.channel()
                            && a.offset() + a.len() == src.offset()
                            && b.offset() + b.len() == dst.offset()
                            && a.len() + src.len() <= u64::from(tt_isa::eth::mover::MAX_LEN)
                    }) {
                        *a = a
                            .channel()
                            .range(a.offset(), a.len() + src.len())
                            .expect("adjacent source slots");
                        *b = b
                            .channel()
                            .range(b.offset(), b.len() + dst.len())
                            .expect("adjacent destination slots");
                    } else {
                        ranges.push((src, dst));
                    }
                }
            }
            for (source, destination) in ranges {
                for hop in 0..route.len() - 1 {
                    let (p, q) = (route[hop], route[hop + 1]);
                    let src = if hop == 0 {
                        Source::Dram(source)
                    } else {
                        Source::Tensix(self.chips[p].relay, RELAY_AT)
                    };
                    let dst = if hop + 2 == route.len() {
                        Dest::Dram(destination)
                    } else {
                        Dest::Tensix(self.chips[q].relay, RELAY_AT)
                    };
                    self.hop(p, q, src, dst, source.len() as u32).map_err(|e| {
                        TensorError::Shape(format!("resident Ethernet transfer: {e}"))
                    })?;
                }
            }
            Ok(())
        })();
        if let Err(e) = result {
            let _ = self.chips[to].session().free(out);
            return Err(e);
        }
        out.set_pad(t.pad());
        Ok(out)
    }

    /// Split output columns across reachable chips, retaining inputs/results
    /// in GDDR. All copies, transposes and arithmetic execute on devices.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_resident(
        &mut self,
        a: &crate::tensor::DramTensor,
        ta: bool,
        b: &crate::tensor::DramTensor,
        tb: bool,
        route: SrcRoute,
        fidelity: Fidelity,
        budget: u64,
    ) -> Result<crate::tensor::DramTensor, crate::tensor::TensorError> {
        use crate::tensor::{BlockMove, DramTensor, TensorError};
        fn transpose<T: Transport>(
            session: &mut crate::session::Session<T>,
            t: &DramTensor,
        ) -> Result<DramTensor, TensorError> {
            session.copy_blocks(
                t,
                &[BlockMove {
                    from: [0, 0],
                    to: [0, 0],
                    extent: [t.cols, t.rows],
                    transposed: true,
                }],
                [t.cols, t.rows],
            )
        }
        let mut temps: Vec<(usize, DramTensor)> = Vec::new();
        let result = (|| {
            let a = if ta {
                let t = transpose(self.chips[0].session(), a)?;
                temps.push((0, t.clone()));
                t
            } else {
                a.clone()
            };
            let b = if tb {
                let t = transpose(self.chips[0].session(), b)?;
                temps.push((0, t.clone()));
                t
            } else {
                b.clone()
            };
            if a.cols != b.rows {
                return Err(TensorError::Shape(
                    "mesh matmul inner dimensions differ".into(),
                ));
            }
            let peers: Vec<_> = (0..self.len())
                .filter(|&c| self.routes[c].is_some())
                .collect();
            let per = b.cols.div_ceil(32).div_ceil(peers.len());
            let mut results = Vec::new();
            for (i, &peer) in peers.iter().enumerate() {
                let start = (i * per * 32).min(b.cols);
                let end = ((i + 1) * per * 32).min(b.cols);
                if start == end {
                    continue;
                }
                let width = end - start;
                let slice = self.chips[0].session().copy_blocks(
                    &b,
                    &[BlockMove {
                        from: [0, start],
                        to: [0, 0],
                        extent: [b.rows, width],
                        transposed: false,
                    }],
                    [b.rows, width],
                )?;
                temps.push((0, slice.clone()));
                let (pa, pb) = if peer == 0 {
                    (a.clone(), slice)
                } else {
                    let pa = self.transfer_tensor(0, &a, peer)?;
                    temps.push((peer, pa.clone()));
                    let pb = self.transfer_tensor(0, &slice, peer)?;
                    temps.push((peer, pb.clone()));
                    (pa, pb)
                };
                let c = self.chips[peer]
                    .session()
                    .matmul_dram(&pa, false, &pb, false, route, fidelity, budget)?;
                temps.push((peer, c.clone()));
                let c = transpose(self.chips[peer].session(), &c)?;
                temps.push((peer, c.clone()));
                let c = if peer == 0 {
                    c
                } else {
                    let t = self.transfer_tensor(peer, &c, 0)?;
                    temps.push((0, t.clone()));
                    t
                };
                results.push(c);
            }
            let refs: Vec<_> = results.iter().collect();
            let rows: Vec<_> = results
                .iter()
                .enumerate()
                .flat_map(|(i, t)| (0..t.rows).map(move |r| (i, r)))
                .collect();
            let combined = self.chips[0].session().gather_rows(&refs, &rows, a.rows)?;
            temps.push((0, combined.clone()));
            transpose(self.chips[0].session(), &combined)
        })();
        for (peer, t) in temps.into_iter().rev() {
            let _ = self.chips[peer].session().free(t);
        }
        result
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
        let (dev, window) = c.parts();
        h.mover.send(dev, window, dir, src, dst, len)?;
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
                let (dev, window) = c.parts();
                h.mover.stage(dev, window, dir, piece)?;
                h.mover.send(dev, window, dir, Source::Staged, d, len)?;
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
            let relay = c.relay;
            let (dev, window) = c.parts();
            dev.read(
                window,
                relay,
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
            let dev = self.chips[chip].device();
            reset_tile(dev, tile)?;
            reset_thread_state(dev, tile, &images)?;
        }
        if chip == 0 {
            let c = &mut self.chips[0];
            return Ok(matmul::matmul(
                c.device(),
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
        runtime::run(self.chips[chip].device(), tile, &images, &kernel, budget)?;
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
        if self.chips.iter().any(|chip| chip.session.is_some()) {
            return Err(ShardError::Shape(
                "resident fabric: use matmul_resident; legacy kernels would reset resident state"
                    .into(),
            ));
        }
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

    /// Full F32 sum/mean for legacy fabrics without resident sessions.
    /// Input and intermediate tiles are staged through L1 on chip 0.
    /// Both reduction passes must fit the L1 arena and role program slots;
    /// unsupported shapes return an error instead of using host arithmetic.
    pub fn full_reduce(
        &mut self,
        values: &[f32],
        [rows, cols]: [usize; 2],
        mean: bool,
        budget: u64,
    ) -> Result<f32, ShardError> {
        if self.chips.iter().any(|chip| chip.session.is_some()) {
            return Err(ShardError::Shape("resident fabric: use session reductions; legacy kernels would reset resident state".into()));
        }
        use crate::sfpu::reduce::Axis;
        if rows == 0 || cols == 0 || values.len() != rows * cols {
            return Err(ShardError::Shape(
                "native full reduction needs a nonempty matrix".into(),
            ));
        }
        if self.is_empty() {
            return Err(ShardError::Unreachable(0));
        }
        let column = self.reduce_staged(values, [rows, cols], Axis::Cols, None, budget)?;
        let sum = self.reduce_staged(
            &column,
            [rows, 1],
            Axis::Rows,
            mean.then_some((rows * cols) as f32),
            budget,
        )?;
        Ok(sum[0])
    }

    fn reduce_staged(
        &mut self,
        values: &[f32],
        [rows, cols]: [usize; 2],
        axis: crate::sfpu::reduce::Axis,
        divisor: Option<f32>,
        budget: u64,
    ) -> Result<Vec<f32>, ShardError> {
        use crate::sfpu::kernel::{A_ROW, OUT_ROW};
        use crate::sfpu::ops::{kind_sfpu, program};
        use crate::sfpu::reduce::{math_programs, plan_layout, roles, Axis, ReduceOp};
        use crate::sfpu::{Format, LReg, LoopPolicy, Program};
        use tt_isa::dm::{TILE_DATA, TILE_SLOT};

        let [rt, ct] = [rows.div_ceil(32), cols.div_ceil(32)];
        let (outputs, per, last, dims) = match axis {
            Axis::Cols => (rt, ct, cols, [rows, 1]),
            Axis::Rows => (ct, rt, rows, [1, cols]),
        };
        let valid = ((last - 1) % 32 + 1) as u32;
        let layout = plan_layout(outputs, per).map_err(|e| ShardError::Shape(e.to_string()))?;
        let (inputs, mut finish) = math_programs(ReduceOp::Sum, axis, per, valid);
        if let Some(divisor) = divisor {
            // The division program reads A and writes OUT. Copy the sum
            // within Dst before running it; no scalar arithmetic on the host.
            let mut copy = Program::with_policy(LoopPolicy::Unrolled);
            for offset in (0..64).step_by(2) {
                copy.load(LReg::L0, Format::Fp32, OUT_ROW + offset);
                copy.store(LReg::L0, Format::Fp32, A_ROW + offset);
            }
            finish.extend(copy.finish());
            finish.extend(
                program(kind_sfpu::DIV_SCALAR, divisor)
                    .expect("SFPU division")
                    .1,
            );
        }
        let code = roles(&layout, &inputs, &finish);
        let tiled = matmul::tilize_f32_fp32(values, rows, cols);
        let mut stages = Vec::with_capacity(outputs * per);
        for k in 0..outputs {
            for n in 0..per {
                let tile = match axis {
                    Axis::Cols => k * ct + n,
                    Axis::Rows => n * ct + k,
                };
                stages.push((
                    layout.in_at + (k * per + n) as u64 * TILE_SLOT,
                    &tiled[tile * matmul::TILE_IMAGE_BYTES..(tile + 1) * matmul::TILE_IMAGE_BYTES],
                ));
            }
        }
        let reads: Vec<_> = (0..outputs)
            .map(|k| (layout.out_at + k as u64 * TILE_SLOT + TILE_DATA, 4096))
            .collect();
        let mut kernel = Kernel::new(
            [&code[0], &code[1], &code[2]],
            Schedule::Concurrent(&layout.init),
        );
        kernel.stage = &stages;
        kernel.read_back = &reads;
        let tile = self.chips[0].compute;
        let dev = self.chips[0].device();
        reset_tile(dev, tile)?;
        reset_thread_state(dev, tile, &self.images)?;
        let outcome = runtime::run(dev, tile, &self.images, &kernel, budget)?;
        let packed: Vec<_> = outcome.l1.into_iter().flatten().collect();
        Ok(matmul::detilize_packed(&packed, dims[0], dims[1]))
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
                if c.dev.is_some() || c.session.is_some() {
                    let (dev, window) = c.parts();
                    let _ = dev.park_e1(window, tile);
                }
            }
        }
    }
}
