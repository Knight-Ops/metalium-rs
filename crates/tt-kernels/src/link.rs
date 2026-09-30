//! Moving data between chips over Ethernet.
//!
//! A [`Link`] is a pair of cabled Ethernet tiles, one on each of two chips. A
//! [`Mover`] is the `eth_e1` image (`tt_isa::eth::mover`) running on E1 at both
//! ends of it. With one, a buffer in a Tensix tile of one chip reaches a Tensix
//! tile of the other with no host access to the second chip: the sender pulls
//! it over the NoC, TT-link carries it, the receiver checks it against the
//! record's checksum and pushes it over its own NoC, then acknowledges. The host
//! waits for that acknowledgement on the *sending* chip.

use std::time::{Duration, Instant};

use tt_device::core_control::CYCLES_PER_POLL;
use tt_device::{Device, Transport, TransportError, Window};
use tt_isa::eth::{mover, EthTile, Ethernet};
use tt_isa::mailbox::status;
use tt_isa::noc::{Noc0, NocCoord};

/// Why a link operation did not complete.
#[derive(Debug)]
pub enum LinkError {
    Transport(TransportError),
    /// The two chips name no cabled pair of tiles, or not the one asked for.
    NoLink(&'static str),
    /// A request the mover would refuse, refused here first.
    Invalid(&'static str),
    /// The mover reported [`mover::ERROR`].
    Mover {
        code: u32,
    },
    /// No acknowledgement within the deadline. The link may have dropped.
    TimedOut {
        seq: u32,
        acked: u32,
    },
    /// E1 did not reach its prologue.
    NotStarted,
}

impl From<TransportError> for LinkError {
    fn from(e: TransportError) -> Self {
        LinkError::Transport(e)
    }
}

impl std::fmt::Display for LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkError::Transport(e) => write!(f, "{e}"),
            LinkError::NoLink(why) => write!(f, "no link: {why}"),
            LinkError::Invalid(why) => write!(f, "refused: {why}"),
            LinkError::Mover { code } => write!(f, "the E1 mover gave up with code {code}"),
            LinkError::TimedOut { seq, acked } => {
                write!(
                    f,
                    "transfer {seq} was not acknowledged (last acknowledged: {acked})"
                )
            }
            LinkError::NotStarted => write!(f, "E1 did not start"),
        }
    }
}

impl std::error::Error for LinkError {}

pub type Result<T> = std::result::Result<T, LinkError>;

/// A cabled pair: `a` is a tile of the first chip, `b` of the second.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Link {
    pub a: EthTile,
    pub b: EthTile,
}

/// Every cabled pair between two chips, read from the chips.
///
/// A tile on `a` and one on `b` are a pair when each is Up and each names the
/// other's MAC as its partner (the base firmware's chip-info exchange). Both
/// directions are required, so a tile whose exchange is stale cannot pair.
pub fn discover<T: Transport, U: Transport>(
    a: &mut Device<T>,
    wa: &Window,
    ga: &Ethernet,
    b: &mut Device<U>,
    wb: &Window,
    gb: &Ethernet,
) -> Result<Vec<Link>> {
    let mut theirs = Vec::new();
    for t in gb.tiles().filter_map(|x| gb.tile(x)) {
        let s = b.eth_link_state(wb, t)?;
        if s.is_up() {
            theirs.push((t, s));
        }
    }
    let mut links = Vec::new();
    for t in ga.tiles().filter_map(|x| ga.tile(x)) {
        let s = a.eth_link_state(wa, t)?;
        if !s.is_up() {
            continue;
        }
        if let Some((u, _)) = theirs
            .iter()
            .find(|(_, o)| Some(o.own_mac) == s.peer_mac && o.peer_mac == Some(s.own_mac))
        {
            links.push(Link { a: t, b: *u });
        }
    }
    Ok(links)
}

/// Where data comes from on the sending chip.
#[derive(Copy, Clone, Debug)]
pub enum Source {
    /// A Tensix tile's L1.
    Tensix(NocCoord<Noc0>, u32),
    /// The sender's staging buffer, as the host left it ([`Mover::stage`]).
    Staged,
}

/// Where data goes on the receiving chip.
#[derive(Copy, Clone, Debug)]
pub enum Dest {
    Tensix(NocCoord<Noc0>, u32),
    /// Left in the receiver's landing buffer ([`mover::RX_LAND`]).
    Landed,
}

/// Which end sends.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Dir {
    AToB,
    BToA,
}

/// The mover running on both ends of a link.
pub struct Mover {
    link: Link,
    seq: [u32; 2],
    deadline: Duration,
}

/// The bytes of the mover's mailbox the host zeroes before start, so a record
/// or sequence number from a previous run cannot be mistaken for a new one.
const MAILBOX_SPAN: usize = (mover::ACK_STAGE + 16 - tt_isa::eth::MAILBOX_BASE) as usize;

fn start_end<T: Transport>(d: &mut Device<T>, w: &Window, t: EthTile, image: &[u8]) -> Result<()> {
    d.park_e1(w, t)?;
    let mut mb = vec![0u8; MAILBOX_SPAN];
    let put = |mb: &mut Vec<u8>, addr: u64, v: u32| {
        let i = (addr - tt_isa::eth::MAILBOX_BASE) as usize;
        mb[i..i + 4].copy_from_slice(&v.to_le_bytes());
    };
    let c = t.coord();
    put(&mut mb, mover::MY_X, c.x() as u32);
    put(&mut mb, mover::MY_Y, c.y() as u32);
    put(
        &mut mb,
        mover::LANDING_WAIT,
        u32::from(!d.transport().is_simulated()),
    );
    d.eth_write(w, t, tt_isa::eth::MAILBOX_BASE, &mb)?;
    d.load_and_start_e1(w, t, image)?;
    match d.wait_for_e1_status(w, t, 400_000, |s| s == status::RUNNING)? {
        Ok(_) => Ok(()),
        Err(_) => Err(LinkError::NotStarted),
    }
}

impl Mover {
    /// Start `image` (the `eth_e1` mover) on E1 at both ends of `link`.
    #[allow(clippy::too_many_arguments)]
    pub fn start<T: Transport, U: Transport>(
        a: &mut Device<T>,
        wa: &Window,
        b: &mut Device<U>,
        wb: &Window,
        link: Link,
        image: &[u8],
    ) -> Result<Self> {
        start_end(a, wa, link.a, image)?;
        start_end(b, wb, link.b, image)?;
        Ok(Mover {
            link,
            seq: [0, 0],
            deadline: Duration::from_secs(2),
        })
    }

    pub fn link(&self) -> Link {
        self.link
    }

    fn end(&self, dir: Dir) -> (EthTile, usize) {
        match dir {
            Dir::AToB => (self.link.a, 0),
            Dir::BToA => (self.link.b, 1),
        }
    }

    /// Write `data` into the sending end's staging buffer, for [`Source::Staged`].
    pub fn stage<T: Transport>(
        &self,
        d: &mut Device<T>,
        w: &Window,
        dir: Dir,
        data: &[u8],
    ) -> Result<()> {
        if data.len() > mover::MAX_LEN as usize {
            return Err(LinkError::Invalid("more than one transfer's worth"));
        }
        let t = self.end(dir).0;
        d.eth_write(w, t, mover::TX_STAGE, data)?;
        // Host writes are posted: without a read back, E1 can be told to send
        // before the last of them has landed, and would send stale bytes. This
        // is the silicon hazard `silicon_eth_bench::raw_no_receiver_polling`
        // measured (57 of 480 transfers with a late tail, 0 of 480 fenced).
        if let Some(last) = data.len().checked_sub(4) {
            let mut word = [0u8; 4];
            d.eth_read(w, t, mover::TX_STAGE + last as u64, &mut word)?;
        }
        Ok(())
    }

    /// Read `out.len()` bytes of the receiving end's landing buffer.
    pub fn landed<T: Transport>(
        &self,
        d: &mut Device<T>,
        w: &Window,
        dir: Dir,
        out: &mut [u8],
    ) -> Result<()> {
        let to = match dir {
            Dir::AToB => self.link.b,
            Dir::BToA => self.link.a,
        };
        d.eth_read(w, to, mover::RX_LAND, out)?;
        Ok(())
    }

    /// Move `len` bytes from `src` on the sending chip to `dst` on the other,
    /// and wait for the receiver's acknowledgement -- which arrives on the
    /// sending chip, so `d` is the only device this touches.
    pub fn send<T: Transport>(
        &mut self,
        d: &mut Device<T>,
        w: &Window,
        dir: Dir,
        src: Source,
        dst: Dest,
        len: u32,
    ) -> Result<()> {
        if len == 0 || len % 16 != 0 || len > mover::MAX_LEN {
            return Err(LinkError::Invalid(
                "length must be a nonzero multiple of 16, at most MAX_LEN",
            ));
        }
        let (tile, i) = self.end(dir);
        let (sx, sy, sa) = match src {
            Source::Tensix(c, addr) => (c.x() as u32, c.y() as u32, addr),
            Source::Staged => (mover::NO_TILE, 0, 0),
        };
        let (dx, dy, da) = match dst {
            Dest::Tensix(c, addr) => (c.x() as u32, c.y() as u32, addr),
            Dest::Landed => (mover::NO_TILE, 0, 0),
        };
        if sa % 16 != 0 || da % 16 != 0 {
            return Err(LinkError::Invalid(
                "Tensix addresses must be 16-byte aligned",
            ));
        }
        self.seq[i] = self.seq[i].wrapping_add(1).max(1);
        let seq = self.seq[i];
        let c = tile.coord();
        for (addr, v) in [
            (mover::SEND_SRC_X, sx),
            (mover::SEND_SRC_Y, sy),
            (mover::SEND_SRC_ADDR, sa),
            (mover::SEND_LEN, len),
            (mover::SEND_DST_X, dx),
            (mover::SEND_DST_Y, dy),
            (mover::SEND_DST_ADDR, da),
            (mover::SEND_SEQ, seq),
        ] {
            d.write32(w, c, addr, v)?;
        }
        self.wait_acked(d, w, tile, seq)
    }

    fn wait_acked<T: Transport>(
        &self,
        d: &mut Device<T>,
        w: &Window,
        t: EthTile,
        seq: u32,
    ) -> Result<()> {
        let started = Instant::now();
        let simulated = d.transport().is_simulated();
        let mut ticks = 0u64;
        loop {
            let acked = d.read32(w, t.coord(), mover::ACKED)?;
            if acked == seq {
                return Ok(());
            }
            let code = d.read32(w, t.coord(), mover::ERROR)?;
            if code != 0 {
                return Err(LinkError::Mover { code });
            }
            let expired = if simulated {
                ticks > 20_000_000
            } else {
                started.elapsed() > self.deadline
            };
            if expired {
                return Err(LinkError::TimedOut { seq, acked });
            }
            d.tick(CYCLES_PER_POLL);
            ticks += CYCLES_PER_POLL as u64;
        }
    }
}
