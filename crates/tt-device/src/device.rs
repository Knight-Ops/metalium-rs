//! A Blackhole device reached through TLB windows.
//!
//! Wraps a [`Transport`] with the two things every caller above it needs: a window
//! allocator, and transfers that are split so no access ever straddles a window.

use std::collections::BTreeMap;

use tt_isa::noc::{niu, ChipId, NocCoord, NocId, TileType};

use crate::tlb::{
    self, TlbConfig, WindowKind, KERNEL_RESERVED_WINDOW, NUM_2MIB_WINDOWS, NUM_WINDOWS,
};
use crate::{Bar, Result, Transport, TransportError};

/// A TLB window reserved for this `Device`'s use.
///
/// Returned by [`Device::alloc_window`] and released with [`Device::free_window`].
/// Not `Copy`: two callers holding the same index would retarget the window under
/// each other.
#[derive(Debug, PartialEq, Eq)]
pub struct Window {
    index: u16,
    kind: WindowKind,
}

impl Window {
    pub fn index(&self) -> u16 {
        self.index
    }

    pub fn kind(&self) -> WindowKind {
        self.kind
    }
}

/// What a window is currently pointing at, so that redundant reconfiguration can be
/// skipped.
///
/// Shadowing is not an optimisation, it is a necessity: the configuration registers
/// are write-only. It happens to also be an optimisation, since each configuration
/// write is a slow uncached MMIO round trip — the same reason `ethdump.c:264` caches
/// them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Shadow {
    words: [u32; 3],
    readable: bool,
}

/// A tile found by [`Device::discover_tiles`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile<N: NocId> {
    pub coord: NocCoord<N>,
    pub kind: TileType,
    /// Index of this tile within its type.
    pub index: u8,
    /// True if the tile is fused off and must not be used.
    pub harvested: bool,
}

pub struct Device<T: Transport> {
    transport: T,
    chip: ChipId,
    /// Windows not currently handed out, lowest first.
    free: Vec<u16>,
    shadow: BTreeMap<u16, Shadow>,
}

impl<T: Transport> Device<T> {
    /// Take ownership of a transport and confirm it is a Blackhole.
    ///
    /// Every address constant in this crate is Blackhole-specific; against another
    /// part they are merely plausible, which is the worst kind of wrong.
    pub fn open(mut transport: T) -> Result<Self> {
        transport.verify_is_blackhole()?;

        // Window 201 belongs to the kernel driver and must never be handed out --
        // honoured on the simulator too, where there is no driver to collide with,
        // so that allocation behaviour is identical on silicon.
        let free: Vec<u16> = (0..NUM_WINDOWS)
            .filter(|&i| i != KERNEL_RESERVED_WINDOW)
            .collect();

        // Taken from the transport rather than assumed: a transport is one chip,
        // and it is the only thing that knows which. Asserting it here instead
        // would let a `Device` claim to be a chip it does not address.
        let chip = transport.chip();

        Ok(Device {
            transport,
            chip,
            free,
            shadow: BTreeMap::new(),
        })
    }

    pub fn chip(&self) -> ChipId {
        self.chip
    }

    pub fn transport(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Advance simulated time. A no-op against silicon.
    pub fn tick(&mut self, n: u32) {
        self.transport.tick(n);
    }

    /// Reserve a window of the requested geometry.
    pub fn alloc_window(&mut self, kind: WindowKind) -> Result<Window> {
        let position = self
            .free
            .iter()
            .position(|&i| tlb::window_kind(i) == Some(kind));
        match position {
            Some(p) => {
                let index = self.free.remove(p);
                Ok(Window { index, kind })
            }
            None => Err(TransportError::OutOfBounds {
                bar: kind.bar(),
                offset: 0,
                len: 0,
            }),
        }
    }

    /// Return a window to the pool.
    ///
    /// The hardware configuration is left as it was: there is no "unconfigured"
    /// state to restore it to, and the next allocator will reconfigure before use.
    /// The shadow entry is dropped so the next holder cannot inherit a stale belief
    /// about where it points.
    pub fn free_window(&mut self, window: Window) {
        self.shadow.remove(&window.index);
        let index = window.index;
        let insert_at = self.free.partition_point(|&i| i < index);
        self.free.insert(insert_at, index);
    }

    /// Point a window at a device address in a tile, if it is not already there.
    fn retarget<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        base_address: u64,
    ) -> Result<()> {
        let config = TlbConfig::unicast(base_address, coord);
        let words = config
            .encode(window.kind)
            .map_err(|_| TransportError::Misaligned {
                bar: window.kind.bar(),
                offset: base_address,
                len: window.kind.size(),
                reason: "device address is not window-aligned",
            })?;
        let shadow = Shadow {
            words,
            readable: config.is_readable(),
        };

        if self.shadow.get(&window.index) == Some(&shadow) {
            return Ok(());
        }
        tlb::write_config(&mut self.transport, window.index, &config)?;
        self.shadow.insert(window.index, shadow);
        Ok(())
    }

    /// Write `data` to `address` in the tile at `coord`, through `window`.
    ///
    /// Splits the transfer so that no single access straddles a window boundary,
    /// retargeting the window as it goes.
    pub fn write<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        data: &[u8],
    ) -> Result<()> {
        self.transfer(
            window,
            coord,
            address,
            data.len(),
            |dev, bar, off, range| dev.transport.bar_write(bar, off, &data[range]),
        )
    }

    /// Read from `address` in the tile at `coord` into `out`, through `window`.
    pub fn read<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        out: &mut [u8],
    ) -> Result<()> {
        // Split borrows: the closure needs `out` mutably while `transfer` needs
        // `self`, so the chunking is done against a raw range and applied here.
        let len = out.len();
        let mut done = 0usize;
        let chunks = self.plan(window, address, len)?;
        for chunk in chunks {
            self.retarget(window, coord, chunk.window_base)?;
            let bar_offset = tlb::window_bar_offset(window.index).expect("allocated window")
                + chunk.offset_in_window;
            self.transport.bar_read(
                window.kind.bar(),
                bar_offset,
                &mut out[done..done + chunk.len],
            )?;
            done += chunk.len;
        }
        debug_assert_eq!(done, len);
        Ok(())
    }

    fn transfer<N: NocId, F>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        len: usize,
        mut apply: F,
    ) -> Result<()>
    where
        F: FnMut(&mut Self, Bar, u64, std::ops::Range<usize>) -> Result<()>,
    {
        let mut done = 0usize;
        for chunk in self.plan(window, address, len)? {
            self.retarget(window, coord, chunk.window_base)?;
            let bar_offset = tlb::window_bar_offset(window.index).expect("allocated window")
                + chunk.offset_in_window;
            apply(self, window.kind.bar(), bar_offset, done..done + chunk.len)?;
            done += chunk.len;
        }
        debug_assert_eq!(done, len);
        Ok(())
    }

    /// Break a transfer into per-window pieces.
    fn plan(&self, window: &Window, address: u64, len: usize) -> Result<Vec<Chunk>> {
        let size = window.kind.size();
        let mut chunks = Vec::new();
        let mut address = address;
        let mut remaining = len as u64;

        while remaining > 0 {
            let window_base = address & !(size - 1);
            let offset_in_window = address - window_base;
            let here = remaining.min(size - offset_in_window);
            chunks.push(Chunk {
                window_base,
                offset_in_window,
                len: here as usize,
            });
            address = address
                .checked_add(here)
                .ok_or(TransportError::OutOfBounds {
                    bar: window.kind.bar(),
                    offset: address,
                    len: remaining,
                })?;
            remaining -= here;
        }
        Ok(chunks)
    }

    /// Read one dword from a tile.
    pub fn read32<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
    ) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read(window, coord, address, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Write one dword to a tile.
    pub fn write32<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
        address: u64,
        value: u32,
    ) -> Result<()> {
        self.write(window, coord, address, &value.to_le_bytes())
    }

    /// Identify the tile at `coord`, or `None` if nothing answers there.
    ///
    /// Reads `NOC_ENDPOINT_ID` and `NIU_CFG_0` through NoC #0's NIU, the same probe
    /// `ethdump.c:462-474` uses to walk the grid.
    pub fn probe_tile<N: NocId>(
        &mut self,
        window: &Window,
        coord: NocCoord<N>,
    ) -> Result<Option<Tile<N>>> {
        let endpoint = self.read32(window, coord, niu::NOC0_BASE + niu::NOC_ENDPOINT_ID)?;
        if endpoint == 0 || endpoint == u32::MAX {
            return Ok(None);
        }
        let cfg = self.read32(window, coord, niu::NOC0_BASE + niu::NIU_CFG_0)?;
        Ok(Some(Tile {
            coord,
            kind: TileType::from_endpoint_id(endpoint),
            index: tt_isa::noc::endpoint_tile_index(endpoint),
            harvested: cfg & niu::NIU_CFG_0_HARVESTED != 0,
        }))
    }

    /// Walk a rectangle of the NoC grid, reporting every tile that answers.
    ///
    /// Discovery rather than hardcoding: which tiles are present depends on
    /// harvesting, and whether coordinates are translated depends on `NIU_CFG_0`
    /// bit 14. Probing survives both.
    pub fn discover_tiles<N: NocId>(
        &mut self,
        window: &Window,
        xs: std::ops::Range<u8>,
        ys: std::ops::Range<u8>,
    ) -> Result<Vec<Tile<N>>> {
        let mut found = Vec::new();
        for y in ys.clone() {
            for x in xs.clone() {
                let Some(coord) = NocCoord::<N>::new(x, y) else {
                    continue;
                };
                if let Some(tile) = self.probe_tile(window, coord)? {
                    found.push(tile);
                }
            }
        }
        Ok(found)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Chunk {
    /// Window-aligned device address this piece needs the window pointed at.
    window_base: u64,
    offset_in_window: u64,
    len: usize,
}

/// Number of 2 MiB windows available to callers, after the kernel reservation.
pub const USABLE_2MIB_WINDOWS: u16 = NUM_2MIB_WINDOWS - 1;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bar, ConfigOffset};
    use tt_isa::noc::Noc0;

    /// A transport that models just enough of the chip to test windowing: it
    /// decodes TLB configuration writes and routes data accesses to a per-tile
    /// sparse memory, exactly as the hardware's address translation would.
    ///
    /// Without the decode step two windows onto the same tile address would appear
    /// to be different memory, and a round-trip through two windows -- the whole
    /// point of the step 2 gate -- could not be expressed here at all. The
    /// authoritative version of that gate runs against ttsim; this one keeps the
    /// chunking and shadowing logic testable without a simulator.
    #[derive(Default)]
    struct FakeTransport {
        writes: Vec<(Bar, u64, usize)>,
        reads: Vec<(Bar, u64, usize)>,
        /// Raw configuration words, indexed by window.
        tlb_cfg: BTreeMap<u16, [u32; 3]>,
        /// Tile memory, keyed by (packed coordinate, device address).
        mem: BTreeMap<(u16, u64), u8>,
    }

    impl FakeTransport {
        /// Reverse of `TlbConfig::encode` for a 2 MiB window: recover the tile and
        /// device address a BAR0 offset resolves to.
        fn translate(&self, offset: u64) -> Option<(u16, u64)> {
            let index = (offset / WindowKind::TwoMib.size()) as u16;
            let offset_in_window = offset % WindowKind::TwoMib.size();
            let w = self.tlb_cfg.get(&index)?;
            let local_offset = (w[0] as u64) | (((w[1] & 0x7FF) as u64) << 32);
            let x = ((w[1] >> 11) & 0x3F) as u16;
            let y = ((w[1] >> 17) & 0x3F) as u16;
            Some((x | (y << 6), (local_offset << 21) | offset_in_window))
        }

        fn is_config(bar: Bar, offset: u64) -> bool {
            bar == Bar::Bar0 && offset >= tlb::CONFIG_BASE
        }
    }

    impl Transport for FakeTransport {
        fn bar_read(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) -> Result<()> {
            self.reads.push((bar, offset, dst.len()));
            assert!(
                !Self::is_config(bar, offset),
                "TLB config registers are write-only"
            );
            let (tile, addr) = self.translate(offset).expect("window is configured");
            for (i, b) in dst.iter_mut().enumerate() {
                *b = self.mem.get(&(tile, addr + i as u64)).copied().unwrap_or(0);
            }
            Ok(())
        }

        fn bar_write(&mut self, bar: Bar, offset: u64, src: &[u8]) -> Result<()> {
            self.writes.push((bar, offset, src.len()));
            if Self::is_config(bar, offset) {
                let reg = (offset - tlb::CONFIG_BASE) / 4;
                let (index, word) = ((reg / 3) as u16, (reg % 3) as usize);
                assert_eq!(src.len(), 4, "config registers take dwords");
                self.tlb_cfg.entry(index).or_default()[word] =
                    u32::from_le_bytes(src.try_into().unwrap());
                return Ok(());
            }
            let (tile, addr) = self.translate(offset).expect("window is configured");
            for (i, b) in src.iter().enumerate() {
                self.mem.insert((tile, addr + i as u64), *b);
            }
            Ok(())
        }

        fn config_read32(&mut self, offset: ConfigOffset) -> Result<u32> {
            Ok(match offset {
                ConfigOffset::VendorDevice => 0xB140_1E52,
                _ => 0,
            })
        }

        fn tick(&mut self, _n: u32) {}

        fn is_simulated(&self) -> bool {
            true
        }
    }

    fn device() -> Device<FakeTransport> {
        Device::open(FakeTransport::default()).unwrap()
    }

    fn c(x: u8, y: u8) -> NocCoord<Noc0> {
        NocCoord::new(x, y).unwrap()
    }

    /// Data writes only; the three dwords of TLB configuration are filtered out.
    fn data_writes(d: &Device<FakeTransport>) -> Vec<(Bar, u64, usize)> {
        d.transport
            .writes
            .iter()
            .copied()
            .filter(|(bar, off, _)| !(*bar == Bar::Bar0 && *off >= tlb::CONFIG_BASE))
            .collect()
    }

    fn config_write_count(d: &Device<FakeTransport>) -> usize {
        d.transport
            .writes
            .iter()
            .filter(|(bar, off, _)| *bar == Bar::Bar0 && *off >= tlb::CONFIG_BASE)
            .count()
    }

    #[test]
    fn kernel_window_is_never_allocated() {
        let mut d = device();
        let mut seen = Vec::new();
        // Drain every 2 MiB window.
        for _ in 0..USABLE_2MIB_WINDOWS {
            seen.push(d.alloc_window(WindowKind::TwoMib).unwrap().index);
        }
        assert!(!seen.contains(&KERNEL_RESERVED_WINDOW));
        assert_eq!(seen.len(), 201);
        // And the pool is now empty for that geometry, but not for the other.
        assert!(d.alloc_window(WindowKind::TwoMib).is_err());
        assert!(d.alloc_window(WindowKind::FourGib).is_ok());
    }

    #[test]
    fn freed_windows_are_reused_and_lose_their_shadow() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write(&w, c(1, 2), 0, &[0xAA; 4]).unwrap();
        let before = config_write_count(&d);
        assert!(before > 0);

        let index = w.index();
        d.free_window(w);
        let w2 = d.alloc_window(WindowKind::TwoMib).unwrap();
        assert_eq!(
            w2.index(),
            index,
            "lowest free window should come back first"
        );

        // The new holder must not inherit the old shadow, or it would skip the
        // reconfiguration it needs.
        d.write(&w2, c(1, 2), 0, &[0xAA; 4]).unwrap();
        assert!(
            config_write_count(&d) > before,
            "reconfiguration must be repeated"
        );
    }

    #[test]
    fn transfers_split_at_window_boundaries() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let size = WindowKind::TwoMib.size();

        // Start four bytes before a window boundary, run eight bytes past it.
        d.write(&w, c(1, 2), size - 4, &[0x5A; 8]).unwrap();
        let writes = data_writes(&d);
        assert_eq!(
            writes.len(),
            2,
            "should have split into two accesses: {writes:?}"
        );
        assert_eq!(writes[0].2, 4, "first piece runs to the window boundary");
        assert_eq!(writes[1].2, 4, "second piece starts the next window");
        // The second piece lands at the start of the window aperture, because the
        // window was retargeted rather than the offset being advanced.
        assert_eq!(writes[1].1, tlb::window_bar_offset(w.index()).unwrap());
    }

    #[test]
    fn a_transfer_spanning_many_windows_is_fully_chunked() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let size = WindowKind::TwoMib.size() as usize;
        let data = vec![7u8; size * 2 + 16];
        d.write(&w, c(3, 4), (size - 8) as u64, &data).unwrap();

        let writes = data_writes(&d);
        assert_eq!(writes.iter().map(|w| w.2).sum::<usize>(), data.len());
        assert_eq!(writes.len(), 4, "8 + 2MiB + 2MiB + 8 = four accesses");
        assert!(writes.iter().all(|(_, _, len)| *len <= size));
    }

    #[test]
    fn aligned_transfers_are_not_split() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write(&w, c(1, 1), 0x1000, &[0u8; 4096]).unwrap();
        assert_eq!(data_writes(&d).len(), 1);
    }

    #[test]
    fn repeated_access_to_one_window_reconfigures_once() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        d.write32(&w, c(1, 2), 0x100, 1).unwrap();
        let after_first = config_write_count(&d);
        assert_eq!(after_first, 3, "three dwords of configuration");

        // Same tile, same window -- no reconfiguration.
        d.write32(&w, c(1, 2), 0x200, 2).unwrap();
        assert_eq!(config_write_count(&d), after_first);

        // Different tile -- reconfiguration.
        d.write32(&w, c(5, 6), 0x200, 3).unwrap();
        assert_eq!(config_write_count(&d), after_first + 3);
    }

    #[test]
    fn round_trip_through_two_different_windows() {
        let mut d = device();
        let a = d.alloc_window(WindowKind::TwoMib).unwrap();
        let b = d.alloc_window(WindowKind::TwoMib).unwrap();
        assert_ne!(a.index(), b.index());

        let payload: Vec<u8> = (0..255u8).collect();
        d.write(&a, c(2, 3), 0x4000, &payload).unwrap();
        let mut back = vec![0u8; payload.len()];
        d.read(&b, c(2, 3), 0x4000, &mut back).unwrap();
        assert_eq!(
            back, payload,
            "a second window must see the first window's writes"
        );
    }

    #[test]
    fn probe_reports_absent_tiles_as_none() {
        let mut d = device();
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        // The fake serves zeroes, which stands in for "nothing there".
        assert_eq!(d.probe_tile(&w, c(9, 9)).unwrap(), None);
    }
}
