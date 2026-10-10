//! X7: the posted-write fence as API (`Device::write_fenced` family).
//!
//! `Device::write`, `l1_write` and `write32` are *posted*: they return when the
//! bytes have left the CPU, not when they have landed, and only a read through
//! the same path says they have (divergence row AA: an Ethernet transfer landed
//! before the host's sentinel did; 3 failures in 40 runs without a read-back, 0
//! in 60 with it). `write_fenced`, `write32_fenced`, `l1_write_fenced`,
//! `eth_write_fenced` and `FencedWrite` carry the rule.
//!
//! # What ttsim can and cannot show
//!
//! ttsim applies a BAR write synchronously (`pci_mem_wr_bytes` has landed when
//! it returns), so against the bare simulator *no* write is ever late and the
//! race cannot be exhibited: `ttsim_round_trips_*` below proves the fence moves
//! the right bytes through the real transport and costs exactly one read, nothing
//! more. The ordering claim is proven against [`Posted`], a transport that wraps
//! the simulator and models the documented rule and nothing else: every write is
//! held back, in order, until some read through the same transport passes it. A
//! second agent (`peek`) reads the simulator's memory *without* that read, as an
//! Ethernet transfer or a mover would. With it, `write` is demonstrably not
//! visible to the agent and the fenced writes are visible before they return.
//!
//! Gates, with the mutant each one catches:
//!
//! * `device_write_is_posted_in_the_model`: control. Without the fence the agent
//!   sees stale bytes, so the model is not vacuous.
//! * `fenced_writes_are_visible_to_a_second_agent_before_they_return`: skip the
//!   read-back (`FencedWrite::commit`, `l1_write_fenced`, `write_range_fenced`)
//!   and the agent sees the old bytes.
//! * `one_fence_orders_every_earlier_write_through_the_path`: the batch and the
//!   plain write issued *before* a fenced one are visible too.
//! * `ttsim_round_trips_*` and the `tt-device` unit tests (`device::fence::tests`)
//!   fix the read-back's address and width (a read of the wrong word fails them),
//!   alignment refusal, and that a failed write reads nothing back.
//!
//! Silicon (`silicon_gates`, `#[cfg(feature = "silicon")]`): a documented-class
//! test on a Tensix L1 range, no Ethernet, no register.

#[cfg(not(feature = "silicon"))]
mod model {

    use tt_device::tlb::{self, WindowKind};
    use tt_device::{Bar, ConfigOffset, Device, FencedWrite, Result, Transport, Window};
    use tt_isa::noc::{Noc0, NocCoord};
    use tt_ttsim::{fork_scope, Simulator};

    const ADDR: u64 = 0x2_0000;

    fn tile() -> NocCoord<Noc0> {
        NocCoord::new(3, 4).unwrap()
    }

    enum Held {
        Plain,
        Bulk,
    }

    /// The simulator behind the documented posted-write rule: a write is held, in
    /// order, until a read through this transport comes after it.
    struct Posted<T: Transport> {
        inner: T,
        held: Vec<(Held, Bar, u64, Vec<u8>)>,
        /// Reads issued (each one passes everything held before it).
        reads: usize,
    }

    impl<T: Transport> Posted<T> {
        fn new(inner: T) -> Self {
            Self {
                inner,
                held: Vec::new(),
                reads: 0,
            }
        }

        fn pass_held_writes(&mut self) -> Result<()> {
            for (kind, bar, offset, data) in std::mem::take(&mut self.held) {
                match kind {
                    Held::Plain => self.inner.bar_write(bar, offset, &data)?,
                    Held::Bulk => self.inner.bar_write_bulk(bar, offset, &data)?,
                }
            }
            Ok(())
        }

        /// The second agent: a read of the simulator's memory that does not pass the
        /// host's held writes.
        fn peek(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) {
            self.inner.bar_read(bar, offset, dst).unwrap();
        }

        fn held(&self) -> usize {
            self.held.len()
        }
    }

    impl<T: Transport> Transport for Posted<T> {
        fn bar_read(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) -> Result<()> {
            self.reads += 1;
            self.pass_held_writes()?;
            self.inner.bar_read(bar, offset, dst)
        }
        fn bar_write(&mut self, bar: Bar, offset: u64, src: &[u8]) -> Result<()> {
            self.held.push((Held::Plain, bar, offset, src.to_vec()));
            Ok(())
        }
        fn bar_read_bulk(&mut self, bar: Bar, offset: u64, dst: &mut [u8]) -> Result<()> {
            self.reads += 1;
            self.pass_held_writes()?;
            self.inner.bar_read_bulk(bar, offset, dst)
        }
        fn bar_write_bulk(&mut self, bar: Bar, offset: u64, src: &[u8]) -> Result<()> {
            self.held.push((Held::Bulk, bar, offset, src.to_vec()));
            Ok(())
        }
        fn config_read32(&mut self, offset: ConfigOffset) -> Result<u32> {
            self.inner.config_read32(offset)
        }
        fn tick(&mut self, n: u32) {
            self.inner.tick(n)
        }
        fn is_simulated(&self) -> bool {
            self.inner.is_simulated()
        }
        fn chip(&self) -> tt_isa::noc::ChipId {
            self.inner.chip()
        }
    }

    /// Run `f` against a posted-write model over a fresh simulator.
    fn in_posted_model(f: impl FnOnce(&mut Device<Posted<tt_ttsim::LibTtsim<'_>>>)) {
        if let Err(e) = fork_scope(|| {
            let mut sim =
                Simulator::open().unwrap_or_else(|e| panic!("could not open simulator: {e}"));
            let mut dev =
                Device::open(Posted::new(sim.transport())).unwrap_or_else(|e| panic!("{e}"));
            f(&mut dev);
        }) {
            panic!("{e}");
        }
    }

    /// What the second agent sees at `ADDR + at`, `len` bytes, through `agent` (a
    /// window the host has configured onto `tile()` and then left alone).
    fn agent_sees<T: Transport>(
        dev: &mut Device<Posted<T>>,
        agent: &Window,
        at: u64,
        len: usize,
    ) -> Vec<u8> {
        let offset = tlb::window_bar_offset(agent.index()).unwrap()
            + (ADDR + at) % WindowKind::TwoMib.size();
        let mut out = vec![0u8; len];
        dev.transport().peek(agent.kind().bar(), offset, &mut out);
        out
    }

    /// A host window and the agent's, both on `tile()`; the agent's is configured
    /// (by one read, which also passes every held write) and not touched again.
    fn two_windows<T: Transport>(dev: &mut Device<Posted<T>>) -> (Window, Window) {
        let host = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let agent = dev.alloc_window(WindowKind::TwoMib).unwrap();
        dev.read32(&agent, tile(), ADDR).unwrap();
        (host, agent)
    }

    fn bytes(seed: u8, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
            .collect()
    }

    #[test]
    fn device_write_is_posted_in_the_model() {
        in_posted_model(|dev| {
            let (host, agent) = two_windows(dev);
            let data = bytes(1, 64);
            dev.write(&host, tile(), ADDR, &data).unwrap();
            dev.l1_write(&host, tile(), ADDR + 64, &data).unwrap();
            dev.write32(&host, tile(), ADDR + 128, 0xA5A5_A5A5).unwrap();
            assert!(dev.transport().held() >= 3);
            // The control: the agent has not been given any of it.
            assert_eq!(agent_sees(dev, &agent, 0, 64), vec![0; 64]);
            assert_eq!(agent_sees(dev, &agent, 64, 64), vec![0; 64]);
            assert_eq!(agent_sees(dev, &agent, 128, 4), vec![0; 4]);
            // Any read through the path passes them (the rule the fence relies on).
            dev.read32(&host, tile(), ADDR).unwrap();
            assert_eq!(dev.transport().held(), 0);
            assert_eq!(agent_sees(dev, &agent, 0, 64), data);
        });
    }

    #[test]
    fn fenced_writes_are_visible_to_a_second_agent_before_they_return() {
        in_posted_model(|dev| {
            let (host, agent) = two_windows(dev);

            let a = bytes(2, 256);
            dev.write_fenced(&host, tile(), ADDR, &a).unwrap();
            assert_eq!(
                dev.transport().held(),
                0,
                "write_fenced returned with a write held"
            );
            assert_eq!(agent_sees(dev, &agent, 0, 256), a, "write_fenced");

            dev.write32_fenced(&host, tile(), ADDR + 0x100, 0x1234_5678)
                .unwrap();
            assert_eq!(
                agent_sees(dev, &agent, 0x100, 4),
                0x1234_5678u32.to_le_bytes()
            );

            // Ragged byte range through the bulk path.
            let b = bytes(3, 37);
            dev.l1_write_fenced(&host, tile(), ADDR + 0x203, &b)
                .unwrap();
            assert_eq!(
                dev.transport().held(),
                0,
                "l1_write_fenced returned with a write held"
            );
            assert_eq!(agent_sees(dev, &agent, 0x203, 37), b, "l1_write_fenced");

            // A batch: every write is visible, in order (the later one wins).
            FencedWrite::new(&host, tile())
                .word(ADDR + 0x300, 1)
                .word(ADDR + 0x304, 2)
                .word(ADDR + 0x300, 3)
                .commit(dev)
                .unwrap();
            assert_eq!(dev.transport().held(), 0);
            assert_eq!(agent_sees(dev, &agent, 0x300, 8), [3, 0, 0, 0, 2, 0, 0, 0]);
        });
    }

    #[test]
    fn one_fence_orders_every_earlier_write_through_the_path() {
        in_posted_model(|dev| {
            let (host, agent) = two_windows(dev);
            // The receiver's sentinel and a payload, both plain (as the hand-fenced
            // link test did), then one fenced write: all of them must have landed.
            let sentinel = vec![0xEE; 128];
            dev.write(&host, tile(), ADDR, &sentinel).unwrap();
            dev.write32(&host, tile(), ADDR + 0x80, 7).unwrap();
            assert_eq!(
                agent_sees(dev, &agent, 0, 128),
                vec![0; 128],
                "still posted"
            );
            dev.write32_fenced(&host, tile(), ADDR + 0x84, 9).unwrap();
            assert_eq!(agent_sees(dev, &agent, 0, 128), sentinel);
            assert_eq!(agent_sees(dev, &agent, 0x80, 8), [7, 0, 0, 0, 9, 0, 0, 0]);
        });
    }

    #[test]
    fn a_refused_fenced_write_leaves_the_agent_and_the_path_untouched() {
        in_posted_model(|dev| {
            let (host, agent) = two_windows(dev);
            let reads = dev.transport().reads;
            assert!(dev.write_fenced(&host, tile(), ADDR + 2, &[1; 4]).is_err());
            assert!(dev.write_fenced(&host, tile(), ADDR, &[1; 6]).is_err());
            assert!(dev
                .l1_write_fenced(&host, NocCoord::<Noc0>::new(8, 0).unwrap(), 0x1000, &[1; 4])
                .is_err());
            assert_eq!(dev.transport().held(), 0, "nothing was written");
            assert_eq!(dev.transport().reads, reads, "and nothing was read back");
            assert_eq!(agent_sees(dev, &agent, 0, 8), vec![0; 8]);
        });
    }

    #[test]
    fn ttsim_round_trips_every_fenced_width_with_exactly_one_read() {
        use tt_tests::harness::in_device;
        in_device(|dev| {
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            let r = dev.alloc_window(WindowKind::TwoMib).unwrap();
            let t = tile();
            // Settle both windows' targets so the traffic below is only the access.
            dev.read32(&w, t, ADDR).unwrap();
            dev.read32(&r, t, ADDR).unwrap();

            let before = dev.traffic();
            dev.write32_fenced(&w, t, ADDR, 0xCAFE_F00D).unwrap();
            let d = dev.traffic() - before;
            assert_eq!(
                (d.write_calls, d.bytes_written, d.read_calls, d.bytes_read),
                (1, 4, 1, 4)
            );
            assert_eq!(dev.read32(&r, t, ADDR).unwrap(), 0xCAFE_F00D);

            for len in [4usize, 16, 64, 1000, 4096] {
                let data = bytes(len as u8, len);
                let before = dev.traffic();
                dev.write_fenced(&w, t, ADDR + 0x1000, &data).unwrap();
                let d = dev.traffic() - before;
                assert_eq!(
                    (d.read_calls, d.bytes_read, d.bytes_written),
                    (1, 4, len as u64),
                    "len {len}"
                );
                let mut back = vec![0u8; len];
                dev.read(&r, t, ADDR + 0x1000, &mut back).unwrap();
                assert_eq!(back, data, "len {len}");
            }

            for (at, len) in [(1u64, 1usize), (3, 2), (5, 7), (0, 33), (2, 4093)] {
                let data = bytes(len as u8, len);
                let before = dev.traffic();
                dev.l1_write_fenced(&w, t, ADDR + 0x3000 + at, &data)
                    .unwrap();
                let d = dev.traffic() - before;
                assert_eq!(
                    (d.read_calls, d.bytes_read, d.bytes_written),
                    (1, 4, len as u64),
                    "at {at} len {len}"
                );
                let mut back = vec![0u8; len];
                dev.l1_read(&r, t, ADDR + 0x3000 + at, &mut back).unwrap();
                assert_eq!(back, data, "at {at} len {len}");
            }

            // Refusals never reach the simulator (which would exit on a bad access).
            let before = dev.traffic();
            assert!(dev.write32_fenced(&w, t, ADDR + 2, 1).is_err());
            assert!(dev.write_fenced(&w, t, ADDR, &[0; 5]).is_err());
            assert!(dev.write_fenced(&w, t, 0xFFB1_4000, &[0; 4]).is_err());
            assert!(dev
                .l1_write_fenced(&w, NocCoord::<Noc0>::new(8, 0).unwrap(), ADDR, &[0; 4])
                .is_err());
            assert_eq!(dev.traffic(), before);
        });
    }
}

// ---------------------------------------------------------------------------
// Silicon: documented class (posted L1 writes and reads through a Tensix
// tile's TLB window; no register, no Ethernet, no second agent). The race
// itself needs another agent and is covered by `silicon_eth_link` /
// `step13_ethernet`, which now use `eth_write_fenced`.
// ---------------------------------------------------------------------------

#[cfg(feature = "silicon")]
mod silicon_gates {
    use tt_device::tlb::WindowKind;
    use tt_tests::harness::{assert_on_silicon, in_device, tile};

    const ADDR: u64 = 0x2_0000;

    /// A posted L1 write, then fenced writes of every width, read back through
    /// a second window: every byte as written, and a refusal reaches no card.
    /// Mutants that fail it: a fence that writes the wrong address or width.
    #[test]
    fn posted_then_fenced_l1_writes_read_back_exactly() {
        assert_on_silicon();
        in_device(|dev| {
            let t = tile(dev, 3, 4);
            let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
            let r = dev.alloc_window(WindowKind::TwoMib).unwrap();
            let pat = |seed: u8, n: usize| -> Vec<u8> {
                (0..n)
                    .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
                    .collect()
            };
            let posted = pat(1, 4096);
            dev.l1_write(&w, t, ADDR, &posted).unwrap();
            let fenced = pat(2, 4096);
            dev.l1_write_fenced(&w, t, ADDR + 0x2000, &fenced).unwrap();
            let word = 0x0BAD_CAFEu32;
            dev.write32_fenced(&w, t, ADDR + 0x4000, word).unwrap();
            let wide = pat(3, 1024);
            dev.write_fenced(&w, t, ADDR + 0x5000, &wide).unwrap();
            let ragged = pat(4, 37);
            dev.l1_write_fenced(&w, t, ADDR + 0x6003, &ragged).unwrap();

            let mut back = vec![0u8; 4096];
            dev.l1_read(&r, t, ADDR, &mut back).unwrap();
            assert_eq!(back, posted, "the posted write");
            dev.l1_read(&r, t, ADDR + 0x2000, &mut back).unwrap();
            assert_eq!(back, fenced, "l1_write_fenced");
            assert_eq!(dev.read32(&r, t, ADDR + 0x4000).unwrap(), word);
            let mut back = vec![0u8; 1024];
            dev.read(&r, t, ADDR + 0x5000, &mut back).unwrap();
            assert_eq!(back, wide, "write_fenced");
            let mut back = vec![0u8; 37];
            dev.l1_read(&r, t, ADDR + 0x6003, &mut back).unwrap();
            assert_eq!(back, ragged, "ragged l1_write_fenced");

            let before = dev.traffic();
            assert!(dev.write32_fenced(&w, t, ADDR + 2, 1).is_err());
            assert_eq!(dev.traffic(), before);
        });
    }
}
