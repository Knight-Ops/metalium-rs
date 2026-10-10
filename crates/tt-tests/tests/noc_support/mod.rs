//! Shared by `step115_noc_multicast`, `step116_noc_atomics`,
//! `step117_niu_completion` and `step118_mover_fast_path`: stage requests for the NoC probe image
//! (`tt-firmware/src/bin/noc_probe_b.rs`), run it on RISCV B of one tile, read
//! back what it recorded.
//!
//! Silicon note: this runs the same image on silicon. Nothing here names a tile
//! that was not obtained through `harness::tile` / `harness::tensix_grid`.

#![allow(dead_code)]

use tt_device::tlb::WindowKind;
use tt_isa::mailbox::{self, status as mbox};
use tt_isa::noc::probe::{self, result};
use tt_isa::noc::{Noc0, NocCoord};
use tt_tests::harness::Dev;

/// One request for the probe: the register values (ten, or eleven for a
/// multicast), from the typed encoders.
#[derive(Clone, Debug)]
pub struct Req {
    pub kind: u32,
    pub flags: u32,
    pub txn: u32,
    pub expected_acks: u32,
    pub regs: Vec<u32>,
    /// With `probe::flag::PARTIAL`: the registers the image writes
    /// (`probe::word::MASK`).
    pub mask: u32,
}

/// What the probe recorded for one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Res {
    pub status: u32,
    pub ack_before: u32,
    pub ack_after: u32,
    pub outstanding_before: u32,
    pub outstanding_after: u32,
    pub rtz_before: u32,
    pub rtz_after: u32,
    pub polls: u32,
}

/// Simulated cycles one probe run may take before it is called hung. On
/// silicon the device layer converts it with a one second floor.
pub const BUDGET: u64 = 6_000_000;
/// Polls of a completion counter before a request is reported timed out.
pub const SPIN_BUDGET: u32 = 20_000;

fn values(regs: &[(u64, u32)]) -> Vec<u32> {
    // The encoders list registers in `probe::REGISTERS` order; the image writes
    // them by position, so check it.
    for (i, (offset, _)) in regs.iter().enumerate() {
        assert_eq!(*offset, probe::REGISTERS[i], "register {i} out of order");
    }
    regs.iter().map(|(_, v)| *v).collect()
}

pub fn unicast(regs: &[(u64, u32)], txn: u32, flags: u32) -> Req {
    assert_eq!(regs.len(), 10);
    Req {
        kind: probe::kind::UNICAST,
        flags,
        txn,
        expected_acks: 0,
        regs: values(regs),
        mask: u32::MAX,
    }
}

/// A unicast that writes only the registers `mask` selects (bit `i` is
/// `probe::REGISTERS[i]`) and leaves the rest as the previous request left them,
/// with `PARTIAL` set so the image snapshots the initiator afterwards. `regs`
/// still holds all ten values: the unselected ones are not written.
pub fn partial(regs: &[(u64, u32)], mask: u32, txn: u32, flags: u32) -> Req {
    Req {
        mask,
        ..unicast(regs, txn, flags | probe::flag::PARTIAL)
    }
}

pub fn multicast(regs: &[(u64, u32)], txn: u32, flags: u32, acks: u32) -> Req {
    assert_eq!(regs.len(), 11);
    Req {
        kind: probe::kind::MULTICAST,
        flags,
        txn,
        expected_acks: acks,
        regs: values(regs),
        mask: u32::MAX,
    }
}

/// Stage `reqs` in `tile`'s L1 and run the probe image on its RISCV B.
///
/// On a hang the error names the last breadcrumb, and B is held in reset before
/// returning.
pub fn run_probe(
    dev: &mut Dev<'_>,
    tile: NocCoord<Noc0>,
    reqs: &[Req],
    stage: &[(u64, &[u8])],
) -> Result<Vec<Res>, String> {
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    for (at, bytes) in stage {
        dev.write(&w, tile, *at, bytes).unwrap();
    }
    dev.write32(&w, tile, probe::STAGE, 0).unwrap();
    dev.write32(&w, tile, probe::SCRIPT, reqs.len() as u32)
        .unwrap();
    dev.write32(&w, tile, probe::SCRIPT + 4, SPIN_BUDGET)
        .unwrap();
    for (k, r) in reqs.iter().enumerate() {
        let at = probe::REQUESTS + k as u64 * probe::REQUEST_STRIDE;
        let mut words = vec![r.kind, r.flags, r.txn, r.expected_acks];
        words.extend(&r.regs);
        words.resize((probe::word::MASK / 4) as usize, 0);
        words.push(r.mask);
        let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
        dev.write(&w, tile, at, &bytes).unwrap();
        // Poison the record so an unwritten result cannot pass for a zero.
        let poison = [0xA5u8; probe::RESULT_STRIDE as usize];
        dev.write(
            &w,
            tile,
            probe::RESULTS + k as u64 * probe::RESULT_STRIDE,
            &poison,
        )
        .unwrap();
    }
    dev.write32(&w, tile, mailbox::STATUS, 0).unwrap();
    let (core, image, at) = tt_firmware_images::NOC_PROBE_B;
    dev.load_and_start(&w, tile, core, image, at).unwrap();
    let done = dev
        .wait_for_status(&w, tile, BUDGET, |x| x == mbox::DONE)
        .unwrap();
    if let Err(e) = done {
        let stage = dev.read32(&w, tile, probe::STAGE).unwrap_or(0xDEAD);
        let _ = dev.set_core_reset(&w, tile, core, true);
        return Err(format!(
            "probe did not finish: {e}; last stage request {} phase {} ({})",
            stage >> 8,
            stage & 0xFF,
            probe::phase::name(stage & 0xFF)
        ));
    }
    let mut out = vec![];
    for k in 0..reqs.len() as u64 {
        let base = probe::RESULTS + k * probe::RESULT_STRIDE;
        let mut word = |o| dev.read32(&w, tile, base + o).unwrap();
        out.push(Res {
            status: word(result::STATUS),
            ack_before: word(result::ACK_BEFORE),
            ack_after: word(result::ACK_AFTER),
            outstanding_before: word(result::OUTSTANDING_BEFORE),
            outstanding_after: word(result::OUTSTANDING_AFTER),
            rtz_before: word(result::RTZ_BEFORE),
            rtz_after: word(result::RTZ_AFTER),
            polls: word(result::POLLS),
        });
    }
    dev.set_core_reset(&w, tile, core, true).unwrap();
    Ok(out)
}

pub fn read_bytes(dev: &mut Dev<'_>, tile: NocCoord<Noc0>, at: u64, len: usize) -> Vec<u8> {
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    let mut out = vec![0u8; len];
    dev.read(&w, tile, at, &mut out).unwrap();
    out
}

pub fn write_bytes(dev: &mut Dev<'_>, tile: NocCoord<Noc0>, at: u64, bytes: &[u8]) {
    let w = dev.alloc_window(WindowKind::TwoMib).unwrap();
    dev.write(&w, tile, at, bytes).unwrap();
}

/// The initiator's ten words as the probe read them back after request `k`
/// (`probe::flag::PARTIAL` requests only), in `probe::REGISTERS` order.
pub fn snapshot(dev: &mut Dev<'_>, tile: NocCoord<Noc0>, k: usize) -> [u32; 10] {
    let bytes = read_bytes(
        dev,
        tile,
        probe::SNAPSHOTS + k as u64 * probe::SNAPSHOT_STRIDE,
        40,
    );
    let mut out = [0u32; 10];
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *o = u32::from_le_bytes(c.try_into().unwrap());
    }
    out
}
