//! The card moving host memory itself (`tt_isa::dm::op::HOST_READ`,
//! `HOST_WRITE`): a tile's mover reads host memory into its L1, and writes
//! its L1 into host memory, through the host-connected PCIe tile, at the NoC
//! address the transport gives (`Transport::host_memory`: memory pinned for
//! the card on silicon, a region the DMA callbacks serve on ttsim).
//!
//! The claims: every byte lands, at sizes of one request up to several and at
//! offsets into the buffer; and a move outside the PCIe tile's two windows to
//! the host, or misaligned, is refused before it reaches the mover.

use tt_device::tlb::WindowKind;
use tt_device::Transport;
use tt_isa::dm::{error, op};
use tt_kernels::dm::{DataMover, DmError};
use tt_tests::backend::GATE_TILE;
use tt_tests::harness::{in_device, tile};

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s as u8
        })
        .collect()
}

const L1_AT: u32 = 0x2_0000;

fn entry(op: u32, host: u64, l1: u32, len: u32) -> [u32; 8] {
    [op, host as u32, (host >> 32) as u32, 0, l1, len, 0, 0]
}

#[test]
fn the_mover_reads_and_writes_host_memory() {
    in_device(|d| {
        let w = d.alloc_window(WindowKind::TwoMib).unwrap();
        let dram = d.dram_grid(&w).unwrap();
        let t = tile(d, GATE_TILE.0, GATE_TILE.1);
        let mut host = d.transport().host_memory(1 << 20).unwrap();
        let base = host.noc_address();
        println!("host memory at NoC address {base:#x}");
        let mut m = DataMover::start(d, &w, t, &dram, tt_firmware_images::DM_B.1).unwrap();
        // One request, several (16 KiB each), and a short last one.
        for (k, (offset, len)) in [(0usize, 64usize), (4096, 4096), (64 << 10, 40 << 10 | 64)]
            .into_iter()
            .enumerate()
        {
            // Host -> L1.
            let data = pattern(len, k as u32 * 2 + 1);
            host.write(offset, &data);
            d.l1_write(&w, t, L1_AT as u64, &vec![0u8; len]).unwrap();
            let n = m
                .enqueue(
                    d,
                    &w,
                    &[entry(
                        op::HOST_READ,
                        base + offset as u64,
                        L1_AT,
                        len as u32,
                    )],
                )
                .unwrap();
            m.wait_for(d, &w, n).unwrap();
            let mut back = vec![0u8; len];
            d.l1_read(&w, t, L1_AT as u64, &mut back).unwrap();
            assert!(back == data, "{len} B at +{offset}: the read did not land");
            // L1 -> host.
            let data = pattern(len, k as u32 * 2 + 2);
            d.l1_write(&w, t, L1_AT as u64, &data).unwrap();
            host.write(offset, &vec![0u8; len]);
            let n = m
                .enqueue(
                    d,
                    &w,
                    &[entry(
                        op::HOST_WRITE,
                        base + offset as u64,
                        L1_AT,
                        len as u32,
                    )],
                )
                .unwrap();
            m.wait_for(d, &w, n).unwrap();
            let mut back = vec![0u8; len];
            host.read(offset, &mut back);
            assert!(back == data, "{len} B at +{offset}: the write did not land");
            println!("{len} B at +{offset}: read and write landed");
        }
        // Refused on the host: the PCIe controller's DBI window, and a host
        // address not congruent with L1 mod 64.
        for (bad, code) in [
            (
                entry(op::HOST_WRITE, 0xF800_0000_0000_0000, L1_AT, 64),
                error::OP,
            ),
            (entry(op::HOST_READ, base + 16, L1_AT, 64), error::ALIGNMENT),
        ] {
            let e = m.enqueue(d, &w, &[bad]).unwrap_err();
            assert!(
                matches!(e, DmError::Invalid(c) if c == code),
                "{bad:x?}: {e}"
            );
        }
        m.stop(d, &w).unwrap();
    });
}
