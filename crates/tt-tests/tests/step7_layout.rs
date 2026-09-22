//! Step 7 gate: a tiled tensor survives the trip into a Tensix tile's L1 and back.
//!
//! # What this adds over the host-side gates
//!
//! `tt-layout`'s own tests prove the byte image is right. They run on a `Vec<u8>`
//! and say nothing about the transport. This gate puts the same image through the
//! TLB window path -- window retargeting, boundary splitting, real L1 -- because
//! that path has its own failure modes and none of them are visible host-side.
//!
//! It is therefore deliberately a *thin* gate with a small number of shapes. The
//! shape sweep belongs in `tt-layout`, where it costs nothing; here each case forks
//! a process and boots a simulator.

use tt_device::{tlb::WindowKind, Device};
use tt_isa::noc::{grid, Noc0, NocCoord};
use tt_isa::tile::L1Format;
use tt_layout::{detilize, tilize, HostDtype, Layout, TensorView, TensorViewMut};
use tt_ttsim::{fork_scope, Simulator};

type Dev<'a> = Device<tt_ttsim::LibTtsim<'a>>;

#[track_caller]
fn in_device(f: impl FnOnce(&mut Dev<'_>)) {
    let result = fork_scope(|| {
        let mut sim = Simulator::open().unwrap_or_else(|e| panic!("could not open simulator: {e}"));
        let mut dev = Device::open(sim.transport()).unwrap_or_else(|e| panic!("{e}"));
        f(&mut dev);
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}

fn tensix(x: u8, y: u8) -> NocCoord<Noc0> {
    assert!(
        grid::is_tensix_geometry(x, y),
        "({x},{y}) is not a Tensix tile"
    );
    NocCoord::new(x, y).unwrap()
}

/// Where the tiled buffer is staged.
///
/// Inside L1 (1536 KiB), clear of the firmware load address (`0x6000`) and of the
/// mailbox regions at `0x10_0000` and above, and 16-byte aligned as a tile base must
/// be (`UNPACR_Regular.md:113`).
const STAGE: u64 = 0x2_0000;

/// Values unique per element and exactly representable in every format in scope, so
/// a mismatch localises to an element rather than to rounding.
fn source(dtype: HostDtype, count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count * dtype.bytes());
    for i in 0..count {
        let v = (i % 2048) as i32 - 1024;
        let bits = (v as f32).to_bits();
        match dtype {
            HostDtype::F32 => out.extend_from_slice(&bits.to_le_bytes()),
            HostDtype::Bf16 => {
                out.extend_from_slice(&tt_isa::tile::fp32_to_bf16_round(bits).to_le_bytes())
            }
            HostDtype::F16 => {
                out.extend_from_slice(&tt_isa::tile::fp32_to_fp16(bits).unwrap().to_le_bytes())
            }
        }
    }
    out
}

/// Stage a tiled tensor in L1 and read it back, through two *different* windows.
///
/// Two windows for the reason `step2_tlb.rs` gives: a single window would pass even
/// if window configuration were ignored entirely.
fn stage_and_recover(dev: &mut Dev<'_>, dtype: HostDtype, shape: [usize; 3]) -> (Vec<u8>, Vec<u8>) {
    let write_window = dev.alloc_window(WindowKind::TwoMib).unwrap();
    let read_window = dev.alloc_window(WindowKind::TwoMib).unwrap();
    let tile = tensix(3, 4);

    let layout = Layout::tt_metal_32x32(dtype.identical_l1_format(), dtype, shape).unwrap();
    let src_bytes = source(dtype, shape[0] * shape[1] * shape[2]);
    let src = TensorView::contiguous(&src_bytes, dtype, shape);
    let tiled = tilize(&src, &layout).unwrap();

    dev.write(&write_window, tile, STAGE, &tiled).unwrap();
    let mut read_back = vec![0u8; tiled.len()];
    dev.read(&read_window, tile, STAGE, &mut read_back).unwrap();
    assert_eq!(read_back, tiled, "the tiled image did not survive L1");

    let mut out_bytes = vec![0u8; src_bytes.len()];
    let mut dst = TensorViewMut::contiguous(&mut out_bytes, dtype, shape);
    detilize(&read_back, &layout, &mut dst).unwrap();

    dev.free_window(write_window);
    dev.free_window(read_window);
    (src_bytes, out_bytes)
}

#[test]
fn a_tiled_tensor_round_trips_through_l1() {
    in_device(|dev| {
        for dtype in [HostDtype::F32, HostDtype::Bf16, HostDtype::F16] {
            for shape in [[1usize, 32, 32], [1, 13, 47], [2, 33, 65]] {
                let (src, out) = stage_and_recover(dev, dtype, shape);
                assert_eq!(
                    out, src,
                    "{dtype:?} {shape:?} did not survive the round trip"
                );
            }
        }
    });
}

/// The largest transfer this path can actually carry, at the top of L1.
///
/// There is deliberately no window-straddling case: a Tensix tile's 1536 KiB of L1
/// is *smaller than one 2 MiB window*, so no L1 transfer can straddle a window
/// boundary at all -- `step2_tlb.rs` records the same fact and reaches the splitting
/// logic by other means. What is reachable, and what this covers, is a large
/// multi-chunk transfer ending exactly at the top of the aperture, where an
/// off-by-one in the chunk planner would run off the end.
#[test]
fn a_large_tile_grid_at_the_top_of_l1_survives() {
    in_device(|dev| {
        let write_window = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let read_window = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix(5, 6);

        let dtype = HostDtype::Bf16;
        let shape = [1usize, 128, 128];
        let layout = Layout::tt_metal_32x32(L1Format::Bf16, dtype, shape).unwrap();
        let src_bytes = source(dtype, 128 * 128);
        let src = TensorView::contiguous(&src_bytes, dtype, shape);
        let tiled = tilize(&src, &layout).unwrap();

        assert_eq!(layout.grid_tiles(), 16, "four tiles each way");
        let base = grid::TENSIX_L1_SIZE - tiled.len() as u64;
        assert_eq!(base % 16, 0, "a tile base must be 16-byte aligned");

        dev.write(&write_window, tile, base, &tiled).unwrap();
        let mut back = vec![0u8; tiled.len()];
        dev.read(&read_window, tile, base, &mut back).unwrap();
        assert_eq!(back, tiled, "the image did not survive the top of L1");

        let mut out_bytes = vec![0u8; src_bytes.len()];
        let mut dst = TensorViewMut::contiguous(&mut out_bytes, dtype, shape);
        detilize(&back, &layout, &mut dst).unwrap();
        assert_eq!(out_bytes, src_bytes);
    });
}

/// The gate rejects a corrupted image.
///
/// Without this, every assertion above would also pass against a device that
/// returned the buffer the host already had. One datum is flipped in L1 and the
/// recovered tensor must differ.
#[test]
fn a_single_corrupted_datum_is_detected() {
    in_device(|dev| {
        let window = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix(3, 4);

        let dtype = HostDtype::F32;
        let shape = [1usize, 32, 32];
        let layout = Layout::tt_metal_32x32(L1Format::Fp32, dtype, shape).unwrap();
        let src_bytes = source(dtype, 32 * 32);
        let src = TensorView::contiguous(&src_bytes, dtype, shape);
        let tiled = tilize(&src, &layout).unwrap();

        dev.write(&window, tile, STAGE, &tiled).unwrap();

        // Flip one datum in the middle of the third face, in L1 rather than in the
        // host copy, so the corruption is on the device side of the transport.
        let image = layout.image();
        let victim = image.datum_bit_offset(image.datum_index(0, 2, 7, 9)) as u64 / 8;
        dev.write32(&window, tile, STAGE + victim, 0x1234_5678)
            .unwrap();

        let mut back = vec![0u8; tiled.len()];
        dev.read(&window, tile, STAGE, &mut back).unwrap();
        assert_ne!(back, tiled, "the corrupting write did not reach L1");

        let mut out_bytes = vec![0u8; src_bytes.len()];
        let mut dst = TensorViewMut::contiguous(&mut out_bytes, dtype, shape);
        detilize(&back, &layout, &mut dst).unwrap();
        assert_ne!(
            out_bytes, src_bytes,
            "a corrupted datum produced the original tensor, so the gate proves nothing"
        );

        // And it is the element the face convention says it should be.
        let (row, col) = layout.logical_coord(0, 0, 0, 2, 7, 9);
        let at = (row * 32 + col) * 4;
        assert_ne!(
            out_bytes[at..at + 4],
            src_bytes[at..at + 4],
            "the corruption landed somewhere other than logical ({row}, {col})"
        );
        let mut others_intact = true;
        for i in 0..32 * 32 {
            if i * 4 == at {
                continue;
            }
            if out_bytes[i * 4..i * 4 + 4] != src_bytes[i * 4..i * 4 + 4] {
                others_intact = false;
            }
        }
        assert!(
            others_intact,
            "one corrupted datum changed more than one element"
        );
    });
}

/// **Gate (silicon).** The same round trips on a real p150.
///
/// Compiled always so it cannot rot, run only with `--features silicon`.
#[cfg(feature = "silicon")]
#[test]
fn a_tiled_tensor_round_trips_on_silicon() {
    in_device(|dev| {
        for dtype in [HostDtype::F32, HostDtype::Bf16, HostDtype::F16] {
            for shape in [[1usize, 32, 32], [1, 13, 47], [1, 1, 1024], [2, 33, 65]] {
                let (src, out) = stage_and_recover(dev, dtype, shape);
                assert_eq!(out, src, "{dtype:?} {shape:?} did not survive on silicon");
            }
        }
    });
}

/// **Gate (silicon).** Misaligned tile bases, which the documentation says are fine.
///
/// # Why this is a real question
///
/// `BlackholeA0/NoC/Alignment.md` does not exist, and violations of the alignment
/// rules are `UndefinedBehavior`. The Wormhole page's table says that for data
/// travelling **from the host via PCIe to an L1 address** the requirement is
/// "Any: No restrictions (at least for x86 / x86-64 hosts)"
/// (`WormholeB0/NoC/Alignment.md:19,23`) -- so the host staging path this crate uses
/// should tolerate any base address at all, including odd ones.
///
/// That is a Wormhole-sourced fact about a page Blackhole does not carry, which this
/// project treats as a hypothesis until silicon says otherwise. The simulator cannot
/// settle it: ttsim models the transport, not the NIU's alignment checks. So this
/// asserts the documented "Any", and a failure here is a finding worth reporting
/// rather than a bug in this crate.
///
/// Note what this does **not** cover: the C16 congruence required when an L1 address
/// is the *source* -- that is the unpacker and packer driving the NoC themselves,
/// which arrives in Phase 6.
#[cfg(feature = "silicon")]
#[test]
fn the_host_path_tolerates_any_tile_base_alignment() {
    in_device(|dev| {
        let window = dev.alloc_window(WindowKind::TwoMib).unwrap();
        let tile = tensix(3, 4);

        let dtype = HostDtype::F32;
        let shape = [1usize, 32, 32];
        let layout = Layout::tt_metal_32x32(L1Format::Fp32, dtype, shape).unwrap();
        let src_bytes = source(dtype, 32 * 32);
        let src = TensorView::contiguous(&src_bytes, dtype, shape);
        let tiled = tilize(&src, &layout).unwrap();

        // 16 is the alignment a tile base is *architecturally* expected to have; the
        // others are deliberate violations of it that the host path should still
        // carry, because the restriction belongs to the unpacker, not to PCIe.
        for offset in [0u64, 1, 2, 3, 4, 8, 15, 16, 17, 31, 33] {
            let base = STAGE + offset;
            dev.write(&window, tile, base, &tiled).unwrap();
            let mut back = vec![0u8; tiled.len()];
            dev.read(&window, tile, base, &mut back).unwrap();
            assert_eq!(
                back, tiled,
                "a tile image staged at base {base:#x} (offset {offset} from a \
                 16-byte boundary) did not survive; Alignment.md says the host to L1 \
                 path has no restrictions"
            );

            let mut out_bytes = vec![0u8; src_bytes.len()];
            let mut dst = TensorViewMut::contiguous(&mut out_bytes, dtype, shape);
            detilize(&back, &layout, &mut dst).unwrap();
            assert_eq!(out_bytes, src_bytes, "offset {offset}");
        }
    });
}
