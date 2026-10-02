//! Phase 10 gate (F1): whole FP32 tiles through `Dst` and back.
//!
//! Phase 5 moved 128 datums; an SFPU tile kernel moves 32x32 tiles. A tile
//! image is a 16-byte header and 1024 datums in `tt_layout`'s face order, and
//! one flat 1024-datum `UnpackToDst` lays the four faces down sixty-four `Dst`
//! rows (`datapath::tile_descriptor`), which is exactly what the packer reads
//! back. The claims, every datum checked:
//!
//! - two tiles unpack to `Dst` rows 0 and 64 and pack back from either, in
//!   either order, bit for bit (the unpacker's destination and the packer's
//!   `Dst` offset retargeted between tiles);
//! - the SFPU reaches every datum of a tile exactly once by walking sixteen
//!   row groups and both column halves of each (`SFPLOAD`/`SFPSTORE` cover a
//!   four-row group's even or odd columns), which is how the SFPU program
//!   builder iterates a tile: a copy so walked, to rows 128..192, packs back
//!   as the tile it copied.

use tt_isa::backend::ConfigWords;
use tt_isa::backend::{self, Before};
use tt_isa::sfpu::{self, mod0_fmt, DST_ODD_COLUMNS};
use tt_isa::tile::L1Format;
use tt_isa::tile::TileImage;
use tt_kernels::datapath::{
    config_program, pack_tile_from_dst, state_id, thread_config, tile_descriptor,
    tile_unpack_config, unpack_tile_to_dst, OUT, STAGE, TILE_DATUMS,
};
use tt_tests::harness::{self, Roles, Run};

/// Where the second input tile is staged: one tile slot on, as in GDDR.
const STAGE_B: u64 = STAGE + tt_isa::dm::TILE_SLOT;

fn tile(seed: u32) -> Vec<u32> {
    let specials = [
        0.0f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        1.0,
    ];
    let mut s = seed | 1;
    (0..TILE_DATUMS as usize)
        .map(|i| {
            if i % 37 == 0 {
                return specials[(i / 37) % specials.len()].to_bits();
            }
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // A normal float of either sign, exponents across the range.
            (s & 0x8000_0000) | (((s >> 8) % 250 + 2) << 23) | (s & 0x7f_ffff)
        })
        .collect()
}

fn image(datums: &[u32]) -> Vec<u8> {
    let img = TileImage::new(tile_descriptor(), L1Format::Fp32).unwrap();
    let mut b = vec![0u8; img.total_bytes()];
    for (i, d) in datums.iter().enumerate() {
        let at = img.datum_bit_offset(i) / 8;
        b[at..at + 4].copy_from_slice(&d.to_le_bytes());
    }
    b
}

/// Copy `Dst` rows `from..from + 64` to `to..to + 64` through `LReg` 0, one
/// row group and column half at a time.
fn sfpu_copy(from: u32, to: u32) -> Vec<tt_isa::isa::Instruction> {
    let mut p = Vec::new();
    for g in 0..16 {
        for half in [0, DST_ODD_COLUMNS] {
            p.push(sfpu::load(0, mod0_fmt::FP32, 0, from + 4 * g + half).unwrap());
            p.push(sfpu::store(0, mod0_fmt::FP32, 0, to + 4 * g + half).unwrap());
        }
    }
    p
}

#[test]
fn whole_tiles_round_trip_through_dst_and_the_sfpu_walks_every_datum() {
    let (a, b) = (tile(1), tile(2));
    let (ia, ib) = (image(&a), image(&b));

    let mut unpack = thread_config();
    let mut words = ConfigWords::new();
    tile_unpack_config(&mut words, STAGE);
    unpack.extend(config_program(&words));
    unpack.extend(unpack_tile_to_dst(STAGE, 0));
    unpack.extend(unpack_tile_to_dst(STAGE_B, 64));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());

    let mut math = vec![state_id()];
    math.extend(sfpu_copy(0, 128));
    math.push(backend::wait_for_sfpu(Before::EVERYTHING).unwrap());

    // B first, from row 64: the offset is honoured, not just row 0's default.
    let mut pack = vec![state_id()];
    pack.extend(pack_tile_from_dst(OUT, 64));
    pack.extend(pack_tile_from_dst(OUT + 4096, 0));
    pack.extend(pack_tile_from_dst(OUT + 8192, 128));
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());

    let sentinel = vec![0xA5u8; 3 * 4096];
    harness::in_device(|dev| {
        let out = harness::run(
            dev,
            &Run::roles(Roles {
                unpack: &unpack,
                math: &math,
                pack: &pack,
            })
            .stage(&[(STAGE, &ia), (STAGE_B, &ib), (OUT, &sentinel)])
            .dump_rows(0)
            .read_back(&[(OUT, 3 * 4096)]),
        );
        let got: Vec<u32> = out.l1[0]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (k, (want, what)) in [
            (&b, "B from row 64"),
            (&a, "A from row 0"),
            (&a, "A, copied by the SFPU"),
        ]
        .into_iter()
        .enumerate()
        {
            let got = &got[k * 1024..(k + 1) * 1024];
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g,
                    w,
                    "{what}: datum {i} (face {}, row {}, col {}): {g:#010x} vs {w:#010x}",
                    i / 256,
                    i % 256 / 16,
                    i % 16
                );
            }
        }
    });
}
