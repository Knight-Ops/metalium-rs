//! Staging for the elementwise gates: two operands in, one result out.
//!
//! Split from the gates themselves so the assertions read as arithmetic rather than
//! as address arithmetic.

use tt_isa::isa::Instruction;
use tt_isa::tile::TileDescriptor;
use tt_tests::datapath::{dst_round_trip_roles, OUT, STAGE};
use tt_tests::harness::{self, Roles, Run};

/// Datums staged in L1.
///
/// One unpack fills `Dst` rows 0..8: rows 0..4 are operand A and rows 4..8 are
/// operand B, which is what lets the kernel address them as two row groups four
/// apart: 128 datums.
pub const DATUMS: u32 = 128;

/// `Dst` datums per aligned group of four rows -- the span one `SFPLOAD` pair
/// covers, and the operand size.
pub const GROUP_DATUMS: usize = 64;

/// Which staged datum a flat `Dst` position holds, if any.
///
/// The unpacker places datum `i` at `OutAddr = dst_base + i` and the row/column
/// split is `Row = OutAddr/16 - 4`, `Col = OutAddr & 15`
/// (`UNPACR_Regular.md:394-396`), so the flat position is just `i`, for every
/// position the operands cover. (It was once `4 + i` with four positions no datum
/// reached: `REG3_Base_address` pointed one unit early and the unpacker read the
/// tile header as datums. See `datapath::tile_base_units`.)
pub const fn datum_at_dst_flat(flat: usize) -> Option<usize> {
    if flat < DATUMS as usize {
        Some(flat)
    } else {
        None
    }
}

/// L1 words the packer writes: four `Dst` rows of sixteen.
pub const PACKED_DATUMS: usize = 64;

const L1_SENTINEL: u32 = 0xA5A5_5A5A;

/// The three role programs: configure and unpack both operands on thread 0,
/// run `kernel` on thread 1, pack out on thread 2.
///
/// One unpack moves the whole flat run: operand A lands in `Dst` rows 0..4 and
/// operand B immediately after it, which is what lets the kernel address them as
/// two row groups four apart.
pub fn kernel_roles(
    descriptor: TileDescriptor,
    datums: u32,
    kernel: &[Instruction],
) -> [Vec<Instruction>; 3] {
    dst_round_trip_roles(descriptor, datums, kernel, 0b1111)
}

/// Run `kernel` over `staged` and return the packed L1 words.
pub fn run_kernel(
    dev: &mut harness::Dev<'_>,
    descriptor: TileDescriptor,
    staged: &[u8],
    kernel: &[Instruction],
) -> Vec<u32> {
    let [unpack, math, pack] = kernel_roles(descriptor, DATUMS, kernel);
    let sentinel: Vec<u8> = L1_SENTINEL
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(PACKED_DATUMS * 4)
        .collect();
    let out = harness::run(
        dev,
        &Run::roles(Roles {
            unpack: &unpack,
            math: &math,
            pack: &pack,
        })
        .stage(&[(STAGE, staged), (OUT, &sentinel)])
        .dump_rows(8)
        .read_back(&[(OUT, PACKED_DATUMS * 4)]),
    );
    let packed: Vec<u32> = out.l1[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(
        packed.iter().any(|&w| w != L1_SENTINEL),
        "the packer wrote nothing"
    );
    packed
}
