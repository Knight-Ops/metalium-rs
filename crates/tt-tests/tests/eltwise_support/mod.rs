//! Staging for the elementwise gates: two operands in, one result out.
//!
//! Split from the gates themselves so the assertions read as arithmetic rather than
//! as address arithmetic.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::isa::Instruction;
use tt_isa::sfpu;
use tt_isa::tile::TileDescriptor;
use tt_tests::datapath::{
    pack_config, pack_instruction, set_adc_x_pack, set_adc_x_unpack, thread_config, unpack_config,
    unpack_instruction, OUT, SCRATCH_GPR, STAGE,
};
use tt_tests::harness::{self, Run};

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

/// The full program: configure, unpack both operands, run `kernel`, pack out.
pub fn kernel_program(
    descriptor: TileDescriptor,
    datums: u32,
    kernel: &[Instruction],
) -> Vec<Instruction> {
    let mut p = thread_config();
    let mut words = ConfigWords::new();
    unpack_config(&mut words, descriptor, STAGE);
    pack_config(&mut words, OUT);
    let mut buf = [sfpu::nop(); 160];
    let n = words.program(SCRATCH_GPR, &mut buf).unwrap();
    p.extend_from_slice(&buf[..n]);

    // One unpack moves the whole flat run: operand A lands in `Dst` rows 0..1 and
    // operand B immediately after it, which is what lets the kernel address them as
    // two row groups four apart.
    p.push(set_adc_x_unpack(0, datums - 1));
    p.push(unpack_instruction());
    // The kernel's `SFPLOAD`s read what the unpacker writes, so the wait holds the
    // SFPU, not only the unpackers (`backend::Before`).
    p.push(backend::wait_for_unpacker0(Before::SFPU).unwrap());

    p.extend_from_slice(kernel);
    // The packer must not read `Dst` before the SFPU has written it: C11, holding
    // the packer back.
    p.push(backend::wait_for_sfpu(Before::PACKER).unwrap());

    p.push(set_adc_x_pack(0, 15));
    p.push(pack_instruction(0b1111, true));
    p.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    p
}

/// Run `kernel` over `staged` and return the packed L1 words.
pub fn run_kernel(
    dev: &mut harness::Dev<'_>,
    descriptor: TileDescriptor,
    staged: &[u8],
    kernel: &[Instruction],
) -> Vec<u32> {
    let program = kernel_program(descriptor, DATUMS, kernel);
    let sentinel: Vec<u8> = L1_SENTINEL
        .to_le_bytes()
        .iter()
        .copied()
        .cycle()
        .take(PACKED_DATUMS * 4)
        .collect();
    let out = harness::run(
        dev,
        &Run::new(&program)
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
    let _ = descriptor;
    packed
}
