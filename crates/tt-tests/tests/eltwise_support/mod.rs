//! Staging for the elementwise gates: two operands in, one result out.
//!
//! Split from the gates themselves so the assertions read as arithmetic rather than
//! as address arithmetic.

use tt_isa::backend::{self, ConfigWords};
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
/// apart. That is 128 datums, plus the four the unpacker drops off the end
/// (`docs/ttsim-divergence.md` row 30).
pub const DATUMS: u32 = 132;

/// `Dst` datums per aligned group of four rows -- the span one `SFPLOAD` pair
/// covers, and the operand size.
pub const GROUP_DATUMS: usize = 64;

/// Which staged datum a flat `Dst` position holds, if any.
///
/// The unpacker places datum `i` at `OutAddr = dst_base + 4 + i` and the row/column
/// split is `Row = OutAddr/16 - 4`, `Col = OutAddr & 15`
/// (`UNPACR_Regular.md:394-396`), so the flat position is just `4 + i`. The `4` is
/// ttsim holding `UNP0_ADDR_BASE_REG_1_Base` at 16; see divergence row 30.
///
/// `None` for the first four positions, which no datum reaches. `Dst` has no
/// power-on reset value (`Dst.md:15`), so those read `UnpredictableValue` and the
/// gates exclude them rather than asserting the zero ttsim happens to give.
pub const fn datum_at_dst_flat(flat: usize) -> Option<usize> {
    if flat < 4 {
        None
    } else {
        Some(flat - 4)
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
    p.push(backend::wait_for_unpacker0().unwrap());

    p.extend_from_slice(kernel);
    // The packer must not read `Dst` before the SFPU has written it: C11 with block
    // bit B8 (`STALLWAIT.md`).
    p.push(backend::wait_for_sfpu().unwrap());

    p.push(set_adc_x_pack(0, 15));
    p.push(pack_instruction(0b1111, true));
    p.push(backend::wait_for_packer().unwrap());
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
