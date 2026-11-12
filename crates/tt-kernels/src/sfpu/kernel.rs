//! The SFPU tile kernel (`hardware-coverage.md` F3): a run of tiles through
//! `Dst`, an SFPU program over each.
//!
//! Three roles, as LLK splits them: T0 unpacks each tile's operands into
//! `Dst` -- `A` to rows 0..64, `B` to rows 64..128 -- T1 runs the op's program,
//! which writes its result to rows 128..192, and T2 packs those rows to the
//! tile's output slot. The data mover gathers the run's operands into L1
//! before the kernel and scatters the outputs after it
//! (`tt_isa::dm::record::{READ_RUN, WRITE_RUN}`), so the kernel sees only L1.
//!
//! The roles hand each tile on through three semaphores, declared through
//! `crate::l1` like every kernel's: `unpacked` (T0 -> T1), `computed` (T1 ->
//! T2) and `free` (T2 -> T0: the packer has read `Dst`, the next tile may be
//! unpacked over it). The consumer waits -- `STALLWAIT` blocks the
//! *consumer*'s unit (divergence row 46) -- and each producer posts only once
//! its unit has finished. A run takes every one it posts, so it leaves them as
//! it found them (`unpacked` and `computed` at zero, `free` at one) and the
//! mover can run kernels back to back (`Kernel::restores_semaphores`).
//!
//! One tile at a time: the three roles overlap across a tile's stages only in
//! that each starts its next tile's setup while waiting. Double-buffering
//! `Dst` is 9.8's.

use tt_isa::backend::{self, Before, ConfigWords};
use tt_isa::dm::TILE_SLOT;
use tt_isa::isa::Instruction;
use tt_isa::sync::{self, Semaphore, Unit};

use crate::datapath::{
    config_program, pack_tile_from_dst, state_id, thread_config, tile_unpack_config,
    unpack_datums_to_dst, unpack_tile_to_dst, TILE_DATUMS,
};
use crate::l1::{PlanError, Requirements};
use crate::runtime::SemaphoreInit;

/// Where an op's operands are in `Dst`, and its result.
pub const A_ROW: u32 = 0;
pub const B_ROW: u32 = 64;
pub const OUT_ROW: u32 = 128;
/// A ternary op's third operand (`Operands::Ternary`): inside 32-bit `Dst`'s
/// 512 rows (`Dst.md`).
pub const C_ROW: u32 = 192;
/// Rows a program may spill registers to (`log1p`, `pow`): no unpacker writes
/// them, no packer reads them.
pub const SPILL_ROW: u32 = 256;

/// What a tile's second operand is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Operands {
    /// `A` only.
    Unary,
    /// `A` and `B`, the same shape.
    Binary,
    /// `A`, and `B`'s first row broadcast down every row: `B`'s row 0 is laid
    /// into `Dst` four times over, its first face's sixteen columns in rows
    /// `B_ROW..B_ROW + 4` and its second face's in `B_ROW + 4..B_ROW + 8`, so
    /// one `SFPLOAD` of either group gives every lane its column's value
    /// (`Program`s read them with [`bias_row`]).
    RowBroadcast,
    /// `A`, and `B`'s first column broadcast across every column: the mover
    /// reads `B`'s tile as it lies (`record::READ_RUN_COL_AS_IT_LIES`), the
    /// kernel unpacks it as [`Operands::Binary`] does, and the program copies
    /// column 0 into every column first ([`crate::sfpu::Program::broadcast_col0`]).
    ColBroadcast,
    /// `A`, `B` and `C`, the same shape: `C` to rows `C_ROW..C_ROW + 64`.
    Ternary,
}

/// How a run holds its operands and its result (roadmap W1, native BF16 arithmetic;
/// `docs/plans/tt-metal-parity.md` §5 W1). A BF16 run computes what widening each operand
/// (`bf16::expand`), the F32 op, and narrowing its result (`bf16::compress`) compute, bit for
/// bit: its prologue is the expand's own sequence -- each operand through `SrcA` in eight chunks
/// of 128 datums, a `MOVA2D` each into its `Dst` rows, FP32 there -- its program is the F32 op's,
/// and its result is rounded by the compress's own SFPU quantize (`bf16::quantize_code`) before
/// the pack. The caller appends that quantize to the op's program
/// (`sfpu::ops::code_for_rows_as`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Formats {
    /// Every operand a BF16 tile in its L1 slot: its header, then 1,024 two-byte datums (as
    /// `record::READ_RUN` reads a BF16 tensor's slot whole).
    pub bf16_in: bool,
    pub out: Out,
}

impl Formats {
    /// FP32 operands, an FP32 result: every run before native BF16.
    pub const FP32: Formats = Formats {
        bf16_in: false,
        out: Out::Fp32,
    };

    /// The format the packer writes.
    pub fn pack(&self) -> tt_isa::tile::L1Format {
        match self.out {
            Out::Bf16 | Out::Bf16Unrounded => tt_isa::tile::L1Format::Bf16,
            Out::Fp32 | Out::Bf16AsFp32 => tt_isa::tile::L1Format::Fp32,
        }
    }
}

/// What a run's result is ([`Formats`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Out {
    /// FP32 datums, as computed.
    #[default]
    Fp32,
    /// Rounded to BF16 and packed as BF16: 2,048 bytes of data a tile.
    Bf16,
    /// Rounded to BF16 and packed as FP32, the BF16 bits in each datum's high half: the
    /// simulator gate's form of [`Out::Bf16`], whose pack ttsim refuses (`PACR` mode 0x105).
    Bf16AsFp32,
    /// Packed as BF16 without the rounding, which the packer then truncates: a gate's negative
    /// control only (`tensor::probe_bf16_result`), never a result.
    Bf16Unrounded,
}

impl Out {
    /// Whether the result is rounded to BF16 before its pack.
    pub fn quantized(self) -> bool {
        matches!(self, Out::Bf16 | Out::Bf16AsFp32)
    }
}

/// The `Dst` row of the broadcast row's group for row group `g` of a tile:
/// faces 0 and 2 (row groups 0..4 and 8..12) take columns 0..16, faces 1 and 3
/// columns 16..32.
pub const fn bias_row(g: u32) -> u32 {
    if (g / 4) % 2 == 0 {
        B_ROW
    } else {
        B_ROW + 4
    }
}

/// The kernel's three semaphores, as a plan placed them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SfpuSemaphores {
    pub unpacked: Semaphore,
    pub computed: Semaphore,
    pub free: Semaphore,
}

/// Where a run of `tiles` tiles lives in L1, and its semaphores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub tiles: usize,
    /// The first operand's slots, one per tile.
    pub a_at: u64,
    /// The second operand's, unless unary.
    pub b_at: Option<u64>,
    /// `B` is one slot, read for every tile: a `[rows, 1]` column that every tile of a row
    /// broadcasts ([`Operands::ColBroadcast`]), held in L1 once (`crate::attention`).
    pub b_once: bool,
    /// The third operand's, for a ternary op.
    pub c_at: Option<u64>,
    /// The packer's output slots: datums at `TILE_DATA` past each slot's start.
    pub out_at: u64,
    pub sems: SfpuSemaphores,
    pub init: Vec<SemaphoreInit>,
}

impl Layout {
    /// This layout with every slot `by` bytes on, its semaphores where they
    /// were: the second half of a double-buffered pair.
    pub fn shifted(&self, by: u64) -> Layout {
        Layout {
            a_at: self.a_at + by,
            b_at: self.b_at.map(|b| b + by),
            c_at: self.c_at.map(|c| c + by),
            out_at: self.out_at + by,
            ..self.clone()
        }
    }
}

/// Plan a run of `tiles` tiles in the data arena: slots for the operands and
/// the outputs, and the three semaphores.
pub fn plan_layout(tiles: usize, operands: Operands) -> Result<Layout, PlanError> {
    plan_layout_in(tiles, operands, crate::l1::arena())
}

/// [`plan_layout`] in `arena`: half the data arena, for a run double-buffered
/// with the next (`crate::matmul::half_arena`, then [`Layout::shifted`]).
pub fn plan_layout_in(
    tiles: usize,
    operands: Operands,
    arena: tt_isa::l1::Region,
) -> Result<Layout, PlanError> {
    let req = std::sync::Arc::new(requirements(tiles, operands));
    let plan = req.plan(arena)?;
    Ok(Layout::placed(
        tiles,
        operands,
        &crate::l1::Placed::new(req, plan, 0),
    ))
}

/// What a run of `tiles` tiles needs: slots for the operands and the outputs, and the three
/// semaphores ([`Layout::placed`] reads them back).
pub fn requirements(tiles: usize, operands: Operands) -> Requirements {
    let mut req = Requirements::new(1);
    let bytes = tiles as u64 * TILE_SLOT;
    let align = tt_isa::dram::ALIGN;
    req.scratch("sfpu A slots", bytes, align, 0..1);
    if operands != Operands::Unary {
        req.scratch("sfpu B slots", bytes, align, 0..1);
    }
    if operands == Operands::Ternary {
        req.scratch("sfpu C slots", bytes, align, 0..1);
    }
    req.scratch("sfpu output slots", bytes, align, 0..1);
    // Declared in this order so the planner numbers them as a matmul's are --
    // the first starting at zero, the second at one -- and a tile that ran
    // one needs no setup run before the other (`runtime::Resident`).
    req.semaphore("sfpu unpacked", 0, 0..1);
    req.semaphore("sfpu Dst free", 1, 0..1);
    req.semaphore("sfpu computed", 0, 0..1);
    req
}

impl Layout {
    /// A run of `tiles` tiles where its [`requirements`] were placed.
    pub fn placed(tiles: usize, operands: Operands, l1: &crate::l1::Placed) -> Layout {
        Layout {
            tiles,
            a_at: l1.addr("sfpu A slots"),
            b_at: (operands != Operands::Unary).then(|| l1.addr("sfpu B slots")),
            b_once: false,
            c_at: (operands == Operands::Ternary).then(|| l1.addr("sfpu C slots")),
            out_at: l1.addr("sfpu output slots"),
            sems: SfpuSemaphores {
                unpacked: l1.semaphore("sfpu unpacked"),
                computed: l1.semaphore("sfpu computed"),
                free: l1.semaphore("sfpu Dst free"),
            },
            init: l1.semaphore_init(),
        }
    }
}

/// Most tiles a run may have: a slot per operand and one for the output, each,
/// in the data arena.
pub fn max_tiles(operands: Operands) -> usize {
    max_tiles_in(operands, crate::l1::arena().len())
}

/// [`max_tiles`] in an arena of `bytes`.
pub fn max_tiles_in(operands: Operands, bytes: u64) -> usize {
    let per = match operands {
        Operands::Unary => 2,
        Operands::Ternary => 4,
        _ => 3,
    };
    (bytes / (per * TILE_SLOT)) as usize
}

/// The three role programs of a run over `layout`'s tiles, `math` -- one
/// tile's SFPU program, reading `A_ROW` (and `B_ROW`) and writing `OUT_ROW`
/// -- run once per tile.
pub fn roles(layout: &Layout, operands: Operands, math: &[Instruction]) -> [Vec<Instruction>; 3] {
    let (code, loops) = roles_code(layout, operands, &crate::code::Code::plain(math.to_vec()));
    std::array::from_fn(|t| {
        crate::code::Code {
            ins: code[t].clone(),
            loops: loops[t].clone(),
        }
        .expand()
    })
}

/// [`roles`] as the role slots hold them, with each role's block repeats
/// (`crate::code`): the math role's per-tile block -- the same for every tile
/// -- stored once and repeated once per tile, and `math`'s own row loops
/// inside it, so a run's math program is one tile's whatever its length.
pub fn roles_code(
    layout: &Layout,
    operands: Operands,
    math: &crate::code::Code,
) -> ([Vec<Instruction>; 3], [Vec<crate::code::Loop>; 3]) {
    roles_code_as(layout, operands, math, Formats::FP32)
}

/// [`roles_code`] with the operands and result held as `formats` says. A BF16 run's `math`
/// already ends in the quantize its result takes, where it takes one ([`Formats`]).
pub fn roles_code_as(
    layout: &Layout,
    operands: Operands,
    math: &crate::code::Code,
    formats: Formats,
) -> ([Vec<Instruction>; 3], [Vec<crate::code::Loop>; 3]) {
    roles_code_validated(layout, operands, math, false, formats)
}

pub(crate) fn roles_code_validated(
    layout: &Layout,
    operands: Operands,
    math: &crate::code::Code,
    checked: bool,
    formats: Formats,
) -> ([Vec<Instruction>; 3], [Vec<crate::code::Loop>; 3]) {
    assert!(
        !formats.bf16_in || widened_rows(operands).is_some(),
        "{operands:?} has no BF16 form"
    );
    assert!(
        !(checked && formats != Formats::FP32),
        "a checked run is FP32"
    );
    roles_looped(layout, operands, math, checked, formats)
        .unwrap_or_else(|| roles_unrolled(layout, operands, math, checked, formats))
}

/// The `Dst` rows a BF16 run widens its operands' tiles into, in the order it reads them; `None`
/// for a row broadcast, whose kernel lays `B`'s row out by face rows (no BF16 form yet).
pub fn widened_rows(operands: Operands) -> Option<&'static [u32]> {
    match operands {
        Operands::Unary => Some(&[A_ROW]),
        Operands::Binary | Operands::ColBroadcast => Some(&[A_ROW, B_ROW]),
        Operands::Ternary => Some(&[A_ROW, B_ROW, C_ROW]),
        Operands::RowBroadcast => None,
    }
}

/// One chunk of a BF16 widening, as `bf16::expand` moves it: the `UNPACR` that hands `SrcA` to
/// the Matrix Unit (T0), and the `MOVA2D` of its eight rows to `Dst` row `row`, the wait for it
/// and the bank's release (T1).
fn bf16_chunk(row: u32) -> (Instruction, [Instruction; 3]) {
    use tt_isa::isa::generated::encode::{Mova2D, UnpacrRegular};
    let (unpack, banks) = tt_isa::matrix::Banks::after_reset()
        .unpack_a(UnpacrRegular::ZERO.multi_context_mode(1))
        .unwrap();
    let (mova, banks) = banks
        .mova2d(Mova2D::ZERO.move8_rows(1).dst_row(row))
        .unwrap();
    let (release, _) = banks.release_a().unwrap();
    (
        unpack,
        [
            mova,
            backend::wait_for_matrix(Before::EVERYTHING).unwrap(),
            release,
        ],
    )
}

/// Chunks a BF16 tile is widened in, and the bytes of L1 each covers.
const CHUNKS: u32 = TILE_DATUMS / crate::datapath::BF16_CHUNK_DATUMS;
const CHUNK_BYTES: u64 = crate::datapath::BF16_CHUNK_DATUMS as u64 * 2;

/// The math role's start of a tile of a BF16 run: each chunk taken from the unpacker as it lands
/// and moved to its operand's `Dst` rows, `free` handed back after every chunk but the tile's
/// last (the packer hands back that one); the SFPU's program follows, with the operands in
/// `Dst`. As `bf16::expand`'s math role does each tile's eight.
fn widen_math(operands: Operands, s: SfpuSemaphores) -> Vec<Instruction> {
    let rows = widened_rows(operands).expect("checked by the caller");
    let mut m = Vec::new();
    for (k, &row) in rows.iter().enumerate() {
        for c in 0..CHUNKS {
            m.extend(sync::take(s.unpacked, Before::MATRIX));
            m.extend(bf16_chunk(row + 8 * c).1);
            if !(k + 1 == rows.len() && c + 1 == CHUNKS) {
                m.extend(sync::post_after(Unit::Matrix, s.free));
            }
        }
    }
    m
}

/// The unpack role's start: everything the unpacker's address generators read, set here -- a
/// kernel inherits the thread's ADCs from whatever ran before (a matmul leaves Z at its last
/// face), and a stale counter moves where a tile is read from (`tt-metal-concepts-review.md`
/// G11) -- and its configuration: whole FP32 tiles into `Dst`, or a BF16 run's chunks into
/// `SrcA` (`datapath::bf16_chunk_config`, with the `Src` path's `ThreadConfig`).
fn unpack_prelude(layout: &Layout, bf16: bool) -> Vec<Instruction> {
    let mut unpack = if bf16 {
        crate::datapath::src_thread_config()
    } else {
        thread_config()
    };
    unpack.extend(crate::datapath::clear_unpacker0_adcs());
    let mut words = ConfigWords::new();
    if bf16 {
        crate::datapath::bf16_chunk_config(&mut words, layout.a_at);
    } else {
        tile_unpack_config(&mut words, layout.a_at);
    }
    unpack.extend(config_program(&words));
    unpack
}

/// The math role's start: a BF16 run's moves take the Matrix Unit's address counters and
/// modifiers as `bf16::expand`'s do (`matmul::math_prelude`, which leaves address modifier 0
/// stepping nothing, as the SFPU's `Dst` accesses take it).
fn math_prelude(bf16: bool) -> Vec<Instruction> {
    if bf16 {
        crate::matmul::math_prelude()
    } else {
        vec![state_id()]
    }
}

/// GPRs the looped roles keep their tiles' addresses in (`roles_looped`): clear of
/// `datapath::SCRATCH_GPR`, the matmul's 24..28, `local_movement`'s 12..16 and the
/// guard's reserved 61..64. Numbered apart on the unpack and pack threads although
/// each thread has its own.
const A_GPR: u32 = 32;
const B_GPR: u32 = 33;
/// [`Operands::RowBroadcast`]'s second face row.
const B_FACE1_GPR: u32 = 34;
const C_GPR: u32 = 35;
const UNPACK_STRIDE_GPR: u32 = 36;
const OUT_GPR: u32 = 37;
const STATUS_GPR: u32 = 38;
const PACK_STRIDE_GPR: u32 = 39;
/// A BF16 run's chunk being read, the step to the next chunk (256 bytes), and zero.
const CHUNK_GPR: u32 = 40;
const CHUNK_STRIDE_GPR: u32 = 41;
const NO_STEP_GPR: u32 = 42;

/// The roles with every tile's unpack and pack the same words, its addresses
/// stepped in GPRs a slot a tile, the tile's block stored once and repeated
/// (`crate::code::Loop`): a program's size is one tile's, whatever the run's
/// length (`docs/plans/long-context.md` 1.A). The unpackers and packer see the
/// same addresses, in the same order, as [`roles_unrolled`] gives them; the
/// math role is the same. `None` where there is nothing to repeat, or the run
/// is longer than one loop counts.
fn roles_looped(
    layout: &Layout,
    operands: Operands,
    math: &crate::code::Code,
    checked: bool,
    formats: Formats,
) -> Option<crate::code::LoopedRoles> {
    use crate::code::Loop;
    use crate::datapath::{pack_dest_units, pack_tile_from_dst_gpr, unpack_datums_to_dst_gpr};
    use tt_isa::backend::set_gpr;
    let tiles = layout.tiles as u32;
    if tiles < 2 {
        return None;
    }
    let s = layout.sems;
    let stride = crate::datapath::tile_base_units(TILE_SLOT);
    debug_assert_eq!(stride as u64 * 16, TILE_SLOT);
    // Each stepped GPR starts a slot before its first tile: the block adds first.
    let back = |units: u32| units.checked_sub(stride).expect("a slot below the arena");
    let base = |l1: u64| crate::datapath::tile_base_units(l1);

    let bf16 = formats.bf16_in;
    let mut unpack = unpack_prelude(layout, bf16);
    let set = |p: &mut Vec<Instruction>, gpr: u32, v: u32| p.extend(set_gpr(gpr, v).unwrap());
    set(&mut unpack, UNPACK_STRIDE_GPR, stride);
    if bf16 {
        set(&mut unpack, CHUNK_STRIDE_GPR, (CHUNK_BYTES / 16) as u32);
        set(&mut unpack, NO_STEP_GPR, 0);
    }
    set(&mut unpack, A_GPR, back(base(layout.a_at)));
    let step = Some(UNPACK_STRIDE_GPR);
    // `B` held once is read from one slot: never stepped.
    let b_step = if layout.b_once { None } else { step };
    let b_start = |b: u64| {
        if layout.b_once {
            base(b)
        } else {
            back(base(b))
        }
    };
    match (operands, layout.b_at) {
        (Operands::Binary | Operands::ColBroadcast, Some(b)) => set(&mut unpack, B_GPR, b_start(b)),
        (Operands::Ternary, Some(b)) => {
            set(&mut unpack, B_GPR, back(base(b)));
            let c = layout.c_at.expect("a ternary layout has C slots");
            set(&mut unpack, C_GPR, back(base(c)));
        }
        (Operands::RowBroadcast, Some(b)) => {
            // Row 0 of faces 0 and 1: datums 0..16 and 256..272. Stepped a tile
            // whatever `b_once` says, as `roles_unrolled` reads it.
            set(&mut unpack, B_GPR, back(base(b)));
            set(&mut unpack, B_FACE1_GPR, back(base(b + 4 * 256)));
        }
        (Operands::Unary, _) => {}
        _ => panic!("{operands:?} with no B slots"),
    }
    let unpack_block = unpack.len() as u32;
    if bf16 {
        // Each operand's tile in chunks: the chunk's base is the tile's (stepped a tile first,
        // unless held once) plus 256 bytes a chunk.
        let mut tiles = vec![(A_GPR, step)];
        match operands {
            Operands::Binary | Operands::ColBroadcast => tiles.push((B_GPR, b_step)),
            Operands::Ternary => tiles.extend([(B_GPR, step), (C_GPR, step)]),
            _ => {}
        }
        for (gpr, step) in tiles {
            for c in 0..CHUNKS {
                unpack.extend(sync::take(s.free, Before::UNPACKER));
                let mut adds = Vec::new();
                if c == 0 {
                    if let Some(stride) = step {
                        adds.push((gpr, stride, gpr));
                    }
                    adds.push((CHUNK_GPR, NO_STEP_GPR, gpr));
                } else {
                    adds.push((CHUNK_GPR, CHUNK_STRIDE_GPR, CHUNK_GPR));
                }
                unpack.extend(crate::datapath::unpack_src_chunk_gpr(
                    &adds,
                    CHUNK_GPR,
                    bf16_chunk(0).0,
                ));
                unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
            }
        }
    } else {
        unpack.extend(sync::take(s.free, Before::UNPACKER));
        unpack.extend(unpack_datums_to_dst_gpr(A_GPR, step, TILE_DATUMS, A_ROW));
    }
    match operands {
        _ if bf16 => {}
        Operands::Binary | Operands::ColBroadcast => {
            unpack.extend(unpack_datums_to_dst_gpr(B_GPR, b_step, TILE_DATUMS, B_ROW));
        }
        Operands::Ternary => {
            unpack.extend(unpack_datums_to_dst_gpr(B_GPR, step, TILE_DATUMS, B_ROW));
            unpack.extend(unpack_datums_to_dst_gpr(C_GPR, step, TILE_DATUMS, C_ROW));
        }
        Operands::RowBroadcast => {
            for (face, gpr) in [(0, B_GPR), (1, B_FACE1_GPR)] {
                for r in 0..4 {
                    // Each face row's base steps once a tile, at its first unpack.
                    let step = if r == 0 { step } else { None };
                    unpack.extend(unpack_datums_to_dst_gpr(
                        gpr,
                        step,
                        16,
                        B_ROW + 4 * face + r,
                    ));
                }
            }
        }
        Operands::Unary => {}
    }
    if !bf16 {
        unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
    }
    let unpack_loop = Loop {
        start: unpack_block,
        len: unpack.len() as u32 - unpack_block,
        count: tiles,
    };

    let mut m = math_prelude(bf16);
    let block = m.len() as u32;
    if bf16 {
        m.extend(widen_math(operands, s));
    } else {
        m.extend(sync::take(s.unpacked, Before::SFPU));
    }
    let at = m.len() as u32;
    m.extend_from_slice(&math.ins);
    m.extend(sync::post_after(Unit::Sfpu, s.computed));
    let mut math_loops = vec![Loop {
        start: block,
        len: m.len() as u32 - block,
        count: tiles,
    }];
    math_loops.extend(math.loops.iter().map(|l| Loop {
        start: l.start + at,
        ..*l
    }));

    let mut pack = vec![state_id()];
    set(&mut pack, PACK_STRIDE_GPR, stride);
    let out = |at: u64| back(pack_dest_units(at + tt_isa::dm::TILE_DATA));
    set(&mut pack, OUT_GPR, out(layout.out_at));
    if checked {
        set(
            &mut pack,
            STATUS_GPR,
            out(layout.c_at.expect("domain status slots")),
        );
    }
    let pack_block = pack.len() as u32;
    pack.extend(sync::take(s.computed, Before::PACKER));
    pack.extend(crate::datapath::pack_tile_from_dst_gpr_as(
        OUT_GPR,
        Some(PACK_STRIDE_GPR),
        OUT_ROW,
        formats.pack(),
    ));
    if checked {
        pack.extend(pack_tile_from_dst_gpr(
            STATUS_GPR,
            Some(PACK_STRIDE_GPR),
            C_ROW,
        ));
    }
    pack.extend(sync::post_after(Unit::Packer, s.free));
    let pack_loop = Loop {
        start: pack_block,
        len: pack.len() as u32 - pack_block,
        count: tiles,
    };

    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    let code = [unpack, m, pack];
    let loops = [vec![unpack_loop], math_loops, vec![pack_loop]];
    let fits = (0..3).all(|t| crate::code::check(&loops[t], code[t].len() as u32).is_ok());
    fits.then_some((code, loops))
}

/// The roles with every tile's unpack and pack written out, its addresses
/// immediates: only the math role repeats.
fn roles_unrolled(
    layout: &Layout,
    operands: Operands,
    math: &crate::code::Code,
    checked: bool,
    formats: Formats,
) -> ([Vec<Instruction>; 3], [Vec<crate::code::Loop>; 3]) {
    use crate::code::Loop;
    let s = layout.sems;
    let slot = |base: u64, n: usize| base + n as u64 * TILE_SLOT;
    let b_slot = |base: u64, n: usize| slot(base, if layout.b_once { 0 } else { n });
    let bf16 = formats.bf16_in;

    let mut unpack = unpack_prelude(layout, bf16);
    let mut m = math_prelude(bf16);
    let mut math_loops = Vec::new();
    let mut pack = vec![state_id()];
    for n in 0..layout.tiles {
        if bf16 {
            let mut tiles = vec![slot(layout.a_at, n)];
            match (operands, layout.b_at) {
                (Operands::Binary | Operands::ColBroadcast, Some(b)) => tiles.push(b_slot(b, n)),
                (Operands::Ternary, Some(b)) => {
                    let c = layout.c_at.expect("a ternary layout has C slots");
                    tiles.extend([slot(b, n), slot(c, n)]);
                }
                (Operands::Unary, _) => {}
                _ => panic!("{operands:?} with no B slots"),
            }
            for at in tiles {
                for c in 0..CHUNKS {
                    unpack.extend(sync::take(s.free, Before::UNPACKER));
                    unpack.extend(crate::datapath::unpack_src_chunk(
                        at + c as u64 * CHUNK_BYTES,
                        bf16_chunk(0).0,
                    ));
                    unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
                }
            }
        } else {
            unpack.extend(sync::take(s.free, Before::UNPACKER));
            unpack.extend(unpack_tile_to_dst(slot(layout.a_at, n), A_ROW));
        }
        match (operands, layout.b_at) {
            _ if bf16 => {}
            (Operands::Binary | Operands::ColBroadcast, Some(b)) => {
                unpack.extend(unpack_tile_to_dst(b_slot(b, n), B_ROW));
            }
            (Operands::Ternary, Some(b)) => {
                unpack.extend(unpack_tile_to_dst(slot(b, n), B_ROW));
                let c = layout.c_at.expect("a ternary layout has C slots");
                unpack.extend(unpack_tile_to_dst(slot(c, n), C_ROW));
            }
            (Operands::RowBroadcast, Some(b)) => {
                // Row 0 of faces 0 and 1: datums 0..16 and 256..272.
                for (face, first) in [(0, 0), (1, 256)] {
                    for r in 0..4 {
                        unpack.extend(unpack_datums_to_dst(
                            slot(b, n),
                            first,
                            16,
                            B_ROW + 4 * face + r,
                        ));
                    }
                }
            }
            (Operands::Unary, _) => {}
            _ => panic!("{operands:?} with no B slots"),
        }
        if !bf16 {
            unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
        }

        if n == 0 {
            let block = m.len() as u32;
            if bf16 {
                m.extend(widen_math(operands, s));
            } else {
                m.extend(sync::take(s.unpacked, Before::SFPU));
            }
            let at = m.len() as u32;
            m.extend_from_slice(&math.ins);
            m.extend(sync::post_after(Unit::Sfpu, s.computed));
            if layout.tiles > 1 {
                math_loops.push(Loop {
                    start: block,
                    len: m.len() as u32 - block,
                    count: layout.tiles as u32,
                });
            }
            math_loops.extend(math.loops.iter().map(|l| Loop {
                start: l.start + at,
                ..*l
            }));
        }

        pack.extend(sync::take(s.computed, Before::PACKER));
        pack.extend(crate::datapath::pack_tile_from_dst_as(
            slot(layout.out_at, n) + tt_isa::dm::TILE_DATA,
            OUT_ROW,
            formats.pack(),
        ));
        if checked {
            pack.extend(pack_tile_from_dst(
                slot(layout.c_at.expect("domain status slots"), n) + tt_isa::dm::TILE_DATA,
                C_ROW,
            ));
        }
        pack.extend(sync::post_after(Unit::Packer, s.free));
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    ([unpack, m, pack], [Vec::new(), math_loops, Vec::new()])
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::code::transfers as evaluate;

    /// The looped roles present every unpack and pack the configuration the
    /// unrolled ones do, and are otherwise the same stream, for every operand
    /// kind and run length; and their stored size does not grow with the run.
    #[test]
    fn looped_roles_unpack_and_pack_what_the_unrolled_ones_do() {
        let math = crate::code::Code::plain(vec![tt_isa::sfpu::nop(); 3]);
        let stored = |code: &[Instruction], loops: &[crate::code::Loop]| {
            crate::code::Code {
                ins: code.to_vec(),
                loops: loops.to_vec(),
            }
            .stored()
            .unwrap()
            .0
            .len()
        };
        let cases = [
            (Operands::Unary, false, false),
            (Operands::Binary, false, false),
            (Operands::Binary, false, true),
            (Operands::RowBroadcast, false, false),
            (Operands::ColBroadcast, false, false),
            (Operands::ColBroadcast, true, false),
            (Operands::Ternary, false, false),
        ];
        let bf16 = |out| Formats { bf16_in: true, out };
        let formats = [Formats::FP32, bf16(Out::Bf16), bf16(Out::Bf16AsFp32)];
        let cases = cases
            .into_iter()
            .flat_map(|c| formats.map(|f| (c, f)))
            .filter(|&((operands, _, checked), f)| {
                f == Formats::FP32 || (!checked && widened_rows(operands).is_some())
            });
        for ((operands, b_once, checked), formats) in cases {
            let mut sizes = Vec::new();
            // A checked op packs its status to the `C` slots.
            let planned = if checked { Operands::Ternary } else { operands };
            for tiles in [2, 3, 7, 32, 64].map(|n| n.min(max_tiles(planned))) {
                let mut layout = plan_layout(tiles, planned).unwrap();
                layout.b_once = b_once;
                let (looped, loops) = roles_looped(&layout, operands, &math, checked, formats)
                    .expect("a loop counts it");
                let (unrolled, unrolled_loops) =
                    roles_unrolled(&layout, operands, &math, checked, formats);
                for t in 0..3 {
                    let expand = |ins: &[Instruction], ls: &[crate::code::Loop]| {
                        crate::code::Code {
                            ins: ins.to_vec(),
                            loops: ls.to_vec(),
                        }
                        .expand()
                    };
                    let (a, b) = (
                        evaluate(&expand(&looped[t], &loops[t])),
                        evaluate(&expand(&unrolled[t], &unrolled_loops[t])),
                    );
                    let what = format!(
                        "{operands:?} b_once {b_once} checked {checked} {formats:?}, {tiles} \
                         tiles, role {t}"
                    );
                    assert_eq!(a.0.len(), b.0.len(), "{what}: unpacks and packs");
                    for (n, (x, y)) in a.0.iter().zip(&b.0).enumerate() {
                        // The looped roles also stage their GPRs' values nowhere in
                        // `Config`: every word the unrolled roles wrote must agree.
                        for (addr, v) in y {
                            assert_eq!(x.get(addr), Some(v), "{what}: transfer {n}, word {addr}");
                        }
                    }
                    assert_eq!(a.1, b.1, "{what}: the stream outside the addresses");
                }
                sizes.push(std::array::from_fn::<_, 3, _>(|t| {
                    stored(&looped[t], &loops[t])
                }));
            }
            assert!(
                sizes.windows(2).all(|w| w[0] == w[1]),
                "{operands:?}: stored words by run length {sizes:?}"
            );
        }
    }

    /// The evaluator sees a stride off by a slot: the looped roles with their
    /// stride wrong unpack from other addresses than the unrolled ones.
    #[test]
    fn the_evaluator_sees_a_wrong_stride() {
        let math = crate::code::Code::plain(vec![tt_isa::sfpu::nop()]);
        let layout = plan_layout(4, Operands::Binary).unwrap();
        let (mut looped, loops) =
            roles_looped(&layout, Operands::Binary, &math, false, Formats::FP32).unwrap();
        let wrong = tt_isa::backend::set_gpr(UNPACK_STRIDE_GPR, 2 * 260).unwrap();
        let at = looped[0]
            .windows(2)
            .position(|w| w == tt_isa::backend::set_gpr(UNPACK_STRIDE_GPR, 260).unwrap())
            .expect("the stride is set");
        looped[0][at..at + 2].copy_from_slice(&wrong);
        let (unrolled, _) = roles_unrolled(&layout, Operands::Binary, &math, false, Formats::FP32);
        let a = evaluate(
            &crate::code::Code {
                ins: looped[0].clone(),
                loops: loops[0].clone(),
            }
            .expand(),
        );
        let b = evaluate(&unrolled[0]);
        assert_ne!(a.0, b.0);
    }

    /// The SFPU kernel's semaphores agree with a matmul's on every one they
    /// share, so the two alternate on a tile with no setup between them.
    #[test]
    fn the_semaphores_agree_with_a_matmul_s() {
        let (_, matmul) = crate::matmul::MatmulSemaphores::alone();
        for operands in [
            Operands::Unary,
            Operands::Binary,
            Operands::RowBroadcast,
            Operands::ColBroadcast,
            Operands::Ternary,
        ] {
            let sfpu = plan_layout(8, operands).unwrap().init;
            for m in &matmul {
                assert!(
                    sfpu.iter().all(|s| s.0 != m.0 || s == m),
                    "{operands:?}: {sfpu:?} against {matmul:?}"
                );
            }
        }
    }
}
