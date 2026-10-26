//! The Matrix Unit datapath, shared by the matmul gates: `Src` staging, the
//! unpack role's configuration, and the math role's RWC and address-modifier
//! setup.
//!
//! Established one block at a time by `tests/step9_matmul.rs`; it lives here so
//! the tile and multi-tile gates build on the same configuration rather than on
//! a copy of it.

use tt_isa::backend::ConfigWords;
use tt_isa::cfg::generated::{alu, thread};
use tt_isa::isa::generated::encode;
use tt_isa::isa::Instruction;
use tt_isa::tile::{L1Format, TileImage};

use crate::datapath::{
    config_program, flat_descriptor, src_thread_config, state_id, thread_entry, unpack_src_config,
    Unpacker,
};

/// Datums in one `Src` or `Dst` row.
pub const ROW: usize = 16;
/// Where `SrcA` operands sit: row 0, straight from the start of their L1 run.
///
/// It was row 16, with sixteen rows of zeros staged ahead of every operand,
/// from when a datum seemed to land four columns late (divergence rows 30 and
/// 35, both our tile-header off-by-one). With that fixed, row 0 passes on ttsim
/// and both cards, and a tile's faces can be unpacked as they sit in L1. Row 8
/// is still refused by ttsim (row 41).
pub const SRC_A_ROW: usize = 0;
/// Where `SrcB` operands sit; row 0, as `SrcA`.
pub const SRC_B_ROW: usize = 0;
/// `OutDataFormat` TF32: what FP32 in L1 becomes in `Src`.
pub const TF32_CODE: u32 = 4;

/// Stage `rows` as a flat FP32 run that lands at `Src` row `src_row`, column
/// 0: `src_row` rows of zeros, then the operand. Returns the image and its
/// datum count.
pub fn stage_operand(src_row: usize, rows: &[[f32; 16]]) -> (Vec<u8>, u32) {
    let mut datums = vec![0u32; src_row * ROW];
    datums.extend(rows.iter().flatten().map(|v| v.to_bits()));
    let n = datums.len() as u32;
    let image = TileImage::new(flat_descriptor(n), L1Format::Fp32).unwrap();
    let mut staged = vec![0u8; image.total_bytes()];
    for (i, d) in datums.iter().enumerate() {
        let off = image.datum_bit_offset(i) / 8;
        staged[off..off + 4].copy_from_slice(&d.to_le_bytes());
    }
    (staged, n)
}

/// Where the two operands of a block are staged, and how many datums each is.
#[derive(Copy, Clone, Debug)]
pub struct Operands {
    pub a_addr: u64,
    pub na: u32,
    pub b_addr: u64,
    pub nb: u32,
    /// `OutDataFormat` for both unpackers.
    pub out: u32,
}

/// The unpack role's configuration: its `ThreadConfig`, both unpackers'
/// `Config`, and FP32 `Dst` with `Zero_Flag_disabled_src` clear (the other
/// setting is the "keep SrcB denormals" mode `MVMUL.md` says must not be used).
pub fn unpack_prelude(op: Operands) -> Vec<Instruction> {
    let mut p = src_thread_config();
    let mut words = ConfigWords::new();
    unpack_src_config(
        &mut words,
        Unpacker::SrcA,
        flat_descriptor(op.na),
        op.a_addr,
        op.out,
    );
    unpack_src_config(
        &mut words,
        Unpacker::SrcB,
        flat_descriptor(op.nb),
        op.b_addr,
        op.out,
    );
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    words
        .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
        .unwrap();
    p.extend(config_program(&words));
    p
}

/// The math role's setup: address modifiers, fidelity phase 0, the RWCs
/// pointed at the operands, and all of `Dst` cleared.
///
/// Address modifier 0 moves nothing; 1 advances the fidelity phase only (with
/// the measured Blackhole `AddrMod` position; divergence row 42). These are the
/// *math* thread's `ThreadConfig` and counters, so they are set there.
pub fn math_prelude() -> Vec<Instruction> {
    let mut p = vec![state_id()];
    p.push(thread_entry(thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0));
    p.push(thread_entry(thread::ADDR_MOD_DST_SEC0_DestIncr, 0));
    p.push(thread_entry(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0));
    p.push(thread_entry(thread::ADDR_MOD_DST_SEC1_FidelityIncr, 1));
    p.extend(math_addr_mods());
    p.push(thread_entry(thread::FIDELITY_BASE_Phase, 0));
    // `SrcAVal` is four bits; a row of 16 or more would take a second
    // `SETRWC` with `SrcACr` (`SETRWC.md`: `if (SrcACr) SrcAVal += RWC.SrcA_Cr`).
    const _: () = assert!(SRC_A_ROW < 16 && SRC_B_ROW < 16);
    p.push(
        encode::Setrwc::ZERO
            .src_a(1)
            .src_a_val(SRC_A_ROW as u32)
            .src_b(1)
            .src_b_val(SRC_B_ROW as u32)
            .dst(1)
            .dst_val(0)
            .fidelity(1)
            .encode()
            .unwrap(),
    );
    // All of `Dst`: mode 3 is `CLR_ALL` in LLK's encoding too (row 40).
    // (mode, use_dst32b, addr_mod, imm10)
    p.push(encode::zeroacc(3, 0, 0, 0).unwrap());
    p
}

/// The math role's address modifiers for the matmul's `MVMUL`s (`RWCs.md`,
/// `ApplyAddrMod`; Blackhole's `MVMUL` takes a three-bit `AddrMod`,
/// divergence row 42). Each `MVMUL` names its `Dst` row within an `fi`, and
/// only plain increments move the counters:
///
/// * [`MATH_AM_PHASE`] (1): the next fidelity phase;
/// * [`MATH_AM_HALF`] (2): the end of a group's first `SrcB` half -- phase
///   back to 0, `SrcB` on eight rows;
/// * [`MATH_AM_GROUP`] (3): the end of a group, but an `fi`'s last -- phase
///   and `SrcB` back to 0;
/// * [`MATH_AM_FI`] (4): the end of an `fi` -- phase and `SrcB` back to 0,
///   `Dst` on 32 rows (two faces), the next `fi`'s base.
///
/// 0 moves nothing, as before.
pub const MATH_AM_PHASE: u32 = 1;
pub const MATH_AM_HALF: u32 = 2;
pub const MATH_AM_GROUP: u32 = 3;
pub const MATH_AM_FI: u32 = 4;

fn math_addr_mods() -> Vec<Instruction> {
    use tt_isa::backend::ThreadConfigEntry as E;
    let ab = |srcb_incr: u16,
              srcb_clear: u16,
              fields: (
        tt_isa::cfg::ThreadConfigField,
        tt_isa::cfg::ThreadConfigField,
    )| {
        E::zeroed(fields.0.addr32())
            .set(fields.0, srcb_incr)
            .unwrap()
            .set(fields.1, srcb_clear)
            .unwrap()
            .encode()
            .unwrap()
    };
    // (DestIncr, DestCR, FidelityClear)
    let dst = |incr: u16,
               cr: u16,
               fclear: u16,
               f: (
        tt_isa::cfg::ThreadConfigField,
        tt_isa::cfg::ThreadConfigField,
        tt_isa::cfg::ThreadConfigField,
    )| {
        E::zeroed(f.0.addr32())
            .set(f.0, incr)
            .unwrap()
            .set(f.1, cr)
            .unwrap()
            .set(f.2, fclear)
            .unwrap()
            .encode()
            .unwrap()
    };
    use thread::*;
    vec![
        // 2: end of a first half.
        ab(
            8,
            0,
            (ADDR_MOD_AB_SEC2_SrcBIncr, ADDR_MOD_AB_SEC2_SrcBClear),
        ),
        dst(
            0,
            0,
            1,
            (
                ADDR_MOD_DST_SEC2_DestIncr,
                ADDR_MOD_DST_SEC2_DestCR,
                ADDR_MOD_DST_SEC2_FidelityClear,
            ),
        ),
        // 3: end of a group.
        ab(
            0,
            1,
            (ADDR_MOD_AB_SEC3_SrcBIncr, ADDR_MOD_AB_SEC3_SrcBClear),
        ),
        dst(
            0,
            0,
            1,
            (
                ADDR_MOD_DST_SEC3_DestIncr,
                ADDR_MOD_DST_SEC3_DestCR,
                ADDR_MOD_DST_SEC3_FidelityClear,
            ),
        ),
        // 4: end of an `fi`, the base on two faces.
        ab(
            0,
            1,
            (ADDR_MOD_AB_SEC4_SrcBIncr, ADDR_MOD_AB_SEC4_SrcBClear),
        ),
        dst(
            32,
            0,
            1,
            (
                ADDR_MOD_DST_SEC4_DestIncr,
                ADDR_MOD_DST_SEC4_DestCR,
                ADDR_MOD_DST_SEC4_FidelityClear,
            ),
        ),
    ]
}

// --- Whole tiles --------------------------------------------------------------

use crate::loops::Item;
use tt_isa::backend::{self, Before};
use tt_isa::cfg::generated::thcon;
use tt_isa::frontend::mop::MopConfig;
use tt_isa::matrix::{Banks, Loaded};
use tt_isa::sync::{self, Semaphore, Unit};
use tt_isa::tile::TileDescriptor;

/// The matmul's names for the two semaphores its math and pack roles hand
/// `Dst` over with, as its plan assigned them. The semaphores themselves are
/// ordinary `crate::l1` declarations, as any kernel's are -- a starting value
/// and a live range -- and what a run initialises comes from the plan
/// ([`Layout::init`]), so nothing here is special to a matmul but which is
/// which.
///
/// `ready` is math -> pack, "`Dst` holds a finished output tile"; `free` is
/// pack -> math, "the packer has finished reading `Dst`; it may be cleared",
/// and starts at one, so the first output tile does not wait.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct MatmulSemaphores {
    pub ready: Semaphore,
    pub free: Semaphore,
}

impl MatmulSemaphores {
    /// Declare the pair in `req`, live in `live`: `ready` starting at zero
    /// and `free` at one. A run leaves both where they started -- each `ready`
    /// posted is taken, and `free` is taken and given back once per output
    /// tile -- which is what lets a resident tile skip the setup run between
    /// matmuls (`Kernel::restores_semaphores`).
    pub fn declare(
        req: &mut crate::l1::Requirements,
        live: std::ops::Range<u32>,
    ) -> (crate::l1::Sem, crate::l1::Sem) {
        (
            req.semaphore("matmul Dst ready", 0, live.clone()),
            req.semaphore("matmul Dst free", 1, live),
        )
    }

    /// The pair as `plan` placed it.
    pub fn planned(
        plan: &crate::l1::Plan,
        (ready, free): (crate::l1::Sem, crate::l1::Sem),
    ) -> Self {
        MatmulSemaphores {
            ready: plan.semaphore(ready),
            free: plan.semaphore(free),
        }
    }

    /// The pair, and their initialisation, for a matmul that is the whole of
    /// a tile's work: a plan of nothing else.
    pub fn alone() -> (Self, Vec<crate::runtime::SemaphoreInit>) {
        let mut req = crate::l1::Requirements::new(1);
        let sems = Self::declare(&mut req, 0..1);
        let plan = req
            .plan(crate::l1::arena())
            .expect("two semaphores always fit");
        (Self::planned(&plan, sems), plan.semaphore_init())
    }
}

/// One output tile: the `(A, B)` tile-image pairs it accumulates, in order, and
/// where its 1024 datums are packed.
#[derive(Clone, Debug)]
pub struct OutputTile {
    pub pairs: Vec<(u64, u64)>,
    pub out: u64,
}

/// The unpacker's view of a `tt_layout::Layout::tt_metal_32x32` tile in
/// `format`: four 16x16 faces, one per `Z` plane.
///
/// `ZDim` sits in descriptor word 1, which ttsim models; words 2 and 3 stay
/// zero (divergence row 29).
pub fn tile_descriptor(format: L1Format) -> TileDescriptor {
    TileDescriptor::zeroed()
        .with_x_dim(16)
        .with_y_dim(16)
        .with_z_dim(4)
        .with_is_uncompressed(true)
        .with_in_data_format_raw(format.code().expect("only measured format codes"))
}

/// `OutDataFormat` BF16 (divergence row H).
pub const BF16_CODE: u32 = 5;

/// Datums in one face.
pub const FACE_DATUMS: u32 = 256;
/// `Dst` rows one 32x32 FP32 output tile occupies: four faces of sixteen.
pub const TILE_DST_ROWS: u32 = 64;

/// GPRs the matmul's unpack role steps its tiles with: each operand's base
/// address (in `tile_base_units`) and its stride from one pair to the next.
/// Clear of `datapath::SCRATCH_GPR`, the only other GPR a kernel uses.
const A_BASE_GPR: u32 = 24;
const B_BASE_GPR: u32 = 25;
const A_STRIDE_GPR: u32 = 26;
const B_STRIDE_GPR: u32 = 27;

/// The constant step from each tile pair to the next, in bytes, if there is
/// one: `(da, db)`, both non-negative multiples of the tile alignment. One
/// pair has none to need.
fn even_strides(pairs: &[(u64, u64)]) -> Option<(u64, u64)> {
    if pairs.len() < 2 {
        return None;
    }
    let step = |i: usize| {
        let (a0, b0) = pairs[i];
        let (a1, b1) = pairs[i + 1];
        Some((a1.checked_sub(a0)?, b1.checked_sub(b0)?))
    };
    let first = step(0)?;
    let align = TileImage::ALIGNMENT as u64;
    if first.0 % align != 0 || first.1 % align != 0 {
        return None;
    }
    (1..pairs.len() - 1)
        .all(|i| step(i) == Some(first))
        .then_some(first)
}

/// Fewest pairs a run of [`even_runs`] may hold for a kernel to step by runs rather than
/// retarget every pair.
const MIN_EVEN_RUN: usize = 4;

/// `pairs` cut into maximal runs each evenly spaced as [`even_strides`] needs: `(first, length,
/// (da, db))`, a run of one taking a zero step.
fn even_runs(pairs: &[(u64, u64)]) -> Vec<(usize, usize, (u64, u64))> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < pairs.len() {
        let mut end = start + 1;
        let mut stride = (0, 0);
        if end < pairs.len() {
            if let Some(s) = even_strides(&pairs[start..=end]) {
                stride = s;
                while end + 1 < pairs.len() && even_strides(&pairs[end..=end + 1]) == Some(s) {
                    end += 1;
                }
                end += 1;
            }
        }
        runs.push((start, end - start, stride));
        start = end;
    }
    runs
}

/// [`retarget`] to the pair `(a_tile, b_tile)`, with each base and its stride
/// left in the GPRs [`gpr_step`] adds: `SETDMAREG`s, then the bases written
/// to the unpackers.
fn gpr_retarget(a_tile: u64, b_tile: u64, da: u64, db: u64) -> Vec<Instruction> {
    let units = |x: u64| crate::datapath::tile_base_units(x);
    let mut p = vec![
        backend::wait_for_unpacker0(Before::CONFIG).unwrap(),
        backend::wait_for_unpacker1(Before::CONFIG).unwrap(),
    ];
    for (gpr, v) in [
        (A_BASE_GPR, units(a_tile)),
        (B_BASE_GPR, units(b_tile)),
        (A_STRIDE_GPR, units(da)),
        (B_STRIDE_GPR, units(db)),
    ] {
        p.extend(backend::set_gpr(gpr, v).unwrap());
    }
    p.extend(write_bases());
    p
}

/// Both bases from their GPRs into the unpackers' configuration, and the
/// separation `WRCFG.md` asks for before an unpack reads it. `A` is in0 and
/// goes to `SrcB` (unpacker 1, `THCON_SEC1`); `B` to `SrcA` (`THCON_SEC0`).
fn write_bases() -> [Instruction; 4] {
    [
        backend::write_word(A_BASE_GPR, thcon::THCON_SEC1_REG3_Base_address.addr32()).unwrap(),
        backend::write_word(B_BASE_GPR, thcon::THCON_SEC0_REG3_Base_address.addr32()).unwrap(),
        tt_isa::sfpu::nop(),
        backend::stallwait(
            backend::block::UNPACKER | backend::block::CONFIG,
            backend::cond::CONFIG_BUSY,
        )
        .unwrap(),
    ]
}

/// On to the next tile pair: once both unpackers have drained, each base
/// plus its stride (`ADDDMAREG`, `step38_gpr_add`), written back. The same
/// eight words every pair.
fn gpr_step() -> Vec<Instruction> {
    let mut p = Vec::new();
    if !PROBE_NO_DRAIN.load(std::sync::atomic::Ordering::Relaxed) {
        p.extend([
            backend::wait_for_unpacker0(Before::CONFIG).unwrap(),
            backend::wait_for_unpacker1(Before::CONFIG).unwrap(),
        ]);
    }
    p.extend([
        encode::adddmareg(A_BASE_GPR, A_STRIDE_GPR, A_BASE_GPR).unwrap(),
        encode::adddmareg(B_BASE_GPR, B_STRIDE_GPR, B_BASE_GPR).unwrap(),
    ]);
    p.extend(write_bases());
    p
}

/// A probe only (`silicon_probe_pair_cost`): [`gpr_step`] without its wait for both unpackers to
/// drain, so the next pair's bases may land under an unpack still reading: the products are
/// **wrong**, and only their time means anything (how much of a K-tile pair the drain costs).
/// Process-wide, and set before any product's programs are built (the program caches do not key
/// on it).
static PROBE_NO_DRAIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A probe only (`silicon_probe_pair_cost`): each unpacker's `Throttle_mode` in a product's
/// `Src` configuration, unpacker 0 (`B`, into `SrcA`) in the low byte and unpacker 1 (`A`, into
/// `SrcB`) in the next; 0 (the default) is x1, 16 bytes a cycle from L1, and 1 and 2 ask for x2
/// and x4 (Wormhole's `UNPACR_Regular.md`, "Performance"; not documented for Blackhole).
/// Process-wide, set before any product's programs are built.
static PROBE_THROTTLE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[doc(hidden)]
pub fn probe_throttle(unpacker0: u32, unpacker1: u32) {
    PROBE_THROTTLE.store(
        unpacker0 | unpacker1 << 8,
        std::sync::atomic::Ordering::Relaxed,
    );
}

#[doc(hidden)]
pub fn probe_no_drain(on: bool) {
    PROBE_NO_DRAIN.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Point both unpackers at a new pair of tile images, between `UNPACR`s.
///
/// `REG3_Base_address` is sampled by the unpacker, so the previous unpack on
/// each must have drained first, and the next must not start before the write
/// has landed (`WRCFG` is not visible for two cycles, `ConfigurationUnit.md`).
fn retarget(a_tile: u64, b_tile: u64) -> Vec<Instruction> {
    let mut p = vec![
        backend::wait_for_unpacker0(Before::CONFIG).unwrap(),
        backend::wait_for_unpacker1(Before::CONFIG).unwrap(),
    ];
    let mut words = ConfigWords::new();
    // `A` is in0 and goes to `SrcB`; `B` is in1 and goes to `SrcA`, as in LLK.
    words
        .set(
            thcon::THCON_SEC1_REG3_Base_address,
            crate::datapath::tile_base_units(a_tile),
        )
        .unwrap()
        .set(
            thcon::THCON_SEC0_REG3_Base_address,
            crate::datapath::tile_base_units(b_tile),
        )
        .unwrap();
    p.extend(config_program(&words));
    p.push(
        backend::stallwait(
            backend::block::UNPACKER | backend::block::CONFIG,
            backend::cond::CONFIG_BUSY,
        )
        .unwrap(),
    );
    p
}

/// The three role programs of one 32x32 output tile, `C = sum over k of
/// A_k @ B_k`, for the `in_fmt` tile images at `pairs[k] = (A_k, B_k)`,
/// converted to `out_fmt` in `Src`, packed as FP32 in
/// sixty-four contiguous `Dst` rows -- the output tile's four faces in order,
/// without a header -- to `out`.
///
/// Output face `(i, j)` lives in `Dst` rows `16 * (2i + j)..`; for each `k` it
/// takes `A_k` face `(i, k')` into `SrcB` and `B_k` face `(k', j)` into `SrcA`,
/// `k' = 0, 1`, and two `MVMUL`s, one per eight-row half of `SrcB`. That is
/// sixteen unpack pairs per `k` through two banks each, so the roles must run
/// concurrently (`harness::Run::concurrent` with the plan's `Layout::init`).
pub fn tile_roles(
    pairs: &[(u64, u64)],
    sems: MatmulSemaphores,
    in_fmt: L1Format,
    out_fmt: u32,
    fidelity: Fidelity,
    out: u64,
) -> [Vec<Instruction>; 3] {
    matmul_roles(
        &[OutputTile {
            pairs: pairs.to_vec(),
            out,
        }],
        sems,
        in_fmt,
        out_fmt,
        fidelity,
    )
}

/// How many of the Matrix Unit's four fidelity phases each `MVMUL` block runs
/// (`MatrixUnit.md:143-165`; LLK's `MathFidelity`).
///
/// Phase `p` multiplies one slice of `SrcA`'s mantissa by one slice of `SrcB`'s,
/// and the four together are the whole product: [`Fidelity::HiFi4`] is exact
/// wherever the `Src` values and their FP32 sums are, and [`Fidelity::Lo`] keeps
/// only the high slices -- exact for operands with at most seven significant bits
/// in `A` and five in `B`, and otherwise the least precise product on offer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Fidelity {
    /// Phase 0 only.
    Lo,
    /// Phases 0 and 1.
    HiFi2,
    /// Phases 0, 1 and 2.
    HiFi3,
    /// All four phases.
    HiFi4,
}

impl Fidelity {
    /// The number of phases, 1 to 4.
    pub const fn phases(self) -> u32 {
        match self {
            Fidelity::Lo => 1,
            Fidelity::HiFi2 => 2,
            Fidelity::HiFi3 => 3,
            Fidelity::HiFi4 => 4,
        }
    }
}

/// [`tile_roles`] for several output tiles, one after another through `Dst`.
///
/// The math role clears `Dst` for each output tile only once the pack role has
/// finished reading the last one (`sems.free`), and the pack role packs each
/// only once the math role has finished writing it (`sems.ready`). One output
/// tile is in `Dst` at a time; double-buffering it is Phase 9.
///
/// With more than one fidelity phase, each `SrcB` half gets one `MVMUL` per
/// phase on the same banks, each with address modifier 1, whose only effect is
/// `FidelityIncr` ([`math_prelude`]; `RWCs.md`, `ApplyAddrMod`), and the
/// `SETRWC` that selects the next half also puts the phase back to zero -- so
/// nothing depends on whether the phase counter wraps.
pub fn matmul_roles(
    outputs: &[OutputTile],
    sems: MatmulSemaphores,
    in_fmt: L1Format,
    out_fmt: u32,
    fidelity: Fidelity,
) -> [Vec<Instruction>; 3] {
    matmul_roles_with(outputs, sems, SrcFormats::same(in_fmt, out_fmt), fidelity)
}

/// [`matmul_roles`] with each operand in its own format ([`SrcFormats`]).
pub fn matmul_roles_with(
    outputs: &[OutputTile],
    sems: MatmulSemaphores,
    fmts: SrcFormats,
    fidelity: Fidelity,
) -> [Vec<Instruction>; 3] {
    let [unpack, math, pack] =
        matmul_items_reloading(outputs, sems, fmts, fidelity, None, false, LiveRows::All);
    // `REPLAY` alone for now: a kernel step does not carry a MOP
    // configuration to the roles' mailboxes yet (X2b).
    let lower = |items: &[Item]| crate::loops::lower_with(items, false).words;
    [lower(&unpack), lower(&math), lower(&pack)]
}

/// [`matmul_roles`], lowered with the MOP as well: each role's words and the
/// MOP configuration its loops took, for a kernel that carries it to the
/// roles' mailboxes (`runtime::Kernel::mop`, `tensor::Step::Kernel`).
///
/// `allow_mop` false lowers with `REPLAY` alone: on ttsim, whose frontend
/// FIFO takes a `MOP`'s whole expansion without backpressure and overflows
/// where silicon would stall the expander (divergence row 68).
#[allow(clippy::too_many_arguments)]
pub fn matmul_kernel(
    outputs: &[OutputTile],
    sems: MatmulSemaphores,
    in_fmt: L1Format,
    out_fmt: u32,
    fidelity: Fidelity,
    allow_mop: bool,
    b_transposed: bool,
    live: LiveRows,
) -> ([Vec<Instruction>; 3], [Option<MopConfig>; 3]) {
    matmul_kernel_with(
        outputs,
        sems,
        SrcFormats::same(in_fmt, out_fmt),
        fidelity,
        allow_mop,
        b_transposed,
        live,
    )
}

/// [`matmul_kernel`] with each operand in its own format ([`SrcFormats`]).
pub fn matmul_kernel_with(
    outputs: &[OutputTile],
    sems: MatmulSemaphores,
    fmts: SrcFormats,
    fidelity: Fidelity,
    allow_mop: bool,
    b_transposed: bool,
    live: LiveRows,
) -> ([Vec<Instruction>; 3], [Option<MopConfig>; 3]) {
    let [unpack, math, pack] =
        matmul_items_reloading(outputs, sems, fmts, fidelity, None, b_transposed, live);
    // With `allow_mop`, a `MOP` on the math role only: on silicon (row AE) one
    // on the unpack or pack role made MNIST inference slower (0.487 -> 0.500,
    // 0.528 ms a batch), and the math role's is no faster end to end on the
    // fixed mover firmware, so the session passes `false`.
    let (u, m, p) = (
        crate::loops::lower_with(&unpack, false),
        crate::loops::lower_with(&math, allow_mop),
        crate::loops::lower_with(&pack, false),
    );
    ([u.words, m.words, p.words], [u.mop, m.mop, p.mop])
}

/// Key of the matmul's shared block (`crate::loops::Item::Shared`): one tile
/// pair's face block on the unpack role, where the pairs are not evenly
/// spaced.
const UNPACK_FACES: u32 = 1;

/// Whether a product's last tile row computes only its own rows ([`LiveRows::for_job`]): on by
/// default.
static TRIM_ROWS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Turn [`LiveRows::for_job`]'s trim on or off (a gate's control, a benchmark's comparison).
/// Process-wide; the program caches key on the rows computed.
pub fn set_trim_rows(on: bool) {
    TRIM_ROWS.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The rows of each output tile a kernel computes: the rest are left as `ZEROACC` (or a K
/// block's reload) left them. A product whose output rows past `m` are padding (a decode
/// product's one row) needs only the `MVMUL`s that write its rows: `A`'s faces `(0, k)` and the
/// first eight rows of `SrcB`. Each computed row takes the same products in the same order as
/// [`LiveRows::All`]'s, so its bits are the same; the rows left out are zero, which every padding
/// claim ([`crate::tensor::Pad`]) allows.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LiveRows {
    /// All 32.
    All,
    /// Rows 0-15: `A`'s faces `(0, k)` only.
    Sixteen,
    /// Rows 0-7: `A`'s faces `(0, k)` and the first half of `SrcB`.
    Eight,
}

impl LiveRows {
    /// The fewest rows that compute the first `rows` of a tile (`1..=32`).
    pub fn for_rows(rows: usize) -> Self {
        match rows {
            1..=8 => LiveRows::Eight,
            9..=16 => LiveRows::Sixteen,
            _ => LiveRows::All,
        }
    }

    /// What a job of `rows` output tile rows from tile row `i0` computes, of a product of `m`
    /// rows: its rows' worth when it is the product's last tile row alone, else every row.
    /// Every row while [`set_trim_rows`] has it off.
    pub fn for_job(m: usize, i0: usize, rows: usize) -> Self {
        let mt = m.div_ceil(32);
        if rows == 1 && i0 + 1 == mt && TRIM_ROWS.load(std::sync::atomic::Ordering::Relaxed) {
            Self::for_rows(m - 32 * (mt - 1))
        } else {
            LiveRows::All
        }
    }

    /// `A`'s face rows (`fi`) unpacked: 1 or 2.
    fn face_rows(self) -> u32 {
        match self {
            LiveRows::All => 2,
            LiveRows::Sixteen | LiveRows::Eight => 1,
        }
    }

    /// `SrcB`'s eight-row halves multiplied: 1 or 2.
    fn halves(self) -> u32 {
        match self {
            LiveRows::Eight => 1,
            LiveRows::All | LiveRows::Sixteen => 2,
        }
    }
}

/// `outputs` with each pair's `A` tile where a packed `A` ([`crate::tensor::pack_row`]) holds
/// it: its packed tiles read into the slots from `a_at`, the first holding `K` tile `k0 / 32 *
/// 32`, and pair `q` this block's `K` tile `k0 + q`. The unpacker reads a face as 16 rows of 64
/// bytes from its base (past a 16-byte header): `K` tile `kk`'s faces `(0, 0)` and `(0, 1)`
/// start at row `kk % 16` of the packed faces `((kk % 32) / 16, 0)` and `((kk % 32) / 16, 1)`,
/// which are 1024 bytes apart as a tile's faces `(0, 0)` and `(0, 1)` are. So `SrcB`'s row 0 is
/// `K` tile `kk`'s row 0, and its other rows hold later packed rows, which only reach output
/// rows past the one a packed product has.
pub fn packed_a_pairs(outputs: &[OutputTile], a_at: u64, k0: usize) -> Vec<OutputTile> {
    packed_a_pairs_from(outputs, a_at, k0 / 32, k0)
}

/// [`packed_a_pairs`] with the slots from `a_at` holding the packed tiles from `first` on (a
/// K-streamed product's whole packed `A`, `first` 0).
pub fn packed_a_pairs_from(
    outputs: &[OutputTile],
    a_at: u64,
    first: usize,
    k0: usize,
) -> Vec<OutputTile> {
    outputs
        .iter()
        .map(|o| OutputTile {
            pairs: o
                .pairs
                .iter()
                .enumerate()
                .map(|(q, &(_, b))| {
                    let kk = k0 + q;
                    let tile = (kk / 32 - first) as u64;
                    let r = (kk % 32) as u64;
                    let a = a_at + tile * tt_isa::dm::TILE_SLOT + (r / 16) * 2048 + (r % 16) * 64;
                    (a, b)
                })
                .collect(),
            out: o.out,
        })
        .collect()
}

/// Continuation state: the prior FP32 tile and the two-way reset/load handshake.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Reload {
    pub prior: u64,
    pub cleared: Semaphore,
    pub loaded: Semaphore,
}

/// One-output K block, with a separate prior tile and declared handshakes.
pub(crate) fn k_block_layout(kt: usize) -> Result<(Layout, Reload), crate::runtime::RunError> {
    k_block_layout_with(kt, kt, tt_isa::dm::TILE_SLOT)
}

/// [`k_block_layout`] with `a_slots` slots for `A`: a packed `A`'s, `ceil(kt / 32) + 1` (a block's
/// `K` range may start inside a packed tile), and `B`'s tiles `b_slot` bytes apart
/// ([`SrcRoute::b_slot`]).
pub(crate) fn k_block_layout_with(
    kt: usize,
    a_slots: usize,
    b_slot: u64,
) -> Result<(Layout, Reload), crate::runtime::RunError> {
    let mut m = matmul_requirements_placed([1, kt, 1], a_slots, None, b_slot);
    let prior = m
        .req
        .scratch("matmul prior", tt_isa::dm::TILE_SLOT, 64, 0..1);
    let cleared = m.req.semaphore("matmul cleared", 0, 0..1);
    let loaded = m.req.semaphore("matmul loaded", 0, 0..1);
    let plan =
        m.req
            .plan(crate::l1::arena())
            .map_err(|_| crate::runtime::RunError::DoesNotFit {
                what: "a K-block matmul",
                bytes: (2 * kt + 2) as u64 * tt_isa::dm::TILE_SLOT,
                limit: crate::l1::arena().len(),
            })?;
    let (a_at, b_at) = (plan.addr(m.a), plan.addr(m.b));
    let layout = Layout {
        a_at,
        b_at,
        outputs: vec![OutputTile {
            pairs: (0..kt)
                .map(|k| {
                    (
                        a_at + k as u64 * tt_isa::dm::TILE_SLOT,
                        b_at + k as u64 * b_slot,
                    )
                })
                .collect(),
            out: plan.addr(m.out) + tt_isa::dm::TILE_DATA,
        }],
        sems: MatmulSemaphores::planned(&plan, m.sems),
        init: plan.semaphore_init(),
    };
    Ok((
        layout,
        Reload {
            prior: plan.addr(prior),
            cleared: plan.semaphore(cleared),
            loaded: plan.semaphore(loaded),
        },
    ))
}

/// The L1 of a one-row product streamed over `K` in chunks of `kq` tiles ([`k_stream_layout`]):
/// a packed `A` read once, and two halves, each a chunk's `B` and an output slot. Chunk `c` runs
/// in half `c % 2`; it reloads the partial chunk `c - 1` packed in the other half's output slot
/// ([`Reload`], the K block's FP32 reload, from L1 rather than GDDR), and packs its own into its
/// half's. So the mover reads chunk `c + 1` while chunk `c` computes, and only a column's last
/// chunk is written out.
#[derive(Clone, Debug)]
pub(crate) struct StreamLayout {
    pub a_at: u64,
    pub b_at: [u64; 2],
    /// `B`'s tiles apart ([`SrcRoute::b_slot`]).
    pub b_slot: u64,
    /// Each half's output slot (its first byte: the packer writes past the header).
    pub out: [u64; 2],
    pub sems: MatmulSemaphores,
    pub cleared: Semaphore,
    pub loaded: Semaphore,
    pub init: Vec<crate::runtime::SemaphoreInit>,
}

impl StreamLayout {
    /// Chunk `c` (its `K` tiles from `k0`, `kq` of them) as a K block: its layout, `A`'s pairs
    /// in the packed `A`, and its reload of the partial before it.
    pub fn chunk(&self, c: usize, k0: usize, kq: usize) -> (Layout, Option<Reload>) {
        let h = c % 2;
        let outputs = vec![OutputTile {
            pairs: (0..kq)
                .map(|q| (self.a_at, self.b_at[h] + q as u64 * self.b_slot))
                .collect(),
            out: self.out[h] + tt_isa::dm::TILE_DATA,
        }];
        let layout = Layout {
            a_at: self.a_at,
            b_at: self.b_at[h],
            outputs: packed_a_pairs_from(&outputs, self.a_at, 0, k0),
            sems: self.sems,
            init: self.init.clone(),
        };
        let reload = (c > 0).then_some(Reload {
            prior: self.out[1 - h],
            cleared: self.cleared,
            loaded: self.loaded,
        });
        (layout, reload)
    }
}

/// [`StreamLayout`] for chunks of `kq` `K` tiles, a packed `A` of `a_slots` tiles and `B`'s
/// tiles `b_slot` bytes apart ([`SrcRoute::b_slot`]).
pub(crate) fn k_stream_layout(
    kq: usize,
    a_slots: usize,
    b_slot: u64,
) -> Result<StreamLayout, crate::runtime::RunError> {
    use crate::l1::{Endpoint, Requirements};
    use tt_isa::dm::TILE_SLOT;
    let mut req = Requirements::new(1);
    let mut ring = |name, page: u64, pages: usize, from, to| {
        req.cb(
            name,
            page,
            pages as u32,
            tt_isa::dram::ALIGN,
            from,
            to,
            0..1,
        )
    };
    let (input, output) = (
        (Endpoint::Mover, Endpoint::Unpack),
        (Endpoint::Pack, Endpoint::Mover),
    );
    let a = ring("matmul A", TILE_SLOT, a_slots, input.0, input.1);
    let b = ["matmul B, half 0", "matmul B, half 1"]
        .map(|name| ring(name, b_slot, kq, input.0, input.1));
    let out = ["matmul out, half 0", "matmul out, half 1"]
        .map(|name| ring(name, TILE_SLOT, 1, output.0, output.1));
    let sems = MatmulSemaphores::declare(&mut req, 0..1);
    let cleared = req.semaphore("matmul cleared", 0, 0..1);
    let loaded = req.semaphore("matmul loaded", 0, 0..1);
    let plan = req
        .plan(crate::l1::arena())
        .map_err(|_| crate::runtime::RunError::DoesNotFit {
            what: "a K-streamed matmul",
            bytes: (a_slots + 2) as u64 * TILE_SLOT + 2 * kq as u64 * b_slot,
            limit: crate::l1::arena().len(),
        })?;
    Ok(StreamLayout {
        a_at: plan.addr(a),
        b_at: b.map(|b| plan.addr(b)),
        b_slot,
        out: out.map(|o| plan.addr(o)),
        sems: MatmulSemaphores::planned(&plan, sems),
        cleared: plan.semaphore(cleared),
        loaded: plan.semaphore(loaded),
        init: plan.semaphore_init(),
    })
}

/// The L1 of a 2-D blocked product streamed over `K` ([`block_stream_layout`],
/// `docs/plans/prefill-operand-reuse.md`): a job's `mb x nb` output tiles, its `K` in chunks of at
/// most `kq` tiles. Chunk `c` gathers `A`'s `mb x kq` tiles and `B`'s `kq x nb` into input half
/// `c % 2`, reloads each output's partial from output set `1 - c % 2` (none for the first chunk)
/// and packs it into set `c % 2`: [`StreamLayout`]'s halves, for a block. The sets alternate as
/// the inputs do because a unit's chunks are double-buffered and a chunk's gather is only ordered
/// after the scatter two chunks back, so a partial packed in place could land on slots the last
/// job's final scatter still reads.
#[derive(Clone, Debug)]
pub(crate) struct BlockStreamLayout {
    /// `[mb, kq, nb]`: the block's output tiles and the longest chunk.
    pub tiles: [usize; 3],
    pub a_at: [u64; 2],
    pub b_at: [u64; 2],
    /// `B`'s tiles apart ([`SrcRoute::b_slot`]).
    pub b_slot: u64,
    /// Each set's first output slot (its first byte: the packer writes past the header).
    pub out: [u64; 2],
    pub sems: MatmulSemaphores,
    pub cleared: Semaphore,
    pub loaded: Semaphore,
    pub init: Vec<crate::runtime::SemaphoreInit>,
}

impl BlockStreamLayout {
    /// Output tile `(i, j)` of a block `cols` wide: its slot in set `h`, header included, row-major
    /// as `record::SCATTER` writes a block out.
    pub fn out_slot(&self, h: usize, cols: usize, i: usize, j: usize) -> u64 {
        self.out[h] + (i * cols + j) as u64 * tt_isa::dm::TILE_SLOT
    }

    /// Chunk `c` of a block of `rows x cols` output tiles, its `count` `K` tiles, as the
    /// gather stages it (`record::GATHER`: `A` tile `(i, kk)` at `a_at + (i count + kk)`
    /// slots, `B` tile `(kk, j)` at `b_at + (kk cols + j)`): its layout, its outputs row-major,
    /// and their priors when it continues.
    pub fn chunk(&self, c: usize, [rows, count, cols]: [usize; 3]) -> (Layout, Option<Vec<u64>>) {
        use tt_isa::dm::{TILE_DATA, TILE_SLOT};
        let [mb, kq, nb] = self.tiles;
        assert!(
            (1..=mb).contains(&rows) && (1..=kq).contains(&count) && (1..=nb).contains(&cols),
            "a chunk of {rows} x {count} x {cols} tiles in a {mb} x {kq} x {nb} layout"
        );
        let h = c % 2;
        let (a_at, b_at) = (self.a_at[h], self.b_at[h]);
        let mut outputs = Vec::with_capacity(rows * cols);
        let mut priors = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                let pairs = (0..count)
                    .map(|kk| {
                        (
                            a_at + (i * count + kk) as u64 * TILE_SLOT,
                            b_at + (kk * cols + j) as u64 * self.b_slot,
                        )
                    })
                    .collect();
                outputs.push(OutputTile {
                    pairs,
                    out: self.out_slot(h, cols, i, j) + TILE_DATA,
                });
                priors.push(self.out_slot(1 - h, cols, i, j));
            }
        }
        let layout = Layout {
            a_at,
            b_at,
            outputs,
            sems: self.sems,
            init: self.init.clone(),
        };
        (layout, (c > 0).then_some(priors))
    }
}

/// [`BlockStreamLayout`] for `[mb, kq, nb]` and `B`'s tiles `b_slot` bytes apart
/// ([`SrcRoute::b_slot`]), planned in the current arena (`crate::l1::arena`).
pub(crate) fn block_stream_layout(
    [mb, kq, nb]: [usize; 3],
    b_slot: u64,
) -> Result<BlockStreamLayout, crate::runtime::RunError> {
    use crate::l1::{Endpoint, PlanError, Requirements};
    use tt_isa::dm::TILE_SLOT;
    let mut req = Requirements::new(1);
    let mut ring = |name, page: u64, pages: usize, from, to| {
        req.cb(
            name,
            page,
            pages as u32,
            tt_isa::dram::ALIGN,
            from,
            to,
            0..1,
        )
    };
    let (input, output) = (
        (Endpoint::Mover, Endpoint::Unpack),
        (Endpoint::Pack, Endpoint::Mover),
    );
    let a = ["matmul A, half 0", "matmul A, half 1"]
        .map(|name| ring(name, TILE_SLOT, mb * kq, input.0, input.1));
    let b = ["matmul B, half 0", "matmul B, half 1"]
        .map(|name| ring(name, b_slot, kq * nb, input.0, input.1));
    let out = ["matmul out, set 0", "matmul out, set 1"]
        .map(|name| ring(name, TILE_SLOT, mb * nb, output.0, output.1));
    let sems = MatmulSemaphores::declare(&mut req, 0..1);
    let cleared = req.semaphore("matmul cleared", 0, 0..1);
    let loaded = req.semaphore("matmul loaded", 0, 0..1);
    let bytes = block_stream_bytes([mb, kq, nb], b_slot);
    let plan = req.plan(crate::l1::arena()).map_err(|e| match e {
        PlanError::DoesNotFit { .. } | PlanError::TooManyPages { .. } => {
            crate::runtime::RunError::DoesNotFit {
                what: "a K-streamed blocked matmul",
                bytes,
                limit: crate::l1::arena().len(),
            }
        }
        e => panic!("the blocked matmul's own requirements are invalid: {e}"),
    })?;
    Ok(BlockStreamLayout {
        tiles: [mb, kq, nb],
        a_at: a.map(|a| plan.addr(a)),
        b_at: b.map(|b| plan.addr(b)),
        b_slot,
        out: out.map(|o| plan.addr(o)),
        sems: MatmulSemaphores::planned(&plan, sems),
        cleared: plan.semaphore(cleared),
        loaded: plan.semaphore(loaded),
        init: plan.semaphore_init(),
    })
}

/// The bytes of a [`block_stream_layout`]'s slots: two halves of `A` and `B` and two output sets.
pub(crate) fn block_stream_bytes([mb, kq, nb]: [usize; 3], b_slot: u64) -> u64 {
    2 * ((kq * mb + mb * nb) as u64 * tt_isa::dm::TILE_SLOT + (kq * nb) as u64 * b_slot)
}

/// The three role programs of a [`BlockStreamLayout`] chunk: every output tile of the block over
/// the chunk's pairs, continuing from `priors` when given.
pub(crate) fn block_stream_kernel(
    layout: &Layout,
    priors: Option<&[u64]>,
    reload: (Semaphore, Semaphore),
    route: SrcRoute,
    fidelity: Fidelity,
    b_transposed: bool,
    live: LiveRows,
) -> [Vec<Instruction>; 3] {
    let items = matmul_items_continuing(
        &layout.outputs,
        layout.sems,
        route.src_formats(),
        fidelity,
        priors.map(|priors| Continuation {
            priors,
            cleared: reload.0,
            loaded: reload.1,
        }),
        b_transposed,
        live,
    );
    items.map(|items| crate::loops::lower_with(&items, false).words)
}

/// GPRs the looped block kernel ([`block_stream_kernel_looped`]) steps its addresses with, beside
/// [`A_BASE_GPR`]..[`B_STRIDE_GPR`]: a slot, `A`'s row of the block and its step, `B`'s column, the
/// prior partial, zero, and the packer's output.
const SLOT_GPR: u32 = 28;
const A_ROW_GPR: u32 = 29;
const A_ROW_STEP_GPR: u32 = 30;
const B_COL_GPR: u32 = 31;
const PRIOR_GPR: u32 = 40;
const ZERO_GPR: u32 = 41;
const OUT_GPR: u32 = 42;
/// `B`'s slot, which `B`'s column steps by ([`BlockStreamLayout::b_slot`]).
const B_SLOT_GPR: u32 = 43;

/// [`block_stream_kernel`] with each output tile's block of each role stored once and repeated
/// by the role runner (`crate::code::Loop`; `docs/plans/prefill-operand-reuse.md` P1b): its
/// addresses -- `A`'s row and `B`'s column, the prior partial, the output -- stepped in GPRs, so a
/// role's program does not grow with the block. The unpack role repeats an output a column inside
/// a row; the math and pack roles an output a tile. Every output takes its pairs in the same order
/// from the same slots, so the products are [`block_stream_kernel`]'s. `None` where the loops do not
/// fit the runner's header.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_stream_kernel_looped(
    layout: &BlockStreamLayout,
    c: usize,
    [rows, count, cols]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    b_transposed: bool,
    live: LiveRows,
) -> Option<crate::code::LoopedRoles> {
    use crate::code::Loop;
    use crate::datapath::{pack_dest_units, tile_base_units as units};
    use tt_isa::dm::{TILE_DATA, TILE_SLOT};
    let h = c % 2;
    let continuing = c > 0;
    let (a_at, b_at) = (layout.a_at[h], layout.b_at[h]);
    let slot = units(TILE_SLOT);
    let set =
        |p: &mut Vec<Instruction>, gpr: u32, v: u32| p.extend(backend::set_gpr(gpr, v).unwrap());
    let lowered = |items: Vec<Item>| crate::loops::lower_with(&items, false).words;
    let PairCode {
        prelude,
        words,
        ub,
        pair_math,
    } = pair_code(
        (a_at, b_at),
        route.src_formats(),
        fidelity,
        b_transposed,
        live,
    );

    // Unpack: the strides `gpr_step` adds, and each stepped GPR a step before its first use (the
    // blocks add first).
    let mut unpack = prelude;
    set(&mut unpack, A_STRIDE_GPR, slot);
    let b_slot = units(layout.b_slot);
    set(
        &mut unpack,
        B_STRIDE_GPR,
        units(cols as u64 * layout.b_slot),
    );
    set(&mut unpack, SLOT_GPR, slot);
    set(&mut unpack, B_SLOT_GPR, b_slot);
    set(&mut unpack, ZERO_GPR, 0);
    let a_row_step = units(count as u64 * TILE_SLOT);
    set(&mut unpack, A_ROW_STEP_GPR, a_row_step);
    set(&mut unpack, A_ROW_GPR, units(a_at) - a_row_step);
    let prior0 = layout.out_slot(1 - h, cols, 0, 0);
    if continuing {
        set(&mut unpack, PRIOR_GPR, units(prior0) - slot);
    }
    let row_block = unpack.len() as u32;
    set(&mut unpack, B_COL_GPR, units(b_at) - b_slot);
    unpack.push(encode::adddmareg(A_ROW_GPR, A_ROW_STEP_GPR, A_ROW_GPR).unwrap());
    let col_block = unpack.len() as u32;
    if continuing {
        unpack.extend(sync::take(layout.cleared, Before::UNPACKER));
        unpack.extend(crate::datapath::thread_config());
        unpack.extend(crate::datapath::clear_unpacker0_adcs());
        let mut prior_words = ConfigWords::new();
        // Any prior: the base is the GPR's, written below.
        crate::datapath::tile_unpack_config(&mut prior_words, prior0);
        unpack.extend(config_program(&prior_words));
        unpack.extend(crate::datapath::unpack_datums_to_dst_gpr(
            PRIOR_GPR,
            Some(SLOT_GPR),
            crate::datapath::TILE_DATUMS,
            0,
        ));
        unpack.extend(sync::post_after(Unit::Unpacker0, layout.loaded));
        // The whole Src configuration again: `UnpackToDst` leaves different state behind.
        unpack.extend(src_thread_config());
        unpack.extend(crate::datapath::clear_unpacker0_adcs());
        unpack.extend(config_program(&words));
        for u in [Unpacker::SrcA, Unpacker::SrcB] {
            unpack.push(crate::datapath::set_adc_x(u, 0, FACE_DATUMS - 1));
        }
    }
    // This output's first pair: `A`'s row and `B`'s column, as `gpr_retarget` sets them.
    unpack.push(encode::adddmareg(B_COL_GPR, B_SLOT_GPR, B_COL_GPR).unwrap());
    unpack.push(backend::wait_for_unpacker0(Before::CONFIG).unwrap());
    unpack.push(backend::wait_for_unpacker1(Before::CONFIG).unwrap());
    // `ADDDMAREG` names its result first.
    unpack.push(encode::adddmareg(A_BASE_GPR, ZERO_GPR, A_ROW_GPR).unwrap());
    unpack.push(encode::adddmareg(B_BASE_GPR, ZERO_GPR, B_COL_GPR).unwrap());
    unpack.extend(write_bases());
    let body: Vec<Item> = ub.iter().copied().chain(gpr_step()).map(Item::I).collect();
    unpack.extend(lowered(vec![Item::Repeat {
        times: count as u32,
        body,
    }]));
    let end = unpack.len() as u32;
    let mut unpack_loops = vec![Loop {
        start: col_block,
        len: end - col_block,
        count: cols as u32,
    }];
    if rows > 1 {
        unpack_loops.push(Loop {
            start: row_block,
            len: end - row_block,
            count: rows as u32,
        });
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    unpack.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());

    // Math: the same words every output.
    let mut math = math_prelude();
    let block = math.len() as u32;
    math.extend(sync::take(layout.sems.free, Before::MATRIX));
    math.push(encode::zeroacc(3, 0, 0, 0).unwrap());
    if continuing {
        math.extend(sync::post_after(Unit::Matrix, layout.cleared));
        math.extend(sync::take(layout.loaded, Before::MATRIX));
    }
    math.extend(lowered(vec![Item::Repeat {
        times: count as u32,
        body: pair_math,
    }]));
    math.extend(sync::post_after(Unit::Matrix, layout.sems.ready));
    let outputs = (rows * cols) as u32;
    let math_loops = vec![Loop {
        start: block,
        len: math.len() as u32 - block,
        count: outputs,
    }];

    // Pack: the whole configuration once, then each output's destination from its GPR.
    let out0 = layout.out_slot(h, cols, 0, 0) + TILE_DATA;
    let mut pw = ConfigWords::new();
    crate::datapath::pack_config(&mut pw, out0);
    let mut pack = config_program(&pw);
    set(&mut pack, SLOT_GPR, slot);
    set(&mut pack, OUT_GPR, pack_dest_units(out0) - slot);
    let block = pack.len() as u32;
    pack.push(backend::wait_for_packer(Before::CONFIG).unwrap());
    pack.push(encode::adddmareg(OUT_GPR, SLOT_GPR, OUT_GPR).unwrap());
    pack.push(crate::datapath::gpr_settled());
    pack.push(
        backend::write_word(
            OUT_GPR,
            tt_isa::cfg::generated::thcon::THCON_SEC0_REG1_L1_Dest_addr.addr32(),
        )
        .unwrap(),
    );
    pack.push(tt_isa::sfpu::nop());
    pack.push(
        backend::stallwait(
            backend::block::PACKER | backend::block::CONFIG,
            backend::cond::CONFIG_BUSY,
        )
        .unwrap(),
    );
    pack.extend(sync::take(layout.sems.ready, Before::PACKER));
    pack.extend(lowered(crate::datapath::pack_rows_items(TILE_DST_ROWS)));
    pack.extend(sync::post_after(Unit::Packer, layout.sems.free));
    let pack_loops = vec![Loop {
        start: block,
        len: pack.len() as u32 - block,
        count: outputs,
    }];
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());

    let code = [unpack, math, pack];
    let loops = [unpack_loops, math_loops, pack_loops];
    let fits = (0..3).all(|t| crate::code::check(&loops[t], code[t].len() as u32).is_ok());
    fits.then_some((code, loops))
}

/// Whether a blocked product's roles repeat an output's block ([`block_stream_kernel_looped`]) rather
/// than write every output out ([`block_stream_kernel`]): on by default.
static BLOCK_LOOPS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Turn [`block_stream_kernel_looped`] on or off, process-wide (a gate's comparison). The program
/// cache keys on it.
pub fn set_block_loops(on: bool) {
    BLOCK_LOOPS.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// A kernel's three role programs and their block repeats, shared.
pub(crate) type SharedLoopedRoles = (
    std::sync::Arc<[Vec<Instruction>; 3]>,
    std::sync::Arc<[Vec<crate::code::Loop>; 3]>,
);

/// A blocked product's chunk's programs and their loops: [`block_stream_kernel_looped`]'s where
/// [`set_block_loops`] has them on and they fit, else [`block_stream_kernel`]'s. Built once a process
/// for each chunk shape, half, layout and arena.
pub(crate) fn block_stream_programs(
    layout: &BlockStreamLayout,
    c: usize,
    tiles: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    b_transposed: bool,
    live: LiveRows,
) -> SharedLoopedRoles {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    type Key = (
        [usize; 3],
        [usize; 3],
        usize,
        bool,
        SrcRoute,
        Fidelity,
        bool,
        LiveRows,
        u64,
        MatmulSemaphores,
        bool,
    );
    static CACHE: OnceLock<Mutex<HashMap<Key, SharedLoopedRoles>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let looped = BLOCK_LOOPS.load(std::sync::atomic::Ordering::Relaxed);
    let key = (
        tiles,
        layout.tiles,
        c % 2,
        c > 0,
        route,
        fidelity,
        b_transposed,
        live,
        crate::l1::arena().end,
        layout.sems,
        looped,
    );
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return v.clone();
    }
    let built = looped
        .then(|| block_stream_kernel_looped(layout, c, tiles, route, fidelity, b_transposed, live))
        .flatten()
        .unwrap_or_else(|| {
            let (l, priors) = layout.chunk(c, tiles);
            let roles = block_stream_kernel(
                &l,
                priors.as_deref(),
                (layout.cleared, layout.loaded),
                route,
                fidelity,
                b_transposed,
                live,
            );
            (roles, Default::default())
        });
    let value = (Arc::new(built.0), Arc::new(built.1));
    cache.lock().unwrap().insert(key, value.clone());
    value
}

/// First/continuation programs share the Matrix Unit's original product order.
pub(crate) fn k_block_kernel(
    layout: &Layout,
    reload: Option<Reload>,
    route: SrcRoute,
    fidelity: Fidelity,
    b_transposed: bool,
    live: LiveRows,
) -> [Vec<Instruction>; 3] {
    let items = matmul_items_reloading(
        &layout.outputs,
        layout.sems,
        route.src_formats(),
        fidelity,
        reload,
        b_transposed,
        live,
    );
    items.map(|items| crate::loops::lower_with(&items, false).words)
}

/// The three role programs of [`matmul_roles`] as loop items, before
/// lowering: each tile pair's face block -- the same words for every pair --
/// a shared block on the unpack and the math role, between the pairs' own
/// retargeting (`crate::loops`).
///
/// The face loop runs `fi`, `k`, `fj`: `Dst` face `(fi, fj)` still takes its
/// `k = 0` product before its `k = 1` one within each pair, and pairs in
/// order, so every output datum accumulates exactly as before -- but each `A`
/// face is unpacked once and read by the two `MVMUL` groups that need it,
/// released by the second. Each face is picked by `SETADCZW` alone --
/// `FirstDatum = ((W * ZDim + Z) * YDim + Y) * XDim + X`, `Z` from the issuing
/// thread's ADCs and `X`/`Y` from `ContextADC`'s, thread 0 for both on the
/// unpack role (`UNPACR_Regular.md:136-182`) -- and the `X` range is the whole
/// face throughout, so it is set once at the start rather than before every
/// unpack. A pair's block is 24 words on the
/// unpack role, which the replay buffer holds; on the math role it is 32 words
/// at LoFi, and longer, so written out, at higher fidelities.
pub fn matmul_items(
    outputs: &[OutputTile],
    sems: MatmulSemaphores,
    in_fmt: L1Format,
    out_fmt: u32,
    fidelity: Fidelity,
) -> [Vec<Item>; 3] {
    matmul_items_reloading(
        outputs,
        sems,
        SrcFormats::same(in_fmt, out_fmt),
        fidelity,
        None,
        false,
        LiveRows::All,
    )
}

/// [`matmul_items`], continuing a K block's accumulator from `reload`'s prior tile when given,
/// and with each `B` tile the transpose of the tile in its slot when `b_transposed`: unpacker 0
/// transposes within each face (`Haloize_mode`, [`tt_isa::unpacker::stage_transpose`]) and `B`'s
/// face `(k, fj)` is unpacked from the slot's face `(fj, k)`, so the mover reads the stored tile
/// as it lies (`record::GATHER_B_UNPACK_TRANSPOSE`). The unpack converts each datum as it does
/// untransposed, so `SrcA` holds the same values the mover's own transpose would give it. Only
/// `live`'s rows of each output tile are computed ([`LiveRows`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_items_reloading(
    outputs: &[OutputTile],
    sems: MatmulSemaphores,
    fmts: SrcFormats,
    fidelity: Fidelity,
    reload: Option<Reload>,
    b_transposed: bool,
    live: LiveRows,
) -> [Vec<Item>; 3] {
    if reload.is_some() {
        assert_eq!(
            outputs.len(),
            1,
            "a K block's continuation is one output tile"
        );
    }
    let priors = reload.map(|r| [r.prior]);
    matmul_items_continuing(
        outputs,
        sems,
        fmts,
        fidelity,
        reload.zip(priors.as_ref()).map(|(r, p)| Continuation {
            priors: p,
            cleared: r.cleared,
            loaded: r.loaded,
        }),
        b_transposed,
        live,
    )
}

/// Each output tile's prior FP32 partial, in `outputs`' order, and the reset/load handshake a
/// continuation reloads them through ([`Reload`] for a block of outputs).
#[derive(Copy, Clone, Debug)]
pub(crate) struct Continuation<'a> {
    pub priors: &'a [u64],
    pub cleared: Semaphore,
    pub loaded: Semaphore,
}

/// What every tile pair of a product shares, built once: the unpack role's prelude and the `Src`
/// configuration it writes (`words`, which a reload writes again after unpacking a prior to
/// `Dst`), the pair's face block on the unpack role (`ub`, 24 words: the replay buffer holds it with
/// [`gpr_step`]), and the pair on the math role.
struct PairCode {
    prelude: Vec<Instruction>,
    words: ConfigWords,
    ub: Vec<Instruction>,
    pair_math: Vec<Item>,
}

/// [`PairCode`] for a product whose first pair is `(a0, b0)` (the bases its configuration names
/// until a retarget moves them).
fn pair_code(
    (a0, b0): (u64, u64),
    fmts: SrcFormats,
    fidelity: Fidelity,
    b_transposed: bool,
    live: LiveRows,
) -> PairCode {
    let mut prelude = src_thread_config();
    let mut words = ConfigWords::new();
    unpack_src_config(
        &mut words,
        Unpacker::SrcA,
        tile_descriptor(fmts.b),
        b0,
        fmts.b_out,
    );
    if b_transposed {
        tt_isa::unpacker::stage_transpose(&mut words, false).expect("a Src unpack");
    }
    unpack_src_config(
        &mut words,
        Unpacker::SrcB,
        tile_descriptor(fmts.a),
        a0,
        fmts.a_out,
    );
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    words
        .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
        .unwrap();
    let throttle = PROBE_THROTTLE.load(std::sync::atomic::Ordering::Relaxed);
    if throttle != 0 {
        words
            .set(thcon::THCON_SEC0_REG2_Throttle_mode, throttle & 0xff)
            .unwrap()
            .set(thcon::THCON_SEC1_REG2_Throttle_mode, throttle >> 8)
            .unwrap();
    }
    prelude.extend(config_program(&words));
    for u in [Unpacker::SrcA, Unpacker::SrcB] {
        prelude.push(crate::datapath::set_adc_x(u, 0, FACE_DATUMS - 1));
    }
    let base = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    let phases = fidelity.phases();
    let (face_rows, halves) = (live.face_rows(), live.halves());
    // Each `MVMUL` names its `Dst` row within its `fi` -- the same for both --
    // and the RWCs' `Dst` holds the `fi`'s base, which only the last `MVMUL`
    // of an `fi` moves ([`math_prelude`], `MATH_AM_*`).
    let mv = |row: u32, am: u32| encode::Mvmul::ZERO.dst_row(row).addr_mod(am);
    // One `MVMUL` group into rows `dst..dst + 8 halves` of the `fi` -- each
    // `SrcB` half it computes, every phase -- all but its last `MVMUL`, which
    // the caller issues at `dst + 8 (halves - 1)` with the group's flips and
    // its end.
    let group = |out: &mut Vec<Instruction>, dst: u32, mut loaded: Banks<Loaded, Loaded>| {
        for half in 0..halves {
            let last = if half + 1 < halves {
                phases
            } else {
                phases - 1
            };
            for phase in 0..last {
                let am = if half + 1 < halves && phase + 1 == phases {
                    MATH_AM_HALF
                } else {
                    MATH_AM_PHASE
                };
                let (i, next) = loaded.mvmul(mv(dst + 8 * half, am)).unwrap();
                out.push(i);
                loaded = next;
            }
        }
        loaded
    };
    let zw = |u: Unpacker, face: u32| {
        let z = encode::Setadczw::ZERO.z0(1).z0_val(face);
        match u {
            Unpacker::SrcA => z.u0(1),
            Unpacker::SrcB => z.u1(1),
        }
        .encode()
        .unwrap()
    };
    // One tile pair's face block on the unpack role, and on the math role one
    // `fi` unit -- its two `k` halves, the second ending by moving the `Dst`
    // base on two faces: the same words for every pair and, on the math role,
    // for both `fi` (the banks start and end empty), so built once.
    // `B`'s face `(k, fj)`: the slot's own, or its `(fj, k)` transposed by the unpacker.
    let b_face = |k: u32, fj: u32| if b_transposed { 2 * fj + k } else { 2 * k + fj };
    let mut ub = Vec::new();
    let mut unit: Option<Vec<Instruction>> = None;
    let mut banks = Banks::after_reset();
    let last_row = 8 * (halves - 1);
    for fi in 0..face_rows {
        let mut mb = Vec::new();
        for k in 0..2u32 {
            // `A` is in0 and goes to `SrcB`: its face `(fi, k)`.
            ub.push(zw(Unpacker::SrcB, 2 * fi + k));
            let (u, with_b) = banks.unpack_b(base).unwrap();
            ub.push(u);
            // `B` is in1 and goes to `SrcA`: its faces `(k, fj)`.
            ub.push(zw(Unpacker::SrcA, b_face(k, 0)));
            let (u, both) = with_b.unpack_a(base).unwrap();
            ub.push(u);
            let loaded = group(&mut mb, 0, both);
            let (m, only_b) = loaded.mvmul_release_a(mv(last_row, MATH_AM_GROUP)).unwrap();
            mb.push(m);
            ub.push(zw(Unpacker::SrcA, b_face(k, 1)));
            let (u, both) = only_b.unpack_a(base).unwrap();
            ub.push(u);
            let loaded = group(&mut mb, 16, both);
            let end = if k == 0 { MATH_AM_GROUP } else { MATH_AM_FI };
            let (m, empty) = loaded.mvmul_release_both(mv(16 + last_row, end)).unwrap();
            mb.push(m);
            banks = empty;
        }
        match &unit {
            None => unit = Some(mb),
            Some(u) => debug_assert_eq!(u, &mb, "both `fi` units are the same words"),
        }
    }
    let _ = banks;
    let unit = unit.expect("two units");
    // A pair on the math role: `Dst`, its base, `SrcB` and the phase back to
    // zero, then the unit for each `fi`.
    let pair_math: Vec<Item> = {
        let reset = encode::Setrwc::ZERO
            .src_b(1)
            .src_b_val(0)
            .dst(1)
            .dst_val(0)
            .fidelity(1)
            .encode()
            .unwrap();
        vec![
            Item::I(reset),
            Item::Repeat {
                times: face_rows,
                body: unit.iter().copied().map(Item::I).collect(),
            },
        ]
    };
    PairCode {
        prelude,
        words,
        ub,
        pair_math,
    }
}

/// [`matmul_items_reloading`] for any number of outputs, each continuing from its own prior when
/// `reload` is given: `Dst` is cleared before each output after the first (the first relies on
/// [`math_prelude`]'s clear, as a K block's one output does), then the output's prior is loaded
/// and its pairs accumulate onto it. One output in `Dst` at a time, as without a reload.
#[allow(clippy::too_many_arguments)]
pub(crate) fn matmul_items_continuing(
    outputs: &[OutputTile],
    sems: MatmulSemaphores,
    fmts: SrcFormats,
    fidelity: Fidelity,
    reload: Option<Continuation<'_>>,
    b_transposed: bool,
    live: LiveRows,
) -> [Vec<Item>; 3] {
    if let Some(r) = reload {
        assert_eq!(
            r.priors.len(),
            outputs.len(),
            "a prior for every output tile"
        );
    }
    assert!(
        outputs.iter().all(|o| !o.pairs.is_empty()),
        "every output tile needs at least one operand pair"
    );
    let (a0, b0) = outputs[0].pairs[0];
    let PairCode {
        prelude,
        words,
        ub,
        pair_math,
    } = pair_code((a0, b0), fmts, fidelity, b_transposed, live);
    fn i(v: impl IntoIterator<Item = Instruction>) -> impl Iterator<Item = Item> {
        v.into_iter().map(Item::I)
    }
    let mut unpack: Vec<Item> = i(prelude).collect();
    let mut math: Vec<Item> = i(math_prelude()).collect();
    let mut pack: Vec<Item> = Vec::new();
    let mut current = (a0, b0);
    for (o, output) in outputs.iter().enumerate() {
        math.extend(i(sync::take(sems.free, Before::MATRIX)));
        // (mode, use_dst32b, addr_mod, imm10): all of `Dst`.
        if let Some(reload) = reload {
            // math_prelude clears Dst before allowing T0 to load the first prior; each later
            // output's clear comes once the packer has read the one before (`sems.free`), and
            // before T0 loads its prior, so no clear erases a loaded prior. Both handshakes
            // finish back at zero.
            let prior = reload.priors[o];
            if o > 0 {
                math.push(Item::I(encode::zeroacc(3, 0, 0, 0).unwrap()));
            }
            math.extend(i(sync::post_after(Unit::Matrix, reload.cleared)));
            math.extend(i(sync::take(reload.loaded, Before::MATRIX)));
            unpack.extend(i(sync::take(reload.cleared, Before::UNPACKER)));
            unpack.extend(i(crate::datapath::thread_config()));
            unpack.extend(i(crate::datapath::clear_unpacker0_adcs()));
            let mut prior_words = ConfigWords::new();
            crate::datapath::tile_unpack_config(&mut prior_words, prior);
            unpack.extend(i(config_program(&prior_words)));
            unpack.extend(i(crate::datapath::unpack_tile_to_dst(prior, 0)));
            unpack.extend(i(sync::post_after(Unit::Unpacker0, reload.loaded)));
            // Restore the complete Src configuration, including descriptor
            // words and ADCs; UnpackToDst leaves different state behind.
            unpack.extend(i(src_thread_config()));
            unpack.extend(i(crate::datapath::clear_unpacker0_adcs()));
            unpack.extend(i(config_program(&words)));
            for u in [Unpacker::SrcA, Unpacker::SrcB] {
                unpack.push(Item::I(crate::datapath::set_adc_x(u, 0, FACE_DATUMS - 1)));
            }
            current = (u64::MAX, u64::MAX);
        } else {
            math.push(Item::I(encode::zeroacc(3, 0, 0, 0).unwrap()));
        }
        let runs = even_runs(&output.pairs);
        match even_strides(&output.pairs) {
            // Pairs evenly spaced in L1 (how the gather stages them): the
            // tiles are stepped by GPR arithmetic, so every pair is the same
            // words -- one loop, which the replay buffer and the MOP take.
            Some((da, db)) => {
                let (a, b) = output.pairs[0];
                unpack.extend(i(gpr_retarget(a, b, da, db)));
                let body: Vec<Item> = i(ub.iter().copied()).chain(i(gpr_step())).collect();
                let n = output.pairs.len() as u32;
                unpack.push(Item::Repeat { times: n, body });
                math.push(Item::Repeat {
                    times: n,
                    body: pair_math.clone(),
                });
                // Where the last step left the bases: one stride past the last pair.
                current = (u64::MAX, u64::MAX);
            }
            // Runs of evenly spaced pairs (a packed `A`'s 16 rows a face, [`packed_a_pairs`]):
            // each run a loop of its own, from bases set once a run.
            None if runs.iter().all(|r| r.1 >= MIN_EVEN_RUN) => {
                for &(start, n, (da, db)) in &runs {
                    let (a, b) = output.pairs[start];
                    unpack.extend(i(gpr_retarget(a, b, da, db)));
                    let body: Vec<Item> = i(ub.iter().copied()).chain(i(gpr_step())).collect();
                    unpack.push(Item::Repeat {
                        times: n as u32,
                        body,
                    });
                }
                math.push(Item::Repeat {
                    times: output.pairs.len() as u32,
                    body: pair_math.clone(),
                });
                current = (u64::MAX, u64::MAX);
            }
            None => {
                for &(a, b) in &output.pairs {
                    if (a, b) != current {
                        unpack.extend(i(retarget(a, b)));
                        current = (a, b);
                    }
                    unpack.push(Item::Shared {
                        key: UNPACK_FACES,
                        body: ub.clone(),
                    });
                    math.extend(pair_math.iter().cloned());
                }
            }
        }
        math.extend(i(sync::post_after(Unit::Matrix, sems.ready)));

        // The packer's whole configuration before the first output; then only where each output
        // goes, the one word that differs (`docs/plans/long-context.md` A3): every `PACR` sees what
        // a whole rewrite a tile gave it, at a few words an output instead of the configuration's
        // dozens, so a block's pack role no longer grows by those with its outputs.
        let mut pw = ConfigWords::new();
        if pack.is_empty() {
            crate::datapath::pack_config(&mut pw, output.out);
        } else {
            pw.set(
                tt_isa::cfg::generated::thcon::THCON_SEC0_REG1_L1_Dest_addr,
                crate::datapath::pack_dest_units(output.out),
            )
            .unwrap();
        }
        pack.extend(i(config_program(&pw)));
        pack.extend(i(sync::take(sems.ready, Before::PACKER)));
        pack.extend(crate::datapath::pack_rows_items(TILE_DST_ROWS));
        pack.extend(i(sync::post_after(Unit::Packer, sems.free)));
    }
    unpack.push(Item::I(
        backend::wait_for_unpacker0(Before::EVERYTHING).unwrap(),
    ));
    unpack.push(Item::I(
        backend::wait_for_unpacker1(Before::EVERYTHING).unwrap(),
    ));
    pack.push(Item::I(
        backend::wait_for_packer(Before::EVERYTHING).unwrap(),
    ));
    [unpack, math, pack]
}

/// A row-major `[rows, cols]` matrix as `tt_metal_32x32` tile images in
/// `format`, and the layout that describes them.
pub fn tilize_f32(
    values: &[f32],
    rows: usize,
    cols: usize,
    format: L1Format,
) -> (Vec<u8>, tt_layout::Layout) {
    use tt_layout::{tilize, HostDtype, Layout, TensorView};
    assert_eq!(values.len(), rows * cols);
    let layout = Layout::tt_metal_32x32(format, HostDtype::F32, [1, rows, cols]).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = TensorView::contiguous(&bytes, HostDtype::F32, [1, rows, cols]);
    (tilize(&view, &layout).unwrap(), layout)
}

/// Bytes in one FP32 `tt_metal_32x32` tile image: the 16-byte header, then
/// 1024 datums.
pub const TILE_IMAGE_BYTES: usize = 16 + 1024 * 4;

/// [`tilize_f32`] for FP32 tiles, as one copy pattern: each 32x32 tile, in
/// row-major tile order, its 16-byte header then four 16x16 faces (top left,
/// top right, bottom left, bottom right), each row-major; zeros past `rows`
/// and `cols`. `tt_layout`'s tilizer handles every format element by element,
/// ~18 us a tile on the host -- more than the card's DMA takes to move it
/// (`silicon_bench_host_dma::session_transfers`). Checked against it
/// (`fast_tilize_is_tt_layouts`).
pub fn tilize_f32_fp32(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(values.len(), rows * cols);
    let tiles = rows.div_ceil(32).max(1) * cols.div_ceil(32).max(1);
    let mut out = vec![0u8; tiles * TILE_IMAGE_BYTES];
    tilize_into(
        |i| values[i].to_bits(),
        rows,
        cols,
        0..tiles,
        &mut out,
        TILE_IMAGE_BYTES,
    );
    out
}

/// Tiles `tiles` of a row-major `[rows, cols]` matrix of 32-bit datums
/// (`get(i)` is datum `i`'s bits), as [`tilize_f32_fp32`] lays each out, into
/// `out` one every `stride` bytes from its start: straight into the pinned
/// buffer a DMA upload sends from, with no copy of its own.
pub fn tilize_into(
    get: impl Fn(usize) -> u32 + Sync,
    rows: usize,
    cols: usize,
    tiles: std::ops::Range<usize>,
    out: &mut [u8],
    stride: usize,
) {
    let n = tiles.len();
    let threads = host_threads(n);
    if threads <= 1 {
        return tilize_serial(&get, rows, cols, tiles, out, stride);
    }
    // Contiguous runs of tiles, each thread its own run of `out`.
    let per = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let get = &get;
        for (k, chunk) in out[..n * stride].chunks_mut(per * stride).enumerate() {
            let first = tiles.start + k * per;
            let run = first..(first + per).min(tiles.end);
            scope.spawn(move || tilize_serial(get, rows, cols, run, chunk, stride));
        }
    });
}

/// Threads to split `tiles` tiles of host tile conversion over: one per
/// [`HOST_TILES_PER_THREAD`], up to the host's cores. A tile is ~0.5 us of
/// conversion, a thread ~20 to spawn: on card 0's host, 64 tiles a thread
/// left a 1024-tile upload's tilize at 0.51 ms, 256 took it to 0.28.
fn host_threads(tiles: usize) -> usize {
    if tiles < 2 * HOST_TILES_PER_THREAD {
        return 1;
    }
    // Asked once: `available_parallelism` reads the cgroup's CPU quota
    // files on every call, ~40 us -- more than a small transfer's DMA.
    static CORES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cores = *CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    (tiles / HOST_TILES_PER_THREAD).clamp(1, cores.min(16))
}

/// Fewest tiles a thread of host tile conversion is given.
const HOST_TILES_PER_THREAD: usize = 256;

fn tilize_serial(
    get: &impl Fn(usize) -> u32,
    rows: usize,
    cols: usize,
    tiles: std::ops::Range<usize>,
    out: &mut [u8],
    stride: usize,
) {
    let ct = cols.div_ceil(32).max(1);
    let header = fp32_tile_header();
    for (n, t) in tiles.enumerate() {
        let image = &mut out[n * stride..][..TILE_IMAGE_BYTES];
        let (i, j) = (t / ct, t % ct);
        image[..16].copy_from_slice(&header);
        for face in 0..4 {
            let (fr, fc) = (face / 2, face % 2);
            let col0 = 32 * j + 16 * fc;
            let n = cols.saturating_sub(col0).min(16);
            for r in 0..16 {
                let row = 32 * i + 16 * fr + r;
                let at = 16 + (face * 256 + r * 16) * 4;
                let dst = &mut image[at..at + 64];
                if row >= rows || n == 0 {
                    dst.fill(0);
                    continue;
                }
                let base = row * cols + col0;
                for k in 0..16 {
                    let v = if k < n { get(base + k) } else { 0 };
                    dst[4 * k..4 * k + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
}

/// The header `tt_layout` gives every FP32 tile.
fn fp32_tile_header() -> [u8; 16] {
    use std::sync::OnceLock;
    static HEADER: OnceLock<[u8; 16]> = OnceLock::new();
    *HEADER.get_or_init(|| {
        let (image, _) = tilize_f32(&[0.0; 1024], 32, 32, L1Format::Fp32);
        image[..16].try_into().unwrap()
    })
}

/// Row-major `[rows, cols]` FP32 values from packed output tiles: each tile's
/// 1024 datums as [`tile_roles`] packs them, with no header, one after another
/// in row-major tile order -- [`tilize_f32_fp32`]'s faces, read back.
pub fn detilize_packed(packed: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let tiles = rows.div_ceil(32).max(1) * cols.div_ceil(32).max(1);
    assert!(packed.len() >= tiles * 4096, "not enough packed tiles");
    let mut out = vec![0f32; rows * cols];
    detilize_from(packed, 4096, 0, rows, cols, 0..tiles, &mut out);
    out
}

/// Tiles `tiles` of a `[rows, cols]` matrix into row-major `out`, from
/// `src`: tile `first + n`'s 1024 datums at `n * stride + skip` -- straight
/// out of the pinned buffer a DMA download lands in.
pub fn detilize_from(
    src: &[u8],
    stride: usize,
    skip: usize,
    rows: usize,
    cols: usize,
    tiles: std::ops::Range<usize>,
    out: &mut [f32],
) {
    let ct = cols.div_ceil(32).max(1);
    let threads = host_threads(tiles.len());
    // Split by whole tile rows, so each thread writes its own rows of `out`.
    let whole_rows = tiles.start % ct == 0 && tiles.end % ct == 0;
    if threads <= 1 || !whole_rows {
        return detilize_rows(src, stride, skip, rows, cols, tiles, out, 0);
    }
    let (r0, r1) = (tiles.start / ct, tiles.end / ct);
    let per = (r1 - r0).div_ceil(threads);
    let mine = &mut out[(32 * r0).min(rows) * cols..(32 * r1).min(rows) * cols];
    std::thread::scope(|scope| {
        for (k, chunk) in mine.chunks_mut(32 * per * cols).enumerate() {
            let (a, b) = (r0 + k * per, (r0 + (k + 1) * per).min(r1));
            let run = a * ct..b * ct;
            let src = &src[(run.start - tiles.start) * stride..];
            scope.spawn(move || detilize_rows(src, stride, skip, rows, cols, run, chunk, 32 * a));
        }
    });
}

/// [`detilize_from`] on one thread, into `out` holding rows from `row0`.
#[allow(clippy::too_many_arguments)]
fn detilize_rows(
    src: &[u8],
    stride: usize,
    skip: usize,
    rows: usize,
    cols: usize,
    tiles: std::ops::Range<usize>,
    out: &mut [f32],
    row0: usize,
) {
    let ct = cols.div_ceil(32).max(1);
    for (n, t) in tiles.enumerate() {
        let (i, j) = (t / ct, t % ct);
        let tile = &src[n * stride + skip..][..4096];
        for face in 0..4 {
            let (fr, fc) = (face / 2, face % 2);
            let col0 = 32 * j + 16 * fc;
            if col0 >= cols {
                continue;
            }
            let m = (cols - col0).min(16);
            for r in 0..16 {
                let row = 32 * i + 16 * fr + r;
                if row >= rows {
                    break;
                }
                let at = (face * 256 + r * 16) * 4;
                for (k, v) in out[(row - row0) * cols + col0..][..m]
                    .iter_mut()
                    .enumerate()
                {
                    let b = &tile[at + 4 * k..at + 4 * k + 4];
                    *v = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        }
    }
}

/// [`detilize_packed`] through `tt_layout`'s detilizer: the oracle the fast
/// one is checked against.
#[cfg(test)]
fn detilize_packed_generic(packed: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    use tt_layout::{detilize, HostDtype, Layout, TensorViewMut};
    let layout = Layout::tt_metal_32x32(L1Format::Fp32, HostDtype::F32, [1, rows, cols]).unwrap();
    let mut images = Vec::with_capacity(layout.total_bytes());
    for tile in packed.chunks_exact(1024 * 4).take(layout.grid_tiles()) {
        images.extend_from_slice(&[0u8; 16]);
        images.extend_from_slice(tile);
    }
    assert_eq!(
        images.len(),
        layout.total_bytes(),
        "not enough packed tiles"
    );
    let mut out = vec![0u8; rows * cols * 4];
    let mut view = TensorViewMut::contiguous(&mut out, HostDtype::F32, [1, rows, cols]);
    detilize(&images, &layout, &mut view).unwrap();
    out.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Where [`stage_matmul`] puts things in L1 on the host-staged path: `A`'s
/// tiles from here, then `B`'s, then the packed output tiles from
/// [`MATMUL_OUT`]. The start of the data arena (`tt_isa::l1::DATA`); the
/// GDDR path's layout is planned there instead ([`plan_layout_in`] with
/// [`Staging::Slots`], through `crate::l1`).
pub const MATMUL_STAGE: u64 = tt_isa::l1::DATA.base;
/// Where the packed output tiles go on the host-staged path.
pub const MATMUL_OUT: u64 = 0x8_0000;
const _: () = assert!(MATMUL_OUT > MATMUL_STAGE && MATMUL_OUT < tt_isa::l1::DATA.end);

/// A matmul `C[m, n] = A[m, k] @ B[k, n]` laid out for the device: both
/// operands tiled and padded by `tt_layout`, and the output tiles planned.
pub struct StagedMatmul {
    pub m: usize,
    pub n: usize,
    pub a: Vec<u8>,
    pub b: Vec<u8>,
    pub b_at: u64,
    pub outputs: Vec<OutputTile>,
    pub sems: MatmulSemaphores,
    pub init: Vec<crate::runtime::SemaphoreInit>,
}

impl StagedMatmul {
    /// Bytes of packed output: one 1024-datum FP32 tile per output tile.
    pub fn out_bytes(&self) -> usize {
        self.outputs.len() * 1024 * 4
    }
}

/// Bytes of one `tt_metal_32x32` tile image in `format`, header included.
pub fn tile_image_bytes(format: L1Format) -> u64 {
    tt_layout::Layout::tt_metal_32x32(format, tt_layout::HostDtype::F32, [1, 32, 32])
        .expect("one 32x32 tile is a valid layout")
        .image()
        .total_bytes() as u64
}

/// Where a matmul run's operands and outputs go in L1, and which semaphores
/// it hands `Dst` over with.
#[derive(Clone, Debug)]
pub struct Layout {
    /// `A`'s first byte.
    pub a_at: u64,
    /// `B`'s first byte.
    pub b_at: u64,
    /// One per output tile, row-major.
    pub outputs: Vec<OutputTile>,
    pub sems: MatmulSemaphores,
    /// What a concurrent run of it initialises: every semaphore of its plan.
    pub init: Vec<crate::runtime::SemaphoreInit>,
}

impl Layout {
    /// The same run with its operands and outputs `by` bytes further on and
    /// its semaphores where they were: the other half of a double-buffered
    /// pair ([`Staging::SlotsHalf`], [`half`]). Two runs that share
    /// semaphores never overlap -- the second starts once the first has
    /// finished, and each leaves its semaphores as it found them.
    pub fn shifted(&self, by: u64) -> Layout {
        Layout {
            a_at: self.a_at + by,
            b_at: self.b_at + by,
            outputs: self
                .outputs
                .iter()
                .map(|o| OutputTile {
                    pairs: o.pairs.iter().map(|&(a, b)| (a + by, b + by)).collect(),
                    out: o.out + by,
                })
                .collect(),
            sems: self.sems,
            init: self.init.clone(),
        }
    }
}

/// Where a `[mt, kt] @ [kt, nt]`-tile matmul's operands and outputs go in L1,
/// without the data: `B`'s first byte, one [`OutputTile`] per output tile, and
/// its semaphores.
///
/// Refuses a shape whose operands would overrun [`MATMUL_OUT`] or whose output
/// would overrun the mailbox, as [`RunError::DoesNotFit`](crate::runtime::RunError).
pub fn plan_layout(
    tiles: [usize; 3],
    in_fmt: L1Format,
) -> Result<Layout, crate::runtime::RunError> {
    plan_layout_in(tiles, in_fmt, Staging::Host)
}

/// How a run's tiles sit in L1.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Staging {
    /// Tile images back to back, as the host tilizes them, and packed output
    /// tiles back to back: [`plan_layout`].
    Host,
    /// One `tt_isa::dm::TILE_SLOT` per tile, operands and outputs alike, with
    /// each output's datums at `TILE_DATA` into its slot -- so the data mover
    /// can copy any tile between GDDR and L1 under the C64 rule, and an output
    /// slot is an operand slot. FP32 in L1 only.
    Slots,
    /// [`Staging::Slots`] in the first half of the data arena, so that a
    /// second block can be staged in the other half ([`Layout::shifted`] by
    /// [`half`]) while this one computes (checklist 9.15).
    SlotsHalf,
}

/// How far the second half of the data arena ([`crate::l1::arena`]) is from the first, for
/// [`Staging::SlotsHalf`]: half the arena, on a slot-friendly boundary.
pub fn half() -> u64 {
    (crate::l1::arena().len() / 2) / 4096 * 4096
}

/// The first half of the data arena ([`Staging::SlotsHalf`]).
pub(crate) fn half_arena() -> tt_isa::l1::Region {
    tt_isa::l1::Region {
        name: "first half of the data arena",
        base: tt_isa::l1::DATA.base,
        end: tt_isa::l1::DATA.base + half(),
    }
}

/// [`plan_layout_in`] in slots ([`Staging::Slots`] or [`Staging::SlotsHalf`]) for a product
/// whose `A` is read packed (`crate::tensor`'s packed `A`): its ring holds the `a_slots` slots
/// the packed rows span, not `mt * kt` (at `K` = 2048, 3 slots against 64: 254 KB of a unit's
/// arena freed). Under [`Staging::Slots`], `B`'s ring sits at the arena's end, `A` and the
/// outputs at its start ([`matmul_requirements_top_b`]), so a prefetch of `B` lands above what the
/// ops before the product plan. The pairs name `A`'s unpacked slots as [`plan_layout_in`]'s do;
/// the caller relays them onto the packed rows.
pub fn plan_layout_packed_a(
    tiles: [usize; 3],
    in_fmt: L1Format,
    staging: Staging,
    a_slots: usize,
    b_slot: u64,
) -> Result<Layout, crate::runtime::RunError> {
    assert!(staging != Staging::Host, "a packed A is read into slots");
    planned_slots(
        tiles,
        in_fmt,
        staging,
        a_slots,
        staging == Staging::Slots,
        b_slot,
    )
}

/// [`plan_layout`] for either [`Staging`].
pub fn plan_layout_in(
    tiles: [usize; 3],
    in_fmt: L1Format,
    staging: Staging,
) -> Result<Layout, crate::runtime::RunError> {
    plan_layout_route(tiles, in_fmt, staging, tt_isa::dm::TILE_SLOT)
}

/// [`plan_layout_in`] with a slot staging's `B` tiles `b_slot` bytes apart ([`SrcRoute::b_slot`]).
/// [`Staging::Host`] lays both operands out at `in_fmt`'s image, whatever `b_slot`.
pub fn plan_layout_route(
    [mt, kt, nt]: [usize; 3],
    in_fmt: L1Format,
    staging: Staging,
    b_slot: u64,
) -> Result<Layout, crate::runtime::RunError> {
    use crate::runtime::RunError;
    match staging {
        Staging::Slots | Staging::SlotsHalf => {
            return planned_slots([mt, kt, nt], in_fmt, staging, mt * kt, false, b_slot)
        }
        Staging::Host => {}
    }
    let (img, align, out_stride, out_skip) = match staging {
        Staging::Host => (tile_image_bytes(in_fmt), 16, 1024 * 4, 0),
        Staging::Slots | Staging::SlotsHalf => unreachable!("planned by plan_slots"),
    };
    let a_bytes = (mt * kt) as u64 * img;
    let b_at = (MATMUL_STAGE + a_bytes).next_multiple_of(align);
    let b_end = b_at + (kt * nt) as u64 * img;
    if b_end > MATMUL_OUT {
        return Err(RunError::DoesNotFit {
            what: "the operand tiles",
            bytes: b_end - MATMUL_STAGE,
            limit: MATMUL_OUT - MATMUL_STAGE,
        });
    }
    let out_bytes = (mt * nt) as u64 * out_stride;
    // The arena's end: the mailboxes, or a planned capture's persistent region (`crate::l1::arena`).
    let end = crate::l1::arena().end;
    if MATMUL_OUT + out_bytes > end {
        return Err(RunError::DoesNotFit {
            what: "the output tiles",
            bytes: out_bytes,
            limit: end.saturating_sub(MATMUL_OUT),
        });
    }
    crate::l1::note_planned_top(MATMUL_OUT + out_bytes);
    let mut outputs = Vec::with_capacity(mt * nt);
    for i in 0..mt {
        for j in 0..nt {
            let pairs = (0..kt)
                .map(|kk| {
                    (
                        MATMUL_STAGE + (i * kt + kk) as u64 * img,
                        b_at + (kk * nt + j) as u64 * img,
                    )
                })
                .collect();
            outputs.push(OutputTile {
                pairs,
                out: MATMUL_OUT + (i * nt + j) as u64 * out_stride + out_skip,
            });
        }
    }
    let (sems, init) = MatmulSemaphores::alone();
    Ok(Layout {
        a_at: MATMUL_STAGE,
        b_at,
        outputs,
        sems,
        init,
    })
}

/// The GDDR path's L1 layout ([`Staging::Slots`]): its kernel's needs as
/// `crate::l1` requirements -- `A`'s and `B`'s tiles, rings from the mover to
/// the unpacker, and the output tiles, a ring from the packer back to the
/// mover -- planned in the data arena. One stage, so all three live at once.
pub fn matmul_requirements([mt, kt, nt]: [usize; 3]) -> MatmulBuffers {
    matmul_requirements_with([mt, kt, nt], mt * kt)
}

/// [`matmul_requirements`] with `a_slots` slots for `A` (a packed `A`'s few,
/// [`packed_a_pairs`]).
pub fn matmul_requirements_with(tiles: [usize; 3], a_slots: usize) -> MatmulBuffers {
    matmul_requirements_placed(tiles, a_slots, None, tt_isa::dm::TILE_SLOT)
}

/// [`matmul_requirements_with`] with `B`'s ring at the end of `arena_bytes` of arena: `A` and the
/// outputs first, then a gap that fills the rest, then `B`, so first fit puts `B` against the
/// arena's end (`docs/plans/decode-to-bandwidth.md` Phase 8: the ops between two products plan
/// low, and a prefetch of the next product's `B` lands above them). `None` if the three do not
/// fit, as the plain order would not.
pub fn matmul_requirements_top_b(
    tiles: [usize; 3],
    a_slots: usize,
    arena_bytes: u64,
    b_slot: u64,
) -> Option<MatmulBuffers> {
    let gap = arena_bytes.checked_sub(slot_bytes(tiles, a_slots, b_slot))?;
    Some(matmul_requirements_placed(
        tiles,
        a_slots,
        Some(gap),
        b_slot,
    ))
}

/// The bytes of a slot staging's rings: `a_slots` of `A` and the outputs' FP32 slots, and `B`'s
/// `kt * nt` tiles `b_slot` apart.
fn slot_bytes([mt, kt, nt]: [usize; 3], a_slots: usize, b_slot: u64) -> u64 {
    (a_slots + mt * nt) as u64 * tt_isa::dm::TILE_SLOT + (kt * nt) as u64 * b_slot
}

fn matmul_requirements_placed(
    [mt, kt, nt]: [usize; 3],
    a_slots: usize,
    gap: Option<u64>,
    b_slot: u64,
) -> MatmulBuffers {
    use crate::l1::{Endpoint, Requirements};
    use tt_isa::dm::TILE_SLOT;
    let mut req = Requirements::new(1);
    let ring = |req: &mut Requirements, name, page: u64, pages: usize, from, to| {
        // C64: a slot is read from GDDR under the 64-byte congruence rule
        // (divergence row 64), and a slot is a whole number of 64-byte units.
        req.cb(
            name,
            page,
            pages as u32,
            tt_isa::dram::ALIGN,
            from,
            to,
            0..1,
        )
    };
    let a = ring(
        &mut req,
        "matmul A",
        TILE_SLOT,
        a_slots,
        Endpoint::Mover,
        Endpoint::Unpack,
    );
    let b_ring = |req: &mut Requirements| {
        ring(
            req,
            "matmul B",
            b_slot,
            kt * nt,
            Endpoint::Mover,
            Endpoint::Unpack,
        )
    };
    let out_ring = |req: &mut Requirements| {
        ring(
            req,
            "matmul out",
            TILE_SLOT,
            mt * nt,
            Endpoint::Pack,
            Endpoint::Mover,
        )
    };
    let (b, out) = match gap {
        None => {
            let b = b_ring(&mut req);
            (b, out_ring(&mut req))
        }
        Some(gap) => {
            let out = out_ring(&mut req);
            if gap > 0 {
                req.slack("matmul gap", gap, tt_isa::dram::ALIGN, 0..1);
            }
            (b_ring(&mut req), out)
        }
    };
    let sems = MatmulSemaphores::declare(&mut req, 0..1);
    MatmulBuffers {
        req,
        a,
        b,
        out,
        sems,
    }
}

/// A matmul's declared buffers ([`matmul_requirements`]).
pub struct MatmulBuffers {
    pub req: crate::l1::Requirements,
    pub a: crate::l1::Buf,
    pub b: crate::l1::Buf,
    pub out: crate::l1::Buf,
    /// `Dst` ready and free ([`MatmulSemaphores`]).
    pub sems: (crate::l1::Sem, crate::l1::Sem),
}

/// [`plan_slots`] for a slot staging, planned once per process for each
/// shape: an op of many same-shaped blocks plans each of them, and the L1
/// planner was most of the host's time for a pipelined 1024^3 matmul's 512
/// blocks.
fn planned_slots(
    tiles: [usize; 3],
    in_fmt: L1Format,
    staging: Staging,
    a_slots: usize,
    top_b: bool,
    b_slot: u64,
) -> Result<Layout, crate::runtime::RunError> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = ([usize; 3], L1Format, Staging, usize, bool, u64, u64, u64);
    // Each layout with what its plan reached and needed, noted again on every hit
    // (`crate::l1::note_planned`); by the arena's end too, and the end `B`'s ring sits against
    // (`crate::l1::ring_end`).
    static CACHE: OnceLock<Mutex<HashMap<Key, (Layout, crate::l1::Planned)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (
        tiles,
        in_fmt,
        staging,
        a_slots,
        top_b,
        crate::l1::arena().end,
        crate::l1::ring_end(),
        b_slot,
    );
    if let Some((l, p)) = cache.lock().unwrap().get(&key) {
        crate::l1::note_planned(*p);
        return Ok(l.clone());
    }
    let arena = match staging {
        Staging::SlotsHalf => half_arena(),
        // `B` against the ring's end, `A` and the outputs at the arena's start.
        _ if top_b => tt_isa::l1::Region {
            end: crate::l1::ring_end(),
            ..crate::l1::arena()
        },
        _ => crate::l1::arena(),
    };
    let before = crate::l1::take_planned();
    let planned = plan_slots(tiles, in_fmt, arena, a_slots, top_b, b_slot);
    let p = crate::l1::take_planned();
    crate::l1::note_planned(before);
    crate::l1::note_planned(p);
    let l = planned?;
    cache.lock().unwrap().insert(key, (l.clone(), p));
    Ok(l)
}

fn plan_slots(
    [mt, kt, nt]: [usize; 3],
    in_fmt: L1Format,
    arena: tt_isa::l1::Region,
    a_slots: usize,
    top_b: bool,
    b_slot: u64,
) -> Result<Layout, crate::runtime::RunError> {
    use crate::l1::PlanError;
    use crate::runtime::RunError;
    use tt_isa::dm::{TILE_DATA, TILE_SLOT};
    assert!(
        tile_image_bytes(in_fmt) <= TILE_DATA + 4096,
        "slot staging must fit the operand image"
    );
    let m = match top_b {
        true => matmul_requirements_top_b([mt, kt, nt], a_slots, arena.len(), b_slot).ok_or(
            RunError::DoesNotFit {
                what: "the operand tiles",
                bytes: slot_bytes([mt, kt, nt], a_slots, b_slot),
                limit: arena.len(),
            },
        )?,
        false => matmul_requirements_placed([mt, kt, nt], a_slots, None, b_slot),
    };
    let plan = m.req.plan(arena).map_err(|e| match e {
        PlanError::DoesNotFit { name, bytes, arena } => RunError::DoesNotFit {
            what: name,
            bytes,
            limit: arena,
        },
        // A K-block too long for one circular buffer's page counters is too large,
        // not malformed: report it as not fitting, so the chunk search
        // (`plan_uncached`) tries a shorter K-block as it does for L1.
        PlanError::TooManyPages { name, pages, max } => RunError::DoesNotFit {
            what: name,
            bytes: pages as u64 * TILE_SLOT,
            limit: max as u64 * TILE_SLOT,
        },
        e => panic!("the matmul's own requirements are invalid: {e}"),
    })?;
    let (a_at, b_at, out_at) = (plan.addr(m.a), plan.addr(m.b), plan.addr(m.out));
    let mut outputs = Vec::with_capacity(mt * nt);
    for i in 0..mt {
        for j in 0..nt {
            let pairs = (0..kt)
                .map(|kk| {
                    (
                        a_at + (i * kt + kk) as u64 * TILE_SLOT,
                        b_at + (kk * nt + j) as u64 * b_slot,
                    )
                })
                .collect();
            outputs.push(OutputTile {
                pairs,
                out: out_at + (i * nt + j) as u64 * TILE_SLOT + TILE_DATA,
            });
        }
    }
    Ok(Layout {
        a_at,
        b_at,
        outputs,
        sems: MatmulSemaphores::planned(&plan, m.sems),
        init: plan.semaphore_init(),
    })
}

/// Tile `a` and `b` (row-major) in `in_fmt` and plan one [`OutputTile`] per
/// output tile, row-major, each accumulating its `k / 32` tile pairs in order
/// ([`plan_layout`]).
///
/// Padding is `tt_layout`'s zero, which is the identity for the accumulation.
pub fn stage_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    in_fmt: L1Format,
) -> Result<StagedMatmul, crate::runtime::RunError> {
    stage_matmul_with(a, b, [m, k, n], (in_fmt, in_fmt))
}

/// [`stage_matmul`] with `A` tiled in `formats.0` and `B` in `formats.1`, each `B` tile at the
/// stride of an `A` one (`formats.1`'s image no larger), as [`plan_layout`] places them.
pub fn stage_matmul_with(
    a: &[f32],
    b: &[f32],
    [m, k, n]: [usize; 3],
    (a_fmt, b_fmt): (L1Format, L1Format),
) -> Result<StagedMatmul, crate::runtime::RunError> {
    let tiles = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)];
    let Layout {
        b_at,
        outputs,
        a_at: _,
        sems,
        init,
    } = plan_layout(tiles, a_fmt)?;
    let (ta, la) = tilize_f32(a, m, k, a_fmt);
    let (mut tb, lb) = tilize_f32(b, k, n, b_fmt);
    assert_eq!(la.tiles_per_matrix(), tiles[0] * tiles[1]);
    assert_eq!(lb.tiles_per_matrix(), tiles[1] * tiles[2]);
    assert_eq!(la.image().total_bytes() as u64, tile_image_bytes(a_fmt));
    let (img_a, img_b) = (
        tile_image_bytes(a_fmt) as usize,
        tile_image_bytes(b_fmt) as usize,
    );
    assert!(img_b <= img_a, "B's tile image fits A's stride");
    if img_b < img_a {
        let mut spread = vec![0u8; tb.len() / img_b * img_a];
        for (t, image) in tb.chunks_exact(img_b).enumerate() {
            spread[t * img_a..t * img_a + img_b].copy_from_slice(image);
        }
        tb = spread;
    }
    debug_assert_eq!(b_at, (MATMUL_STAGE + ta.len() as u64).next_multiple_of(16));
    Ok(StagedMatmul {
        m,
        n,
        a: ta,
        b: tb,
        b_at,
        outputs,
        sems,
        init,
    })
}

/// How operands reach `Src`, and so the precision the Matrix Unit multiplies
/// at (`UNPACR_Regular.md:495-520`; codes measured, divergence rows G and H).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SrcRoute {
    /// FP32 in L1, truncated to TF32 by the unpacker.
    Tf32FromFp32,
    /// FP32 in L1, converted to BF16 by the unpacker.
    Bf16FromFp32,
    /// BF16 in L1, moved as is.
    Bf16FromBf16,
    /// Packed block-float operands moved directly to Src.
    Bfp(crate::bfp::BfpFormat),
    /// `A` as [`SrcRoute::Tf32FromFp32`]; `B` BF16 in L1 (a weight stored in half the bytes),
    /// moved as is ([`BF16_CODE`]; BF16 in with TF32 out is undefined, which ttsim refuses). A
    /// BF16 value in `Src` is a TF32 one whose low three mantissa bits are clear, and the product
    /// is [`SrcRoute::Tf32FromFp32`]'s over the same values bit for bit, specials and every
    /// fidelity included (gate `step232`; `docs/plans/bf16-weight-storage.md`).
    Tf32WithBf16B,
}

/// The L1 format and `OutDataFormat` of each operand of a product: `A` (in0, into `SrcB`) and
/// `B` (in1, into `SrcA`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SrcFormats {
    pub a: L1Format,
    pub a_out: u32,
    pub b: L1Format,
    pub b_out: u32,
}

impl SrcFormats {
    /// Both operands in `in_fmt`, unpacked under `out`.
    pub const fn same(in_fmt: L1Format, out: u32) -> Self {
        SrcFormats {
            a: in_fmt,
            a_out: out,
            b: in_fmt,
            b_out: out,
        }
    }
}

impl SrcRoute {
    /// The L1 format the operands are staged in, and the `OutDataFormat` code.
    pub const fn formats(self) -> (L1Format, u32) {
        match self {
            SrcRoute::Tf32FromFp32 => (L1Format::Fp32, TF32_CODE),
            SrcRoute::Bf16FromFp32 => (L1Format::Fp32, BF16_CODE),
            SrcRoute::Bf16FromBf16 => (L1Format::Bf16, BF16_CODE),
            SrcRoute::Bfp(format) => (
                format.l1_format(),
                match format {
                    crate::bfp::BfpFormat::Bfp8 => 6,
                    crate::bfp::BfpFormat::Bfp4 => 7,
                    crate::bfp::BfpFormat::Bfp2 => 15,
                },
            ),
            SrcRoute::Tf32WithBf16B => (L1Format::Fp32, TF32_CODE),
        }
    }

    /// Each operand's L1 format and `OutDataFormat`. [`SrcRoute::formats`] is `A`'s, and the
    /// slot size of every L1 plan's `A` and outputs; `B`'s is [`SrcRoute::b_slot`].
    pub const fn src_formats(self) -> SrcFormats {
        match self {
            SrcRoute::Tf32WithBf16B => SrcFormats {
                a: L1Format::Fp32,
                a_out: TF32_CODE,
                b: L1Format::Bf16,
                b_out: BF16_CODE,
            },
            _ => {
                let (in_fmt, out) = self.formats();
                SrcFormats::same(in_fmt, out)
            }
        }
    }

    /// Whether `B` is BF16 in L1 while `A` is FP32 ([`SrcRoute::Tf32WithBf16B`]).
    pub const fn bf16_b(self) -> bool {
        matches!(self, SrcRoute::Tf32WithBf16B)
    }

    /// The bytes a `B` tile takes in a GDDR-staged plan's L1: a BF16 `B`'s own 2,112-byte slot
    /// ([`tt_isa::dm::record::GATHER_B_BF16`] reads it there), so a ring holds twice the tiles
    /// (`docs/plans/bf16-weight-storage.md` A5b); [`tt_isa::dm::TILE_SLOT`] otherwise.
    pub const fn b_slot(self) -> u64 {
        if self.bf16_b() {
            tt_isa::dm::BF16_TILE_SLOT
        } else {
            tt_isa::dm::TILE_SLOT
        }
    }
}

/// `C[m, n] = A[m, k] @ B[k, n]` for row-major `f32` operands, on the Tensix
/// tile at `tile`, in one run: tiled and zero-padded by `tt_layout`, multiplied
/// through `route` at `fidelity`, accumulated in FP32 `Dst`, packed as FP32 and
/// de-tiled.
///
/// A shape too large for one run is refused ([`RunError::DoesNotFit`],
/// [`RunError::ProgramTooLong`]); [`plan`] and [`matmul_chunked`] split one.
///
/// [`RunError::DoesNotFit`]: crate::runtime::RunError::DoesNotFit
/// [`RunError::ProgramTooLong`]: crate::runtime::RunError::ProgramTooLong
#[allow(clippy::too_many_arguments)]
pub fn matmul<T: tt_device::Transport, N: tt_isa::noc::NocId>(
    dev: &mut tt_device::Device<T>,
    tile: tt_isa::noc::NocCoord<N>,
    images: &crate::runtime::RoleImages<'_>,
    a: &[f32],
    b: &[f32],
    mkn: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    budget: u64,
) -> Result<Vec<f32>, crate::runtime::RunError> {
    matmul_with(a, b, mkn, route, fidelity, |kernel| {
        crate::runtime::run(dev, tile, images, kernel, budget)
    })
}

/// [`matmul`], with the kernel handed to `run` -- [`crate::runtime::run`], or a
/// [`crate::runtime::Resident`]'s `run` -- instead of a fixed runner.
pub fn matmul_with(
    a: &[f32],
    b: &[f32],
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    run: impl FnOnce(
        &crate::runtime::Kernel<'_>,
    ) -> Result<crate::runtime::Outcome, crate::runtime::RunError>,
) -> Result<Vec<f32>, crate::runtime::RunError> {
    use crate::runtime::{Kernel, Schedule};
    let fmts = route.src_formats();
    let staged = stage_matmul_with(a, b, [m, k, n], (fmts.a, fmts.b))?;
    let tiles = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)];
    let roles = programs(
        (tiles, Staging::Host),
        route,
        fidelity,
        staged.sems,
        LiveRows::All,
        || matmul_roles_with(&staged.outputs, staged.sems, fmts, fidelity),
    );
    let [unpack, math, pack] = &*roles;
    let stage = [
        (MATMUL_STAGE, staged.a.as_slice()),
        (staged.b_at, staged.b.as_slice()),
    ];
    let read_back = [(MATMUL_OUT, staged.out_bytes())];
    let kernel = Kernel {
        stage: &stage,
        read_back: &read_back,
        // Every run leaves its semaphores as it found them (`MatmulSemaphores`).
        restores_semaphores: true,
        ..Kernel::new([unpack, math, pack], Schedule::Concurrent(&staged.init))
    };
    let out = run(&kernel)?;
    Ok(detilize_packed(&out.l1[0], m, n))
}

// --- Chunking -----------------------------------------------------------------

/// The shape of every run a large matmul is split into, in 32x32 tiles:
/// `[mc, kc, nc]`. The last chunk along each axis may be smaller.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ChunkShape {
    pub tiles: [usize; 3],
}

/// Whether a `[mc, kc, nc]`-tile run fits: its L1 layout ([`plan_layout`]) and
/// every role program in its slot, measured by building the programs rather than
/// by a formula that could drift from them.
pub fn chunk_fits(tiles: [usize; 3], route: SrcRoute, fidelity: Fidelity) -> bool {
    chunk_fits_in(tiles, route, fidelity, Staging::Host)
}

/// How many columns a job of a product with a BF16 `B` (compact rings, [`SrcRoute::b_slot`]) is
/// planned to take (`docs/plans/bf16-weight-storage.md` A5b). Its rings are compact whichever.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Bf16Plans {
    /// An FP32 `B`'s columns: the same jobs in about half the bytes.
    F32,
    /// As many columns as fit the bytes an FP32 `B`'s ring takes (3 against 2 at `K` = 2048): more
    /// of a product read ahead a unit, its ring no lower in `DATA`.
    Ring,
    /// As many as fit the arena (twice an FP32 `B`'s at `K` = 2048). Measured slower on the 1.7B:
    /// its rings reach below the ops between products, so busy units no longer read them ahead.
    Wide,
}

/// [`Bf16Plans`], process-wide: `TT_BF16_PLANS=f32|ring|wide` at first use, [`set_bf16_plans`].
static BF16_PLANS: std::sync::OnceLock<std::sync::atomic::AtomicU8> = std::sync::OnceLock::new();

fn bf16_plans_cell() -> &'static std::sync::atomic::AtomicU8 {
    BF16_PLANS.get_or_init(|| {
        let mode = match std::env::var("TT_BF16_PLANS").as_deref() {
            Ok("f32") | Err(_) => Bf16Plans::F32,
            Ok("ring") => Bf16Plans::Ring,
            Ok("wide") => Bf16Plans::Wide,
            Ok(other) => panic!("TT_BF16_PLANS={other}: f32, ring or wide"),
        };
        std::sync::atomic::AtomicU8::new(mode as u8)
    })
}

/// The current [`Bf16Plans`].
pub fn bf16_plans() -> Bf16Plans {
    match bf16_plans_cell().load(std::sync::atomic::Ordering::Relaxed) {
        0 => Bf16Plans::F32,
        1 => Bf16Plans::Ring,
        _ => Bf16Plans::Wide,
    }
}

/// Set [`Bf16Plans`], process-wide. The plan caches key on it.
pub fn set_bf16_plans(mode: Bf16Plans) {
    bf16_plans_cell().store(mode as u8, std::sync::atomic::Ordering::Relaxed);
}

/// The `B` slot a product's width is first planned with ([`plan_uncached`]): [`SrcRoute::b_slot`]
/// under [`Bf16Plans::Wide`], else an FP32 `B`'s.
fn planned_b_slot(route: SrcRoute) -> u64 {
    match bf16_plans() {
        Bf16Plans::Wide => route.b_slot(),
        _ => tt_isa::dm::TILE_SLOT,
    }
}

/// [`chunk_fits`] for either [`Staging`]. Its probe plans `B` at [`planned_b_slot`].
pub fn chunk_fits_in(
    tiles: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    staging: Staging,
) -> bool {
    chunk_fits_at(tiles, route, fidelity, staging, planned_b_slot(route))
}

/// [`chunk_fits_in`] with `B`'s tiles `b_slot` apart.
fn chunk_fits_at(
    tiles: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    staging: Staging,
    b_slot: u64,
) -> bool {
    let in_fmt = route.formats().0;
    if staging != Staging::Host && !matches!(in_fmt, L1Format::Fp32 | L1Format::Bf16) {
        return false;
    }
    let Ok(layout) = plan_layout_route(tiles, in_fmt, staging, b_slot) else {
        return false;
    };
    let roles = programs(
        (tiles, staging),
        route,
        fidelity,
        layout.sems,
        LiveRows::All,
        || matmul_roles_with(&layout.outputs, layout.sems, route.src_formats(), fidelity),
    );
    roles
        .iter()
        .all(|p| p.len() <= tt_isa::mailbox::PROGRAM_MAX as usize)
}

/// The three role programs of a `tiles`-shaped run, built once per process.
///
/// They depend on nothing but the shape, the route and the fidelity -- the L1
/// layout is a function of the shape ([`plan_layout`]) -- and building them was
/// most of a chunk's host time: [`plan`] builds every candidate it probes, on
/// every call. `build` must be `matmul_roles` over that shape's layout.
pub(crate) fn programs(
    tiles: ([usize; 3], Staging),
    route: SrcRoute,
    fidelity: Fidelity,
    sems: MatmulSemaphores,
    live: LiveRows,
    build: impl FnOnce() -> [Vec<Instruction>; 3],
) -> std::sync::Arc<[Vec<Instruction>; 3]> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    // The semaphores are part of the programs: the same shape planned beside
    // another kernel may be given different ones.
    // The programs name the layout's addresses, planned in the arena (`crate::l1::arena`).
    type Key = (
        ([usize; 3], Staging),
        SrcRoute,
        Fidelity,
        MatmulSemaphores,
        LiveRows,
        u64,
    );
    type Roles = Arc<[Vec<Instruction>; 3]>;
    static CACHE: OnceLock<Mutex<HashMap<Key, Roles>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (tiles, route, fidelity, sems, live, crate::l1::arena().end);
    if let Some(p) = cache.lock().unwrap().get(&key) {
        return p.clone();
    }
    let built = Arc::new(build());
    cache.lock().unwrap().insert(key, built.clone());
    built
}

/// [`programs`] for [`matmul_kernel`]: the programs and their MOP
/// configurations, built once per process.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kernel_programs(
    tiles: [usize; 3],
    half: u32,
    route: SrcRoute,
    fidelity: Fidelity,
    sems: MatmulSemaphores,
    allow_mop: bool,
    b_transposed: bool,
    live: LiveRows,
    build: impl FnOnce() -> ([Vec<Instruction>; 3], [Option<MopConfig>; 3]),
) -> (
    std::sync::Arc<[Vec<Instruction>; 3]>,
    [Option<MopConfig>; 3],
) {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    // `half`: the half of a double-buffered pair (`Staging::SlotsHalf`), whose
    // programs name addresses `half()` apart under the same semaphores -- or any
    // other layout a caller tells apart (a K-streamed chunk's).
    type Key = (
        [usize; 3],
        u32,
        SrcRoute,
        Fidelity,
        MatmulSemaphores,
        bool,
        bool,
        LiveRows,
        u64,
        u64,
    );
    type Value = (Arc<[Vec<Instruction>; 3]>, [Option<MopConfig>; 3]);
    static CACHE: OnceLock<Mutex<HashMap<Key, Value>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (
        tiles,
        half,
        route,
        fidelity,
        sems,
        allow_mop,
        b_transposed,
        live,
        // The layout was planned in the arena (`crate::l1::arena`), `B`'s ring against
        // `crate::l1::ring_end`.
        crate::l1::arena().end,
        crate::l1::ring_end(),
    );
    if let Some(p) = cache.lock().unwrap().get(&key) {
        return p.clone();
    }
    let (roles, mop) = build();
    let built = (Arc::new(roles), mop);
    cache.lock().unwrap().insert(key, built.clone());
    built
}

/// The chunk shape for `A[m, k] @ B[k, n]`: `K` whole if at all possible,
/// because splitting it moves part of the accumulation to the host; then the
/// largest `mc * nc` that fits.
///
/// `None` only if not even a single `[1, 1, 1]`-tile run fits, which would be a
/// bug in this crate.
pub fn plan(mkn: [usize; 3], route: SrcRoute, fidelity: Fidelity) -> Option<ChunkShape> {
    plan_in(mkn, route, fidelity, Staging::Host)
}

/// [`plan`] for either [`Staging`], cached per process. It notes, hit or miss, the arena end its
/// answer needs ([`plan_floor`]), not the probes it made to find it.
pub fn plan_in(
    mkn: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    staging: Staging,
) -> Option<ChunkShape> {
    let (p, floor) = plan_floor(mkn, route, fidelity, staging);
    crate::l1::note_planned(crate::l1::Planned {
        top: 0,
        need: floor,
    });
    p
}

/// [`plan_in`] without noting, and the lowest arena end ([`crate::l1::arena`]) at which its
/// answer is the same: the chunk it chose is the widest that fits, so it stands while that chunk's
/// own probe layout fits (`A` unpacked, as [`chunk_fits_in`] lays it out), and a smaller arena
/// narrows it (`docs/plans/planned-l1-residency.md` 2h: at `K` = 2048, `DATA.end` less 110,464
/// bytes). [`Staging::SlotsHalf`] holds it in [`half`] the arena, which keeps it while the arena
/// is twice the probe on [`half`]'s 4096-byte boundary. 0 for [`Staging::Host`], whose layout is
/// not planned in the arena.
pub fn plan_floor(
    mkn: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    staging: Staging,
) -> (Option<ChunkShape>, u64) {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    // By the arena's end too (`crate::l1::arena`): what fits it.
    type Key = ([usize; 3], SrcRoute, Fidelity, Staging, u64, Bf16Plans);
    type Value = (Option<ChunkShape>, u64);
    static CACHE: OnceLock<Mutex<HashMap<Key, Value>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (
        mkn,
        route,
        fidelity,
        staging,
        crate::l1::arena().end,
        bf16_plans(),
    );
    if let Some(p) = cache.lock().unwrap().get(&key) {
        return *p;
    }
    let p = crate::l1::unnoted(|| {
        let mut p = plan_uncached(mkn, route, fidelity, staging);
        // A BF16 `B`'s ring in the bytes an FP32 one's takes: the most columns up to that which fit.
        let ring = route.bf16_b() && bf16_plans() == Bf16Plans::Ring && staging != Staging::Host;
        if let (Some(s), true) = (p.as_mut(), ring) {
            let [mc, kc, nc] = s.tiles;
            let most = (nc as u64 * tt_isa::dm::TILE_SLOT / route.b_slot()) as usize;
            let nt = mkn[2].div_ceil(32).max(1);
            if let Some(wider) = (nc + 1..=most.min(nt))
                .rev()
                .find(|&w| chunk_fits_at([mc, kc, w], route, fidelity, staging, route.b_slot()))
            {
                s.tiles = [mc, kc, wider];
            }
        }
        // The floor of what is laid out: compact past `Bf16Plans::F32` (whose floor stays an FP32
        // `B`'s probe, as before compact rings).
        let floor_slot = match bf16_plans() {
            Bf16Plans::F32 => planned_b_slot(route),
            _ => route.b_slot(),
        };
        let floor = match (p, staging) {
            (None, _) | (_, Staging::Host) => 0,
            (Some(s), _) => {
                crate::l1::take_planned();
                let laid =
                    plan_layout_route(s.tiles, route.formats().0, staging, floor_slot).is_ok();
                let need = crate::l1::take_planned().need;
                let base = tt_isa::l1::DATA.base;
                match staging {
                    _ if !laid || need <= base => 0,
                    Staging::SlotsHalf => base + 2 * (need - base).next_multiple_of(4096),
                    _ => need,
                }
            }
        };
        (p, floor)
    });
    cache.lock().unwrap().insert(key, p);
    p
}

/// [`plan_floor`] in an arena ending at `arena_end`: what a capture keeping a region down to there
/// would plan. Dev-facing: a probe reporting the block width a region leaves a product.
#[doc(hidden)]
pub fn probe_plan_at(
    arena_end: u64,
    mkn: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    staging: Staging,
) -> (Option<ChunkShape>, u64) {
    let was = crate::l1::set_arena_end(arena_end);
    let p = plan_floor(mkn, route, fidelity, staging);
    crate::l1::set_arena_end(was);
    p
}

fn plan_uncached(
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    staging: Staging,
) -> Option<ChunkShape> {
    let [mt, kt, nt] = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)].map(|t| t.max(1));
    let mut kc = kt;
    loop {
        // For each nc, the largest mc that fits, by bisection: fitting is
        // monotone in each axis.
        let mut best: Option<[usize; 3]> = None;
        for nc in (1..=nt).rev() {
            if let Some(b) = best {
                if nc * mt <= b[0] * b[2] {
                    break;
                }
            }
            let fits = |mc: usize| chunk_fits_in([mc, kc, nc], route, fidelity, staging);
            if !fits(1) {
                continue;
            }
            let (mut lo, mut hi) = (1, mt);
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                if fits(mid) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            if best.is_none_or(|b| lo * nc > b[0] * b[2]) {
                best = Some([lo, kc, nc]);
            }
        }
        if let Some(tiles) = best {
            return Some(ChunkShape { tiles });
        }
        if kc == 1 {
            return None;
        }
        kc = kc.div_ceil(2);
    }
}

/// `A[m, k] @ B[k, n]`, row-major, in as many runs of `run` as [`plan`] says,
/// each on a row-major block `A[mi, ki] @ B[ki, nj]` given as `(a, b, [rows,
/// inner, cols])`.
///
/// When `K` is split, the partial products are summed on the host in FP32, in
/// `K` order, **which is not the accumulation order of one run** -- the device
/// sums the whole of `K` in `Dst` -- so a split matmul is not bit-identical to an
/// unsplit one unless every partial sum is exact. [`plan`] keeps `K` whole
/// whenever it can.
pub fn matmul_chunked<E>(
    a: &[f32],
    b: &[f32],
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    fidelity: Fidelity,
    mut run: impl FnMut(&[f32], &[f32], [usize; 3]) -> Result<Vec<f32>, E>,
) -> Result<Vec<f32>, E>
where
    E: From<crate::runtime::RunError>,
{
    assert_eq!(a.len(), m * k, "A is not [m, k]");
    assert_eq!(b.len(), k * n, "B is not [k, n]");
    let mut c = vec![0f32; m * n];
    if m == 0 || n == 0 {
        return Ok(c);
    }
    if k == 0 {
        return Ok(c);
    }
    let ChunkShape {
        tiles: [mc, kc, nc],
    } = plan([m, k, n], route, fidelity).ok_or(crate::runtime::RunError::DoesNotFit {
        what: "a single-tile matmul",
        bytes: 0,
        limit: 0,
    })?;
    let [mr, kr, nr] = [mc * 32, kc * 32, nc * 32];
    for i0 in (0..m).step_by(mr) {
        let rows = mr.min(m - i0);
        for j0 in (0..n).step_by(nr) {
            let cols = nr.min(n - j0);
            for (step, k0) in (0..k).step_by(kr).enumerate() {
                let inner = kr.min(k - k0);
                let sub_a: Vec<f32> = (i0..i0 + rows)
                    .flat_map(|r| a[r * k + k0..r * k + k0 + inner].iter().copied())
                    .collect();
                let sub_b: Vec<f32> = (k0..k0 + inner)
                    .flat_map(|r| b[r * n + j0..r * n + j0 + cols].iter().copied())
                    .collect();
                let part = run(&sub_a, &sub_b, [rows, inner, cols])?;
                for r in 0..rows {
                    for q in 0..cols {
                        let dst = &mut c[(i0 + r) * n + j0 + q];
                        let v = part[r * cols + q];
                        *dst = if step == 0 { v } else { *dst + v };
                    }
                }
            }
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `PACR` of a kernel's pack role sees the packer configuration a whole rewrite for its
    /// output tile would give it, though only the first output's is written whole and each later
    /// one's destination alone; and the pack role grows by a few words an output, not dozens.
    #[test]
    fn the_pack_role_configures_each_output_as_a_whole_rewrite_would() {
        let pacrs = crate::code::transfers(&crate::datapath::pack_rows(TILE_DST_ROWS))
            .0
            .len();
        let mut per_output = Vec::new();
        for tiles in [[1, 2, 1], [2, 3, 5], [1, 2, 20]] {
            let layout = plan_layout(tiles, L1Format::Fp32).unwrap();
            let [_, _, pack] = matmul_items_reloading(
                &layout.outputs,
                layout.sems,
                SrcFormats::same(L1Format::Fp32, 0),
                Fidelity::HiFi4,
                None,
                false,
                LiveRows::All,
            );
            let stream = crate::loops::Item::unrolled(&pack);
            let (seen, _) = crate::code::transfers(&stream);
            assert_eq!(seen.len(), pacrs * layout.outputs.len(), "{tiles:?}");
            for (k, output) in layout.outputs.iter().enumerate() {
                let mut whole = ConfigWords::new();
                crate::datapath::pack_config(&mut whole, output.out);
                for config in &seen[k * pacrs..(k + 1) * pacrs] {
                    for &(addr, value) in whole.words() {
                        assert_eq!(
                            config.get(&(addr as u32)),
                            Some(&value),
                            "{tiles:?} output {k} word {addr}"
                        );
                    }
                }
            }
            let stored = crate::loops::lower_with(&pack, false).words.len();
            per_output.push((stored, layout.outputs.len()));
        }
        let (a, b) = (per_output[1], per_output[2]);
        let words_an_output = (b.0 - a.0) / (b.1 - a.1);
        // 74 words an output when each wrote the whole configuration (16 words of three
        // instructions and a separating `NOP`); its destination takes 4, the semaphores and
        // `pack_rows` the rest.
        assert!(
            words_an_output <= 32,
            "{words_an_output} pack words an output"
        );
    }

    #[test]
    fn lo_fidelity_is_the_established_encoding() {
        // Two MVMULs per face pair at LoFi, as the backend receives them.
        let l = plan_layout([1, 1, 1], L1Format::Fp32).unwrap();
        let [_, math, _] =
            matmul_roles(&l.outputs, l.sems, L1Format::Fp32, TF32_CODE, Fidelity::Lo);
        // As the backend receives them: the replays expanded.
        let math = crate::loops::frontend_stream(&math, None).unwrap();
        let mvmuls = math
            .iter()
            .filter(|i| i.word() >> 24 == encode::Mvmul::ZERO.encode().unwrap().word() >> 24)
            .count();
        assert_eq!(mvmuls, 16);
    }

    /// The semaphores are the plan's, not constants: a matmul planned alone
    /// gets two distinct ones, and the same layout with the pair swapped builds
    /// programs that differ exactly where they post and take them.
    #[test]
    fn the_roles_use_the_semaphores_they_are_given() {
        let l = plan_layout([1, 2, 1], L1Format::Fp32).unwrap();
        assert_ne!(l.sems.ready, l.sems.free);
        let swapped = MatmulSemaphores {
            ready: l.sems.free,
            free: l.sems.ready,
        };
        let a = matmul_roles(&l.outputs, l.sems, L1Format::Fp32, TF32_CODE, Fidelity::Lo);
        let b = matmul_roles(&l.outputs, swapped, L1Format::Fp32, TF32_CODE, Fidelity::Lo);
        assert_eq!(a[0], b[0], "the unpacker uses no semaphore");
        for role in [1, 2] {
            assert_eq!(a[role].len(), b[role].len());
            let differ = a[role]
                .iter()
                .zip(&b[role])
                .filter(|(x, y)| x.word() != y.word())
                .count();
            // One output tile: a take (`SEMWAIT`, `SEMGET`) and a post
            // (`SEMPOST`) per role, each naming its semaphore.
            assert_eq!(differ, 3, "role {role}");
        }
    }

    #[test]
    fn each_phase_is_one_more_mvmul_per_half() {
        let l = plan_layout([1, 1, 1], L1Format::Fp32).unwrap();
        let opcode = encode::Mvmul::ZERO.encode().unwrap().word() >> 24;
        for f in [
            Fidelity::Lo,
            Fidelity::HiFi2,
            Fidelity::HiFi3,
            Fidelity::HiFi4,
        ] {
            let [_, math, _] = matmul_roles(&l.outputs, l.sems, L1Format::Fp32, TF32_CODE, f);
            let math = crate::loops::frontend_stream(&math, None).unwrap();
            let n = math.iter().filter(|i| i.word() >> 24 == opcode).count();
            assert_eq!(n as u32, 16 * f.phases(), "{f:?}");
        }
    }

    #[test]
    fn layout_refuses_rather_than_overrunning() {
        use crate::runtime::RunError;
        // 64 FP32 A tiles alone are 263 KiB; with 64 B tiles they overrun.
        assert!(matches!(
            plan_layout([8, 8, 8], L1Format::Fp32),
            Err(RunError::DoesNotFit {
                what: "the operand tiles",
                ..
            })
        ));
        // 129 output tiles overrun the mailbox.
        assert!(matches!(
            plan_layout([129, 1, 1], L1Format::Fp32),
            Err(RunError::DoesNotFit { .. })
        ));
        assert!(plan_layout([2, 2, 2], L1Format::Fp32).is_ok());
    }

    /// A K too long for one circular buffer's page counters (found by Burn's own
    /// `transaction` test: `[408, 261120] @ [261120, 408]`) is planned in shorter K-blocks, as a K too
    /// long for L1 is. It used to panic the planner, and with it the device's server thread.
    #[test]
    fn a_k_longer_than_a_circular_buffer_holds_is_planned_in_blocks() {
        use crate::l1::MAX_CB_PAGES;
        // One block of this many K tiles needs more pages than the counters hold.
        let kt = MAX_CB_PAGES as usize + 1;
        assert!(
            !chunk_fits_in(
                [1, kt, 1],
                SrcRoute::Tf32FromFp32,
                Fidelity::HiFi4,
                Staging::Slots
            ),
            "an over-long block is refused, not a panic"
        );
        let shape = plan_in(
            [408, 261_120, 408],
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            Staging::Slots,
        )
        .expect("a long K is split into blocks that fit");
        let [_, kc, _] = shape.tiles;
        assert!(kc >= 1 && kc * 32 < 261_120, "{shape:?}");
        assert!(chunk_fits_in(
            shape.tiles,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            Staging::Slots
        ));
    }

    /// `docs/plans/planned-l1-residency.md` 2h: a product's plan stands in any arena ending at or
    /// above its floor, and narrows 64 bytes below it. For a one-row product with `K` whole, the
    /// floor is the widest block's probe with `A` unpacked, `(kt + 1)(nc + 1) - 1` slots: at `K` =
    /// 2048, 110,464 bytes of `DATA` left for a region. `plan_in` notes it, hit or miss.
    #[test]
    fn a_plan_stands_down_to_its_floor_and_narrows_below_it() {
        use tt_isa::{dm::TILE_SLOT, l1::DATA};
        let (route, fid) = (SrcRoute::Tf32FromFp32, Fidelity::HiFi4);
        let at = |end: u64, mkn: [usize; 3], st: Staging| {
            let was = crate::l1::set_arena_end(end);
            let p = plan_floor(mkn, route, fid, st).0;
            crate::l1::set_arena_end(was);
            p
        };
        // 1.7B `qkv`, `gate|up`, head; Qwen2.5-1.5B's `gate|up`.
        for (k, n, nc, left) in [
            (2048, 6144, 2, 110_464),
            (2048, 16_384, 2, 110_464),
            (2048, 49_152, 2, 110_464),
            (1536, 17_920, 3, 106_304),
        ] {
            let mkn = [1, k, n];
            let kt = k / 32;
            let (p, floor) = plan_floor(mkn, route, fid, Staging::Slots);
            assert_eq!(p.unwrap().tiles, [1, kt, nc], "{mkn:?}");
            assert_eq!(
                floor,
                DATA.base + TILE_SLOT * ((kt + 1) * (nc + 1) - 1) as u64,
                "{mkn:?}"
            );
            assert_eq!(DATA.end - floor, left, "{mkn:?}");
            assert_eq!(at(floor, mkn, Staging::Slots), p, "{mkn:?} at its floor");
            assert_ne!(at(floor - 64, mkn, Staging::Slots), p, "{mkn:?} below it");
            // Noted on every call, the cached ones too.
            for _ in 0..2 {
                crate::l1::take_planned();
                plan_in(mkn, route, fid, Staging::Slots);
                assert_eq!(crate::l1::take_planned().need, floor, "{mkn:?}");
            }
        }
        // Half the arena: the arena twice the probe, on `half`'s 4096-byte boundary.
        let mkn = [1024, 1024, 1024];
        let (p, floor) = plan_floor(mkn, route, fid, Staging::SlotsHalf);
        assert!(p.is_some() && floor > DATA.base);
        assert_eq!((floor - DATA.base) % (2 * 4096), 0);
        assert_eq!(at(floor, mkn, Staging::SlotsHalf), p);
        assert_ne!(at(floor - 64, mkn, Staging::SlotsHalf), p);
    }

    /// A BF16 `B` ([`SrcRoute::Tf32WithBf16B`]) takes 2,112-byte slots in every slot layout
    /// (`docs/plans/bf16-weight-storage.md` A5b): its pairs step by them, its rings take that many
    /// bytes a page, and a decode product's plan holds more columns than an FP32 `B`'s in the
    /// same arena. An FP32 `B`'s plans are unchanged.
    #[test]
    fn a_bf16_b_plans_compact_slots() {
        use tt_isa::dm::{BF16_TILE_SLOT, TILE_SLOT};
        let fid = Fidelity::HiFi4;
        let (f32_b, bf16_b) = (SrcRoute::Tf32FromFp32, SrcRoute::Tf32WithBf16B);
        assert_eq!(
            (f32_b.b_slot(), bf16_b.b_slot()),
            (TILE_SLOT, BF16_TILE_SLOT)
        );
        // 1.7B `qkv`, `gate|up`, `o`; Qwen2.5-1.5B's `gate|up`: an FP32 `B`'s columns a job, as
        // many as its ring's bytes hold, or twice them.
        let plans = |mode, mkn| {
            set_bf16_plans(mode);
            let p = plan_floor(mkn, bf16_b, fid, Staging::Slots)
                .0
                .unwrap()
                .tiles;
            set_bf16_plans(Bf16Plans::F32);
            p
        };
        for (k, n, nc) in [
            (2048, 6144, 2),
            (2048, 16_384, 2),
            (2048, 2048, 2),
            (1536, 17_920, 3),
        ] {
            let f = plan_floor([1, k, n], f32_b, fid, Staging::Slots);
            let kt = k / 32;
            assert_eq!(f.0.unwrap().tiles, [1, kt, nc], "[1, {k}, {n}], FP32 B");
            assert_eq!(
                plans(Bf16Plans::F32, [1, k, n]),
                [1, kt, nc],
                "[1, {k}, {n}], BF16 B"
            );
            let ring = (nc as u64 * TILE_SLOT / BF16_TILE_SLOT) as usize;
            assert_eq!(
                plans(Bf16Plans::Ring, [1, k, n]),
                [1, kt, ring],
                "[1, {k}, {n}], ring"
            );
            assert_eq!(
                plans(Bf16Plans::Wide, [1, k, n]),
                [1, kt, 2 * nc],
                "[1, {k}, {n}], wide"
            );
        }
        let tiles = [2, 5, 3];
        for st in [Staging::Slots, Staging::SlotsHalf] {
            let l = plan_layout_route(tiles, L1Format::Fp32, st, BF16_TILE_SLOT).unwrap();
            for (o, out) in l.outputs.iter().enumerate() {
                let (i, j) = (o / 3, o % 3);
                for (kk, &(a, b)) in out.pairs.iter().enumerate() {
                    assert_eq!(a, l.a_at + ((i * 5 + kk) as u64) * TILE_SLOT);
                    assert_eq!(b, l.b_at + ((kk * 3 + j) as u64) * BF16_TILE_SLOT);
                }
            }
        }
        let (kb, _) = k_block_layout_with(7, 2, BF16_TILE_SLOT).unwrap();
        assert!(kb.outputs[0]
            .pairs
            .windows(2)
            .all(|p| p[1].1 - p[0].1 == BF16_TILE_SLOT));
        let s = k_stream_layout(16, 2, BF16_TILE_SLOT).unwrap();
        let (c, _) = s.chunk(1, 16, 16);
        assert!(c.outputs[0]
            .pairs
            .windows(2)
            .all(|p| p[1].1 - p[0].1 == BF16_TILE_SLOT));
        assert!(s.b_at[0].abs_diff(s.b_at[1]) >= 16 * BF16_TILE_SLOT);
        let bl = block_stream_layout([2, 4, 3], BF16_TILE_SLOT).unwrap();
        let (c, _) = bl.chunk(0, [2, 4, 3]);
        assert_eq!(
            c.outputs[4].pairs[3].1,
            bl.b_at[0] + (3 * 3 + 1) * BF16_TILE_SLOT
        );
        assert_eq!(
            block_stream_bytes([2, 4, 3], BF16_TILE_SLOT),
            2 * ((8 + 6) * TILE_SLOT + 12 * BF16_TILE_SLOT)
        );
    }

    #[test]
    fn the_plan_fits_and_is_maximal_along_n() {
        for (shape, fid) in [
            ([64, 784, 128], Fidelity::HiFi4),
            ([64, 784, 128], Fidelity::Lo),
            ([64, 128, 10], Fidelity::HiFi4),
            ([1, 1, 1], Fidelity::HiFi4),
            ([2048, 32, 2048], Fidelity::Lo),
        ] {
            let ChunkShape { tiles } = plan(shape, SrcRoute::Tf32FromFp32, fid).unwrap();
            assert!(
                chunk_fits(tiles, SrcRoute::Tf32FromFp32, fid),
                "{shape:?} {tiles:?}"
            );
            let nt = shape[2].div_ceil(32);
            if tiles[2] < nt {
                let wider = [tiles[0], tiles[1], tiles[2] + 1];
                assert!(
                    !chunk_fits(wider, SrcRoute::Tf32FromFp32, fid),
                    "{shape:?} {tiles:?}"
                );
            }
        }
    }

    #[test]
    fn chunking_reassembles_the_whole_product() {
        // Small-integer operands, so every order of summation is exact and the
        // host reference is the answer. The "device" here is the host.
        let [m, k, n] = [70, 900, 150];
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 5) as f32 - 2.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 3) % 7) as f32 - 3.0).collect();
        let host = |a: &[f32], b: &[f32], [m, k, n]: [usize; 3]| -> Vec<f32> {
            (0..m * n)
                .map(|x| (0..k).map(|q| a[x / n * k + q] * b[q * n + x % n]).sum())
                .collect()
        };
        let mut runs = 0;
        let c = matmul_chunked::<crate::runtime::RunError>(
            &a,
            &b,
            [m, k, n],
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            |a, b, mkn| {
                runs += 1;
                Ok(host(a, b, mkn))
            },
        )
        .unwrap();
        assert!(runs > 1, "the shape must need more than one run");
        assert_eq!(c, host(&a, &b, [m, k, n]));
    }
}

#[cfg(test)]
mod loop_tests {
    use super::*;

    /// The lowered roles are the unrolled ones, as the frontend expands them
    /// (`crate::loops::frontend_stream`), at every fidelity and over block
    /// shapes with several outputs and tile pairs; and the face blocks are
    /// shared: a pair costs one `REPLAY` on the math role at LoFi.
    #[test]
    fn the_lowered_roles_are_the_unrolled_ones() {
        for tiles in [[1usize, 1, 1], [2, 16, 3], [1, 25, 4]] {
            let layout = plan_layout_in(tiles, L1Format::Fp32, Staging::Slots).unwrap();
            for f in [Fidelity::Lo, Fidelity::HiFi2, Fidelity::HiFi4] {
                let items = matmul_items(&layout.outputs, layout.sems, L1Format::Fp32, 0, f);
                for (r, it) in items.iter().enumerate() {
                    for allow_mop in [false, true] {
                        let l = crate::loops::lower_with(it, allow_mop);
                        let got = crate::loops::frontend_stream(&l.words, l.mop.as_ref()).unwrap();
                        assert!(
                            got == Item::unrolled(it),
                            "{tiles:?} {f:?} role {r} mop {allow_mop}"
                        );
                    }
                }
                // What a GDDR kernel carries: the words with their MOP configurations.
                let (words, mop) = matmul_kernel(
                    &layout.outputs,
                    layout.sems,
                    L1Format::Fp32,
                    0,
                    f,
                    true,
                    false,
                    LiveRows::All,
                );
                for r in 0..3 {
                    let got = crate::loops::frontend_stream(&words[r], mop[r].as_ref()).unwrap();
                    assert!(
                        got == Item::unrolled(&items[r]),
                        "{tiles:?} {f:?} kernel role {r}"
                    );
                }
                if f == Fidelity::Lo && tiles[1] > 1 {
                    // A pair costs about one word on the math role at LoFi:
                    // the body recorded once, every other pair a REPLAY.
                    let math = crate::loops::lower_with(&items[1], false);
                    let pairs = tiles[0] * tiles[1] * tiles[2];
                    assert!(
                        math.words.len() < pairs + 80,
                        "{tiles:?}: {} words for {pairs} pairs",
                        math.words.len()
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod fast_layout {
    use super::*;

    fn values(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.5 - 7.25).collect()
    }

    #[test]
    fn fast_tilize_is_tt_layouts() {
        for (rows, cols) in [
            (1usize, 1usize),
            (32, 32),
            (64, 784),
            (33, 17),
            (100, 70),
            (7, 129),
            // Enough tiles to split over threads, with ragged edges.
            (1000, 1000),
            (4096, 600),
        ] {
            let v = values(rows * cols);
            let (want, _) = tilize_f32(&v, rows, cols, L1Format::Fp32);
            assert!(tilize_f32_fp32(&v, rows, cols) == want, "[{rows}, {cols}]");
        }
    }

    /// `cargo test --release -p tt-kernels --lib tilize_speed -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn tilize_speed() {
        let (rows, cols) = (1024, 1024);
        let v = values(rows * cols);
        let t = std::time::Instant::now();
        let bits: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
        let back: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        println!("to_bits and back {:?}", t.elapsed());
        let t = std::time::Instant::now();
        let img = tilize_f32_fp32(&back, rows, cols);
        println!("tilize {:?} ({} B)", t.elapsed(), img.len());
    }

    #[test]
    fn fast_detilize_is_tt_layouts() {
        for (rows, cols) in [
            (1usize, 1usize),
            (32, 32),
            (64, 784),
            (33, 17),
            (100, 70),
            (7, 129),
            // Enough tiles to split over threads, with ragged edges.
            (1000, 1000),
            (4096, 600),
        ] {
            let (rt, ct) = (rows.div_ceil(32), cols.div_ceil(32));
            let packed: Vec<u8> = (0..rt * ct * 4096).map(|i| (i * 31 % 251) as u8).collect();
            let a = detilize_packed(&packed, rows, cols);
            let b = detilize_packed_generic(&packed, rows, cols);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "[{rows}, {cols}]"
            );
        }
    }
}
