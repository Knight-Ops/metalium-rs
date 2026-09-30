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

// --- Whole tiles --------------------------------------------------------------

use tt_isa::backend::{self, Before};
use tt_isa::cfg::generated::thcon;
use tt_isa::matrix::Banks;
use tt_isa::sync::{self, Semaphore, Unit};
use tt_isa::tile::TileDescriptor;

/// Math -> pack: "`Dst` holds a finished output tile".
pub const DST_READY: Semaphore = match Semaphore::new(0) {
    Some(s) => s,
    None => unreachable!(),
};

/// Pack -> math: "the packer has finished reading `Dst`; it may be cleared".
/// Starts at one, so the first output tile does not wait.
pub const DST_FREE: Semaphore = match Semaphore::new(1) {
    Some(s) => s,
    None => unreachable!(),
};

/// The semaphores [`tile_roles`] and [`matmul_roles`] expect a concurrent run to
/// initialise. Every run leaves them as it found them: each `DST_READY` posted
/// is taken, and `DST_FREE` is taken once per tile and given back once per tile.
pub const TILE_SEMAPHORES: [(Semaphore, u8, u8); 2] = [
    (DST_READY, 0, sync::MAX_VALUE),
    (DST_FREE, 1, sync::MAX_VALUE),
];

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

/// Unpack face `face` of the tile `unpacker` is configured for: all 256 of its
/// datums, from the `Z` plane the unpack thread's ADC names.
///
/// `FirstDatum = ((W * ZDim + Z) * YDim + Y) * XDim + X`, with `Z` from the
/// *issuing* thread's ADCs and `X`/`Y` from `ContextADC`'s -- thread 0 for
/// both, on the unpack role (`UNPACR_Regular.md:136-182`). `X` runs over the
/// whole face, so one `UNPACR` moves 16 rows.
fn select_face(unpacker: Unpacker, face: u32) -> [Instruction; 2] {
    let z = encode::Setadczw::ZERO.z0(1).z0_val(face);
    let z = match unpacker {
        Unpacker::SrcA => z.u0(1),
        Unpacker::SrcB => z.u1(1),
    };
    [
        crate::datapath::set_adc_x(unpacker, 0, FACE_DATUMS - 1),
        z.encode().unwrap(),
    ]
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
/// concurrently (`harness::Run::concurrent` with [`TILE_SEMAPHORES`]).
pub fn tile_roles(
    pairs: &[(u64, u64)],
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
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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
/// finished reading the last one ([`DST_FREE`]), and the pack role packs each
/// only once the math role has finished writing it ([`DST_READY`]). One output
/// tile is in `Dst` at a time; double-buffering it is Phase 9.
///
/// With more than one fidelity phase, each `SrcB` half gets one `MVMUL` per
/// phase on the same banks, each with address modifier 1, whose only effect is
/// `FidelityIncr` ([`math_prelude`]; `RWCs.md`, `ApplyAddrMod`), and the
/// `SETRWC` that selects the next half also puts the phase back to zero -- so
/// nothing depends on whether the phase counter wraps.
pub fn matmul_roles(
    outputs: &[OutputTile],
    in_fmt: L1Format,
    out_fmt: u32,
    fidelity: Fidelity,
) -> [Vec<Instruction>; 3] {
    assert!(
        outputs.iter().all(|o| !o.pairs.is_empty()),
        "every output tile needs at least one operand pair"
    );
    let (a0, b0) = outputs[0].pairs[0];
    let mut unpack = src_thread_config();
    let mut words = ConfigWords::new();
    let descriptor = tile_descriptor(in_fmt);
    unpack_src_config(&mut words, Unpacker::SrcA, descriptor, b0, out_fmt);
    unpack_src_config(&mut words, Unpacker::SrcB, descriptor, a0, out_fmt);
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    words
        .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
        .unwrap();
    unpack.extend(config_program(&words));

    let mut math = math_prelude();
    let mut pack = Vec::new();
    let base = encode::UnpacrRegular::ZERO.multi_context_mode(1);
    let mut banks = Banks::after_reset();
    let mut current = (a0, b0);
    for output in outputs {
        math.extend(sync::take(DST_FREE, Before::MATRIX));
        // (mode, use_dst32b, addr_mod, imm10): all of `Dst`.
        math.push(encode::zeroacc(3, 0, 0, 0).unwrap());
        for &(a, b) in &output.pairs {
            if (a, b) != current {
                unpack.extend(retarget(a, b));
                current = (a, b);
            }
            for fi in 0..2u32 {
                for fj in 0..2u32 {
                    for k in 0..2u32 {
                        unpack.extend(select_face(Unpacker::SrcA, 2 * k + fj));
                        let (i, loaded_a) = banks.unpack_a(base).unwrap();
                        unpack.push(i);
                        unpack.extend(select_face(Unpacker::SrcB, 2 * fi + k));
                        let (i, loaded) = loaded_a.unpack_b(base).unwrap();
                        unpack.push(i);

                        let dst = 16 * (2 * fi + fj);
                        let phases = fidelity.phases();
                        // Lo keeps the encoding the gates established; with more
                        // phases, each MVMUL steps the phase (addr_mod 1).
                        let step = |row: u32| {
                            let m = encode::Mvmul::ZERO.dst_row(row);
                            if phases > 1 {
                                m.addr_mod(1)
                            } else {
                                m
                            }
                        };
                        let mut loaded = loaded;
                        for half in 0..2u32 {
                            math.push(set_src_b_row(8 * half, phases > 1));
                            for phase in 0..phases {
                                if half == 1 && phase + 1 == phases {
                                    break;
                                }
                                let (i, next) = loaded.mvmul(step(dst + 8 * half)).unwrap();
                                math.push(i);
                                loaded = next;
                            }
                        }
                        let (i, empty) = loaded.mvmul_release_both(step(dst + 8)).unwrap();
                        math.push(i);
                        banks = empty;
                    }
                }
            }
        }
        math.extend(sync::post_after(Unit::Matrix, DST_READY));

        let mut pw = ConfigWords::new();
        crate::datapath::pack_config(&mut pw, output.out);
        pack.extend(config_program(&pw));
        pack.extend(sync::take(DST_READY, Before::PACKER));
        pack.extend(crate::datapath::pack_rows(TILE_DST_ROWS));
        pack.extend(sync::post_after(Unit::Packer, DST_FREE));
    }
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    unpack.push(backend::wait_for_unpacker1(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, math, pack]
}

/// `SETRWC` of the `SrcB` counter: which eight-row half of `SrcB` the next
/// `MVMUL` reads (`SETRWC.md`; `MVMUL.md` takes `RWCs.SrcB & 0x38`), and, with
/// `reset_phase`, the fidelity phase back to zero.
fn set_src_b_row(row: u32, reset_phase: bool) -> Instruction {
    encode::Setrwc::ZERO
        .src_b(1)
        .src_b_val(row)
        .fidelity(u32::from(reset_phase))
        .encode()
        .unwrap()
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

/// Row-major `[rows, cols]` FP32 values from packed output tiles: each tile's
/// 1024 datums as [`tile_roles`] packs them, with no header, one after another
/// in the layout's tile order.
pub fn detilize_packed(packed: &[u8], rows: usize, cols: usize) -> Vec<f32> {
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

/// Where [`stage_matmul`] puts things in L1: `A`'s tiles from here, then
/// `B`'s, then the packed output tiles from [`MATMUL_OUT`].
pub const MATMUL_STAGE: u64 = 0x2_0000;
/// Where the packed output tiles go.
pub const MATMUL_OUT: u64 = 0x8_0000;

/// A matmul `C[m, n] = A[m, k] @ B[k, n]` laid out for the device: both
/// operands tiled and padded by `tt_layout`, and the output tiles planned.
pub struct StagedMatmul {
    pub m: usize,
    pub n: usize,
    pub a: Vec<u8>,
    pub b: Vec<u8>,
    pub b_at: u64,
    pub outputs: Vec<OutputTile>,
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

/// Where a `[mt, kt] @ [kt, nt]`-tile matmul's operands and outputs go in L1,
/// without the data: `B`'s first byte, and one [`OutputTile`] per output tile.
///
/// Refuses a shape whose operands would overrun [`MATMUL_OUT`] or whose output
/// would overrun the mailbox, as [`RunError::DoesNotFit`](crate::runtime::RunError).
pub fn plan_layout(
    [mt, kt, nt]: [usize; 3],
    in_fmt: L1Format,
) -> Result<(u64, Vec<OutputTile>), crate::runtime::RunError> {
    use crate::runtime::RunError;
    let img = tile_image_bytes(in_fmt);
    let a_bytes = (mt * kt) as u64 * img;
    let b_at = (MATMUL_STAGE + a_bytes).next_multiple_of(16);
    let b_end = b_at + (kt * nt) as u64 * img;
    if b_end > MATMUL_OUT {
        return Err(RunError::DoesNotFit {
            what: "the operand tiles",
            bytes: b_end - MATMUL_STAGE,
            limit: MATMUL_OUT - MATMUL_STAGE,
        });
    }
    let out_bytes = (mt * nt) as u64 * 1024 * 4;
    if MATMUL_OUT + out_bytes > tt_isa::mailbox::MAILBOX_BASE {
        return Err(RunError::DoesNotFit {
            what: "the output tiles",
            bytes: out_bytes,
            limit: tt_isa::mailbox::MAILBOX_BASE - MATMUL_OUT,
        });
    }
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
                out: MATMUL_OUT + (i * nt + j) as u64 * 1024 * 4,
            });
        }
    }
    Ok((b_at, outputs))
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
    let tiles = [m.div_ceil(32), k.div_ceil(32), n.div_ceil(32)];
    let (b_at, outputs) = plan_layout(tiles, in_fmt)?;
    let (ta, la) = tilize_f32(a, m, k, in_fmt);
    let (tb, lb) = tilize_f32(b, k, n, in_fmt);
    assert_eq!(la.tiles_per_matrix(), tiles[0] * tiles[1]);
    assert_eq!(lb.tiles_per_matrix(), tiles[1] * tiles[2]);
    assert_eq!(la.image().total_bytes() as u64, tile_image_bytes(in_fmt));
    debug_assert_eq!(b_at, (MATMUL_STAGE + ta.len() as u64).next_multiple_of(16));
    Ok(StagedMatmul {
        m,
        n,
        a: ta,
        b: tb,
        b_at,
        outputs,
    })
}

/// How operands reach `Src`, and so the precision the Matrix Unit multiplies
/// at (`UNPACR_Regular.md:495-520`; codes measured, divergence rows G and H).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SrcRoute {
    /// FP32 in L1, truncated to TF32 by the unpacker.
    Tf32FromFp32,
    /// FP32 in L1, converted to BF16 by the unpacker.
    Bf16FromFp32,
    /// BF16 in L1, moved as is.
    Bf16FromBf16,
}

impl SrcRoute {
    /// The L1 format the operands are staged in, and the `OutDataFormat` code.
    pub const fn formats(self) -> (L1Format, u32) {
        match self {
            SrcRoute::Tf32FromFp32 => (L1Format::Fp32, TF32_CODE),
            SrcRoute::Bf16FromFp32 => (L1Format::Fp32, BF16_CODE),
            SrcRoute::Bf16FromBf16 => (L1Format::Bf16, BF16_CODE),
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
    let (in_fmt, out_fmt) = route.formats();
    let staged = stage_matmul(a, b, m, k, n, in_fmt)?;
    let [unpack, math, pack] = matmul_roles(&staged.outputs, in_fmt, out_fmt, fidelity);
    let stage = [
        (MATMUL_STAGE, staged.a.as_slice()),
        (staged.b_at, staged.b.as_slice()),
    ];
    let read_back = [(MATMUL_OUT, staged.out_bytes())];
    let kernel = Kernel {
        stage: &stage,
        read_back: &read_back,
        ..Kernel::new(
            [&unpack, &math, &pack],
            Schedule::Concurrent(&TILE_SEMAPHORES),
        )
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
    let (in_fmt, out_fmt) = route.formats();
    let Ok((_, outputs)) = plan_layout(tiles, in_fmt) else {
        return false;
    };
    let roles = matmul_roles(&outputs, in_fmt, out_fmt, fidelity);
    roles
        .iter()
        .all(|p| p.len() <= tt_isa::mailbox::PROGRAM_MAX as usize)
}

/// The chunk shape for `A[m, k] @ B[k, n]`: `K` whole if at all possible,
/// because splitting it moves part of the accumulation to the host; then the
/// largest `mc * nc` that fits.
///
/// `None` only if not even a single `[1, 1, 1]`-tile run fits, which would be a
/// bug in this crate.
pub fn plan([m, k, n]: [usize; 3], route: SrcRoute, fidelity: Fidelity) -> Option<ChunkShape> {
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
            let fits = |mc: usize| chunk_fits([mc, kc, nc], route, fidelity);
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

    #[test]
    fn lo_fidelity_is_the_established_encoding() {
        // Two MVMULs per face pair, neither with an address modifier, and a
        // SETRWC without a fidelity reset before each: what every gate before
        // fidelity was a parameter ran.
        let (b_at, outputs) = plan_layout([1, 1, 1], L1Format::Fp32).unwrap();
        let _ = b_at;
        let [_, math, _] = matmul_roles(&outputs, L1Format::Fp32, TF32_CODE, Fidelity::Lo);
        let mvmuls = math
            .iter()
            .filter(|i| i.word() >> 24 == encode::Mvmul::ZERO.encode().unwrap().word() >> 24)
            .count();
        assert_eq!(mvmuls, 16);
    }

    #[test]
    fn each_phase_is_one_more_mvmul_per_half() {
        let (_, outputs) = plan_layout([1, 1, 1], L1Format::Fp32).unwrap();
        let opcode = encode::Mvmul::ZERO.encode().unwrap().word() >> 24;
        for f in [
            Fidelity::Lo,
            Fidelity::HiFi2,
            Fidelity::HiFi3,
            Fidelity::HiFi4,
        ] {
            let [_, math, _] = matmul_roles(&outputs, L1Format::Fp32, TF32_CODE, f);
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
