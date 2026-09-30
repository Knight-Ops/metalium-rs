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
    out: u64,
) -> [Vec<Instruction>; 3] {
    matmul_roles(
        &[OutputTile {
            pairs: pairs.to_vec(),
            out,
        }],
        in_fmt,
        out_fmt,
    )
}

/// [`tile_roles`] for several output tiles, one after another through `Dst`.
///
/// The math role clears `Dst` for each output tile only once the pack role has
/// finished reading the last one ([`DST_FREE`]), and the pack role packs each
/// only once the math role has finished writing it ([`DST_READY`]). One output
/// tile is in `Dst` at a time; double-buffering it is Phase 9.
pub fn matmul_roles(
    outputs: &[OutputTile],
    in_fmt: L1Format,
    out_fmt: u32,
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
                        math.push(set_src_b_row(0));
                        let (i, loaded) = loaded.mvmul(encode::Mvmul::ZERO.dst_row(dst)).unwrap();
                        math.push(i);
                        math.push(set_src_b_row(8));
                        let (i, empty) = loaded
                            .mvmul_release_both(encode::Mvmul::ZERO.dst_row(dst + 8))
                            .unwrap();
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

/// `SETRWC` of the `SrcB` counter alone: which eight-row half of `SrcB` the
/// next `MVMUL` reads (`SETRWC.md`; `MVMUL.md` takes `RWCs.SrcB & 0x38`).
fn set_src_b_row(row: u32) -> Instruction {
    encode::Setrwc::ZERO
        .src_b(1)
        .src_b_val(row)
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

/// Tile `a` and `b` (row-major) in `in_fmt` and plan one [`OutputTile`] per
/// output tile, row-major, each accumulating its `k / 32` tile pairs in order.
///
/// Padding is `tt_layout`'s zero, which is the identity for the accumulation.
pub fn stage_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    in_fmt: L1Format,
) -> StagedMatmul {
    let (ta, la) = tilize_f32(a, m, k, in_fmt);
    let (tb, lb) = tilize_f32(b, k, n, in_fmt);
    let (mt, kt) = (m.div_ceil(32), k.div_ceil(32));
    let nt = n.div_ceil(32);
    assert_eq!(la.tiles_per_matrix(), mt * kt);
    assert_eq!(lb.tiles_per_matrix(), kt * nt);
    let a_img = la.image().total_bytes() as u64;
    let b_img = lb.image().total_bytes() as u64;
    let b_at = (MATMUL_STAGE + ta.len() as u64).next_multiple_of(16);
    assert!(
        b_at + tb.len() as u64 <= MATMUL_OUT,
        "operands overrun the output region"
    );
    let mut outputs = Vec::new();
    for i in 0..mt {
        for j in 0..nt {
            let pairs = (0..kt)
                .map(|kk| {
                    (
                        MATMUL_STAGE + (i * kt + kk) as u64 * a_img,
                        b_at + (kk * nt + j) as u64 * b_img,
                    )
                })
                .collect();
            outputs.push(OutputTile {
                pairs,
                out: MATMUL_OUT + (i * nt + j) as u64 * 1024 * 4,
            });
        }
    }
    assert!(
        MATMUL_OUT + (outputs.len() * 1024 * 4) as u64 <= tt_isa::mailbox::MAILBOX_BASE,
        "the output tiles overrun the mailbox"
    );
    StagedMatmul {
        m,
        n,
        a: ta,
        b: tb,
        b_at,
        outputs,
    }
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
/// tile at `tile`: tiled and zero-padded by `tt_layout`, multiplied with
/// fidelity phase 0 through `route`, accumulated in FP32 `Dst`, packed as FP32
/// and de-tiled.
///
/// Phase 0 is exact for operands whose `Src` values carry at most five
/// significant bits in `B` and seven in `A` (`MatrixUnit.md:143-165`); beyond
/// that it is the fastest and least precise of the four phases, which is what a
/// first training backend runs and what Phase 9 revisits.
#[allow(clippy::too_many_arguments)]
pub fn matmul<T: tt_device::Transport, N: tt_isa::noc::NocId>(
    dev: &mut tt_device::Device<T>,
    tile: tt_isa::noc::NocCoord<N>,
    images: &crate::runtime::RoleImages<'_>,
    a: &[f32],
    b: &[f32],
    [m, k, n]: [usize; 3],
    route: SrcRoute,
    budget: u64,
) -> Result<Vec<f32>, crate::runtime::RunError> {
    use crate::runtime::{self, Kernel, Schedule};
    let (in_fmt, out_fmt) = route.formats();
    let staged = stage_matmul(a, b, m, k, n, in_fmt);
    let [unpack, math, pack] = matmul_roles(&staged.outputs, in_fmt, out_fmt);
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
    let out = runtime::run(dev, tile, images, &kernel, budget)?;
    Ok(detilize_packed(&out.l1[0], m, n))
}
