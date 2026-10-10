//! Physical BF16 tiles. This type cannot enter a 32-bit tensor transfer by
//! accident. Conversions run on Tensix; B/NC only copy the packed payload.
//! Conversion rounds ties-even, quiets NaNs and flushes BF16 subnormals to
//! signed zero, matching Blackhole late pack narrowing.
use std::sync::Arc;

use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::{global, thcon},
    dm::{op, TILE_DATA},
    dram::DramRange,
    isa::Instruction,
    sync::{self, Unit},
};

use crate::{
    datapath,
    sfpu::{kernel, Cond, Format, LReg, Program},
    tensor::{DramAlloc, DramTensor, Elem, Job, Pad, Placement, Result, Step, TensorError, Work},
};

/// 1024 two-byte datums, a 16-byte header and 64-byte slot alignment.
pub const TILE_SLOT: u64 = tt_isa::dm::BF16_TILE_SLOT;

/// A resident BF16 matrix, with an explicit physical allocation. It is
/// intentionally distinct from `DramTensor`'s 32-bit datum contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bf16Tensor {
    pub rows: usize,
    pub cols: usize,
    pub(crate) placement: Placement,
}

impl Bf16Tensor {
    pub fn rows_view(&self, first_row: usize, rows: usize) -> Result<Self> {
        let end = first_row
            .checked_add(rows)
            .ok_or_else(|| TensorError::Shape("BF16 row view overflow".into()))?;
        if rows == 0
            || first_row % 32 != 0
            || end > self.rows
            || (end % 32 != 0 && end != self.rows)
        {
            return Err(TensorError::Shape(
                "BF16 row view requires whole tile rows".into(),
            ));
        }
        let ct = self.cols.div_ceil(32);
        Ok(Self {
            rows,
            cols: self.cols,
            placement: self
                .placement
                .borrowed_tiles(first_row / 32 * ct, rows.div_ceil(32) * ct),
        })
    }
    pub fn tile_count(&self) -> usize {
        self.placement.tiles()
    }

    /// Physical slot range, useful for inspecting actual format bytes.
    pub fn slot(&self, tile: usize) -> DramRange {
        self.placement.slot(tile)
    }
}

pub(crate) fn transfer(range: DramRange, read: bool, l1: u64, bytes: u32) -> [u32; 8] {
    [
        if read { op::READ } else { op::WRITE },
        range.channel().index() as u32,
        0,
        (range.offset() + if read { 0 } else { TILE_DATA }) as u32,
        (l1 + if read { 0 } else { TILE_DATA }) as u32,
        bytes,
        0,
        0,
    ]
}

/// Exact ties-even BF16 conversion on raw FP32 bits. Quiet NaNs retain their
/// sign/high payload and cannot become infinity when the payload is discarded.
fn quantize() -> Vec<Instruction> {
    let mut p = Program::new();
    p.for_each_row_group(64, |p, row| {
        p.load(LReg::L0, Format::Int32, row);
        p.loadi_bits(LReg::L3, 0x7fffffff);
        p.and(LReg::L0, LReg::L3, LReg::L1);
        p.loadi_bits(LReg::L3, 0x7f800000);
        p.if_else(
            Cond::Less(LReg::L3, LReg::L1),
            |p| {
                p.loadi_bits(LReg::L3, 0x00400000);
                p.or(LReg::L0, LReg::L3, LReg::L0);
            },
            |p| {
                p.loadi_bits(LReg::L3, 0x10000);
                p.and(LReg::L0, LReg::L3, LReg::L1);
                p.loadi_bits(LReg::L3, (-16i32) as u32);
                p.shr_by(LReg::L3, LReg::L1);
                p.loadi_bits(LReg::L2, 0x7fff);
                p.iadd(LReg::L2, LReg::L1);
                p.iadd(LReg::L1, LReg::L0);
            },
        );
        p.loadi_bits(LReg::L3, 0xffff0000);
        p.and(LReg::L0, LReg::L3, LReg::L0);
        p.loadi_bits(LReg::L3, 0x7fffffff);
        p.and(LReg::L0, LReg::L3, LReg::L1);
        p.loadi_bits(LReg::L3, 0x00800000);
        p.if_(Cond::Less(LReg::L1, LReg::L3), |p| {
            p.loadi_bits(LReg::L3, 0x80000000);
            p.and(LReg::L0, LReg::L3, LReg::L0);
        });
        p.store(LReg::L0, Format::Int32, row);
    });
    p.finish()
}

fn roles(layout: &kernel::Layout, compress: bool) -> [Vec<Instruction>; 3] {
    let s = layout.sems;
    let mut unpack = datapath::thread_config();
    unpack.extend(datapath::clear_unpacker0_adcs());
    let mut math = crate::matmul::math_prelude();
    if compress {
        unpack.extend(sync::take(s.free, Before::UNPACKER));
        let mut words = ConfigWords::new();
        datapath::tile_unpack_config(&mut words, layout.a_at);
        unpack.extend(datapath::config_program(&words));
        unpack.extend(datapath::unpack_tile_to_dst(layout.a_at, 0));
        unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
        math.extend(sync::take(s.unpacked, Before::SFPU));
        math.extend(quantize());
        math.extend(sync::post_after(Unit::Sfpu, s.computed));
    } else {
        // The Src route supports BF16 in both ttsim and silicon. Convert
        // eight 128-datum chunks by MOVA2D instead of UnpackToDst, which
        // cannot widen this format. Each bank is released exactly once.
        unpack = datapath::src_thread_config();
        let mut empty = tt_isa::matrix::Banks::after_reset();
        for chunk in 0..8 {
            unpack.extend(sync::take(s.free, Before::UNPACKER));
            let mut words = ConfigWords::new();
            datapath::unpack_src_config(
                &mut words,
                datapath::Unpacker::SrcA,
                datapath::flat_descriptor(128).with_in_data_format_raw(5),
                layout.a_at + chunk * 256,
                5,
            );
            words
                .set(tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                .unwrap();
            words
                .set(
                    tt_isa::cfg::generated::alu::ALU_ACC_CTRL_Zero_Flag_disabled_src,
                    1,
                )
                .unwrap();
            unpack.extend(datapath::config_program(&words));
            unpack.push(datapath::set_adc_x(datapath::Unpacker::SrcA, 0, 127));
            let (i, banks) = empty
                .unpack_a(tt_isa::isa::generated::encode::UnpacrRegular::ZERO.multi_context_mode(1))
                .unwrap();
            unpack.push(i);
            unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
            math.extend(sync::take(s.unpacked, Before::MATRIX));
            let (i, banks) = banks
                .mova2d(
                    tt_isa::isa::generated::encode::Mova2D::ZERO
                        .move8_rows(1)
                        .dst_row(chunk as u32 * 8),
                )
                .unwrap();
            math.push(i);
            math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
            let (i, next) = banks.release_a().unwrap();
            empty = next;
            math.push(i);
            math.extend(sync::post_after(
                Unit::Matrix,
                if chunk == 7 { s.computed } else { s.free },
            ));
        }
    }
    let mut pack = vec![datapath::state_id()];
    pack.extend(sync::take(s.computed, Before::PACKER));
    pack.push(backend::wait_for_packer(Before::CONFIG).unwrap());
    let mut words = ConfigWords::new();
    datapath::pack_config(&mut words, layout.out_at + TILE_DATA);
    if compress {
        words
            .set(thcon::THCON_SEC0_REG1_Out_data_format, 5)
            .unwrap();
    }
    words
        .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, 0)
        .unwrap();
    pack.extend(datapath::config_program(&words));
    pack.extend(datapath::pack_rows(64));
    pack.extend(sync::post_after(Unit::Packer, s.free));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, math, pack]
}

fn jobs(input: &Placement, output: &Placement, compress: bool) -> Result<Vec<Job>> {
    let layout = kernel::plan_layout(1, kernel::Operands::Unary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let roles = Arc::new(roles(&layout, compress));
    Ok((0..input.tiles())
        .map(|t| {
            vec![
                Step::List {
                    what: "BF16 gather",
                    entries: vec![transfer(
                        input.slot(t),
                        true,
                        layout.a_at,
                        if compress {
                            tt_isa::dm::TILE_SLOT as u32
                        } else {
                            TILE_SLOT as u32
                        },
                    )],
                },
                Step::Kernel {
                    roles: roles.clone(),
                    init: layout.init.clone(),
                    mop: Box::new([None; 3]),
                    loops: Default::default(),
                    half: None,
                },
                Step::List {
                    what: "BF16 scatter",
                    entries: vec![transfer(
                        output.slot(t),
                        false,
                        layout.out_at,
                        if compress { 2048 } else { 4096 },
                    )],
                },
            ]
        })
        .collect())
}

pub(crate) fn compress(
    alloc: &mut DramAlloc,
    input: &DramTensor,
) -> Result<(Bf16Tensor, Vec<Job>)> {
    input.expect("BF16 conversion", Elem::F32)?;
    let output = Bf16Tensor {
        rows: input.rows,
        cols: input.cols,
        placement: alloc.alloc_slots(input.placement.tiles(), TILE_SLOT)?,
    };
    let jobs = jobs(&input.placement, &output.placement, true)?;
    Ok((output, jobs))
}

pub(crate) fn expand(alloc: &mut DramAlloc, input: &Bf16Tensor) -> Result<Work> {
    let out = DramTensor::alloc(alloc, input.rows, input.cols)?;
    let jobs = jobs(&input.placement, &out.placement, false)?;
    out.set_pad(Pad::Undefined);
    Ok(Work { out, jobs })
}

/// Bit-preserving BF16 repack. The host builds coordinates; B copies halfwords
/// in L1 and NC writes the result. No floating-point instruction is involved.
pub(crate) fn repack(
    alloc: &mut DramAlloc,
    input: &Bf16Tensor,
    sources: &[[usize; 2]],
    [rows, cols]: [usize; 2],
) -> Result<(Bf16Tensor, Vec<Job>)> {
    let sources: Vec<_> = sources.iter().copied().map(Some).collect();
    repack_padded(alloc, input, &sources, [rows, cols], 0)
}

pub(crate) fn repack_padded(
    alloc: &mut DramAlloc,
    input: &Bf16Tensor,
    sources: &[Option<[usize; 2]>],
    [rows, cols]: [usize; 2],
    padding: u16,
) -> Result<(Bf16Tensor, Vec<Job>)> {
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(sources.len()) {
        return Err(TensorError::Shape("invalid BF16 repack mapping".into()));
    }
    let output = Bf16Tensor {
        rows,
        cols,
        placement: alloc.alloc_slots(rows.div_ceil(32) * cols.div_ceil(32), TILE_SLOT)?,
    };
    match repack_padded_into(input, sources, &output, padding) {
        Ok(jobs) => Ok((output, jobs)),
        Err(error) => {
            alloc.free(&output.placement);
            Err(error)
        }
    }
}

/// Reuse a bounded staging allocation. Every output tile is initialized before
/// copying; ordered submission preserves producer/consumer and trace lifetimes.
pub(crate) fn repack_padded_into(
    input: &Bf16Tensor,
    sources: &[Option<[usize; 2]>],
    output: &Bf16Tensor,
    padding: u16,
) -> Result<Vec<Job>> {
    let mapping: Vec<_> = sources.iter().map(|x| x.map(|at| (0, at))).collect();
    repack_many_into(&[input], &mapping, output, padding)
}

pub(crate) fn repack_many(
    alloc: &mut DramAlloc,
    inputs: &[&Bf16Tensor],
    sources: &[(usize, [usize; 2])],
    [rows, cols]: [usize; 2],
) -> Result<(Bf16Tensor, Vec<Job>)> {
    if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(sources.len()) {
        return Err(TensorError::Shape("invalid BF16 repack mapping".into()));
    }
    let output = Bf16Tensor {
        rows,
        cols,
        placement: alloc.alloc_slots(rows.div_ceil(32) * cols.div_ceil(32), TILE_SLOT)?,
    };
    let mapping: Vec<_> = sources.iter().copied().map(Some).collect();
    match repack_many_into(inputs, &mapping, &output, 0) {
        Ok(jobs) => Ok((output, jobs)),
        Err(error) => {
            alloc.free(&output.placement);
            Err(error)
        }
    }
}

pub(crate) fn zeros(
    alloc: &mut DramAlloc,
    [rows, cols]: [usize; 2],
) -> Result<(Bf16Tensor, Vec<Job>)> {
    let count = rows
        .checked_mul(cols)
        .filter(|&n| n > 0)
        .ok_or_else(|| TensorError::Shape("invalid BF16 zero shape".into()))?;
    let output = Bf16Tensor {
        rows,
        cols,
        placement: alloc.alloc_slots(rows.div_ceil(32) * cols.div_ceil(32), TILE_SLOT)?,
    };
    match repack_many_into(&[], &vec![None; count], &output, 0) {
        Ok(jobs) => Ok((output, jobs)),
        Err(error) => {
            alloc.free(&output.placement);
            Err(error)
        }
    }
}

fn repack_many_into(
    inputs: &[&Bf16Tensor],
    sources: &[Option<(usize, [usize; 2])>],
    output: &Bf16Tensor,
    padding: u16,
) -> Result<Vec<Job>> {
    let (rows, cols) = (output.rows, output.cols);
    use crate::tensor::TransferBatch;
    use tt_isa::dm::face_index;
    if rows == 0
        || cols == 0
        || rows.checked_mul(cols) != Some(sources.len())
        || sources
            .iter()
            .flatten()
            .any(|&(i, [r, c])| inputs.get(i).is_none_or(|x| r >= x.rows || c >= x.cols))
    {
        return Err(TensorError::Shape("invalid BF16 repack mapping".into()));
    }
    let mut req = crate::l1::Requirements::new(1);
    let src = req.scratch("BF16 repack source", TILE_SLOT, 64, 0..1);
    let dst = req.scratch("BF16 repack output", TILE_SLOT, 64, 0..1);
    let constant = req.scratch("BF16 repack padding", TILE_SLOT, 64, 0..1);
    let layout = req
        .plan(tt_isa::l1::DATA)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let (src, dst) = (layout.addr(src), layout.addr(dst));
    let constant = layout.addr(constant);
    let ct = cols.div_ceil(32);
    let mut jobs = Vec::new();
    for tile in 0..output.tile_count() {
        let mut by_tile = std::collections::BTreeMap::<(usize, usize), Vec<(usize, usize)>>::new();
        for r in 0..32.min(rows - tile / ct * 32) {
            for c in 0..32.min(cols - tile % ct * 32) {
                let Some((source, [sr, sc])) =
                    sources[(tile / ct * 32 + r) * cols + tile % ct * 32 + c]
                else {
                    continue;
                };
                by_tile
                    .entry((source, sr / 32 * inputs[source].cols.div_ceil(32) + sc / 32))
                    .or_default()
                    .push((face_index(sr % 32, sc % 32), face_index(r, c)));
            }
        }
        let out_range = output.slot(tile);
        let mut batches = Vec::new();
        let initialise = || {
            vec![
                [
                    op::FILL_HALFWORDS,
                    padding as u32,
                    1 | (1 << 8),
                    constant as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    op::COPY_HALFWORDS,
                    (constant + TILE_DATA + 2) as u32,
                    (dst + TILE_DATA) as u32,
                    1024,
                    0,
                    2,
                    0,
                    0,
                ],
            ]
        };
        for ((source, source_tile), pairs) in by_tile {
            let input = inputs[source];
            for chunk in pairs.chunks(200) {
                let mut read = Vec::new();
                if !batches.is_empty() {
                    read.push(transfer(out_range, true, dst, TILE_SLOT as u32));
                } else {
                    read.extend(initialise());
                }
                read.push(transfer(
                    input.slot(source_tile),
                    true,
                    src,
                    TILE_SLOT as u32,
                ));
                read.extend(chunk.iter().map(|&(from, to)| {
                    [
                        op::COPY_HALFWORDS,
                        (src + TILE_DATA + from as u64 * 2) as u32,
                        (dst + TILE_DATA + to as u64 * 2) as u32,
                        1,
                        2,
                        2,
                        0,
                        0,
                    ]
                }));
                batches.push(TransferBatch {
                    read,
                    write: vec![transfer(out_range, false, dst, 2048)],
                });
            }
        }
        if batches.is_empty() {
            batches.push(TransferBatch {
                read: initialise(),
                write: vec![transfer(out_range, false, dst, 2048)],
            });
        }
        jobs.push(vec![Step::Transfer {
            what: "BF16 bit repack",
            depth: 1,
            batches,
        }]);
    }
    Ok(jobs)
}

fn gather(
    a: &Bf16Tensor,
    b: &Bf16Tensor,
    at: [u64; 2],
    block: [usize; 4],
    k0: usize,
    kt: usize,
) -> Vec<[u32; 8]> {
    use tt_isa::dm::record;
    let encode = |input: &Bf16Tensor, origin: [usize; 2]| {
        let mut encoded = input.placement.tensor_ref(input.cols.div_ceil(32)).encode();
        encoded[0][4] = input.rows as u32;
        encoded[0][5] = input.cols as u32;
        encoded[0][6] = origin[0] as u32;
        encoded[0][7] = origin[1] as u32;
        encoded
    };
    let [a0, a1] = encode(a, [0, k0]);
    let [b0, b1] = encode(b, [k0, 0]);
    let [i0, rows, j0, cols] = block;
    vec![
        [
            record::GATHER,
            record::GATHER_BF16 | (kt as u32) << 8,
            at[0] as u32,
            at[1] as u32,
            i0 as u32,
            rows as u32,
            j0 as u32,
            cols as u32,
        ],
        a0,
        a1,
        b0,
        b1,
    ]
}

/// Direct packed BF16 inputs, FP32 matrix accumulation and FP32 output.
/// K blocks reload the prior FP32 accumulator after NC releases it.
pub(crate) fn matmul(
    alloc: &mut DramAlloc,
    a: &Bf16Tensor,
    b: &Bf16Tensor,
    fidelity: crate::matmul::Fidelity,
    units: usize,
    k_limit: Option<usize>,
) -> Result<Work> {
    use crate::matmul::{self, SrcRoute, Staging};
    use tt_isa::{dm::record, tile::L1Format};
    if [a.rows, a.cols, b.rows, b.cols].contains(&0) || a.cols != b.rows {
        return Err(TensorError::Shape(
            "BF16 matmul requires nonempty compatible matrices".into(),
        ));
    }
    let [mt, kt, nt] = [
        a.rows.div_ceil(32),
        a.cols.div_ceil(32),
        b.cols.div_ceil(32),
    ];
    let shape = matmul::plan_in(
        [a.rows, a.cols, b.cols],
        SrcRoute::Bf16FromBf16,
        fidelity,
        Staging::Slots,
    )
    .ok_or_else(|| TensorError::Shape("no BF16 matmul block fits".into()))?;
    let kc = shape.tiles[1].min(k_limit.unwrap_or(kt));
    if kc < kt {
        return matmul_continuing(alloc, a, b, fidelity, kc);
    }
    let [mc, nc] = crate::tensor::blocks([mt, nt], [shape.tiles[0], shape.tiles[2]], units);
    let out = DramTensor::alloc(alloc, a.rows, b.cols)?;
    let mut jobs = Vec::new();
    for i0 in (0..mt).step_by(mc) {
        for j0 in (0..nt).step_by(nc) {
            let (rows, cols) = (mc.min(mt - i0), nc.min(nt - j0));
            let tiles = [rows, kt, cols];
            let layout = match matmul::plan_layout_in(tiles, L1Format::Bf16, Staging::Slots) {
                Ok(layout) => layout,
                Err(error) => {
                    alloc.free(&out.placement);
                    return Err(error.into());
                }
            };
            let gather = gather(
                a,
                b,
                [layout.a_at, layout.b_at],
                [i0, rows, j0, cols],
                0,
                kt,
            );
            let (roles, mop) = matmul::kernel_programs(
                tiles,
                0x90,
                SrcRoute::Bf16FromBf16,
                fidelity,
                layout.sems,
                false,
                || {
                    matmul::matmul_kernel(
                        &layout.outputs,
                        layout.sems,
                        L1Format::Bf16,
                        5,
                        fidelity,
                        false,
                    )
                },
            );
            let scatter = vec![
                [
                    record::SCATTER,
                    layout.outputs[0].out as u32,
                    tt_isa::dm::TILE_SLOT as u32,
                    i0 as u32,
                    rows as u32,
                    j0 as u32,
                    cols as u32,
                    0,
                ],
                out.tensor_ref().encode()[0],
                out.tensor_ref().encode()[1],
            ];
            jobs.push(vec![
                Step::List {
                    what: "packed BF16 matmul gather",
                    entries: gather,
                },
                Step::Kernel {
                    roles,
                    init: layout.init,
                    mop: Box::new(mop),
                    loops: Default::default(),
                    half: None,
                },
                Step::List {
                    what: "packed BF16 matmul scatter",
                    entries: scatter,
                },
            ]);
        }
    }
    out.set_pad(Pad::Undefined);
    Ok(Work { out, jobs })
}

fn matmul_continuing(
    alloc: &mut DramAlloc,
    a: &Bf16Tensor,
    b: &Bf16Tensor,
    fidelity: crate::matmul::Fidelity,
    kc: usize,
) -> Result<Work> {
    use crate::matmul::{self, SrcRoute};
    use tt_isa::dm::{record, TILE_SLOT};
    let route = SrcRoute::Bf16FromBf16;
    let kc = crate::tensor::k_block_size(kc, route, fidelity)?;
    let out = DramTensor::alloc(alloc, a.rows, b.cols)?;
    let build = || -> Result<Vec<Job>> {
        let mut jobs = Vec::new();
        let rc = out.tensor_ref().encode();
        for i in 0..a.rows.div_ceil(32) {
            for j in 0..b.cols.div_ceil(32) {
                let mut steps = Vec::new();
                for k0 in (0..a.cols.div_ceil(32)).step_by(kc) {
                    let count = kc.min(a.cols.div_ceil(32) - k0);
                    let (layout, reload) = matmul::k_block_layout(count)?;
                    let mut entries =
                        gather(a, b, [layout.a_at, layout.b_at], [i, 1, j, 1], k0, count);
                    if k0 > 0 {
                        // Waiting for NC release preserves the prior tile and
                        // makes the gather conservative for scheduler overlap.
                        entries.insert(0, [op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
                        entries.extend([
                            [
                                record::READ_RUN,
                                (i * b.cols.div_ceil(32) + j) as u32,
                                1,
                                reload.prior as u32,
                                0,
                                0,
                                0,
                                0,
                            ],
                            rc[0],
                            rc[1],
                        ]);
                    }
                    let (roles, _) = matmul::kernel_programs(
                        [1, count, 1],
                        if k0 > 0 { 0x93 } else { 0x92 },
                        route,
                        fidelity,
                        layout.sems,
                        false,
                        || {
                            (
                                matmul::k_block_kernel(
                                    &layout,
                                    (k0 > 0).then_some(reload),
                                    route,
                                    fidelity,
                                ),
                                [None; 3],
                            )
                        },
                    );
                    steps.extend([
                        Step::List {
                            what: "packed BF16 K-block gather",
                            entries,
                        },
                        Step::Kernel {
                            roles,
                            init: layout.init,
                            mop: Box::new([None; 3]),
                            loops: Default::default(),
                            half: None,
                        },
                        Step::List {
                            what: "packed BF16 K-block scatter",
                            entries: vec![
                                [
                                    record::SCATTER,
                                    layout.outputs[0].out as u32,
                                    TILE_SLOT as u32,
                                    i as u32,
                                    1,
                                    j as u32,
                                    1,
                                    0,
                                ],
                                rc[0],
                                rc[1],
                            ],
                        },
                    ]);
                }
                jobs.push(steps);
            }
        }
        Ok(jobs)
    };
    match build() {
        Ok(jobs) => {
            out.set_pad(Pad::Undefined);
            Ok(Work { out, jobs })
        }
        Err(error) => {
            alloc.free(&out.placement);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn quantization_preserves_zero_and_rounds_ties_even() {
        let values: Vec<_> = (0..1024)
            .map(|i| [0, 0x80000000, 0x3f808000, 0x3f818000, 0x7f800001][i % 5])
            .collect();
        let mut model = crate::sfpu::interp::Vector::new();
        model.put_tile(0, &values);
        model.run(&super::quantize()).unwrap();
        for (i, got) in model.tile(0).into_iter().enumerate() {
            assert_eq!(
                got,
                [0, 0x80000000, 0x3f800000, 0x3f820000, 0x7fc00000][i % 5],
                "{i}"
            );
        }
    }
}
