//! Explicit resident block-float storage. Conversion runs on Tensix; movers
//! only transfer bytes. Logical values are decoded F32, not a Burn dtype.
use crate::{
    datapath,
    sfpu::kernel,
    tensor::{DramAlloc, DramTensor, Elem, Job, Pad, Placement, Result, Step, TensorError, Work},
};
use std::sync::Arc;
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::{alu, thcon},
    dm::{op, TILE_DATA},
    dram::DramRange,
    isa::{generated::encode, Instruction},
    matrix::Banks,
    sync::{self, Unit},
    tile::{L1Format, TileImage},
};

/// Physical storage precision, ordered from lower to higher precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum BfpFormat {
    Bfp2,
    Bfp4,
    Bfp8,
}

impl BfpFormat {
    pub const fn l1_format(self) -> L1Format {
        match self {
            Self::Bfp8 => L1Format::Bfp8,
            Self::Bfp4 => L1Format::Bfp4,
            Self::Bfp2 => L1Format::Bfp2,
        }
    }
    pub fn tile_image(self) -> TileImage {
        TileImage::new(datapath::tile_descriptor(), self.l1_format())
            .expect("fixed BFP tile layout")
    }
    /// Includes header, exponent section, packed datums and 64-byte alignment.
    pub fn slot_bytes(self) -> u64 {
        (self.tile_image().total_bytes().div_ceil(64) * 64) as u64
    }
    fn code(self) -> u32 {
        self.l1_format().code().expect("measured BFP format")
    }
}

/// A matrix of resident packed tiles, each four 16x16 faces. Every consecutive
/// 16 face-ordered datums shares an exponent. Padding is explicitly zero.
/// Fields are private so shape/format cannot be changed without conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BfpTensor {
    rows: usize,
    cols: usize,
    format: BfpFormat,
    pub(crate) placement: Placement,
}

impl BfpTensor {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn format(&self) -> BfpFormat {
        self.format
    }
    pub fn pad(&self) -> Pad {
        Pad::Zero
    }
    pub fn tile_count(&self) -> usize {
        self.placement.tiles()
    }
    pub fn slot(&self, tile: usize) -> DramRange {
        self.placement.slot(tile)
    }
    /// Shares whole tile rows and their original exponent groups. Freeing the
    /// view does not free its parent; the parent must remain live while used.
    pub fn rows_view(&self, first: usize, rows: usize) -> Result<Self> {
        let end = first
            .checked_add(rows)
            .ok_or_else(|| TensorError::Shape("BFP row view overflow".into()))?;
        if rows == 0 || first % 32 != 0 || end > self.rows || (end % 32 != 0 && end != self.rows) {
            return Err(TensorError::Shape(
                "BFP row view requires whole tile rows".into(),
            ));
        }
        let ct = self.cols.div_ceil(32);
        Ok(Self {
            rows,
            cols: self.cols,
            format: self.format,
            placement: self
                .placement
                .borrowed_tiles(first / 32 * ct, rows.div_ceil(32) * ct),
        })
    }
}

fn transfer(range: DramRange, read: bool, offset: u64, l1: u64, bytes: u32) -> [u32; 8] {
    [
        if read { op::READ } else { op::WRITE },
        range.channel().index() as u32,
        0,
        (range.offset() + offset) as u32,
        l1 as u32,
        bytes,
        0,
        0,
    ]
}

fn roles(layout: &kernel::Layout, format: BfpFormat, compress: bool) -> [Vec<Instruction>; 3] {
    let s = layout.sems;
    let mut unpack = if compress {
        datapath::thread_config()
    } else {
        datapath::src_thread_config()
    };
    unpack.extend(datapath::clear_unpacker0_adcs());
    unpack.extend(sync::take(s.free, Before::UNPACKER));
    let mut words = ConfigWords::new();
    let mut math = crate::matmul::math_prelude();
    math.extend(sync::take(s.unpacked, Before::MATRIX));
    if compress {
        datapath::tile_unpack_config(&mut words, layout.a_at);
        unpack.extend(datapath::config_program(&words));
        unpack.extend(datapath::unpack_tile_to_dst(layout.a_at, 0));
    } else {
        datapath::unpack_src_config(
            &mut words,
            datapath::Unpacker::SrcA,
            datapath::flat_descriptor(128).with_in_data_format_raw(format.code()),
            layout.a_at,
            format.code(),
        );
        words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
        words
            .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 1)
            .unwrap();
        unpack.extend(datapath::config_program(&words));
        unpack.push(datapath::set_adc_x(datapath::Unpacker::SrcA, 0, 127));
        let (i, banks) = Banks::after_reset()
            .unpack_a(encode::UnpacrRegular::ZERO.multi_context_mode(1))
            .unwrap();
        unpack.push(i);
        let (i, banks) = banks.mova2d(encode::Mova2D::ZERO.move8_rows(1)).unwrap();
        math.push(i);
        math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
        let (i, _) = banks.release_a().unwrap();
        math.push(i);
    }
    unpack.extend(sync::post_after(Unit::Unpacker0, s.unpacked));
    math.extend(sync::post_after(Unit::Matrix, s.computed));
    let mut pack = vec![datapath::state_id()];
    pack.extend(sync::take(s.computed, Before::PACKER));
    pack.push(backend::wait_for_packer(Before::CONFIG).unwrap());
    let mut words = ConfigWords::new();
    datapath::pack_config(&mut words, layout.out_at + TILE_DATA);
    words.set(alu::ALU_ROUNDING_MODE_Packer_srnd_en, 0).unwrap();
    if compress {
        words
            .set(thcon::THCON_SEC0_REG1_Out_data_format, format.code())
            .unwrap();
        words
            .set(
                thcon::THCON_SEC0_REG1_Exp_section_size,
                (format.tile_image().exponent_section_bytes() / 16) as u32,
            )
            .unwrap();
    }
    pack.extend(datapath::config_program(&words));
    pack.extend(datapath::pack_rows(if compress { 64 } else { 8 }));
    pack.extend(sync::post_after(Unit::Packer, s.free));
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    pack.push(backend::wait_for_packer(Before::EVERYTHING).unwrap());
    [unpack, math, pack]
}

fn kernel_step(layout: &kernel::Layout, roles: &Arc<[Vec<Instruction>; 3]>) -> Step {
    Step::Kernel {
        roles: roles.clone(),
        init: layout.init.clone(),
        mop: Box::new([None; 3]),
        loops: Default::default(),
        half: None,
    }
}

pub(crate) fn compress(
    alloc: &mut DramAlloc,
    input: &DramTensor,
    format: BfpFormat,
) -> Result<(BfpTensor, Vec<Job>)> {
    input.expect("BFP conversion", Elem::F32)?;
    if input.pad() != Pad::Zero {
        return Err(TensorError::Shape(
            "BFP conversion requires zero padding".into(),
        ));
    }
    let layout = kernel::plan_layout(1, kernel::Operands::Unary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let roles = Arc::new(roles(&layout, format, true));
    let out = BfpTensor {
        rows: input.rows,
        cols: input.cols,
        format,
        placement: alloc.alloc_slots(input.placement.tiles(), format.slot_bytes())?,
    };
    let jobs = (0..input.placement.tiles())
        .map(|t| {
            vec![
                Step::List {
                    what: "BFP pack gather",
                    entries: vec![transfer(
                        input.slot(t),
                        true,
                        0,
                        layout.a_at,
                        tt_isa::dm::TILE_SLOT as u32,
                    )],
                },
                kernel_step(&layout, &roles),
                Step::List {
                    what: "BFP pack scatter",
                    entries: vec![transfer(
                        out.slot(t),
                        false,
                        TILE_DATA,
                        layout.out_at + TILE_DATA,
                        (format.tile_image().total_bytes() - TILE_DATA as usize) as u32,
                    )],
                },
            ]
        })
        .collect();
    Ok((out, jobs))
}

pub(crate) fn expand(alloc: &mut DramAlloc, input: &BfpTensor) -> Result<Work> {
    // The additional declared operand slot holds the exponent stream across
    // eight bounded Src bank conversions. No packed numerical work runs on B.
    let layout = kernel::plan_layout(1, kernel::Operands::Binary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let scratch = layout.b_at.expect("declared exponent and datum scratch");
    let exponents = scratch + 16;
    let roles = Arc::new(roles(&layout, input.format, false));
    let out = DramTensor::alloc(alloc, input.rows, input.cols)?;
    let chunk_bytes = 128 * input.format.l1_format().datum_bits() / 8;
    let mut jobs = Vec::new();
    for t in 0..input.tile_count() {
        let mut job = vec![Step::List {
            what: "BFP exponent gather",
            entries: vec![transfer(input.slot(t), true, TILE_DATA, exponents, 64)],
        }];
        for chunk in 0..8u64 {
            let offset =
                input.format.tile_image().datums_offset() as u64 + chunk * chunk_bytes as u64;
            // NoC reads require congruent GDDR/L1 low six address bits.
            // Read into declared scratch, then byte-preserving halfword copy
            // to the compact unpack image. Exponents remain live at +16..80.
            let datums = scratch + 128 + offset % 64;
            job.push(Step::List {
                what: "BFP unpack gather",
                entries: vec![
                    transfer(input.slot(t), true, offset, datums, chunk_bytes),
                    [
                        op::COPY_HALFWORDS,
                        datums as u32,
                        (layout.a_at + 32) as u32,
                        chunk_bytes / 2,
                        2,
                        2,
                        0,
                        0,
                    ],
                    [
                        op::COPY_HALFWORDS,
                        (exponents + chunk * 8) as u32,
                        (layout.a_at + 16) as u32,
                        4,
                        2,
                        2,
                        0,
                        0,
                    ],
                ],
            });
            job.push(kernel_step(&layout, &roles));
            job.push(Step::List {
                what: "BFP unpack scatter",
                entries: vec![transfer(
                    out.slot(t),
                    false,
                    TILE_DATA + chunk * 512,
                    layout.out_at + TILE_DATA,
                    512,
                )],
            });
        }
        jobs.push(job);
    }
    out.set_pad(Pad::Zero);
    Ok(Work { out, jobs })
}

/// Full exponent groups reach Src directly. Each output tile uses the existing
/// matrix face order and F32 accumulator reload between bounded K blocks.
pub(crate) fn matmul(
    alloc: &mut DramAlloc,
    a: &BfpTensor,
    b: &BfpTensor,
    fidelity: crate::matmul::Fidelity,
    k_limit: Option<usize>,
) -> Result<Work> {
    use crate::matmul::{self, SrcRoute};
    use tt_isa::dm::{record, TILE_SLOT};
    if a.format != b.format || a.cols != b.rows || [a.rows, a.cols, b.rows, b.cols].contains(&0) {
        return Err(TensorError::Shape(
            "packed BFP matmul requires compatible matrices of one format".into(),
        ));
    }
    let route = SrcRoute::Bfp(a.format);
    let kt = a.cols.div_ceil(32);
    let kc = crate::tensor::k_block_size(kt.min(k_limit.unwrap_or(8)), route, fidelity)?;
    let out = DramTensor::alloc(alloc, a.rows, b.cols)?;
    let build = || -> Result<Vec<Job>> {
        let mut jobs = Vec::new();
        let rc = out.tensor_ref().encode();
        for i in 0..a.rows.div_ceil(32) {
            for j in 0..b.cols.div_ceil(32) {
                let mut job = Vec::new();
                for k0 in (0..kt).step_by(kc) {
                    let count = kc.min(kt - k0);
                    let (layout, reload) = matmul::k_block_layout(count)?;
                    let mut reads = Vec::new();
                    if k0 > 0 {
                        reads.push([op::WAIT, 0, 0, 0, 0, 0, 0, 0]);
                        reads.extend([
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
                    for q in 0..count {
                        reads.push(transfer(
                            a.slot(i * kt + k0 + q),
                            true,
                            0,
                            layout.a_at + q as u64 * TILE_SLOT,
                            a.format.slot_bytes() as u32,
                        ));
                        reads.push(transfer(
                            b.slot((k0 + q) * b.cols.div_ceil(32) + j),
                            true,
                            0,
                            layout.b_at + q as u64 * TILE_SLOT,
                            b.format.slot_bytes() as u32,
                        ));
                    }
                    let (roles, _) = matmul::kernel_programs(
                        [1, count, 1],
                        if k0 > 0 { 0xb1 } else { 0xb0 },
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
                    job.extend([
                        Step::List {
                            what: "packed BFP matmul gather",
                            entries: reads,
                        },
                        Step::Kernel {
                            roles,
                            init: layout.init,
                            mop: Box::new([None; 3]),
                            loops: Default::default(),
                            half: None,
                        },
                        Step::List {
                            what: "packed BFP matmul scatter",
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
                jobs.push(job);
            }
        }
        Ok(jobs)
    };
    match build() {
        Ok(jobs) => {
            out.set_pad(Pad::Undefined);
            Ok(Work { out, jobs })
        }
        Err(e) => {
            alloc.free(&out.placement);
            Err(e)
        }
    }
}

pub(crate) fn copy_into(src: &BfpTensor, dst: &BfpTensor) -> Result<Vec<Job>> {
    if (src.rows, src.cols, src.format) != (dst.rows, dst.cols, dst.format) {
        return Err(TensorError::Shape(
            "BFP copy_into shapes or formats differ".into(),
        ));
    }
    let layout = kernel::plan_layout(1, kernel::Operands::Unary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    Ok((0..src.tile_count())
        .map(|t| {
            vec![Step::Transfer {
                what: "BFP physical copy",
                depth: 1,
                batches: vec![crate::tensor::TransferBatch {
                    read: vec![transfer(
                        src.slot(t),
                        true,
                        0,
                        layout.a_at,
                        src.format.slot_bytes() as u32,
                    )],
                    write: vec![transfer(
                        dst.slot(t),
                        false,
                        0,
                        layout.a_at,
                        dst.format.slot_bytes() as u32,
                    )],
                }],
            }]
        })
        .collect())
}
