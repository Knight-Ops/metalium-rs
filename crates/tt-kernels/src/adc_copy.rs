//! Bit-preserving rectangular F32 copies using unpacker XY counters.
//! Column boundaries are whole 16-datum faces; rows may start anywhere.
use crate::{
    datapath,
    sfpu::kernel,
    tensor::{
        DramAlloc, DramTensor, Elem, OpPadding, Pad, PadNeed, Result, Step, TensorError, Work,
    },
};
use std::{collections::BTreeMap, sync::Arc};
use tt_isa::{
    adc::{self, Coordinates, Targets, Xy},
    backend::{self, Before, ConfigWords},
    cfg::generated::{alu, thcon, unpack1},
    dm::{record, TILE_DATA, TILE_SLOT},
    isa::{generated::encode, Instruction},
    sync::{self, Unit},
};

/// The copy reads logical datums only and initializes all destination padding.
pub struct RectPadding;
impl OpPadding for RectPadding {
    fn requires(&self, _: usize) -> PadNeed {
        PadNeed::Any
    }
    fn produces(&self, _: &[&DramTensor]) -> Pad {
        Pad::Zero
    }
}

#[derive(Clone, Copy, Debug)]
struct Run {
    tile: usize,
    face: usize,
    src_row: usize,
    dst_row: usize,
    rows: usize,
}

fn geometry(source: &DramTensor, origin: [usize; 2], dims: [usize; 2]) -> Result<()> {
    source.expect("ADC rectangle copy", Elem::F32)?;
    if dims.contains(&0)
        || origin[1] % 16 != 0
        || dims[1] % 16 != 0
        || origin[0]
            .checked_add(dims[0])
            .is_none_or(|n| n > source.rows)
        || origin[1]
            .checked_add(dims[1])
            .is_none_or(|n| n > source.cols)
        || dims[0].checked_mul(dims[1]).is_none()
    {
        return Err(TensorError::Shape(
            "ADC copy requires an in-bounds nonempty rectangle with 16-column boundaries".into(),
        ));
    }
    Ok(())
}

fn runs(source: &DramTensor, origin: [usize; 2], dims: [usize; 2], tile: usize) -> Vec<Run> {
    let ct = dims[1].div_ceil(32);
    let (out_r, out_c) = (tile / ct * 32, tile % ct * 32);
    let src_ct = source.grid()[1];
    let mut result = Vec::new();
    for face in 0..4 {
        let (r0, c0) = (out_r + face / 2 * 16, out_c + face % 2 * 16);
        if r0 >= dims[0] || c0 >= dims[1] {
            continue;
        }
        let mut row = 0;
        let height = 16.min(dims[0] - r0);
        while row < height {
            let (sr, sc) = (origin[0] + r0 + row, origin[1] + c0);
            let rows = (16 - sr % 16).min(height - row);
            result.push(Run {
                tile: sr / 32 * src_ct + sc / 32,
                face: sr % 32 / 16 * 2 + sc % 32 / 16,
                src_row: sr % 16,
                dst_row: face * 16 + row,
                rows,
            });
            row += rows;
        }
    }
    result
}

fn cursor_rows(program: &mut Vec<Instruction>, mut input: usize, mut output: usize) {
    while input != 0 || output != 0 {
        let (a, b) = (input.min(7), output.min(7));
        program.push(
            adc::advance_cursor(
                Targets::UNPACKER0,
                Coordinates::ROWS,
                Xy {
                    y0: a as u32,
                    y1: b as u32,
                    ..Xy::default()
                },
            )
            .unwrap(),
        );
        input -= a;
        output -= b;
    }
}

fn roles(
    layout: &kernel::Layout,
    sources: &BTreeMap<usize, usize>,
    runs: &[Run],
) -> [Vec<Instruction>; 3] {
    let sems = layout.sems;
    // Clear before any unpack. Reversing the usual math/unpack handoff is
    // intentional: computed means cleared, unpacked means ready to pack.
    let mut math = vec![datapath::state_id()];
    math.extend(sync::take(sems.free, Before::MATRIX));
    let mut words = ConfigWords::new();
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    math.extend(datapath::config_program(&words));
    math.push(encode::Setrwc::ZERO.dst(1).dst_val(0).encode().unwrap());
    math.push(encode::zeroacc(3, 0, 0, 0).unwrap());
    math.extend(sync::post_after(Unit::Matrix, sems.computed));

    let mut unpack = datapath::thread_config();
    unpack.extend(sync::take(sems.computed, Before::UNPACKER));
    for run in runs {
        unpack.push(backend::wait_for_unpacker0(Before::CONFIG).unwrap());
        let mut words = ConfigWords::new();
        let at = layout.a_at + sources[&run.tile] as u64 * TILE_SLOT + run.face as u64 * 1024;
        datapath::unpack_config(&mut words, datapath::flat_descriptor(16).with_y_dim(16), at);
        // Input is sixteen contiguous datums per source face row. Output
        // channel Y selects one sixteen-datum Dst row (byte stride 64).
        words
            .set(unpack1::UNP0_ADDR_CTRL_XY_REG_1_Ystride, 64)
            .unwrap();
        words
            .set(
                thcon::THCON_SEC0_REG5_Dest_cntx0_address,
                datapath::DST_BASE,
            )
            .unwrap();
        unpack.extend(datapath::config_program(&words));
        unpack.extend(datapath::clear_unpacker0_adcs());
        unpack.push(datapath::set_adc_x_unpack(0, 15));
        cursor_rows(&mut unpack, run.src_row, run.dst_row);
        let mut remaining = run.rows;
        while remaining != 0 {
            let count = remaining.min(7);
            for r in 0..count {
                unpack.push(datapath::unpack_instruction());
                // Counter mutation must follow retirement, not merely issue.
                unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
                if r + 1 < count {
                    unpack.push(
                        adc::increment(
                            Targets::UNPACKER0,
                            Xy {
                                y0: 1,
                                y1: 1,
                                ..Xy::default()
                            },
                        )
                        .unwrap(),
                    );
                }
            }
            remaining -= count;
            if remaining != 0 {
                // The live counters are anchor + count - 1. Restore them
                // from anchor + count, rather than adding count to live.
                cursor_rows(&mut unpack, count, count);
            }
        }
    }
    // Other Src programs reuse this thread's counters. Retire the final
    // unpack and restore the zero XY/ZW baseline before handing Dst off.
    unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    unpack.extend(datapath::clear_unpacker0_adcs());
    unpack.extend(sync::post_after(Unit::Unpacker0, sems.unpacked));
    let mut pack = vec![datapath::state_id()];
    pack.extend(sync::take(sems.unpacked, Before::EVERYTHING));
    pack.push(
        encode::Setadcxy::ZERO
            .pk(1)
            .x0(1)
            .x1(1)
            .y0(1)
            .y1(1)
            .encode()
            .unwrap(),
    );
    pack.push(
        encode::Setadczw::ZERO
            .pk(1)
            .z0(1)
            .z1(1)
            .w0(1)
            .w1(1)
            .encode()
            .unwrap(),
    );
    pack.extend(datapath::pack_tile_from_dst(layout.out_at + TILE_DATA, 0));
    pack.extend(sync::post_after(Unit::Packer, sems.free));
    [unpack, math, pack]
}

pub(crate) fn build(
    alloc: &mut DramAlloc,
    source: &DramTensor,
    origin: [usize; 2],
    dims: [usize; 2],
) -> Result<Work> {
    geometry(source, origin, dims)?;
    // An output can touch up to two source tile columns and two tile rows.
    let layout = kernel::plan_layout(4, kernel::Operands::Unary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let out = DramTensor::alloc_elem(alloc, dims[0], dims[1], Elem::F32)?;
    out.set_pad(RectPadding.produces(&[source]));
    let jobs = (0..out.grid()[0] * out.grid()[1])
        .map(|tile| {
            let runs = runs(source, origin, dims, tile);
            let mut sources = BTreeMap::new();
            for run in &runs {
                let index = sources.len();
                sources.entry(run.tile).or_insert(index);
            }
            let mut read = Vec::new();
            for (&src, &index) in &sources {
                read.push([
                    record::READ_RUN,
                    src as u32,
                    1,
                    (layout.a_at + index as u64 * TILE_SLOT) as u32,
                    0,
                    0,
                    0,
                    0,
                ]);
                read.extend(source.tensor_ref().encode());
            }
            let mut write = vec![[
                record::WRITE_RUN,
                tile as u32,
                1,
                layout.out_at as u32,
                0,
                0,
                0,
                0,
            ]];
            write.extend(out.tensor_ref().encode());
            vec![
                Step::List {
                    what: "ADC rectangle gather",
                    entries: read,
                },
                Step::Kernel {
                    roles: Arc::new(roles(&layout, &sources, &runs)),
                    init: layout.init.clone(),
                    mop: Box::new([None; 3]),
                    loops: Default::default(),
                    half: None,
                },
                Step::List {
                    what: "ADC rectangle scatter",
                    entries: write,
                },
            ]
        })
        .collect();
    Ok(Work { out, jobs })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangle_runs_cover_each_output_datum_and_audit_the_pipeline() {
        let mut alloc = DramAlloc::new(&tt_isa::dram::Dram::FULL);
        let source = DramTensor::alloc_elem(&mut alloc, 97, 99, Elem::F32).unwrap();
        let work = build(&mut alloc, &source, [15, 16], [65, 80]).unwrap();
        for (tile, job) in work.jobs.iter().enumerate() {
            let runs = runs(&source, [15, 16], [65, 80], tile);
            assert_eq!(
                runs.iter().map(|r| r.rows * 16).sum::<usize>(),
                (65 - tile / 3 * 32).min(32) * (80 - tile % 3 * 32).min(32)
            );
            let Step::List { entries, .. } = &job[0] else {
                panic!("gather")
            };
            assert!(entries.chunks_exact(3).all(|r| r[0][0] == record::READ_RUN));
            let Step::Kernel { roles, .. } = &job[1] else {
                panic!("kernel")
            };
            let names: Vec<_> = roles[0].iter().map(|i| i.def().mnemonic()).collect();
            if runs.iter().any(|r| r.rows > 1) {
                assert!(names.contains(&"INCADCXY"));
            }
            assert!(names.contains(&"ADDRCRXY"));
            assert!(!roles
                .iter()
                .flatten()
                .any(|i| i.def().mnemonic().starts_with("SFP")));
        }
    }
}

/// A logical row fragment, independently mapped across plane/tile boundaries.
#[derive(Clone, Copy, Debug)]
struct Fragment {
    tile: usize,
    word: usize,
    count: usize,
    slab: usize,
}

fn plane_geometry(
    source: &DramTensor,
    shape: [usize; 4],
    origin: [usize; 2],
    dims: [usize; 2],
) -> Result<[usize; 2]> {
    source.expect("ADC plane copy", Elem::F32)?;
    let [w, z, y, x] = shape;
    let rows = w.checked_mul(z).and_then(|n| n.checked_mul(y));
    let out_rows = dims[0].checked_mul(dims[1]).and_then(|n| n.checked_mul(y));
    if shape.contains(&0)
        || dims.contains(&0)
        || rows != Some(source.rows)
        || x != source.cols
        || origin[0].checked_add(dims[0]).is_none_or(|n| n > w)
        || origin[1].checked_add(dims[1]).is_none_or(|n| n > z)
        || out_rows.and_then(|n| n.checked_mul(x)).is_none()
    {
        return Err(TensorError::Shape(
            "ADC plane copy requires matching nonempty F32 storage and an in-bounds W/Z rectangle"
                .into(),
        ));
    }
    Ok([out_rows.unwrap(), x])
}

fn fragments(
    source: &DramTensor,
    shape: [usize; 4],
    origin: [usize; 2],
    dims: [usize; 2],
    tile: usize,
) -> Vec<Fragment> {
    let [_, z, y, x] = shape;
    let rows = dims[0] * dims[1] * y;
    let ct = x.div_ceil(32);
    let mut result = Vec::new();
    for slab in 0..64 {
        let face = slab / 16;
        let row = tile / ct * 32 + face / 2 * 16 + slab % 16;
        let col = tile % ct * 32 + face % 2 * 16;
        if row >= rows || col >= x {
            continue;
        }
        let plane = row / y;
        let sr = ((origin[0] + plane / dims[1]) * z + origin[1] + plane % dims[1]) * y + row % y;
        result.push(Fragment {
            tile: sr / 32 * source.grid()[1] + col / 32,
            word: (sr % 32 / 16 * 2 + col % 32 / 16) * 256 + sr % 16 * 16,
            count: (x - col).min(16),
            slab,
        });
    }
    result
}

fn plane_roles(layout: &kernel::Layout) -> [Vec<Instruction>; 3] {
    let mut roles = roles(layout, &BTreeMap::new(), &[]);
    let mut unpack = datapath::thread_config();
    unpack.extend(sync::take(layout.sems.computed, Before::UNPACKER));
    unpack.push(backend::wait_for_unpacker0(Before::CONFIG).unwrap());
    let mut words = ConfigWords::new();
    // Eight Z rows per W group. The descriptor supplies input byte strides;
    // output strides explicitly map the same microplanes to complete Dst rows.
    datapath::unpack_config(
        &mut words,
        datapath::flat_descriptor(16).with_z_dim(8),
        layout.a_at + TILE_SLOT,
    );
    words
        .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Zstride, 64)
        .unwrap();
    words
        .set(unpack1::UNP0_ADDR_CTRL_ZW_REG_1_Wstride, 512)
        .unwrap();
    unpack.extend(datapath::config_program(&words));
    unpack.extend(datapath::clear_unpacker0_adcs());
    unpack.push(datapath::set_adc_x_unpack(0, 15));
    for w in 0..8 {
        for z in 0..8 {
            unpack.push(datapath::unpack_instruction());
            unpack.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
            if z != 7 {
                unpack.push(
                    adc::increment_zw(
                        Targets::UNPACKER0,
                        adc::Zw {
                            z0: 1,
                            z1: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                );
            }
        }
        if w != 7 {
            unpack.push(
                adc::advance_cursor_zw(
                    Targets::UNPACKER0,
                    adc::ZwCoordinates::ALL,
                    adc::Zw {
                        w0: 1,
                        w1: 1,
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        }
    }
    unpack.extend(datapath::clear_unpacker0_adcs());
    unpack.extend(sync::post_after(Unit::Unpacker0, layout.sems.unpacked));
    roles[0] = unpack;
    roles
}

pub(crate) fn build_planes(
    alloc: &mut DramAlloc,
    source: &DramTensor,
    shape: [usize; 4],
    origin: [usize; 2],
    dims: [usize; 2],
) -> Result<Work> {
    use tt_isa::dm::{op, LIST_MAX};
    let [rows, cols] = plane_geometry(source, shape, origin, dims)?;
    // All three scratch slots, output slots and semaphores have overlapping
    // declared lifetimes. Scratch is bounded independently of global planes.
    let layout = kernel::plan_layout(3, kernel::Operands::Unary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let stage = layout.a_at + TILE_SLOT;
    let zero = stage + TILE_SLOT;
    let out = DramTensor::alloc_elem(alloc, rows, cols, Elem::F32)?;
    out.set_pad(RectPadding.produces(&[source]));
    let programs = Arc::new(plane_roles(&layout));
    let mut jobs = Vec::new();
    for tile in 0..out.grid()[0] * out.grid()[1] {
        let fs = fragments(source, shape, origin, dims, tile);
        let mut groups: BTreeMap<usize, Vec<Fragment>> = BTreeMap::new();
        for f in fs {
            groups.entry(f.tile).or_default().push(f);
        }
        let mut read = vec![
            [op::FILL, 0, 1 | (1 << 8), zero as u32, 0, 0, 0, 0],
            [
                op::COPY_WORDS,
                (zero + TILE_DATA + 4) as u32,
                (stage + TILE_DATA) as u32,
                1024,
                0,
                4,
                0,
                0,
            ],
        ];
        for (src, fs) in groups {
            read.push([
                record::READ_RUN,
                src as u32,
                1,
                layout.a_at as u32,
                0,
                0,
                0,
                0,
            ]);
            read.extend(source.tensor_ref().encode());
            for f in fs {
                read.push([
                    op::COPY_WORDS,
                    (layout.a_at + TILE_DATA + f.word as u64 * 4) as u32,
                    (stage + TILE_DATA + f.slab as u64 * 64) as u32,
                    f.count as u32,
                    4,
                    4,
                    0,
                    0,
                ]);
            }
        }
        // At most 64 source reads (three records each), 64 fragments and two
        // zeroing records: well within the existing mover list limit.
        assert!(read.len() <= LIST_MAX as usize);
        let mut write = vec![[
            record::WRITE_RUN,
            tile as u32,
            1,
            layout.out_at as u32,
            0,
            0,
            0,
            0,
        ]];
        write.extend(out.tensor_ref().encode());
        jobs.push(vec![
            Step::List {
                what: "ADC plane gather",
                entries: read,
            },
            Step::Kernel {
                roles: programs.clone(),
                init: layout.init.clone(),
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
            },
            Step::List {
                what: "ADC plane scatter",
                entries: write,
            },
        ]);
    }
    Ok(Work { out, jobs })
}

#[cfg(test)]
mod plane_tests {
    use super::*;
    #[test]
    fn plane_fragments_match_flat_selection_and_program_uses_both_instructions() {
        let mut alloc = DramAlloc::new(&tt_isa::dram::Dram::FULL);
        for y in [1, 17, 33] {
            for x in [1, 17, 33] {
                let source = DramTensor::alloc_elem(&mut alloc, 27 * y, x, Elem::F32).unwrap();
                let shape = [3, 9, y, x];
                let origin = [1, 2];
                let dims = [2, 7];
                let work = build_planes(&mut alloc, &source, shape, origin, dims).unwrap();
                let mut covered = vec![false; 14 * y * x];
                for (tile, job) in work.jobs.iter().enumerate() {
                    for f in fragments(&source, shape, origin, dims, tile) {
                        let face = f.slab / 16;
                        let row = tile / x.div_ceil(32) * 32 + face / 2 * 16 + f.slab % 16;
                        let col = tile % x.div_ceil(32) * 32 + face % 2 * 16;
                        let physical_row = f.tile / source.grid()[1] * 32
                            + (f.word / 256 / 2) * 16
                            + f.word % 256 / 16;
                        let physical_col = f.tile % source.grid()[1] * 32 + (f.word / 256 % 2) * 16;
                        let selected_plane = row / y;
                        let expected_row =
                            ((1 + selected_plane / 7) * 9 + 2 + selected_plane % 7) * y + row % y;
                        assert_eq!((physical_row, physical_col), (expected_row, col));
                        for c in col..col + f.count {
                            assert!(!covered[row * x + c]);
                            covered[row * x + c] = true;
                        }
                    }
                    let Step::Kernel { roles, .. } = &job[1] else {
                        panic!("kernel")
                    };
                    let names: Vec<_> = roles[0].iter().map(|i| i.def().mnemonic()).collect();
                    assert_eq!(names.iter().filter(|&&n| n == "INCADCZW").count(), 56);
                    assert_eq!(names.iter().filter(|&&n| n == "ADDRCRZW").count(), 7);
                    assert!(!roles
                        .iter()
                        .flatten()
                        .any(|i| i.def().mnemonic().starts_with("SFP")));
                }
                assert!(covered.into_iter().all(|v| v));
            }
        }
    }
}
