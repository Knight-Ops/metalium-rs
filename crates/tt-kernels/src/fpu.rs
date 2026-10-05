//! Explicit reduced-precision FPU pooling. These entry points never replace
//! an F32 SFPU reduction implicitly. GMPOOL is not an IEEE max instruction.
use std::sync::Arc;

use crate::{
    datapath::{self, Unpacker},
    sfpu::kernel,
    tensor::{DramAlloc, DramTensor, Elem, Pad, Placement, Result, Step, TensorError, Work},
};
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::alu,
    dm::{op, record},
    isa::generated::encode,
    matrix::Banks,
    sync::{self, Unit},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PoolOp {
    Max,
    Sum,
    Mean,
}

/// Pool one aligned 16x16 SrcA block over rows, producing 16 FP32 datums.
/// F32 inputs explicitly undergo BF16 Src truncation. Max uses exponent-only
/// unit scaling; sum/mean use four fidelity phases of GAPOOL with explicit
/// weights. This initial path refuses other shapes instead of ignoring ragged
/// lanes or silently changing general F32 reduction semantics.
pub(crate) fn block(alloc: &mut DramAlloc, a: &DramTensor, op: PoolOp) -> Result<Work> {
    a.expect("BF16 FPU pooling", Elem::F32)?;
    build(alloc, &a.placement, a.rows, a.cols, false, op)
}

/// GAPOOL sum blocks with FP32 continuation and explicit mean divisors.
/// Geometry is metadata; no tensor datum is read by the host.
pub(crate) fn windows<T: tt_device::Transport>(
    s: &mut crate::session::Session<T>,
    input: &crate::bf16::Bf16Tensor,
    windows: &[Vec<[usize; 2]>],
    divisors: &[usize],
    dims: [usize; 2],
) -> Result<DramTensor> {
    use crate::sfpu::ops::kind_sfpu;
    if dims[0].checked_mul(dims[1]) != Some(windows.len())
        || windows.is_empty()
        || divisors.len() != windows.len()
        || divisors.iter().any(|&n| n == 0 || n > 1 << 24)
        || windows
            .iter()
            .any(|w| w.is_empty() || w.iter().any(|&[r, c]| r >= input.rows || c >= input.cols))
    {
        return Err(TensorError::Shape("invalid BF16 pool windows".into()));
    }
    let mut temps = Vec::new();
    let result = (|| {
        let mut outputs = Vec::new();
        for (group_index, group) in windows.chunks(16).enumerate() {
            let mut acc: Option<DramTensor> = None;
            for first in (0..group.iter().map(Vec::len).max().unwrap()).step_by(16) {
                let sources: Vec<_> = (0..256)
                    .map(|i| {
                        group
                            .get(i % 16)
                            .and_then(|w| w.get(first + i / 16))
                            .copied()
                    })
                    .collect();
                let packed = s.repack_bf16_padded(input, &sources, [16, 16], 0)?;
                let partial = s.bf16_pool_block(&packed, PoolOp::Sum);
                let _ = s.free_bf16(packed);
                let partial = partial?;
                temps.push(partial.clone());
                acc = Some(if let Some(prior) = acc {
                    let out = s.eltwise(
                        crate::tensor::Eltwise {
                            kind: crate::kind::ADD,
                            scalar: 0.0,
                            scalar2: 0.0,
                        },
                        &prior,
                        Some(&partial),
                    )?;
                    temps.push(out.clone());
                    out
                } else {
                    partial
                });
            }
            let counts: Vec<_> = (0..16)
                .map(|i| divisors.get(group_index * 16 + i).copied().unwrap_or(1) as f32)
                .collect();
            let counts: Vec<_> = counts.into_iter().map(f32::to_bits).collect();
            let counts = s.metadata(&counts, [1, 16], Elem::F32)?;
            temps.push(counts.clone());
            let out = s.eltwise(
                crate::tensor::Eltwise {
                    kind: kind_sfpu::DIV,
                    scalar: 0.0,
                    scalar2: 0.0,
                },
                &acc.unwrap(),
                Some(&counts),
            )?;
            temps.push(out.clone());
            outputs.push(out);
        }
        let refs: Vec<_> = outputs.iter().collect();
        let rows: Vec<_> = (0..outputs.len()).map(|i| (i, 0)).collect();
        let joined = s.gather_rows(&refs, &rows, 16)?;
        temps.push(joined.clone());
        let sources: Vec<_> = (0..windows.len()).map(|i| [i / 16, i % 16]).collect();
        s.repack(&joined, &sources, dims)
    })();
    for t in temps {
        let _ = s.free(t);
    }
    result
}

pub(crate) fn bf16_block(
    alloc: &mut DramAlloc,
    a: &crate::bf16::Bf16Tensor,
    op: PoolOp,
) -> Result<Work> {
    build(alloc, &a.placement, a.rows, a.cols, true, op)
}

fn build(
    alloc: &mut DramAlloc,
    placement: &Placement,
    rows: usize,
    cols: usize,
    bf16: bool,
    op: PoolOp,
) -> Result<Work> {
    if [rows, cols] != [16, 16] {
        return Err(TensorError::Shape(
            "FPU block pooling requires [16,16]".into(),
        ));
    }
    let layout = kernel::plan_layout(1, kernel::Operands::Binary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let b_at = layout.b_at.expect("pool weights");
    let sems = layout.sems;
    let mut up = datapath::src_thread_config();
    // Prior matmuls leave face/context counters at their last tile. A flat
    // pooling run must establish both unpackers' complete ADC state.
    up.push(
        encode::Setadcxy::ZERO
            .u0(1)
            .u1(1)
            .x0(1)
            .x1(1)
            .y0(1)
            .y1(1)
            .encode()
            .unwrap(),
    );
    up.push(
        encode::Setadczw::ZERO
            .u0(1)
            .u1(1)
            .z0(1)
            .z1(1)
            .w0(1)
            .w1(1)
            .encode()
            .unwrap(),
    );
    up.extend(sync::take(sems.free, Before::UNPACKER));
    let mut words = ConfigWords::new();
    datapath::unpack_src_config(
        &mut words,
        Unpacker::SrcA,
        datapath::flat_descriptor(256).with_in_data_format_raw(if bf16 { 5 } else { 0 }),
        layout.a_at,
        5,
    );
    datapath::unpack_src_config(
        &mut words,
        Unpacker::SrcB,
        datapath::flat_descriptor(64),
        b_at + 16,
        5,
    );
    words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
    words
        .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
        .unwrap();
    up.extend(datapath::config_program(&words));
    up.push(datapath::set_adc_x(Unpacker::SrcA, 0, 255));
    up.push(datapath::set_adc_x(Unpacker::SrcB, 0, 63));
    let (i, banks) = Banks::after_reset()
        .unpack_a(encode::UnpacrRegular::ZERO.multi_context_mode(1))
        .unwrap();
    up.push(i);
    let (i, mut banks) = banks
        .unpack_b(encode::UnpacrRegular::ZERO.multi_context_mode(1))
        .unwrap();
    up.push(i);
    up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
    up.extend(sync::post_after(Unit::Unpacker1, sems.unpacked));

    let mut math = crate::matmul::math_prelude();
    math.extend(sync::take(sems.unpacked, Before::MATRIX));
    match op {
        PoolOp::Max => {
            let (i, next) = banks.gmpool(encode::Gmpool::ZERO).unwrap();
            math.push(i);
            banks = next;
        }
        PoolOp::Sum | PoolOp::Mean => {
            for _ in 0..4 {
                let (i, next) = banks.gapool(crate::matmul::MATH_AM_PHASE, 0).unwrap();
                math.push(i);
                banks = next;
            }
        }
    }
    math.push(backend::wait_for_matrix(Before::EVERYTHING).unwrap());
    let (i, banks) = banks.release_a().unwrap();
    math.push(i);
    let (i, _) = banks.release_b().unwrap();
    math.push(i);
    math.extend(sync::post_after(Unit::Matrix, sems.computed));
    let mut pack = vec![datapath::state_id()];
    pack.extend(sync::take(sems.computed, Before::PACKER));
    pack.extend(datapath::pack_tile_from_dst(
        layout.out_at + tt_isa::dm::TILE_DATA,
        0,
    ));
    pack.extend(sync::post_after(Unit::Packer, sems.free));
    let out = DramTensor::alloc(alloc, 1, 16)?;
    out.set_pad(Pad::Undefined);
    let slot = placement.slot(0);
    let mut gather = vec![[
        op::READ,
        slot.channel().index() as u32,
        0,
        slot.offset() as u32,
        layout.a_at as u32,
        slot.len() as u32,
        0,
        0,
    ]];
    let weight: f32 = if op == PoolOp::Mean { 1.0 / 16.0 } else { 1.0 };
    // FILL preserves its valid 1x1 datum; the unpack starts four datums
    // later, in the filled region, with a 16-byte aligned base.
    gather.push([
        op::FILL,
        weight.to_bits(),
        1 | (1 << 8),
        b_at as u32,
        0,
        0,
        0,
        0,
    ]);
    let mut scatter = vec![[record::WRITE_RUN, 0, 1, layout.out_at as u32, 0, 0, 0, 0]];
    scatter.extend(out.tensor_ref().encode());
    Ok(Work {
        out,
        jobs: vec![vec![
            Step::List {
                what: "FPU pool gather",
                entries: gather,
            },
            Step::Kernel {
                roles: Arc::new([up, math, pack]),
                init: layout.init,
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
            },
            Step::List {
                what: "FPU pool scatter",
                entries: scatter,
            },
        ]],
    })
}
