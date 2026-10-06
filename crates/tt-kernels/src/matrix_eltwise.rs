//! Opt-in matrix-unit elementwise arithmetic with reduced Src precision.
use crate::{
    datapath::{self, Unpacker},
    matmul::Fidelity,
    sfpu::kernel,
    tensor::{DramAlloc, DramTensor, Elem, Pad, Placement, Result, Step, TensorError, Work},
};
use std::sync::{Arc, Mutex, OnceLock};
use tt_isa::{
    backend::{self, Before, ConfigWords},
    cfg::generated::{alu, thread},
    dm::{op, record},
    isa::{generated::encode, Instruction},
    matrix::Banks,
    sync::{self, Unit},
};
pub use tt_isa::{matrix::SrcBroadcast, numerics::MatrixEltwiseOp};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SrcPrecision {
    Tf32,
    Bf16,
}
impl SrcPrecision {
    pub const fn code(self) -> u32 {
        match self {
            Self::Tf32 => 4,
            Self::Bf16 => 5,
        }
    }
}

/// Immutable engine selection. SFPU preserves the existing numerical contract.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum ElementwiseMode {
    #[default]
    Sfpu,
    Matrix {
        precision: SrcPrecision,
        fidelity: Fidelity,
    },
}

pub(crate) fn broadcast(a: [usize; 2], b: [usize; 2]) -> Result<SrcBroadcast> {
    if a == b {
        Ok(SrcBroadcast::None)
    } else if b == [1, 1] {
        Ok(SrcBroadcast::Scalar)
    } else if b == [1, a[1]] {
        Ok(SrcBroadcast::Row)
    } else if b == [a[0], 1] {
        Ok(SrcBroadcast::Column)
    } else {
        Err(TensorError::Shape(format!("matrix elementwise requires equal shapes or RHS row/column/scalar broadcast: {a:?}, {b:?}")))
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct Config {
    pub packed: bool,
    pub simulated: bool,
    pub precision: SrcPrecision,
    pub fidelity: Fidelity,
    pub op: MatrixEltwiseOp,
    pub broadcast: SrcBroadcast,
}

/// One physical tile, four face unpacks and eight aligned 8×16 consumers.
/// All state is established per invocation; four releases leave bank pointers
/// in lockstep. Dst is cleared before any multiplication phase.
fn roles(
    packed: bool,
    precision: SrcPrecision,
    fidelity: Fidelity,
    op: MatrixEltwiseOp,
    broadcast: SrcBroadcast,
    simulated: bool,
) -> Arc<[Vec<Instruction>; 3]> {
    type Key = (
        bool,
        SrcPrecision,
        Fidelity,
        MatrixEltwiseOp,
        SrcBroadcast,
        bool,
    );
    type Cache = std::collections::HashMap<Key, Arc<[Vec<Instruction>; 3]>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache
        .entry((packed, precision, fidelity, op, broadcast, simulated))
        .or_insert_with(|| {
            let l = kernel::plan_layout(1, kernel::Operands::Binary).unwrap();
            let sems = l.sems;
            let mut up = datapath::src_thread_config();
            up.extend(sync::take(sems.free, Before::UNPACKER));
            // Word 7 also holds both CLR_DVALID disable bits: writing it whole
            // establishes release behavior as well as the math Dst offset.
            let mut math = vec![
                datapath::state_id(),
                datapath::thread_entry(thread::DEST_TARGET_REG_CFG_MATH_Offset, 0),
            ];
            let mut config = ConfigWords::new();
            // ttsim refuses this register even at zero (divergence row 21)
            // and implicitly models the reset base. Silicon must write it.
            if !simulated {
                config.set(alu::DEST_REGW_BASE_Base, 0).unwrap();
            }
            config.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
            config
                .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
                .unwrap();
            math.extend(datapath::config_program(&config));
            math.extend(crate::matmul::math_prelude());
            let mut empty = Banks::after_reset();
            for face in 0..4 {
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
                let bytes = if packed { 2 } else { 4 };
                let mut words = ConfigWords::new();
                for (unpacker, base) in
                    [(Unpacker::SrcA, l.a_at), (Unpacker::SrcB, l.b_at.unwrap())]
                {
                    let source_face = if unpacker == Unpacker::SrcB {
                        match broadcast {
                            SrcBroadcast::None => face,
                            SrcBroadcast::Row => face % 2,
                            SrcBroadcast::Column => face / 2 * 2,
                            SrcBroadcast::Scalar => 0,
                        }
                    } else {
                        face
                    };
                    datapath::unpack_src_config(
                        &mut words,
                        unpacker,
                        datapath::flat_descriptor(256).with_in_data_format_raw(if packed {
                            5
                        } else {
                            0
                        }),
                        base + source_face * 256 * bytes,
                        precision.code(),
                    );
                }
                words.set(alu::ALU_ACC_CTRL_Fp32_enabled, 1).unwrap();
                words
                    .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
                    .unwrap();
                up.extend(datapath::config_program(&words));
                up.push(datapath::set_adc_x(Unpacker::SrcA, 0, 255));
                up.push(datapath::set_adc_x(Unpacker::SrcB, 0, 255));
                let (i, banks) = empty
                    .unpack_a(encode::UnpacrRegular::ZERO.multi_context_mode(1))
                    .unwrap();
                up.push(i);
                let (i, mut banks) = banks
                    .unpack_b(encode::UnpacrRegular::ZERO.multi_context_mode(1))
                    .unwrap();
                up.push(i);
                up.push(backend::wait_for_unpacker0(Before::EVERYTHING).unwrap());
                up.extend(sync::post_after(Unit::Unpacker1, sems.unpacked));
                math.extend(sync::take(sems.unpacked, Before::MATRIX));
                for half in 0..2 {
                    math.push(
                        encode::Setrwc::ZERO
                            .src_a(1)
                            .src_a_val(half * 8)
                            .src_b(1)
                            .src_b_val(if broadcast.row() != 0 { 0 } else { half * 8 })
                            .dst(1)
                            .dst_val(0)
                            .fidelity(1)
                            .encode()
                            .unwrap(),
                    );
                    let phases = if op == MatrixEltwiseOp::Mul {
                        fidelity.phases()
                    } else {
                        1
                    };
                    // The last consumer below releases both banks exactly once.
                    for _ in 0..phases - u32::from(half == 1) {
                        let dst = face as u32 * 16 + half * 8;
                        let (i, next) = match op {
                            MatrixEltwiseOp::Add => {
                                banks.elwadd(encode::Elwadd::ZERO.dst_row(dst), broadcast)
                            }
                            MatrixEltwiseOp::Sub => {
                                banks.elwsub(encode::Elwsub::ZERO.dst_row(dst), broadcast)
                            }
                            MatrixEltwiseOp::Mul => banks.elwmul(
                                encode::Elwmul::ZERO
                                    .dst_row(dst)
                                    .addr_mod(crate::matmul::MATH_AM_PHASE),
                                broadcast,
                            ),
                        }
                        .unwrap();
                        math.push(i);
                        banks = next;
                    }
                }
                let dst = face as u32 * 16 + 8;
                let (i, next) = match op {
                    MatrixEltwiseOp::Add => {
                        banks.elwadd_release_both(encode::Elwadd::ZERO.dst_row(dst), broadcast)
                    }
                    MatrixEltwiseOp::Sub => {
                        banks.elwsub_release_both(encode::Elwsub::ZERO.dst_row(dst), broadcast)
                    }
                    MatrixEltwiseOp::Mul => banks.elwmul_release_both(
                        encode::Elwmul::ZERO
                            .dst_row(dst)
                            .addr_mod(crate::matmul::MATH_AM_PHASE),
                        broadcast,
                    ),
                }
                .unwrap();
                math.push(i);
                empty = next;
            }
            math.extend(sync::post_after(Unit::Matrix, sems.computed));
            let mut pack = vec![datapath::state_id()];
            pack.extend(sync::take(sems.computed, Before::PACKER));
            pack.extend(datapath::pack_tile_from_dst(
                l.out_at + tt_isa::dm::TILE_DATA,
                0,
            ));
            pack.extend(sync::post_after(Unit::Packer, sems.free));
            Arc::new([up, math, pack])
        })
        .clone()
}

pub(crate) fn build(
    alloc: &mut DramAlloc,
    a: &Placement,
    b: &Placement,
    dims: [usize; 2],
    config: Config,
    units: usize,
) -> Result<Work> {
    let Config {
        packed,
        simulated,
        precision,
        fidelity,
        op,
        broadcast,
    } = config;
    let l = kernel::plan_layout(1, kernel::Operands::Binary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let roles = roles(packed, precision, fidelity, op, broadcast, simulated);
    let out = DramTensor::alloc_elem(alloc, dims[0], dims[1], Elem::F32)?;
    out.set_pad(Pad::Undefined);
    let mut jobs = vec![vec![]; units];
    for tile in 0..out.placement.tiles() {
        // Read the original RHS tile. Face and RWC selection in the unpack/math
        // streams then broadcasts its valid row/column/scalar directly in SrcB.
        // No expanded GDDR tensor or per-datum coordinate metadata is needed.
        let cols = dims[1].div_ceil(32);
        let rhs_tile = match broadcast {
            SrcBroadcast::None => tile,
            SrcBroadcast::Row => tile % cols,
            SrcBroadcast::Column => tile / cols,
            SrcBroadcast::Scalar => 0,
        };
        let gather = [(a, tile, l.a_at), (b, rhs_tile, l.b_at.unwrap())]
            .map(|(p, tile, at)| {
                let slot = p.slot(tile);
                [
                    op::READ,
                    slot.channel().index() as u32,
                    0,
                    slot.offset() as u32,
                    at as u32,
                    slot.len() as u32,
                    0,
                    0,
                ]
            })
            .to_vec();
        let mut scatter = vec![[
            record::WRITE_RUN,
            tile as u32,
            1,
            l.out_at as u32,
            0,
            0,
            0,
            0,
        ]];
        scatter.extend(out.tensor_ref().encode());
        jobs[tile % units].extend([
            Step::List {
                what: "matrix elementwise gather",
                entries: gather,
            },
            Step::Kernel {
                roles: roles.clone(),
                init: l.init.clone(),
                mop: Box::new([None; 3]),
                loops: Default::default(),
                half: None,
            },
            Step::List {
                what: "matrix elementwise scatter",
                entries: scatter,
            },
        ]);
    }
    Ok(Work { out, jobs })
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Audit submitted arithmetic: device numerical gates then establish
    /// execution. A route tag or absence of downloads alone is insufficient.
    #[test]
    fn cached_streams_issue_matrix_consumers_and_release_each_face_once() {
        for op in [
            MatrixEltwiseOp::Add,
            MatrixEltwiseOp::Sub,
            MatrixEltwiseOp::Mul,
        ] {
            for broadcast in [
                SrcBroadcast::None,
                SrcBroadcast::Row,
                SrcBroadcast::Column,
                SrcBroadcast::Scalar,
            ] {
                for precision in [SrcPrecision::Tf32, SrcPrecision::Bf16] {
                    for fidelity in [
                        Fidelity::Lo,
                        Fidelity::HiFi2,
                        Fidelity::HiFi3,
                        Fidelity::HiFi4,
                    ] {
                        let program = roles(false, precision, fidelity, op, broadcast, false);
                        assert!(Arc::ptr_eq(
                            &program,
                            &roles(false, precision, fidelity, op, broadcast, false)
                        ));
                        let math = &program[1];
                        let elw: Vec<_> = math
                            .iter()
                            .filter(|i| i.def().mnemonic().starts_with("ELW"))
                            .collect();
                        assert_eq!(
                            elw.len(),
                            8 * if op == MatrixEltwiseOp::Mul {
                                fidelity.phases() as usize
                            } else {
                                1
                            }
                        );
                        assert_eq!(elw.iter().filter(|i| i.word() & (3 << 22) != 0).count(), 4);
                        assert!(math.iter().all(|i| !i.def().mnemonic().starts_with("SFP")));
                        for i in elw {
                            assert_eq!((i.word() >> 19) & 1, broadcast.column());
                            assert_eq!((i.word() >> 20) & 1, broadcast.row());
                        }
                        assert!(math.iter().any(|i| i.def().mnemonic() == "ZEROACC"));
                    }
                }
            }
        }
    }
}
