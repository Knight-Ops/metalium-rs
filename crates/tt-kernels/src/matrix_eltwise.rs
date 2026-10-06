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

/// Second stage of a resident matrix chain. Dst-to-Src conversion truncates
/// the intermediate to TF32 (10 fraction bits) or BF16 (7 fraction bits).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MatrixChainTail {
    /// `op(Q(first(a, b)), b)`.
    WithRhs(MatrixEltwiseOp),
    /// `op(a, Q(first(a, b)))`.
    WithLhs(MatrixEltwiseOp),
    /// `Q(first(a, b)) * Q(first(a, b))`.
    Square,
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
    pub tail: Option<MatrixChainTail>,
}

/// One physical tile, four face unpacks and eight aligned 8×16 consumers.
/// All state is established per invocation; each bank is released once per
/// face, leaving bank pointers in lockstep. Multiplication starts with clear Dst.
fn roles(
    packed: bool,
    precision: SrcPrecision,
    fidelity: Fidelity,
    op: MatrixEltwiseOp,
    broadcast: SrcBroadcast,
    simulated: bool,
    tail: Option<MatrixChainTail>,
) -> Arc<[Vec<Instruction>; 3]> {
    type Key = (
        bool,
        SrcPrecision,
        Fidelity,
        MatrixEltwiseOp,
        SrcBroadcast,
        bool,
        Option<MatrixChainTail>,
    );
    type Cache = std::collections::HashMap<Key, Arc<[Vec<Instruction>; 3]>>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache
        .entry((packed, precision, fidelity, op, broadcast, simulated, tail))
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
            if tail.is_some() {
                math.push(datapath::thread_entry(
                    thread::DISABLE_IMPLIED_SRCA_FMT_Base,
                    1,
                ));
                math.push(datapath::thread_entry(
                    thread::DISABLE_IMPLIED_SRCB_FMT_Base,
                    1,
                ));
                config
                    .set(alu::ALU_FORMAT_SPEC_REG_SrcA_override, 1)
                    .unwrap();
                config
                    .set(alu::ALU_FORMAT_SPEC_REG_SrcA_val, precision.code())
                    .unwrap();
                config
                    .set(alu::ALU_FORMAT_SPEC_REG_SrcB_override, 1)
                    .unwrap();
                config
                    .set(alu::ALU_FORMAT_SPEC_REG_SrcB_val, precision.code())
                    .unwrap();
            }
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
                if let Some(tail) = tail {
                    for half in 0..2 {
                        let dst = face as u32 * 16 + half * 8;
                        math.push(chain_rwc(half));
                        banks = emit_stage(&mut math, banks, op, fidelity, dst);
                        for row in [0, 4] {
                            let (i, next) = match tail {
                                MatrixChainTail::WithRhs(_) => banks.movd2a(
                                    encode::Movd2A::ZERO
                                        .move4_rows(1)
                                        .src_row(row)
                                        .dst_row(dst + row),
                                ),
                                _ => banks.movd2b(
                                    encode::Movd2B::ZERO
                                        .move4_rows(1)
                                        .src_row(row)
                                        .dst_row(dst + row),
                                ),
                            }
                            .unwrap();
                            math.push(i);
                            banks = next;
                        }
                        if tail == MatrixChainTail::Square {
                            for row in [0, 4] {
                                let (i, next) = banks.movb2a(row, 0, true, row).unwrap();
                                math.push(i);
                                banks = next;
                            }
                        }
                        // Only this half is cleared: earlier final rows stay live
                        // until the packer consumes the entire tile.
                        // Sixteen physical rows are eight FP32 rows. Avoid
                        // one-row ZEROACC's silicon physical-address behavior
                        // (divergence 51) while preserving earlier final halves.
                        math.push(encode::zeroacc(1, 0, 0, dst / 8).unwrap());
                        math.push(chain_rwc(half));
                        let second = match tail {
                            MatrixChainTail::WithRhs(op) | MatrixChainTail::WithLhs(op) => op,
                            MatrixChainTail::Square => MatrixEltwiseOp::Mul,
                        };
                        banks = emit_stage(&mut math, banks, second, fidelity, dst);
                    }
                    let (i, next) = banks.release_a().unwrap();
                    math.push(i);
                    let (i, next) = next.release_b().unwrap();
                    math.push(i);
                    empty = next;
                } else {
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
                    let (i, next) =
                        match op {
                            MatrixEltwiseOp::Add => banks
                                .elwadd_release_both(encode::Elwadd::ZERO.dst_row(dst), broadcast),
                            MatrixEltwiseOp::Sub => banks
                                .elwsub_release_both(encode::Elwsub::ZERO.dst_row(dst), broadcast),
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
            }
            if tail.is_some() {
                // Subsequent ordinary kernels use unpacker's implied formats.
                // Source ownership is already released; restore format selection
                // without changing DVALID or bank pointers.
                math.push(datapath::thread_entry(
                    thread::DISABLE_IMPLIED_SRCA_FMT_Base,
                    0,
                ));
                math.push(datapath::thread_entry(
                    thread::DISABLE_IMPLIED_SRCB_FMT_Base,
                    0,
                ));
                let mut defaults = ConfigWords::new();
                defaults
                    .set(alu::ALU_FORMAT_SPEC_REG_SrcA_override, 0)
                    .unwrap();
                defaults
                    .set(alu::ALU_FORMAT_SPEC_REG_SrcB_override, 0)
                    .unwrap();
                math.extend(datapath::config_program(&defaults));
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

fn chain_rwc(half: u32) -> Instruction {
    encode::Setrwc::ZERO
        .src_a(1)
        .src_a_val(half * 8)
        .src_b(1)
        .src_b_val(half * 8)
        .dst(1)
        .dst_val(0)
        .fidelity(1)
        .encode()
        .unwrap()
}

fn emit_stage(
    p: &mut Vec<Instruction>,
    mut banks: Banks<tt_isa::matrix::Loaded, tt_isa::matrix::Loaded>,
    op: MatrixEltwiseOp,
    fidelity: Fidelity,
    dst: u32,
) -> Banks<tt_isa::matrix::Loaded, tt_isa::matrix::Loaded> {
    for _ in 0..if op == MatrixEltwiseOp::Mul {
        fidelity.phases()
    } else {
        1
    } {
        let (i, next) = match op {
            MatrixEltwiseOp::Add => {
                banks.elwadd(encode::Elwadd::ZERO.dst_row(dst), SrcBroadcast::None)
            }
            MatrixEltwiseOp::Sub => {
                banks.elwsub(encode::Elwsub::ZERO.dst_row(dst), SrcBroadcast::None)
            }
            MatrixEltwiseOp::Mul => banks.elwmul(
                encode::Elwmul::ZERO
                    .dst_row(dst)
                    .addr_mod(crate::matmul::MATH_AM_PHASE),
                SrcBroadcast::None,
            ),
        }
        .unwrap();
        p.push(i);
        banks = next;
    }
    banks
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
        tail,
    } = config;
    let l = kernel::plan_layout(1, kernel::Operands::Binary)
        .map_err(|e| TensorError::Shape(e.to_string()))?;
    let roles = roles(packed, precision, fidelity, op, broadcast, simulated, tail);
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
    #[test]
    fn chain_streams_reuse_registers_and_submit_only_two_reads_one_write() {
        use tt_isa::dram::Dram;
        for first in [
            MatrixEltwiseOp::Add,
            MatrixEltwiseOp::Sub,
            MatrixEltwiseOp::Mul,
        ] {
            for tail in [
                MatrixChainTail::WithRhs(MatrixEltwiseOp::Add),
                MatrixChainTail::WithLhs(MatrixEltwiseOp::Sub),
                MatrixChainTail::Square,
            ] {
                for precision in [SrcPrecision::Tf32, SrcPrecision::Bf16] {
                    for simulated in [false, true] {
                        let program = roles(
                            false,
                            precision,
                            Fidelity::HiFi4,
                            first,
                            SrcBroadcast::None,
                            simulated,
                            Some(tail),
                        );
                        assert!(Arc::ptr_eq(
                            &program,
                            &roles(
                                false,
                                precision,
                                Fidelity::HiFi4,
                                first,
                                SrcBroadcast::None,
                                simulated,
                                Some(tail)
                            )
                        ));
                        let math = &program[1];
                        let count =
                            |name| math.iter().filter(|i| i.def().mnemonic() == name).count();
                        assert_eq!(
                            count("MOVD2A"),
                            if matches!(tail, MatrixChainTail::WithRhs(_)) {
                                16
                            } else {
                                0
                            }
                        );
                        assert_eq!(
                            count("MOVD2B"),
                            if matches!(tail, MatrixChainTail::WithRhs(_)) {
                                0
                            } else {
                                16
                            }
                        );
                        assert_eq!(
                            count("MOVB2A"),
                            if tail == MatrixChainTail::Square {
                                16
                            } else {
                                0
                            }
                        );
                        assert_eq!(count("ZEROACC"), 9);
                        assert!(math
                            .iter()
                            .filter(|i| i.def().mnemonic().starts_with("ELW"))
                            .all(|i| i.word() & (3 << 22) == 0));
                        assert_eq!(
                            math.iter()
                                .filter(|i| i.def().mnemonic() == "SETRWC"
                                    && i.word() & i.def().field("FlipSrcA").unwrap().place(1) != 0)
                                .count(),
                            4
                        );
                        assert_eq!(
                            math.iter()
                                .filter(|i| i.def().mnemonic() == "SETRWC"
                                    && i.word() & i.def().field("FlipSrcB").unwrap().place(1) != 0)
                                .count(),
                            4
                        );
                        assert!(math.iter().all(|i| !i.def().mnemonic().starts_with("SFP")));
                        let mut alloc = DramAlloc::new(&Dram::from_usable_mask(1));
                        let a = DramTensor::alloc(&mut alloc, 37, 65).unwrap();
                        let b = DramTensor::alloc(&mut alloc, 37, 65).unwrap();
                        let before = alloc.free_bytes();
                        let work = build(
                            &mut alloc,
                            &a.placement,
                            &b.placement,
                            [37, 65],
                            Config {
                                packed: false,
                                simulated,
                                precision,
                                fidelity: Fidelity::HiFi4,
                                op: first,
                                broadcast: SrcBroadcast::None,
                                tail: Some(tail),
                            },
                            2,
                        )
                        .unwrap();
                        assert_eq!(
                            before - alloc.free_bytes(),
                            work.out.placement.tiles() as u64 * tt_isa::dm::TILE_SLOT
                        );
                        for steps in &work.jobs {
                            for tile in steps.chunks_exact(3) {
                                let Step::List { entries, .. } = &tile[0] else {
                                    panic!("gather")
                                };
                                assert_eq!(entries.len(), 2);
                                assert!(entries.iter().all(|e| e[0] == op::READ));
                                assert!(matches!(tile[1], Step::Kernel { .. }));
                                let Step::List { entries, .. } = &tile[2] else {
                                    panic!("scatter")
                                };
                                assert_eq!(
                                    entries.iter().filter(|e| e[0] == record::WRITE_RUN).count(),
                                    1
                                );
                            }
                        }
                    }
                }
            }
        }
    }

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
                        let program = roles(false, precision, fidelity, op, broadcast, false, None);
                        assert!(Arc::ptr_eq(
                            &program,
                            &roles(false, precision, fidelity, op, broadcast, false, None)
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
