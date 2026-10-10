//! The hazard checker (`tt_isa::hazard`) over the builders that are not public:
//! pooling, the Src transpose, rectangle copies and the Matrix Unit element-wise
//! kernels. Their role programs are carried in the `Step::Kernel`s of the `Work`
//! each builds, which needs no device. The public builders (matmul, SFPU
//! kernels, reductions, scans) are `tt-tests`'s `step121_wait_planner`.

use tt_isa::dram::Dram;
use tt_isa::hazard::{self, Kind};
use tt_isa::isa::Instruction;

use crate::code::Code;
use crate::fpu::PoolOp;
use crate::loops::frontend_stream;
use crate::matmul::{Fidelity, SrcRoute};
use crate::matrix_eltwise::{Config, MatrixChainTail, MatrixEltwiseOp, SrcBroadcast, SrcPrecision};
use crate::tensor::{DramAlloc, DramTensor, Elem, Step, Work};

/// Every kernel of `work`, each role as the backend receives it.
fn kernels(work: &Work) -> Vec<(usize, [Vec<Instruction>; 3])> {
    let mut out = Vec::new();
    for job in &work.jobs {
        for (n, step) in job.iter().enumerate() {
            if let Step::Kernel {
                roles, mop, loops, ..
            } = step
            {
                let streams = std::array::from_fn(|t| {
                    let code = Code {
                        ins: roles[t].clone(),
                        loops: loops[t].to_vec(),
                    };
                    frontend_stream(&code.expand(), mop[t].as_ref())
                        .expect("the frontend model accepts the program")
                });
                out.push((n, streams));
            }
        }
    }
    out
}

/// Check every role of every kernel of `work`; returns the kernels seen and the
/// misses, as text.
fn misses(label: &str, work: &Work) -> (usize, Vec<String>) {
    let ks = kernels(work);
    let mut bad = Vec::new();
    for (n, roles) in &ks {
        for (t, stream) in roles.iter().enumerate() {
            hazard::check(stream.iter().copied(), true, |f| {
                if f.kind.is_miss() {
                    bad.push(format!(
                        "{label} step {n} role {t}: {:?} at word {} ({}), unprotected {:?}",
                        f.kind, f.index, f.consumer, f.producer
                    ));
                }
            });
        }
    }
    (ks.len(), bad)
}

#[test]
fn the_pooling_copy_and_matrix_eltwise_builders_have_every_wait_their_hazards_need() {
    let mut seen = 0;
    let mut bad = Vec::new();
    let mut add = |label: &str, work: &Work| {
        let (n, b) = misses(label, work);
        seen += n;
        bad.extend(b);
    };

    // Pooling on the Matrix Unit, and the Src transpose.
    for op in [
        PoolOp::Max,
        PoolOp::MaxIndexProbe,
        PoolOp::Sum,
        PoolOp::Mean,
    ] {
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let a = DramTensor::alloc_elem(&mut alloc, 16, 16, Elem::F32).unwrap();
        let work = crate::fpu::block(&mut alloc, &a, op).unwrap();
        add(&format!("pool {op:?}"), &work);
    }
    for route in [SrcRoute::Tf32FromFp32, SrcRoute::Bf16FromFp32] {
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let a = DramTensor::alloc_elem(&mut alloc, 16, 16, Elem::F32).unwrap();
        let work = crate::fpu::transpose_block(&mut alloc, &a, route).unwrap();
        add(&format!("src transpose {route:?}"), &work);
    }

    // Rectangle copies through the ADC address generators.
    for (origin, dims) in [
        ([15, 16], [65, 80]),
        ([0, 0], [32, 32]),
        ([3, 32], [17, 48]),
    ] {
        let mut alloc = DramAlloc::new(&Dram::FULL);
        let source = DramTensor::alloc_elem(&mut alloc, 97, 99, Elem::F32).unwrap();
        let work = crate::adc_copy::build(&mut alloc, &source, origin, dims).unwrap();
        add(&format!("rect copy {origin:?} {dims:?}"), &work);
    }

    // Matrix Unit element-wise, every op, precision and chain tail.
    for first in [
        MatrixEltwiseOp::Add,
        MatrixEltwiseOp::Sub,
        MatrixEltwiseOp::Mul,
    ] {
        for tail in [
            None,
            Some(MatrixChainTail::WithRhs(MatrixEltwiseOp::Add)),
            Some(MatrixChainTail::WithLhs(MatrixEltwiseOp::Sub)),
            Some(MatrixChainTail::Square),
        ] {
            for precision in [SrcPrecision::Tf32, SrcPrecision::Bf16] {
                for (simulated, broadcast) in [
                    (false, SrcBroadcast::None),
                    (true, SrcBroadcast::None),
                    (false, SrcBroadcast::Row),
                    (false, SrcBroadcast::Column),
                    (false, SrcBroadcast::Scalar),
                ] {
                    let mut alloc = DramAlloc::new(&Dram::from_usable_mask(1));
                    let a = DramTensor::alloc(&mut alloc, 37, 65).unwrap();
                    let b = DramTensor::alloc(&mut alloc, 37, 65).unwrap();
                    let built = crate::matrix_eltwise::build(
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
                            broadcast,
                            tail,
                        },
                        2,
                    );
                    // Combinations the builder refuses (a broadcast with a
                    // tail, say) are not programs.
                    if let Ok(work) = built {
                        add(
                            &format!(
                                "elw {first:?} {tail:?} {precision:?} sim={simulated} {broadcast:?}"
                            ),
                            &work,
                        );
                    }
                }
            }
        }
    }

    assert!(seen > 20, "only {seen} kernels reached the checker");
    assert!(bad.is_empty(), "missed waits:\n{}", bad.join("\n"));
    let _ = Kind::EndsBusy;
}
