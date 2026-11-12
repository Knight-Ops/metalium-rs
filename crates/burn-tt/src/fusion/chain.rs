//! A fuser for linear chains of operations: Burn compositions that reach the stream as a run of
//! small ops, each reading the one before (Burn's default `sigmoid`, `silu(gate) * up`).
//!
//! A [`Pattern`] is a list of [`Step`]s and the step counts at which it is complete. The
//! [`ChainFuser`] follows the stream while it matches and keeps the longest complete match, so
//! `sigmoid`, `silu` and `silu(gate) * up` are one pattern with three ends. A match is only
//! complete when every tensor it made, other than its result, dies inside it: read for the last
//! time (`ReadWrite`) or dropped. A tensor read again later would otherwise have no handle,
//! since the optimization registers only its result.
//!
//! Scalars arrive as ids, bound only at execution (a plan is cached and reused with other
//! scalars), so a match records each with what it must be; [`ChainMatch::bind`] checks them.

use burn_backend::DType;
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, CastOpIr, FloatOperationIr, NumericOperationIr, OperationIr,
    ReduceDimOpIr, ScalarIr, ScalarOpIr, TensorId, TensorIr, TensorStatus, UnaryOpIr,
};

use crate::ops::norm::{fused_eps, FUSED_ID};
use serde::{Deserialize, Serialize};

use super::TtOptimization;

/// An operand of a binary step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arg {
    /// The previous step's result.
    Chain,
    /// Step `k`'s result (from 0), an earlier step than the previous one.
    Step(usize),
    /// The `k`th tensor from outside the chain, bound where it is first read. Input 0 is what
    /// the first step reads.
    Input(usize),
}

/// What a scalar must be.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expect {
    /// Exactly this value (as Burn binds it, an `f64`).
    Exact(f64),
    /// Any value; the optimization reads it.
    Bind,
}

/// One operation of a pattern. Unary steps read the chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Step {
    /// A cast to F32.
    CastF32,
    /// A cast back to input 0's dtype.
    CastBack,
    MulScalar(Expect),
    AddScalar(Expect),
    Exp,
    Log,
    Sqrt,
    /// The mean over the last dimension.
    MeanLast,
    Mul(Arg, Arg),
    Add(Arg, Arg),
    Div(Arg, Arg),
    /// `burn_tt::rms_norm`'s custom operation (`x`, `gamma`): RMSNorm with its scale, `eps`
    /// bound from the operation's id.
    ScaledNorm(Arg, Arg),
}

/// What a complete match computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainKind {
    /// Burn's default `sigmoid(x)`.
    Sigmoid,
    /// `x * sigmoid(x)`, Burn's `silu`.
    Silu,
    /// `silu(x) * y`.
    SiluMul,
    /// Burn's `RmsNorm` without its scale: `x / sqrt(mean(x x) + eps)` over the last dimension.
    RmsNorm,
    /// `a b + c d`, both products dying in the sum (RoPE's `x cos + rotate(x) sin`).
    MulAdd,
    /// `s = a + b` and Burn's `RmsNorm` of `s` up to its scale, `s` kept: a residual sum and the
    /// norm after it.
    AddRmsNorm,
    /// `s = a + b` and `burn_tt::rms_norm(s, gamma)`, `s` kept.
    AddRmsNormScaled,
}

impl ChainKind {
    /// The name burn-fusion and the fusion counters know it by.
    pub fn name(self) -> &'static str {
        match self {
            ChainKind::Sigmoid => "tt_sigmoid",
            ChainKind::Silu => "tt_silu",
            ChainKind::SiluMul => "tt_silu_mul",
            ChainKind::RmsNorm => "tt_rms_norm",
            ChainKind::MulAdd => "tt_mul_add",
            ChainKind::AddRmsNorm => "tt_add_rms_norm",
            ChainKind::AddRmsNormScaled => "tt_add_rms_norm_scaled",
        }
    }
}

/// A chain and where it may end.
#[derive(Debug)]
pub(crate) struct Pattern {
    pub steps: &'static [Step],
    /// `(steps matched, what the match is)`, shortest first.
    pub ends: &'static [(usize, ChainKind)],
    /// Steps whose results may be read after the match: registered with its result
    /// ([`ChainMatch::kept`]). Every other tensor it makes must die inside it.
    pub keep: &'static [usize],
}

/// Burn's default `sigmoid` (`burn-backend` `ActivationOps::sigmoid`), which burn-fusion 0.21 does
/// not forward to the backend: `exp(-log(exp(-x) + 1))` in F32 between two casts (`float_neg`
/// is `mul_scalar(-1)`). Then `silu` (`x * sigmoid(x)`) and `silu(x) * y`, in Burn's operand
/// order. A chain written by hand through the tensor API has no casts, and is left alone.
pub(crate) const SILU_MUL: Pattern = Pattern {
    steps: &[
        Step::CastF32,
        Step::MulScalar(Expect::Exact(-1.0)),
        Step::Exp,
        Step::AddScalar(Expect::Exact(1.0)),
        Step::Log,
        Step::MulScalar(Expect::Exact(-1.0)),
        Step::Exp,
        Step::CastBack,
        Step::Mul(Arg::Input(0), Arg::Chain),
        Step::Mul(Arg::Chain, Arg::Input(1)),
    ],
    ends: &[
        (8, ChainKind::Sigmoid),
        (9, ChainKind::Silu),
        (10, ChainKind::SiluMul),
    ],
    keep: &[],
};

/// Burn's `RmsNorm::forward` (`burn-nn` `norm/rms.rs`) on F32, up to its scale: `x.cast(F32)` and
/// `rms.cast(dtype)` are no ops then (`Tensor::cast` skips a cast to the same dtype), and
/// `square` is `x * x`. The scale is not part of it: its `unsqueeze` is a `Reshape` of a tensor
/// the chain never read, which burn-fusion puts in a block of its own, and the product by it then
/// joins neither block (observed with `BURN_FUSION_LOG=full`).
pub(crate) const RMS_NORM: Pattern = Pattern {
    steps: &[
        Step::Mul(Arg::Input(0), Arg::Input(0)),
        Step::MeanLast,
        Step::AddScalar(Expect::Bind),
        Step::Sqrt,
        Step::Div(Arg::Input(0), Arg::Chain),
    ],
    ends: &[(5, ChainKind::RmsNorm)],
    keep: &[],
};

/// `a b + c d`, each product read only by the sum: RoPE's `x cos + cat(x2, x1) sin`, which
/// reaches the stream as `Mul(x, cos)`, `Mul(rotated, sin)`, `Add(first, second)` (the `cat`
/// before them ends the block before). The first product runs as itself and the second with
/// the sum as one pass of `MUL_ADD` (`a b` its `c`): one op fewer, bit for bit.
pub(crate) const MUL_ADD: Pattern = Pattern {
    steps: &[
        Step::Mul(Arg::Input(0), Arg::Input(1)),
        Step::Mul(Arg::Input(2), Arg::Input(3)),
        Step::Add(Arg::Step(0), Arg::Chain),
    ],
    ends: &[(3, ChainKind::MulAdd)],
    keep: &[],
};

/// A residual sum and Burn's `RmsNorm` of it up to its scale ([`RMS_NORM`] reading the sum),
/// the sum kept for the next residual: one op (`Session::add_rms_norm`) where the sum and the
/// norm were two. Any pre-norm model's `x + f(x)` then norm.
pub(crate) const ADD_RMS_NORM: Pattern = Pattern {
    steps: &[
        Step::Add(Arg::Input(0), Arg::Input(1)),
        Step::Mul(Arg::Chain, Arg::Chain),
        Step::MeanLast,
        Step::AddScalar(Expect::Bind),
        Step::Sqrt,
        Step::Div(Arg::Step(0), Arg::Chain),
    ],
    ends: &[(6, ChainKind::AddRmsNorm)],
    keep: &[0],
};

/// A residual sum and `burn_tt::rms_norm` of it (the norm with its scale, a model's call), the
/// sum kept.
pub(crate) const ADD_RMS_NORM_SCALED: Pattern = Pattern {
    steps: &[
        Step::Add(Arg::Input(0), Arg::Input(1)),
        Step::ScaledNorm(Arg::Chain, Arg::Input(2)),
    ],
    ends: &[(2, ChainKind::AddRmsNormScaled)],
    keep: &[0],
};

/// Most operations a block may hold before a chain begins. burn-fusion groups an operation
/// with the block of the tensors it reads, and a block lasts while any fuser is open: a residual
/// `Add` keeps `ElementWiseFuser` open (it may be `add_relu`), so the norm that reads the sum
/// joins the `Add`'s block, one operation in. Only such an operation is passed over
/// ([`opens_another_fuser`]): passing over any other would keep blocks open that end today, and
/// a fuser that must begin a block (`ElementWiseFuser` on a matmul's bias `Add`) would never
/// begin one.
pub(crate) const MAX_LEAD: usize = 1;

/// The operations [`super::eltwise::ElementWiseFuser`] stays open on as a block's first.
fn opens_another_fuser(operation: &OperationIr) -> bool {
    matches!(
        operation,
        OperationIr::NumericFloat(
            _,
            NumericOperationIr::Add(_) | NumericOperationIr::LowerEqualElem(_)
        )
    )
}

/// A scalar operand as the fuser saw it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ScalarRef {
    /// A relative id, bound at execution.
    Id(u64),
    /// A literal value.
    Value(f64),
}

/// A complete match: what [`TtOptimization::Chain`] executes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainMatch {
    pub kind: ChainKind,
    /// Each input as last read inside the match (its status says whether the match consumes it).
    pub inputs: Vec<TensorIr>,
    pub out: TensorIr,
    /// The results of the pattern's [`Pattern::keep`] steps, read after the match.
    pub kept: Vec<TensorIr>,
    pub scalars: Vec<(ScalarRef, Expect)>,
    /// Positions of the `Drop`s inside the match, run after the result is registered.
    pub drops: Vec<usize>,
    /// Operations before the chain, in the same block: run as themselves, first.
    pub lead: usize,
    pub num_ops: usize,
}

impl ChainMatch {
    /// The values of the bound ([`Expect::Bind`]) scalars, in order, or `None` when an exact one
    /// differs (or is missing): the chain is then not the pattern, and runs as its operations.
    /// `lookup` finds a relative id's value (`Context::scalars`).
    pub fn bind(&self, lookup: impl Fn(u64) -> Option<ScalarIr>) -> Option<Vec<f64>> {
        let mut bound = Vec::new();
        for (scalar, expect) in &self.scalars {
            let value = match scalar {
                ScalarRef::Value(v) => *v,
                ScalarRef::Id(id) => match lookup(*id)? {
                    ScalarIr::Float(v) => v,
                    _ => return None,
                },
            };
            match expect {
                Expect::Exact(e) if value.to_bits() != e.to_bits() => return None,
                Expect::Exact(_) => {}
                Expect::Bind => bound.push(value),
            }
        }
        Some(bound)
    }
}

/// The match in progress.
#[derive(Clone, Debug, Default)]
struct Progress {
    /// Steps matched.
    steps: usize,
    /// Operations taken, `Drop`s included.
    ops: usize,
    /// The last step's result.
    chain: Option<TensorIr>,
    /// Each input as last read.
    inputs: Vec<TensorIr>,
    /// The tensors the chain made, a step each, and whether each has died inside it.
    made: Vec<(TensorId, bool)>,
    /// The same tensors as made.
    outs: Vec<TensorIr>,
    scalars: Vec<(ScalarRef, Expect)>,
    drops: Vec<usize>,
    /// Operations taken before the chain's first step ([`MAX_LEAD`]).
    lead: usize,
}

/// Fuser for one [`Pattern`].
#[derive(Clone, Debug)]
pub(crate) struct ChainFuser {
    pattern: &'static Pattern,
    progress: Progress,
    /// The longest complete match so far.
    best: Option<ChainMatch>,
    /// No further operation can extend the match.
    closed: bool,
}

impl ChainFuser {
    pub(crate) fn new(pattern: &'static Pattern) -> Self {
        Self {
            pattern,
            progress: Progress::default(),
            best: None,
            closed: false,
        }
    }

    /// Takes `operation` as the next step, or returns `false` (the progress is then stale and
    /// the fuser closes).
    fn step(&mut self, operation: &OperationIr) -> bool {
        let p = &mut self.progress;
        if let OperationIr::Drop(dropped) = operation {
            if let Some(made) = p.made.iter_mut().find(|(id, _)| *id == dropped.id) {
                made.1 = true;
            } else if !p.inputs.iter().any(|t| t.id == dropped.id) {
                return false;
            }
            p.drops.push(p.ops);
            p.ops += 1;
            return true;
        }
        let Some(step) = self.pattern.steps.get(p.steps).copied() else {
            return false;
        };
        let out = match (step, operation) {
            (
                Step::CastF32,
                OperationIr::BaseFloat(BaseOperationIr::Cast(CastOpIr { input, out })),
            ) if input.dtype.is_float() && out.dtype == DType::F32 => {
                if !p.read(Arg::Chain, input) {
                    return false;
                }
                out
            }
            (
                Step::CastBack,
                OperationIr::BaseFloat(BaseOperationIr::Cast(CastOpIr { input, out })),
            ) if p.inputs.first().is_some_and(|x| x.dtype == out.dtype) => {
                if !p.read(Arg::Chain, input) {
                    return false;
                }
                out
            }
            (
                Step::MulScalar(expect),
                OperationIr::NumericFloat(
                    _,
                    NumericOperationIr::MulScalar(ScalarOpIr { lhs, rhs, out }),
                ),
            )
            | (
                Step::AddScalar(expect),
                OperationIr::NumericFloat(
                    _,
                    NumericOperationIr::AddScalar(ScalarOpIr { lhs, rhs, out }),
                ),
            ) => {
                let scalar = match rhs {
                    ScalarIr::UInt(id) => ScalarRef::Id(*id),
                    ScalarIr::Float(v) => ScalarRef::Value(*v),
                    _ => return false,
                };
                if !p.read(Arg::Chain, lhs) {
                    return false;
                }
                p.scalars.push((scalar, expect));
                out
            }
            (Step::Exp, OperationIr::Float(_, FloatOperationIr::Exp(UnaryOpIr { input, out })))
            | (Step::Log, OperationIr::Float(_, FloatOperationIr::Log(UnaryOpIr { input, out })))
            | (
                Step::Sqrt,
                OperationIr::Float(_, FloatOperationIr::Sqrt(UnaryOpIr { input, out })),
            ) => {
                if !p.read(Arg::Chain, input) {
                    return false;
                }
                out
            }
            (
                Step::MeanLast,
                OperationIr::NumericFloat(
                    _,
                    NumericOperationIr::MeanDim(ReduceDimOpIr {
                        input, out, axis, ..
                    }),
                ),
            ) if *axis + 1 == input.shape.rank() => {
                if !p.read(Arg::Chain, input) {
                    return false;
                }
                out
            }
            (
                Step::Mul(a, b),
                OperationIr::NumericFloat(_, NumericOperationIr::Mul(BinaryOpIr { lhs, rhs, out })),
            )
            | (
                Step::Add(a, b),
                OperationIr::NumericFloat(_, NumericOperationIr::Add(BinaryOpIr { lhs, rhs, out })),
            )
            | (
                Step::Div(a, b),
                OperationIr::NumericFloat(_, NumericOperationIr::Div(BinaryOpIr { lhs, rhs, out })),
            ) => {
                if !p.read(a, lhs) || !p.read(b, rhs) {
                    return false;
                }
                out
            }
            (Step::ScaledNorm(x, gamma), OperationIr::Custom(desc))
                if desc.id.starts_with(FUSED_ID)
                    && desc.inputs.len() == 2
                    && desc.outputs.len() == 1 =>
            {
                let Some(eps) = fused_eps(&desc.id) else {
                    return false;
                };
                if !p.read(x, &desc.inputs[0]) || !p.read(gamma, &desc.inputs[1]) {
                    return false;
                }
                p.scalars.push((ScalarRef::Value(eps as f64), Expect::Bind));
                &desc.outputs[0]
            }
            _ => return false,
        };
        p.made.push((out.id, false));
        p.outs.push(out.clone());
        p.chain = Some(out.clone());
        p.steps += 1;
        p.ops += 1;
        true
    }

    /// Records a complete match if the pattern ends here and everything else the chain made
    /// is dead.
    fn record_end(&mut self) {
        let p = &self.progress;
        let Some(&(_, kind)) = self.pattern.ends.iter().find(|(n, _)| *n == p.steps) else {
            return;
        };
        let Some(out) = &p.chain else { return };
        let keep = self.pattern.keep;
        if p.made
            .iter()
            .enumerate()
            .any(|(k, (id, dead))| *id != out.id && !dead && !keep.contains(&k))
        {
            return;
        }
        self.best = Some(ChainMatch {
            kind,
            inputs: p.inputs.clone(),
            out: out.clone(),
            kept: keep.iter().map(|&k| p.outs[k].clone()).collect(),
            scalars: p.scalars.clone(),
            drops: p.drops.clone(),
            lead: p.lead,
            num_ops: p.ops,
        });
    }
}

impl Progress {
    /// Reads `tensor` as `arg`, or returns `false` when it is not that operand.
    fn read(&mut self, arg: Arg, tensor: &TensorIr) -> bool {
        let arg = match (arg, &self.chain) {
            // The first step's chain operand is input 0.
            (Arg::Chain, None) => Arg::Input(0),
            (arg, _) => arg,
        };
        match arg {
            Arg::Chain => {
                if self.chain.as_ref().map(|c| c.id) != Some(tensor.id) {
                    return false;
                }
            }
            Arg::Step(k) => {
                if self.made.get(k).map(|(id, _)| *id) != Some(tensor.id) {
                    return false;
                }
            }
            Arg::Input(k) => {
                if k < self.inputs.len() {
                    if self.inputs[k].id != tensor.id {
                        return false;
                    }
                    self.inputs[k] = tensor.clone();
                } else if k == self.inputs.len()
                    && !self.made.iter().any(|(id, _)| *id == tensor.id)
                    && !self.inputs.iter().any(|t| t.id == tensor.id)
                {
                    self.inputs.push(tensor.clone());
                } else {
                    return false;
                }
                return true;
            }
        }
        if tensor.status == TensorStatus::ReadWrite {
            if let Some(made) = self.made.iter_mut().find(|(id, _)| *id == tensor.id) {
                made.1 = true;
            }
        }
        true
    }
}

impl OperationFuser<TtOptimization> for ChainFuser {
    fn fuse(&mut self, operation: &OperationIr) {
        if self.closed {
            return;
        }
        if self.step(operation) {
            self.record_end();
            if self.progress.steps == self.pattern.steps.len() {
                self.closed = true;
            }
        } else if self.progress.steps == 0
            && self.progress.lead < MAX_LEAD
            && opens_another_fuser(operation)
        {
            // Not begun: the operation runs as itself, before the chain (a step may have bound
            // an operand before refusing, so the progress starts again).
            let (lead, ops) = (self.progress.lead + 1, self.progress.ops + 1);
            self.progress = Progress {
                lead,
                ops,
                ..Progress::default()
            };
        } else {
            self.closed = true;
        }
    }

    fn finish(&mut self) -> TtOptimization {
        TtOptimization::Chain(
            self.best
                .take()
                .expect("finish called on a fuser with no match"),
        )
    }

    fn reset(&mut self) {
        self.progress = Progress::default();
        self.best = None;
        self.closed = false;
    }

    fn status(&self) -> FuserStatus {
        if self.closed {
            FuserStatus::Closed
        } else {
            FuserStatus::Open
        }
    }

    fn properties(&self) -> FuserProperties {
        match &self.best {
            // Above `ElementWiseFuser`'s 5 a fused op: a chain replaces more dispatches. The
            // operations before it are not fused, and do not count.
            Some(m) => FuserProperties {
                score: ((m.num_ops - m.lead) * 10) as u64,
                ready: true,
            },
            None => FuserProperties {
                score: 0,
                ready: false,
            },
        }
    }

    fn len(&self) -> usize {
        match &self.best {
            Some(m) => m.num_ops,
            None => self.progress.ops,
        }
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<TtOptimization>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::Shape;

    fn t(id: u64, status: TensorStatus, dtype: DType) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            shape: Shape::from(vec![2, 3]),
            status,
            dtype,
        }
    }
    const RW: TensorStatus = TensorStatus::ReadWrite;
    const RO: TensorStatus = TensorStatus::ReadOnly;
    const NEW: TensorStatus = TensorStatus::NotInit;
    const F32: DType = DType::F32;

    fn cast(input: TensorIr, out: TensorIr) -> OperationIr {
        OperationIr::BaseFloat(BaseOperationIr::Cast(CastOpIr { input, out }))
    }
    fn scalar(add: bool, lhs: TensorIr, id: u64, out: TensorIr) -> OperationIr {
        let op = ScalarOpIr {
            lhs,
            rhs: ScalarIr::UInt(id),
            out,
        };
        OperationIr::NumericFloat(
            F32,
            if add {
                NumericOperationIr::AddScalar(op)
            } else {
                NumericOperationIr::MulScalar(op)
            },
        )
    }
    fn unary(log: bool, input: TensorIr, out: TensorIr) -> OperationIr {
        let op = UnaryOpIr { input, out };
        OperationIr::Float(
            F32,
            if log {
                FloatOperationIr::Log(op)
            } else {
                FloatOperationIr::Exp(op)
            },
        )
    }
    fn mul(lhs: TensorIr, rhs: TensorIr, out: TensorIr) -> OperationIr {
        OperationIr::NumericFloat(F32, NumericOperationIr::Mul(BinaryOpIr { lhs, rhs, out }))
    }

    /// Burn's default sigmoid of tensor 1 (read `x_status`) as burn-fusion records it, ending
    /// in tensor 18; `keep` leaves one intermediate (`exp(-x)`, tensor 12) read-only.
    fn sigmoid(x_status: TensorStatus, dtype: DType, keep: bool) -> Vec<OperationIr> {
        let mid = |id| t(id, if keep && id == 12 { RO } else { RW }, F32);
        vec![
            cast(t(1, x_status, dtype), t(10, NEW, F32)),
            scalar(false, mid(10), 0, t(11, NEW, F32)),
            unary(false, mid(11), t(12, NEW, F32)),
            scalar(true, mid(12), 1, t(13, NEW, F32)),
            unary(true, mid(13), t(14, NEW, F32)),
            scalar(false, mid(14), 2, t(15, NEW, F32)),
            unary(false, mid(15), t(16, NEW, F32)),
            cast(mid(16), t(18, NEW, dtype)),
        ]
    }

    fn run(ops: &[OperationIr]) -> ChainFuser {
        let mut f = ChainFuser::new(&SILU_MUL);
        for op in ops {
            f.fuse(op);
        }
        f
    }

    #[test]
    fn burns_sigmoid_matches_with_its_scalars() {
        let f = run(&sigmoid(RW, F32, false));
        assert!(f.properties().ready);
        assert_eq!(f.len(), 8);
        let m = f.best.clone().unwrap();
        assert_eq!(m.kind, ChainKind::Sigmoid);
        assert_eq!(m.inputs[0].id, TensorId::new(1));
        assert_eq!(m.out.id, TensorId::new(18));
        assert_eq!(
            m.scalars,
            vec![
                (ScalarRef::Id(0), Expect::Exact(-1.0)),
                (ScalarRef::Id(1), Expect::Exact(1.0)),
                (ScalarRef::Id(2), Expect::Exact(-1.0)),
            ]
        );
        // A BF16 input: the casts are to F32 and back.
        assert!(run(&sigmoid(RW, DType::BF16, false)).properties().ready);
    }

    #[test]
    fn silu_mul_is_the_longest_end_and_a_later_op_is_ignored() {
        let mut ops = sigmoid(RO, F32, false);
        ops.push(mul(t(1, RW, F32), t(18, RW, F32), t(19, NEW, F32)));
        let f = run(&ops);
        assert_eq!(f.best.as_ref().unwrap().kind, ChainKind::Silu);
        assert!(f.status() == FuserStatus::Open);
        ops.push(mul(t(19, RW, F32), t(2, RO, F32), t(20, NEW, F32)));
        ops.push(unary(false, t(20, RW, F32), t(21, NEW, F32)));
        let f = run(&ops);
        let m = f.best.clone().unwrap();
        assert_eq!((m.kind, m.num_ops), (ChainKind::SiluMul, 10));
        assert_eq!(
            m.inputs.iter().map(|t| t.id.value()).collect::<Vec<_>>(),
            [1, 2]
        );
        // `x` as last read: the match consumes it.
        assert_eq!(m.inputs[0].status, RW);
        assert!(f.status() == FuserStatus::Closed);
    }

    /// RoPE's `x cos + rotate(x) sin` after its `cat`: `Mul(x, cos)`, `Mul(rotated, sin)`,
    /// `Add(first, second)` is `MulAdd` over the four inputs in that order. A product read
    /// again later, or the sum's operands the other way round, is not.
    #[test]
    fn a_sum_of_two_products_is_mul_add() {
        let add = |lhs, rhs, out| {
            OperationIr::NumericFloat(F32, NumericOperationIr::Add(BinaryOpIr { lhs, rhs, out }))
        };
        let fuse = |ops: &[OperationIr]| {
            let mut f = ChainFuser::new(&MUL_ADD);
            for op in ops {
                f.fuse(op);
            }
            f
        };
        let rope = |first: TensorStatus, swap: bool| {
            let (l, r) = (t(10, first, F32), t(11, RW, F32));
            vec![
                mul(t(1, RW, F32), t(2, RO, F32), t(10, NEW, F32)),
                mul(t(3, RW, F32), t(4, RO, F32), t(11, NEW, F32)),
                if swap {
                    add(r, l, t(12, NEW, F32))
                } else {
                    add(l, r, t(12, NEW, F32))
                },
            ]
        };
        let f = fuse(&rope(RW, false));
        let m = f.best.clone().unwrap();
        assert_eq!((m.kind, m.num_ops), (ChainKind::MulAdd, 3));
        assert_eq!(
            m.inputs.iter().map(|t| t.id.value()).collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
        assert_eq!(m.out.id, TensorId::new(12));
        assert!(
            fuse(&rope(RO, false)).best.is_none(),
            "the first product read again"
        );
        assert!(
            fuse(&rope(RW, true)).best.is_none(),
            "the sum the other way round"
        );
    }

    #[test]
    fn an_intermediate_read_after_the_chain_is_not_fused() {
        let f = run(&sigmoid(RW, F32, true));
        assert!(!f.properties().ready);
        // Unless it is dropped inside the chain.
        let mut ops = sigmoid(RW, F32, true);
        ops.insert(4, OperationIr::Drop(t(12, RW, F32)));
        let f = run(&ops);
        let m = f.best.clone().unwrap();
        assert_eq!((m.num_ops, m.drops.clone()), (9, vec![4]));
    }

    #[test]
    fn a_silu_whose_sigmoid_is_kept_fuses_only_the_sigmoid() {
        let mut ops = sigmoid(RO, F32, false);
        ops.push(mul(t(1, RW, F32), t(18, RO, F32), t(19, NEW, F32)));
        let f = run(&ops);
        assert_eq!(f.best.as_ref().unwrap().kind, ChainKind::Sigmoid);
        assert_eq!(f.len(), 8);
    }

    #[test]
    fn near_misses_are_not_matched() {
        // No casts: a chain written through the tensor API.
        assert!(!run(&sigmoid(RW, F32, false)[1..7]).properties().ready);
        // A step reading something other than the chain.
        let mut ops = sigmoid(RW, F32, false);
        ops[4] = unary(true, t(12, RW, F32), t(14, NEW, F32));
        assert!(!run(&ops).properties().ready);
        // The swapped `silu(x)` operand order and a `Mul` by another tensor.
        for wrong in [
            mul(t(18, RW, F32), t(1, RW, F32), t(19, NEW, F32)),
            mul(t(3, RW, F32), t(18, RW, F32), t(19, NEW, F32)),
        ] {
            let mut ops = sigmoid(RO, F32, false);
            ops.push(wrong);
            assert_eq!(run(&ops).best.unwrap().kind, ChainKind::Sigmoid);
        }
        // `up * silu(x)`.
        let mut ops = sigmoid(RO, F32, false);
        ops.push(mul(t(1, RW, F32), t(18, RW, F32), t(19, NEW, F32)));
        ops.push(mul(t(2, RO, F32), t(19, RW, F32), t(20, NEW, F32)));
        assert_eq!(run(&ops).best.unwrap().kind, ChainKind::Silu);
        // A drop of an unrelated tensor ends the match where it is.
        let mut ops = sigmoid(RW, F32, false);
        ops.insert(3, OperationIr::Drop(t(7, RW, F32)));
        assert!(!run(&ops).properties().ready);
    }

    /// A chain one operation into its block (a residual `Add` whose sum it reads): the `Add`
    /// runs first, as itself, and only the chain scores. Past [`MAX_LEAD`], or after any other
    /// operation, nothing matches.
    #[test]
    fn a_chain_may_begin_a_few_operations_in() {
        let add = |out: u64| {
            OperationIr::NumericFloat(
                F32,
                NumericOperationIr::Add(BinaryOpIr {
                    lhs: t(30, RW, F32),
                    rhs: t(31, RW, F32),
                    out: t(out, NEW, F32),
                }),
            )
        };
        let mut ops = vec![add(1)];
        ops.extend(sigmoid(RW, F32, false));
        let f = run(&ops);
        let m = f.best.clone().unwrap();
        assert_eq!((m.kind, m.lead, m.num_ops), (ChainKind::Sigmoid, 1, 9));
        assert_eq!(f.properties().score, 80);
        let mut ops: Vec<_> = (40..41 + MAX_LEAD as u64).map(add).collect();
        ops.extend(sigmoid(RW, F32, false));
        assert!(!run(&ops).properties().ready);
        // Nor after an operation no other fuser stays open on: the block ends there today, and
        // must, so that the next operation begins one (a matmul's bias `Add` for `add_relu`).
        let mut ops = vec![unary(false, t(30, RW, F32), t(1, NEW, F32))];
        ops.extend(sigmoid(RW, F32, false));
        let f = run(&ops);
        assert!(!f.properties().ready && f.status() == FuserStatus::Closed);
    }

    #[test]
    fn exact_scalars_are_checked_at_binding() {
        let m = run(&sigmoid(RW, F32, false)).best.unwrap();
        let bind = |v: [f64; 3]| m.bind(|id| v.get(id as usize).map(|v| ScalarIr::Float(*v)));
        assert_eq!(bind([-1.0, 1.0, -1.0]), Some(vec![]));
        assert_eq!(bind([-1.0, 2.0, -1.0]), None);
        assert_eq!(bind([-1.0, 1.0, -1.0 - f64::EPSILON]), None);
        assert_eq!(m.bind(|_| None), None);
        assert_eq!(m.bind(|_| Some(ScalarIr::Int(-1))), None);
    }
}
