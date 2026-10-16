//! Element-wise operation fusers for Tenstorrent Blackhole.
//!
//! Fuses compound element-wise patterns such as:
//! - `add(lhs, rhs) -> tmp` followed by `relu(tmp) -> out` into a single compound
//!   Tensix SFPU dispatch `float_add_relu(lhs, rhs)`.
//! - `relu(input) -> out` (lowered by Burn into `lower_equal_elem(0.0)` + `mask_fill(0.0)`)
//!   into a direct device `activation::relu(input)` kernel call.

use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, MaskFillOpIr, NumericOperationIr, OperationIr, ScalarIr,
    ScalarOpIr, TensorIr,
};

use super::TtOptimization;

/// Fuser for element-wise operations on Tenstorrent Blackhole.
#[derive(Clone, Debug, Default)]
pub struct ElementWiseFuser {
    state: FuserState,
}

#[derive(Clone, Debug, Default)]
enum FuserState {
    #[default]
    Empty,
    Add {
        lhs: TensorIr,
        rhs: TensorIr,
        out: TensorIr,
        num_ops: usize,
    },
    AddLowerEqual {
        lhs: TensorIr,
        rhs: TensorIr,
        add_out: TensorIr,
        mask: TensorIr,
        zeros: Vec<u64>,
        num_ops: usize,
    },
    LowerEqual {
        input: TensorIr,
        mask: TensorIr,
        zeros: Vec<u64>,
        num_ops: usize,
    },
    AddReluMatched {
        lhs: TensorIr,
        rhs: TensorIr,
        out: TensorIr,
        zeros: Vec<u64>,
        num_ops: usize,
    },
    ReluMatched {
        input: TensorIr,
        out: TensorIr,
        zeros: Vec<u64>,
        num_ops: usize,
    },
    Closed {
        num_ops: usize,
    },
}

impl OperationFuser<TtOptimization> for ElementWiseFuser {
    fn fuse(&mut self, operation: &OperationIr) {
        // burn-fusion keeps offering a block's operations while any other fuser is open; a
        // match (or a miss) is final, so a later operation must not undo it.
        if matches!(self.status(), FuserStatus::Closed) {
            return;
        }
        match operation {
            OperationIr::NumericFloat(
                _dtype,
                NumericOperationIr::Add(BinaryOpIr { lhs, rhs, out }),
            ) => match self.state {
                FuserState::Empty => {
                    self.state = FuserState::Add {
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        out: out.clone(),
                        num_ops: 1,
                    };
                }
                _ => {
                    self.state = FuserState::Closed {
                        num_ops: self.len() + 1,
                    };
                }
            },
            OperationIr::NumericFloat(
                _dtype,
                NumericOperationIr::LowerEqualElem(ScalarOpIr {
                    lhs: input,
                    rhs: scalar,
                    out,
                }),
            ) if zero_scalar_ids(scalar).is_some() => match &self.state {
                FuserState::Add {
                    lhs,
                    rhs,
                    out: add_out,
                    num_ops,
                } if input.id == add_out.id => {
                    self.state = FuserState::AddLowerEqual {
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        add_out: add_out.clone(),
                        mask: out.clone(),
                        zeros: zero_scalar_ids(scalar).unwrap_or_default(),
                        num_ops: num_ops + 1,
                    };
                }
                FuserState::Empty => {
                    self.state = FuserState::LowerEqual {
                        input: input.clone(),
                        mask: out.clone(),
                        zeros: zero_scalar_ids(scalar).unwrap_or_default(),
                        num_ops: 1,
                    };
                }
                _ => {
                    self.state = FuserState::Closed {
                        num_ops: self.len() + 1,
                    };
                }
            },
            OperationIr::BaseFloat(BaseOperationIr::MaskFill(MaskFillOpIr {
                tensor,
                mask,
                value,
                out,
            })) if zero_scalar_ids(value).is_some() => match &self.state {
                FuserState::AddLowerEqual {
                    lhs,
                    rhs,
                    add_out,
                    mask: expected_mask,
                    zeros,
                    num_ops,
                } if tensor.id == add_out.id && mask.id == expected_mask.id => {
                    self.state = FuserState::AddReluMatched {
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        out: out.clone(),
                        zeros: with_ids(zeros, value),
                        num_ops: num_ops + 1,
                    };
                }
                FuserState::LowerEqual {
                    input,
                    mask: expected_mask,
                    zeros,
                    num_ops,
                } if tensor.id == input.id && mask.id == expected_mask.id => {
                    self.state = FuserState::ReluMatched {
                        input: input.clone(),
                        out: out.clone(),
                        zeros: with_ids(zeros, value),
                        num_ops: num_ops + 1,
                    };
                }
                _ => {
                    self.state = FuserState::Closed {
                        num_ops: self.len() + 1,
                    };
                }
            },
            OperationIr::Drop(dropped) => match &mut self.state {
                FuserState::AddLowerEqual {
                    add_out,
                    mask,
                    num_ops,
                    ..
                } if dropped.id == add_out.id || dropped.id == mask.id => {
                    *num_ops += 1;
                }
                FuserState::LowerEqual { mask, num_ops, .. } if dropped.id == mask.id => {
                    *num_ops += 1;
                }
                _ => {
                    self.state = FuserState::Closed {
                        num_ops: self.len() + 1,
                    };
                }
            },
            _ => {
                self.state = FuserState::Closed {
                    num_ops: self.len() + 1,
                };
            }
        }
    }

    fn finish(&mut self) -> TtOptimization {
        match std::mem::take(&mut self.state) {
            FuserState::AddReluMatched {
                lhs,
                rhs,
                out,
                zeros,
                num_ops,
            } => TtOptimization::AddRelu {
                lhs,
                rhs,
                out,
                zeros,
                num_ops,
            },
            FuserState::ReluMatched {
                input,
                out,
                zeros,
                num_ops,
            } => TtOptimization::Relu {
                input,
                out,
                zeros,
                num_ops,
            },
            _ => panic!("finish called on non-ready fuser"),
        }
    }

    fn reset(&mut self) {
        self.state = FuserState::Empty;
    }

    fn status(&self) -> FuserStatus {
        match self.state {
            FuserState::Empty
            | FuserState::Add { .. }
            | FuserState::AddLowerEqual { .. }
            | FuserState::LowerEqual { .. } => FuserStatus::Open,
            FuserState::AddReluMatched { .. }
            | FuserState::ReluMatched { .. }
            | FuserState::Closed { .. } => FuserStatus::Closed,
        }
    }

    fn properties(&self) -> FuserProperties {
        match self.state {
            FuserState::AddReluMatched { num_ops, .. } => FuserProperties {
                score: (num_ops * 10) as u64,
                ready: true,
            },
            FuserState::ReluMatched { num_ops, .. } => FuserProperties {
                score: (num_ops * 5) as u64,
                ready: true,
            },
            _ => FuserProperties {
                score: 0,
                ready: false,
            },
        }
    }

    fn len(&self) -> usize {
        match self.state {
            FuserState::Empty => 0,
            FuserState::Add { num_ops, .. }
            | FuserState::AddLowerEqual { num_ops, .. }
            | FuserState::LowerEqual { num_ops, .. }
            | FuserState::AddReluMatched { num_ops, .. }
            | FuserState::ReluMatched { num_ops, .. }
            | FuserState::Closed { num_ops, .. } => num_ops,
        }
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<TtOptimization>> {
        Box::new(self.clone())
    }
}

/// The runtime scalar ids a match assumes to be `+0.0`, or `None` when the scalar cannot be.
///
/// burn-fusion hands a fuser every scalar as `ScalarIr::UInt(id)`: the value is bound only when
/// the plan executes (a cached plan is reused for the same graph with other scalars). A match
/// therefore cannot decide `relu` against `thresholded_relu(alpha)` or `mask_fill(.., -5.0)`;
/// the optimization records the ids and checks them at execution (`TtOptimization::execute`).
fn zero_scalar_ids(s: &ScalarIr) -> Option<Vec<u64>> {
    match s {
        ScalarIr::Float(v) if v.to_bits() == 0 => Some(Vec::new()),
        ScalarIr::UInt(id) => Some(vec![*id]),
        _ => None,
    }
}

fn with_ids(ids: &[u64], scalar: &ScalarIr) -> Vec<u64> {
    let mut ids = ids.to_vec();
    ids.extend(zero_scalar_ids(scalar).unwrap_or_default());
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DType, Shape};
    use burn_ir::{FloatOperationIr, TensorId, TensorStatus, UnaryOpIr};

    fn t(id: u64, status: TensorStatus) -> TensorIr {
        TensorIr {
            id: TensorId::new(id),
            shape: Shape::from(vec![4]),
            status,
            dtype: DType::F32,
        }
    }

    /// burn-fusion offers a block's later operations to a fuser that has closed while another
    /// fuser is still open; a matched relu must stay matched.
    #[test]
    fn a_match_survives_the_operations_after_it() {
        use TensorStatus::{NotInit, ReadOnly, ReadWrite};
        let mut f = ElementWiseFuser::default();
        let ops = [
            OperationIr::NumericFloat(
                DType::F32,
                NumericOperationIr::Add(BinaryOpIr {
                    lhs: t(1, ReadWrite),
                    rhs: t(2, ReadWrite),
                    out: t(3, NotInit),
                }),
            ),
            OperationIr::NumericFloat(
                DType::F32,
                NumericOperationIr::LowerEqualElem(ScalarOpIr {
                    lhs: t(3, ReadOnly),
                    rhs: ScalarIr::UInt(0),
                    out: t(4, NotInit),
                }),
            ),
            OperationIr::BaseFloat(BaseOperationIr::MaskFill(MaskFillOpIr {
                tensor: t(3, ReadWrite),
                mask: t(4, ReadWrite),
                value: ScalarIr::UInt(1),
                out: t(5, NotInit),
            })),
        ];
        for op in &ops {
            f.fuse(op);
        }
        assert!(f.properties().ready);
        f.fuse(&OperationIr::Float(
            DType::F32,
            FloatOperationIr::Exp(UnaryOpIr {
                input: t(5, ReadWrite),
                out: t(6, NotInit),
            }),
        ));
        assert!(f.properties().ready);
        assert_eq!(f.len(), 3);
        assert!(matches!(
            f.finish(),
            TtOptimization::AddRelu { num_ops: 3, .. }
        ));
    }
}
