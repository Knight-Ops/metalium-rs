//! Element-wise operation fusers for Tenstorrent Blackhole.
//!
//! Fuses compound element-wise patterns such as:
//! - `add(lhs, rhs) -> tmp` followed by `relu(tmp) -> out` into a single compound
//!   Tensix SFPU dispatch `float_add_relu(lhs, rhs)`.
//! - `relu(input) -> out` (lowered by Burn into `lower_equal_elem(0.0)` + `mask_fill(0.0)`)
//!   into a direct device `activation::relu(input)` kernel call.

use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, MaskFillOpIr, NumericOperationIr, OperationIr, ScalarOpIr,
    TensorIr,
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
        num_ops: usize,
    },
    LowerEqual {
        input: TensorIr,
        mask: TensorIr,
        num_ops: usize,
    },
    AddReluMatched {
        lhs: TensorIr,
        rhs: TensorIr,
        out: TensorIr,
        num_ops: usize,
    },
    ReluMatched {
        input: TensorIr,
        out: TensorIr,
        num_ops: usize,
    },
    Closed {
        num_ops: usize,
    },
}

impl OperationFuser<TtOptimization> for ElementWiseFuser {
    fn fuse(&mut self, operation: &OperationIr) {
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
            ) if is_zero_scalar(scalar) => match &self.state {
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
                        num_ops: num_ops + 1,
                    };
                }
                FuserState::Empty => {
                    self.state = FuserState::LowerEqual {
                        input: input.clone(),
                        mask: out.clone(),
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
            })) if is_zero_scalar(value) => match &self.state {
                FuserState::AddLowerEqual {
                    lhs,
                    rhs,
                    add_out,
                    mask: expected_mask,
                    num_ops,
                } if tensor.id == add_out.id && mask.id == expected_mask.id => {
                    self.state = FuserState::AddReluMatched {
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        out: out.clone(),
                        num_ops: num_ops + 1,
                    };
                }
                FuserState::LowerEqual {
                    input,
                    mask: expected_mask,
                    num_ops,
                } if tensor.id == input.id && mask.id == expected_mask.id => {
                    self.state = FuserState::ReluMatched {
                        input: input.clone(),
                        out: out.clone(),
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
                num_ops,
            } => TtOptimization::AddRelu {
                lhs,
                rhs,
                out,
                num_ops,
            },
            FuserState::ReluMatched {
                input,
                out,
                num_ops,
            } => TtOptimization::Relu {
                input,
                out,
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

fn is_zero_scalar(s: &burn_ir::ScalarIr) -> bool {
    match s {
        burn_ir::ScalarIr::Float(v) => *v == 0.0,
        burn_ir::ScalarIr::UInt(_) => true,
        _ => false,
    }
}
