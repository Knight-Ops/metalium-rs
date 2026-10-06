//! Fusion runtime and backend implementation for Tenstorrent Blackhole.

pub mod eltwise;

use crate::{TtBackend, TtDevice, TtHandle};
use burn_backend::tensor::FloatTensor;
use burn_backend::DType;
use burn_fusion::{FusionBackend, FusionRuntime, NumOperations, OperationFuser, Optimization};
use burn_ir::TensorIr;
use eltwise::ElementWiseFuser;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

static FUSED_ADD_RELU_EXECUTIONS: AtomicUsize = AtomicUsize::new(0);
static FUSED_RELU_EXECUTIONS: AtomicUsize = AtomicUsize::new(0);

/// Number of times compound `AddRelu` has been executed by the fusion runtime.
pub fn fused_add_relu_count() -> usize {
    FUSED_ADD_RELU_EXECUTIONS.load(Ordering::Relaxed)
}

/// Number of times `Relu` has been executed by the fusion runtime.
pub fn fused_relu_count() -> usize {
    FUSED_RELU_EXECUTIONS.load(Ordering::Relaxed)
}

/// Reset fusion execution counters.
pub fn reset_fusion_counters() {
    FUSED_ADD_RELU_EXECUTIONS.store(0, Ordering::Relaxed);
    FUSED_RELU_EXECUTIONS.store(0, Ordering::Relaxed);
}

/// Resolve a fused float tensor primitive to an underlying device `TtTensor`.
pub fn resolve_float_tensor(
    tensor: &burn_fusion::FusionTensor<TtFusionRuntime>,
) -> crate::TtTensor {
    let client = burn_fusion::get_client::<TtBackend>(tensor.client.device());
    client.resolve_tensor_float::<TtBackend>(tensor.clone())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TtOptimizationState {
    AddRelu {
        lhs: TensorIr,
        rhs: TensorIr,
        out: TensorIr,
        num_ops: usize,
    },
    Relu {
        input: TensorIr,
        out: TensorIr,
        num_ops: usize,
    },
}

#[derive(Debug)]
pub enum TtOptimization {
    AddRelu {
        lhs: TensorIr,
        rhs: TensorIr,
        out: TensorIr,
        num_ops: usize,
    },
    Relu {
        input: TensorIr,
        out: TensorIr,
        num_ops: usize,
    },
}

impl NumOperations for TtOptimization {
    fn len(&self) -> usize {
        match self {
            TtOptimization::AddRelu { num_ops, .. } => *num_ops,
            TtOptimization::Relu { num_ops, .. } => *num_ops,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            TtOptimization::AddRelu { .. } => "tt_add_relu",
            TtOptimization::Relu { .. } => "tt_relu",
        }
    }
}

impl Optimization<TtFusionRuntime> for TtOptimization {
    fn execute(
        &mut self,
        context: &mut burn_fusion::stream::Context<TtHandle>,
        _execution: &burn_fusion::stream::OrderedExecution<TtFusionRuntime>,
    ) {
        match self {
            TtOptimization::AddRelu { lhs, rhs, out, .. } => {
                FUSED_ADD_RELU_EXECUTIONS.fetch_add(1, Ordering::Relaxed);
                let lhs_global = context
                    .tensors
                    .get(&lhs.id)
                    .cloned()
                    .unwrap_or_else(|| lhs.clone());
                let rhs_global = context
                    .tensors
                    .get(&rhs.id)
                    .cloned()
                    .unwrap_or_else(|| rhs.clone());
                let out_global = context
                    .tensors
                    .get(&out.id)
                    .cloned()
                    .unwrap_or_else(|| out.clone());

                let lhs_tensor = context.handles.get_float_tensor::<TtBackend>(&lhs_global);
                let rhs_tensor = context.handles.get_float_tensor::<TtBackend>(&rhs_global);

                let result = if lhs_tensor.storage_format() != crate::storage::StorageFormat::F32
                    || rhs_tensor.storage_format() != crate::storage::StorageFormat::F32
                {
                    let sum =
                        <TtBackend as burn_backend::ops::FloatTensorOps<TtBackend>>::float_add(
                            lhs_tensor, rhs_tensor,
                        );
                    <TtBackend as burn_backend::ops::ActivationOps<TtBackend>>::relu(sum)
                } else {
                    crate::ops::float::float_add_relu(lhs_tensor, rhs_tensor)
                };

                context
                    .handles
                    .register_float_tensor::<TtBackend>(&out_global.id, result);
            }
            TtOptimization::Relu { input, out, .. } => {
                FUSED_RELU_EXECUTIONS.fetch_add(1, Ordering::Relaxed);
                let in_global = context
                    .tensors
                    .get(&input.id)
                    .cloned()
                    .unwrap_or_else(|| input.clone());
                let out_global = context
                    .tensors
                    .get(&out.id)
                    .cloned()
                    .unwrap_or_else(|| out.clone());

                let in_tensor = context.handles.get_float_tensor::<TtBackend>(&in_global);
                let result =
                    <TtBackend as burn_backend::ops::ActivationOps<TtBackend>>::relu(in_tensor);

                context
                    .handles
                    .register_float_tensor::<TtBackend>(&out_global.id, result);
            }
        }
    }

    fn to_state(&self) -> TtOptimizationState {
        match self {
            TtOptimization::AddRelu {
                lhs,
                rhs,
                out,
                num_ops,
            } => TtOptimizationState::AddRelu {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                out: out.clone(),
                num_ops: *num_ops,
            },
            TtOptimization::Relu {
                input,
                out,
                num_ops,
            } => TtOptimizationState::Relu {
                input: input.clone(),
                out: out.clone(),
                num_ops: *num_ops,
            },
        }
    }

    fn from_state(_device: &TtDevice, state: TtOptimizationState) -> Self {
        match state {
            TtOptimizationState::AddRelu {
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
            TtOptimizationState::Relu {
                input,
                out,
                num_ops,
            } => TtOptimization::Relu {
                input,
                out,
                num_ops,
            },
        }
    }
}

#[derive(Debug)]
pub struct TtFusionRuntime;

impl FusionRuntime for TtFusionRuntime {
    type OptimizationState = TtOptimizationState;
    type Optimization = TtOptimization;
    type FusionHandle = TtHandle;
    type FusionDevice = TtDevice;

    fn fusers(_device: TtDevice) -> Vec<Box<dyn OperationFuser<Self::Optimization>>> {
        vec![Box::new(ElementWiseFuser::default())]
    }
}

impl FusionBackend for TtBackend {
    type FusionRuntime = TtFusionRuntime;
    type FullPrecisionBackend = TtBackend;

    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle {
        TtHandle::Tensor(crate::ops::cast_native(tensor, dtype))
    }
}
