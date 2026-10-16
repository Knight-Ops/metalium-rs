//! Fusion runtime and backend implementation for Tenstorrent Blackhole.

pub mod chain;
pub mod eltwise;

use crate::{TtBackend, TtDevice, TtHandle};
use burn_backend::ops::{ActivationOps, FloatTensorOps};
use burn_backend::tensor::FloatTensor;
use burn_backend::DType;
use burn_fusion::{FusionBackend, FusionRuntime, NumOperations, OperationFuser, Optimization};
use burn_ir::TensorIr;
use chain::{ChainFuser, ChainKind, ChainMatch};
use eltwise::ElementWiseFuser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

/// Fused executions by optimization name (`NumOperations::name`): counted only when the fused
/// path runs, not when a scalar check sends the operations through unfused.
static FUSED: Mutex<Option<HashMap<&'static str, usize>>> = Mutex::new(None);

fn count(name: &'static str) {
    *FUSED
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry(name)
        .or_default() += 1;
}

/// Number of times the optimization named `name` (`tt_add_relu`, `tt_relu`, `tt_sigmoid`,
/// `tt_silu`, `tt_silu_mul`) has run fused.
pub fn fused_count(name: &str) -> usize {
    FUSED
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()
        .and_then(|m| m.get(name).copied())
        .unwrap_or(0)
}

/// Number of times compound `AddRelu` has been executed by the fusion runtime.
pub fn fused_add_relu_count() -> usize {
    fused_count("tt_add_relu")
}

/// Number of times `Relu` has been executed by the fusion runtime.
pub fn fused_relu_count() -> usize {
    fused_count("tt_relu")
}

/// Reset fusion execution counters.
pub fn reset_fusion_counters() {
    *FUSED.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// The global tensor for `relative`, read with `relative`'s status. The context's global
/// description carries the status of the id's last use in the whole converted stream, which
/// may be a read after this optimization: taking it would consume a handle still needed.
fn global(context: &burn_fusion::stream::Context<TtHandle>, relative: &TensorIr) -> TensorIr {
    let mut global = context
        .tensors
        .get(&relative.id)
        .cloned()
        .unwrap_or_else(|| relative.clone());
    global.status = relative.status;
    global
}

/// Runs an optimization's operations one by one: its pattern did not hold at execution.
fn run_unfused(
    len: usize,
    context: &mut burn_fusion::stream::Context<TtHandle>,
    execution: &burn_fusion::stream::OrderedExecution<TtFusionRuntime>,
) {
    for index in 0..len {
        execution
            .operation_within_optimization(index)
            .execute(&mut context.handles);
    }
}

/// Resolve a fused float tensor primitive to an underlying device `TtTensor`.
pub fn resolve_float_tensor(
    tensor: &burn_fusion::FusionTensor<TtFusionRuntime>,
) -> crate::TtTensor {
    let client = burn_fusion::get_client::<TtBackend>(tensor.client.device());
    client.resolve_tensor_float::<TtBackend>(tensor.clone())
}

/// [`resolve_float_tensor`] for an int tensor: a trace's token and position inputs, made under
/// `Tt`, as the device tensors the trace reads.
#[doc(hidden)]
pub fn resolve_int_tensor(tensor: &burn_fusion::FusionTensor<TtFusionRuntime>) -> crate::TtTensor {
    let client = burn_fusion::get_client::<TtBackend>(tensor.client.device());
    client.resolve_tensor_int::<TtBackend>(tensor.clone())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TtOptimizationState {
    AddRelu {
        lhs: TensorIr,
        rhs: TensorIr,
        out: TensorIr,
        zeros: Vec<u64>,
        num_ops: usize,
    },
    Relu {
        input: TensorIr,
        out: TensorIr,
        zeros: Vec<u64>,
        num_ops: usize,
    },
    Chain(ChainMatch),
}

/// A fused pattern. `zeros` are the runtime scalars (relative ids) the match took to be `+0.0`:
/// the fuser sees every scalar as an id whose value is bound only at execution, so
/// [`Optimization::execute`] checks them and runs the original operations when one differs.
#[derive(Debug)]
pub enum TtOptimization {
    AddRelu {
        lhs: TensorIr,
        rhs: TensorIr,
        out: TensorIr,
        zeros: Vec<u64>,
        num_ops: usize,
    },
    Relu {
        input: TensorIr,
        out: TensorIr,
        zeros: Vec<u64>,
        num_ops: usize,
    },
    /// A [`chain`] pattern.
    Chain(ChainMatch),
}

impl NumOperations for TtOptimization {
    fn len(&self) -> usize {
        match self {
            TtOptimization::AddRelu { num_ops, .. } => *num_ops,
            TtOptimization::Relu { num_ops, .. } => *num_ops,
            TtOptimization::Chain(m) => m.num_ops,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            TtOptimization::AddRelu { .. } => "tt_add_relu",
            TtOptimization::Relu { .. } => "tt_relu",
            TtOptimization::Chain(m) => m.kind.name(),
        }
    }
}

/// Runs a chain whose scalars held: the operation it composes, on `TtBackend`.
/// `bound`: the values of its bound scalars ([`ChainMatch::bind`]).
fn execute_chain(
    m: &ChainMatch,
    bound: &[f64],
    context: &mut burn_fusion::stream::Context<TtHandle>,
    execution: &burn_fusion::stream::OrderedExecution<TtFusionRuntime>,
) {
    count(m.kind.name());
    for index in 0..m.lead {
        execution
            .operation_within_optimization(index)
            .execute(&mut context.handles);
    }
    let inputs: Vec<_> = m
        .inputs
        .iter()
        .map(|t| {
            let t = global(context, t);
            context.handles.get_float_tensor::<TtBackend>(&t)
        })
        .collect();
    let result = match m.kind {
        // The generated dispatch: BF16 and block-float inputs widen and round exactly as
        // Burn's composition's casts do.
        ChainKind::Sigmoid => TtBackend::sigmoid(inputs[0].clone()),
        // One SFPU pass where the separate ops would be SFPU ops on plain F32 (the same
        // instructions, so the same bits), else those ops.
        ChainKind::Silu | ChainKind::SiluMul => {
            let x = &inputs[0];
            let y = inputs.get(1);
            let fused = {
                let _op = crate::report::enter(m.kind.name(), true);
                crate::ops::float::float_silu_mul(x, y)
            };
            fused.unwrap_or_else(|| {
                let s = TtBackend::sigmoid(x.clone());
                let t = TtBackend::float_mul(x.clone(), s);
                match y {
                    Some(y) => TtBackend::float_mul(t, y.clone()),
                    None => t,
                }
            })
        }
        // One kernel where the separate ops would be SFPU ops on plain F32 and the row fits a
        // job, else those ops: `eps` converted as `float_add_scalar` converts it.
        ChainKind::RmsNorm => {
            let x = &inputs[0];
            let eps = bound[0];
            let fused = {
                let _op = crate::report::enter(m.kind.name(), true);
                crate::ops::float::float_rms_norm(x, eps as f32)
            };
            fused.unwrap_or_else(|| {
                use burn_backend::TensorMetadata;
                let last = x.shape().num_dims() - 1;
                let squares = TtBackend::float_mul(x.clone(), x.clone());
                let mean = TtBackend::float_mean_dim(squares, last);
                let shifted = TtBackend::float_add_scalar(mean, eps.into());
                let rms = TtBackend::float_sqrt(shifted);
                TtBackend::float_div(x.clone(), rms)
            })
        }
    };
    let out = global(context, &m.out);
    context
        .handles
        .register_float_tensor::<TtBackend>(&out.id, result);
    for &index in &m.drops {
        execution
            .operation_within_optimization(index)
            .execute(&mut context.handles);
    }
}

impl Optimization<TtFusionRuntime> for TtOptimization {
    fn execute(
        &mut self,
        context: &mut burn_fusion::stream::Context<TtHandle>,
        execution: &burn_fusion::stream::OrderedExecution<TtFusionRuntime>,
    ) {
        let zeros = match &*self {
            TtOptimization::AddRelu { zeros, .. } | TtOptimization::Relu { zeros, .. } => zeros,
            TtOptimization::Chain(m) => {
                let scalars = &context.scalars;
                let bound = m.bind(|id| {
                    scalars
                        .get(&burn_fusion::stream::ScalarId { value: id })
                        .copied()
                });
                if let Some(bound) = bound {
                    execute_chain(m, &bound, context, execution);
                } else {
                    run_unfused(m.num_ops, context, execution);
                }
                return;
            }
        };
        let all_zero = zeros.iter().all(|&id| {
            matches!(
                context.scalars.get(&burn_fusion::stream::ScalarId { value: id }),
                Some(burn_ir::ScalarIr::Float(v)) if v.to_bits() == 0
            )
        });
        if !all_zero {
            // `relu(x)` is `mask_fill(x, x <= 0, 0)`; `thresholded_relu(x, alpha)` and
            // `mask_fill(x, x <= t, v)` have the same shape with other scalars.
            run_unfused(self.len(), context, execution);
            return;
        }
        match self {
            TtOptimization::AddRelu { lhs, rhs, out, .. } => {
                count("tt_add_relu");
                let lhs_global = global(context, lhs);
                let rhs_global = global(context, rhs);
                let out_global = global(context, out);

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
                    let _op = crate::report::enter("add_relu", true);
                    crate::ops::float::float_add_relu(lhs_tensor, rhs_tensor)
                };

                context
                    .handles
                    .register_float_tensor::<TtBackend>(&out_global.id, result);
            }
            TtOptimization::Chain(_) => unreachable!("chains return above"),
            TtOptimization::Relu { input, out, .. } => {
                count("tt_relu");
                let in_global = global(context, input);
                let out_global = global(context, out);

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
                zeros,
                num_ops,
            } => TtOptimizationState::AddRelu {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                out: out.clone(),
                zeros: zeros.clone(),
                num_ops: *num_ops,
            },
            TtOptimization::Relu {
                input,
                out,
                zeros,
                num_ops,
            } => TtOptimizationState::Relu {
                input: input.clone(),
                out: out.clone(),
                zeros: zeros.clone(),
                num_ops: *num_ops,
            },
            TtOptimization::Chain(m) => TtOptimizationState::Chain(m.clone()),
        }
    }

    fn from_state(_device: &TtDevice, state: TtOptimizationState) -> Self {
        match state {
            TtOptimizationState::AddRelu {
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
            TtOptimizationState::Relu {
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
            TtOptimizationState::Chain(m) => TtOptimization::Chain(m),
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
        vec![
            Box::new(ElementWiseFuser::default()),
            Box::new(ChainFuser::new(&chain::SILU_MUL)),
            Box::new(ChainFuser::new(&chain::RMS_NORM)),
        ]
    }
}

impl FusionBackend for TtBackend {
    type FusionRuntime = TtFusionRuntime;
    type FullPrecisionBackend = TtBackend;

    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle {
        TtHandle::Tensor(crate::ops::cast_native(tensor, dtype))
    }
}
