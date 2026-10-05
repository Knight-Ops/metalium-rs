//! Generalized parameter collection and utilities for traced execution in Burn models.

use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorPrimitive;
use burn_tt::{TtBackend, TtTensor};
use std::collections::BTreeMap;

/// Visits all float parameters in a module and collects their underlying `TtTensor` primitives.
pub struct ParamCollector {
    pub params: BTreeMap<ParamId, TtTensor>,
}

impl<B: AutodiffBackend<InnerBackend = TtBackend>> ModuleVisitor<B> for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<burn::tensor::Tensor<B, D>>) {
        let t = param.val().inner();
        let prim = match t.into_primitive() {
            TensorPrimitive::Float(p) => p,
            _ => unreachable!("expected float tensor"),
        };
        self.params.insert(param.id, prim);
    }
}

/// Collects `(new_param, orig_param)` pairs across all parameters of a model,
/// matching them by their unique `ParamId`.
///
/// This is used by `TracedTrainingStep::capture` to record on-device `copy_into`
/// operations that write optimizer updates back into persistent parameter buffers in-place.
pub fn collect_parameter_updates<M: Module<B>, B: AutodiffBackend<InnerBackend = TtBackend>>(
    old_model: &M,
    new_model: &M,
) -> Vec<(TtTensor, TtTensor)> {
    let mut old_collector = ParamCollector {
        params: BTreeMap::new(),
    };
    old_model.visit(&mut old_collector);

    let mut new_collector = ParamCollector {
        params: BTreeMap::new(),
    };
    new_model.visit(&mut new_collector);

    let mut updates = Vec::new();
    for (id, orig_param) in old_collector.params {
        if let Some(new_param) = new_collector.params.remove(&id) {
            updates.push((new_param, orig_param));
        }
    }
    updates
}

/// Ensure all parameters of a model are uploaded and resident in GDDR on the device.
///
/// Must be called before `TracedTrainingStep::capture` to guarantee no host-to-device
/// uploads occur during trace recording (which hardware trace capture strictly forbids).
pub fn ensure_resident<M: Module<B>, B: AutodiffBackend<InnerBackend = TtBackend>>(model: &M) {
    let mut collector = ParamCollector {
        params: BTreeMap::new(),
    };
    model.visit(&mut collector);
    for (_, prim) in collector.params {
        prim.ensure_resident();
    }
}

/// Extract underlying `TtTensor` from an autodiff float tensor.
pub fn primitive_float<B: AutodiffBackend<InnerBackend = TtBackend>, const D: usize>(
    t: burn::tensor::Tensor<B, D>,
) -> TtTensor {
    match t.inner().into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("expected float tensor"),
    }
}

/// Extract underlying `TtTensor` from an autodiff int tensor.
pub fn primitive_int<B: AutodiffBackend<InnerBackend = TtBackend>, const D: usize>(
    t: burn::tensor::Tensor<B, D, burn::tensor::Int>,
) -> TtTensor {
    t.inner().into_primitive()
}

/// Extract underlying `TtTensor` from a base `TtBackend` float tensor.
pub fn primitive_float_inner<const D: usize>(t: burn::tensor::Tensor<TtBackend, D>) -> TtTensor {
    match t.into_primitive() {
        TensorPrimitive::Float(p) => p,
        _ => unreachable!("expected float tensor"),
    }
}

/// Extract underlying `TtTensor` from a base `TtBackend` int tensor.
pub fn primitive_int_inner<const D: usize>(
    t: burn::tensor::Tensor<TtBackend, D, burn::tensor::Int>,
) -> TtTensor {
    t.into_primitive()
}
