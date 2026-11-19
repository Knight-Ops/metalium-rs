//! A PyTorch-to-Burn adapter for loading checkpoints, burn-store's `PyTorchToBurnAdapter` with a
//! fast transpose.
//!
//! A PyTorch `Linear` stores its weight `[out, in]`, Burn `[in, out]`, so every projection of a
//! checkpoint is transposed as it loads. burn-store 0.21 does it a byte at a time on one thread,
//! writing down columns: ~17 s for one Qwen3.5-27B layer's 0.8 GB (PROTOKOLL #137), about 20
//! minutes for the model. Here the transpose copies `BLOCK x BLOCK` tiles (both a tile's rows and
//! its columns stay in L1), whole elements at a time, the output's row blocks spread over every
//! core. It moves the same bytes to the same places: bit for bit burn-store's result. Everything
//! else (the norms' `weight`/`bias` names) is burn-store's adapter's.

use std::rc::Rc;

use burn::tensor::TensorData;
use burn_store::{ModuleAdapter, PyTorchToBurnAdapter, TensorSnapshot};
use rayon::prelude::*;

/// A transpose tile's side, in elements.
const BLOCK: usize = 64;

/// [`PyTorchToBurnAdapter`] with a blocked, parallel transpose of `Linear` weights.
#[derive(Debug, Clone, Default)]
pub struct PyTorchAdapter;

impl ModuleAdapter for PyTorchAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let linear_weight = snapshot.shape.len() == 2
            && snapshot.module_type().as_deref() == Some("Struct:Linear")
            && snapshot
                .path_stack
                .as_ref()
                .and_then(|p| p.last())
                .is_some_and(|n| n == "weight");
        if !linear_weight {
            return PyTorchToBurnAdapter.adapt(snapshot);
        }
        let data = snapshot.clone_data_fn();
        let (rows, cols) = (snapshot.shape[0], snapshot.shape[1]);
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(transposed(data()?))),
            snapshot.dtype,
            vec![cols, rows].into(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        PyTorchToBurnAdapter.get_alternative_param_name(param_name, container_type)
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// `data` (`[rows, cols]`) transposed to `[cols, rows]`.
fn transposed(data: TensorData) -> TensorData {
    let (rows, cols) = (data.shape[0], data.shape[1]);
    let size = data.dtype.size();
    let mut out = vec![0u8; data.as_bytes().len()];
    transpose_bytes(data.as_bytes(), &mut out, rows, cols, size);
    TensorData::from_bytes_vec(out, vec![cols, rows], data.dtype)
}

/// `src` (`rows x cols` elements of `size` bytes) transposed into `dst` (`cols x rows`).
fn transpose_bytes(src: &[u8], dst: &mut [u8], rows: usize, cols: usize, size: usize) {
    assert_eq!(src.len(), rows * cols * size);
    assert_eq!(dst.len(), src.len());
    if src.is_empty() {
        return;
    }
    // Each task writes `BLOCK` whole output rows (input columns `c0..c1`), tile by tile down the
    // input's rows.
    dst.par_chunks_mut(BLOCK * rows * size)
        .enumerate()
        .for_each(|(b, out)| {
            let c0 = b * BLOCK;
            let c1 = (c0 + BLOCK).min(cols);
            for r0 in (0..rows).step_by(BLOCK) {
                let r1 = (r0 + BLOCK).min(rows);
                for c in c0..c1 {
                    let line = &mut out[(c - c0) * rows * size..][..rows * size];
                    for r in r0..r1 {
                        let from = (r * cols + c) * size;
                        line[r * size..(r + 1) * size].copy_from_slice(&src[from..from + size]);
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::DType;

    /// Bit for bit burn-store's own adapter, for BF16 and F32 weights whose sides are and are not
    /// multiples of a tile.
    #[test]
    fn transposes_as_burn_store_does() {
        for (rows, cols, dtype) in [
            (3, 5, DType::F32),
            (64, 128, DType::BF16),
            (130, 67, DType::BF16),
            (257, 1, DType::F32),
        ] {
            let size = dtype.size();
            let bytes: Vec<u8> = (0..rows * cols * size)
                .map(|i| (i * 7 + i / 3) as u8)
                .collect();
            let data = TensorData::from_bytes_vec(bytes, vec![rows, cols], dtype);
            let snapshot = |d: TensorData| {
                TensorSnapshot::from_data(
                    d,
                    vec!["proj".into(), "weight".into()],
                    vec!["Struct:Linear".into(), "Struct:Linear".into()],
                    Default::default(),
                )
            };
            let ours = PyTorchAdapter
                .adapt(&snapshot(data.clone()))
                .to_data()
                .unwrap();
            let theirs = PyTorchToBurnAdapter
                .adapt(&snapshot(data))
                .to_data()
                .unwrap();
            assert_eq!(ours.shape, theirs.shape, "{rows}x{cols}");
            assert_eq!(
                ours.as_bytes(),
                theirs.as_bytes(),
                "{rows}x{cols} {dtype:?}"
            );
        }
    }
}
