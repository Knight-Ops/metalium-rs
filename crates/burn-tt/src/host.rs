//! Owned tensor bytes used for input staging and explicit readback.
use burn_backend::{DType, Shape, TensorData};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct HostBuffer {
    data: Arc<TensorData>,
    shape: Shape,
}

impl HostBuffer {
    pub(crate) fn from_data(data: TensorData) -> Self {
        assert_eq!(
            data.bytes.len(),
            data.shape.num_elements() * data.dtype.size(),
            "burn-tt: invalid tensor bytes for shape {:?}, dtype {:?}",
            data.shape,
            data.dtype
        );
        Self {
            shape: data.shape.clone(),
            data: Arc::new(data),
        }
    }

    pub(crate) fn shape(&self) -> Shape {
        self.shape.clone()
    }
    pub(crate) fn dtype(&self) -> DType {
        self.data.dtype
    }
    pub(crate) fn into_data(self) -> TensorData {
        let mut data = Arc::try_unwrap(self.data).unwrap_or_else(|data| (*data).clone());
        data.shape = self.shape;
        data
    }

    pub(crate) fn reshape(mut self, shape: Shape) -> Self {
        assert_eq!(
            shape.num_elements(),
            self.shape.num_elements(),
            "burn-tt: invalid reshape {:?} -> {:?}",
            self.shape,
            shape
        );
        self.shape = shape;
        self
    }

    /// Repack host input bytes without interpreting or computing on values.
    pub(crate) fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        let from = self.shape().to_vec();
        let mut to = from.clone();
        to.swap(dim1, dim2);
        self.repack(to, |mut coords| {
            coords.swap(dim1, dim2);
            coords
        })
    }

    pub(crate) fn slice(self, slices: &[burn_backend::Slice]) -> Self {
        let from = self.shape().to_vec();
        assert!(
            slices.len() <= from.len(),
            "burn-tt: too many slice dimensions"
        );
        let mut ranges: Vec<Vec<usize>> = from.iter().map(|&n| (0..n).collect()).collect();
        for (dim, slice) in slices.iter().enumerate() {
            assert!(
                slice.step > 0,
                "burn-tt: host input slicing requires a positive step"
            );
            let n = from[dim] as isize;
            let bound = |i: isize| if i < 0 { n + i } else { i };
            let start = bound(slice.start);
            let end = bound(slice.end.unwrap_or(n));
            assert!(
                start >= 0 && start <= end && end <= n,
                "burn-tt: invalid slice {slice:?} for shape {from:?}"
            );
            ranges[dim] = (start as usize..end as usize)
                .step_by(slice.step as usize)
                .collect();
        }
        let to = ranges.iter().map(Vec::len).collect();
        self.repack(to, |coords| {
            coords
                .iter()
                .enumerate()
                .map(|(d, &i)| ranges[d][i])
                .collect()
        })
    }

    pub(crate) fn expand(self, shape: Shape) -> Self {
        let from = self.shape().to_vec();
        let to = shape.to_vec();
        assert!(
            to.len() >= from.len(),
            "burn-tt: invalid expand {from:?} -> {to:?}"
        );
        let pad = to.len() - from.len();
        assert!(
            from.iter().zip(&to[pad..]).all(|(&a, &b)| a == b || a == 1),
            "burn-tt: invalid expand {from:?} -> {to:?}"
        );
        self.repack(to, |coords| {
            from.iter()
                .enumerate()
                .map(|(d, &n)| if n == 1 { 0 } else { coords[d + pad] })
                .collect()
        })
    }

    fn repack(self, shape: Vec<usize>, coords: impl Fn(Vec<usize>) -> Vec<usize>) -> Self {
        let from = self.shape().to_vec();
        let width = self.dtype().size();
        let mut bytes = Vec::with_capacity(shape.iter().product::<usize>() * width);
        for flat in 0..shape.iter().product() {
            let mut remainder = flat;
            let mut at = vec![0; shape.len()];
            for d in (0..shape.len()).rev() {
                at[d] = remainder % shape[d];
                remainder /= shape[d];
            }
            let at = coords(at);
            let offset = at
                .iter()
                .zip(&from)
                .fold(0, |index, (&i, &n)| index * n + i)
                * width;
            bytes.extend_from_slice(&self.data.bytes[offset..offset + width]);
        }
        Self::from_data(TensorData::from_bytes_vec(bytes, shape, self.dtype()))
    }
}
