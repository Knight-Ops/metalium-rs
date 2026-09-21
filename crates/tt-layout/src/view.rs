//! Strided views over host buffers.
//!
//! Strided rather than contiguous because that is the shape Burn hands tensors in,
//! and adapting at the boundary later would mean a copy this crate is otherwise in a
//! position to avoid.

use crate::convert::HostDtype;

/// A read-only view of a host tensor: `[batch, rows, cols]`.
///
/// Strides are in **elements**, not bytes, and may be zero (a broadcast) or
/// negative (a reversed axis). `offset` is the element index of `[0, 0, 0]`.
#[derive(Copy, Clone, Debug)]
pub struct TensorView<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) dtype: HostDtype,
    pub(crate) shape: [usize; 3],
    pub(crate) strides: [isize; 3],
    pub(crate) offset: usize,
}

impl<'a> TensorView<'a> {
    /// A view over a contiguous, row-major `[batch, rows, cols]` buffer.
    pub fn contiguous(data: &'a [u8], dtype: HostDtype, shape: [usize; 3]) -> Self {
        let [_, rows, cols] = shape;
        TensorView {
            data,
            dtype,
            shape,
            strides: [(rows * cols) as isize, cols as isize, 1],
            offset: 0,
        }
    }

    /// A view with explicit element strides.
    pub fn strided(
        data: &'a [u8],
        dtype: HostDtype,
        shape: [usize; 3],
        strides: [isize; 3],
        offset: usize,
    ) -> Self {
        TensorView {
            data,
            dtype,
            shape,
            strides,
            offset,
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub fn dtype(&self) -> HostDtype {
        self.dtype
    }

    /// Raw bits of one element, or `None` if the computed index falls outside the
    /// buffer -- which a hand-built stride set can do.
    pub(crate) fn element(&self, b: usize, r: usize, c: usize) -> Option<u32> {
        let index = self.offset as isize
            + b as isize * self.strides[0]
            + r as isize * self.strides[1]
            + c as isize * self.strides[2];
        let index = usize::try_from(index).ok()?;
        let width = self.dtype.bytes();
        let start = index.checked_mul(width)?;
        let bytes = self.data.get(start..start + width)?;
        Some(match width {
            4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            _ => u16::from_le_bytes([bytes[0], bytes[1]]) as u32,
        })
    }
}

/// A writable view, used as the destination of [`crate::detilize`].
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    pub(crate) data: &'a mut [u8],
    pub(crate) dtype: HostDtype,
    pub(crate) shape: [usize; 3],
    pub(crate) strides: [isize; 3],
    pub(crate) offset: usize,
}

impl<'a> TensorViewMut<'a> {
    pub fn contiguous(data: &'a mut [u8], dtype: HostDtype, shape: [usize; 3]) -> Self {
        let [_, rows, cols] = shape;
        TensorViewMut {
            data,
            dtype,
            shape,
            strides: [(rows * cols) as isize, cols as isize, 1],
            offset: 0,
        }
    }

    pub fn strided(
        data: &'a mut [u8],
        dtype: HostDtype,
        shape: [usize; 3],
        strides: [isize; 3],
        offset: usize,
    ) -> Self {
        TensorViewMut {
            data,
            dtype,
            shape,
            strides,
            offset,
        }
    }

    pub fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub fn dtype(&self) -> HostDtype {
        self.dtype
    }

    pub(crate) fn set_element(&mut self, b: usize, r: usize, c: usize, bits: u32) {
        let index = self.offset as isize
            + b as isize * self.strides[0]
            + r as isize * self.strides[1]
            + c as isize * self.strides[2];
        let Ok(index) = usize::try_from(index) else {
            return;
        };
        let width = self.dtype.bytes();
        let Some(start) = index.checked_mul(width) else {
            return;
        };
        let Some(slot) = self.data.get_mut(start..start + width) else {
            return;
        };
        match width {
            4 => slot.copy_from_slice(&bits.to_le_bytes()),
            _ => slot.copy_from_slice(&(bits as u16).to_le_bytes()),
        }
    }
}
