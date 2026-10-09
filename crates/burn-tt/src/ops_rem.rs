//! Lane T4 exact remainder: native implementations. Routed from `ops.rs`'s `float`, `int`
//! and `bool` modules by glob re-export, and listed in
//! `xtask/src/gen_burn/overrides/`.
//!
//! `float_remainder{,_scalar}` are `burn-flex`'s `((a % b) + b) % b`, bit for
//! bit, on the device (`tt_kernels::sfpu::ops::rem`): Rust's `%` is an exact
//! `fmod`, so neither `a - b * trunc(a / b)` nor any approximation is it.
//! Every input is Flex's bits -- denormals, signed zeros, infinities, huge
//! quotients -- but a NaN's payload (the program stores the canonical NaN).
//!
//! A divisor that is not `lhs`'s shape is expanded on the device to the common
//! shape first (a copy: no arithmetic, nothing downloaded), because the
//! program's row-broadcast form -- unrolled over a tile's row groups -- would
//! not fit a role's program slot; the same-shape program serves every shape.

#![allow(unused_imports)]

pub mod float {
    use crate::ops::*;
    use burn_backend::Scalar;
    use num_traits::ToPrimitive;

    /// `((lhs % rhs) + rhs) % rhs` element-wise, Flex's formula, on the device
    /// where the data is; operands that do not fit the device's element-wise
    /// path fail, naming the operation and both operands.
    pub fn float_remainder(
        lhs: FloatTensor<TtBackend>,
        rhs: FloatTensor<TtBackend>,
    ) -> FloatTensor<TtBackend> {
        let (sl, sr) = (lhs.shape().to_vec(), rhs.shape().to_vec());
        let (lhs, rhs) = if sl == sr {
            (lhs, rhs)
        } else {
            let Some(common) = broadcast_shape(&sl, &sr) else {
                fail(
                    "float_remainder",
                    format_args!(
                        "lhs=({}), rhs=({}): the shapes do not broadcast",
                        context(&lhs),
                        context(&rhs)
                    ),
                )
            };
            let shape = burn_backend::Shape::from(common);
            (
                expanded(&lhs, shape.clone(), "float_remainder"),
                expanded(&rhs, shape, "float_remainder"),
            )
        };
        device_eltwise(kind_sfpu::REM, 0.0, &lhs, Some(&rhs)).unwrap_or_else(|| {
            fail(
                "float_remainder",
                format_args!("lhs=({}), rhs=({})", context(&lhs), context(&rhs)),
            )
        })
    }

    /// `((lhs % s) + s) % s` with the scalar converted as Flex converts it
    /// (`to_f64() as f32`, `burn-flex` `ops/binary.rs`), on the device where
    /// the data is.
    pub fn float_remainder_scalar(
        lhs: FloatTensor<TtBackend>,
        rhs: Scalar,
    ) -> FloatTensor<TtBackend> {
        let s = rhs.to_f64().expect("a float scalar") as f32;
        device_eltwise(kind_sfpu::REM_S, s, &lhs, None).unwrap_or_else(|| {
            fail(
                "float_remainder_scalar",
                format_args!("lhs=({}), rhs={rhs:?}", context(&lhs)),
            )
        })
    }
}

pub mod int {
    use crate::ops::*;
}

pub mod bool {
    use crate::ops::*;
}
