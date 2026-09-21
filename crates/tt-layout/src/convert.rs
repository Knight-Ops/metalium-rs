//! Host element types, and conversion between them and L1 datums.

use tt_isa::tile::{self, Fp16EncodeError, Fp16Reading, L1Format};

/// The element type of a host buffer.
///
/// Three, matching the scope of this phase. Integer tensors and block float are not
/// here: the first is not needed until Burn wants index tensors, and the second
/// cannot be checked against anything until the packers run.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum HostDtype {
    /// IEEE754 FP32, the host's `f32`.
    F32,
    /// BF16, stored little-endian in two bytes.
    Bf16,
    /// IEEE754 FP16, stored little-endian in two bytes. **The device does not read
    /// every FP16 bit pattern the way the host writes it** — see
    /// [`tt_isa::tile::Fp16Reading`].
    F16,
}

impl HostDtype {
    pub const fn bytes(self) -> usize {
        match self {
            HostDtype::F32 => 4,
            HostDtype::Bf16 | HostDtype::F16 => 2,
        }
    }

    /// The L1 format with the identical bit layout, if any.
    ///
    /// When a layout's L1 format equals this, tilization copies bits and converts
    /// nothing: a conversion nobody asked for is a chance to lose a value.
    pub const fn identical_l1_format(self) -> L1Format {
        match self {
            HostDtype::F32 => L1Format::Fp32,
            HostDtype::Bf16 => L1Format::Bf16,
            HostDtype::F16 => L1Format::Fp16,
        }
    }
}

/// Why a tensor could not be laid out.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LayoutError {
    /// The descriptor describes a tile this crate cannot walk. Named rather than
    /// silently worked around.
    UnsupportedDescriptor { why: &'static str },
    /// An L1 format with no conversion implemented in this phase.
    UnsupportedFormat { format: L1Format },
    /// `ZDim` is not the product of the face grid the layout was built with.
    FaceGridMismatch {
        z_dim: u32,
        faces_down: usize,
        faces_across: usize,
    },
    /// The view's shape or dtype does not match what the layout was built for.
    ShapeMismatch {
        expected: [usize; 3],
        found: [usize; 3],
    },
    /// The view's element type is not the layout's.
    DtypeMismatch {
        expected: HostDtype,
        found: HostDtype,
    },
    /// A tiled buffer of the wrong size.
    BufferLength { expected: usize, found: usize },
    /// A value with no representation the device reads the same way.
    Unrepresentable {
        /// The FP32 bit pattern that could not be encoded.
        value: u32,
        reason: Fp16EncodeError,
    },
    /// An L1 FP16 datum whose meaning depends on which unit reads it, so no host
    /// value is the right answer (`FloatBitPatterns.md:73-77`).
    ConsumerDependentDatum { datum: u16 },
}

impl core::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LayoutError::UnsupportedDescriptor { why } => write!(f, "unsupported tile: {why}"),
            LayoutError::UnsupportedFormat { format } => {
                write!(f, "{format:?} has no host conversion in this phase")
            }
            LayoutError::FaceGridMismatch {
                z_dim,
                faces_down,
                faces_across,
            } => write!(
                f,
                "ZDim = {z_dim} is not {faces_down} x {faces_across} faces"
            ),
            LayoutError::ShapeMismatch { expected, found } => {
                write!(f, "expected shape {expected:?}, found {found:?}")
            }
            LayoutError::DtypeMismatch { expected, found } => {
                write!(f, "expected {expected:?}, found {found:?}")
            }
            LayoutError::BufferLength { expected, found } => {
                write!(f, "expected {expected} bytes of tiles, found {found}")
            }
            LayoutError::Unrepresentable { value, reason } => write!(
                f,
                "FP32 {value:#010x} has no FP16 encoding the device reads back the \
                 same way: {reason:?}"
            ),
            LayoutError::ConsumerDependentDatum { datum } => write!(
                f,
                "FP16 datum {datum:#06x} is a denormal, whose value depends on which \
                 unit reads it"
            ),
        }
    }
}

/// Is this format one the host can convert to and from in this phase?
pub(crate) const fn is_supported(format: L1Format) -> bool {
    matches!(format, L1Format::Fp32 | L1Format::Bf16 | L1Format::Fp16)
}

/// Convert one host element, given as its raw little-endian bits, into the bits of
/// an L1 datum.
pub(crate) fn host_to_l1(dtype: HostDtype, raw: u32, format: L1Format) -> Result<u32, LayoutError> {
    if dtype.identical_l1_format() == format {
        return Ok(raw);
    }
    // Everything goes through FP32 as the common currency, which is exact for BF16
    // and, where it is not exact for FP16, is an error rather than a guess.
    let fp32 = match dtype {
        HostDtype::F32 => raw,
        HostDtype::Bf16 => tile::bf16_to_fp32(raw as u16),
        HostDtype::F16 => match tile::fp16_to_fp32(raw as u16) {
            Fp16Reading::Agreed(b) | Fp16Reading::NotNanOrInfinity(b) => b,
            Fp16Reading::ConsumerDependent => {
                return Err(LayoutError::ConsumerDependentDatum { datum: raw as u16 })
            }
        },
    };
    match format {
        L1Format::Fp32 => Ok(fp32),
        // Rounding rather than truncating: the packer offers both and neither is the
        // default, but round-to-nearest loses less and is what a host-side
        // conversion is normally expected to do. `tt_isa::tile` exposes the other.
        L1Format::Bf16 => Ok(tile::fp32_to_bf16_round(fp32) as u32),
        L1Format::Fp16 => tile::fp32_to_fp16(fp32)
            .map(|v| v as u32)
            .map_err(|reason| LayoutError::Unrepresentable {
                value: fp32,
                reason,
            }),
        other => Err(LayoutError::UnsupportedFormat { format: other }),
    }
}

/// The inverse: an L1 datum's bits back to a host element's bits.
pub(crate) fn l1_to_host(
    format: L1Format,
    datum: u32,
    dtype: HostDtype,
) -> Result<u32, LayoutError> {
    if dtype.identical_l1_format() == format {
        return Ok(datum);
    }
    let fp32 = match format {
        L1Format::Fp32 => datum,
        L1Format::Bf16 => tile::bf16_to_fp32(datum as u16),
        L1Format::Fp16 => match tile::fp16_to_fp32(datum as u16) {
            Fp16Reading::Agreed(b) | Fp16Reading::NotNanOrInfinity(b) => b,
            Fp16Reading::ConsumerDependent => {
                return Err(LayoutError::ConsumerDependentDatum {
                    datum: datum as u16,
                })
            }
        },
        other => return Err(LayoutError::UnsupportedFormat { format: other }),
    };
    match dtype {
        HostDtype::F32 => Ok(fp32),
        HostDtype::Bf16 => Ok(tile::fp32_to_bf16_round(fp32) as u32),
        HostDtype::F16 => tile::fp32_to_fp16(fp32)
            .map(|v| v as u32)
            .map_err(|reason| LayoutError::Unrepresentable {
                value: fp32,
                reason,
            }),
    }
}
