//! Methods lane T4 exact remainder routes to the device: `(trait, &[methods])`, each
//! implemented in the matching `burn-tt/src/ops_*.rs` module.

pub const OVERRIDDEN: &[(&str, &[&str])] = &[(
    "FloatTensorOps",
    &["float_remainder", "float_remainder_scalar"],
)];
