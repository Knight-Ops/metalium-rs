//! Methods lane T2 indexing compositions routes to the device: `(trait, &[methods])`, each
//! implemented in the matching `burn-tt/src/ops_*.rs` module.

pub const OVERRIDDEN: &[(&str, &[&str])] = &[
    (
        "FloatTensorOps",
        &["float_gather_nd", "float_scatter_nd", "float_cross"],
    ),
    (
        "IntTensorOps",
        &["int_gather_nd", "int_scatter_nd", "int_matmul"],
    ),
];
