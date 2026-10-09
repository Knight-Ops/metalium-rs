//! Methods lane T1 int and bool wiring routes to the device: `(trait, &[methods])`, each
//! implemented in the matching `burn-tt/src/ops_*.rs` module.

pub const OVERRIDDEN: &[(&str, &[&str])] = &[
    (
        "IntTensorOps",
        &[
            "int_abs",
            "int_cast",
            "int_flip",
            "int_mask_fill",
            "int_mask_where",
            "int_permute",
            "int_unfold",
        ],
    ),
    (
        "BoolTensorOps",
        &[
            "bool_flip",
            "bool_mask_fill",
            "bool_mask_where",
            "bool_permute",
            "bool_unfold",
        ],
    ),
];
