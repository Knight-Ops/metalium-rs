//! Methods lane T5 device sort family routes to the device: `(trait, &[methods])`, each
//! implemented in the matching `burn-tt/src/ops_*.rs` module.

pub const OVERRIDDEN: &[(&str, &[&str])] = &[
    (
        "FloatTensorOps",
        &[
            "float_sort",
            "float_sort_with_indices",
            "float_argsort",
            "float_argtopk",
            "float_topk",
        ],
    ),
    (
        "IntTensorOps",
        &[
            "int_sort",
            "int_sort_with_indices",
            "int_argsort",
            "int_argtopk",
            "int_topk",
        ],
    ),
];
