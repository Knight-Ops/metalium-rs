//! Methods lane T3 scans and integer arg-extremes routes to the device: `(trait, &[methods])`, each
//! implemented in the matching `burn-tt/src/ops_*.rs` module.

pub const OVERRIDDEN: &[(&str, &[&str])] = &[
    ("FloatTensorOps", &["float_cummin", "float_cummax"]),
    (
        "IntTensorOps",
        &[
            "int_cumsum",
            "int_cumprod",
            "int_cummin",
            "int_cummax",
            "int_argmax",
            "int_argmin",
        ],
    ),
];
