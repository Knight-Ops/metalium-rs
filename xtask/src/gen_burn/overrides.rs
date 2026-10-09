//! One override list per work lane (see `docs/plans/hardware-coverage-closeout.md`).
//! A lane appends its methods to its own file only.

#[path = "overrides/t10_misc.rs"]
mod t10_misc;
#[path = "overrides/t1_intbool.rs"]
mod t1_intbool;
#[path = "overrides/t2_index.rs"]
mod t2_index;
#[path = "overrides/t3_scan.rs"]
mod t3_scan;
#[path = "overrides/t4_rem.rs"]
mod t4_rem;
#[path = "overrides/t5_sort.rs"]
mod t5_sort;
#[path = "overrides/t6_random.rs"]
mod t6_random;
#[path = "overrides/t7_dtype.rs"]
mod t7_dtype;
#[path = "overrides/t8_mathmode.rs"]
mod t8_mathmode;
#[path = "overrides/t9_mesh.rs"]
mod t9_mesh;

type List = &'static [(&'static str, &'static [&'static str])];

pub const LANES: &[(&str, List)] = &[
    ("t1_intbool", t1_intbool::OVERRIDDEN),
    ("t2_index", t2_index::OVERRIDDEN),
    ("t3_scan", t3_scan::OVERRIDDEN),
    ("t4_rem", t4_rem::OVERRIDDEN),
    ("t5_sort", t5_sort::OVERRIDDEN),
    ("t6_random", t6_random::OVERRIDDEN),
    ("t7_dtype", t7_dtype::OVERRIDDEN),
    ("t8_mathmode", t8_mathmode::OVERRIDDEN),
    ("t9_mesh", t9_mesh::OVERRIDDEN),
    ("t10_misc", t10_misc::OVERRIDDEN),
];
