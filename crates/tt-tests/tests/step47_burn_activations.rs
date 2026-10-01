//! Phase 10 gate (milestone 10.2, Burn routing): what 10.2 puts on the device,
//! through Burn, against `burn-flex`, on resident tensors, downloading
//! nothing. In the silicon smoke tier (`xtask/src/silicon.rs`, `SMOKE`).
//!
//! - D3: `Int` (`i32`) and `Bool` tensors live on the card. `to_device`
//!   uploads them; reshapes that keep the stored matrix, whole-tile-row
//!   slices and transposes are views; the logic ops run there.

use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend};
use tt_tests::burn_device::{with_device, Config};

/// Was `p` computed on the device: a device copy and no host one?
fn on_device(p: &burn_tt::TtTensor) -> bool {
    p.computed_on_device()
}

fn ints(seed: u64, n: usize) -> Vec<i32> {
    let mut s = seed | 1;
    let specials = [0, 1, -1, i32::MIN, i32::MAX, 0x7f80_0001, -0x0080_0000];
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 11 == 0 {
                specials[(i / 11) % specials.len()]
            } else {
                (s >> 32) as i32 >> (s % 31)
            }
        })
        .collect()
}

fn bools(seed: u64, n: usize) -> Vec<bool> {
    ints(seed, n)
        .iter()
        .map(|x| x.count_ones() % 2 == 1)
        .collect()
}

/// `f` moves nothing across PCIe.
fn resident<T>(what: &str, f: impl FnOnce() -> T) -> T {
    let before = tensor_traffic();
    let out = f();
    let during = tensor_traffic() - before;
    assert_eq!(
        (during.uploads, during.downloads),
        (0, 0),
        "{what}: {during:?}"
    );
    out
}

#[test]
fn integers_and_booleans_stay_on_the_card() {
    with_device(Config::default(), |d| {
        for [r, c] in [[37, 70], [96, 64]] {
            let iv = ints(1, r * c);
            let ti =
                Tensor::<TtBackend, 2, Int>::from_data(TensorData::new(iv.clone(), [r, c]), &d)
                    .to_device(&d);
            let fi =
                Tensor::<Flex, 2, Int>::from_data(TensorData::new(iv.clone(), [r, c]), &FlexDevice);
            let int_eq = |t: Tensor<TtBackend, 2, Int>, f: Tensor<Flex, 2, Int>, what: &str| {
                assert!(
                    on_device(&t.clone().into_primitive()),
                    "{what}: not on the device"
                );
                assert_eq!(
                    t.into_data().to_vec::<i32>().unwrap(),
                    f.into_data().to_vec::<i32>().unwrap(),
                    "{what} [{r}, {c}]"
                );
            };
            let t = resident("int transpose", || ti.clone().transpose());
            int_eq(t, fi.clone().transpose(), "an Int transposed");
            let t = resident("int reshape", || {
                ti.clone().reshape([1, r, c]).reshape([r, c])
            });
            int_eq(t, fi.clone(), "an Int reshaped there and back");
            if r >= 64 {
                let t = resident("int slice", || ti.clone().slice([32..64, 0..c]));
                int_eq(t, fi.clone().slice([32..64, 0..c]), "an Int's tile rows");
            }

            let (av, bv, rowv) = (bools(2, r * c), bools(3, r * c), bools(4, c));
            let tb = |v: &[bool], s: [usize; 2]| {
                Tensor::<TtBackend, 2, Bool>::from_data(TensorData::new(v.to_vec(), s), &d)
                    .to_device(&d)
            };
            let fb = |v: &[bool], s: [usize; 2]| {
                Tensor::<Flex, 2, Bool>::from_data(TensorData::new(v.to_vec(), s), &FlexDevice)
            };
            let (a, b, row) = (tb(&av, [r, c]), tb(&bv, [r, c]), tb(&rowv, [1, c]));
            let (fa, fbb, frow) = (fb(&av, [r, c]), fb(&bv, [r, c]), fb(&rowv, [1, c]));
            let bool_eq = |t: Tensor<TtBackend, 2, Bool>, f: Tensor<Flex, 2, Bool>, what: &str| {
                assert!(
                    on_device(&t.clone().into_primitive()),
                    "{what}: not on the device"
                );
                let got = t.into_data();
                assert_eq!(got.dtype, f.dtype(), "{what}: dtype");
                assert_eq!(
                    got.to_vec::<bool>().unwrap(),
                    f.into_data().to_vec::<bool>().unwrap(),
                    "{what} [{r}, {c}]"
                );
            };
            let t = resident("not", || a.clone().bool_not());
            bool_eq(t, fa.clone().bool_not(), "!a");
            let t = resident("and", || a.clone().bool_and(b.clone()));
            bool_eq(t, fa.clone().bool_and(fbb.clone()), "a && b");
            let t = resident("or", || a.clone().bool_or(b.clone()));
            bool_eq(t, fa.clone().bool_or(fbb.clone()), "a || b");
            let t = resident("xor", || a.clone().bool_xor(b.clone()));
            bool_eq(t, fa.clone().bool_xor(fbb.clone()), "a ^ b");
            let t = resident("and, a row broadcast", || a.clone().bool_and(row.clone()));
            bool_eq(t, fa.clone().bool_and(frow.clone()), "a && row");
            // A transpose is a view; element-wise ops do not read one yet (M3),
            // so the claim is the view's, not a logic op's on it.
            let t = resident("bool transpose", || a.clone().transpose());
            bool_eq(t, fa.clone().transpose(), "a^T");
        }
    });
}
