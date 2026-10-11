//! `argwhere` (and with it `nonzero`) of a Boolean tensor on the device.
//!
//! Found by Burn's own backend tests (`burn-tt-conformance`: `argwhere_nonzero::*` and the autodiff
//! `nonzero`): `bool_argwhere` was a stub that failed unconditionally, while the generated coverage
//! table listed it as native, because it was hand-written. Its result has a data-dependent shape --
//! `[count, rank]` -- so the one thing the host must learn is the count; everything else is device work:
//! the flat positions of the `true` elements are the first `count` of a stable argsort of the negated
//! mask (ties keep their order, so the positions come out ascending), and each coordinate is
//! `(position / stride) % extent` in native `I32` arithmetic.
//!
//! The oracle is exact -- positions are integers -- against Flex's, over every rank from one to four,
//! masks from empty to full and every density between, and shapes ragged against the tile. Beyond it:
//!
//! * the **traffic claim**: exactly one device-to-host read, of the 4-byte count, and no upload
//!   but the two constant rows of radices (`rank` words each); the result is computed on the device;
//! * a mask longer than the device sort's limit is **refused by name** (`bool_argwhere`), not by the
//!   sort's own failure from inside;
//! * negative control: the coordinates in column-major order are not the answer, so the comparison can
//!   tell an order from its transpose.

use burn::tensor::{Bool, Int, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{tensor_traffic, TtBackend, TtDevice};
use tt_tests::burn_device::{with_device, Config};

/// A mask of `shape` with roughly `percent` percent set, deterministically.
fn mask(n: usize, percent: usize, seed: usize) -> Vec<bool> {
    (0..n)
        .map(|i| ((i * 2654435761usize).wrapping_add(seed * 40503) >> 7) % 100 < percent)
        .collect()
}

fn coords<B: burn::tensor::backend::Backend>(t: Tensor<B, 2, Int>) -> (Vec<usize>, Vec<i32>) {
    let data = t.try_into_data().expect("argwhere reads back");
    (
        data.shape.to_vec(),
        data.convert::<i32>().to_vec::<i32>().unwrap(),
    )
}

fn case<const D: usize>(d: &TtDevice, shape: [usize; D], percent: usize) {
    let n: usize = shape.iter().product();
    let m = mask(n, percent, n + percent);
    let t =
        Tensor::<TtBackend, D, Bool>::from_data(TensorData::new(m.clone(), shape), d).to_device(d);
    let f = Tensor::<Flex, D, Bool>::from_data(TensorData::new(m, shape), &FlexDevice);

    let before = tensor_traffic();
    let out = t.argwhere();
    let during = tensor_traffic() - before;
    let want = coords(f.argwhere());
    let found = want.0[0];
    // Exactly the count comes back (one 4-byte word, for any non-empty mask), and when there is
    // something to compute the two constant rows of radices go up (`rank` words each).
    assert_eq!(
        (during.downloads, during.downloaded),
        if n > 0 { (1, 4) } else { (0, 0) },
        "{shape:?} {percent}%: {during:?}"
    );
    assert_eq!(
        (during.uploads, during.uploaded),
        if found > 0 {
            (2, 2 * 4 * D as u64)
        } else {
            (0, 0)
        },
        "{shape:?} {percent}%: {during:?}"
    );
    assert_eq!(out.dims()[1], D);
    assert_eq!(coords(out), want, "{shape:?} {percent}%");
}

#[test]
fn argwhere_is_exactly_flexs_for_every_rank_and_density() {
    with_device(Config::default(), |d| {
        for percent in [0, 1, 10, 50, 90, 100] {
            case(&d, [37], percent);
            case(&d, [5, 7], percent);
            case(&d, [3, 70], percent);
            case(&d, [2, 3, 5], percent);
            case(&d, [2, 2, 3, 9], percent);
        }
        // Burn's own shapes and one a tile boundary lands inside.
        case(&d, [4], 50);
        case(&d, [3, 2], 50);
        case(&d, [2, 2, 3], 50);
        case(&d, [1, 33], 30);
        case(&d, [32, 32], 25);
    });
}

/// The longest mask the device sort takes is exactly right; the next is refused by this op's name.
#[test]
fn the_sorts_limit_is_the_masks_and_past_it_the_refusal_names_the_op() {
    with_device(Config::default(), |d| {
        let limit = tt_kernels::sfpu::sort::max_axis(true);
        case(&d, [limit], 40);

        let m = vec![true; limit + 1];
        let t = Tensor::<TtBackend, 1, Bool>::from_data(TensorData::new(m, [limit + 1]), &d)
            .to_device(&d);
        let panic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.argwhere().into_data()))
                .expect_err("a mask past the limit is refused");
        let text = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            text.contains("bool_argwhere") && text.contains("limit"),
            "the refusal does not name the op and the limit: {text}"
        );
    });
}

/// The comparison tells row-major order from column-major: the same coordinates of a non-symmetric
/// mask, transposed, are a different answer.
#[test]
fn negative_control_a_transposed_order_is_not_equal() {
    with_device(Config::default(), |d| {
        let m = mask(6 * 4, 40, 3);
        let t = Tensor::<TtBackend, 2, Bool>::from_data(TensorData::new(m.clone(), [6, 4]), &d)
            .to_device(&d);
        let f = Tensor::<Flex, 2, Bool>::from_data(TensorData::new(m, [6, 4]), &FlexDevice);
        let ours = coords(t.argwhere());
        assert_eq!(ours, coords(f.clone().argwhere()));
        assert_ne!(
            ours,
            coords(f.transpose().argwhere()),
            "the comparison accepts a transposed order"
        );
    });
}
