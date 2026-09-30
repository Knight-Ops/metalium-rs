//! MNIST, from the IDX files `cargo xtask fetch-mnist` puts in `vendor/mnist/`.
//!
//! Twenty lines of parser rather than a dataset crate: the format is a
//! big-endian magic, the dimensions, then bytes, and the files are pinned by
//! hash, so there is nothing for a dependency to add.

use std::path::PathBuf;

/// One split: `n` images of 28 x 28 as `f32` in `[0, 1]`, row-major, and their
/// labels.
pub struct Split {
    pub images: Vec<f32>,
    pub labels: Vec<u8>,
    pub n: usize,
}

pub const PIXELS: usize = 28 * 28;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor/mnist")
}

fn read(name: &str) -> Vec<u8> {
    let path = dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Run `cargo xtask fetch-mnist` first.",
            path.display()
        )
    })
}

fn header(bytes: &[u8], magic: u32, dims: usize) -> Vec<usize> {
    let word = |i: usize| u32::from_be_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!(word(0), magic, "not an IDX file of the expected kind");
    (1..=dims).map(|i| word(i) as usize).collect()
}

/// The training split (`train = true`, 60 000 images) or the test split
/// (10 000).
pub fn load(train: bool) -> Split {
    let prefix = if train { "train" } else { "t10k" };
    let images = read(&format!("{prefix}-images-idx3-ubyte"));
    let labels = read(&format!("{prefix}-labels-idx1-ubyte"));
    let d = header(&images, 0x0000_0803, 3);
    assert_eq!((d[1], d[2]), (28, 28));
    let l = header(&labels, 0x0000_0801, 1);
    assert_eq!(d[0], l[0], "images and labels disagree on the count");
    Split {
        images: images[16..].iter().map(|&p| f32::from(p) / 255.0).collect(),
        labels: labels[8..].to_vec(),
        n: d[0],
    }
}
