//! Embed MNIST, compressed, so the binary needs nothing beside it.
//!
//! Reads the IDX files `cargo xtask fetch-mnist` puts in `vendor/mnist/`
//! (pinned by hash in `PINS.toml`) and deflates each into `OUT_DIR`; `main.rs`
//! includes them and inflates them in memory at start-up. 55 MB of IDX becomes
//! about 11 MB of binary.

use std::path::PathBuf;

const FILES: [&str; 4] = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("../../vendor/mnist");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    for name in FILES {
        let path = src.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        let raw = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e}. The binary embeds MNIST: run `cargo xtask fetch-mnist` first.",
                path.display()
            )
        });
        let packed = miniz_oxide::deflate::compress_to_vec(&raw, 6);
        std::fs::write(out.join(format!("{name}.deflate")), packed).unwrap();
    }
}
