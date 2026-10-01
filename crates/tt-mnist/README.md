# tt-mnist

Trains a Burn MLP on MNIST on a Blackhole card, from one self-contained binary. Pure
Rust all the way down: `burn-tt` -> `tt-kernels` -> `tt-device` -> `tt-kmd`, with the
firmware (`tt-firmware-images`) and the dataset (compressed by `build.rs`) embedded.
The target machine needs only the card and the `tenstorrent` kernel driver (ioctl
API 2). No Python, C++ or TT-Metalium.

- **Model:** Burn `nn::Linear` x2 with ReLU (784-128-10), cross-entropy, SGD,
  batch 64, Burn autodiff. Nothing in the model knows about the hardware.
- **On the card:** every matmul (forward and backward), bias adds, ReLU and its
  gradient, bias-gradient sums, SGD updates. Dataset, weights, activations and
  gradients stay resident in GDDR6.
- **On the host:** the cross-entropy loss.

## Run

```text
tt-mnist [--card N | --cards 0,1] [--tiles T] [--epochs E] [--steps S] [--host]
```

| Flag | Meaning |
|---|---|
| `--card N` | Train on `/dev/tenstorrent/N` (default 0). |
| `--cards 0,1` | Several cabled cards, matmuls sharded over Ethernet. Not GDDR-resident, one tile per card, so much slower; cannot combine with `--tiles`. |
| `--tiles T` | Compute on `T` Tensix tiles of the card, or `all` (default 1). |
| `--epochs E` | Passes over the 60 000 training images (default 1). |
| `--steps S` | Stop after `S` steps. |
| `--host` | Also train the same model from the same initial weights on the CPU (`burn-flex`) and compare. |

The binary does not read `TT_TILES` or `TT_TOPOLOGY`; use the flags. It prints the
loss every 100 steps, then timings, test accuracy and the tensor data that crossed
PCIe.

## Build

The binary embeds MNIST, so fetch it first (pinned by hash in `PINS.toml`). Without
it the build script panics, which also breaks a plain workspace `cargo build` or
`cargo test`.

```sh
cargo xtask fetch-mnist
cargo build --release -p tt-mnist
```

A static binary that runs on any x86-64 Linux:

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release -p tt-mnist --target x86_64-unknown-linux-musl
strip -o tt-mnist target/x86_64-unknown-linux-musl/release/tt-mnist
```

## Gotchas

- Card arithmetic is TF32 operands (`SrcRoute::Tf32FromFp32`, `Fidelity::HiFi4`)
  with FP32 accumulation, so loss curves match the host closely, not bit for bit.
  The bit-exact reference is `tt-tests`' `step12_mnist` golden.
- Do not run another compute process on the same card at the same time: the busy
  operating point and tile state are chip-wide (`tt_device::PowerPolicy`).
