# tt-mnist

Trains a Burn MLP or CNN on MNIST on a Blackhole card, from one self-contained binary. Pure
Rust all the way down: `burn-tt` -> `tt-kernels` -> `tt-device` -> `tt-kmd`, with the
firmware (`tt-firmware-images`) and the dataset (compressed by `build.rs`) embedded.
The target machine needs only the card and the `tenstorrent` kernel driver (ioctl
API 2). No Python, C++ or TT-Metalium.

- **Model:** Burn `nn::Linear` x2 with ReLU (784-128-10), cross-entropy, SGD,
  batch 64, Burn autodiff. Nothing in the model knows about the hardware.
- **On the card:** every matmul (forward and backward), bias adds, ReLU and its
  gradient, bias-gradient sums, SGD updates. Dataset, weights, activations and
  gradients stay resident in GDDR6.
- **On the card:** cross-entropy and its backward pass as native kernels.

## Run

```text
tt-mnist [--model mnist|cnn] [--card N | --cards 0,1] [--tiles T] [--epochs E] [--steps S] [--host]
```

| Flag | Meaning |
| `--model cnn` | Conv2D 1→8 (5×5, stride 4), ReLU, average pool 2×2, Linear 72→10; 938 parameters. Training/inference and single-card traces share the MLP execution paths. |
|---|---|
| `--card N` | Train on `/dev/tenstorrent/N` (default 0). |
| `--cards 0,1` | Several cabled cards, matmuls sharded over Ethernet. Not GDDR-resident, one tile per card, so much slower; cannot combine with `--tiles`. |
| `--tiles T` | Compute on `T` Tensix tiles of the card, or `all` (default 1). |
| `--epochs E` | Passes over the 60 000 training images (default 1). |
| `--steps S` | Stop after `S` steps. |
| `--host` | Also train the same model from the same initial weights on the CPU (`burn-flex`) and compare. |
| `--bf16` | Store single-card MNIST parameters, data and activations in BF16. Matrix accumulation/loss and trace input/output remain F32. Mesh and transformer modes reject this flag. |

The binary does not read `TT_TILES` or `TT_TOPOLOGY`; use the flags. It prints the
loss every 100 steps, then timings, test accuracy and the tensor data that crossed
PCIe.

BF16 is opt-in. Raw storage uses half-sized datums; most arithmetic widens on
Tensix to F32 and rounds ties-even back to BF16 at operation boundaries, with
BF16 subnormals flushed to signed zero on device conversion. The current packed
gather route is slower than TF32 on MNIST's measured forward GEMMs (about
662/81 us versus 94/18 us, one tile, release, both cards). This mode has native
SGD and trace gates; full MNIST BF16 accuracy has not been measured. See
[the implementation record](../../docs/plans/tensix-next-features.md).

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


## Convolutional MNIST

```sh
cargo run --release -p tt-mnist -- --model cnn --card 0 --epochs 1 --host
cargo run --release -p tt-mnist -- --model cnn --card 0 --bf16 --train-trace --steps 100
cargo run --release -p tt-mnist -- --model cnn --card 0 --infer --trace --batch 64
```

The CNN takes the same normalized 28×28 dataset rows and reshapes them to NCHW.
Its convolution, pooling, classifier, backward passes and SGD updates run
natively. F32 and BF16 parameters use F32 cross-entropy loss. The deterministic
initial values also feed the optional Flex comparison. `--hidden` and
`--activation` configure only the MLP; the CNN architecture uses ReLU. Mesh
training uses fresh dispatch; hardware traces require one card.

`step89_mnist_cnn` checks actual MNIST learning, convolution weight updates,
scalar-only per-step downloads, and changed-batch training replay against fresh
execution with one/two Tensix tiles. The full-epoch accuracy/timing gate is
explicitly ignored in smoke and runs through the isolated runner:

```sh
cargo xtask silicon --release --device 0 --include-ignored --timeout-secs 900 \
  --filter convolutional_mnist_full_epoch_accuracy_and_timing
```


Initial one-epoch F32 result: **78.60%** on all 10,000 test images, versus Flex
78.66% from identical weights (59,968 training images, 937 steps;
run `1791252065`). This compact model is a learning/residency example, not an
accuracy improvement over the MLP. Fresh native dispatch measured 226.410 ms/step
versus Flex 1.040 ms/step, including preload and scalar loss reads in a single
release run with concurrent host regression/build work. It is not a controlled
median comparison; bounded staging and descriptor construction still need
performance work. F32/BF16 single-card replay correctness is validated separately.
