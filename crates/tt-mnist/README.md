# tt-mnist

Train a neural network on MNIST, **in pure Rust, on a Tenstorrent Blackhole card**,
from one self-contained binary.

- **The model** is ordinary [Burn](https://burn.dev): two `nn::Linear` layers
  with a ReLU between them (784-128-10), cross-entropy loss, SGD, batch 64,
  trained by Burn's autodiff. Nothing in the model knows about the hardware.
- **Burn's backend** is `burn-tt`, this workspace's Blackhole backend.
- **Below `burn-tt`**, everything is also Rust and also in this workspace:
  kernels (`tt-kernels`), device access (`tt-device`), the driver interface
  (`tt-kmd`), and the firmware for the card's RISC-V cores
  (`tt-firmware`, built for `riscv32im` and embedded).
- **Not used:** Python, C++, TT-Metalium.
- **On the card:**
  - every matmul, forward and backward;
  - the bias adds, the ReLU and its gradient, the bias-gradient sums and the
    SGD updates;
  - the dataset, weights, activations and gradients, all resident in the
    card's 32 GB of GDDR6.
- **On the host:** only the cross-entropy loss. Per training step about 16 KB
  crosses PCIe.
- **Inside the binary:** the MNIST dataset, compressed, alongside the firmware.
  The target machine needs only the card and Tenstorrent's kernel driver
  (`tenstorrent` ≥ 2.11, ioctl API 2).

## Run

```text
tt-mnist [--card N | --cards 0,1] [--epochs E] [--steps S] [--host]
```

- **`--host`** also trains the same model, from the same initial weights, on
  the CPU through `burn-flex`, for comparison.
- **`--cards 0,1`** uses two cabled cards (matmuls sharded over Ethernet).
  That path is not yet device-resident, so it is much slower.

Sample output: one p150a card, one epoch, static binary, 2026-09-30.

```text
tt-mnist: a 784-128-10 network learning MNIST, in Rust, on Tenstorrent Blackhole
  device   /dev/tenstorrent/0, one Tensix tile, tensors resident in GDDR6
  model    Burn nn::Linear x2 + ReLU, cross-entropy, SGD lr 0.1, batch 64
  data     60000 training / 10000 test images, embedded (264.18ms to unpack)

  step     0   loss 2.3211
  step   100   loss 0.5137
  step   200   loss 0.3612
  step   300   loss 0.4191
  step   400   loss 0.2551
  step   500   loss 0.3423
  step   600   loss 0.2284
  step   700   loss 0.3498
  step   800   loss 0.3174
  step   900   loss 0.3726

on the card:
  model and 59968 images uploaded once       2.97s
  937 training steps                        6.37s  (6.8 ms/step)
  loss                                      2.3211 -> 0.3726
  test accuracy                             91.96%
  tensor data over PCIe, whole run          222.7 MB up (incl. the dataset), 3.32 MB down

on the host CPU (burn-flex), same model, same initial weights:
  937 training steps                        891.17ms  (1.0 ms/step)
  test accuracy                             91.97%
  loss curves agree to 0.1% at 9 of 10 checkpoints
```

- **Accuracy** on the card and on the host differs by one test image in
  10 000.
- **The loss curves** agree to within 0.1% at nine of ten checkpoints. The
  card's matmuls take TF32 operands through the Tensix Matrix Unit and
  accumulate in FP32; the host computes in full FP32.
- **The card's time** is almost all compute on **one** of the chip's 120
  Tensix tiles, one operation after another. Spreading work over the tiles
  is the next step (`docs/implementation-checklist.md`, Phase 9 next steps).

## Build

The binary embeds MNIST, so fetch it first. The files are pinned by hash in
`PINS.toml`.

```sh
cargo xtask fetch-mnist
```

A **static binary** with no dynamic dependencies runs on any x86-64 Linux.
Built this way it is about 15 MB once stripped:

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release -p tt-mnist --target x86_64-unknown-linux-musl
strip -o tt-mnist target/x86_64-unknown-linux-musl/release/tt-mnist
```

A **glibc build** (`cargo build --release -p tt-mnist`) is about 15% faster
on the host-side parts, thanks to glibc's allocator: 5.8 against 6.8 ms/step.
