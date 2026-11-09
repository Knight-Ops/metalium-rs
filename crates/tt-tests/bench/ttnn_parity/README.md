# tt-metal parity scripts (development only)

These scripts measure tt-metal (ttnn) on the same card as this stack, to fill the parity table in
[performance-scoreboard.md](../../../../docs/learnings/performance-scoreboard.md). They are measuring sticks, never part of a
shipped build. Roadmap W0; [tt-metal-parity.md](../../../../docs/plans/tt-metal-parity.md) §2.3.

**First run:** 2026-11-09 on card 1 (p150a, `0000:02:00.0`, AICLK 1350 MHz). Results are in
[`results-2026-11-09/`](results-2026-11-09); the summary is in
[performance-log.md](../../../../docs/learnings/performance-log.md#tt-metal-on-the-same-card-smollm2-17b-and-its-products-2026-11-09-card-1).

## Environment

- **Tree:** a built tt-metal tree with its own SFPI. The first run used `/home/carl/projects/tt-metal-v077` (`d53c2a71fb0`, ttnn
  `0.77.1.dev0`, SFPI 7.69) and its `python_env`.
- **Variables:**
  ```bash
  export TT_METAL_HOME=/home/carl/projects/tt-metal-v077 ARCH_NAME=blackhole PYTHONPATH=$TT_METAL_HOME
  export TT_VISIBLE_DEVICES=1        # fence tt-metal to one card; check /sys/class/tenstorrent/tenstorrent!1 is the PCI address you mean
  export TT_AICLK_SYSFS='/sys/class/tenstorrent/tenstorrent!1/tt_aiclk'
  source $TT_METAL_HOME/python_env/bin/activate
  ```
- **Before the first tt-metal run after this repo's gates,** reset that card alone: `tt-smi -r <PCI address>`. Without the reset, 2026-11-09's first op hung silently. **After the tt-metal runs,** reset it again before this repo's gates. Mesh gates need both cards reset ([silicon-operating-notes.md](../../../../docs/learnings/silicon-operating-notes.md)).

## Scripts

| Script | What it measures |
|---|---|
| `../ttnn_reference.py --device 0` | add, matmul, embedding: eager, batched and traced, host-timed |
| `decode_mm.py [--plain]` | M=32 decode products at SmolLM2-1.7B shapes, DRAM-sharded (as `tt_transformers`), each weight format and fidelity; the per-op time is the slope between a 1-op and a 21-op trace replay |
| `decode_sweep.py` | the DRAM-sharded config's in0 grid and `in0_block_w` swept, best kept |
| `prefill_mm.py`, `prefill_2d.py` | prefill-shape GEMMs, ttnn's auto config and swept 2-D multicast configs, TFLOPS from the trace slope |
| `all_bf16_*.json` | `tt_transformers` decoder precision overrides for an all-BF16 model |

## The model

The model run is `tt_transformers`' `simple_text_demo` with `HF_MODEL` set to a directory holding the SmolLM2-1.7B-Instruct config
and tokenizer, and `vendor/weights/smollm2-1.7b-instruct/model.safetensors` (symlinks):

```bash
HF_MODEL=<dir> TT_CACHE_PATH=<cache dir> MESH_DEVICE=P150 \
  pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1" -s --timeout 2400
```

Other selections: `accuracy`, and `batch-32`. Pass a decoder JSON for all-BF16.

## Qwen3.5-4B

tt-metal's `models/demos/blackhole/qwen36` (written for Qwen3.5-9B on one P150) reads the 4B's config unchanged. `HF_MODEL`
is a directory of symlinks: the two shards in `vendor/weights/qwen3.5-4b/`, the pinned revision's config, tokenizer and chat
template from the Hugging Face cache, and a `model.safetensors.index.json` made from the shards' headers:

```bash
HF_MODEL=<dir> TT_CACHE_PATH=<cache dir> MESH_DEVICE=P150 HF_HUB_OFFLINE=1 \
  pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128 and not 128k" --timeout 1200
```

`-k traced_128` alone also selects `traced_128k`. The demo's decode time is host-timed and moves with host load;
`results-2026-11-09/qwen35-4b-timing-plugin.py` (`-p`) adds back-to-back trace replays for the time on the card. One device
has no batched path (batch above 1 is tensor-parallel only).
