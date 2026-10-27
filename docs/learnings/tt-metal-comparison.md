# Where we stand against tt-metal: findings and fixes to investigate

**Snapshot of 2026-10-26.** Compared against the local tt-metal checkout `/mnt/nvme/tt-metal` at
`a11d86c19d3` (`v0.81.0-dev20261013-1`, committed 2026-10-12), with this repository at `22a265f` on branch
`traced-generation`. It has two halves: how far the stack is from a viable Rust inference engine for
Blackhole, and how each of our performance results and limits compares with what tt-metal does.

It is a dated snapshot, not a living tracker. **To re-run the analysis, follow
[tt-metal-comparison-method.md](tt-metal-comparison-method.md)**, which holds the procedure, the claims sheet to
refresh and the prompts; then add a new dated section or replace this one and diff the verdict tables. The
concept-level comparison (G1-G16, 2026-10-01) is [tt-metal-concepts-review.md](tt-metal-concepts-review.md); this
file is about numbers and limits.

**How to read the evidence marks.** Findings about this repository come from its own docs and code; they were
read, not re-measured, and no build or test was run for this review. Findings about tt-metal were gathered by
reading its tree. **✓** marks a tt-metal claim re-checked directly in the source after the agents reported it
(grep of the cited line). Unmarked tt-metal claims are as reported by the reading agents with the file and
line they gave, at the revision above. Figures marked *derived* (implied GB/s, effective TFLOPS) are our
arithmetic from published times and assumed weight sizes, not published numbers.

## 1. Verdict

- A very strong **single-card, batch-1, bit-exact inference core**: Burn backend, firmware, kernels, traces, a
  profile mode and strict loading, with no TT-Metalium or Python dependency.
- **Not yet a viable LLM inference engine.** It serves one request at a time, runs one model family beyond toy
  sizes, samples fast only greedily, has no multi-card decode and no tokenizer or serving layer.
- Against tt-metal, **decode bandwidth efficiency and the per-op fixed cost are in the same band**; **prefill,
  sampling, quantised formats, batching, serving and multi-device are well behind**; and three of our limits
  (programs that do not fit, trace size, attention's 512-position reach) are consequences of **our design, not of
  the hardware**.
- No same-card, same-model measurement against tt-metal exists in this repository. Every comparison below is
  against published or derived figures for a different model mix. Fix F1 closes that.

## 2. How far we have come (state of the project)

Sources: [roadmap.md](../plans/roadmap.md), [performance-scoreboard.md](performance-scoreboard.md),
`crates/burn-tt-conformance/README.md`, the proposals, `crates/tt-models`.

**Foundations done.** Host driver, ttsim, firmware, a native Burn backend, ISA coverage (Phase 10 closed; the Burn
operation traits have no unsupported float, int or bool method). Burn's own suite on silicon, card 0, `--strict`:
F32 and F16 3,001 of 3,025 pass (24 listed: 19 `precision`, 5 F64); BF16 1,098 of 1,187 pass (89 listed). None
listed is a missing op. On ttsim the default profile lists 52 (24 `precision`, 23 `timeout`, 5 F64).

**Models on silicon** (traced, every tile, each checked against burn-flex on the same weights; SmolLM2 also
against HuggingFace `transformers` on the host):

| Model | On card | burn-flex (CPU) |
|---|---|---|
| ResNet-18 | 13.7 ms | 155 ms |
| ViT-Tiny | 8.3 ms | 105 ms |
| SmolLM2-1.7B-Instruct decode, BF16 weights | **11.76 / 12.24 / 12.95 ms** (about 85 / 82 / 77 tokens/s) at 32 / 128 / 256 tokens | 299 / 306 / 327 ms |
| same, F32 weights | 16.79 / 17.19 / 18.09 ms | same |
| SmolLM2-1.7B prefill (F32) | 315 / 741 / 861 tokens/s at 32 / 128 / 256 | 6-22 tokens/s |
| 1.7B at 8,192 positions | 7,992-token prompt in 6.62 s (chunks of 512) | 1,847 s |

The 1.7B BF16 decode is bit for bit the F32 step (the checkpoint is BF16 widened at load).

### Biggest wins

1. **Correctness discipline.** Every result is bit for bit against an independent oracle; gates carry negative
   controls; traced is checked against eager token for token; vendored tolerances are never loosened. It has
   caught real defects (decode attention's L1 reach: a traced 1.7B generation chose other tokens at a 256-token
   prompt; fixed, gate `step209`).
2. **Decode performance.** A 1.7B step fell from 73 ms (2026-10-16) to 11.8 ms through a position-major cache,
   thin products at the read ceiling, a weight-prefetch pump, planned L1 residency and fused `.ttinsn` push.
   F32 weights stream at about 84% of 512 GB/s over the replay window (429 GB/s).
3. **General architecture.** Fusion through burn-fusion fusers, a driver generic over decoders, one profile mode
   that costs nothing when it is not built in.
4. **Long-context prefill.** 7,992 tokens fell from 18.3 s to 6.6 s (blocked products, replayable softmax, data
   passes); an 8K prompt agrees with burn-flex's tokens.
5. **Hardware safety.** Checked APIs, a build-time firmware instruction gate, pinned inputs with SHA-256.

### Largest gaps, ranked

1. **Batch 1 only.** A step reads every weight once whatever the batch, so batching is the largest throughput
   knob (the roofline's "realistic" batch 4-32 is 132-863 tokens/s against about 80 today). Design only:
   [batched-decode.md](../proposals/batched-decode.md).
2. **No serving layer.** No tokenizer, chat template, scheduler, continuous batching, paged or prefix cache, API.
3. **Narrow model coverage.** Only Llama-style GQA with plain RoPE. No QKV bias, RoPE scaling, sliding window, MoE
   or linear attention; no Llama-3, Mistral or Qwen checkpoint has run.
4. **Sampling.** Only greedy argmax is fast (10.5 us token readback). Temperature, top-k and top-p run on the
   card but eagerly: 112 ms a token (top-k 50, top-p 0.9; `performance-log.md` 2026-10, "not yet fast; a native
   top-k is the follow-up"). Top-k is capped at 512 (`SORT_MAX / 2`); a logit tied with the k-th is left out
   where `transformers` keeps it.
5. **Multi-card is a proof of concept.** Matmul, conv and attention split over Ethernet; every other op on chip
   0; no all-reduce; no LLM run across cards; card-to-card `to_device` is a host round trip; both 800G links
   reach 27 GB/s; the hop latency was never measured; nothing validated on a physical p300.
6. **Numerics and dtypes.** The default matmul is TF32-grade (24 conformance `precision` entries; `--accurate`
   costs 19-64x). KV cache, activations and embedding lookup are F32 (a BF16 lookup is slow); no INT8; BFP formats
   are checked for correctness on a 135M but have no decode numbers on the 1.7B.
7. **Prefill and long context.** Products plateau at about 5 TFLOPS, GDDR-bound, 1-2% of the card's compute. The
   fast attention op fits L1 only to about 512 positions at `head_dim` 64; past 8K the traces do not fit the
   program cache; first trace builds take 10-15 minutes at 8K.
8. **The BF16 step.** Weights floor at about 7.3 ms of an 11.76 ms step; the other ~4.4 ms is small ops and per-op
   cost. BF16 reaches only about 57% of 512 GB/s (292 GB/s); the 90-95% target is met for F32 only.

### Distance to a viable engine (estimate, ours)

| Milestone | Missing | State |
|---|---|---|
| Single-user, single-card 1-2B Llama chat | tokenizer and chat template, a minimal server, fast top-k/top-p | Close: glue plus one on-card top-k |
| A second model family | RoPE scaling, QKV bias, traces past the program-cache limit | Moderate |
| Real serving throughput | batched decode, paged cache, scheduler, BF16 cache and activations | The largest chunk |
| 7-8B models | BF16 cache, activations and embedding; the 32 GB limit; likely both cards | After the above |
| Multi-card inference | Ethernet transfers with no host, per-card submission, tensor placement | Roadmap foundation list all unchecked |

### Repository risks found (code-health review)

1. **Silicon has no automated gate.** CI is GitHub-hosted and only type-checks silicon code. The full simulator
   suite runs only on schedule, dispatch or push to `main` (`.github/workflows/ci.yml`); `main` is at
   `b22a245` (2026-10-14), 55 commits behind this branch, with 12 unmerged branches.
2. **Host-crash hazard.** A hung NoC has dropped the PCIe link and rebooted the host (operating notes). The
   safeguards are software-side: the harness reports the ARC watchdog value but never refuses to run. Our E1
   image left on the cards breaks the next tt-metal run until `tt-smi -r`.
3. **Single maintainer and machine.** All 167 commits are one person under three identities; cards are passed
   through to one VM; bench scripts hard-code `/home/carl/...`; no minimum host is documented.
4. **Mega-files.** `tt-kernels/session.rs` 10,268 lines, `tensor.rs` 9,175 (about 4,400 tests), `burn-tt/ops.rs`
   5,865, `server.rs` 5,380. About 495 non-test `.unwrap()` in `tt-kernels`, many "this Option is set in this
   phase" invariants that panic inside a live device session when violated.
5. **Floating toolchain with tight firmware budgets.** `rust-toolchain.toml` is `stable` while CI uses
   `-D warnings`; firmware images fit 10-24 KiB slots and have shifted by about 900 bytes with compiler
   changes; `ttinsn_t1` already overflows its slot by 216 bytes.
6. **Licensing.** There is no `LICENSE` file (confirmed: `git ls-files` finds none) though `Cargo.toml` declares
   Apache-2.0; `tt-isa/src/isa/generated.rs` (14k lines) is generated from third-party "CC BY-ND" spec content
   and committed; ttsim is a closed prebuilt binary (its terms were not checked); MNIST is embedded in a
   shippable binary (terms not checked).
7. **A blurrier ship boundary.** `tt-mnist` is `SHIPPABLE` and depends on `burn-flex` (confirmed in its
   `Cargo.toml`); `check-no-flex-in-backend` covers only `burn-tt`; the sim check looks for two crate names via
   `cargo tree --edges normal`. `PINS.toml` still says `burn-flex` is a dev-dependency of `tt-tests` only.
8. **Stale documentation.** Roadmap line 17 says the 1.7B decodes at 22.1 tokens/s (about 85 now); its
   conformance headline says 32 listed against 52; the scoreboard's "now" roofline row and one "not measured"
   traced row contradict rows beside them; `roadmap.md` claims to hold only open items but carries 37 `[x]`;
   finished plans (`ttinsn-push-experiment`, `bf16-weight-storage`, `replay-fixed-cost`) still sit in
   `plans/`; six plans and one proposal are missing from `docs/README.md`; `tt-models/README.md` says "three
   models" where there are four rungs.
9. **Weak baselines.** Every claimed win is against burn-flex on the CPU. Until section 3's baseline exists
   there is no tt-metal end-to-end number in this repository.
10. **Untracked clutter.** `.claude/worktrees/` holds a stale 6 GB agent worktree; `.agents/`, `.claude/` and
    `.mcp.json` are untracked and not ignored.
11. **Silent oracle loss.** `tt-tests/build.rs` skips the C oracle build with a cargo warning if the spec checkout
    or a C compiler is missing, so `fma_oracle` and `sfpu_models_oracle` then run no differential test.
12. **No firmware or ARC version check.** Nothing in `tt-device` or `tt-kmd` reads the card's firmware bundle
    version; ARC telemetry tags (34, 35, 36, 22) are hard-coded from UMD. A card on another bundle could report
    different tags and nothing would refuse to run. tt-metal does gate on firmware versions (section 4.9).

## 3. tt-metal's published numbers, beside ours

### 3.1 LLM performance on Blackhole

All figures are decode tokens/s/user unless noted. tt-metal publishes **no p150 figure for Llama-3.2-1B, 3B or
Qwen**, and none for a model near ours. The 1B and 3B rows are Wormhole N150 (288 GB/s), not comparable in
absolute terms. Weight sizes assume BFP8 = 1.0625 B/element, BFP4 = 0.5625, LM head BFP8, embedding and KV
excluded.

| Source (tt-metal) | Model, hardware | Batch | Precision | Result | Implied GB/s *(derived)* |
|---|---|---|---|---|---|
| `models/model_targets.yaml:685-701` (`bh_p150`, seq 712) | Llama-3.1-8B, p150 | 32 | BFP4 FF1/FF3, BFP8 rest, BFP8 KV, LoFi MLP, HiFi2 attention | 42.0 t/s/u (23.8 ms), TTFT 53 ms; "on-device argmax fast-path and two DRAM readers per bank" | about 259 (51% of 512) |
| `models/README.md:21` (updated 2025-12-08) | same | 32, 128-token prompt | same | 33.1 t/s/u, TTFT 57 ms | about 204 (40%) |
| `models/README.md:20` | Llama-3.1-8B, p100 | 32 | same | 29.0 t/s/u, TTFT 61 ms | p100 bandwidth not stated |
| `tt_transformers/PERF.md:30` (deprecated) | same, p150 | **1** | same | 33.6 t/s/u, TTFT 76 ms, top-1 90 / top-5 98 | about 207 (40%) |
| `model_targets.yaml:748-760` | Llama-3.1-8B, DP=4 on p300x2 | 1 per die | same | 52.0 t/s/u per die, 208 total, TTFT 14.4 ms | about 320 per die if 512 each |
| `PERF.md:19,23` | Llama-3.2-1B / 3B, **N150 (Wormhole)** | 1 | same | 87.8 / 54.0 t/s/u | 1B about 92 (32% of 288), 3B about 146 (51%) |
| `model_targets.yaml:537-548` | gpt-oss-20b, p150 | 1 | MoE | 14.73 t/s/u, TTFT 541 ms | not computed |

**Ours (single p150a, batch 1):** 1.7B BF16 weights 3.4 GB in 11.76 ms is about 292 GB/s (57% of 512); F32 weights
6.96 GB in 16.79 ms about 414-429 GB/s (about 84%). So **we sit in the same 40-60% band as tt-metal's best
single-die Blackhole decode while moving two to four times more bytes per parameter**, and our F32 path is
higher. Different models, batch sizes and fidelities: a "same band" statement, not a ranking.

Staleness: `models/README.md` carries "Blackhole software optimization is under active development"; `PERF.md`
is deprecated in favour of the yaml, which has **no batch-1 p150 perf entry** (`perf: {}`, lines 685-693); the
README (line 270) says top-p and batch>1 argmax run on host while `model_targets.yaml:670` contradicts it.

### 3.2 Decode techniques tt-metal uses

- **DRAM-sharded matmul** for all decode matmuls with width-sharded L1 activations
  (`tech_reports/LLMs/llms.md:1586-1626`; config `model_config.py:3646-3895`). p150 has 8 DRAM banks; activation
  grids come from `find_grid_k_n` (line 3813), **capped at 8x8 = 64 cores**; Llama-8B on p150 uses 2 reader
  workers per bank (`get_dram_sharded_matmul_num_workers`, line 3855). We use 64-104 units; the cap is theirs.
- **Stated utilisation is Wormhole only.** `Saturating_DRAM_bandwidth.md:61-69`: 267 of 288 GB/s ("over 92%")
  microbenchmark, matmuls 82-90%, Llama-70B decode 239-260 GB/s. `llms.md:1844-1845`: about 190 GB/s interleaved,
  about 240 GB/s DRAM-sharded. **No Blackhole LLM utilisation figure exists in the tree.**
- **Fused ops** (`attention.py`): `nlp_create_qkv_heads_decode`, `rotary_embedding_llama_fused_qk`,
  `paged_update_cache`, `paged_scaled_dot_product_attention_decode`, `nlp_concat_heads_decode`, multi-device
  `all_gather_matmul_async`; sharded `ttnn.rms_norm` and distributed `rms_norm_pre/post_all_gather`; SiLU fused
  into FF1 (`model_config.py:996`).
- **DRAM prefetcher.** Blackhole only, opt-in `--use_prefetcher`, restricted to Llama-8B
  (`simple_text_demo.py:936-941`); it disables on-device sampling (`model.py:281`).
- **Trace.** Decode always traced (`generator.py:2391`); "a whole model iteration takes about 10 ms with
  microsecond-long ops" (`llms.md:1462-1470`).

### 3.3 Prefill

- 2D multicast matmuls (`MatmulMultiCoreReuseMultiCastProgramConfig`), DRAM-interleaved in and out
  (`llms.md:1535-1570`). **On Blackhole M is capped at 512 rows** (`prefill_len_cutoff`, `model_config.py:661`;
  1024 on Wormhole); longer sequences are reshaped into 512-row chunks (`mlp.py:193-195`). Grid at most 8x8
  (`find_prefill_grid`, line 3783, "TODO Improve configuration for BH"); `minimal_matmul` for sequences over 128
  (line 1869).
- SDPA q/k chunks 64 below 2,048 tokens, 256 from 2,048; chunked prefill via
  `chunked_scaled_dot_product_attention` (`attention.py:1225`).
- **Max chunk:** p150 is absent from `MAX_PREFILL_CHUNK_SIZES_DIV1024` (`model_config.py:2573-2650`), so
  Llama-8B falls back to 4k with a warning.
- **Prefill trace:** only sequence-length buckets 128 and 1024 on p150 (`model_config.py:2649-2700`); traced
  prefill needs paged attention (`generator.py:1155`). `can_enable_trace`: "for bigger sequence lengths, op2op
  gaps are already small, so we don't need tracing" (line 3242); `llms.md:1465-1470`: "we typically use tracing for
  the decode pass of LLMs but not the prefill pass".
- **Effective rate.** Llama-8B TTFT 53-76 ms at 128 tokens is about 27-38 TFLOPS *(derived)*. Ours at 256 tokens
  is about 2.9 TFLOPS *(derived: 861 tokens/s x 2 x 1.7 B)*. About 10x, with a fidelity caveat: theirs is
  BFP8/BFP4 at HiFi2/LoFi, ours F32 at HiFi4. No LLM prefill TFLOPS figure is published; the GEMM report
  (`GEMM_FLOPS.md:83-113`) cites a 13x10 grid on p150, a 580 TFLOPS peak and 96% best-case utilisation on tuned
  large shapes.
- Long context: only Wormhole numbers (`PERF.md:115-139`, Llama-8B at 64k takes 36 s on N300). README lines
  324-341 say 128k; no per-model maximum for p150 was found.

### 3.4 On-device sampling

`models/common/sampling/` (`TTSampling`, `SamplingGenerator`): temperature, top-k, top-p, seed, penalties and
log-probs using `ttnn.topk` and `ttnn.sampling`. **`max_top_k` is 32** (`tt_sampling.py:304,1224`). Greedy
collapses to `ttnn.argmax` with deterministic tie-breaking (lines 29-45, 1147). It is **trace-compatible**:
`generator.py:602-635` captures a separate sampling trace per configuration; it is skipped only when a
per-request seed is active (line 666). No ms/token figure is published. On a single device the vocab is split
into chunks of up to 64k logits, so 49,152 qualifies.

### 3.5 Batching, paged KV, serving

Batch 1-32; paged attention is used by the demos and is mandatory for vLLM and for prefill trace; a non-paged
path exists (`attention.py:406-447`). **KV cache dtype defaults to BFP8** (`model_config.py:394`), BF16 only for
Qwen2.5-7B and a few others; ours is F32 (393 KB a position for the 1.7B). vLLM: `tt/generator_vllm.py`
(`LlamaForCausalLM`, line 681) advertises prefix caching, async decode, on-device sampling, device grammar,
`max_device_top_k: 32` and chunked prefill; the plugin is `tenstorrent/vllm-tt-plugin`
(`tech_reports/LLMs/vLLM_integration.md`). A Blackhole QB2 path (`models/demos/llama31_8b_qb2`) supports batch up
to 32 and a 131,072-token shared cache arena.

### 3.6 Accuracy philosophy

Tolerance-based, not bit-exact: top-1/top-5 against HuggingFace with teacher forcing (`PERF.md:7`; threshold
target minus 0.5, `simple_text_demo.py:82-90`); unit PCC 0.99 for MLP and attention
(`tests/test_mlp.py:145`, `tests/test_attention.py:73`), 0.86-0.94 for the full model
(`tests/test_model.py:139-141`). Default `_default_settings` (`model_config.py:388-419`): BFP8 weights, HiFi2 with
FP16 accumulate for MLP, SDPA prefill HiFi4. "Performance" puts BFP4 FF1/FF3 at LoFi (Llama-8B keeps decoder 31
at BFP8); "accuracy" mode keeps BFP8 attention and KV for Llama and Mistral. Non-determinism is accepted and
documented: identical users in batched prefill can differ in argmax (Galaxy README lines 60-70), and Blackhole
prefill logits are batch-variant (`model_config.py:752-760`).

### 3.7 Model coverage

`tt_transformers/README.md:206-214`: LlamaForCausalLM, Qwen2, Qwen3, Mistral, Mistral3, Phi3, Exaone4_5 (text
only); SmolLM2-1.7B-Instruct is listed (line 88) on N150 as experimental **with no numbers**. Qwen QKV bias
(`attention.py:182-204`), RoPE variants llama3/yarn/linear (`rope.py`), sliding window
(`model_config.py:2917-3042`), Mixtral MoE (`mixtral_moe.py`), a 9B hybrid DeltaNet Qwen on one p150
(`models/demos/blackhole/qwen36`, MTP speculative decode by default). Attention heads must divide the device
count (README line 204). This is the same list our roadmap names as missing.

## 4. Highlight-by-highlight comparison

Verdict key: **inherent** (a hardware fact), **design** (a consequence of how we built it), **not comparable**.

| # | Our result or limit | tt-metal | Verdict |
|---|---|---|---|
| 1 | Decode 11.8 ms, 57% of 512 GB/s (BF16); F32 about 84% | Llama-8B on p150 batch 32 about 51% *(derived)*, batch 1 about 40%; DRAM-sharded, 2 readers a bank, 64-core cap | Same band; not comparable in absolute terms |
| 2 | Fixed cost per op about 4.1 us (3.5 control + 0.6 barrier) | Traced dispatch 1.29 us minimal, 4.6 us with kernel groups or 1 runtime arg on all cores; op-to-op gate 2.08 us (4.60 pack-to-unpack) | Comparable; a few us is inherent to go-signal plus completion sync; 2-3x headroom to their best case is design |
| 3 | Program cache: 240 KiB a tile holds every op's program; "a list's programs do not fit"; unrolled unpack/pack grow with length | Kernels live in DRAM; per-tile kernel-config **ring of 69 KiB** ✓; Belady-style trace allocator ✓; prefetcher binary cache 1 MiB ✓ (864 KB with two command queues, 67 KB in another configuration); only one program must fit; kernels take `num_tiles` as a runtime argument | **Design** |
| 4 | Trace size 1.5 GB for the 1.7B (2026-10-15; may be stale) | 8B batch-32 decode traces 56-70 MB ✓ (a comment in `model_trace_region_sizes.yaml`); 70B prefill traces exceed 128 MB; one multicast command packet a program | **Design** (about 20x larger, a per-unit entry stream) |
| 5 | Static L1 residency plan over a captured graph | Hand-configured L1 sharded activations in decode; no whole-graph plan found; Blackhole DRISC tensor prefetcher | Ours is more ambitious; DRISC is a hardware analogue to study |
| 6 | Prefill about 2.9 TFLOPS | about 27-38 TFLOPS *(derived)*, at BFP8/BFP4 HiFi2/LoFi | Behind; fidelity explains part |
| 7 | Attention reaches about 512 positions at head_dim 64 | K/V streamed from DRAM in chunks of up to 512, runtime length, split over up to 16 cores | **Design**; closing it needs the order change we parked |
| 8 | SFPU push 3.31 vs 1.19 cycles a word | LLK emits `.ttinsn` for every `TTI_*`; answers push-bound code with replay plus `SFPLOADMACRO` | Our measurement is new; their method is stronger for transcendentals |
| 9 | DRAM: **tt-metal's own gate mirrored on our movers: 507.7-508.3 GB/s** (fast-reads, 16 KiB packet entries) and **511.2** (128 KiB entries) on card 1; 480-496 GB/s on 120 tiles; 487-489 per streamed weight | CI gate 490-512 GB/s on 8 banks ✓ (`test_moreh_microbenchmark.py:815`, `bw_lower_bound = 490.0`) | **Meets the gate** with a `fast-reads` image (the safe image 414 with packet entries: a safe entry costs 359 cycles). The 487-489 and 496 figures are not the channels' limit (section 4.8) |
| 10 | Sampling 112 ms a token | On-device, trace-compatible, top-k at most 32 | Behind |
| 11 | Matmul TF32-grade by default | Documents the same TF32 SrcA/SrcB path; `ttnn.matmul` defaults HiFi2/LoFi | We are more accurate than their LLM stack |
| 12 | Multi-chip: hop latency unmeasured, 27 GB/s both links | About 5 us all-gather at 4-16 KiB; fabric with device-to-device sockets | Behind; the go/no-go for two-card TP now looks favourable |

### 4.1 Traces and the program cache (rows 3 and 4)

How tt-metal avoids "programs do not fit":
- Capture puts the command queue in bypass mode and records the dispatch command stream
  (`tt_metal/distributed/fd_mesh_command_queue.cpp:1389-1402`); `record_end` lays it out and writes it to a DRAM
  buffer (`mesh_trace.cpp:36-141`). Replay sends a go-signal with `RUN_MSG_REPLAY_TRACE`, then one
  `add_prefetch_exec_buf` that streams the trace from DRAM through the prefetcher to the dispatcher
  (`tt_metal/impl/trace/dispatch.cpp:76-141`). Trace pages are 1-8 KiB.
- `trace_region_size` is a device parameter; default 0 means trace buffers go top-down in ordinary DRAM
  (`common_values.hpp:16`). Overflow error: "Creating trace buffers of size {}B on MeshDevice {}, but only {}B is
  allocated for trace region." (`mesh_trace.cpp:79`). **No hard maximum trace size**; the limit is DRAM.
- **Binaries are not in the trace.** They sit in a per-program DRAM buffer
  (`program.cpp:3005`); the trace holds per-program command packets (runtime args, CB configs, semaphores, launch
  and go messages, stall counts; `trace_node.hpp:36-52`). A "paged to ringbuffer" prefetch command brings
  binaries in (`impl/program/dispatch.cpp:3233-3261`). The prefetcher core keeps an SRAM cache of binaries
  (`dispatch_settings.cpp:76,98,117` ✓: 1024 KB, 864 KB with several command queues, 67 KB in a third
  configuration).
- **`SimpleTraceAllocator`** (`tt_metal/impl/dispatch/simple_trace_allocator.cpp`) schedules each traced program's
  runtime-arg block and binary in the per-tile ring with **Belady-style eviction** ✓ (lines 61-64: "evict the
  region that will be used the farthest in the future"). Binaries reused later are packed bottom-up and kept
  resident; one-shot data goes top-down (lines 266-290); each node sets `send_binary = !all_binaries_cached` (line
  342); syncs keep the launch-message window within 7 entries (lines 316-340). After a trace the host's
  config-buffer manager is marked completely full (`impl/trace/dispatch.cpp:210`). This is our "paging a trace's
  programs into the cache", done automatically.
- **Rings and limits.** Launch-message ring 8 entries (`dev_msgs.h:430`); **kernel-config ring on a Blackhole
  Tensix `default_l1_kernel_config_size = 69 * 1024`** ✓ (`bh_hal_tensix.cpp:43`), configurable via
  `worker_l1_size`; it holds runtime args, CB configs, semaphores **and the kernel binaries** on Blackhole. 64
  circular buffers a core. Per-processor kernel size nominally 1,497 KB (`MEM_MAX_KERNEL_SIZE`,
  `dev_mem_map.h:80`); no instruction-memory constraint. The one hard failure: "Program size ({}) too large for
  kernel config buffer ({}) on {}" (`program.cpp:3666-3671`).
- **Requirements** (`TraceCorrectness.md:44-62`): every program compiled and in the program cache before
  capture (a miss during capture is an error); program-cache buffers allocated after capture can be corrupted by
  replay.
- **Kernel size does not scale with tensor size.** For example `eltwise_sfpu.cpp:18-23` takes `num_tiles` as a
  runtime argument and loops. Runtime args have a cap (`kernel.cpp:891`); `llms.md:1823` advises converting
  runtime to compile-time arguments to cut dispatch time.
- **Sizes seen.** `model_trace_region_sizes.yaml`: Llama-3.1-8B on `bh_p150` is 0 (dynamic) with the comment
  "TTTv2 P150 batch-32 captures measured 55,787,520 and 70,017,024 bytes" ✓; 80 MB for the 8B family and 90 MB
  for Llama-70B and Qwen3-32B on `bh_p150`/`bh_p300`; Qwen3-32B on p300x2 captured 92,299,264 bytes; Gemma-4-31B
  reserved 512 MiB for 60 layers plus sampling; Llama-70B on p300x2 exceeded 128 MB after decode plus Q128 prefill
  batches 32, 1 and 2 (reserved 224 MB, lines 95-97).

Where the comparison cuts: our 240 KiB cache is **larger** than their 69 KiB ring. We fail because we hold
**every** op's program resident and because some kernels (element-wise unpack and pack, reductions) are unrolled
a tile and so grow with length (roadmap, "The composition does not build at long lengths": unit 0 at 232 of 240
KB at 512 positions). tt-metal's only fit constraint is one program in the ring. L1 holding code is a hardware
fact; programs that scale with length and an all-resident cache are not. We already built the small version of
their answer (looped programs, paging a trace's programs into the cache). Our trace is also a per-unit entry
stream (about 25 32-byte entries a section on each of up to 120 units) where theirs is one multicast packet a
program. The 1.5 GB figure is from 2026-10-15 for the 1.7B; later packing cut the 135M's four traces from 3.75 GB
to 563 MB, so re-measure before quoting it.

### 4.2 Dispatch cost (row 2)

Blackhole traced dispatch, `tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/pgm_dispatch_blackhole_golden.json`
(kernels of 256 B to 12 KiB): about 1.29 us a program minimal, 1.67 us with CBs on all cores, about 4.6 us with 8
kernel groups or 1 runtime argument on all cores, 5.7 us with 128 runtime arguments, 8.65 us kernel groups plus 1
argument, 9.8 us groups plus 128. Untraced: about 2.6 us minimal, 12-20 us with kernel groups, growing with
kernel size (2.6 to 3.85 us at 8 KiB). **Op-to-op gap** (`op_to_op_latency/op_to_op_blackhole_golden.json` ✓):
`official_op2op_us` 2.08, `pack_to_unpack_op2op_us` 4.60, 8 traced programs. `llms.md:1826`: tracing reduces the
gap "below 6us"; an untraced example shows 756 us between ops against 362 us on device (architecture not
stated). Dispatch cores: default type `WORKER` (Tensix), Blackhole defaults to the column axis
(`dispatch_core_common.cpp:19-41`), `dispatch_s` co-located with the dispatcher; about 2 Tensix tiles a command
queue on one chip *(inferred, not confirmed)*. No published trace-replay start latency (ours is about 170 us).
Our 4.1 us an op is inside their 1.3-4.6 us traced range.

### 4.3 L1 residency, sharding and weight streaming (row 5)

- Allocator (`tech_reports/memory/allocator.md:13,36-44`): DRAM bottom-up, L1 top-down, CBs after the reserved
  region; a validation step before any program runs checks CBs and L1 buffers do not overlap (`program.cpp:2537`).
- L1-sharded activations between ops are configured by hand per model ("activations in L1, weights in DRAM",
  `llms.md:1355`; the residual stream stays width-sharded, `llms.md:877`; prefill intermediates stay in DRAM,
  `llms.md:1140`). In a trace addresses are frozen: an L1 input must be re-allocated at the same address
  (`AdvancedPerformanceOptimizationsForModels.md:126-150`). **No static plan over a captured graph was found.**
- Weight streaming: `prefetcher_matmul_design.md` has a worker-core prefetcher (`ttnn.dram_prefetcher`, a global
  circular-buffer ring) and a **Blackhole DRISC tensor prefetcher** (`ttnn.experimental.start_tensor_prefetcher`,
  queueable; needs firmware 19.12.0.0 or newer; about 70 KB working L1; "can run at higher throughput on
  production shapes"). On Llama-3.1-8B it covers FF1, QKV and O with ring 64; FF2 uses a plain DRAM-sharded
  matmul.

### 4.4 Attention (row 7)

`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/`, `tech_reports/FlashAttention/FlashDecode.md`. K and V are
streamed from DRAM in chunks (validation: "buffer type must be DRAM", `sdpa_decode_device_operation.cpp:170`);
default chunk `min(512, largest power-of-2 divisor of s)` (`sdpa_decode.cpp:21-35`); K/V CBs are double-buffered
at `Sk_chunk_t * DHt * 2` tiles (`sdpa_decode_program_factory.cpp:448-449`), so the L1 footprint depends on chunk
size and head dimension, **not on kv length**; the kv loop is driven by `cur_pos` at runtime. Split-K: several
cores a kv head with a tree reduction and semaphores, at most 16 cores a head-batch
(`max_cores_per_head_batch`; the factory defaults to 16, line 144, while `sdpa_decode.cpp:19` has
`kDefaultMaxCoresPerHeadBatch = 1`: not reconciled). Paged mode takes a page table and `cur_pos` tensor; dtypes
BF16, BF8 and BF4. The report benchmarks head_dim 64/128/256 up to 128K positions; about 180 of 250 GB/s observed
maximum on Wormhole. Our limit (a head's whole K and V in one unit's L1; 512 positions at head_dim 64, about 864
at most) is **design**. tt-metal's chunk is also 512, but it is a streaming chunk. Closing it for us means a
streamed attention, and flash-decode's order change is the lever our bit-for-bit rule parked
([split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md)).

### 4.5 SFPU, replay and approximate math (row 8)

- tt-metal already uses `.ttinsn` for every `TTI_*` macro: `ckernel_ops.h:7-8` (`TTI_INSN` is `.ttinsn %0`;
  `TT_INSN` at line 9 is the MMIO store). The Blackhole march string is
  `rv32im_zmmul_zaamo_zba_zbb_xtttensixbh_zve32f` (`bh_hal.cpp:271`). **No cycles-per-word figure appears in the
  tree**; our 1.19 against 3.31 is not documented upstream.
- **Instruction gathering** matches our finding with an opt-in. `firmware_common.h:300-320`
  (`configure_gathering`): on Blackhole, unless `ENABLE_GATHERING` is defined, bit 18 of CSR 0x7c0 is set citing
  tt-metal#16439. `TT_METAL_ENABLE_GATHERING=1` sets the JIT define (`rtoptions.cpp:769-773`,
  `jit_build/build.cpp:469-471`; PR #22424). With it on, `load_replay_buf` (`ckernel.h:680-697`) brackets each
  record with `disable_gathering()`/`enable_gathering()`. Hazard: a replay record while gathering is on. If we ever
  re-enable gathering, bracket our record windows the same way.
- **Replay buffer** `REPLAY_BUF_SIZE = 32` per thread (`ckernel_structs.h:48`), conventionally split at offset 16
  between FPU and SFPU users (`cmath_common.h:29`).
- **tt-metal's answer to push-bound code is replay plus `SFPLOADMACRO`**
  (`hw/ckernels/blackhole/metal/llk_api/llk_sfpu/`): accurate bf16 exp records a 15-17 instruction body once and
  issues one `TTI_REPLAY` a row (`ckernel_sfpu_exp.h:185-345`); approximate exp uses `SFPLOADMACRO` plus replay,
  "~20 cycles for 8 elements = 2.5 cycles/element", "~68 cycles for 32 elements = 2.125 cycles/element"
  (lines 649-697); reciprocal "approximate, with throughput of 1c/32", "FP32 reciprocal, with throughput of
  5c/32" (`ckernel_sfpu_recip.h:61,157`, 6-slot replay at line 363, `DISABLE_SFPLOADMACRO` fallback). **Precise
  gelu has no replay** (plain SFPI loops, `ckernel_sfpu_gelu.h:295-340`), consistent with our push-bound finding.
- **Defaults.** `ttnn.exp/erf/log/rsqrt/sqrt` default `fast_and_approximate_mode=False`; `ttnn.gelu` defaults
  `GeluVariant.Accurate` (`unary_nanobind.cpp:317,859,904,973`). Softmax defaults `math_approx_mode=true`, HiFi4,
  fp32 accumulation true (sharded: only for fp32 input; `softmax_device_operation.cpp:403-434`); layernorm
  approx false, fp32 accumulation true, HiFi4; rmsnorm approx **true**, fp32 accumulation false, HiFi4.
  Softmax kernels are chain-fused (`softmax_sharded.cpp`: sub-bcast, exp, pack in one `eltwise_chain`). No
  per-tile cycle cost for softmax, layernorm or rmsnorm is documented.

### 4.6 Matrix-unit numerics (row 11)

`matrix_engine.md:64-71` (Wormhole-oriented, notes the numbers apply to Blackhole): 5b x 7b multipliers; LoFi
uses the 4 and 6 most significant mantissa bits, HiFi2/3/4 add least-significant passes. Blackhole peak per engine
at 1.35 GHz: LoFi about 5.4, HiFi2 about 2.7, HiFi4 about 1.35 TFLOPS (`GEMM_FLOPS.md:56-69`).
`accuracy_tips.md:58,142`: FP32 operands pass through TF32 SrcA/SrcB even with an fp32 destination, the closest
upstream statement to our "TF32-grade" finding. `ComputeKernelConfig` struct defaults: LoFi, approx true, fp32
accumulation false, L1 accumulation false. **`ttnn.matmul` actual defaults** (`matmul_device_operation.cpp:2942-2992`):
HiFi2 with no program config and no user grid unless both inputs are bfp8/bfp4 (then LoFi); HiFi4 on Blackhole
when both inputs are fp32; approx false; fp32 accumulation only for an fp32 output; `packer_l1_acc = !fp32_output`.
`tt_transformers` (`model_config.py:392-416`): BFP8 weights with HiFi2; MLP `hifi2_fp16` (fp32 destination off,
L1 accumulation on); QKV, O and SDPA decode HiFi2 with fp32 destination on; SDPA prefill HiFi4; the comment at
`:920-922` says fp32 accumulation "does not appear to be needed". **Our two-groups-of-eight, align-once MVMUL
model (`step158`) is not documented in tt-metal.** Our 24 `precision` conformance failures therefore compare
against a stack that runs lower fidelity by default.

### 4.7 Multi-chip (row 12)

- **Supported.** `ClusterType::P150_X2/X4/X8` are auto-detected by chip count (`tt_cluster.cpp:187-198`; the
  description is "2 Blackhole single card, ethernet connected"). `p150_x2_mesh_graph_descriptor.textproto` is a
  [1,2] mesh with 4 channels; `test_system_health.cpp:416-418` expects 2 chips and 4 connections a side. `tt_ccl.py`
  lists p300 and p150x4 at (2,2) links; `CCL_Performance_Best_Practices.md:49` says "2 for BH". Whether our two 800G
  QSFP links are those four channels was not verified.
- **Latency model** (`tech_reports/CCLs/CCL_bandwidth.md`): line rate 400 Gbps = 50 GB/s a link a direction, about
  96% payload (48 GB/s; our 48-51 GB/s one-link result matches); **hop latency 515 ns 1D, 619 ns 2D**
  (lines 97-106); fixed cost 1.4 us launch, 0.7 us open, 0.3 us + 0.64 us a hop barrier (lines 112-120); BH
  `all_gather` line = 3.7 us + 1.3 us x (N-1) + (N-1) x bytes / 48 GB/s (lines 221-226). Galaxy links run at half
  rate, so those tables normalise against 25 GB/s (line 201).
- **Small transfers, n=2, line, DRAM, 8 traced calls** (`results/FULL_blackhole_bfloat16_8192.md:573-585`):

  | Op | 4 KiB | 8 KiB | 16 KiB |
  |---|---|---|---|
  | `all_gather` | 4.97 us | 4.55 us | 4.86 us |
  | `reduce_scatter` | 4.70 us | 4.74 us | 5.05 us |
  | `all_reduce` | 9.50 us | 9.99 us | 10.75 us |

  These come from Blackhole Galaxy-style runs, **not cabled p150 cards**. On p150_x4 the fabric golden
  (`tt_fabric/test_infra/golden/golden_latency_summary_blackhole_p150_x4.csv`) shows about 1.36 us net and 0.68
  us a hop at 16 B, about 1.64 us net and 0.82 us a hop at 4 KiB (the hop count behind "per hop" was not
  confirmed). Bandwidth golden: 38.4 GB/s at 4,096 B, 41-45 GB/s at 8-15 KB packets, **not scaling with
  `num_links`**. Mux throughput golden (on-chip, not wire): 10.0, 19.5, 39.1, 52.0 GB/s at 1, 2, 4, 8 buffers a
  channel: pipelining depth is the lever. Nothing in the tree reproduces our 27 GB/s on both links; CCLs use
  `num_links=2` on Blackhole (`llama31_8b_qb2/tt/decoder.py:443-468`).
- **Device-to-device copy with no host:** `ttnn.experimental.send_async`/`recv_async` over `MeshSocket`
  (`Programming_Multiple_Meshes.md:509-530`); TT-Fabric has routing planes (2 a direction on Blackhole) and uses
  both ERISCs (`TT-Fabric-Architecture.md:388,422`). We have no equivalent. CCLs: `all_gather_async`,
  `reduce_scatter`, `all_reduce`, `all_reduce_async`, `rms_allgather`, `llama_reduce_scatter`, `minimal_matmul`
  under `ttnn/cpp/ttnn/operations/{ccl,experimental/ccl}`.
- **Tensor-parallel decode on Blackhole:** only the QB2 TP4 demo (4 dies, 1x4 ring): per layer `all_gather_async`
  plus `reduce_scatter`, decode settings `chunks_per_sync=1`, `num_workers_per_link=1`, `num_links=2`, persistent
  buffers (`decoder.py:439-470`); the README reports 131.08 t/s/u in CI. `models/README.md:22` lists Llama-3.1-8B
  DP=2 on two p150 at 18.6 t/s/u (**data-parallel, no inter-card traffic**). There is no TP=2-on-two-p150 demo
  (`tttv2_llama3_8b_bh_required_capabilities.json` has only `p300_tp1_dp2` and `p150x4_tp4_dp1`). One decode
  datapoint: BH `all_gather` of `[1,1,768,256]` takes about 54 us naive, about 45 us with pre-allocated buffers and
  8,192 B packets (`CCL_Performance_Best_Practices.md:127`).
- **What it means for us.** [sram-residency-and-multi-card-decode.md](../proposals/sram-residency-and-multi-card-decode.md)
  says two-card tensor parallelism is dead above about 20 us a hop. tt-metal's all-gather is about 5 us at 4-16
  KiB, so 96 gathers a step would cost about 0.5 ms, inside the proposal's 0.2-0.5 ms estimate. **That is
  tt-metal's fabric on different cabling, not our E1 movers**; the first step is still our own measurement (F2).

### 4.8 DRAM bandwidth and the NoC (row 9)

- Blackhole: 8 banks of about 4 GB, 64 B read alignment (`BlackholeBringUpProgrammingGuide.md:42-57`); 8
  channels with 3 NoC endpoint tiles each (`blackhole_140_arch.yaml`). `CCL_bandwidth.md:84-93`: "~64 GB/s (8
  channels, ~512 GB/s aggregate)", measured 64 GB/s a bank.
- Microbenchmark `perf_microbenchmark/8_dram_adjacent_core_read/test_dram_read.cpp:282,552-553`: spec constant
  `BH_DRAM_BANDWIDTH_GB_PER_SEC = 512`, pass goal 90%. **CI gate** `tests/scripts/test_moreh_microbenchmark.py`
  ✓: `bw_lower_bound = 490.0` (line 815; Wormhole 250-270). The gate's pattern: **8 readers, one on each bank,
  each on the tile beside the bank's NoC #0 endpoint, each streaming its own bank** in packets of up to 16 KiB
  (`get_optimal_dram_bank_to_logical_worker_assignment`, `reader_dram.cpp`: three blocks in flight on transaction IDs
  1-3, a static VC `bank & 3`); the CI shapes are `k` 524,288 with `n` 1,024 (BFP8, BF16) or 2,048 (BFP4), and a
  padded FF1 shape of under 2 MiB a bank; the metric is all bytes over the kernel's device duration.
- **We mirrored it** (`silicon_bench_memory::gddr_adjacent_banks`, 2026-10-26, card 1, runs `1793028574` and
  `1793028275`; [log](performance-log.md#tt-metals-adjacent-core-dram-read-gate-on-our-mover-2026-10-26-card-1)):
  placement from the same SOC descriptor (on card 1 the east readers are at raw x 11, since column 10 is harvested,
  as tt-metal's own rule gives), the gate's metric and bound, tt-metal's packet sizes.

  | | safe image | `fast-reads` |
  |---|--:|--:|
  | BFP8 / BF16 / BFP4, packet entries (16,320 / 16,384 / 16,128 B) | 414.1 / 413.9 / 409.2 | **508.2 / 507.7 / 507.8** |
  | FF1 shape, packet entries | 405.7 | **496.1** |
  | BFP8 / BF16 / BFP4, 128 KiB entries | 511.1 / 511.2 / 511.2 | **511.2** |

  **The gate (490-512) is met as shipped with a `fast-reads` image, for every format**; the safe image misses it with
  packet entries (one tile issues at most about 52 GB/s of 16 KiB reads at 359 cycles an entry, against the 64 a bank
  supplies) and meets it with long ones. One reader on one endpoint in long runs gives **63.9 GB/s a channel, 99.8% of
  the reasoned 64**: so **our 487-489 GB/s weight stream and 496 GB/s 120-tile aggregate are not DRAM's limit**. What
  holds them below it is the many readers an endpoint serves and the crossing traffic, and the access pattern this gate
  uses (a bank read by the tile beside it) is what tt-metal's DRAM-sharded matmul is built on; whether a bank-sharded
  weight layout would raise decode's rate is F27. Differences from tt-metal's kernel (a compiled reader with a static
  VC and two 1 MiB blocks in flight; each of our readers streams 8-64 MiB, a list holds 512 entries, against the
  gate's 68-128; no fast-dispatch core reservation) are listed in the log.
- **Earlier, against v0.7.7, interleaved reads from N cores** (`crates/tt-tests/bench/tt_metal_dram_read/`): at 4 KiB
  on the full grid ours is 491 GB/s (120 tiles; tt-metal's 110 cores) against 415, at 16 KiB 496 against 387, at 2
  KiB 446 against 378; tt-metal wins at 1-8 cores (a compiled reader, about 125 cycles a 4 KiB page against our 175
  fast or 359 safe) and ours wins at 16 KiB on one core (85 against 69 GB/s: our window of requests slides, theirs
  drains at each barrier) ([log](performance-log.md#tt-metal-reference-dram-reads-by-core-count-2026-10-13-card-1)).
  tt-metal reserves cores for fast dispatch (110 of 120), and that tree is v0.7.7. tt-metal's own binary for the
  new gate has not been run on this card.
- **SYS-1419 is documented upstream.** `tests/tt_metal/tt_metal/api/test_dram.cpp:363-366`: "DRAM arbiter seems to
  drop requests from one noc when both nocs are issuing requests to the same DRAM endpoint at a fast rate"; the
  repro is kept DISABLED (a reshape test also cites it, `test_reshape.py:985`). `blackhole_140_arch.yaml:25`
  gives each DRAM view a different subchannel per NoC with NoC0 pinned to the endpoint the syseng firmware uses;
  `metal_soc_descriptor.hpp:56-62` excludes that endpoint from usable DRAM cores. Same one-NoC-per-endpoint policy
  as our `DramChannel::owns`; the port numbering was not compared. `kernel_types.hpp:141-153`: NoC0 is the default
  read NoC, NoC1 the default write NoC. No "NoC 1 split" document or one-link cap on NoC1 reads (our 58-86 GB/s
  observation) was found.
- **A hazard worth checking in our readers:** `test_dram_sharded_then_1d_matmul.py:1-24` documents a NoC VC state
  leak after the DRAM-sharded reader that costs about 20% of DRAM read bandwidth; the fix is to reset VC=1.

### 4.9 Firmware, driver and Ethernet cores

- `INSTALLING.md:29-33`: Blackhole needs KMD 2.8.0 or above, firmware bundle 19.8.1 and tt-smi 5.0.0 or above;
  the README release table stops at 0.68 and is stale. `llrt.cpp:587` hard minimum: tt-firmware 18.10.0.
  `firmware_capability.cpp:29,47` gates features on Blackhole: 2-ERISC mode needs eth firmware 1.7.0+; DRAM
  programmable cores need bundle 19.12.0+ ("earlier firmware puts syseng firmware on a DRAM core"); PTP trace
  needs eth firmware 1.8.0+. The `tt-umd` submodule was not checked out, so the Oct-2026 KMD minimum was not read;
  the v0.7.7 checkout lists 2.0.0 minimum, 2.9.0 for read-only page pinning, 2.10.0.1 for TLB dmabuf. Our pin
  `ttkmd-2.11.0` is above all of them.
- Reset advice (`CONTRIBUTING.md:577`): `tt-smi -r 0`, power-cycle if that fails. v0.7.7 UMD hang tests say a NoC
  hang on Blackhole recovers by warm reset, but on ARM64 p100 it hung the whole host. No ARC-watchdog
  documentation for Blackhole was found.
- **E0/E1 matches our practice.** `active_erisc.cc:298-302`: kernels on ERISC1 use NoC1 "which ensures no
  conflict with base firmware running concurrently on ERISC0"; line 240: "Base FW on ERISC0 uses NoC0
  concurrently, so only touch our own NoC". `enable_2_erisc_mode` defaults true (`rtoptions.hpp:347`);
  `TT_METAL_DISABLE_MULTI_AERISC=1` falls back to one ERISC. tt-metal's 2-ERISC mode resets E0 after base
  firmware saves state (`active_erisc.cc:157-160`, `bh_hal.cpp:117`); we never touch E0.
- Harvesting: `blackhole_140_arch.yaml` has 140 functional workers with no harvesting; handling lives in UMD,
  which was not read.

## 5. Ahead, behind, not comparable

**Ahead of tt-metal**
- Bit-exactness against an independent reference (tt-metal does not attempt it and documents batch-variant
  results); strict native-execution checks; negative-control gates.
- F32 and BF16 numerics at comparable bandwidth efficiency; default matmul more accurate than their LLM stack's.
- A static whole-graph L1 residency plan (theirs is hand-configured sharding).
- Per-op fixed cost inside their traced range; DRAM reads meet their CI gate (507.7-511.2 GB/s against 490-512,
  `fast-reads`, card 1) and a bank read from the tile beside it reaches 99.8% of the channel.
- A pure-Rust stack: no Python, no TTNN, no FFI in shipped crates.
- Measurements upstream does not state: `.ttinsn` push cost (3.31 vs 1.19 cycles a word), push-bound word counts
  (precise gelu 516 stored and 11,645 pushed; softmax `SUB_EXP`/`DIV` bodies of 57 and 62 words overflowing the
  32-entry replay buffer), the MVMUL group-of-eight alignment model, the two-link 27 GB/s result, the 120-tile
  DRAM aggregate.

**Behind tt-metal**
- Prefill throughput (about 10x, partly fidelity), and quantised formats (BFP8 KV, BFP4/BFP8 weights).
- On-device trace-compatible sampling (about 100x slower for non-greedy).
- Batching up to 32, paged KV, prefix caching, vLLM integration, async decode, grammar.
- Multi-device: fabric, CCLs, sockets, tensor and data parallelism; prefetcher (DRISC).
- Model coverage: QKV bias, RoPE scaling, sliding window, MoE, DeltaNet.
- Streamed attention with a runtime length; traces independent of kernel size.
- Firmware and driver version gating.

**Not comparable**
- Model and hardware: their Llama-8B (8 KV heads, GQA) and Wormhole 1B/3B numbers against our 32-head MHA 1.7B on
  p150a; BFP4/BFP8 reduces bytes read, so their percentage of peak can be lower while their absolute speed is
  higher; their TTFT is per user at 128 tokens and includes first-token sampling.
- Galaxy CCL tables and mux throughput against two cabled p150s and raw Ethernet.

**Not found or stale in tt-metal** (do not infer from absence): a Blackhole LLM DRAM-utilisation figure; Blackhole
LLM prefill TFLOPS; p150 numbers for Llama-3.2-1B/3B or Qwen; a published cost for on-device sampling; a
per-model maximum context for p150; a batch-1 p150 yaml perf entry; whether a p300 die has 512 GB/s (assumed);
the replay-start latency; bytes an op adds to a trace; trace capture time; a confirmed Blackhole dispatch-core
count; any cabled-p150 hop latency; the Oct-2026 UMD KMD minimum; per-tile softmax, layernorm and rmsnorm cycle
costs; a Blackhole DRAM-sharded matmul GB/s.

## 6. Fixes to investigate

Each item names why, the first measurement, and what would settle it. Nothing here is scheduled; the roadmap
links this section. Priorities are our judgement. Sections 4.x have the evidence.

### Close the evidence gap first

- **F1. Head-to-head baseline on card 0 (highest priority).** Run tt_transformers on the same card and model
  (SmolLM2-1.7B, listed experimental, or Llama-3.2-1B) and record decode ms/token, TTFT, prefill tokens/s, and
  the precision settings, beside our numbers on the same day. Without it every "same band" above stays derived.
  Needs a tt-metal build (`/mnt/nvme/tt-metal/build` exists; version-match the SFPI and firmware, as
  `crates/tt-tests/bench/ttnn_reference.py` notes), card 0 only, and `tt-smi -r` first because our E1 image
  stays on the cards. Also fixes the weights-format mismatch by running both at BF16 and at their default BFP8.
- **F11. (Mostly done 2026-10-26.) The DRAM read gate.** Our mirror of tt-metal's `8_dram_adjacent_core_read` gate
  exists and passes (section 4.8: `gddr_adjacent_banks`, 507.7-511.2 GB/s with `fast-reads`, card 1). Left: run
  tt-metal's own binary for it on card 1 (build `a11d86c19d3` or newer, `TT_VISIBLE_DEVICES=1`, `tt-smi -r` first
  because our E1 image stays on the cards) and set the two beside each other; rebuild the v0.7.7 interleaved harness
  for the current tree; the same gate on card 0; and make `gddr_adjacent_banks` with `ADJ_GATE=1` part of the silicon
  smoke or a nightly so a regression of the read path is a failure, not a number.
- **F12. Check our readers for the VC-state leak** that costs tt-metal about 20% of DRAM read bandwidth after a
  DRAM-sharded reader (`test_dram_sharded_then_1d_matmul.py:1-24`): does any of our reader or NC path leave a NoC
  virtual-channel state that slows the next product? Probe: the same product measured first and after a
  channel-changing op.

### Multi-card

- **F2. Measure one card-to-card Ethernet hop with our E1 movers** (4-16 KiB, L1 to L1, no host, median of warm
  runs on cards 0 and 1) and record it in the scoreboard. This is the proposal's go/no-go (kill threshold about
  20 us); tt-metal's published all-gather is about 5 us. Then study why our two links give 27 GB/s where theirs
  reaches the per-link rate with `num_links=2` and deeper pipelining (mux 1 to 8 buffers a channel: 10.0 to 52.0
  GB/s), and read `send_async`/`recv_async` and the fabric router as the design for a host-free card-to-card
  copy.
- **F13. Two-card topology.** Confirm whether our two QSFP links are the four channels in `p150_x2` and whether
  tt-metal's `p150_x2` fabric runs on our cabling at all; this decides whether we can borrow their fabric as an
  oracle.

### Programs, traces and attention

- **F3. Make program size independent of length.** List every kernel still unrolled per tile (element-wise
  unpack/pack, reductions; roadmap "The composition does not build at long lengths") and move the tile count into
  an argument and a loop, as `eltwise_sfpu.cpp:18-23` does. Measure the per-tile cost of the loop against the
  unrolled body; a replay-friendly loop may be free. Settles the 2,048 and 8,192 build failures at the root.
- **F4. Trace size.** Re-measure the 1.7B's decode and prefill traces (the 1.5 GB figure is from 2026-10-15) and
  compare with tt-metal's 56-70 MB for an 8B decode. Investigate sharing identical entries across units (a
  multicast or repeated-section form; our traces are per-unit streams, theirs one packet a program) and what the
  sub-trace work (5.6x) leaves. Report GB of trace per layer.
- **F5. Streamed attention.** A decode attention whose K and V are read from GDDR in chunks with a runtime length
  would remove the 512-position reach and the composition fallback. First decide whether a **streamed variant that
  keeps the composition's reduction order** (multiple passes over K chunks, no split) is possible, since
  flash-decode's order change was parked
  ([split-k-and-flash-decode.md](../proposals/split-k-and-flash-decode.md)); tt-metal accepts the order change
  (split across up to 16 cores, tree reduction). Measure at 512, 1,024 and 8,192 positions.
- **F14. Trace the prefill only where it pays.** tt-metal traces prefill only at 128 and 1,024 tokens and
  declines it for longer sequences ("op2op gaps are already small"). Check whether our long-prefill traces
  (8K, 893 s first build at chunks of 512; sub-traces make a prefill trace's layers 8.4x smaller) buy anything over eager chunks past a length, since a
  trace's cost there is build time and GDDR, not step time (build time is not a target, but GDDR is).

### Speed

- **F6. SFPU transcendentals by replay and `SFPLOADMACRO`.** Read `ckernel_sfpu_exp.h:185-345` (accurate bf16 exp,
  15-17 instruction body, one replay a row) and `ckernel_sfpu_recip.h` and compare with our staged row loops and
  fused `.ttinsn` push for `SUB_EXP` and `DIV`. The question is whether their accurate (not approximate) path is
  bit-exact against our oracle; our bit-for-bit rule forbids the approximate paths, but the accurate ones may be
  cheaper than our staging. Measure cycles a tile on card 0 for `exp`, `recip` and softmax.
- **F9. Separate fidelity from structure in prefill.** Our products at about 5-18 TFLOPS are F32 at HiFi4; theirs
  are BFP8/BFP4 at HiFi2/LoFi with 2D multicast. Measure our blocked products at HiFi2 and LoFi to see how much of
  the 10x is fidelity, then use the multicast primitive (G7: done, gated, unused by `tt-kernels`) for Step 2 of
  [prefill-operand-reuse.md](../plans/prefill-operand-reuse.md). Note their M cap of 512 rows and 8x8 grid on
  Blackhole.
- **F10. DRISC tensor prefetcher.** Read `prefetcher_matmul_design.md` and `tensor_prefetcher.hpp`: does a DRAM
  programmable core prefetch weights without a Tensix unit, and what firmware (19.12.0.0+) does our card run?
  Compare with our idle-unit prefetch and the NoC-endpoint rules (SYS-1419); it might free the L1 and units our
  prefetch uses.
- **F16. Per-op fixed cost.** Our 4.1 us against their 1.29 us minimal: split our 3.5 us of control (descriptor
  pokes, `PAIR_CALL` launch and credits) from the barrier, and find what a minimal program would cost. The roadmap
  puts barriers alone at about 1.5 ms of a step (replay-fixed-cost P1); each microsecond saved is worth as many
  microseconds a step as the step has ops, so count the current ops a step with the profile mode first.

### Features

- **F7. Native top-k and trace-compatible sampling.** 112 ms a token against 12 ms greedy; tt-metal uses
  `ttnn.topk` plus `ttnn.sampling` in a separate sampling trace, top-k at most 32, per-request seed skipping the
  trace. Decide our cap (512 today), tie rule (ours drops a tie with the k-th where `transformers` keeps it) and
  whether sampling replays inside the decode trace.
- **F8. Quantised weights and KV as opt-in modes.** BFP8 weights and a BFP8 KV cache are tt-metal's defaults and
  would roughly halve the BF16 floor again (7.3 ms of an 11.76 ms step). Our BFP formats have only correctness
  checks (135M, six layers). Measure BFP8 and BFP4 decode on the 1.7B and the accuracy cost (top-1 agreement,
  logit drift), keeping BF16-weights exact decode the default; the user approved BF16 only.
- **F15. Model coverage.** Read tt_transformers' handling of QKV bias (`attention.py:182-204`), RoPE scaling
  (`rope.py`), sliding window (`model_config.py:2917-3042`) and MoE, and list what each needs in `tt-models`
  (`rope_theta` is one float today; `bias: None`). The first candidate is Llama-3.2-1B or Qwen2.5-1.5B.
- **F17. Batching, paged KV and serving.** Their pattern: batch 32, paged cache, page table, prefix cache,
  per-batch sampling trace, vLLM plugin. Reconcile with [batched-decode.md](../proposals/batched-decode.md) and
  add their KV-capacity ceiling (every request needs its own KV on every dense-layer device).

### Hardware, drivers and safety

- **F18. Firmware and driver gating.** Read the ARC firmware bundle version at attach and refuse (by name) a
  bundle below a tested minimum; tt-metal gates features at 18.10.0 (hard minimum), 19.8.1 and 19.12.0 (see
  4.9). Record the version in every silicon log header.
- **F19. Upstream's errata as a checklist.** Re-read tt-metal#16439 (gathering), SYS-1419 and the VC leak above
  against `silicon-operating-notes.md`; add what they know and we do not.

### Repository hygiene (from section 2)

- **F20.** Run the full simulator suite on branch pushes (or at least the affected gates), and publish a silicon
  smoke result per commit from a manual or self-hosted runner.
- **F21.** Add a `LICENSE` file, decide the CC BY-ND question for `tt-isa/src/isa/generated.rs`, and check
  ttsim's and MNIST's terms.
- **F22.** Pin the Rust toolchain to a version and add `timeout-minutes` to CI jobs; revisit the firmware
  slots' compiler dependence.
- **F23.** Replace the "Option set at this phase" unwraps in `tt-kernels/session.rs` with typed errors, starting
  with paths reachable from a live device session; split `session.rs`, `tensor.rs`, `ops.rs`, `server.rs`.
- **F24.** Tighten the ship checks: extend `check-no-flex-in-backend` beyond `burn-tt` and the two sim crate
  names to FFI crates generally, resolve `tt-mnist`'s `burn-flex` dependency, and fail (not warn) when the C
  oracle is skipped on a CI or silicon run.
- **F25.** Refresh stale docs: roadmap lines 17 and the conformance headline, the scoreboard's roofline "now" row
  and the contradictory traced rows, `tt-models/README.md`'s "three models", the missing `docs/README.md`
  entries, finished plans into `completed-plans/`, `PINS.toml`'s `burn-flex` comment, and the `README.md` claims
  about generated files and when CI runs e2e.
- **F26.** Remove the stale `.claude/worktrees/` and ignore `.agents/`, `.claude/`, `.mcp.json`.

### From the measured gate

- **F27. Bank-sharded weights, each channel read by the tiles beside it.** The mirrored gate moves 63.9 GB/s a channel
  from one tile (99.8% of 64); decode's weight stream reaches 429 GB/s of 512 (F32) and the 120-tile aggregate 496,
  with each weight's tiles spread round-robin over the channels and read by whichever units own the column. tt-metal's
  DRAM-sharded matmul is the bank-local layout. Investigate, before building anything: (1) the same gate with two,
  four and eight tiles beside each channel (does a channel still give 63.9 GB/s when 2-4 tiles share its endpoint, and
  from what count does it fall), to find what the crossing traffic and the shared endpoint cost; (2) with that, what
  a decode product's rate would be if each unit read only the weight tiles of its adjacent channel (the deal of
  columns to units would follow the channel, not the column); (3) whether `gate|up`/`down`'s units can take the
  activations multicast it needs and what the L1 and the prefetch ring cost. The decode floor at BF16 is about 7.3
  ms of weights at 469 GB/s; at 511 it is 6.7 ms. A hypothesis, not a measurement of a decode step.
  **Probed 2026-10-27 and stopped** (card 0, [log](performance-log.md#f27-probed-bank-local-reads-and-what-does-not-bind-a-decode-product-2026-10-27-card-0)):
  (1) at decode's 2,112-byte entries a tile reads ~6.5 GB/s (safe) or ~20-35 (`fast-reads`) whatever channel it reads, so 8
  readers beside each channel give 425 / 466 GB/s, the same as today's interleaved deal on as many units (424 / 466; 96
  units 449 / 491); with 16 KiB or longer entries 2-4 readers a channel give 509-511. The decode products already read every
  channel evenly (`channels.tsv`). (2) With reads 5x cheaper (`fast-reads`) the products' spans move 4% (`down` 120.8 ->
  114.9 us, the step 10.82 -> 10.60 ms): a unit's read rate does not bind them, so neither a bank-local deal nor a
  long-entry layout pays for decode. Not built.

## 7. Caveats

- tt-metal moves fast; line numbers are for `a11d86c19d3` and will drift. The method file explains how to read
  them at a pinned revision without disturbing the working checkout.
- The reading agents mixed Wormhole and Blackhole sources in places; every Wormhole-only figure above is
  labelled. The README and the yaml disagree in places (3.1); the yaml is the newer source.
- "Implied GB/s" and "effective TFLOPS" are derived; a published number is always preferable. F1 replaces them.
- This repository's own figures were taken from its docs and not re-measured; the roadmap and scoreboard carry
  known staleness (section 2, risk 8). Check the run IDs before quoting.
