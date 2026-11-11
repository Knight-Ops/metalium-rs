# Achievements

This is the record of what the project has achieved, moved out of
[`../plans/roadmap.md`](../plans/roadmap.md) on 2026-11-09 so that the roadmap holds only open work. Each entry
keeps the numbers as they were recorded at the time, with their date and run IDs; later work has moved most of
them. The current numbers live in [performance-scoreboard.md](../learnings/performance-scoreboard.md), and the run
history in [performance-log.md](../learnings/performance-log.md). Approaches that were tried and killed or retired
are recorded separately; here they appear only where a win replaced them.

Cards: card 0 runs gates, card 1 runs benches, unless an entry says otherwise. "Bit for bit" means the device result
equals the reference (eager path, host model or burn-flex, as stated) in every bit.

## Foundations and hardware coverage

- **Hardware instructions: done.** Phase 10 closed ([close-out record](hardware-coverage-closeout.md),
  [coverage matrix](hardware-coverage.md), [phase 0-10 checklist](phases-0-10-checklist.md)). The Burn operation
  traits have no unsupported float, int or bool method ([burn-op-coverage.md](../plans/burn-op-coverage.md)).
- **The Matrix Unit's group adder, modelled** (2026-10-11, gate `step158_matmul_accumulation`, run `1791754387`):
  `MVMUL` adds its 16 products in two groups of eight, aligned once to a quantum of 2^(largest exponent sum - 10)
  with half-up rounding; the model equals the device bit for bit
  ([silicon-operating-notes.md](../learnings/silicon-operating-notes.md), "Matmul accumulation").
- **Host-encoded reads, `fast-reads`** (2026-10-13, then fast-mode hardening P0-P2, P7): a read entry 359 -> 175 ->
  91 device cycles (109 with `encoded-guard`); one tile reads 57 GB/s at 4 KiB (3.8x the safe image), 8 tiles 340
  GB/s (2.9x). The checksum is bound to a per-start key, tile, chip and op; `enqueue_checked` validates every
  entry; gate `step173_mover_differential`. Reads and writes name only the data arena, checked on the host in both
  modes ([log](../learnings/performance-log.md#host-encoded-read-entries-2026-10-13-card-0),
  [read-loop probe](../learnings/performance-log.md#read-loop-probe-what-the-interpreter-costs-per-read-entry-2026-10-13-card-0)).
- **Firmware image budget** (2026-10-13): the role images repacked into 10 KiB slots (`tt_isa::l1::roles`), so B's
  mover slot is 42 KiB, not 24, at the same speed (smoke 1,030 of 1,030, `1791855592`;
  [log](../learnings/performance-log.md#firmware-image-budget-and-core-assignment-2026-10-13-card-0)). `cargo clippy`
  had inflated every image by 900 bytes; `build.rs` now strips `RUSTC_WORKSPACE_WRAPPER`.
- **The firmware diet** (2026-11-06, cards 0 and 1,
  [log](../learnings/performance-log.md#the-firmware-diet-bs-room-180---10356-bytes-at-the-same-speed-h16-and-h17-killed-2026-11-06-cards-0-and-1)):
  `cargo xtask fw-size`; guarded runs out of the role images (slots 9,600 -> 7,168 bytes); layer sub-traces and
  X9's two-copy layout retired. B's free room 180 -> 10,356 bytes in the tightest build, 4,568 -> 14,628 in the
  safe one, every model within noise.
- **Instruction gathering off in every shipped Tensix image** (2026-10-11, `tt_firmware::disable_gathering`, read
  back at start, silicon-verified on both cards). The build-time instruction gate's `.word` hole was closed: data
  in code is refused in every image.
- **Barriers only on the units an op ran on** (2026-10-14): units without a job catch up with a wait-only
  `BARRIER`. The all-tiles transformer 404-422 -> 96-98 ms/step, all-tiles MNIST 12.9-13.4 -> 6.7
  ([log](../learnings/performance-log.md#barriers-on-the-units-an-op-ran-on-2026-10-14-card-0)).
- **Per-op tile spread** (2026-10-14, `Session::set_spread`, `TT_SPREAD`): a session opens every tile and sizes each
  op's unit count; a catch-up rides in a unit's first list. `Work { share: 64, min: 4 }` is as fast on small ops and
  1.45x on a 2048^2 add ([log](../learnings/performance-log.md#open-all-made-strictly-better-2026-10-14-cards-0-and-1)).
- **GDDR read rate at tt-metal's level** (F11, 2026-10-26): our mirror of tt-metal's adjacent-core DRAM gate,
  `silicon_bench_memory::gddr_adjacent_banks`, reads 507.7-511.2 GB/s with `fast-reads` (card 1, `1793028574`,
  `1793028275`; [tt-metal-comparison.md](../learnings/tt-metal-comparison.md)).

## Burn backend and conformance

Detail: [burn-backend-parity.md](../plans/burn-backend-parity.md), [conformance-to-feature.md](../plans/conformance-to-feature.md).

- **B1, typed `TtError`** (2026-10-11, `burn-tt/src/error.rs`): `try_into_data` and `Backend::sync` return the
  sticky error; strict mode keeps the panic.
- **B2, lazy attach** (2026-10-11): engine registry, `init`/`detach`/`TtConfig`, the card lock in every
  `Session::open_card`, `TtDevice::enumerate`/`device_count`.
- **B10/B11, Burn's own backend tests** (2026-10-11, [`burn-tt-conformance`](../../crates/burn-tt-conformance/README.md),
  `cargo xtask conformance --release`): 1,838 tensor and autodiff tests on ttsim, one a process; 1,806 pass, 32
  listed with a cause (from 163 listed; rounds one to five closed 131). `--exact` 1,818 pass; `--fusion` 1,803.
- **`--accurate` profile** (2026-10-12): 2,989 of 3,025 pass, the 36 listed are F64 and ttsim timeouts that pass on
  card 0 (F32 `1791819292`, F16 `1791820653`); no `precision` entry left.
- **D1, F16 arithmetic** (2026-10-12, gate `step162_f16_arithmetic`, card 0 `1791814459`): F16 widens to F32 on the
  card and rounds once, as BF16 does; two latent BF16 defects fixed on the way. The F16 suites, 1,187 tests: 1,165
  pass.
- **D2, BF16 suites and a silicon conformance lane** (2026-10-12): `tests/tensor_bf16.rs` and `autodiff_bf16.rs`
  (upstream has none), silicon only; 1,187 tests, 1,098 pass, 89 listed with causes, 0 unexpected (`--strict`,
  `1791830816`). `cargo xtask conformance --silicon` and `--reference-flex` added. `grid_sample_2d` in BF16 fixed
  (gate `step164_grid_sample_bf16`, `1791830746`; all 15 `grid_sample` tests pass, `1791830775`).
- **First full silicon pass of F32 and F16** (2026-10-12, `--strict`, card 0, `1791833700`): 3,025 tests, 3,001
  pass, 24 listed (5 F64, 19 `precision`), 0 unexpected, nothing downloaded or staged.
- **No cheating found** (audit 2026-10-12): `burn-tt` has no host arithmetic on tensor values and the conformance
  path links no CPU backend.
- **`interpolate` / `interpolate_backward`**, nearest, bilinear, bicubic and Lanczos3 (2026-10-11,
  [`ops/interpolate.rs`](../../crates/burn-tt/src/ops/interpolate.rs), gate `step155_interpolate`, `1791733008`):
  no new kernel; all 22 of Burn's tests pass; nearest and bilinear equal Flex bit for bit.
- **`scatter_nd` with Mul, Min and Max** (2026-10-11, gate `step156_scatter_nd_reduce`, `1791741849`): all 26 of
  Burn's tests pass; Min/Max equal Flex bit for bit over every special.
- **`conv3d` and `conv_transpose3d`**, forward and three gradients (2026-10-11, gate `step157_conv3d`,
  `1791751188`): the patch tables made generic over spatial dimensions (`Geometry<D>`); bit for bit Flex on exact
  operands.
- **`deform_conv2d` and its backward** (2026-10-12, gate `step160_deform_conv2d`, `1791767178`): forward and all
  five gradients equal Flex bit for bit on exact data; five mutations each fail the gate.
- **Exact matmul, opt-in** (2026-10-11, `MatmulMode::Exact`, gate `step159_exact_matmul`, `1791761930`): F32 SFPU
  products summed in increasing `k` from `+0.0`; it settled that all 21 `precision` failures were the Matrix Unit.
  Price: 1,000x-22,000x the fast matmul, ~0.5 us a product (`1791761944`,
  [log](../learnings/performance-log.md#exact-matmul-cost-2026-10-11-card-0)).
- **Accurate matmul, opt-in** (2026-10-12, `MatmulMode::Accurate`, gate `step163_accurate_matmul`, `1791818525`,
  `1791818535`, bench `1791818553`): power-of-two scales and 7/5-bit integer slices make every LoFi product exact;
  error ~1e-7 of the largest entry against the fast mode's 1e-3; 19x-64x the fast matmul, 50x-750x faster than
  exact ([log](../learnings/performance-log.md#accurate-matmul-cost-2026-10-12-card-0)). New SFPU kinds `POW2_FLOOR`,
  `POW2_FLOOR_INV`.
- **F16 CTC gradients** (2026-10-12, `ops/ctc.rs`, gate `step162`, `1791818535`): native `ctc_loss` and its
  backward, widened to F32 and rounded once.
- **Fuser scalar guard** (2026-10-12, gate `step161_fusion_scalar_guard`, `1791772236`): a fuser took every scalar id
  for zero, so `thresholded_relu` ran as `relu`; now checked at execution.
- **B15, loading and checkpoints** (2026-10-12, [`crates/tt-models`](../../crates/tt-models/README.md)): strict
  safetensors loading via `burn-store` (every mismatch an error); weights pinned in `PINS.toml`, fetched by
  `cargo xtask fetch-weights`. Gate `step170_checkpoint_round_trip` (`1791847160`): train, save, load, continue, bit
  for bit an uninterrupted run.
- **Copy-shaped ops on the card** (2026-10-14, [decode-host-cost.md](decode-host-cost.md)): native `cat` and
  `repeat_dim` (gate `step180_native_cat`), whole-row `slice_assign` as a tile copy (`step181_region_assign`),
  positive-step slices and `swap_dims` as views (`step182_region_views`).
- **Gather without staging** (2026-10-13, gate `step171_select_rows`, `1791851883`): B's `SELECT_ROWS` record reads the
  rows a lookup returns, no host address table. One token from SmolLM2's 49,152 x 576 table **15.9 s -> 260 us**
  (`1791851940`); the F32 135M forward 575 -> 33 s (`1791851959`)
  ([log](../learnings/performance-log.md#row-select-an-embedding-lookup-without-staging-2026-10-13-card-0)).
- **Chunk-free convolution and pooling** (2026-10-14, gates `step178_select_taps`, `step179_nhwc_conv_pool`): a 2-D
  convolution is one tap select and one product over NHWC; max pooling one select and a fold. ResNet-18 forward
  16,288 -> 782.3 ms on one tile, **131.1 ms on every tile against burn-flex's 151.4** (`1791996580`, `1791996716`;
  [log](../learnings/performance-log.md#chunk-free-convolution-and-max-pooling-2026-10-14-cards-0-and-1)).
- **Attention in few jobs** (2026-10-14/15, [attention-jobs.md](attention-jobs.md), gates
  `step183_attention_jobs`, `step184_ragged_batched_matmul`): ragged batched matmuls as one job; jobs a call decode
  64 -> 10, ViT 28 -> 10, prefill 67 -> 11. ViT-Tiny every tile 283.4 -> 129.1 ms (burn-flex 109.3, `1792026124`;
  [log](../learnings/performance-log.md#attention-products-in-one-job-2026-10-14-cards-0-and-1)).

## Models on silicon

All run natively (`assert_native_model`: no staged compute, host arithmetic or downloaded operand) and agree with
burn-flex on the same weights.

- **Model ladder, rungs 1 to 3** (2026-10-12, card 0): ResNet-18 (`step165_resnet18`, `1791847160`, same top-5 as
  Flex), ViT-Tiny/16 (`step166_vit_tiny`, same top-5), SmolLM2-135M (`step167_smollm2`, `1791844568`, F32, 30
  layers: the same 16 greedy tokens as Flex; the host model reproduces HuggingFace `transformers` to 5e-5). The
  checkpoint's own BF16 also agrees (6 layers, `1791848957`).
- **Mixed precision in one model** (2026-10-12, [`quant.rs`](../../crates/tt-models/src/quant.rs), gate
  `step168_smollm2_mixed_precision`, `1791845324`): a format per matrix and activation boundary; BFP8/4/2, BF16, F16
  and F32 in one model, each plan against an independent host model.
- **SmolLM2 in `SMOKE`** (2026-10-13): `step167` F32 (33 s) and `step168` (`1791852139`).
- **SmolLM2-1.7B-Instruct, the first model past 1B** (2026-10-15, gate `step188_smollm2_1_7b`, `1792080640`): no
  model code changes (preset `SMOLLM2_1_7B_INSTRUCT`); a pinned `transformers` reference
  (`reference/llama_reference.py`); the same 16 greedy tokens as Flex, traced bit for bit eager. First measured
  (card 1, `1792082994`): decode 13.6 tokens/s against burn-flex's 3.4, prefill 267-771 tokens/s against 7.6-24.7
  ([log](../learnings/performance-log.md#smollm2-17b-instruct-on-the-card-2026-10-15-card-1)).
- **Three architectures past Llama** (2026-10-30, [new-architectures.md](../plans/new-architectures.md)): Qwen3.5-4B
  (`step260_qwen35`, F32 and BF16-stored weights), Nemotron-3.5-Bolt / Gemma 4 E4B (`step261_gemma4`) and
  MOSS-Image (`step262_moss_image`) reproduce their references on the host and agree with burn-flex on card 0.
- **Qwen3.5-4B decodes traced** (2026-10-30, `qwen35::HybridCache`, gate `step263_qwen35_traced`, card 1
  `1793345534`): 64.2 / 64.6 ms a step at 32 / 256 tokens with BF16-stored weights (`1793346925`), against ~1.6 s
  eager.
- **Qwen3.5-4B's step attributed on the card** (2026-11-08, card 1,
  [log](../learnings/performance-log.md#qwen35-4bs-step-on-the-card-a-third-of-it-is-the-greedy-pick-the-even-deals-04-is-the-head-2026-11-08-cards-0-and-1)):
  a replay is 64.34 ms on the card against 62.52 host-timed; the 21.5 ms in no span is `generate::greedy`'s eight
  31,040-column argmax parts on one unit each.

## Decode performance

Detail: [decode-to-bandwidth.md](decode-to-bandwidth.md) (the umbrella plan),
[decode-step-bandwidth.md](decode-step-bandwidth.md), [decode-matmul-bandwidth.md](decode-matmul-bandwidth.md),
[decode-small-ops.md](decode-small-ops.md), [decode-attention.md](decode-attention.md),
[in-place-cache-write.md](in-place-cache-write.md), [planned-l1-residency.md](planned-l1-residency.md),
[bf16-weight-storage.md](bf16-weight-storage.md), [decode-bytes-in-flight.md](../plans/decode-bytes-in-flight.md).

### Chronology: the SmolLM2-1.7B traced decode step

Batch 1, at a 32-token prompt unless noted, bit for bit the step before. F32 weights until 2026-10-26, BF16 since.

| Date | Change | ms a step | Run ID |
|---|---|--:|---|
| 2026-10-15 | First traced generation (13.6 tokens/s) | ~74 (74-79) | `1792082994` |
| 2026-10-16 | Cache write as one row scatter at resident indices (`step189`) | 73.65 -> 57.21 | `1792109477` |
| 2026-10-16 | `m = 1` products read only used rows; fused `q\|k\|v`, `gate\|up` (`step190`, `step191`) | 47.90 | `1792161623` |
| 2026-10-16 | Element-wise ops in place; RoPE in the projection's layout (S1, S2a, `step192`) | 45.23 | `1792174499` |
| 2026-10-16 | Three burn-fusion fusers: sigmoid, `silu_mul`, RMSNorm statistic | 43.42 | `1792186180` |
| 2026-10-16 | RMSNorm statistic over only occupied row groups (`step196`) | 42.97 | `1792189445` |
| 2026-10-17 | `A` read once a unit (`GATHER_A_RESIDENT`, B1, `step198`) | 41.83 | `1792209913` |
| 2026-10-17 | In-place cache write, no whole-cache copies (D1-D4, `step199`) | 38.97 | `1792263398` |
| 2026-10-18 | Position-major cache, streamed `K`, packed `A`, next weights prefetched | 33.02 -> 26.47 | `1792364210` |
| 2026-10-19 | Small ops: SFPU column broadcast, row gathers by rows, packed twins | 24.04 | `1792383988` |
| 2026-10-19 | Decode attention as one op (`Session::attention_rows`, `step209`) | 22.78 | `1792425331` |
| 2026-10-19 | One region a job; RMSNorm statistic and `DIV` as one op (`step210`) | 22.24 | `1792432143` |
| 2026-10-20 | Token on the card, RoPE as one op, three copies gone (`step211`-`step213`) | 20.95 | `1792444936` |
| 2026-10-21 | L1 residency on by default (planned capture, `step217`-`step220`) | 20.07 | `1792556241` |
| 2026-10-21 | Products scatter into readers' L1 (residency Phase 3) | 19.50 | `1792595329` |
| 2026-10-21 | Busy prefetch on by default (only B drains the prefetch transaction ID) | 18.39 | `1792615740` |
| 2026-10-22 | Weight stream at 2 reads in flight (`GATHER_STREAM`) | 17.27 | `1792637346` |
| 2026-10-22 | `down` prefetched by 40 idle units | 16.87 | `1792679220` |
| 2026-10-22 | Prefetch anchored at a unit's last product | 16.79 | `1792695341` |
| 2026-10-26 | BF16 weight storage (`step232`-`step234`) | 12.95 | `1792990193` |
| 2026-10-26 | BF16 stream on 96 units, 3 in flight (A5a) | 11.76 | `1793002379` |
| 2026-10-26 | Compact 2,112-byte BF16 slots (`GATHER_B_BF16`) | 11.68 | profile `1793023409` |
| 2026-10-26 | Barriers the data does not need pruned (S5, `step235`) | 11.59 | (card 0) |
| 2026-10-27 | S5 merged with prefill P4 | 11.32 | (log) |
| 2026-10-27 | RMSNorm's scale inside the norm's op | 11.18 | `1793114653` |
| 2026-10-29 | F4a.2: a product's units read the next weights before their arrival | 10.95 | (log) |
| 2026-10-30 | F5 `Dst` subblocks merged | 10.93 | `1793396168` |
| 2026-11-03 | Interleaved baseline on the day | 10.96 | `1793723839` |
| 2026-11-04 | C1: weights laid by column, one copy, 16 KiB reads | 10.06 | `1793808030` |
| 2026-11-07 | The even deal: `gate\|up`'s columns over its prefetching units too | **9.90** | `1794024329`-`1794025917` |

Links for the rows: [cache write](../learnings/performance-log.md#a-cache-write-of-one-row-a-head-2026-10-16-cards-0-and-1),
[matmul rows](../learnings/performance-log.md#decode-matmuls-read-only-the-activation-rows-they-use-2026-10-16-cards-0-and-1),
[fusers](../learnings/performance-log.md#decode-fusers-sigmoid-silu-and-rmsnorm-2026-10-16-cards-0-and-1),
[larger changes](decode-to-bandwidth.md#the-larger-changes-compared-2026-10-18),
[bandwidth 2026-10-19](../learnings/performance-log.md#where-decodes-bandwidth-stands-and-a-column-broadcast-in-the-sfpu-2026-10-19-cards-0-and-1),
[BF16 slots](../learnings/performance-log.md#compact-bf16-slots-and-where-a-bf16-decode-step-goes-2026-10-26-cards-0-and-1),
[barriers](../learnings/performance-log.md#which-barriers-a-decode-step-needs-2026-10-26-card-0),
[merged](../learnings/performance-log.md#the-decode-and-prefill-streams-merged-2026-10-27-card-0),
[F4a.2](../learnings/performance-log.md#f4a2-a-products-units-read-the-next-weights-before-their-arrival-2026-10-29-cards-0-and-1),
[C1](../plans/decode-bytes-in-flight.md#c1-final-the-rules-on-the-code-as-built-2026-11-04-card-1),
[even deal](../plans/decode-bytes-in-flight.md#the-even-deal-the-default-2026-11-07-cards-0-and-1).
The header of the roadmap also cites `1794024534`-`1794025528` for the 9.90 ms. In tokens/s: 13.6 (2026-10-15), 88
(2026-10-27), ~101 (2026-11-07), against burn-flex's 3.4. X9's two-copy layout reached 9.75-9.90 ms on 2026-11-03
(`1793663823`); it did not scale and was retired on 2026-11-06, when C1 and the even deal took its gain in one copy.

### What each change built

- **Profile-mode byte counts** (B0, gate `step197`): every captured op's GDDR bytes and entries. A 1.7B step then
  moved 9.6 GB against 42.3 ms.
- **Decode attention as one op** (Phase 6a, gate `step209`): a head a unit, seven kernels in L1. 135M 9.58 ms
  (`1792425715`).
- **Planned L1 residency** (2026-10-20/21, [planned-l1-residency.md](planned-l1-residency.md), gates
  `step217`-`step220`, `silicon_probe_region_cliff`): a static plan over each captured graph; a 1.7B step keeps
  ~400 of 548 tensors in L1, sparing 294 MB of reads and 123 MB of writes a step. On by default (`TT_RESIDENCY=0`).
- **Busy prefetch, stream caps and anchors** (2026-10-21/22): idle units read the next product's `B`; `dm::TXN_FETCH`
  for a replay's next reader section; `record::GATHER_STREAM` with `dm::STREAM_CAP`; `Session::set_prefetch_anchor`.
  The step's weights 389 -> 426 GB/s.
- **Mailboxes packed** (M2, 2026-10-22): 40 KiB above the program slots, freeing 88 KiB of capture-only L1; smoke
  1,188 of 1,188 (`1792711476`).
- **BF16 weight storage** (2026-10-26, [bf16-weight-storage.md](bf16-weight-storage.md)): every weight but the
  embedding stored BF16, read as BF16, bit for bit the F32 step (the checkpoint is BF16); `Llama::with_bf16_weights`,
  the benches' default. 135M 7.30-8.55 ms. Decided by the owner once the F32 levers were spent (F32's floor ~14.3 ms).
- **`fast-reads` on decode** (2026-10-26, card 1): 11.74 -> 11.54 ms, all of it the products' read entries.
- **Barrier pruning** (S5, `tt_kernels::barriers`, `Session::set_barrier_prune`, gate `step235`): 194 of the 135M
  step's 591 barriers, 82 of the 1.7B's 403. 135M 7.35 -> 6.97 ms.
- **Fixed on the way:** a ragged RMS statistic past a program slot hung the card (`step196`); decode attention's job
  recorded too low an L1 top, so busy prefetch overwrote `K`/`V` at `Step(512)` (`step209`).
- **The card's reads metered** (A4, profile `1793141447`,
  [log](../learnings/performance-log.md#the-cards-gddr-reads-over-a-decode-step-2026-10-27-card-0)): every unit's
  GDDR reads summed into each op's span in eighths; per-channel counts (`channels.tsv`).
- **A decode product's pipeline metered from inside** (follow-up 2, 2026-11-02, [decode-product-meter.md](decode-product-meter.md),
  [log](../learnings/performance-log.md#a-decode-products-pipeline-metered-from-inside-read-bound-by-its-reads-in-flight-2026-11-02-cards-0-and-1)):
  `Detail::Pipeline`, B's L1 batch log (`dm::BATCH_LOG`), gate `step266_batch_tables`. Verdict (`1793637811`,
  `1793638678`): read-bound by each unit's bytes in flight; T0 starves 52-86% of the slowest unit's work.
- **More bytes in flight** (follow-up 2b, [decode-bytes-in-flight.md](../plans/decode-bytes-in-flight.md),
  [log](../learnings/performance-log.md#h11s-then-step-contiguous-weights-in-16-kib-reads-2026-11-02-card-1)):
  Phase A bounded it (16 KiB entries make `down` 1.34-1.43x faster). Phase C1 (2026-11-04) is the scalable form:
  column `j`'s tiles contiguous in channel `j % 8`, decode reading 16 KiB pieces, blocks a tile at a time, one copy,
  the default (`TT_WEIGHT_COLUMNS=0` off). Qwen3.5-4B 2.1% faster; prefill at or above every bracket.
- **The even deal** (X10, 2026-11-07, `TT_EVEN_DEAL=0` off): 10.14 / 11.18 -> 9.90 / 10.95 ms at 32 / 256 tokens;
  card 0 `--smoke` 1,274 of 1,274. Its 0.4% on Qwen3.5-4B located to the tied head (`1794101823`, `1794103109`).
- **135M** over the same period: 24.40 ms (2026-10-16) -> 6.92 / 7.47 / 8.14 ms at 32 / 128 / 256 tokens
  (2026-10-27), against burn-flex's 26.5 / 29.1 / 30.5.

- **W1 bounded: the performance contract's products measured before any plumbing** (2026-11-09, card 1,
  `silicon_probe_performance_contract` `1794257871`/`1794258133`, the 1.7B per weight format `1794260351`,
  `1794261034`, `1794261722`; [log](../learnings/performance-log.md#w1-bounded-the-performance-contract-on-this-stack-before-any-plumbing-2026-11-09-cards-0-and-1)). BFP8 weights alone take the 1.7B's decode step 9.83 -> **7.51 ms** (BFP4 6.34),
  F32 activations at HiFi4, against tt-metal's 10.55 on the same card: W1's 8.0 ms kill does not fire. Fidelity
  moves no product at `m` = 1. A BF16 `A` is slower today (decode products 1.4x, prefill at HiFi4 1.4-2.0x): a
  packed `A` takes the faces pair order and misses the one-row packing and weight prefetch. H13 killed; packer L1
  accumulation's ceiling, up to 24% of a product, shows only under a packed `A`.

- **W1, first box: BFP8 weights the inference default, behind the contract switch and a model gate** (2026-11-10;
  gates card 0, benches card 1; [log](../learnings/performance-log.md#w1-bfp8-weights-the-inference-default-behind-the-contract-switch-and-a-model-gate-2026-11-10-cards-0-and-1)).
  - **The contract switch:** `burn_tt::Numerics`.
    - `TtConfig::numerics` / `TT_NUMERICS` set it; the default is `Reference`. `with_numerics` scopes it.
    - A trace records its contract; the residency-plan key includes it; a capture refuses a switch part way (`step40`).
  - **The models ask for it:**
    - `Llama::for_inference` / `Qwen35::for_inference` store the weights in the contract's formats
      (`quant::Plan::for_numerics`).
    - `TracedGenerator` captures under the model's contract.
  - **The model gate, `step269_performance_contract`:** the 1.7B traced, teacher-forced over eight prompts (232
    positions) against HuggingFace (`llama_reference.py --prompt-set`), PCC against burn-flex (held to HuggingFace
    within 1.07e-4).
    - **BFP8: top-1 98.28%, top-5 100%, PCC 0.99968**, against the reference contract's 100% / 100% / 0.999998
      (`1794287910`).
    - The BFP4 `gate|up` option scores 86.64% / 0.93, below tt-metal's floors on this model; it is recorded, not gated.
  - **The 1.7B's decode step under the performance contract: 7.52 ms** (133 tokens/s; reference 9.81; `1794291915`,
    `1794292593`). That is 0.71x tt-metal's 10.55 ms in its performance mode on the same card.
  - **Qwen3.5-4B under the performance contract: 56.93 ms** (reference 62.49 the same day; `1794293972`,
    `1794295094`). Its tied head stays the F32 table; its quality gate is still open.
  - **A fused, stored Llama held each fused matrix's parts a second time.** SmolLM2-135M BF16: 543 MB against 399 MB
    unfused (`1794286677`). `fuse` now drops the parts: 399.0 MB (`1794286904`, gate
    `step240::a_fused_stored_model_holds_each_weight_once`).

- **W1: a packed tile whole in `SrcB`** (2026-11-10, card 0;
  [log](../learnings/performance-log.md#w1-a-packed-tile-whole-in-srcb-a-bf16-a-past-the-fp32-as-rate-2026-11-10-card-0)).
  - A BF16 (as a weight or an activation), BFP8, BFP4 or BFP2 `A` unpacked whole into `SrcB`, its first face row alone
    where only that is live: bit for bit the faces order on ttsim and card 0 (`probe_pair_src_rows`,
    `step241::whole_tile_orders_give_every_packed_a_its_bits` with its control `probe_half_whole_srcb`;
    `1794328961`-`1794328978`). `TileFormat::whole_tile_unpack` licenses it; `PairOrder::Auto` holds both tiles whole
    for every pair of edge formats. No default path changed (FP32-`A` programs word for word the same).
  - A BF16 `A`'s prefill products at HiFi4 22-30 -> **29-70 TFLOPS**, 1.13-1.32x the FP32 `A` (it was 0.51-0.96x);
    fidelity no longer moves them, `A`'s gathers bind instead. H13's chunk 1.27x the reference (1.34x with BFP8
    weights), so its kill no longer fires; packer L1 accumulation's ceiling falls to at most 2.5% of a product.

- **W1: the one-row path for a BF16 `A`** (2026-11-10, gates card 0, the 1.7B card 1;
  [log](../learnings/performance-log.md#w1-the-one-row-path-for-a-bf16-a-its-decode-products-at-the-fp32-as-step-2026-11-10-cards-0-and-1)).
  - A one-row BF16 `A` is packed `[K / 32, 32]` in its own format (`tensor::pack_row`), its tiles in FP32-sized slots,
    and read from 32-byte face rows (`matmul::packed_a_pairs`, `SrcRoute::packed_row_bytes`). So it takes the weight
    prefetch, the busy prefetch and the streamed `K` as an FP32 row does. Bit for bit the row read whole and an FP32
    `A` of its values, on ttsim and card 0: the isolated probe `probe_pair_src_rows::srcb_packed_bf16_row`
    (`1794347716`), then `step204`, `step205` and `step216` (`1794348020`). The control,
    `matmul::probe_fp32_row_strides`, changes every column.
  - Decode products a 1.7B step at `m` = 1: BF16 x BFP8 HiFi2 10.74 -> **6.99 ms** (FP32 `A` 7.00), BF16 x BF16 14.33 ->
    10.34 (10.19); `down` 177 -> 71 us (`1794332034` -> `1794348097`). A BF16 `A` is no slower than an FP32 one at
    any `m`; no model gains until BF16 activations (native BF16 arithmetic).
  - Nothing else moved: the 1.7B 7.52 / 9.82 ms a step under the two contracts (`1794349089`, `1794349769`), the
    pinned jobs grown by BF16 cases only. Found on the way: `Session::download_bf16` refused a BF16 tensor in
    FP32-sized slots; it reads each slot at its own length.

- **W1: BF16 tensors in their own slots** (2026-11-11, gates card 0, the 1.7B card 1;
  [log](../learnings/performance-log.md#w1-bf16-tensors-in-their-own-slots-named-in-the-tensor-reference-2026-11-11-cards-0-and-1)).
  A tensor reference names its slots (`TensorRef::slot`): `READ_RUN`, `WRITE_RUN` and `SCATTER` step and size GDDR by
  them, every other record refuses a BF16 slot by name, and `alloc_elem` gives a BF16 tensor 2,112-byte slots (the
  new-architectures defect "a BF16 tensor takes an F32 slot", for `alloc_elem` and every op allocating through it;
  Nemotron's fit not re-run). Gate `step270_native_bf16` with its control `tensor::probe_unnamed_slot`; the 1.7B within
  its run-to-run band under both contracts.

- **W1: element-wise ops on BF16 operands, with BF16 results** (2026-11-11, ttsim and card 0, `1794407220`). The SFPU
  kernel reads each BF16 operand as the expand widens it (`SrcA`, eight `MOVA2D` chunks into its `Dst` rows), runs the
  F32 op's program, and rounds the result with the compress's own quantize before a BF16 pack
  (`sfpu::kernel::Formats`): bit for bit `bf16_from_f32(op(bf16_to_f32(x)))` over 15 kinds -- the 1.7B's `silu`,
  `silu * up` and `add`, unary, scalar, binary, column and block broadcasts, ternary, a comparison, and a decode row --
  and every BF16 class, with one op instead of three or four and no F32 tensor between them. The looped roles stay one
  tile long. Measured on the way: the pack from 32-bit `Dst` to BF16 truncates (silicon-operating-notes), which is why
  the quantize stays. A row broadcast and kinds taking `Bool` or `I32` still widen.

- **W1: products with a BF16 result, and a BF16 packed twin** (2026-11-11, ttsim and card 0 `1794408773`, cost card 1
  `1794408399`; [log](../learnings/performance-log.md#w1-products-with-a-bf16-result-and-a-bf16-packed-twin-2026-11-11-cards-0-and-1)).
  `Session::matmul_dram_as(.., Elem::BF16)` rounds each output in `Dst` once its `K` chain is done and packs it as
  BF16: bit for bit `bf16_from_f32` of the F32 product over every schedule (blocks, the one-row packed path, streamed
  `K`, the block stream's three kernels) and FP32, BF16 and BFP8 operands; `K` blocks refused by name. A one-row BF16
  result keeps a BF16 twin that products read. The quantize costs 6-9% of a short-`K` prefill product, ~0 of `down`.

- **W1: burn-tt routes BF16 natively** (2026-11-11, card 0). Twenty-one Burn ops that are one element-wise op on the card
  (`add`, `sub`, `mul`, `div` and their scalar forms, `exp`, `log`, `neg`, `abs`, `sqrt`, `recip`, `tanh`, `erf`,
  `sin`, `cos`, `log1p`, `relu`, `sigmoid`) take BF16 operands as they lie (`gen_burn.rs`'s `NATIVE_BF16`): one job each, no cast,
  bit for bit `op(x.cast(F32)).cast(BF16)` (`step76_burn_bf16::bf16_ops_run_natively_bit_for_bit_their_widened_composition`);
  forms the kernel has no BF16 form of (a row broadcast, kinds taking `Bool`/`I32`) widen and narrow. A BF16 product's
  result is BF16 from the card (`Engine::matmul_dram_bf16`), and a logical BF16 weight may be stored BFP
  (`with_storage`), which a BF16 activation multiplies with a BF16 result. Conformance unchanged (ttsim 3,025, and the
  BF16 suites on card 0); `step12_mnist` 8/8; tt-mnist BF16 5.32 -> 5.04 ms a traced step (burn-flex 3.49, F32 1.36).
  Found on the way: 17 BF16 conv entries of the silicon expected-failures list pass, and already did at `9f873b8`;
  they left the list.

- **W1: the casts over runs of tiles; RMSNorm on BF16 in one statistic** (2026-11-11, ttsim and card 0).
  `Session::bf16_to_f32` and `bf16_from_f32` run the SFPU kernel over runs of tiles (`tile_op::cast`: the expand's
  widening, a raw copy, the compress's quantize), a job a run where they took a job a tile, bit for bit
  (`step270::casts_over_runs_are_the_casts_a_tile`, every pattern and class). `burn_tt::rms_norm` takes a BF16 `x` and
  scale: the F32 statistic in one op, cast to BF16, then `x / rms` and `* gamma` natively -- Burn's module bit for bit,
  5 jobs against its 9 at a decode row, 8 against 12 at a block (`step76::bf16_rms_norm_is_the_module_bit_for_bit`).
  `sigmoid` joins the native list, and an element-wise op over blocks reads BF16 buffers.

- **W1: the 1.7B on a BF16 stream, an option** (2026-11-11, cards 0 and 1;
  [log](../learnings/performance-log.md#w1-burn-tt-on-bf16-and-the-17b-on-a-bf16-stream-an-option-2026-11-11-cards-0-and-1)).
  `WeightOptions::bf16_stream` (`TT_BF16_STREAM=1` in the bench): BF16 activations between ops, the body's weights
  logical BF16 stored BFP8, the attention block and the head F32. Decode 9.20 / 9.73 / 10.36 ms, prefill 704 / 2,137 /
  2,936 tokens/s, behind the FP32 stream's 7.51 ms and 4,004 tokens/s (from 12.60 ms and 346 tokens/s at its first
  run, the casts of views removed); `step269` top-1 97.84%, one position under its threshold. The default stays the
  FP32 stream; the profile names what the option lacks (the norm, `silu * up` and the attention block on BF16).

## Prefill and long context

Detail: [prefill-operand-reuse.md](../plans/prefill-operand-reuse.md), [long-context.md](../plans/long-context.md).

- **Prefill traced in chunks** (2026-10-15, `Llama::step_chunk`, gate `step185`): 256 tokens in 79 ms against
  burn-flex's 1,040; then 73.1 ms (3,500 tokens/s)
  ([log](../learnings/performance-log.md#smollm2-prefill-chunk-size-2026-10-15-cards-0-and-1)).
- **The 1.7B at its native 8,192 positions** (2026-10-23, card 0): builds and decodes (17.0 / 26.0 ms a step after
  32 / 1,000 tokens, `1792793709`); a trace's programs paged into the program cache, so only one trace need fit; two
  GDDR leaks fixed. A 7,992-token prompt gives burn-flex's tokens in 38.7 s against 1,847 s (`1792800061`).
- **32,768 positions** (synthetic GQA models, traced equal to eager, `1792799289`).
- **A chunk attends only to its prefix; the GQA mask read as a block** (D1-D3, 2026-10-24): 1,000 / 7,992 tokens
  4.69 / 36.63 -> 1.24 / 18.33 s (`1792808294`, `1792809505`).
- **A capture's intermediates give their GDDR to later ops** (1.G, `TT_CHUNK_REUSE=0` off): 2.10 -> 17.78 GB free
  after the 1.7B's build; chunks of 512 capture at 8,192 positions.
- **Blocked prefill products** (Step 1, 2026-10-24, on by default, gate `step229`): a job owns an `mb x nb` output
  block and streams `K`; 2.9-3.6x at `m` >= 512 (`1792857503`, `1792857489`); 7,992 tokens 18.33 -> 12.59 s
  (`1792858000`).
- **Replayable `SUB_EXP` and `DIV`** (2026-10-25, `sfpu::ops::staged`): row bodies split into replayable loops; 7,992
  tokens 15.24 -> 12.28 s at chunks of 64, 12.58 -> 10.00 s at 512 (`1792939867`, `1792940254`;
  [log](../learnings/performance-log.md#the-long-context-softmax-sub_exp-and-div-push-bound-their-row-loops-staged-to-replay-2026-10-25-card-0)).
- **Fused `.ttinsn` push** on the same prefill: 10.00 -> 9.84 s (`1792962530`).
- **The softmax's data passes** (1.H, 2026-10-26): column broadcasts read their `B` tile once; each statistic in the
  pass that makes its input. 9.84 -> 8.42 s (`1792980190`, `1792980549`;
  [log](../learnings/performance-log.md#the-long-prompt-softmax-a-column-read-once-a-run-the-statistics-in-the-passes-that-make-their-input-2026-10-26-card-0)).
- **Attention's batched products as 2-D blocks** (P6, 2026-10-26): 8.42 -> 6.62 s (`1792992817`).
- **The K-tile pair's cost** (P4, 2026-10-26, card 1): each operand's tile whole in its `Src`, two unpacks a pair
  instead of twelve, checked by `matmul::order`; kernel 1.5-1.9x; BF16 prefill of 7,992 tokens 6,386 -> 5,984 ms
  (`1793060076`; [log](../learnings/performance-log.md#a-k-tile-pairs-order-each-tile-whole-in-its-src-2026-10-26-card-1)).
- **Measured blocks and pulled operands** (F4b.2, below): the 1.7B prefill 1.04-1.34x. **Short chunks** (H14,
  below): 1.11x at 64 / 256 tokens. **Program restores on the card** (1a, below): 128 / 256 tokens 2,546 / 2,744 ->
  3,040 / 3,827 tokens/s.
- As of 2026-11-07 the 1.7B prefills 7,992 tokens in 5.98 s.

- **No 1.7B prefill regression** (2026-11-09, `1794260351`, [log](../learnings/performance-log.md#w1-bounded-the-performance-contract-on-this-stack-before-any-plumbing-2026-11-09-cards-0-and-1)): `traced_generation` reads 1,098 / 3,084 /
  3,890 tokens/s at 32 / 128 / 256 tokens, against H15's 3,827 at 256. The ~2,090 quoted since 2026-11-03 were
  `traced_generation_stream_ceiling`'s rows (prefill in chunks of 64), which run under the same filter. Model
  benchmarks now attach every tile unless `TT_TILES` says otherwise (the gate tile alone overflowed the 1.7B's
  program cache).

## Traced execution and replay

Detail: [traced-execution.md](../plans/traced-execution.md), [replay-fixed-cost.md](replay-fixed-cost.md),
[decode-length-cost.md](decode-length-cost.md).

- **ResNet-18 traced** (2026-10-14, `resnet18_forward_traced`, `1792000003`): 17.6 ms a replay against 205.7 ms
  untraced and burn-flex's 151.7, bit for bit eager.
- **ViT-Tiny traced** (2026-10-15, `1792026627`): 12.6 ms against burn-flex's 107.6
  ([log](../learnings/performance-log.md#vit-tiny-traced-2026-10-15-card-1)). After the fused push: 8.32 ms.
- **SmolLM2 decode step traced** (2026-10-15, `Llama::step_fixed`, gate `step185_fixed_decode`): 37.1 ms
  (`1792029144`), then **26.46 ms against burn-flex's 26.42** (`1792036613`) after profiling the ~33 ms fixed cost:
  padding refilled only along `K`, the tied head transposed once, the mask expanded once a step, thin downloads
  untilized on the card ([log](../learnings/performance-log.md#smollm2-decode-step-traced-2026-10-15-cards-0-and-1)).
- **A generic prefill-and-decode driver** (2026-10-15, `tt_models::generate::TracedGenerator` over any
  `FixedShapeDecoder`, `burn_tt::SharedStateTraces`, `FixedCache`, `Rope`; gate `step187_traced_generation`,
  `1792071747`): bit for bit against eager over SmolLM2, GQA, multi-head, untied heads and a GPT-2-style decoder.
  The token lookup replays inside the trace
  ([log](../learnings/performance-log.md#traced-generation-end-to-end-2026-10-15-cards-0-and-1)).
- **A step costs the context, not the cache** (2026-10-15): GQA without `repeat_kv` (5 ms a step) and step traces per
  attended length. A whole generation 0.60x-0.90x burn-flex's time (`1792078489`).
- **Trace size in GDDR** (2026-10-15, A0-A1, gate `step174`): SmolLM2's traces 3.75 GB -> 563 MB, each unit's stream
  and sections one region; the driver's build 44.9 -> 18.8 s.
- **Traced, all three models beat burn-flex** (2026-10-15): SmolLM2 decode 26.46 ms against 26.42, ViT-Tiny 11.6
  against 112, ResNet-18 17.2 against 164.
- **Program restores on the card** (follow-up 1a, H15, 2026-11-01, `Session::restore_programs`,
  `dm::check_program_restore`; `TT_MOVER_RESTORE=0` off): a replay's B mover copies evicted programs back from GDDR
  ahead of its `CALL`; the host writes 0 bytes; its restore bookkeeping 611 -> 43 us a trace (`1793549551`;
  [log](../learnings/performance-log.md#program-restores-on-the-card-paged-programs-copied-back-by-the-b-movers-2026-11-01-cards-0-and-1)).

## Dataflow compiler (F1-F6)

Plan: [dataflow-compiler.md](dataflow-compiler.md). Design reference and ADRs 0003-0005:
[dataflow-compiler.md](../learnings/dataflow-compiler.md). Every stage was a hypothesis with a kill criterion.

- **Stage 0** (2026-10-28): card-1 baselines.
- **F1, one matmul for every format pair** (H3, 2026-10-28,
  [log](../learnings/performance-log.md#one-matmul-for-every-format-pair-and-bfp-weights-through-the-model-2026-10-28-cards-0-and-1)):
  `TileFormat`; one builder for any edge-format pair, three old builders deleted; weights stored in any packed format
  (`Llama::with_stored_weights`), bit for bit; BFP weights at prefill shapes reach BF16's rate (H7).
- **F2, typed channels and the cross-core credit** (2026-10-28, gate `step242`,
  [log](../learnings/performance-log.md#a-cross-core-credit-against-the-barrier-h8-2026-10-28-card-1)):
  `op::CREDIT`/`op::WAIT_CREDIT` over remote, broadcast and GDDR rings; `channel::Graph` refuses unbalanced rates and
  deadlocks. H8: a credited handoff 1.81 us against a two-unit barrier's 2.6.
- **F3, the tile-generic op** (H6 held, 2026-10-28,
  [log](../learnings/performance-log.md#f3-the-tile-generic-op-pointwise-and-reduction-ported-before-and-after-2026-10-28-cards-0-and-1)):
  pointwise, reduction and every matmul form ported word for word through one lowering; allocation
  (`tile_op::allocate`) and channel checks as passes; the job and L1 goldens found two defects. `--smoke` 1,241 of
  1,241 (`1793221000`).
- **F4b, operand reuse across cores** (H1, 2026-10-29): `A` pulled from a few holders, 1.33-1.44x at prefill shapes
  ([log](../learnings/performance-log.md#d5-decided-a-pulled-from-a-few-holders-a-packed-bs-block-is-the-next-lever-2026-10-29-card-1)).
  F4b.2: the block and the pull measured on the card at a capture's first plan, the default (H9): 1.02-2.25x per
  product over 50 cases; H1 through Burn 19.2 -> 29.0 TFLOPS; 1.7B prefill 1.04-1.34x
  ([log](../learnings/performance-log.md#f4b2-a-block-measured-on-the-card-a-pulled-where-it-gains-2026-10-29-card-1)).
  `--smoke` 1,264 of 1,264 (`1793306388`). Leftovers bounded and located 2026-10-31
  ([log](../learnings/performance-log.md#f4bs-leftovers-the-batched-pull-bounded-the-eager-slowdown-located-2026-10-31-card-0)).
- **F4a, channels in decode** (H4, closed 2026-10-30): the kill fired for credits, but its first lever is the default:
  a product's units read the next weights before their arrival (F4a.2), 1.7B 11.32 -> 10.95 ms. Per-edge credits
  (`barriers::credit`, gate `step265`) built, opt-in; their build check made linear (a Qwen3.5 build had passed 40
  minutes).
- **F5, `Dst` subblocks and half-sync** (H2, 2026-10-30, gate `step246`, `TT_DST_SUBBLOCKS=0` off,
  [log](../learnings/performance-log.md#f5-dst-subblocks-and-half-sync-2026-10-30-card-0)): up to four FP32 tiles a
  `Dst` half, math and pack alternating; H1 x five formats 1.12x (FP32 32.3 -> 34.4 TFLOPS); MOSS-Image DiT `qkv`
  15.0 -> 27.3. `--smoke` 1,265 of 1,265 (`1793397733`).
- **F6, register fusion and spatial pipelining bounded** (H5, H10, 2026-10-31,
  [log](../learnings/performance-log.md#f60-register-fusion-and-spatial-pipelining-bounded-on-the-card-2026-10-31-card-1)):
  both bounded on the card before any build and both killed, saving the build. The plan is complete.
- **Follow-up 1, short chunks** (H14, 2026-10-31,
  [log](../learnings/performance-log.md#h14-short-chunks-products-and-the-profile-modes-measured-blocks-2026-10-31-cards-0-and-1)):
  `tensor::SHORT_MEASURED_BLOCKS` for products of at most eight tile rows; 64-row products 1.19x, prefill 1.11x
  (`1793477637`). Found and fixed: the profile mode skewed measured blocks (`Session::unprofiled`).
- **Follow-up 1a, program restores on the card** (H15, 2026-11-01): see "Traced execution".
- **Follow-up 2, a decode product metered from inside** (2026-11-02): see "Decode performance".
- **Follow-up 3, a replay's wall on the card** (2026-11-08): `replays.tsv`, unit 0's `LIST_BEGIN`/`LIST_END`; see
  "Models on silicon".
- **Follow-up 5, copy-free levers** (decided 2026-11-07): the even deal made the default.

## Fusion and `.ttinsn`

Detail: [kernel-fusion.md](../plans/kernel-fusion.md), [ttinsn-push.md](ttinsn-push.md),
[ttinsn-push-experiment.md](ttinsn-push-experiment.md).

- **Decided: burn-fusion fusers are the end state** (2026-10-16). `Fusion<TtBackend>` runs Burn's suites (B13a,
  2026-10-12: 1,803 pass).
- **Three fusers** (2026-10-16, `burn-tt` `fusion/chain.rs`, gates `step193`, `step194`, `step195`): Burn's sigmoid
  composition as one op; `silu(gate) * up` as one SFPU pass (`SILU_MUL`); RMSNorm's statistic in one kernel
  (`sfpu::norm`). Decode moved to `Tt` (`TracedGenerator`'s default)
  ([log](../learnings/performance-log.md#decode-fusers-sigmoid-silu-and-rmsnorm-2026-10-16-cards-0-and-1)).
- **RMSNorm's scale in the norm's op** (2026-10-27, `burn_tt::rms_norm`, gates `step210`, `step195`).
- **The `.ttinsn` experiment** (2026-10-10, card 0,
  [log](../learnings/performance-log.md#ttinsn-push-fusion-precise-gelu-2026-10-10-card-0)): baked fused words
  2.78x on `gelu`'s push phase, bit for bit. Learned: the gain is not hardware gathering; T1 can write code and
  invalidate its own instruction cache (~330 cycles); ttsim runs `.ttinsn` bit for bit silicon.
- **Fused `.ttinsn` push, shipped and on by default** (2026-10-25, `TT_TTINSN=0` off, gate `step231`, models
  `1792958143`, [log](../learnings/performance-log.md#fused-ttinsn-push-from-the-host-the-default-2026-10-25-card-0)):
  the host emits fused code for programs of 256 pushed words or more, validated by `tt_isa::ttinsn::validate`, run
  on the host word for word against the expansion. 1.7B prefill 1.6-3.9% faster, ViT-Tiny 8.97 -> 8.32 ms, ResNet-18
  13.88 -> 13.74. The profile mode measures push-boundness (`ProfileOptions::push`).

## Training (tt-mnist)

- **The MNIST MLP, CNN and TinyTransformer train from scratch** in `tt-mnist`; `step12_mnist` is the arithmetic
  regression gate, with a golden (`mnist_reduced.txt`).
- **The training regression found and fixed** (2026-10-27,
  [log](../learnings/performance-log.md#the-tt-mnist-regression-found-and-rmsnorms-scale-in-its-op-2026-10-27-cards-0-and-1)):
  the transformer's 9.2 -> 92 ms/step bisected to `c4255c7` and `6d979e7` (the embedding gradient at ~20 ops an
  index, now one; cross-entropy's gather and scatter-add, now one and a few). MNIST 784-128-10 1.39 ms/step, the
  transformer 8.21. `bench_training` is in the default bench.
- **Checkpoints round-trip** (`step170_checkpoint_round_trip`, above).

## Multi-chip proof of concept

- **Matmul, conv and attention split across two cards over Ethernet** (`MeshEngine`, `Topology::Cards`: a matmul
  split along `N`, bit-identical to one card); mesh inference traces work (`step147`); both 800G links together
  reach 27 GB/s ([phases-0-10-checklist.md](phases-0-10-checklist.md) Phases 8-9).
- **Design decisions recorded** (2026-10-14, in the roadmap's Track 3): a Burn device is one card; data-parallel
  training through a native `DistributedBackend` (B14); a fixed-order all-reduce as the numerics contract;
  `burn-collective` reviewed and kept as a fallback.

## Tooling

- **The profile mode** (2026-10-15, `TT_PROFILE=<dir>`, `cargo xtask bench|silicon --profile`, `burn_tt::profile`,
  `tt_kernels::profile_mode`, gates `silicon_profile_mode`, `step23`): each Burn op's device time split into
  barrier, kernels, control and data; traces covered window by window (`dm::TRACE_WINDOW`); `summary.md`, `ops.tsv`,
  `trace.json`. Stamping only with the firmware `profile` feature (0.46 ms a 135M step when compiled in). Since
  extended with `channels.tsv`, `tails.tsv`, `skew.tsv`, push totals, `Detail::Pipeline` and `replays.tsv`.
- **`cargo xtask pipeline-report`** (with `--against`): puts a step's spread in a class of ops and a unit.
- **`cargo xtask fw-size`**: firmware image room per role and build.
- **`cargo xtask conformance --silicon`, `--reference-flex`, `--exact`, `--accurate`, `--fusion`** (above).
- **`cargo xtask fetch-weights`**, size-scaled fetch timeouts, SHA-256 checked.
- **ttsim for small spikes only** (2026-10-20): model gates took 50-275 s on ttsim and seconds on card 0; the rule
  is now in [test-tiers.md](../learnings/test-tiers.md).
- **xtask waits for a timed-out test's whole process group** (2026-11-01).
