# Re-running the tt-metal comparison

How the 2026-10-26 snapshot, [tt-metal-comparison.md](tt-metal-comparison.md), was made, so it can be repeated
against a newer tt-metal and a newer version of this repository, and the two snapshots diffed. It is a procedure,
not a tool: five read-only agents, a claims sheet that keeps them honest, and a spot-check of what they report.

Cost of the first run: five agents, 100-510 s each, about 130-270k tokens each (about one million tokens in
all), plus about 20 minutes of the operator's time to check and write up. **Nothing in the procedure touches a
card.** The one thing that needs hardware is fix F1 in the findings (a same-card baseline); a re-run should
report whether F1 has since been done and, if so, replace the derived figures with measured ones.

## When to re-run

- A new tt-metal release or a large change to `models/tt_transformers`, `tt_metal/impl/trace`, `tt_fabric` or
  the Blackhole LLK (`git log` on those paths since the recorded revision).
- After one of the findings' fixes lands (F1, F2, F3, F5, F6, F7, F8): the verdict row for it should change.
- After a large change here to decode, prefill, traces or the multi-card path (a new scoreboard row).
- Before choosing what to work on next from the roadmap's tt-metal items.

## 0. Preconditions

- A local tt-metal checkout. The first run used `/mnt/nvme/tt-metal` (`a11d86c19d3`,
  `v0.81.0-dev20261013-1`); `/home/carl/projects/tt-metal-v077` is an old v0.7.7 tree that this repository's earlier
  benchmarks compared against (useful only for UMD sources the newer tree lacks).
- **Do not move the checkout other work uses.** To read a newer revision, add a worktree beside it instead:

  ```bash
  git -C /mnt/nvme/tt-metal fetch origin
  git -C /mnt/nvme/tt-metal worktree add /mnt/nvme/tt-metal-<shortrev> origin/main
  ```

  The agents and the spot-check then read `/mnt/nvme/tt-metal-<shortrev>`. Remove the worktree when done
  (`git worktree remove`). Initialise the `tt-umd` submodule in it if the KMD and firmware minimums are wanted
  (the first run could not read them: the submodule was not checked out).
- Record in the new snapshot: both repositories' `git rev-parse --short HEAD`, the tt-metal `git describe --tags`,
  the date, and whether `git status` of either was dirty.
- Optional, in-session web search for anything the tree cannot answer (a Blackhole number in a release note, an
  issue such as tt-metal#16439). Mark such figures as web-sourced; they are not pinned to the revision.

## 1. Refresh the claims sheet

The agents compare against *our* claims, so those must be current. Re-read the sources below and rewrite the sheet
(section 3) with today's numbers and run IDs. Do not paste a roadmap headline without checking it against the
scoreboard row: the 2026-10-26 run found the roadmap and scoreboard stale in several places.

| Claim | Where it lives |
|---|---|
| Decode ms and tokens/s (1.7B BF16 and F32, 135M), bandwidth % of 512 | `performance-scoreboard.md` (decode rows, "roofline"), `performance-log.md` newest entries, `bf16-weight-storage.md` |
| Prefill tokens/s, 8K prompt time, TFLOPS | scoreboard prefill rows; `prefill-operand-reuse.md`; `long-context.md` |
| Per-op fixed cost, replay start | `replay-fixed-cost.md` |
| Program cache size, trace sizes, build times | `traced-execution.md`, `long-context.md`, `layer-sub-traces.md`, roadmap Track 2 item 6 |
| L1 residency numbers | `planned-l1-residency.md` |
| SFPU push, `.ttinsn`, approximate-mode costs | `ttinsn-push.md`, `performance-log.md` ("ttinsn push fusion", "fused .ttinsn push") |
| DRAM bandwidth (aggregate and streamed) | scoreboard top table; `performance-log.md` DRAM entries |
| Sampling cost | `performance-log.md` ("sampling", on-card top-k) |
| Multi-card and Ethernet | scoreboard mesh rows; `sram-residency-and-multi-card-decode.md`; roadmap Track 3 |
| Conformance counts | `crates/burn-tt-conformance/README.md` results table and `expected-failures*.toml` entry counts |
| Models and dtypes | `crates/tt-models/README.md`, `quant.rs`, `bf16-weight-storage.md` |

Counts that are cheap to recompute rather than trust:

```bash
grep -c '^\[\[' crates/burn-tt-conformance/expected-failures.toml        # listed groups
git log main..HEAD --oneline | wc -l                                      # commits ahead of main
find crates -name '*.rs' -not -path '*/.claude/*' | xargs wc -l | sort -n | tail -15   # mega-files
```

## 2. Launch the agents

Use `Explore` agents (read-only), in parallel, one message. T1-T3 are the tt-metal comparison; P1-P2 refresh the
state-of-the-project half and can be skipped when only tt-metal changed. Give each the claims sheet (section 3).
Each prompt in section 4 asks for citations (path and line), exact quotes for numbers, an explicit "not found"
list, and the Wormhole-or-Blackhole, batch size and hardware of every figure. **These requirements are the point of
the procedure**; keep them when editing a prompt.

## 3. Claims sheet (the 2026-10-26 values; replace each time)

Hardware: single Blackhole p150a, 32 GB GDDR6, 512 GB/s peak, 469 GB/s measured read ceiling (120 tiles); two
cabled p150a for multi-card.

- SmolLM2-1.7B-Instruct decode: 11.76 / 12.24 / 12.95 ms (BF16 weights, 32 / 128 / 256 tokens, run
  `1793002379`), 16.79 / 17.19 / 18.09 ms (F32, `1793002966`); BF16 weights 3.4 GB, about 57% of 512 GB/s; F32 about
  84%. F32 activations and KV cache (393 KB a position). Greedy argmax only is fast.
- Prefill (F32): 315 / 741 / 861 tokens/s at 32 / 128 / 256 tokens; 7,992 tokens in 6.62 s at chunks of 512 and an
  8,192-position cache; products 5-18 TFLOPS, 1-2% of compute.
- Sampling: temperature, top-k, top-p on the card but eager, 112 ms a token.
- Batch 1 only; no paged cache; no tokenizer or server; Llama-style models only.
- Per-op fixed cost about 4.1 us (3.5 control + 0.6 barrier); replay start about 170 us.
- Per-tile program cache 240 KiB holds every op's program; "a list's programs do not fit" at 2,048 and 8,192
  positions (unrolled unpack/pack grow with length); looped programs and paging added; trace streams 1.5 GB for the
  1.7B (2026-10-15, may be stale) and about 25 32-byte entries a section on each of up to 120 units; sub-traces
  5.6x (decode) and 8.4x (prefill) smaller; first trace build 569-893 s at 8K.
- Planned L1 residency: 393 of 548 tensors resident in a 1.7B step; weights prefetched into L1 on idle units.
- SFPU push: 3.31 cycles a word (`lw`+`sw`) against 1.19 with `.ttinsn`; instruction gathering off in every image;
  approximate math 1.2x-8.2x faster than precise; softmax `SUB_EXP`/`DIV` bodies of 57 and 62 words overflow the
  32-entry replay buffer.
- Matmul: TF32-grade by default (MVMUL adds 16 products in two groups of eight, aligned once); `--accurate` 19-64x.
- DRAM: `silicon_bench_memory::gddr_adjacent_banks` mirrors tt-metal's Blackhole gate (`cargo xtask bench --device 1
  --feature profile --feature fast-reads --filter silicon_bench_memory::gddr_adjacent_banks`; the default image has no
  stamps, so `--feature profile` is required): 507.7-508.3 GB/s with 16 KiB packet entries and 511.2 with 128 KiB entries
  (safe image 414 / 511), against tt-metal's 490-512; `gddr_aggregate` (the v0.7.7 counterpart is
  `crates/tt-tests/bench/tt_metal_dram_read/`): 480-496 GB/s on 120 tiles at 4-16 KiB (446 at 2 KiB), 469 GB/s in the
  older scoreboard row, 487-489 GB/s per streamed weight; NoC0 reads, NoC1 writes, each GDDR endpoint owned by
  one NoC (SYS-1419).
- Ethernet: one link 48-51 GB/s, both 800G links 27 GB/s, store-and-forward; hop latency unmeasured; card-to-card
  `to_device` is a host round trip; no all-reduce; no LLM across cards.
- Firmware: E1 customer code, E0 left alone; ARC firmware version not checked; ttkmd 2.11.0.
- Attention: fast op holds a head's K and V in one unit's L1 (about 512 positions at head_dim 64); longer runs use
  the composition; flash-decode parked because it changes accumulation order.

## 4. Prompts

Replace `{{TT}}` with the tt-metal path, `{{REPO}}` with this repository's path, `{{CLAIMS}}` with the claims sheet.
Every prompt begins with: *Read-only research, very thorough. Do not edit anything in either tree.*

### P1. Project status (models and performance)

> Repo `{{REPO}}` is a Rust stack for Tenstorrent Blackhole (host access, firmware, kernels, Burn backend).
> Determine how close it is to a viable LLM inference engine from the evidence in docs and code. Read
> `docs/learnings/performance-scoreboard.md`, the newest entries of `docs/learnings/performance-log.md`,
> `docs/plans/burn-backend-parity.md`, `burn-native-cutover.md`, `traced-execution.md`, `kernel-fusion.md`,
> `docs/proposals/*`, `crates/tt-models` (README and layout), `crates/burn-tt-conformance/expected-failures.toml`.
> Report with concrete numbers: (1) which models run on silicon and their tokens/s or latency against the memory
> roofline and any cited tt-metal reference; (2) the share of GDDR bandwidth decode reaches; (3) prefill;
> (4) KV cache, sampling, batching, sequence length limits, dtype and quantisation support; (5) multi-card, mesh and
> Ethernet; (6) conformance pass and fail counts; (7) stated blockers, known-wrong results, disabled features, TODOs
> and any stale or contradictory claim you notice. Quote paths. Do not run anything.

### P2. Code health

> Repo `{{REPO}}`. Assess code health and risk. (1) Lines of Rust per crate under `crates/` and files over about
> 3,000 lines (exclude `.claude/worktrees`, which is a second copy of the tree); (2) test infrastructure: test
> files and `#[test]` counts, silicon-only against simulator-only, `#[ignore]` counts and reasons, CI
> (`.github/workflows`, `prek.toml`) and whether CI can run silicon; (3) `unsafe` density in `tt-kmd`, `tt-device`,
> `tt-firmware`, and `unwrap`/`expect`/`panic!`/`todo!` in shipped non-test code, TODO/FIXME counts; (4) shipped-crate
> dependency hygiene against `xtask/src/ship.rs` (SHIPPABLE and DEV_ONLY) and any FFI or simulator or flex leak;
> (5) docs sprawl and contradictions between roadmap, README, AGENTS.md and code; (6) `vendor/` and `PINS.toml`
> (external binaries, redistribution and licence concerns, a missing LICENSE); (7) reproducibility (machine, kernel
> module, firmware blobs, toolchain pin). Report file paths and numbers, and call out the biggest risks. Do not
> build or run tests.

### T1. tt-transformers performance, techniques and serving

> tt-metal is at `{{TT}}` (note its `git describe`). Our stack's claims: `{{CLAIMS}}`. Find what tt-metal's own LLM
> stack (`models/tt_transformers`, `models/demos/llama3_70b_galaxy`, `models/demos/blackhole`, `models/perf`,
> `models/model_targets.yaml`, `models/tt_transformers/PERF.md`, `tech_reports/LLMs/*`) achieves and does for each
> point, with file paths, line references and exact quoted numbers. Report: (1) published decode and prefill
> performance on Blackhole (p100, p150, p300) for the nearest-size models: tokens/s/user, ms/token, TTFT, batch size,
> precision (BFP4/BFP8/BF16, LoFi/HiFi2/HiFi4, KV dtype), and an implied GB/s against the card's DRAM bandwidth,
> stating batch 1 or 32 and whether it is the same hardware; (2) the decode techniques (DRAM-sharded matmul,
> L1-sharded activations, prefetcher, fused ops, trace, core counts) and their stated DRAM utilisation;
> (3) prefill program configs, TFLOPS or utilisation, chunking, maximum length, prefill trace handling; (4) on-device
> sampling implementation, limits, trace compatibility and cost; (5) batching, paged attention, KV dtype, vLLM
> integration, maximum context per model and device; (6) the accuracy philosophy (PCC and top-k targets, default
> fidelity) against our bit-exact goal; (7) model coverage and how Qwen bias, RoPE scaling, MoE and sliding window are
> handled. Give a table of numbers, then where we are ahead, behind or not comparable, and flag every number you
> could not find or that is stale for this checkout. Under 1,500 words.

### T2. Traces, dispatch and attention limits

> tt-metal is at `{{TT}}`; ours is `{{REPO}}`. Our design facts: `{{CLAIMS}}` (especially the program cache, trace
> sizes, per-op cost, L1 residency and attention fit). Find in tt-metal, with paths, lines and quotes: (1) how traces
> work (capture and replay, `trace_region_size`, where the trace lives, how it reaches the dispatch cores, limits,
> the error when it does not fit, whether program binaries are in the trace, the kernel-config, launch-message and
> prefetcher ring sizes, Blackhole per-core L1 limits on binaries, whether replay re-uploads binaries); read
> `models/model_trace_region_sizes.yaml` and any guidance on how large LLM traces get and advice on tracing prefill;
> (2) the fixed per-op cost under fast dispatch against traced (goldens in `tests/tt_metal/tt_metal/perf_microbenchmark`,
> `tech_reports/AdvancedPerformanceOptimizationsForModels`), dispatch core counts on Blackhole; (3) the ttnn program
> cache (host-side) against kernel JIT, why they avoid "programs do not fit", runtime against compile-time
> arguments; (4) the L1 and DRAM allocator, sharded memory configs and circular buffers, DRAM-sharded matmul and the
> prefetchers, with the bandwidth figures they publish; (5) SDPA decode and flash-decode: limits on sequence
> length and head dimension, the L1 fit rule, split-K across cores, paged mode. Per topic give tt-metal's approach,
> its numbers and limits, and a one-line verdict on whether our limitation is inherent to the hardware or a
> consequence of our design. Under 1,500 words.

### T3. SFPU, matrix numerics, multi-chip, DRAM, errata

> tt-metal is at `{{TT}}`; ours is `{{REPO}}`; our claims are `{{CLAIMS}}`. (A) How tt-metal and the LLK handle
> push-bound SFPU code: `tt_metal/tt-llk`, `TTI_*` and `.ttinsn`, replay buffer use, instruction gathering
> configuration (tt-metal#16439), SFPU exp, gelu and reciprocal, approximate against precise defaults, how softmax,
> layernorm and rmsnorm are implemented. (B) Matrix-unit numerics and defaults: math fidelity, `fp32_dest_acc_en`,
> `packer_l1_acc`, what `ttnn.matmul` and tt_transformers default to, what accumulation guarantees are documented.
> (C) Multi-chip: fabric, ethernet data movement, CCL ops, Blackhole p300 and p150 cabled support, topology and link
> counts, published latencies (microseconds a hop) and bandwidths, small (4-16 KiB) transfers, all-gather in decode,
> and how Llama tensor-parallel decode runs on Blackhole. (D) DRAM bandwidth: Blackhole read microbenchmarks and any
> CI floor, banks and channels, NoC assignment, the NoC1 split (SYS-1419), DRAM-sharded matmul GB/s. (E) Errata and
> versions: firmware, KMD and tt-smi minimums, reset behaviour, harvesting, Ethernet E0 and E1 use. Quote paths,
> lines and numbers. End with: where tt-metal does it better, where our finding is new or contradicts its practice,
> where it is not comparable, and what you could not confirm. Under 1,500 words.

## 5. Spot-check before writing

Agents sometimes cite a line that has moved, or merge Wormhole and Blackhole figures. **Re-check the claims the
verdicts lean on**, run from the tt-metal path, and mark each one that held with ✓ in the snapshot. At the first
run these six held:

```bash
cd {{TT}}
grep -n default_l1_kernel_config_size tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp        # 69 * 1024
sed -n 55,70p tt_metal/impl/dispatch/simple_trace_allocator.cpp                                    # Belady comment
grep -n prefetch_ringbuffer_size tt_metal/impl/dispatch/util/dispatch_settings.cpp                 # 1024 KB, 864 KB, 67 KB
grep -n bw_lower_bound tests/scripts/test_moreh_microbenchmark.py                                  # 490.0 (Blackhole)
grep -n -B1 -A3 '55,787,520' models/model_trace_region_sizes.yaml                                  # 8B batch-32 trace size
grep -n official_op2op tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/op_to_op_blackhole_golden.json
```

Also re-check, each time: the headline throughput rows (`grep -n -A12 'bh_p150' models/model_targets.yaml`, and
`models/README.md` lines 19-22), any figure that drives a fix (a hop latency, a trace size), and one claim the
agents flagged as unconfirmed. If a claim fails the check, correct it in the snapshot and say so. Never repeat an
agent's number that you did not look up when it carries a verdict.

## 6. Write the snapshot

Copy the structure of [tt-metal-comparison.md](tt-metal-comparison.md) into a new dated file (or replace the
sections in place and keep the old figures in a short "changes since" list, whichever the reader of the history
prefers):

1. Header with both revisions, the date and the meaning of ✓ and *derived*.
2. Verdict (3-4 bullets).
3. State of the project (from P1 and P2), then the tt-metal tables (T1), then the highlight table (one row per
   claim in the sheet, verdict **inherent / design / not comparable**), then the detail sections.
4. Ahead, behind, not comparable; the "not found" list (absence in the tree is not evidence of absence upstream).
5. Fixes to investigate, **each with why, the first measurement, and what would settle it**. Mark fixes that have
   since been done and cite the run IDs.
6. Caveats.

Diff against the previous snapshot: for each highlight row, did our number move, did tt-metal's move, did the
verdict change? List the rows that changed at the top. Update the roadmap's tt-metal item (checkboxes for F-items
that closed) and the `docs/README.md` entry if files were added.

## 7. Pitfalls met so far

- **Wormhole against Blackhole.** `llms.md` and the DRAM-saturation report are Wormhole. Only the Blackhole rows
  in `model_targets.yaml` and the Blackhole goldens apply to a p150.
- **Batch 1 against batch 32.** The published p150 figures are mostly batch 32; the batch-1 figure sits in a
  deprecated `PERF.md`. Never quote one without its batch size.
- **Different models.** There is no tt-metal number for a 1.7B MHA model on a p150. Say "same band", not "faster".
- **Derived figures.** Implied GB/s and effective TFLOPS depend on assumed weight sizes and on the TTFT's
  definition (theirs is per user at 128 tokens and includes first-token sampling). Label them.
- **Stale or contradictory sources.** The README table, `PERF.md` and the yaml disagree; the yaml is the newest.
  Our roadmap and scoreboard also lag; check run IDs.
- **Galaxy against cabled cards.** CCL latency tables are Galaxy-style runs; the mux throughput golden is on-chip
  data movement, not the Ethernet wire. Say which a number is.
- **Line numbers drift** between revisions; pin the revision and re-grep rather than trusting an old line.
- **Agent hypotheses.** An agent handed our hypothesis tends to confirm it. The prompts therefore state our
  facts and ask for the other side's, and the spot-check looks for the failing case.
