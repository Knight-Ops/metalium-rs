# Test tiers

Which harness a gate runs on, and why. This was audited on 2026-10-29: every file in
`crates/tt-tests/tests` plus the other crates' test directories.

## The rule

| Tier | Harness | For | Not for |
|---|---|---|---|
| Validation | ttsim, plain `cargo test -p tt-tests` | Instruction encodings, single kernels and ops, mover and firmware records, residency, and simulator refusals. Also any risky instruction, which runs here before it runs on silicon (a hang on silicon reboots the host). | **Model workloads:** MNIST (training or inference), `tt_mnist` / `tt_models` models, synthetic or pretrained decoders, training loops, multi-step generation, multi-layer stacks. A single layer only when a mechanism needs it. |
| Oracle | burn-flex (CPU), inside the gates | The reference that model-scale gates compare burn-tt against: forward passes, training curves, top-1. Runs on every core. | Being the thing under test. |
| Silicon | `cargo xtask silicon --release --device 0` | Everything else: every validation gate again, every model workload, all Burn ops through `--smoke`. | Running the same single-card gate on card 1 as well (it adds nothing). |

**The mechanism** (`crates/tt-tests/Cargo.toml`):
- A model-scale test is
  `#[cfg_attr(not(feature = "silicon"), ignore = "model workload, silicon tier: ...")]`.
  It still compiles on ttsim and shows as ignored rather than passing quietly.
- `cargo xtask silicon` runs ignored tests, so the same function runs on the card.
- Its only routine run is `--smoke`, so it **must** match a `SMOKE` filter in
  `xtask/src/silicon.rs`.
- Pretrained models also need `cargo xtask fetch-weights`.
- The `e2e` and `weights` features of `tt-tests` are gone; `silicon` covers both.

**The MNIST golden** (`tests/golden/mnist_reduced.txt`) is blessed on silicon:
- Command: `TT_BLESS=1 cargo xtask silicon --release --device 0 --filter
  step12_mnist::the_mlp_trains_on_a_reduced_dataset`.
- Re-blessing on 2026-10-29 changed only the header: every loss bit equals what ttsim wrote,
  which ttsim-divergence row I had already recorded.
- The training regression for arithmetic changes is
  `cargo xtask silicon --release --device 0 --filter step12_mnist`. CI no longer runs it,
  because CI has no card.

## What moved off ttsim (2026-10-29)

**Model workloads that ran under plain `cargo test`.** The nightly CI ran them twice, plain
and with `fast-reads`:

| Gate | Workload |
|---|---|
| `step12_mnist`: the three first-forward-pass bounds, `compare_all_four_execution_modes...` | MNIST MLP 784-128-10 on real MNIST |
| `step89_mnist_cnn::convolutional_mnist_trace_matches_fresh_changed_batches` | MNIST CNN, traced SGD steps |
| `step96_mixed_bfp_mnist`, both | MNIST MLP and CNN, 4 storage policies × 4 SGD steps |
| `step59_burn_transformer`, `step40_burn_trace::transformer_...`, `step170_checkpoint_round_trip` | TinyTransformer training |
| `step187_traced_generation` (9), `step191_fused_projections` (2), `step193_fusion_decode_trace` (3), `step197_trace_bytes`, `step199_in_place_cache_write` (2), `step212_device_sampling` (2), `step217_residency_plan` (2), `step219_residency_decode` (2), `step240_bfp_weight_storage` (3) | Synthetic 2–4-layer Llama and GPT-2 decoders, traced and eager generation (up to 26 decode steps; one of them a 1024-wide model on every tile) |
| `step34_batching::several_lists_per_unit_queued_are_the_unbatched_bits` | MLP-shaped race check: weak under ttsim's lockstep |
| `step174_trace_footprint` (3) | About 14 MLP passes captured; a 4,096-tile element-wise op |
| `step223_sub_traces`: the 6-layer chain and its negative control | A multi-layer chain |

**Model workloads behind features.**
- Under `e2e`:
  - `step12_mnist`: the reduced run, four tiles, and two chips.
  - `step89_mnist_cnn::convolutional_mnist_learns_with_resident_gradients`.
- Under `weights`, or `-- --ignored`, on ttsim:
  - `step165`, `step166`, `step167`, `step185`;
  - `step187`, `step188`, `step191` and `step193`'s SmolLM2 tests;
  - `step234`, `step240`.

**Dropped:** `step12_mnist::the_mlp_trains_sharded_round_a_four_chip_ring_matching_the_golden`.
It ran only on the x4 simulator, since there are two cards. Four-chip ring coverage remains in:
- `step14_shard::a_matmul_sharded_round_a_four_chip_ring_is_bit_identical`
- `step14_shard::the_four_chip_ring_routes_chip_2_through_a_neighbour`
- `step65_burn_mesh_and_argmax::four_chip_products_are_resident_and_exact_through_relays`

**Kept on ttsim on purpose:** each is a single op or a single layer, testing a mechanism:
- `step17_resident` (MNIST-sized single matmuls);
- `step39_traces` (an MLP-shaped capture);
- `step179_nhwc_conv_pool::a_resnet_block_runs_on_views_with_no_repack_once_warm` (one
  block);
- `step40_burn_trace::traced_training_step_updates_weights_in_place_and_reduces_loss` (one
  `Linear(2→1)`);
- `step83` attention and `step85/154/157/160` convolutions (single ops);
- `step14_shard` (sharded single matmuls on the x2/x4 simulators).

**Other crates:**
- `burn-tt-conformance`'s ttsim lane (`cargo xtask conformance`) is Burn's op-level backend
  tests: in tier.
- `tt-ttsim/tests/fatality.rs` tests the simulator itself.
- `tt-models/tests/*_host.rs` are burn-flex only.

## burn-flex on every core

- **Why it was single-core:** burn-flex was built with `default-features = false,
  features = ["std"]`, which drops its `rayon` feature, so its matmul ran with
  `Parallelism::None`. The root `Cargo.toml` now enables `rayon`.
- **Numerics:**
  - gemm splits only the output's m/n blocks, and its k-blocking depends on the shape, not
    the thread count. Every other parallel path splits outputs too. So each element
    accumulates in the same order on any number of threads.
  - The one exception: a contiguous F32 `sum`/`mean` over the whole tensor, of 4Mi elements
    or more, combines per-chunk partials in work-stealing order. No gate pins those bits.
  - No golden file is burn-flex output.
- **`simd` stays off.** It changes the bits of full and last-dim sums and of `layer_norm`, and
  they depend on the host CPU's vector width.
- **The fork rule:** a forked child inherits rayon's global pool without its threads, so once
  the parent has used the pool, the child's first parallel job waits forever. `fork_scope`
  (`crates/tt-ttsim/src/fork.rs`, used by the ttsim and the silicon harness alike) runs the
  child in a fresh pool that takes the forking thread as worker 0. `f` keeps its thread and
  its thread-locals.
- **Conformance:** `cargo xtask conformance` runs one test per CPU, so it sets
  `RAYON_NUM_THREADS=1` on its children.
- **Benchmarks:** burn-flex baseline figures from before 2026-10-29 were single-core
  ([scoreboard](performance-scoreboard.md)).

## Where the simulator suite's time goes

After the re-tiering (2026-10-29, release, binaries run one after another as `cargo test`
does, per-test times from libtest's `--report-time`):
- **Total:** the 290 binaries took 1,189 s, all passing.
- **Moved off ttsim:** `step212_device_sampling::sampled_tokens_are_burn_flex_s` alone had
  taken 1,825 s. It is about 216 sampling ops against burn-flex, and it moved with the model
  workloads.
- **What is left is single ops** (in tier, but heavy):
  - `step209_attention_rows::positions_in_another_order_give_other_bits` 191 s
  - `step211_argmax_row` 85 s
  - `step160_deform_conv2d::deform_conv2d_matches_flex` 76 s
  - `step155_interpolate` (14 tests) about 58 s wall
  - `step56_scatter_on_nc` 56 s
  - `step163_accurate_matmul` 41 s
- **Next candidates** if the simulator suite should shrink further: the Burn-level sweeps among
  these, such as interpolate's gradient checks, which the silicon smoke run repeats.

## Where the silicon suite's time goes

Smoke run `1793244532` (card 0, release): 1,252 tests in 1,683 s, one test per process,
sequentially.
- **Median test:** 0.81 s, p10 0.76 s and p90 1.56 s; 993 of them took 0.5–1 s.
- **Fixed cost per device test:** about 0.7 s. Host-only tests take 0.05 s. Across about
  1,217 device tests that is roughly 900 s, about 54% of the run (breakdown below).
- **Runner overhead:** about 0.05 s per test is process spawn and the old 50 ms poll.
- **Conformance binaries:** another 0.38–0.45 s each after libtest finishes.
- **Build:** a filtered run used to build and link every test binary of `tt-tests`,
  `burn-tt-conformance` and `tt-mnist` (about 300), roughly 150 s before the first test.
- **Slowest in smoke:**
  - `step168_smollm2_mixed_precision::bf16_weights` 104 s
  - `step240::smollm2_with_bfp_weights...` 50 s
  - `step105_mutex::contended_handoff...` 38 s
  - `step114_bf16_unpack_to_dst...` 26 s
  - `step191::smollm2_fused_equals_unfused` 22 s
  - `step212::sampled_tokens...` 20 s
  - `step187::smollm2_generates_as_eagerly` 19 s
- **Outside smoke:** `step188::the_native_context_agrees_with_flex` took 2,190 s, 1,472 s of
  it burn-flex's reference, which rayon now spreads over the cores.

**Runner changes (2026-10-29), none of which drops a test:**
- A filtered run builds only the binaries its filters can select from. A filter's part
  before `::` must be in a binary's name; when it is in none, everything is built. Example:
  `--filter step12_mnist --filter step174_trace_footprint::` built two binaries in 46 s
  after a dependency change.
- Binaries are listed in parallel.
- A test's exit is waited for on a thread, not polled every 50 ms.
- `cargo xtask conformance` starts each lane's slowest tests first, from the previous run's
  times in `target/conformance/times-<lane>.tsv`. The ~31 tests that run to the 120 s limit
  then overlap the rest instead of leaving idle stretches and a 110 s tail (471 s on
  2026-10-28).

**Not done, by decision:** splitting the queue over cards 0 and 1 would roughly halve a smoke
run. Validation stays on card 0.

### The 0.7 s a device test pays

Measured with `strace` ([operating notes](silicon-operating-notes.md#opening-a-card-costs-07-s-most-of-it-bar4s-page-tables-2026-10-29-card-0)):
- Mapping BAR4, uncached and write-combining, 32 GiB each: 230 ms.
- Unmapping them: 372 ms.
- The AICLK busy and idle settles: about 110 ms.
- The 1 GiB hugepage buffer: 52 ms.

That is about 0.76 s of a 0.79 s gate.

**Proposals (each needs sign-off: they touch the shippable `tt-kmd` and the hardware contracts):**
1. **Map BAR4 lazily, per 4 GiB window, on first use** (`tt_kmd::Kmd::open`, `Kmd::bar`). A gate
   that never opens a 4 GiB window saves about 0.6 s. One that does pays about 1/8 of the
   cost per window it touches. Applications open faster for the same reason. First measure
   how many windows a Burn gate touches.
2. **Keep the clock busy for a whole run.** This is the clock-settle share of item 4 in the
   plan. The runner would hold one busy request while each child checks AICLK instead of
   settling. Saves about 110 ms a test, but it bends "one `Device` per chip" (power policy is
   chip-wide), so it needs a design.
