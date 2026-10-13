# Experiment: does `.ttinsn` push fusion speed up SFPU kernels?

**Status:** complete 2026-10-10 on card 0. **Result: go.** A fused push is 2.78x faster on the
push phase and 2.77x on a whole three-role `gelu` kernel; see [Results](#results). The follow-on (a shipped mechanism) is **parked** by owner decision until a real model shows push-bound
SFPU device time; see the roadmap's "Parked" section for the reasoning, trigger and work list.
**Decisions:** recorded at the bottom.
**Reopens:** the P9 `[-]` for `.ttinsn` fusion
([close-out line 157](../completed-plans/hardware-coverage-closeout.md)). P9's own rule was
"measured first; `[-]` if no measurable push bottleneck or if it weakens the build-time
instruction gate". Nothing was ever measured, so this experiment does that measurement.
Linked from [roadmap.md](roadmap.md), Track 2.

## Question

Today the T1 runner pushes a program with `lw` then `sw` to `INSTRN_BUF`, sixteen words at a
time, about 2.8 cycles per word (`crates/tt-firmware/src/corpus.rs:851-872`; 7.6 one at a
time, 3.5 eight at a time). `.ttinsn` encodes the same push as one instruction, and the T0/T1/T2
instruction caches fuse up to four adjacent ones into a single cycle
(`vendor/tt-isa-documentation/BlackholeA0/TensixTile/BabyRISCV/PushTensixInstruction.md:19`).
The frontend still drains only one instruction per thread per cycle, so the best case is
about 1 cycle per word. **Does a fused `.ttinsn` image make the push phase of a real SFPU
kernel measurably shorter, and does that survive into op time?**

## What we know, and what we do not

- **The existing "push-bound" figure is a model, not a measurement.** "39 of 73 SFPU kinds above
  1, median 1.17, max 2.87" comes from
  `how_push_bound_are_the_sfpu_math_programs` (`crates/tt-tests/tests/step121_wait_planner.rs:594-646`):
  `2.8 * words / frontend_stream_len`, where the backend is assumed to retire one instruction per
  cycle. It prints only min/q1/median/q3/max, so **no per-kind ratios are recorded anywhere**.
  If the real backend is slower than one per cycle, the true push-bound share is smaller.
- **Measured:** `silicon_perf::role_push_rate` (`silicon_perf.rs:884-960`) measured 2.8 cycles
  per word on 4000 NOPs on T1, and that matmul unpack roles push at the runner ceiling
  ([log, "Core path"](../learnings/performance-log.md)). It did not measure SFPU kinds.
- **The "28.46 µs per tile" for precise `gelu` is host wall time** for the whole
  `Session::eltwise` op (mover, unpack, pack included: `(t64 - t4)/60`,
  `silicon_perf.rs:970-1060`). It is not a push-phase time and must not be used as the baseline.
- The fixed costs `.ttinsn` cannot touch: about 171-183 cycles from role wake to first push, and
  9-10 cycles from last push to retire
  ([log, "Core path"](../learnings/performance-log.md)).
- **ttsim probably cannot decode `.ttinsn`.** No mention in `ttsim-divergence.md` or
  `crates/tt-ttsim*`, and `strings` on `vendor/libttsim_bh.so` shows no `ttinsn` (only
  `rv32_unimplemented`). A simulator gate can at best be a decode-or-refuse probe. Treat
  silicon as the only oracle for the pushed stream.

## Feasibility facts (from code, for the design)

- The T1 math role is static per SFPU kind apart from scalar immediates: operands sit at fixed
  `Dst` rows (`A_ROW=0`, `B_ROW=64`, `OUT_ROW=128`, `sfpu/kernel.rs:41-45`), row loops use RWC
  stepping (`sfpu/mod.rs:697-710`), and tile count is only the block-repeat count wrapped
  around a per-tile block (`kernel.rs:191-195`). Scalar kinds bake `scalars[0].to_bits()`
  into `loadi_bits` immediates (`ops.rs:2578`), so they would need one image per scalar.
- A scalar-free unary kind therefore fits a **build-time image per kind** with no run-time code
  generator. The row-loop body is about 300 words (about 1.2 KiB), which fits the ~4 KiB I-cache
  (`performance-log.md:357-370`); the whole precise-`gelu` program is about 9,200 words.
- The T1 image is linked in T1's role slot, 0xD000, LENGTH 0x2800 (`crates/tt-firmware/link_role_t1.x`; it was 0xA000, LENGTH 0x4000, `link_t1.x`, before the role images were repacked).

## Constraints that shape the plan

- `AGENTS.md`: do not emit compressed instructions, and preserve the build-time instruction gate.
  `.ttinsn` words live in the compressed encoding space (low two bits 00, 01 or 10), so objdump
  without the C extension will print `<unknown>` or `c.*`, which the gate rejects. The
  experiment must therefore be an **isolated, opt-in image** that never weakens the gate for
  any other image.
- The gate is `check_instruction_set` in `crates/tt-firmware-images/build.rs:149-198`
  (`forbidden_reason` at `:230-284`), not `crates/tt-tests/build.rs`. That crate embeds the
  shipped firmware images. **Open question 1** asks whether the experiment image may live
  there at all or must live in a dev-only crate.
- Hardware rules
  ([silicon-operating-notes.md:791-807](../learnings/silicon-operating-notes.md)): an unverified
  encoding gets a ttsim pass or a recorded refusal first, then an isolated minimal probe on
  silicon. A hang becomes a refusal, never a retry. One silicon test per process, never concurrent
  with other silicon work. Tile 0 on card 0 first. A bad word can wedge the tile until the next
  session reset.
- Silicon runs only through `cargo xtask silicon --release --device N --filter <gate>`, never
  plain `cargo test`. `--list` touches no card.

## Phases (as planned; [Results](#results) says what happened)

Each phase ends with a go/no-go. Stop at the first no-go and record why.

### Phase 0: per-kind model data (host only, no hardware)

Extend `how_push_bound_are_the_sfpu_math_programs` to print the ratio and word count **per kind**
(and mark scalar-free unary kinds). Archive the table in
[performance-log.md](../learnings/performance-log.md).
**Go if:** at least a few scalar-free unary kinds have ratio > 1.3 and a program long enough to
matter. Pick the candidate (precise `gelu`, `kind_sfpu::GELU = 0x12f`, is the first guess; use
the table, not the guess).
**No-go if:** no kind is meaningfully push-bound; record "model says no" and stop.

### Phase 1: build-only probe (host only)

Emit rotated words with `Instruction::ttinsn_word` (`crates/tt-isa/src/isa/mod.rs:467`) via
`asm!(".word 0x...")` or `.4byte` after `.balign 4` into a named section such as `.text.ttinsn`
with start and end symbols. Do not use `.insn` (its 32-bit form requires low bits 11), and keep
`-C target-feature=-c`. Disassemble it and record exactly what `llvm-objdump` prints (unverified:
`<unknown>` vs `c.*`, and whether LLVM adds `$d` mapping symbols for `.word` in `.text`).
**Go if:** the section is bounded, deterministic and its words round-trip
(`w & 3 != 3`, `w.rotate_right(2)` equals a table-encodable word).

### Phase 2: gate change, minimal and opt-in (host only)

In `check_instruction_set` add `allow_ttinsn: bool`, true only for the experiment image name.
Skip only that section's address range in the line loop; validate each word in it from
`llvm-objcopy --dump-section` by the round-trip rule above. Everything outside the range, and
every other image, goes through the unchanged gate. **Negative controls that must fail:** a stray
`.word` outside the section, a `c.nop` anywhere, and the same image built without the allow flag.
Watch each fail before trusting the pass. Do not touch `link.x` or the role images. The image is
built only when a cargo feature (proposed `ttinsn-experiment`, off by default) is on and is not
part of the default embed.

### Phase 3: ttsim decode-or-refuse probe

In `tt_ttsim::fork_scope`, run the smallest image (one `.ttinsn` NOP). Expected outcome: refusal.
Record a `ttsim-divergence.md` row either way (row number next in sequence). If ttsim refuses, the
pushed stream cannot be validated on the simulator; say so in the gate's docs.
**Go:** proceed to silicon regardless, because a refusal is the recorded precondition.

### Phase 4: silicon ladder (card 0, tile 0, one test per process)

Template: `crates/tt-tests/tests/probe_icache.rs` (`in_device`, `GATE_TILE`, images loaded
manually). Run `cargo xtask silicon --list` first, then each rung as its own filtered run:

1. One `.ttinsn` NOP on T1: the tile retires it, no hang.
2. Four adjacent `.ttinsn` NOPs (the fusion case): retired count is 4, no hang.
3. A short real program (a few dozen SFPU words) from the candidate kind: `Dst` bit-exact against
   the interpreter and `reference`.
4. The full candidate kind: bit-exact, then timing.

Any hang or wrong result stops the ladder, is written up as a refusal in
`silicon-operating-notes.md`, and is not retried. Run on card 1 only after card 0 is clean.

### Phase 5: measurement

Use `role_push_rate`'s `spans()` (`silicon_perf.rs:887-896`) with the math-only `Run::roles`
(no mover, so no concurrent timestamp-export corruption; do **not** use NC or B timestamp
exports, `AGENTS.md`). Three variants of the same program on T1:

- **A, baseline:** the existing runner path (`lw` + `sw`, batches of sixteen).
- **B, control:** an unfused straight-line `sw` image (removes the L1 read but not the fusion).
- **C, fused:** the `.ttinsn` image.

Report `START -> PUSHED` and `PUSHED -> RETIRED` separately (this isolates the fixed 171-183
and 9-10 cycle costs from the push phase). Median of 9 (`tt_tests::bench::{median, REPS}`) after
warm-ups, output validated bit-exact on every run, run IDs recorded. Then, only if the push phase
improved, a second measurement at op level: the same kind through `Session::eltwise` on a resident
GDDR tensor across many tiles, because the end-to-end gain is bounded by the push fraction of
the op.

### Phase 6: decision and records

Proposed thresholds (**open question 2**): continue to a build-time-image-per-kind design only
if the push phase is at least 1.5x faster in variant C against A **and** the op-level time on a
large resident tensor improves by at least 1.15x. Otherwise leave P9 as `[-]` with the new
evidence.

Either way, update: the performance scoreboard and log with conditions and run IDs, a
`ttsim-divergence.md` row, `silicon-operating-notes.md` if anything hung or misbehaved, and the
`.ttinsn` line in [roadmap.md](roadmap.md) Track 2. If the answer is yes, the follow-on design
(image-per-kind at build time, scalar kinds, I-cache fit, how `Session` selects an image, a
gate story that ships) is a new plan, not part of this one.

## Risks and unknowns

- The ratio is a model; the backend may retire fewer than one instruction per cycle, so the
  ceiling may be lower than 2.87x (precise `gelu` is only about 67% push by a rough count).
- Whether fusion works for the T1 pairing and alongside Auto TTSync is undocumented on silicon
  here; rung 2 exists for that.
- objdump behaviour on `.word` in `.text` is unverified (Phase 1).
- A wedged tile is possible; the ladder keeps blast radius to tile 0 on one card.
- Host-bound workloads (all current Burn training) will not see any of this; the benefit is
  only for device-bound elementwise work at scale.

## Results

Numbers and run IDs are in [performance-log.md](../learnings/performance-log.md#ttinsn-push-fusion-precise-gelu-2026-10-10-card-0).

| Phase | Outcome |
|---|---|
| 0, model per kind | Go. 23 scalar-free unary kinds sit at the model's 2.80 ceiling, with stored programs of 41-368 words (precise `gelu`: 368 stored, 11,495 pushed), so a looped `.ttinsn` image is small. `per_kind_push_table` in `step121_wait_planner.rs` prints the table. |
| 1, build probe | `llvm-objdump` prints `.word` for the words (the assembler marks them data), not `<unknown>`: see the gate finding below. The image is 14,768 bytes of the 16,384 the T1 slot holds. |
| 2, gate | Done, behind the feature. Negative controls watched failing: a stray inline `.word` in the experiment image and in an ordinary image, a run word with the low bits `11`, a run word whose opcode is not a Tensix instruction. Default firmware images are byte-identical to `HEAD`'s (checked against a scratch build; only embedded source paths differ in `dm_b`/`dm_nc`). |
| 3, ttsim | ttsim **executes** the fused words (it did not refuse); results match the runner's bit for bit for every program, including the 11,581-word `gelu`, and match silicon. Divergence row AQ. |
| 4, silicon ladder | Rungs 1-4 pass on card 0 (one fused NOP; four adjacent; a 130-word program; the full `gelu`), no hang, bit-exact. |
| 5, measurement | Push phase 2.78x; whole kernel 2.77x (1 tile 2.65x, 16 tiles 2.77x); repeats agree. |
| 6, decision | Both thresholds cleared: push phase >= 1.5x (2.78x) and op level >= 1.15x (2.77x **at the kernel level**; the Session level was not measured). Continue to a design. |

### Where this differed from the plan

- **The image is the real runner with two hooks, not a separate probe.** `corpus.rs` gained two
  feature-gated hooks (`experiment::TRY_PUSH`, `AFTER_RETIRE`); the experiment image `ttinsn_t1`
  installs them. A `MODE` word in L1 picks the runner's own push (0) or the fused push (1), so the baseline runs
  the same binary at the same addresses. That let the experiment reuse the host staging, the three-role
  kernel and the timestamper unchanged.
- **Variant B (an unfused straight-line `sw` image) was dropped.** Materialising each word's immediate is
  `lui` + `addi` + `sw`, three instructions a push, so it cannot beat the runner's 3.32 cycles a word and
  would teach nothing about the L1 read.
- **ttsim could be a real gate.** The plan expected a refusal; ttsim executes the words, so Phase 3 is a
  bit-exact comparison with controls, not a refusal record.
- **The first comparison was unsound twice, and the controls caught it.** (1) A checksum that xor-rotated
  cancelled to zero on the repeated four-row pattern. (2) On silicon `Dst` keeps its contents between runs, so a push
  that wrote nothing matched the previous run's results and a control that halved the loop count did
  not change them; the rung 3 pass was redone. The prelude now poisons rows 64..256 with a sentinel, and the
  kernel-level test poisons the result rows before each run.
- **The gate had a hole.** `.word` is data to the disassembler, so the existing check would have let a
  compressed-space word through in any image. It now refuses data in code for every image (none had any),
  and the experiment image's runs are validated directly. `AGENTS.md` records the one exception.
- **Cargo:** `cargo xtask silicon` gained `--feature NAME` so the experiment's tests can build through the
  runner silicon runs require.

### What this does not show

- The **Session level**: the mover and host around the kernel. The earlier Session-level per-tile slope for
  precise `gelu` (28.46 us) equals the kernel's 38,053 cycles at ~1.35 GHz, so a large op is probably
  kernel-bound and would follow, but that is an inference, not a measurement.
- Kinds other than `gelu`, the matmul (backend-bound), and card 1 (rerun there before any adoption).
- That a **general mechanism** ships. The baked experiment matches a few programs built into the image;
  the run-time translator (below) removes the image-per-kind problem, but a real design still needs the
  translator in the shipped runner, the streamed `push_body` path hooked as well as `run_in`, a planned L1
  code buffer (`l1::Requirements`) with a translation cache, a size threshold for when to translate, and a gate
  story that ships (generated code cannot be seen by the build-time gate; the translator is its own gate:
  `.ttinsn` words only, validated, in a fixed loop skeleton). That is the next plan.

## Run-time translator (follow-on probe, 2026-10-10)

Instead of an image per kind, the runner **translates the staged program into fused code in L1 at run
time**, invalidates T1's instruction cache, and calls it (`MODE` 2 of the experiment image;
`crates/tt-firmware/src/ttinsn.rs`). Numbers: [performance-log.md](../learnings/performance-log.md#ttinsn-push-fusion-precise-gelu-2026-10-10-card-0).

| Rung | Outcome |
|---|---|
| 0, mechanism, no `.ttinsn` | Passes on card 0 with a working negative control: code rewritten without invalidation runs **stale** (returns the old value); after the documented invalidation sequence it runs the new code. T1 invalidating its own cache works as documented. |
| ttsim | The translated programs, run by ttsim's core, match the runner for every program and for the 1 and 16-tile kernels (this checks the emitted counters, loops, branch offsets and return). The invalidate write runs; ttsim has no cache. |
| 5-7, silicon | Four NOPs; a 194-word program; the full precise `gelu` (516 stored words, 11,645 pushed): translated, bit-exact against the runner. |
| measurement | Whole 16-tile `gelu` kernel 608,840 -> 228,421 cycles (2.67x; baked is 2.77x); 1 tile 39,328 -> 23,219 (1.69x). Translating costs 13-22 cycles a word plus ~330 fixed, so it loses on a program without loops (194 words: 4,791 against the runner's 818) and is 4% of a 16-tile `gelu` kernel. |

Verdict: the mechanism works and needs no per-kind images, scalars or tile counts. It pays only for programs
that are long expanded and short stored, which is exactly the SFPU row-loop kernels that are push-bound.

### Instruction gathering and tt-metal (checked against tt-metal `6024ed40f33`)

The plan assumed the speedup was the instruction caches **gathering** four adjacent `.ttinsn` pushes. tt-metal
turns gathering **off** (tt-metal#16439: a `REPLAY` record window can be corrupted), and this repo's firmware
never touched the CSR, so all measurements ran with it on. Measured on card 0 with the same image
(`1791671588`): gathering on or off changes nothing (baked `gelu` push 13,868 / 13,868 cycles; 16-tile
kernel 219,767 / 219,788), and every mode stays bit-exact. **The gain is `.ttinsn` being a single store of a
compile-time immediate** (the runner's `lw` and loop bookkeeping gone), at the frontend's one instruction a
cycle. Consequences: a shipped fused path should disable gathering as tt-metal does (free, and it removes
the replay-record hazard); the headline `gelu` programs contain no `REPLAY`, so they were never exposed,
while the 194-word `short` program (32 `REPLAY`) passed bit-exact with gathering on, which proves nothing
given tt-metal's "codegen lottery". tt-metal itself gets `.ttinsn` pushes from its compiler: LLK's
`TTI_*` macros pass compile-time constants to `.ttinsn` (`TTI_INSN` in
`tt_llk_blackhole/common/inc/ckernel_ops.h`), runtime-valued `TT_*` become a store, and a JIT build per
kernel configuration (cached in `~/.cache/tt-metal-cache`) bakes data formats, loop counts and `Dst` sync mode
into each binary. The baked image and the run-time translator are this repo's stand-ins for that compiler.

**Done 2026-10-11: gathering is off in every shipped Tensix image.** `tt_firmware::disable_gathering` (tt-metal's
sequence, then a read-back of `cfg0` that publishes `panic_code::GATHERING_STILL_ON` if bit 18 did not stick) runs
at the start of the runner (T0, T1, T2, and the corpus images) and of both movers (B, NC); the Ethernet image is
left alone. This experiment image now measures *against* the default: `GATHERING_ON` (bit 11 of `MODE`) clears
the bit explicitly (`cfg0` can survive a core reset, as it does on ttsim), and the experiment records T1's
`cfg0` after each run (`gathering_is_off_by_default_and_the_flag_keeps_it_on`, ttsim and both cards, run
`1791685705`). The earlier "gathering on" numbers above are therefore the boot default, not the shipped one.

### Which core should translate

- **The mask selects any core**: B, T0, T1 and T2 (not NC) can write `RISCV_IC_INVALIDATE` for any other
  core (bit 0 B, 1 T0, 2 T1, 3 T2, 4 NC), so translating on one core and invalidating T1's cache from it is
  possible. Only T1 invalidating itself is tested.
- **B is the worst choice in this stack.** It is the streaming reader on NoC0, its per-entry path is
  latency-critical and sensitive to code layout (its instruction cache is small), and translation would
  compete with it and pollute that cache. **T0 and T2 are the idle cores**: in a push-bound SFPU kernel they
  push a few words a tile and wait on semaphores almost all the time.
- **Whether to move it at all depends on the cost.** Translation is once per kernel launch, not per tile. At
  13-22 cycles a word it is 4% of a 16-tile `gelu` kernel but 40% of a one-tile kernel, and the fixed part is
  ~330 cycles. Moving it to T0 would take it off T1's critical path and let T0 translate the next program
  while T1 runs the current one; caching the translated code across launches (the buffer survives in L1)
  would remove it for repeated ops, which is most of a training loop. Both are worth trying only if the
  Session-level measurement shows the cost matters; T1 translating its own program is the simplest thing and
  is what was measured.

## Decisions (owner, 2026-10-10)

1. **Image location:** the experiment image lives in `crates/tt-firmware-images` behind a cargo
   feature (`ttinsn-experiment`), off by default. Default and shipped builds never contain
   `.ttinsn` words and the gate is unchanged for them.
2. **Thresholds:** as proposed in Phase 6 (push phase at least 1.5x, op level at least 1.15x).
3. **First card:** card 0.

## Done when

The phase records exist and the roadmap line says either "measured: not worth it, with numbers" or
"measured: worth it, follow-on plan at <path>".
