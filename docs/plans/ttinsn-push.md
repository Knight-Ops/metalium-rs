# Fused `.ttinsn` push for the math role (host-side expansion)

The follow-on plan the [`.ttinsn` experiment](ttinsn-push-experiment.md) asked for ("Done when"), and the
roadmap's "Fused `.ttinsn` push for SFPU kernels" (triggered 2026-10-25, raised by the user). Started
2026-10-25.

## What it is

A role runner pushes a staged program word by word: a load and a store a word, 3.31 cycles on silicon.
`.ttinsn` pushes the same word as one instruction, the Tensix word rotated left two bits and executed
(1.19 cycles a word, the experiment's measurement). The experiment did it with baked images and a
translator on the card; this does it on the **host**:

- the host turns the math role's program (`code::Code`: instructions and block repeats) into **fused
  code**, RISC-V that pushes the program's words by executing them, its block repeats counted loops
  (`tt_isa::ttinsn::emit`, `code::Code::fused`);
- it **checks** the code before it leaves: `tt_isa::ttinsn::validate` (the gate for code written at run
  time: every word a table instruction with its fixed fields, or the loop skeleton's word at its place,
  one return at the end), then runs it on the host (`tt_isa::ttinsn::execute`) against the program's own
  expansion, word for word;
- it places the code in the program cache like any program, its length flagged `tt_isa::ttinsn::FUSED`;
- the runner (`corpus.rs`, both `run_once` and the streamed `push_body`) **invalidates its instruction
  cache** once a run (`tt_firmware::fused`, the documented sequence, ~330 cycles) and **calls** the code.

No translation on the card, no image per kind. On by default since the measurements below
(`Session::set_ttinsn`; `TT_TTINSN=0` turns it off). The same words reach the coprocessor either way, so
results are bit for bit.

### Fused code (`tt_isa::ttinsn`)

| Part | Words |
|---|---|
| A Tensix word | rotated left two bits; low bits never `0b11` (it must be below `0xC000_0000`), and rotated back a table instruction with its fixed fields |
| A loop | `addi tN, x0, count` (1..=2047), the body, `addi tN, tN, -1`, `bne tN, x0, <body>`; `t0` outer, `t1` inner, two deep at most |
| A far loop (body past `bne`'s 4 KiB) | as a loop, closed by `beq tN, x0, 8` and `jal x0, <body>` (a row broadcast's 1,990-word `DIV` body in a tile loop) |
| The end | `jalr x0, 0(ra)`, once, last |

A program with a seed directive (no `.ttinsn` form) is pushed as before. The runner, like `Code::expand`,
does not loop a block repeat that spans exactly the range being walked; the emitter walks as the runner does.

## Checklist

### Phase 0: what is push-bound (the profile mode)

- [x] **Push totals in the profile mode** (`ProfileOptions::push`, on by default, `TT_PROFILE_PUSH=0` off).
  The math role totals, per run, the cycles it spent pushing, its wait after pushing for the coprocessor,
  and the words it pushed (`mailbox::PUSH_CYCLES`, `DRAIN_CYCLES`, `PUSH_WORDS`, written before it
  acknowledges, only in a `profile` image and only when its `TRACE` word has `trace_word::PUSH`); the mover
  stamps them after the kernel's `ROLES_DONE` inside its window (`trace::ROLE_PUSH`, `ROLE_DRAIN`,
  `ROLE_WORDS`), so role and mover stamps never share the stream and a replay's window still bounds them.
  Attributed per op and unit (`UnitTime::push`, `drain`, `words`), in `ops.tsv` / `skew.tsv` and a summary
  table by label. **Push time alone is not push-boundness:** a push into a full FIFO stalls, so a
  backend-bound kernel pushes "slowly" too (the matmul showed 87-91% of its work pushing). Push-bound is
  pushing at the runner's own rate, ~3.3 cycles a word (`PUSH_BOUND_CYCLES_A_WORD` = 4 counts).
- [x] **`per_kind_push_table` over every form** (`step121_wait_planner.rs`): same-shape, row, column, the
  decode row form (`rows1`), staged, and the fused kinds at `0x1f0`+ (the loop stopped at `BOOL_TO_I32`
  before, so `SILU_MUL`, `SUB_EXP` and `SCALE_MASK_FILL` were never in it). Model figures below.
- [x] **Profiled on card 0** (the 1.7B, `TT_TILES=all`, 8,192 positions, a 1,000-token prompt in chunks of 64;
  runs `1792956684` off, `1792957400` with `TT_TTINSN=1`; the profile-mode spans below are device time). **Nothing
  in the 1.7B pushes at the runner's rate any more.** The math role's pushing runs at 5.6 cycles a word in
  `silu_mul` and 7.3 in RMSNorm (decode), 12-16 in the softmax, ~40-500 in the products (backend-bound): the
  staging of `SUB_EXP` and `DIV` (2026-10-25) took what was push-bound. What is left gains a little:

  | Replay (profile mode, covered ops) | Off | Fused | |
  |---|--:|--:|--:|
  | `Chunk { c: 64, len: 64 }` | 5,388.8 us | 5,248.6 | -2.6% |
  | `Chunk { c: 64, len: 512 }` | 24,606.2 | 23,956.2 | -2.6% |
  | `Step(1024)` (decode) | 17,434.7 | 17,351.0 | -0.5% |
  | of it: RMSNorm (decode, 36 ops) | 417.5 | 375.6 | -10% |
  | of it: `silu_mul` (decode, 17 ops) | 283.8 | 241.2 | -15% |

  The benchmarks (host-timed, below) decide the default. ViT's precise `GELU` (11,495 words a tile, pushed at the
  runner's rate in the experiment) is the clearest push-bound kernel left in a model.

### Phase 1: the shared emitter and validator

- [x] `tt_isa::ttinsn` (`no_std`): `emit`, `validate`, `execute`, the encoders (`addi`, `bne`, `beq`,
  `jump`, `RET`), `FUSED`. Unit tests: the encoders against the assembler's words; the round trip over
  nested, adjacent and far loops; every refusal (no form, not a table word, three deep, a count of 0 or
  2048, overlapping repeats, no room); **14 mutations of good code each refused** (a stray `lw`, a
  non-table word, a branch one word short or long or on the other counter, a foreign register, the inner
  loop on `t0`, a decrement by two, a count of zero, a missing or mid-code return, an unclosed loop, a third
  level, a far close where a branch reaches) and four of a far close.
- [x] `code::Code::fused` with the host round trip; `every_sfpu_kernel_fuses`: every SFPU kind's three-role
  kernel (same-shape, row, column, the decode form; 1 and 3 tiles) fuses and pushes its expansion.
- [x] The experiment's translator uses the shared encoders (`tt_firmware::ttinsn`).

### Phase 2: host expansion and placement

- [x] `FUSED` in a program's length word (`dataflow::program` refuses it with `LOOPED`; `LENGTH_MASK`
  strips it), `Session::set_ttinsn` / `TT_TTINSN=1`; `stored_program` fuses the math role's program when
  it pushes `FUSE_MIN_WORDS` (256) words or more; memoised per form; a capture keeps the form it was
  captured with (its scripts name the fused programs, and a replay restores them as any other).

### Phase 3: the runner

- [x] `tt_firmware::fused` (shipped): `invalidate_icache`, `call`. `run_stream` invalidates once a run
  before its first fused batch; `push_body` and `run_once` call fused code. T0 and T2 refuse `FUSED`
  (`dm::error::PROGRAM` / the runner's panic), as does a looped, streamed-in-the-descriptor or guarded run.
  T1's image 7,696 of 10,240 bytes. Firmware lints clean with and without `profile`, `encoded-guard`,
  `ttinsn-experiment`.
- [ ] **The experiment image does not link** (`ttinsn_t1`): its `.text` alone overflows T1's 10 KiB slot by
  216 bytes, before its baked programs (the slot shrank from 16 KiB in `36ea1dc`, 2026-10-14). The shipped
  path replaces its baked and translated modes; slim it (drop the baked programs and the translator) or retire it.

### Phase 4: gates, measurement, rollout

- [x] **Gate `step231_ttinsn_push`** (ttsim and card 0, run `1792956507`): 14 ops (precise `GELU`,
  `SIGMOID`, `TANH`, `ERF`, `EXP`, `LOG`, `SQRT`, `SILU`, `SILU_MUL`; `DIV` same-shape, row and column;
  `SUB_EXP` row and column) over signed zeros, subnormals, NaNs, infinities and `f32::MAX`, fused against
  pushed **bit for bit**, the fused path seen taken; a capture with fused programs replays as the ops run
  fresh; **a fused program run from the address another fused program ran from** (the cache forgotten
  between them) computes itself, not the other (on silicon, where the instruction cache is real).
  Negative control: fused code with its `SFPLOADI` immediates changed after the host's checks gives other
  bits (5,216 of 6,144 differ), so the card ran the fused code. (Changing only the first `SFPLOADI` changed
  nothing: precise `GELU`'s first constant reaches none of the outputs.)
- [x] **Model gates with `TT_TTINSN=1`**, card 0, run `1792958143`, 14 of 14: `step188` (the 1.7B against
  burn-flex, at the native 8K context too, and traced equal to eager), `step166` (ViT), `step165` (ResNet),
  `step209`, `step230`, `step231`. On ttsim: `step12_mnist` 8 of 8 against its goldens; Burn's conformance
  (`target/conformance/1792958174`) 2,973 of 3,025, the 52 listed failures and no other.
- [x] **Benchmarks, off and on** (card 0, release, medians; [log](../learnings/performance-log.md#fused-ttinsn-push-from-the-host-the-default-2026-10-25-card-0)):
  nothing slower. The 1.7B's prefill 1.6-3.9% faster (7,992 tokens at chunks of 512 9,999 -> 9,842 ms,
  `1792961449` / `1792962530`), its decode 0.3-1.7% (16.83 / 18.28 -> 16.78 / 17.97 ms, `1792963641` /
  `1792963811`), ViT-Tiny 8.97 -> 8.32 ms (-7.2%), ResNet-18 13.88 -> 13.74 (`1792963983` / `1792963996`).
- [x] **Rollout:** the default (`TT_TTINSN=0` off); `AGENTS.md` records the contract (the shipped runner calls
  host-made, host-validated `.ttinsn` code in the program cache after invalidating its instruction cache;
  `tt_isa::ttinsn::validate` is that code's gate); `step231` in `--smoke`. `--smoke` on card 0 with the
  default: 1,210 of 1,210 (`1792964273`). Mover gates `step172`/`step173` in the safe, fast and guarded
  modes on ttsim (the mover's stamps changed).

## Model figures (`per_kind_push_table`, 2026-10-25)

`2.8 * pushed / backend` with the backend at one instruction a cycle: a ratio of 2.8 is a body pushed word for
word (past the replay buffer), below 1 a body that replays. What the models run:

| Kind | Form | Stored | Pushed a tile | Ratio |
|---|---|--:|--:|--:|
| `SILU_MUL` (the 1.7B's MLP) | same-shape | 130 | 3,879 | 2.80 |
| `SILU_MUL` | decode row | 487 | 487 | 2.80 |
| `SILU` | same-shape | 128 | 3,815 | 2.80 |
| `GELU` (ViT, precise) | same-shape | 368 | 11,495 | 2.80 |
| `SUB_EXP` | column (attention) | 285 | 285 | 0.36 (staged) |
| `SUB_EXP` | row | 1,827 | 1,827 | 2.80 |
| `DIV` | column (norms, softmax) | 330 | 330 | 0.38 (staged) |
| `DIV` | row | 1,990 | 1,990 | 2.80 |
| `DIV` | decode row | 254 | 254 | 2.80 |
| `SCALE_MASK_FILL` | same-shape | 49 | 49 | 0.52 |

Every long transcendental (`TAN`, `ASINH`, `SIN`, `COS`, `ERF`, `LOG1P`, `POW`, ...) sits at 2.80.
