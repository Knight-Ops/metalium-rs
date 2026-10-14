# New op: definition of done

What a new kernel or op must have before it is done. This is the standing contract that
`AGENTS.md` points at ("Follow the coverage definition of done"). It was extracted from the
Phase 10 tracker, now archived as [hardware-coverage.md](../completed-plans/hardware-coverage.md);
item ids cited below and in source comments (X6, S6, F0, ...) refer to that file. Open work is
in [roadmap.md](roadmap.md); the generated Burn status table is
[burn-op-coverage.md](burn-op-coverage.md), with per-method notes in
[burn-op-notes.md](burn-op-notes.md).

How to get from a failing Burn conformance test to an op that meets this contract, step by step, is
[conformance-to-feature.md](conformance-to-feature.md).

The two-gate rule, applied per op. An item is `[x]` only when every line below holds for
it, and a line that does not apply says why in the item.

1. **A `tt-isa` helper**, typed so that the hazards are in the API (the Cross-cutting rules in [the Phase 0-10 checklist](../completed-plans/phases-0-10-checklist.md#cross-cutting)),
   with unit tests of its encoding.
2. **An oracle that is not an epsilon** (the [Tolerance policy](../completed-plans/phases-0-10-checklist.md#tolerance-policy)):
   - *bit-exact* wherever the hardware's arithmetic is documented: `numerics::fma_bh` for
     anything `SFPMAD`-shaped, and a port of the page's functional model for `SFPLUT`,
     `SFPLUTFP32`, `SFPARECIP`, `SFPEXEXP`/`SFPSETEXP`/`SFPEXMAN`, `SFPSTOCHRND`,
     `SFPCAST` -- each port checked against the page's pseudocode the way `fma_oracle`
     checks `fma_bh` against `fma.c`;
   - for an *approximation* (a polynomial, a Newton step, a range reduction), the device
     result is still predicted bit for bit by running the same instruction sequence through
     the ported models, **and** its distance from the true function is held to a bound
     *derived* in a comment next to the gate (polynomial remainder, Newton's quadratic
     convergence, the reduction's error), never a guessed number.
3. **A ttsim gate** in `crates/tt-tests/tests/stepNN_*.rs`, inside `fork_scope`,
   watched failing at least once (an empty kernel, a wrong constant, a swapped operand).
4. **A silicon gate**, the same test through `cargo xtask silicon`. Ordinary
   validation uses one card (card 0): the same test on the other card gives no new
   information. Mesh or multi-card ops, and models too large for one card, use both.
5. **Burn routing**: the method overridden in `burn-tt/src/ops.rs` and listed in
   `OVERRIDDEN` (then `cargo xtask gen-burn-ops` and its `--check` mode), agreeing with `burn-flex` bit for
   bit or within the item's derived bound; a residency check that the op downloads
   nothing (`tensor_traffic`, `TT_TRACE_FALLBACK=1` silent); and an entry in the silicon
   smoke tier (`xtask/src/silicon.rs`, `SMOKE`), where burn-tt is compared with
   burn-flex on the card.
6. **Findings logged**: every ttsim refusal or ttsim/silicon disagreement met on the way
   is a row in [`ttsim-divergence.md`](../learnings/ttsim-divergence.md), and every measured fact a ["Measured, not quoted"](../learnings/ttsim-divergence.md#measured-not-quoted)
   entry.
7. **Padding declared** (F0 in [the archived tracker](../completed-plans/hardware-coverage.md), from [`tt-metal-concepts-review.md`](../learnings/tt-metal-concepts-review.md) G1): the op states the pad
   it needs from each input and the pad it leaves (`OpPadding`), and a ragged-shape gate
   chains it into an accumulation.

An op that works only on silicon (ttsim refuses an instruction it needs) can be ticked
with the simulator line `[-]` and the divergence row cited, as `DOTPV` is today.

## Cross-check against SFPI (development only)

SFPI is Tenstorrent's GCC toolchain (`riscv-tt-elf-g++`, SFPI releases at
`github.com/tenstorrent/sfpi`). It compiles C++ kernels written with `sfpi::vFloat` and friends into SFPU
instructions, and it is what tt-metal's LLK kernels are built with. We do not use it to build anything:
our SFPU programs are built on the host from typed encoders (`tt_kernels::sfpu`, `tt_isa::isa`), and
firmware is Rust. It is useful as an **independent check of what we encode**, in the same class as the C/C++
specification models: a development-only oracle, never a dependency of a shipped crate.

**What it can tell you**
- Whether an instruction word we encode is the word Tenstorrent's compiler emits for Blackhole
  (opcode, field positions, field widths). This is the check that matters for `UNVERIFIED` and
  Wormhole-only layouts, which `AGENTS.md` says never establish Blackhole behaviour.
- How many instructions an equivalent kernel needs (for comparing the size of our programs; the roles are
  push-bound, so words pushed is time).

**What it cannot tell you**
- Results. The compiler may schedule, reorder, fuse and allocate registers differently, so a program that
  differs from ours is not wrong. Numerical oracles stay the specification models in point 2 above.
- Silicon behaviour. Agreement with the compiler is evidence for an encoding, not a promotion: an
  `UNVERIFIED` encoding is still promoted by the rule in `AGENTS.md` (`Bits32_BH.lua`, evidence in
  `measured.rs`, a gate).

**Recipe** (checked 2026-10-13 with the SFPI 7.83.0 at `/opt/tenstorrent/sfpi`, which is not pinned):

```cpp
// k.cc -- the stub stands in for tt-metal's ckernel headers, which SFPI's headers expect
#include <cstdint>
namespace ckernel { extern volatile std::uint32_t instrn_buffer[]; }
#include <sfpi.h>
using namespace sfpi;
void square_plus_one() {
    for (int i = 0; i < 8; i++) { vFloat x = dst_reg[0]; dst_reg[0] = x * x + 1.0f; dst_reg++; }
}
```

```sh
SFPI=/opt/tenstorrent/sfpi
$SFPI/compiler/bin/riscv-tt-elf-g++ -mcpu=tt-bh-tensix -O2 -std=c++17 -fno-exceptions -fno-rtti \
    -I$SFPI/include -c -o k.o k.cc
$SFPI/compiler/bin/riscv-tt-elf-objdump -d k.o
```

Use `-mcpu=tt-bh-tensix` (compute) or `tt-bh` (data movement), as tt-metal's build does. The Tensix
instructions come out as immediate words in the instruction stream (the `.ttinsn` form): a word is the
`TT_OP` instruction **rotated left two bits**, so rotate right two bits before decoding it against
`tt_isa::isa` (`ttinsn_word` in `crates/tt-isa/src/isa/mod.rs` is the inverse). Compare opcode and each
field with the generated definition of that instruction; do not hand-edit the generated tables.

**Worked example** (the kernel above; the four instructions in its loop):

| Compiler word | After rotating right 2 | Ours |
|---|---|---|
| `sfpload L0,0,0,7` `0xc0038001` | `0x7000e000` | opcode `0x70`, `AddrMod` = 7 in the Blackhole 3-bit field at bit 13 (the Wormhole 2-bit field cannot hold 7) |
| `sfpmad L0,L0,L0,L10,0` `0x10002802` | `0x84000a00` | opcode `0x84`, `VA`=0, `VB`=0, `VC`=10, `VD`=0, `Mod1`=0 |
| `sfpstore L0,0,0,7` `0xc8038001` | `0x7200e000` | opcode `0x72`, same fields as `SFPLOAD` |
| `ttincrwc 0,2,0,0` `0xe0020000` | `0x38008000` | opcode `0x38`, `DstInc` = 2 at bit 14: agrees with our `INCRWC` layout, which is `UNVERIFIED` (a Wormhole page) for every other field |

**Constraints**
- **Pin before relying on it.** `PINS.toml` states that an oracle that can change without the pin noticing is
  not an oracle. The SFPI under `/opt/tenstorrent/sfpi` is whatever the machine has; before any gate or CI
  step runs it, pin the release and hash there and add a `cargo xtask fetch-*` task for it as the other
  inputs have. Until then, record the compiler version (`riscv-tt-elf-g++ --version`) in any note that
  quotes a result.
- **Compiler and headers must come from the same release.** ttnn 0.72's bundled LLK headers (SFPI 7.52
  era) fail to compile with the 7.83.0 compiler (`sfpi::reinterpret` deleted, `RoundMode::NearestEven`
  missing). Use the `include/` of the SFPI you run, not tt-metal's.
- **Not a build dependency.** The shippable-crate rule applies: no shipped crate may need this toolchain to
  build. (`cargo xtask check-no-sim-in-ship` checks the simulator binding, not this; nothing enforces it
  mechanically yet.)
- **Not exercised by any gate yet.** The table above is a one-off check; turning it into a test means a
  pinned toolchain, a fetch task and a differential test in `tt-tests` (compare each emitted word against
  `tt_isa::isa` decoding), and that is open work.
