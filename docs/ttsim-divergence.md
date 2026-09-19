# Simulator divergence log

Every place the simulator and the specification disagree, or where the simulator
declines to model something. Each entry is a finding: either ttsim, the documentation,
or our understanding is wrong, and all three are worth knowing about.

The log exists from the first commit deliberately. There is no silicon on this machine,
so the "silicon" column is empty everywhere — but the file and the habit need to predate
the pressure to skip them.

## Not modelled by ttsim

These raise `UnimplementedFunctionality` or `UnsupportedFunctionality`, which terminates
the process. Each one that we hit is either routed around or deferred to a silicon gate.

| # | What | ttsim says | Consequence here |
|--:|---|---|---|
| 1 | `libttsim_tile_rd_bytes` / `wr_bytes` | `UnsupportedFunctionality` on Blackhole | There is **no backdoor into tile memory**. Everything goes through PCIe TLB windows, which puts the TLB layer on the critical path for every later step rather than making it an optimisation. |
| 2 | `NOC_ENDPOINT_ID` (`+0x048`) | `base_noc_regs_rd32: UnimplementedFunctionality` | Tile discovery by probe — what `ethdump.c:462` does — is impossible. The grid topology was measured instead, by memory size; see `tt_isa::noc::grid`. |
| 3 | `pc` snapshot registers (`0xFFB1_3138`+) | `t_tile_mmio_rd32: UnimplementedFunctionality` | The step 3 gate cannot cross-check *where* a core is executing, only that it is. `pc_snapshot_lands_in_the_loaded_image` is `#[cfg(feature = "silicon")]`. |
| 4 | TLB `strided[32]` array (`0x1FC0_09D8`) | Not decoded; the config region stops at the end of `windows[210]` | Non-rectangular multicast cannot be configured on the simulator at all. Rejected in `tt_ttsim::transport` with a message naming the cause. |
| 5 | TLB config word 2, bits outside `0xE7` | `UnimplementedFunctionality` | A window must target **NoC #0**, and may not set `linked`, `static_vc`, `static_vc_buddy` or `static_vc_class`. `linked` is never safe from the host anyway; the NoC #1 restriction is real and unremarked in the docs. |
| 6 | PCI config space at or beyond `0x40` | `UnimplementedFunctionality` | There is no capability list to walk (`CapabilitiesPointer` reads 0). `ConfigOffset` is a closed enum so the obvious "dump config space" routine cannot be written. |
| 7 | `SFPLOADMACRO` | `tensix_sfploadmacro: explicitly out of scope` | **Now watched, not quoted.** `step5_corpus.rs::instructions_ttsim_declines_to_execute` pushes one and asserts the child dies, with a control program of the same shape that survives. Excluded from the simulator corpus; belongs in the silicon-only suite. |
| 12 | RISC-V reads of `Dst`, from any core but **T1** | `tensix_dst_rd32: pipe=N` — its handler asserts `pipe == 1` | The step 4 firmware runs on T1, not T0. The image is identical; `load_and_start` points the core's reset-PC override at it. A hardware constraint this is not. |
| 13 | `RISC_DEST_ACCESS_CTRL_SEC1.fmt` outside {0, 2, 3}, or `unsigned_int` set | `UnimplementedFunctionality` | `fmt = 0` (FP32) is the reset default and what the baseline uses, so this has not bitten. |
| 14 | `pause` (Zihintpause), which shares an encoding with `fence w, 0` | `rv32_fence: fence_mode=0x10` | `core::hint::spin_loop()` emits it. `tt_firmware::spin` uses an empty asm block instead, and the instruction-set gate rejects the encoding — Blackhole does not list Zihintpause among its extensions either, so this is very likely faithful. |
| 15 | `fence` orderings outside {`iorw,iorw`, `rw,rw`, `rw,w`, `r,rw`} | `rv32_fence: fence_mode=0x...` | The gate in `crates/tt-tests/build.rs` allows only those four. |
| 16 | Writes to `SOFT_RESET_0` setting any bit outside `0x4_7800` | `UnimplementedFunctionality` | Only the five baby RISC-V bits exist as far as ttsim is concerned. The backend reset bits can be neither set nor meaningfully cleared. |
| 19 | RISC-V reads of `ThreadConfig` | `tensix_cfg_rd32: reg=N` | ttsim maps the entire `TENSIX_CFG_BASE` aperture to a flat `Config` array — bank hardcoded to zero, no `ThreadConfig` region — so any address past `Config` falls off its register switch. `tt_firmware::cfg::active_bank` therefore cannot run there, and firmware that must work on the simulator assumes bank 0 and says why. |
| 20 | `Config` bank 1 through the RISC-V aperture | `tensix_cfg_rd32: reg=N` | Same cause: the decode passes a hardcoded bank of 0 and derives the register index from the raw byte offset, so a bank-1 address is read as a register index past the end of bank 0. |
| 21 | `Config` registers ttsim has not modelled individually | `tensix_cfg_rd32`/`wr32: reg=N` | The implementation is a `switch` over specific registers rather than a backing array, topping out around register 220. A legitimate `Config` word that no workload has exercised is fatal rather than merely uninitialised. |
| 17 | `SFPMUL` (opcode `0x86`) with `Mod1 > 1` | `tensix_sfpmul: instr_mod1=2` | Its `SFPMAD` handler (opcode `0x84`) accepts `Mod1 <= 3`, and the two are documented as the same instruction, so `tt_isa::sfpu` emits the `SFPMAD` spelling for any modified multiply. |

## Behaviour worth knowing, not strictly divergence

| # | What | Note |
|--:|---|---|
| 8 | DRAM tiles fault on Tensix NIU addresses | A DRAM tile's NIU is not at `0xFFB2_0000`, so a blind grid walk reading that address dies at the first DRAM column. This is most likely faithful, not a simulator artefact — it is why `ethdump.c` scans only the row it already knows is Ethernet. Our scans fork per coordinate. |
| 9 | BAR bases are compile-time constants | `libttsim_init` pre-programs config space to match them, and config-space BAR *writes are accepted but change nothing*. Relocating a BAR the way a BIOS would makes config space lie while decoding continues against the constants. `Simulator::verify` refuses to start if the two disagree. |
| 10 | Every contract violation calls `_Exit` | No return code, no unwinding, no destructors, no panic hook. This is why `tt_ttsim::transport` validates before calling and why tests run inside `fork_scope`. `crates/tt-ttsim/tests/fatality.rs` pins five cases as genuinely fatal, so the validation layer cannot quietly become unnecessary. |
| 18 | Tensix backend soft reset is not modelled | `SOFT_RESET_0` comes up at `0x0004_7800` — only the RISC-V bits — and ttsim's handler acts on those alone. The Vector Unit is therefore always released, and `release_tensix_backend` is a no-op there. It is kept because it is required on silicon, where holding bit 10 means SFPU instructions "might or might not be silently discarded". The negative control that checks this is `#[cfg(feature = "silicon")]`. |
| 11 | Timing is not modelled | ttsim claims bit-exactness, not cycle accuracy, and says operations may be evaluated in any order software synchronisation permits. No performance number may come from here. |

## Measured, not quoted

Facts we established empirically because the documentation does not state them for
Blackhole, or ttsim does not expose what the documentation says to read. Each needs
re-deriving at the first silicon gate.

| # | Fact | How | Cross-check |
|--:|---|---|---|
| A | Tensix tiles occupy raw NoC #0 `x ∈ {1..7, 10..16}`, `y ∈ 2..11` | Bisected the highest writable address at each coordinate (`crates/tt-tests/tests/probe_niu.rs`) | Gives 14 × 10 = 140 tiles, exactly the documented Tensix count; and 1536 KiB / 512 KiB / `0xFF000000` match the documented Tensix L1, Ethernet L1 and DRAM channel sizes |
| B | Row `y = 1` is Ethernet, columns `x ∈ {0, 9}` are DRAM, column `x = 8` is neither | Same measurement | `ethdump.c:462` scans `y = 1` for Ethernet and skips `x = 8, 9` |

## Numerics worth knowing

| # | Behaviour | Note |
|--:|---|---|
| C | The SFPU multiply drops the sign of a zero result unless both operand exponents are non-zero and sum into range | So `-1.0 * 0.0` gives `+0.0` where IEEE754 gives `-0.0`. `SFPMUL.md` names the fix: set `SFPMAD_MOD1_NEGATE_VC` so the addend is `-0` — the identity element of floating-point addition — rather than `+0`. `tt_isa::sfpu::mul` does this by default, and with it every case in `it_computes_rather_than_returning_a_constant` matches the host bit-for-bit. Found by differential testing, not by reading: the doc note is easy to skim past. |
| D | Denormal operands and results are flushed to zero, and every NaN is canonicalised to `0x7FC0_0000` | Visible in ttsim's `sfpu_mul`. Not yet exercised by a gate — it is the obvious next differential test. |
