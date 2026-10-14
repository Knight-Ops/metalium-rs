# tt-metal DRAM read bandwidth (development only)

The counterpart, on tt-metal, of `gddr_aggregate`'s Read mix: N cores each read their share of one DRAM buffer into
their own L1, `--batch` requests per barrier, timed by the tile wall clock from inside the kernel, every iteration's
bytes checked against a poisoned window. Not part of any build or suite; it needs a built tt-metal checkout
(`/home/carl/projects/tt-metal-v077`, `build_Release`).

```sh
crates/tt-tests/bench/tt_metal_dram_read/build.sh           # clang++ against build_Release, flags from `ninja -t commands`
CORES=1,2,4,8,16,32,64,110 crates/tt-tests/bench/tt_metal_dram_read/run_bw.sh out.log \
    --page 4096 --batch 8 --noc 0 [--pinned 1] --iters 7
```

`--page` bytes per page (2048 and 4096 are BF16 and F32 tiles), `--batch` requests per barrier, `--noc 0|1` (BRISC on
NoC0, or NCRISC on NoC1, tt-metal's default reader), `--pinned 1` core i reads DRAM bank i mod 8 directly instead of the
interleaved accessor. `run_bw.sh` sets `TT_VISIBLE_DEVICES=1` (card 1 only: tt-metal's cluster start touches every card
it can see) and fast dispatch, which reserves cores: the grid is 11x10 = 110 here, not 120.

**Before the first run after this repo's gates, reset the board** (`tt-smi -r <PCI address>`): our Ethernet image stays
on the E1 cores across processes and tt-metal times out returning them to base firmware
(`silicon-operating-notes.md`, "Running tt-metal after this repo's gates"). Results and what they mean:
[performance-log.md](../../../../docs/learnings/performance-log.md#tt-metal-reference-dram-reads-by-core-count-2026-10-13-card-1).
