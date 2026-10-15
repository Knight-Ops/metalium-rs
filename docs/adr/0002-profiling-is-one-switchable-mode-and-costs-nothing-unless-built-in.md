---
number: 2
title: "Profiling is one switchable mode, and costs nothing unless built in"
status: "accepted"
affected_paths:
  - "crates/tt-kernels/src/profile_mode.rs"
  - "crates/tt-kernels/src/session.rs"
  - "crates/burn-tt/src/profile.rs"
  - "crates/tt-firmware/src/mover.rs"
  - "crates/tt-firmware/src/corpus.rs"
  - "crates/tt-isa/src/dm.rs"
  - "crates/tt-firmware-images/build.rs"
  - "xtask/src/silicon.rs"
  - "crates/tt-tests/tests/silicon_profile_mode.rs"
---

# 2. Profiling is one switchable mode, and costs nothing unless built in

## Context
Per-op device timing existed only as custom analysis inside one benchmark (smollm2_replay_profile) and a probe, worked only on a model cut to one layer, and took 128 KiB from the program cache. Host counters were scattered across several reports. Meanwhile the data movers and role runners carried stamping code in every image: compiled in but switched off, it cost 0.46 ms (1.8%) of a SmolLM2-135M decode step (card 1, runs 1792088908 and 1792089070). The owner directed on 2026-10-15 that profiling be a mode one enables rather than custom profiling per workload, and that tracing cost nothing unless needed, as a feature flag.

## Decision
There is one profile mode: TT_PROFILE=<dir> (or cargo xtask bench|silicon --profile, or burn_tt::profile::profile_mode) attributes every eager op and every traced op to its Burn op or op part, split into barrier, kernels, control and data, with trace names, replay splits and the host's counters, written as summary.md, ops.tsv and trace.json (tt_kernels::profile_mode, burn_tt::profile). Long traces are covered by a rotating stamp window per replay (dm::TRACE_WINDOW), never by repeating or extending a replay. Stamping is compiled into the firmware only with its profile feature (tt-firmware, tt-firmware-images, tt-tests/profile, burn-tt/profile; reported as dm::caps::STAMPS): default images contain no stamping code and no check of whether to stamp, and the host refuses to profile without it by name. Profiling never fails a run: a stamp overflow is reported in the profile, not raised.

## Consequences
New diagnostics extend the profile mode instead of adding per-benchmark analysis. Any diagnostic hook added to firmware or a hot host path is gated by a build feature or a const that compiles it out, and its off-cost is measured. A firmware change is linted and gated with and without the profile feature (step172/step173 in every read mode, silicon_profile_mode on silicon). Tests that need stamps compile only with the feature. Performance numbers are always taken on default (non-profile) builds.
