//! `cargo xtask silicon`: run the silicon suite one test per process, with a log
//! that survives the host dying.
//!
//! Every rule in here was paid for during Phase 1's bring-up
//! (`docs/learnings/silicon-operating-notes.md`):
//!
//! * **One test per process, one thread.** All gates share one physical card, and
//!   window allocation is global card state. Running them one at a time is also
//!   what identified the failing access when the host went down — the output of a
//!   whole-suite run died with the session.
//! * **A log `fsync`'d after every line, carrying `boot_id`.** A hard kill loses
//!   the journal's last minutes and unflushed file data, so the last `START` line
//!   without a matching `END` names the test that took the machine down, and a
//!   changed `boot_id` on the next run makes a reboot evidence rather than
//!   inference.
//! * **Stop at the first failure** unless told otherwise. A failing silicon gate
//!   may have left the card in a state the next gate should not inherit.
//!
//! The test binaries are built once and run directly, not through `cargo test`,
//! so a run does not rebuild between tests and the per-test timeout measures the
//! test rather than cargo.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::testrun::{build, list, run_one, Built, Verdict};
use crate::util::workspace_root;

struct Opts {
    devices: Vec<u16>,
    filters: Vec<String>,
    include_ignored: bool,
    keep_going: bool,
    list_only: bool,
    release: bool,
    /// Extra `tt-tests` features for the build, beyond `silicon`.
    features: Vec<String>,
    timeout: Duration,
}

fn parse(args: impl Iterator<Item = String>) -> Result<Opts, String> {
    let mut o = Opts {
        devices: vec![0],
        filters: Vec::new(),
        include_ignored: false,
        keep_going: false,
        list_only: false,
        release: false,
        features: Vec::new(),
        timeout: Duration::from_secs(120),
    };
    let mut smoke = false;
    let mut args = args.peekable();
    while let Some(a) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("`{name}` needs a value\n\n{USAGE}"))
        };
        match a.as_str() {
            "--device" => {
                let v = value("--device")?;
                o.devices = if v == "all" {
                    all_devices()?
                } else {
                    vec![v
                        .parse()
                        .map_err(|_| format!("`--device {v}`: expected a number or `all`"))?]
                };
            }
            "--filter" => o.filters.push(value("--filter")?),
            "--feature" => o.features.push(value("--feature")?),
            "--timeout-secs" => {
                let v = value("--timeout-secs")?;
                o.timeout = Duration::from_secs(
                    v.parse()
                        .map_err(|_| format!("`--timeout-secs {v}`: expected a number"))?,
                );
            }
            "--smoke" => smoke = true,
            "--include-ignored" => o.include_ignored = true,
            "--keep-going" => o.keep_going = true,
            "--list" => o.list_only = true,
            "--release" => o.release = true,
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown option `{other}`\n\n{USAGE}")),
        }
    }
    if smoke {
        o.filters.extend(SMOKE.iter().map(|s| s.to_string()));
    }
    Ok(o)
}

/// What `--smoke` runs, in order: the device backend against `burn-flex`,
/// cheapest claim first, so a broken op fails before a training run spends a
/// minute finding it. Simulator validation is plain `cargo test`; this is the
/// tier that says burn-tt still computes what Burn's CPU backend computes, on
/// the hardware it is for.
pub const SMOKE: &[&str] = &[
    "step19_eltwise::",
    "step27_burn_eltwise::",
    "step32_burn_softmax::",
    "step35_burn_rank_n::",
    "step47_burn_activations::",
    "step11_burn::",
    // Burn's own backend tests (`crates/burn-tt-conformance`, B11): whole upstream
    // modules, each a single-op claim checked against literals Burn's CPU backend
    // meets. Only modules with no entry in `expected-failures.toml`, so every test
    // matched here is meant to pass; on card 0, since they use `TtDevice::default()`.
    // Upstream's own `#[ignore]`d tests are not run (the runner does not list them).
    "tensor::float::ops::matmul::",
    "tensor::float::ops::add::",
    "tensor::float::ops::sub::",
    "tensor::float::ops::mul::",
    "tensor::float::ops::div::",
    "tensor::float::ops::exp::",
    "tensor::float::ops::neg::",
    "tensor::float::activation::relu::",
    "tensor::float::ops::aggregation::",
    "tensor::float::ops::transpose::",
    "tensor::float::ops::permute::",
    "tensor::float::ops::reshape::",
    "tensor::float::ops::expand::",
    "tensor::float::ops::slice_assign::",
    "tensor::int::ops::add::",
    "tensor::int::ops::matmul::",
    "tensor::int::ops::slice::",
    "tensor::multi_threads::",
    "autodiff::base::matmul::",
    "autodiff::base::relu::",
    "autodiff::base::add::",
    "autodiff::base::slice::",
    // The modules the first conformance round made pass (docs/plans/conformance-to-feature.md).
    "tensor::float::activation::softmax::",
    "tensor::float::ops::mask::",
    "tensor::float::ops::powf::",
    "tensor::float::ops::fmod::",
    "tensor::float::ops::cat::",
    "tensor::int::ops::cat::",
    "tensor::bool::ops::cat::",
    "tensor::bool::ops::argwhere_nonzero::",
    "tensor::bool::ops::reshape::",
    "tensor::int::ops::reshape::",
    "tensor::int::ops::stack::",
    "tensor::float::ops::one_hot::",
    "autodiff::base::nonzero::",
    "autodiff::base::reshape::",
    "tensor::float::module::nearest_interpolate::",
    "tensor::float::module::bilinear_interpolate::",
    "tensor::float::module::bicubic_interpolate::",
    "tensor::float::module::lanczos3_interpolate::",
    "autodiff::base::nearest_interpolate::",
    "tensor::float::ops::gather_scatter_nd::",
    "tensor::int::ops::gather_scatter_nd::",
    "autodiff::base::gather_scatter_nd::",
    // The gates written for them: exact oracles, exact traffic, each watched failing.
    "step149_softmax_nd::",
    "step150_int_bool_reshape::",
    "step151_broadcast_mask_pow::",
    "step152_empty_tensors::",
    "step153_argwhere::",
    "step154_conv_transpose_padding_out::",
    "step155_interpolate::",
    "step156_scatter_nd_reduce::",
    "step157_conv3d::",
    "step158_matmul_accumulation::",
    "step159_exact_matmul::",
    "step160_deform_conv2d::",
    "step161_fusion_scalar_guard::",
    "step162_f16_arithmetic::",
    // What that gate made pass in Burn's own suite.
    // `test_conv3d_complex` and `_groups` are `precision` entries; the rest of the module passes.
    "tensor::float::module::conv3d::test_conv3d_simple",
    "tensor::float::module::conv_transpose3d::",
    "autodiff::base::conv3d::",
    "autodiff::base::conv_transpose3d::",
    "tensor::float::module::deform_conv2d::",
    "autodiff::base::deform_conv2d::",
    // The same suites with a BF16 float element (D2), silicon only: ttsim refuses the BF16
    // narrowing. Modules with no entry in `expected-failures-silicon.toml`.
    "tensor_bf16::bf16::ops::add::",
    "tensor_bf16::bf16::ops::mul::",
    "tensor_bf16::bf16::ops::exp::",
    "tensor_bf16::bf16::ops::cat::",
    "tensor_bf16::bf16::ops::gather_scatter::",
    "tensor_bf16::bf16::ops::sort_argsort::",
    "tensor_bf16::bf16::ops::grid_sample::",
    "tensor_bf16::bf16::module::avgpool2d::",
    "tensor_bf16::bf16::module::maxpool2d::",
    "tensor_bf16::bf16::module::conv_transpose2d::",
    "autodiff_bf16::base::add::",
    "autodiff_bf16::base::mul::",
    "autodiff_bf16::base::matmul::",
    "step164_grid_sample_bf16::",
    "step20_many_tiles::eltwise",
    "step20_many_tiles::column_sums",
    "silicon_card_lock::a_second",
    "silicon_lazy_attach::",
    "step12_mnist::the_first_forward_pass",
    "step12_mnist::the_mlp_trains_on_a_reduced_dataset",
    "step12_mnist::the_mlp_trains_on_four_tiles",
    "step59_burn_transformer::",
    "step60_batched_blocks::",
    "step61_burn_gather::",
    "step62_gather_rows::",
    "step63_burn_full_reduce::",
    "step66_burn_small_ops::",
    "step67_general_reduce::",
    "step68_k_block_matmul::",
    "step69_reduction_primitives::",
    "step70_native_norms::",
    "step71_integer_alu::",
    "step72_round_cast::",
    "step73_fpu_transpose::",
    "step74_bf16_storage::",
    "step75_fpu_pooling::",
    "step76_burn_bf16::",
    "step77_bf16_matmul::",
    "step78_burn_pooling::",
    "step79_extremum_scans::",
    "step80_integer_reductions::",
    "step81_hardware_rounding::",
    "step82_integer_division::",
    "step83_attention::",
    "step84_slice_assignment::",
    "step85_convolution::",
    "step86_resident_indices::",
    "step87_src_transpose::",
    "step88_mesh_module_reference::",
    "step89_mnist_cnn::",
    "step90_matrix_eltwise::",
    "step91_seeded_prng::",
    "step92_bfp_formats::",
    "step93_exponent_history::",
    "step94_bfp_storage::",
    "step95_burn_bfp::",
    "step96_mixed_bfp_mnist::",
    "step97_matrix_register_moves::",
    "step98_adc_copy::",
    "step99_adc_planes::",
    "step100_scalar_config::",
    "step101_l1_movement::",
    "step102_xmov_tensor::",
    "step103_source_banks::",
    "step104_unpacker_handover::",
    "step130_float_extremum_scans::",
    "step131_int_scans::",
    "step132_int_arg::",
    "step125_int_bool_views::",
    "step126_int_mask_abs::",
    "step127_nd_indexing::",
    "step128_cross_int_matmul::",
    "step129_prelu_residency::",
    "step133_remainder::",
    "step134_burn_remainder::",
    "step105_mutex::probe_uncontended_acquire_release_survives",
    "step105_mutex::uncontended_t",
    "step105_mutex::state_does_not_leak_from_one_run_into_the_next",
    "step105_mutex::silicon_release_of_unheld_and_reacquire_complete_without_effect",
    "step105_mutex::contended_handoff_follows_round_robin_for_every_mutex_and_holder",
    "step105_mutex::control_an_unlocked_waiter_is_not_excluded_and_the_gate_notices",
    "step105_mutex::control_a_waiter_on_another_mutex_is_not_excluded",
    "step105_mutex::deadline_reports_blocked_and_the_host_release_completes_the_role",
    "step106_l1_atomics::silicon_gates::probe_",
    "step106_l1_atomics::silicon_gates::atincget_wraps",
    "step106_l1_atomics::silicon_gates::control_atincget",
    "step106_l1_atomics::silicon_gates::atincget_is_atomic",
    "step106_l1_atomics::silicon_gates::atswap_group_form",
    "step106_l1_atomics::silicon_gates::control_atswap",
    "step106_l1_atomics::silicon_gates::atcas_with_the_compare_already_met",
    "step106_l1_atomics::silicon_gates::atincgetptr_follows",
    "step106_l1_atomics::silicon_gates::atcas_blocks_until_the_host",
    "step106_l1_atomics::silicon_gates::blocked_atomic_monopolizes_the_scalar_unit",
    "step106_l1_atomics::silicon_gates::atcas_is_freed_by_the_other_roles_risc_v_core",
    "step106_l1_atomics::silicon_gates::atincgetptr_pop_blocks_until_the_write_counter_is_poked",
    "step106_l1_atomics::silicon_gates::atincgetptr_push_blocks_until_the_read_counter_is_poked",
    "step105_mutex::a_poke_stores_a_word_from_the_runner_core",
    "step110_sfploadmacro::s0",
    "step110_sfploadmacro::s1",
    "step110_sfploadmacro::s_restore",
    "step119_device_write_fence::silicon_gates::posted_then_fenced_l1_writes_read_back_exactly",
    "step114_bf16_unpack_to_dst::silicon_probe_bf16_unpack_to_dst_isolated",
    "step114_bf16_unpack_to_dst::bf16_unpack_to_dst_reads_back_widened",
    "step114_bf16_unpack_to_dst::silicon_bf16_dst16_pack_round_trip",
    "step112_packer_relu_edge::silicon_",
    "step112_packer_relu_edge::packer_",
    "step112_packer_relu_edge::relu_unspecified_classes",
    "step113_unpacker_modes::silicon_probe_",
    "step113_unpacker_modes::tileize_strips_match_the_row_stride_model",
    "step113_unpacker_modes::unpacker_transpose_swaps_row_and_column_low_nibbles",
    "step143_fp16_storage::raw_fp16",
    "step143_fp16_storage::widen",
    "step143_fp16_storage::narrowing_matches",
    "step143_fp16_storage::physical_fp16",
    "step144_burn_fp16::",
    "step116_noc_atomics::silicon::silicon_noc_atomic_",
    "step116_noc_atomics::silicon::silicon_rejects_",
    "step117_niu_completion::silicon::silicon_niu_rtz_minimal_probe",
    "step117_niu_completion::silicon::silicon_niu_rtz_source_follows_completion",
    "step117_niu_completion::silicon::silicon_niu_rtz_source_is_sticky_until_cleared",
    "step117_niu_completion::silicon::silicon_niu_rtz_is_not_a_multicast_completion",
    "step118_mover_fast_path::silicon::silicon_x6_fast_read_path_matches_the_slow_one",
    "step118_mover_fast_path::silicon::silicon_x6_rejects_a_skipped_destination",
    "step118_mover_fast_path::silicon::silicon_x6_oracle_rejects_a_flipped_bit",
    "step115_noc_multicast::silicon::",
    "step107_mmio_regs::silicon::",
    "step140_prng_model::lane_initialisation_matches_the_model",
    "step141_random_kernel::",
    "step142_burn_random::",
    "step145_approx_math::",
    "step146_burn_approx::",
    "step147_mesh_trace::",
    "step111_matrix_debug::probe_movdbga2d_minimal_survives",
    "step111_matrix_debug::movdbga2d_addr_mod_",
    "step111_matrix_debug::mova2d_control_addr_mod_",
    "step111_matrix_debug::movdbga2d_eight_rows",
    "step111_matrix_debug::movdbga2d_one_row",
    "step111_matrix_debug::movdbga2d_addr_mod_1_two_blocks",
    "step111_matrix_debug::movdbga2d_format_override",
    "step111_matrix_debug::movdbga2d_denormal",
    "step111_matrix_debug::movdbga2d_data_mutants",
    "step111_matrix_debug::mova2d_control_one_row",
    "step111_matrix_debug::movdbga2d_reads_a_bank",
    "step111_matrix_debug::probe_gatesrcrst",
    "step111_matrix_debug::gatesrcrst_stale",
    "step135_sort::",
    "step136_burn_sort::",
    "step9_matmul::elw_broadcast_assignment_and_destination_fields",
    "step9_matmul::elw_oracles_reject_safe_instruction_mutants",
];

pub const USAGE: &str = "\
usage: cargo xtask silicon [options]

  --device N|all        which /dev/tenstorrent/N to run against (default 0);
                        `all` runs the whole selection on each card in turn
  --filter S            run only tests whose `binary::test` name contains S;
                        repeatable, and the selection runs in filter order
  --smoke               the everyday smoke test: burn-tt against burn-flex,
                        from single ops up to the reduced training runs
                        (adds the filters in `SMOKE`, after any given)
  --include-ignored     also run #[ignore] tests (the exploratory probes)
  --keep-going          do not stop at the first failure
  --timeout-secs N      per-test wall-clock limit (default 120)
  --feature NAME        also build `tt-tests` with this feature (repeatable); only for
                        opt-in experiments such as `ttinsn-experiment`, never a suite
  --release             build the suite optimised; every timing worth quoting
                        comes from this, since the dev build is not a baseline
  --list                print the selection and exit without touching a card";

/// Every card the driver has enumerated.
fn all_devices() -> Result<Vec<u16>, String> {
    let mut v: Vec<u16> = std::fs::read_dir("/dev/tenstorrent")
        .map_err(|e| format!("/dev/tenstorrent: {e}"))?
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .collect();
    v.sort();
    if v.is_empty() {
        return Err("no cards under /dev/tenstorrent".into());
    }
    Ok(v)
}

pub fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    let o = parse(args)?;
    let (_, results) = run_opts(&o)?;
    if results.iter().all(|r| r.2 == Verdict::Pass) {
        Ok(())
    } else {
        Err("silicon suite failed".into())
    }
}

/// What `cargo xtask bench` runs by default, in order: device-timed memory,
/// the core path, the Ethernet counter's own gate, then Ethernet -- which
/// needs that gate to have passed -- and the host link last.
pub const BENCH: &[&str] = &[
    "silicon_bench_memory::",
    "silicon_bench_path::",
    "silicon_eth_clock::",
    "silicon_bench_eth::",
    "silicon_perf::pcie_l1",
    "silicon_perf::pcie_dram",
];

pub const BENCH_USAGE: &str = "\
usage: cargo xtask bench [silicon options]

  Runs the firmware benchmarks (`BENCH` in xtask/src/silicon.rs, or the
  given --filter selection) as `cargo xtask silicon --release
  --include-ignored`, with a 900 s per-test limit unless --timeout-secs says
  otherwise, then collects every `BENCH {json}` line the tests printed into

    target/silicon/bench/<stamp>.jsonl   one record per line, with its test
    target/silicon/bench/<stamp>.md      the same as a table

  Takes every `cargo xtask silicon` option; see `cargo xtask silicon --help`.";

/// `cargo xtask bench`: the benchmark preset of the silicon runner, plus the
/// collection of what the benchmarks reported.
pub fn bench(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args: Vec<String> = args.collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        return Err(BENCH_USAGE.into());
    }
    let mut o = parse(args.iter().cloned())?;
    o.release = true;
    o.include_ignored = true;
    if !args.iter().any(|a| a == "--timeout-secs") {
        o.timeout = Duration::from_secs(900);
    }
    if o.filters.is_empty() {
        o.filters.extend(BENCH.iter().map(|s| s.to_string()));
    }
    let (stamp, results) = run_opts(&o)?;
    if o.list_only {
        return Ok(());
    }
    let dir = workspace_root().join("target/silicon/bench");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut jsonl = String::new();
    let mut md = String::from(
        "| test | dev | result | timed | median | unit | p10 | p90 | GB/s | % of peak | peak |\n\
         |---|--:|---|---|--:|---|--:|--:|--:|--:|---|\n",
    );
    let mut conditions = String::new();
    for (dev, full, verdict, out) in &results {
        let text = std::fs::read_to_string(out).unwrap_or_default();
        for rec in text.lines().filter_map(|l| l.strip_prefix("BENCH ")) {
            jsonl += &format!("{{\"test\":\"{full}\",\"dev\":{dev},\"verdict\":\"{verdict}\",\"record\":{rec}}}\n");
            if rec.contains("\"kind\":\"conditions\"") {
                if conditions.is_empty() {
                    conditions = rec.to_string();
                }
                continue;
            }
            let f = |k: &str| json_field(rec, k).unwrap_or_default();
            md += &format!(
                "| {} | {dev} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                full.split("::").last().unwrap_or(full),
                f("key"),
                f("timed"),
                num(&f("median")),
                f("unit"),
                num(&f("p10")),
                num(&f("p90")),
                f("gbps"),
                f("pct_of_peak"),
                f("peak"),
            );
        }
    }
    let jsonl_path = dir.join(format!("{stamp}.jsonl"));
    let md_path = dir.join(format!("{stamp}.md"));
    std::fs::write(&jsonl_path, jsonl).map_err(|e| format!("{}: {e}", jsonl_path.display()))?;
    std::fs::write(&md_path, format!("conditions: `{conditions}`\n\n{md}"))
        .map_err(|e| format!("{}: {e}", md_path.display()))?;
    println!(
        "benchmarks collected into {} and {}",
        jsonl_path.display(),
        md_path.display()
    );
    if results.iter().all(|r| r.2 == Verdict::Pass) {
        Ok(())
    } else {
        Err("a benchmark failed; what the others reported was still collected".into())
    }
}

/// A value from one flat JSON object, as its text, quotes removed.
fn json_field(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":");
    let start = obj.find(&pat)? + pat.len();
    let rest = &obj[start..];
    if let Some(r) = rest.strip_prefix('"') {
        return Some(r[..r.find('"')?].to_string());
    }
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

fn num(s: &str) -> String {
    s.parse::<f64>()
        .map_or(s.to_string(), |v| format!("{v:.3}"))
}

type Results = Vec<(u16, String, Verdict, PathBuf)>;

fn run_opts(o: &Opts) -> Result<(u64, Results), String> {
    let root = workspace_root();
    let binaries = build(&root, o.release, &select(&o.features))?;

    let mut selection = Vec::new();
    for b in &binaries {
        for test in list(&b.exe, o.include_ignored)? {
            let full = format!("{}::{test}", b.name);
            selection.push((full, b.clone(), test));
        }
    }
    if !o.filters.is_empty() {
        // Filter order is run order, so a caller can say "Phase 2, then Phase 3"
        // and have the riskier group gate the next one.
        // A filter that matches nothing is a typo or a renamed test, and with
        // `--smoke` a stale preset entry; either way, say so rather than run less.
        let unmatched: Vec<&String> = o
            .filters
            .iter()
            .filter(|f| !selection.iter().any(|s| s.0.contains(f.as_str())))
            .collect();
        if !unmatched.is_empty() {
            return Err(format!("no test matches the filter(s) {unmatched:?}"));
        }
        let mut ordered = Vec::new();
        for f in &o.filters {
            for s in &selection {
                if s.0.contains(f.as_str()) && !ordered.iter().any(|o: &(String, _, _)| o.0 == s.0)
                {
                    ordered.push(s.clone());
                }
            }
        }
        selection = ordered;
    }
    if selection.is_empty() {
        return Err("no tests selected".into());
    }
    if o.list_only {
        for (full, _, _) in &selection {
            println!("{full}");
        }
        return Ok((0, Vec::new()));
    }

    preflight(&o.devices)?;

    let dir = root.join("target/silicon");
    std::fs::create_dir_all(dir.join("out")).map_err(|e| format!("{}: {e}", dir.display()))?;
    let stamp = unix_secs();
    let log_path = dir.join(format!("{stamp}.log"));
    let mut log = Log::open(&log_path)?;
    let boot = boot_id();
    println!("logging to {}", log_path.display());
    // A timing in the log is only comparable with one from the same profile.
    let profile = if o.release { "release" } else { "dev" };
    log.line(&format!("{} boot={boot} PROFILE {profile}", unix_secs()))?;

    let mut results = Vec::new();
    'cards: for &dev in &o.devices {
        for (full, bin, test) in &selection {
            if dev != 0 && card0_only(bin) {
                println!("dev{dev} SKIP    {full}: runs on card 0 only");
                continue;
            }
            log.line(&format!(
                "{} boot={boot} dev={dev} START {full}",
                unix_secs()
            ))?;
            let out_path = dir
                .join("out")
                .join(format!("{stamp}-dev{dev}-{}.txt", full.replace("::", "__")));
            let started = Instant::now();
            // `cargo test` runs a package's tests from its own directory.
            let env = [("TT_SILICON_DEVICE", dev.to_string())];
            let verdict = run_one(&bin.exe, test, &bin.dir, &env, o.timeout, &out_path)?;
            let secs = started.elapsed().as_secs_f64();
            log.line(&format!(
                "{} boot={boot} dev={dev} END {full} {verdict} {secs:.2}s",
                unix_secs()
            ))?;
            println!("dev{dev} {verdict:<7} {secs:>7.2}s  {full}");
            let failed = verdict != Verdict::Pass;
            results.push((dev, full.clone(), verdict, out_path));
            if failed && !o.keep_going {
                break 'cards;
            }
        }
    }

    let failures: Vec<_> = results.iter().filter(|r| r.2 != Verdict::Pass).collect();
    println!(
        "\n{} run, {} passed, {} failed",
        results.len(),
        results.len() - failures.len(),
        failures.len()
    );
    for (dev, full, verdict, out) in &failures {
        println!("  dev{dev} {verdict} {full}\n    output: {}", out.display());
    }
    Ok((stamp, results))
}

/// The cargo selection for the silicon test binaries: `tt-tests` with its `silicon`
/// feature (and any experiment features the caller asked for), and the Burn
/// conformance suite built for silicon too, whose smoke subset `SMOKE` lists.
fn select(features: &[String]) -> Vec<String> {
    let features: Vec<String> = std::iter::once("silicon".to_string())
        .chain(features.iter().cloned())
        .map(|f| format!("tt-tests/{f}"))
        .chain(std::iter::once("burn-tt-conformance/silicon".to_string()))
        .collect();
    vec![
        "-p".into(),
        "tt-tests".into(),
        "-p".into(),
        "burn-tt-conformance".into(),
        "--features".into(),
        features.join(","),
    ]
}

/// Binaries whose tests always run on card 0 (`TtDevice::default()`), whatever
/// `--device` says: running them on another card would test card 0 again.
fn card0_only(bin: &Built) -> bool {
    bin.dir.ends_with("burn-tt-conformance")
}

/// Refuse to start against a card that is not there, and say out loud whether the
/// ARC watchdog is armed.
pub(crate) fn preflight(devices: &[u16]) -> Result<(), String> {
    for d in devices {
        let p = format!("/dev/tenstorrent/{d}");
        if !Path::new(&p).exists() {
            return Err(format!("{p} does not exist"));
        }
    }
    let param = "/sys/module/tenstorrent/parameters/auto_reset_timeout";
    match std::fs::read_to_string(param).map(|v| v.trim().to_string()) {
        Ok(v) if v == "0" => {
            println!("auto_reset_timeout=0: a hung NoC wedges the card, not the host")
        }
        Ok(v) => println!(
            "note: auto_reset_timeout={v} -- the ARC watchdog is armed, so a hung NoC \
             resets the chip and drops the PCIe link"
        ),
        Err(e) => println!("note: cannot read {param} ({e})"),
    }
    Ok(())
}

pub(crate) struct Log(File);

impl Log {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map(Log)
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Append and `fsync`: a line that is not on disk when the host dies is a line
    /// that was never written.
    pub(crate) fn line(&mut self, s: &str) -> Result<(), String> {
        writeln!(self.0, "{s}").map_err(|e| e.to_string())?;
        self.0.sync_all().map_err(|e| e.to_string())
    }
}

pub(crate) fn boot_id() -> String {
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

pub(crate) fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
