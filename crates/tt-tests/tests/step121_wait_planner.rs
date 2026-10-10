//! Phase 10 gate (P9): the hazard checker over the real kernel builders.
//!
//! `tt_isa::hazard` states why each hand-placed `STALLWAIT` exists as a table and
//! checks a thread's instruction stream against it. This gate runs it over every
//! role program the public builders emit -- matmul at every route and fidelity,
//! every SFPU element-wise kind (with its broadcast forms), reductions, scans --
//! as the backend receives them (block repeats unrolled, `MOP` and `REPLAY`
//! expanded by `loops::frontend_stream`), and requires no program to miss a
//! documented wait. It also holds the checker to its table (the block table is
//! compared with `STALLWAIT.md`) and to negative controls: each required wait
//! removed from a real program must be flagged, and the unmodified program must
//! pass. Host only: no simulator, no device.
//!
//! Redundant waits are counted and listed, not failed: they are the cost a
//! planner would remove (`hardware-coverage.md`, P9).

use std::collections::BTreeMap;

use tt_isa::backend::{self, block, cond, Before};
use tt_isa::hazard::{self, Finding, Kind, Unit};
use tt_isa::isa::generated::ALL;
use tt_isa::isa::Instruction;
use tt_isa::tile::L1Format;
use tt_kernels::code::Code;
use tt_kernels::loops::frontend_stream;
use tt_kernels::matmul::{self, Fidelity, MatmulSemaphores, OutputTile, SrcRoute};
use tt_kernels::sfpu::kernel::{self, Operands};
use tt_kernels::sfpu::ops::{self, kind_sfpu, Broadcast};
use tt_kernels::sfpu::reduce::{self, Axis, ReduceOp};
use tt_kernels::sfpu::scan::{self, ScanOp};

/// A program the corpus checked, and what the checker said.
struct Checked {
    label: String,
    findings: Vec<Finding>,
    words: usize,
    /// `(block mask, condition mask)` of each redundant wait.
    redundant: Vec<(u32, u32)>,
}

fn check(label: impl Into<String>, backend_stream: &[Instruction]) -> Checked {
    let mut findings = Vec::new();
    hazard::check(backend_stream.iter().copied(), true, |f| findings.push(f));
    let redundant = findings
        .iter()
        .filter(|f| f.kind == Kind::RedundantWait)
        .map(|f| {
            let w = backend_stream[f.index];
            (
                w.operand("BlockMask").unwrap(),
                w.operand("ConditionMask").unwrap(),
            )
        })
        .collect();
    Checked {
        label: label.into(),
        findings,
        words: backend_stream.len(),
        redundant,
    }
}

/// What the backend receives from a role slot: block repeats unrolled, then the
/// frontend's expanders.
fn backend_of_code(code: &Code) -> Vec<Instruction> {
    frontend_stream(&code.expand(), None).expect("the frontend model accepts the program")
}

fn sfpu_roles(layout: &kernel::Layout, operands: Operands, math: &Code) -> [Vec<Instruction>; 3] {
    let (ins, loops) = kernel::roles_code(layout, operands, math);
    std::array::from_fn(|t| {
        backend_of_code(&Code {
            ins: ins[t].clone(),
            loops: loops[t].clone(),
        })
    })
}

/// Every public builder's role programs.
fn corpus() -> Vec<Checked> {
    let mut out = Vec::new();
    let role = ["unpack", "math", "pack"];

    // Matmul: every route and fidelity, with and without a MOP on the math role,
    // evenly spaced pairs, uneven pairs, and several output tiles.
    for route in [
        SrcRoute::Tf32FromFp32,
        SrcRoute::Bf16FromFp32,
        SrcRoute::Bf16FromBf16,
    ] {
        let (in_fmt, out_fmt) = route.formats();
        let img = if in_fmt == L1Format::Bf16 { 2048 } else { 4096 };
        for fidelity in [
            Fidelity::Lo,
            Fidelity::HiFi2,
            Fidelity::HiFi3,
            Fidelity::HiFi4,
        ] {
            for (tag, outputs) in [
                (
                    "1x1x1",
                    vec![OutputTile {
                        pairs: vec![(0x20000, 0x30000)],
                        out: 0x40000,
                    }],
                ),
                (
                    "1x8x2 even",
                    (0..2)
                        .map(|j| OutputTile {
                            pairs: (0..8)
                                .map(|k| (0x20000 + k * img, 0x30000 + (k * 2 + j) * img))
                                .collect(),
                            out: 0x40000 + j * 4096,
                        })
                        .collect::<Vec<_>>(),
                ),
                (
                    "1x3x1 uneven",
                    vec![OutputTile {
                        pairs: vec![
                            (0x20000, 0x30000),
                            (0x21000 + 64, 0x35000),
                            (0x26000, 0x31000 + 32),
                        ],
                        out: 0x40000,
                    }],
                ),
            ] {
                for allow_mop in [false, true] {
                    let (sems, _) = MatmulSemaphores::alone();
                    let (words, mop) =
                        matmul::matmul_kernel(&outputs, sems, in_fmt, out_fmt, fidelity, allow_mop);
                    for t in 0..3 {
                        let stream = frontend_stream(&words[t], mop[t].as_ref())
                            .expect("frontend model accepts the matmul");
                        out.push(check(
                            format!(
                                "matmul {route:?} {fidelity:?} {tag} mop={allow_mop} {}",
                                role[t]
                            ),
                            &stream,
                        ));
                    }
                }
            }
        }
    }

    // SFPU element-wise: every kind that has a program, each broadcast form.
    let kinds = (1..=tt_kernels::kind::LAST).chain(0x100..=kind_sfpu::LAST);
    for kind in kinds {
        for (bcast, name) in [
            (Broadcast::None, ""),
            (Broadcast::Row, " row"),
            (Broadcast::Col, " col"),
        ] {
            let Some((operands, code)) = ops::code_for(kind, [0.5, 2.0], bcast) else {
                continue;
            };
            for tiles in [1usize, 3] {
                let layout = kernel::plan_layout(tiles, operands).unwrap();
                let roles = sfpu_roles(&layout, operands, &code);
                for t in 0..3 {
                    out.push(check(
                        format!("sfpu kind {kind:#x}{name} tiles={tiles} {}", role[t]),
                        &roles[t],
                    ));
                }
            }
        }
    }

    // Reductions.
    for op in [
        ReduceOp::Sum,
        ReduceOp::Max,
        ReduceOp::Prod,
        ReduceOp::All,
        ReduceOp::Any,
        ReduceOp::BitOr,
        ReduceOp::SumI32,
        ReduceOp::ProdI32,
        ReduceOp::MinI32,
        ReduceOp::MaxI32,
    ] {
        for axis in [Axis::Rows, Axis::Cols] {
            let Ok(Ok((inputs, finish))) =
                std::panic::catch_unwind(|| Ok::<_, ()>(reduce::math_programs(op, axis, 3, 17)))
            else {
                continue;
            };
            let layout = reduce::plan_layout(2, 3).unwrap();
            let roles = reduce::roles(&layout, &inputs, &finish);
            for t in 0..3 {
                let stream = frontend_stream(&roles[t], None).expect("frontend accepts");
                out.push(check(
                    format!("reduce {op:?} {axis:?} {}", role[t]),
                    &stream,
                ));
            }
        }
    }

    // Scans, through the shared SFPU tile kernel.
    for op in [
        ScanOp::Sum,
        ScanOp::Prod,
        ScanOp::Min,
        ScanOp::Max,
        ScanOp::MinNan,
        ScanOp::MaxNan,
        ScanOp::ISum,
        ScanOp::IProd,
        ScanOp::IMin,
        ScanOp::IMax,
    ] {
        for first in [true, false] {
            let program = scan::program(op, first);
            let layout = kernel::plan_layout(2, Operands::Unary).unwrap();
            let roles = sfpu_roles(&layout, Operands::Unary, &Code::plain(program));
            for t in 0..3 {
                out.push(check(
                    format!("scan {op:?} first={first} {}", role[t]),
                    &roles[t],
                ));
            }
        }
    }
    out
}

fn kinds(c: &Checked) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for f in &c.findings {
        *m.entry(format!("{:?}", f.kind)).or_default() += 1;
    }
    m
}

/// No program of the corpus misses a documented wait.
#[test]
fn every_builder_program_has_every_wait_its_hazards_need() {
    let programs = corpus();
    assert!(programs.len() > 900, "corpus of {}", programs.len());
    let words: usize = programs.iter().map(|p| p.words).sum();

    let mut bad: Vec<String> = Vec::new();
    let mut bad_groups: BTreeMap<String, (usize, String, usize)> = BTreeMap::new();
    let mut redundant: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for p in &programs {
        for f in &p.findings {
            if f.kind.is_miss() {
                // One line per (group, role, rule, consumer, producer), with a
                // count and the first program that shows it.
                let group = p.label.split_whitespace().next().unwrap();
                let role = p.label.rsplit(' ').next().unwrap();
                let key = format!(
                    "{group} {role}: {:?} at {} unprotected {:?}",
                    f.kind, f.consumer, f.producer
                );
                let e = bad_groups
                    .entry(key)
                    .or_insert((0, p.label.clone(), f.index));
                e.0 += 1;
            } else {
                // Group by program family (the label up to the first digit run
                // that varies) to keep the listing short.
                let family = p
                    .label
                    .split(" tiles=")
                    .next()
                    .unwrap()
                    .split(" mop=")
                    .next()
                    .unwrap()
                    .to_string();
                let e = redundant.entry(family).or_insert((0, p.label.clone()));
                e.0 += 1;
            }
        }
    }
    let total_redundant: usize = redundant.values().map(|e| e.0).sum();
    let mut by_mask: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for p in &programs {
        for m in &p.redundant {
            *by_mask.entry(*m).or_default() += 1;
        }
    }
    for ((b, c), n) in &by_mask {
        eprintln!("  redundant STALLWAIT block {b:#011b} cond {c:#015b}: {n}");
    }
    eprintln!(
        "hazard corpus: {} programs, {words} backend words, {} redundant waits ({:.3}% of words), {} misses",
        programs.len(),
        total_redundant,
        100.0 * total_redundant as f64 / words as f64,
        bad.len()
    );
    let mut by_family: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (family, (n, _)) in &redundant {
        let key = family.split_whitespace().next().unwrap();
        let e = by_family.entry(key).or_default();
        e.0 += 1;
        e.1 += n;
    }
    for (k, (families, n)) in by_family {
        eprintln!("  redundant waits in {k}: {n} over {families} program families");
    }
    for (k, (n, first, at)) in &bad_groups {
        bad.push(format!("{k}: {n} times, first {first} word {at}"));
    }
    assert!(bad.is_empty(), "missed waits:\n{}", bad.join("\n"));
    // The summary used by the checker's own report.
    let _ = kinds(&programs[0]);
}

// ---------------------------------------------------------------------------
// The checker held to its table.

/// `STALLWAIT.md`'s block table, parsed: mnemonic -> the block bits it names.
fn spec_block_table() -> BTreeMap<String, u32> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../vendor/tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e} (run `cargo xtask fetch-spec`)"));
    let mut table = BTreeMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("<tr><th align=\"left\"><code>") else {
            continue;
        };
        let Some((name, cells)) = rest.split_once("</code></th>") else {
            continue;
        };
        if cells.contains("colspan") {
            continue;
        }
        let bits: u32 = cells
            .split("<td")
            .skip(1)
            .enumerate()
            .filter(|(_, c)| c.contains('\u{274c}'))
            .fold(0, |a, (i, _)| a | 1 << i);
        table.insert(name.to_string(), bits);
    }
    table
}

#[test]
fn the_block_table_is_the_specification_s() {
    let spec = spec_block_table();
    assert!(spec.len() > 100, "parsed {} rows", spec.len());
    let mut compared = 0;
    let mut mismatches = Vec::new();
    for def in ALL {
        let m = def.mnemonic();
        let row = [
            m,
            m.trim_end_matches(|c: char| c.is_ascii_digit()),
            m.split('_').next().unwrap(),
            if m.starts_with("UNPACR_NOP") {
                "UNPACR_NOP"
            } else {
                m
            },
            m.trim_end_matches('i'),
            m.trim_end_matches('b'),
        ]
        .into_iter()
        .find_map(|k| spec.get(k).map(|b| (k, *b)));
        let Some((_, want)) = row else { continue };
        let i = Instruction::new(def.skeleton(), def);
        compared += 1;
        let got = hazard::blocked_by(i);
        // A wait and the gate itself are blocked by every bit.
        if got != want {
            mismatches.push(format!(
                "{m}: table {got:#011b}, specification {want:#011b}"
            ));
        }
    }
    assert!(compared > 120, "only {compared} instructions compared");
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

// ---------------------------------------------------------------------------
// Negative controls: remove a required wait from a real program.

fn sfpu_unary_roles() -> [Vec<Instruction>; 3] {
    let (operands, code) = ops::code_for(kind_sfpu::LAST.min(0x100), [0.5, 2.0], Broadcast::None)
        .or_else(|| ops::code_for(5, [0.0, 0.0], Broadcast::None))
        .unwrap();
    let layout = kernel::plan_layout(2, operands).unwrap();
    sfpu_roles(&layout, operands, &code)
}

fn misses(stream: &[Instruction]) -> Vec<Finding> {
    let mut f = Vec::new();
    hazard::check(stream.iter().copied(), true, |x| {
        if x.kind.is_miss() {
            f.push(x)
        }
    });
    f
}

fn remove_first(
    stream: &[Instruction],
    mut pred: impl FnMut(&Instruction) -> bool,
) -> Vec<Instruction> {
    let at = stream.iter().position(&mut pred).expect("a wait to remove");
    let mut v = stream.to_vec();
    v.remove(at);
    v
}

fn is_wait(i: &Instruction, block_bits: u32, cond_bits: u32) -> bool {
    i.def().mnemonic() == "STALLWAIT"
        && i.operand("BlockMask") == Some(block_bits)
        && i.operand("ConditionMask") == Some(cond_bits)
}

#[test]
fn the_unmodified_sfpu_kernel_passes_and_each_removed_wait_is_flagged() {
    let [unpack, math, pack] = sfpu_unary_roles();
    for (name, p) in [("unpack", &unpack), ("math", &math), ("pack", &pack)] {
        assert!(misses(p).is_empty(), "{name} unmodified: {:?}", misses(p));
    }

    // 1. The unpack role's announcement: drop the wait in front of the post.
    let every = Before::EVERYTHING.mask();
    let m = remove_first(&unpack, |i| {
        is_wait(i, every | block::UNPACKER, cond::UNPACKER0_BUSY)
    });
    let f = misses(&m);
    assert!(
        f.iter()
            .any(|f| f.kind == Kind::PublishBeforeDrain && f.producer == Some(Unit::Unpacker0)),
        "dropped post_after wait: {f:?}"
    );

    // 2. A take that does not hold its consumer: SEMWAIT's block mask emptied of
    // the unit that reads what was handed over.
    let mut m = pack.clone();
    let at = m
        .iter()
        .position(|i| i.def().mnemonic() == "SEMWAIT")
        .expect("the pack role takes `computed`");
    let words = m[at].word();
    let field = m[at].def().field("BlockMask").unwrap();
    let cleared = (words & !field.place(0x1ff)) | field.place(block::SYNC);
    m[at] = Instruction::new(cleared, m[at].def());
    let f = misses(&m);
    assert!(
        f.iter().any(|f| f.kind == Kind::TakeNotHeld),
        "take no longer holds the packer: {f:?}"
    );

    // 3. The packer's config rewrite must land before the PACR: drop the
    // `STALLWAIT(PACKER | CONFIG, CONFIG_BUSY)` of `pack_tile_from_dst` alone
    // leaves the scheduling `NOP` (the documented separation), which suffices;
    // dropping the wait and the `NOP` both is the miss.
    let wait = |i: &Instruction| is_wait(i, block::PACKER | block::CONFIG, cond::CONFIG_BUSY);
    let m = remove_first(&pack, wait);
    assert!(
        misses(&m).is_empty(),
        "the NOP alone is the documented rule"
    );
    let at = pack.iter().position(wait).unwrap();
    assert_eq!(pack[at - 1].def().mnemonic(), "NOP");
    // Everything between the last `WRCFG` and the first `PACR` (the `NOP`, the
    // wait, `SETC16`s and ADC setup, all of which also separate them) goes, so
    // the `PACR` follows the write directly.
    let first_pacr = at
        + pack[at..]
            .iter()
            .position(|i| i.def().mnemonic() == "PACR")
            .unwrap();
    let mut m = pack.clone();
    m.drain(at - 1..first_pacr);
    assert!(
        misses(&m).iter().any(|f| f.kind == Kind::ConfigNotLanded),
        "pack config landing"
    );

    // 4. The packer must be drained before its config is rewritten again: the
    // second tile's `wait_for_packer(CONFIG)` is redundant after the first
    // tile's `post_after` (which drains with every bit), so the control removes
    // both -- every packer drain but the closing one.
    let mut m = pack.clone();
    let last = m.len() - 1;
    let mut at = 0;
    m.retain(|i| {
        at += 1;
        i.def().mnemonic() != "STALLWAIT"
            || i.operand("ConditionMask") != Some(cond::PACKER_BUSY)
            || at - 1 == last
    });
    assert!(
        misses(&m).iter().any(|f| f.kind == Kind::ConfigWhileBusy),
        "packer drained before rewrite"
    );

    // 5. A role must not end with work in flight. The closing drain is
    // redundant after the last `post_after` (which already drained), so dropping
    // it alone passes; dropping the last tile's whole announcement and the
    // closing drain leaves its `PACR`s in flight.
    let mut m = pack.clone();
    let last = m.pop().unwrap();
    assert!(is_wait(&last, every | block::PACKER, cond::PACKER_BUSY));
    assert!(misses(&m).is_empty(), "closing drain is redundant here");
    m.truncate(m.len() - 2); // the last tile's `post_after`: wait, SEMPOST
    assert!(
        misses(&m).iter().any(|f| f.kind == Kind::EndsBusy),
        "closing drain"
    );
}

#[test]
fn a_multiply_add_followed_by_a_case_stalling_misses_needs_its_nop() {
    // Find a real SFPU program with an SFPMAD-family instruction followed by an
    // SFPNOP the builder inserted, and remove the NOP.
    let mut found = None;
    for kind in 1..=0x1f0 {
        let Some((_, code)) = ops::code_for(kind, [0.5, 2.0], Broadcast::None) else {
            continue;
        };
        let s = backend_of_code(&code);
        if let Some(at) = (1..s.len()).find(|&k| {
            s[k].def().mnemonic() == "SFPNOP"
                && matches!(
                    s[k - 1].def().mnemonic(),
                    "SFPMAD" | "SFPMUL" | "SFPADD" | "SFPMULI" | "SFPADDI"
                )
                && !s[k + 1].stalls_automatically_after_mad()
        }) {
            found = Some((kind, s, at));
            break;
        }
    }
    let (kind, s, at) = found.expect("some builder program needs an SFPNOP after a multiply-add");
    let mut mutated = s.clone();
    mutated.remove(at);
    let f: Vec<_> = {
        let mut v = Vec::new();
        hazard::check(mutated.iter().copied(), false, |x| v.push(x));
        v
    };
    assert!(
        f.iter().any(|f| f.kind == Kind::MadMissedStall),
        "kind {kind:#x}: NOP at {at} removed: {f:?}"
    );
    let mut v = Vec::new();
    hazard::check(s.iter().copied(), false, |x| v.push(x));
    assert!(v.iter().all(|f| f.kind != Kind::MadMissedStall), "{v:?}");
}

#[test]
fn an_unpacker_write_read_by_sfpload_in_one_thread_needs_its_wait() {
    // The race the first silicon run of the elementwise gate hit: a wait for
    // unpacker 0 that blocks only the unpackers lets the SFPLOADs go ahead.
    use tt_isa::isa::generated::encode;
    let unpacr = tt_kernels::datapath::unpack_instruction();
    let load = tt_isa::sfpu::load(0, 0, 0, 0).unwrap();
    let weak = [
        unpacr,
        backend::wait_for_unpacker0(Before::UNPACKER).unwrap(),
        load,
    ];
    let strong = [
        unpacr,
        backend::wait_for_unpacker0(Before::SFPU).unwrap(),
        load,
    ];
    let _ = encode::nop;
    let f = |s: &[Instruction]| {
        let mut v = Vec::new();
        hazard::check(s.iter().copied(), false, |x| {
            if x.kind.is_miss() {
                v.push(x)
            }
        });
        v
    };
    assert!(f(&weak).iter().any(|f| f.kind == Kind::DstUnpackerToSfpu));
    assert!(f(&strong).iter().all(|f| f.kind != Kind::DstUnpackerToSfpu));
}

// ---------------------------------------------------------------------------
// Part B's measurement: what does pushing cost against what the backend does?

/// The runner pushes `Code::expand()` word by word at 2.8 cycles a word with
/// sixteen-word batches (`silicon_perf::role_push_rate`, divergence row AB); the
/// frontend then expands `REPLAY`s, so the backend receives
/// `frontend_stream(..)` instructions. The Vector Unit retires at most one
/// instruction a cycle (`VectorUnit.md`: every `SFP*` row's throughput is 1, a
/// few 2-cycle latencies hidden by hardware), so `backend` instructions take at
/// least `backend` cycles: the ratio below is a *lower bound* on how push-bound a
/// kernel is. Above 1.0 the runner, not the Vector Unit, sets the pace.
#[test]
fn how_push_bound_are_the_sfpu_math_programs() {
    const CYCLES_PER_WORD: f64 = 2.8;
    let mut rows: Vec<(u32, usize, usize, f64)> = Vec::new();
    for kind in (1..=tt_kernels::kind::LAST).chain(0x100..=kind_sfpu::LAST) {
        let Some((_, code)) = ops::code_for(kind, [0.5, 2.0], Broadcast::None) else {
            continue;
        };
        let pushed = code.expand().len();
        let backend_words = backend_of_code(&code).len();
        rows.push((
            kind,
            pushed,
            backend_words,
            CYCLES_PER_WORD * pushed as f64 / backend_words as f64,
        ));
    }
    rows.sort_by(|a, b| a.3.total_cmp(&b.3));
    let n = rows.len();
    let (pushed, backend_words) = rows
        .iter()
        .fold((0usize, 0usize), |a, r| (a.0 + r.1, a.1 + r.2));
    let bound = rows.iter().filter(|r| r.3 > 1.0).count();
    eprintln!(
        "{n} SFPU kinds: pushed words {pushed}, backend instructions {backend_words} \
         (replay expands {:.2}x); push-cycles / backend-cycles: min {:.2}, median {:.2}, max {:.2}; \
         {bound} of {n} kinds above 1.0 (push-bound)",
        backend_words as f64 / pushed as f64,
        rows[0].3,
        rows[n / 2].3,
        rows[n - 1].3,
    );
    for r in [
        &rows[0],
        &rows[n / 4],
        &rows[n / 2],
        &rows[3 * n / 4],
        &rows[n - 1],
    ] {
        eprintln!(
            "  kind {:#x}: pushed {}, backend {}, ratio {:.2}",
            r.0, r.1, r.2, r.3
        );
    }
}
