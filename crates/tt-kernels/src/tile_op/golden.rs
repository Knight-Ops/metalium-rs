//! Every job the builders [`super`] replaces emit, pinned word for word (F3.1).
//!
//! One golden a client, under `golden/`: `jobs_eltwise.txt` (the pointwise SFPU family,
//! [`tensor::sfpu_eltwise`] and [`super::eltwise`], which replaced `tensor::sfpu_eltwise_blocks`), `jobs_reduce.txt`
//! ([`tensor::sfpu_reduce_then`], with its chunked fallback), `jobs_matmul.txt` (every lowering
//! behind [`tensor::matmul_dram_prefetching`], the builder `Session::matmul_dram` calls) and
//! `jobs_batched.txt` ([`tensor::matmul_dram_batched_limited`], the batched products of blocks
//! `Session::matmul_dram_batched` builds, which share those lowerings). Each
//! case is built on a fresh allocator, so its output and addresses are its own. A job is dumped as
//! its steps: a list or transfer as every entry's eight words, a kernel as each role's length and a
//! hash of its instruction words, with its semaphores, `MOP`s, loops and half; a refusal as its
//! message. The committed golden folds each job's dump to one hash ([`compact`]); the full dump is
//! kept under `target/` for diffing ([`pinned`]). `TT_BLESS=1` rewrites the goldens.
//!
//! Each case's L1 notes (`l1::take_planned`: the top and need a capture keeps in its `OpRecord`)
//! are pinned beside its jobs, in `golden/planned_<client>.txt`. Each test builds its cases twice,
//! the second time from the builders' memos, and requires the same jobs and notes both times.
//!
//! The builders read a few process-wide switches (`TT_PAIR_ORDER`, `TT_BF16_PLANS`,
//! `TT_MATMUL_BLOCK`, `TT_COL_ONCE`, `TT_MATH`): the goldens are their defaults, so the test
//! refuses to run with any of them set, and holds [`tensor::SWITCHES`] while it reads them.

use std::fmt::Write as _;

use tt_isa::dram::Dram;
use tt_isa::format::TileFormat;

use crate::kind;
use crate::matmul::{Fidelity, PairOrder, SrcRoute, ORDER_HERE};
use crate::sfpu::kernel::Operands;
use crate::sfpu::ops::{self, kind_sfpu};
use crate::sfpu::reduce::{Axis, ReduceOp};
use crate::tensor::{
    self, Block, BlockOperand, BlockStream, DramAlloc, DramTensor, Elem, Eltwise, Job, Overlap,
    Reads, Step, Work,
};

/// FNV-1a, 64 bits: a stable hash of words.
fn fnv(words: impl IntoIterator<Item = u32>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn entries(out: &mut String, label: &str, entries: &[[u32; 8]]) {
    writeln!(out, "    {label} {}", entries.len()).unwrap();
    for e in entries {
        let words: Vec<String> = e.iter().map(|w| format!("{w:08x}")).collect();
        writeln!(out, "      {}", words.join(" ")).unwrap();
    }
}

fn job(out: &mut String, k: usize, job: &Job) {
    writeln!(out, "  job {k}").unwrap();
    for step in job {
        match step {
            Step::List { what, entries: e } => entries(out, &format!("list {what:?}"), e),
            Step::Transfer {
                what,
                depth,
                batches,
            } => {
                writeln!(
                    out,
                    "    transfer {what:?} depth {depth} batches {}",
                    batches.len()
                )
                .unwrap();
                for b in batches {
                    entries(out, "read", &b.read);
                    entries(out, "write", &b.write);
                }
            }
            Step::Kernel {
                roles,
                init,
                mop,
                loops,
                half,
            } => {
                let roles: Vec<String> = roles
                    .iter()
                    .map(|r| format!("{}:{:016x}", r.len(), fnv(r.iter().map(|i| i.word()))))
                    .collect();
                let init: Vec<String> = init
                    .iter()
                    .map(|(s, v, m)| format!("{}={v}/{m}", s.index()))
                    .collect();
                let mop: Vec<String> = mop
                    .iter()
                    .map(|m| match m {
                        None => "-".into(),
                        Some(m) => match m.config_words() {
                            Ok(w) => format!("{:016x}", fnv(w)),
                            Err(e) => format!("{e:?}"),
                        },
                    })
                    .collect();
                let loops: Vec<String> = loops
                    .iter()
                    .map(|l| {
                        l.iter()
                            .map(|l| format!("{}+{}x{}", l.start, l.len, l.count))
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .collect();
                writeln!(
                    out,
                    "    kernel half {half:?} roles {} init [{}] mop [{}] loops [{}]",
                    roles.join(" "),
                    init.join(" "),
                    mop.join(" "),
                    loops.join(" | ")
                )
                .unwrap();
            }
        }
    }
}

std::thread_local! {
    /// Each case's notes since the last ([`case`]).
    static NOTES: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// One case: its name, and what its builder returned. What the build noted of L1 goes to
/// [`NOTES`].
fn case(out: &mut String, name: &str, work: Result<Option<Work>, tensor::TensorError>) {
    let planned = crate::l1::take_planned();
    NOTES.with(|n| {
        writeln!(
            n.borrow_mut(),
            "case {name}\n  planned top {:#x} need {:#x}",
            planned.top,
            planned.need
        )
        .unwrap()
    });
    writeln!(out, "case {name}").unwrap();
    match work {
        Err(e) => writeln!(out, "  refused: {e}").unwrap(),
        Ok(None) => writeln!(out, "  no program").unwrap(),
        Ok(Some(w)) => {
            writeln!(
                out,
                "  out {}x{} {:?} at {:?} jobs {}",
                w.out.rows,
                w.out.cols,
                w.out.elem,
                w.out.tensor_ref().encode(),
                w.jobs.len()
            )
            .unwrap();
            for (k, j) in w.jobs.iter().enumerate() {
                job(out, k, j);
            }
        }
    }
}

/// The full dump with each job's lines folded to one hash (FNV-1a of the lines, 32 bits), sixteen
/// to a line: what is committed.
fn compact(full: &str) -> String {
    let mut out = String::new();
    let mut hashes: Vec<String> = Vec::new();
    let mut job: Option<u64> = None;
    let flush = |out: &mut String, hashes: &mut Vec<String>| {
        for row in hashes.chunks(16) {
            writeln!(out, "  jobs {}", row.join(" ")).unwrap();
        }
        hashes.clear();
    };
    for line in full.lines() {
        if line.starts_with("    ") && job.is_some() {
            job = job.map(|h| fnv_bytes(h, line));
            continue;
        }
        if let Some(h) = job.take() {
            hashes.push(format!("{:08x}", (h ^ (h >> 32)) as u32));
        }
        if line.starts_with("  job ") {
            job = Some(fnv_bytes(0xcbf2_9ce4_8422_2325, line));
            continue;
        }
        flush(&mut out, &mut hashes);
        writeln!(out, "{line}").unwrap();
    }
    if let Some(h) = job {
        hashes.push(format!("{:08x}", (h ^ (h >> 32)) as u32));
    }
    flush(&mut out, &mut hashes);
    out
}

fn fnv_bytes(mut h: u64, line: &str) -> u64 {
    for b in line.bytes().chain(*b"\n") {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Checked against `golden/<file>` ([`compact`]), or written there with `TT_BLESS`. The full
/// dump, every word, is written to `target/tile_op_golden/<file>` on a mismatch or with
/// `TT_GOLDEN_FULL=1`: dump both commits' and diff them.
fn pinned(file: &str, full: &str) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("golden").join(file);
    let dump = compact(full);
    let keep_full = || {
        let dir = root.join("../../target/tile_op_golden");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), full).unwrap();
        dir.join(file)
    };
    if std::env::var_os("TT_GOLDEN_FULL").is_some() {
        keep_full();
    }
    if std::env::var_os("TT_BLESS").is_some() {
        std::fs::write(&path, &dump).unwrap();
    }
    let golden = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("{} is missing: bless it with TT_BLESS=1", path.display()));
    if golden == dump {
        return;
    }
    let full_at = keep_full();
    let mut case = "";
    for (n, (g, d)) in golden.lines().zip(dump.lines()).enumerate() {
        if let Some(c) = d.strip_prefix("case ") {
            case = c;
        }
        assert_eq!(
            g,
            d,
            "{file} line {}, in case {case} (full dump: {})",
            n + 1,
            full_at.display()
        );
    }
    panic!(
        "{file}: {} lines pinned, {} built (full dump: {})",
        golden.lines().count(),
        dump.lines().count(),
        full_at.display()
    );
}

/// `cases` built twice, the second time from the builders' memos: the jobs and notes of both
/// must agree, and are pinned in `golden/jobs_<client>.txt` and `golden/planned_<client>.txt`.
fn pinned_twice(client: &str, cases: impl Fn(&mut String)) {
    let build = || {
        crate::l1::take_planned();
        NOTES.with(|n| n.borrow_mut().clear());
        let mut out = String::new();
        cases(&mut out);
        (out, NOTES.with(|n| n.take()))
    };
    let (jobs, notes) = build();
    let (again, renoted) = build();
    let mut case = "";
    for (a, b) in notes.lines().zip(renoted.lines()) {
        if let Some(c) = a.strip_prefix("case ") {
            case = c;
        }
        assert_eq!(a, b, "case {case}: noted otherwise when built again");
    }
    assert!(jobs == again, "{client}: jobs built again differ");
    pinned(&format!("jobs_{client}.txt"), &jobs);
    pinned(&format!("planned_{client}.txt"), &notes);
}

/// The switches the goldens are the defaults of, held for the test's length.
fn defaults() -> std::sync::MutexGuard<'static, ()> {
    for var in [
        "TT_PAIR_ORDER",
        "TT_BF16_PLANS",
        "TT_MATMUL_BLOCK",
        "TT_COL_ONCE",
        "TT_MATH",
    ] {
        assert!(
            std::env::var_os(var).is_none(),
            "{var} is set: the job goldens are the builders' defaults"
        );
    }
    let held = tensor::SWITCHES.lock().unwrap_or_else(|e| e.into_inner());
    ORDER_HERE.set(Some(PairOrder::Auto));
    held
}

fn fresh() -> DramAlloc {
    DramAlloc::new(&Dram::FULL)
}

/// An operand of `elem`, `rows x cols`.
fn operand(alloc: &mut DramAlloc, [rows, cols]: [usize; 2], elem: Elem) -> DramTensor {
    DramTensor::alloc_elem(alloc, rows, cols, elem).unwrap()
}

fn ew(kind: u32, scalar: f32, scalar2: f32) -> Eltwise {
    Eltwise {
        kind,
        scalar,
        scalar2,
    }
}

/// Every kind with an SFPU program, from `kind` and `kind_sfpu`'s ranges.
fn kinds() -> impl Iterator<Item = u32> {
    (1..=kind::LAST).chain(0x100..=0x1ff)
}

/// Each operand's shape: `a`'s, then `b`'s and `c`'s as `operands` reads them.
fn operand_shapes(operands: Operands, a: [usize; 2]) -> Vec<[usize; 2]> {
    let [r, c] = a;
    match operands {
        Operands::Unary => vec![a],
        Operands::Binary => vec![a, a],
        Operands::RowBroadcast => vec![a, [1, c]],
        Operands::ColBroadcast => vec![a, [r, 1]],
        Operands::Ternary => vec![a, a, a],
    }
}

fn eltwise_case(
    out: &mut String,
    op: Eltwise,
    shapes: &[[usize; 2]],
    units: usize,
    overlap: Overlap,
    twin: bool,
) {
    let mut alloc = fresh();
    let sig = ops::elems(op.kind);
    let ts: Vec<DramTensor> = shapes
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let elem = sig.inputs.get(i).or(sig.inputs.last()).copied();
            operand(&mut alloc, s, elem.unwrap_or(Elem::F32))
        })
        .collect();
    let twin = twin.then(|| operand(&mut alloc, [shapes[0][1] / 32, 32], Elem::F32));
    let name = format!(
        "{:#x} s {:?}/{:?} {shapes:?} units {units} {overlap:?} twin {}",
        op.kind,
        op.scalar,
        op.scalar2,
        twin.is_some()
    );
    let work = tensor::sfpu_eltwise(
        &mut alloc,
        op,
        &ts[0],
        ts.get(1),
        ts.get(2),
        units,
        overlap,
        twin.as_ref(),
    );
    case(out, &name, work);
}

#[test]
fn eltwise_jobs_are_pinned() {
    let _held = defaults();
    pinned_twice("eltwise", |out| {
        // Every kind, ragged over a few tiles and whole over more, plain and pipelined.
        for kind in kinds() {
            let Some(operands) = ops::operands(kind) else {
                continue;
            };
            for (a, units, overlap) in [([37, 70], 3, Overlap::Fresh), ([64, 128], 1, Overlap::Off)]
            {
                let shapes = operand_shapes(operands, a);
                eltwise_case(out, ew(kind, 0.5, 2.0), &shapes, units, overlap, false);
            }
        }
        // Distribution and pipelining across unit counts and overlaps, past one list.
        for kind in [
            kind::ADD,
            kind_sfpu::EXP,
            kind_sfpu::DIV,
            kind_sfpu::SILU_MUL,
        ] {
            let operands = ops::operands(kind).unwrap();
            for units in [1, 3, 8, 120] {
                for overlap in [Overlap::Off, Overlap::Fresh, Overlap::Captured] {
                    let shapes = operand_shapes(operands, [256, 512]);
                    eltwise_case(out, ew(kind, 0.0, 0.0), &shapes, units, overlap, false);
                }
            }
        }
        // Broadcasts: a row, a column and a block of rows, against a ragged `a` and a whole one.
        for kind in [kind::ADD, kind::MUL, kind_sfpu::DIV, kind_sfpu::SUB_EXP] {
            for [r, c] in [[37, 70], [256, 512]] {
                for b in [[1, c], [r, 1], [64.min(r), c]] {
                    for (units, overlap) in [
                        (1, Overlap::Off),
                        (5, Overlap::Captured),
                        (4, Overlap::Fresh),
                    ] {
                        eltwise_case(out, ew(kind, 0.0, 0.0), &[[r, c], b], units, overlap, false);
                    }
                }
            }
        }
        // A one-row output past `PACKED_A_K_TILES` columns, with its packed twin.
        for units in [1, 8] {
            eltwise_case(
                out,
                ew(kind_sfpu::SILU, 0.0, 0.0),
                &[[1, 2048]],
                units,
                Overlap::Fresh,
                true,
            );
        }
        // Blocks read where they lie: a head's columns, a gate's half of a fused projection.
        for (parent, at, dims) in [
            ([64, 512], [0, 128], [64, 128]),
            ([96, 640], [32, 320], [64, 320]),
            ([37, 512], [0, 256], [37, 256]),
        ] {
            for (units, overlap) in [
                (1, Overlap::Off),
                (3, Overlap::Fresh),
                (8, Overlap::Captured),
            ] {
                let mut alloc = fresh();
                let pa = operand(&mut alloc, parent, Elem::F32);
                let pb = operand(&mut alloc, parent, Elem::F32);
                let block = |t| BlockOperand {
                    tensor: t,
                    at,
                    dims,
                };
                let work = crate::tile_op::eltwise(
                    ew(kind_sfpu::SILU_MUL, 0.0, 0.0),
                    block(&pa),
                    Some(block(&pb)),
                    None,
                    None,
                )
                .and_then(|o| {
                    o.map(|mut o| o.build(&mut alloc, units, overlap))
                        .transpose()
                });
                case(
                    out,
                    &format!("blocks {parent:?} at {at:?} dims {dims:?} units {units} {overlap:?}"),
                    work,
                );
            }
        }
    });
}

#[test]
fn reduce_jobs_are_pinned() {
    let _held = defaults();
    pinned_twice("reduce", |out| {
        let ops = [
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
        ];
        let mut reduce = |shape: [usize; 2],
                          op: ReduceOp,
                          axis: Axis,
                          units: usize,
                          overlap: Overlap,
                          then: tensor::ReduceThen| {
            let mut alloc = fresh();
            let a = operand(&mut alloc, shape, op.elem());
            let work = tensor::sfpu_reduce_then(&mut alloc, &a, op, axis, units, overlap, then);
            case(
                out,
                &format!("{op:?} {axis:?} {shape:?} units {units} {overlap:?} then {then:?}"),
                work.map(Some),
            );
        };
        for op in ops {
            for axis in [Axis::Cols, Axis::Rows] {
                for (shape, units, overlap) in [
                    ([37, 70], 1, Overlap::Off),
                    ([37, 70], 3, Overlap::Fresh),
                    ([256, 512], 8, Overlap::Fresh),
                    ([256, 512], 120, Overlap::Captured),
                ] {
                    reduce(shape, op, axis, units, overlap, None);
                }
            }
        }
        // A softmax's clamps: the finishing kind after the fold.
        for (op, then) in [
            (
                ReduceOp::Max,
                Some((kind_sfpu::CLAMP_MIN, f32::MIN.to_bits())),
            ),
            (
                ReduceOp::Sum,
                Some((kind_sfpu::CLAMP_MIN, 1e-30f32.to_bits())),
            ),
        ] {
            for (units, overlap) in [(1, Overlap::Off), (8, Overlap::Captured)] {
                reduce([256, 512], op, Axis::Cols, units, overlap, then);
            }
        }
        // Rows too long for one tile's programs: the chunked fallback, and its refusal of a `then`.
        for (shape, axis) in [([64, 65536], Axis::Cols), ([65536, 64], Axis::Rows)] {
            for units in [1, 8] {
                reduce(shape, ReduceOp::Sum, axis, units, Overlap::Fresh, None);
            }
            reduce(
                shape,
                ReduceOp::Sum,
                axis,
                1,
                Overlap::Fresh,
                Some((kind_sfpu::CLAMP_MIN, 0)),
            );
        }
    });
}

/// `Session::matmul_dram`'s reads on a fresh session.
fn session_reads() -> Reads {
    Reads {
        thin: true,
        resident_a: true,
        unpack_b: true,
        packed_a: None,
        stream_k: true,
        stream_prefetch: true,
        prefetch_units: [crate::session::PREFETCH_UNITS; 2],
        handoff: None,
        second_job: false,
        balanced_deal: false,
        block_stream: BlockStream::Auto,
        reuse: tensor::Reuse::Off,
    }
}

/// One product's shape and how it is read and dealt.
#[derive(Clone)]
struct Product {
    a: [usize; 2],
    a_transposed: bool,
    b: [usize; 2],
    b_transposed: bool,
    formats: [TileFormat; 2],
    units: usize,
    pipeline: bool,
    max_k_tiles: Option<usize>,
    /// `A` packed one row (`tensor::pack_row`'s layout).
    packed_a: bool,
    reads: Reads,
    /// Units past `units`, below `spread`, that read a job's `B` ahead.
    prefetch: bool,
    spread: usize,
    /// Every unit free below the arena (`Some(DATA.base)`), for reading the first job's `B` ahead.
    busy: bool,
}

impl Product {
    fn new(a: [usize; 2], b: [usize; 2], units: usize) -> Product {
        Product {
            a,
            a_transposed: false,
            b,
            b_transposed: false,
            formats: [TileFormat::FP32; 2],
            units,
            pipeline: true,
            max_k_tiles: None,
            packed_a: false,
            reads: session_reads(),
            prefetch: false,
            spread: units,
            busy: false,
        }
    }

    fn run(&self, out: &mut String, order: PairOrder) {
        ORDER_HERE.set(Some(order));
        let mut alloc = fresh();
        let elem = |f: TileFormat| {
            if f == TileFormat::FP32 {
                Elem::F32
            } else {
                Elem::Packed(f)
            }
        };
        let a = operand(&mut alloc, self.a, elem(self.formats[0]));
        let b = operand(&mut alloc, self.b, elem(self.formats[1]));
        let k = if self.a_transposed {
            self.a[0]
        } else {
            self.a[1]
        };
        let p = operand(&mut alloc, [k / 32, 32], Elem::F32);
        let mut reads = self.reads;
        // As `Session::matmul_dram` reads a packed `A`: whole tiles of its format.
        reads.thin &= self.formats[0] == TileFormat::FP32;
        if self.packed_a {
            reads.packed_a = Some(p.tensor_ref());
        }
        let route = if self.formats == [TileFormat::FP32; 2] {
            SrcRoute::Tf32FromFp32
        } else {
            SrcRoute::edges(self.formats[0], self.formats[1]).unwrap()
        };
        let prefetch: Vec<usize> = if self.prefetch {
            (self.units..self.spread).collect()
        } else {
            Vec::new()
        };
        let busy = if self.busy {
            vec![Some(tt_isa::l1::DATA.base); self.units]
        } else {
            Vec::new()
        };
        let name = format!(
            "{:?}{} @ {:?}{} {:?} units {} spread {} pipeline {} k {:?} packed_a {} prefetch {} busy {} {:?} {order:?}",
            self.a,
            if self.a_transposed { "^T" } else { "" },
            self.b,
            if self.b_transposed { "^T" } else { "" },
            self.formats.map(|f| f.l1()),
            self.units,
            self.spread,
            self.pipeline,
            self.max_k_tiles,
            self.packed_a,
            self.prefetch,
            self.busy,
            reads_label(&self.reads),
        );
        let built = tensor::matmul_dram_prefetching(
            &mut alloc,
            &a,
            self.a_transposed,
            &b,
            self.b_transposed,
            route,
            Fidelity::HiFi4,
            self.units,
            false,
            self.pipeline,
            self.max_k_tiles.and_then(std::num::NonZeroUsize::new),
            reads,
            &prefetch,
            self.spread,
            &busy,
            &[],
        );
        match built {
            Ok((work, dealt, handoff)) => {
                case(out, &name, Ok(Some(work)));
                writeln!(out, "  dealt {dealt:?}").unwrap();
                if let Some(h) = handoff {
                    writeln!(
                        out,
                        "  handoff partials {:?} jobs {}",
                        h.partials.tensor_ref().encode(),
                        h.jobs.len()
                    )
                    .unwrap();
                    for (k, j) in h.jobs.iter().enumerate() {
                        job(out, k, j);
                    }
                }
            }
            Err(e) => case(out, &name, Err(e)),
        }
        ORDER_HERE.set(Some(PairOrder::Auto));
    }
}

fn reads_label(r: &Reads) -> String {
    format!(
        "thin {} resident {} unpack_b {} stream_k {} stream_prefetch {} prefetch_units {:?} handoff {:?} second {} balanced {} block {:?}",
        r.thin as u8,
        r.resident_a as u8,
        r.unpack_b as u8,
        r.stream_k as u8,
        r.stream_prefetch as u8,
        r.prefetch_units,
        r.handoff,
        r.second_job as u8,
        r.balanced_deal as u8,
        r.block_stream,
    )
}

#[test]
fn matmul_jobs_are_pinned() {
    let _held = defaults();
    pinned_twice("matmul", |out| {
        // Blocked and K-blocked products, with and without transposes and pipelining.
        let general = [
            ([64, 784], false, [784, 128], false),
            ([64, 128], false, [128, 10], false),
            ([64, 10], false, [128, 10], true),
            ([64, 128], true, [64, 10], false),
            ([37, 45], true, [37, 70], false),
            ([96, 320], false, [320, 288], false),
            ([256, 256], false, [256, 256], false),
            ([512, 1024], false, [1024, 512], false),
        ];
        for (a, at, b, bt) in general {
            for units in [1, 8, 120] {
                for pipeline in [true, false] {
                    let mut p = Product::new(a, b, units);
                    (p.a_transposed, p.b_transposed, p.pipeline) = (at, bt, pipeline);
                    p.run(out, PairOrder::Auto);
                }
            }
            // A transposed `B` read transposed by the mover rather than the unpacker; whole tiles.
            if bt {
                let mut p = Product::new(a, b, 8);
                p.b_transposed = true;
                p.reads.unpack_b = false;
                p.run(out, PairOrder::Auto);
            }
            let mut p = Product::new(a, b, 8);
            (p.a_transposed, p.b_transposed) = (at, bt);
            p.reads.thin = false;
            p.reads.resident_a = false;
            p.run(out, PairOrder::Auto);
        }
        // Each pair order, on a product past one K tile.
        for order in PairOrder::ALL {
            Product::new([96, 320], [320, 288], 3).run(out, order);
        }
        // The block-stream path: chosen by its model, fixed blocks (ragged against the product),
        // off, and its unlooped roles.
        for (a, b, units) in [
            ([1024, 2048], [2048, 2048], 120),
            ([2048, 2048], [2048, 1024], 64),
            ([512, 1024], [1024, 416], 8),
        ] {
            Product::new(a, b, units).run(out, PairOrder::Auto);
        }
        for block in [[3, 4, 5], [2, 2, 2], [8, 8, 4]] {
            let mut p = Product::new([256, 512], [512, 416], 3);
            p.reads.block_stream = BlockStream::Fixed(block);
            p.run(out, PairOrder::Auto);
        }
        let mut p = Product::new([1024, 2048], [2048, 2048], 120);
        p.reads.block_stream = BlockStream::Off;
        p.run(out, PairOrder::Auto);
        crate::matmul::LOOPS_HERE.set(Some(false));
        let mut p = Product::new([256, 512], [512, 416], 3);
        p.reads.block_stream = BlockStream::Fixed([3, 4, 5]);
        p.run(out, PairOrder::Auto);
        crate::matmul::LOOPS_HERE.set(None);
        // K blocked by a limit, and by the program slot alone.
        for (a, b, limit) in [
            ([64, 8192], [8192, 512], Some(64)),
            ([64, 8192], [8192, 512], None),
            ([37, 2048], [2048, 300], Some(16)),
        ] {
            for units in [1, 8] {
                let mut p = Product::new(a, b, units);
                p.max_k_tiles = limit;
                p.run(out, PairOrder::Auto);
            }
        }
        // One row (decode): thin and resident `A`, packed `A`, `K` streamed, a K-block limit.
        for (k, n) in [(2048, 2048), (2048, 6144), (8192, 2048), (1056, 200)] {
            for units in [8, 64] {
                for (packed, stream_k, limit) in [
                    (false, false, None),
                    (true, false, None),
                    (true, true, None),
                    (true, true, Some(16)),
                    (true, false, Some(64)),
                ] {
                    let mut p = Product::new([1, k], [k, n], units);
                    p.packed_a = packed;
                    p.reads.stream_k = stream_k;
                    p.max_k_tiles = limit;
                    p.run(out, PairOrder::Auto);
                }
            }
        }
        // Capture-time reads ahead: idle units prefetching, a second job each, a balanced deal, a
        // streamed column handed between two units, and the first job's `B` read while ops run.
        for (k, n, stream_k) in [(2048, 6144, false), (8192, 2048, true), (1056, 200, true)] {
            for (second, balanced, handoff) in [
                (false, false, None),
                (true, false, None),
                (false, true, None),
                (false, false, Some(0)),
                (false, false, Some(3)),
            ] {
                for busy in [false, true] {
                    let mut p = Product::new([1, k], [k, n], 64);
                    p.packed_a = true;
                    p.reads.stream_k = stream_k;
                    p.reads.second_job = second;
                    p.reads.balanced_deal = balanced;
                    p.reads.handoff = handoff;
                    p.reads.prefetch_units = [crate::session::PREFETCH_UNITS, 40];
                    p.prefetch = true;
                    p.spread = 120;
                    p.busy = busy;
                    p.run(out, PairOrder::Auto);
                }
            }
        }
        // Packed weights and activations: every edge pair F1 admits, one row and blocked.
        let formats = [
            TileFormat::FP32,
            TileFormat::BF16,
            TileFormat::BFP8,
            TileFormat::BFP4,
            TileFormat::BFP2,
        ];
        for fa in formats {
            for fb in formats {
                if SrcRoute::edges(fa, fb).is_none() || [fa, fb] == [TileFormat::FP32; 2] {
                    continue;
                }
                for (a, b, units) in [
                    ([1, 2048], [2048, 2048], 64),
                    ([96, 320], [320, 288], 3),
                    ([1024, 2048], [2048, 2048], 120),
                ] {
                    let mut p = Product::new(a, b, units);
                    p.formats = [fa, fb];
                    p.run(out, PairOrder::Auto);
                }
                // A packed `B` read ahead by idle units, beside a packed one-row `A`.
                if fa == TileFormat::FP32 {
                    let mut p = Product::new([1, 2048], [2048, 6144], 64);
                    p.formats = [fa, fb];
                    p.packed_a = true;
                    p.prefetch = true;
                    p.spread = 120;
                    p.run(out, PairOrder::Auto);
                }
            }
        }
    });
}

/// One batched product: blocks of `a` and `b`, each `[m, k] @ [k, n]` after its transposes.
struct Batched {
    a: [usize; 2],
    b: [usize; 2],
    items: Vec<(Block, Block)>,
    mkn: [usize; 3],
    units: usize,
    pipeline: bool,
    max_k_tiles: Option<usize>,
    padded: [bool; 2],
    unpack_b: bool,
    block_stream: BlockStream,
}

impl Batched {
    fn new(a: [usize; 2], b: [usize; 2], items: Vec<(Block, Block)>, mkn: [usize; 3]) -> Batched {
        // `Session::matmul_dram_batched_with`'s reads on a fresh session.
        Batched {
            a,
            b,
            items,
            mkn,
            units: 8,
            pipeline: true,
            max_k_tiles: None,
            padded: [false; 2],
            unpack_b: true,
            block_stream: BlockStream::Auto,
        }
    }

    fn run(&self, out: &mut String) {
        let mut alloc = fresh();
        let a = operand(&mut alloc, self.a, Elem::F32);
        let b = operand(&mut alloc, self.b, Elem::F32);
        let items: Vec<String> = self
            .items
            .iter()
            .map(|(x, y)| {
                format!(
                    "{:?}{}/{:?}{}",
                    x.at,
                    if x.transposed { "^T" } else { "" },
                    y.at,
                    if y.transposed { "^T" } else { "" }
                )
            })
            .collect();
        let name = format!(
            "{:?} {:?} {:?} items [{}] units {} pipeline {} k {:?} padded {:?} unpack_b {} block {:?}",
            self.a,
            self.b,
            self.mkn,
            items.join(" "),
            self.units,
            self.pipeline,
            self.max_k_tiles,
            self.padded,
            self.unpack_b,
            self.block_stream,
        );
        let work = tensor::matmul_dram_batched_limited(
            &mut alloc,
            &a,
            &b,
            &self.items,
            self.mkn,
            SrcRoute::Tf32FromFp32,
            Fidelity::HiFi4,
            self.units,
            false,
            self.pipeline,
            self.max_k_tiles.and_then(std::num::NonZeroUsize::new),
            self.padded,
            self.unpack_b,
            self.block_stream,
        );
        case(out, &name, work.map(Some));
    }
}

fn block(at: [usize; 2], transposed: bool) -> Block {
    Block { at, transposed }
}

#[test]
fn batched_jobs_are_pinned() {
    let _held = defaults();
    pinned_twice("batched", |out| {
        // Attention's two products over heads of a projection: `Q K^T` (each head's columns of both,
        // `K` read transposed) and `P V` (each head's row block of `P`, its columns of `V`).
        for (s, dk, heads) in [(128usize, 64usize, 4usize), (35, 32, 3), (64, 96, 2)] {
            let qk: Vec<_> = (0..heads)
                .map(|h| (block([0, h * dk], false), block([0, h * dk], true)))
                .collect();
            let pv: Vec<_> = (0..heads)
                .map(|h| {
                    (
                        block([h * s.div_ceil(32) * 32, 0], false),
                        block([0, h * dk], false),
                    )
                })
                .collect();
            let p_rows = heads * s.div_ceil(32) * 32;
            for units in [1, 8, 120] {
                for pipeline in [true, false] {
                    let mut p =
                        Batched::new([s, heads * dk], [s, heads * dk], qk.clone(), [s, dk, s]);
                    (p.units, p.pipeline) = (units, pipeline);
                    p.run(out);
                    let mut p = Batched::new([p_rows, s], [s, heads * dk], pv.clone(), [s, s, dk]);
                    (p.units, p.pipeline, p.padded) = (units, pipeline, [true, false]);
                    p.run(out);
                }
            }
            // `K` transposed by the mover rather than the unpacker.
            let mut p = Batched::new([s, heads * dk], [s, heads * dk], qk.clone(), [s, dk, s]);
            p.unpack_b = false;
            p.run(out);
        }
        // One row a head (decode): `q K^T` over a cache of `t` positions, then `p V`.
        for t in [64, 300] {
            let heads = 3;
            let qk: Vec<_> = (0..heads)
                .map(|h| (block([0, h * 64], false), block([0, h * 64], true)))
                .collect();
            let pv: Vec<_> = (0..heads)
                .map(|h| (block([h * 32, 0], false), block([0, h * 64], false)))
                .collect();
            for units in [8, 64] {
                let mut p = Batched::new([1, heads * 64], [t, heads * 64], qk.clone(), [1, 64, t]);
                p.units = units;
                p.run(out);
                let mut p = Batched::new([heads * 32, t], [t, heads * 64], pv.clone(), [1, t, 64]);
                (p.units, p.padded) = (units, [true, false]);
                p.run(out);
            }
        }
        // A long prefix: `P V` with `K` past one block, by the block-stream model, fixed blocks and
        // off, and `K` blocked by a limit.
        let pv: Vec<_> = (0..2)
            .map(|h| (block([h * 512, 0], false), block([0, h * 64], false)))
            .collect();
        for (blocks, limit) in [
            (BlockStream::Auto, None),
            (BlockStream::Fixed([4, 8, 2]), None),
            (BlockStream::Off, None),
            (BlockStream::Auto, Some(16)),
        ] {
            for units in [8, 120] {
                let mut p = Batched::new([1024, 2048], [2048, 128], pv.clone(), [512, 2048, 64]);
                (p.units, p.block_stream, p.max_k_tiles) = (units, blocks, limit);
                p.run(out);
            }
        }
        // A batch of one, and refusals: a block past its tensor, and nothing to compute.
        Batched::new(
            [96, 320],
            [320, 288],
            vec![(block([0, 0], false), block([0, 0], false))],
            [96, 320, 288],
        )
        .run(out);
        Batched::new(
            [64, 64],
            [64, 64],
            vec![(block([0, 32], false), block([0, 0], false))],
            [64, 64, 64],
        )
        .run(out);
        Batched::new([64, 64], [64, 64], Vec::new(), [64, 64, 64]).run(out);
    });
}
