//! `cargo xtask pipeline-report`: a decode step's products, unit against unit, from a profile.
//!
//! Reads `skew.tsv` (and `pipeline.tsv`, when present, for batch counts) from a profile directory
//! written with `TT_PROFILE_DETAIL=pipeline` (`burn_tt::profile`) and prints, per product class,
//! the median over its layers of: the slowest unit's work, the mean unit's, and what two re-deals
//! of the same in-span bytes would take. This is the method roadmap item 2b's plan
//! (`docs/plans/decode-bytes-in-flight.md`, "Conditions and the baseline, re-read") describes in
//! prose, as code, and X10's bound.
//!
//! **Classes.** A product is an op whose label starts with `float_matmul`, in index order within
//! one source (`Step(64)` by default). The first and the last (a decoder's layer 0 `qkv`, fused
//! with its rope, and the head) are classes of their own. The rest repeat with the layers: their
//! period is the shortest one under which at most a tenth of the ops differ in `what` or unit
//! count from the op one period before, and each position in the period is a class. `--names` names the positions
//! (the SmolLM2 / Llama order is `o,gate|up,down,qkv`); nothing here knows a model.
//!
//! **The model of a unit's work** (per op, every unit with a line): `work = a + b * span_mb`, a
//! least-squares line through the units, where `span_mb` is what the unit read in the op's own
//! span (not ahead of it). Then:
//! - *even*: every unit given the same in-span bytes, `a + b * mean(span_mb)`;
//! - *ideal*: in-span bytes dealt in proportion to each unit's own rate (a unit that read at least
//!   half the largest share is a reader, its cost a MB `(work - a) / span_mb`; others take the
//!   readers' median), every unit finishing together: `a + total / sum(1 / cost)`, and never
//!   faster than the card reads (`--peak-gbps`, 512 by default): `total / peak`.
//! - *ideal, whole columns*: the same, but the bytes dealt in whole columns (the granule every
//!   unit's share is a multiple of, inferred), each to the unit it would finish soonest on;
//! - *saving*: the slowest unit less the ideal (in whole columns where a column shows), summed
//!   over the class's ops: what a perfect X10 deal could take off a step at today's rates, before
//!   anything it changes elsewhere. Rates measured under one deal's contention need not hold under
//!   another: this bounds, the probe measures.
//!
//! **What explains the spread** (X10's bound, rule 3's attribution):
//! - *share*: the line's R² (how much of the units' spread their bytes explain);
//! - *position*: among readers with the same share, the rank correlation of each unit's residual
//!   between consecutive ops of the class (each layer's weight is dealt alike, so a unit that is
//!   slow in every layer is slow by where it is or which channels it reads, not by chance);
//! - *chance*: the readers' median p10..p90 work over replays (`skew.tsv`'s last columns) beside
//!   the spread of residuals between readers.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

pub const USAGE: &str = "\
usage: cargo xtask pipeline-report <profile dir> [options]

  A decode step's products, unit against unit, from a TT_PROFILE_DETAIL=pipeline
  profile (the directory holding skew.tsv, or one above it: every skew.tsv below
  is reported). Prints each product class's median over its layers of the slowest
  and mean unit's work, an even and a rate-weighted re-deal of the same bytes, and
  what explains the spread (share, position, chance).

options:
  --source <name>       the trace to read (default Step(64))
  --names <a,b,..>      names for the repeating classes in order, e.g.
                        o,gate|up,down,qkv for a SmolLM2 / Llama decoder
  --units <class>       also list that class's units: each one's median over the
                        class's ops
  --peak-gbps <rate>    the card's read rate the ideal may not beat (default 512)
  --column <class>=<MB> that class's column, in place of the one inferred from its
                        shares (repeatable): a share read partly ahead looks like
                        a smaller column than any deal can move";

/// What the re-deals assume beyond the profile.
#[derive(Clone, Debug)]
struct Model {
    peak_gbps: f64,
    /// A class's column, MB, where its shares do not show it.
    columns: BTreeMap<String, f64>,
}

/// The card's GDDR read rate the ideal may not beat, GB/s: `tt_kernels::profile_mode`'s
/// `CARD_READ_GBPS` (8 channels of GDDR6 at 16 Gb/s a pin on a p150).
const PEAK_GBPS: f64 = 512.0;

pub fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args: Vec<String> = args.collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        return Err(USAGE.into());
    }
    let mut dir = None;
    let mut source = "Step(64)".to_string();
    let mut names = Vec::new();
    let mut units = None;
    let mut model = Model {
        peak_gbps: PEAK_GBPS,
        columns: BTreeMap::new(),
    };
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let mut value = |flag: &str| it.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--source" => source = value("--source")?,
            "--names" => {
                names = value("--names")?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect()
            }
            "--units" => units = Some(value("--units")?),
            "--column" => {
                let v = value("--column")?;
                let (class, mb) = v
                    .rsplit_once('=')
                    .ok_or_else(|| format!("--column {v}: not <class>=<MB>"))?;
                let mb: f64 = mb.parse().map_err(|e| format!("--column {v}: {e}"))?;
                model.columns.insert(class.to_string(), mb);
            }
            "--peak-gbps" => {
                model.peak_gbps = value("--peak-gbps")?
                    .parse()
                    .map_err(|e| format!("--peak-gbps: {e}"))?
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n\n{USAGE}")),
            _ if dir.is_none() => dir = Some(PathBuf::from(a)),
            _ => return Err(format!("one directory only\n\n{USAGE}")),
        }
    }
    let dir = dir.ok_or(USAGE)?;
    let mut found = Vec::new();
    find_skews(&dir, 4, &mut found);
    if found.is_empty() {
        return Err(format!("no skew.tsv in or below {}", dir.display()));
    }
    found.sort();
    for d in found {
        let read = |f: &str| std::fs::read_to_string(d.join(f));
        let skew = read("skew.tsv").map_err(|e| format!("{}: {e}", d.display()))?;
        let pipeline = read("pipeline.tsv").ok();
        let ops = parse(&skew, pipeline.as_deref(), &source)?;
        println!("## {}\n", d.display());
        print!("{}", report(&ops, &names, units.as_deref(), &model));
    }
    Ok(())
}

/// Every directory at most `depth` below `dir` (itself included) that holds a `skew.tsv`.
fn find_skews(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if dir.join("skew.tsv").is_file() {
        out.push(dir.to_path_buf());
        return;
    }
    if depth == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        if e.path().is_dir() {
            find_skews(&e.path(), depth - 1, out);
        }
    }
}

/// One unit's line on one op.
#[derive(Clone, Debug, PartialEq)]
struct Unit {
    unit: u16,
    work: f64,
    span_mb: f64,
    /// `(p10, p90)` of `work` over the replays, where the profile has them.
    spread: Option<(f64, f64)>,
    /// Its streamed batches in the replay `pipeline.tsv` holds, where it has them.
    batches: Option<u32>,
}

/// One product op of the source.
#[derive(Clone, Debug, PartialEq)]
struct Op {
    index: u32,
    what: String,
    units: Vec<Unit>,
}

/// The source's product ops, in index order, each with its units.
fn parse(skew: &str, pipeline: Option<&str>, source: &str) -> Result<Vec<Op>, String> {
    let mut lines = skew.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or("skew.tsv is empty")?
        .split('\t')
        .collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| format!("skew.tsv has no `{name}` column"))
    };
    let [c_source, c_index, c_label, c_what, c_unit, c_work, c_span] = [
        "source", "index", "label", "what", "unit", "work_us", "span_mb",
    ]
    .map(col);
    let (c_source, c_index, c_label, c_what, c_unit, c_work, c_span) = (
        c_source?, c_index?, c_label?, c_what?, c_unit?, c_work?, c_span?,
    );
    let c_p10 = col("work_p10_us").ok();
    let c_p90 = col("work_p90_us").ok();
    let batches = pipeline.map(|p| batch_counts(p, source)).transpose()?;
    let mut ops: BTreeMap<u32, Op> = BTreeMap::new();
    for (n, line) in lines.enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < header.len()
            || f[c_source] != source
            || !f[c_label].starts_with("float_matmul")
        {
            continue;
        }
        let at = |what: &str| format!("skew.tsv line {}: {what}", n + 2);
        let num = |c: usize| -> Result<f64, String> {
            f[c].parse().map_err(|e| at(&format!("{}: {e}", header[c])))
        };
        let index: u32 = f[c_index].parse().map_err(|e| at(&format!("index: {e}")))?;
        let unit: u16 = f[c_unit].parse().map_err(|e| at(&format!("unit: {e}")))?;
        let spread = match (c_p10, c_p90) {
            (Some(a), Some(b)) => Some((num(a)?, num(b)?)),
            _ => None,
        };
        let op = ops.entry(index).or_insert_with(|| Op {
            index,
            what: f[c_what].to_string(),
            units: Vec::new(),
        });
        op.units.push(Unit {
            unit,
            work: num(c_work)?,
            span_mb: num(c_span)?,
            spread,
            batches: batches
                .as_ref()
                .and_then(|b| b.get(&(index, unit)).copied()),
        });
    }
    Ok(ops.into_values().filter(|o| !o.units.is_empty()).collect())
}

/// `(op index, unit)` -> its batches in `pipeline.tsv` (one line a batch).
fn batch_counts(pipeline: &str, source: &str) -> Result<BTreeMap<(u32, u16), u32>, String> {
    let mut out = BTreeMap::new();
    for (n, line) in pipeline.lines().enumerate().skip(1) {
        let mut f = line.split('\t');
        if f.next() != Some(source) {
            continue;
        }
        let index = f.next().and_then(|s| s.parse().ok());
        let unit = f.nth(2).and_then(|s| s.parse().ok());
        let (Some(index), Some(unit)) = (index, unit) else {
            return Err(format!("pipeline.tsv line {}: no index or unit", n + 1));
        };
        *out.entry((index, unit)).or_default() += 1;
    }
    Ok(out)
}

/// Each product op's class: `first`, `last`, or its position in the layers' period, named.
fn classes(ops: &[Op], names: &[String]) -> Vec<String> {
    let n = ops.len();
    if n <= 2 {
        return (0..n).map(|i| format!("op {}", ops[i].index)).collect();
    }
    let middle = &ops[1..n - 1];
    let key = |o: &Op| (o.what.clone(), o.units.len());
    // The shortest period under which at most a tenth of the ops differ from the op a period
    // before (a layer whose product the session put on fewer units is still that product).
    let differ = |p: usize| {
        (p..middle.len())
            .filter(|&i| key(&middle[i]) != key(&middle[i - p]))
            .count()
    };
    let period = (1..=middle.len() / 2)
        .find(|&p| differ(p) * 10 <= middle.len())
        .unwrap_or(middle.len());
    let slot = |k: usize| {
        names.get(k).cloned().unwrap_or_else(|| {
            let (what, units) = key(&middle[k]);
            format!("slot {k} ({what}, {units} units)")
        })
    };
    let mut out = vec!["first".to_string()];
    out.extend((0..middle.len()).map(|i| slot(i % period)));
    out.push("last".to_string());
    out
}

/// One op's units against each other ([the module](self)'s model).
#[derive(Clone, Debug, PartialEq)]
struct Fit {
    slowest: f64,
    slowest_unit: u16,
    mean: f64,
    total_mb: f64,
    /// The line `work = a + b * span_mb`, and its R² (`None` where every unit read alike).
    a: f64,
    b: f64,
    r2: Option<f64>,
    even: f64,
    ideal: f64,
    /// The ideal dealt in whole shares of `granule` MB (a column), `None` where no granule shows.
    whole: Option<f64>,
    granule: Option<f64>,
    /// Each unit's work less the line's.
    residuals: BTreeMap<u16, f64>,
    /// The units whose share is the most common one (within 1%), for the rank correlation.
    modal: Vec<u16>,
}

fn fit(op: &Op, peak_gbps: f64, column: Option<f64>) -> Fit {
    let u = &op.units;
    let n = u.len() as f64;
    let mean_x = u.iter().map(|x| x.span_mb).sum::<f64>() / n;
    let mean = u.iter().map(|x| x.work).sum::<f64>() / n;
    let sxx: f64 = u.iter().map(|x| (x.span_mb - mean_x).powi(2)).sum();
    let sxy: f64 = u
        .iter()
        .map(|x| (x.span_mb - mean_x) * (x.work - mean))
        .sum();
    let syy: f64 = u.iter().map(|x| (x.work - mean).powi(2)).sum();
    // A spread of shares under a thousandth of a MB is no spread: the line is the mean.
    let (a, b, r2) = if sxx > 1e-6 * n {
        let b = sxy / sxx;
        let r2 = if syy > 0.0 {
            sxy * sxy / (sxx * syy)
        } else {
            1.0
        };
        (mean - b * mean_x, b, Some(r2))
    } else {
        (mean, 0.0, None)
    };
    let total_mb = mean_x * n;
    let (slowest, slowest_unit) = u
        .iter()
        .map(|x| (x.work, x.unit))
        .fold((f64::MIN, 0), |m, x| if x.0 > m.0 { x } else { m });
    let even = a + b * mean_x;
    // Rate-weighted: each reader's own cost a MB above the shared overhead.
    let base = a.max(0.0);
    let largest = u.iter().map(|x| x.span_mb).fold(0.0, f64::max);
    let mut costs: Vec<f64> = u
        .iter()
        .filter(|x| largest > 0.0 && x.span_mb >= 0.5 * largest)
        .map(|x| ((x.work - base) / x.span_mb).max(f64::MIN_POSITIVE))
        .collect();
    // No deal reads faster than the card: MB over GB/s is ms.
    let floor = total_mb / peak_gbps * 1e3;
    let reader = |x: &Unit| largest > 0.0 && x.span_mb >= 0.5 * largest;
    let (ideal, whole, granule) = if costs.is_empty() || total_mb <= 0.0 {
        (even, None, None)
    } else {
        costs.sort_by(f64::total_cmp);
        let median = costs[costs.len() / 2];
        let others = u.len() - costs.len();
        let speed: f64 = costs.iter().map(|c| 1.0 / c).sum::<f64>() + others as f64 / median;
        let ideal = (base + total_mb / speed).max(floor);
        // Whole shares: each in turn to the unit it would finish soonest on, at its own cost a
        // MB (a reader's, or the readers' median).
        let granule = column.or_else(|| granule(u.iter().map(|x| x.span_mb), largest));
        let whole = granule.map(|g| {
            let cost: Vec<f64> = u
                .iter()
                .map(|x| {
                    if reader(x) {
                        ((x.work - base) / x.span_mb).max(f64::MIN_POSITIVE)
                    } else {
                        median
                    }
                })
                .collect();
            let mut held = vec![0u32; u.len()];
            for _ in 0..(total_mb / g).round() as u64 {
                let next = |i: usize| base + f64::from(held[i] + 1) * g * cost[i];
                let i = (0..u.len())
                    .min_by(|&i, &j| next(i).total_cmp(&next(j)))
                    .expect("units");
                held[i] += 1;
            }
            (0..u.len())
                .map(|i| base + f64::from(held[i]) * g * cost[i])
                .fold(floor, f64::max)
        });
        (ideal, whole, granule)
    };
    let residuals = u
        .iter()
        .map(|x| (x.unit, x.work - (a + b * x.span_mb)))
        .collect();
    // The most common share, by count within 1% of the largest.
    let tol = 0.01 * largest.max(1e-9);
    let modal_x = u
        .iter()
        .map(|x| x.span_mb)
        .max_by_key(|&s| u.iter().filter(|y| (y.span_mb - s).abs() <= tol).count())
        .unwrap_or(0.0);
    let modal = u
        .iter()
        .filter(|y| (y.span_mb - modal_x).abs() <= tol)
        .map(|y| y.unit)
        .collect();
    Fit {
        slowest,
        slowest_unit,
        mean,
        total_mb,
        a,
        b,
        r2,
        even,
        ideal,
        whole,
        granule,
        residuals,
        modal,
    }
}

/// The share every unit's in-span MB is a whole number of (a column): the least of the positive
/// shares and of the gaps between distinct shares, each at least a hundredth of the largest, if
/// every share is within 2% of a whole number of it.
fn granule(shares: impl Iterator<Item = f64>, largest: f64) -> Option<f64> {
    let tol = 0.01 * largest;
    let mut distinct: Vec<f64> = Vec::new();
    for s in shares {
        if !distinct.iter().any(|d| (d - s).abs() <= tol) {
            distinct.push(s);
        }
    }
    distinct.sort_by(f64::total_cmp);
    let g = distinct
        .iter()
        .filter(|&&s| s > tol)
        .copied()
        .chain(distinct.windows(2).map(|w| w[1] - w[0]))
        .filter(|&d| d > tol)
        .fold(f64::INFINITY, f64::min);
    (g.is_finite()
        && distinct
            .iter()
            .all(|s| ((s / g) - (s / g).round()).abs() <= 0.02 * (s / g).max(1.0)))
    .then_some(g)
}

/// Spearman's rank correlation of two equally long samples (ties take their mean rank).
fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let ranks = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0.0; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            for &k in &idx[i..=j] {
                r[k] = (i + j) as f64 / 2.0;
            }
            i = j + 1;
        }
        r
    };
    let (rx, ry) = (ranks(x), ranks(y));
    let n = x.len() as f64;
    let m = (n - 1.0) / 2.0;
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - m) * (b - m)).sum();
    let var = |r: &[f64]| r.iter().map(|a| (a - m).powi(2)).sum::<f64>();
    let d = (var(&rx) * var(&ry)).sqrt();
    (d > 0.0).then(|| cov / d)
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The report: a table of classes, then (with `units`) that class's units.
fn report(ops: &[Op], names: &[String], units: Option<&str>, model: &Model) -> String {
    let mut out = String::new();
    if ops.is_empty() {
        out.push_str("no products in this source\n\n");
        return out;
    }
    let class = classes(ops, names);
    let mut by: BTreeMap<&str, Vec<(&Op, Fit)>> = BTreeMap::new();
    let mut order: Vec<&str> = Vec::new();
    for (op, c) in ops.iter().zip(&class) {
        if !by.contains_key(c.as_str()) {
            order.push(c);
        }
        let column = model.columns.get(c.as_str()).copied();
        by.entry(c)
            .or_default()
            .push((op, fit(op, model.peak_gbps, column)));
    }
    let _ = writeln!(
        out,
        "Medians over each class's ops (us; MB read in the op's own span). saving: slowest less \
         ideal (in whole columns where a column shows), summed over the class's ops, ms a step.\n\n\
         | class | ops | units | batches a unit | MB | slowest | mean | slowest / mean | even | \
         ideal | column MB | ideal, whole columns | saving ms | R² share | rank corr. position | \
         replay p10..p90 | residual p10..p90 | slowest most often |\n\
         |---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"
    );
    let mut total_saving = 0.0;
    for c in &order {
        let rows = &by[c];
        let m = |f: &dyn Fn(&Fit) -> f64| median(rows.iter().map(|(_, x)| f(x)).collect());
        let saving: f64 = rows
            .iter()
            .map(|(_, x)| x.slowest - x.whole.unwrap_or(x.ideal))
            .sum::<f64>()
            / 1e3;
        let whole = median(rows.iter().filter_map(|(_, x)| x.whole).collect());
        let granule = median(rows.iter().filter_map(|(_, x)| x.granule).collect());
        total_saving += saving;
        let r2 = median(rows.iter().filter_map(|(_, x)| x.r2).collect());
        // Consecutive ops' residuals over the units of the common share in both.
        let corr = median(
            rows.windows(2)
                .filter_map(|w| {
                    let (p, q) = (&w[0].1, &w[1].1);
                    let common: Vec<u16> = p
                        .modal
                        .iter()
                        .copied()
                        .filter(|u| q.modal.contains(u))
                        .collect();
                    let x: Vec<f64> = common.iter().map(|u| p.residuals[u]).collect();
                    let y: Vec<f64> = common.iter().map(|u| q.residuals[u]).collect();
                    spearman(&x, &y)
                })
                .collect(),
        );
        // Chance: a modal unit's own spread over replays, against the spread between them.
        let replay = median(
            rows.iter()
                .flat_map(|(op, f)| {
                    op.units
                        .iter()
                        .filter(|x| f.modal.contains(&x.unit))
                        .filter_map(|x| x.spread.map(|(lo, hi)| hi - lo))
                })
                .collect(),
        );
        let between = median(
            rows.iter()
                .map(|(_, f)| {
                    let mut r: Vec<f64> = f.modal.iter().map(|u| f.residuals[u]).collect();
                    r.sort_by(f64::total_cmp);
                    let at = |q: f64| r[((r.len() - 1) as f64 * q).round() as usize];
                    if r.is_empty() {
                        f64::NAN
                    } else {
                        at(0.9) - at(0.1)
                    }
                })
                .collect(),
        );
        let batches = median(
            rows.iter()
                .filter_map(|(op, _)| {
                    let b: Vec<f64> = op
                        .units
                        .iter()
                        .filter_map(|x| x.batches.map(f64::from))
                        .collect();
                    (!b.is_empty()).then(|| b.iter().sum::<f64>() / b.len() as f64)
                })
                .collect(),
        );
        let mut often: BTreeMap<u16, u32> = BTreeMap::new();
        for (_, f) in rows {
            *often.entry(f.slowest_unit).or_default() += 1;
        }
        let mut often: Vec<_> = often.into_iter().collect();
        often.sort_by_key(|&(u, n)| (std::cmp::Reverse(n), u));
        let often: Vec<String> = often
            .iter()
            .take(4)
            .map(|(u, n)| format!("{u} x{n}"))
            .collect();
        let opt = |v: f64, digits: usize| {
            if v.is_nan() {
                "-".to_string()
            } else {
                format!("{v:.digits$}")
            }
        };
        let _ = writeln!(
            out,
            "| {c} | {} | {} | {} | {:.2} | {:.1} | {:.1} | {:.2} | {:.1} | {:.1} | {} | {} | {:.3} | {} | {} | {} | {} | {} |",
            rows.len(),
            median(rows.iter().map(|(op, _)| op.units.len() as f64).collect()),
            opt(batches, 2),
            m(&|f| f.total_mb),
            m(&|f| f.slowest),
            m(&|f| f.mean),
            m(&|f| f.slowest / f.mean),
            m(&|f| f.even),
            m(&|f| f.ideal),
            opt(granule, 3),
            opt(whole, 1),
            saving,
            opt(r2, 2),
            opt(corr, 2),
            opt(replay, 1),
            opt(between, 1),
            often.join(", "),
        );
    }
    let _ = writeln!(out, "\nAll classes' saving: {total_saving:.3} ms a step.\n");
    if let Some(name) = units {
        let Some(rows) = by.get(name) else {
            let _ = writeln!(out, "no class `{name}` (classes: {})\n", order.join(", "));
            return out;
        };
        let _ = writeln!(
            out,
            "Class `{name}`'s units, each a median over its {} ops:\n\n\
             | unit | MB | work us | residual us | replay p10..p90 us | batches | slowest in |\n\
             |--:|--:|--:|--:|--:|--:|--:|",
            rows.len()
        );
        /// One unit's lines over the class's ops.
        #[derive(Default)]
        struct Lines {
            mb: Vec<f64>,
            work: Vec<f64>,
            residual: Vec<f64>,
            spread: Vec<f64>,
            batches: Vec<f64>,
            slowest: u32,
        }
        let mut per: BTreeMap<u16, Lines> = BTreeMap::new();
        for (op, f) in rows {
            for x in &op.units {
                let e = per.entry(x.unit).or_default();
                e.mb.push(x.span_mb);
                e.work.push(x.work);
                e.residual.push(f.residuals[&x.unit]);
                if let Some((lo, hi)) = x.spread {
                    e.spread.push(hi - lo);
                }
                if let Some(b) = x.batches {
                    e.batches.push(f64::from(b));
                }
                if f.slowest_unit == x.unit {
                    e.slowest += 1;
                }
            }
        }
        for (u, l) in per {
            let Lines {
                mb,
                work,
                residual: res,
                spread,
                batches,
                slowest,
            } = l;
            let opt = |v: f64| {
                if v.is_nan() {
                    "-".to_string()
                } else {
                    format!("{v:.1}")
                }
            };
            let _ = writeln!(
                out,
                "| {u} | {:.3} | {:.1} | {:+.1} | {} | {} | {slowest} |",
                median(mb),
                median(work),
                median(res),
                opt(median(spread)),
                opt(median(batches)),
            );
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "source\tindex\tlabel\twhat\tunit\tsamples\twork_us\twait_us\tspan_mb\t\
                          ahead_mb\tspan_gbps\tpush_us\tdrain_us\tpush_words\twork_p10_us\twork_p90_us";

    fn line(index: u32, label: &str, what: &str, unit: u16, work: f64, mb: f64) -> String {
        format!(
            "Step(64)\t{index}\t{label}\t{what}\t{unit}\t5\t{work}\t0\t{mb}\t0\t0\t0\t0\t0\t{}\t{}",
            work - 1.0,
            work + 1.0
        )
    }

    /// Two layers of `o` (2 units), `gate|up` (4 units), between a first and a last product, and
    /// an op that is no product.
    fn skew() -> String {
        let mut s = vec![HEADER.to_string()];
        s.push(line(0, "float_matmul + rope", "first", 0, 10.0, 1.0));
        s.push(line(1, "rms_norm", "norm", 0, 99.0, 0.0));
        for (base, slow) in [(2, 0u16), (4, 1)] {
            s.push(line(base, "float_matmul", "o", 0, 20.0, 1.0));
            s.push(line(base, "float_matmul", "o", 1, 20.0, 1.0));
            // gate|up: units 0-2 read 2 MB, unit 3 reads none; one reader is slower.
            for u in 0..4u16 {
                let (work, mb) = match u {
                    3 => (5.0, 0.0),
                    _ if u == slow => (45.0, 2.0),
                    _ => (25.0, 2.0),
                };
                s.push(line(base + 1, "float_matmul", "gu", u, work, mb));
            }
        }
        s.push(line(9, "float_matmul", "head", 0, 50.0, 3.0));
        s.push(line(0, "float_matmul", "first", 0, 1.0, 1.0).replace("Step(64)", "Step(128)"));
        s.join("\n")
    }

    #[test]
    fn products_are_read_from_their_source_and_classed_by_the_layers_period() {
        let ops = parse(&skew(), None, "Step(64)").unwrap();
        assert_eq!(
            ops.iter().map(|o| o.index).collect::<Vec<_>>(),
            [0, 2, 3, 4, 5, 9],
            "products only, from Step(64) only"
        );
        assert_eq!(
            ops[2].units[0].spread,
            Some((44.0, 46.0)),
            "the slow unit, work 45"
        );
        let names = ["o".to_string(), "gate|up".to_string()];
        assert_eq!(
            classes(&ops, &names),
            ["first", "o", "gate|up", "o", "gate|up", "last"]
        );
        assert_eq!(classes(&ops, &[])[1], "slot 0 (o, 2 units)");
        // Ten layers, one of whose `o` ran on one unit: still `o`.
        let op = |index, what: &str, units: u16| Op {
            index,
            what: what.into(),
            units: (0..units)
                .map(|unit| Unit {
                    unit,
                    work: 1.0,
                    span_mb: 0.0,
                    spread: None,
                    batches: None,
                })
                .collect(),
        };
        let mut many = vec![op(0, "first", 1)];
        for l in 0..10u32 {
            many.push(op(2 * l + 1, "o", if l == 6 { 1 } else { 2 }));
            many.push(op(2 * l + 2, "gu", 3));
        }
        many.push(op(99, "head", 3));
        let c = classes(&many, &names);
        assert_eq!(c.iter().filter(|c| *c == "o").count(), 10, "{c:?}");
        assert_eq!(c.iter().filter(|c| *c == "gate|up").count(), 10);
    }

    #[test]
    fn the_even_and_rate_weighted_deals_follow_from_the_units_line() {
        let unit = |unit, work, span_mb| Unit {
            unit,
            work,
            span_mb,
            spread: None,
            batches: None,
        };
        let op = Op {
            index: 3,
            what: "gu".into(),
            units: vec![unit(0, 45.0, 2.0), unit(1, 25.0, 2.0), unit(2, 5.0, 0.0)],
        };
        let f = fit(&op, PEAK_GBPS, None); // works 45, 25, 5 at 2, 2, 0 MB
        assert_eq!((f.slowest, f.slowest_unit), (45.0, 0));
        assert!((f.mean - 25.0).abs() < 1e-9);
        // The line: x = 2, 2, 0 and y = 45, 25, 5: b = 15 us a MB, a = 5 us.
        assert!(
            (f.b - 15.0).abs() < 1e-9 && (f.a - 5.0).abs() < 1e-9,
            "{f:?}"
        );
        // Even: 4 MB over three units at the line.
        assert!((f.even - (5.0 + 15.0 * 4.0 / 3.0)).abs() < 1e-9);
        // Ideal: readers cost 20 and 10 us a MB above 5 us; unit 2 takes their median (the
        // upper of the two, 20). 4 MB / (1/20 + 1/10 + 1/20) + 5 = 25 us.
        assert!((f.ideal - 25.0).abs() < 1e-9, "{}", f.ideal);
        // Whole 2 MB columns: the first to unit 1 (5 + 2 x 10 = 25 us); the second costs 45 us
        // anywhere (unit 1 again, or 20 us a MB on unit 0 or on unit 2 at the median): whole
        // columns cannot beat today's 45 here, though the fluid ideal says 25.
        assert_eq!(f.granule, Some(2.0));
        assert!((f.whole.unwrap() - 45.0).abs() < 1e-9, "{f:?}");
        // A card that reads 0.1 GB/s takes 4 MB in 40 ms, not 25 us.
        assert!((fit(&op, 0.1, None).ideal - 40_000.0).abs() < 1e-6);
        // A column of 4 MB named: one column, on unit 1 (5 + 4 x 10 = 45 us).
        let named = fit(&op, PEAK_GBPS, Some(4.0));
        assert_eq!(named.granule, Some(4.0));
        assert!((named.whole.unwrap() - 45.0).abs() < 1e-9, "{named:?}");
        assert_eq!(f.modal, [0, 1]);
        assert!((f.residuals[&0] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn the_report_names_the_slow_unit_and_its_saving_and_lists_a_class() {
        let ops = parse(&skew(), None, "Step(64)").unwrap();
        let names = ["o".to_string(), "gate|up".to_string()];
        let model = Model {
            peak_gbps: PEAK_GBPS,
            columns: BTreeMap::new(),
        };
        let r = report(&ops, &names, Some("gate|up"), &model);
        // gate|up: works 45, 25, 25, 5 at 2, 2, 2, 0 MB: the line is 5 + 40/3 us a MB, the
        // readers cost 20, 10 and 10 us a MB above 5, unit 3 takes their median (10), so the
        // fluid ideal is 5 + 6 / (1/20 + 3/10) = 22.1 us; in whole 2 MB columns, one each to the
        // three 10 us a MB units, 25 us, and the saving 2 x 20 us = 0.040 ms.
        let row = r.lines().find(|l| l.starts_with("| gate|up |")).unwrap();
        assert!(row.contains("| 22.1 | 2.000 | 25.0 | 0.040 |"), "{row}");
        // Unit 0 is slow in layer 0 and unit 1 in layer 1: residual ranks (2, 0.5, 0.5) against
        // (0.5, 2, 0.5) correlate at -0.5, no stable position.
        assert!(row.contains("| -0.50 |"), "{row}");
        assert!(row.contains("0 x1, 1 x1"), "{row}");
        // `o` reads alike on every unit: no line, no saving.
        let o = r.lines().find(|l| l.starts_with("| o |")).unwrap();
        assert!(o.contains("| 0.000 |") && o.contains("| - |"), "{o}");
        assert!(r.contains("| 3 | 0.000 | 5.0 |"), "unit 3 listed:\n{r}");
    }

    #[test]
    fn batch_counts_are_lines_per_unit_and_op() {
        let p = "source\tindex\tlabel\twhat\tunit\tbatch\n\
                 Step(64)\t3\tfloat_matmul\tgu\t1\t0\n\
                 Step(64)\t3\tfloat_matmul\tgu\t1\t1\n\
                 Step(128)\t3\tfloat_matmul\tgu\t1\t0\n";
        let b = batch_counts(p, "Step(64)").unwrap();
        assert_eq!(b.get(&(3, 1)), Some(&2));
        let ops = parse(&skew(), Some(p), "Step(64)").unwrap();
        assert_eq!(ops[2].units[1].batches, Some(2));
        assert_eq!(ops[2].units[0].batches, None);
        assert_eq!(classes(&ops, &[]).len(), ops.len());
    }

    #[test]
    fn a_column_is_the_least_share_or_gap_that_divides_every_share() {
        let g = granule([0.406, 0.270, 0.0, 0.406].into_iter(), 0.406).unwrap();
        assert!((g - 0.136).abs() < 1e-9, "{g}");
        assert_eq!(granule([2.0, 3.0].into_iter(), 3.0), Some(1.0));
        // 1.0 and 1.5 leave a gap of 0.5, which divides both; 1.0 and 1.3 leave none that does.
        assert_eq!(granule([1.0, 1.5].into_iter(), 1.5), Some(0.5));
        assert_eq!(granule([1.0, 1.3].into_iter(), 1.3), None);
        assert_eq!(granule([0.0, 0.0].into_iter(), 0.0), None);
    }

    #[test]
    fn spearman_is_one_for_the_same_order_and_minus_one_reversed() {
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]), Some(1.0));
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), Some(-1.0));
        assert_eq!(spearman(&[1.0, 2.0], &[1.0, 2.0]), None);
    }
}
