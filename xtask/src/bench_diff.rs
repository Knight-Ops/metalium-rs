//! `cargo xtask bench-diff`: two `cargo xtask bench` collections side by side.
//!
//! Each `target/silicon/bench/<stamp>.jsonl` line is one `BENCH` record with its test. Results are
//! joined by test and key (a key repeated within a test by its occurrence), and each pair is
//! printed with its ratio. A pair whose p10..p90 spreads do not overlap has moved; one whose spreads
//! overlap is within the benchmark's own noise. The noise band is the samples' spread, never a
//! chosen tolerance.

use std::collections::BTreeMap;

pub const USAGE: &str = "\
usage: cargo xtask bench-diff <before.jsonl> <after.jsonl>

  Joins two `cargo xtask bench` collections (target/silicon/bench/<stamp>.jsonl)
  by test and key and prints a Markdown table: each result's median before and
  after, the ratio, and whether it moved (p10..p90 spreads apart) or is within
  noise (spreads overlap). Results in only one collection are listed after it.";

/// One result record: its median and spread, in `unit`.
#[derive(Clone, Debug, PartialEq)]
struct Result_ {
    unit: String,
    median: f64,
    p10: f64,
    p90: f64,
}

type Results = BTreeMap<(String, String), Result_>;

pub fn run(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args: Vec<String> = args.collect();
    if args.len() != 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        return Err(USAGE.into());
    }
    let read = |p: &str| std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"));
    let before = parse(&read(&args[0])?)?;
    let after = parse(&read(&args[1])?)?;
    print!("{}", table(&args[0], &args[1], &before, &after));
    Ok(())
}

/// Every `result` record of a collection, keyed by (test, key).
fn parse(text: &str) -> Result<Results, String> {
    let mut out = Results::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let at = |what: &str| format!("line {}: {what}", n + 1);
        let test = crate::silicon::json_field(line, "test").ok_or_else(|| at("no test"))?;
        let rec_at = line.find("\"record\":").ok_or_else(|| at("no record"))?;
        let rec = &line[rec_at + "\"record\":".len()..];
        if crate::silicon::json_field(rec, "kind").as_deref() != Some("result") {
            continue;
        }
        let field = |k: &str| crate::silicon::json_field(rec, k).ok_or_else(|| at(k));
        let number = |k: &str| -> Result<f64, String> {
            field(k)?.parse().map_err(|e| at(&format!("{k}: {e}")))
        };
        let key = field("key")?;
        let r = Result_ {
            unit: field("unit")?,
            median: number("median")?,
            p10: number("p10")?,
            p90: number("p90")?,
        };
        // A key a test reports twice (a repeated configuration) is told apart by occurrence.
        let mut k = (test.clone(), key.clone());
        let mut i = 1;
        while out.contains_key(&k) {
            i += 1;
            k = (test.clone(), format!("{key} #{i}"));
        }
        out.insert(k, r);
    }
    Ok(out)
}

/// Whether a larger value of `unit` is better (a rate) rather than worse (a time).
fn higher_is_better(unit: &str) -> Option<bool> {
    let u = unit.to_ascii_lowercase();
    if u.contains("/s") || u.contains("flops") || u.contains("tok") || u.contains("gbps") {
        Some(true)
    } else if ["ns", "us", "ms", "s", "cycles"]
        .iter()
        .any(|t| u == *t || u.starts_with(&format!("{t}/")))
    {
        Some(false)
    } else {
        None
    }
}

fn verdict(b: &Result_, a: &Result_) -> &'static str {
    if a.p10 <= b.p90 && b.p10 <= a.p90 {
        return "noise";
    }
    match higher_is_better(&a.unit) {
        Some(up) if (a.median > b.median) == up => "better",
        Some(_) => "**worse**",
        None => "moved",
    }
}

fn table(before_name: &str, after_name: &str, before: &Results, after: &Results) -> String {
    let short = |t: &str| t.rsplit("::").next().unwrap_or(t).to_string();
    let mut s = format!(
        "before: `{before_name}`\nafter: `{after_name}`\n\n\
         | test | key | unit | before | after | after/before | verdict |\n\
         |---|---|---|--:|--:|--:|---|\n"
    );
    let mut only = Vec::new();
    for ((test, key), b) in before {
        match after.get(&(test.clone(), key.clone())) {
            Some(a) => {
                s += &format!(
                    "| {} | {key} | {} | {:.3} | {:.3} | {:.3} | {} |\n",
                    short(test),
                    a.unit,
                    b.median,
                    a.median,
                    a.median / b.median,
                    verdict(b, a)
                );
            }
            None => only.push(format!("before only: {} / {key}", short(test))),
        }
    }
    for (test, key) in after.keys() {
        if !before.contains_key(&(test.clone(), key.clone())) {
            only.push(format!("after only: {} / {key}", short(test)));
        }
    }
    if !only.is_empty() {
        s += "\n";
        for o in only {
            s += &format!("- {o}\n");
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    const BEFORE: &str = r#"{"test":"t::a","dev":1,"verdict":"PASS","record":{"kind":"conditions","card":1,"sha":"x"}}
{"test":"t::a","dev":1,"verdict":"PASS","record":{"kind":"result","key":"add 512","unit":"us","timed":"host","median":10.0,"p10":9.5,"p90":10.5,"n":9}}
{"test":"t::a","dev":1,"verdict":"PASS","record":{"kind":"result","key":"add 512","unit":"us","timed":"host","median":20.0,"p10":19.0,"p90":21.0,"n":9}}
{"test":"t::b","dev":1,"verdict":"PASS","record":{"kind":"result","key":"matmul","unit":"TFLOPS","timed":"device","median":14.0,"p10":13.9,"p90":14.1,"n":7}}
{"test":"t::b","dev":1,"verdict":"PASS","record":{"kind":"result","key":"gone","unit":"ms","timed":"host","median":1.0,"p10":1.0,"p90":1.0,"n":7}}
"#;
    const AFTER: &str = r#"{"test":"t::a","dev":1,"verdict":"PASS","record":{"kind":"result","key":"add 512","unit":"us","timed":"host","median":10.2,"p10":9.8,"p90":10.6,"n":9}}
{"test":"t::a","dev":1,"verdict":"PASS","record":{"kind":"result","key":"add 512","unit":"us","timed":"host","median":25.0,"p10":24.0,"p90":26.0,"n":9}}
{"test":"t::b","dev":1,"verdict":"PASS","record":{"kind":"result","key":"matmul","unit":"TFLOPS","timed":"device","median":15.0,"p10":14.9,"p90":15.1,"n":7}}
{"test":"t::b","dev":1,"verdict":"PASS","record":{"kind":"result","key":"new","unit":"ms","timed":"host","median":1.0,"p10":1.0,"p90":1.0,"n":7}}
"#;

    #[test]
    fn joins_by_test_and_key_and_judges_by_spread() {
        let (b, a) = (parse(BEFORE).unwrap(), parse(AFTER).unwrap());
        assert_eq!(
            b.len(),
            4,
            "conditions are skipped, a repeated key kept apart"
        );
        let t = table("b.jsonl", "a.jsonl", &b, &a);
        assert!(
            t.contains("| a | add 512 | us | 10.000 | 10.200 | 1.020 | noise |"),
            "{t}"
        );
        assert!(
            t.contains("| a | add 512 #2 | us | 20.000 | 25.000 | 1.250 | **worse** |"),
            "{t}"
        );
        assert!(
            t.contains("| b | matmul | TFLOPS | 14.000 | 15.000 | 1.071 | better |"),
            "{t}"
        );
        assert!(t.contains("- before only: b / gone"), "{t}");
        assert!(t.contains("- after only: b / new"), "{t}");
    }

    #[test]
    fn a_line_without_a_record_is_refused_by_line() {
        let e = parse("{\"test\":\"t::a\"}\n").unwrap_err();
        assert!(e.contains("line 1"), "{e}");
    }

    #[test]
    fn units_say_which_way_is_better() {
        assert_eq!(higher_is_better("us/op"), Some(false));
        assert_eq!(higher_is_better("ms"), Some(false));
        assert_eq!(higher_is_better("TFLOPS"), Some(true));
        assert_eq!(higher_is_better("tokens/s"), Some(true));
        assert_eq!(higher_is_better("count"), None);
    }
}
