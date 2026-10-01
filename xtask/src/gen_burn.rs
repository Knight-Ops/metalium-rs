//! `cargo xtask gen-burn-delegate`: the part of `burn-tt` that forwards to
//! `burn-flex`, generated from the pinned `burn-backend`'s op traits.
//!
//! Burn 0.21 has about two hundred op methods without a default, and `burn-tt`
//! starts as a backend that runs every one of them on the host through
//! `burn-flex` and routes a chosen few to a Tensix tile. Writing the forwarding
//! by hand would be transcription of exactly the kind this workspace generates
//! its way out of: an op forwarded to the wrong Flex op compiles when the two
//! signatures agree, and nothing but a test of that particular op would notice.
//!
//! So this reads each op trait's source, and for every method Flex itself
//! implements -- every required one, and each defaulted one Flex overrides --
//! emits a forward that converts each argument whose type
//! names the backend with `IntoFlex`, calls the same method on `Flex`, and
//! converts the result back with `FromFlex`, tagged with the device of the first
//! argument that carries one (`HasDevice`). The generator is syntactic on
//! purpose: what a type converts to is decided by those three traits in
//! `burn-tt/src/convert.rs`, where rustc checks it, not by a table here.
//!
//! **A defaulted method Flex does not override is left to its default**, which
//! then composes `burn-tt`'s own ops -- exactly as it composes Flex's for Flex.
//! That is what makes the two backends agree everywhere but the device ops,
//! *and* what lets a device op reach the defaults built on it. Forwarding the
//! default to Flex instead would run it on Flex's ops: `ModuleOps::linear` is
//! a default over `float_matmul` that Flex does not override, and `nn::Linear`
//! calls it, so forwarding it kept every `Linear` layer off the device (found
//! by `step12_mnist`'s first-forward gate, which asserts the device ran).
//! Which methods Flex implements is read from its own `impl ... for Flex`
//! blocks, from the pinned `burn-flex` source.
//!
//! Methods named in [`OVERRIDDEN`] are forwarded to a hand-written function in
//! `burn-tt/src/ops.rs` with the same signature instead. The generator refuses:
//! a method signature it cannot parse, a `where` clause or generic parameters
//! (none exist in 0.21; a new one needs looking at), a method whose result
//! names the backend but none of whose arguments carries a device, an `impl
//! Future` result not in [`OVERRIDDEN`], an identifier with no known import, and
//! an [`OVERRIDDEN`] entry that is not a method of its trait.
//!
//! Dependency-free, like the rest of xtask: `cargo metadata` locates the pinned
//! source (whose integrity `Cargo.lock`'s checksum already guarantees), and the
//! traits are brace-matched by hand, which Burn's regular trait syntax allows.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::util::{rustfmt, workspace_root};

/// The pinned version; `Cargo.toml` pins the crate with `=`.
pub const BURN_VERSION: &str = "0.21.0";

/// The op traits `Backend` requires, and the file under `src/backend/` each
/// lives in.
const TRAITS: &[(&str, &str)] = &[
    ("FloatTensorOps", "ops/tensor.rs"),
    ("IntTensorOps", "ops/int_tensor.rs"),
    ("BoolTensorOps", "ops/bool_tensor.rs"),
    ("ModuleOps", "ops/modules/base.rs"),
    ("ActivationOps", "ops/activation.rs"),
    ("QTensorOps", "ops/qtensor.rs"),
    ("TransactionOps", "ops/transaction.rs"),
];

/// Methods implemented by hand in `burn-tt/src/ops.rs`, by trait: the ones that
/// run on the device, the ones that say which device a tensor is on or move it,
/// and the ones returning futures.
pub const OVERRIDDEN: &[(&str, &[&str])] = &[
    (
        "FloatTensorOps",
        &[
            "float_matmul",
            "float_add",
            "float_sub",
            "float_mul",
            "float_mul_scalar",
            "float_add_scalar",
            "float_sub_scalar",
            "float_div",
            "float_div_scalar",
            "float_recip",
            "float_exp",
            "float_log",
            "float_sum_dim",
            "float_max_dim",
            "float_reshape",
            "float_slice",
            "float_swap_dims",
            "float_transpose",
            "float_device",
            "float_to_device",
            "float_into_data",
            "float_neg",
            "float_abs",
            "float_sign",
            "float_clamp",
            "float_clamp_min",
            "float_clamp_max",
            "float_equal",
            "float_not_equal",
            "float_greater",
            "float_greater_equal",
            "float_lower",
            "float_lower_equal",
            "float_equal_elem",
            "float_not_equal_elem",
            "float_greater_elem",
            "float_greater_equal_elem",
            "float_lower_elem",
            "float_lower_equal_elem",
            "float_is_nan",
            "float_is_inf",
            "float_mask_fill",
            "float_mask_where",
            "float_cast",
            "float_sqrt",
            "float_log1p",
            "float_powf",
            "float_powi",
            "float_powf_scalar",
            "float_powf_scalar_impl",
            "float_powi_scalar",
        ],
    ),
    (
        "IntTensorOps",
        &[
            "int_device",
            "int_to_device",
            "int_into_data",
            "int_into_float",
            "int_reshape",
            "int_slice",
            "int_swap_dims",
            "int_transpose",
        ],
    ),
    (
        "BoolTensorOps",
        &[
            "bool_device",
            "bool_to_device",
            "bool_into_data",
            "bool_argwhere",
            "bool_reshape",
            "bool_slice",
            "bool_swap_dims",
            "bool_transpose",
            "bool_not",
            "bool_and",
            "bool_or",
            "bool_xor",
        ],
    ),
    (
        "ActivationOps",
        &[
            "relu",
            "relu_backward",
            "softmax",
            "log_softmax",
            "leaky_relu",
            "hard_sigmoid",
            "prelu",
        ],
    ),
    ("QTensorOps", &["q_device", "q_to_device", "q_into_data"]),
    ("TransactionOps", &["tr_execute"]),
];

/// Where each identifier the signatures use is imported from.
const IMPORTS: &[(&str, &str)] = &[
    ("FloatTensor", "burn_backend::tensor::FloatTensor"),
    ("IntTensor", "burn_backend::tensor::IntTensor"),
    ("BoolTensor", "burn_backend::tensor::BoolTensor"),
    ("QuantizedTensor", "burn_backend::tensor::QuantizedTensor"),
    ("Device", "burn_backend::tensor::Device"),
    ("TensorPrimitive", "burn_backend::TensorPrimitive"),
    ("Scalar", "burn_backend::Scalar"),
    ("BoolDType", "burn_backend::BoolDType"),
    ("IntDType", "burn_backend::IntDType"),
    ("FloatDType", "burn_backend::FloatDType"),
    ("Shape", "burn_backend::Shape"),
    ("Slice", "burn_backend::Slice"),
    ("TensorData", "burn_backend::TensorData"),
    ("Distribution", "burn_backend::Distribution"),
    ("ExecutionError", "burn_backend::ExecutionError"),
    ("Range", "core::ops::Range"),
    ("Vec", "alloc_vec::Vec"),
    ("Option", "core::option::Option"),
    ("ConvOptions", "burn_backend::ops::ConvOptions"),
    (
        "ConvTransposeOptions",
        "burn_backend::ops::ConvTransposeOptions",
    ),
    ("DeformConvOptions", "burn_backend::ops::DeformConvOptions"),
    (
        "DeformConv2dBackward",
        "burn_backend::ops::DeformConv2dBackward",
    ),
    (
        "InterpolateOptions",
        "burn_backend::ops::InterpolateOptions",
    ),
    ("GridSampleOptions", "burn_backend::ops::GridSampleOptions"),
    ("UnfoldOptions", "burn_backend::ops::UnfoldOptions"),
    (
        "AttentionModuleOptions",
        "burn_backend::ops::AttentionModuleOptions",
    ),
    (
        "MaxPool1dWithIndices",
        "burn_backend::ops::MaxPool1dWithIndices",
    ),
    ("MaxPool1dBackward", "burn_backend::ops::MaxPool1dBackward"),
    (
        "MaxPool2dWithIndices",
        "burn_backend::ops::MaxPool2dWithIndices",
    ),
    ("MaxPool2dBackward", "burn_backend::ops::MaxPool2dBackward"),
    (
        "TransactionPrimitive",
        "burn_backend::ops::TransactionPrimitive",
    ),
    (
        "TransactionPrimitiveData",
        "burn_backend::ops::TransactionPrimitiveData",
    ),
    ("QuantScheme", "burn_backend::quantization::QuantScheme"),
    (
        "QuantizationParametersPrimitive",
        "burn_backend::quantization::QuantizationParametersPrimitive",
    ),
    ("IndexingUpdateOp", "burn_backend::tensor::IndexingUpdateOp"),
    ("Future", "core::future::Future"),
];

/// Identifiers that need no import: the backend's own generic parameter,
/// primitive types, and paths.
const BUILTIN: &[&str] = &[
    "B", "usize", "bool", "f64", "f32", "i64", "u64", "u32", "i32", "Output", "Send", "Result",
    "static", "impl", "crate", "tensor", "mut",
];

/// One method of an op trait.
#[derive(Debug, Clone)]
pub struct Method {
    pub name: String,
    /// The trait gives it a body.
    pub defaulted: bool,
    /// `(pattern, type)`, with the type as written.
    pub args: Vec<(String, String)>,
    /// `None` for `()`.
    pub ret: Option<String>,
}

pub fn generate(check_only: bool) -> Result<(), String> {
    let root = workspace_root();
    let src = crate_source(&root, "burn-backend")?;
    let flex = flex_implements(&crate_source(&root, "burn-flex")?)?;
    let mut traits = Vec::new();
    for (name, file) in TRAITS {
        let path = src.join("src/backend").join(file);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        let methods = parse_trait(&text, name)?;
        let implemented = flex.iter().find(|(t, _)| t == name).map(|(_, m)| m);
        traits.push((*name, forwarded(name, methods, implemented)?));
    }
    check_overridden(&traits, OVERRIDDEN)?;
    let rendered = render(&traits, OVERRIDDEN)?;
    let generated = rustfmt(&rendered, &root)?;
    let dest = root.join("crates/burn-tt/src/generated/delegate.rs");
    if check_only {
        let current = std::fs::read_to_string(&dest)
            .map_err(|e| format!("reading {}: {e}", dest.display()))?;
        if current == generated {
            println!(
                "ok: {} is up to date with burn-backend {BURN_VERSION}",
                dest.display()
            );
            Ok(())
        } else {
            Err(format!(
                "{} is out of date with burn-backend {BURN_VERSION}.\n\
                 Run `cargo xtask gen-burn-delegate` and commit the result.",
                dest.display()
            ))
        }
    } else {
        std::fs::write(&dest, generated).map_err(|e| format!("writing {}: {e}", dest.display()))?;
        let count: usize = traits.iter().map(|(_, m)| m.len()).sum();
        println!("wrote {} ({count} methods)", dest.display());
        Ok(())
    }
}

/// The methods `burn-tt` must emit: every required one, every defaulted one
/// Flex implements (so its answer is Flex's), and every hand-written one.
/// Refuses a required method Flex does not implement, which would mean the
/// Flex parser has stopped reading Flex.
pub fn forwarded(
    tr: &str,
    methods: Vec<Method>,
    implemented: Option<&BTreeSet<String>>,
) -> Result<Vec<Method>, String> {
    let empty = BTreeSet::new();
    let implemented = implemented.unwrap_or(&empty);
    let hand: &[&str] = OVERRIDDEN
        .iter()
        .find(|(t, _)| *t == tr)
        .map(|(_, n)| *n)
        .unwrap_or(&[]);
    let mut out = Vec::new();
    for m in methods {
        let flex_has = implemented.contains(&m.name);
        if !m.defaulted && !flex_has {
            return Err(format!(
                "{tr}::{} is required but burn-flex's impl does not seem to define it; \
                 the Flex parser is not reading Flex",
                m.name
            ));
        }
        if !m.defaulted || flex_has || hand.contains(&m.name.as_str()) {
            out.push(m);
        }
    }
    Ok(out)
}

/// For each op trait, the methods `impl {Trait}<Flex> for Flex` defines,
/// across every file of `burn-flex`'s source.
fn flex_implements(flex: &Path) -> Result<Vec<(String, BTreeSet<String>)>, String> {
    let mut found: Vec<(String, BTreeSet<String>)> = Vec::new();
    let mut stack = vec![flex.join("src")];
    while let Some(dir) = stack.pop() {
        let entries =
            std::fs::read_dir(&dir).map_err(|e| format!("reading {}: {e}", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("reading {}: {e}", path.display()))?;
            for (tr, _) in TRAITS {
                for names in impl_methods(&text, &format!("impl {tr}<Flex> for Flex")) {
                    match found.iter_mut().find(|(t, _)| t == tr) {
                        Some((_, set)) => set.extend(names),
                        None => found.push((tr.to_string(), names)),
                    }
                }
            }
        }
    }
    Ok(found)
}

/// The names of the `fn`s directly inside each `{head} { ... }` in `source`.
pub fn impl_methods(source: &str, head: &str) -> Vec<BTreeSet<String>> {
    let s = strip_comments(source);
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(at) = s[from..].find(head) {
        let start = from + at;
        let Some(open) = s[start..].find(['{', ';']).map(|o| start + o) else {
            break;
        };
        from = open + 1;
        if s.as_bytes()[open] == b';' {
            continue;
        }
        let Ok(body) = matching_body(&s, open) else {
            break;
        };
        let b = body.as_bytes();
        let mut names = BTreeSet::new();
        let mut depth = 0;
        for i in 0..b.len() {
            match b[i] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                b'f' if depth == 0
                    && body[i..].starts_with("fn ")
                    && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) =>
                {
                    let rest = &body[i + 3..];
                    let end = rest
                        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                        .unwrap_or(rest.len());
                    names.insert(rest[..end].to_string());
                }
                _ => {}
            }
        }
        out.push(names);
    }
    out
}

/// A pinned crate's source directory, from `cargo metadata`.
fn crate_source(root: &Path, krate: &str) -> Result<PathBuf, String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let out = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--locked"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let json = String::from_utf8_lossy(&out.stdout);
    let needle = format!("{krate}-{BURN_VERSION}/Cargo.toml");
    let hit = json
        .split("\"manifest_path\":\"")
        .skip(1)
        .map(|rest| &rest[..rest.find('"').unwrap_or(0)])
        .find(|p| p.ends_with(&needle))
        .ok_or_else(|| format!("{krate} {BURN_VERSION} is not in the dependency graph"))?;
    Ok(Path::new(hit)
        .parent()
        .expect("a manifest has a directory")
        .to_path_buf())
}

fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
        } else if b[i] == b'"' {
            // String literals in default bodies may hold braces.
            out.push('"');
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            out.push('"');
            i += 1;
        } else if b[i] == b'\'' && b.get(i + 2) == Some(&b'\'') {
            // A char literal such as '{'.
            out.push_str("' '");
            i += 3;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// Every method of `pub trait {name}` in `source`, in order.
pub fn parse_trait(source: &str, name: &str) -> Result<Vec<Method>, String> {
    let s = strip_comments(source);
    let head = format!("pub trait {name}<B: Backend>");
    let start = s.find(&head).ok_or_else(|| format!("`{head}` not found"))?;
    let open = start + s[start..].find('{').ok_or("trait has no body")?;
    let body = matching_body(&s, open)?;
    let b = body.as_bytes();
    let mut methods = Vec::new();
    let mut depth = 0i32;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b'f' if depth == 0
                && body[i..].starts_with("fn ")
                && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) =>
            {
                let (sig, end) = signature(body, i)?;
                let mut m = parse_signature(&sig).map_err(|e| format!("{name}: {e}"))?;
                m.defaulted = b[end] == b'{';
                methods.push(m);
                i = end;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    if methods.is_empty() {
        return Err(format!(
            "{name} has no methods; the parser is not reading it"
        ));
    }
    Ok(methods)
}

/// The text between the brace at `open` and its match.
fn matching_body(s: &str, open: usize) -> Result<&str, String> {
    let mut depth = 0;
    for (i, c) in s[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(&s[open + 1..open + i]);
                }
            }
            _ => {}
        }
    }
    Err("unbalanced braces".into())
}

/// The signature starting at `at` (`fn ...`), up to its `;` or `{` outside
/// parentheses, whitespace-normalised; and where it ends.
fn signature(body: &str, at: usize) -> Result<(String, usize), String> {
    let mut paren = 0;
    for (off, c) in body[at..].char_indices() {
        match c {
            '(' => paren += 1,
            ')' => paren -= 1,
            ';' | '{' if paren == 0 => {
                let sig = body[at..at + off].split_whitespace().collect::<Vec<_>>();
                return Ok((sig.join(" "), at + off));
            }
            _ => {}
        }
    }
    Err(format!(
        "unterminated signature at `{}`",
        &body[at..at + 40]
    ))
}

fn parse_signature(sig: &str) -> Result<Method, String> {
    let rest = sig.strip_prefix("fn ").ok_or("not a fn")?;
    let paren = rest
        .find('(')
        .ok_or_else(|| format!("no argument list: `{sig}`"))?;
    let name = rest[..paren].trim().to_string();
    if name.contains('<') {
        return Err(format!("`{name}` has generic parameters; look at it"));
    }
    let close = close_paren(rest, paren)?;
    let args_text = &rest[paren + 1..close];
    let tail = rest[close + 1..].trim();
    if tail.contains(" where ") || tail.starts_with("where") {
        return Err(format!("`{name}` has a where clause; look at it"));
    }
    let ret = match tail.strip_prefix("->") {
        Some(r) => Some(r.trim().to_string()),
        None if tail.is_empty() => None,
        None => return Err(format!("`{name}`: unexpected `{tail}` after the arguments")),
    };
    let mut args = Vec::new();
    for part in split_top(args_text) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part.contains("self") {
            return Err(format!("`{name}` takes self; op traits are static"));
        }
        let colon = part
            .find(':')
            .ok_or_else(|| format!("`{name}`: argument `{part}` has no type"))?;
        args.push((
            part[..colon].trim().to_string(),
            part[colon + 1..].trim().to_string(),
        ));
    }
    Ok(Method {
        name,
        defaulted: false,
        args,
        ret,
    })
}

fn close_paren(s: &str, open: usize) -> Result<usize, String> {
    let mut depth = 0;
    for (i, c) in s[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(open + i);
                }
            }
            _ => {}
        }
    }
    Err(format!("unbalanced parentheses in `{s}`"))
}

/// `s` split at commas outside `<>`, `()` and `[]`.
fn split_top(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    let b = s.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'<' | b'(' | b'[' => depth += 1,
            b'>' if i > 0 && b[i - 1] == b'-' => {}
            b'>' | b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Whether a type names the backend, and so converts.
fn names_backend(ty: &str) -> bool {
    words(ty).any(|w| w == "B")
}

/// Whether an argument of this type can say which device it is on.
fn carries_device(ty: &str) -> bool {
    let t = ty.trim_start_matches('&').trim();
    [
        "FloatTensor<B>",
        "IntTensor<B>",
        "BoolTensor<B>",
        "QuantizedTensor<B>",
        "Device<B>",
        "TensorPrimitive<B>",
    ]
    .contains(&t)
        || [
            "Vec<FloatTensor<B>>",
            "Vec<IntTensor<B>>",
            "Vec<BoolTensor<B>>",
            "Vec<QuantizedTensor<B>>",
        ]
        .contains(&t)
}

fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty() && !w.chars().next().unwrap().is_ascii_digit())
}

/// `B` replaced by the backend, and `crate::` by `burn_backend::`.
fn for_backend(ty: &str) -> String {
    let ty = ty.replace("crate::", "burn_backend::");
    let mut out = String::new();
    let mut word = String::new();
    for c in ty.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_alphanumeric() || c == '_' {
            word.push(c);
        } else {
            if word == "B" {
                out.push_str("TtBackend");
            } else {
                out.push_str(&word);
            }
            word.clear();
            out.push(c);
        }
    }
    out.pop();
    out
}

pub fn check_overridden(
    traits: &[(&str, Vec<Method>)],
    overridden: &[(&str, &[&str])],
) -> Result<(), String> {
    for (tr, names) in overridden {
        let Some((_, methods)) = traits.iter().find(|(t, _)| t == tr) else {
            return Err(format!("OVERRIDDEN names `{tr}`, which is not an op trait"));
        };
        for n in names.iter() {
            if !methods.iter().any(|m| m.name == *n) {
                return Err(format!(
                    "OVERRIDDEN names `{tr}::{n}`, which burn-backend {BURN_VERSION} \
                     does not have"
                ));
            }
        }
    }
    Ok(())
}

pub fn render(
    traits: &[(&str, Vec<Method>)],
    overridden: &[(&str, &[&str])],
) -> Result<String, String> {
    let mut used = BTreeSet::new();
    let mut impls = String::new();
    for (tr, methods) in traits {
        let hand: &[&str] = overridden
            .iter()
            .find(|(t, _)| t == tr)
            .map(|(_, n)| *n)
            .unwrap_or(&[]);
        impls.push_str(&format!("impl {tr}<TtBackend> for TtBackend {{\n"));
        for m in methods {
            for (_, ty) in &m.args {
                used.extend(words(ty).map(str::to_string));
            }
            if let Some(r) = &m.ret {
                used.extend(words(r).map(str::to_string));
            }
            let params: Vec<String> = m
                .args
                .iter()
                .map(|(p, t)| format!("{p}: {}", for_backend(t)))
                .collect();
            let ret = m
                .ret
                .as_ref()
                .map(|r| format!(" -> {}", for_backend(r)))
                .unwrap_or_default();
            let names: Vec<&str> = m
                .args
                .iter()
                .map(|(p, _)| p.trim_start_matches("mut ").trim())
                .collect();
            impls.push_str(&format!("fn {}({}){ret} {{\n", m.name, params.join(", ")));
            if hand.contains(&m.name.as_str()) {
                impls.push_str(&format!(
                    "crate::ops::{}::{}({})\n}}\n",
                    module_of(tr),
                    m.name,
                    names.join(", ")
                ));
                continue;
            }
            if m.ret.as_deref().is_some_and(|r| r.contains("Future")) {
                return Err(format!(
                    "{tr}::{} returns a future; implement it in burn-tt/src/ops.rs and \
                     name it in OVERRIDDEN",
                    m.name
                ));
            }
            let converts_ret = m.ret.as_deref().is_some_and(names_backend);
            let device_arg = m
                .args
                .iter()
                .zip(&names)
                .find(|((_, t), _)| carries_device(t))
                .map(|(_, n)| *n);
            if converts_ret {
                let d = device_arg.ok_or_else(|| {
                    format!(
                        "{tr}::{}: the result names the backend but no argument \
                         says which device it is on; implement it by hand",
                        m.name
                    )
                })?;
                impls.push_str(&format!("let device = HasDevice::tt_device(&{d});\n"));
            }
            let call_args: Vec<String> = m
                .args
                .iter()
                .zip(&names)
                .map(|((_, t), n)| {
                    if names_backend(t) {
                        format!("{n}.into_flex()")
                    } else {
                        n.to_string()
                    }
                })
                .collect();
            let call = format!("<Flex as {tr}<Flex>>::{}({})", m.name, call_args.join(", "));
            if converts_ret {
                impls.push_str(&format!("FromFlex::from_flex({call}, device)\n}}\n"));
            } else {
                impls.push_str(&format!("{call}\n}}\n"));
            }
        }
        impls.push_str("}\n\n");
    }

    let mut imports = String::new();
    for w in &used {
        if BUILTIN.contains(&w.as_str()) || TRAITS.iter().any(|(t, _)| t == w) {
            continue;
        }
        let Some((_, path)) = IMPORTS.iter().find(|(n, _)| n == w) else {
            return Err(format!(
                "a signature uses `{w}`, which gen_burn's IMPORTS does not know how to import"
            ));
        };
        if path.starts_with("alloc_vec") || path.starts_with("core::option") {
            continue;
        }
        imports.push_str(&format!("#[allow(unused_imports)]\nuse {path};\n"));
    }
    let trait_imports: Vec<&str> = TRAITS.iter().map(|(t, _)| *t).collect();
    Ok(format!(
        "//! Forwarding of every `burn-backend` {BURN_VERSION} op to `burn-flex`, except\n\
         //! the ones `burn-tt/src/ops.rs` implements.\n\
         //!\n\
         //! @generated by `cargo xtask gen-burn-delegate` from the pinned\n\
         //! burn-backend's op traits. Do not edit: `gen-burn-delegate --check` fails\n\
         //! if this file and its source disagree.\n\n\
         #![allow(clippy::too_many_arguments, unused_variables)]\n\n\
         use burn_backend::ops::{{{}}};\n\
         use burn_flex::Flex;\n\n\
         {imports}\n\
         use crate::convert::{{FromFlex, HasDevice, IntoFlex}};\n\
         use crate::TtBackend;\n\n\
         {impls}",
        trait_imports.join(", ")
    ))
}

/// The `burn-tt/src/ops.rs` module holding a trait's hand-written methods.
fn module_of(tr: &str) -> &'static str {
    match tr {
        "FloatTensorOps" => "float",
        "IntTensorOps" => "int",
        "BoolTensorOps" => "bool",
        "ModuleOps" => "module",
        "ActivationOps" => "activation",
        "QTensorOps" => "quantized",
        "TransactionOps" => "transaction",
        _ => unreachable!("TRAITS lists every trait"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
pub trait FloatTensorOps<B: Backend> {
    /// A comment with a { brace.
    fn float_add(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;
    fn float_zeros(shape: Shape, device: &Device<B>, dtype: FloatDType) -> FloatTensor<B> {
        let x = "}";
        todo!()
    }
    fn float_sort_with_indices(
        tensor: FloatTensor<B>,
        dim: usize,
    ) -> (FloatTensor<B>, IntTensor<B>);
}
"#;

    #[test]
    fn parses_required_and_defaulted_methods_across_lines() {
        let m = parse_trait(SAMPLE, "FloatTensorOps").unwrap();
        let names: Vec<&str> = m.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            ["float_add", "float_zeros", "float_sort_with_indices"]
        );
        assert_eq!(m[1].args[1], ("device".into(), "&Device<B>".into()));
        assert_eq!(m[2].ret.as_deref(), Some("(FloatTensor<B>, IntTensor<B>)"));
    }

    #[test]
    fn forwards_with_the_first_argument_that_carries_a_device() {
        let m = parse_trait(SAMPLE, "FloatTensorOps").unwrap();
        let out = render(&[("FloatTensorOps", m)], &[]).unwrap();
        assert!(out.contains("let device = HasDevice::tt_device(&lhs);"));
        assert!(out.contains(
            "<Flex as FloatTensorOps<Flex>>::float_add(lhs.into_flex(), rhs.into_flex())"
        ));
        // `shape` and `dtype` do not name the backend and pass through.
        assert!(out.contains("float_zeros(shape, device.into_flex(), dtype)"));
    }

    #[test]
    fn records_which_methods_have_defaults() {
        let m = parse_trait(SAMPLE, "FloatTensorOps").unwrap();
        let d: Vec<bool> = m.iter().map(|m| m.defaulted).collect();
        assert_eq!(d, [false, true, false]);
    }

    #[test]
    fn a_default_flex_does_not_override_is_left_to_the_default() {
        let m = parse_trait(SAMPLE, "FloatTensorOps").unwrap();
        let flex: BTreeSet<String> = ["float_add", "float_sort_with_indices"]
            .map(String::from)
            .into();
        let kept: Vec<String> = forwarded("FloatTensorOps", m.clone(), Some(&flex))
            .unwrap()
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(kept, ["float_add", "float_sort_with_indices"]);
        // Flex overriding the default brings it back.
        let flex: BTreeSet<String> = ["float_add", "float_zeros", "float_sort_with_indices"]
            .map(String::from)
            .into();
        assert_eq!(
            forwarded("FloatTensorOps", m.clone(), Some(&flex))
                .unwrap()
                .len(),
            3
        );
        // A required method Flex does not define means the parser is lost.
        let flex: BTreeSet<String> = ["float_add"].map(String::from).into();
        assert!(forwarded("FloatTensorOps", m, Some(&flex)).is_err());
    }

    #[test]
    fn reads_the_methods_of_an_impl_block() {
        let src = "impl FloatTensorOps<Flex> for Flex {\n fn a(x: u8) -> u8 { if x > 0 { 1 } else { 0 } }\n async fn b() {}\n}\nimpl TransactionOps<Flex> for Flex {}";
        let got = impl_methods(src, "impl FloatTensorOps<Flex> for Flex");
        assert_eq!(got, vec![["a", "b"].map(String::from).into()]);
        assert_eq!(
            impl_methods(src, "impl TransactionOps<Flex> for Flex"),
            vec![BTreeSet::new()]
        );
    }

    #[test]
    fn refuses_an_override_that_is_not_a_method() {
        let m = parse_trait(SAMPLE, "FloatTensorOps").unwrap();
        let traits = [("FloatTensorOps", m)];
        let err =
            check_overridden(&traits, &[("FloatTensorOps", &["float_nonexistent"])]).unwrap_err();
        assert!(err.contains("float_nonexistent"), "{err}");
        assert!(check_overridden(&traits, &[("FloatTensorOps", &["float_add"])]).is_ok());
    }

    #[test]
    fn refuses_a_result_with_no_device_to_tag_it_with() {
        let src = "pub trait FloatTensorOps<B: Backend> { fn f(x: usize) -> FloatTensor<B>; }";
        let m = parse_trait(src, "FloatTensorOps").unwrap();
        let err = render(&[("FloatTensorOps", m)], &[]).unwrap_err();
        assert!(err.contains("no argument"), "{err}");
    }

    #[test]
    fn refuses_an_unknown_identifier() {
        let src = "pub trait FloatTensorOps<B: Backend> { fn f(x: FloatTensor<B>, y: Mystery) -> FloatTensor<B>; }";
        let m = parse_trait(src, "FloatTensorOps").unwrap();
        let err = render(&[("FloatTensorOps", m)], &[]).unwrap_err();
        assert!(err.contains("Mystery"), "{err}");
    }

    #[test]
    fn refuses_generics_and_futures() {
        let g = "pub trait FloatTensorOps<B: Backend> { fn f<T>(x: FloatTensor<B>) -> T; }";
        assert!(parse_trait(g, "FloatTensorOps").is_err());
        let f = "pub trait FloatTensorOps<B: Backend> { fn f(x: FloatTensor<B>) -> impl Future<Output = TensorData> + Send; }";
        let m = parse_trait(f, "FloatTensorOps").unwrap();
        assert!(render(&[("FloatTensorOps", m)], &[])
            .unwrap_err()
            .contains("future"));
    }
}
