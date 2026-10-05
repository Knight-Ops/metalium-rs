//! Which op ran where, and what each one moved across PCIe.
//!
//! Generated dispatch (`xtask/src/gen_burn.rs`) opens an [`enter`] guard
//! for native and unsupported methods. Async methods enter when polled.
//! Downloads and uploads (`server.rs`) are charged
//! to the innermost op running on the calling thread, and the tensor
//! constructors (`tensor.rs`) say whether an op's result was made on the
//! device or on the host. [`report`] is the process-wide table; [`with_report`]
//! is one thread's, for a gate.
//!
//! `TT_REPORT=1` prints the table at exit. `TT_STRICT=1` (or [`set_strict`],
//! or [`strictly`] for one thread's block)
//! turns a download caused by a host op -- the device had the data and a
//! host op needed it -- and a host-staged matmul into a panic naming the op,
//! except where a caller has said host work is intended ([`host_ok`]).
//! Reading a tensor back (`*_into_data`, transactions) is never a fallback.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, Once};

/// One op's counts. `on_device` and `on_host` count calls by where their
/// result was made; a call whose result is neither (a shape, a view of a
/// host tensor, a future) counts in `calls` only.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OpStat {
    pub op: &'static str,
    /// Implemented in `burn-tt`, or an explicit unsupported method.
    pub hand_written: bool,
    pub calls: u64,
    pub on_device: u64,
    pub on_host: u64,
    pub downloads: u64,
    pub downloaded: u64,
    pub uploads: u64,
    pub uploaded: u64,
    /// Bytes a host-staged matmul copied to the device and back: operands
    /// that were not resident, per batch element (`ops.rs`'s batched path).
    pub staged: u64,
    /// The stored `[rows, cols]` of the first tensor this op downloaded.
    pub first_download: Option<[usize; 2]>,
}

impl OpStat {
    /// Bytes this op moved across PCIe, all ways.
    pub fn moved(&self) -> u64 {
        self.downloaded + self.uploaded + self.staged
    }

    fn add(&mut self, o: &OpStat) {
        self.calls += o.calls;
        self.on_device += o.on_device;
        self.on_host += o.on_host;
        self.downloads += o.downloads;
        self.downloaded += o.downloaded;
        self.uploads += o.uploads;
        self.uploaded += o.uploaded;
        self.staged += o.staged;
        self.first_download = self.first_download.or(o.first_download);
    }
}

/// A snapshot: every op seen, most bytes moved first, then most host results.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report(pub Vec<OpStat>);

impl Report {
    pub fn op(&self, name: &str) -> Option<&OpStat> {
        self.0.iter().find(|s| s.op == name)
    }

    /// Bytes moved by every op.
    pub fn moved(&self) -> u64 {
        self.0.iter().map(OpStat::moved).sum()
    }

    /// The ops that produced a result on the host, or moved bytes: what is
    /// left to bring to the device. An op that makes a tensor from no tensor
    /// (`from_data`, `zeros`, `random`, ...) is left out unless it moved
    /// bytes: its host result is uploaded by whichever device op needs it.
    pub fn off_device(&self) -> impl Iterator<Item = &OpStat> {
        self.0
            .iter()
            .filter(|s| s.moved() > 0 || (s.on_host > 0 && !is_creation(s.op)))
    }

    fn from_map(m: &HashMap<&'static str, OpStat>) -> Report {
        let mut v: Vec<OpStat> = m.values().cloned().collect();
        v.sort_by(|a, b| {
            (b.moved(), b.on_host, b.calls, a.op).cmp(&(a.moved(), a.on_host, a.calls, b.op))
        });
        Report(v)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn bytes(b: u64) -> String {
            match b {
                0 => "-".into(),
                b if b < 1 << 10 => format!("{b} B"),
                b if b < 1 << 20 => format!("{:.1} KiB", b as f64 / 1024.0),
                b => format!("{:.1} MiB", b as f64 / (1 << 20) as f64),
            }
        }
        writeln!(
            f,
            "{:<34} {:>5} {:>8} {:>8} {:>8} {:>11} {:>11} {:>11}  first download",
            "op", "path", "calls", "device", "host", "down", "up", "staged"
        )?;
        for s in &self.0 {
            writeln!(
                f,
                "{:<34} {:>5} {:>8} {:>8} {:>8} {:>11} {:>11} {:>11}  {}",
                s.op,
                if s.hand_written { "tt" } else { "unsupported" },
                s.calls,
                s.on_device,
                s.on_host,
                bytes(s.downloaded),
                bytes(s.uploaded),
                bytes(s.staged),
                s.first_download
                    .map(|[r, c]| format!("[{r}, {c}]"))
                    .unwrap_or_default()
            )?;
        }
        Ok(())
    }
}

/// Every thread's counts.
static GLOBAL: Mutex<Option<HashMap<&'static str, OpStat>>> = Mutex::new(None);

struct Frame {
    op: &'static str,
    /// Where the last tensor made inside this op (not inside a nested one)
    /// was made: `Some(true)` on the device. The last one is the result.
    made_on_device: Option<bool>,
}

thread_local! {
    static STACK: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
    static LOCAL: RefCell<HashMap<&'static str, OpStat>> = RefCell::new(HashMap::new());
    static HOST_OK: Cell<u32> = const { Cell::new(0) };
    static STRICT_HERE: Cell<u32> = const { Cell::new(0) };
}

/// Apply `f` to `op`'s counts, on this thread's table and the global one.
fn bump(op: &'static str, hand_written: Option<bool>, f: impl Fn(&mut OpStat)) {
    let apply = |m: &mut HashMap<&'static str, OpStat>| {
        let s = m.entry(op).or_insert_with(|| OpStat {
            op,
            ..OpStat::default()
        });
        if let Some(h) = hand_written {
            s.hand_written = h;
        }
        f(s);
    };
    LOCAL.with(|l| apply(&mut l.borrow_mut()));
    apply(
        GLOBAL
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get_or_insert_with(HashMap::new),
    );
}

/// The op running on this thread, innermost first.
fn current() -> Option<&'static str> {
    STACK.with(|s| s.borrow().last().map(|f| f.op))
}

/// Charges what happens on this thread to `op` until dropped.
#[must_use]
pub struct OpGuard {
    op: &'static str,
}

/// Enter op `op` (`hand_written`: `burn-tt` implements it; otherwise it is
/// explicitly unsupported). Called by generated dispatch.
pub fn enter(op: &'static str, hand_written: bool) -> OpGuard {
    static AT_EXIT: Once = Once::new();
    AT_EXIT.call_once(|| {
        if std::env::var("TT_REPORT").is_ok_and(|v| v == "1") {
            extern "C" fn print() {
                eprintln!("burn-tt report (TT_REPORT=1):\n{}", report());
            }
            // SAFETY: registers a plain `extern "C"` function with no
            // arguments, as `atexit` requires.
            unsafe { libc::atexit(print) };
        }
    });
    STACK.with(|s| {
        s.borrow_mut().push(Frame {
            op,
            made_on_device: None,
        })
    });
    bump(op, Some(hand_written), |s| s.calls += 1);
    OpGuard { op }
}

impl Drop for OpGuard {
    fn drop(&mut self) {
        let made = STACK.with(|s| s.borrow_mut().pop().and_then(|f| f.made_on_device));
        match made {
            Some(true) => bump(self.op, None, |s| s.on_device += 1),
            Some(false) => bump(self.op, None, |s| s.on_host += 1),
            None => {}
        }
    }
}

/// A tensor was made, on the device or on the host, inside the current op.
pub(crate) fn made(on_device: bool) {
    STACK.with(|s| {
        if let Some(f) = s.borrow_mut().last_mut() {
            f.made_on_device = Some(on_device);
        }
    });
}

/// The name a transfer outside every op is charged to: a tensor built by a
/// caller with `burn-tt`'s own API rather than through Burn.
const OUTSIDE: &str = "(outside an op)";

/// Ops that make a tensor from no tensor operand.
fn is_creation(op: &str) -> bool {
    [
        "_from_data",
        "_zeros",
        "_ones",
        "_full",
        "_empty",
        "_random",
        "_arange",
        "_arange_step",
    ]
    .iter()
    .any(|s| op.ends_with(s))
}

/// Ops whose download is the caller reading a result, never a fallback.
fn is_readback(op: &str) -> bool {
    op.ends_with("_into_data") || op == "tr_execute"
}

/// A tensor of stored `[rows, cols]` was downloaded.
pub(crate) fn downloaded(rows: usize, cols: usize) {
    downloaded_width(rows, cols, 4);
}

pub(crate) fn downloaded_width(rows: usize, cols: usize, width: usize) {
    let op = current().unwrap_or(OUTSIDE);
    let bytes = (rows * cols * width) as u64;
    bump(op, None, |s| {
        s.downloads += 1;
        s.downloaded += bytes;
        s.first_download.get_or_insert([rows, cols]);
    });
    if refuses() && !is_readback(op) {
        panic!(
            "burn-tt strict mode: `{op}` downloaded a [{rows}, {cols}] device tensor to \
             run on the host. Bring the op to the device, or wrap intended host work \
             in burn_tt::host_ok(..)\n{}",
            report_of_thread()
        );
    }
}

/// A tensor of stored `[rows, cols]` was uploaded.
pub(crate) fn uploaded(rows: usize, cols: usize) {
    uploaded_width(rows, cols, 4);
}

pub(crate) fn uploaded_width(rows: usize, cols: usize, width: usize) {
    let op = current().unwrap_or(OUTSIDE);
    let bytes = (rows * cols * width) as u64;
    bump(op, None, |s| {
        s.uploads += 1;
        s.uploaded += bytes;
    });
}

/// A native reduction staged its input, column result and scalar through L1.
pub(crate) fn staged_reduce([rows, cols]: [usize; 2]) {
    let op = current().unwrap_or(OUTSIDE);
    // Input, column-reduction readback and reupload, scalar readback.
    let bytes = ((rows * cols + 2 * rows + 1) * 4) as u64;
    bump(op, None, |s| s.staged += bytes);
    if refuses() {
        panic!("burn-tt strict mode: `{op}` staged a native full reduction through the host (engine has no resident tensor storage)");
    }
}

/// A matmul staged its operands through the host: `bytes` up and back.
pub(crate) fn staged(mkn: [usize; 3]) {
    let op = current().unwrap_or(OUTSIDE);
    let [m, k, n] = mkn;
    let bytes = ((m * k + k * n + m * n) * 4) as u64;
    bump(op, None, |s| s.staged += bytes);
    if refuses() {
        panic!(
            "burn-tt strict mode: `{op}` staged a [{m}, {k}] @ [{k}, {n}] matmul through \
             the host (operands not resident as matrices). Wrap intended host work in \
             burn_tt::host_ok(..)"
        );
    }
}

/// Everything every thread has done so far.
pub fn report() -> Report {
    Report::from_map(
        GLOBAL
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get_or_insert_with(HashMap::new),
    )
}

fn report_of_thread() -> Report {
    LOCAL.with(|l| Report::from_map(&l.borrow()))
}

/// Forget everything counted so far, on every thread's global table and on
/// this thread's.
pub fn report_reset() {
    *GLOBAL.lock().unwrap_or_else(|p| p.into_inner()) = None;
    LOCAL.with(|l| l.borrow_mut().clear());
}

/// Run `f` and return what *this thread's* ops did during it: exact for a
/// gate, whatever other test threads do meanwhile.
pub fn with_report<R>(f: impl FnOnce() -> R) -> (R, Report) {
    let before = LOCAL.with(|l| l.borrow().clone());
    let r = f();
    let after = LOCAL.with(|l| l.borrow().clone());
    let mut diff: HashMap<&'static str, OpStat> = HashMap::new();
    for (op, a) in &after {
        let b = before.get(op).cloned().unwrap_or_default();
        let d = OpStat {
            op,
            hand_written: a.hand_written,
            calls: a.calls - b.calls,
            on_device: a.on_device - b.on_device,
            on_host: a.on_host - b.on_host,
            downloads: a.downloads - b.downloads,
            downloaded: a.downloaded - b.downloaded,
            uploads: a.uploads - b.uploads,
            uploaded: a.uploaded - b.uploaded,
            staged: a.staged - b.staged,
            first_download: if a.downloads > b.downloads {
                a.first_download
            } else {
                None
            },
        };
        if d.calls > 0 || d.moved() > 0 {
            diff.entry(op).or_default().add(&d);
            diff.get_mut(op).expect("inserted").op = op;
            diff.get_mut(op).expect("inserted").hand_written = a.hand_written;
        }
    }
    (r, Report::from_map(&diff))
}

/// Run `f` with strict mode's panics off on this thread: host work the caller
/// intends, such as a loss that has no device kernel yet. Still counted.
pub fn host_ok<R>(f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            HOST_OK.with(|h| h.set(h.get() - 1));
        }
    }
    HOST_OK.with(|h| h.set(h.get() + 1));
    let _restore = Restore;
    f()
}

/// Run `f` in strict mode on this thread only, whatever the process-wide
/// setting: what a gate uses to say a block of ops moves nothing it should
/// not, without changing what other test threads may do.
pub fn strictly<R>(f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            STRICT_HERE.with(|h| h.set(h.get() - 1));
        }
    }
    STRICT_HERE.with(|h| h.set(h.get() + 1));
    let _restore = Restore;
    f()
}

/// Does a fallback here panic: strict, and not inside [`host_ok`]?
fn refuses() -> bool {
    (STRICT_HERE.with(Cell::get) > 0 || strict()) && HOST_OK.with(Cell::get) == 0
}

/// Turn strict mode on or off for the process (`TT_STRICT=1` sets it on).
pub fn set_strict(on: bool) {
    STRICT.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}

/// Is strict mode on ([`set_strict`], or `TT_STRICT=1`)?
pub fn strict() -> bool {
    match STRICT.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var("TT_STRICT").is_ok_and(|v| v == "1");
            STRICT.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        v => v == 2,
    }
}

/// 0: not yet read from the environment; 1: off; 2: on.
static STRICT: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfers_are_charged_to_the_innermost_op_and_results_to_their_op() {
        let ((), r) = with_report(|| {
            let _outer = enter("outer_op", true);
            uploaded(32, 32);
            {
                let _inner = enter("inner_op", false);
                downloaded(64, 10);
                made(false);
            }
            made(true);
        });
        let outer = r.op("outer_op").unwrap();
        assert_eq!(
            (outer.calls, outer.on_device, outer.on_host, outer.uploaded),
            (1, 1, 0, 32 * 32 * 4)
        );
        let inner = r.op("inner_op").unwrap();
        assert_eq!(
            (
                inner.calls,
                inner.on_host,
                inner.downloaded,
                inner.hand_written
            ),
            (1, 1, 64 * 10 * 4, false)
        );
        assert_eq!(inner.first_download, Some([64, 10]));
        assert_eq!(r.0[0].op, "outer_op", "most bytes first:\n{r}");
    }

    #[test]
    fn a_readback_is_not_a_fallback_and_host_ok_allows_one() {
        let caught = strictly(|| {
            {
                let _g = enter("float_into_data", true);
                downloaded(1, 1);
            }
            host_ok(|| {
                let _g = enter("float_gather", false);
                downloaded(1, 1);
            });
            std::panic::catch_unwind(|| {
                let _g = enter("float_gather", false);
                downloaded(1, 1);
            })
        });
        let msg = caught.unwrap_err();
        let msg = msg
            .downcast_ref::<String>()
            .expect("a formatted panic message");
        assert!(msg.contains("`float_gather`"), "{msg}");
    }
}
