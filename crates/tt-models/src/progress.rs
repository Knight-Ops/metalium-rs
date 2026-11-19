//! Where a long load or generation is, for a front end that shows it: one process-wide hook,
//! called from the loader and the traced generator. Without a hook (the default) a report costs
//! one atomic load.

use std::sync::{OnceLock, RwLock};

/// A step of a load or a generation, `done` of `total` (`done` counts the finished ones, so a
/// report with `done == total` closes the phase).
#[derive(Clone, Debug, PartialEq)]
pub enum Progress {
    /// Decoder layers loaded, prepared and stored ([`crate::qwen35::Qwen35::streamed`]): the
    /// last one's name (`layers.5`), kind (`DeltaNet`, `Attention`), its bytes in the checkpoint
    /// and the seconds it took from read to stored.
    Layer {
        done: usize,
        total: usize,
        name: String,
        kind: &'static str,
        bytes: u64,
        secs: f64,
    },
    /// Traces warmed up and captured (`TracedGenerator::new`), the last one named.
    Trace {
        done: usize,
        total: usize,
        key: String,
    },
    /// Prompt chunks fed.
    Prefill { done: usize, total: usize },
    /// Tokens generated after the prompt, the last one given.
    Token {
        done: usize,
        total: usize,
        token: i32,
    },
}

type Hook = Box<dyn Fn(&Progress) + Send + Sync>;

fn hook() -> &'static RwLock<Option<Hook>> {
    static HOOK: OnceLock<RwLock<Option<Hook>>> = OnceLock::new();
    HOOK.get_or_init(|| RwLock::new(None))
}

static SET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Call `f` with every report from now on (replacing an earlier hook); `None` removes it.
pub fn set_hook(f: Option<Hook>) {
    let on = f.is_some();
    *hook().write().unwrap_or_else(|e| e.into_inner()) = f;
    SET.store(on, std::sync::atomic::Ordering::Release);
}

/// Report `p` to the hook, if one is set.
pub(crate) fn report(p: Progress) {
    if !SET.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    if let Some(f) = hook().read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        f(&p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_hook_sees_reports_until_it_is_removed() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        set_hook(Some(Box::new(move |p| s.lock().unwrap().push(p.clone()))));
        let prefill = |done| Progress::Prefill { done, total: 2 };
        report(prefill(1));
        set_hook(None);
        report(prefill(2));
        assert_eq!(*seen.lock().unwrap(), [prefill(1)]);
    }
}
