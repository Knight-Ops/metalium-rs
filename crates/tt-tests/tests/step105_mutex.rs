//! Tensix mutexes (`ATGETM`, `ATRELM`): independent arbitration model,
//! uncontended and contended gates, and the guarded-run machinery that makes a
//! blocking instruction safe to drive (role-side deadline, host-visible blocked
//! status, release from the host).
//!
//! The oracle is the pinned Blackhole pages (`ATGETM.md`, `ATRELM.md`,
//! `SyncUnit.md`) and nothing in `tt-isa`: the arbitration rule below is
//! written from the page text, not from `tt_isa::sync::mutex`.
//!
//! The threads report "I hold it" by posting a semaphore of their own, which the
//! guarded runner publishes (`Guard::snapshot`). Not an L1 store: ttsim refuses
//! `STOREIND` (divergence row 66's family), and this keeps one path for both
//! targets.
#[cfg(not(feature = "silicon"))]
use tt_isa::mailbox::guard;
use tt_isa::{
    backend::{self, Before},
    isa::Instruction,
    mailbox::guard::PollMode,
    scalar::atomic::{self as atomic},
    sync::mutex::{self, Mutex},
    sync::{self, Semaphore},
};
use tt_kernels::{
    atomics::{self, Budget, GuardSpec, GuardedProgram, Launch, RoleProgram, RoleState, Spec},
    l1::{Plan, Requirements},
};
use tt_tests::harness::{self, Run};

// ---------------------------------------------------------------------------
// Independent model: what the pages say, and nothing about the implementation.
// ---------------------------------------------------------------------------

/// `ATRELM.md`: "If a mutex is released by thread `i`, and both of the other
/// threads are trying to acquire it, then thread `(i + 1) % 3` is always chosen
/// to be the acquirer." With one waiter, that waiter acquires it (`ATGETM.md`:
/// a thread waits only while another holds it). With none, nobody does.
fn next_acquirer(releaser: usize, waiting: [bool; 3]) -> Option<usize> {
    let first = (releaser + 1) % 3;
    let second = (releaser + 2) % 3;
    if waiting[first] {
        Some(first)
    } else if waiting[second] {
        Some(second)
    } else {
        None
    }
}

/// The order in which the two non-holders acquire after `holder` releases,
/// when both are already waiting and each releases as soon as it has acquired.
fn model_order(holder: usize) -> [usize; 2] {
    let mut waiting = [true; 3];
    waiting[holder] = false;
    let first = next_acquirer(holder, waiting).unwrap();
    waiting[first] = false;
    let second = next_acquirer(first, waiting).unwrap();
    [first, second]
}

#[test]
fn model_matches_the_pages_worked_cases() {
    // ATRELM.md: released by i with both others waiting -> (i + 1) % 3.
    for i in 0..3 {
        let mut w = [true; 3];
        w[i] = false;
        assert_eq!(next_acquirer(i, w), Some((i + 1) % 3));
        // One waiter: that one, whichever it is.
        for j in 0..3 {
            if j != i {
                let mut one = [false; 3];
                one[j] = true;
                assert_eq!(next_acquirer(i, one), Some(j));
            }
        }
        assert_eq!(next_acquirer(i, [false; 3]), None);
    }
    assert_eq!(model_order(0), [1, 2]);
    assert_eq!(model_order(1), [2, 0]);
    assert_eq!(model_order(2), [0, 1]);
}

#[test]
fn only_indices_0_2_3_4_are_representable_and_encode_as_the_pages_say() {
    // SyncUnit.md: "four mutexes (index 0, and then indices 2 through 4)";
    // ATGETM.md / ATRELM.md: any other index waits forever.
    for index in 0..=70_000u32 {
        assert_eq!(
            Mutex::new(index).is_some(),
            [0, 2, 3, 4].contains(&index),
            "index {index}"
        );
    }
    for index in [0u32, 2, 3, 4] {
        let m = Mutex::new(index).unwrap();
        // TT_OP_ATGETM / TT_OP_ATRELM: opcode in the top byte, the index in the
        // low sixteen bits, everything between unused.
        assert_eq!(mutex::acquire(m).word(), 0xa0 << 24 | index);
        assert_eq!(mutex::release(m).word(), 0xa1 << 24 | index);
    }
    assert!(mutex::check_scope(&[mutex::acquire(Mutex::ALL[0])]).is_err());
}

// ---------------------------------------------------------------------------
// Programs.
// ---------------------------------------------------------------------------

/// Eight semaphores, all of them: `acq`, `go`, a "holds it" marker per thread
/// and a completion semaphore per guarded role.
struct Layout {
    plan: Plan,
    acq: Semaphore,
    go: Semaphore,
    mark: [Semaphore; 3],
    done: [Semaphore; 3],
}

impl Layout {
    fn new() -> Self {
        let mut req = Requirements::new(1);
        let acq = req.semaphore("mutex held", 0, 0..1);
        let go = req.semaphore("go", 0, 0..1);
        let mark = [
            req.semaphore("T0 holds", 0, 0..1),
            req.semaphore("T1 holds", 0, 0..1),
            req.semaphore("T2 holds", 0, 0..1),
        ];
        let done = [
            req.semaphore("T0 complete", 0, 0..1),
            req.semaphore("T1 complete", 0, 0..1),
            req.semaphore("T2 complete", 0, 0..1),
        ];
        let plan = req.plan(tt_isa::l1::DATA).unwrap();
        Layout {
            acq: plan.semaphore(acq),
            go: plan.semaphore(go),
            mark: mark.map(|s| plan.semaphore(s)),
            done: done.map(|s| plan.semaphore(s)),
            plan,
        }
    }
}

/// The holder: take `mutex`, say so, tell both waiters, then keep it until the
/// host posts `go`.
fn holder_body(l: &Layout, thread: usize, mutex: Mutex) -> Vec<Instruction> {
    let mut p = vec![
        mutex::acquire(mutex),
        sync::post(l.mark[thread]),
        sync::post(l.acq),
    ];
    p.extend(sync::take(l.go, Before::EVERYTHING));
    p.push(mutex::release(mutex));
    p
}

/// What a waiter does in place of the mutex: nothing (`None`, the "unlocked"
/// mutant: the acquire and release are NOPs) or another mutex (the "wrong
/// index" mutant).
type Which = Option<Mutex>;

/// A waiter: only once the holder holds the mutex, try to take it, say so, and
/// keep it until the host posts `go`.
fn waiter_body(l: &Layout, thread: usize, mutex: Which) -> Vec<Instruction> {
    // Waits for `acq` without taking it: two waiters taking one post would race
    // to a SEMGET on zero.
    let mut p = vec![sync::wait_nonzero(l.acq, Before::EVERYTHING)];
    p.push(mutex.map_or(backend::nop(), mutex::acquire));
    p.push(sync::post(l.mark[thread]));
    p.extend(sync::take(l.go, Before::EVERYTHING));
    p.push(mutex.map_or(backend::nop(), mutex::release));
    p
}

fn guard_spec() -> GuardSpec {
    // Silicon: the deadline is a wall-clock target turned into polls by
    // `atomics::SILICON_POLLS_PER_SECOND`, a conservative guess until
    // `guard_poll_rate_calibration` has measured it. ttsim counts in simulated
    // cycles.
    if harness::ON_SILICON {
        GuardSpec::for_seconds(
            DEADLINE_SECONDS,
            GRACE_SECONDS,
            atomics::SILICON_POLLS_PER_SECOND,
        )
        .unwrap()
    } else {
        GuardSpec::new(3_000, 3_000_000).unwrap()
    }
}

/// When a blocked role should report BLOCKED, and how long it keeps polling
/// afterwards for a release before giving up, on silicon.
const DEADLINE_SECONDS: f64 = 3.0;
const GRACE_SECONDS: f64 = 30.0;

/// How long the host waits for any one thing: three deadlines and some slack,
/// so a role that is polling at the assumed rate has time to report BLOCKED.
fn budget() -> Budget {
    Budget::new(
        60_000_000,
        std::time::Duration::from_secs_f64(3.0 * DEADLINE_SECONDS + 10.0),
    )
}

/// [`harness::in_device`] whose failure names the step. A fork that panics
/// reports only "the forked child panicked", and `in_device` itself runs the
/// thread-state reset (`harness::run`, a plain role run with a 1 s floor) before
/// the body: a tile left with a parked Tensix thread by an earlier run fails
/// *there*, and the panic text would otherwise carry no trace of which gate
/// was starting.
fn labelled(label: impl std::fmt::Display, f: impl FnOnce(&mut harness::Dev<'_>)) {
    let label = label.to_string();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| harness::in_device(f)));
    if let Err(payload) = result {
        let text = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "(non-text panic)".into());
        panic!("[{label}] {text}");
    }
}

/// `result` of a host wait, or a panic that says what the roles were doing.
fn expect_wait(
    dev: &mut harness::Dev<'_>,
    l: &Layout,
    launch: &Run3,
    result: Result<(), atomics::AtomicsError>,
    what: &str,
) {
    if let Err(e) = result {
        let state = diagnose(dev, l, launch);
        let _ = launch.abort_and_recover(dev, &tt_tests::firmware::ROLES);
        panic!("{what}: {e}; host-readable state:{state}");
    }
}

type Run3 = Launch<tt_isa::noc::Noc0>;

/// Which threads have posted their "holds it" marker. Markers are never taken
/// down, so the largest value any role's snapshot shows is the truth (a role
/// that has finished stops publishing).
fn marks(dev: &mut harness::Dev<'_>, l: &Layout, launch: &Run3) -> [u32; 3] {
    let mut out = [0u32; 3];
    for role in 0..3 {
        let sems = launch.semaphores(dev, role).unwrap();
        for t in 0..3 {
            out[t] = out[t].max(sems[l.mark[t].index() as usize]);
        }
    }
    out
}

fn settle(dev: &mut harness::Dev<'_>) {
    for _ in 0..60 {
        harness::advance(dev, 4096);
    }
}

/// One contended hand-off: `holder` takes its mutex while the other two, each
/// with its own entry of `mutexes`, wait behind it. Returns the order the
/// others acquired in, or why the mutual exclusion failed.
fn contended(
    dev: &mut harness::Dev<'_>,
    holder: usize,
    mutexes: [Which; 3],
) -> Result<[usize; 2], String> {
    let l = Layout::new();
    let spec = guard_spec();
    let programs: Vec<GuardedProgram> = (0..3)
        .map(|t| {
            let body = if t == holder {
                holder_body(&l, t, mutexes[t].expect("the holder takes a real mutex"))
            } else {
                waiter_body(&l, t, mutexes[t])
            };
            GuardedProgram::new(t, &body, spec, l.done[t]).unwrap()
        })
        .collect();
    let init = l.plan.semaphore_init();
    let run = Spec {
        roles: [
            RoleProgram::Guarded(&programs[0]),
            RoleProgram::Guarded(&programs[1]),
            RoleProgram::Guarded(&programs[2]),
        ],
        semaphores: &init,
        stage: &[],
    };
    let tile = harness::tensix_tile();
    let launch = atomics::start(dev, tile, &tt_tests::firmware::ROLES, &run, budget()).unwrap();
    let outcome = (|| -> Result<[usize; 2], String> {
        // Everyone is parked: the holder at its `go` wait, each waiter behind
        // the held mutex. A role reports BLOCKED only because its deadline
        // expired without its completion post.
        launch
            .wait(
                dev,
                budget(),
                "all three roles to report BLOCKED",
                |dev, l2| {
                    for t in 0..3 {
                        if l2.state(dev, t)? != RoleState::Blocked {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                },
            )
            .map_err(|e| e.to_string())?;
        settle(dev);
        let m = marks(dev, &l, &launch);
        if m[holder] != 1 {
            return Err(format!(
                "the holder never reported holding its mutex: {m:?}"
            ));
        }
        if (0..3).any(|t| t != holder && m[t] != 0) {
            return Err(format!("a waiter holds while the holder does: {m:?}"));
        }
        let mut order = Vec::new();
        let mut current = holder;
        for _ in 0..2 {
            let before = marks(dev, &l, &launch);
            launch
                .release(dev, current, 1 << l.go.index(), budget())
                .map_err(|e| e.to_string())?;
            // Exactly one waiter takes over.
            launch
                .wait(dev, budget(), "a waiter to take the mutex", |dev, l2| {
                    let now = marks(dev, &l, l2);
                    Ok((0..3).any(|t| now[t] > before[t]))
                })
                .map_err(|e| e.to_string())?;
            settle(dev);
            let after = marks(dev, &l, &launch);
            let new: Vec<usize> = (0..3).filter(|t| after[*t] > before[*t]).collect();
            if new.len() != 1 {
                return Err(format!("{} threads acquired at once: {after:?}", new.len()));
            }
            order.push(new[0]);
            current = new[0];
        }
        // The last holder lets go; everyone finishes.
        launch
            .release(dev, current, 1 << l.go.index(), budget())
            .map_err(|e| e.to_string())?;
        Ok([order[0], order[1]])
    })();
    // A failure says what every role was doing: its breadcrumb stage and live
    // poll count (still polling, or stuck), status and semaphores.
    let outcome = match outcome {
        Err(e) => Err(format!(
            "{e}; host-readable state:{}",
            diagnose(dev, &l, &launch)
        )),
        ok => ok,
    };
    let finished = match &outcome {
        Ok(_) => launch
            .finish(dev, budget())
            .map(|_| ())
            .map_err(|e| e.to_string()),
        Err(_) => launch
            .abort_and_recover(dev, &tt_tests::firmware::ROLES)
            .map_err(|e| e.to_string()),
    };
    match (outcome, finished) {
        (Ok(o), Ok(())) => Ok(o),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(format!("finish: {e}")),
    }
}

/// The mutexes a gate exercises: ttsim models only index 0 (divergence row
/// 76); silicon has all four.
fn mutexes_under_test() -> &'static [Mutex] {
    if harness::ON_SILICON {
        &Mutex::ALL
    } else {
        &Mutex::ALL[..1]
    }
}

#[test]
fn contended_handoff_follows_round_robin_for_every_mutex_and_holder() {
    for &mutex in mutexes_under_test() {
        for holder in 0..3 {
            let label = format!(
                "contended handoff, mutex {} held by thread {holder}",
                mutex.index()
            );
            labelled(label, |dev| {
                assert_eq!(
                    contended(dev, holder, [Some(mutex); 3]),
                    Ok(model_order(holder)),
                    "mutex {} held by thread {holder}",
                    mutex.index()
                );
            });
        }
    }
}

/// Negative control: with one waiter's acquire replaced by a NOP (the
/// "unlocked" mutant) it is not excluded, and the contended gate's exclusion
/// check must report it. Both waiters in turn, for every holder.
#[test]
fn control_an_unlocked_waiter_is_not_excluded_and_the_gate_notices() {
    for holder in 0..3 {
        for other in (0..3).filter(|t| *t != holder) {
            let mut mutexes = [Some(Mutex::ALL[0]); 3];
            mutexes[other] = None;
            let label = format!("control, thread {holder} holds, thread {other} unlocked");
            labelled(label, |dev| {
                let got = contended(dev, holder, mutexes);
                assert!(
                    matches!(&got, Err(e) if e.contains("a waiter holds while the holder does")),
                    "holder {holder}, unlocked {other}: {got:?}"
                );
            });
        }
    }
}

/// Mutexes are independent (`SyncUnit.md`): a waiter on a *different* mutex is
/// not serialized behind the holder, so the same check reports it. Silicon
/// only, since ttsim refuses indices 2 to 4.
#[cfg(feature = "silicon")]
#[test]
fn control_a_waiter_on_another_mutex_is_not_excluded() {
    for holder in 0..3 {
        let other = (holder + 1) % 3;
        let mut mutexes = [Some(Mutex::ALL[0]); 3];
        mutexes[other] = Some(Mutex::ALL[1]);
        let label = format!("control, thread {holder} holds, thread {other} on mutex 2");
        labelled(label, |dev| {
            let got = contended(dev, holder, mutexes);
            assert!(
                matches!(&got, Err(e) if e.contains("a waiter holds while the holder does")),
                "holder {holder}: {got:?}"
            );
        });
    }
}

// ---------------------------------------------------------------------------
// Uncontended use, on every mutex and every thread.
// ---------------------------------------------------------------------------

/// What the host can read about a role without touching its Tensix thread:
/// the raw status word and the decoded state, the guard's counters, and the
/// runner's semaphore snapshot (the mark and completion semaphores in it).
fn diagnose(dev: &mut harness::Dev<'_>, l: &Layout, launch: &Run3) -> String {
    use tt_isa::mailbox::role::Mailbox;
    let mut out = String::new();
    for role in 0..3 {
        let mb = Mailbox::of(role as u32);
        let raw = launch.read32(dev, mb.status()).unwrap_or(0xffff_ffff);
        let panic = launch.read32(dev, mb.panic_code()).unwrap_or(0xffff_ffff);
        let g = tt_isa::mailbox::guard::Guard::of(mb);
        let arm = launch.read32(dev, g.arm()).unwrap_or(0xffff_ffff);
        let polls = launch.read32(dev, g.polls()).unwrap_or(0xffff_ffff);
        let sems = launch.semaphores(dev, role);
        out += &format!(
            "\n  role {role}: status {raw:#010x} ({:?}) panic_code {panic:#x} \
             arm_word {arm:#x} polls {polls}\n    breadcrumbs: {}\n    semaphores {:?}",
            launch.state(dev, role),
            launch
                .breadcrumbs(dev, role)
                .unwrap_or_else(|e| format!("unreadable: {e}")),
            sems
        );
        if let Ok(s) = sems {
            out += &format!(
                "\n    marks {:?} completion {:?}",
                l.mark.map(|m| s[m.index() as usize]),
                l.done.map(|m| s[m.index() as usize])
            );
        }
    }
    out
}

/// One thread, one mutex: take it, mark, drop it, take it again (a released
/// mutex is available), mark again, drop it. On a timeout the host prints
/// what it can read without touching the hung thread, then holds the cores,
/// before panicking with the thread and mutex named.
fn uncontended(thread: usize, mutex: Mutex) {
    uncontended_in(thread, mutex, guard_spec().mode(), false);
}

/// [`uncontended`] with the poll design chosen. With `verbose` the decoded
/// breadcrumbs of every role are printed even on success.
fn uncontended_in(thread: usize, mutex: Mutex, mode: PollMode, verbose: bool) {
    let label = format!(
        "uncontended thread {thread} mutex {} {mode:?}",
        mutex.index()
    );
    labelled(label, |dev| {
        uncontended_on(dev, thread, mutex, mode, verbose)
    });
}

/// [`uncontended_in`] on an open device, so several runs can share a process
/// (and, on silicon, the semaphore and L1 state the previous one left).
fn uncontended_on(
    dev: &mut harness::Dev<'_>,
    thread: usize,
    mutex: Mutex,
    mode: PollMode,
    verbose: bool,
) {
    {
        let l = Layout::new();
        let body = [
            mutex::acquire(mutex),
            sync::post(l.mark[thread]),
            mutex::release(mutex),
            mutex::acquire(mutex),
            sync::post(l.mark[thread]),
            mutex::release(mutex),
        ];
        let p = GuardedProgram::new(thread, &body, guard_spec().with_mode(mode), l.done[thread])
            .unwrap();
        let mut roles = [RoleProgram::Idle; 3];
        roles[thread] = RoleProgram::Guarded(&p);
        let init = l.plan.semaphore_init();
        let launch = atomics::start(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            &Spec {
                roles,
                semaphores: &init,
                stage: &[],
            },
            budget(),
        )
        .unwrap();
        let waited = launch.wait(dev, budget(), "the thread to finish", |dev, l2| {
            Ok(l2.state(dev, thread)? == RoleState::Done)
        });
        if let Err(e) = waited {
            let state = diagnose(dev, &l, &launch);
            let _ = launch.abort_and_recover(dev, &tt_tests::firmware::ROLES);
            panic!(
                "uncontended thread {thread} mutex {} ({mode:?}): {e}; host-readable state:{state}",
                mutex.index()
            );
        }
        if verbose {
            println!(
                "BREADCRUMBS thread {thread} mutex {} {mode:?}: {}",
                mutex.index(),
                launch.breadcrumbs(dev, thread).unwrap()
            );
        }
        // The L1-word design has no semaphore snapshot to read the marks from.
        if mode != PollMode::L1Word {
            let m = marks(dev, &l, &launch);
            let mut want = [0; 3];
            want[thread] = 2;
            assert_eq!(m, want, "thread {thread} mutex {}", mutex.index());
        }
        launch.finish(dev, budget()).unwrap();
    }
}

/// Regression for state leaking between runs. A run on thread 0 leaves mark
/// semaphore 0 at 2, thread 0's snapshot word at the same, and (on silicon)
/// all of it in place for the next process. A following run on another thread
/// must start from a clean slate: marks `[0, 2, 0]`, not `[2, 2, 0]` (the
/// first silicon failure of `uncontended_t1_m0`, where an unguarded role's
/// stale snapshot was read back). Both runs share one device and process, on
/// either target.
#[test]
fn state_does_not_leak_from_one_run_into_the_next() {
    harness::in_device(|dev| {
        for (first, second) in [(0, 1), (1, 2), (2, 0)] {
            uncontended_on(dev, first, Mutex::ALL[0], guard_spec().mode(), false);
            uncontended_on(dev, second, Mutex::ALL[0], guard_spec().mode(), false);
        }
    });
}

/// Same gate, the cheaper poll (`PollMode::Light`): on ttsim as well, so the
/// designs stay in parity there.
#[test]
fn uncontended_t0_m0_light_poll() {
    uncontended_in(0, Mutex::ALL[0], PollMode::Light, false);
}

/// On ttsim every poll design that does not need `STOREIND` ends at the last
/// breadcrumb, having passed through the whole sequence; the host decodes it.
#[cfg(not(feature = "silicon"))]
#[test]
fn guard_breadcrumbs_reach_drained_on_the_simulator() {
    for mode in [PollMode::Full, PollMode::Light] {
        harness::in_device(|dev| {
            let l = Layout::new();
            let body = [mutex::acquire(Mutex::ALL[0]), mutex::release(Mutex::ALL[0])];
            let p = GuardedProgram::new(0, &body, guard_spec().with_mode(mode), l.done[0]).unwrap();
            let init = l.plan.semaphore_init();
            let launch = atomics::start(
                dev,
                harness::tensix_tile(),
                &tt_tests::firmware::ROLES,
                &Spec {
                    roles: [
                        RoleProgram::Guarded(&p),
                        RoleProgram::Idle,
                        RoleProgram::Idle,
                    ],
                    semaphores: &init,
                    stage: &[],
                },
                budget(),
            )
            .unwrap();
            launch
                .wait(dev, budget(), "DONE", |dev, l2| {
                    Ok(l2.state(dev, 0)? == RoleState::Done)
                })
                .unwrap();
            let text = launch.breadcrumbs(dev, 0).unwrap();
            assert!(text.contains("(DRAINED)"), "{mode:?}: {text}");
            launch.finish(dev, budget()).unwrap();
        });
    }
}

/// Silicon, one case, printing the decoded breadcrumbs even on success: the
/// coordinator's bisect of the guard's poll. The three designs (full snapshot
/// poll, light poll, L1-word poll) run in separate processes.
#[cfg(feature = "silicon")]
#[test]
fn guard_breadcrumbs_t0_m0() {
    uncontended_in(0, Mutex::ALL[0], PollMode::Full, true);
}

#[cfg(feature = "silicon")]
#[test]
fn guard_breadcrumbs_t0_m0_light() {
    uncontended_in(0, Mutex::ALL[0], PollMode::Light, true);
}

/// Calibration, silicon only, one poll design. Arms a program that blocks on a
/// semaphore nobody posts, with a deadline too large to expire, and reads the
/// runner's live poll count (written every 16 polls) once a second for three
/// seconds, printing `POLLRATE <mode> <polls/s>` and the decoded breadcrumbs
/// each time, so a role that is stuck rather than slow shows its stage and a
/// frozen count. Then releases the role through the host RELEASE path and
/// confirms it completes (a hung role fails there, with the diagnostics).
///
/// Put the Light figure (the design in use) into
/// `tt_kernels::atomics::SILICON_POLLS_PER_SECOND`.
#[cfg(feature = "silicon")]
fn calibrate(mode: PollMode) {
    harness::in_device(|dev| {
        use std::time::{Duration, Instant};
        let l = Layout::new();
        let body = sync::take(l.go, Before::EVERYTHING).to_vec();
        let spec = GuardSpec::new(1 << 29, 1 << 29).unwrap().with_mode(mode);
        let p = GuardedProgram::new(0, &body, spec, l.done[0]).unwrap();
        let init = l.plan.semaphore_init();
        let launch = atomics::start(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            &Spec {
                roles: [
                    RoleProgram::Guarded(&p),
                    RoleProgram::Idle,
                    RoleProgram::Idle,
                ],
                semaphores: &init,
                stage: &[],
            },
            budget(),
        )
        .unwrap();
        let g = p.guard();
        // In its poll loop: the first poll finished.
        let polling = launch.wait(dev, budget(), "the first poll", |dev, l2| {
            Ok(l2.read32(dev, g.stage())? >= tt_isa::mailbox::guard::stage::FIRST_POLL_DONE)
        });
        expect_wait(dev, &l, &launch, polling, "waiting for the poll loop");
        let start = Instant::now();
        let mut last = (Duration::ZERO, launch.read32(dev, g.polls()).unwrap());
        for _ in 0..3 {
            std::thread::sleep(Duration::from_secs(1));
            let now = (start.elapsed(), launch.read32(dev, g.polls()).unwrap());
            let seconds = (now.0 - last.0).as_secs_f64();
            println!(
                "POLLRATE {mode:?} {:.1} polls/s (live count {} -> {} over {seconds:.3} s)",
                (now.1 - last.1) as f64 / seconds,
                last.1,
                now.1
            );
            println!("  {}", launch.breadcrumbs(dev, 0).unwrap());
            if now.1 == last.1 {
                println!(
                    "POLLRATE {mode:?}: the live count did not move (stuck or under 16 polls/s)"
                );
            }
            last = now;
        }
        // The program is still blocked (the deadline is 2^29 polls); the host
        // release lets it go.
        let released = launch.release(dev, 0, 1 << l.go.index(), budget());
        expect_wait(dev, &l, &launch, released, "the host release");
        let done = launch.wait(dev, budget(), "DONE", |dev, l2| {
            Ok(l2.state(dev, 0)? == RoleState::Done)
        });
        expect_wait(dev, &l, &launch, done, "waiting for DONE after the release");
        launch.finish(dev, budget()).unwrap();
    });
}

/// The finding: `PollMode::Full` does not get past its first poll on silicon
/// when the program has to wait (see its doc comment). Kept so the evidence can
/// be reproduced; it is expected to fail there, and is not part of the smoke set.
#[cfg(feature = "silicon")]
#[test]
fn guard_poll_rate_calibration_full() {
    calibrate(PollMode::Full);
}

#[cfg(feature = "silicon")]
#[test]
fn guard_poll_rate_calibration_light() {
    calibrate(PollMode::Light);
}

#[cfg(feature = "silicon")]
#[test]
fn guard_poll_rate_calibration_l1_word() {
    calibrate(PollMode::L1Word);
}

/// Parity on ttsim for the design silicon cannot use: the full-snapshot poll
/// still completes there.
#[cfg(not(feature = "silicon"))]
#[test]
fn uncontended_t0_m0_full_poll() {
    uncontended_in(0, Mutex::ALL[0], PollMode::Full, false);
}

#[cfg(feature = "silicon")]
#[test]
fn guard_breadcrumbs_t0_m0_l1_word() {
    uncontended_in(0, Mutex::ALL[0], PollMode::L1Word, true);
}

/// One test per (thread, mutex) so a silicon filter runs each alone. ttsim
/// models mutex 0 only (divergence row 76), so the others are silicon gates.
macro_rules! uncontended_tests {
    ($($name:ident: $thread:expr, $mutex:expr $(, $silicon:meta)?;)*) => {$(
        #[test]
        $(#[$silicon])?
        fn $name() {
            uncontended($thread, Mutex::new($mutex).unwrap());
        }
    )*};
}

uncontended_tests! {
    uncontended_t0_m0: 0, 0;
    uncontended_t1_m0: 1, 0;
    uncontended_t2_m0: 2, 0;
    uncontended_t0_m2: 0, 2, cfg(feature = "silicon");
    uncontended_t1_m2: 1, 2, cfg(feature = "silicon");
    uncontended_t2_m2: 2, 2, cfg(feature = "silicon");
    uncontended_t0_m3: 0, 3, cfg(feature = "silicon");
    uncontended_t1_m3: 1, 3, cfg(feature = "silicon");
    uncontended_t2_m3: 2, 3, cfg(feature = "silicon");
    uncontended_t0_m4: 0, 4, cfg(feature = "silicon");
    uncontended_t1_m4: 1, 4, cfg(feature = "silicon");
    uncontended_t2_m4: 2, 4, cfg(feature = "silicon");
}

/// The minimal isolated probe for the coordinator: one thread, one mutex,
/// acquire and release, inside its own child process, no guard.
#[test]
fn probe_uncontended_acquire_release_survives() {
    let m = Mutex::ALL[0];
    let p = [mutex::acquire(m), mutex::release(m), atomic::consume()];
    assert!(harness::survives(|dev| {
        harness::run(dev, &Run::new(&p).dump_rows(0));
    }));
}

// ---------------------------------------------------------------------------
// The guard: deadline, blocked status, host release, one-shot arming.
// ---------------------------------------------------------------------------

/// A guarded program parked at a semaphore wait reports BLOCKED once its
/// deadline expires, and the host's release posts the semaphore from the
/// role's own runner and lets it finish.
#[test]
fn deadline_reports_blocked_and_the_host_release_completes_the_role() {
    harness::in_device(|dev| {
        let l = Layout::new();
        let body = sync::take(l.go, Before::EVERYTHING).to_vec();
        let p = GuardedProgram::new(0, &body, guard_spec(), l.done[0]).unwrap();
        let init = l.plan.semaphore_init();
        let run = Spec {
            roles: [
                RoleProgram::Guarded(&p),
                RoleProgram::Idle,
                RoleProgram::Idle,
            ],
            semaphores: &init,
            stage: &[],
        };
        let launch = atomics::start(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            &run,
            budget(),
        )
        .unwrap();
        let blocked = launch.wait(dev, budget(), "BLOCKED", |dev, l2| {
            Ok(l2.state(dev, 0)? == RoleState::Blocked)
        });
        expect_wait(dev, &l, &launch, blocked, "waiting for BLOCKED");
        // Still parked a while later: BLOCKED is not a transient.
        settle(dev);
        assert_eq!(launch.state(dev, 0).unwrap(), RoleState::Blocked);
        assert_eq!(
            launch.semaphores(dev, 0).unwrap()[l.done[0].index() as usize],
            0
        );
        let released = launch.release(dev, 0, 1 << l.go.index(), budget());
        expect_wait(dev, &l, &launch, released, "the host release");
        let done = launch.wait(dev, budget(), "DONE", |dev, l2| {
            Ok(l2.state(dev, 0)? == RoleState::Done)
        });
        expect_wait(dev, &l, &launch, done, "waiting for DONE after the release");
        launch.finish(dev, budget()).unwrap();
    });
}

/// Past the grace period a role that nothing releases gives up, publishing a
/// panic the host can see, and never issues the load that would hang it.
/// Simulator only: on silicon the Tensix thread would stay parked in its
/// `SEMWAIT` (divergence row 65), so the gate does not strand a real tile.
#[cfg(not(feature = "silicon"))]
#[test]
fn an_unreleased_role_gives_up_after_the_grace_period() {
    harness::in_device(|dev| {
        let l = Layout::new();
        let body = sync::take(l.go, Before::EVERYTHING).to_vec();
        let p =
            GuardedProgram::new(0, &body, GuardSpec::new(500, 500).unwrap(), l.done[0]).unwrap();
        let init = l.plan.semaphore_init();
        let run = Spec {
            roles: [
                RoleProgram::Guarded(&p),
                RoleProgram::Idle,
                RoleProgram::Idle,
            ],
            semaphores: &init,
            stage: &[],
        };
        let launch = atomics::start(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            &run,
            budget(),
        )
        .unwrap();
        // `wait` reports a panic as an error: that is the host-visible outcome.
        let seen = launch.wait(dev, budget(), "the runner to give up", |dev, l2| {
            Ok(l2.state(dev, 0)? == RoleState::Done)
        });
        match seen {
            Err(atomics::AtomicsError::Panicked { thread: 0, code }) => {
                assert_eq!(code, guard::ABANDONED)
            }
            other => panic!("expected the role to be abandoned, got {other:?}"),
        }
        launch
            .abort_and_recover(dev, &tt_tests::firmware::ROLES)
            .unwrap();
    });
}

/// Arming is consumed by the run it guards: afterwards the word is zero, so a
/// stale one (L1 survives between processes on silicon) cannot guard a later,
/// unrelated run; and the completion post was taken down again.
#[test]
fn arming_is_consumed_by_the_run_it_guards() {
    harness::in_device(|dev| {
        let l = Layout::new();
        let p = GuardedProgram::new(1, &[backend::nop()], guard_spec(), l.done[1]).unwrap();
        let init = l.plan.semaphore_init();
        let run = Spec {
            roles: [
                RoleProgram::Idle,
                RoleProgram::Guarded(&p),
                RoleProgram::Idle,
            ],
            semaphores: &init,
            stage: &[],
        };
        let launch = atomics::start(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            &run,
            budget(),
        )
        .unwrap();
        launch
            .wait(dev, budget(), "DONE", |dev, l2| {
                Ok(l2.state(dev, 1)? == RoleState::Done)
            })
            .unwrap();
        assert_eq!(launch.read32(dev, p.guard().arm()).unwrap(), 0);
        assert_eq!(
            launch.semaphores(dev, 1).unwrap()[l.done[1].index() as usize],
            0
        );
        launch.finish(dev, budget()).unwrap();
    });
}

/// A poke is a store by the role's RISC-V core to a word of the data arena, the
/// agent that frees an `ATCAS` or `ATINCGETPTR` polling that word (no Tensix
/// thread can: the Scalar Unit serves one instruction at a time for all three).
/// Runs on both targets with a role parked at a semaphore wait.
#[test]
fn a_poke_stores_a_word_from_the_runner_core() {
    labelled("poke from the runner core", |dev| {
        let l = Layout::new();
        let body = sync::take(l.go, Before::EVERYTHING).to_vec();
        let p = GuardedProgram::new(1, &body, guard_spec(), l.done[1]).unwrap();
        let init = l.plan.semaphore_init();
        let at = tt_isa::l1::DATA.base + 0x1000;
        let launch = atomics::start(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            &Spec {
                roles: [
                    RoleProgram::Idle,
                    RoleProgram::Guarded(&p),
                    RoleProgram::Idle,
                ],
                semaphores: &init,
                stage: &[(at, &[0u8; 4])],
            },
            budget(),
        )
        .unwrap();
        let blocked = launch.wait(dev, budget(), "BLOCKED", |dev, l2| {
            Ok(l2.state(dev, 1)? == RoleState::Blocked)
        });
        expect_wait(dev, &l, &launch, blocked, "waiting for BLOCKED");
        assert_eq!(launch.read32(dev, at).unwrap(), 0);
        let poked = launch.poke(dev, 1, at, 0xa5a5_1234, budget());
        expect_wait(dev, &l, &launch, poked, "the poke");
        assert_eq!(launch.read32(dev, at).unwrap(), 0xa5a5_1234);
        // The host refuses an address outside the arena before asking.
        assert!(launch
            .poke(dev, 1, tt_isa::mailbox::MAILBOX_BASE, 1, budget())
            .is_err());
        let released = launch.release(dev, 1, 1 << l.go.index(), budget());
        expect_wait(dev, &l, &launch, released, "the host release");
        launch.finish(dev, budget()).unwrap();
    });
}

/// The runner checks a poke itself: an address outside the data arena (written
/// past the host's check) panics it with `guard::REFUSED` rather than writing
/// a mailbox. Simulator only: the role's parked thread would stay parked on
/// silicon.
#[cfg(not(feature = "silicon"))]
#[test]
fn the_runner_refuses_a_poke_outside_the_data_arena() {
    harness::in_device(|dev| {
        let l = Layout::new();
        let body = sync::take(l.go, Before::EVERYTHING).to_vec();
        let p = GuardedProgram::new(1, &body, guard_spec(), l.done[1]).unwrap();
        let init = l.plan.semaphore_init();
        let launch = atomics::start(
            dev,
            harness::tensix_tile(),
            &tt_tests::firmware::ROLES,
            &Spec {
                roles: [
                    RoleProgram::Idle,
                    RoleProgram::Guarded(&p),
                    RoleProgram::Idle,
                ],
                semaphores: &init,
                stage: &[],
            },
            budget(),
        )
        .unwrap();
        launch
            .wait(dev, budget(), "BLOCKED", |dev, l2| {
                Ok(l2.state(dev, 1)? == RoleState::Blocked)
            })
            .unwrap();
        let g = p.guard();
        launch
            .write32(dev, g.poke_addr(), tt_isa::mailbox::MAILBOX_BASE as u32)
            .unwrap();
        launch.write32(dev, g.poke_value(), 7).unwrap();
        launch.write32(dev, g.release(), guard::POKE).unwrap();
        let seen = launch.wait(dev, budget(), "the runner to refuse", |dev, l2| {
            Ok(l2.state(dev, 1)? == RoleState::Done)
        });
        match seen {
            Err(atomics::AtomicsError::Panicked { thread: 1, code }) => {
                assert_eq!(code, guard::REFUSED)
            }
            other => panic!("expected the runner to refuse, got {other:?}"),
        }
        assert_ne!(
            launch.read32(dev, tt_isa::mailbox::MAILBOX_BASE).unwrap(),
            7,
            "nothing was written"
        );
        launch.abort(dev).unwrap();
    });
}

/// A program the guard cannot take is refused before anything is pushed: the
/// runner panics with `guard::REFUSED` rather than pushing into a FIFO a
/// blocked thread could fill. Likewise a completion semaphore that does not
/// exist.
#[cfg(not(feature = "silicon"))]
#[test]
fn the_runner_refuses_an_oversize_program_and_a_bad_completion_semaphore() {
    let g = guard::Guard::of(tt_isa::mailbox::role::Mailbox::single_core());
    for (words, complete) in [(guard::MAX_WORDS as usize + 1, 7), (2, 8)] {
        harness::in_device(|dev| {
            let program = vec![backend::nop(); words];
            let stage: Vec<(u64, [u8; 4])> = g
                .arm_writes(100, 100, complete, guard::PollMode::Full)
                .iter()
                .map(|(a, v)| (*a, v.to_le_bytes()))
                .collect();
            let staged: Vec<(u64, &[u8])> = stage.iter().map(|(a, b)| (*a, &b[..])).collect();
            let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                harness::run(dev, &Run::new(&program).dump_rows(0).stage(&staged));
            }));
            let message = *refused
                .expect_err("the guard must refuse this program")
                .downcast::<String>()
                .unwrap();
            assert!(
                message.contains(&format!("code {}", guard::REFUSED)),
                "{message}"
            );
        });
    }
}

// ---------------------------------------------------------------------------
// ttsim's treatment of mutexes (divergence row 76).
// ---------------------------------------------------------------------------

/// ttsim implements the Sync Unit mutexes, but refuses two things the page says
/// complete: releasing a mutex the thread does not hold ("no state is
/// changed"), and re-acquiring one it already holds ("maybe wait a cycle or
/// two"). Both are fatal there (`NonContractualBehavior`), so
/// `mutex::check_scope` refuses them on the host. Controls of the same shape
/// survive.
#[cfg(not(feature = "silicon"))]
#[test]
fn simulator_refuses_release_of_unheld_and_reacquire_with_surviving_controls() {
    let m = Mutex::ALL[0];
    let run = |p: Vec<Instruction>| {
        harness::survives(|dev| {
            harness::run(dev, &Run::new(&p).dump_rows(0));
        })
    };
    assert!(run(vec![mutex::acquire(m), mutex::release(m)]), "control");
    assert!(
        run(vec![mutex::acquire(m)]),
        "control: a mutex held at the end is not refused by ttsim (the host check is stricter)"
    );
    assert!(
        !run(vec![mutex::release(m)]),
        "ttsim now accepts ATRELM of an unheld mutex: update divergence row 76"
    );
    assert!(
        !run(vec![
            mutex::acquire(m),
            mutex::acquire(m),
            mutex::release(m)
        ]),
        "ttsim now accepts a re-acquire: update divergence row 76"
    );
    // ttsim models index 0 only: 2, 3 and 4 are `UntestedFunctionality`, the
    // indices Blackhole lacks `UndefinedBehavior`.
    for &other in &Mutex::ALL[1..] {
        assert!(
            !run(vec![mutex::acquire(other), mutex::release(other)]),
            "ttsim now models mutex {}: update divergence row 76 and run the gates on it",
            other.index()
        );
    }
    assert!(mutex::check_scope(&[mutex::release(m)]).is_err());
    assert!(mutex::check_scope(&[mutex::acquire(m), mutex::acquire(m)]).is_err());
}

/// What the page says and ttsim refuses, on silicon: both complete without
/// effect. Isolated, one child process each; run after the uncontended probe.
#[cfg(feature = "silicon")]
#[test]
fn silicon_release_of_unheld_and_reacquire_complete_without_effect() {
    let m = Mutex::ALL[0];
    for p in [
        vec![mutex::release(m), atomic::consume()],
        vec![
            mutex::acquire(m),
            mutex::acquire(m),
            mutex::release(m),
            atomic::consume(),
        ],
    ] {
        assert!(harness::survives(|dev| {
            harness::run(dev, &Run::new(&p).dump_rows(0));
        }));
    }
}
