//! B2 gate on real hardware: a card has one owner.
//!
//! `Session::open_card` takes the driver's lock on the card (`Kmd::lock_card`).
//! The driver ties the lock to the open file description and drops it when that
//! closes, so a second session -- in this process or another -- is refused with
//! `CardBusy` while the first lives, and succeeds once it is gone. Before the
//! lock a second process silently shared the chip and reset tiles under the
//! first.
//!
//! The controls are the success paths: the open that is refused works after
//! the owner is dropped (or exits), so the refusal is the lock's and not a card
//! that cannot be opened at all.

#![cfg(feature = "silicon")]

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use tt_kernels::session::{Session, SessionError, TileChoice};
use tt_tests::backend::{device_index, GATE_TILE};
use tt_tests::harness::assert_on_silicon;

/// Set in the child process that holds the card for the parent to collide with.
const CHILD: &str = "TT_CARD_LOCK_CHILD";

fn open(card: u16) -> Result<Session<tt_kmd::Kmd>, SessionError> {
    Session::open_card(
        card,
        tt_firmware_images::ROLES,
        TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
    )
}

#[track_caller]
fn assert_busy(card: u16) {
    match open(card) {
        Err(SessionError::CardBusy { card: busy }) => assert_eq!(busy, card),
        Err(other) => panic!("expected CardBusy, got: {other}"),
        Ok(_) => panic!("card {card} was opened twice"),
    }
}

#[test]
fn a_second_session_is_refused_while_the_first_lives_and_succeeds_after() {
    assert_on_silicon();
    let card = device_index();
    let first = open(card).unwrap_or_else(|e| panic!("{e}"));
    assert_busy(card);
    drop(first);
    drop(open(card).unwrap_or_else(|e| panic!("after the owner dropped: {e}")));
}

/// The process the next test starts: holds the card until its stdin closes.
/// Does nothing unless started that way.
#[test]
fn silicon_card_lock_child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    let session = open(device_index()).unwrap_or_else(|e| panic!("child: {e}"));
    println!("HELD");
    let mut line = String::new();
    // EOF: the parent is done with us.
    let _ = std::io::stdin().read_line(&mut line);
    drop(session);
}

#[test]
fn a_second_process_is_refused_while_the_first_lives_and_succeeds_after() {
    assert_on_silicon();
    let card = device_index();
    let mut child = Command::new(std::env::current_exe().expect("test binary"))
        .args(["--exact", "silicon_card_lock_child", "--nocapture"])
        .env(CHILD, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start the holder process");
    let mut out = BufReader::new(child.stdout.take().expect("piped"));
    let mut line = String::new();
    loop {
        line.clear();
        if out.read_line(&mut line).expect("read the holder") == 0 {
            panic!(
                "the holder exited without taking the card: {:?}",
                child.wait()
            );
        }
        if line.trim() == "HELD" {
            break;
        }
    }
    assert_busy(card);
    // Release it: closing stdin ends the holder, which drops its session.
    drop(child.stdin.take());
    assert!(child.wait().expect("wait").success());
    drop(open(card).unwrap_or_else(|e| panic!("after the holder exited: {e}")));
}
